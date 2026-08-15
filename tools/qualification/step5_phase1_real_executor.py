#!/usr/bin/env python3
"""Real, process-isolated GPU executor for the Step-5 Phase-1 campaign.

The campaign deliberately owns the oracle while this module owns device
construction and transport.  Every replicate is a separate worker process.
That boundary is important: terminal lane A is allowed to wedge after B3, so
destroying an epoch must not depend on the CUDA worker returning to Python.

The executor is intended to run in the qualified runtime image.  Configuration
is supplied as canonical JSON through ``NEMOTRON_PHASE1_EXECUTOR_CONFIG``; no
model or image identity is guessed from a running production service.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from step5_phase1_campaign import (
    CaseSpec,
    ExecutionResult,
    JsonValue,
    MaterializedFixture,
    OraclePlan,
    Phase1Executor,
    load_frozen_fixture,
    sha256_file,
    stable_digest,
    validate_materialized_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCE = Path(__file__).with_name("step5_phase1_gpu_worker.py")
TYPED_CARRIER_SOURCE = Path(__file__).with_name("step5_phase1_typed_carrier.py")
POCKET_WORKER_SOURCE = REPO_ROOT / "src/nemotron_voicechat_runtime/pocket_worker.py"
RUNTIME_OPTIMIZATIONS_SOURCE = REPO_ROOT / "src/nemotron_voicechat_runtime/runtime_optimizations.py"
SERVER_SOURCE = REPO_ROOT / "src/nemotron_voicechat_runtime/server.py"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.42
DEFAULT_RPC_TIMEOUT_SECONDS = 120.0
DEFAULT_CONSTRUCTION_TIMEOUT_SECONDS = 900.0
DEFAULT_TERMINAL_TIMEOUT_SECONDS = 30.0
CONFIG_ENV = "NEMOTRON_PHASE1_EXECUTOR_CONFIG"
USED_IMAGE_ENV = "NEMOTRON_PHASE1_USED_IMAGE_SHA256"
FROZEN_CONFIG_PATH = REPO_ROOT / "reports/step5-wedge/phase1-real-executor-config.json"
SEALED_WORKER_ENVIRONMENT = {
    "EA_DISABLE_CYCLIC_GC": "1",
    "EA_PERCEPTION_DTYPE": "bf16",
    "HF_HUB_OFFLINE": "1",
    "S2S_TTS_TEXT_TOKEN_MIN": "5",
    "S2S_TTS_TEXT_TOKEN_RATIO_CAP": "16",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    "VOICECHAT_EARTTS_DECODE_PAD_SILENCE": "1",
    "VOICECHAT_EARTTS_IDLE_PAD_BYPASS": "1",
    "VOICECHAT_EARTTS_PREPARED_EPOCH": "1",
    "VOICECHAT_EARTTS_RESET_ON_BOS": "1",
    "VOICECHAT_NANO_PAD_PAIR": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "0",
    "VOICECHAT_NANO_PAD_PAIR_TRACE": "1",
    "VOICECHAT_NANO_PAIR_FULL_GRAPH": "0",
    "VOICECHAT_STREAMING_MAX_LEN": "12288",
    "VOICECHAT_USE_PERCEPTION_CUDAGRAPH": "1",
    "VOICECHAT_VLLM_DELTA_OUTPUT": "1",
    "VOICECHAT_VLLM_DELTA_OUTPUT_TARGET": "eartts",
    "VOICECHAT_VLLM_EARTTS_ENFORCE_EAGER": "0",
    "VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION": "0.10",
    "VOICECHAT_VLLM_MAX_MODEL_LEN": "12288",
    "VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION": "0.42",
}


def _tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Return the complete regular-file inventory and reject ambiguous entries."""
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"sealed tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"sealed tree contains a non-regular entry: {path}")
        inventory[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_model_file(path),
        }
    if not inventory:
        raise ValueError(f"hash tree is empty: {root}")
    return inventory


def _inventory_digest(inventory: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative, seal in inventory.items():
        digest.update(relative.encode() + b"\0")
        digest.update(str(seal["bytes"]).encode() + b"\0")
        digest.update(str(seal["sha256"]).encode() + b"\n")
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    """Hash all regular files without following links."""
    return _inventory_digest(_tree_inventory(root))


def _sha256_model_file(path: Path) -> str:
    """Hash a large model file without pinning its pages before GPU construction."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
        try:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            # Identity remains exact where the advisory is unavailable; the
            # qualified Linux launcher separately checks that the GPU is idle.
            pass
    return digest.hexdigest()


def _verify_model_artifacts(config: "RealExecutorConfig") -> dict[str, Any]:
    """Hash the files that the selected pipeline will actually load."""
    manifest = json.loads(config.vllm_manifest.read_text(encoding="utf-8"))
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise ValueError("vLLM model manifest lacks components")
    roots = {"nano": config.nano_vllm_path, "eartts": config.eartts_vllm_path}
    observed: dict[str, Any] = {}
    for name, root in roots.items():
        root = root.resolve()
        component = components.get(name)
        files = component.get("files") if isinstance(component, dict) else None
        model = component.get("model") if isinstance(component, dict) else None
        if not isinstance(files, dict) or not isinstance(model, dict):
            raise ValueError(f"model manifest component is incomplete: {name}")
        expected_inventory: dict[str, dict[str, Any]] = {}
        for relative, expected in sorted(files.items()):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"model manifest path escapes {name} root: {relative}")
            _require_file(path, f"{name} artifact {relative}")
            expected_inventory[relative] = expected
        actual_inventory = _tree_inventory(root)
        if actual_inventory != expected_inventory:
            raise ValueError(f"used {name} artifact tree differs from the manifest file set")
        observed[name] = {
            "files": actual_inventory,
            "manifest_model_sha256": model.get("sha256"),
            "used_tree_sha256": _inventory_digest(actual_inventory),
        }
    source = manifest.get("source")
    expected_parent = source.get("model", {}).get("sha256") if isinstance(source, dict) else None
    checkpoint_root = config.checkpoint_root.resolve()
    checkpoint_inventory = _tree_inventory(checkpoint_root)
    for relative in ("config.json", "model.safetensors", "rnnt_tokenizer/tokenizer.model"):
        if relative not in checkpoint_inventory:
            raise ValueError(f"used parent checkpoint artifact is absent: {relative}")
    parent_sha = checkpoint_inventory["model.safetensors"]["sha256"]
    if parent_sha != expected_parent:
        raise ValueError("used parent checkpoint differs from the model manifest")
    speech_runtime_files: dict[str, dict[str, Any]] = {}
    for relative in (
        "examples/speechlm2/function_calling/template.jinja",
        "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
    ):
        path = _require_file(config.speech_root / relative, f"Speech runtime artifact {relative}")
        speech_runtime_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_model_file(path),
        }
    identity = {
        "checkpoint_files": checkpoint_inventory,
        "checkpoint_tree_sha256": _inventory_digest(checkpoint_inventory),
        "eartts_model_sha256": observed["eartts"]["used_tree_sha256"],
        "hf_skeleton_tree_sha256": _hash_tree(config.hf_skeleton),
        "manifest_sha256": _sha256_model_file(config.vllm_manifest),
        "nano_model_sha256": observed["nano"]["used_tree_sha256"],
        "parent_model_sha256": parent_sha,
        "speech_runtime_files": speech_runtime_files,
        "verified_files": observed,
    }
    identity["model_sha256"] = stable_digest(identity)
    return identity


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is absent: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is absent: {resolved}")
    return resolved


@dataclass(frozen=True)
class RealExecutorConfig:
    """Sealed paths and limits for the qualified GPU worker."""

    patched_vllm_root: Path
    patch_manifest: Path
    image_sha256: str
    model_sha256: str
    speech_root: Path = Path("/opt/Speech")
    checkpoint_root: Path = Path("/models/voicechat")
    hf_skeleton: Path = Path("/models/NVIDIA-Nemotron-Nano-9B-v2")
    vllm_manifest: Path = Path("/models/candidates/fhw8-candidate-manifest.json")
    nano_vllm_path: Path = Path("/models/candidates/nano-fhw8")
    eartts_vllm_path: Path = Path("/models/derived/eartts")
    source_pcm: Path = Path(
        "/home/khkramer/.local/state/nemotron-voicechat/traces/model/"
        "2d08e743-39bb-4119-866d-8ab5c215581c/input-16000-mono-s16le.pcm"
    )
    source_typed_carrier_witness: Path = Path(
        "/home/khkramer/.local/state/nemotron-voicechat/traces/"
        "nano-sequential-browser-r1.log"
    )
    pocket_python: Path = Path("/opt/pocket-tts/bin/python")
    pocket_hf_hub: Path = Path("/models/huggingface/hub")
    system_prompt: str = "Respond concisely in one sentence. Do not repeat yourself."
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    construction_timeout_seconds: float = DEFAULT_CONSTRUCTION_TIMEOUT_SECONDS
    rpc_timeout_seconds: float = DEFAULT_RPC_TIMEOUT_SECONDS
    terminal_timeout_seconds: float = DEFAULT_TERMINAL_TIMEOUT_SECONDS
    python_executable: str = sys.executable
    work_root: Path | None = None
    selected_config_path: Path | None = None
    used_image_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RealExecutorConfig":
        required = {"patched_vllm_root", "patch_manifest", "image_sha256", "model_sha256"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"real executor configuration keys absent: {sorted(missing)}")
        path_fields = {
            "patched_vllm_root",
            "patch_manifest",
            "speech_root",
            "checkpoint_root",
            "hf_skeleton",
            "vllm_manifest",
            "nano_vllm_path",
            "eartts_vllm_path",
            "source_pcm",
            "source_typed_carrier_witness",
            "pocket_python",
            "pocket_hf_hub",
            "work_root",
            "selected_config_path",
        }
        converted = {
            key: (None if item is None else Path(item)) if key in path_fields else item
            for key, item in value.items()
        }
        return cls(**converted)

    def validated(self) -> "RealExecutorConfig":
        if self.gpu_memory_utilization != DEFAULT_GPU_MEMORY_UTILIZATION:
            raise ValueError(
                "Phase-1 gpu_memory_utilization must remain at the sealed harness default 0.42"
            )
        for name in ("image_sha256", "model_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if self.used_image_sha256 is not None and self.used_image_sha256 != self.image_sha256:
            raise ValueError("used runtime image differs from the frozen image identity")
        _require_dir(self.patched_vllm_root, "patched vLLM root")
        _require_file(self.patch_manifest, "Phase-1 patch manifest")
        _require_dir(self.speech_root, "Speech root")
        _require_dir(self.checkpoint_root, "VoiceChat checkpoint")
        _require_dir(self.hf_skeleton, "Nano skeleton")
        _require_file(self.vllm_manifest, "vLLM model manifest")
        _require_dir(self.nano_vllm_path, "Nano vLLM model")
        _require_dir(self.eartts_vllm_path, "EarTTS vLLM model")
        _require_file(self.source_pcm, "frozen source PCM")
        _require_file(self.source_typed_carrier_witness, "historical typed-carrier witness")
        _require_file(self.pocket_python, "Pocket Python interpreter")
        _require_dir(self.pocket_hf_hub, "Pocket Hugging Face hub")
        if not WORKER_SOURCE.is_file():
            raise ValueError(f"Phase-1 GPU worker is absent: {WORKER_SOURCE}")
        if not TYPED_CARRIER_SOURCE.is_file():
            raise ValueError(f"typed-carrier replay helper is absent: {TYPED_CARRIER_SOURCE}")
        return self

    def canonical(self) -> dict[str, JsonValue]:
        return {
            "checkpoint_root": str(self.checkpoint_root.expanduser().resolve()),
            "construction_timeout_seconds": self.construction_timeout_seconds,
            "eartts_vllm_path": str(self.eartts_vllm_path.expanduser().resolve()),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "hf_skeleton": str(self.hf_skeleton.expanduser().resolve()),
            "image_sha256": self.image_sha256,
            "model_sha256": self.model_sha256,
            "nano_vllm_path": str(self.nano_vllm_path.expanduser().resolve()),
            "patch_manifest": str(self.patch_manifest.expanduser().resolve()),
            "patched_vllm_root": str(self.patched_vllm_root.expanduser().resolve()),
            "python_executable": str(Path(self.python_executable).expanduser().resolve()),
            "rpc_timeout_seconds": self.rpc_timeout_seconds,
            "pocket_hf_hub": str(self.pocket_hf_hub.expanduser().resolve()),
            "pocket_python": str(self.pocket_python.expanduser().resolve()),
            "source_pcm": str(self.source_pcm.expanduser().resolve()),
            "source_typed_carrier_witness": str(
                self.source_typed_carrier_witness.expanduser().resolve()
            ),
            "speech_root": str(self.speech_root.expanduser().resolve()),
            "system_prompt": self.system_prompt,
            "terminal_timeout_seconds": self.terminal_timeout_seconds,
            "vllm_manifest": str(self.vllm_manifest.expanduser().resolve()),
            "work_root": (
                None if self.work_root is None else str(self.work_root.expanduser().resolve())
            ),
            "worker_protocol": 1,
        }


class EpochClient(Protocol):
    async def request(self, operation: str, payload: dict[str, Any], timeout: float) -> Any: ...

    async def close(self, reason: str) -> None: ...


class WorkerBackend(Protocol):
    async def materialize(self, fixture: dict[str, JsonValue]) -> MaterializedFixture: ...

    async def create_epoch(self, replicate: int) -> EpochClient: ...


class WorkerProtocolError(RuntimeError):
    pass


class _SubprocessEpoch:
    def __init__(self, process: asyncio.subprocess.Process, root: Path) -> None:
        self.process = process
        self.root = root
        self.sequence = 0
        self.construction: dict[str, Any] | None = None
        self.stderr_task: asyncio.Task[bytes] | None = None
        if process.stderr is not None:
            self.stderr_task = asyncio.create_task(process.stderr.read())

    async def request(self, operation: str, payload: dict[str, Any], timeout: float) -> Any:
        if self.process.returncode is not None:
            raise WorkerProtocolError(
                f"Phase-1 worker exited before {operation}: {self.process.returncode}"
            )
        if self.process.stdin is None or self.process.stdout is None:
            raise WorkerProtocolError("Phase-1 worker pipes are unavailable")
        self.sequence += 1
        output = self.root / f"response-{self.sequence:04d}.pt"
        payload_path = self.root / f"request-{self.sequence:04d}.pt"
        torch.save(payload, payload_path)
        command = {
            "operation": operation,
            "output": str(output),
            "payload_path": str(payload_path),
            "sequence": self.sequence,
        }
        self.process.stdin.write(
            (json.dumps(command, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        await self.process.stdin.drain()
        try:
            raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Phase-1 worker timed out during {operation}") from None
        if not raw:
            raise WorkerProtocolError(f"Phase-1 worker closed stdout during {operation}")
        response = json.loads(raw)
        if response.get("sequence") != self.sequence:
            raise WorkerProtocolError("Phase-1 worker response sequence mismatch")
        if not response.get("ok"):
            traceback_text = str(response.get("traceback", "")).strip()
            raise WorkerProtocolError(
                f"Phase-1 worker {operation} failed: "
                f"{response.get('error', 'unknown error')}\n{traceback_text}"
            )
        if not output.is_file():
            raise WorkerProtocolError(f"Phase-1 worker omitted payload for {operation}")
        return torch.load(output, map_location="cpu", weights_only=False)

    async def close(self, reason: str, *, force: bool = False) -> None:
        def kill_process_group() -> None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                if self.process.returncode is None:
                    self.process.kill()

        if self.process.returncode is None:
            if force:
                kill_process_group()
            else:
                try:
                    await self.request("shutdown", {"reason": reason}, 20.0)
                except Exception:
                    kill_process_group()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30.0)
            except TimeoutError:
                kill_process_group()
                await self.process.wait()
        # The worker owns a new process session.  Remove any engine-core
        # grandchildren even if the Python worker exited before closing its
        # inherited pipes.
        kill_process_group()
        if self.stderr_task is not None:
            try:
                stderr = await asyncio.wait_for(self.stderr_task, timeout=5.0)
            except TimeoutError:
                self.stderr_task.cancel()
                try:
                    await self.stderr_task
                except asyncio.CancelledError:
                    pass
                stderr = b""
            if stderr:
                (self.root / "worker-stderr.log").write_bytes(stderr)


class SubprocessWorkerBackend:
    """Spawn qualified worker processes and exchange tensor payloads by file."""

    def __init__(self, config: RealExecutorConfig, runtime_identity: dict[str, JsonValue]) -> None:
        self.config = config.validated()
        self.worker_config = self.config.canonical() | runtime_identity
        parent = self.config.work_root
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(
            tempfile.mkdtemp(prefix="step5-phase1-real-", dir=None if parent is None else parent)
        )
        self.prefix_rows_path: Path | None = None
        self.prefix_payload_sha256: str | None = None

    def _environment(self, trace_dir: Path) -> dict[str, str]:
        semantic_prefixes = ("EA_", "S2S_", "VLLM_", "VOICECHAT_")
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith(semantic_prefixes)
        }
        environment.update(SEALED_WORKER_ENVIRONMENT)
        environment.update(
            {
                "NEMOTRON_WEDGE_TRACE": "1",
                "NEMOTRON_WEDGE_TRACE_CAPACITY": "4096",
                "NEMOTRON_WEDGE_TRACE_DIR": str(trace_dir),
                "NEMOTRON_WEDGE_TRACE_RUN_UUID": trace_dir.name,
                "NEMOTRON_WEDGE_TRACE_PROCESS_ROLE": "phase1-worker",
            }
        )
        return environment

    async def _spawn(self, label: str, mode: str) -> _SubprocessEpoch:
        root = self.root / label
        trace = root / "boundary-trace"
        trace.mkdir(parents=True, exist_ok=False)
        config_path = root / "worker-config.json"
        config_path.write_text(
            json.dumps(self.worker_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            self.config.python_executable,
            str(WORKER_SOURCE),
            "--serve",
            "--config",
            str(config_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment(trace),
            cwd=REPO_ROOT,
            start_new_session=True,
        )
        epoch = _SubprocessEpoch(process, root)
        construction = await epoch.request(
            "construct", {"mode": mode}, self.config.construction_timeout_seconds
        )
        if not isinstance(construction, dict):
            await epoch.close("invalid-construction-receipt", force=True)
            raise WorkerProtocolError("worker returned an invalid construction receipt")
        epoch.construction = construction
        return epoch

    async def materialize(self, fixture: dict[str, JsonValue]) -> MaterializedFixture:
        epoch = await self._spawn("materializer", "materializer")
        try:
            result = await epoch.request(
                "materialize", {"fixture": fixture}, self.config.construction_timeout_seconds
            )
            if not isinstance(result, dict) or not isinstance(
                result.get("materialized"), MaterializedFixture
            ):
                raise WorkerProtocolError("materializer returned the wrong payload type")
            prefix = Path(str(result.get("prefix_rows_path", "")))
            if not prefix.is_file():
                raise WorkerProtocolError("materializer omitted retained prefix rows")
            claimed_prefix_sha256 = result.get("prefix_payload_sha256")
            if (
                not isinstance(claimed_prefix_sha256, str)
                or sha256_file(prefix) != claimed_prefix_sha256
            ):
                raise WorkerProtocolError("materializer retained prefix payload is unsealed")
            self.prefix_rows_path = prefix
            self.prefix_payload_sha256 = claimed_prefix_sha256
            return result["materialized"]
        finally:
            await epoch.close("materialization-complete")

    async def create_epoch(self, replicate: int) -> EpochClient:
        if self.prefix_rows_path is None:
            raise WorkerProtocolError("cannot create an epoch before fixture materialization")
        return await self._spawn(f"replicate-{replicate:02d}", "epoch")


class RealPhase1Executor(Phase1Executor):
    """Concrete real-GPU implementation of the sealed Phase1Executor API."""

    supports_lane_local_dispatch = True

    def __init__(
        self,
        config: RealExecutorConfig,
        *,
        backend_factory: Callable[..., WorkerBackend] = SubprocessWorkerBackend,
    ) -> None:
        self.config = config.validated()
        patched_inventory = _tree_inventory(self.config.patched_vllm_root)
        patched_tree_sha256 = _inventory_digest(patched_inventory)
        model_identity = _verify_model_artifacts(self.config)
        if model_identity["model_sha256"] != self.config.model_sha256:
            raise ValueError("used model artifacts differ from the frozen model identity")
        if self.config.used_image_sha256 is None:
            # Unit-test backends do not run a container. Production construction
            # is bound by executor_from_environment below.
            used_image_sha256 = self.config.image_sha256
        else:
            used_image_sha256 = self.config.used_image_sha256
        configuration_sha256 = (
            sha256_file(self.config.selected_config_path)
            if self.config.selected_config_path is not None
            else stable_digest(self.config.canonical())
        )
        self._runtime_identity: dict[str, JsonValue] = {
            "configuration_sha256": configuration_sha256,
            "executor_source_sha256": sha256_file(Path(__file__)),
            "image_sha256": used_image_sha256,
            "model_artifact_identity": model_identity,
            "model_sha256": str(model_identity["model_sha256"]),
            "patch_manifest_sha256": sha256_file(self.config.patch_manifest),
            "patched_runner_sha256": sha256_file(
                self.config.patched_vllm_root / "v1/worker/gpu_model_runner.py"
            ),
            "patched_vllm_file_count": len(patched_inventory),
            "patched_vllm_tree_sha256": patched_tree_sha256,
            "runtime_optimizations_source_sha256": sha256_file(RUNTIME_OPTIMIZATIONS_SOURCE),
            "pocket_worker_source_sha256": sha256_file(POCKET_WORKER_SOURCE),
            "typed_carrier_source_sha256": sha256_file(TYPED_CARRIER_SOURCE),
            "server_source_sha256": sha256_file(SERVER_SOURCE),
            "worker_environment": dict(SEALED_WORKER_ENVIRONMENT),
            "worker_source_sha256": sha256_file(WORKER_SOURCE),
            "worker_environment_sha256": stable_digest(SEALED_WORKER_ENVIRONMENT),
        }
        self.backend = backend_factory(config, self._runtime_identity)
        self._materialized: MaterializedFixture | None = None
        self._materialization_sha256: str | None = None

    async def _ensure_materialized(
        self, fixture: dict[str, JsonValue] | None = None
    ) -> MaterializedFixture:
        if self._materialized is None:
            frozen = load_frozen_fixture() if fixture is None else fixture
            self._materialized = await self.backend.materialize(frozen)
            seal = validate_materialized_fixture(self._materialized, frozen)
            self._materialization_sha256 = str(seal["fixture_materialization_sha256"])
        return self._materialized

    def _materialize_synchronously(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._ensure_materialized())
            return
        raise RuntimeError(
            "sealed_identity cannot first materialize from an active event loop; "
            "call materialize_fixture before requesting identity"
        )

    def sealed_identity(self) -> dict[str, Any]:
        if self._materialization_sha256 is None:
            self._materialize_synchronously()
        return {
            "configuration_sha256": self._runtime_identity["configuration_sha256"],
            "executor_source_sha256": self._runtime_identity["executor_source_sha256"],
            "fixture_materialization_sha256": self._materialization_sha256,
            "image_sha256": self._runtime_identity["image_sha256"],
            "kind": "real-gpu-process-isolated",
            "model_artifact_identity_sha256": stable_digest(
                self._runtime_identity["model_artifact_identity"]
            ),
            "model_sha256": self._runtime_identity["model_sha256"],
            "patch_manifest_sha256": self._runtime_identity["patch_manifest_sha256"],
            "patched_runner_sha256": self._runtime_identity["patched_runner_sha256"],
            "patched_vllm_file_count": self._runtime_identity["patched_vllm_file_count"],
            "patched_vllm_tree_sha256": self._runtime_identity["patched_vllm_tree_sha256"],
            "runtime_optimizations_source_sha256": self._runtime_identity[
                "runtime_optimizations_source_sha256"
            ],
            "pocket_worker_source_sha256": self._runtime_identity[
                "pocket_worker_source_sha256"
            ],
            "typed_carrier_source_sha256": self._runtime_identity[
                "typed_carrier_source_sha256"
            ],
            "schema": 4,
            "server_source_sha256": self._runtime_identity["server_source_sha256"],
            "worker_source_sha256": self._runtime_identity["worker_source_sha256"],
            "worker_environment_sha256": self._runtime_identity["worker_environment_sha256"],
        }

    async def materialize_fixture(self, fixture: dict[str, JsonValue]) -> MaterializedFixture:
        return await self._ensure_materialized(fixture)

    async def create_epoch(self, replicate: int) -> dict[str, Any]:
        client = await self.backend.create_epoch(replicate)
        return {"client": client, "replicate": replicate}

    async def restore_lane(
        self, epoch: Any, lane: str, fixture: dict[str, Any], request_id: str
    ) -> str:
        materialized = await self._ensure_materialized(fixture)
        result = await epoch["client"].request(
            "restore_lane",
            {
                "fixture_sha256": fixture["fixture_sha256"],
                "lane": lane,
                "materialized": materialized,
                "prefix_rows_path": str(getattr(self.backend, "prefix_rows_path", "") or ""),
                "prefix_payload_sha256": str(
                    getattr(self.backend, "prefix_payload_sha256", "") or ""
                ),
                "request_id": request_id,
            },
            self.config.rpc_timeout_seconds,
        )
        if not isinstance(result, str):
            raise WorkerProtocolError("restore_lane returned a non-string prefix identity")
        return result

    async def execute_case(
        self,
        epoch: Any,
        lane: str,
        case: CaseSpec,
        plan: OraclePlan,
        execution: str,
        request_id: str,
    ) -> ExecutionResult:
        timeout = (
            self.config.terminal_timeout_seconds
            if lane == "A" and case.causal_terminal
            else self.config.rpc_timeout_seconds
        )
        try:
            result = await epoch["client"].request(
                "execute_case",
                {
                    "case": case,
                    "execution": execution,
                    "lane": lane,
                    "plan": plan,
                    "request_id": request_id,
                },
                timeout,
            )
        except TimeoutError:
            try:
                boundaries, dispatch_mode, dispatch_evidence = await self._terminal_trace(
                    epoch, request_id
                )
            except Exception as trace_error:
                if lane != "A" or not case.causal_terminal:
                    raise TimeoutError(
                        f"nonterminal Phase-1 timeout without complete trace: "
                        f"{lane}/{case.name}: {trace_error}"
                    ) from None
                raise
            if lane != "A" or not case.causal_terminal:
                raise TimeoutError(
                    "nonterminal Phase-1 timeout: "
                    f"{lane}/{case.name}; boundaries={boundaries!r}; "
                    f"dispatch={dispatch_mode!r}; evidence={dispatch_evidence!r}"
                ) from None
            return ExecutionResult(
                snapshot=None,
                boundaries=boundaries,
                completed=False,
                terminal_unknown=True,
                dispatch_mode=dispatch_mode,
                prefix_sha256=load_frozen_fixture()["fixture_sha256"],
                observed_plan=plan,
                dispatch_evidence=dispatch_evidence,
            )
        if not isinstance(result, ExecutionResult):
            raise WorkerProtocolError("execute_case returned the wrong payload type")
        return result

    async def _terminal_boundaries(self, epoch: Any, request_id: str) -> tuple[str, ...]:
        client = epoch["client"]
        trace_dir = getattr(client, "root", Path()) / "boundary-trace"
        from step5_phase1_gpu_worker import read_boundary_prefix

        return read_boundary_prefix(trace_dir, request_id)

    async def _terminal_trace(
        self, epoch: Any, request_id: str
    ) -> tuple[tuple[str, ...], str, dict[str, Any]]:
        client = epoch["client"]
        trace_dir = getattr(client, "root", Path()) / "boundary-trace"
        from step5_phase1_gpu_worker import observed_dispatch, read_boundary_prefix

        dispatch_mode, evidence = observed_dispatch(trace_dir, request_id)
        return read_boundary_prefix(trace_dir, request_id), dispatch_mode, evidence

    async def destroy_epoch(self, epoch: Any, reason: str) -> None:
        await epoch["client"].close(reason)


def executor_from_environment() -> RealPhase1Executor:
    path_text = os.environ.get(CONFIG_ENV, "").strip()
    if not path_text:
        raise RuntimeError(f"{CONFIG_ENV} must name the sealed executor JSON configuration")
    path = _require_file(Path(path_text), "real executor configuration")
    if path.read_bytes() != FROZEN_CONFIG_PATH.read_bytes():
        raise ValueError("selected executor configuration differs from the frozen repository file")
    image_sha256 = os.environ.get(USED_IMAGE_ENV, "").removeprefix("sha256:").strip()
    if len(image_sha256) != 64:
        raise RuntimeError(
            f"{USED_IMAGE_ENV} must be derived from the container image selected by the launcher"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("real executor configuration must be a JSON object")
    config = replace(
        RealExecutorConfig.from_mapping(value),
        selected_config_path=path,
        used_image_sha256=image_sha256,
    )
    return RealPhase1Executor(config)


def wait_for_gpu_idle(timeout_seconds: float, poll_seconds: float = 2.0) -> None:
    """Fail closed unless NVML reports no active compute process."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not probe.stdout.strip():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU did not become idle before the Phase-1 smoke deadline")
        time.sleep(poll_seconds)


async def run_single_engine_smoke(executor: RealPhase1Executor, output: Path) -> dict[str, Any]:
    """Run real PCM materialization and all S/B/A structure in one worker.

    The receipt is explicitly non-campaign evidence.  Reusing the materializer's
    Nano engine keeps this authorization to one runtime construction while still
    exercising real prefix restoration, pair-eager selection, controls, abort,
    and terminal worker destruction.
    """
    from step5_phase1_campaign import LANE_ORDER, _result_summary, _run_lane
    from step5_phase1_oracle import compare_snapshots

    if not isinstance(executor.backend, SubprocessWorkerBackend):
        raise TypeError("single-engine smoke requires the subprocess GPU backend")
    fixture = load_frozen_fixture()
    client = await executor.backend._spawn("single-engine-smoke", "materializer")
    lane_results: dict[str, dict[str, ExecutionResult]] = {}
    exactness_errors: list[str] = []
    cross_lane_observations: dict[str, dict[str, Any]] = {}
    terminal_unknown = False
    destroy_reason = "single-engine-smoke-complete"
    materialization_seal: dict[str, Any] | None = None
    try:
        response = await client.request(
            "materialize",
            {"fixture": fixture},
            executor.config.construction_timeout_seconds,
        )
        if not isinstance(response, dict) or not isinstance(
            response.get("materialized"), MaterializedFixture
        ):
            raise WorkerProtocolError("single-engine materializer returned invalid data")
        prefix = Path(str(response.get("prefix_rows_path", "")))
        prefix_sha256 = response.get("prefix_payload_sha256")
        if (
            not prefix.is_file()
            or not isinstance(prefix_sha256, str)
            or sha256_file(prefix) != prefix_sha256
        ):
            raise WorkerProtocolError("single-engine materializer prefix is unsealed")
        materialized = response["materialized"]
        materialization_seal = validate_materialized_fixture(materialized, fixture)
        executor.backend.prefix_rows_path = prefix
        executor.backend.prefix_payload_sha256 = prefix_sha256
        executor._materialized = materialized
        executor._materialization_sha256 = str(
            materialization_seal["fixture_materialization_sha256"]
        )
        epoch = {"client": client, "replicate": 0}
        for lane in LANE_ORDER:
            oracle_results = None if lane == "S" else lane_results.get("S")
            results, errors = await _run_lane(
                executor,
                epoch,
                lane,
                fixture,
                0,
                oracle_results,
            )
            lane_results[lane] = results
            exactness_errors.extend(errors)
            if lane != "S" and "S" in lane_results:
                comparisons: dict[str, Any] = {}
                for case_name, result in results.items():
                    baseline = lane_results["S"].get(case_name)
                    if (
                        result.snapshot is None
                        or baseline is None
                        or baseline.snapshot is None
                    ):
                        comparisons[case_name] = {
                            "comparable": False,
                            "differences": [],
                            "exact": False,
                        }
                        continue
                    comparison = compare_snapshots(baseline.snapshot, result.snapshot)
                    comparisons[case_name] = {
                        "comparable": True,
                        "differences": [
                            {
                                "classification": "genuine_semantic_divergence",
                                "path": item.path,
                                "reason": item.reason,
                            }
                            for item in comparison.differences
                        ],
                        "exact": comparison.exact,
                    }
                cross_lane_observations[lane] = comparisons
            if errors:
                destroy_reason = "single-engine-smoke-exactness-failure"
                break
            if any(result.terminal_unknown for result in results.values()):
                terminal_unknown = True
                destroy_reason = "single-engine-smoke-terminal-destruction"
                break
    except BaseException:
        destroy_reason = "single-engine-smoke-exception"
        raise
    finally:
        await client.close(destroy_reason, force=terminal_unknown)
    payload: dict[str, Any] = {
        "campaign_admissible": False,
        "construction": client.construction,
        "cross_lane_observations": cross_lane_observations,
        "destroy_reason": destroy_reason,
        "exactness_errors": exactness_errors,
        "executor_identity": executor.sealed_identity(),
        "fixture_materialization": materialization_seal,
        "gpu_memory_utilization": executor.config.gpu_memory_utilization,
        "kind": "phase1-real-pcm-single-engine-structural-smoke",
        "lanes": {
            lane: {name: _result_summary(result) for name, result in results.items()}
            for lane, results in lane_results.items()
        },
        "real_runtime_constructions": 1,
        "comparison_contract": {
            "exact_fields": [
                "emitted text/function/conjunction",
                "accepted count and selected packed/source row",
                "selected custom outputs",
                "committed logical KV tokens",
                "committed Mamba conv/SSM",
                "transaction canonical rollback boundary",
                "request-relative shadow/virtual-engine identities",
                "host pending rows and correction state",
            ],
            "retained_not_cross_compared": [
                "lane-absolute shared-control-owner frames/state",
                "physical shadow locations and raw inactive shadow leaves",
                "transition/provenance/pending/whole-state digests",
                "position-one and uncommitted-position whole-state digests",
            ],
        },
        "round3_difference_audit": {
            "function_logits": (
                "selected committed output; any corrected-comparator difference is genuine"
            ),
            "kv": (
                "unused physical block tail was a representation artifact; any difference "
                "within the committed-token projection is genuine"
            ),
            "mamba_conv_ssm": (
                "already lane-invariant committed state; any remaining difference is genuine"
            ),
            "rollback_shadow": (
                "raw layout/history was a representation artifact; any difference in the "
                "canonical transaction rollback projection is genuine"
            ),
            "shared_control_and_whole_state_hashes": (
                "lane-absolute retained evidence, excluded from cross-lane equality"
            ),
        },
        "schema": 3,
        "terminal_worker_destroyed": client.process.returncode is not None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if exactness_errors or not payload["terminal_worker_destroyed"]:
        raise RuntimeError(
            "single-engine structural smoke failed closed after retaining receipt: "
            + json.dumps(
                {
                    "exactness_errors": exactness_errors,
                    "receipt": str(output),
                    "terminal_worker_destroyed": payload["terminal_worker_destroyed"],
                },
                sort_keys=True,
            )
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-engine-smoke-output", type=Path, required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    wait_for_gpu_idle(args.idle_timeout_seconds)
    executor = executor_from_environment()
    asyncio.run(run_single_engine_smoke(executor, args.single_engine_smoke_output))


if __name__ == "__main__":
    main()
