#!/usr/bin/env python3
"""Minimal persistent WebSocket server for NVIDIA's VoiceChat source pipeline.

This implements the strict version-3 protocol used by the project-owned Pipecat
service. One active stream is supported because the qualified DGX Spark
configuration has batch size one and one set of recurrent model state. Only the
public combined Hugging Face checkpoint is supported.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import functools
import gc
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pocket_controller import (
    PocketSynthesisCancelled,
    PocketWorkerError,
    PocketWorkerManager,
)
from .protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RealtimeProtocolSession,
    protocol_capabilities,
)
from .provenance import RUNTIME_SOURCE_PATHS

PINNED_COMMIT = "911ec674ab40f04302ef33672be4179f45a7310f"
PRODUCTION_ENGINE_TYPE = "vllm_llm_vllm_eartts"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
FRAME_SECONDS = 0.08
FRAME_SAMPLES = int(INPUT_SAMPLE_RATE * FRAME_SECONDS)
MAX_INPUT_MESSAGE_BYTES = FRAME_SAMPLES * 2 * 16
MAX_TYPED_INPUT_CHARS = 1_000
FUNCTION_CALL_TIMEOUT_SECONDS = 30.0
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
TOOL_GROUNDING_INSTRUCTIONS = (
    "Never invent or imply a tool result without calling the tool. If you say you "
    "will use a tool, emit that tool call before ending the turn. After a tool "
    "response arrives, answer the user and explicitly include the exact returned "
    "result."
)
PUBLIC_VLLM_MANIFEST_KINDS = frozenset(
    {
        "exact_public_vllm_quantized_nano_eartts",
        "nemotron_voicechat_dgx_spark_hf_release",
    }
)
LOGGER = logging.getLogger(__name__)


def typed_input_trailing_silence_seconds(
    *, rnnt_eou_frames: int, margin: float, injection_frame_ms: int = 20
) -> float:
    """Return EOU-derived silence rounded to an 80 ms model-frame boundary."""

    raw = rnnt_eou_frames * FRAME_SECONDS * margin + injection_frame_ms / 1_000
    return math.ceil(raw / FRAME_SECONDS) * FRAME_SECONDS


class RealtimeProtocolError(Exception):
    """A client-visible protocol error with explicit connection semantics."""

    def __init__(self, code: str, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.code = code
        self.fatal = fatal


class SessionTerminated(Exception):
    """Internal control flow after a terminal session event was emitted."""


@dataclass
class ActiveTypedInput:
    job_id: str
    text: str
    cancel_event: asyncio.Event
    done: asyncio.Event
    audio_started: bool = False
    replacement_requested: bool = False


def parse_nvidia_driver_version(version_text: str) -> str | None:
    """Extract the host NVIDIA driver version from /proc's NVRM banner."""
    match = re.search(r"\b(\d{3,4}(?:\.\d+){1,3})\s+Release Build\b", version_text)
    return match.group(1) if match else None


def host_runtime_provenance() -> dict[str, str | None]:
    """Return host facts that materially affect CUDA runtime reproducibility."""
    driver_version = None
    nvrm_version = Path("/proc/driver/nvidia/version")
    try:
        driver_version = parse_nvidia_driver_version(nvrm_version.read_text(encoding="utf-8"))
    except OSError:
        pass
    if driver_version is None:
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            driver_version = next(
                (line.strip() for line in output.splitlines() if line.strip()), None
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "driver_version": driver_version,
        "kernel_version": platform.release(),
    }


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment switch with an explicit default."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be a boolean, got {value!r}")


def websocket_ping_interval() -> float | None:
    """Return the Uvicorn server-ping interval, allowing an explicit disable."""
    value = os.environ.get("VOICECHAT_WEB_WS_PING_INTERVAL", "20").strip().lower()
    if value in {"none", "off", "disabled"}:
        return None
    interval = float(value)
    if interval <= 0:
        raise SystemExit("VOICECHAT_WEB_WS_PING_INTERVAL must be positive or 'none'")
    return interval


def websocket_ping_timeout() -> float | None:
    """Disable ping timeouts whenever server-originated pings are disabled."""
    if websocket_ping_interval() is None:
        return None
    value = os.environ.get("VOICECHAT_WEB_WS_PING_TIMEOUT", "20").strip().lower()
    if value in {"none", "off", "disabled"}:
        return None
    timeout = float(value)
    if timeout <= 0:
        raise SystemExit("VOICECHAT_WEB_WS_PING_TIMEOUT must be positive or 'none'")
    return timeout


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifested_model(root: Path, component: dict) -> dict:
    """Validate a monolithic or sharded vLLM model declared by its manifest."""
    declared_path = component.get("path")
    if declared_path and Path(declared_path).resolve() != root.resolve():
        raise SystemExit(f"Manifest component path mismatch: {declared_path} != {root}")

    validated_files: dict[str, dict[str, Any]] = {}
    for name, expected in sorted((component.get("files") or {}).items()):
        if not name or Path(name).name != name:
            raise SystemExit(f"Unsafe manifested artifact filename: {name!r}")
        path = root / name
        if not path.is_file():
            raise SystemExit(f"Missing manifested artifact file: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != {
            "bytes": expected.get("bytes"),
            "sha256": expected.get("sha256"),
        }:
            raise SystemExit(f"Manifest mismatch for artifact file: {path}")
        validated_files[name] = actual

    model = component.get("model") or {}
    declared_files = model.get("files")
    if declared_files:
        digest = hashlib.sha256()
        total_bytes = 0
        for item in sorted(declared_files, key=lambda value: value["name"]):
            name = item.get("name")
            if not name or Path(name).name != name:
                raise SystemExit(f"Unsafe manifested model filename: {name!r}")
            path = root / name
            if not path.is_file():
                raise SystemExit(f"Missing manifested model shard: {path}")
            cached = validated_files.get(name)
            actual_bytes = cached["bytes"] if cached else path.stat().st_size
            actual_sha = cached["sha256"] if cached else sha256_file(path)
            if actual_bytes != item.get("bytes") or actual_sha != item.get("sha256"):
                raise SystemExit(f"Manifest mismatch for model shard: {path}")
            total_bytes += actual_bytes
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(actual_bytes).encode("ascii"))
            digest.update(b"\0")
            digest.update(actual_sha.encode("ascii"))
            digest.update(b"\n")
        actual_sha = digest.hexdigest()
        if total_bytes != model.get("bytes") or actual_sha != model.get("sha256"):
            raise SystemExit(f"Composite model manifest mismatch: {root}")
        return {
            "bytes": total_bytes,
            "sha256": actual_sha,
            "files": declared_files,
            "artifact_files": validated_files or None,
        }

    model_file = root / "model.safetensors"
    expected_sha = model.get("sha256")
    if not expected_sha or not model_file.is_file():
        raise SystemExit(f"Incomplete public vLLM artifact: {root}")
    cached = validated_files.get(model_file.name)
    actual_sha = cached["sha256"] if cached else sha256_file(model_file)
    if actual_sha != expected_sha:
        raise SystemExit(f"Public vLLM SHA mismatch: {actual_sha} != {expected_sha}")
    return {
        "bytes": model_file.stat().st_size,
        "sha256": actual_sha,
        "artifact_files": validated_files or None,
    }


def release_component_manifest(release: dict[str, Any], name: str) -> dict[str, Any]:
    """Adapt the signed Hugging Face release inventory to component validation."""

    prefix = f"{name}/"
    files = {
        item["path"].removeprefix(prefix): {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in release.get("files", [])
        if item.get("path", "").startswith(prefix)
    }
    shards = [
        {"name": filename, **details}
        for filename, details in files.items()
        if filename.startswith("model-") and filename.endswith(".safetensors")
    ]
    composite = release.get(f"{name}_composite_sha256")
    if not files or not shards or not composite:
        raise SystemExit(f"Signed release has no complete {name} component")
    return {
        "files": files,
        "model": {
            "files": shards,
            "bytes": sum(item["bytes"] for item in shards),
            "sha256": composite,
        },
        "reproducibility": {"runs": 2, "byte_identical": True},
        "quantization": {"release": release.get("candidate")},
    }


def validate_nano_runtime_optimization(nano_root: Path, model_cfg: Any) -> dict | None:
    """Fail closed when a Nano artifact omits the custom text-logit head.

    vLLM still evaluates its ordinary language-model head to choose the token.
    Omitting the second, custom copy is equivalent only while the surrounding
    NVIDIA wrapper accepts that greedy token without applying another sampler.
    """
    config_path = nano_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    optimization = config.get("voicechat_skip_custom_text_logits") or {}
    if not optimization.get("enabled"):
        return None

    custom_outputs = list(config.get("custom_outputs") or [])
    if "text_logits" in custom_outputs:
        raise SystemExit("Nano skip-custom-text-logits artifact still declares text_logits")
    required_outputs = {"function_tokens", "function_logits"}
    if not required_outputs <= set(custom_outputs):
        raise SystemExit("Nano skip-custom-text-logits artifact must retain both function outputs")

    actual = {
        "temperature": float(model_cfg.get("temperature", 1.0)),
        "top_p": float(model_cfg.get("top_p", 1.0)),
        "repetition_penalty": float(model_cfg.get("repetition_penalty", 1.0)),
    }
    expected = {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}
    if actual != expected:
        raise SystemExit(
            "Nano skip-custom-text-logits requires greedy wrapper sampling: "
            f"expected {expected}, got {actual}"
        )
    return {
        "skip_custom_text_logits": True,
        "sampling_contract": actual,
        "retained_custom_outputs": custom_outputs,
    }


def validate_nano_pad_pair_runtime(
    nano_root: Path, manifest: dict[str, Any], requested: bool
) -> dict[str, Any]:
    """Fail closed unless the live artifact declares the qualified pair contract."""
    config = json.loads((nano_root / "config.json").read_text(encoding="utf-8"))
    contract = config.get("voicechat_pad_pair_contract") or {}
    result = {
        "enabled": bool(requested),
        "conditional_previous_effective_pad": env_bool(
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL", False
        ),
        "pad_token_id": config.get("voicechat_pad_pair_token_id"),
        "mamba_chunk_size": config.get("chunk_size"),
        "vllm_max_model_len": int(os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")),
        "streaming_max_len": int(os.environ.get("VOICECHAT_STREAMING_MAX_LEN", "12288")),
        "max_session_model_frames": int(
            os.environ.get("VOICECHAT_WEB_MAX_SESSION_MODEL_FRAMES", "12000")
        ),
        "contract": contract,
    }
    if not requested:
        return result
    required = {
        "scope": "production live runtime",
        "max_positions": 2,
        "production_mamba_rollback": True,
        "function_channel_rejection": True,
        "post_fc_fusion_correction": True,
        "mamba_chunk_size": 1,
    }
    actual = {key: contract.get(key) for key in required}
    if actual != required or config.get("voicechat_pad_pair_token_id") is None:
        raise SystemExit(f"Nano PAD-pair production contract mismatch: {actual} != {required}")
    if manifest.get("kind") == "nemotron_voicechat_dgx_spark_hf_release":
        if manifest.get("candidate") != "production-candidate-1":
            raise SystemExit("Unsupported signed VoiceChat artifact candidate")
        release_contract = manifest.get("runtime_contract") or {}
        expected_release_contract = {
            "nano_pad_pair": True,
            "nano_pad_pair_conditional": True,
            "nano_pad_pair_control_barrier": True,
            "vllm_max_model_len": 12288,
            "streaming_max_len": 12288,
            "max_session_model_frames": 12000,
            "agent_no_text_frames": 30,
            "agent_no_audio_frames": 30,
            "agent_silence_eos_frames": 20,
        }
        actual_release_contract = {
            key: release_contract.get(key) for key in expected_release_contract
        }
        if actual_release_contract != expected_release_contract:
            raise SystemExit("Signed release runtime contract mismatch")
        if not result["conditional_previous_effective_pad"]:
            raise SystemExit("Production Candidate 1 requires conditional PAD drafting")
        for key in ("vllm_max_model_len", "streaming_max_len", "max_session_model_frames"):
            if result[key] != expected_release_contract[key]:
                raise SystemExit(f"Production Candidate 1 {key} mismatch")
        result["production_candidate"] = {
            "number": 1,
            "status": "signed_release",
            "release_sha256": manifest.get("release_sha256"),
        }
        return result
    manifest_contract = manifest.get("runtime_optimization") or {}
    required_manifest = {
        "pad_pair_production": True,
        "production_mamba_rollback": True,
        "function_channel_rejection": True,
        "post_fc_fusion_correction": True,
    }
    if any(manifest_contract.get(key) is not value for key, value in required_manifest.items()):
        raise SystemExit("Nano PAD-pair manifest lacks the production contract")
    candidate = manifest.get("production_candidate")
    if candidate is not None:
        if candidate.get("number") != 1 or candidate.get("status") != "frozen":
            raise SystemExit("Unsupported or unfrozen VoiceChat production candidate")
        if not result["conditional_previous_effective_pad"]:
            raise SystemExit("Production Candidate 1 requires conditional PAD drafting")
        conditional_contract = {
            "conditional_previous_effective_pad": True,
            "function_async_sequential_latch": True,
            "function_async_pending_drain": True,
            "agent_no_text_frames": 30,
            "agent_no_audio_frames": 30,
            "agent_decoded_silence_frames": 20,
            "vllm_max_model_len": 12288,
            "streaming_max_len": 12288,
            "max_session_model_frames": 12000,
            "production_candidate": 1,
        }
        actual_conditional = {key: contract.get(key) for key in conditional_contract}
        if actual_conditional != conditional_contract:
            raise SystemExit(
                "Production Candidate 1 config contract mismatch: "
                f"{actual_conditional} != {conditional_contract}"
            )
        if result["vllm_max_model_len"] != conditional_contract["vllm_max_model_len"]:
            raise SystemExit(
                "Production Candidate 1 max-model-length mismatch: "
                f"{result['vllm_max_model_len']} != "
                f"{conditional_contract['vllm_max_model_len']}"
            )
        if result["streaming_max_len"] != conditional_contract["streaming_max_len"]:
            raise SystemExit(
                "Production Candidate 1 streaming-buffer-length mismatch: "
                f"{result['streaming_max_len']} != "
                f"{conditional_contract['streaming_max_len']}"
            )
        if result["max_session_model_frames"] != conditional_contract["max_session_model_frames"]:
            raise SystemExit(
                "Production Candidate 1 session-frame guard mismatch: "
                f"{result['max_session_model_frames']} != "
                f"{conditional_contract['max_session_model_frames']}"
            )
        if any(manifest_contract.get(key) != value for key, value in conditional_contract.items()):
            raise SystemExit("Production Candidate 1 manifest contract mismatch")
        source_root = Path(__file__).resolve().parents[2]
        runtime_sources = candidate.get("runtime_sources") or {}
        required_sources = set(RUNTIME_SOURCE_PATHS)
        if set(runtime_sources) != required_sources:
            raise SystemExit("Production Candidate 1 runtime source set mismatch")
        for name, expected in runtime_sources.items():
            path = source_root / name
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                raise SystemExit(f"Production Candidate 1 source size mismatch: {name}")
            actual_sha = sha256_file(path)
            if actual_sha != expected["sha256"]:
                raise SystemExit(
                    f"Production Candidate 1 source SHA mismatch for {name}: "
                    f"{actual_sha} != {expected['sha256']}"
                )
        result["production_candidate"] = {
            "number": 1,
            "status": "frozen",
            "runtime_image": candidate.get("runtime_image"),
            "runtime_image_id": candidate.get("runtime_image_id"),
            "runtime_sources": runtime_sources,
        }
    return result


def pcm16_bytes_to_float32(payload: bytes) -> np.ndarray:
    """Decode little-endian mono PCM16 to normalized float32."""
    if len(payload) % 2:
        raise ValueError("PCM16 payload has an odd byte count")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Encode normalized audio as clipped little-endian PCM16."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


def clean_display_text(value: str) -> str:
    """Remove model control/timestamp markers without changing human text."""
    value = re.sub(r"</?s>|<SPECIAL_12>|<\$[\d.]+\$>|<\|[\d.]+\|>", " ", value or "")
    return " ".join(value.lstrip("^").replace("\u0120", " ").split())


def normalize_tool_definitions(value: Any) -> list[dict[str, Any]]:
    """Validate OpenAI- or NVIDIA-style function schemas for one session."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("session.tools must be an array")
    if len(value) > 32:
        raise ValueError("session.tools supports at most 32 functions")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"session.tools[{index}] must be an object")
        function = item.get("function", item)
        if not isinstance(function, dict):
            raise ValueError(f"session.tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not TOOL_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"session.tools[{index}] has an invalid function name")
        registry_name = name.replace("-", "_")
        if registry_name in seen:
            raise ValueError(f"duplicate function name after normalization: {name}")
        seen.add(registry_name)
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ValueError(f"session.tools[{index}].parameters must be an object")
        normalized.append(
            {
                "name": name,
                "description": str(function.get("description", "")),
                "parameters": parameters,
            }
        )
    return normalized


def render_tool_system_prompt(
    template_path: Path, system_message: str, tools: list[dict[str, Any]]
) -> str:
    """Render NVIDIA's function prompt plus the live grounding requirements."""
    if not tools:
        return system_message
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
        render_fc_system_prompt,
    )

    rendered = render_fc_system_prompt(str(template_path), system_message, tools)
    return f"{rendered}\n\n{TOOL_GROUNDING_INSTRUCTIONS}"


def coerce_tool_arguments(arguments: Any, schema: Any) -> Any:
    """Conservatively restore JSON scalar types declared by a tool schema.

    NVIDIA's public function head sometimes spells numeric arguments as JSON
    strings.  The external transport must still honor the schema advertised by
    the client.  Only exact, lossless scalar spellings are coerced; all other
    values are preserved for the tool implementation to validate.
    """
    if not isinstance(schema, dict):
        return arguments
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(arguments, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return arguments
        return {
            key: coerce_tool_arguments(value, properties.get(key))
            for key, value in arguments.items()
        }
    if schema_type == "array" and isinstance(arguments, list):
        item_schema = schema.get("items")
        return [coerce_tool_arguments(value, item_schema) for value in arguments]
    if not isinstance(arguments, str):
        return arguments
    stripped = arguments.strip()
    if schema_type == "integer" and re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", stripped):
        return int(stripped)
    if schema_type == "number" and re.fullmatch(
        r"[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        stripped,
    ):
        value = float(stripped)
        return value if math.isfinite(value) else arguments
    if schema_type == "boolean" and stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    return arguments


@dataclass
class PendingExternalToolCall:
    call_id: str
    name: str
    arguments: str
    completed: threading.Event
    output: str | None = None


class ExternalToolBridge:
    """Bridge NVIDIA's blocking tool hook to two-phase WebSocket events."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        emit: Any,
        *,
        emit_timeout: Any | None = None,
        timeout_seconds: float = FUNCTION_CALL_TIMEOUT_SECONDS,
    ):
        self.loop = loop
        self.emit = emit
        self.emit_timeout = emit_timeout
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, PendingExternalToolCall] = {}
        self._closed = False

    @staticmethod
    def _arguments_json(arguments: Any) -> str:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"value": arguments}
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        return json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"))

    def handlers(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        handlers: dict[str, Any] = {}
        for tool in tools:
            advertised_name = tool["name"]
            parameters = tool.get("parameters", {})

            def handler(
                arguments: Any,
                *,
                _name: str = advertised_name,
                _parameters: Any = parameters,
            ) -> str:
                return self._invoke(_name, coerce_tool_arguments(arguments, _parameters))

            handlers[advertised_name.replace("-", "_")] = handler
        return handlers

    def _invoke(self, name: str, arguments: Any) -> str:
        call = PendingExternalToolCall(
            call_id=f"call_{uuid.uuid4().hex}",
            name=name,
            arguments=self._arguments_json(arguments),
            completed=threading.Event(),
        )
        with self._lock:
            if self._closed:
                return json.dumps({"error": "session_closed"})
            self._pending[call.call_id] = call
        future = asyncio.run_coroutine_threadsafe(self.emit(call), self.loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:
            with self._lock:
                self._pending.pop(call.call_id, None)
            return json.dumps({"error": f"function_call_event_failed: {exc}"})
        if not call.completed.wait(self.timeout_seconds):
            with self._lock:
                self._pending.pop(call.call_id, None)
            if self.emit_timeout is not None:
                timeout_future = asyncio.run_coroutine_threadsafe(
                    self.emit_timeout(call), self.loop
                )
                try:
                    timeout_future.result(timeout=5.0)
                except Exception:
                    pass
            return json.dumps({"error": "function_call_output_timeout"})
        with self._lock:
            self._pending.pop(call.call_id, None)
        return call.output if call.output is not None else json.dumps({"error": "empty_output"})

    def submit(self, call_id: str, output: Any) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("function_call_output requires call_id")
        with self._lock:
            call = self._pending.get(call_id)
            if call is None:
                raise ValueError(f"unknown or completed call_id: {call_id}")
            if call.completed.is_set():
                raise ValueError(f"duplicate function output for call_id: {call_id}")
            call.output = output if isinstance(output, str) else json.dumps(output)
            call.completed.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._pending.values())
        for call in pending:
            call.output = json.dumps({"error": "session_closed"})
            call.completed.set()


def audio_stats(samples: np.ndarray) -> dict[str, float | int]:
    """Return compact signal diagnostics for one PCM payload or model frame."""
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size:
        return {"samples": 0, "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0}
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    return {
        "samples": int(values.size),
        "rms": round(rms, 8),
        "rms_dbfs": round(max(-120.0, 20.0 * math.log10(max(rms, 1e-6))), 3),
        "peak": round(float(np.max(np.abs(values))), 8),
    }


def response_audio_is_deliverable(turn_state: dict[str, Any]) -> bool:
    """Expose decoded audio only while a tokenized assistant turn is open."""
    boundary = turn_state.get("response_boundary")
    if not isinstance(boundary, dict):
        return False
    return (
        boundary.get("phase") in {"responding", "tail_draining"} or boundary.get("event") == "start"
    )


class SessionTrace:
    """Lossless application-payload capture plus structured streaming events."""

    def __init__(self, root: Path | None, session_id: str):
        self.session_id = session_id
        self.directory: Path | None = None
        self._events = None
        self._input_pcm = None
        self._output_pcm = None
        if root is not None:
            self.directory = root / session_id
            self.directory.mkdir(parents=True, exist_ok=False)
            self._events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
            self._input_pcm = (self.directory / "input-16000-mono-s16le.pcm").open("wb")
            self._output_pcm = (self.directory / "output-22050-mono-s16le.pcm").open("wb")
        self.event(
            "session_trace_started",
            capture_directory=str(self.directory) if self.directory else None,
        )

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "trace": "voicechat",
            "event": kind,
            "session_id": self.session_id,
            "monotonic_s": round(time.monotonic(), 6),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        if self._events is not None:
            self._events.write(line + "\n")
            self._events.flush()

    def input_pcm(self, payload: bytes) -> None:
        if self._input_pcm is not None:
            self._input_pcm.write(payload)
            self._input_pcm.flush()

    def output_pcm(self, payload: bytes) -> None:
        if self._output_pcm is not None:
            self._output_pcm.write(payload)
            self._output_pcm.flush()

    def close(self, reason: str) -> None:
        self.event("session_trace_stopped", reason=reason)
        for stream in (self._events, self._input_pcm, self._output_pcm):
            if stream is not None:
                stream.close()
        self._events = None
        self._input_pcm = None
        self._output_pcm = None


class PcmFrameBuffer:
    """Reassemble arbitrary PCM packets into exact 80 ms model frames."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES):
        self.frame_samples = frame_samples
        self._pending = np.empty(0, dtype=np.float32)

    def push(self, payload: bytes) -> list[np.ndarray]:
        decoded = pcm16_bytes_to_float32(payload)
        if decoded.size:
            self._pending = np.concatenate((self._pending, decoded))
        frames: list[np.ndarray] = []
        while self._pending.size >= self.frame_samples:
            frames.append(self._pending[: self.frame_samples].copy())
            self._pending = self._pending[self.frame_samples :]
        return frames

    def flush(self) -> np.ndarray:
        """Return one zero-padded final frame, including for an empty buffer."""
        frame = np.zeros(self.frame_samples, dtype=np.float32)
        if self._pending.size:
            frame[: self._pending.size] = self._pending
        self._pending = np.empty(0, dtype=np.float32)
        return frame

    def discard(self) -> int:
        """Discard a partial packet boundary and return its sample count."""

        samples = int(self._pending.size)
        self._pending = np.empty(0, dtype=np.float32)
        return samples


@dataclass
class StepResult:
    audio: np.ndarray
    user_text: str
    assistant_delta: str
    assistant_text: str
    inference_ms: float
    server_step_ms: float
    frame_index: int
    input_audio: dict[str, float | int]
    output_audio: dict[str, float | int]
    turn_state: dict[str, Any]
    function_delta: str = ""
    function_text: str = ""
    trace_diagnostics: dict[str, Any] | None = None


@dataclass
class AgentSilenceEosWatchdog:
    """Request model-state EOS when an opened answer stops making progress."""

    threshold_dbfs: float = -90.0
    required_frames: int = 12
    no_text_required_frames: int = 30
    no_audio_required_frames: int = 30
    agent_open: bool = False
    text_seen: bool = False
    function_seen: bool = False
    audible_seen: bool = False
    silent_frames: int = 0
    no_text_frames: int = 0
    no_audio_frames: int = 0
    request_pending: bool = False
    request_reason: str | None = None

    def reset(self) -> None:
        self.agent_open = False
        self.text_seen = False
        self.function_seen = False
        self.audible_seen = False
        self.silent_frames = 0
        self.no_text_frames = 0
        self.no_audio_frames = 0
        self.request_pending = False
        self.request_reason = None

    def observe(
        self,
        control: str | None,
        text_delta: str,
        output_dbfs: float,
        *,
        function_delta: str = "",
    ) -> bool:
        if control == "agent_bos" and not self.agent_open:
            self.agent_open = True
            self.text_seen = False
            self.function_seen = False
            self.audible_seen = False
            self.silent_frames = 0
            self.no_text_frames = 0
            self.no_audio_frames = 0
            self.request_pending = False
            self.request_reason = None

        if not self.agent_open:
            return False

        if text_delta:
            self.text_seen = True
        if function_delta:
            self.function_seen = True
        if output_dbfs > self.threshold_dbfs:
            self.audible_seen = True
        if self.function_seen or self.text_seen:
            self.no_text_frames = 0
        else:
            self.no_text_frames += 1
        if self.function_seen or self.audible_seen:
            self.no_audio_frames = 0
        else:
            self.no_audio_frames += 1
        if self.audible_seen and not text_delta and output_dbfs <= self.threshold_dbfs:
            self.silent_frames += 1
        else:
            self.silent_frames = 0

        no_text_stalled = (
            self.no_text_required_frames > 0 and self.no_text_frames >= self.no_text_required_frames
        )
        no_audio_stalled = (
            self.no_audio_required_frames > 0
            and self.no_audio_frames >= self.no_audio_required_frames
        )
        decoded_silence = self.required_frames > 0 and self.silent_frames >= self.required_frames
        should_request = (
            no_text_stalled or no_audio_stalled or decoded_silence
        ) and not self.request_pending
        if should_request:
            self.request_pending = True
            if no_text_stalled:
                self.request_reason = "no_text_since_bos_watchdog"
            elif no_audio_stalled:
                self.request_reason = "no_audio_since_bos_watchdog"
            else:
                self.request_reason = "decoded_silence_watchdog"

        if control == "agent_eos":
            self.agent_open = False
            self.text_seen = False
            self.function_seen = False
            self.audible_seen = False
            self.silent_frames = 0
            self.no_text_frames = 0
            self.no_audio_frames = 0
            self.request_pending = False
            self.request_reason = None
            return False
        return should_request

    def snapshot(self) -> dict[str, Any]:
        return {
            "threshold_dbfs": self.threshold_dbfs,
            "required_frames": self.required_frames,
            "no_text_required_frames": self.no_text_required_frames,
            "no_audio_required_frames": self.no_audio_required_frames,
            "agent_open": self.agent_open,
            "text_seen": self.text_seen,
            "function_seen": self.function_seen,
            "audible_seen": self.audible_seen,
            "silent_frames": self.silent_frames,
            "no_text_frames": self.no_text_frames,
            "no_audio_frames": self.no_audio_frames,
            "request_pending": self.request_pending,
            "request_reason": self.request_reason,
        }


@dataclass
class TransportModelGate:
    """Gate only leading silence; preserve NVIDIA's continuous stream thereafter."""

    threshold_dbfs: float = -50.0
    continuous_after_speech: bool = True
    min_active_frames: int = 1
    speech_started: bool = False
    consecutive_active_frames: int = 0
    input_active: bool = False

    def reset(self) -> None:
        self.speech_started = False
        self.consecutive_active_frames = 0
        self.input_active = False

    def should_advance(self, input_dbfs: float) -> bool:
        above_threshold = input_dbfs >= self.threshold_dbfs
        if above_threshold:
            self.consecutive_active_frames += 1
        else:
            self.consecutive_active_frames = 0
        self.input_active = self.consecutive_active_frames >= max(1, self.min_active_frames)
        if self.input_active:
            self.speech_started = True
        return self.speech_started

    def observe_response_end(self) -> None:
        """Rearm only after the decoded-audio boundary has fully closed."""
        if not self.continuous_after_speech:
            self.speech_started = False

    def snapshot(self, input_dbfs: float, advanced: bool) -> dict[str, Any]:
        return {
            "continuous_after_speech": self.continuous_after_speech,
            "speech_started": self.speech_started,
            "input_active": self.input_active,
            "above_threshold": input_dbfs >= self.threshold_dbfs,
            "consecutive_active_frames": self.consecutive_active_frames,
            "min_active_frames": self.min_active_frames,
            "model_advanced": advanced,
            "speech_gate_dbfs": self.threshold_dbfs,
        }


@dataclass
class AcousticResponseBoundary:
    """Close assistant turns on decoded-audio silence after text EOS."""

    threshold_dbfs: float = -90.0
    required_silent_frames: int = 3
    max_tail_frames: int = 25
    delivery_silence_frames: int = 12
    open: bool = False
    text_seen: bool = False
    acoustic_complete: bool = False
    bos_frame: int | None = None
    eos_frame: int | None = None
    eos_reason: str | None = None
    silent_frames: int = 0
    post_eos_frames: int = 0

    def reset(self) -> None:
        self.open = False
        self.text_seen = False
        self.acoustic_complete = False
        self.bos_frame = None
        self.eos_frame = None
        self.eos_reason = None
        self.silent_frames = 0
        self.post_eos_frames = 0

    def observe(
        self,
        control: str | None,
        output_dbfs: float,
        frame_index: int,
        *,
        text_delta: str = "",
        watchdog_eos_pending: bool = False,
        watchdog_eos_reason: str | None = None,
        max_response_frames: int = 0,
    ) -> dict[str, Any]:
        event: str | None = None
        boundary_reason: str | None = None
        tail_timed_out = False

        if control == "agent_bos" and not self.open:
            self.open = True
            self.text_seen = False
            self.acoustic_complete = False
            self.bos_frame = frame_index
            self.eos_frame = None
            self.eos_reason = None
            self.silent_frames = 0
            self.post_eos_frames = 0
            event = "start"

        if self.open:
            if text_delta:
                self.text_seen = True
            if output_dbfs <= self.threshold_dbfs:
                self.silent_frames += 1
            else:
                self.silent_frames = 0
                # Delivery silence is a reversible acoustic observation, not a
                # model-state boundary. Resume delivery if speech returns after
                # a long prosodic pause before text EOS.
                if self.eos_frame is None:
                    self.acoustic_complete = False

            if (
                self.eos_frame is None
                and self.text_seen
                and self.delivery_silence_frames > 0
                and self.silent_frames >= self.delivery_silence_frames
            ):
                self.acoustic_complete = True

            if control == "agent_eos" and self.eos_frame is None:
                self.eos_frame = frame_index
                self.post_eos_frames = 0
                if watchdog_eos_pending:
                    self.eos_reason = watchdog_eos_reason or "decoded_silence_watchdog"
                elif (
                    self.bos_frame is not None
                    and max_response_frames > 0
                    and frame_index - self.bos_frame >= max_response_frames
                ):
                    self.eos_reason = "max_response_cap"
                else:
                    self.eos_reason = "model_or_turn_taking"
            elif self.eos_frame is not None and frame_index > self.eos_frame:
                self.post_eos_frames += 1

            enough_silence = (
                self.eos_frame is not None
                and self.required_silent_frames > 0
                and self.silent_frames >= self.required_silent_frames
            )
            tail_deadline = (
                self.eos_frame is not None
                and self.max_tail_frames > 0
                and self.post_eos_frames >= self.max_tail_frames
            )
            if enough_silence or tail_deadline:
                event = "end"
                tail_timed_out = tail_deadline and not enough_silence
                boundary_reason = (
                    "post_eos_audio_silence" if enough_silence else "post_eos_tail_deadline"
                )
                snapshot = {
                    "event": event,
                    "phase": "complete",
                    "bos_frame": self.bos_frame,
                    "text_eos_frame": self.eos_frame,
                    "eos_reason": self.eos_reason,
                    "boundary_reason": boundary_reason,
                    "silent_frames": self.silent_frames,
                    "post_eos_frames": self.post_eos_frames,
                    "tail_timed_out": tail_timed_out,
                    "acoustic_complete": self.acoustic_complete,
                    "delivery_silence_frames": self.delivery_silence_frames,
                    "threshold_dbfs": self.threshold_dbfs,
                    "required_silent_frames": self.required_silent_frames,
                    "max_tail_frames": self.max_tail_frames,
                }
                self.reset()
                return snapshot

        phase = "idle"
        if self.open:
            if self.eos_frame is not None:
                phase = "tail_draining"
            elif self.acoustic_complete:
                phase = "internal_drain"
            else:
                phase = "responding"
        return {
            "event": event,
            "phase": phase,
            "bos_frame": self.bos_frame,
            "text_eos_frame": self.eos_frame,
            "eos_reason": self.eos_reason,
            "boundary_reason": boundary_reason,
            "silent_frames": self.silent_frames,
            "post_eos_frames": self.post_eos_frames,
            "tail_timed_out": tail_timed_out,
            "acoustic_complete": self.acoustic_complete,
            "delivery_silence_frames": self.delivery_silence_frames,
            "threshold_dbfs": self.threshold_dbfs,
            "required_silent_frames": self.required_silent_frames,
            "max_tail_frames": self.max_tail_frames,
        }


def clear_stale_agent_eos_latch(pipeline: Any, stream_id: int) -> bool:
    """Consume an EOS edge before an unrelated future FC worker can see it.

    NVIDIA's non-blocking function loop consumes this boolean only while a
    worker is active. Without this guard an ordinary turn ending remains
    latched and kills the next worker immediately after SOTC.
    """
    active = getattr(pipeline, "_fc_async_bg", {})
    if stream_id in active:
        return False
    wrapper = pipeline.s2s_model
    if not getattr(wrapper, "_agent_eos_just_fired", False):
        return False
    wrapper._agent_eos_just_fired = False
    return True


class VoiceChatEngine:
    """Small stateful adapter around NVIDIA's incremental pipeline API."""

    def __init__(
        self,
        pipeline: Any,
        *,
        agent_silence_eos_dbfs: float = -90.0,
        agent_silence_eos_frames: int = 12,
        agent_no_text_frames: int = 30,
        agent_no_audio_frames: int = 30,
        response_tail_dbfs: float = -60.0,
        response_tail_silence_frames: int = 3,
        response_tail_max_frames: int = 25,
        delivery_silence_frames: int = 12,
    ):
        self.pipeline = pipeline
        self.stream_id = 0
        self.frame_index = 0
        self.started = False
        self.system_prompt = ""
        self.assistant_position = 0
        self.function_position = 0
        self.user_text = ""
        self.last_rnnt_decoded_count = 0
        self.vllm_request_position_baseline: dict[str, int] = {}
        self.agent_silence_watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=agent_silence_eos_dbfs,
            required_frames=agent_silence_eos_frames,
            no_text_required_frames=agent_no_text_frames,
            no_audio_required_frames=agent_no_audio_frames,
        )
        self.response_boundary = AcousticResponseBoundary(
            threshold_dbfs=response_tail_dbfs,
            required_silent_frames=response_tail_silence_frames,
            max_tail_frames=response_tail_max_frames,
            delivery_silence_frames=delivery_silence_frames,
        )
        self.pad_pair_idle_no_buffer_frames = 0
        self.pad_pair_watchdog_emitted = False
        self.pad_pair_trace_errors = 0

    def _reset_wrapper_session_state(self) -> None:
        reset_fn = getattr(self.pipeline.s2s_model, "reset_transport_session_state", None)
        if not callable(reset_fn):
            raise RuntimeError(
                "inference wrapper has no reset_transport_session_state(); refusing "
                "to risk carrying wrapper-global turn state across sessions"
            )
        reset_fn()

    @staticmethod
    def _first_scalar(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().reshape(-1)
            return value[0].item() if value.numel() else None
        if isinstance(value, (list, tuple)):
            return VoiceChatEngine._first_scalar(value[0]) if value else None
        if isinstance(value, (bool, int, float, str)):
            return value
        return None

    def _turn_state(self, context: Any, frame_position: int) -> dict[str, Any]:
        if context is None or frame_position < 0:
            return {}
        stt = self.pipeline.s2s_model.model.stt_model
        agent_token_id = int(context.gen_text[0, frame_position].item())
        asr_token_id = int(context.gen_asr_text[0, frame_position].item())
        control_ids = {
            "pad": int(stt.text_pad_id),
            "agent_bos": int(stt.text_bos_id),
            "agent_eos": int(stt.text_eos_id),
            "user_bos": int(stt.user_bos_id),
        }

        def label(token_id: int, *, asr: bool = False) -> str | None:
            if token_id == control_ids["pad"]:
                return "pad"
            if asr and token_id == control_ids["user_bos"]:
                return "user_bos"
            if token_id == control_ids["agent_bos"]:
                return "agent_bos"
            if token_id == control_ids["agent_eos"]:
                return "agent_eos"
            return None

        result: dict[str, Any] = {
            "agent_token_id": agent_token_id,
            "agent_control": label(agent_token_id),
            "asr_token_id": asr_token_id,
            "asr_control": label(asr_token_id, asr=True),
            "control_ids": control_ids,
        }
        rnnt = getattr(context, "rnnt_partial_hypotheses", None)
        if isinstance(rnnt, dict):
            result["rnnt"] = {
                key: self._first_scalar(rnnt.get(key))
                for key in (
                    "blank_count",
                    "nonblank_consec",
                    "nonblank_total",
                    "speech_confirmed",
                    "agent_speaking",
                    "first_turn",
                    "user_first_turn",
                    "rolling_density",
                    "forced_bos",
                    "post_eos_fired",
                )
                if key in rnnt
            }
            result["rnnt"]["decoded_token_count"] = len(rnnt.get("y_sequence", []))
            for source_key, result_key in (
                ("_turn_text_tokens", "turn_text_tokens"),
                ("_agent_talking_frames", "agent_talking_frames"),
            ):
                counter = rnnt.get(source_key)
                if isinstance(counter, dict):
                    result["rnnt"][result_key] = self._first_scalar(counter.get(0))
        fc_state = getattr(context, "fc_state", None)
        if isinstance(fc_state, dict):
            result["function_calling"] = {
                "active": bool(fc_state.get("active", False)),
                "awaiting_response": bool(fc_state.get("awaiting_response", False)),
                "injecting_response": bool(fc_state.get("injecting_response", False)),
                "forced_tokens": len(fc_state.get("forced_function_tokens") or []),
                "completed_calls": len(fc_state.get("completed_calls") or []),
                "background_active": self.stream_id in getattr(self.pipeline, "_fc_async_bg", {}),
                "agent_eos_edge_pending": bool(
                    getattr(self.pipeline.s2s_model, "_agent_eos_just_fired", False)
                ),
            }
        perception_cache = getattr(context, "perception_cache", None)
        if perception_cache is not None:
            cache_channel = getattr(perception_cache, "cache_last_channel", None)
            cache_time = getattr(perception_cache, "cache_last_time", None)
            cache_len = getattr(perception_cache, "cache_last_channel_len", None)
            initialized = getattr(perception_cache, "is_initialized", None)
            result["perception_cache"] = {
                "initialized": bool(initialized()) if callable(initialized) else None,
                "cache_last_channel_shape": (
                    list(cache_channel.shape) if cache_channel is not None else None
                ),
                "cache_last_time_shape": (
                    list(cache_time.shape) if cache_time is not None else None
                ),
                "cache_last_channel_len": self._first_scalar(cache_len),
            }
        return result

    def _vllm_request_positions(self) -> dict[str, Any]:
        """Read live request lengths without synchronizing or touching tensors."""
        request_id_fn = getattr(self.pipeline, "_request_id_for_stream", None)
        request_id = (
            str(request_id_fn(self.stream_id)) if callable(request_id_fn) else str(self.stream_id)
        )
        wrapper = self.pipeline.s2s_model
        interfaces = {
            "nano": getattr(wrapper, "model_llm_interface", None),
            "eartts": getattr(
                getattr(getattr(wrapper, "model", None), "tts_model", None),
                "tts_model",
                None,
            ),
        }
        result: dict[str, Any] = {"request_id": request_id}
        for name, interface in interfaces.items():
            engine = getattr(interface, "engine", None)
            requests = getattr(engine, "requests", None)
            state = requests.get(request_id) if hasattr(requests, "get") else None
            generated = getattr(state, "generated_tokens", None)
            count = len(generated) if generated is not None else None
            baseline = self.vllm_request_position_baseline.get(name)
            result[name] = {
                "generated_tokens": count,
                "prefill_baseline": baseline,
                "session_positions": (
                    count - baseline if count is not None and baseline is not None else None
                ),
            }
        return result

    def _pad_pair_trace_diagnostics(
        self,
        context: Any,
        frame_position: int,
        turn_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Collect opt-in evidence without allowing diagnostics to break inference."""
        from .runtime_optimizations import enabled

        if not enabled("VOICECHAT_NANO_PAD_PAIR_TRACE"):
            return None
        try:
            return self._collect_pad_pair_trace_diagnostics(
                context, frame_position, turn_state
            )
        except Exception as exc:
            self.pad_pair_trace_errors += 1
            LOGGER.exception("Dropped Nano PAD-pair model-step diagnostics")
            return {
                "trace_error": type(exc).__name__,
                "trace_record_errors": self.pad_pair_trace_errors,
            }

    def _collect_pad_pair_trace_diagnostics(
        self,
        context: Any,
        frame_position: int,
        turn_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the enabled scalar snapshot; caller owns exception isolation."""
        from .runtime_optimizations import consume_pad_pair_trace

        request_id_fn = getattr(self.pipeline, "_request_id_for_stream", None)
        request_id = (
            str(request_id_fn(self.stream_id)) if callable(request_id_fn) else str(self.stream_id)
        )
        snapshot = consume_pad_pair_trace(self.pipeline.s2s_model, request_id)
        finalized = snapshot.get("last_effective_tokens")
        effective: dict[str, Any] | None = None
        if context is not None and frame_position >= 0:
            gen_function = getattr(context, "gen_function_text", None)
            if gen_function is not None:
                text_token = int(context.gen_text[0, frame_position].item())
                function_token = int(gen_function[0, frame_position].item())
                scheduler_pad = (finalized or {}).get("pad")
                pad_token = (
                    int(scheduler_pad)
                    if scheduler_pad is not None
                    else int(self.pipeline.s2s_model.model.stt_model.text_pad_id)
                )
                effective = {
                    "frame_idx": frame_position,
                    "text": text_token,
                    "function": function_token,
                    "pad": pad_token,
                    "text_is_pad": text_token == pad_token,
                    "function_is_pad": function_token == pad_token,
                    "both_pad": text_token == pad_token and function_token == pad_token,
                }

        fc = turn_state.get("function_calling") or {}
        fc_idle = not any(
            (
                fc.get("active"),
                fc.get("awaiting_response"),
                fc.get("injecting_response"),
                fc.get("forced_tokens"),
                fc.get("background_active"),
            )
        )
        buffered = any(event.get("decision") == "buffered" for event in snapshot["events"])
        watching = bool(
            snapshot.get("enabled") and fc_idle and turn_state.get("agent_control") == "pad"
        )
        watchdog = None
        if watching and not buffered:
            self.pad_pair_idle_no_buffer_frames += 1
            if self.pad_pair_idle_no_buffer_frames >= 25 and not self.pad_pair_watchdog_emitted:
                self.pad_pair_watchdog_emitted = True
                watchdog = {
                    "event": "idle_agent_without_pair_buffer",
                    "consecutive_frames": self.pad_pair_idle_no_buffer_frames,
                    "threshold_frames": 25,
                    "effective_tokens": effective,
                    "scheduler": snapshot.get("state"),
                }
        else:
            self.pad_pair_idle_no_buffer_frames = 0
            self.pad_pair_watchdog_emitted = False
        effective_matches_finalize = None
        if effective is not None and finalized is not None:
            effective_matches_finalize = all(
                effective.get(key) == finalized.get(key)
                for key in ("frame_idx", "text", "function", "pad", "both_pad")
            )
        return {
            "effective_tokens": effective,
            "effective_tokens_match_finalize": effective_matches_finalize,
            "pad_pair": snapshot,
            "idle_no_buffer_frames": self.pad_pair_idle_no_buffer_frames,
            "watchdog": watchdog,
        }

    def start(self, system_prompt: str = "") -> None:
        if self.started:
            raise RuntimeError("session already started")
        self._reset_wrapper_session_state()
        self.stream_id += 1
        self.frame_index = 0
        self.assistant_position = 0
        self.function_position = 0
        self.user_text = ""
        self.last_rnnt_decoded_count = 0
        self.vllm_request_position_baseline = {}
        self.agent_silence_watchdog.reset()
        self.response_boundary.reset()
        self.pad_pair_idle_no_buffer_frames = 0
        self.pad_pair_watchdog_emitted = False
        self.pad_pair_trace_errors = 0
        self.pipeline.open_session()
        self.system_prompt = system_prompt.strip()
        # Prime the two vLLM request streams without consuming an audio frame.
        # In particular, EarTTS must receive its prepared speaker/prompt inputs
        # before infer_one_step appends the first text/acoustic frame.  Starting
        # EarTTS from that first frame omits the 37-token Aria prompt and yields
        # a different acoustic trajectory from frame zero.  Use NVIDIA's
        # dedicated non-audio API rather than a synthetic zero-length Frame so
        # perception/RNNT frame zero remains the first microphone frame.
        prefill = getattr(self.pipeline, "prefill_for_new_stream", None)
        if not callable(prefill):
            raise RuntimeError(
                "streaming pipeline lacks prefill_for_new_stream(); refusing "
                "to start an unprimed EarTTS request"
            )
        prefill(self.stream_id, self.system_prompt)
        prefill_positions = self._vllm_request_positions()
        for name in ("nano", "eartts"):
            count = prefill_positions[name]["generated_tokens"]
            if count is not None:
                self.vllm_request_position_baseline[name] = count
        self.started = True

    def process(
        self,
        samples: np.ndarray,
        *,
        is_last: bool = False,
        input_active: bool | None = None,
    ) -> StepResult:
        if not self.started:
            self.start()
        if samples.shape != (FRAME_SAMPLES,):
            raise ValueError(f"expected {FRAME_SAMPLES} samples, received {samples.shape}")

        import torch
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import (
            S2SRequestOptions,
        )

        if input_active is not None:
            observe_activity = getattr(
                self.pipeline.s2s_model, "observe_transport_user_activity", None
            )
            if not callable(observe_activity):
                raise RuntimeError(
                    "inference wrapper has no observe_transport_user_activity(); "
                    "refusing to run without the short-utterance EOU backstop"
                )
            observe_activity(input_active, stream_id=self.stream_id)

        first_frame = self.frame_index == 0
        frame = Frame(
            samples=torch.from_numpy(np.asarray(samples, dtype=np.float32)),
            stream_id=self.stream_id,
            is_first=first_frame,
            is_last=is_last,
            options=(S2SRequestOptions(system_prompt=self.system_prompt) if first_frame else None),
        )
        started_at = time.perf_counter()
        clear_stale_agent_eos_latch(self.pipeline, self.stream_id)
        self.pipeline.generate_step([frame])
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        state = self.pipeline.get_or_create_state(self.stream_id)
        if is_last:
            # Exact-public fixed-stage optimizations are opt-in. Draining here
            # is required by the WebSocket step API, which does not call the
            # file-oriented StreamingS2SPipeline finalizer.
            from .runtime_optimizations import finalize_public_fixed_stages

            finalize_public_fixed_stages(self.pipeline.s2s_model, self.stream_id, state)
        audio_tensor = state.audio_buffer.detach().to(torch.float32).cpu()
        audio = audio_tensor.reshape(-1).numpy().copy()
        assistant_text = clean_display_text(state.get_output_text())
        function_text = state.get_output_function_text()
        rnnt_display = clean_display_text(state.get_output_asr_text())
        # Keep ASR-head decoding as a fallback for checkpoints with no RNNT.
        token_data = state.get_token_tensors() if is_last else None
        context = None
        manager = self.pipeline.context_manager
        slot = manager.streamidx2slotidx.get(self.stream_id)
        if slot is not None:
            context = manager.slot_contexts[slot]
        if rnnt_display:
            self.user_text = rnnt_display
        if token_data is not None:
            _, asr_tokens, total_frames, _ = token_data
        else:
            asr_tokens = None
            total_frames = 0
            if context is not None:
                asr_tokens = context.gen_asr_text
                total_frames = context.frame_idx
        if (
            getattr(self.pipeline, "ui_asr_source", "asr_head") != "rnnt"
            and not rnnt_display
            and asr_tokens is not None
            and total_frames > 0
        ):
            from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str

            lengths = torch.tensor([total_frames], dtype=torch.long)
            decoded = tokens_to_str(
                asr_tokens[:, :total_frames],
                lengths,
                tokenizer=self.pipeline.s2s_model.tokenizer,
                pad_id=self.pipeline.s2s_model.model.stt_model.text_pad_id,
                eval_text_turn_taking=False,
            )[0]
            displayed = clean_display_text(decoded)
            if displayed:
                self.user_text = displayed
        user_text = self.user_text
        assistant_delta = assistant_text[self.assistant_position :]
        self.assistant_position = len(assistant_text)
        function_delta = function_text[self.function_position :]
        self.function_position = len(function_text)
        state.cleanup_after_response()
        server_step_ms = (time.perf_counter() - started_at) * 1000.0

        turn_state = self._turn_state(context, total_frames - 1)
        turn_state["vllm_request_positions"] = self._vllm_request_positions()
        trace_diagnostics = self._pad_pair_trace_diagnostics(
            context, total_frames - 1, turn_state
        )
        rnnt_state = turn_state.get("rnnt")
        if isinstance(rnnt_state, dict):
            decoded_count = int(rnnt_state.get("decoded_token_count") or 0)
            rnnt_state["emitted_this_frame"] = max(0, decoded_count - self.last_rnnt_decoded_count)
            self.last_rnnt_decoded_count = decoded_count
        result = StepResult(
            audio=audio,
            user_text=user_text,
            assistant_delta=assistant_delta,
            assistant_text=assistant_text,
            inference_ms=inference_ms,
            server_step_ms=server_step_ms,
            frame_index=self.frame_index,
            input_audio=audio_stats(samples),
            output_audio=audio_stats(audio),
            turn_state=turn_state,
            function_delta=function_delta,
            function_text=function_text,
            trace_diagnostics=trace_diagnostics,
        )
        control = result.turn_state.get("agent_control")
        rnnt_metrics = result.turn_state.get("rnnt", {})
        text_tokens = int(rnnt_metrics.get("turn_text_tokens") or 0)
        talking_frames = int(rnnt_metrics.get("agent_talking_frames") or 0)
        tts_ratio_cap = float(
            self.pipeline.s2s_model.model_cfg.get("tts_text_token_ratio_cap", 16.0)
        )
        tts_token_min = int(self.pipeline.s2s_model.model_cfg.get("tts_text_token_min", 5))
        if isinstance(rnnt_metrics, dict):
            rnnt_metrics["tts_ratio_cap"] = tts_ratio_cap
            rnnt_metrics["tts_token_min"] = tts_token_min
            rnnt_metrics["tts_ratio_guard_armed"] = bool(
                tts_ratio_cap > 0 and text_tokens >= tts_token_min
            )
            rnnt_metrics["tts_ratio_remaining_frames"] = (
                max(0.0, tts_ratio_cap * text_tokens - talking_frames)
                if tts_ratio_cap > 0 and text_tokens >= tts_token_min
                else None
            )
        watchdog_eos_pending = self.agent_silence_watchdog.request_pending
        watchdog_eos_reason = self.agent_silence_watchdog.request_reason
        request_eos = self.agent_silence_watchdog.observe(
            control,
            result.assistant_delta,
            float(result.output_audio["rms_dbfs"]),
            function_delta=result.function_delta,
        )
        result.turn_state["agent_silence_watchdog"] = self.agent_silence_watchdog.snapshot()
        result.turn_state["response_boundary"] = self.response_boundary.observe(
            control,
            float(result.output_audio["rms_dbfs"]),
            self.frame_index,
            text_delta=result.assistant_delta,
            watchdog_eos_pending=watchdog_eos_pending,
            watchdog_eos_reason=watchdog_eos_reason,
            max_response_frames=int(
                getattr(self.pipeline.s2s_model, "_max_agent_response_frames", 0)
            ),
        )
        if request_eos:
            request_fn = getattr(self.pipeline.s2s_model, "request_agent_eos", None)
            if not callable(request_fn):
                raise RuntimeError(
                    "agent silence watchdog fired but the inference wrapper has no "
                    "request_agent_eos()"
                )
            request_fn()
        self.frame_index += 1
        if is_last:
            self.pipeline.delete_state(self.stream_id)
            self._reset_wrapper_session_state()
            self.started = False
        return result

    def abort(self) -> None:
        if self.started:
            # Drive NVIDIA's normal end-of-stream path so ordered local codec
            # pipelines drain/close instead of retaining one pending frame into
            # the next WebSocket session. The result is intentionally discarded.
            try:
                self.process(
                    np.zeros(FRAME_SAMPLES, dtype=np.float32),
                    is_last=True,
                    input_active=False,
                )
            except Exception:
                self.pipeline.reset_session()
        self._reset_wrapper_session_state()
        self.started = False
        self.system_prompt = ""
        self.agent_silence_watchdog.reset()
        self.response_boundary.reset()
        self.last_rnnt_decoded_count = 0

    def configure_external_tools(self, handlers: dict[str, Any]) -> None:
        """Authorize exactly the functions advertised by the active client."""
        if self.started:
            raise RuntimeError("tools must be configured before session start")
        self.pipeline.tool_registry = dict(handlers)


def checked_speech_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    entrypoint = root / "examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py"
    if not entrypoint.is_file():
        raise SystemExit(f"Missing upstream streaming entrypoint: {entrypoint}")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"Speech checkout is {commit}; expected {PINNED_COMMIT}")
    reviewed_paths = [
        ("nemo/collections/speechlm2/inference/model_wrappers/model_factory.py"),
        (
            "nemo/collections/speechlm2/inference/model_wrappers/"
            "nemotron_voicechat_inference_wrapper.py"
        ),
        "nemo/collections/speechlm2/inference/vllm/streaming_llm_engine.py",
    ]
    retained_patch = (
        Path(__file__).resolve().parent / "patches/nemotron-voicechat-rnnt-turn-taking.patch"
    )
    if not retained_patch.is_file():
        raise SystemExit(f"Missing retained Speech source patch: {retained_patch}")
    actual_patch = subprocess.check_output(["git", "-C", str(root), "diff", "--", *reviewed_paths])
    expected_patch = retained_patch.read_bytes()
    if actual_patch != expected_patch:
        raise SystemExit(
            "Speech runtime working-tree diff does not match the retained, "
            f"reviewed patch: {retained_patch}"
        )
    return root


def attach_separate_rnnt_runtime(pipeline: Any, checkpoint: Path) -> dict[str, Any]:
    """Attach decoder/joint omitted by the old separate-RNNT source path.

    The pinned source restores the separate ASR checkpoint to initialize the
    perception encoder, but deliberately discards its decoder and joint. The
    realtime wrapper looks only for underscore-prefixed aliases, so RNNT turn
    detection is otherwise silently disabled even when configured as `rnnt`.
    """
    wrapper = pipeline.s2s_model
    stt = wrapper.model.stt_model
    if getattr(stt, "_rnnt_decoder", None) is not None:
        return {"attached": False, "reason": "already_available"}
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Separate RNNT checkpoint is missing: {checkpoint}")

    import nemo.collections.asr as nemo_asr
    import torch

    started_at = time.perf_counter()
    restored = nemo_asr.models.ASRModel.restore_from(str(checkpoint), map_location="cpu")
    decoder = restored.decoder.to(device=wrapper.device, dtype=wrapper.dtype).eval()
    joint = restored.joint.to(device=wrapper.device, dtype=wrapper.dtype).eval()
    tokenizer = restored.tokenizer
    vocabulary = list(getattr(joint, "vocabulary", None) or [])
    blank_id = getattr(getattr(restored, "decoding", None), "blank_id", None)
    if blank_id is None:
        blank_id = getattr(decoder, "blank_idx", len(vocabulary))

    stt.rnnt_decoder = decoder
    stt.rnnt_joint = joint
    stt.rnnt_tokenizer = tokenizer
    object.__setattr__(stt, "_rnnt_decoder", decoder)
    object.__setattr__(stt, "_rnnt_joint", joint)
    object.__setattr__(stt, "_rnnt_blank_id", int(blank_id))

    wrapper._rnnt_is_start_tokens = [token.startswith("\u2581") for token in vocabulary]
    wrapper._rnnt_unk_id = next(
        (index for index, token in enumerate(vocabulary) if token in ("<unk>", "\u2047")),
        None,
    )
    wrapper._rnnt_punct_ids = []
    wrapper._rnnt_punct_ids_set = set()
    wrapper._rnnt_punct_bias_increments = {}
    increments = wrapper.model_cfg.get("rnnt_punct_bias_increments", {}) or {}
    if wrapper.model_cfg.get("rnnt_punct_bias_enabled", False):
        for punctuation in wrapper.model_cfg.get("rnnt_punct_bias_tokens", [".", ",", "?", "!"]):
            for candidate in (punctuation, "\u2581" + punctuation):
                if candidate in vocabulary:
                    token_id = vocabulary.index(candidate)
                    if token_id not in wrapper._rnnt_punct_ids_set:
                        wrapper._rnnt_punct_ids.append(token_id)
                        wrapper._rnnt_punct_ids_set.add(token_id)
                        wrapper._rnnt_punct_bias_increments[token_id] = float(
                            increments.get(punctuation, 1.0)
                        )

    wrapper.model_cfg["rnnt_self_play_suppression"] = True
    wrapper.model_cfg["force_turn_taking"] = True
    pipeline.ui_asr_source = "rnnt"
    del restored
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "attached": True,
        "checkpoint": str(checkpoint),
        "blank_id": int(blank_id),
        "vocabulary_size": len(vocabulary),
        "load_seconds": round(time.perf_counter() - started_at, 3),
    }
    print(f"RNNT runtime attached: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def _install_local_nano_skeleton(skeleton: Path) -> None:
    """Route architecture/tokenizer construction to an exact local skeleton.

    NVIDIA's combined checkpoint deliberately contains all trained VoiceChat
    weights but its config still names the Nano architecture repository. This
    narrow hook changes only that construction-time path and keeps Hub access
    disabled; it does not replace or merge any model weights.
    """
    required = (
        skeleton / "config.json",
        skeleton / "configuration_nemotron_h.py",
        skeleton / "modeling_nemotron_h.py",
        skeleton / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Incomplete local Nano skeleton: {missing}")

    from nemo.collections.speechlm2.inference.model_wrappers import (
        nemotron_voicechat_inference_wrapper as wrapper_module,
    )

    wrapper_class = wrapper_module.NemotronVoicechatInferenceWrapper
    if getattr(wrapper_class, "_voicechat_local_skeleton_installed", False):
        return
    upstream_loader = wrapper_class._load_and_merge_configs

    def load_with_local_skeleton(wrapper: Any) -> Any:
        cfg = upstream_loader(wrapper)
        cfg.model.stt.model.pretrained_llm = str(skeleton)
        cfg.model.speech_generation.model.pretrained_lm_name = str(skeleton)
        return cfg

    wrapper_class._load_and_merge_configs = load_with_local_skeleton
    wrapper_class._voicechat_local_skeleton_installed = True


def build_public_pipeline(args: argparse.Namespace) -> Any:
    """Build NVIDIA's streamer from the exact public combined weights."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

    from omegaconf import OmegaConf

    from .audio_compat import install_torchaudio_soundfile_io

    speech_root = checked_speech_root(args.speech_root)
    checkpoint = Path(args.checkpoint_root).expanduser().resolve()
    skeleton = Path(args.hf_skeleton).expanduser().resolve()
    required = (
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "rnnt_tokenizer" / "tokenizer.model",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Incomplete public VoiceChat checkpoint: {missing}")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    rnnt_info = config.get("_rnnt_merge_info") or {}
    if not rnnt_info.get("decoder_config") or not rnnt_info.get("joint_config"):
        raise SystemExit("Public checkpoint lacks embedded RNNT decoder/joint metadata")

    sys.path.insert(0, str(speech_root))
    install_torchaudio_soundfile_io()
    _install_local_nano_skeleton(skeleton)
    from .runtime_optimizations import install_public_wrapper_hooks

    install_public_wrapper_hooks()

    config_path = (
        speech_root / "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml"
    )
    cfg = OmegaConf.load(config_path)
    manifest_path = Path(args.vllm_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in PUBLIC_VLLM_MANIFEST_KINDS:
        raise SystemExit(f"Unexpected public vLLM manifest kind: {manifest_path}")
    is_hf_release = manifest.get("kind") == "nemotron_voicechat_dgx_spark_hf_release"
    source = manifest.get("parent") if is_hf_release else manifest.get("source")
    source = source or {}
    if (
        source.get("repository") != "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
        or source.get("revision") != args.checkpoint_revision
        or (
            source.get("model_sha256")
            if is_hf_release
            else (source.get("model") or {}).get("sha256")
        )
        != args.checkpoint_sha256
        or manifest.get("derived_from_ea") is not False
    ):
        raise SystemExit("Public vLLM manifest does not match the active checkpoint")
    resolved_artifacts = {}
    for name, requested in (
        ("nano", args.nano_vllm_path),
        ("eartts", args.eartts_vllm_path),
    ):
        root = Path(requested).expanduser().resolve()
        component = (
            release_component_manifest(manifest, name)
            if is_hf_release
            else (manifest.get("components") or {}).get(name) or {}
        )
        validated = validate_manifested_model(root, component)
        if not (component.get("reproducibility") or {}).get("byte_identical"):
            raise SystemExit(f"Public {name} vLLM artifact lacks reproducibility proof")
        resolved_artifacts[name] = {
            "path": str(root),
            "bytes": validated["bytes"],
            "sha256": validated["sha256"],
            "files": validated.get("files"),
            "artifact_files": validated.get("artifact_files"),
            "tensor_count": (component.get("model") or {}).get("inventory", {}).get("tensor_count"),
            "reproducibility_runs": component["reproducibility"].get("runs"),
            "quantization": component.get("quantization"),
            "runtime_optimization": component.get("runtime_optimization"),
        }
    artifact_provenance = {
        "manifest": str(manifest_path),
        "derived_from_ea": False,
        "runtime": source.get("runtime"),
        "runtime_optimization": manifest.get("runtime_optimization"),
        "components": resolved_artifacts,
    }

    updates = {
        "audio_file": "/dev/null",
        "output_dir": "/tmp/voicechat-web",
        "s2s.model_path": str(checkpoint),
        "s2s.llm_checkpoint_path": str(checkpoint),
        "s2s.speaker_reference": None,
        "s2s.speaker_name": args.speaker_name,
        "s2s.engine_type": PRODUCTION_ENGINE_TYPE,
        "s2s.use_separate_rnnt_ckpt": False,
        "s2s.pretrained_rnnt_asr": None,
        "s2s.turn_taking_source": "rnnt",
        "s2s.ui_asr_source": "rnnt",
        # NVIDIA's public configuration enables this. The environment switch is
        # diagnostic-only and defaults to the published behavior so a run can
        # isolate perception graph replay without changing EarTTS graph mode.
        "s2s.use_perception_cudagraph": env_bool("VOICECHAT_USE_PERCEPTION_CUDAGRAPH", True),
        "s2s.rnnt_eou_frames": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.rnnt_bou_frames": int(os.environ.get("VOICECHAT_RNNT_BOU_FRAMES", "3")),
        "s2s.rnnt_fc_interrupt_ms": int(os.environ.get("VOICECHAT_RNNT_FC_INTERRUPT_MS", "240")),
        "s2s.asr_eou_first_turn": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.transport_vad_eou_frames": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.transport_vad_min_rnnt_tokens": int(
            os.environ.get("VOICECHAT_TRANSPORT_VAD_MIN_RNNT_TOKENS", "1")
        ),
        "s2s.system_prompt": "",
        "s2s.max_agent_response_sec": float(
            os.environ.get("VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC", "0")
        ),
        "s2s.tts_text_token_ratio_cap": float(os.environ.get("S2S_TTS_TEXT_TOKEN_RATIO_CAP", "16")),
        "s2s.tts_text_token_min": int(os.environ.get("S2S_TTS_TEXT_TOKEN_MIN", "5")),
        "streaming.chunk_size_in_secs": FRAME_SECONDS,
        "streaming.max_len": int(os.environ.get("VOICECHAT_STREAMING_MAX_LEN", "12288")),
    }
    updates.update(
        {
            "s2s.vllm_llm_config.model_path": str(Path(args.nano_vllm_path).expanduser().resolve()),
            "s2s.vllm_llm_config.engine_path": str(
                Path(args.nano_vllm_path).expanduser().resolve()
            ),
            "s2s.vllm_llm_config.gpu_memory_utilization": float(
                os.environ.get("VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION", "0.42")
            ),
            "s2s.vllm_llm_config.dtype": "bfloat16",
            "s2s.vllm_llm_config.max_model_len": int(
                os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")
            ),
        }
    )
    updates.update(
        {
            "s2s.vllm_tts_config.model_path": str(
                Path(args.eartts_vllm_path).expanduser().resolve()
            ),
            "s2s.vllm_tts_config.engine_path": str(
                Path(args.eartts_vllm_path).expanduser().resolve()
            ),
            "s2s.vllm_tts_config.gpu_memory_utilization": float(
                os.environ.get("VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION", "0.10")
            ),
            "s2s.vllm_tts_config.dtype": "float32",
            "s2s.vllm_tts_config.max_model_len": int(
                os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")
            ),
            "s2s.vllm_tts_config.enforce_eager": os.environ.get(
                "VOICECHAT_VLLM_EARTTS_ENFORCE_EAGER", "0"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
        }
    )
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.resolve(cfg)

    nano_runtime_optimization = validate_nano_runtime_optimization(
        Path(args.nano_vllm_path).expanduser().resolve(), cfg.s2s
    )
    nano_pad_pair_runtime = validate_nano_pad_pair_runtime(
        Path(args.nano_vllm_path).expanduser().resolve(),
        manifest,
        env_bool("VOICECHAT_NANO_PAD_PAIR", False),
    )

    import torch
    from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import (
        S2SPipelineBuilder,
    )

    torch.set_float32_matmul_precision(str(cfg.get("matmul_precision", "high")))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = S2SPipelineBuilder.build_pipeline(cfg)
    if nano_runtime_optimization is not None:
        llm_interface = pipeline.s2s_model.model_llm_interface
        interface_sampling = {
            "temperature": float(llm_interface.temperature),
            "top_p": float(llm_interface.top_p),
            "repetition_penalty": float(llm_interface.repetition_penalty),
        }
        if interface_sampling != nano_runtime_optimization["sampling_contract"]:
            raise RuntimeError(
                "Nano wrapper sampling changed after the runtime-optimization gate: "
                f"{interface_sampling}"
            )
        engine_temperature = float(llm_interface.engine.sampling_params.temperature)
        if engine_temperature != 0.0:
            raise RuntimeError(
                "Nano vLLM engine must remain greedy when custom text logits are omitted"
            )
        nano_runtime_optimization["vllm_temperature"] = engine_temperature
    from .runtime_optimizations import prepare_public_fixed_stages

    fixed_stage_optimizations = prepare_public_fixed_stages(pipeline, checkpoint)
    # NVIDIA's wrapper currently overwrites this setting to False during
    # initialization. Restore the requested public-runtime policy afterwards so
    # native BOS cannot self-trigger a new answer during post-turn silence.
    pipeline.s2s_model.model_cfg["rnnt_self_play_suppression"] = True
    stt = pipeline.s2s_model.model.stt_model
    if getattr(stt, "_rnnt_decoder", None) is None or getattr(stt, "_rnnt_joint", None) is None:
        raise RuntimeError("Exact public checkpoint did not initialize its embedded RNNT")
    pipeline.checkpoint_provenance = {
        "kind": "public_checkpoint_production_candidate_1",
        "repository": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        "revision": args.checkpoint_revision,
        "model_sha256": args.checkpoint_sha256,
        "checkpoint": str(checkpoint),
        "nano_skeleton": str(skeleton),
        "speaker": args.speaker_name,
        "turn_taking_source": "rnnt",
        "rnnt_self_play_suppression": True,
        "rnnt_eou_frames": int(pipeline.s2s_model.model_cfg["rnnt_eou_frames"]),
        "rnnt_bou_frames": int(pipeline.s2s_model.model_cfg["rnnt_bou_frames"]),
        "rnnt_fc_interrupt_ms": int(pipeline.s2s_model.model_cfg["rnnt_fc_interrupt_ms"]),
        "ui_asr_source": "rnnt",
        "rnnt_source": "embedded_combined_checkpoint",
        "use_perception_cudagraph": bool(pipeline.s2s_model.model_cfg["use_perception_cudagraph"]),
        "engine_type": PRODUCTION_ENGINE_TYPE,
        "vllm_artifacts": artifact_provenance,
        "nano_runtime_optimization": nano_runtime_optimization,
        "nano_pad_pair_runtime": nano_pad_pair_runtime,
        "fixed_stage_optimizations": fixed_stage_optimizations,
        "host_runtime": host_runtime_provenance(),
    }
    print(
        f"Checkpoint provenance: {json.dumps(pipeline.checkpoint_provenance, sort_keys=True)}",
        flush=True,
    )
    return pipeline


def build_pipeline(args: argparse.Namespace) -> Any:
    """Build only the qualified public combined-checkpoint pipeline."""
    return build_public_pipeline(args)


def warm_pipeline(pipeline: Any, warmup_wav: str | None) -> None:
    if not warmup_wav:
        pipeline.warmup()
        return
    path = Path(warmup_wav).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Warmup WAV does not exist: {path}")
    from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import (
        S2SRequestOptions,
    )

    print(f"Running full streaming warmup with {path}", flush=True)
    started_at = time.perf_counter()
    pipeline.run([str(path)], options=[S2SRequestOptions(system_prompt="")])
    print(
        f"Full streaming warmup finished in {time.perf_counter() - started_at:.2f}s",
        flush=True,
    )


def warm_realtime_engine(
    engine: Any, frame_count: int = 4, system_prompt: str = ""
) -> dict[str, Any]:
    """Exercise the exact WebSocket continuation shape before health-ready.

    Pipeline prefill warmup does not reach Nano's paired continuation path. On
    DGX Spark that path pays a one-time multi-second initialization on its third
    model frame, which must not be charged to the first client session.
    """
    if frame_count < 3:
        raise ValueError("realtime warmup requires at least three model frames")
    started_at = time.perf_counter()
    frame_ms: list[float] = []
    engine.start(system_prompt=system_prompt)
    try:
        for frame_index in range(frame_count):
            result = engine.process(
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                is_last=frame_index == frame_count - 1,
                input_active=True,
            )
            frame_ms.append(float(result.inference_ms))
    finally:
        if engine.started:
            engine.abort()
    details = {
        "frames": frame_count,
        "frame_ms": frame_ms,
        "system_prompt": system_prompt,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    print(f"Realtime continuation warmup: {json.dumps(details)}", flush=True)
    return details


async def send_step(
    websocket: Any,
    result: StepResult,
    last_user_text: str,
    trace: SessionTrace,
    transport_frame: int,
    protocol: RealtimeProtocolSession,
) -> str:
    output_bytes = b""
    audio_delivered = response_audio_is_deliverable(result.turn_state)
    delivered_output_audio: dict[str, float | int]
    if audio_delivered:
        delivered_output_audio = result.output_audio
    else:
        delivered_output_audio = {
            "samples": 0,
            "rms": 0.0,
            "rms_dbfs": -120.0,
            "peak": 0.0,
        }
    user_text_changed = result.user_text != last_user_text
    if result.audio.size and audio_delivered:
        output_bytes = float32_to_pcm16_bytes(result.audio)
        trace.output_pcm(output_bytes)
    for event in protocol.step_events(
        result,
        output_pcm=output_bytes,
        audio_delivered=audio_delivered,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
    ):
        await websocket.send_json(event)
    if result.user_text != last_user_text:
        last_user_text = result.user_text
    if result.function_delta:
        await websocket.send_json(
            protocol.diagnostic_event(
                "voicechat.function_text.delta",
                delta=result.function_delta,
                text=result.function_text,
                turn_id=protocol.turn_id,
                response_id=protocol.response_id,
            )
        )
    await websocket.send_json(
        protocol.diagnostic_event(
            "voicechat.metrics",
            frame=result.frame_index,
            transport_frame=transport_frame,
            inference_ms=round(result.inference_ms, 3),
            server_step_ms=round(result.server_step_ms, 3),
            budget_ms=FRAME_SECONDS * 1000,
            over_budget=result.server_step_ms > FRAME_SECONDS * 1000,
            input_audio=result.input_audio,
            output_audio=delivered_output_audio,
            model_output_audio=result.output_audio,
            audio_delivered=audio_delivered,
            turn_state=result.turn_state,
            turn_id=protocol.turn_id,
            response_id=protocol.response_id,
        )
    )
    trace_fields = {
        "frame": result.frame_index,
        "transport_frame": transport_frame,
        "input_audio": result.input_audio,
        "output_audio": {**delivered_output_audio, "bytes": len(output_bytes)},
        "model_output_audio": result.output_audio,
        "audio_delivered": audio_delivered,
        "inference_ms": round(result.inference_ms, 3),
        "server_step_ms": round(result.server_step_ms, 3),
        "user_text": result.user_text,
        "user_text_changed": user_text_changed,
        "assistant_delta": result.assistant_delta,
        "function_delta": result.function_delta,
        "function_text": result.function_text,
        "turn_state": result.turn_state,
    }
    if result.trace_diagnostics is not None:
        trace_fields["runtime_diagnostics"] = result.trace_diagnostics
    trace.event(
        "model_step",
        **trace_fields,
    )
    if result.trace_diagnostics and result.trace_diagnostics.get("watchdog"):
        trace.event(
            "nano_pad_pair_watchdog",
            frame=result.frame_index,
            transport_frame=transport_frame,
            **result.trace_diagnostics["watchdog"],
        )
    return last_user_text


async def send_idle_step(
    websocket: Any,
    trace: SessionTrace,
    transport_frame: int,
    input_audio: dict[str, float | int],
    gate_dbfs: float,
    transport_state: dict[str, Any] | None = None,
    protocol: RealtimeProtocolSession | None = None,
) -> None:
    output = {"samples": 0, "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0}
    turn_state = {
        "gate": "leading_silence",
        "speech_gate_dbfs": gate_dbfs,
        "transport": transport_state or {},
    }
    event_fields = {
        "frame": None,
        "transport_frame": transport_frame,
        "inference_ms": 0.0,
        "server_step_ms": 0.0,
        "budget_ms": FRAME_SECONDS * 1000,
        "over_budget": False,
        "input_audio": input_audio,
        "output_audio": output,
        "turn_state": turn_state,
    }
    await websocket.send_json(
        protocol.diagnostic_event("voicechat.metrics", **event_fields)
        if protocol is not None
        else {"type": "voicechat.metrics", **event_fields}
    )
    trace.event(
        "idle_step",
        transport_frame=transport_frame,
        input_audio=input_audio,
        output_audio=output,
        turn_state=turn_state,
    )


def session_position_limit_event(model_frames: int, max_model_frames: int) -> dict[str, Any] | None:
    if max_model_frames <= 0 or model_frames < max_model_frames:
        return None
    return {
        "type": "error",
        "error": {
            "code": "session_position_limit",
            "message": (
                f"Continuous model session reached its supported {max_model_frames}-frame limit"
            ),
            "model_frames": model_frames,
            "max_model_frames": max_model_frames,
        },
    }


def create_app(
    engine: VoiceChatEngine,
    trace_dir: Path | None = None,
    speech_gate_dbfs: float = -50.0,
    speech_gate_min_frames: int = 1,
    continuous_after_speech: bool = True,
    default_system_prompt: str = "",
    function_template_path: Path | None = None,
    max_session_model_frames: int = 12_000,
    pocket_worker: PocketWorkerManager | None = None,
) -> Any:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    # With postponed annotations, FastAPI resolves route parameter types from
    # module globals. Keep the heavyweight optional import lazy for unit tests,
    # but publish WebSocket so the upgrade route is recognized correctly.
    globals()["WebSocket"] = WebSocket

    @asynccontextmanager
    async def lifespan(_app):
        if pocket_worker is not None:
            # Loading and one full prewarm synthesis are readiness requirements.
            await pocket_worker.start()
        try:
            yield
        finally:
            if pocket_worker is not None:
                await pocket_worker.stop()

    app = FastAPI(title="Nemotron VoiceChat DGX Spark server", lifespan=lifespan)
    active_client = asyncio.Lock()
    model_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="voicechat-model"
    )
    loaded_at = time.time()
    checkpoint_provenance = dict(getattr(engine.pipeline, "checkpoint_provenance", {}) or {})
    host_provenance = host_runtime_provenance()
    checkpoint_provenance.setdefault("host_runtime", host_provenance)

    async def model_call(function: Any, *args: Any, **kwargs: Any) -> Any:
        """Run synchronous NVIDIA wrappers away from Uvicorn's asyncio loop."""
        loop = asyncio.get_running_loop()
        call = functools.partial(function, *args, **kwargs)
        return await loop.run_in_executor(model_executor, call)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        nano_pad_pair = {"enabled": False, "eligible": False, "requests": {}}
        try:
            from .runtime_optimizations import public_pad_pair_status

            nano_pad_pair = public_pad_pair_status(engine.pipeline.s2s_model)
        except (AttributeError, ImportError):
            pass
        typed_input_health = pocket_worker.health if pocket_worker is not None else {"ready": False}
        return {
            "status": "ready"
            if pocket_worker is None or typed_input_health["ready"]
            else "degraded",
            "model": "ready",
            "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            "auth_required": False,
            "checkpoint": checkpoint_provenance,
            "host_runtime": host_provenance,
            "driver_version": host_provenance["driver_version"],
            "kernel_version": host_provenance["kernel_version"],
            "nano_pad_pair": nano_pad_pair,
            "typed_input": typed_input_health,
            "active_client": active_client.locked(),
            "uptime_seconds": round(time.time() - loaded_at, 1),
        }

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        if active_client.locked():
            busy_id = str(uuid.uuid4())
            busy_protocol = RealtimeProtocolSession(
                busy_id,
                protocol_capabilities(
                    function_call_timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
                    max_session_model_frames=max_session_model_frames,
                    typed_input=pocket_worker is not None and pocket_worker.ready,
                ),
            )
            await websocket.send_json(
                busy_protocol.error(
                    "server_busy",
                    "Only one active client is supported",
                    fatal=True,
                )
            )
            await websocket.close(code=1013)
            return

        await active_client.acquire()
        session_id = str(uuid.uuid4())
        capabilities = protocol_capabilities(
            function_call_timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
            max_session_model_frames=max_session_model_frames,
            typed_input=pocket_worker is not None and pocket_worker.ready,
        )
        protocol = RealtimeProtocolSession(session_id, capabilities)
        trace = SessionTrace(trace_dir, session_id)
        pcm_buffer = PcmFrameBuffer()
        last_user_text = ""
        configured = False
        session_started = False
        close_reason = "disconnect"
        transport_frame = -1
        system_message = default_system_prompt
        system_prompt = default_system_prompt
        session_tools: list[dict[str, Any]] = []
        transport_gate = TransportModelGate(
            threshold_dbfs=speech_gate_dbfs,
            continuous_after_speech=continuous_after_speech,
            min_active_frames=speech_gate_min_frames,
        )
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        model_input_lock = asyncio.Lock()
        active_typed: ActiveTypedInput | None = None
        typed_task: asyncio.Task | None = None
        seen_typed_job_ids: set[str] = set()

        async def process_pcm(payload: bytes, *, source: str, job_id: str | None) -> None:
            """Feed PCM through the single RNNT-owned model input path."""

            nonlocal transport_frame, session_started, last_user_text, close_reason
            decoded_packet = pcm16_bytes_to_float32(payload)
            trace.input_pcm(payload)
            trace.event(
                "input_packet",
                bytes=len(payload),
                source=source,
                job_id=job_id,
                audio=audio_stats(decoded_packet),
            )
            async with model_input_lock:
                for frame in pcm_buffer.push(payload):
                    protocol.set_input_source(source, job_id)
                    transport_frame += 1
                    frame_stats = audio_stats(frame)
                    advance_model = transport_gate.should_advance(
                        float(frame_stats["rms_dbfs"])
                    )
                    if not advance_model:
                        async with send_lock:
                            await send_idle_step(
                                websocket,
                                trace,
                                transport_frame,
                                frame_stats,
                                speech_gate_dbfs,
                                transport_gate.snapshot(
                                    float(frame_stats["rms_dbfs"]), False
                                ),
                                protocol,
                            )
                        continue
                    if not session_started:
                        await model_call(engine.start, system_prompt)
                        session_started = True
                    limit_event = session_position_limit_event(
                        engine.frame_index, max_session_model_frames
                    )
                    if limit_event is not None:
                        trace.event(
                            "session_position_limit",
                            model_frames=engine.frame_index,
                            max_model_frames=max_session_model_frames,
                        )
                        async with send_lock:
                            await websocket.send_json(
                                protocol.error(
                                    "session_position_limit",
                                    limit_event["error"]["message"],
                                    fatal=True,
                                    model_frames=engine.frame_index,
                                    max_model_frames=max_session_model_frames,
                                )
                            )
                            for event in protocol.close_events(
                                "session_position_limit",
                                cumulative_user_text=last_user_text,
                                status="failed",
                            ):
                                await websocket.send_json(event)
                        close_reason = "session_position_limit"
                        await websocket.close(code=1000, reason=close_reason)
                        raise SessionTerminated
                    result = await model_call(
                        engine.process,
                        frame,
                        input_active=transport_gate.input_active,
                    )
                    result.turn_state["transport"] = transport_gate.snapshot(
                        float(frame_stats["rms_dbfs"]), True
                    )
                    async with send_lock:
                        last_user_text = await send_step(
                            websocket,
                            result,
                            last_user_text,
                            trace,
                            transport_frame,
                            protocol,
                        )
                    boundary = result.turn_state.get("response_boundary") or {}
                    if boundary.get("event") == "end":
                        transport_gate.observe_response_end()

        async def inject_trailing_silence(job: ActiveTypedInput) -> None:
            rnnt_eou_frames = int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "20"))
            margin = float(os.environ.get("VOICECHAT_TYPED_INPUT_EOU_MARGIN", "1.25"))
            seconds = typed_input_trailing_silence_seconds(
                rnnt_eou_frames=rnnt_eou_frames,
                margin=margin,
            )
            frame = b"\x00\x00" * (INPUT_SAMPLE_RATE // 50)
            for _ in range(math.ceil(seconds / 0.02)):
                await process_pcm(frame, source="typed", job_id=job.job_id)

        async def run_typed_job(job: ActiveTypedInput) -> None:
            nonlocal active_typed
            disposition = "completed"
            try:
                if pocket_worker is None or not pocket_worker.ready:
                    raise PocketWorkerError("Pocket TTS worker is unavailable")
                synthesis = await pocket_worker.synthesize(
                    job.text,
                    job_id=job.job_id,
                    cancel_event=job.cancel_event,
                )
                if synthesis.sample_rate != INPUT_SAMPLE_RATE:
                    raise PocketWorkerError(
                        f"Pocket worker returned {synthesis.sample_rate} Hz"
                    )
                if job.cancel_event.is_set():
                    disposition = "replaced" if job.replacement_requested else "cancelled"
                else:
                    async with send_lock:
                        await websocket.send_json(protocol.typed_input_started(job.job_id))
                    trace.event(
                        "typed_input_injection_started",
                        job_id=job.job_id,
                        generated_seconds=synthesis.generated_seconds,
                        synthesis_seconds=synthesis.wall_seconds,
                        time_to_first_audio_seconds=synthesis.time_to_first_audio_seconds,
                    )
                    bytes_per_frame = INPUT_SAMPLE_RATE * 20 // 1_000 * 2
                    next_send = time.monotonic()
                    for offset in range(0, len(synthesis.pcm16), bytes_per_frame):
                        if job.cancel_event.is_set():
                            disposition = (
                                "replaced" if job.replacement_requested else "cancelled"
                            )
                            break
                        delay = next_send - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        chunk = synthesis.pcm16[offset : offset + bytes_per_frame]
                        if not chunk:
                            break
                        job.audio_started = True
                        await process_pcm(chunk, source="typed", job_id=job.job_id)
                        next_send = max(next_send, time.monotonic()) + len(chunk) / (
                            2 * INPUT_SAMPLE_RATE
                        )
            except PocketSynthesisCancelled:
                disposition = "replaced" if job.replacement_requested else "cancelled"
            except SessionTerminated:
                disposition = "error"
            except Exception as exc:  # noqa: BLE001 - contain typed-input failure
                disposition = "error"
                trace.event(
                    "typed_input_error",
                    job_id=job.job_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    async with send_lock:
                        await websocket.send_json(
                            protocol.error(
                                "typed_input_failed",
                                str(exc),
                                client_event_id=job.job_id,
                            )
                        )
                except Exception:
                    pass
            finally:
                # Once any synthetic audio entered RNNT, cancellation and failure
                # must still close that partial utterance before a replacement.
                if job.audio_started and close_reason != "session_position_limit":
                    try:
                        await inject_trailing_silence(job)
                    except Exception:
                        disposition = "error"
                protocol.set_input_source("microphone")
                async with model_input_lock:
                    discarded = pcm_buffer.discard()
                if discarded:
                    trace.event(
                        "typed_input_tail_discarded",
                        job_id=job.job_id,
                        samples=discarded,
                    )
                try:
                    async with send_lock:
                        await websocket.send_json(
                            protocol.typed_input_finished(job.job_id, disposition)
                        )
                except Exception:
                    pass
                trace.event(
                    "typed_input_injection_finished",
                    job_id=job.job_id,
                    disposition=disposition,
                    audio_started=job.audio_started,
                )
                if active_typed is job:
                    active_typed = None
                job.done.set()

        async def emit_external_tool_call(call: PendingExternalToolCall) -> None:
            trace.event(
                "function_call_requested",
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
            async with send_lock:
                for event in protocol.function_call_events(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                ):
                    await websocket.send_json(event)

        async def emit_external_tool_timeout(call: PendingExternalToolCall) -> None:
            trace.event(
                "function_call_timeout",
                call_id=call.call_id,
                timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
            )
            async with send_lock:
                await websocket.send_json(
                    protocol.function_call_failed_event(
                        call_id=call.call_id,
                        reason="function_call_output_timeout",
                    )
                )

        tool_bridge = ExternalToolBridge(
            loop,
            emit_external_tool_call,
            emit_timeout=emit_external_tool_timeout,
        )
        engine.configure_external_tools({})
        try:
            await websocket.send_json(
                protocol.session_created(
                    {
                        "input_audio_format": "pcm16",
                        "input_audio_sample_rate": INPUT_SAMPLE_RATE,
                        "output_audio_format": "pcm16",
                        "output_audio_sample_rate": OUTPUT_SAMPLE_RATE,
                        "frame_duration_ms": int(FRAME_SECONDS * 1000),
                        "checkpoint": checkpoint_provenance,
                    }
                )
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await websocket.send_json(
                        protocol.error(
                            "binary_audio_unsupported",
                            "Send JSON input_audio_buffer.append with base64 PCM16 audio",
                        )
                    )
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        protocol.error("invalid_json", "Message is not valid JSON")
                    )
                    continue
                if not isinstance(body, dict):
                    await websocket.send_json(
                        protocol.error("invalid_event", "Event must be a JSON object")
                    )
                    continue
                message_type = body.get("type", "")
                client_event_id = body.get("event_id")

                if message_type == "session.update":
                    trace.event("client_event", type=message_type)
                    if session_started:
                        await websocket.send_json(
                            protocol.error(
                                "settings_locked",
                                "session.update must precede the first model frame",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    session_update = body.get("session", {})
                    if not isinstance(session_update, dict):
                        await websocket.send_json(
                            protocol.error(
                                "invalid_session_update",
                                "session.update.session must be an object",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    requested_version = session_update.get("protocol_version")
                    if not configured and requested_version != PROTOCOL_VERSION:
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_protocol_version",
                                f"{PROTOCOL_NAME} version {PROTOCOL_VERSION} is required",
                                fatal=True,
                                client_event_id=client_event_id,
                                requested_version=requested_version,
                            )
                        )
                        await websocket.send_json(
                            protocol.session_closed("unsupported_protocol_version", status="failed")
                        )
                        close_reason = "unsupported_protocol_version"
                        await websocket.close(code=1002, reason=close_reason)
                        return
                    if configured and requested_version not in (None, PROTOCOL_VERSION):
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_protocol_version",
                                f"protocol version cannot change from {PROTOCOL_VERSION}",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if "instructions" in session_update:
                        instructions = session_update.get("instructions")
                        system_message = (
                            default_system_prompt if instructions is None else str(instructions)
                        )
                    if "tools" in session_update:
                        try:
                            session_tools = normalize_tool_definitions(session_update["tools"])
                        except (TypeError, ValueError) as exc:
                            await websocket.send_json(
                                protocol.error(
                                    "invalid_tools",
                                    str(exc),
                                    client_event_id=client_event_id,
                                )
                            )
                            continue
                    if session_tools:
                        if function_template_path is None:
                            await websocket.send_json(
                                protocol.error(
                                    "function_calling_unavailable",
                                    "Function-calling template is unavailable",
                                    fatal=True,
                                    client_event_id=client_event_id,
                                )
                            )
                            close_reason = "function_calling_unavailable"
                            await websocket.close(code=1011, reason=close_reason)
                            return
                        system_prompt = render_tool_system_prompt(
                            function_template_path, system_message, session_tools
                        )
                    else:
                        system_prompt = system_message
                    engine.configure_external_tools(tool_bridge.handlers(session_tools))
                    configured = True
                    trace.event(
                        "session_tools_configured",
                        tools=[tool["name"] for tool in session_tools],
                        rendered_prompt_bytes=len(system_prompt.encode("utf-8")),
                    )
                    await websocket.send_json(
                        protocol.session_updated(
                            {
                                "instructions": system_message,
                                "tools": session_tools,
                                "input_audio_format": "pcm16",
                                "input_audio_sample_rate": INPUT_SAMPLE_RATE,
                                "output_audio_format": "pcm16",
                                "output_audio_sample_rate": OUTPUT_SAMPLE_RATE,
                            }
                        )
                    )
                    continue
                if message_type == "conversation.item.create":
                    if not configured:
                        await websocket.send_json(
                            protocol.error(
                                "session_not_configured",
                                "Complete session.update before sending conversation items",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    item = body.get("item", {})
                    if not isinstance(item, dict) or item.get("type") != "function_call_output":
                        await websocket.send_json(
                            protocol.error(
                                "invalid_conversation_item",
                                "conversation.item.create requires a function_call_output item",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if not session_tools:
                        await websocket.send_json(
                            protocol.error(
                                "tools_not_set",
                                "No tools were configured in session.update",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    call_id = item.get("call_id")
                    try:
                        tool_bridge.submit(call_id, item.get("output", ""))
                    except ValueError as exc:
                        await websocket.send_json(
                            protocol.error(
                                "invalid_function_call_output",
                                str(exc),
                                client_event_id=client_event_id,
                                call_id=call_id,
                            )
                        )
                        continue
                    protocol.function_call_completed(call_id)
                    trace.event(
                        "function_call_output_received",
                        call_id=call_id,
                        output_bytes=len(str(item.get("output", "")).encode("utf-8")),
                    )
                    continue
                if message_type == "input_text.request":
                    if not configured:
                        await websocket.send_json(
                            protocol.typed_input_rejected(
                                body.get("job_id"), "session_not_configured"
                            )
                        )
                        continue
                    job_id = body.get("job_id")
                    typed_text = body.get("text")
                    rejection = None
                    if pocket_worker is None or not pocket_worker.ready:
                        rejection = "typed_input_unavailable"
                    elif not isinstance(job_id, str) or not job_id or len(job_id) > 128:
                        rejection = "invalid_job_id"
                    elif job_id in seen_typed_job_ids:
                        rejection = "duplicate_job_id"
                    elif not isinstance(typed_text, str) or not typed_text.strip():
                        rejection = "empty_text"
                    elif len(typed_text) > MAX_TYPED_INPUT_CHARS:
                        rejection = "text_too_long"
                    if rejection is not None:
                        await websocket.send_json(
                            protocol.typed_input_rejected(
                                job_id if isinstance(job_id, str) else None,
                                rejection,
                            )
                        )
                        continue
                    if active_typed is not None:
                        active_typed.replacement_requested = True
                        active_typed.cancel_event.set()
                        await active_typed.done.wait()
                        if typed_task is not None:
                            await asyncio.gather(typed_task, return_exceptions=True)
                    seen_typed_job_ids.add(job_id)
                    async with model_input_lock:
                        discarded = pcm_buffer.discard()
                    if discarded:
                        trace.event(
                            "microphone_partial_discarded_for_typed_input",
                            job_id=job_id,
                            samples=discarded,
                        )
                    active_typed = ActiveTypedInput(
                        job_id=job_id,
                        text=typed_text,
                        cancel_event=asyncio.Event(),
                        done=asyncio.Event(),
                    )
                    trace.event(
                        "typed_input_accepted",
                        job_id=job_id,
                        text_bytes=len(typed_text.encode("utf-8")),
                    )
                    await websocket.send_json(protocol.typed_input_accepted(job_id))
                    typed_task = asyncio.create_task(
                        run_typed_job(active_typed),
                        name=f"typed-input-{job_id}",
                    )
                    continue
                if message_type == "input_audio_buffer.append":
                    if not configured:
                        await websocket.send_json(
                            protocol.error(
                                "session_not_configured",
                                "Complete session.update before sending audio",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    try:
                        if body.get("encoding") != "pcm16":
                            raise RealtimeProtocolError(
                                "unsupported_audio_encoding",
                                "Only pcm16 input is supported",
                            )
                        if int(body.get("sample_rate", 0)) != INPUT_SAMPLE_RATE:
                            raise RealtimeProtocolError(
                                "unsupported_sample_rate",
                                f"Input must be {INPUT_SAMPLE_RATE} Hz",
                            )
                        if int(body.get("channels", 0)) != 1:
                            raise RealtimeProtocolError(
                                "unsupported_channel_count", "Input must be mono"
                            )
                        payload = base64.b64decode(body.get("audio", ""), validate=True)
                        if not payload:
                            raise RealtimeProtocolError(
                                "empty_audio", "Audio payload must not be empty"
                            )
                        if len(payload) % 2:
                            raise RealtimeProtocolError(
                                "invalid_audio",
                                "PCM16 audio must contain whole samples",
                            )
                        if len(payload) > MAX_INPUT_MESSAGE_BYTES:
                            raise RealtimeProtocolError(
                                "audio_packet_too_large", "Audio packet is too large"
                            )
                    except (ValueError, TypeError, base64.binascii.Error) as exc:
                        await websocket.send_json(
                            protocol.error(
                                "invalid_audio",
                                str(exc),
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    except RealtimeProtocolError as exc:
                        await websocket.send_json(
                            protocol.error(
                                exc.code,
                                str(exc),
                                fatal=exc.fatal,
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if active_typed is not None:
                        trace.event(
                            "microphone_packet_gated",
                            bytes=len(payload),
                            active_job_id=active_typed.job_id,
                        )
                        continue
                    await process_pcm(payload, source="microphone", job_id=None)
                    continue
                if message_type == "session.stop":
                    trace.event("client_event", type=message_type)
                    if active_typed is not None:
                        active_typed.cancel_event.set()
                        try:
                            await asyncio.wait_for(active_typed.done.wait(), timeout=10.0)
                        except TimeoutError:
                            pass
                    if session_started:
                        transport_frame += 1
                        final_frame = pcm_buffer.flush()
                        final_stats = audio_stats(final_frame)
                        result = await model_call(
                            engine.process,
                            final_frame,
                            is_last=True,
                            input_active=False,
                        )
                        result.turn_state["transport"] = transport_gate.snapshot(
                            float(final_stats["rms_dbfs"]), True
                        )
                        async with send_lock:
                            last_user_text = await send_step(
                                websocket,
                                result,
                                last_user_text,
                                trace,
                                transport_frame,
                                protocol,
                            )
                        session_started = False
                        transport_gate.reset()
                    for event in protocol.close_events(
                        "client_stop", cumulative_user_text=last_user_text
                    ):
                        await websocket.send_json(event)
                    close_reason = "client_stop"
                    await websocket.close(code=1000, reason=close_reason)
                    return
                await websocket.send_json(
                    protocol.error(
                        "unsupported_event",
                        f"Unsupported event: {message_type}",
                        client_event_id=client_event_id,
                    )
                )
        except SessionTerminated:
            pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await websocket.send_json(
                    protocol.error(
                        "internal_error",
                        f"{type(exc).__name__}: {exc}",
                        fatal=True,
                    )
                )
                await websocket.send_json(
                    protocol.session_closed("internal_error", status="failed")
                )
                close_reason = "internal_error"
                await websocket.close(code=1011, reason=close_reason)
            except Exception:
                pass
        finally:
            if active_typed is not None:
                active_typed.cancel_event.set()
            if typed_task is not None and not typed_task.done():
                try:
                    await asyncio.wait_for(typed_task, timeout=3.0)
                except TimeoutError:
                    typed_task.cancel()
                    await asyncio.gather(typed_task, return_exceptions=True)
            tool_bridge.close()
            await model_call(engine.abort)
            trace.close(close_reason)
            active_client.release()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--hf-skeleton", required=True)
    parser.add_argument(
        "--checkpoint-revision",
        default="fb0f94eaf4d03ddc430f39565229393fa1b50c26",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default="d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b",
    )
    parser.add_argument("--speaker-name", default="Aria")
    parser.add_argument("--nano-vllm-path", required=True)
    parser.add_argument("--eartts-vllm-path", required=True)
    parser.add_argument("--vllm-manifest", required=True)
    parser.add_argument("--warmup-wav")
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("VOICECHAT_WEB_SYSTEM_PROMPT", ""),
        help="Default prompt used when session.start does not supply one",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--trace-dir",
        help="Write per-session JSONL plus exact input/output PCM payloads here",
    )
    parser.add_argument(
        "--speech-gate-dbfs",
        type=float,
        default=float(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_DBFS", "-50")),
        help="RMS threshold for transport speech evidence",
    )
    parser.add_argument(
        "--speech-gate-min-frames",
        type=int,
        default=int(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_MIN_FRAMES", "1")),
        help="Consecutive 80 ms frames required for transport speech evidence",
    )
    parser.add_argument(
        "--continuous-after-speech",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VOICECHAT_WEB_CONTINUOUS_AFTER_SPEECH", "1")
        not in {"0", "false", "False"},
        help="After first speech, advance every received frame including silence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    function_template = (
        Path(args.speech_root).expanduser().resolve()
        / "examples/speechlm2/function_calling/template.jinja"
    )
    if not function_template.is_file():
        raise SystemExit(f"Missing NVIDIA function-calling template: {function_template}")
    pipeline = build_pipeline(args)
    warm_pipeline(pipeline, args.warmup_wav)
    agent_no_text_frames = int(os.environ.get("VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES", "30"))
    agent_no_audio_frames = int(os.environ.get("VOICECHAT_WEB_AGENT_NO_AUDIO_FRAMES", "30"))
    agent_decoded_silence_frames = int(
        os.environ.get("VOICECHAT_WEB_AGENT_SILENCE_EOS_FRAMES", "20")
    )
    candidate_runtime = pipeline.checkpoint_provenance.get("nano_pad_pair_runtime", {}).get(
        "production_candidate"
    )
    if candidate_runtime is not None:
        candidate_contract = pipeline.checkpoint_provenance["nano_pad_pair_runtime"]["contract"]
        expected_no_text_frames = int(candidate_contract["agent_no_text_frames"])
        expected_no_audio_frames = int(candidate_contract["agent_no_audio_frames"])
        expected_decoded_silence_frames = int(candidate_contract["agent_decoded_silence_frames"])
        if (
            agent_no_text_frames != expected_no_text_frames
            or agent_no_audio_frames != expected_no_audio_frames
            or agent_decoded_silence_frames != expected_decoded_silence_frames
        ):
            raise RuntimeError(
                "Production candidate BOS progress deadline mismatch: "
                f"text={agent_no_text_frames}/{expected_no_text_frames}, "
                f"audio={agent_no_audio_frames}/{expected_no_audio_frames}, "
                "decoded_silence="
                f"{agent_decoded_silence_frames}/{expected_decoded_silence_frames}"
            )
    engine = VoiceChatEngine(
        pipeline,
        agent_silence_eos_dbfs=float(os.environ.get("VOICECHAT_WEB_AGENT_SILENCE_EOS_DBFS", "-90")),
        agent_silence_eos_frames=agent_decoded_silence_frames,
        agent_no_text_frames=agent_no_text_frames,
        agent_no_audio_frames=agent_no_audio_frames,
        response_tail_dbfs=float(os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_DBFS", "-60")),
        response_tail_silence_frames=int(
            os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_SILENCE_FRAMES", "3")
        ),
        response_tail_max_frames=int(
            os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_MAX_FRAMES", "25")
        ),
        delivery_silence_frames=int(os.environ.get("VOICECHAT_WEB_DELIVERY_SILENCE_FRAMES", "12")),
    )
    if os.environ.get("VOICECHAT_WEB_REALTIME_WARMUP", "1") == "1":
        realtime_warmup = warm_realtime_engine(engine, system_prompt=args.system_prompt)
        pipeline.checkpoint_provenance["realtime_continuation_warmup"] = realtime_warmup
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"WebSocket traces: {trace_dir}", flush=True)
    app = create_app(
        engine,
        trace_dir,
        args.speech_gate_dbfs,
        args.speech_gate_min_frames,
        args.continuous_after_speech,
        args.system_prompt,
        function_template,
        max_session_model_frames=int(
            os.environ.get("VOICECHAT_WEB_MAX_SESSION_MODEL_FRAMES", "12000")
        ),
        pocket_worker=PocketWorkerManager(
            executable=os.environ.get(
                "VOICECHAT_POCKET_PYTHON", "/opt/pocket-tts/bin/python"
            ),
            socket_path=Path(
                os.environ.get("VOICECHAT_POCKET_SOCKET", "/tmp/voicechat-pocket.sock")
            ),
            language=os.environ.get(
                "VOICECHAT_TYPED_INPUT_LANGUAGE", "english_2026-04"
            ),
            voice=os.environ.get("VOICECHAT_TYPED_INPUT_VOICE", "alba"),
            threads=int(os.environ.get("VOICECHAT_TYPED_INPUT_TORCH_THREADS", "4")),
            cpus=tuple(
                int(item)
                for item in os.environ.get(
                    "VOICECHAT_TYPED_INPUT_CPU_CORES", "7,8,9,15"
                ).split(",")
            ),
        ),
    )

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=websocket_ping_interval(),
        ws_ping_timeout=websocket_ping_timeout(),
    )


if __name__ == "__main__":
    main()
