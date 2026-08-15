"""Foreground, credential-free lifecycle for the local Voicechat stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    Layout,
    atomic_json,
    default_layout,
    host_facts,
    load_config,
    require_free_space,
    sha256_file,
)
from .provenance import valid_runtime_image_id
from .public_runtime_identity import (
    PUBLIC_RUNTIME_PAYLOAD_LABEL,
    PUBLIC_RUNTIME_RECIPE_LABEL,
    PUBLIC_RUNTIME_SOURCE_LABEL,
    public_runtime_payload_sha256,
    public_runtime_recipe_sha256,
    public_runtime_source_sha256,
)

BOT_SOURCE_ROOT = REPO_ROOT / "src/nemotron_voicechat_pipecat"
BOT_RELOAD_DEBOUNCE_SECONDS = 0.5
BOT_RESTART_TIMEOUT_SECONDS = 45.0


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"missing required executable `{name}`")
    return path


def _image_exists(image: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _runtime_image_matches_checkout(image: str) -> bool:
    """Require a tagged image to contain the exact current runtime build inputs."""

    try:
        inspected = json.loads(
            subprocess.run(
                ["docker", "image", "inspect", image],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        labels = inspected[0].get("Config", {}).get("Labels") or {}
        expected = {
            PUBLIC_RUNTIME_SOURCE_LABEL: public_runtime_source_sha256(REPO_ROOT),
            PUBLIC_RUNTIME_PAYLOAD_LABEL: public_runtime_payload_sha256(REPO_ROOT),
            PUBLIC_RUNTIME_RECIPE_LABEL: public_runtime_recipe_sha256(REPO_ROOT),
        }
        return all(labels.get(name) == value for name, value in expected.items())
    except (OSError, IndexError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError):
        return False


def _container_exists(name: str) -> bool:
    return (
        subprocess.run(
            ["docker", "container", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as stream:
        # Match normal server restart semantics: recently closed health-check
        # connections may remain in TIME_WAIT but do not prevent a listener
        # using SO_REUSEADDR from rebinding the port.
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            stream.bind((host, port))
        except OSError:
            return False
    return True


def _wait_ports_available(endpoints: list[tuple[str, int]], timeout: float = 30.0) -> None:
    """Do not return from foreground shutdown until an immediate restart can bind."""

    deadline = time.monotonic() + timeout
    pending = endpoints
    while pending and time.monotonic() < deadline:
        pending = [endpoint for endpoint in endpoints if not _port_available(*endpoint)]
        if pending:
            time.sleep(0.1)
    if pending:
        rendered = ", ".join(f"{host}:{port}" for host, port in pending)
        raise RuntimeError(f"ports were not released after stack shutdown: {rendered}")


def _bootstrap_ready(config: dict[str, Any], layout: Layout) -> tuple[bool, str]:
    artifact_state = layout.cache / "artifact-state.json"
    if not artifact_state.is_file():
        return False, "model artifacts have not been downloaded"
    try:
        state = json.loads(artifact_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "artifact state is unreadable"
    release = config["artifacts"]["release"]
    if (
        state.get("release_sha256") != release["release_sha256"]
        or state.get("release_revision") != release["revision"]
    ):
        return False, "artifact state does not match checked-in configuration"
    if not _image_exists(config["image"]["runtime"]):
        return False, "public runtime image has not been built"
    if not _runtime_image_matches_checkout(config["image"]["runtime"]):
        return False, "public runtime image does not match this checkout"
    if not layout.state_file.is_file():
        return False, "bootstrap did not finish its final state commit"
    try:
        bootstrap_state = json.loads(layout.state_file.read_text(encoding="utf-8"))
        image_id = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                config["image"]["runtime"],
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, json.JSONDecodeError, subprocess.CalledProcessError):
        return False, "bootstrap state or runtime image is unreadable"
    if (
        bootstrap_state.get("release_sha256") != release["release_sha256"]
        or bootstrap_state.get("runtime_image_id") != image_id
    ):
        return False, "bootstrap state does not match the runtime image"
    return True, "ready"


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    support = config["support"]
    facts = host_facts()
    venv_python = layout.cache / "venv/bin/python"
    if venv_python.is_file():
        pinned_python = (
            subprocess.run([str(venv_python), "--version"], capture_output=True, text=True)
            .stdout.strip()
            .removeprefix("Python ")
        )
    else:
        pinned_python = "missing"
    checks: list[tuple[str, bool, str]] = [
        (
            "architecture",
            facts["architecture"] in {support["architecture"], "arm64"},
            f"{facts['architecture']} (expected {support['architecture']})",
        ),
        (
            "kernel",
            facts["kernel_version"] == support["kernel_version"],
            f"{facts['kernel_version']} (qualified {support['kernel_version']})",
        ),
        (
            "python",
            pinned_python == support["python"],
            f"{pinned_python} (pinned {support['python']})",
        ),
    ]
    for executable in ("uv", "docker", "git", "curl"):
        checks.append(
            (
                executable,
                shutil.which(executable) is not None,
                shutil.which(executable) or "missing",
            )
        )
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        nvidia = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        checks.append(
            (
                "NVIDIA driver",
                support["gpu"] in nvidia and support["driver_version"] in nvidia,
                nvidia,
            )
        )
    else:
        checks.append(("NVIDIA driver", False, "nvidia-smi missing"))
    docker_ready = (
        subprocess.run(
            ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
        if shutil.which("docker")
        else False
    )
    checks.append(("Docker daemon", docker_ready, "available" if docker_ready else "unavailable"))
    if docker_ready:
        nvidia_cli = shutil.which("nvidia-container-cli")
        nvidia_runtime_ready = (
            bool(nvidia_cli)
            and subprocess.run(
                [nvidia_cli, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        checks.append(
            (
                "NVIDIA container runtime",
                nvidia_runtime_ready,
                nvidia_cli or "nvidia-container-cli missing",
            )
        )
        sbom_generator = REPO_ROOT / "tools/runtime/generate_runtime_sbom.py"
        checks.append(
            (
                "runtime SBOM generator",
                sbom_generator.is_file(),
                str(sbom_generator),
            )
        )
    deployment = config["deployment"]
    checks.extend(
        [
            (
                "model loopback port",
                _port_available("127.0.0.1", deployment["model_port"]),
                str(deployment["model_port"]),
            ),
            (
                "Pipecat port",
                _port_available(deployment["pipecat_host"], deployment["pipecat_port"]),
                f"{deployment['pipecat_host']}:{deployment['pipecat_port']}",
            ),
        ]
    )
    ready, reason = _bootstrap_ready(config, layout)
    checks.append(("bootstrap state", ready, reason))
    print(f"cache:  {layout.cache}")
    print(f"traces: {layout.traces}")
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def _uv_environment(layout: Layout) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(layout.cache / "venv"),
            "UV_CACHE_DIR": str(layout.cache / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(layout.cache / "python"),
        }
    )
    return environment


def _sync_host(layout: Layout, offline: bool) -> None:
    command = [_executable("uv"), "sync", "--frozen"]
    if offline:
        command.append("--offline")
    _run(command, cwd=REPO_ROOT, env=_uv_environment(layout))


def _download_artifacts(args: argparse.Namespace, layout: Layout) -> None:
    command = [
        str(layout.cache / "venv/bin/python"),
        "-m",
        "nemotron_voicechat_runtime.bootstrap_download",
        "--config",
        str(args.config),
        "--cache-root",
        str(layout.cache),
        "--trace-root",
        str(layout.traces),
    ]
    if args.offline:
        command.append("--offline")
    _run(command, cwd=REPO_ROOT)


def _build_image(config: dict[str, Any], layout: Layout, *, offline: bool, no_cache: bool) -> None:
    image = config["image"]["runtime"]
    image_exists = _image_exists(image)
    image_matches = image_exists and _runtime_image_matches_checkout(image)
    if image_matches and not no_cache:
        print(f"[resume] runtime image exists: {image}")
        _run(
            [
                str(REPO_ROOT / "container/audit-public-runtime.sh"),
                image,
                str(layout.cache / "build/audit"),
            ],
            cwd=REPO_ROOT,
        )
    else:
        if offline:
            reason = "missing" if not image_exists else "stale"
            raise RuntimeError(
                f"offline bootstrap cannot build {reason} image {image}; "
                "complete one online bootstrap first"
            )
        environment = os.environ.copy()
        environment["VOICECHAT_RUNTIME_IMAGE"] = image
        environment["VOICECHAT_BUILD_CACHE"] = str(layout.cache / "build")
        if no_cache:
            environment["VOICECHAT_BUILD_NO_CACHE"] = "1"
        _run(
            [str(REPO_ROOT / "container/build-public-runtime.sh")],
            cwd=REPO_ROOT,
            env=environment,
        )


def _ensure_asr_evaluator(config: dict[str, Any], *, offline: bool, no_cache: bool) -> str:
    """Build or audit the qualification-only ASR image and return its immutable ID."""

    asr_image = config["image"]["asr_evaluator"]
    if _image_exists(asr_image) and not no_cache:
        print(f"[resume] ASR evaluator image exists: {asr_image}")
    else:
        if offline:
            raise RuntimeError(
                f"offline bootstrap cannot build missing image {asr_image}; "
                "complete one online bootstrap first"
            )
        environment = os.environ.copy()
        environment["VOICECHAT_ASR_EVALUATOR_IMAGE"] = asr_image
        if no_cache:
            environment["VOICECHAT_BUILD_NO_CACHE"] = "1"
        _run(
            [str(REPO_ROOT / "container/build-asr-evaluator.sh")],
            cwd=REPO_ROOT,
            env=environment,
        )
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", asr_image],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run(
        [str(REPO_ROOT / "container/audit-asr-evaluator.sh"), image_id],
        cwd=REPO_ROOT,
    )
    return image_id


def _link_or_copy(source: str, destination: str) -> str:
    """Hardlink cache-local release files, with a cross-device fallback."""

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _remove_stale_conversion_activations(layout: Layout) -> None:
    """Remove interrupted swap state after the active release is verified."""

    for path in layout.artifacts.glob(".source-converted-release-*"):
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"unexpected source-conversion activation path: {path}")
        shutil.rmtree(path)


def _tree_copy_gib(roots: tuple[Path, ...]) -> int:
    """Return the whole-GiB disk reservation for independent component copies."""

    total = sum(path.stat().st_size for root in roots for path in root.rglob("*") if path.is_file())
    gib = 1024**3
    return (total + gib - 1) // gib


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


_ASR_NUMBER_EQUIVALENTS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _asr_normalize(text: str) -> list[str]:
    normalized = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        token = _ASR_NUMBER_EQUIVALENTS.get(token, token)
        normalized.append(str(int(token)) if token.isdigit() else token)
    return normalized


def _asr_word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _asr_normalize(reference)
    actual = _asr_normalize(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for index, expected_token in enumerate(expected, 1):
        current = [index]
        for offset, actual_token in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (expected_token != actual_token),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _asr_provenance_contract() -> dict[str, Any]:
    manifest_path = REPO_ROOT / "config/nemotron-asr-en-0.6b.json"
    backend_path = REPO_ROOT / "src/nemotron_voicechat_asr_evaluator/backend.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference = manifest["inference"]
    return {
        "backend": inference["backend"],
        "backend_path": "/opt/voicechat-asr/nemotron_voicechat_asr_evaluator/backend.py",
        "backend_sha256": sha256_file(backend_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_id": manifest["model_id"],
        "hf_revision": manifest["revision"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "decoder": {
            "strategy": "greedy",
            "stopping": "encoder_exhaustion",
            "num_lookahead_tokens": inference["num_lookahead_tokens"],
            "attention_implementation": "sdpa",
            "streaming": False,
        },
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "float32_matmul_precision": "highest",
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "trust_remote_code": False,
        "network_required": False,
    }


def _validated_external_asr_report(
    work: Path, report_path: Path, source_dir: Path, evaluator_image_id: str
) -> dict[str, Any]:
    """Re-derive the source-conversion ASR evidence from its report and retained WAVs."""

    work = work.resolve()
    report_path = report_path.resolve()
    source_dir = source_dir.resolve()
    if work not in report_path.parents or work not in source_dir.parents:
        raise RuntimeError("external ASR evidence path escapes the conversion workspace")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_report_path = source_dir / "report.json"
    if (
        report.get("passed") is not True
        or report.get("pending_external_asr") is not False
        or report.get("pre_asr_report_sha256") != sha256_file(source_report_path)
    ):
        raise RuntimeError(f"invalid finalized ASR report: {report_path}")
    provenance = report.get("external_asr") or {}
    if not valid_runtime_image_id(evaluator_image_id):
        raise RuntimeError("external ASR evaluator ID is not immutable")
    contract = _asr_provenance_contract()
    if provenance.get("evaluator_image_id") != evaluator_image_id or any(
        provenance.get(key) != value for key, value in contract.items()
    ):
        raise RuntimeError(f"ASR provenance does not identify the audited evaluator: {report_path}")

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    inputs = source_report.get("asr_inputs") or {}
    diagnostics = (report.get("offline_diagnostic") or {}).get("external_asr") or {}
    results = {"streaming": provenance, "offline": diagnostics}
    input_records: dict[str, Any] = {}
    for name in ("streaming", "offline"):
        expected = inputs.get(name) or {}
        wav = (source_dir / str(expected.get("path") or "")).resolve()
        if wav.parent != source_dir.resolve() or not wav.is_file():
            raise RuntimeError(f"ASR WAV is missing or escapes its source directory: {wav}")
        observed = {"bytes": wav.stat().st_size, "sha256": sha256_file(wav)}
        if observed != {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}:
            raise RuntimeError(f"ASR WAV identity changed after transcription: {wav}")
        result = results[name]
        if (
            result.get("input_bytes") != observed["bytes"]
            or result.get("input_sha256") != observed["sha256"]
        ):
            raise RuntimeError(f"ASR result does not identify its input WAV: {wav}")
        with wave.open(str(wav), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 16_000
                or source.getnframes() <= 0
                or source.getcomptype() != "NONE"
            ):
                raise RuntimeError(f"ASR evidence WAV is not PCM16 mono 16 kHz: {wav}")
        input_records[name] = {"path": str(wav.relative_to(work)), **observed}

    reference = str((source_report.get("fixture") or {}).get("reference_text") or "")
    max_wer = (source_report.get("external_asr") or {}).get("max_wer")
    transcript = provenance.get("transcript")
    if (
        not reference
        or isinstance(max_wer, bool)
        or not isinstance(max_wer, (int, float))
        or not 0 <= max_wer <= 1
        or not isinstance(transcript, str)
    ):
        raise RuntimeError(f"invalid ASR scoring inputs: {report_path}")
    rederived_wer = _asr_word_error_rate(reference, transcript)
    rederived_passed = bool(
        transcript.strip() and rederived_wer <= max_wer and "5" in _asr_normalize(transcript)
    )
    if (
        not rederived_passed
        or report.get("passed") is not True
        or provenance.get("passed") is not True
        or provenance.get("max_wer") != max_wer
        or provenance.get("wer") != rederived_wer
    ):
        raise RuntimeError(
            f"external ASR verdict does not rederive from retained evidence: {report_path}"
        )
    return {
        "report": str(report_path.relative_to(work)),
        "report_sha256": sha256_file(report_path),
        "pre_asr_report": str(source_report_path.relative_to(work)),
        "pre_asr_report_sha256": sha256_file(source_report_path),
        "inputs": input_records,
        "external_asr": provenance,
    }


def _finalize_source_asr_stage(work: Path, evaluator_image_id: str) -> dict[str, Any]:
    """Run or resume external ASR as its own content-addressed conversion stage."""

    marker_path = work / "external-asr/stage.json"
    names = ("eartts-eager", "eartts-graph")
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        recorded_sha = marker.pop("sha256", None)
        if _canonical_json_sha256(marker) != recorded_sha:
            raise RuntimeError(f"corrupt external ASR stage marker: {marker_path}")
        if marker.get("evaluator_image_id") != evaluator_image_id:
            raise RuntimeError("external ASR stage was produced by a different evaluator image")
        derived = {
            name: _validated_external_asr_report(
                work, work / f"external-asr/{name}.json", work / f"gates/{name}", evaluator_image_id
            )
            for name in names
        }
        if marker.get("reports") != derived:
            raise RuntimeError("external ASR stage evidence no longer matches retained files")
        print("[resume] external ASR stage", flush=True)
        return derived

    outputs = work / "external-asr"
    outputs.mkdir(parents=True, exist_ok=True)
    derived: dict[str, Any] = {}
    for name in names:
        report_path = outputs / f"{name}.json"
        source_dir = work / f"gates/{name}"
        if not report_path.exists():
            _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "--ipc",
                    "host",
                    "--network",
                    "none",
                    "-e",
                    f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID={evaluator_image_id}",
                    "-v",
                    f"{REPO_ROOT}:/workspace/project:ro",
                    "-v",
                    f"{work}:/work",
                    evaluator_image_id,
                    "python3",
                    "-P",
                    "/workspace/project/tools/qualification/finalize_eartts_component_asr.py",
                    "--output-dir",
                    f"/work/gates/{name}",
                    "--report-out",
                    f"/work/external-asr/{name}.json",
                ]
            )
        derived[name] = _validated_external_asr_report(
            work, report_path, source_dir, evaluator_image_id
        )
    marker = {
        "schema": 1,
        "stage": "external-asr",
        "evaluator_image_id": evaluator_image_id,
        "reports": derived,
    }
    marker["sha256"] = _canonical_json_sha256(marker)
    atomic_json(marker_path, marker)
    return derived


def _activate_converted_release(
    layout: Layout,
    converted: Path,
    expected_release_sha: str,
    expected_asr_image_id: str,
) -> dict[str, Any]:
    """Atomically activate reproduced files after full signed-release verification."""

    from .bootstrap_download import verify_release

    report_path = converted.parent / "external-asr/reproduction.json"
    if not report_path.is_file():
        raise RuntimeError(f"source conversion report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_report_sha = report.get("sha256")
    if report.get("kind") != "voicechat_release_artifact_reproduction" or not isinstance(
        recorded_report_sha, str
    ):
        raise RuntimeError("source conversion report has an invalid identity")
    unsigned_report = dict(report)
    unsigned_report.pop("sha256")
    if _canonical_json_sha256(unsigned_report) != recorded_report_sha:
        raise RuntimeError("source conversion report failed its self-hash")
    if report.get("asr_evaluator_image_id") != expected_asr_image_id:
        raise RuntimeError("source conversion report has the wrong ASR evaluator identity")
    expected_asr = report.get("component_gates_external_asr")
    if not isinstance(expected_asr, dict) or set(expected_asr) != {
        "eartts-eager",
        "eartts-graph",
    }:
        raise RuntimeError("source conversion report is missing external ASR evidence")
    work = converted.parent
    for name, expected in expected_asr.items():
        observed = _validated_external_asr_report(
            work,
            work / str(expected.get("report") or ""),
            work / f"gates/{name}",
            expected_asr_image_id,
        )
        if observed != expected:
            raise RuntimeError(f"source conversion ASR evidence changed for {name}")
    for component in ("nano", "eartts"):
        if not (converted / component).is_dir():
            raise RuntimeError(f"source conversion output is missing {component}")

    temporary = Path(tempfile.mkdtemp(prefix=".source-converted-release-", dir=layout.artifacts))
    candidate = temporary / "release"
    backup = temporary / "downloaded-release"
    try:
        shutil.copytree(layout.release, candidate, copy_function=_link_or_copy)
        for component in ("nano", "eartts"):
            shutil.rmtree(candidate / component)
            shutil.copytree(
                converted / component,
                candidate / component,
                # Keep the active release independent from the resumable
                # conversion work tree. In-place diagnostic edits there must
                # never alias into serving artifacts through hardlinks.
                copy_function=shutil.copy2,
            )
        verify_release(candidate, expected_release_sha)

        os.replace(layout.release, backup)
        try:
            os.replace(candidate, layout.release)
            verify_release(layout.release, expected_release_sha)
        except BaseException:
            if layout.release.exists():
                shutil.rmtree(layout.release)
            os.replace(backup, layout.release)
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    print(
        "[conversion] activated reproduced Nano and EarTTS after full signed-release verification",
        flush=True,
    )
    return {
        "artifact_materialization": "source-converted",
        "reproduction_sha256": report["sha256"],
    }


def _convert_from_source(
    args: argparse.Namespace,
    config: dict[str, Any],
    layout: Layout,
    asr_image_id: str,
) -> dict[str, Any]:
    remaining = config["deployment"]["minimum_conversion_remaining_gib"]
    output_estimate = config["deployment"]["conversion_output_estimate_gib"]
    activation_copy = _tree_copy_gib((layout.release / "nano", layout.release / "eartts"))
    require_free_space(
        layout.cache,
        remaining + output_estimate + activation_copy,
        "source conversion and independent activation copy",
    )

    corpus = layout.cache / "conversion-corpus"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    if not corpus.exists():
        temporary = Path(tempfile.mkdtemp(prefix=f".{corpus.name}.tmp-", dir=corpus.parent))
        try:
            with tarfile.open(layout.release / "calibration/replay.tar") as archive:
                archive.extractall(temporary, filter="data")
            os.replace(temporary, corpus)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    work = layout.cache / "conversion"
    work.mkdir(parents=True, exist_ok=True)
    (work / "external-asr").mkdir(exist_ok=True)
    image = config["image"]["runtime"]
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{layout.parent}:/models/parent:ro",
        "-v",
        f"{layout.nano_skeleton}:/models/NVIDIA-Nemotron-Nano-9B-v2:ro",
        "-v",
        f"{corpus}:/corpus:ro",
        "-v",
        f"{work}:/work",
        image,
        "python3",
        "/workspace/project/tools/conversion/reproduce_release_artifacts.py",
        "--speech-root",
        "/opt/Speech",
        "--checkpoint-root",
        "/models/parent",
        "--nano-skeleton",
        "/models/NVIDIA-Nemotron-Nano-9B-v2",
        "--calibration-root",
        "/corpus/calibration/0",
        "--calibration-root",
        "/corpus/calibration/1",
        "--evaluation-root",
        "/corpus/evaluation/0",
        "--work-root",
        "/work",
    ]
    _run(command)
    finalized = _finalize_source_asr_stage(work, asr_image_id)
    base_reproduction_path = work / "release/reproduction.json"
    reproduction_path = work / "external-asr/reproduction.json"
    reproduction = json.loads(base_reproduction_path.read_text(encoding="utf-8"))
    reproduction.pop("sha256", None)
    reproduction["asr_evaluator_image_id"] = asr_image_id
    reproduction["component_gates_external_asr"] = finalized
    reproduction["sha256"] = _canonical_json_sha256(reproduction)
    atomic_json(reproduction_path, reproduction)
    return _activate_converted_release(
        layout,
        work / "release",
        config["artifacts"]["release"]["release_sha256"],
        asr_image_id,
    )


def command_bootstrap(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    deployment = config["deployment"]
    print("Voicechat bootstrap estimates:")
    print("  downloads: about 65 GiB (parent + converted release + Pocket assets)")
    print("  image build: 45-120 minutes on the first run; BuildKit cache is retained")
    print(f"  cache root: {layout.cache}")
    require_free_space(layout.cache, deployment["minimum_download_free_gib"], "artifact download")
    _executable("uv")
    _executable("docker")
    _sync_host(layout, args.offline)
    _download_artifacts(args, layout)
    # Artifact download/verification above guarantees that the active release
    # exists and is valid. It is now safe to discard any hidden backup left by
    # a process or machine failure during an earlier two-directory swap.
    _remove_stale_conversion_activations(layout)
    require_free_space(layout.cache, deployment["minimum_build_free_gib"], "runtime build")
    _build_image(config, layout, offline=args.offline, no_cache=args.no_cache)
    conversion_state: dict[str, Any] = {"artifact_materialization": "downloaded"}
    if args.convert_from_source:
        asr_image_id = _ensure_asr_evaluator(config, offline=args.offline, no_cache=args.no_cache)
        conversion_state = _convert_from_source(args, config, layout, asr_image_id)
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", config["image"]["runtime"]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = {
        "schema": 1,
        "candidate": config["candidate"],
        "release_sha256": config["artifacts"]["release"]["release_sha256"],
        "release_revision": config["artifacts"]["release"]["revision"],
        "runtime_image": config["image"]["runtime"],
        "runtime_image_id": image_id,
        **conversion_state,
    }
    atomic_json(layout.state_file, state)
    print("Bootstrap complete. Run `./voicechat up`.")
    return 0


def model_container_command(
    config: dict[str, Any],
    layout: Layout,
    model_port: int,
    *,
    trace_pad_pair: bool = False,
    trace_post_fc_tokens: int = 0,
    trace_post_fc_agent_logits: int = 0,
    trace_fc_async_heartbeat: bool = False,
    trace_fc_async_benchmark: bool = False,
    runtime_image_id: str | None = None,
) -> list[str]:
    environment = config["runtime"]["environment"] | {
        "HF_HOME": "/models/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/models/huggingface/hub",
        # Make diagnostic and release evidence self-identifying. Docker does
        # not expose the originating image tag from inside a running container.
        "VOICECHAT_RUNTIME_IMAGE": config["image"]["runtime"],
        # Transformers materializes trusted local model code in a dynamic-module
        # cache even in offline mode. Keep downloaded assets read-only while
        # giving that derived, per-container state an ephemeral writable home.
        "HF_MODULES_CACHE": "/tmp/voicechat-hf-modules",
    }
    if runtime_image_id is not None:
        environment["VOICECHAT_RUNTIME_IMAGE_ID"] = runtime_image_id
    if trace_pad_pair:
        environment["VOICECHAT_NANO_PAD_PAIR_TRACE"] = "1"
    if not 0 <= trace_post_fc_tokens <= 32:
        raise ValueError("trace_post_fc_tokens must be from 0 through 32")
    if trace_post_fc_tokens:
        environment["S2S_POST_FC_TOKEN_TRACE_FRAMES"] = str(trace_post_fc_tokens)
    if not 0 <= trace_post_fc_agent_logits <= 20:
        raise ValueError("trace_post_fc_agent_logits must be from 0 through 20")
    if trace_post_fc_agent_logits and not trace_post_fc_tokens:
        raise ValueError("trace_post_fc_agent_logits requires a nonzero post-FC token trace")
    if trace_post_fc_agent_logits:
        environment["S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK"] = str(trace_post_fc_agent_logits)
    if trace_fc_async_heartbeat:
        environment["S2S_FC_ASYNC_HEARTBEAT"] = "1"
    if trace_fc_async_benchmark:
        environment["S2S_FC_ASYNC_BENCHMARK"] = "1"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        config["deployment"]["container_name"],
        "--init",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--security-opt",
        "label=disable",
        "-p",
        f"127.0.0.1:{model_port}:8786",
        "-v",
        f"{layout.parent}:/models/voicechat:ro",
        "-v",
        f"{layout.nano_skeleton}:/models/NVIDIA-Nemotron-Nano-9B-v2:ro",
        "-v",
        f"{layout.release}:/models/derived:ro",
        "-v",
        f"{layout.hf_home}:/models/huggingface:ro",
        "-v",
        f"{layout.traces / 'model'}:/trace",
    ]
    for name, value in sorted(environment.items()):
        command.extend(["-e", f"{name}={value}"])
    command.extend(
        [
            config["image"]["runtime"],
            "python3",
            "-m",
            "nemotron_voicechat_runtime.server",
            "--speech-root",
            "/opt/Speech",
            "--checkpoint-root",
            "/models/voicechat",
            "--hf-skeleton",
            "/models/NVIDIA-Nemotron-Nano-9B-v2",
            "--vllm-manifest",
            "/models/derived/manifests/release.json",
            "--nano-vllm-path",
            "/models/derived/nano",
            "--eartts-vllm-path",
            "/models/derived/eartts",
            "--speaker-name",
            "Aria",
            "--host",
            "0.0.0.0",
            "--port",
            "8786",
            "--trace-dir",
            "/trace",
        ]
    )
    return command


def _wait_health(url: str, process: subprocess.Popen, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model container exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            if payload.get("status") == "ready" and payload.get("typed_input", {}).get("ready"):
                return payload
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for model health at {url}: {last_error}")


def _stop_stack(
    model: subprocess.Popen | None, pipecat: subprocess.Popen | None, name: str
) -> None:
    _stop_pipecat(pipecat)
    if model is not None and model.poll() is None:
        subprocess.run(["docker", "stop", "--timeout", "20", name], check=False)
        try:
            model.wait(timeout=25)
        except subprocess.TimeoutExpired:
            model.kill()


def _stop_pipecat(pipecat: subprocess.Popen | None) -> None:
    if pipecat is None or pipecat.poll() is not None:
        return
    try:
        pipecat.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        pipecat.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            pipecat.kill()
        except ProcessLookupError:
            return
        pipecat.wait(timeout=5)


def _pipecat_command(layout: Layout, host: str, port: int) -> list[str]:
    return [
        str(layout.cache / "venv/bin/python"),
        "-m",
        "nemotron_voicechat_pipecat.demo",
        "-t",
        "webrtc",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _start_pipecat(layout: Layout, model_port: int, host: str, port: int) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["NEMOTRON_VOICECHAT_WS_URL"] = f"ws://127.0.0.1:{model_port}/v1/realtime"
    return subprocess.Popen(
        _pipecat_command(layout, host, port),
        cwd=REPO_ROOT,
        env=environment,
    )


def _process_start_time(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        return (proc_root / str(pid) / "stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        return None


def _owned_pipecat_process(
    pid: int, proc_root: Path = Path("/proc"), expected_start_time: str | None = None
) -> bool:
    try:
        command = (proc_root / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"-m nemotron_voicechat_pipecat.demo" in command and (
        expected_start_time is None
        or _process_start_time(pid, proc_root=proc_root) == expected_start_time
    )


def _owned_launcher_process(
    pid: int, proc_root: Path = Path("/proc"), expected_start_time: str | None = None
) -> bool:
    try:
        arguments = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return (
        any(
            argument == b"-m"
            and index + 1 < len(arguments)
            and arguments[index + 1] == b"nemotron_voicechat_runtime.cli"
            for index, argument in enumerate(arguments)
        )
        and b"up" in arguments
        and (
            expected_start_time is None
            or _process_start_time(pid, proc_root=proc_root) == expected_start_time
        )
    )


def _bot_source_snapshot(root: Path = BOT_SOURCE_ROOT) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap, stable fingerprint for the bot's Python source tree."""

    snapshot: list[tuple[str, int, int]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            # Editors commonly replace files atomically. The next poll observes
            # the replacement and extends the debounce window.
            continue
        snapshot.append((str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size))
    return tuple(snapshot)


def _runtime_state(
    *,
    name: str,
    model_port: int,
    pipecat_host: str,
    pipecat_port: int,
    pipecat: subprocess.Popen | None,
    bot_generation: int,
    bot_status: str = "running",
    bot_exit_code: int | None = None,
) -> dict[str, Any]:
    state = {
        "schema": 2,
        "launcher_pid": os.getpid(),
        "launcher_start_time": _process_start_time(os.getpid()),
        "bot_generation": bot_generation,
        "bot_status": bot_status,
        "container_name": name,
        "model_port": model_port,
        "pipecat_host": pipecat_host,
        "pipecat_port": pipecat_port,
    }
    if pipecat is not None:
        state.update(
            {
                "pipecat_pid": pipecat.pid,
                "pipecat_start_time": _process_start_time(pipecat.pid),
            }
        )
    if bot_exit_code is not None:
        state["bot_exit_code"] = bot_exit_code
    return state


def _stop_orphan_pipecat(pid: int, expected_start_time: str | None = None) -> None:
    if not _owned_pipecat_process(pid, expected_start_time=expected_start_time):
        print(f"Refusing to signal PID {pid}: it is not the recorded Pipecat bot")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _owned_pipecat_process(pid, expected_start_time=expected_start_time):
            return
        time.sleep(0.1)
    if _owned_pipecat_process(pid, expected_start_time=expected_start_time):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def command_up(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    ready, reason = _bootstrap_ready(config, layout)
    if not ready:
        raise RuntimeError(f"stack is not bootstrapped: {reason}; run `./voicechat bootstrap`")
    deployment = config["deployment"]
    name = deployment["container_name"]
    if _container_exists(name):
        raise RuntimeError(f"stale container {name} exists; run `./voicechat down` first")
    model_port = args.model_port or deployment["model_port"]
    if not _port_available("127.0.0.1", model_port):
        raise RuntimeError(f"model port 127.0.0.1:{model_port} is already in use")
    pipecat_host = args.host or deployment["pipecat_host"]
    pipecat_port = args.port or deployment["pipecat_port"]
    if not _port_available(pipecat_host, pipecat_port):
        raise RuntimeError(f"Pipecat port {pipecat_host}:{pipecat_port} is already in use")
    (layout.traces / "model").mkdir(parents=True, exist_ok=True)
    model: subprocess.Popen | None = None
    pipecat: subprocess.Popen | None = None
    bot_generation = 0
    restart_requested = False
    reload_snapshot: tuple[tuple[str, int, int], ...] | None = None
    reload_after: float | None = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigusr1 = signal.getsignal(signal.SIGUSR1)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    def request_bot_restart(_signum, _frame):
        nonlocal restart_requested
        restart_requested = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGUSR1, request_bot_restart)
    try:
        bootstrap_state = json.loads(layout.state_file.read_text(encoding="utf-8"))
        runtime_image_id = bootstrap_state.get("runtime_image_id")
        if not valid_runtime_image_id(runtime_image_id):
            raise RuntimeError("bootstrap state has no immutable runtime image ID")
        model = subprocess.Popen(
            model_container_command(
                config,
                layout,
                model_port,
                trace_pad_pair=args.trace_pad_pair,
                trace_post_fc_tokens=args.trace_post_fc_tokens,
                trace_post_fc_agent_logits=args.trace_post_fc_agent_logits,
                trace_fc_async_heartbeat=getattr(args, "trace_fc_async_heartbeat", False),
                trace_fc_async_benchmark=getattr(args, "trace_fc_async_benchmark", False),
                runtime_image_id=runtime_image_id,
            )
        )
        health = _wait_health(
            f"http://127.0.0.1:{model_port}/health",
            model,
            deployment["health_timeout_seconds"],
        )
        print(f"Model ready: protocol={health.get('protocol')} typed_input=ready", flush=True)
        pipecat = _start_pipecat(layout, model_port, pipecat_host, pipecat_port)
        atomic_json(
            layout.runtime_state_file,
            _runtime_state(
                name=name,
                model_port=model_port,
                pipecat_host=pipecat_host,
                pipecat_port=pipecat_port,
                pipecat=pipecat,
                bot_generation=bot_generation,
            ),
        )
        print(
            f"Pipecat Playground: http://{pipecat_host}:{pipecat_port}/client/",
            flush=True,
        )
        if args.reload_bot:
            reload_snapshot = _bot_source_snapshot()
            print(f"Watching {BOT_SOURCE_ROOT} for bot source changes", flush=True)
        while True:
            if args.reload_bot:
                current_snapshot = _bot_source_snapshot()
                if current_snapshot != reload_snapshot:
                    reload_snapshot = current_snapshot
                    reload_after = time.monotonic() + BOT_RELOAD_DEBOUNCE_SECONDS
                elif reload_after is not None and time.monotonic() >= reload_after:
                    reload_after = None
                    restart_requested = True
                    print("Bot source changed; restarting Pipecat...", flush=True)

            if restart_requested:
                restart_requested = False
                _stop_pipecat(pipecat)
                _wait_ports_available([(pipecat_host, pipecat_port)])
                if model.poll() is not None:
                    return model.returncode or 1
                pipecat = _start_pipecat(layout, model_port, pipecat_host, pipecat_port)
                bot_generation += 1
                atomic_json(
                    layout.runtime_state_file,
                    _runtime_state(
                        name=name,
                        model_port=model_port,
                        pipecat_host=pipecat_host,
                        pipecat_port=pipecat_port,
                        pipecat=pipecat,
                        bot_generation=bot_generation,
                    ),
                )
                print(
                    f"Pipecat bot restarted (generation {bot_generation}); "
                    "reconnect the Playground session",
                    flush=True,
                )
                continue

            model_status = model.poll()
            pipecat_status = pipecat.poll() if pipecat is not None else None
            if model_status is not None:
                return model_status or 1
            if pipecat_status is not None:
                if args.reload_bot or bot_generation > 0:
                    print(
                        f"Pipecat bot exited with status {pipecat_status}; model remains ready. "
                        "Fix the bot and run `./voicechat restart-bot` or save a watched file.",
                        file=sys.stderr,
                        flush=True,
                    )
                    atomic_json(
                        layout.runtime_state_file,
                        _runtime_state(
                            name=name,
                            model_port=model_port,
                            pipecat_host=pipecat_host,
                            pipecat_port=pipecat_port,
                            pipecat=None,
                            bot_generation=bot_generation,
                            bot_status="exited",
                            bot_exit_code=pipecat_status,
                        ),
                    )
                    pipecat = None
                    time.sleep(0.5)
                    continue
                return pipecat_status or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            _stop_stack(model, pipecat, name)
            _wait_ports_available([("127.0.0.1", model_port), (pipecat_host, pipecat_port)])
        finally:
            layout.runtime_state_file.unlink(missing_ok=True)
            signal.signal(signal.SIGUSR1, previous_sigusr1)


def command_restart_bot(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    layout = default_layout(args.cache_root, args.trace_root)
    try:
        state = json.loads(layout.runtime_state_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("the stack is not running; start it with `./voicechat up`") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"runtime state is unreadable: {exc}") from exc

    launcher_pid = state.get("launcher_pid")
    launcher_start_time = state.get("launcher_start_time")
    generation = state.get("bot_generation")
    if (
        state.get("schema") != 2
        or not isinstance(launcher_pid, int)
        or not isinstance(launcher_start_time, str)
        or not isinstance(generation, int)
    ):
        raise RuntimeError(
            "the running stack predates bot reload support; restart `./voicechat up` once"
        )
    if not _owned_launcher_process(launcher_pid, expected_start_time=launcher_start_time):
        raise RuntimeError("runtime state does not identify a live `voicechat up` launcher")

    try:
        os.kill(launcher_pid, signal.SIGUSR1)
    except OSError as exc:
        raise RuntimeError(f"could not signal the `voicechat up` launcher: {exc}") from exc

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not _owned_launcher_process(launcher_pid, expected_start_time=launcher_start_time):
            raise RuntimeError("the `voicechat up` launcher exited while restarting the bot")
        try:
            current = json.loads(layout.runtime_state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        new_pid = current.get("pipecat_pid")
        new_start_time = current.get("pipecat_start_time")
        current_generation = current.get("bot_generation")
        host = current.get("pipecat_host")
        port = current.get("pipecat_port")
        if (
            current.get("launcher_pid") == launcher_pid
            and isinstance(current_generation, int)
            and current_generation > generation
            and isinstance(new_pid, int)
            and isinstance(new_start_time, str)
            and isinstance(host, str)
            and isinstance(port, int)
            and _owned_pipecat_process(new_pid, expected_start_time=new_start_time)
            and not _port_available(host, port)
        ):
            print(
                f"Pipecat bot restarted (generation {current_generation}); "
                "reconnect the Playground session"
            )
            return 0
        time.sleep(0.1)
    raise RuntimeError(f"timed out after {args.timeout:g}s waiting for the Pipecat bot restart")


def command_down(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    name = config["deployment"]["container_name"]
    if layout.runtime_state_file.is_file():
        try:
            state = json.loads(layout.runtime_state_file.read_text(encoding="utf-8"))
            if state.get("container_name") == name and isinstance(state.get("pipecat_pid"), int):
                _stop_orphan_pipecat(
                    state["pipecat_pid"], expected_start_time=state.get("pipecat_start_time")
                )
        except (OSError, ValueError) as exc:
            print(f"Runtime-state cleanup skipped: {exc}")
        finally:
            layout.runtime_state_file.unlink(missing_ok=True)
    if not _container_exists(name):
        print(f"No stale container: {name}")
        return 0
    _run(["docker", "rm", "-f", name])
    return 0


def command_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    ready, reason = _bootstrap_ready(config, layout)
    print(f"bootstrap: {'ready' if ready else 'not ready'} ({reason})")
    name = config["deployment"]["container_name"]
    subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^/{name}$",
            "--format",
            "{{.Names}}\t{{.Status}}\t{{.Ports}}",
        ],
        check=False,
    )
    return 0 if ready else 1


def command_test(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    python = layout.cache / "venv/bin/python"
    if not python.is_file():
        raise RuntimeError("host environment is missing; run `./voicechat bootstrap`")
    command = [str(python), "-m", "pytest", "-q", "--ignore", "tests/e2e"]
    if not args.live:
        return _run(command, cwd=REPO_ROOT).returncode
    status = _run(command, cwd=REPO_ROOT).returncode
    if status:
        return status
    asr_image_id = _ensure_asr_evaluator(config, offline=False, no_cache=False)
    manifest = Path(args.fixture_manifest).expanduser().resolve() if args.fixture_manifest else None
    if manifest is not None and not manifest.is_file():
        raise RuntimeError(f"fixture manifest does not exist: {manifest}")
    browser_mic = (
        Path(args.browser_mic_wav).expanduser().resolve() if args.browser_mic_wav else None
    )
    for name, path in (("browser microphone WAV", browser_mic),):
        if path is not None and not path.is_file():
            raise RuntimeError(f"{name} does not exist: {path}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else layout.traces / f"live-suite-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    return _run(
        [
            str(python),
            str(REPO_ROOT / "tools/qualification/run_live_suite.py"),
            "--python",
            str(python),
            "--cache-root",
            str(layout.cache),
            "--trace-root",
            str(layout.traces),
            "--output",
            str(output),
            "--asr-evaluator-image",
            asr_image_id,
            "--runtime-image",
            config["image"]["runtime"],
            "--duration-seconds",
            str(args.duration_seconds),
            "--restart-cycles",
            str(args.restart_cycles),
            "--model-port",
            str(args.model_port or config["deployment"]["model_port"]),
            "--pipecat-port",
            str(args.port or config["deployment"]["pipecat_port"]),
            "--ready-budget-seconds",
            str(config["deployment"]["cold_start_budget_seconds"]),
            (
                "--expect-session-limit"
                if args.expect_session_limit
                else "--no-expect-session-limit"
            ),
            *(["--fixture-manifest", str(manifest)] if manifest else []),
            *(["--browser-mic-wav", str(browser_mic)] if browser_mic else []),
        ],
        cwd=REPO_ROOT,
    ).returncode


def command_release(args: argparse.Namespace) -> int:
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise RuntimeError(
            "release is maintainer-only; pass arguments for tools/release/upload_hf.py"
        )
    layout = default_layout(args.cache_root, args.trace_root)
    return _run(
        [
            str(layout.cache / "venv/bin/python"),
            str(REPO_ROOT / "tools/release/upload_hf.py"),
            *arguments,
        ],
        cwd=REPO_ROOT,
    ).returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--cache-root")
    result.add_argument("--trace-root")
    subparsers = result.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--offline", action="store_true")
    bootstrap.add_argument("--convert-from-source", action="store_true")
    bootstrap.add_argument("--no-cache", action="store_true", help="force the final release build")
    bootstrap.set_defaults(handler=command_bootstrap)
    subparsers.add_parser("doctor").set_defaults(handler=command_doctor)
    up = subparsers.add_parser("up")
    up.add_argument("--host")
    up.add_argument("--port", type=int)
    up.add_argument("--model-port", type=int)
    up.add_argument(
        "--reload-bot",
        action="store_true",
        help="restart only Pipecat when its Python source changes",
    )
    up.add_argument(
        "--trace-pad-pair",
        action="store_true",
        help="capture diagnostic-only Nano pair-scheduler decisions in model traces",
    )
    up.add_argument(
        "--trace-post-fc-tokens",
        type=int,
        default=0,
        metavar="FRAMES",
        help="capture the forced BOS and a bounded post-FC raw/effective token window",
    )
    up.add_argument(
        "--trace-post-fc-agent-logits",
        type=int,
        default=0,
        metavar="TOP_K",
        help="capture bounded agent-head top-k values on post-FC token-trace positions",
    )
    up.add_argument(
        "--trace-fc-async-heartbeat",
        action="store_true",
        help="publish FC async stage heartbeats and enable SIGUSR1 all-thread dumps",
    )
    up.add_argument(
        "--trace-fc-async-benchmark",
        action="store_true",
        help="publish the opt-in FC async per-phase and per-position latency ledger",
    )
    up.set_defaults(handler=command_up)
    restart_bot = subparsers.add_parser(
        "restart-bot", help="restart Pipecat without restarting the model container"
    )
    restart_bot.add_argument("--timeout", type=float, default=BOT_RESTART_TIMEOUT_SECONDS)
    restart_bot.set_defaults(handler=command_restart_bot)
    subparsers.add_parser("down").set_defaults(handler=command_down)
    subparsers.add_parser("status").set_defaults(handler=command_status)
    test = subparsers.add_parser("test")
    test.add_argument("--live", action="store_true")
    test.add_argument("--fixture-manifest", type=Path)
    test.add_argument("--browser-mic-wav", type=Path)
    test.add_argument("--duration-seconds", type=float, default=1200)
    test.add_argument("--restart-cycles", type=int, default=3)
    test.add_argument("--model-port", type=int)
    test.add_argument("--port", type=int)
    test.add_argument("--output", type=Path)
    test.add_argument("--expect-session-limit", action=argparse.BooleanOptionalAction, default=True)
    test.set_defaults(handler=command_test)
    release = subparsers.add_parser("release")
    release.add_argument("arguments", nargs=argparse.REMAINDER)
    release.set_defaults(handler=command_release)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        status = args.handler(args)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(status)


if __name__ == "__main__":
    main()
