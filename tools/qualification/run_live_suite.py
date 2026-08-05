#!/usr/bin/env python3
"""Orchestrate the DGX-only restart, browser, voice, sustained, and ASR gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from nemotron_voicechat_runtime.artifacts import atomic_json
from nemotron_voicechat_runtime.bootstrap_download import extract_replay_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_checked(
    command: list[str],
    *,
    timeout: float,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if log is None:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=timeout)
        return
    with log.open("wb") as output:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def run_browser_gate(args: argparse.Namespace, environment: dict[str, str]) -> dict[str, Any]:
    """Run the browser gate with one bounded retry and retain every outcome."""

    command = [
        str(args.python),
        "-m",
        "pytest",
        "-q",
        "tests/e2e/test_browser_smallwebrtc_e2e.py::test_live_playground_typed_turn_and_audio",
    ]
    report_path = args.output / "browser-attempts.json"
    report: dict[str, Any] = {
        "passed": False,
        "max_attempts": 2,
        "policy": "bounded retry for documented upstream wrong-tool base rate",
        "attempts": [],
    }
    for attempt in range(1, report["max_attempts"] + 1):
        log_path = args.output / f"browser-attempt-{attempt}.log"
        started = time.monotonic()
        outcome: dict[str, Any] = {
            "attempt": attempt,
            "log": log_path.name,
            "passed": False,
        }
        try:
            with log_path.open("wb") as output:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=600,
                )
            outcome["returncode"] = completed.returncode
            outcome["passed"] = completed.returncode == 0
        except subprocess.TimeoutExpired:
            outcome["timed_out"] = True
        finally:
            outcome["duration_seconds"] = round(time.monotonic() - started, 3)
            if log_path.is_file():
                outcome["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
            report["attempts"].append(outcome)
            report["passed"] = any(item["passed"] for item in report["attempts"])
            atomic_json(report_path, report)
        if outcome["passed"]:
            return report
        if attempt < report["max_attempts"]:
            wait_model_idle(args.model_port)

    raise RuntimeError(
        f"browser gate failed {report['max_attempts']} recorded attempts; "
        f"see {report_path}"
    )


def up_command(args: argparse.Namespace) -> list[str]:
    return [
        str(REPO_ROOT / "voicechat"),
        "--cache-root",
        str(args.cache_root),
        "--trace-root",
        str(args.trace_root),
        "up",
        "--model-port",
        str(args.model_port),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.pipecat_port),
    ]


def wait_for_playground(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"stack exited before Playground readiness: {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/client/", timeout=2):
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Playground readiness timed out: {last_error}")


def wait_model_idle(port: int, timeout: float = 30) -> dict[str, Any]:
    """Wait for the strict single-client model socket to become available."""

    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_health: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                last_health = json.load(response)
            if last_health.get("status") == "ready" and not last_health.get("active_client"):
                return last_health
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"model did not become idle within {timeout}s: health={last_health}, error={last_error}"
    )


def asr_command(
    args: argparse.Namespace, run_dir: Path, report_name: str
) -> tuple[list[str], Path]:
    asr_model = args.asr_model.expanduser().resolve()
    output = run_dir.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{output}:/qualification",
        "-v",
        f"{asr_model}:/models/asr.nemo:ro",
        args.runtime_image,
        "python3",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/qualification",
        "--model",
        "/models/asr.nemo",
    ]
    return command, args.output / report_name


def gpu_container_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--network",
        "none",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'parent'}:/models/voicechat:ro",
        "-v",
        (
            f"{args.cache_root / 'artifacts' / 'nano-skeleton'}:"
            "/models/NVIDIA-Nemotron-Nano-9B-v2:ro"
        ),
        "-v",
        f"{args.cache_root / 'artifacts' / 'release'}:/models/derived:ro",
        "-v",
        f"{args.asr_model.expanduser().resolve()}:/models/asr.nemo:ro",
    ]


def run_component_gates(args: argparse.Namespace) -> None:
    components = args.output / "components"
    components.mkdir(parents=True, exist_ok=True)
    corpus = args.cache_root / "qualification-corpus"
    extract_replay_corpus(args.cache_root / "artifacts" / "release", corpus)
    prefix = gpu_container_prefix(args)
    run_checked(
        [
            *prefix,
            "-v",
            f"{corpus}:/corpus:ro",
            "-v",
            f"{components}:/qualification",
            args.runtime_image,
            "python3",
            "/workspace/project/tools/conversion/nano_component_gate.py",
            "--speech-root",
            "/opt/Speech",
            "--nano-vllm-path",
            "/models/derived/nano",
            "--teacher-replay",
            "/corpus/evaluation/0",
            "--output-dir",
            "/qualification/nano",
            "--max-calls",
            "1172",
        ],
        log=args.output / "nano-component.log",
        timeout=3600,
    )
    for name, graph_arguments in (
        ("eartts-eager", []),
        ("eartts-graph", ["--no-enforce-eager"]),
    ):
        run_checked(
            [
                *prefix,
                "-v",
                f"{components}:/qualification",
                args.runtime_image,
                "python3",
                "/workspace/project/tools/conversion/eartts_component_gate.py",
                "--speech-root",
                "/opt/Speech",
                "--checkpoint-root",
                "/models/voicechat",
                "--nano-skeleton",
                "/models/NVIDIA-Nemotron-Nano-9B-v2",
                "--eartts-vllm-path",
                "/models/derived/eartts",
                "--asr-model",
                "/models/asr.nemo",
                "--output-dir",
                f"/qualification/{name}",
                *graph_arguments,
            ],
            log=args.output / f"{name}-component.log",
            timeout=3600,
        )
    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-e",
            "HF_HOME=/models/huggingface",
            "-e",
            "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
            "-v",
            f"{REPO_ROOT}:/workspace/project:ro",
            "-v",
            f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
            "-v",
            f"{components}:/qualification",
            args.runtime_image,
            "python3",
            "/workspace/project/tools/qualification/pocket_worker_gate.py",
            "--output",
            "/qualification/pocket",
        ],
        log=args.output / "pocket-component.log",
        timeout=900,
    )


def run_conversion_tensor_tests(args: argparse.Namespace) -> None:
    """Run torch-dependent conversion contracts in the pinned CUDA image.

    The Pipecat host environment deliberately has no PyTorch. These tests are
    collected as visible skips there and executed here before any GPU gate.
    """

    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--workdir",
            "/workspace/project",
            "-v",
            f"{REPO_ROOT}:/workspace/project:ro",
            args.runtime_image,
            "python3",
            "-m",
            "pytest",
            "-q",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "tests/conversion/test_nano_attribution.py",
            "tests/conversion/test_nano_gptq_calibration.py",
            "tests/conversion/test_package_replay_corpus.py",
            "tests/conversion/test_nano_replay.py",
            "tests/conversion/test_eartts_component_gate.py",
        ],
        log=args.output / "conversion-tensor-tests.log",
        timeout=300,
    )


def prepare_fixtures(args: argparse.Namespace) -> None:
    if args.fixture_manifest is None:
        run_checked(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-e",
                "HF_HOME=/models/huggingface",
                "-e",
                "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
                "-v",
                f"{REPO_ROOT}:/workspace/project:ro",
                "-v",
                f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
                "-v",
                f"{args.output}:/qualification",
                args.runtime_image,
                "python3",
                "/workspace/project/tools/qualification/generate_multiturn_fixtures.py",
                "--output",
                "/qualification/fixtures",
            ],
            log=args.output / "fixture-generation.log",
            timeout=900,
        )
        generated_manifest = args.output / "fixtures/manifest.json"
        generated = json.loads(generated_manifest.read_text(encoding="utf-8"))
        for turn in generated["turns"]:
            turn["audio_path"] = str(generated_manifest.parent / Path(turn["audio_path"]).name)
        generated["browser_microphone_wav"] = str(
            generated_manifest.parent / Path(generated["browser_microphone_wav"]).name
        )
        # Preserve the container-produced manifest as immutable evidence. The
        # host owns the run root and writes its path-remapped view alongside it.
        args.fixture_manifest = args.output / "fixture-manifest-host.json"
        args.fixture_manifest.write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if args.browser_mic_wav is None:
        configured = fixture.get("browser_microphone_wav")
        args.browser_mic_wav = (
            Path(configured) if configured else Path(fixture["turns"][0]["audio_path"])
        )


def require_browser_executable() -> Path:
    """Fail before GPU work when Playwright's pinned Chromium is absent."""

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = Path(playwright.chromium.executable_path)
    if not executable.is_file():
        raise RuntimeError(
            f"Playwright Chromium is not installed at {executable}; run "
            "`uv run --frozen playwright install chromium` before the live suite"
        )
    return executable


def run(args: argparse.Namespace) -> dict[str, Any]:
    chromium = require_browser_executable()
    args.output.mkdir(parents=True, exist_ok=False)
    run_conversion_tensor_tests(args)
    prepare_fixtures(args)
    run_component_gates(args)
    if args.restart_cycles:
        run_checked(
            [
                str(args.python),
                str(REPO_ROOT / "tools/qualification/restart_cycle_gate.py"),
                "--cache-root",
                str(args.cache_root),
                "--trace-root",
                str(args.trace_root),
                "--output",
                str(args.output / "restart-cycles"),
                "--cycles",
                str(args.restart_cycles),
                "--model-port",
                str(args.model_port),
                "--pipecat-port",
                str(args.pipecat_port),
                "--ready-budget-seconds",
                str(args.ready_budget_seconds),
            ],
            timeout=(args.restart_cycles + 2) * (args.startup_timeout + 120),
        )

    stack_log = (args.output / "stack.log").open("wb")
    stack_started = time.monotonic()
    stack: subprocess.Popen[bytes] = subprocess.Popen(
        up_command(args), cwd=REPO_ROOT, stdout=stack_log, stderr=subprocess.STDOUT
    )
    try:
        wait_for_playground(stack, args.pipecat_port, args.startup_timeout)
        stack_ready_seconds = time.monotonic() - stack_started
        if stack_ready_seconds > args.ready_budget_seconds:
            raise RuntimeError(
                f"stack readiness {stack_ready_seconds:.3f}s exceeded "
                f"{args.ready_budget_seconds:.3f}s budget"
            )
        wait_model_idle(args.model_port)
        browser_environment = os.environ.copy()
        browser_environment.update(
            {
                "VOICECHAT_LIVE_PLAYGROUND_URL": f"http://127.0.0.1:{args.pipecat_port}/client/",
                "VOICECHAT_LIVE_MIC_WAV": str(args.browser_mic_wav),
                "PLAYWRIGHT_CHROMIUM_EXECUTABLE": str(chromium),
            }
        )
        browser_report = run_browser_gate(args, browser_environment)
        wait_model_idle(args.model_port)
        run_checked(
            [
                str(args.python),
                str(REPO_ROOT / "tools/qualification/multiturn_strict_v3.py"),
                "--url",
                f"ws://127.0.0.1:{args.model_port}/v1/realtime",
                "--manifest",
                str(args.fixture_manifest),
                "--output",
                str(args.output / "multiturn"),
            ],
            log=args.output / "multiturn.log",
            timeout=900,
        )
        wait_model_idle(args.model_port)
        sustained_command = [
            str(args.python),
            str(REPO_ROOT / "tools/qualification/sustained_strict_v3.py"),
            "--url",
            f"ws://127.0.0.1:{args.model_port}/v1/realtime",
            "--output",
            str(args.output / "sustained"),
            "--duration-seconds",
            str(args.duration_seconds),
        ]
        sustained_command.append(
            "--expect-session-limit" if args.expect_session_limit else "--no-expect-session-limit"
        )
        run_checked(
            sustained_command,
            log=args.output / "sustained.log",
            timeout=args.duration_seconds + 900,
        )
    finally:
        if stack.poll() is None:
            stack.send_signal(signal.SIGINT)
            try:
                stack.wait(timeout=90)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait(timeout=10)
        stack_log.close()

    for run_dir, report_name in (
        (args.output / "multiturn", "multiturn-asr.log"),
        (args.output / "sustained", "sustained-asr.log"),
    ):
        command, report_log = asr_command(args, run_dir, report_name)
        run_checked(command, log=report_log, timeout=1800)

    report = {
        "passed": True,
        "browser": browser_report,
        "multiturn": json.loads((args.output / "multiturn/report.json").read_text()),
        "sustained": json.loads((args.output / "sustained/report.json").read_text()),
        "restart_cycles": args.restart_cycles,
        "cold_start_to_playground_ready_seconds": round(stack_ready_seconds, 3),
        "cold_start_budget_seconds": args.ready_budget_seconds,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--browser-mic-wav", type=Path)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--duration-seconds", type=float, default=1200)
    parser.add_argument("--restart-cycles", type=int, default=3)
    parser.add_argument("--model-port", type=int, default=8786)
    parser.add_argument("--pipecat-port", type=int, default=7860)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--ready-budget-seconds", type=float, required=True)
    parser.add_argument(
        "--expect-session-limit", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
