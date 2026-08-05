#!/usr/bin/env python3
"""Qualify foreground teardown, immediate restart, and SIGKILL recovery."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class RunningLauncher:
    def __init__(self, process: subprocess.Popen[bytes], log: Any) -> None:
        self.process = process
        self.log = log


def wait_json(url: str, process: subprocess.Popen, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"launcher exited before readiness: {process.returncode}")
        try:
            with NO_PROXY_OPENER.open(url, timeout=2) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"readiness timeout for {url}: {last_error}")


def wait_http(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"launcher exited before Pipecat readiness: {process.returncode}")
        try:
            with NO_PROXY_OPENER.open(url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError(f"readiness timeout for {url}")


def host_non_loopback_addresses() -> list[str]:
    values: list[str] = []
    try:
        interfaces = json.loads(
            subprocess.run(
                ["ip", "-j", "-4", "address", "show"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        for interface in interfaces:
            for info in interface.get("addr_info", []):
                address = info.get("local")
                if (
                    info.get("family") == "inet"
                    and isinstance(address, str)
                    and not address.startswith("127.")
                    and address not in values
                ):
                    values.append(address)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        address = info[4][0]
        if not address.startswith("127.") and address not in values:
            values.append(address)
    return values


def assert_model_not_exposed(port: int) -> None:
    addresses = host_non_loopback_addresses()
    if not addresses:
        raise RuntimeError(
            "cannot verify model isolation: host has no discoverable non-loopback IPv4"
        )
    for address in addresses:
        try:
            with NO_PROXY_OPENER.open(f"http://{address}:{port}/health", timeout=1):
                pass
        except urllib.error.HTTPError:
            raise RuntimeError(
                f"raw model port is reachable through non-loopback {address}:{port}"
            ) from None
        except (OSError, urllib.error.URLError):
            continue
        raise RuntimeError(f"raw model port is reachable through non-loopback {address}:{port}")


def launcher_command(args: argparse.Namespace) -> list[str]:
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


def start_and_wait(args: argparse.Namespace, log_path: Path) -> tuple[RunningLauncher, float]:
    started = time.monotonic()
    log = log_path.open("wb")
    process: subprocess.Popen[bytes] = subprocess.Popen(
        launcher_command(args),
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    running = RunningLauncher(process=process, log=log)
    try:
        health = wait_json(
            f"http://127.0.0.1:{args.model_port}/health", process, args.startup_timeout
        )
        if (
            health.get("status") != "ready"
            or health.get("protocol") != {"name": "voicechat.realtime", "version": 3}
            or health.get("auth_required") is not False
            or not health.get("typed_input", {}).get("ready")
        ):
            raise RuntimeError(f"unexpected model health contract: {health}")
        wait_http(
            f"http://127.0.0.1:{args.pipecat_port}/client/",
            process,
            args.startup_timeout,
        )
        assert_model_not_exposed(args.model_port)
        return running, time.monotonic() - started
    except Exception:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        log.close()
        raise


def finish(running: RunningLauncher, sig: signal.Signals, timeout: float = 90) -> int:
    running.process.send_signal(sig)
    try:
        try:
            status = running.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            running.process.kill()
            running.process.wait(timeout=10)
            raise
    finally:
        running.log.close()
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    cycles = []
    for index in range(args.cycles):
        running, ready_seconds = start_and_wait(args, args.output / f"cycle-{index + 1}.log")
        if ready_seconds > args.ready_budget_seconds:
            finish(running, signal.SIGINT)
            raise RuntimeError(
                f"SIGINT cycle {index + 1} readiness {ready_seconds:.3f}s exceeded "
                f"{args.ready_budget_seconds:.3f}s budget"
            )
        status = finish(running, signal.SIGINT)
        if status != 0:
            raise RuntimeError(f"SIGINT cycle {index + 1} exited {status}")
        cycles.append({"cycle": index + 1, "ready_seconds": round(ready_seconds, 3)})

    running, ready_seconds = start_and_wait(args, args.output / "sigkill.log")
    if ready_seconds > args.ready_budget_seconds:
        finish(running, signal.SIGINT)
        raise RuntimeError(
            f"SIGKILL setup readiness {ready_seconds:.3f}s exceeded "
            f"{args.ready_budget_seconds:.3f}s budget"
        )
    running.process.kill()
    running.process.wait(timeout=10)
    running.log.close()
    subprocess.run(
        [
            str(REPO_ROOT / "voicechat"),
            "--cache-root",
            str(args.cache_root),
            "--trace-root",
            str(args.trace_root),
            "down",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    recovered, recovery_ready = start_and_wait(args, args.output / "recovery.log")
    if recovery_ready > args.ready_budget_seconds:
        finish(recovered, signal.SIGINT)
        raise RuntimeError(
            f"recovery readiness {recovery_ready:.3f}s exceeded "
            f"{args.ready_budget_seconds:.3f}s budget"
        )
    status = finish(recovered, signal.SIGINT)
    if status != 0:
        raise RuntimeError(f"post-SIGKILL recovery exited {status}")
    report = {
        "passed": True,
        "sigint_cycles": cycles,
        "sigkill_initial_ready_seconds": round(ready_seconds, 3),
        "recovery_ready_seconds": round(recovery_ready, 3),
        "raw_model_loopback_only": True,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--model-port", type=int, default=8786)
    parser.add_argument("--pipecat-port", type=int, default=7860)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--ready-budget-seconds", type=float, required=True)
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
