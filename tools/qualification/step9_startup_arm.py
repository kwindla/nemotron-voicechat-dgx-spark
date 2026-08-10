#!/usr/bin/env python3
"""Run one guarded, sequential Step-9 cold-to-ready startup arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nemotron_voicechat_runtime.artifacts import default_layout, load_config
from nemotron_voicechat_runtime.cli import model_container_command

KIB_PER_GIB = 1024 * 1024


def mem_available_gib() -> float:
    value = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    return value / KIB_PER_GIB


def preflight() -> dict[str, object]:
    running = subprocess.run(
        ["docker", "ps", "-q"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    available = mem_available_gib()
    if running:
        raise RuntimeError(f"startup arm requires docker ps -q empty: {running}")
    if available <= 100.0:
        raise RuntimeError(f"startup arm requires MemAvailable >100 GiB: {available:.3f}")
    return {"running_containers": running, "mem_available_gib": available}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--port", type=int, default=8786)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--keep-container", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    initial = preflight()

    config = load_config(args.config)
    config["image"]["runtime"] = args.image
    config["deployment"]["container_name"] = args.name
    config["runtime"]["environment"].update(
        {
            "VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION": "0.35",
            "VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION": "0.10",
            "VOICECHAT_NANO_PAIR_FULL_GRAPH": "1",
        }
    )
    for assignment in args.env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise SystemExit(f"invalid --env assignment: {assignment!r}")
        config["runtime"]["environment"][name] = value
    layout = default_layout(args.cache_root, str(args.trace_root))
    (layout.traces / "model").mkdir(parents=True, exist_ok=True)
    command = model_container_command(config, layout, args.port)
    command[2:2] = ["--memory", "90g", "--memory-swap", "90g"]
    command.remove("--rm")
    command.insert(2, "-d")

    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    container_id = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()
    watchdog = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("step7_memory_watchdog.py")),
            "--container",
            args.name,
            "--output",
            str(args.output_dir / "memory-watchdog.jsonl"),
            "--interval-seconds",
            "10",
            "--kill-below-gib",
            "15",
        ]
    )
    ready_payload = None
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", args.name],
            capture_output=True,
            text=True,
        )
        if state.returncode or state.stdout.strip() != "true":
            raise RuntimeError(f"container exited before readiness: {args.name}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=2
            ) as response:
                candidate = json.load(response)
            if candidate.get("status") == "ready" and candidate.get("typed_input", {}).get(
                "ready"
            ):
                ready_payload = candidate
                break
        except Exception:
            pass
        elapsed = (time.monotonic_ns() - launch_monotonic_ns) / 1e9
        print(
            json.dumps(
                {"event": "startup_wait", "elapsed_seconds": round(elapsed, 3)},
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(2)
    if ready_payload is None:
        subprocess.run(["docker", "kill", args.name], check=False)
        raise RuntimeError("startup health timeout")
    ready_wall_ns = time.time_ns()
    ready_monotonic_ns = time.monotonic_ns()
    host = {
        "schema": 1,
        "image": args.image,
        "container_id": container_id,
        "container_name": args.name,
        "launch_utc": datetime.fromtimestamp(
            launch_wall_ns / 1e9, tz=timezone.utc
        ).isoformat(),
        "launch_wall_ns": launch_wall_ns,
        "launch_monotonic_ns": launch_monotonic_ns,
        "ready_wall_ns": ready_wall_ns,
        "ready_monotonic_ns": ready_monotonic_ns,
        "wall_seconds": (ready_monotonic_ns - launch_monotonic_ns) / 1e9,
        "preflight": initial,
        "health": ready_payload,
    }
    (args.output_dir / "host-timeline.json").write_text(
        json.dumps(host, indent=2, sort_keys=True) + "\n"
    )
    logs = subprocess.run(
        ["docker", "logs", "--timestamps", args.name],
        check=True,
        capture_output=True,
        text=True,
    )
    (args.output_dir / "server.log").write_text(logs.stdout + logs.stderr)
    inspect = subprocess.run(
        ["docker", "inspect", args.name], check=True, capture_output=True, text=True
    ).stdout
    (args.output_dir / "container-inspect.json").write_text(inspect)
    print(json.dumps({"event": "ready", **host}, sort_keys=True), flush=True)
    if not args.keep_container:
        subprocess.run(["docker", "stop", "--timeout", "20", args.name], check=True)
        watchdog.wait(timeout=60)
        subprocess.run(["docker", "rm", args.name], check=True)


if __name__ == "__main__":
    main()
