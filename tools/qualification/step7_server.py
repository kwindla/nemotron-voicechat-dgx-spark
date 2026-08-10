#!/usr/bin/env python3
"""Launch or stop the frozen server contract with a Step-7 overlay image."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from nemotron_voicechat_runtime.artifacts import default_layout, load_config
from nemotron_voicechat_runtime.cli import model_container_command

KIB_PER_GIB = 1024 * 1024


def require_launch_headroom() -> None:
    running = subprocess.run(
        ["docker", "ps", "-q"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if running:
        raise RuntimeError(f"Step 7 requires an empty Docker runtime before launch: {running}")
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available_kib <= 100 * KIB_PER_GIB:
        raise RuntimeError(
            "Step 7 requires host MemAvailable > 100 GiB before launch; "
            f"found {available_kib / KIB_PER_GIB:.3f} GiB"
        )


def start(args: argparse.Namespace) -> None:
    if not 0.0 < args.nano_gpu_memory_utilization <= 0.40:
        raise ValueError("--nano-gpu-memory-utilization must be in (0, 0.40] for Step 7")
    if args.memory.lower() != "90g" or args.memory_swap.lower() != "90g":
        raise ValueError("Step 7 requires --memory 90g --memory-swap 90g")
    require_launch_headroom()
    config = load_config(args.config)
    config["image"]["runtime"] = args.image
    config["deployment"]["container_name"] = args.name
    config["runtime"]["environment"]["VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION"] = str(
        args.nano_gpu_memory_utilization
    )
    config["runtime"]["environment"]["VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION"] = str(
        args.eartts_gpu_memory_utilization
    )
    for assignment in args.env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --env assignment: {assignment!r}")
        config["runtime"]["environment"][name] = value
    layout = default_layout(args.cache_root, str(args.trace_root))
    (layout.traces / "model").mkdir(parents=True, exist_ok=True)
    command = model_container_command(config, layout, args.port, trace_pad_pair=args.trace_pad_pair)
    command[2:2] = [
        "--memory",
        args.memory,
        "--memory-swap",
        args.memory_swap,
    ]
    command.remove("--rm")
    command.insert(2, "-d")
    container_id = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()
    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", args.name],
            capture_output=True,
            text=True,
        )
        if state.returncode or state.stdout.strip() != "true":
            raise RuntimeError(f"container exited during startup: {args.name}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=2
            ) as response:
                health = json.load(response)
            if health.get("status") == "ready" and health.get("typed_input", {}).get("ready"):
                args.health.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n")
                print(json.dumps({"container_id": container_id, "health": health}))
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"health timeout: {last_error}")


def stop(args: argparse.Namespace) -> None:
    subprocess.run(["docker", "stop", "--timeout", "20", args.name], check=False)
    subprocess.run(["docker", "rm", args.name], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("start")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--image", required=True)
    launch.add_argument("--name", required=True)
    launch.add_argument("--port", type=int, default=8786)
    launch.add_argument("--cache-root")
    launch.add_argument("--trace-root", type=Path, required=True)
    launch.add_argument("--health", type=Path, required=True)
    launch.add_argument("--timeout", type=float, default=900)
    launch.add_argument("--nano-gpu-memory-utilization", type=float, required=True)
    launch.add_argument("--eartts-gpu-memory-utilization", type=float, default=0.10)
    launch.add_argument("--memory", default="90g")
    launch.add_argument("--memory-swap", default="90g")
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("--trace-pad-pair", action="store_true")
    launch.set_defaults(func=start)
    terminate = subparsers.add_parser("stop")
    terminate.add_argument("--name", required=True)
    terminate.set_defaults(func=stop)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
