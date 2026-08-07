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


def start(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config["image"]["runtime"] = args.image
    config["deployment"]["container_name"] = args.name
    for assignment in args.env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --env assignment: {assignment!r}")
        config["runtime"]["environment"][name] = value
    layout = default_layout(args.cache_root, str(args.trace_root))
    (layout.traces / "model").mkdir(parents=True, exist_ok=True)
    command = model_container_command(config, layout, args.port, trace_pad_pair=args.trace_pad_pair)
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
            if health.get("status") == "ready" and health.get("typed_input", {}).get(
                "ready"
            ):
                args.health.write_text(
                    json.dumps(health, indent=2, sort_keys=True) + "\n"
                )
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
