#!/usr/bin/env python3
"""Kill a Step-7 container if host MemAvailable crosses the safety floor."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

KIB_PER_GIB = 1024 * 1024


def memory_snapshot() -> dict[str, int | float | str]:
    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, separator, rest = line.partition(":")
        if separator and name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            fields[name] = int(rest.split()[0])
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_s": time.monotonic(),
        **{f"{name}_kib": value for name, value in fields.items()},
        "mem_available_gib": fields["MemAvailable"] / KIB_PER_GIB,
    }


def container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--kill-below-gib", type=float, default=15.0)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    deadline = time.monotonic() + args.wait_seconds
    while not container_running(args.container):
        if time.monotonic() >= deadline:
            append_event(
                args.output,
                {
                    "event": "container_not_started",
                    "container": args.container,
                    **memory_snapshot(),
                },
            )
            raise SystemExit(f"container did not start: {args.container}")
        time.sleep(0.25)

    listed = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if listed != [args.container]:
        append_event(
            args.output,
            {
                "event": "concurrent_container_abort",
                "container": args.container,
                "running_containers": listed,
                **memory_snapshot(),
            },
        )
        subprocess.run(["docker", "kill", args.container], check=False)
        raise SystemExit(f"unexpected concurrent containers: {listed}")

    while container_running(args.container):
        snapshot = memory_snapshot()
        event = {"event": "sample", "container": args.container, **snapshot}
        append_event(args.output, event)
        if float(snapshot["mem_available_gib"]) < args.kill_below_gib:
            result = subprocess.run(
                ["docker", "kill", args.container], capture_output=True, text=True
            )
            append_event(
                args.output,
                {
                    "event": "low_memory_kill",
                    "container": args.container,
                    "threshold_gib": args.kill_below_gib,
                    "docker_kill_returncode": result.returncode,
                    **memory_snapshot(),
                },
            )
            raise SystemExit(70)
        time.sleep(args.interval_seconds)

    append_event(
        args.output,
        {"event": "container_stopped", "container": args.container, **memory_snapshot()},
    )


if __name__ == "__main__":
    main()
