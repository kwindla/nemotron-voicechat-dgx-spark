#!/usr/bin/env python3
"""Build a gap-free additive Step-9 startup account from one arm's evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def docker_timestamp_ns(line: str) -> int:
    stamp = line.split(" ", 1)[0]
    if not stamp.endswith("Z") or "." not in stamp:
        raise ValueError(f"line has no Docker timestamp: {line[:120]!r}")
    head, fraction = stamp[:-1].split(".", 1)
    seconds = int(datetime.fromisoformat(head + "+00:00").timestamp())
    return seconds * 1_000_000_000 + int(fraction.ljust(9, "0")[:9])


def startup_events(lines: list[str]) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in lines:
        marker = "VOICECHAT_STARTUP "
        if marker not in line:
            continue
        payload = json.loads(line.split(marker, 1)[1])
        # EngineCore is spawned with the same module. The parent event is first.
        events.setdefault(payload["event"], int(payload["wall_time_ns"]))
    return events


def matching_times(lines: list[str], needle: str) -> list[int]:
    return [docker_timestamp_ns(line) for line in lines if needle in line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host = json.loads((args.arm / "host-timeline.json").read_text())
    lines = (args.arm / "server.log").read_text(errors="replace").splitlines()
    event = startup_events(lines)
    model_start = matching_times(lines, "Initializing model structure...")[0]
    model_finish = matching_times(lines, "Model structure initialized")[0]
    engine_start = matching_times(lines, "Initializing vLLM engine (this may take")[0:2]
    engine_ready = matching_times(lines, "vLLM engine ready!")[0:2]
    boundaries = [
        ("container launch and Python import", int(host["launch_wall_ns"])),
        ("module import to main entry", event["server_module_imported"]),
        ("main dispatch to pipeline build", event["main_entered"]),
        ("pre-model pipeline setup", event["pipeline_build_started"]),
        ("NeMo CPU model-structure initialization", model_start),
        ("checkpoint hydration and EarTTS handoff", model_finish),
        ("EarTTS vLLM engine initialization", engine_start[0]),
        ("inter-engine fixed setup", engine_ready[0]),
        ("Nano vLLM engine initialization", engine_start[1]),
        ("post-engine pipeline setup", engine_ready[1]),
        ("pipeline warmup", event["pipeline_build_finished"]),
        ("realtime engine construction", event["pipeline_warmup_finished"]),
        ("realtime exact-prompt continuation warmup", event["realtime_engine_create_finished"]),
        ("application construction and server entry", event["realtime_warmup_finished"]),
        ("Uvicorn startup", event["uvicorn_run_entered"]),
        ("ready sentinel to first successful health poll", event["application_ready"]),
        ("ready", int(host["ready_wall_ns"])),
    ]
    phases = []
    for (name, start), (_, finish) in zip(boundaries, boundaries[1:]):
        if finish < start:
            raise RuntimeError(f"non-monotonic boundary for {name}: {start} -> {finish}")
        phases.append({"phase": name, "seconds": (finish - start) / 1e9})
    total = sum(phase["seconds"] for phase in phases)
    result = {
        "schema": 1,
        "arm": str(args.arm),
        "phases": phases,
        "sum_seconds": total,
        "host_wall_seconds": host["wall_seconds"],
        "accounting_error_seconds": total - float(host["wall_seconds"]),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
