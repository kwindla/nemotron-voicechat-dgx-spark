#!/usr/bin/env python3
"""Extract phase-correlated vLLM child-process timing from Docker logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

MARKER = "[VOICECHAT_ENGINE_STEP] "


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "schedule_ms_mean": statistics.mean(row["schedule_ms"] for row in rows),
        "model_executor_ms_mean": statistics.mean(
            row["model_executor_ms"] for row in rows
        ),
        "scheduler_update_ms_mean": statistics.mean(
            row["scheduler_update_ms"] for row in rows
        ),
        "total_ms_mean": statistics.mean(row["total_ms"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--cell-report", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.cell_report.read_text())
    completed = subprocess.run(
        ["docker", "logs", args.container],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    records = []
    for line in completed.stdout.splitlines():
        if MARKER not in line:
            continue
        try:
            records.append(json.loads(line.split(MARKER, 1)[1]))
        except json.JSONDecodeError:
            continue
    phases = []
    for phase in report["phases"]:
        start = phase["phase_start_monotonic_s"]
        end = phase["phase_end_monotonic_s"]
        in_window = [
            row
            for row in records
            if start <= row["monotonic_s"] <= end and row["scheduled_tokens"]
        ]
        nano = [row for row in in_window if args.request_id in row["request_ids"]]
        eartts = [
            row
            for row in in_window
            if any(
                item.startswith(f"{args.request_id}__eartts_generation_")
                for item in row["request_ids"]
            )
        ]
        phases.append(
            {
                "phase": phase["phase"],
                "positions": phase["positions_count"],
                "wall_ms": phase["wall_ms"],
                "nano": summarize(nano),
                "eartts": summarize(eartts),
                "records": {"nano": nano, "eartts": eartts},
            }
        )
    value = {
        "schema": 1,
        "kind": "tool_call_vllm_engine_ledger",
        "container": args.container,
        "cell_report": str(args.cell_report.resolve()),
        "cell_report_sha256": sha256_file(args.cell_report),
        "request_id": args.request_id,
        "phases": phases,
        "passed": all(
            row["nano"]["count"] == row["positions"]
            and row["eartts"]["count"] in {row["positions"], row["positions"] + 1}
            for row in phases
        ),
    }
    atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["passed"] else 1)


if __name__ == "__main__":
    main()
