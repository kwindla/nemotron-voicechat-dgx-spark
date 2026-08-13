#!/usr/bin/env python3
"""Poll Step 4 host telemetry at a monotonic-clock cadence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

GPU_FIELDS = (
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
)
CPU_IDS = (5, 6, 7, 8, 9)
PRESSURE_KINDS = ("cpu", "memory", "io")


def parse_number(value: str) -> float | None:
    stripped = value.strip()
    if stripped in {"", "N/A", "[N/A]"}:
        return None
    number = float(stripped)
    if not math.isfinite(number):
        raise ValueError(f"non-finite telemetry value: {value!r}")
    return number


def parse_pressure(line: str) -> dict[str, float]:
    fields = line.strip().split()
    if not fields or fields[0] != "some":
        raise ValueError(f"unexpected pressure record: {line!r}")
    parsed: dict[str, float] = {}
    for field in fields[1:]:
        key, raw = field.split("=", 1)
        parsed[key] = float(raw)
    if set(parsed) != {"avg10", "avg60", "avg300", "total"}:
        raise ValueError(f"unexpected pressure fields: {line!r}")
    return parsed


def _boundary() -> dict[str, float]:
    return {"monotonic_s": time.monotonic(), "wall_time_s": time.time()}


def _interval(start: dict[str, float], end: dict[str, float]) -> dict[str, float]:
    return {
        "monotonic_start_s": round(start["monotonic_s"], 9),
        "monotonic_end_s": round(end["monotonic_s"], 9),
        "wall_start_s": round(start["wall_time_s"], 9),
        "wall_end_s": round(end["wall_time_s"], 9),
        "duration_ms": round((end["monotonic_s"] - start["monotonic_s"]) * 1000.0, 6),
    }


def sample() -> dict[str, object]:
    acquisition_start = _boundary()
    query = ",".join(GPU_FIELDS)
    gpu_start = _boundary()
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_end = _boundary()
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one GPU, received {len(rows)}")
    values = [parse_number(value) for value in rows[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        raise RuntimeError("nvidia-smi returned an unexpected field count")
    cpu_khz: dict[str, int | None] = {}
    cpu_start = _boundary()
    for cpu_id in CPU_IDS:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_cur_freq")
        cpu_khz[str(cpu_id)] = int(path.read_text(encoding="utf-8").strip())
    cpu_end = _boundary()
    pressure_start = _boundary()
    pressure = {
        kind: parse_pressure(
            Path(f"/proc/pressure/{kind}").read_text(encoding="utf-8").splitlines()[0]
        )
        for kind in PRESSURE_KINDS
    }
    pressure_end = _boundary()
    acquisition_end = _boundary()
    return {
        "schema": "nemotron_voicechat.step4_telemetry.v2",
        # These aliases remain the acquisition-start clock for readers of v1.
        "monotonic_s": round(acquisition_start["monotonic_s"], 9),
        "wall_time_s": round(acquisition_start["wall_time_s"], 9),
        "wall_minus_monotonic_s": round(
            acquisition_start["wall_time_s"] - acquisition_start["monotonic_s"], 9
        ),
        "acquisition": _interval(acquisition_start, acquisition_end),
        "source_acquisition": {
            "gpu": _interval(gpu_start, gpu_end),
            "cpu_scaling_cur_freq": _interval(cpu_start, cpu_end),
            "pressure": _interval(pressure_start, pressure_end),
        },
        "gpu": dict(zip(GPU_FIELDS, values, strict=True)),
        "cpu_scaling_cur_freq_khz": cpu_khz,
        "pressure": pressure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("durations must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration_seconds
    next_sample = time.monotonic()
    with args.output.open("x", encoding="utf-8") as output:
        while True:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(next_sample - now)
            record = sample()
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            if time.monotonic() >= deadline:
                break
            next_sample += args.interval_seconds
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
