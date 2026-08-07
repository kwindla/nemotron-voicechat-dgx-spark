#!/usr/bin/env python3
"""Reduce sequential strict-v3 soak arms into settled raw/two-frame statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "max_ms": round(max(values), 3),
        "over_80ms": sum(value > 80.0 for value in values),
    }


def load_settled(path: Path, settle_frames: int) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        value = float(event.get("server_step_ms", 0.0))
        if event.get("type") == "voicechat.metrics" and value > 0.0:
            values.append(value)
    if len(values) <= settle_frames:
        raise ValueError(f"{path}: only {len(values)} positive frames")
    return values[settle_frames:]


def summarize(path: Path, settle_frames: int) -> dict[str, Any]:
    raw = load_settled(path, settle_frames)
    paired = [(left + right) / 2.0 for left, right in zip(raw, raw[1:])]
    return {
        "events": str(path.resolve()),
        "settle_frames_discarded": settle_frames,
        "raw": stats(raw),
        "sliding_two_frame": stats(paired),
    }


def deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for distribution in ("raw", "sliding_two_frame"):
        result[distribution] = {}
        for name in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            old = float(baseline[distribution][name])
            new = float(candidate[distribution][name])
            result[distribution][name] = {
                "candidate_minus_baseline_ms": round(new - old, 3),
                "candidate_minus_baseline_percent": round((new - old) * 100 / old, 3),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-frames", type=int, default=1_500)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    baseline = summarize(args.baseline, args.settle_frames)
    candidate = summarize(args.candidate, args.settle_frames)
    report = {
        "schema": 1,
        "method": "server_step_ms > 0; discard first settled frames; linear percentiles",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas(candidate, baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
