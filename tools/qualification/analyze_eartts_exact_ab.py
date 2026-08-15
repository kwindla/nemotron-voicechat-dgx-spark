#!/usr/bin/env python3
"""Reduce bounded strict-v3 EarTTS A/B logs into distribution summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

TTS_RE = re.compile(r"Time taken for tts_model: ([0-9.]+)s")


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean_ms": round(float(array.mean()), 4),
        "std_ms": round(float(array.std()), 4),
        "min_ms": round(float(array.min()), 4),
        "p05_ms": round(float(np.percentile(array, 5)), 4),
        "p25_ms": round(float(np.percentile(array, 25)), 4),
        "p50_ms": round(float(np.percentile(array, 50)), 4),
        "p75_ms": round(float(np.percentile(array, 75)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "p99_ms": round(float(np.percentile(array, 99)), 4),
        "max_ms": round(float(array.max()), 4),
    }


def session_ids(root: Path) -> list[str]:
    result = []
    report_paths = sorted(
        path
        for path in root.glob("run-*/report.json")
        if re.fullmatch(r"run-\d+", path.parent.name)
    )
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result.append(report["session_closed"]["session_id"])
    return result


def parse_server(root: Path) -> dict[str, list[dict[str, Any]]]:
    wanted = set(session_ids(root))
    frames = {session_id: [] for session_id in wanted}
    pending_tts_ms: float | None = None
    for line in (root / "server.log").read_text(encoding="utf-8", errors="replace").splitlines():
        match = TTS_RE.search(line)
        if match:
            pending_tts_ms = float(match.group(1)) * 1_000.0
            continue
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            event = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        if event.get("event") != "model_step" or event.get("session_id") not in wanted:
            continue
        event["tts_stage_ms"] = pending_tts_ms
        pending_tts_ms = None
        diagnostics = event.get("runtime_diagnostics") or {}
        pad_pair = diagnostics.get("pad_pair") or {}
        decisions = [item.get("decision") for item in pad_pair.get("events", [])]
        event["frame_class"] = decisions[-1] if decisions else "unclassified"
        frames[event["session_id"]].append(event)
    return frames


def summarize_side(root: Path) -> dict[str, Any]:
    ids = session_ids(root)
    frames = parse_server(root)
    runs = []
    pooled_wall: list[float] = []
    pooled_tts: list[float] = []
    by_class: dict[str, dict[str, list[float]]] = {}
    for index, session_id in enumerate(ids, 1):
        values = frames[session_id]
        wall = [float(item["server_step_ms"]) for item in values if item["server_step_ms"] > 0]
        tts = [float(item["tts_stage_ms"]) for item in values if item["tts_stage_ms"] is not None]
        pooled_wall.extend(wall)
        pooled_tts.extend(tts)
        for item in values:
            if item["server_step_ms"] <= 0 or item["tts_stage_ms"] is None:
                continue
            bucket = by_class.setdefault(item["frame_class"], {"wall": [], "tts": []})
            bucket["wall"].append(float(item["server_step_ms"]))
            bucket["tts"].append(float(item["tts_stage_ms"]))
        runs.append(
            {
                "run": index,
                "session_id": session_id,
                "wall": stats(wall),
                "tts_stage": stats(tts),
            }
        )
    return {
        "root": str(root.resolve()),
        "runs": runs,
        "run_mean_distribution": {
            "wall": stats([run["wall"]["mean_ms"] for run in runs]),
            "tts_stage": stats([run["tts_stage"]["mean_ms"] for run in runs]),
        },
        "pooled": {"wall": stats(pooled_wall), "tts_stage": stats(pooled_tts)},
        "by_frame_class": {
            name: {"wall": stats(values["wall"]), "tts_stage": stats(values["tts"])}
            for name, values in sorted(by_class.items())
        },
    }


def delta(candidate: dict[str, Any], baseline: dict[str, Any], field: str) -> dict[str, float]:
    result = {}
    for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        if metric in candidate[field] and metric in baseline[field]:
            result[metric] = round(candidate[field][metric] - baseline[field][metric], 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = summarize_side(args.baseline)
    candidate = summarize_side(args.candidate)
    report = {
        "schema": "voicechat-eartts-exact-ab-v1",
        "label": args.label,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline_ms": {
            "pooled_wall": delta(candidate["pooled"], baseline["pooled"], "wall"),
            "pooled_tts_stage": delta(
                candidate["pooled"], baseline["pooled"], "tts_stage"
            ),
            "run_mean_wall": delta(
                candidate["run_mean_distribution"],
                baseline["run_mean_distribution"],
                "wall",
            ),
            "run_mean_tts_stage": delta(
                candidate["run_mean_distribution"],
                baseline["run_mean_distribution"],
                "tts_stage",
            ),
        },
        "notes": [
            "tts_stage is parsed from the existing host log and quantized to 1 ms",
            "bounded 90 s runs are below the fixture's 1500-frame endurance-settle threshold",
            "negative deltas favor the candidate",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["candidate_minus_baseline_ms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
