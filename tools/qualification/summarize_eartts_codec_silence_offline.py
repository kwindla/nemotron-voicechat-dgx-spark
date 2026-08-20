#!/usr/bin/env python3
"""Consolidate offline codec variants with their captured raw code sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_VARIANTS = ("production", "pointwise_off", "persistent_off", "reference")


def _read_positions(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["frame"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--response", type=int, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    import torch

    analysis = json.loads(args.analysis.read_text())
    target = next(
        row
        for row in analysis["responses"]
        if row["attempt"] == args.attempt and row["response"] == args.response
    )
    start = int(target["start_frame"])
    end = int(target["end_frame"])
    onset = int(target["final_pcm_silence_start_frame"])

    variants = {}
    summaries = {}
    for variant_dir in sorted(path for path in args.results.iterdir() if path.is_dir()):
        positions_path = variant_dir / "positions.jsonl"
        summary_path = variant_dir / "summary.json"
        if positions_path.exists() and summary_path.exists():
            variants[variant_dir.name] = _read_positions(positions_path)
            summaries[variant_dir.name] = json.loads(summary_path.read_text())
    missing = sorted(set(REQUIRED_VARIANTS) - variants.keys())
    if missing:
        raise RuntimeError(f"missing required variants: {missing}")
    expected_frames = set(range(start, end + 1))
    for name, rows in variants.items():
        if set(rows) != expected_frames:
            raise RuntimeError(f"{name} does not contain the expected frame range")

    run_dirs = [path for path in args.capture_root.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one capture run, found {len(run_dirs)}")
    request_id = str(args.attempt + 1)
    records = {}
    for frame_path in sorted(run_dirs[0].glob("frame-*.pt")):
        record = torch.load(frame_path, map_location="cpu", weights_only=False)
        if str(record["request_id"]) == request_id:
            frame = int(record["frame_index"])
            if start <= frame <= end:
                records[frame] = record
    if set(records) != expected_frames:
        raise RuntimeError("capture does not contain the expected raw code sequence")

    matrix_rows = []
    for frame in range(start, end + 1):
        record = records[frame]
        matrix_rows.append(
            {
                "frame": frame,
                "raw_generated_code": [
                    int(value) for value in record["raw_generated_code"].reshape(-1)
                ],
                "raw_code_shape": list(record["raw_generated_code"].shape),
                "raw_equals_post_policy": torch.equal(
                    record["raw_generated_code"],
                    record["post_policy_recurrent_code"],
                ),
                "post_policy_equals_decoder": torch.equal(
                    record["post_policy_recurrent_code"], record["decoder_code"]
                ),
                "policy_applied": bool(record["decoder_pad_policy_applied"]),
                "dbfs": {
                    name: rows[frame]["raw_dbfs"] for name, rows in variants.items()
                },
                "captured_dbfs": variants["production"][frame]["captured_dbfs"],
            }
        )
    with (args.results / "matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in matrix_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    required_spreads = []
    for row in matrix_rows:
        values = [row["dbfs"][name] for name in REQUIRED_VARIANTS]
        required_spreads.append(max(values) - min(values))
    summary = {
        "schema": "eartts-codec-silence-matrix-v1",
        "attempt": args.attempt,
        "response": args.response,
        "session_id": target["session_id"],
        "capture_run": run_dirs[0].name,
        "start_frame": start,
        "end_frame": end,
        "positions": end - start + 1,
        "captured_final_silence_start_frame": onset,
        "required_variants": list(REQUIRED_VARIANTS),
        "additional_variants": sorted(set(variants) - set(REQUIRED_VARIANTS)),
        "production_faithfulness": summaries["production"]["faithfulness"],
        "onset_dbfs": {name: variants[name][onset]["raw_dbfs"] for name in variants},
        "audible_positions_at_or_after_captured_silence_start": {
            name: summaries[name]["raw_decode"][
                "audible_positions_at_or_after_captured_silence_start"
            ]
            for name in variants
        },
        "max_required_variant_dbfs_spread": max(required_spreads),
        "onset_required_variant_dbfs_spread": required_spreads[onset - start],
        "first_required_variant_spread_above_0_1_db": next(
            (
                row["frame"]
                for row, spread in zip(matrix_rows, required_spreads, strict=True)
                if spread > 0.1
            ),
            None,
        ),
    }
    (args.results / "matrix-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
