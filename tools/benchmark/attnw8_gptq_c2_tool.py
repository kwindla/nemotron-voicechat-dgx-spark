#!/usr/bin/env python3
"""Compare both packed-runtime lanes on the exact Seal-1 C2 tool rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def scalar(value: Any) -> int:
    if value is None:
        return -1
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError(f"expected scalar token, got {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def manifest(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]


def load_calls(root: Path):
    import torch

    return {
        int(record["call_index"]): torch.load(
            root / record["file"], map_location="cpu", weights_only=False
        )
        for record in manifest(root)
    }


def outputs(call: dict[str, Any]):
    public = call["outputs"]["interface_result"]
    engine = (call["outputs"].get("engine_results") or [])[-1]
    custom = engine.get("custom_outputs") or {}
    text_logits = custom["text_logits"].float().reshape(-1, custom["text_logits"].shape[-1])[-1]
    function_logits = custom["function_logits"].float().reshape(
        -1, custom["function_logits"].shape[-1]
    )[-1]
    return {
        "text_token": scalar(public[0]),
        "function_token": scalar(public[1]),
        "margin": float(function_logits.max().item() - text_logits.max().item()),
        "text_logits_sha256": hashlib.sha256(text_logits.numpy().tobytes()).hexdigest(),
        "function_logits_sha256": hashlib.sha256(function_logits.numpy().tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal1", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--baseline-capture", type=Path, required=True)
    parser.add_argument("--candidate-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal1 = json.loads(args.seal1.read_text())
    a0 = json.loads(args.a0.read_text())
    if not seal1.get("passed") or seal1["a0_sha256"] != hashlib.sha256(args.a0.read_bytes()).hexdigest():
        raise SystemExit("C2 requires the passing hash-matched Seal-1 A0")
    base = load_calls(args.baseline_capture)
    candidate = load_calls(args.candidate_capture)
    rows = []
    passed = True
    for frozen in a0["c2_rows"]:
        call_index = int(frozen["nano_call"])
        b = outputs(base[call_index])
        c = outputs(candidate[call_index])
        baseline_index_match = (
            b["text_token"] == frozen["fhw8_emitted_text_token"]
            and b["function_token"] == frozen["fhw8_emitted_function_token"]
        )
        categorical_equal = (
            b["text_token"] == c["text_token"]
            and b["function_token"] == c["function_token"]
        )
        erosion = c["margin"] - b["margin"]
        relative = None if b["margin"] == 0 else erosion / abs(b["margin"])
        item = {
            **frozen,
            "baseline": b,
            "candidate": c,
            "baseline_index_match": baseline_index_match,
            "categorical_equal": categorical_equal,
            "absolute_margin_erosion": erosion,
            "relative_margin_erosion": relative,
        }
        passed &= baseline_index_match and categorical_equal
        rows.append(item)
    result = {
        "schema": 1,
        "kind": "attnw8_gptq_c2_tool_dual_replay",
        "margin_convention": "max(function_logits)-max(text_logits)",
        "row_count": len(rows),
        "rows": rows,
        "categorical_flips": sum(not r["categorical_equal"] for r in rows),
        "baseline_index_mismatches": sum(not r["baseline_index_match"] for r in rows),
        "passed": bool(passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
