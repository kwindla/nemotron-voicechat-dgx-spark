#!/usr/bin/env python3
"""Analyze retained tool-call benchmark phase and per-position ledgers."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STAGES = (
    "embed_ms",
    "perception_poll_ms",
    "fusion_ms",
    "nano_interface_ms",
    "eartts_infer_ms",
    "codec_decode_ms",
    "codec_cpu_copy_ms",
    "rnnt_ms",
    "residual_ms",
)
NANO_ENGINE_STAGES = (
    "input_copy_ms",
    "submit_ms",
    "output_wait_ms",
    "parse_ms",
    "residual_ms",
)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260810)
    bootstrap_means = np.mean(rng.choice(array, size=(10_000, len(array)), replace=True), axis=1)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean_bootstrap_ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
    }


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, Any]:
    if len(xs) < 2 or len(set(xs)) < 2:
        return {"count": len(xs)}
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    predicted = slope * np.asarray(xs) + intercept
    residual = np.asarray(ys) - predicted
    total = np.asarray(ys) - np.mean(ys)
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
    return {
        "count": len(xs),
        "slope_ms_per_position": float(slope),
        "intercept_ms": float(intercept),
        "r_squared": r_squared,
    }


def load_cells(root: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(root.glob("cell-*/report.json")):
        report = json.loads(path.read_text())
        report["_path"] = str(path.resolve())
        reports.append(report)
    return reports


def injection_contract(cell: dict[str, Any]) -> dict[str, Any]:
    payload = cell["payload"]
    use_model_output = cell.get("model_output_enabled") is True
    injected_tokens = int(
        payload["model_output_tokens"] if use_model_output else payload["realized_tokens"]
    )
    response_phases = [
        phase for phase in cell["phases"] if phase.get("phase") == "tool_response"
    ]
    actual_positions = (
        response_phases[0].get("positions_count") if len(response_phases) == 1 else None
    )
    expected_positions = injected_tokens + 1
    acknowledged_tokens = cell.get("function_output_tokens_ack")
    acknowledgement_matches = acknowledged_tokens is None or (
        isinstance(acknowledged_tokens, int)
        and not isinstance(acknowledged_tokens, bool)
        and acknowledged_tokens == injected_tokens
    )
    return {
        "ordinal": cell["ordinal"],
        "full_output_tokens": int(payload["realized_tokens"]),
        "injection_source": "model_output" if use_model_output else "output",
        "injected_tokens": injected_tokens,
        "expected_positions": expected_positions,
        "actual_positions": actual_positions,
        "function_output_tokens_ack": acknowledged_tokens,
        "acknowledgement_matches": acknowledgement_matches,
        "passed": actual_positions == expected_positions and acknowledgement_matches,
    }


def cell_contract_passed(cell: dict[str, Any]) -> bool:
    phases = cell.get("phases")
    if not isinstance(phases, list):
        return False
    return bool(
        not cell.get("errors")
        and isinstance(cell.get("call_id"), str)
        and len(phases) == 2
        and {phase.get("phase") for phase in phases} == {"tool_call", "tool_response"}
        and all(phase.get("positions_truncated") is False for phase in phases)
        and injection_contract(cell)["passed"]
        and cell["payload"]["verification_word"].casefold()
        in cell.get("text", "").casefold()
        and cell.get("first_audible_monotonic_s") is not None
    )


def analyze(root: Path, *, expected_cells: int | None) -> dict[str, Any]:
    cells = load_cells(root)
    if expected_cells is not None and len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} cells, found {len(cells)}")
    if not cells:
        raise RuntimeError("no benchmark cells found")
    phase_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        for phase in cell["phases"]:
            row = {"cell": cell, "phase": phase}
            phase_rows[phase["phase"]].append(row)
            by_target[int(cell["payload"]["realized_tokens"])].append(row)
    phase_analysis = {}
    for phase_name, rows in phase_rows.items():
        wall = [float(row["phase"]["wall_ms"]) for row in rows]
        positions = [float(row["phase"]["positions_count"]) for row in rows]
        positions_flat = [position for row in rows for position in row["phase"]["positions"]]
        phase_analysis[phase_name] = {
            "cycles": len(rows),
            "wall_ms": distribution(wall),
            "positions": distribution(positions),
            "wall_fit": linear_fit(positions, wall),
            "per_position": {
                stage: distribution([float(position[stage]) for position in positions_flat])
                for stage in STAGES
            },
            "nano_engine_per_position": {
                stage: distribution(
                    [float(position["nano_engine"][stage]) for position in positions_flat]
                )
                for stage in NANO_ENGINE_STAGES
            },
            "phase_residual_ms": distribution([float(row["phase"]["residual_ms"]) for row in rows]),
        }
    target_analysis = {}
    for target, rows in sorted(by_target.items()):
        response_rows = [row for row in rows if row["phase"]["phase"] == "tool_response"]
        target_analysis[str(target)] = {
            "cycles": len(response_rows),
            "wall_ms": distribution([float(row["phase"]["wall_ms"]) for row in response_rows]),
            "positions": distribution(
                [float(row["phase"]["positions_count"]) for row in response_rows]
            ),
        }
    response_rows = phase_rows.get("tool_response", [])
    response_positions = [float(row["phase"]["positions_count"]) for row in response_rows]
    response_wall = [float(row["phase"]["wall_ms"]) for row in response_rows]
    ordering = [
        {
            "ordinal": cell["ordinal"],
            "block": cell["block"],
            "tokens": cell["payload"]["realized_tokens"],
            "source_passed": cell["passed"],
            "passed": cell_contract_passed(cell),
        }
        for cell in cells
    ]
    injection_contracts = [injection_contract(cell) for cell in cells]
    rederived_cell_passed = [cell_contract_passed(cell) for cell in cells]
    return {
        "schema": 2,
        "kind": "tool_call_latency_analysis_v2",
        "passed": bool(
            all(rederived_cell_passed)
            and all(
                abs(float(phase["residual_ms"])) <= 150 and phase["positions_truncated"] is False
                for cell in cells
                for phase in cell["phases"]
            )
        ),
        "cells": len(cells),
        "cell_passed": sum(rederived_cell_passed),
        "source_cell_passed": sum(bool(cell["passed"]) for cell in cells),
        "ordering": ordering,
        "injection_contracts": injection_contracts,
        "phase_analysis": phase_analysis,
        "tool_response_primary_fit": linear_fit(response_positions, response_wall),
        "targets": target_analysis,
        "text_semantic_passed": sum(
            cell["payload"]["verification_word"].casefold() in cell["text"].casefold()
            for cell in cells
        ),
        "audible_passed": sum(cell["first_audible_monotonic_s"] is not None for cell in cells),
        "mean_end_to_end_seconds": statistics.mean(
            float(cell["response_done_monotonic_s"] - cell["client_commit_monotonic_s"])
            for cell in cells
        ),
        "source_reports": [cell["_path"] for cell in cells],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int)
    args = parser.parse_args()
    try:
        report = analyze(args.input, expected_cells=args.expected_cells)
    except Exception as exc:
        report = {
            "schema": 2,
            "kind": "tool_call_latency_analysis_v2",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
