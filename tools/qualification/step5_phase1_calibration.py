#!/usr/bin/env python3
"""Run the independently sealed Step-5 Phase-1 S+A calibration campaign."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from step5_phase1_campaign import (
    _result_summary,
    _run_lane,
    atomic_json,
    case_matrix,
    load_frozen_fixture,
    sha256_file,
    stable_digest,
    validate_materialized_fixture,
)
from step5_phase1_oracle import (
    NUMERIC_DRIFT_ORIENTATION,
    NUMERIC_STATISTIC_KEYS,
    characterize_snapshot_drift,
)
from step5_phase1_real_executor import (
    RealPhase1Executor,
    SubprocessWorkerBackend,
    WorkerProtocolError,
    executor_from_environment,
    wait_for_gpu_idle,
)

CALIBRATION_CONSTRUCTIONS = 5
CALIBRATION_LANES = ("S", "A")
COMPLETED_BOUNDARY = "B9"
ABORT_BOUNDARY = "NO_MODEL_BOUNDARY"
TERMINAL_CAUSAL_CASE = "frozen-prefix-pad-accept"
ENVELOPE_NAME = "phase1-directional-envelopes.json"
RECEIPT_NAME = "calibration-receipt.json"
MANIFEST_NAME = "artifact-manifest.json"


def _contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("calibration contract differs from the pre-run seal")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "acceptance_rule",
        "case_boundary_population",
        "construction_count",
        "held_out_phase1_replicates",
        "lane_population",
        "orientation",
        "schema_version",
        "statistics",
    }
    if set(contract) != required:
        raise ValueError(f"calibration contract keys differ: {sorted(set(contract) ^ required)}")
    if contract["schema_version"] != 1:
        raise ValueError("calibration contract schema must be 1")
    if contract["construction_count"] != CALIBRATION_CONSTRUCTIONS:
        raise ValueError("calibration construction count must be exactly five")
    if contract["lane_population"] != list(CALIBRATION_LANES):
        raise ValueError("calibration population must be fresh single-engine S+A")
    if contract["held_out_phase1_replicates"] != 12:
        raise ValueError("all 12 Phase-1 replicates must remain held out")
    if contract["orientation"] != NUMERIC_DRIFT_ORIENTATION:
        raise ValueError("calibration orientation mismatch")
    if set(contract["statistics"]) != set(NUMERIC_STATISTIC_KEYS):
        raise ValueError("calibration statistic set is incomplete")
    expected_population = [
        {
            "boundary": ABORT_BOUNDARY if case.name == "abort" else COMPLETED_BOUNDARY,
            "case": case.name,
        }
        for case in case_matrix(12)
        if case.name != TERMINAL_CAUSAL_CASE
    ]
    if contract["case_boundary_population"] != expected_population:
        raise ValueError("calibration does not cover every applicable case/boundary class")
    return contract


def _boundary(case: str) -> str:
    return ABORT_BOUNDARY if case == "abort" else COMPLETED_BOUNDARY


def _envelope_rule(observations: list[Mapping[str, Any]], path: str) -> dict[str, Any]:
    if len(observations) != CALIBRATION_CONSTRUCTIONS:
        raise ValueError(f"{path}: incomplete construction coverage")
    for statistic in observations:
        if set(statistic) != set(NUMERIC_STATISTIC_KEYS):
            raise ValueError(f"{path}: missing statistic")
        if statistic["orientation"] != NUMERIC_DRIFT_ORIENTATION:
            raise ValueError(f"{path}: orientation mismatch")
    metadata = ("dtype", "shape", "element_count")
    for key in metadata:
        if any(item[key] != observations[0][key] for item in observations[1:]):
            raise ValueError(f"{path}: calibration metadata changed for {key}")
    rule: dict[str, Any] = {
        "calibration_observations": len(observations),
        "dtype": observations[0]["dtype"],
        "element_count": observations[0]["element_count"],
        "orientation": NUMERIC_DRIFT_ORIENTATION,
        "shape": observations[0]["shape"],
        "signed_delta_lower": min(item["signed_delta_lower"] for item in observations),
        "signed_delta_upper": max(item["signed_delta_upper"] for item in observations),
    }
    for key in (
        "different_element_count",
        "positive_count",
        "negative_count",
        "zero_count",
    ):
        rule[f"{key}_lower"] = min(item[key] for item in observations)
        rule[f"{key}_upper"] = max(item[key] for item in observations)
    for key in ("l1_norm", "l2_norm", "linf_norm"):
        rule[f"{key}_ceiling"] = max(item[key] for item in observations)
    rule["zero_required"] = all(
        item["different_element_count"] == 0
        and item["positive_count"] == 0
        and item["negative_count"] == 0
        and item["l1_norm"] == 0
        and item["l2_norm"] == 0
        and item["linf_norm"] == 0
        and item["signed_delta_lower"] == 0
        and item["signed_delta_upper"] == 0
        for item in observations
    )
    return rule


def derive_envelopes(
    receipt: Mapping[str, Any], *, contract_sha256: str, receipt_sha256: str
) -> dict[str, Any]:
    constructions = receipt.get("constructions")
    if not isinstance(constructions, list) or len(constructions) != CALIBRATION_CONSTRUCTIONS:
        raise ValueError("cannot derive envelopes from an incomplete calibration")
    if receipt.get("accepted") is not True:
        raise ValueError("cannot derive envelopes from a rejected calibration")
    observations: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for construction in constructions:
        comparisons = construction["comparisons"]
        for case, comparison in comparisons.items():
            boundary = comparison["boundary"]
            for path, statistic in comparison["statistics"].items():
                observations.setdefault((case, boundary, path), []).append(statistic)

    a_lane: dict[str, dict[str, dict[str, Any]]] = {}
    for (case, boundary, path), items in sorted(observations.items()):
        a_lane.setdefault(case, {}).setdefault(boundary, {})[path] = _envelope_rule(items, path)
    expected_cases = {item["case"] for item in receipt["case_boundary_population"]}
    if set(a_lane) != expected_cases:
        raise ValueError("derived envelope case coverage is incomplete")

    conv_zero_paths = [
        path
        for case in a_lane.values()
        for boundary in case.values()
        for path, rule in boundary.items()
        if "committed_mamba_conv_state" in path and "model.layers.0.mixer[0][0]" in path
        and rule["zero_required"]
    ]
    if not conv_zero_paths:
        raise ValueError("the required conv-0 zero envelope is absent")

    payload: dict[str, Any] = {
        "acceptance_semantics": {
            "ceilings_are_inclusive": True,
            "equality_at_ceiling_passes": True,
            "failure_conditions": [
                "missing statistic",
                "orientation mismatch",
                "new or missing numeric leaf",
                "dtype/shape/element-count mismatch",
                "non-finite oracle or candidate value",
                "sign-count or differing-support range violation",
                "signed lower/upper bound violation",
                "L1/L2/Linf norm ceiling exceedance",
                "conv-0 zero-envelope violation",
            ],
        },
        "calibration_contract_sha256": contract_sha256,
        "calibration_receipt_sha256": receipt_sha256,
        "construction_count": CALIBRATION_CONSTRUCTIONS,
        "derived_lane_policy": {
            "A": "observed released packed lane",
            "B": (
                "pre-causal ordinary-packed baseline copied from independently observed A; "
                "not an artifact-matched causal-case oracle"
            ),
        },
        "envelopes": {"A": a_lane, "B": copy.deepcopy(a_lane)},
        "held_out_phase1_replicates": 12,
        "orientation": NUMERIC_DRIFT_ORIENTATION,
        "phase1_recalibration_authorized": False,
        "schema_version": 1,
        "terminal_causal_case": {
            "artifact_matched_oracle_available": False,
            "envelope_authorized": False,
            "semantic_equivalence_authorized": False,
        },
    }
    payload["envelope_sha256"] = stable_digest(payload)
    return payload


async def _one_construction(
    executor: RealPhase1Executor,
    fixture: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    if not isinstance(executor.backend, SubprocessWorkerBackend):
        raise TypeError("calibration requires the subprocess GPU backend")
    client = await executor.backend._spawn(f"calibration-{index:02d}", "materializer")
    errors: list[str] = []
    lane_results: dict[str, dict[str, Any]] = {}
    materialization_seal: dict[str, Any] | None = None
    destroy_reason = "calibration-construction-complete"
    terminal_unknown = False
    try:
        response = await client.request(
            "materialize",
            {"fixture": fixture},
            executor.config.construction_timeout_seconds,
        )
        if not isinstance(response, dict) or "materialized" not in response:
            raise WorkerProtocolError("calibration materializer returned invalid data")
        prefix = Path(str(response.get("prefix_rows_path", "")))
        prefix_sha256 = response.get("prefix_payload_sha256")
        if (
            not prefix.is_file()
            or not isinstance(prefix_sha256, str)
            or sha256_file(prefix) != prefix_sha256
        ):
            raise WorkerProtocolError("calibration materializer prefix is unsealed")
        materialized = response["materialized"]
        materialization_seal = validate_materialized_fixture(materialized, fixture)
        executor.backend.prefix_rows_path = prefix
        executor.backend.prefix_payload_sha256 = prefix_sha256
        executor._materialized = materialized
        executor._materialization_sha256 = str(
            materialization_seal["fixture_materialization_sha256"]
        )
        epoch = {"client": client, "replicate": index}
        for lane in CALIBRATION_LANES:
            results, lane_errors = await _run_lane(
                executor,
                epoch,
                lane,
                fixture,
                index,
                None,
            )
            lane_results[lane] = results
            errors.extend(lane_errors)
            if lane_errors:
                destroy_reason = "calibration-protocol-failure"
                break
            if any(result.terminal_unknown for result in results.values()):
                terminal_unknown = True
                destroy_reason = "calibration-terminal-A-destruction"
                break

        comparisons: dict[str, Any] = {}
        for case in case_matrix(fixture["pad_token_id"]):
            if case.name == TERMINAL_CAUSAL_CASE:
                continue
            s_result = lane_results.get("S", {}).get(case.name)
            a_result = lane_results.get("A", {}).get(case.name)
            if (
                s_result is None
                or a_result is None
                or s_result.snapshot is None
                or a_result.snapshot is None
            ):
                errors.append(f"A/{case.name}: applicable calibration snapshot absent")
                continue
            characterized = characterize_snapshot_drift(s_result.snapshot, a_result.snapshot)
            errors.extend(
                f"A/{case.name}: {difference.path}: {difference.reason}"
                for difference in characterized.differences
            )
            comparisons[case.name] = {
                "boundary": _boundary(case.name),
                "exact_metadata": characterized.exact_metadata,
                "statistics": characterized.statistics,
            }
        return {
            "comparisons": comparisons,
            "construction": client.construction,
            "construction_index": index,
            "errors": errors,
            "fixture_materialization_sha256": (
                None
                if materialization_seal is None
                else materialization_seal["fixture_materialization_sha256"]
            ),
            "lanes": {
                lane: {name: _result_summary(result) for name, result in results.items()}
                for lane, results in lane_results.items()
            },
            "terminal_a_observed": terminal_unknown,
            "worker_pid": client.process.pid,
        }
    except BaseException:
        destroy_reason = "calibration-construction-exception"
        raise
    finally:
        await client.close(destroy_reason, force=terminal_unknown)


def _manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / MANIFEST_NAME
    files = {
        item.relative_to(output_dir).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(output_dir.rglob("*"))
        if item.is_file() and item != path
    }
    payload: dict[str, Any] = {"files": files, "schema_version": 1}
    payload["self_canonical_sha256"] = stable_digest(payload)
    return payload


async def run_calibration(
    executor: RealPhase1Executor,
    output_dir: Path,
    *,
    contract_path: Path,
    contract_sha256: str,
    resume: bool = False,
) -> dict[str, Any]:
    contract = _contract(contract_path, contract_sha256)
    fixture = load_frozen_fixture()
    receipt_path = output_dir / RECEIPT_NAME
    if resume:
        if not receipt_path.is_file():
            raise FileNotFoundError("calibration resume receipt is absent")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("calibration_contract_sha256") != contract_sha256:
            raise ValueError("calibration resume contract seal mismatch")
        if len(receipt.get("constructions", [])) >= CALIBRATION_CONSTRUCTIONS:
            raise ValueError("calibration resume has no remaining constructions")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        receipt = {
            "accepted": False,
            "calibration_contract_path": str(contract_path),
            "calibration_contract_sha256": contract_sha256,
            "case_boundary_population": contract["case_boundary_population"],
            "constructions": [],
            "held_out_phase1_replicates": 12,
            "kind": "phase1-independent-s-a-directional-calibration",
            "lane_population": list(CALIBRATION_LANES),
            "orientation": NUMERIC_DRIFT_ORIENTATION,
            "schema_version": 1,
        }
    receipt["runner_source_sha256_by_construction"] = receipt.get(
        "runner_source_sha256_by_construction",
        ["18788387c4dae0b733771efdb5d0e92832e210233465fa34a4b1d72bb5b30c8b"]
        * len(receipt["constructions"]),
    )
    for index in range(len(receipt["constructions"]), CALIBRATION_CONSTRUCTIONS):
        construction = await _one_construction(executor, fixture, index)
        receipt["constructions"].append(construction)
        receipt["runner_source_sha256_by_construction"].append(sha256_file(Path(__file__)))
        atomic_json(output_dir / RECEIPT_NAME, receipt)
    receipt["accepted"] = (
        len(receipt["constructions"]) == CALIBRATION_CONSTRUCTIONS
        and all(not item["errors"] for item in receipt["constructions"])
    )
    receipt["executor_identity"] = executor.sealed_identity()
    receipt["fresh_worker_pids"] = [item["worker_pid"] for item in receipt["constructions"]]
    receipt["fresh_materialization_seals"] = [
        item["fixture_materialization_sha256"] for item in receipt["constructions"]
    ]
    if len(set(receipt["fresh_worker_pids"])) != len(receipt["fresh_worker_pids"]):
        receipt["accepted"] = False
    atomic_json(output_dir / RECEIPT_NAME, receipt)
    receipt_sha256 = sha256_file(output_dir / RECEIPT_NAME)
    if not receipt["accepted"]:
        atomic_json(output_dir / MANIFEST_NAME, _manifest(output_dir))
        raise RuntimeError(f"calibration failed closed; receipt={output_dir / RECEIPT_NAME}")
    envelope = derive_envelopes(
        receipt,
        contract_sha256=contract_sha256,
        receipt_sha256=receipt_sha256,
    )
    atomic_json(output_dir / ENVELOPE_NAME, envelope)
    atomic_json(output_dir / MANIFEST_NAME, _manifest(output_dir))
    return {
        "accepted": True,
        "calibration_contract_sha256": contract_sha256,
        "calibration_receipt_sha256": receipt_sha256,
        "construction_count": CALIBRATION_CONSTRUCTIONS,
        "envelope_file_sha256": sha256_file(output_dir / ENVELOPE_NAME),
        "envelope_sha256": envelope["envelope_sha256"],
        "manifest_sha256": sha256_file(output_dir / MANIFEST_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    wait_for_gpu_idle(args.idle_timeout_seconds)
    executor = executor_from_environment()
    result = asyncio.run(
        run_calibration(
            executor,
            args.output_dir,
            contract_path=args.contract,
            contract_sha256=args.contract_sha256,
            resume=args.resume,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
