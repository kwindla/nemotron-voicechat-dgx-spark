#!/usr/bin/env python3
"""Response-clustered Step 4 A/B comparison over qualified analyzer strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from stratified_latency_analyzer import (
    _membership_identifier,
    _normalize_model_rows,
    _stage_timings,
    classify_populations,
    distribution_stats,
    nearest_rank,
    read_jsonl,
    response_clustered_stats,
    source_identity,
)

JsonObject = dict[str, Any]
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_ITERATIONS = 10_000


def _clusters(
    rows: list[JsonObject], value: Callable[[JsonObject], float | None]
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        measured = value(row)
        response_id = row.get("response_group")
        if measured is not None and isinstance(response_id, str):
            result[response_id].append(float(measured))
    return dict(result)


def _paired_cluster_difference(
    arm_a: dict[str, list[float]],
    arm_b: dict[str, list[float]],
    pairing: list[JsonObject] | None = None,
) -> JsonObject:
    names_a = list(arm_a)
    names_b = list(arm_b)
    if len(names_a) != len(names_b) or not names_a:
        raise ValueError("A/B response cluster cardinality differs or is empty")
    if pairing is None:
        pairing = [
            {
                "prompt_ordinal": ordinal,
                "arm_a_response_group": name_a,
                "arm_b_response_group": name_b,
            }
            for ordinal, (name_a, name_b) in enumerate(
                zip(sorted(names_a), sorted(names_b), strict=True), 1
            )
        ]
    if len(pairing) != len(names_a):
        raise ValueError("A/B prompt mapping and response cluster cardinality differ")
    pairs = [
        (str(item["arm_a_response_group"]), str(item["arm_b_response_group"]), item)
        for item in pairing
    ]
    if {pair[0] for pair in pairs} != set(names_a) or {pair[1] for pair in pairs} != set(names_b):
        raise ValueError("A/B prompt mapping does not exactly cover response clusters")

    def pooled_difference(selected: list[tuple[str, str, JsonObject]]) -> float:
        values_a = [value for name_a, _name_b, _item in selected for value in arm_a[name_a]]
        values_b = [value for _name_a, name_b, _item in selected for value in arm_b[name_b]]
        return sum(values_a) / len(values_a) - sum(values_b) / len(values_b)

    def equal_response_difference(selected: list[tuple[str, str, JsonObject]]) -> float:
        effects = [
            sum(arm_a[name_a]) / len(arm_a[name_a]) - sum(arm_b[name_b]) / len(arm_b[name_b])
            for name_a, name_b, _item in selected
        ]
        return sum(effects) / len(effects)

    generator = random.Random(BOOTSTRAP_SEED)
    selections = [[generator.choice(pairs) for _ in pairs] for _ in range(BOOTSTRAP_ITERATIONS)]
    frame_bootstrap = [pooled_difference(selected) for selected in selections]
    response_bootstrap = [equal_response_difference(selected) for selected in selections]
    per_response = []
    for name_a, name_b, pair_identity in pairs:
        mean_a = sum(arm_a[name_a]) / len(arm_a[name_a])
        mean_b = sum(arm_b[name_b]) / len(arm_b[name_b])
        per_response.append(
            {
                **pair_identity,
                "arm_a_response_id": name_a,
                "arm_b_response_id": name_b,
                "arm_a_n": len(arm_a[name_a]),
                "arm_b_n": len(arm_b[name_b]),
                "arm_a_mean_ms": mean_a,
                "arm_b_mean_ms": mean_b,
                "trace_on_minus_trace_off_ms": mean_a - mean_b,
            }
        )
    return {
        "effect_definition": "arm A trace-on minus arm B trace-off",
        "n_paired_response_clusters": len(pairs),
        "effect_ms": pooled_difference(pairs),
        "bootstrap_95_ci_ms": [
            nearest_rank(frame_bootstrap, 2.5),
            nearest_rank(frame_bootstrap, 97.5),
        ],
        "declared_estimands": {
            "frame_weighted": {
                "definition": "pooled frame mean in arm A minus pooled frame mean in arm B",
                "effect_ms": pooled_difference(pairs),
                "bootstrap_95_ci_ms": [
                    nearest_rank(frame_bootstrap, 2.5),
                    nearest_rank(frame_bootstrap, 97.5),
                ],
            },
            "equal_response": {
                "definition": "mean of paired response-level arm differences",
                "effect_ms": equal_response_difference(pairs),
                "bootstrap_95_ci_ms": [
                    nearest_rank(response_bootstrap, 2.5),
                    nearest_rank(response_bootstrap, 97.5),
                ],
            },
            "equal_run": {
                "definition": (
                    "equal-weighted run means; with one run period per arm this equals the "
                    "observed pooled-period difference and has no run-level uncertainty"
                ),
                "effect_ms": pooled_difference(pairs),
                "uncertainty_status": "unavailable-one-run-period-per-arm",
            },
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "per_response": per_response,
    }


def _clock_value(row: JsonObject) -> float | None:
    value = row["event"].get("server_step_ms")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _residual_value(row: JsonObject) -> float | None:
    value = _stage_timings(row["event"]).get("wrapper_residual_ms")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metric_report(
    rows_a: list[JsonObject],
    rows_b: list[JsonObject],
    value: Callable[[JsonObject], float | None],
    pairing: list[JsonObject],
) -> JsonObject:
    clusters_a = _clusters(rows_a, value)
    clusters_b = _clusters(rows_b, value)
    values_a = [item for values in clusters_a.values() for item in values]
    values_b = [item for values in clusters_b.values() for item in values]
    return {
        "arm_a": distribution_stats(values_a),
        "arm_b": distribution_stats(values_b),
        "paired_response_clustered_difference": _paired_cluster_difference(
            clusters_a, clusters_b, pairing
        ),
    }


def _membership(rows: list[JsonObject]) -> JsonObject:
    members = [_membership_identifier(row) for row in rows]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return {"n": len(members), "sha256": hashlib.sha256(encoded).hexdigest(), "members": members}


def _load_jobs(report_path: Path) -> list[JsonObject]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("job_records")
    if not isinstance(records, list) or not records:
        raise ValueError("run report lacks retained prompt/job mapping records")
    ordinals = [record.get("prompt_ordinal") for record in records]
    if ordinals != list(range(1, len(records) + 1)):
        raise ValueError("run report has missing or duplicated prompt ordinal")
    for field in ("job_id", "prompt_id", "prompt_text", "response_id"):
        values = [record.get(field) for record in records]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"run report has missing {field}")
        if field in {"job_id", "response_id"} and len(set(values)) != len(values):
            raise ValueError(f"run report has duplicated {field}")
    if any(record.get("terminal_status") != "completed" for record in records):
        raise ValueError("run report contains a non-completed prompt job")
    return records


def _load_arm(
    analyzer_path: Path, trace_path: Path, report_path: Path
) -> tuple[JsonObject, dict[str, Any], list[JsonObject]]:
    analyzer = json.loads(analyzer_path.read_text(encoding="utf-8"))
    if analyzer.get("schema") != "generation-frame-latency-step0-v2":
        raise ValueError("input is not a qualified Step 0 analyzer result")
    if len(analyzer.get("sources", [])) != 1:
        raise ValueError("A/B comparison requires one source per analyzer result")
    source = analyzer["sources"][0]
    identity = source_identity(trace_path)
    if source["source"] != identity:
        raise ValueError("trace identity does not match qualified analyzer result")
    populations = classify_populations(_normalize_model_rows(read_jsonl(trace_path)))
    for name, rows in populations.items():
        if _membership(rows) != source["populations"][name]["membership"]:
            raise ValueError(f"qualified population exact membership mismatch: {name}")
    return source, populations, _load_jobs(report_path)


def _pairing_manifest(
    rows_a: list[JsonObject],
    rows_b: list[JsonObject],
    jobs_a: list[JsonObject],
    jobs_b: list[JsonObject],
) -> list[JsonObject]:
    groups_a = list(_clusters(rows_a, _clock_value))
    groups_b = list(_clusters(rows_b, _clock_value))
    if len(groups_a) != len(jobs_a) or len(groups_b) != len(jobs_b):
        raise ValueError("prompt jobs and analyzer response cluster cardinality differ")
    pairing = []
    for job_a, job_b, group_a, group_b in zip(jobs_a, jobs_b, groups_a, groups_b, strict=True):
        if job_a["prompt_ordinal"] != job_b["prompt_ordinal"]:
            raise ValueError("A/B prompt ordinals differ")
        if (job_a["prompt_id"], job_a["prompt_text"]) != (
            job_b["prompt_id"],
            job_b["prompt_text"],
        ):
            raise ValueError(f"A/B prompt identity differs at ordinal {job_a['prompt_ordinal']}")
        pairing.append(
            {
                "prompt_ordinal": job_a["prompt_ordinal"],
                "prompt_id": job_a["prompt_id"],
                "prompt_text": job_a["prompt_text"],
                "arm_a_job_id": job_a["job_id"],
                "arm_b_job_id": job_b["job_id"],
                "arm_a_retained_response_id": job_a["response_id"],
                "arm_b_retained_response_id": job_b["response_id"],
                "arm_a_response_group": group_a,
                "arm_b_response_group": group_b,
            }
        )
    return pairing


def analyze(
    analyzer_a: Path,
    trace_a: Path,
    run_report_a: Path,
    analyzer_b: Path,
    trace_b: Path,
    run_report_b: Path,
) -> JsonObject:
    source_a, populations_a, jobs_a = _load_arm(analyzer_a, trace_a, run_report_a)
    source_b, populations_b, jobs_b = _load_arm(analyzer_b, trace_b, run_report_b)
    delivered_nonbos_a = populations_a["delivered-nonBOS"]
    delivered_nonbos_b = populations_b["delivered-nonBOS"]
    residual_high_a = populations_a["residual-high"]
    residual_high_b = populations_b["residual-high"]
    pairing = _pairing_manifest(delivered_nonbos_a, delivered_nonbos_b, jobs_a, jobs_b)
    return {
        "schema": "nemotron_voicechat.step4_trace_overhead_ab.v2",
        "inference_scope": (
            "descriptive one fixed on-then-off run sequence; response bootstrap is conditional "
            "on these two periods and does not estimate run/order uncertainty"
        ),
        "pairing": pairing,
        "arms": {
            "A_trace_on": {
                "analyzer_source": source_a["source"],
                "delivered_nonbos_membership": source_a["populations"]["delivered-nonBOS"][
                    "membership"
                ],
                "residual_high_membership": source_a["populations"]["residual-high"]["membership"],
            },
            "B_trace_off": {
                "analyzer_source": source_b["source"],
                "delivered_nonbos_membership": source_b["populations"]["delivered-nonBOS"][
                    "membership"
                ],
                "residual_high_membership": source_b["populations"]["residual-high"]["membership"],
            },
        },
        "delivered_generation_server_step_ms": _metric_report(
            delivered_nonbos_a, delivered_nonbos_b, _clock_value, pairing
        ),
        "delivered_generation_wrapper_residual_ms": _metric_report(
            delivered_nonbos_a, delivered_nonbos_b, _residual_value, pairing
        ),
        "qualified_delivered_clustered_server_step_ms": {
            "arm_a": response_clustered_stats(populations_a["delivered"], "server_step_ms"),
            "arm_b": response_clustered_stats(populations_b["delivered"], "server_step_ms"),
        },
        "residual_high": {
            "definition": "qualified delivered-nonBOS wrapper_residual_ms >= 10 ms",
            "arm_a_n": len(residual_high_a),
            "arm_b_n": len(residual_high_b),
            "effect_status": "unavailable-empty-both-arms"
            if not residual_high_a and not residual_high_b
            else "observed",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer-a", type=Path, required=True)
    parser.add_argument("--trace-a", type=Path, required=True)
    parser.add_argument("--run-report-a", type=Path, required=True)
    parser.add_argument("--analyzer-b", type=Path, required=True)
    parser.add_argument("--trace-b", type=Path, required=True)
    parser.add_argument("--run-report-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.analyzer_a,
        args.trace_a,
        args.run_report_a,
        args.analyzer_b,
        args.trace_b,
        args.run_report_b,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
