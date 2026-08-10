#!/usr/bin/env python3
"""Compare Pocket, FP32-direct, and W8-direct semantic-corpus evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemotron_voicechat_runtime.direct_semantic_probe import _transaction_checks
from nemotron_voicechat_runtime.provenance import valid_runtime_provenance
from nemotron_voicechat_runtime.semantic_corpus import (
    corpus_sha256,
    evaluate_pocket_carrier,
    evaluate_semantic_case,
    load_semantic_corpus,
)

TOKEN_FORMS = ("text", "user_bos_text_user_eos")


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be an object: {path}")
    return value


def _case_map(cases: Any, expected_ids: tuple[str, ...], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(cases, list):
        raise ValueError(f"{label} has no case list")
    mapped = {str(case.get("case_id")): case for case in cases if isinstance(case, dict)}
    if tuple(mapped) != expected_ids or len(mapped) != len(cases):
        raise ValueError(f"{label} case order/set differs from the checked-in corpus")
    return mapped


def _regrade(
    corpus_cases: list[dict[str, Any]], raw_cases: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    grades: dict[str, dict[str, Any]] = {}
    for case in corpus_cases:
        raw = raw_cases[case["id"]]
        grade = evaluate_semantic_case(
            case,
            response_text=str(raw.get("assistant_text") or ""),
            tool_calls=raw.get("tool_calls") or [],
        )
        response_status = raw.get("response_status")
        response_reason = raw.get("response_reason")
        terminal_passed = not (
            response_status == "failed" or response_reason == "function_call_loop_limit"
        )
        grade["response_terminal_passed"] = terminal_passed
        grade["response_status"] = response_status
        grade["response_reason"] = response_reason
        grade["passed"] = bool(grade["passed"] and terminal_passed)
        grades[case["id"]] = grade
    return grades


def _lane(report: dict[str, Any], token_form: str) -> dict[str, Any]:
    matches = [lane for lane in report.get("lanes", []) if lane.get("token_form") == token_form]
    if len(matches) != 1:
        raise ValueError(f"direct report must contain one {token_form} lane")
    return matches[0]


def evaluate(
    *,
    corpus_path: Path,
    pocket: dict[str, Any],
    fp32: dict[str, Any],
    w8: dict[str, Any],
) -> dict[str, Any]:
    corpus = load_semantic_corpus(corpus_path)
    corpus_hash = corpus_sha256(corpus_path)
    expected_ids = tuple(case["id"] for case in corpus["cases"])
    reports = {"pocket": pocket, "fp32": fp32, "w8": w8}
    corpus_identity_passed = all(
        report.get("corpus_sha256") == corpus_hash for report in reports.values()
    )
    provenance_passed = all(
        valid_runtime_provenance(report.get("runtime_provenance")) for report in reports.values()
    )
    source_fingerprints = [
        report.get("runtime_provenance", {}).get("source_sha256") for report in reports.values()
    ]
    source_provenance_consistent = bool(
        source_fingerprints[0]
        and all(item == source_fingerprints[0] for item in source_fingerprints)
    )

    pocket_raw = _case_map(pocket.get("cases"), expected_ids, "Pocket")
    pocket_grade = _regrade(corpus["cases"], pocket_raw)
    pocket_carrier = {
        case["id"]: evaluate_pocket_carrier(
            case,
            transcript=str(pocket_raw[case["id"]].get("input_transcript") or ""),
        )
        for case in corpus["cases"]
    }
    carrier_preserved = [
        case_id for case_id in expected_ids if pocket_carrier[case_id]["passed"]
    ]
    carrier_inconclusive = [
        case_id for case_id in expected_ids if not pocket_carrier[case_id]["passed"]
    ]
    pocket_passed = sum(pocket_grade[case_id]["passed"] for case_id in carrier_preserved)
    pocket_tool_cases = [
        case_id
        for case_id in carrier_preserved
        if pocket_grade[case_id]["expected_tool"] is not None
    ]
    pocket_safety_cases = [
        case_id for case_id in carrier_preserved if pocket_grade[case_id]["safety_critical"]
    ]
    pocket_tool = bool(pocket_tool_cases) and all(
        pocket_grade[case_id]["tool_passed"] for case_id in pocket_tool_cases
    )
    pocket_safety = bool(pocket_safety_cases) and all(
        pocket_grade[case_id]["passed"] for case_id in pocket_safety_cases
    )
    pocket_evidence_complete = bool(
        pocket.get("evidence_complete") is True
        and all(case.get("structural_passed") is True for case in pocket_raw.values())
    )
    pocket_gate = {
        "evidence_complete": pocket_evidence_complete,
        "carrier_preserved_count": len(carrier_preserved),
        "carrier_inconclusive_cases": carrier_inconclusive,
        "eligible_intent_predicate_rate": (
            pocket_passed / len(carrier_preserved) if carrier_preserved else 0.0
        ),
        "eligible_intent_predicate_passed": bool(carrier_preserved)
        and pocket_passed / len(carrier_preserved) >= 0.95,
        "tool_name_and_arguments_passed": pocket_tool,
        "safety_critical_passed": pocket_safety,
        "genuine_semantic_failures": [
            case_id
            for case_id in carrier_preserved
            if not pocket_grade[case_id]["passed"]
        ],
    }
    pocket_gate["reference_passed"] = bool(
        pocket_gate["eligible_intent_predicate_passed"]
        and pocket_tool
        and pocket_safety
    )

    lanes: dict[str, dict[str, Any]] = {}
    grades: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    raw_lanes: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for precision, report in (("fp32", fp32), ("w8", w8)):
        if str(report.get("precision", "")).casefold() != precision:
            raise ValueError(f"{precision} report precision label mismatch")
        for token_form in TOKEN_FORMS:
            lane = _lane(report, token_form)
            raw = _case_map(lane.get("cases"), expected_ids, f"{precision}/{token_form}")
            grade = _regrade(corpus["cases"], raw)
            grades[(precision, token_form)] = grade
            raw_lanes[(precision, token_form)] = raw
            pad_token_id = report.get("pad_token_id")
            hidden_transaction_passed = bool(
                isinstance(pad_token_id, int)
                and all(
                    all(
                        _transaction_checks(
                            case.get("direct_positions") or [], pad_token_id
                        ).values()
                    )
                    for case in raw.values()
                )
            )
            direct_passed_count = sum(grade[case_id]["passed"] for case_id in expected_ids)
            pocket_agreement_count = sum(
                pocket_grade[case_id]["passed"] == grade[case_id]["passed"]
                for case_id in carrier_preserved
            )
            tool_passed = all(
                grade[case_id]["tool_passed"]
                for case_id in expected_ids
                if grade[case_id]["expected_tool"] is not None
            )
            safety_passed = all(
                grade[case_id]["passed"]
                for case_id in expected_ids
                if grade[case_id]["safety_critical"]
            )
            result = {
                "precision": precision,
                "token_form": token_form,
                "structural_passed": all(
                    case.get("structural_passed") is True for case in raw.values()
                )
                and hidden_transaction_passed,
                "hidden_transaction_rederived_passed": hidden_transaction_passed,
                "direct_intent_predicate_count": direct_passed_count,
                "direct_intent_predicate_rate": direct_passed_count / len(expected_ids),
                "direct_intent_predicate_passed": direct_passed_count
                / len(expected_ids)
                >= 0.95,
                "pocket_reference_agreement_count": pocket_agreement_count,
                "pocket_reference_agreement_rate": (
                    pocket_agreement_count / len(carrier_preserved)
                    if carrier_preserved
                    else 0.0
                ),
                "tool_name_and_arguments_passed": tool_passed,
                "safety_critical_passed": safety_passed,
                "failed_cases": [
                    case_id for case_id in expected_ids if not grade[case_id]["passed"]
                ],
            }
            result["passed"] = all(
                value for key, value in result.items() if key.endswith("_passed")
            )
            lanes[f"{precision}/{token_form}"] = result

    token_forms: dict[str, dict[str, Any]] = {}
    for token_form in TOKEN_FORMS:
        regressions = [
            case
            for case in corpus["cases"]
            if grades[("fp32", token_form)][case["id"]]["passed"]
            and not grades[("w8", token_form)][case["id"]]["passed"]
        ]
        critical = [
            case["id"]
            for case in regressions
            if case["predicate"].get("safety_critical") or case["predicate"].get("tool") is not None
        ]
        noncritical = [case["id"] for case in regressions if case["id"] not in critical]
        result = {
            "fp32_lane_passed": lanes[f"fp32/{token_form}"]["passed"],
            "w8_lane_passed": lanes[f"w8/{token_form}"]["passed"],
            "critical_w8_regressions": critical,
            "noncritical_w8_regressions": noncritical,
            "w8_regression_gate_passed": not critical and len(noncritical) <= 1,
        }
        result["qualified"] = bool(
            result["fp32_lane_passed"]
            and result["w8_lane_passed"]
            and result["w8_regression_gate_passed"]
        )
        token_forms[token_form] = result

    qualified = [name for name in TOKEN_FORMS if token_forms[name]["qualified"]]
    selected = qualified[0] if qualified else None
    structural_all_forms = all(lane["structural_passed"] for lane in lanes.values())
    report = {
        "schema": 1,
        "kind": "direct_text_semantic_comparison",
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": corpus_hash,
        "corpus_identity_passed": corpus_identity_passed,
        "runtime_provenance_passed": provenance_passed,
        "source_provenance_consistent": source_provenance_consistent,
        "pocket": {
            **pocket_gate,
            "carrier": pocket_carrier,
            "cases": pocket_grade,
        },
        "lanes": lanes,
        "token_forms": token_forms,
        "structural_all_forms_passed": structural_all_forms,
        "qualified_token_forms": qualified,
        "selected_token_form": selected,
        "limitations": [
            (
                "Safety predicates are deterministic lexical intent checks for this frozen "
                "checkpoint, not a general safety audit."
            ),
            (
                "The corpus is public and model-blind at inference time; it must not be reused "
                "as a holdout after training or calibration exposure."
            ),
            (
                "Pocket is a carrier/reference diagnostic. Carrier-inconclusive cases are "
                "never graded green, and Pocket semantic failures remain visible, but only "
                "the clean-token FP32/W8 lanes qualify direct text."
            ),
        ],
    }
    report["passed"] = bool(
        corpus_identity_passed
        and provenance_passed
        and source_provenance_consistent
        and pocket_evidence_complete
        and structural_all_forms
        and selected is not None
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pocket", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--w8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(
            corpus_path=args.corpus,
            pocket=_load_report(args.pocket),
            fp32=_load_report(args.fp32),
            w8=_load_report(args.w8),
        )
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
