from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "tools" / "benchmark" / "graph_reproducibility.py"
SPEC = importlib.util.spec_from_file_location("graph_reproducibility", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)


def _block(tokens: list[int], *, valid: bool = True) -> dict:
    expected = [7, 8, 9]
    mismatches = [
        {"call_index": index, "expected_token": wanted, "actual_token": observed}
        for index, (wanted, observed) in enumerate(zip(expected, tokens, strict=True))
        if wanted != observed
    ]
    return {
        "actual_token_vector_sha256": repro.stable_digest(tokens),
        "actual_tokens": tokens,
        "calls": len(tokens),
        "cleanup_ok": valid,
        "diagnostics": [],
        "divergence_runs": repro.divergence_runs(expected, tokens),
        "engine_requests_after_cleanup": 0,
        "engine_requests_before": 0,
        "selected_outputs": [],
        "token_mismatches": mismatches,
        "valid": valid,
    }


def _report(
    label: str,
    blocks: list[dict],
    *,
    eager: bool = False,
    sampling_seed: int = 0,
) -> dict:
    return {
        "blocks": blocks,
        "capture": {"max_call_index": 2},
        "component": {
            "config_sha256": "production-config",
            "enforce_eager": eager,
            "gpu_memory_utilization": 0.42,
            "max_model_len": 12288,
            "path": "/models/nano",
        },
        "distinct_token_vectors": len(
            {block["actual_token_vector_sha256"] for block in blocks}
        ),
        "engine_label": label,
        "hypothesis_controls": {
            "parent_seed": 20260813,
            "prelude_calls": 0,
            "reuse_request_id": False,
            "sampling_seed": sampling_seed,
            "sync_each_call": False,
        },
        "kind": "step5_graph_reproducibility_engine_run",
        "passed": all(block["valid"] for block in blocks),
    }


def _localization_block(*, divergent: bool = False) -> dict:
    expected = [13] * 47
    actual = expected.copy()
    if divergent:
        actual[39] = 14
    mismatches = (
        [{"call_index": 39, "expected_token": 13, "actual_token": 14}]
        if divergent
        else []
    )
    return {
        "actual_token_vector_sha256": repro.stable_digest(actual),
        "actual_tokens": actual,
        "calls": 47,
        "cleanup_ok": True,
        "diagnostics": [],
        "divergence_runs": repro.divergence_runs(expected, actual),
        "engine_requests_after_cleanup": 0,
        "engine_requests_before": 0,
        "selected_outputs": [
            {"call_index": call, "reference_match": True} for call in (45, 46)
        ],
        "token_mismatches": mismatches,
        "valid": True,
    }


def _primary_reports() -> dict[str, dict]:
    exact = _localization_block()
    divergent = _localization_block(divergent=True)
    reports = {
        "g0-1.json": _report("g0-1", [copy.deepcopy(exact) for _ in range(3)]),
        "g0-2.json": _report("g0-2", [copy.deepcopy(divergent) for _ in range(3)]),
        "e0-1.json": _report(
            "e0-1", [copy.deepcopy(exact) for _ in range(3)], eager=True
        ),
    }
    for report in reports.values():
        report["capture"]["max_call_index"] = 46
    return reports


def test_divergence_runs_records_reconvergence_and_trailing_nonreconvergence() -> None:
    assert repro.divergence_runs([1, 2, 3, 4], [1, 8, 3, 9]) == [
        {"start_call": 1, "end_call": 1, "length": 1, "reconverged_call": 2},
        {"start_call": 3, "end_call": 3, "length": 1, "reconverged_call": None},
    ]


def test_production_digest_matches_step4_encoder() -> None:
    torch = __import__("pytest").importorskip("torch")
    decomposition_path = ROOT / "tools" / "benchmark" / "nano_interface_decomposition.py"
    spec = importlib.util.spec_from_file_location(
        "nano_interface_decomposition", decomposition_path
    )
    assert spec is not None and spec.loader is not None
    decomposition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decomposition)
    value = {
        "predicted_token": 1046,
        "function_predicted_token": torch.tensor([12]),
        "request_id": "ignored",
        "nano_engine_benchmark": {"ms": 1},
    }
    assert repro.production_output_digest(value) == decomposition.exact_output_digest(value)


def test_bounded_gate_rejects_control_and_multicall_divergence() -> None:
    block = {
        "cleanup_ok": True,
        "calls": 687,
        "divergence_runs": [
            {"start_call": 39, "end_call": 40, "length": 2, "reconverged_call": 41}
        ],
        "selected_outputs": [{"reference_match": True}],
        "token_mismatches": [
            {"call_index": 39, "expected_token": 10592, "actual_token": 1044},
            {"call_index": 40, "expected_token": 12, "actual_token": 1044},
        ],
        "valid": True,
    }
    result = repro.evaluate_bounded_block(block)
    assert not result["mismatches_at_most_one"]
    assert not result["control_token_mismatches_zero"]
    assert not result["max_divergence_run_at_most_one"]


def test_summary_distinguishes_within_engine_stability_from_between_engine_vectors() -> None:
    exact = _block([7, 8, 9])
    divergent = _block([7, 80, 9])
    summary = repro.summarize_reports(
        [_report("a", [exact, exact]), _report("b", [divergent, divergent])]
    )
    cell = summary["cells"][0]
    assert cell["engines"] == 2
    assert cell["blocks"] == 4
    assert cell["distinct_token_vectors"] == 2
    assert cell["within_engine_stable"] is True
    assert cell["exact_prefix_rate"] == 0.5


def test_summary_separates_graph_and_eager_cells() -> None:
    block = _block([7, 8, 9])
    summary = repro.summarize_reports(
        [_report("graph", [block]), _report("eager", [block], eager=True)]
    )
    assert len(summary["cells"]) == 2
    assert {cell["identity"]["enforce_eager"] for cell in summary["cells"]} == {
        False,
        True,
    }


def test_investigation_classifies_reproducible_graph_eager_discriminator_as_defect() -> None:
    summary = repro.build_investigation_summary(_primary_reports())
    assert summary["classification"] == "REAL STATE DEFECT"
    assert summary["classification_basis"]["graph_eager_discriminator"] is True
    assert summary["groups"]["primary_graph_seed0"]["distinct_token_vectors"] == 2
    assert summary["groups"]["primary_eager_seed0"]["exact_blocks"] == 3
    assert summary["bounded_gate"]["status"] == "excluded_real_state_defect"


def test_localization_block_has_no_bounded_rate_or_gate_result() -> None:
    result = repro.evaluate_bounded_block(_localization_block(divergent=True))
    assert result["applicable"] is False
    assert result["mismatch_rate_at_most_one_per_687"] is None
    assert result["passed"] is None


def _empty_selector_groups() -> dict[str, list[dict]]:
    return {
        "primary_graph_seed0": [],
        "graph_seed1": [],
        "graph_history": [],
        "graph_scheduling": [],
    }


def test_seed_conclusion_is_data_derived_but_never_claimed_isolated() -> None:
    exact = _block([7, 8, 9])
    divergent = _block([7, 80, 9])
    groups = _empty_selector_groups()
    groups["primary_graph_seed0"] = [
        _report("g0-exact", [exact] * 3),
        _report("g0-divergent", [divergent] * 3),
    ]
    groups["graph_seed1"] = [
        _report("g1-exact", [exact] * 3, sampling_seed=1),
        _report("g1-divergent", [divergent] * 3, sampling_seed=1),
    ]
    seed = repro._selector_conclusions(groups)["sampler_seed"]
    assert seed["conclusion"] == "no_observed_association_not_isolated"
    assert seed["selects_result"] is None

    changed = _block([7, 81, 9])
    groups["graph_seed1"] = [_report("g1-changed", [changed] * 3, sampling_seed=1)]
    seed = repro._selector_conclusions(groups)["sampler_seed"]
    assert seed["conclusion"] == "observed_association_not_isolated"
    assert seed["observed_association"] is True
    assert seed["selects_result"] is None


def test_history_conclusion_is_derived_and_scoped_to_observed_basin() -> None:
    exact = _block([7, 8, 9])
    blocks = [copy.deepcopy(exact) for _ in range(7)]
    for block, history in zip(
        blocks,
        ["baseline", "post-47", "post-47", "post-47", "post-17", "post-17", "post-17"],
        strict=True,
    ):
        block["history"] = history
    groups = _empty_selector_groups()
    groups["graph_history"] = [_report("history", blocks)]
    history = repro._selector_conclusions(groups)["request_history"]
    assert history["selects_result"] is False
    assert history["tested_basins"] == ["exact"]

    blocks[-1] = _block([7, 80, 9])
    blocks[-1]["history"] = "post-17"
    history = repro._selector_conclusions(groups)["request_history"]
    assert history["selects_result"] is True


def test_request_id_and_synchronization_conclusions_are_data_derived() -> None:
    groups = _empty_selector_groups()
    reports = []
    for label, template in (("exact", _block([7, 8, 9])), ("divergent", _block([7, 80, 9]))):
        blocks = [copy.deepcopy(template) for _ in range(9)]
        for index, block in enumerate(blocks):
            block["request_policy"] = (
                "unique" if index < 3 else "reused" if index < 6 else "synchronized"
            )
            block["sync_each_call"] = index >= 6
        reports.append(_report(label, blocks))
    groups["graph_scheduling"] = reports
    selectors = repro._selector_conclusions(groups)
    assert selectors["request_id_reuse"]["selects_result"] is False
    assert selectors["per_call_synchronization"]["selects_result"] is False
    assert selectors["request_id_reuse"]["tested_basins"] == ["divergent", "exact"]

    reports[0]["blocks"][3].update(_block([7, 81, 9]))
    reports[0]["blocks"][3]["request_policy"] = "reused"
    reports[0]["blocks"][3]["sync_each_call"] = False
    selectors = repro._selector_conclusions(groups)
    assert selectors["request_id_reuse"]["selects_result"] is True


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda block: block.update(valid=False), "recorded_invalid"),
        (lambda block: block.update(cleanup_ok=False), "cleanup_failed"),
        (lambda block: block.update(calls=46), "call_count_mismatch"),
        (
            lambda block: block.update(engine_requests_before=1),
            "requests_present_before_block",
        ),
        (
            lambda block: block.update(engine_requests_after_cleanup=1),
            "requests_present_after_cleanup",
        ),
        (
            lambda block: block["selected_outputs"][0].update(reference_match=False),
            "selected_output_digest_mismatch",
        ),
        (
            lambda block: block.update(
                diagnostics=[{"retained_minus_alternate_logit": math.nan}]
            ),
            "non_finite_diagnostic",
        ),
    ],
)
def test_engine_invalidating_block_conditions_fail_closed(mutation, reason: str) -> None:
    reports = _primary_reports()
    mutation(reports["g0-1.json"]["blocks"][0])
    summary = repro.build_investigation_summary(reports)
    invalid = summary["groups"]["primary_graph_seed0"]["invalid_engine_reasons"]
    assert any(reason in item for item in invalid["g0-1"])
    assert summary["groups"]["primary_graph_seed0"]["distinct_token_vectors"] == 1
    assert summary["classification"] == "INCOMPLETE"
    assert summary["passed"] is False


def test_artifact_or_capture_identity_mismatch_fails_closed() -> None:
    reports = _primary_reports()
    reports["g0-2.json"]["component"]["config_sha256"] = "different-config"
    summary = repro.build_investigation_summary(reports)
    assert summary["classification"] == "INCOMPLETE"
    assert summary["classification_basis"]["all_reports_valid"] is False


def _complete_block() -> dict:
    return {
        "actual_token_vector_sha256": repro.stable_digest([13] * 687),
        "calls": 687,
        "cleanup_ok": True,
        "diagnostics": [],
        "divergence_runs": [],
        "engine_requests_after_cleanup": 0,
        "engine_requests_before": 0,
        "selected_outputs": [
            {"call_index": call, "reference_match": True}
            for call in sorted(repro.SELECTED_CALLS)
        ],
        "token_mismatches": [],
        "valid": True,
    }


def test_global_bounded_gate_enforces_counts_and_production_margin() -> None:
    graph_reports = []
    for engine in range(5):
        report = _report(f"confirm-{engine}", [_complete_block() for _ in range(7)])
        report["capture"]["max_call_index"] = 686
        graph_reports.append(report)
    diagnostic = _report("production-diagnostic", [_block([7, 8, 9])] * 3)
    diagnostic["blocks"][0]["diagnostics"] = [
        {"retained_minus_alternate_logit": 0.04}
    ]
    result = repro.evaluate_global_bounded_gate(
        graph_reports,
        [diagnostic],
        production_config_sha256="production-config",
    )
    assert result["fresh_graph_engines"] == 5
    assert result["measured_687_call_blocks"] == 30
    assert result["production_margin_status"] == "pass"
    assert result["technical_requirements_passed"] is True

    result = repro.evaluate_global_bounded_gate(
        graph_reports[:4], [], production_config_sha256="production-config"
    )
    assert result["requirements"]["fresh_graph_engines_at_least_five"] is False
    assert result["requirements"]["measured_blocks_at_least_30"] is False
    assert result["production_margin_status"] == "unresolved"
    assert result["technical_requirements_passed"] is False
