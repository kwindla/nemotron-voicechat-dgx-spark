from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "tools/qualification/step5_phase1_oracle.py"
SPEC = importlib.util.spec_from_file_location("step5_phase1_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)

StateSnapshot = oracle.StateSnapshot
TransactionKind = oracle.TransactionKind
compare_snapshots = oracle.compare_snapshots
compare_snapshot_to_envelope = oracle.compare_snapshot_to_envelope
characterize_snapshot_drift = oracle.characterize_snapshot_drift
logical_position_plan = oracle.logical_position_plan
validate_rejection_restore = oracle.validate_rejection_restore
validate_pending_drain = oracle.validate_pending_drain
validate_snapshot_presence = oracle.validate_snapshot_presence
validate_snapshot_semantics = oracle.validate_snapshot_semantics


def _snapshot() -> StateSnapshot:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    return StateSnapshot(
        emitted_text_token=12,
        emitted_function_token=12,
        emitted_conjunction=True,
        accepted_token_count=2,
        selected_packed_row=1,
        selected_source_row_identity="c" * 64,
        selected_custom_outputs={"function_tokens": torch.tensor([12])},
        committed_kv_state=tensor,
        committed_mamba_conv_state={"layer": tensor.clone()},
        committed_mamba_ssm_state=(tensor.clone(),),
        rollback_shadow=tensor.clone(),
        shadow_index=2,
        shadow_pointer="shadow/0",
        virtual_engine_identity="ve/0",
        host_pending_rows=[],
        correction_flags={"corrected": False},
        correction_assumptions={"text": 12, "function": 12},
        position_one_boundary_state={"kv": [1], "mamba": [2]},
        uncommitted_second_position_state=None,
        operation_evidence={"accepted": 2, "operation": "transaction"},
        transition_id="frozen-prefix-pad-accept",
        originating_rejection_id=None,
        pending_row_sha256_before=None,
        pending_row_sha256_after=None,
        committed_state_sha256_before="a" * 64,
        committed_state_sha256_after="b" * 64,
    )


@pytest.mark.parametrize(
    ("text", "function", "kind", "positions", "selected"),
    [
        (12, 12, TransactionKind.PAD_ACCEPT, (0, 1), 1),
        (7, 12, TransactionKind.TEXT_REJECTION, (0,), 0),
        (12, 8, TransactionKind.FUNCTION_REJECTION, (0,), 0),
        (7, 8, TransactionKind.CONJUNCTIVE_REJECTION, (0,), 0),
    ],
)
def test_logical_position_oracle_semantics(
    text: int,
    function: int,
    kind: TransactionKind,
    positions: tuple[int, ...],
    selected: int,
) -> None:
    plan = logical_position_plan(text, function, 12)
    assert plan.kind == kind
    assert plan.positions == positions
    assert plan.selected_packed_row == selected
    assert plan.pending_drain_is_separate_transition == (kind != TransactionKind.PAD_ACCEPT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("emitted_text_token", 13),
        ("emitted_function_token", 13),
        ("emitted_conjunction", False),
        ("accepted_token_count", 1),
        ("selected_packed_row", 0),
        ("selected_source_row_identity", "d" * 64),
        ("selected_custom_outputs", {"function_tokens": torch.tensor([13])}),
        ("committed_kv_state", torch.tensor([[1.0, 3.0]])),
        ("committed_mamba_conv_state", {"layer": torch.tensor([[1.0, 3.0]])}),
        ("committed_mamba_ssm_state", (torch.tensor([[1.0, 3.0]]),)),
        ("rollback_shadow", torch.tensor([[1.0, 3.0]])),
        ("shadow_index", 1),
        ("shadow_pointer", "shadow/1"),
        ("virtual_engine_identity", "ve/1"),
        ("host_pending_rows", [{"row": 1}]),
        ("correction_flags", {"corrected": True}),
        ("correction_assumptions", {"text": 0, "function": 12}),
    ],
)
def test_exact_comparator_covers_every_committed_contract_item(field: str, value: object) -> None:
    oracle = _snapshot()
    result = compare_snapshots(oracle, replace(oracle, **{field: value}))
    assert not result.exact
    assert any(field in difference.path for difference in result.differences)


def test_exact_comparator_rejects_tensor_dtype_and_shape_drift() -> None:
    oracle = _snapshot()
    dtype = compare_snapshots(
        oracle, replace(oracle, committed_kv_state=oracle.committed_kv_state.double())
    )
    shape = compare_snapshots(
        oracle, replace(oracle, committed_kv_state=oracle.committed_kv_state.reshape(2, 1))
    )
    assert any("dtype" in item.reason for item in dtype.differences)
    assert any("shape" in item.reason for item in shape.differences)


@pytest.mark.parametrize("field", oracle.CROSS_LANE_RETAINED_EVIDENCE_FIELDS)
def test_exact_comparator_retains_but_excludes_lane_absolute_evidence(field: str) -> None:
    baseline = _snapshot()
    candidate = replace(baseline, **{field: "lane-absolute-evidence"})
    assert compare_snapshots(baseline, candidate).exact


def test_exact_comparator_only_compares_transaction_rollback_projection() -> None:
    baseline = _snapshot()
    changed = replace(baseline, rollback_shadow=torch.tensor([[9.0, 3.0]]))
    assert not compare_snapshots(baseline, changed).exact
    control = replace(baseline, operation_evidence={"operation": "control"})
    changed_control = replace(changed, operation_evidence={"operation": "control"})
    assert compare_snapshots(control, changed_control).exact


def test_tensor_difference_reports_leaf_level_byte_and_value_evidence() -> None:
    baseline = _snapshot()
    changed = replace(baseline, committed_kv_state=torch.tensor([[1.0, 3.0]]))
    difference = compare_snapshots(baseline, changed).differences[0]
    assert "different_bytes=" in difference.reason
    assert "first_element_offset=" in difference.reason
    assert "max_abs_delta=" in difference.reason


def _directional_envelope(statistics: dict[str, dict[str, object]]) -> dict[str, object]:
    leaves = {}
    for path, item in statistics.items():
        leaves[path] = {
            "different_element_count_lower": item["different_element_count"],
            "different_element_count_upper": item["different_element_count"],
            "dtype": item["dtype"],
            "element_count": item["element_count"],
            "l1_norm_ceiling": item["l1_norm"],
            "l2_norm_ceiling": item["l2_norm"],
            "linf_norm_ceiling": item["linf_norm"],
            "negative_count_lower": item["negative_count"],
            "negative_count_upper": item["negative_count"],
            "orientation": oracle.NUMERIC_DRIFT_ORIENTATION,
            "positive_count_lower": item["positive_count"],
            "positive_count_upper": item["positive_count"],
            "shape": item["shape"],
            "signed_delta_lower": item["signed_delta_lower"],
            "signed_delta_upper": item["signed_delta_upper"],
            "zero_count_lower": item["zero_count"],
            "zero_count_upper": item["zero_count"],
            "zero_required": item["different_element_count"] == 0,
        }
    return {
        "orientation": oracle.NUMERIC_DRIFT_ORIENTATION,
        "envelopes": {"A": {"case": {"B9": leaves}}},
    }


def test_directional_characterization_is_candidate_minus_s_and_retains_exact_fields() -> None:
    baseline = _snapshot()
    candidate = replace(
        baseline,
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([0.5, 1.0]),
        },
    )
    baseline = replace(
        baseline,
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([1.0, 0.5]),
        },
    )
    result = characterize_snapshot_drift(baseline, candidate)
    assert result.exact_metadata
    statistic = result.statistics["snapshot.selected_custom_outputs['function_logits']"]
    assert statistic["orientation"] == "candidate_packed_minus_sequential_s"
    assert statistic["signed_delta_lower"] == -0.5
    assert statistic["signed_delta_upper"] == 0.5
    assert statistic["positive_count"] == 1
    assert statistic["negative_count"] == 1
    assert statistic["zero_count"] == 0


def test_directional_envelope_fails_sign_reversal_new_leaf_and_inclusive_exceedance() -> None:
    baseline = replace(
        _snapshot(),
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([0.0, 0.0]),
        },
    )
    candidate = replace(
        baseline,
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([1.0, -1.0]),
        },
    )
    characterized = characterize_snapshot_drift(baseline, candidate)
    envelope = _directional_envelope(dict(characterized.statistics))
    assert compare_snapshot_to_envelope(
        baseline, candidate, lane="A", case="case", boundary="B9", envelope=envelope
    ).exact

    reversed_candidate = replace(
        candidate,
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([-1.0, 1.0]),
        },
    )
    reversed_result = compare_snapshot_to_envelope(
        baseline,
        reversed_candidate,
        lane="A",
        case="case",
        boundary="B9",
        envelope=envelope,
    )
    # Aggregate bounds are symmetric here, but support/norm and counts remain
    # exact; a deliberately asymmetric rule proves orientation is enforced.
    path = "snapshot.selected_custom_outputs['function_logits']"
    envelope["envelopes"]["A"]["case"]["B9"][path]["signed_delta_lower"] = 0.0
    assert not compare_snapshot_to_envelope(
        baseline,
        reversed_candidate,
        lane="A",
        case="case",
        boundary="B9",
        envelope=envelope,
    ).exact
    assert reversed_result.exact

    exceeded = replace(
        candidate,
        selected_custom_outputs={
            "function_tokens": torch.tensor([12]),
            "function_logits": torch.tensor([2.0, -1.0]),
        },
    )
    assert not compare_snapshot_to_envelope(
        baseline, exceeded, lane="A", case="case", boundary="B9", envelope=envelope
    ).exact


def test_directional_envelope_fails_orientation_and_missing_statistic() -> None:
    baseline = _snapshot()
    characterized = characterize_snapshot_drift(baseline, baseline)
    envelope = _directional_envelope(dict(characterized.statistics))
    envelope["orientation"] = "sequential_s_minus_candidate_packed"
    result = compare_snapshot_to_envelope(
        baseline, baseline, lane="A", case="case", boundary="B9", envelope=envelope
    )
    assert not result.exact
    envelope["orientation"] = oracle.NUMERIC_DRIFT_ORIENTATION
    first = next(iter(envelope["envelopes"]["A"]["case"]["B9"].values()))
    first.pop("l2_norm_ceiling")
    assert not compare_snapshot_to_envelope(
        baseline, baseline, lane="A", case="case", boundary="B9", envelope=envelope
    ).exact


def test_rejection_second_position_must_be_absent_or_restored() -> None:
    snapshot = _snapshot()
    assert validate_rejection_restore(snapshot).exact
    restored = replace(
        snapshot,
        uncommitted_second_position_state={"kv": [1], "mamba": [2]},
    )
    assert validate_rejection_restore(restored).exact
    divergent = replace(restored, uncommitted_second_position_state={"kv": [9]})
    assert not validate_rejection_restore(divergent).exact


def test_snapshot_semantics_rejects_wrong_count_and_missing_pending_row() -> None:
    plan = logical_position_plan(7, 12, 12)
    snapshot = replace(
        _snapshot(),
        emitted_text_token=7,
        emitted_function_token=12,
        emitted_conjunction=False,
        accepted_token_count=2,
        host_pending_rows=[],
        correction_flags={"corrected": False},
    )
    result = validate_snapshot_semantics(
        snapshot, plan, text_token=7, function_token=12, pad_token=12
    )
    assert not result.exact
    assert {item.path for item in result.differences} >= {
        "snapshot.accepted_token_count",
        "snapshot.host_pending_rows",
        "snapshot.correction_flags",
    }


@pytest.mark.parametrize(
    "field",
    [
        "committed_kv_state",
        "committed_mamba_conv_state",
        "committed_mamba_ssm_state",
        "rollback_shadow",
    ],
)
def test_snapshot_presence_fails_closed_when_both_lanes_could_omit_tensor(
    field: str,
) -> None:
    snapshot = replace(_snapshot(), **{field: {}})
    result = validate_snapshot_presence(snapshot, logical_position_plan(12, 12, 12))
    assert not result.exact
    assert any(field in item.path for item in result.differences)


def test_snapshot_presence_requires_custom_output_and_selected_row_identity() -> None:
    snapshot = replace(_snapshot(), selected_custom_outputs={}, selected_source_row_identity="")
    result = validate_snapshot_presence(snapshot, logical_position_plan(12, 12, 12))
    assert not result.exact
    assert {item.path for item in result.differences} >= {
        "snapshot.selected_custom_outputs",
        "snapshot.selected_source_row_identity",
    }


def test_pending_drain_is_joined_to_rejection_digest_and_advances_state() -> None:
    rejection = replace(
        _snapshot(),
        transition_id="text-only-rejection",
        originating_rejection_id="text-only-rejection",
        pending_row_sha256_after="c" * 64,
        committed_state_sha256_after="d" * 64,
    )
    drain = replace(
        _snapshot(),
        transition_id="text-only-rejection-pending-drain",
        originating_rejection_id="text-only-rejection",
        pending_row_sha256_before="c" * 64,
        pending_row_sha256_after=None,
        committed_state_sha256_before="d" * 64,
        committed_state_sha256_after="e" * 64,
    )
    assert validate_pending_drain(rejection, drain).exact
    assert not validate_pending_drain(
        rejection, replace(drain, pending_row_sha256_before="f" * 64)
    ).exact
    substituted_identity = replace(drain, selected_source_row_identity="f" * 64)
    mismatch = validate_pending_drain(rejection, substituted_identity)
    assert not mismatch.exact
    assert any(
        item.path == "snapshot.selected_source_row_identity" for item in mismatch.differences
    )
