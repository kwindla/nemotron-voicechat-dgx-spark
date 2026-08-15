#!/usr/bin/env python3
"""Logical-position oracle and exact state comparisons for Step 5 Phase 1."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch

Scalar: TypeAlias = str | int | float | bool | None
ExactValue: TypeAlias = (
    Scalar | torch.Tensor | np.ndarray | Mapping[str, "ExactValue"] | Sequence["ExactValue"]
)
TensorTree: TypeAlias = torch.Tensor | Mapping[str, "TensorTree"] | Sequence["TensorTree"]
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_CUSTOM_OUTPUT_KEYS = frozenset({"function_tokens"})

# The preregistration commits cross-lane equality only at the logical request
# boundary.  The remaining StateSnapshot members are retained, within-lane
# evidence (transition provenance, physical allocation, and pre/post digests),
# and can be lane-absolute by construction.
CROSS_LANE_EXACT_FIELDS = (
    "emitted_text_token",
    "emitted_function_token",
    "emitted_conjunction",
    "accepted_token_count",
    "selected_packed_row",
    "selected_source_row_identity",
    "selected_custom_outputs",
    "committed_kv_state",
    "committed_mamba_conv_state",
    "committed_mamba_ssm_state",
    "shadow_index",
    "shadow_pointer",
    "virtual_engine_identity",
    "host_pending_rows",
    "correction_flags",
    "correction_assumptions",
)
CROSS_LANE_RETAINED_EVIDENCE_FIELDS = (
    "position_one_boundary_state",
    "uncommitted_second_position_state",
    "operation_evidence",
    "transition_id",
    "originating_rejection_id",
    "pending_row_sha256_before",
    "pending_row_sha256_after",
    "committed_state_sha256_before",
    "committed_state_sha256_after",
)


class TransactionKind(str, enum.Enum):
    PAD_ACCEPT = "pad_accept"
    TEXT_REJECTION = "text_only_rejection"
    FUNCTION_REJECTION = "function_only_rejection"
    CONJUNCTIVE_REJECTION = "conjunctive_rejection"
    CONTROL_BOUNDARY = "control_boundary"
    ABORT = "abort"
    PENDING_DRAIN = "pending_drain"


@dataclass(frozen=True)
class OraclePlan:
    kind: TransactionKind
    positions: tuple[int, ...]
    selected_packed_row: int
    accepted_token_count: int
    correct_and_retain_second_row: bool
    pending_drain_is_separate_transition: bool


def logical_position_plan(text_token: int, function_token: int, pad_token: int) -> OraclePlan:
    """Return the preregistered sequential transaction for packed row outputs."""
    text_pad = text_token == pad_token
    function_pad = function_token == pad_token
    if text_pad and function_pad:
        return OraclePlan(
            TransactionKind.PAD_ACCEPT,
            (0, 1),
            1,
            2,
            False,
            False,
        )
    if not text_pad and function_pad:
        kind = TransactionKind.TEXT_REJECTION
    elif text_pad and not function_pad:
        kind = TransactionKind.FUNCTION_REJECTION
    else:
        kind = TransactionKind.CONJUNCTIVE_REJECTION
    return OraclePlan(kind, (0,), 0, 1, True, True)


@dataclass(frozen=True)
class StateSnapshot:
    """Required, typed evidence for one completed Phase-1 transition."""

    emitted_text_token: int | None
    emitted_function_token: int | None
    emitted_conjunction: bool | None
    accepted_token_count: int
    selected_packed_row: int
    selected_source_row_identity: str
    selected_custom_outputs: Mapping[str, ExactValue]
    committed_kv_state: TensorTree
    committed_mamba_conv_state: TensorTree
    committed_mamba_ssm_state: TensorTree
    rollback_shadow: TensorTree
    shadow_index: int
    shadow_pointer: str
    virtual_engine_identity: str
    host_pending_rows: ExactValue
    correction_flags: Mapping[str, bool]
    correction_assumptions: Mapping[str, int]
    position_one_boundary_state: ExactValue
    uncommitted_second_position_state: ExactValue
    operation_evidence: Mapping[str, ExactValue]
    transition_id: str
    originating_rejection_id: str | None
    pending_row_sha256_before: str | None
    pending_row_sha256_after: str | None
    committed_state_sha256_before: str
    committed_state_sha256_after: str


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def seal_exact_value(value: ExactValue | StateSnapshot) -> dict[str, object]:
    """Return canonical, JSON-safe per-leaf evidence without retaining tensors.

    Tensor leaves retain dtype, shape, byte count, and a content digest. Scalar
    leaves retain their exact type/value and a canonical digest. Container
    topology is explicit, so a reviewer can audit every compared path after the
    GPU process and its torch payloads have been destroyed.
    """

    leaves: dict[str, dict[str, object]] = {}

    def visit(item: object, path: str) -> None:
        if isinstance(item, torch.Tensor):
            contents = _tensor_bytes(item)
            leaves[path] = {
                "byte_count": len(contents),
                "dtype": str(item.dtype),
                "kind": "tensor",
                "sha256": hashlib.sha256(contents).hexdigest(),
                "shape": list(item.shape),
            }
            return
        if isinstance(item, np.ndarray):
            contents = np.ascontiguousarray(item).tobytes(order="C")
            leaves[path] = {
                "byte_count": len(contents),
                "dtype": str(item.dtype),
                "kind": "ndarray",
                "sha256": hashlib.sha256(contents).hexdigest(),
                "shape": list(item.shape),
            }
            return
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}")
            return
        if isinstance(item, Mapping):
            leaves[f"{path}.__keys__"] = _scalar_seal([str(key) for key in sorted(item, key=str)])
            for key in sorted(item, key=str):
                visit(item[key], f"{path}[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            leaves[f"{path}.__sequence__"] = _scalar_seal(
                {"length": len(item), "type": type(item).__name__}
            )
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        leaves[path] = _scalar_seal(item)

    def _scalar_seal(item: object) -> dict[str, object]:
        encoded = json.dumps(
            {"type": type(item).__name__, "value": item},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return {
            "byte_count": len(encoded),
            "dtype": type(item).__name__,
            "kind": "scalar",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "shape": [],
            "value": item,
        }

    visit(value, "snapshot")
    payload: dict[str, object] = {"leaves": leaves, "schema_version": 1}
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


@dataclass(frozen=True)
class ExactnessDifference:
    path: str
    reason: str


@dataclass(frozen=True)
class ExactnessResult:
    exact: bool
    differences: tuple[ExactnessDifference, ...]


@dataclass(frozen=True)
class DriftCharacterization:
    """Directional numeric evidence plus fail-closed exact-field findings."""

    exact_metadata: bool
    differences: tuple[ExactnessDifference, ...]
    statistics: Mapping[str, Mapping[str, Any]]


NUMERIC_DRIFT_ORIENTATION = "candidate_packed_minus_sequential_s"
NUMERIC_DRIFT_FIELDS = (
    "committed_mamba_conv_state",
    "committed_mamba_ssm_state",
)
NUMERIC_SELECTED_OUTPUT_KEYS = frozenset({"function_logits"})
NUMERIC_STATISTIC_KEYS = frozenset(
    {
        "different_byte_count",
        "different_element_count",
        "dtype",
        "element_count",
        "first_different_byte_offset",
        "first_different_candidate_value",
        "first_different_element_offset",
        "first_different_oracle_value",
        "l1_norm",
        "l2_norm",
        "linf_norm",
        "negative_count",
        "orientation",
        "positive_count",
        "shape",
        "signed_delta_lower",
        "signed_delta_upper",
        "zero_count",
    }
)


def _compare_exact(
    left: ExactValue | StateSnapshot,
    right: ExactValue | StateSnapshot,
    path: str,
    out: list[ExactnessDifference],
) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            out.append(ExactnessDifference(path, "tensor/type mismatch"))
            return
        if left.dtype != right.dtype:
            out.append(ExactnessDifference(path, f"dtype {left.dtype} != {right.dtype}"))
            return
        if tuple(left.shape) != tuple(right.shape):
            reason = f"shape {tuple(left.shape)} != {tuple(right.shape)}"
            out.append(ExactnessDifference(path, reason))
        else:
            left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
            right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
            if not torch.equal(left_bytes, right_bytes):
                flat_left = left_bytes.reshape(-1)
                flat_right = right_bytes.reshape(-1)
                byte_mask = flat_left != flat_right
                first_byte = int(torch.nonzero(byte_mask, as_tuple=False)[0].item())
                reason = (
                    "tensor bytes differ: "
                    f"different_bytes={int(byte_mask.sum().item())}/{flat_left.numel()}, "
                    f"first_byte_offset={first_byte}, "
                    f"left_byte={int(flat_left[first_byte])}, "
                    f"right_byte={int(flat_right[first_byte])}"
                )
                if left.dtype == right.dtype and (
                    left.dtype.is_floating_point or left.dtype.is_complex
                ):
                    flat_values_left = left.reshape(-1)
                    flat_values_right = right.reshape(-1)
                    value_mask = flat_values_left != flat_values_right
                    value_mask |= torch.isnan(flat_values_left) != torch.isnan(flat_values_right)
                    if bool(value_mask.any()):
                        first_value = int(
                            torch.nonzero(value_mask, as_tuple=False)[0].item()
                        )
                        delta = (
                            flat_values_left.to(torch.complex128)
                            - flat_values_right.to(torch.complex128)
                        ).abs()
                        finite = torch.isfinite(delta)
                        max_delta = (
                            float(delta[finite].max().item())
                            if bool(finite.any())
                            else None
                        )
                        reason += (
                            f", different_elements={int(value_mask.sum().item())}/"
                            f"{flat_values_left.numel()}, first_element_offset={first_value}, "
                            f"left_value={flat_values_left[first_value].item()!r}, "
                            f"right_value={flat_values_right[first_value].item()!r}, "
                            f"max_abs_delta={max_delta!r}"
                        )
                out.append(ExactnessDifference(path, reason))
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            out.append(ExactnessDifference(path, "ndarray/type mismatch"))
            return
        if left.dtype != right.dtype or left.shape != right.shape:
            out.append(ExactnessDifference(path, "ndarray dtype/shape mismatch"))
        elif left.tobytes(order="C") != right.tobytes(order="C"):
            out.append(ExactnessDifference(path, "ndarray bytes differ"))
        return
    if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
        if type(left) is not type(right):
            out.append(ExactnessDifference(path, "dataclass type mismatch"))
            return
        for field in dataclasses.fields(left):
            _compare_exact(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{path}.{field.name}",
                out,
            )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            out.append(ExactnessDifference(path, "mapping/type mismatch"))
            return
        if set(left) != set(right):
            out.append(ExactnessDifference(path, "mapping keys differ"))
            return
        for key in sorted(left, key=str):
            _compare_exact(left[key], right[key], f"{path}[{key!r}]", out)
        return
    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if type(left) is not type(right) or len(left) != len(right):
            out.append(ExactnessDifference(path, "sequence type/length mismatch"))
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_exact(left_item, right_item, f"{path}[{index}]", out)
        return
    if type(left) is not type(right) or left != right:
        out.append(ExactnessDifference(path, f"{left!r} != {right!r}"))


def _flatten_tensor_tree(
    value: object,
    path: str,
    leaves: dict[str, torch.Tensor],
    differences: list[ExactnessDifference],
) -> None:
    if isinstance(value, torch.Tensor):
        leaves[path] = value
        return
    if isinstance(value, Mapping):
        if not value:
            differences.append(ExactnessDifference(path, "numeric tensor tree is empty"))
            return
        for key in sorted(value, key=str):
            _flatten_tensor_tree(value[key], f"{path}[{key!r}]", leaves, differences)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            differences.append(ExactnessDifference(path, "numeric tensor tree is empty"))
            return
        for index, child in enumerate(value):
            _flatten_tensor_tree(child, f"{path}[{index}]", leaves, differences)
        return
    differences.append(ExactnessDifference(path, "numeric leaf is not a tensor"))


def _directional_tensor_statistics(
    oracle: torch.Tensor,
    candidate: torch.Tensor,
    path: str,
    differences: list[ExactnessDifference],
) -> dict[str, Any] | None:
    if oracle.dtype != candidate.dtype:
        differences.append(
            ExactnessDifference(path, f"dtype {oracle.dtype} != {candidate.dtype}")
        )
        return None
    if tuple(oracle.shape) != tuple(candidate.shape):
        differences.append(
            ExactnessDifference(path, f"shape {tuple(oracle.shape)} != {tuple(candidate.shape)}")
        )
        return None
    if not oracle.dtype.is_floating_point or oracle.dtype.is_complex:
        differences.append(
            ExactnessDifference(path, "numeric envelope requires a real float tensor")
        )
        return None
    oracle_cpu = oracle.detach().cpu().contiguous()
    candidate_cpu = candidate.detach().cpu().contiguous()
    if not bool(torch.isfinite(oracle_cpu).all()) or not bool(torch.isfinite(candidate_cpu).all()):
        differences.append(ExactnessDifference(path, "non-finite numeric envelope input"))
        return None

    oracle_flat = oracle_cpu.reshape(-1)
    candidate_flat = candidate_cpu.reshape(-1)
    delta = candidate_flat.to(torch.float64) - oracle_flat.to(torch.float64)
    positive = delta > 0
    negative = delta < 0
    zero = delta == 0
    different = ~zero
    oracle_bytes = oracle_cpu.view(torch.uint8).reshape(-1)
    candidate_bytes = candidate_cpu.view(torch.uint8).reshape(-1)
    byte_mask = oracle_bytes != candidate_bytes
    first_byte = (
        int(torch.nonzero(byte_mask, as_tuple=False)[0].item())
        if bool(byte_mask.any())
        else None
    )
    first_element = (
        int(torch.nonzero(different, as_tuple=False)[0].item())
        if bool(different.any())
        else None
    )
    absolute = delta.abs()
    return {
        "different_byte_count": int(byte_mask.sum().item()),
        "different_element_count": int(different.sum().item()),
        "dtype": str(oracle.dtype),
        "element_count": int(delta.numel()),
        "first_different_byte_offset": first_byte,
        "first_different_candidate_value": (
            None if first_element is None else candidate_flat[first_element].item()
        ),
        "first_different_element_offset": first_element,
        "first_different_oracle_value": (
            None if first_element is None else oracle_flat[first_element].item()
        ),
        "l1_norm": float(absolute.sum().item()),
        "l2_norm": float(torch.linalg.vector_norm(delta, ord=2).item()),
        "linf_norm": float(absolute.max().item()) if delta.numel() else 0.0,
        "negative_count": int(negative.sum().item()),
        "orientation": NUMERIC_DRIFT_ORIENTATION,
        "positive_count": int(positive.sum().item()),
        "shape": list(oracle.shape),
        "signed_delta_lower": float(delta.min().item()) if delta.numel() else 0.0,
        "signed_delta_upper": float(delta.max().item()) if delta.numel() else 0.0,
        "zero_count": int(zero.sum().item()),
    }


def characterize_snapshot_drift(
    oracle: StateSnapshot,
    candidate: StateSnapshot,
) -> DriftCharacterization:
    """Characterize only the sealed pair-vs-S numeric weakening.

    Delta orientation is always ``candidate packed - sequential S``.  All
    metadata, discrete fields, committed KV, and non-waived custom outputs
    remain byte exact.  Transaction rollback uses the same directional
    characterization as committed Mamba state; other operations do not compare
    rollback because they do not refresh the pair shadow.
    """

    differences: list[ExactnessDifference] = []
    statistics: dict[str, Mapping[str, Any]] = {}
    exact_fields = tuple(field for field in CROSS_LANE_EXACT_FIELDS if field not in {
        "selected_custom_outputs",
        *NUMERIC_DRIFT_FIELDS,
    })
    for field in exact_fields:
        _compare_exact(
            getattr(oracle, field),
            getattr(candidate, field),
            f"snapshot.{field}",
            differences,
        )

    oracle_outputs = oracle.selected_custom_outputs
    candidate_outputs = candidate.selected_custom_outputs
    if set(oracle_outputs) != set(candidate_outputs):
        differences.append(
            ExactnessDifference("snapshot.selected_custom_outputs", "mapping keys differ")
        )
    else:
        for key in sorted(oracle_outputs):
            path = f"snapshot.selected_custom_outputs[{key!r}]"
            if key in NUMERIC_SELECTED_OUTPUT_KEYS:
                oracle_leaves: dict[str, torch.Tensor] = {}
                candidate_leaves: dict[str, torch.Tensor] = {}
                _flatten_tensor_tree(oracle_outputs[key], path, oracle_leaves, differences)
                _flatten_tensor_tree(candidate_outputs[key], path, candidate_leaves, differences)
                if set(oracle_leaves) != set(candidate_leaves):
                    differences.append(ExactnessDifference(path, "numeric tensor leaves differ"))
                else:
                    for leaf_path in sorted(oracle_leaves):
                        statistic = _directional_tensor_statistics(
                            oracle_leaves[leaf_path],
                            candidate_leaves[leaf_path],
                            leaf_path,
                            differences,
                        )
                        if statistic is not None:
                            statistics[leaf_path] = statistic
            else:
                _compare_exact(oracle_outputs[key], candidate_outputs[key], path, differences)

    numeric_roots = [
        (field, getattr(oracle, field), getattr(candidate, field))
        for field in NUMERIC_DRIFT_FIELDS
    ]
    oracle_operation = (
        oracle.operation_evidence.get("operation")
        if isinstance(oracle.operation_evidence, Mapping)
        else None
    )
    candidate_operation = (
        candidate.operation_evidence.get("operation")
        if isinstance(candidate.operation_evidence, Mapping)
        else None
    )
    if oracle_operation == "transaction" and candidate_operation == "transaction":
        numeric_roots.append(("rollback_shadow", oracle.rollback_shadow, candidate.rollback_shadow))
    for field, oracle_tree, candidate_tree in numeric_roots:
        path = f"snapshot.{field}"
        oracle_leaves = {}
        candidate_leaves = {}
        _flatten_tensor_tree(oracle_tree, path, oracle_leaves, differences)
        _flatten_tensor_tree(candidate_tree, path, candidate_leaves, differences)
        if set(oracle_leaves) != set(candidate_leaves):
            differences.append(ExactnessDifference(path, "numeric tensor leaves differ"))
            continue
        for leaf_path in sorted(oracle_leaves):
            statistic = _directional_tensor_statistics(
                oracle_leaves[leaf_path],
                candidate_leaves[leaf_path],
                leaf_path,
                differences,
            )
            if statistic is not None:
                statistics[leaf_path] = statistic
    return DriftCharacterization(not differences, tuple(differences), statistics)


def compare_snapshot_to_envelope(
    oracle: StateSnapshot,
    candidate: StateSnapshot,
    *,
    lane: str,
    case: str,
    boundary: str,
    envelope: Mapping[str, Any],
) -> ExactnessResult:
    """Apply one sealed directional, lane/case/boundary-specific envelope."""

    characterization = characterize_snapshot_drift(oracle, candidate)
    differences = list(characterization.differences)
    if envelope.get("orientation") != NUMERIC_DRIFT_ORIENTATION:
        differences.append(ExactnessDifference("envelope.orientation", "orientation mismatch"))
        return ExactnessResult(False, tuple(differences))
    try:
        expected = envelope["envelopes"][lane][case][boundary]
    except (KeyError, TypeError):
        differences.append(
            ExactnessDifference(
                f"envelope[{lane!r}][{case!r}][{boundary!r}]",
                "missing lane/case/boundary envelope",
            )
        )
        return ExactnessResult(False, tuple(differences))
    if not isinstance(expected, Mapping):
        differences.append(ExactnessDifference("envelope", "boundary envelope is not a mapping"))
        return ExactnessResult(False, tuple(differences))
    observed_paths = set(characterization.statistics)
    expected_paths = set(expected)
    if observed_paths != expected_paths:
        missing = sorted(expected_paths - observed_paths)
        new = sorted(observed_paths - expected_paths)
        differences.append(
            ExactnessDifference(
                "envelope.numeric_leaves",
                f"numeric leaf set differs: missing={missing}, new={new}",
            )
        )
        return ExactnessResult(False, tuple(differences))

    required = {
        "different_element_count_lower",
        "different_element_count_upper",
        "dtype",
        "element_count",
        "l1_norm_ceiling",
        "l2_norm_ceiling",
        "linf_norm_ceiling",
        "negative_count_lower",
        "negative_count_upper",
        "orientation",
        "positive_count_lower",
        "positive_count_upper",
        "shape",
        "signed_delta_lower",
        "signed_delta_upper",
        "zero_count_lower",
        "zero_count_upper",
        "zero_required",
    }
    for path in sorted(expected_paths):
        rule = expected[path]
        statistic = characterization.statistics[path]
        if not isinstance(rule, Mapping) or not required <= set(rule):
            differences.append(ExactnessDifference(path, "missing envelope statistic"))
            continue
        if rule["orientation"] != NUMERIC_DRIFT_ORIENTATION:
            differences.append(ExactnessDifference(path, "orientation mismatch"))
            continue
        for key in ("dtype", "shape", "element_count"):
            if statistic[key] != rule[key]:
                differences.append(ExactnessDifference(path, f"{key} metadata mismatch"))
        for key in ("different_element_count", "positive_count", "negative_count", "zero_count"):
            value = statistic[key]
            if value < rule[f"{key}_lower"] or value > rule[f"{key}_upper"]:
                differences.append(ExactnessDifference(path, f"{key} support violation"))
        if statistic["signed_delta_lower"] < rule["signed_delta_lower"]:
            differences.append(ExactnessDifference(path, "signed lower-bound violation"))
        if statistic["signed_delta_upper"] > rule["signed_delta_upper"]:
            differences.append(ExactnessDifference(path, "signed upper-bound violation"))
        for key in ("l1_norm", "l2_norm", "linf_norm"):
            if statistic[key] > rule[f"{key}_ceiling"]:
                differences.append(ExactnessDifference(path, f"{key} inclusive ceiling exceeded"))
        if rule["zero_required"] and any(
            statistic[key] != 0
            for key in (
                "different_element_count",
                "positive_count",
                "negative_count",
                "l1_norm",
                "l2_norm",
                "linf_norm",
                "signed_delta_lower",
                "signed_delta_upper",
            )
        ):
            differences.append(ExactnessDifference(path, "zero envelope violated"))
    return ExactnessResult(not differences, tuple(differences))


def compare_snapshots(oracle: StateSnapshot, candidate: StateSnapshot) -> ExactnessResult:
    """Compare only the preregistered, lane-invariant committed boundary."""
    differences: list[ExactnessDifference] = []
    for field in CROSS_LANE_EXACT_FIELDS:
        _compare_exact(
            getattr(oracle, field),
            getattr(candidate, field),
            f"snapshot.{field}",
            differences,
        )
    oracle_operation = (
        oracle.operation_evidence.get("operation")
        if isinstance(oracle.operation_evidence, Mapping)
        else None
    )
    candidate_operation = (
        candidate.operation_evidence.get("operation")
        if isinstance(candidate.operation_evidence, Mapping)
        else None
    )
    if oracle_operation == "transaction" and candidate_operation == "transaction":
        _compare_exact(
            oracle.rollback_shadow,
            candidate.rollback_shadow,
            "snapshot.rollback_shadow",
            differences,
        )
    return ExactnessResult(not differences, tuple(differences))


def _contains_tensor(value: TensorTree) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, Mapping):
        return bool(value) and all(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_contains_tensor(item) for item in value)
    return False


def validate_snapshot_presence(
    snapshot: StateSnapshot,
    plan: OraclePlan,
) -> ExactnessResult:
    """Fail closed when a required tensor, output, identity, or digest is absent."""
    differences: list[ExactnessDifference] = []
    for field in (
        "committed_kv_state",
        "committed_mamba_conv_state",
        "committed_mamba_ssm_state",
        "rollback_shadow",
    ):
        if not _contains_tensor(getattr(snapshot, field)):
            differences.append(ExactnessDifference(f"snapshot.{field}", "required tensor absent"))
    required_outputs = set() if plan.kind == TransactionKind.ABORT else REQUIRED_CUSTOM_OUTPUT_KEYS
    missing_outputs = required_outputs - set(snapshot.selected_custom_outputs)
    if missing_outputs:
        differences.append(
            ExactnessDifference(
                "snapshot.selected_custom_outputs",
                f"required keys absent: {sorted(missing_outputs)}",
            )
        )
    for key in required_outputs & set(snapshot.selected_custom_outputs):
        if snapshot.selected_custom_outputs[key] is None:
            differences.append(
                ExactnessDifference(
                    f"snapshot.selected_custom_outputs[{key!r}]", "required output absent"
                )
            )
    for field in (
        "selected_source_row_identity",
        "shadow_pointer",
        "virtual_engine_identity",
        "transition_id",
    ):
        if not isinstance(getattr(snapshot, field), str) or not getattr(snapshot, field):
            differences.append(ExactnessDifference(f"snapshot.{field}", "required identity absent"))
    if not snapshot.operation_evidence:
        differences.append(
            ExactnessDifference("snapshot.operation_evidence", "observed operation evidence absent")
        )
    for field in ("committed_state_sha256_before", "committed_state_sha256_after"):
        if not isinstance(getattr(snapshot, field), str) or not SHA256_RE.fullmatch(
            getattr(snapshot, field)
        ):
            differences.append(ExactnessDifference(f"snapshot.{field}", "invalid SHA-256"))
    return ExactnessResult(not differences, tuple(differences))


def validate_snapshot_semantics(
    snapshot: StateSnapshot,
    plan: OraclePlan,
    *,
    text_token: int,
    function_token: int,
    pad_token: int,
) -> ExactnessResult:
    """Validate a lane independently of cross-lane equality."""
    differences: list[ExactnessDifference] = []
    abort = plan.kind == TransactionKind.ABORT
    expected_text = None if abort else text_token
    expected_function = None if abort else function_token
    expected_conjunction = (
        None if abort else bool(text_token == pad_token and function_token == pad_token)
    )
    for field, expected in (
        ("emitted_text_token", expected_text),
        ("emitted_function_token", expected_function),
        ("emitted_conjunction", expected_conjunction),
        ("accepted_token_count", plan.accepted_token_count),
        ("selected_packed_row", plan.selected_packed_row),
    ):
        _compare_exact(getattr(snapshot, field), expected, f"snapshot.{field}", differences)
    if not SHA256_RE.fullmatch(snapshot.selected_source_row_identity):
        differences.append(
            ExactnessDifference(
                "snapshot.selected_source_row_identity", "source-row content SHA-256 absent"
            )
        )
    pending_rows = snapshot.host_pending_rows
    pending = not (
        pending_rows is None
        or (isinstance(pending_rows, (dict, list, set, tuple)) and len(pending_rows) == 0)
    )
    if plan.correct_and_retain_second_row and not pending:
        differences.append(
            ExactnessDifference("snapshot.host_pending_rows", "corrected row is not pending")
        )
    if plan.kind == TransactionKind.PENDING_DRAIN and pending:
        differences.append(
            ExactnessDifference("snapshot.host_pending_rows", "pending drain retained a row")
        )
    if plan.correct_and_retain_second_row:
        if snapshot.correction_flags.get("corrected") is not True:
            differences.append(
                ExactnessDifference(
                    "snapshot.correction_flags", "rejected row lacks correction evidence"
                )
            )
        if not snapshot.correction_assumptions:
            differences.append(
                ExactnessDifference(
                    "snapshot.correction_assumptions", "rejected row lacks assumptions"
                )
            )
        if snapshot.originating_rejection_id != snapshot.transition_id:
            differences.append(
                ExactnessDifference(
                    "snapshot.originating_rejection_id", "rejection is not self-originating"
                )
            )
        if not snapshot.pending_row_sha256_after or not SHA256_RE.fullmatch(
            snapshot.pending_row_sha256_after
        ):
            differences.append(
                ExactnessDifference(
                    "snapshot.pending_row_sha256_after", "corrected row digest absent"
                )
            )
    return ExactnessResult(not differences, tuple(differences))


def validate_rejection_restore(snapshot: StateSnapshot) -> ExactnessResult:
    """Require uncommitted position-two state to be absent or position-one exact."""
    if snapshot.uncommitted_second_position_state is None:
        return ExactnessResult(True, ())
    differences: list[ExactnessDifference] = []
    _compare_exact(
        snapshot.position_one_boundary_state,
        snapshot.uncommitted_second_position_state,
        "snapshot.uncommitted_second_position_state",
        differences,
    )
    return ExactnessResult(not differences, tuple(differences))


def validate_pending_drain(
    rejection: StateSnapshot,
    drain: StateSnapshot,
) -> ExactnessResult:
    """Join a drain to its immediately preceding corrected rejection row."""
    differences: list[ExactnessDifference] = []
    expected_origin = rejection.transition_id
    if drain.originating_rejection_id != expected_origin:
        differences.append(
            ExactnessDifference(
                "snapshot.originating_rejection_id",
                f"{drain.originating_rejection_id!r} != {expected_origin!r}",
            )
        )
    if drain.pending_row_sha256_before != rejection.pending_row_sha256_after:
        differences.append(
            ExactnessDifference("snapshot.pending_row_sha256_before", "origin row digest mismatch")
        )
    if drain.selected_source_row_identity != drain.pending_row_sha256_before:
        differences.append(
            ExactnessDifference(
                "snapshot.selected_source_row_identity",
                "drain source identity does not match the executed pending row",
            )
        )
    if drain.pending_row_sha256_after is not None:
        differences.append(
            ExactnessDifference("snapshot.pending_row_sha256_after", "drained row remains pending")
        )
    if drain.committed_state_sha256_before != rejection.committed_state_sha256_after:
        differences.append(
            ExactnessDifference(
                "snapshot.committed_state_sha256_before", "rejection boundary state mismatch"
            )
        )
    if drain.committed_state_sha256_after == drain.committed_state_sha256_before:
        differences.append(
            ExactnessDifference(
                "snapshot.committed_state_sha256_after", "drain did not advance committed state"
            )
        )
    return ExactnessResult(not differences, tuple(differences))
