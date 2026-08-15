#!/usr/bin/env python3
"""Run the preregistered Step 5 Phase-1 matched-lane campaign."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import torch
from step5_phase1_oracle import (
    OraclePlan,
    StateSnapshot,
    TransactionKind,
    compare_snapshots,
    logical_position_plan,
    seal_exact_value,
    validate_pending_drain,
    validate_rejection_restore,
    validate_snapshot_presence,
    validate_snapshot_semantics,
)

from nemotron_voicechat_runtime.patch_step5_wedge_boundary_trace import (
    PHASE1_B_REQUEST_PREFIX,
    PHASE1_MARKER,
    PHASE1_PAD_TOKEN_ID,
    QUALIFIED_FILE_COUNT,
    QUALIFIED_TREE_SHA256,
)
from nemotron_voicechat_runtime.wedge_boundary_trace import BOUNDARIES

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH = REPO_ROOT / "reports/step5-wedge/phase1-preregistration-v6.md"
FIXTURE_PATH = REPO_ROOT / "reports/step5-wedge/phase1-frozen-fixture.json"
DESIGN_PATH = REPO_ROOT / "docs/step5-pairing-wedge-closure-design.md"
TRACE_SOURCE_PATH = REPO_ROOT / "src/nemotron_voicechat_runtime/wedge_boundary_trace.py"
PATCHER_PATH = REPO_ROOT / "src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py"
APPLICATION_REVISION = "1d8163831ef0ba29a1a41e6cab9f0cb4a411bd87"
V2_APPLICATION_REVISION = "15bcb8e5fb16dfa1fa987def60e7c130c00c82bc"
V4_APPLICATION_REVISION = "cf0e9ca89d2048717a83c15570c2c39630b7e558"
V2_PREREGISTRATION_SHA256 = "f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5"
V3_PREREGISTRATION_SHA256 = "8c3494e8f755f9b6f252ba2306b34af7fcd8cecd38e18285a83ee65f202c1d64"
V4_PREREGISTRATION_SHA256 = "a97b197b1f80c61b438fa5f0c6c15627ca407f56664ef809a1b79a2203e6160d"
V5_PREREGISTRATION_SHA256 = "c685a247623233522993cd0511193fc959e97b3debd42f65a4427cde740c370a"
V3_RESEAL_REASON = (
    "HEAD moved only for unrelated committed FHW8 promotion/qualification and "
    "tool-loop defect fixes; all round-2 frozen Phase-1 hashes were reverified"
)
V4_RESEAL_REASON = (
    "bind the implemented real process-isolated executor, device worker, exact "
    "0.42 configuration, and trace-only B5 list compatibility repair"
)
V5_RESEAL_REASON = (
    "bind the round-3 executor review repairs for executed-row identity, runtime-recorded "
    "correction assumptions, live wrapper control transitions, and child-side complete "
    "model-artifact verification"
)
V6_RESEAL_REASON = (
    "bind the round-4 committed-boundary comparator, canonical rollback/KV "
    "representations, historical UTF-8 carrier replay through Pocket, and S-oracle-gated smoke"
)
REPLICATES = 12
LANE_ORDER = ("S", "B", "A")
PREREGISTERED_A_FIRST_MISSING = "B4"
LEDGER_NAME = "replicates.jsonl"
IDENTITY_NAME = "compatibility-identity-preregistered.json"
SUMMARY_NAME = "phase1-summary.json"
MANIFEST_NAME = "artifact-manifest.json"
MATERIALIZATION_NAME = "fixture-materialization-seal.json"
PATCH_TARGET = "v1/worker/gpu_model_runner.py"
PHASE1_TARGET_SHA256 = "32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    text_token: int
    function_token: int
    action: str = "transaction"
    control_boundary: str | None = None
    causal_terminal: bool = False
    originating_rejection: str | None = None


def case_matrix(pad_token: int) -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("text-only-rejection", 101, pad_token),
        CaseSpec(
            "text-only-rejection-pending-drain",
            pad_token,
            pad_token,
            action="pending_drain",
            originating_rejection="text-only-rejection",
        ),
        CaseSpec("function-only-rejection", pad_token, 202),
        CaseSpec(
            "function-only-rejection-pending-drain",
            pad_token,
            pad_token,
            action="pending_drain",
            originating_rejection="function-only-rejection",
        ),
        CaseSpec("conjunctive-rejection", 101, 202),
        CaseSpec(
            "conjunctive-rejection-pending-drain",
            pad_token,
            pad_token,
            action="pending_drain",
            originating_rejection="conjunctive-rejection",
        ),
        CaseSpec("agent-bos", 1, pad_token, control_boundary="agent_bos"),
        CaseSpec("agent-eos", 2, pad_token, control_boundary="agent_eos"),
        CaseSpec("function-sotc", pad_token, 20, control_boundary="function_sotc"),
        CaseSpec("function-eotr", pad_token, 22, control_boundary="function_eotr"),
        CaseSpec("post-fc-bos", 1, pad_token, control_boundary="post_fc_bos"),
        CaseSpec("abort", pad_token, pad_token, action="abort"),
        CaseSpec("frozen-prefix-pad-accept", pad_token, pad_token, causal_terminal=True),
    )


def plan_for_case(case: CaseSpec, pad_token: int) -> OraclePlan:
    if case.action == "abort":
        return OraclePlan(TransactionKind.ABORT, (), -1, 0, False, False)
    if case.action == "pending_drain":
        return OraclePlan(TransactionKind.PENDING_DRAIN, (0,), 0, 1, False, True)
    if case.control_boundary is not None:
        return OraclePlan(TransactionKind.CONTROL_BOUNDARY, (0,), 0, 1, False, False)
    return logical_position_plan(case.text_token, case.function_token, pad_token)


def requires_pair_eager(lane: str, case: CaseSpec) -> bool:
    """Return whether this case schedules the sealed B-lane packed PAD pair."""
    return lane == "B" and case.action == "transaction" and case.control_boundary is None


@dataclass(frozen=True)
class TensorSeal:
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class Frame22StateSeal:
    frame: int
    nano_generated_tokens: int
    nano_session_positions: int
    pending_row_count: int
    pending_text_token: int
    pending_function_token: int
    function_state: str
    control_barrier_pending: bool
    committed_kv_state: TensorSeal
    committed_mamba_conv_state: TensorSeal
    committed_mamba_ssm_state: TensorSeal
    rollback_shadow: TensorSeal
    shadow_index: int
    shadow_pointer: str
    virtual_engine_identity: str
    construction_metadata_sha256: str


@dataclass(frozen=True)
class MaterializedFrame22State:
    frame: int
    nano_generated_tokens: int
    nano_session_positions: int
    pending_row_count: int
    pending_text_token: int
    pending_function_token: int
    function_state: str
    control_barrier_pending: bool
    committed_kv_state: torch.Tensor
    committed_mamba_conv_state: torch.Tensor
    committed_mamba_ssm_state: torch.Tensor
    rollback_shadow: torch.Tensor
    shadow_index: int
    shadow_pointer: str
    virtual_engine_identity: str
    construction_metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class MaterializedFixture:
    schema_version: int
    fixture_id: str
    fused_rows: tuple[torch.Tensor, torch.Tensor]
    frame_22_state: MaterializedFrame22State


@dataclass(frozen=True)
class ExecutionResult:
    snapshot: StateSnapshot | None
    boundaries: tuple[str, ...]
    completed: bool
    terminal_unknown: bool
    dispatch_mode: str
    prefix_sha256: str
    observed_plan: OraclePlan
    dispatch_evidence: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def first_missing(self) -> str | None:
        observed = set(self.boundaries)
        return next((boundary for boundary in BOUNDARIES if boundary not in observed), None)

    @property
    def last_boundary(self) -> str | None:
        order = {boundary: index for index, boundary in enumerate(BOUNDARIES)}
        return max(self.boundaries, key=order.__getitem__) if self.boundaries else None


class Phase1Executor(ABC):
    """Adapter boundary between the CPU harness and a future real worker."""

    supports_lane_local_dispatch = True

    @abstractmethod
    def sealed_identity(self) -> dict[str, Any]:
        """Return image/model/materialized-fixture/config hashes for sealing."""

    @abstractmethod
    async def materialize_fixture(self, fixture: dict[str, JsonValue]) -> MaterializedFixture:
        """Materialize and return the two fused rows and completed frame-22 state."""

    @abstractmethod
    async def create_epoch(self, replicate: int) -> Any: ...

    @abstractmethod
    async def restore_lane(
        self, epoch: Any, lane: str, fixture: dict[str, Any], request_id: str
    ) -> str: ...

    @abstractmethod
    async def execute_case(
        self,
        epoch: Any,
        lane: str,
        case: CaseSpec,
        plan: OraclePlan,
        execution: str,
        request_id: str,
    ) -> ExecutionResult: ...

    @abstractmethod
    async def destroy_epoch(self, epoch: Any, reason: str) -> None: ...

    async def preclassify_construction(self, epoch: Any) -> str:
        del epoch
        raise RuntimeError("construction fallback requires a non-destructive preclassifier")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a non-null lowercase 64-hex SHA-256")
    return value


def verify_patch_manifest(path: Path, claimed_sha256: str) -> dict[str, Any]:
    """Bind the CLI hash to the exact Phase-1 patch manifest and qualified tree."""
    _require_sha256(claimed_sha256, "patch manifest hash")
    if not path.is_file():
        raise ValueError(f"patch manifest is absent: {path}")
    actual = sha256_file(path)
    if actual != claimed_sha256:
        raise ValueError(f"patch manifest hash mismatch: expected {claimed_sha256}, got {actual}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("patch manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("patch manifest must be a mapping")
    expected_intervention = {
        "kind": "pair-only-graph-disabled-eager-dispatch",
        "marker": PHASE1_MARKER,
        "pad_token_id": PHASE1_PAD_TOKEN_ID,
        "request_prefix": PHASE1_B_REQUEST_PREFIX,
        "target": PATCH_TARGET,
    }
    if manifest.get("baseline_tree_sha256") != QUALIFIED_TREE_SHA256:
        raise ValueError("patch manifest qualified tree digest mismatch")
    if manifest.get("verified_source_file_count") != QUALIFIED_FILE_COUNT:
        raise ValueError("patch manifest qualified file count mismatch")
    grafts = manifest.get("grafted_binary_data_files")
    if not isinstance(grafts, dict):
        raise ValueError("patch manifest lacks the explicit binary/data graft inventory")
    if manifest.get("atomic_source_file_count") != QUALIFIED_FILE_COUNT + len(grafts):
        raise ValueError("patch manifest atomic source file count mismatch")
    for relative, seal in grafts.items():
        if (
            not isinstance(relative, str)
            or relative.endswith((".py", ".pyc"))
            or "__pycache__" in Path(relative).parts
            or not isinstance(seal, dict)
            or not isinstance(seal.get("bytes"), int)
        ):
            raise ValueError("patch manifest contains an inadmissible binary/data graft")
        _require_sha256(seal.get("sha256"), f"patch graft {relative}")
    if manifest.get("phase1_intervention") != expected_intervention:
        raise ValueError("patch manifest Phase-1 intervention identity mismatch")
    if manifest.get("marker") != "step5_wedge_boundary_trace":
        raise ValueError("patch manifest boundary marker mismatch")
    before = manifest.get("before")
    after = manifest.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("patch manifest lacks before/after target mappings")
    index = json.loads(
        (REPO_ROOT / "tools/provenance/qualified-deltas/index.json").read_text(encoding="utf-8")
    )
    expected_runner = next(
        entry["sha256"] for entry in index["files"] if entry["path"] == PATCH_TARGET
    )
    if before.get(PATCH_TARGET) != expected_runner:
        raise ValueError("patch manifest target source hash mismatch")
    if after.get(PATCH_TARGET) != PHASE1_TARGET_SHA256:
        raise ValueError("patch manifest target output hash mismatch")
    if manifest.get("patcher_sha256") != sha256_file(PATCHER_PATH):
        raise ValueError("patch manifest patcher hash mismatch")
    if manifest.get("trace_source_sha256") != sha256_file(TRACE_SOURCE_PATH):
        raise ValueError("patch manifest trace source hash mismatch")
    return manifest


def _seal_tensor(tensor: torch.Tensor, label: str, *, fused_row: bool = False) -> TensorSeal:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
        raise ValueError(f"{label} is not a nonempty tensor")
    shape = tuple(tensor.shape)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{label} has invalid shape")
    if fused_row and shape[-1] != 4480:
        raise ValueError(f"{label} is not 4480-wide")
    contents = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    sha256 = hashlib.sha256(contents).hexdigest()
    _require_sha256(sha256, f"{label} contents")
    return TensorSeal(str(tensor.dtype), shape, len(contents), sha256)


def validate_materialized_fixture(
    materialized: MaterializedFixture,
    fixture: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate and canonically seal materialized rows and completed frame 22."""
    if materialized.schema_version != 1 or materialized.fixture_id != fixture["fixture_id"]:
        raise ValueError("materialized fixture schema or fixture identity mismatch")
    if len(materialized.fused_rows) != 2:
        raise ValueError("materialized fixture must contain exactly two fused rows")
    fused_rows = tuple(
        _seal_tensor(row, f"fused row {index}", fused_row=True)
        for index, row in enumerate(materialized.fused_rows)
    )
    state = materialized.frame_22_state
    if (
        state.frame != 22
        or state.nano_generated_tokens != 23
        or state.nano_session_positions != 22
        or state.pending_row_count != 1
        or state.pending_text_token != fixture["pad_token_id"]
        or state.pending_function_token != fixture["pad_token_id"]
        or state.function_state != "idle"
        or state.control_barrier_pending
    ):
        raise ValueError("materialized frame-22 logical state differs from the frozen prefix")
    if state.shadow_index < 0 or not state.shadow_pointer or not state.virtual_engine_identity:
        raise ValueError("materialized frame-22 shadow/virtual-engine identity is incomplete")
    if not isinstance(state.construction_metadata, dict) or not state.construction_metadata:
        raise ValueError("materialized frame-22 construction metadata is absent")
    try:
        construction_bytes = json.dumps(
            state.construction_metadata,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("construction metadata is not canonical JSON") from error
    state_seal = Frame22StateSeal(
        frame=state.frame,
        nano_generated_tokens=state.nano_generated_tokens,
        nano_session_positions=state.nano_session_positions,
        pending_row_count=state.pending_row_count,
        pending_text_token=state.pending_text_token,
        pending_function_token=state.pending_function_token,
        function_state=state.function_state,
        control_barrier_pending=state.control_barrier_pending,
        committed_kv_state=_seal_tensor(state.committed_kv_state, "frame-22 committed KV"),
        committed_mamba_conv_state=_seal_tensor(
            state.committed_mamba_conv_state, "frame-22 Mamba conv"
        ),
        committed_mamba_ssm_state=_seal_tensor(
            state.committed_mamba_ssm_state, "frame-22 Mamba SSM"
        ),
        rollback_shadow=_seal_tensor(state.rollback_shadow, "frame-22 rollback shadow"),
        shadow_index=state.shadow_index,
        shadow_pointer=state.shadow_pointer,
        virtual_engine_identity=state.virtual_engine_identity,
        construction_metadata_sha256=hashlib.sha256(construction_bytes).hexdigest(),
    )
    payload: dict[str, JsonValue] = {
        "construction_metadata": state.construction_metadata,
        "frame_22_state": asdict(state_seal),
        "fixture_id": materialized.fixture_id,
        "fused_rows": [asdict(row) for row in fused_rows],
        "schema_version": materialized.schema_version,
    }
    payload["fixture_materialization_sha256"] = stable_digest(payload)
    return payload


def application_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_frozen_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "fixture_id",
        "pad_token_id",
        "causal_prefix",
        "source_artifacts",
        "reproduction_recipe",
    }
    if set(fixture) != required:
        raise ValueError(f"frozen fixture keys differ: {sorted(set(fixture) ^ required)}")
    if fixture["schema_version"] != 1 or not fixture["causal_prefix"]:
        raise ValueError("invalid frozen fixture schema or empty causal prefix")
    for artifact in fixture["source_artifacts"]:
        source = Path(artifact["path"])
        if not source.is_file() or sha256_file(source) != artifact["sha256"]:
            raise ValueError(f"frozen source artifact changed or absent: {source}")
    fixture["fixture_sha256"] = sha256_file(path)
    return fixture


FROZEN_HARNESS_FILES = (
    "docs/reviews/step5-wedge-phase1-design-review-r1.md",
    "docs/reviews/step5-wedge-phase1-design-review-r2.md",
    "docs/step5-pairing-wedge-closure-design.md",
    "reports/step5-wedge/phase1-frozen-fixture.json",
    "reports/step5-wedge/phase1-preregistration.md",
    "reports/step5-wedge/phase1-preregistration-v3.md",
    "reports/step5-wedge/phase1-preregistration-v4.md",
    "reports/step5-wedge/phase1-preregistration-v5.md",
    "reports/step5-wedge/phase1-preregistration-v6.md",
    "docs/reviews/step5-wedge-phase1-executor-review-r1.md",
    "docs/reviews/step5-wedge-phase1-executor-review-r2.md",
    "docs/reviews/step5-wedge-phase1-executor-review-r3.md",
    "reports/step5-wedge/phase1-real-executor-config.json",
    "src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py",
    "src/nemotron_voicechat_runtime/runtime_optimizations.py",
    "src/nemotron_voicechat_runtime/pocket_worker.py",
    "src/nemotron_voicechat_runtime/server.py",
    "src/nemotron_voicechat_runtime/wedge_boundary_trace.py",
    "tools/provenance/qualified-deltas/index.json",
    "tools/qualification/step5_phase1_campaign.py",
    "tools/qualification/step5_phase1_oracle.py",
    "tools/qualification/step5_phase1_real_executor.py",
    "tools/qualification/step5_phase1_gpu_worker.py",
    "tools/qualification/step5_phase1_typed_carrier.py",
    "tests/runtime/test_step5_phase1_campaign.py",
    "tests/runtime/test_step5_phase1_oracle.py",
)


def compatibility_identity(
    *,
    patch_manifest_path: Path,
    patch_manifest_sha256: str,
    executor_identity: Any,
) -> dict[str, Any]:
    verify_patch_manifest(patch_manifest_path, patch_manifest_sha256)
    revision = application_revision()
    if revision != APPLICATION_REVISION:
        raise ValueError(
            f"application revision mismatch: expected {APPLICATION_REVISION}, got {revision}"
        )
    fixture = load_frozen_fixture()
    if not isinstance(executor_identity, dict):
        raise ValueError("executor identity must be a hash-bearing mapping")
    required_executor_keys = {
        "configuration_sha256",
        "fixture_materialization_sha256",
        "image_sha256",
        "kind",
        "model_sha256",
        "schema",
    }
    if not required_executor_keys <= set(executor_identity):
        raise ValueError("executor identity lacks required frozen hashes")
    for key in required_executor_keys - {"kind", "schema"}:
        _require_sha256(executor_identity[key], f"executor identity {key}")
    if not isinstance(executor_identity["kind"], str) or not executor_identity["kind"]:
        raise ValueError("executor identity kind is absent")
    if executor_identity["kind"] == "real-gpu-process-isolated":
        real_hashes = {
            "executor_source_sha256",
            "model_artifact_identity_sha256",
            "patch_manifest_sha256",
            "patched_runner_sha256",
            "patched_vllm_tree_sha256",
            "runtime_optimizations_source_sha256",
            "server_source_sha256",
            "worker_environment_sha256",
            "worker_source_sha256",
            "pocket_worker_source_sha256",
            "typed_carrier_source_sha256",
        }
        if not real_hashes <= set(executor_identity):
            raise ValueError("real executor identity lacks implementation hashes")
        for key in real_hashes:
            _require_sha256(executor_identity[key], f"executor identity {key}")
        if executor_identity["schema"] != 4:
            raise ValueError("real executor identity schema must be 4")
        if not isinstance(executor_identity.get("patched_vllm_file_count"), int):
            raise ValueError("real executor identity lacks patched tree file count")
    elif executor_identity["schema"] != 2:
        raise ValueError("executor identity schema must be 2")
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in FROZEN_HARNESS_FILES}
    identity: dict[str, Any] = {
        "application_revision": revision,
        "case_matrix": [asdict(item) for item in case_matrix(fixture["pad_token_id"])],
        "executor_identity": executor_identity,
        "frozen_files": files,
        "lane_order": list(LANE_ORDER),
        "optional_c": "omitted-global-eager-requires-construction-change",
        "patch_manifest_path": str(patch_manifest_path.resolve()),
        "patch_manifest_sha256": patch_manifest_sha256,
        "preregistered_a_first_missing": PREREGISTERED_A_FIRST_MISSING,
        "replicate_shape": {
            "max_logical_positions": 2,
            "replicate_batch_size": 1,
            "requests_per_lane": 1,
        },
        "replicates": REPLICATES,
        "request_prefix_b": PHASE1_B_REQUEST_PREFIX,
        "reseal": {
            "previous_application_revision": V4_APPLICATION_REVISION,
            "previous_v5_preregistration_sha256": V5_PREREGISTRATION_SHA256,
            "reason": V6_RESEAL_REASON,
        },
        "schema_version": 6,
    }
    identity["identity_sha256"] = stable_digest(identity)
    return identity


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def seal_identity(output_dir: Path, identity: dict[str, Any]) -> None:
    path = output_dir / IDENTITY_NAME
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("refusing to replace a different sealed compatibility identity")
        return
    atomic_json(path, identity)


def _validate_resume(path: Path, identity_sha256: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if record["replicate"] != len(records):
                raise ValueError(f"resume replicate gap/overlap at line {line_number}")
            if record["compatibility_identity_sha256"] != identity_sha256:
                raise ValueError(f"resume compatibility mismatch at line {line_number}")
            claimed = record.pop("record_sha256")
            if claimed != stable_digest(record):
                raise ValueError(f"resume record hash mismatch at line {line_number}")
            record["record_sha256"] = claimed
            records.append(record)
    if len(records) > REPLICATES:
        raise ValueError("resume ledger exceeds preregistered replicate count")
    return records


def _result_summary(result: ExecutionResult) -> dict[str, Any]:
    summary = {
        "boundaries": list(result.boundaries),
        "completed": result.completed,
        "dispatch_evidence": result.dispatch_evidence,
        "dispatch_evidence_seal": seal_exact_value(result.dispatch_evidence),
        "dispatch_mode": result.dispatch_mode,
        "first_missing": result.first_missing,
        "last_boundary": result.last_boundary,
        "prefix_sha256": result.prefix_sha256,
        "terminal_unknown": result.terminal_unknown,
        "observed_plan": asdict(result.observed_plan),
    }
    summary["snapshot_seal"] = (
        None if result.snapshot is None else seal_exact_value(result.snapshot)
    )
    return summary


def _validate_dispatch_evidence(
    result: ExecutionResult, *, pair_required: bool, abort: bool
) -> list[str]:
    evidence = result.dispatch_evidence
    if abort:
        return (
            []
            if evidence.get("operation") == "abort_generation"
            else ["abort lacks observed abort_generation evidence"]
        )
    required = {"b2_descriptor_key", "b3_descriptor_key", "graph_mode", "selector_fired"}
    if not required <= set(evidence):
        return ["B2/B3 dispatch evidence is incomplete"]
    if not all(
        isinstance(evidence[key], str) and SHA256_RE.fullmatch(evidence[key])
        for key in ("b2_descriptor_key", "b3_descriptor_key")
    ):
        return ["B2/B3 descriptor key is not a content SHA-256"]
    graph_mode = str(evidence["graph_mode"])
    selector = evidence["selector_fired"]
    if pair_required:
        if selector is not True or graph_mode not in {"NONE", "CUDAGraphMode.NONE"}:
            return ["B selector or B3 graph mode was not observed pair-eager"]
    elif selector is not False:
        return ["non-B operation fired the Phase-1 pair selector"]
    return []


async def _run_lane(
    executor: Phase1Executor,
    epoch: Any,
    lane: str,
    fixture: dict[str, Any],
    replicate: int,
    oracle_results: dict[str, ExecutionResult] | None = None,
) -> tuple[dict[str, ExecutionResult], list[str]]:
    request_id = f"step5-phase1/{lane}/{replicate:02d}"
    restored = await executor.restore_lane(epoch, lane, fixture, request_id)
    if restored != fixture["fixture_sha256"]:
        raise ValueError(f"{lane} restored the wrong frozen prefix")
    results: dict[str, ExecutionResult] = {}
    errors: list[str] = []
    pending_rejection: ExecutionResult | None = None
    for case in case_matrix(fixture["pad_token_id"]):
        plan = plan_for_case(case, fixture["pad_token_id"])
        execution = "sequential-logical-position" if lane == "S" else "packed"
        result = await executor.execute_case(epoch, lane, case, plan, execution, request_id)
        results[case.name] = result
        if result.observed_plan != plan:
            errors.append(f"{lane}/{case.name}: logical-position oracle mismatch")
        if result.prefix_sha256 != fixture["fixture_sha256"]:
            errors.append(f"{lane}/{case.name}: prefix identity mismatch")
        expected_dispatch = "pair-eager" if requires_pair_eager(lane, case) else "normal"
        if result.dispatch_mode != expected_dispatch:
            errors.append(
                f"{lane}/{case.name}: dispatch {result.dispatch_mode!r} != {expected_dispatch!r}"
            )
        errors.extend(
            f"{lane}/{case.name}: {item}"
            for item in _validate_dispatch_evidence(
                result,
                pair_required=requires_pair_eager(lane, case),
                abort=plan.kind == TransactionKind.ABORT,
            )
        )
        if (
            result.completed
            and plan.kind != TransactionKind.ABORT
            and result.boundaries != tuple(BOUNDARIES)
        ):
            errors.append(f"{lane}/{case.name}: completed boundary sequence is not B0-B9")
        if plan.kind == TransactionKind.ABORT and result.boundaries:
            errors.append(f"{lane}/{case.name}: zero-position abort reported model boundaries")
        if result.terminal_unknown:
            missing = result.first_missing
            expected_prefix = (
                () if missing is None else tuple(BOUNDARIES[: BOUNDARIES.index(missing)])
            )
            if missing is None or result.boundaries != expected_prefix:
                errors.append(f"{lane}/{case.name}: terminal boundary prefix is not canonical")
        elif not result.completed:
            errors.append(f"{lane}/{case.name}: nonterminal result did not complete")
        if result.terminal_unknown:
            if result.snapshot is not None:
                errors.append(f"{lane}/{case.name}: terminal unknown returned a snapshot")
        elif result.snapshot is None:
            errors.append(f"{lane}/{case.name}: required state snapshot absent")
        else:
            presence = validate_snapshot_presence(result.snapshot, plan)
            errors.extend(
                f"{lane}/{case.name}: {item.path}: {item.reason}" for item in presence.differences
            )
            semantics = validate_snapshot_semantics(
                result.snapshot,
                plan,
                text_token=case.text_token,
                function_token=case.function_token,
                pad_token=fixture["pad_token_id"],
            )
            errors.extend(
                f"{lane}/{case.name}: {item.path}: {item.reason}" for item in semantics.differences
            )
            if plan.correct_and_retain_second_row:
                restore = validate_rejection_restore(result.snapshot)
                errors.extend(
                    f"{lane}/{case.name}: {item.path}: {item.reason}"
                    for item in restore.differences
                )
                if not plan.pending_drain_is_separate_transition:
                    errors.append(f"{lane}/{case.name}: rejection drain was not declared separate")
                pending_rejection = result
            elif case.action == "pending_drain":
                if pending_rejection is None or pending_rejection.snapshot is None:
                    errors.append(f"{lane}/{case.name}: no immediately preceding origin rejection")
                else:
                    drain = validate_pending_drain(pending_rejection.snapshot, result.snapshot)
                    errors.extend(
                        f"{lane}/{case.name}: {item.path}: {item.reason}"
                        for item in drain.differences
                    )
                pending_rejection = None
            elif pending_rejection is not None:
                errors.append(f"{lane}/{case.name}: corrected rejection row was not drained")
        if oracle_results is not None and result.snapshot is not None:
            oracle = oracle_results.get(case.name)
            if oracle is None or oracle.snapshot is None:
                errors.append(f"{lane}/{case.name}: missing S snapshot")
            else:
                comparison = compare_snapshots(oracle.snapshot, result.snapshot)
                errors.extend(
                    f"{lane}/{case.name}: {item.path}: {item.reason}"
                    for item in comparison.differences
                )
        if result.terminal_unknown:
            if lane != "A" or not case.causal_terminal:
                errors.append(f"{lane}/{case.name}: unknown completion outside terminal A")
            break
        if errors or not result.completed:
            break
    if pending_rejection is not None:
        errors.append(f"{lane}: lane ended with an undrained corrected rejection row")
    return results, errors


async def run_replicate(
    executor: Phase1Executor, replicate: int, fixture: dict[str, Any]
) -> dict[str, Any]:
    if not executor.supports_lane_local_dispatch:
        raise RuntimeError(
            "lane-local dispatch unavailable; use the separately preregistered "
            "construction-stratified runner after non-destructive preclassification"
        )
    epoch = await executor.create_epoch(replicate)
    exactness_errors: list[str] = []
    lane_results: dict[str, dict[str, ExecutionResult]] = {}
    destroy_reason = "completed"
    try:
        for lane in LANE_ORDER:
            results, errors = await _run_lane(
                executor,
                epoch,
                lane,
                fixture,
                replicate,
                lane_results.get("S") if lane != "S" else None,
            )
            lane_results[lane] = results
            exactness_errors.extend(errors)
            if errors:
                destroy_reason = "exactness-or-protocol-failure"
                break
            if any(result.terminal_unknown for result in results.values()):
                destroy_reason = "terminal-A-unknown-completion"
                break
        causal_case = "frozen-prefix-pad-accept"
        b = lane_results.get("B", {}).get(causal_case)
        a = lane_results.get("A", {}).get(causal_case)
        b_exact_complete = bool(
            b
            and b.completed
            and b.first_missing is None
            and b.dispatch_mode == "pair-eager"
            and not any(error.startswith("B/") for error in exactness_errors)
        )
        a_localized_loss = bool(
            a and a.terminal_unknown and a.first_missing == PREREGISTERED_A_FIRST_MISSING
        )
        qualifying = b_exact_complete and a_localized_loss and not exactness_errors
        return {
            "a_localized_loss": a_localized_loss,
            "b_exact_complete": b_exact_complete,
            "destroy_reason": destroy_reason,
            "exactness_errors": exactness_errors,
            "lanes": {
                lane: {name: _result_summary(result) for name, result in results.items()}
                for lane, results in lane_results.items()
            },
            "qualifying_discordance": qualifying,
            "replicate": replicate,
        }
    finally:
        await executor.destroy_epoch(epoch, destroy_reason)


def campaign_verdict(records: list[dict[str, Any]]) -> dict[str, Any]:
    exactness_failures = sum(bool(record["exactness_errors"]) for record in records)
    b_missing = sum(not record["b_exact_complete"] for record in records)
    discordant = sum(record["qualifying_discordance"] for record in records)
    post_forward_losses = 0
    for record in records:
        causal = record.get("lanes", {}).get("A", {}).get("frozen-prefix-pad-accept", {})
        missing = causal.get("first_missing")
        if missing in BOUNDARIES and BOUNDARIES.index(missing) > BOUNDARIES.index("B4"):
            post_forward_losses += 1
    complete = len(records) == REPLICATES
    if not complete:
        verdict = "INCOMPLETE"
    elif exactness_failures or b_missing:
        verdict = "INVALID"
    elif post_forward_losses:
        verdict = "B1_FORWARD_NONRETURN_REJECTED"
    elif discordant >= 2:
        verdict = "B1_SUPPORTED"
    else:
        verdict = "INCOMPLETE"
    return {
        "b_missing_or_inexact_replicates": b_missing,
        "complete": complete,
        "exactness_failure_replicates": exactness_failures,
        "qualifying_discordant_replicates": discordant,
        "post_forward_localized_losses": post_forward_losses,
        "replicates_completed": len(records),
        "verdict": verdict,
    }


def _write_manifest(output_dir: Path) -> bool:
    path = output_dir / MANIFEST_NAME
    files = {
        str(item.relative_to(output_dir)): {
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(output_dir.rglob("*"))
        if item.is_file() and item != path
    }
    manifest: dict[str, Any] = {"files": files, "schema_version": 1}
    manifest["self_canonical_sha256"] = stable_digest(manifest)
    atomic_json(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    claimed = loaded.pop("self_canonical_sha256")
    return claimed == stable_digest(loaded) and all(
        sha256_file(output_dir / relative) == item["sha256"]
        for relative, item in loaded["files"].items()
    )


async def run_campaign(
    executor: Phase1Executor,
    output_dir: Path,
    identity: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed = json.loads((output_dir / IDENTITY_NAME).read_text(encoding="utf-8"))
    if sealed != identity:
        raise ValueError("live compatibility identity differs from preregistration seal")
    ledger = output_dir / LEDGER_NAME
    if ledger.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {ledger}")
    records = _validate_resume(ledger, identity["identity_sha256"]) if resume else []
    fixture = load_frozen_fixture()
    try:
        materialized = await executor.materialize_fixture(fixture)
        materialization_seal = validate_materialized_fixture(materialized, fixture)
        expected_materialization = identity["executor_identity"]["fixture_materialization_sha256"]
        actual_materialization = materialization_seal["fixture_materialization_sha256"]
        if actual_materialization != expected_materialization:
            raise ValueError("materialized fixture differs from sealed executor identity")
        materialization_path = output_dir / MATERIALIZATION_NAME
        if materialization_path.exists():
            canonical_seal = json.loads(json.dumps(materialization_seal, sort_keys=True))
            if json.loads(materialization_path.read_text(encoding="utf-8")) != canonical_seal:
                raise ValueError("materialized fixture differs from retained preflight seal")
        else:
            atomic_json(materialization_path, materialization_seal)
    except Exception as error:
        if records:
            raise ValueError("preflight failed while resuming a non-empty ledger") from error
        summary = campaign_verdict([])
        summary.update(
            {
                "compatibility_identity_sha256": identity["identity_sha256"],
                "preflight_error": f"{type(error).__name__}: {error}",
                "preflight_status": "INCOMPLETE",
            }
        )
        atomic_json(output_dir / SUMMARY_NAME, summary)
        summary["artifact_manifest_verified"] = _write_manifest(output_dir)
        atomic_json(output_dir / SUMMARY_NAME, summary)
        summary["artifact_manifest_verified"] = _write_manifest(output_dir)
        return summary
    with ledger.open("a", encoding="utf-8") as stream:
        for replicate in range(len(records), REPLICATES):
            record = await run_replicate(executor, replicate, fixture)
            record["compatibility_identity_sha256"] = identity["identity_sha256"]
            record["record_sha256"] = stable_digest(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            records.append(record)
    summary = campaign_verdict(records)
    summary["compatibility_identity_sha256"] = identity["identity_sha256"]
    summary["fixture_materialization_sha256"] = materialization_seal[
        "fixture_materialization_sha256"
    ]
    summary["preflight_status"] = "SEALED"
    atomic_json(output_dir / SUMMARY_NAME, summary)
    summary["artifact_manifest_verified"] = _write_manifest(output_dir)
    atomic_json(output_dir / SUMMARY_NAME, summary)
    summary["artifact_manifest_verified"] = _write_manifest(output_dir)
    return summary


class FakeExecutor(Phase1Executor):
    """Deterministic CPU fake proving lane, terminal, and verdict control flow."""

    def __init__(self, *, discordant_replicates: tuple[int, ...] = (0, 1)) -> None:
        self.discordant_replicates = discordant_replicates
        self.operations: list[tuple[Any, ...]] = []
        self.destroyed: list[tuple[int, str]] = []

    def sealed_identity(self) -> dict[str, Any]:
        materialized = self._materialization(load_frozen_fixture()["fixture_id"])
        materialization_seal = validate_materialized_fixture(materialized, load_frozen_fixture())
        return {
            "configuration_sha256": stable_digest("cpu-fake-configuration-v2"),
            "fixture_materialization_sha256": materialization_seal[
                "fixture_materialization_sha256"
            ],
            "image_sha256": stable_digest("cpu-fake-image-v2"),
            "kind": "cpu-fake",
            "model_sha256": stable_digest("cpu-fake-model-v2"),
            "schema": 2,
        }

    @staticmethod
    def _tensor(label: str, shape: tuple[int, ...]) -> torch.Tensor:
        elements = 1
        for dimension in shape:
            elements *= dimension
        seed = int(stable_digest(label)[:8], 16) % 256
        values = (torch.arange(elements, dtype=torch.int32) + seed) % 256
        return values.to(torch.float16).reshape(shape)

    @classmethod
    def _materialization(cls, fixture_id: str) -> MaterializedFixture:
        state = MaterializedFrame22State(
            frame=22,
            nano_generated_tokens=23,
            nano_session_positions=22,
            pending_row_count=1,
            pending_text_token=PHASE1_PAD_TOKEN_ID,
            pending_function_token=PHASE1_PAD_TOKEN_ID,
            function_state="idle",
            control_barrier_pending=False,
            committed_kv_state=cls._tensor("fake-frame22-kv", (1, 2)),
            committed_mamba_conv_state=cls._tensor("fake-frame22-conv", (1, 2)),
            committed_mamba_ssm_state=cls._tensor("fake-frame22-ssm", (1, 2)),
            rollback_shadow=cls._tensor("fake-frame22-shadow", (1, 2)),
            shadow_index=22,
            shadow_pointer="virtual-engine-0/shadow-22",
            virtual_engine_identity="virtual-engine-0",
            construction_metadata={"dispatcher": "cpu-fake", "schema": 2},
        )
        materialized = MaterializedFixture(
            schema_version=1,
            fixture_id=fixture_id,
            fused_rows=(
                cls._tensor("fake-fused-frame22", (1, 4480)),
                cls._tensor("fake-fused-frame23", (1, 4480)),
            ),
            frame_22_state=state,
        )
        return materialized

    async def materialize_fixture(self, fixture: dict[str, JsonValue]) -> MaterializedFixture:
        self.operations.append(("materialize", fixture["fixture_id"]))
        return self._materialization(str(fixture["fixture_id"]))

    async def create_epoch(self, replicate: int) -> dict[str, Any]:
        self.operations.append(("create", replicate))
        return {"replicate": replicate}

    async def restore_lane(
        self, epoch: Any, lane: str, fixture: dict[str, Any], request_id: str
    ) -> str:
        self.operations.append(("restore", epoch["replicate"], lane, request_id))
        return fixture["fixture_sha256"]

    @staticmethod
    def _snapshot(case: CaseSpec, plan: OraclePlan) -> StateSnapshot:
        state = torch.tensor(
            [plan.selected_packed_row, plan.accepted_token_count], dtype=torch.int64
        )
        origin = case.originating_rejection or case.name
        rejected_boundary = stable_digest(f"{origin}/committed-boundary")
        is_drain = plan.kind == TransactionKind.PENDING_DRAIN
        pending = (
            [{"corrected_from": [12, 12], "case": case.name}]
            if plan.correct_and_retain_second_row
            else []
        )
        return StateSnapshot(
            emitted_text_token=(None if plan.kind == TransactionKind.ABORT else case.text_token),
            emitted_function_token=(
                None if plan.kind == TransactionKind.ABORT else case.function_token
            ),
            emitted_conjunction=(
                None
                if plan.kind == TransactionKind.ABORT
                else case.text_token == 12 and case.function_token == 12
            ),
            accepted_token_count=plan.accepted_token_count,
            selected_packed_row=plan.selected_packed_row,
            selected_source_row_identity=(
                stable_digest(f"{origin}/corrected-row")
                if is_drain
                else stable_digest(f"fake-source-row/{plan.selected_packed_row}")
            ),
            selected_custom_outputs=(
                {}
                if plan.kind == TransactionKind.ABORT
                else {"function_tokens": torch.tensor([case.function_token])}
            ),
            committed_kv_state={"kv": state.clone()},
            committed_mamba_conv_state={"conv": state.clone()},
            committed_mamba_ssm_state={"ssm": state.clone()},
            rollback_shadow={"shadow": state.clone()},
            shadow_index=len(plan.positions),
            shadow_pointer="virtual-engine-0/shadow-0",
            virtual_engine_identity="virtual-engine-0",
            host_pending_rows=pending,
            correction_flags={"corrected": plan.correct_and_retain_second_row},
            correction_assumptions={"text": 12, "function": 12},
            position_one_boundary_state={"case": case.name, "position": 1},
            uncommitted_second_position_state=(
                {"case": case.name, "position": 1} if plan.correct_and_retain_second_row else None
            ),
            operation_evidence={
                "accepted": plan.accepted_token_count,
                "operation": case.control_boundary or case.action,
            },
            transition_id=case.name,
            originating_rejection_id=(
                origin if plan.correct_and_retain_second_row or is_drain else None
            ),
            pending_row_sha256_before=(
                stable_digest(f"{origin}/corrected-row") if is_drain else None
            ),
            pending_row_sha256_after=(
                stable_digest(f"{origin}/corrected-row")
                if plan.correct_and_retain_second_row
                else None
            ),
            committed_state_sha256_before=(
                rejected_boundary if is_drain else stable_digest(f"{case.name}/before")
            ),
            committed_state_sha256_after=(
                stable_digest(f"{origin}/drained") if is_drain else rejected_boundary
            ),
        )

    async def execute_case(
        self,
        epoch: Any,
        lane: str,
        case: CaseSpec,
        plan: OraclePlan,
        execution: str,
        request_id: str,
    ) -> ExecutionResult:
        self.operations.append(("execute", epoch["replicate"], lane, case.name, execution))
        terminal = (
            lane == "A"
            and case.causal_terminal
            and epoch["replicate"] in self.discordant_replicates
        )
        boundaries = (
            ()
            if plan.kind == TransactionKind.ABORT
            else tuple(BOUNDARIES[:4])
            if terminal
            else tuple(BOUNDARIES)
        )
        dispatch = "pair-eager" if requires_pair_eager(lane, case) else "normal"
        descriptor = stable_digest(f"fake-descriptor/{epoch['replicate']}/{lane}/{case.name}")
        dispatch_evidence = (
            {"operation": "abort_generation", "request_absent_after": True}
            if plan.kind == TransactionKind.ABORT
            else {
                "b2_descriptor_key": descriptor,
                "b3_descriptor_key": descriptor,
                "graph_mode": (
                    "CUDAGraphMode.NONE" if dispatch == "pair-eager" else "CUDAGraphMode.PIECEWISE"
                ),
                "selector_fired": dispatch == "pair-eager",
            }
        )
        return ExecutionResult(
            None if terminal else self._snapshot(case, plan),
            boundaries,
            not terminal,
            terminal,
            dispatch,
            load_frozen_fixture()["fixture_sha256"],
            plan,
            dispatch_evidence,
        )

    async def destroy_epoch(self, epoch: Any, reason: str) -> None:
        self.operations.append(("destroy", epoch["replicate"], reason))
        self.destroyed.append((epoch["replicate"], reason))


def _load_executor(specification: str) -> Phase1Executor:
    module_name, separator, factory_name = specification.partition(":")
    if not separator:
        raise ValueError("executor must be MODULE:FACTORY")
    executor = getattr(importlib.import_module(module_name), factory_name)()
    if not isinstance(executor, Phase1Executor):
        raise TypeError("executor factory did not return Phase1Executor")
    return executor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest-sha256", required=True)
    parser.add_argument("--executor", help="GPU adapter as MODULE:FACTORY")
    parser.add_argument("--fake", action="store_true", help="CPU-only control-flow dry run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    if args.fake == bool(args.executor):
        parser.error("select exactly one of --fake or --executor")
    executor = FakeExecutor() if args.fake else _load_executor(args.executor)
    identity = compatibility_identity(
        patch_manifest_path=args.patch_manifest,
        patch_manifest_sha256=args.patch_manifest_sha256,
        executor_identity=executor.sealed_identity(),
    )
    seal_identity(args.output_dir, identity)
    if args.seal_only:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    print(
        json.dumps(
            asyncio.run(run_campaign(executor, args.output_dir, identity, resume=args.resume)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
