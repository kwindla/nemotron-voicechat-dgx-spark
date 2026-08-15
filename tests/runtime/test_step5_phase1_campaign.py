from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemotron_voicechat_runtime import patch_step5_wedge_boundary_trace as patcher

TOOLS_PATH = Path(__file__).parents[2] / "tools/qualification"
sys.path.insert(0, str(TOOLS_PATH))
SPEC = importlib.util.spec_from_file_location(
    "step5_phase1_campaign", TOOLS_PATH / "step5_phase1_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)

REAL_SPEC = importlib.util.spec_from_file_location(
    "step5_phase1_real_executor", TOOLS_PATH / "step5_phase1_real_executor.py"
)
assert REAL_SPEC is not None and REAL_SPEC.loader is not None
real_executor = importlib.util.module_from_spec(REAL_SPEC)
sys.modules[REAL_SPEC.name] = real_executor
REAL_SPEC.loader.exec_module(real_executor)
import step5_phase1_gpu_worker as real_worker

CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "step5_phase1_calibration", TOOLS_PATH / "step5_phase1_calibration.py"
)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
calibration = importlib.util.module_from_spec(CALIBRATION_SPEC)
sys.modules[CALIBRATION_SPEC.name] = calibration
CALIBRATION_SPEC.loader.exec_module(calibration)


@pytest.fixture(autouse=True)
def _sealed_application_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign, "application_revision", lambda: campaign.APPLICATION_REVISION)


def _patch_manifest(tmp_path: Path) -> tuple[Path, str]:
    index = json.loads(patcher.INDEX_PATH.read_text(encoding="utf-8"))
    runner_sha = next(
        entry["sha256"] for entry in index["files"] if entry["path"] == campaign.PATCH_TARGET
    )
    payload = {
        "after": {campaign.PATCH_TARGET: campaign.PHASE1_TARGET_SHA256},
        "atomic_source_file_count": patcher.QUALIFIED_FILE_COUNT,
        "baseline_tree_sha256": patcher.QUALIFIED_TREE_SHA256,
        "before": {campaign.PATCH_TARGET: runner_sha},
        "marker": patcher.MARKER,
        "grafted_binary_data_files": {},
        "patcher_sha256": campaign.sha256_file(Path(patcher.__file__)),
        "phase1_intervention": {
            "kind": "pair-only-graph-disabled-eager-dispatch",
            "marker": patcher.PHASE1_MARKER,
            "pad_token_id": patcher.PHASE1_PAD_TOKEN_ID,
            "request_prefix": patcher.PHASE1_B_REQUEST_PREFIX,
            "target": campaign.PATCH_TARGET,
        },
        "trace_source_sha256": campaign.sha256_file(patcher.TRACE_SOURCE),
        "verified_source_file_count": patcher.QUALIFIED_FILE_COUNT,
    }
    path = tmp_path / "patch-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path, campaign.sha256_file(path)


def _identity(tmp_path: Path, executor_identity: object | None = None) -> dict:
    manifest, manifest_sha = _patch_manifest(tmp_path)
    return campaign.compatibility_identity(
        patch_manifest_path=manifest,
        patch_manifest_sha256=manifest_sha,
        executor_identity=executor_identity or campaign.FakeExecutor().sealed_identity(),
    )


def test_frozen_fixture_discloses_missing_historical_tensors_and_verifies_sources() -> None:
    fixture = campaign.load_frozen_fixture()
    assert not fixture["reproduction_recipe"]["exact_tensor_replay"]
    assert "does not contain" in fixture["reproduction_recipe"]["inability_to_replay_disclosure"]
    assert fixture["causal_prefix"][-1]["observed_completion"] is False


def _calibration_statistic() -> dict[str, object]:
    return {
        "different_byte_count": 0,
        "different_element_count": 0,
        "dtype": "torch.float32",
        "element_count": 2,
        "first_different_byte_offset": None,
        "first_different_candidate_value": None,
        "first_different_element_offset": None,
        "first_different_oracle_value": None,
        "l1_norm": 0.0,
        "l2_norm": 0.0,
        "linf_norm": 0.0,
        "negative_count": 0,
        "orientation": "candidate_packed_minus_sequential_s",
        "positive_count": 0,
        "shape": [2],
        "signed_delta_lower": 0.0,
        "signed_delta_upper": 0.0,
        "zero_count": 2,
    }


def test_calibration_derives_lane_case_boundary_envelopes_and_conv0_zero() -> None:
    population = [
        {
            "boundary": (
                calibration.ABORT_BOUNDARY
                if case.name == "abort"
                else calibration.COMPLETED_BOUNDARY
            ),
            "case": case.name,
        }
        for case in campaign.case_matrix(12)
        if case.name != calibration.TERMINAL_CAUSAL_CASE
    ]
    path = "snapshot.committed_mamba_conv_state['model.layers.0.mixer[0][0]']"
    constructions = []
    for index in range(calibration.CALIBRATION_CONSTRUCTIONS):
        comparisons = {
            item["case"]: {
                "boundary": item["boundary"],
                "exact_metadata": True,
                "statistics": {path: _calibration_statistic()},
            }
            for item in population
        }
        constructions.append({"comparisons": comparisons, "construction_index": index})
    receipt = {
        "accepted": True,
        "case_boundary_population": population,
        "constructions": constructions,
    }
    envelope = calibration.derive_envelopes(
        receipt,
        contract_sha256="a" * 64,
        receipt_sha256="b" * 64,
    )
    assert set(envelope["envelopes"]) == {"A", "B"}
    rule = envelope["envelopes"]["A"]["text-only-rejection"]["B9"][path]
    assert rule["calibration_observations"] == 5
    assert rule["zero_required"]
    assert not envelope["terminal_causal_case"]["envelope_authorized"]
    assert not envelope["phase1_recalibration_authorized"]


def test_calibration_fails_closed_on_missing_construction_or_statistic() -> None:
    with pytest.raises(ValueError, match="incomplete calibration"):
        calibration.derive_envelopes(
            {"accepted": True, "constructions": []},
            contract_sha256="a" * 64,
            receipt_sha256="b" * 64,
        )
    statistic = _calibration_statistic()
    statistic.pop("l2_norm")
    with pytest.raises(ValueError, match="missing statistic"):
        calibration._envelope_rule(
            [statistic] * calibration.CALIBRATION_CONSTRUCTIONS,
            "leaf",
        )


def test_boundary_localization_uses_increment1_b0_b9_order() -> None:
    fixture = campaign.load_frozen_fixture()
    plan = campaign.logical_position_plan(12, 12, 12)
    result = campaign.ExecutionResult(
        snapshot=None,
        boundaries=tuple(campaign.BOUNDARIES[:4]),
        completed=False,
        terminal_unknown=True,
        dispatch_mode="normal",
        prefix_sha256=fixture["fixture_sha256"],
        observed_plan=plan,
    )
    assert result.last_boundary == "B3"
    assert result.first_missing == "B4"


def test_control_abort_and_pending_drain_are_separate_single_actions() -> None:
    cases = {item.name: item for item in campaign.case_matrix(12)}
    control = campaign.plan_for_case(cases["agent-bos"], 12)
    abort = campaign.plan_for_case(cases["abort"], 12)
    drain_case = cases["text-only-rejection-pending-drain"]
    drain = campaign.plan_for_case(drain_case, 12)
    assert control.positions == (0,) and control.accepted_token_count == 1
    assert abort.positions == () and abort.accepted_token_count == 0
    assert drain.positions == (0,) and drain.pending_drain_is_separate_transition
    assert drain_case.originating_rejection == "text-only-rejection"


def test_fake_executor_full_dry_run_terminal_order_verdict_and_resume(tmp_path: Path) -> None:
    executor = campaign.FakeExecutor()
    identity = _identity(tmp_path)
    campaign.seal_identity(tmp_path, identity)
    summary = asyncio.run(campaign.run_campaign(executor, tmp_path, identity, resume=False))
    assert summary["verdict"] == "B1_SUPPORTED"
    assert summary["qualifying_discordant_replicates"] == 2
    assert summary["replicates_completed"] == 12
    assert summary["artifact_manifest_verified"]
    for replicate in range(campaign.REPLICATES):
        restores = [
            operation[2]
            for operation in executor.operations
            if operation[:2] == ("restore", replicate)
        ]
        assert restores == list(campaign.LANE_ORDER)
    assert executor.destroyed[:2] == [
        (0, "terminal-A-unknown-completion"),
        (1, "terminal-A-unknown-completion"),
    ]
    before = list(executor.operations)
    resumed = asyncio.run(campaign.run_campaign(executor, tmp_path, identity, resume=True))
    assert resumed["verdict"] == "B1_SUPPORTED"
    assert executor.operations == before + [
        ("materialize", campaign.load_frozen_fixture()["fixture_id"])
    ]


def test_two_discordant_rule_and_incomplete_disposition(tmp_path: Path) -> None:
    executor = campaign.FakeExecutor(discordant_replicates=(0,))
    identity = _identity(tmp_path)
    campaign.seal_identity(tmp_path, identity)
    summary = asyncio.run(campaign.run_campaign(executor, tmp_path, identity, resume=False))
    assert summary["qualifying_discordant_replicates"] == 1
    assert summary["verdict"] == "INCOMPLETE"


def test_resume_and_seal_reject_identity_shape_or_record_drift(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    campaign.seal_identity(tmp_path, identity)
    with pytest.raises(ValueError, match="different sealed"):
        changed = campaign.FakeExecutor().sealed_identity() | {"configuration_sha256": "c" * 64}
        campaign.seal_identity(tmp_path, _identity(tmp_path, changed))
    ledger = tmp_path / campaign.LEDGER_NAME
    record = {
        "compatibility_identity_sha256": identity["identity_sha256"],
        "replicate": 0,
    }
    record["record_sha256"] = campaign.stable_digest(record)
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    record["replicate"] = 1
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gap/overlap"):
        campaign._validate_resume(ledger, identity["identity_sha256"])


def test_real_executor_identity_requires_image_model_fixture_and_config_hashes(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha = _patch_manifest(tmp_path)
    with pytest.raises(ValueError, match="required frozen hashes"):
        campaign.compatibility_identity(
            patch_manifest_path=manifest,
            patch_manifest_sha256=manifest_sha,
            executor_identity={"kind": "real", "schema": 1},
        )


def test_phase1_patch_is_hash_qualified_atomic_and_pair_only(tmp_path: Path) -> None:
    source = Path("/tmp/voicechat-phase1-diff.VbhhUc/qualified-vllm")
    if not source.is_dir():
        pytest.skip("local qualified escrow reconstruction unavailable")
    source_runner_sha = patcher.sha256_file(source / "v1/worker/gpu_model_runner.py")
    output = tmp_path / "patched"
    manifest = patcher.apply_patch(source, output, pair_eager_dispatch=True)
    patched = (output / "v1/worker/gpu_model_runner.py").read_text(encoding="utf-8")
    assert manifest["verified_source_file_count"] == 966
    assert manifest["grafted_binary_data_files"]
    assert manifest["atomic_source_file_count"] == 966 + len(manifest["grafted_binary_data_files"])
    assert not any("__pycache__" in path.parts for path in output.rglob("*"))
    assert manifest["phase1_intervention"]["request_prefix"] == "step5-phase1/B/"
    assert manifest["phase1_intervention"]["pad_token_id"] == 12
    assert "step5_phase1_pair_eager_dispatch" in patched
    assert "int(num_scheduled_tokens) == 2" in patched
    assert "scheduled_spec_decode_tokens[" in patched
    assert "][0]" in patched
    assert "== 12" in patched
    assert "CUDAGraphMode.NONE" in patched
    assert patcher.sha256_file(source / "v1/worker/gpu_model_runner.py") == source_runner_sha


def test_runner_refuses_unstratified_epoch_fallback() -> None:
    executor = campaign.FakeExecutor()
    executor.supports_lane_local_dispatch = False
    fixture = campaign.load_frozen_fixture()
    with pytest.raises(RuntimeError, match="construction-stratified"):
        asyncio.run(campaign.run_replicate(executor, 0, fixture))


def test_exactness_failure_destroys_epoch_before_terminal_a() -> None:
    class DivergentB(campaign.FakeExecutor):
        async def execute_case(self, epoch, lane, case, plan, execution, request_id):
            result = await super().execute_case(epoch, lane, case, plan, execution, request_id)
            if lane == "B" and result.snapshot is not None:
                result = replace(
                    result,
                    snapshot=replace(result.snapshot, committed_kv_state={"diverged": True}),
                )
            return result

    executor = DivergentB()
    fixture = campaign.load_frozen_fixture()
    record = asyncio.run(campaign.run_replicate(executor, 0, fixture))
    assert record["destroy_reason"] == "exactness-or-protocol-failure"
    assert record["exactness_errors"]
    assert not any(operation[:3] == ("restore", 0, "A") for operation in executor.operations)


def test_missing_snapshot_fails_closed_before_terminal_a() -> None:
    class MissingSnapshot(campaign.FakeExecutor):
        async def execute_case(self, epoch, lane, case, plan, execution, request_id):
            result = await super().execute_case(epoch, lane, case, plan, execution, request_id)
            if lane == "S" and result.completed:
                return replace(result, snapshot=None)
            return result

    executor = MissingSnapshot()
    record = asyncio.run(campaign.run_replicate(executor, 0, campaign.load_frozen_fixture()))
    assert any("required state snapshot absent" in item for item in record["exactness_errors"])
    assert not record["qualifying_discordance"]


def test_every_b_packed_pair_case_requires_eager_dispatch() -> None:
    class NormalRejection(campaign.FakeExecutor):
        async def execute_case(self, epoch, lane, case, plan, execution, request_id):
            result = await super().execute_case(epoch, lane, case, plan, execution, request_id)
            if lane == "B" and case.name == "text-only-rejection":
                return replace(result, dispatch_mode="normal")
            return result

    executor = NormalRejection()
    record = asyncio.run(campaign.run_replicate(executor, 0, campaign.load_frozen_fixture()))
    assert any("B/text-only-rejection: dispatch" in item for item in record["exactness_errors"])


def test_wrong_b3_graph_selector_evidence_fails_closed() -> None:
    class WrongB3(campaign.FakeExecutor):
        async def execute_case(self, epoch, lane, case, plan, execution, request_id):
            result = await super().execute_case(epoch, lane, case, plan, execution, request_id)
            if lane == "B" and campaign.requires_pair_eager(lane, case):
                return replace(
                    result,
                    dispatch_evidence=result.dispatch_evidence
                    | {"graph_mode": "CUDAGraphMode.PIECEWISE", "selector_fired": False},
                )
            return result

    record = asyncio.run(campaign.run_replicate(WrongB3(), 0, campaign.load_frozen_fixture()))
    assert any("B selector or B3 graph mode" in item for item in record["exactness_errors"])


def test_wrong_observed_gpu_result_and_hardcoded_snapshot_field_fail_closed() -> None:
    class WrongObserved(campaign.FakeExecutor):
        async def execute_case(self, epoch, lane, case, plan, execution, request_id):
            result = await super().execute_case(epoch, lane, case, plan, execution, request_id)
            if lane == "S" and case.name == "text-only-rejection":
                return replace(
                    result,
                    observed_plan=replace(plan, accepted_token_count=2),
                    snapshot=replace(result.snapshot, emitted_text_token=999),
                )
            return result

    record = asyncio.run(campaign.run_replicate(WrongObserved(), 0, campaign.load_frozen_fixture()))
    assert any("logical-position oracle mismatch" in item for item in record["exactness_errors"])
    assert any("snapshot.emitted_text_token" in item for item in record["exactness_errors"])


def test_patch_manifest_hash_and_content_fail_closed(tmp_path: Path) -> None:
    path, digest = _patch_manifest(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.verify_patch_manifest(path, "a" * 64)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase1_intervention"]["pad_token_id"] = 13
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="intervention identity"):
        campaign.verify_patch_manifest(path, campaign.sha256_file(path))
    with pytest.raises(ValueError, match="64-hex"):
        campaign.verify_patch_manifest(path, "not-a-sha")


def test_preflight_failure_retains_zero_replicate_incomplete_summary(tmp_path: Path) -> None:
    class BrokenPreflight(campaign.FakeExecutor):
        async def materialize_fixture(self, fixture):
            materialized = await super().materialize_fixture(fixture)
            return replace(
                materialized,
                fused_rows=(materialized.fused_rows[0][:, :4479], materialized.fused_rows[1]),
            )

    executor = BrokenPreflight()
    identity = _identity(tmp_path, executor.sealed_identity())
    campaign.seal_identity(tmp_path, identity)
    summary = asyncio.run(campaign.run_campaign(executor, tmp_path, identity, resume=False))
    assert summary["verdict"] == "INCOMPLETE"
    assert summary["replicates_completed"] == 0
    assert summary["preflight_status"] == "INCOMPLETE"
    assert not any(operation[0] == "create" for operation in executor.operations)
    assert not (tmp_path / campaign.LEDGER_NAME).exists()


def test_materialization_seal_is_written_before_replicate_zero(tmp_path: Path) -> None:
    class ObservingExecutor(campaign.FakeExecutor):
        async def create_epoch(self, replicate):
            assert (tmp_path / campaign.MATERIALIZATION_NAME).is_file()
            return await super().create_epoch(replicate)

    executor = ObservingExecutor()
    identity = _identity(tmp_path, executor.sealed_identity())
    campaign.seal_identity(tmp_path, identity)
    summary = asyncio.run(campaign.run_campaign(executor, tmp_path, identity, resume=False))
    assert summary["preflight_status"] == "SEALED"
    assert (
        summary["fixture_materialization_sha256"]
        == executor.sealed_identity()["fixture_materialization_sha256"]
    )


def test_application_revision_seal_remains_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_sha = _patch_manifest(tmp_path)
    monkeypatch.setattr(campaign, "application_revision", lambda: "d" * 40)
    with pytest.raises(ValueError, match="application revision mismatch"):
        campaign.compatibility_identity(
            patch_manifest_path=manifest,
            patch_manifest_sha256=manifest_sha,
            executor_identity=campaign.FakeExecutor().sealed_identity(),
        )


def _real_config(tmp_path: Path, *, utilization: float = 0.42):
    patched = tmp_path / "patched-vllm"
    (patched / "v1/worker").mkdir(parents=True)
    (patched / "__init__.py").write_text("# qualified test tree\n", encoding="utf-8")
    (patched / "v1/worker/gpu_model_runner.py").write_text(
        "# qualified test runner\n", encoding="utf-8"
    )
    files = {}
    for name in (
        "patch-manifest.json",
        "source.pcm",
        "typed-carrier-witness.log",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    directories = {}
    for name in ("speech", "checkpoint", "skeleton", "nano", "eartts", "pocket-hub"):
        path = tmp_path / name
        path.mkdir()
        directories[name] = path
    (directories["checkpoint"] / "model.safetensors").write_bytes(b"parent")
    (directories["checkpoint"] / "config.json").write_text("{}\n", encoding="utf-8")
    (directories["checkpoint"] / "rnnt_tokenizer").mkdir()
    (directories["checkpoint"] / "rnnt_tokenizer" / "tokenizer.model").write_bytes(
        b"tokenizer"
    )
    (directories["skeleton"] / "config.json").write_bytes(b"skeleton")
    for relative in (
        "examples/speechlm2/function_calling/template.jinja",
        "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
    ):
        path = directories["speech"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    component_files = {}
    for name in ("nano", "eartts"):
        artifact = directories[name] / "model.bin"
        artifact.write_bytes(name.encode())
        component_files[name] = {
            "model.bin": {
                "bytes": artifact.stat().st_size,
                "sha256": campaign.sha256_file(artifact),
            }
        }
    manifest_path = tmp_path / "vllm-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "components": {
                    name: {
                        "files": component_files[name],
                        "model": {"sha256": campaign.stable_digest(name)},
                    }
                    for name in ("nano", "eartts")
                },
                "source": {
                    "model": {
                        "sha256": campaign.sha256_file(
                            directories["checkpoint"] / "model.safetensors"
                        )
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config = real_executor.RealExecutorConfig(
        patched_vllm_root=patched,
        patch_manifest=files["patch-manifest.json"],
        image_sha256="a" * 64,
        model_sha256="b" * 64,
        speech_root=directories["speech"],
        checkpoint_root=directories["checkpoint"],
        hf_skeleton=directories["skeleton"],
        vllm_manifest=manifest_path,
        nano_vllm_path=directories["nano"],
        eartts_vllm_path=directories["eartts"],
        source_pcm=files["source.pcm"],
        source_typed_carrier_witness=files["typed-carrier-witness.log"],
        pocket_python=Path(sys.executable),
        pocket_hf_hub=directories["pocket-hub"],
        gpu_memory_utilization=utilization,
    )
    model_identity = real_executor._verify_model_artifacts(config)
    return replace(config, model_sha256=model_identity["model_sha256"])


class _TransportEpoch:
    def __init__(self, fake: Any, replicate: int) -> None:
        self.fake = fake
        self.replicate = replicate
        self.closed: list[str] = []

    async def request(self, operation: str, payload: dict[str, Any], timeout: float) -> Any:
        del timeout
        if operation == "restore_lane":
            return payload["fixture_sha256"]
        if operation == "execute_case":
            case = payload["case"]
            plan = payload["plan"]
            lane = payload["lane"]
            return campaign.ExecutionResult(
                snapshot=self.fake._snapshot(case, plan),
                boundaries=(
                    ()
                    if plan.kind == campaign.TransactionKind.ABORT
                    else tuple(campaign.BOUNDARIES)
                ),
                completed=True,
                terminal_unknown=False,
                dispatch_mode=(
                    "pair-eager" if campaign.requires_pair_eager(lane, case) else "normal"
                ),
                prefix_sha256=campaign.load_frozen_fixture()["fixture_sha256"],
                observed_plan=plan,
                dispatch_evidence=(
                    {"operation": "abort_generation", "request_absent_after": True}
                    if plan.kind == campaign.TransactionKind.ABORT
                    else {
                        "b2_descriptor_key": campaign.stable_digest(
                            f"transport-b2/{self.replicate}/{lane}/{case.name}"
                        ),
                        "b3_descriptor_key": campaign.stable_digest(
                            f"transport-b3/{self.replicate}/{lane}/{case.name}"
                        ),
                        "graph_mode": (
                            "CUDAGraphMode.NONE"
                            if campaign.requires_pair_eager(lane, case)
                            else "CUDAGraphMode.PIECEWISE"
                        ),
                        "selector_fired": campaign.requires_pair_eager(lane, case),
                    }
                ),
            )
        raise AssertionError(operation)

    async def close(self, reason: str) -> None:
        self.closed.append(reason)


class _TransportBackend:
    def __init__(self, config: Any, runtime_identity: Any) -> None:
        del config, runtime_identity
        self.fake = campaign.FakeExecutor(discordant_replicates=())
        self.epochs: list[_TransportEpoch] = []

    async def materialize(self, fixture: dict[str, Any]):
        return self.fake._materialization(fixture["fixture_id"])

    async def create_epoch(self, replicate: int) -> _TransportEpoch:
        epoch = _TransportEpoch(self.fake, replicate)
        self.epochs.append(epoch)
        return epoch


def test_real_executor_is_concrete_hash_sealed_and_transport_parity(tmp_path: Path) -> None:
    executor = real_executor.RealPhase1Executor(
        _real_config(tmp_path), backend_factory=_TransportBackend
    )
    identity = executor.sealed_identity()
    assert identity["kind"] == "real-gpu-process-isolated"
    assert identity["schema"] == 4
    for key in (
        "configuration_sha256",
        "executor_source_sha256",
        "fixture_materialization_sha256",
        "image_sha256",
        "model_sha256",
        "patched_vllm_tree_sha256",
        "runtime_optimizations_source_sha256",
        "pocket_worker_source_sha256",
        "server_source_sha256",
        "worker_environment_sha256",
        "worker_source_sha256",
        "typed_carrier_source_sha256",
    ):
        assert campaign.SHA256_RE.fullmatch(identity[key])
    compatibility = _identity(tmp_path, identity)
    assert compatibility["schema_version"] == 6
    assert (
        compatibility["executor_identity"]["executor_source_sha256"]
        == identity["executor_source_sha256"]
    )
    assert "tools/qualification/step5_phase1_gpu_worker.py" in compatibility["frozen_files"]
    record = asyncio.run(campaign.run_replicate(executor, 0, campaign.load_frozen_fixture()))
    assert not record["exactness_errors"]
    assert record["b_exact_complete"]
    assert not record["qualifying_discordance"]
    assert executor.backend.epochs[0].closed == ["completed"]


def test_real_executor_refuses_nonsealed_gpu_memory_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="0.42"):
        _real_config(tmp_path, utilization=0.41).validated()


def test_real_executor_never_labels_a_nonterminal_timeout_terminal_unknown() -> None:
    class TimeoutClient:
        async def request(self, operation: str, payload: dict[str, Any], timeout: float) -> Any:
            del operation, payload, timeout
            raise TimeoutError

    executor = object.__new__(real_executor.RealPhase1Executor)
    executor.config = SimpleNamespace(rpc_timeout_seconds=1.0, terminal_timeout_seconds=1.0)

    async def observed_trace(epoch: Any, request_id: str):
        del epoch, request_id
        return tuple(campaign.BOUNDARIES[:4]), "normal", {
            "b2_descriptor_key": "a" * 64,
            "b3_descriptor_key": "b" * 64,
            "graph_mode": "CUDAGraphMode.NONE",
            "selector_fired": False,
        }

    executor._terminal_trace = observed_trace
    case = campaign.case_matrix(12)[0]
    with pytest.raises(TimeoutError, match="nonterminal Phase-1 timeout"):
        asyncio.run(
            executor.execute_case(
                {"client": TimeoutClient()},
                "S",
                case,
                campaign.plan_for_case(case, 12),
                "sequential-logical-position",
                "step5-phase1/S/00",
            )
        )


def test_compatibility_reseal_requires_real_implementation_hashes(tmp_path: Path) -> None:
    identity = campaign.FakeExecutor().sealed_identity() | {"kind": "real-gpu-process-isolated"}
    with pytest.raises(ValueError, match="implementation hashes"):
        _identity(tmp_path, identity)


def test_real_executor_boundary_reader_localizes_latest_prefix(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    ring = trace / "wedge-trace-test.ring"
    records = [
        {"boundary": boundary, "monotonic_ns": index, "request_id": "request"}
        for index, boundary in enumerate((*campaign.BOUNDARIES, *campaign.BOUNDARIES[:4]), start=1)
    ]
    payload = b"".join(
        json.dumps(record).encode().ljust(real_worker.RECORD_BYTES, b"\0") for record in records
    )
    ring.write_bytes(payload)
    assert real_worker.read_boundary_prefix(trace, "request") == tuple(campaign.BOUNDARIES[:4])


def test_real_executor_dispatch_reader_uses_recorded_b3_details(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    records = [
        {"boundary": "B0", "monotonic_ns": 1, "request_id": "request"},
        {
            "boundary": "B2",
            "details": {"descriptor_key": "b2-observed"},
            "monotonic_ns": 2,
            "request_id": "request",
        },
        {
            "boundary": "B3",
            "details": {
                "descriptor_key": "b3-observed",
                "graph_mode": "CUDAGraphMode.NONE",
                "phase1_pair_eager_selector": True,
            },
            "monotonic_ns": 3,
            "request_id": "request",
        },
    ]
    (trace / "wedge-trace-test.ring").write_bytes(
        b"".join(
            json.dumps(record).encode().ljust(real_worker.RECORD_BYTES, b"\0")
            for record in records
        )
    )
    mode, evidence = real_worker.observed_dispatch(trace, "request")
    assert mode == "pair-eager"
    assert evidence == {
        "b2_descriptor_key": "b2-observed",
        "b3_descriptor_key": "b3-observed",
        "graph_mode": "CUDAGraphMode.NONE",
        "selector_fired": True,
    }


def test_real_executor_normal_dispatch_allows_runtime_eager_shape(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    records = [
        {"boundary": "B0", "monotonic_ns": 1, "request_id": "request"},
        {
            "boundary": "B2",
            "details": {"descriptor_key": "a" * 64},
            "monotonic_ns": 2,
            "request_id": "request",
        },
        {
            "boundary": "B3",
            "details": {
                "descriptor_key": "b" * 64,
                "graph_mode": "CUDAGraphMode.NONE",
                "phase1_pair_eager_selector": False,
            },
            "monotonic_ns": 3,
            "request_id": "request",
        },
    ]
    (trace / "wedge-trace-test.ring").write_bytes(
        b"".join(
            json.dumps(record).encode().ljust(real_worker.RECORD_BYTES, b"\0")
            for record in records
        )
    )
    mode, evidence = real_worker.observed_dispatch(trace, "request")
    assert mode == "normal"
    assert evidence["selector_fired"] is False
    assert evidence["graph_mode"] == "CUDAGraphMode.NONE"


def test_real_executor_accepts_pinned_none_enum_rendering_for_selected_b(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    records = [
        {"boundary": "B0", "monotonic_ns": 1, "request_id": "request"},
        {
            "boundary": "B2",
            "details": {"descriptor_key": "a" * 64},
            "monotonic_ns": 2,
            "request_id": "request",
        },
        {
            "boundary": "B3",
            "details": {
                "descriptor_key": "b" * 64,
                "graph_mode": "NONE",
                "phase1_pair_eager_selector": True,
            },
            "monotonic_ns": 3,
            "request_id": "request",
        },
    ]
    (trace / "wedge-trace-test.ring").write_bytes(
        b"".join(
            json.dumps(record).encode().ljust(real_worker.RECORD_BYTES, b"\0")
            for record in records
        )
    )
    mode, evidence = real_worker.observed_dispatch(trace, "request")
    assert mode == "pair-eager"
    assert evidence["selector_fired"] is True
    assert evidence["graph_mode"] == "NONE"


def test_real_executor_dispatch_reader_rejects_truncated_b3(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    records = [
        {"boundary": "B0", "monotonic_ns": 1, "request_id": "request"},
        {"boundary": "B2", "details": {}, "monotonic_ns": 2, "request_id": "request"},
        {"boundary": "B3", "monotonic_ns": 3, "request_id": "request", "truncated": True},
    ]
    (trace / "wedge-trace-test.ring").write_bytes(
        b"".join(
            json.dumps(record).encode().ljust(real_worker.RECORD_BYTES, b"\0")
            for record in records
        )
    )
    with pytest.raises(RuntimeError, match="truncated"):
        real_worker.observed_dispatch(trace, "request")


def test_real_worker_retains_block_ownership_after_idle_batch_cleanup() -> None:
    table = SimpleNamespace(
        num_blocks_per_row=[2],
        get_numpy_array=lambda: np.asarray([[7, 9, 0]], dtype=np.int32),
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[table], req_id_to_index={"request": 0}),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
    )
    real_worker._remember_request_blocks(runner)
    runner.input_batch.req_id_to_index.clear()
    real_worker._remember_request_blocks(runner)
    assert runner._phase1_request_blocks == {"request": {0: [7, 9]}}


def test_child_overlay_reverification_rejects_tampered_atomic_tree(tmp_path: Path) -> None:
    config = _real_config(tmp_path)
    inventory = real_executor._tree_inventory(config.patched_vllm_root)
    worker_config = config.canonical() | {
        "patched_runner_sha256": campaign.sha256_file(
            config.patched_vllm_root / "v1/worker/gpu_model_runner.py"
        ),
        "patched_vllm_file_count": len(inventory),
        "patched_vllm_tree_sha256": real_executor._inventory_digest(inventory),
        "patch_manifest_sha256": campaign.sha256_file(config.patch_manifest),
    }
    (config.patched_vllm_root / "leaked-image-package.py").write_text(
        "# must not be accepted\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="differs from the parent seal"):
        real_worker._copy_qualified_overlay(worker_config)


@pytest.mark.parametrize(
    "relative",
    ("config.json", "model.safetensors", "rnnt_tokenizer/tokenizer.model"),
)
def test_child_model_reverification_rejects_tampered_checkpoint_artifact(
    tmp_path: Path, relative: str
) -> None:
    config = _real_config(tmp_path)
    identity = real_executor._verify_model_artifacts(config)
    worker_config = config.canonical() | {
        "model_artifact_identity": identity,
        "model_sha256": identity["model_sha256"],
    }
    (config.checkpoint_root / relative).write_bytes(b"substituted")
    with pytest.raises(RuntimeError, match="differs? from (?:the model manifest|the parent seal)"):
        real_worker._verify_model_artifact_identity(worker_config)


def test_model_hashes_release_page_cache_before_gpu_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"sealed model bytes")
    calls: list[tuple[int, int, int]] = []

    def advise(_fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((offset, length, advice))

    monkeypatch.setattr(real_executor.os, "posix_fadvise", advise)
    real_executor._sha256_model_file(artifact)
    monkeypatch.setattr(real_worker.os, "posix_fadvise", advise)
    real_worker._file_sha256(artifact)
    assert calls == [
        (0, 0, real_executor.os.POSIX_FADV_DONTNEED),
        (0, 0, real_worker.os.POSIX_FADV_DONTNEED),
    ]


def test_runtime_correction_assumptions_reject_substituted_state() -> None:
    prefix = {"assumptions": {23: (12, 12)}}
    assert real_worker._runtime_recorded_assumptions(prefix, 23) == (12, 12)
    with pytest.raises(RuntimeError, match="differ from materialization"):
        real_worker._runtime_recorded_assumptions(prefix, 23, {"assumed": (7, 12)})


def test_named_controls_use_live_wrapper_context_and_streaming_state() -> None:
    class StreamingState:
        agent_idle = True

        def is_agent_idle(self) -> bool:
            return self.agent_idle

    streaming_state = StreamingState()
    context = SimpleNamespace(
        fc_state={
            "active": False,
            "awaiting_response": False,
            "awaiting_eotr": False,
            "injecting_response": False,
            "forced_function_tokens": [],
        },
        frame_idx=0,
        gen_function_text=torch.full((1, 8), 12, dtype=torch.long),
        gen_text=torch.full((1, 8), 12, dtype=torch.long),
        rnnt_partial_hypotheses={"agent_speaking": torch.tensor([False])},
        tool_response_text=None,
    )

    class Wrapper:
        _fc_eotr_id = 22
        _fc_sotc_id = 20
        _post_fc_client_bos_requested = False
        _post_tc_bos_exempt = False
        model = SimpleNamespace(
            stt_model=SimpleNamespace(text_bos_id=1, text_eos_id=2, text_pad_id=12)
        )

        def _get_agent_idle(self, stream_id: int) -> bool:
            assert stream_id == 9
            return streaming_state.agent_idle

        def _set_agent_idle(self, value: bool, stream_id: int) -> None:
            assert stream_id == 9
            streaming_state.agent_idle = value

        def request_user_eou(self) -> None:
            self._external_user_eou_requested = True

        def set_external_user_eou_mode(self, enabled: bool) -> None:
            self._external_user_eou_mode = enabled

        def _apply_rnnt_turn_taking(self, frame, gen_text, blank, state) -> None:
            del blank
            if getattr(self, "_external_user_eou_requested", False):
                gen_text[0, frame] = 1
                self._external_user_eou_requested = False
            token = int(gen_text[0, frame])
            if token == 1:
                state["agent_speaking"][0] = True
            elif token == 2:
                state["agent_speaking"][0] = False

        def _apply_fc_state_machine(self, **kwargs) -> None:
            state = kwargs["fc_state"]
            token = int(kwargs["function_predicted_token"][0])
            if token == 20:
                state["active"] = True
                state["call_tokens"] = []
            elif state["active"] and token != 12:
                state["call_tokens"].append(token)

    wrapper = Wrapper()
    runtime = object.__new__(real_worker.GPUCampaignRuntime)
    runtime.control_engine = SimpleNamespace(stream_id=9)
    runtime._control_owners = lambda: (wrapper, context, streaming_state)

    def set_targets(text: list[int], function: list[int]) -> None:
        runtime.targets = (text[0], function[0])

    runtime._set_targets = set_targets
    runtime._original_step = lambda row, request_id: SimpleNamespace(
        token_id=runtime.targets[0],
        custom_outputs={"function_tokens": torch.tensor([runtime.targets[1]])},
    )
    row = torch.zeros((1, 4))
    mechanisms = []
    for control in (
        "agent_bos",
        "agent_eos",
        "function_sotc",
        "function_eotr",
        "post_fc_bos",
    ):
        _result, evidence = runtime._execute_control_boundary("S", control, row, "request")
        assert evidence["before"] != evidence["after"]
        mechanisms.append(evidence["mechanism"])
    assert mechanisms == [
        "NemotronVoicechatInferenceWrapper._apply_rnnt_turn_taking/_set_agent_idle",
        "NemotronVoicechatInferenceWrapper._apply_rnnt_turn_taking/_set_agent_idle",
        "NemotronVoicechatInferenceWrapper._apply_fc_state_machine",
        "NemotronVoicechatInferenceWrapper._apply_fc_state_machine",
        "NemotronVoicechatInferenceWrapper._apply_rnnt_turn_taking/_set_agent_idle",
    ]
    assert "control_state" not in Path(real_worker.__file__).read_text(encoding="utf-8")


def test_lane_restore_does_not_start_or_reset_the_control_engine() -> None:
    source = Path(real_worker.__file__).read_text(encoding="utf-8")
    start = source.index("    def restore_lane(")
    end = source.index("    def _start_control_engine(")
    restore_source = source[start:end]
    assert "_start_control_engine" not in restore_source
    assert ".start(" not in restore_source
    assert ".reset_session(" not in restore_source


def test_lane_restore_rejects_tampered_retained_prefix_before_replay(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix.pt"
    prefix.write_bytes(b"sealed-prefix")
    claimed = campaign.sha256_file(prefix)
    prefix.write_bytes(b"tampered-prefix")
    runtime = object.__new__(real_worker.GPUCampaignRuntime)
    with pytest.raises(RuntimeError, match="retained prefix payload changed"):
        runtime.restore_lane("S", "request", "a" * 64, None, str(prefix), claimed)


def test_worker_snapshot_fails_closed_without_true_rollback_shadow(tmp_path: Path) -> None:
    kv = SimpleNamespace(kv_cache=torch.zeros((2, 1, 1)))
    mamba = SimpleNamespace(kv_cache=(torch.zeros((1, 2)), torch.zeros((1, 2))))
    groups = [SimpleNamespace(layer_names=["attention", "mamba"])]
    runner = SimpleNamespace(
        _phase1_request_blocks={"request": {0: [0]}},
        engine_index=0,
        input_batch=SimpleNamespace(req_id_to_index={}, block_table=[]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        model=SimpleNamespace(named_modules=lambda: [("attention", kv), ("mamba", mamba)]),
        model_path="model",
        requests={"request": SimpleNamespace(num_computed_tokens=1)},
    )
    with pytest.raises(RuntimeError, match="true rollback shadow"):
        real_worker.capture_worker_state(
            SimpleNamespace(model_runner=runner), "request", str(tmp_path / "state.pt")
        )


def test_snapshot_leaf_seals_are_retained_in_result_summary() -> None:
    case = campaign.case_matrix(12)[0]
    plan = campaign.plan_for_case(case, 12)
    result = campaign.ExecutionResult(
        snapshot=campaign.FakeExecutor()._snapshot(case, plan),
        boundaries=tuple(campaign.BOUNDARIES),
        completed=True,
        terminal_unknown=False,
        dispatch_mode="normal",
        prefix_sha256=campaign.load_frozen_fixture()["fixture_sha256"],
        observed_plan=plan,
    )
    seal = campaign._result_summary(result)["snapshot_seal"]
    assert campaign.SHA256_RE.fullmatch(seal["snapshot_sha256"])
    leaf = seal["leaves"]["snapshot.committed_kv_state['kv']"]
    assert leaf["dtype"] == "torch.int64"
    assert leaf["shape"] == [2]
    assert leaf["byte_count"] == 16
    assert campaign.SHA256_RE.fullmatch(leaf["sha256"])
