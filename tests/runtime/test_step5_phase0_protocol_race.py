from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools/qualification/step5_phase0_protocol_race.py"
SPEC = importlib.util.spec_from_file_location("step5_phase0_protocol_race", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
race = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = race
SPEC.loader.exec_module(race)


def test_splitmix_schedule_seed_is_stable() -> None:
    assert race.splitmix64(race.ROOT_SEED) == 0x511080C9DDC15896
    assert race.splitmix64(race.ROOT_SEED + 999_999) == 0xB72DFE89666FB3CB


def test_domain_separated_fault_seeds_are_stable_and_distinct() -> None:
    seeds = [race.domain_seed(f"fault/{boundary}") for boundary in race.BOUNDARIES]
    assert len(set(seeds)) == 10
    assert seeds[0] == 8685995984842014039


def test_schedule_record_is_versioned_and_replayable() -> None:
    first = race.decision_for(177).expanded()
    second = race.decision_for(177).expanded()
    assert first == second
    assert first["version"] == 3
    assert len(first["replay_vector"]) == 4
    assert len(first["yields"]) == 24


def test_detector_is_independent_of_operation_loss_switch() -> None:
    audit = race.BoundaryAudit()
    for boundary in race.BOUNDARIES[:4]:
        audit.mark(boundary)
    assert audit.first_missing() == "B4"
    assert not hasattr(audit, "operation_loss")


def test_record_suppression_remains_a_separate_trace_self_test() -> None:
    audit = race.BoundaryAudit(record_suppression="B6")
    for boundary in race.BOUNDARIES:
        audit.mark(boundary)
    assert audit.first_missing() == "B6"


def test_exact_class_loader_compiles_selected_qualified_definition(tmp_path: Path) -> None:
    source = tmp_path / "qualified.py"
    source.write_text(
        "from __future__ import annotations\n"
        "class Selected:\n"
        "    def route(self, request_id: str) -> str:\n"
        "        return request_id\n"
        "class NotSelected:\n"
        "    pass\n"
    )
    namespace: dict[str, object] = {}
    loaded = race._compile_classes(source, ("Selected",), namespace)
    assert loaded["Selected"]().route("r7") == "r7"
    assert "NotSelected" not in namespace


def test_resume_rejects_gap_overlap_and_hash_drift(tmp_path: Path) -> None:
    identity = {"identity_sha256": "identity"}
    batch = {
        "compatibility_identity_sha256": "identity",
        "counts": {"schedules": 2},
        "end_schedule_exclusive": 2,
        "equivalence_classes": {"class": 2},
        "representatives": {
            "class": {
                "count": 2,
                "expanded_schedule": {"version": 3},
                "first_index": 0,
                "last_index": 1,
                "realized_events": ["event"],
            }
        },
        "start_schedule": 0,
    }
    batch["batch_sha256"] = race.stable_digest(batch)
    path = tmp_path / "batches.jsonl"
    path.write_text(json.dumps(batch) + "\n")
    assert race._validate_resume(path, identity)[0] == 2
    batch["start_schedule"] = 1
    batch["batch_sha256"] = race.stable_digest(
        {key: value for key, value in batch.items() if key != "batch_sha256"}
    )
    path.write_text(json.dumps(batch) + "\n")
    with pytest.raises(ValueError, match="gap/overlap"):
        race._validate_resume(path, identity)


def test_resume_restores_prefix_representatives(tmp_path: Path) -> None:
    identity = {"identity_sha256": "identity"}
    batch = {
        "compatibility_identity_sha256": "identity",
        "counts": {"schedules": 1},
        "end_schedule_exclusive": 1,
        "equivalence_classes": {"class": 1},
        "representatives": {
            "class": {
                "count": 1,
                "expanded_schedule": {"index": 0, "version": 3},
                "first_index": 0,
                "last_index": 0,
                "realized_events": ["event"],
            }
        },
        "start_schedule": 0,
    }
    batch["batch_sha256"] = race.stable_digest(batch)
    path = tmp_path / "batches.jsonl"
    path.write_text(json.dumps(batch) + "\n")
    end, _totals, classes, representatives = race._validate_resume(path, identity)
    assert end == 1
    assert classes == {"class": 1}
    assert representatives["class"]["expanded_schedule"]["index"] == 0


def test_artifact_gate_covers_nested_ring_summary_and_manifest_itself(
    tmp_path: Path,
) -> None:
    (tmp_path / "phase0-summary-v3.json").write_text('{"passed":true}\n')
    trace = tmp_path / "noninterference-trace" / "trace.ring"
    trace.parent.mkdir()
    trace.write_bytes(b"ring")
    assert race.write_and_verify_artifact_manifest(tmp_path, True)
    manifest = json.loads((tmp_path / race.ARTIFACT_MANIFEST_NAME).read_text())
    assert "phase0-summary-v3.json" in manifest["files"]
    assert "noninterference-trace/trace.ring" in manifest["files"]
    assert manifest["self_canonical_sha256"]
    trace.write_bytes(b"drift")
    assert not race.verify_artifact_manifest(tmp_path)


def test_compatibility_identity_enforces_revision_and_run_shape() -> None:
    identity = race.compatibility_identity(
        batch_size=race.DEFAULT_BATCH_SIZE,
        concurrency=race.DEFAULT_CONCURRENCY,
    )
    assert identity["application_revision"] == race.APPLICATION_REVISION
    assert identity["configuration"]["batch_size"] == race.DEFAULT_BATCH_SIZE
    assert identity["configuration"]["concurrency"] == race.DEFAULT_CONCURRENCY
    changed = race.compatibility_identity(
        batch_size=race.DEFAULT_BATCH_SIZE // 2,
        concurrency=race.DEFAULT_CONCURRENCY,
    )
    assert changed["identity_sha256"] != identity["identity_sha256"]


def test_real_task_fixture_runs_exact_paths_when_qualified_tree_is_available() -> None:
    root = Path("/tmp/voicechat-phase1-diff.VbhhUc/qualified-vllm")
    if not root.is_dir():
        pytest.skip("local qualified escrow reconstruction unavailable")
    result = __import__("asyncio").run(race._fixture_async(root, 16))
    assert result["missing"] == 0
    assert result["routing_mismatches"] == 0
    assert result["production_packed_host"]["append_calls"] == 1


def test_v3_all_yields_are_realized_and_b9_strands_waiting_iterator() -> None:
    root = Path("/tmp/voicechat-phase1-diff.VbhhUc/qualified-vllm")
    if not root.is_dir():
        pytest.skip("local qualified escrow reconstruction unavailable")

    async def probe() -> tuple[set[int], dict]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="step5-v3-test-") as temporary:
            patched = Path(temporary) / "patched"
            race.apply_patch(root, patched)
            qualified = race.load_qualified_classes(root, patched)
            exact_request = qualified.Request(
                request_id="admission-check",
                prompt_token_ids=[1],
                sampling_params=None,
                pooling_params=race._PoolingParams(),
                eos_token_id=None,
            )
            scheduler = race._new_scheduler(qualified, exact_request)
            assert exact_request.status == qualified.RequestStatus.WAITING
            assert len(scheduler.waiting) == 1
            assert scheduler.running == []
            scheduler.set_custom_inputs(
                exact_request.request_id,
                {"combined_embeds": __import__("torch").zeros((2, 1))},
            )
            descriptor = scheduler.schedule()
            assert exact_request.request_id in descriptor.num_scheduled_tokens
            assert len(scheduler.waiting) == 0
            assert scheduler.running == [exact_request]
            runtime = race.RaceRuntime(qualified)
            await runtime.start()
            try:
                results = await race._run_range(runtime, 0, 64, 32)
                b9_index = race.domain_seed("fault/B9") % race.REQUIRED_SCHEDULES
                b9 = await race.run_schedule_async(runtime, b9_index, operation_loss="B9")
            finally:
                await runtime.close()
        points = {
            point
            for result in results
            for point, _decision in result["realized_yields"]
        }
        return points, b9

    points, b9 = __import__("asyncio").run(probe())
    assert points == set(race.YIELD_POINTS)
    assert all(value == "B9" for value in b9["first_missing"])
    assert len(b9["pending_before_cleanup"]) == b9["lane_count"]
