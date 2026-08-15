from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

QUALIFICATION = Path(__file__).resolve().parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION))


def load_module():
    path = Path("tools/qualification/step4c_incidence.py")
    spec = importlib.util.spec_from_file_location("step4c_incidence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bootstrap_resamples_whole_blocks_and_is_deterministic() -> None:
    module = load_module()
    rates = {"direct": [0.0, 0.1, 0.2], "pipecat": [0.0, 0.0, 0.3]}

    first = module.bootstrap_interval(rates, iterations=500, seed=7)
    second = module.bootstrap_interval(rates, iterations=500, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(0.1)
    assert first["resampling_unit"].startswith("run block")
    assert first["interval_95"][0] <= first["mean"] <= first["interval_95"][1]


def test_bootstrap_rejects_empty_transport_or_iteration_set() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="positive iterations"):
        module.bootstrap_interval({"direct": []})
    with pytest.raises(ValueError, match="positive iterations"):
        module.bootstrap_interval({"direct": [0.0]}, iterations=0)


def test_schedule_is_the_preregistered_nine_block_sequence() -> None:
    module = load_module()
    assert module.SCHEDULE == (
        "direct",
        "pipecat",
        "browser",
        "pipecat",
        "browser",
        "direct",
        "browser",
        "direct",
        "pipecat",
    )
    assert {transport: module.SCHEDULE.count(transport) for transport in set(module.SCHEDULE)} == {
        "direct": 3,
        "pipecat": 3,
        "browser": 3,
    }


def test_response_target_allows_spoken_punctuation_but_rejects_refusal() -> None:
    module = load_module()
    rendition = (
        "The quick brown fox jumps over the lazy dog. While seventeen green dragons circle "
        "the ancient stone tower, they count every window, every door, and every flag that "
        "flies above the northern gate."
    )
    assert module.response_target_matches(rendition)
    assert not module.response_target_matches(
        "I cannot process that scene, but I can tell you the current UTC time."
    )


def test_container_identity_ignores_only_mount_order() -> None:
    module = load_module()
    before = {"Id": "same", "Mounts": [{"Destination": "/b"}, {"Destination": "/a"}]}
    after = {"Id": "same", "Mounts": list(reversed(before["Mounts"]))}
    assert module._container_identity(before) == module._container_identity(after)
    after["Id"] = "different"
    assert module._container_identity(before) != module._container_identity(after)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _server_events(module, session_id: str, *, high: bool = False) -> list[dict[str, object]]:
    split = len(module.L1_TEXT) // 2
    rows = []
    for index, text in enumerate((module.L1_TEXT[:split], module.L1_TEXT[split:]), start=1):
        rows.append(
            {
                "event": "model_step",
                "session_id": session_id,
                "frame": index,
                "transport_frame": index,
                "monotonic_s": float(index),
                "audio_delivered": True,
                "assistant_delta": text,
                "output_audio": {"bytes": 320, "samples": 160},
                "turn_state": {
                    "agent_token_id": 12,
                    "model_stage_timings_ms": {
                        "eartts_total_ms": 19.0 if high and index == 1 else 10.0,
                        "wrapper_residual_ms": 1.0,
                    },
                },
            }
        )
    return [
        {"event": "session_trace_started", "session_id": session_id},
        {
            "event": "session_tools_configured",
            "session_id": session_id,
            "tools": [],
        },
        {
            "event": "input_packet",
            "session_id": session_id,
            "source": "microphone",
            "audio": {"peak": 0.0, "rms": 0.0, "samples": 1280},
        },
        {
            "event": "idle_step",
            "session_id": session_id,
            "transport_frame": 0,
        },
        *rows,
    ]


def _refresh_analyzer(module, trace: Path, analyzer: Path) -> None:
    events = module.qualified.read_jsonl(trace)
    populations = module.qualified.classify_populations(
        module.qualified._normalize_model_rows(events)
    )
    _write_json(
        analyzer,
        {
            "sources": [
                {
                    "source": {
                        "path": str(trace),
                        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    },
                    "populations": {
                        name: {
                            "membership": module.qualified._membership_manifest(populations[name])
                        }
                        for name in ("delivered-nonBOS", "EarTTS-high")
                    },
                }
            ]
        },
    )


def _direct_report(module, session_id: str) -> dict[str, object]:
    return {
        "passed": True,
        "session_id": session_id,
        "qualification_fixture": module.QUALIFICATION_FIXTURE,
        "prompt_sha256": module.L1_SHA256,
        "warmup_seconds": module.WARMUP_SECONDS,
        "duration_seconds": module.DURATION_SECONDS,
        "drain_seconds": module.DRAIN_SECONDS,
        "source_frames": module.EXPECTED_SOURCE_FRAMES,
        "typed_completed": 1,
        "typed_answered": 1,
        "typed_unanswered": 0,
        "responses": [{"audio_bytes": 320, "done_reason": "decoded_silence_watchdog"}],
        "errors": [],
        "unexpected_errors": [],
        "session_closed": {
            "session_id": session_id,
            "status": "completed",
            "reason": "client_stop",
        },
    }


def _pipecat_report(module, session_id: str, response_id: str) -> dict[str, object]:
    return {
        "passed": True,
        "session_id": session_id,
        "qualification_fixture": module.QUALIFICATION_FIXTURE,
        "prompt_sha256": module.L1_SHA256,
        "warmup_seconds": module.WARMUP_SECONDS,
        "duration_seconds": module.DURATION_SECONDS,
        "drain_seconds": module.DRAIN_SECONDS,
        "source_frames": module.EXPECTED_SOURCE_FRAMES,
        "prompts_sent": 1,
        "responses_done": 1,
        "audio_bytes": 320,
        "terminal_errors": [],
        "job_records": [{"response_id": response_id, "terminal_status": "completed"}],
        "playout_trace_enabled": True,
        "playout_trace_status": {
            "enabled": True,
            "artifact_published": True,
            "closed": True,
            "valid": True,
            "dropped_records": 0,
            "queued_records": 0,
            "writer_alive": False,
            "error": None,
            "invalid_reason": None,
        },
    }


def _browser_report(module, session_id: str, response_id: str) -> dict[str, object]:
    return {
        "status": "completed",
        "error": None,
        "plan": {
            "schema": "nemotron_voicechat.step4c_fixture_plan.v1",
            "warmup_seconds": module.WARMUP_SECONDS,
            "duration_seconds": module.DURATION_SECONDS,
            "drain_seconds": module.DRAIN_SECONDS,
        },
        "fixture": {
            "typed_text_sha256": module.L1_SHA256,
            "completion_status": "completed",
            "response_ids": [response_id],
            "response_attribution": {
                "primary_response": {
                    "response_id": response_id,
                    "audio_delta_ordinals": [1],
                    "terminal": {"status": "completed"},
                }
            },
        },
        "session": {
            "status": "completed",
            "session_id": session_id,
            "connected_window_seconds": module.DURATION_SECONDS + module.DRAIN_SECONDS,
            "teardown_errors": [],
            "capture_shutdown": {
                "shutdown_error": None,
                "sink_close_error": None,
                "webrtc_disconnect": {"completed": True},
            },
            "browser_artifact_validation": {"valid": True},
            "observed_playout_trace": {
                "valid": True,
                "terminal_status": {
                    "valid": True,
                    "dropped_records": 0,
                    "error": None,
                    "invalid_reason": None,
                },
            },
        },
    }


def _campaign(tmp_path: Path, module, *, high_block: int | None = None) -> Path:
    before = tmp_path / "container-before.json"
    after = tmp_path / "container-after.json"
    _write_json(before, {"Id": "fixed", "Mounts": []})
    _write_json(after, {"Id": "fixed", "Mounts": []})
    blocks = []
    for index, transport in enumerate(module.SCHEDULE, start=1):
        root = tmp_path / f"block-{index:02d}-{transport}"
        session_id = f"session-{index:02d}"
        response_id = f"response-{index:02d}"
        trace = root / "server.jsonl"
        analyzer = root / "analyzer.json"
        health = root / "health.json"
        telemetry = root / "telemetry.jsonl"
        report = root / "report.json"
        _write_jsonl(trace, _server_events(module, session_id, high=index == high_block))
        _refresh_analyzer(module, trace, analyzer)
        _write_json(health, {"active_client": False})
        _write_jsonl(telemetry, [{"sample": 1}])
        block: dict[str, object] = {
            "block": index,
            "transport": transport,
            "session_id": session_id,
            "valid": True,
            "invalid_reason": None,
            "health": str(health),
            "client_report": str(report),
            "telemetry": str(telemetry),
            "trace": str(trace),
            "analyzer": str(analyzer),
        }
        if transport == "direct":
            _write_json(report, _direct_report(module, session_id))
        else:
            playout = root / "playout.jsonl"
            _write_jsonl(
                playout,
                [
                    {"session_id": session_id, "response_id": response_id},
                    {"valid": True, "dropped_records": 0},
                ],
            )
            block["playout_trace"] = str(playout)
            _write_json(
                report,
                _pipecat_report(module, session_id, response_id)
                if transport == "pipecat"
                else _browser_report(module, session_id, response_id),
            )
        blocks.append(block)
    manifest = tmp_path / "campaign.json"
    _write_json(
        manifest,
        {
            "schema": module.MANIFEST_SCHEMA,
            "container_inspect_before": str(before),
            "container_inspect_after": str(after),
            "blocks": blocks,
        },
    )
    return manifest


@pytest.fixture
def synthetic_campaign(tmp_path):
    module = load_module()

    def validate_playout(records):
        terminal = records[-1]
        if terminal.get("valid") is not True or terminal.get("dropped_records") != 0:
            raise ValueError("synthetic terminal playout status is invalid")
        return {"valid": True}

    module.qualified.validate_observed_playout_trace = validate_playout
    return module, _campaign(tmp_path, module)


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_analyze_regenerates_exact_membership_for_all_transports_and_zero_decision(
    synthetic_campaign,
) -> None:
    module, manifest = synthetic_campaign
    result, states = module.analyze(manifest)

    assert result["decision"] == "INSUFFICIENT"
    assert result["valid_blocks"] == 9
    assert result["invalid_blocks"] == 0
    assert result["descriptive_pooled_exposure"] == {
        "eartts_high": 0,
        "delivered_nonbos": 18,
        "incidence": 0.0,
    }
    valid_by_transport = {
        name: summary["valid_blocks"]
        for name, summary in result["transport_summaries"].items()
    }
    assert valid_by_transport == {
        "direct": 3,
        "pipecat": 3,
        "browser": 3,
    }
    assert result["overall_equal_block"] is not None
    assert states["status"] == "not-triggered"


def test_analyze_high_decision_selects_matched_pad_tail_pair(tmp_path) -> None:
    module = load_module()
    module.qualified.validate_observed_playout_trace = lambda _records: {"valid": True}
    manifest = _campaign(tmp_path, module, high_block=5)

    result, states = module.analyze(manifest)

    assert result["decision"] == "RECURRENT"
    assert result["descriptive_pooled_exposure"]["eartts_high"] == 1
    assert states["status"] == "retained-for-nsight"
    assert states["block"] == 5
    assert states["high"]["phase"] == "PAD-tail"
    assert states["low"]["phase"] == "PAD-tail"


def test_analyze_retains_honest_failed_client_report_as_excluded(synthetic_campaign) -> None:
    module, manifest = synthetic_campaign
    document = _manifest(manifest)
    block = document["blocks"][0]
    report_path = Path(block["client_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = False
    _write_json(report_path, report)
    block["valid"] = False
    block["invalid_reason"] = "CLIENT_REPORT_FAILED"
    _write_json(manifest, document)

    result, _states = module.analyze(manifest)

    assert result["valid_blocks"] == 8
    assert result["blocks"][0]["validity_failures"] == ["CLIENT_REPORT_FAILED"]


def test_manifest_flipped_missing_playout_block_fails_closed(synthetic_campaign) -> None:
    module, manifest = synthetic_campaign
    document = _manifest(manifest)
    block = document["blocks"][1]
    block["valid"] = False
    block["invalid_reason"] = "PLAYOUT_TRACE_NOT_PUBLISHED"
    block["playout_trace"] = str(Path(block["playout_trace"]).with_name("missing.jsonl"))
    report_path = Path(block["client_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["playout_trace_status"] = None
    _write_json(report_path, report)
    _write_json(manifest, document)
    result, _states = module.analyze(manifest)
    assert result["valid_blocks"] == 8

    block["valid"] = True
    block["invalid_reason"] = None
    _write_json(manifest, document)
    with pytest.raises(ValueError, match="declared valid.*PLAYOUT_TRACE_NOT_PUBLISHED"):
        module.analyze(manifest)


def test_analyze_allows_browser_connected_window_at_clock_tolerance(
    synthetic_campaign,
) -> None:
    module, manifest = synthetic_campaign
    document = _manifest(manifest)
    block = document["blocks"][2]
    report_path = Path(block["client_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["session"]["connected_window_seconds"] = (
        module.DURATION_SECONDS
        + module.DRAIN_SECONDS
        - module.BROWSER_CONNECTED_WINDOW_TOLERANCE_SECONDS
    )
    _write_json(report_path, report)

    result, _states = module.analyze(manifest)

    assert result["valid_blocks"] == 9


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schedule", "schedule"),
        ("session", "session identity"),
        ("health", "HEALTH_NOT_EXCLUSIVE"),
        ("container", "container identity"),
        ("trace_hash", "source identity"),
        ("continuity", "TRANSPORT_FRAMES_NONCONTIGUOUS"),
        ("acoustic", "ACOUSTIC_INPUT_NOT_EXACT_ZERO"),
        ("fixture", "FIXTURE_IDENTITY_MISMATCH"),
        ("direct_source_frames_truncated", "FIXTURE_TIMING_MISMATCH"),
        ("direct_source_frames_missing", "FIXTURE_TIMING_MISMATCH"),
        ("pipecat_source_frames_truncated", "FIXTURE_TIMING_MISMATCH"),
        ("pipecat_source_frames_missing", "FIXTURE_TIMING_MISMATCH"),
        ("browser_connected_window_truncated", "FIXTURE_TIMING_MISMATCH"),
        ("browser_connected_window_missing", "FIXTURE_TIMING_MISMATCH"),
        ("response", "RESPONSE_CARDINALITY_INVALID"),
        ("teardown", "TEARDOWN_NOT_CLEAN"),
        ("playout_publication", "PLAYOUT_TRACE_NOT_PUBLISHED"),
        ("playout_drop", "PLAYOUT_TRACE_NOT_PUBLISHED"),
        ("invalid_reason", "contradicts evidence"),
    ],
)
def test_analyze_rejects_preregistered_gate_mutations(
    synthetic_campaign, mutation: str, message: str
) -> None:
    module, manifest = synthetic_campaign
    document = _manifest(manifest)
    block = document["blocks"][0]
    if mutation == "schedule":
        block["transport"] = "browser"
    elif mutation == "session":
        block["session_id"] = "different-session"
    elif mutation == "health":
        _write_json(Path(block["health"]), {"active_client": True})
    elif mutation == "container":
        _write_json(Path(document["container_inspect_after"]), {"Id": "changed", "Mounts": []})
    elif mutation == "trace_hash":
        with Path(block["trace"]).open("a", encoding="utf-8") as stream:
            stream.write(" \n")
    elif mutation == "continuity":
        trace = Path(block["trace"])
        events = module.qualified.read_jsonl(trace)
        events[-1]["transport_frame"] = 3
        _write_jsonl(trace, events)
        _refresh_analyzer(module, trace, Path(block["analyzer"]))
    elif mutation == "acoustic":
        trace = Path(block["trace"])
        events = module.qualified.read_jsonl(trace)
        events[2]["audio"]["peak"] = 1 / 32768
        events[2]["audio"]["rms"] = 1 / 32768
        _write_jsonl(trace, events)
        _refresh_analyzer(module, trace, Path(block["analyzer"]))
    elif mutation == "fixture":
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["prompt_sha256"] = "wrong"
        _write_json(report_path, report)
    elif mutation in {"direct_source_frames_truncated", "direct_source_frames_missing"}:
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation.endswith("truncated"):
            report["source_frames"] = 1
        else:
            report.pop("source_frames")
        _write_json(report_path, report)
    elif mutation in {"pipecat_source_frames_truncated", "pipecat_source_frames_missing"}:
        block = document["blocks"][1]
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation.endswith("truncated"):
            report["source_frames"] = 1
        else:
            report.pop("source_frames")
        _write_json(report_path, report)
    elif mutation in {
        "browser_connected_window_truncated",
        "browser_connected_window_missing",
    }:
        block = document["blocks"][2]
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation.endswith("truncated"):
            report["session"]["connected_window_seconds"] = (
                module.DURATION_SECONDS
                + module.DRAIN_SECONDS
                - module.BROWSER_CONNECTED_WINDOW_TOLERANCE_SECONDS
                - 0.000001
            )
        else:
            report["session"].pop("connected_window_seconds")
        _write_json(report_path, report)
    elif mutation == "response":
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["responses"] = []
        _write_json(report_path, report)
    elif mutation == "teardown":
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["session_closed"] = None
        _write_json(report_path, report)
    elif mutation in {"playout_publication", "playout_drop"}:
        block = document["blocks"][1]
        report_path = Path(block["client_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mutation == "playout_publication":
            report["playout_trace_status"]["artifact_published"] = False
        else:
            report["playout_trace_status"]["dropped_records"] = 1
        _write_json(report_path, report)
    else:
        block["valid"] = False
        block["invalid_reason"] = "CLIENT_REPORT_FAILED"
    _write_json(manifest, document)

    with pytest.raises(ValueError, match=message):
        module.analyze(manifest)
