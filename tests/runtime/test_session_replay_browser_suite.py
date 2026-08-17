from __future__ import annotations

import hashlib
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "tools/qualification/session_replay_browser_suite.py"
TRACE_ID = "c42c6701-f14d-4960-83fc-3c24bd269194"
TRACE_PATH = (
    Path.home() / ".local/state/nemotron-voicechat/traces/model" / TRACE_ID / "events.jsonl"
)


def load_suite():
    spec = importlib.util.spec_from_file_location("session_replay_browser_suite", SUITE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pcm_to_wav_preserves_source_bytes_after_explicit_transport_preroll(tmp_path) -> None:
    suite = load_suite()
    source = tmp_path / "input.pcm"
    payload = np.arange(320, dtype="<i2").tobytes()
    source.write_bytes(payload)
    output = tmp_path / "input.wav"

    duration = suite.pcm_to_wav(source, output, leading_silence_seconds=0.25)

    with wave.open(str(output), "rb") as stream:
        rendered = stream.readframes(stream.getnframes())
        assert stream.getframerate() == 16_000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
    preroll_bytes = 2 * round(16_000 * 0.25)
    assert rendered[:preroll_bytes] == bytes(preroll_bytes)
    assert rendered[preroll_bytes:] == payload
    assert duration == 0.25 + len(payload) / 2 / 16_000
    assert hashlib.sha256(rendered[preroll_bytes:]).digest() == hashlib.sha256(payload).digest()


def test_retained_trace_analysis_exposes_all_observed_issue_metrics() -> None:
    suite = load_suite()
    assert TRACE_PATH.is_file()

    records = suite.load_jsonl(TRACE_PATH)
    responses = suite._response_records(records)
    results = suite._tool_results(records)

    assert [response["ordinal"] for response in responses if not response["user_text"]] == [
        2,
        3,
        4,
    ]
    assert results[:3] == [
        "It is currently 6:40 PM UTC.",
        "The time zone is now America/Los_Angeles.",
        "It is currently 11:40 AM PDT.",
    ]
    incomplete = {
        response["ordinal"]: response["i3_classification"]
        for response in responses
        if not response["sentence_boundary"]
    }
    assert incomplete[3] == "model_silence"
    assert incomplete[4] == "model_silence"
    assert incomplete[8] == "model_turn_taking"
    assert not any(value == "runtime_boundary_while_productive" for value in incomplete.values())


def test_retained_freshness_metric_counts_superseded_utc_reports() -> None:
    suite = load_suite()
    analysis = suite.analyze_trace(TRACE_PATH, expected_phase="post")

    assert analysis["response_attribution"]["failures"] == [2, 3, 4]
    freshness = analysis["freshness"]
    assert freshness["status"] == "stale_observed"
    assert freshness["eligibility"]["eligible"] is True
    assert freshness["eligible_responses"] == 6
    assert [item["ordinal"] for item in freshness["stale_responses"]] == [6, 7, 9]
    assert [item["ordinal"] for item in freshness["fresh_responses"]] == [4, 5, 8]
    assert freshness["stale_rate"] == 0.5
    assert freshness["fresh_rate"] == 0.5
    assert [item["classification"] for item in freshness["per_response"]] == [
        "fresh_no_superseded_result",
        "fresh_no_superseded_result",
        "stale_superseded_result",
        "stale_superseded_result",
        "fresh_no_superseded_result",
        "stale_superseded_result",
    ]
    assert freshness["observed_without_stale"] is False
    assert analysis["freshness_classification"] == "reported_metric_not_acceptance_gate"


def test_attempt5_tool_propensity_is_reported_without_failing_runtime_gates() -> None:
    suite = load_suite()
    trace = REPO_ROOT / "reports/session-repair/post-fix-browser-attempt5/model-trace/events.jsonl"
    assert trace.is_file()

    analysis = suite.analyze_trace(trace, expected_phase="post")

    propensity = analysis["tool_propensity"]
    assert propensity["classification"] == "reported_metric_not_acceptance_gate"
    assert propensity["observed_sequence"] == ["get_current_time", "set_timezone"]
    assert propensity["calls_elicited"] == 2
    by_frame = {turn["bos_frame"]: turn for turn in propensity["per_turn"]}
    assert by_frame[28]["call_requested"] is True
    assert by_frame[191]["call_requested"] is True
    assert by_frame[502]["utterance"] == "okay cool what time is it now"
    assert by_frame[502]["call_requested"] is False
    assert analysis["freshness"]["status"] == "not_eligible"
    assert analysis["runtime_gates_passed"] is True


def test_capture_artifacts_are_json_and_jsonl_serializable() -> None:
    suite = load_suite()
    records = suite.load_jsonl(TRACE_PATH)
    payload = {
        "responses": suite._response_records(records),
        "tool_results": suite._tool_results(records),
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
