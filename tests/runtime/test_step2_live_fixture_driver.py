from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

QUALIFICATION_TOOLS = Path(__file__).resolve().parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION_TOOLS))

import step2_live_fixture_driver as driver
from step2_live_fixture_driver import (
    BROWSER_TRACE_SCHEMA,
    CANONICAL_FIXTURE_DOCUMENT,
    CANONICAL_PLAN_SHA256,
    CANONICAL_SCRIPT_HASHES_SHA256,
    MARKER_SCHEMA,
    PLAYOUT_TRACE_PUBLICATION_POLL_S,
    PLAYOUT_TRACE_PUBLICATION_TIMEOUT_S,
    PLAYOUT_TRACE_SCHEMA,
    RTVI_QUIESCENCE_MIN_WINDOW_MS,
    RTVI_QUIESCENCE_WINDOW_MS,
    RTVI_TRANSCRIPT_IDENTITY,
    RTVI_WALL_CLOCK_SKEW_TOLERANCE_S,
    STEP4C_PLAN_SCHEMA,
    STEP4C_SUMMARY_SCHEMA,
    SUMMARY_SCHEMA,
    active_playout_trace_path,
    blank_fixture_result,
    canonical_plan_json,
    connection_playout_trace_path,
    correlate_response_lifecycles,
    fixture_marker,
    parse_fixture_plan,
    parse_rtvi_quiescence_window_ms,
    parse_step4c_fixture_plan,
    plan_document,
    read_playout_trace,
    resolve_step2_paths,
    rtvi_transcript_evidence,
    step4c_plan_json,
    validate_browser_artifact,
    validate_connection_playout_trace_set,
    validate_fixture_summary,
    validate_marker,
    validate_published_playout_artifact,
    validate_step4c_summary,
    wait_for_interruption_deadline,
    wait_for_playout_trace_publication,
)


def canonical_plan():
    return parse_fixture_plan(canonical_plan_json())


def test_rtvi_quiescence_environment_parser_preserves_preregistered_default() -> None:
    assert parse_rtvi_quiescence_window_ms(None) == RTVI_QUIESCENCE_WINDOW_MS


@pytest.mark.parametrize(
    "raw",
    ["0", "499", "-1", "nan", "inf", "-inf", "not-a-number", "60001"],
)
def test_rtvi_quiescence_environment_parser_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match="RTVI quiescence window"):
        parse_rtvi_quiescence_window_ms(raw)


def test_rtvi_quiescence_environment_parser_accepts_positive_floor() -> None:
    assert (
        parse_rtvi_quiescence_window_ms(str(RTVI_QUIESCENCE_MIN_WINDOW_MS))
        == RTVI_QUIESCENCE_MIN_WINDOW_MS
    )


def rtvi_messages() -> list[dict[str, object]]:
    return [
        {
            "receivedPerformanceMs": 10.0,
            "receivedWallTimeS": 100.01,
            "message": {"type": "bot-started-speaking", "data": {}},
        },
        {
            "receivedPerformanceMs": 20.0,
            "receivedWallTimeS": 100.02,
            "message": {"type": "bot-transcription", "data": {"text": "answer"}},
        },
        {
            "receivedPerformanceMs": 30.0,
            "receivedWallTimeS": 100.03,
            "message": {"type": "bot-stopped-speaking", "data": {}},
        },
    ]


def test_rtvi_transcript_indices_survive_message_rematerialization() -> None:
    selected_messages = rtvi_messages()
    rematerialized_messages = copy.deepcopy(selected_messages)

    texts, bracket = rtvi_transcript_evidence(
        rematerialized_messages,
        start_event_index=0,
        terminal_event_index=2,
        start_event=selected_messages[0],
        terminal_event=selected_messages[2],
        response_id="response-1",
        expected_output_text="answer",
        fixture_send_performance_ms=5.0,
        fixture_send_wall_time_s=100.0,
    )

    assert texts == ["answer"]
    assert bracket["start_event_index"] == 0
    assert bracket["terminal_event_index"] == 2


@pytest.mark.parametrize("mismatch", ["type", "performance-clock", "wall-clock"])
def test_rtvi_transcript_indices_reject_mismatched_event_revalidation(
    mismatch: str,
) -> None:
    selected_messages = rtvi_messages()
    rematerialized_messages = copy.deepcopy(selected_messages)
    if mismatch == "type":
        rematerialized_messages[0]["message"]["type"] = "bot-interrupted"
    elif mismatch == "performance-clock":
        rematerialized_messages[0]["receivedPerformanceMs"] = 11.0
    else:
        rematerialized_messages[0]["receivedWallTimeS"] = 100.011

    with pytest.raises(ValueError, match="revalidation mismatch"):
        rtvi_transcript_evidence(
            rematerialized_messages,
            start_event_index=0,
            terminal_event_index=2,
            start_event=selected_messages[0],
            terminal_event=selected_messages[2],
            response_id="response-1",
            expected_output_text="answer",
            fixture_send_performance_ms=5.0,
            fixture_send_wall_time_s=100.0,
        )


@pytest.mark.parametrize(("start_index", "terminal_index"), [(-1, 2), (0, 3)])
def test_rtvi_transcript_indices_are_bounds_checked(start_index: int, terminal_index: int) -> None:
    messages = rtvi_messages()

    with pytest.raises(ValueError, match="out of bounds"):
        rtvi_transcript_evidence(
            messages,
            start_event_index=start_index,
            terminal_event_index=terminal_index,
            start_event=messages[0],
            terminal_event=messages[2],
            response_id="response-1",
            expected_output_text="answer",
            fixture_send_performance_ms=5.0,
            fixture_send_wall_time_s=100.0,
        )


def trace(event_type: str, clock: float, **fields: object) -> dict[str, object]:
    clock_name = {
        "response.created": "client_received_monotonic_s",
        "response.output_text.delta": "client_received_monotonic_s",
        "response.output_audio.delta": "client_received_monotonic_s",
        "response.done": "client_received_monotonic_s",
        "input_audio_buffer.speech_started": "client_received_monotonic_s",
        "voicechat.playout.interruption": "client_interruption_monotonic_s",
        "voicechat.playout.release": "client_release_monotonic_s",
        "voicechat.playout.downstream_push": "client_downstream_push_monotonic_s",
    }[event_type]
    return {
        "trace_schema": PLAYOUT_TRACE_SCHEMA,
        "type": event_type,
        clock_name: clock,
        **fields,
    }


def response_records(
    response_id: str,
    start: float,
    *,
    status: str = "completed",
    clear: bool = False,
    include_typed_send: bool = True,
    output_text: str = "The answer is four.",
) -> list[dict[str, object]]:
    release_reason = "interruption-clear" if clear else "done"
    return [
        *(
            [
                trace(
                    "input_audio_buffer.speech_started",
                    start - 0.1,
                    source="typed",
                    response_id=None,
                )
            ]
            if include_typed_send
            else []
        ),
        trace("response.created", start, response_id=response_id, turn_id=None),
        trace(
            "response.output_text.delta",
            start + 0.05,
            response_id=response_id,
            ordinal=1,
            delta=output_text,
        ),
        trace("response.output_audio.delta", start + 0.1, response_id=response_id, ordinal=1),
        *(
            [
                trace(
                    "response.output_audio.delta",
                    start + 0.11,
                    response_id=response_id,
                    ordinal=2,
                ),
                trace(
                    "voicechat.playout.release",
                    start + 0.12,
                    response_id=response_id,
                    reason="threshold",
                    ordinals=[1, 2],
                    frames_released=2,
                    frames_cleared=0,
                ),
                trace(
                    "voicechat.playout.downstream_push",
                    start + 0.13,
                    response_id=response_id,
                    ordinal=1,
                ),
                trace(
                    "voicechat.playout.downstream_push",
                    start + 0.14,
                    response_id=response_id,
                    ordinal=2,
                ),
                trace(
                    "input_audio_buffer.speech_started",
                    start + 0.2,
                    source="typed",
                    response_id=response_id,
                ),
                trace(
                    "voicechat.playout.release",
                    start + 0.21,
                    response_id=response_id,
                    reason=release_reason,
                    ordinals=[],
                    frames_released=0,
                    frames_cleared=0,
                ),
            ]
            if clear
            else []
        ),
        trace(
            "response.done",
            start + 0.3,
            response_id=response_id,
            status=status,
        ),
        *(
            []
            if clear
            else [
                trace(
                    "voicechat.playout.release",
                    start + 0.31,
                    response_id=response_id,
                    reason=release_reason,
                    ordinals=[1],
                    frames_released=1,
                    frames_cleared=0,
                ),
                trace(
                    "voicechat.playout.downstream_push",
                    start + 0.32,
                    response_id=response_id,
                    ordinal=1,
                ),
            ]
        ),
    ]


def post_release_immediate_push_records(
    response_id: str = "response-1",
) -> list[dict[str, object]]:
    return [
        trace(
            "input_audio_buffer.speech_started",
            0.9,
            source="typed",
            response_id=None,
        ),
        trace("response.created", 1.0, response_id=response_id, turn_id=None),
        trace(
            "response.output_text.delta",
            1.05,
            response_id=response_id,
            ordinal=1,
            delta="The answer is four.",
        ),
        trace("response.output_audio.delta", 1.1, response_id=response_id, ordinal=1),
        trace("response.output_audio.delta", 1.11, response_id=response_id, ordinal=2),
        trace(
            "voicechat.playout.release",
            1.12,
            response_id=response_id,
            reason="threshold",
            ordinals=[1, 2],
            frames_released=2,
            frames_cleared=0,
        ),
        trace(
            "voicechat.playout.downstream_push",
            1.13,
            response_id=response_id,
            ordinal=1,
        ),
        trace(
            "voicechat.playout.downstream_push",
            1.14,
            response_id=response_id,
            ordinal=2,
        ),
        trace("response.output_audio.delta", 1.2, response_id=response_id, ordinal=3),
        trace(
            "voicechat.playout.downstream_push",
            1.21,
            response_id=response_id,
            ordinal=3,
        ),
        trace("response.done", 1.3, response_id=response_id, status="completed"),
    ]


def browser_record(event_type: str, receipt: float, **fields: object) -> dict[str, object]:
    return {
        "trace_schema": BROWSER_TRACE_SCHEMA,
        "type": event_type,
        "playwright_batch_received_monotonic_s": receipt,
        **fields,
    }


def _canonical_browser_records(*, capture_enabled: bool) -> list[dict[str, object]]:
    plan = canonical_plan()
    records: list[dict[str, object]] = []
    receipt = 1.0
    if capture_enabled:
        records.extend(
            [
                browser_record("browser.playout.capability", receipt, audio_worklet=True),
                browser_record(
                    "browser.audio_worklet.registered",
                    receipt + 0.01,
                    worklet_id="worklet-1",
                ),
            ]
        )
        receipt += 0.02
    for fixture in plan.sessions()[0 if capture_enabled else 1][1]:
        boundaries = (
            ("start", "interruption-send", "complete")
            if fixture.interruption
            else ("start", "complete")
        )
        for boundary in boundaries:
            records.append(
                {
                    **fixture_marker(
                        fixture,
                        boundary,
                        browser_performance_ms=receipt * 1000,
                        browser_wall_time_s=100 + receipt,
                    ),
                    "playwright_batch_received_monotonic_s": receipt,
                    "playwright_batch_received_wall_time_s": 100 + receipt,
                }
            )
            receipt += 0.01
    if capture_enabled:
        records.append(
            browser_record(
                "browser.audio_worklet.flush_ack",
                receipt,
                worklet_id="worklet-1",
                final_ordinal=0,
                delivered_last_ordinal=0,
            )
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write_playout_artifact(path: Path, clock: float) -> None:
    _write_jsonl(
        path,
        [
            {
                "trace_schema": PLAYOUT_TRACE_SCHEMA,
                "type": "voicechat.playout.config",
                "client_configured_monotonic_s": clock,
                "configured_prebuffer_ms": 160,
                "sample_rate_hz": 22050,
                "channels": 1,
            },
            {
                "trace_schema": PLAYOUT_TRACE_SCHEMA,
                "type": "voicechat.playout.trace_status",
                "trace_closed_monotonic_s": clock + 1,
                "valid": True,
                "invalid_reason": None,
                "error": None,
                "dropped_records": 0,
            },
        ],
    )


def _artifact_response_records(
    response_id: str,
    start: float,
    *,
    status: str = "completed",
    clear: bool = False,
    include_typed_send: bool = True,
) -> list[dict[str, object]]:
    records = response_records(
        response_id,
        start,
        status=status,
        clear=clear,
        include_typed_send=include_typed_send,
    )
    for record in records:
        if record["type"] == "response.output_audio.delta":
            sample_count = 3527 if clear and record["ordinal"] == 1 else 2 if clear else 1
            record.update(
                sample_count=sample_count,
                encoding="pcm16",
                sample_rate=22050,
                sample_rate_hz=22050,
                channels=1,
                delta=base64.b64encode(b"\0\0" * sample_count).decode(),
            )
        elif record["type"] == "voicechat.playout.release":
            record["sample_count"] = (
                3529 if clear and record["reason"] == "threshold" else 0 if clear else 1
            )
        elif record["type"] == "voicechat.playout.downstream_push":
            record.update(
                sample_count=(3527 if clear and record["ordinal"] == 1 else 2 if clear else 1),
                sample_rate_hz=22050,
                channels=1,
            )
    return records


def _summary_browser_records(
    results: list[dict[str, object]], *, capture_enabled: bool
) -> list[dict[str, object]]:
    records = _canonical_browser_records(capture_enabled=capture_enabled)
    for record in records:
        if record["type"] != "browser.fixture_boundary":
            continue
        result = results[int(record["fixture_index"])]
        boundary = record["boundary"]
        if boundary == "start":
            lower = float(result["send_attempt_wall_time_s"])
            upper = float(result["send_wall_time_s"])
            performance = 1000.0 + int(record["fixture_index"]) * 100.0
        elif boundary == "interruption-send":
            interruption = result["interruption"]
            lower = float(interruption["send_attempt_wall_time_s"])
            upper = float(interruption["send_wall_time_s"])
            performance = 1020.0 + int(record["fixture_index"]) * 100.0
        else:
            lower = float(result["last_audio_wall_time_s"])
            upper = float(result["completion_wall_time_s"])
            performance = (1060.0 if result["interruption"] is not None else 1040.0) + int(
                record["fixture_index"]
            ) * 100.0
        wall = (lower + upper) / 2
        record["browser_performance_ms"] = performance
        record["browser_wall_time_s"] = wall
        record["playwright_batch_received_wall_time_s"] = wall + 0.01
    return records


def completed_summary(tmp_path: Path) -> dict[str, object]:
    plan = canonical_plan()
    results: list[dict[str, object]] = []
    session_records: dict[str, list[dict[str, object]]] = {"on": [], "off": []}
    for fixture in plan.fixtures:
        result = blank_fixture_result(fixture)
        response_ids = (
            [f"response-{fixture.plan_index}-original", f"response-{fixture.plan_index}-new"]
            if fixture.interruption
            else [f"response-{fixture.plan_index}"]
        )
        base_wall = 100.0 + fixture.plan_index * 10
        trace_start = 1.0 + fixture.plan_index * 10.0
        if fixture.interruption:
            records = _artifact_response_records(
                response_ids[0], trace_start, status="cancelled", clear=True
            )
            records += _artifact_response_records(response_ids[1], trace_start + 1.0)
            attribution = correlate_response_lifecycles(records, interruption=True)
        else:
            records = _artifact_response_records(response_ids[0], trace_start)
            attribution = correlate_response_lifecycles(records, interruption=False)
        assert attribution is not None
        session_records[fixture.capture].extend(records)
        result.update(
            {
                "response_ids": response_ids,
                "response_attribution": attribution,
                "send_attempt_wall_time_s": base_wall,
                "send_wall_time_s": base_wall + 0.1,
                "first_audio_wall_time_s": base_wall + 1.0,
                "last_audio_wall_time_s": base_wall + 3.0,
                "completion_wall_time_s": base_wall + 3.1,
                "completion_status": "completed",
                "transcript_evidence": ["The answer is four."],
                "transcript_gate_evidence": driver.transcript_gate_evidence(
                    ["The answer is four."],
                    attribution[
                        "replacement_response" if fixture.interruption else "primary_response"
                    ]["output_text"],
                ),
                "quiescence_barrier": {
                    "identity": RTVI_TRANSCRIPT_IDENTITY,
                    "quiet_window_ms": 2000.0,
                    "window_start_performance_ms": (1050.0 if fixture.interruption else 1030.0)
                    + fixture.plan_index * 100.0,
                    "window_end_performance_ms": (3050.0 if fixture.interruption else 3030.0)
                    + fixture.plan_index * 100.0,
                    "window_start_wall_time_s": base_wall + 3.0,
                    "window_end_wall_time_s": base_wall + 5.0,
                    "start_message_count": fixture.plan_index * 3 + 3,
                    "end_message_count": fixture.plan_index * 3 + 3,
                    "observed_message_count": 0,
                    "bot_message_count": 0,
                    "bot_message_types": [],
                    "passed": True,
                },
            }
        )
        result["marker_emission"]["start"]["status"] = "succeeded"
        result["marker_emission"]["complete"]["status"] = "succeeded"
        if fixture.interruption:
            result["rtvi_window_start_index"] = 0
            result["marker_emission"]["interruption-send"]["status"] = "succeeded"
            result["interruption"].update(
                {
                    "send_attempt_wall_time_s": base_wall + 2.0,
                    "send_wall_time_s": base_wall + 2.1,
                    "rtvi_send_marker_event_index": 1,
                    "actual_offset_s": 4.0,
                    "second_answer_transcript_evidence": ["The answer is four."],
                    "second_answer_gate_evidence": driver.second_answer_gate_evidence(
                        ["The answer is four."],
                        attribution["replacement_response"]["output_text"],
                    ),
                }
            )
            attribution["original_response"].update(
                {
                    "speaking_start_wall_time_s": base_wall + 1.0,
                    "interrupted_wall_time_s": base_wall + 2.0,
                    "rtvi_event_bracket": {
                        "response_id": response_ids[0],
                        "transcript_binding": RTVI_TRANSCRIPT_IDENTITY,
                        "protocol_response_identity": None,
                        "fixture_send_performance_ms": 1000.0 + fixture.plan_index * 100.0,
                        "fixture_send_wall_time_s": base_wall + 0.1,
                        "messages_received_before_send_in_bracket": 0,
                        "start_event_index": 0,
                        "terminal_event_index": 1,
                        "start_event_type": "bot-started-speaking",
                        "terminal_event_type": "bot-interrupted",
                        "start_performance_ms": 1010.0 + fixture.plan_index * 100.0,
                        "terminal_performance_ms": 1030.0 + fixture.plan_index * 100.0,
                        "start_wall_time_s": base_wall + 1.0,
                        "terminal_wall_time_s": base_wall + 2.0,
                        "transcript_event_indices": [],
                        "transcript_performance_ms": [],
                        "output_text_sha256": hashlib.sha256(
                            attribution["original_response"]["output_text"].encode()
                        ).hexdigest(),
                    },
                }
            )
            attribution["replacement_response"].update(
                {
                    "speaking_start_wall_time_s": base_wall + 2.2,
                    "speaking_stop_wall_time_s": base_wall + 3.0,
                    "transcript_evidence": ["The answer is four."],
                    "rtvi_event_bracket": {
                        "response_id": response_ids[1],
                        "transcript_binding": RTVI_TRANSCRIPT_IDENTITY,
                        "protocol_response_identity": None,
                        "fixture_send_performance_ms": 1035.0 + fixture.plan_index * 100.0,
                        "fixture_send_wall_time_s": base_wall + 2.1,
                        "messages_received_before_send_in_bracket": 0,
                        "start_event_index": 2,
                        "terminal_event_index": 4,
                        "start_event_type": "bot-started-speaking",
                        "terminal_event_type": "bot-stopped-speaking",
                        "start_performance_ms": 1040.0 + fixture.plan_index * 100.0,
                        "terminal_performance_ms": 1050.0 + fixture.plan_index * 100.0,
                        "start_wall_time_s": base_wall + 2.2,
                        "terminal_wall_time_s": base_wall + 3.0,
                        "transcript_event_indices": [3],
                        "transcript_performance_ms": [1045.0 + fixture.plan_index * 100.0],
                        "output_text_sha256": hashlib.sha256(
                            attribution["replacement_response"]["output_text"].encode()
                        ).hexdigest(),
                    },
                }
            )
        else:
            attribution["primary_response"].update(
                {
                    "speaking_start_wall_time_s": base_wall + 1.0,
                    "speaking_stop_wall_time_s": base_wall + 3.0,
                    "transcript_evidence": ["The answer is four."],
                    "rtvi_event_bracket": {
                        "response_id": response_ids[0],
                        "transcript_binding": RTVI_TRANSCRIPT_IDENTITY,
                        "protocol_response_identity": None,
                        "fixture_send_performance_ms": 1000.0 + fixture.plan_index * 100.0,
                        "fixture_send_wall_time_s": base_wall + 0.1,
                        "messages_received_before_send_in_bracket": 0,
                        "start_event_index": 0,
                        "terminal_event_index": 2,
                        "start_event_type": "bot-started-speaking",
                        "terminal_event_type": "bot-stopped-speaking",
                        "start_performance_ms": 1010.0 + fixture.plan_index * 100.0,
                        "terminal_performance_ms": 1030.0 + fixture.plan_index * 100.0,
                        "start_wall_time_s": base_wall + 1.0,
                        "terminal_wall_time_s": base_wall + 3.0,
                        "transcript_event_indices": [1],
                        "transcript_performance_ms": [1020.0 + fixture.plan_index * 100.0],
                        "output_text_sha256": hashlib.sha256(
                            attribution["primary_response"]["output_text"].encode()
                        ).hexdigest(),
                    },
                }
            )
        results.append(result)
    paths = {
        "microphone": str(tmp_path / "mic.wav"),
        "browser_capture_on": str(tmp_path / "browser-on.jsonl"),
        "browser_capture_off": str(tmp_path / "browser-off.jsonl"),
        "summary": str(tmp_path / "summary.json"),
        "pipecat_playout_trace": str(tmp_path / "pipecat-{connection}.jsonl"),
        "pipecat_playout_capture_on": str(tmp_path / "pipecat-0001.jsonl"),
        "pipecat_playout_capture_off": str(tmp_path / "pipecat-0002.jsonl"),
        "output_dir": str(tmp_path),
    }
    browser_records = {
        mode: _summary_browser_records(results, capture_enabled=mode == "on")
        for mode in ("on", "off")
    }
    for mode in ("on", "off"):
        _write_jsonl(Path(paths[f"browser_capture_{mode}"]), browser_records[mode])
        records = [
            {
                "trace_schema": PLAYOUT_TRACE_SCHEMA,
                "type": "voicechat.playout.config",
                "client_configured_monotonic_s": 0.0,
                "configured_prebuffer_ms": 160,
                "sample_rate_hz": 22050,
                "channels": 1,
            },
            *session_records[mode],
            {
                "trace_schema": PLAYOUT_TRACE_SCHEMA,
                "type": "voicechat.playout.trace_status",
                "trace_closed_monotonic_s": 100.0,
                "valid": True,
                "invalid_reason": None,
                "error": None,
                "dropped_records": 0,
            },
        ]
        _write_jsonl(Path(paths[f"pipecat_playout_capture_{mode}"]), records)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "started_wall_time_s": 99.0,
        "completed_wall_time_s": 174.0,
        "plan": plan_document(plan),
        "resolved_paths": paths,
        "terminalization_errors": [],
        "sessions": [
            {
                "capture": mode,
                "artifact": paths[f"browser_capture_{mode}"],
                "status": "completed",
                "capture_shutdown": {
                    "shutdown_error": None,
                    "sink_close_error": None,
                    "fixture_marker_records": marker_count,
                },
                "error": None,
                "teardown_errors": [],
                "browser_artifact_validation": {
                    **validate_browser_artifact(
                        browser_records[mode],
                        capture_enabled=mode == "on",
                    )
                },
                "pipecat_artifact": paths[f"pipecat_playout_capture_{mode}"],
                "pipecat_publication_wait_s": 0.0,
                "observed_playout_trace": validate_published_playout_artifact(
                    Path(paths[f"pipecat_playout_capture_{mode}"])
                ),
            }
            for mode, marker_count in (("on", 15), ("off", 2))
        ],
        "fixtures": results,
    }


def test_canonical_plan_has_exact_order_identity_text_and_hashes() -> None:
    plan = canonical_plan()

    assert [fixture.fixture_id for fixture in plan.fixtures] == [
        "S1-1",
        "S1-2",
        "L1-1",
        "L1-2",
        "L2-1",
        "L2-2",
        "I1",
        "capture-off-L1",
    ]
    assert [fixture.script_id for fixture in plan.fixtures] == [
        "S1",
        "S1",
        "L1",
        "L1",
        "L2",
        "L2",
        "I1",
        "L1",
    ]
    assert [mode for mode, _fixtures in plan.sessions()] == ["on", "off"]
    assert len(plan.sessions()[0][1]) == 7
    assert plan.plan_sha256 == CANONICAL_PLAN_SHA256
    assert plan.script_hashes_sha256 == CANONICAL_SCRIPT_HASHES_SHA256


def test_step4c_plan_is_exact_separate_and_cannot_parse_as_step2() -> None:
    raw = step4c_plan_json()
    plan = parse_step4c_fixture_plan(raw)

    assert plan.schema == STEP4C_PLAN_SCHEMA
    assert len(plan.fixtures) == 1
    assert plan.fixtures[0].fixture_id == "step4c-L1"
    assert plan.sessions() == (("on", plan.fixtures),)
    assert plan.acoustic_input_contract == {
        "contract_id": "chrome-fake-mic-webrtc-one-lsb-v1",
        "source_pcm": "exact-zero-mono-pcm16",
        "server_ingress_max_abs_pcm16": 1,
        "known_fixture_limitation": (
            "Chrome file-backed fake microphone/WebRTC may transform exact-zero source WAV "
            "into server-ingress PCM with absolute amplitude one PCM16 LSB."
        ),
    }
    assert (plan.warmup_seconds, plan.duration_seconds, plan.drain_seconds) == (
        10.0,
        120.0,
        15.0,
    )
    with pytest.raises(ValueError, match="canonical|eight"):
        parse_fixture_plan(raw)
    with pytest.raises(ValueError, match="Step 4c"):
        parse_step4c_fixture_plan(canonical_plan_json())


def test_step4c_summary_validator_requires_exact_plan_and_free_lock() -> None:
    plan = parse_step4c_fixture_plan(step4c_plan_json())
    summary = {
        "schema": STEP4C_SUMMARY_SCHEMA,
        "status": "running",
        "started_wall_time_s": 1.0,
        "completed_wall_time_s": None,
        "plan": plan_document(plan),
        "health_immediately_before": {"active_client": False},
        "session": {"status": "running", "session_id": None},
        "fixture": blank_fixture_result(plan.fixtures[0]),
        "error": None,
    }
    validate_step4c_summary(summary)
    summary["health_immediately_before"]["active_client"] = True
    with pytest.raises(ValueError, match="single-client"):
        validate_step4c_summary(summary)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.pop(),
        lambda document: document.reverse(),
        lambda document: document[0].update(fixture_id="invented"),
        lambda document: document[0].update(script_id="L1"),
        lambda document: document[0].update(typed_text=document[0]["typed_text"] + " "),
        lambda document: document[0].update(capture="off"),
        lambda document: document[7].update(capture="on"),
        lambda document: document[6]["interruption"].update(after_first_audio_s=3.9),
        lambda document: document[6]["interruption"].update(typed_text="different"),
        lambda document: document[0]["wait"].update(timeout_s=90),
        lambda document: document[0].update(extra=True),
    ],
)
def test_plan_rejects_every_noncanonical_campaign(mutation) -> None:
    document = copy.deepcopy(CANONICAL_FIXTURE_DOCUMENT)
    mutation(document)
    with pytest.raises(ValueError, match="canonical|preregistered|hash|fields"):
        parse_fixture_plan(json.dumps(document))


def test_marker_carries_distinct_fixture_and_script_identity() -> None:
    fixture = canonical_plan().fixtures[0]
    marker = fixture_marker(
        fixture,
        "start",
        browser_performance_ms=12.0,
        browser_wall_time_s=100.0,
    )

    validate_marker(marker)
    assert marker["marker_schema"] == MARKER_SCHEMA
    assert marker["fixture_id"] == "S1-1"
    assert marker["script_id"] == "S1"


@pytest.mark.asyncio
async def test_interruption_timing_uses_first_audio_deadline_with_fake_clock() -> None:
    class FakeClock:
        now = 12.5
        sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds
            await asyncio.sleep(0)

    clock = FakeClock()
    sent_at = await wait_for_interruption_deadline(
        10.0,
        4.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert sent_at == 14.0
    assert clock.sleeps == [1.5]


def test_normal_response_lifecycle_requires_exact_attributed_audio_chain() -> None:
    attribution = correlate_response_lifecycles(
        response_records("response-1", 1.0), interruption=False
    )

    assert attribution["response_ids"] == ["response-1"]
    assert attribution["primary_response"]["audio_delta_ordinals"] == [1]
    assert attribution["primary_response"]["terminal"]["status"] == "completed"


def test_normal_response_lifecycle_accepts_post_release_immediate_pushes() -> None:
    records = response_records("response-1", 1.0)
    done = next(record for record in records if record["type"] == "response.done")
    release = next(record for record in records if record["type"] == "voicechat.playout.release")
    first_push = next(
        record for record in records if record["type"] == "voicechat.playout.downstream_push"
    )
    release.update(
        reason="threshold",
        client_release_monotonic_s=1.11,
    )
    first_push["client_downstream_push_monotonic_s"] = 1.12
    records.remove(done)
    records.remove(release)
    records.remove(first_push)
    first_delta_index = next(
        index
        for index, record in enumerate(records)
        if record["type"] == "response.output_audio.delta"
    )
    records[first_delta_index + 1 : first_delta_index + 1] = [release, first_push]
    records.extend(
        [
            trace(
                "response.output_audio.delta",
                1.2,
                response_id="response-1",
                ordinal=2,
            ),
            trace(
                "voicechat.playout.downstream_push",
                1.21,
                response_id="response-1",
                ordinal=2,
            ),
            done,
        ]
    )

    attribution = correlate_response_lifecycles(records, interruption=False)

    assert attribution["primary_response"]["release_membership"] == [
        {
            "reason": "threshold",
            "ordinals": [1],
            "frames_released": 1,
            "frames_cleared": 0,
        }
    ]
    assert attribution["primary_response"]["downstream_push_ordinals"] == [1, 2]


def test_response_lifecycle_rejects_post_release_push_after_response_done() -> None:
    records = post_release_immediate_push_records()
    push = next(
        record
        for record in records
        if record["type"] == "voicechat.playout.downstream_push" and record["ordinal"] == 3
    )
    records.remove(push)
    push["client_downstream_push_monotonic_s"] = 1.31
    records.append(push)

    with pytest.raises(ValueError, match="post-release|immediate"):
        correlate_response_lifecycles(records, interruption=False)


def test_response_lifecycle_rejects_reversed_post_release_push_order() -> None:
    records = post_release_immediate_push_records()
    done = records.pop()
    records.extend(
        [
            trace(
                "response.output_audio.delta",
                1.22,
                response_id="response-1",
                ordinal=4,
            ),
            trace(
                "voicechat.playout.downstream_push",
                1.23,
                response_id="response-1",
                ordinal=4,
            ),
        ]
    )
    ordinal_three_push = next(
        record
        for record in records
        if record["type"] == "voicechat.playout.downstream_push" and record["ordinal"] == 3
    )
    records.remove(ordinal_three_push)
    ordinal_three_push["client_downstream_push_monotonic_s"] = 1.24
    records.append(ordinal_three_push)
    records.append(done)

    with pytest.raises(ValueError, match="post-release|immediate"):
        correlate_response_lifecycles(records, interruption=False)


def test_retained_live_trace_correlates_post_release_immediate_pushes() -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T060844Z/driver/pipecat-playout-0001.jsonl"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "acea2107db42d664fb0c9997d62fc8a42998f64bd6be9e2e8c29c6af5654b23b"
    )

    attribution = correlate_response_lifecycles(read_playout_trace(artifact), interruption=False)

    assert attribution is not None
    response = attribution["primary_response"]
    assert response["audio_delta_ordinals"] == list(range(1, 68))
    assert response["release_membership"] == [
        {
            "reason": "threshold",
            "ordinals": [1, 2],
            "frames_released": 2,
            "frames_cleared": 0,
        }
    ]
    assert response["downstream_push_ordinals"] == list(range(1, 68))


def test_i1_requires_cancel_clear_replacement_and_completion_in_order() -> None:
    records = response_records(
        "response-original", 1.0, status="cancelled", clear=True
    ) + response_records("response-replacement", 2.0)

    attribution = correlate_response_lifecycles(records, interruption=True)

    assert attribution["response_ids"] == ["response-original", "response-replacement"]
    assert attribution["typed_send_edge"]["monotonic_s"] == 0.9
    assert attribution["interruption_typed_send_edge"]["monotonic_s"] == 1.9
    assert attribution["clear_edge"]["reason"] == "interruption-clear"
    assert attribution["original_response"]["terminal"]["status"] == "cancelled"
    assert attribution["replacement_response"]["terminal"]["status"] == "completed"


@pytest.mark.parametrize("interruption_send_count", [0, 2])
def test_i1_requires_exactly_one_original_and_one_interruption_typed_send(
    interruption_send_count: int,
) -> None:
    records = response_records("original", 1.0, status="cancelled", clear=True)
    replacement = response_records("replacement", 2.0)
    interruption_send = replacement.pop(0)
    records.extend([copy.deepcopy(interruption_send)] * interruption_send_count)
    records.extend(replacement)

    with pytest.raises(ValueError, match="two typed-send edges"):
        correlate_response_lifecycles(records, interruption=True)


def test_ordinary_fixture_still_requires_exactly_one_typed_send() -> None:
    records = response_records("response-1", 1.0)
    records.insert(
        1,
        trace(
            "input_audio_buffer.speech_started",
            0.95,
            source="typed",
            response_id=None,
        ),
    )

    with pytest.raises(ValueError, match="one typed-send edge"):
        correlate_response_lifecycles(records, interruption=False)


def test_attempt_4_retained_i1_window_owns_both_typed_sends_and_correlates() -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T070325Z/driver/pipecat-playout-0001.jsonl"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "820e15a45992353c58228efc3cb4df588f8642cf7b40f08ad523425000997e3e"
    )
    records = read_playout_trace(artifact)
    capture_on_fixtures = canonical_plan().sessions()[0][1]
    partitions = driver._partition_session_fixture_records(records, capture_on_fixtures)
    typed_send_counts = [
        sum(
            record.get("type") == "input_audio_buffer.speech_started"
            and record.get("source") == "typed"
            and record.get("response_id") is None
            for record in partition
        )
        for partition in partitions
    ]

    assert typed_send_counts == [1, 1, 1, 1, 1, 1, 2]
    i1_records = list(partitions[-1])
    assert correlate_response_lifecycles(i1_records, interruption=True) is None

    replacement_id = next(
        record["response_id"]
        for record in reversed(i1_records)
        if record.get("type") == "response.created"
    )
    last_lifecycle_clock = max(
        driver._clock(record)
        for record in i1_records
        if record.get("type")
        in {
            "response.created",
            "response.output_text.delta",
            "response.output_audio.delta",
            "response.done",
            "input_audio_buffer.speech_started",
            "voicechat.playout.interruption",
            "voicechat.playout.release",
            "voicechat.playout.downstream_push",
        }
    )
    completed_records = list(records)
    completed_records.append(
        trace(
            "response.done",
            last_lifecycle_clock + 0.001,
            response_id=replacement_id,
            status="completed",
        )
    )

    reconstructed = driver.reconstruct_session_response_attributions(
        completed_records, capture_on_fixtures
    )
    attribution = reconstructed[-1]

    assert len(reconstructed) == 7
    assert attribution["response_ids"][-1] == replacement_id
    assert attribution["interruption_typed_send_edge"]["monotonic_s"] == pytest.approx(
        4798052.760756755
    )
    assert attribution["original_response"]["terminal"]["status"] == "cancelled"
    assert attribution["replacement_response"]["terminal"]["status"] == "completed"


def test_attempt_6_retained_i1_accepts_cancel_before_late_typed_send_echo() -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T073549Z/driver/pipecat-playout-0001.jsonl"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "e67e950029c9cb9a3614c94805f67d4374bdebdbdb33056a3952fbadfb1ff748"
    )
    records = read_playout_trace(artifact)
    capture_on_fixtures = canonical_plan().sessions()[0][1]
    i1_records = driver._partition_session_fixture_records(records, capture_on_fixtures)[-1]

    attribution = correlate_response_lifecycles(i1_records, interruption=True)

    assert attribution is not None
    original = attribution["original_response"]
    assert attribution["clear_edge"]["monotonic_s"] == pytest.approx(4799978.482572568)
    assert original["terminal"] == {
        "status": "cancelled",
        "monotonic_s": pytest.approx(4799980.002218832),
    }
    assert attribution["interruption_typed_send_edge"]["monotonic_s"] == pytest.approx(
        4799980.002479973
    )
    assert attribution["replacement_response"]["created_monotonic_s"] == pytest.approx(
        4799982.679757086
    )
    assert (
        attribution["clear_edge"]["monotonic_s"]
        < original["terminal"]["monotonic_s"]
        < attribution["interruption_typed_send_edge"]["monotonic_s"]
        < attribution["replacement_response"]["created_monotonic_s"]
    )


def test_capture_off_partition_retains_its_single_non_i1_send_rule() -> None:
    fixture = canonical_plan().sessions()[1][1][0]
    records = response_records("capture-off-response", 1.0)

    partitions = driver._partition_session_fixture_records(records, [fixture])
    attribution = correlate_response_lifecycles(partitions[0], interruption=False)

    assert len(partitions) == 1
    assert attribution is not None
    assert attribution["response_ids"] == ["capture-off-response"]


@pytest.mark.parametrize("probe", ["audio-after-edge", "push-after-clear"])
def test_i1_rejects_original_audio_that_violates_the_interruption_clear_order(
    probe: str,
) -> None:
    records = response_records("original", 1.0, status="cancelled", clear=True)
    records += response_records("replacement", 2.0)
    edge = next(
        record
        for record in records
        if record["type"] == "input_audio_buffer.speech_started"
        and record.get("response_id") == "original"
    )
    if probe == "audio-after-edge":
        edge["client_received_monotonic_s"] = 1.05
        records.remove(edge)
        created_index = next(
            index
            for index, record in enumerate(records)
            if record["type"] == "response.created" and record["response_id"] == "original"
        )
        records.insert(created_index + 1, edge)
    else:
        push = next(
            record
            for record in records
            if record["type"] == "voicechat.playout.downstream_push"
            and record["response_id"] == "original"
        )
        push["client_downstream_push_monotonic_s"] = 1.22
        records.remove(push)
        clear_index = next(
            index
            for index, record in enumerate(records)
            if record.get("reason") == "interruption-clear"
        )
        records.insert(clear_index + 1, push)

    with pytest.raises(ValueError, match="original audio|downstream push"):
        correlate_response_lifecycles(records, interruption=True)


def test_response_lifecycle_rejects_never_buffered_pre_release_push() -> None:
    records = response_records("response-1", 1.0)
    delta = trace(
        "response.output_audio.delta",
        1.2,
        response_id="response-1",
        ordinal=2,
    )
    push = trace(
        "voicechat.playout.downstream_push",
        1.21,
        response_id="response-1",
        ordinal=2,
    )
    terminal_index = next(
        index for index, record in enumerate(records) if record["type"] == "response.done"
    )
    records[terminal_index:terminal_index] = [delta, push]

    with pytest.raises(ValueError, match="playout started|without release"):
        correlate_response_lifecycles(records, interruption=False)


def test_response_lifecycle_rejects_post_clear_push() -> None:
    records = response_records("original", 1.0, status="cancelled", clear=True)
    records += response_records("replacement", 2.0)
    edge_index = next(
        index
        for index, record in enumerate(records)
        if record["type"] == "input_audio_buffer.speech_started"
        and record.get("response_id") == "original"
    )
    records.insert(
        edge_index,
        trace(
            "response.output_audio.delta",
            1.19,
            response_id="original",
            ordinal=3,
        ),
    )
    clear = next(record for record in records if record.get("reason") == "interruption-clear")
    clear.update(ordinals=[3], frames_cleared=1)
    clear_index = records.index(clear)
    records.insert(
        clear_index + 1,
        trace(
            "voicechat.playout.downstream_push",
            1.22,
            response_id="original",
            ordinal=3,
        ),
    )

    with pytest.raises(ValueError, match="clears|cleared|clear"):
        correlate_response_lifecycles(records, interruption=True)


def test_response_lifecycle_rejects_push_before_its_delta() -> None:
    records = response_records("response-1", 1.0)
    push = next(
        record for record in records if record["type"] == "voicechat.playout.downstream_push"
    )
    records.remove(push)
    records.insert(2, push)
    push["client_downstream_push_monotonic_s"] = 1.01

    with pytest.raises(ValueError, match="release|delta|push"):
        correlate_response_lifecycles(records, interruption=False)


@pytest.mark.parametrize("probe", ["stale", "extra", "missing-clear", "wrong-terminal", "order"])
def test_response_lifecycle_rejects_cross_fixture_and_i1_transition_faults(probe: str) -> None:
    records = response_records("original", 1.0, status="cancelled", clear=True)
    records += response_records("replacement", 2.0)
    if probe == "stale":
        records.insert(0, trace("response.done", 0.9, response_id="stale", status="completed"))
    elif probe == "extra":
        records += response_records("extra", 3.0, include_typed_send=False)
    elif probe == "missing-clear":
        records = [record for record in records if record.get("reason") != "interruption-clear"]
    elif probe == "wrong-terminal":
        next(record for record in records if record["type"] == "response.done")["status"] = (
            "completed"
        )
    else:
        replacement = next(
            record
            for record in records
            if record["type"] == "response.created" and record["response_id"] == "replacement"
        )
        records.remove(replacement)
        records.insert(3, replacement)
    with pytest.raises(ValueError):
        correlate_response_lifecycles(records, interruption=True)


def test_strict_browser_union_accepts_mixed_capture_on_artifact() -> None:
    records = [
        browser_record("browser.playout.capability", 1.0, audio_worklet=True),
        browser_record(
            "browser.clock_mapping",
            1.1,
            phase="startup",
            main_thread_performance_ms=1.0,
            audio_context_time_s=0.0,
            output_context_time_s=None,
            output_performance_time_ms=None,
        ),
        browser_record("browser.audio_worklet.registered", 1.2, worklet_id="worklet-1"),
        browser_record(
            "browser.audio_worklet.dequeue",
            1.3,
            worklet_id="worklet-1",
            main_thread_received_performance_ms=3.0,
            render_audio_context_time_s=0.1,
            render_audio_context_frame=128,
            ordinal=1,
            sample_count=128,
            channel_count=1,
            rms=0.1,
            underrun=False,
            underrun_count=0,
        ),
        browser_record(
            "browser.audio_worklet.onset",
            1.3,
            worklet_id="worklet-1",
            main_thread_received_performance_ms=3.0,
            render_audio_context_time_s=0.1,
            render_audio_context_frame=128,
            ordinal=1,
            sample_count=128,
            channel_count=1,
            rms=0.1,
            underrun=False,
            underrun_count=0,
        ),
        *[
            {
                **record,
                "playwright_batch_received_monotonic_s": 1.4 + index * 0.01,
                "playwright_batch_received_wall_time_s": 101.4 + index * 0.01,
            }
            for index, record in enumerate(
                [
                    item
                    for item in _canonical_browser_records(capture_enabled=True)
                    if item["type"] == "browser.fixture_boundary"
                ]
            )
        ],
        browser_record(
            "browser.audio_worklet.flush_ack",
            2.0,
            worklet_id="worklet-1",
            final_ordinal=1,
            delivered_last_ordinal=1,
        ),
    ]

    validation = validate_browser_artifact(records, capture_enabled=True)

    assert validation["valid"] is True
    assert validation["fixture_marker_records"] == 15
    assert validation["persisted_last_ordinals"] == {"worklet-1": 1}


@pytest.mark.parametrize("probe", ["unknown", "clock-regression", "ordinal-gap", "ack-mismatch"])
def test_strict_browser_union_rejects_invalid_whole_artifact(probe: str) -> None:
    records = [
        browser_record("browser.playout.capability", 1.0, audio_worklet=True),
        browser_record("browser.audio_worklet.registered", 1.1, worklet_id="w"),
        browser_record(
            "browser.audio_worklet.dequeue",
            1.2,
            worklet_id="w",
            main_thread_received_performance_ms=3.0,
            render_audio_context_time_s=0.1,
            render_audio_context_frame=128,
            ordinal=1,
            sample_count=128,
            channel_count=1,
            rms=0.1,
            underrun=False,
            underrun_count=0,
        ),
        browser_record(
            "browser.audio_worklet.flush_ack",
            1.3,
            worklet_id="w",
            final_ordinal=1,
            delivered_last_ordinal=1,
        ),
    ]
    if probe == "unknown":
        records[0]["type"] = "browser.unknown"
    elif probe == "clock-regression":
        records[2]["playwright_batch_received_monotonic_s"] = 0.5
    elif probe == "ordinal-gap":
        records[2]["ordinal"] = 2
    else:
        records[-1]["final_ordinal"] = 2
    with pytest.raises(ValueError):
        validate_browser_artifact(records, capture_enabled=True)


def test_browser_consumer_rejects_canonical_start_only_fixture_markers() -> None:
    records = _canonical_browser_records(capture_enabled=True)
    records = [
        record
        for record in records
        if record.get("type") != "browser.fixture_boundary" or record.get("boundary") == "start"
    ]
    with pytest.raises(ValueError, match="exact canonical fixture marker sequence"):
        validate_browser_artifact(records, capture_enabled=True)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda summary: summary["sessions"][0].update(status="failed"), "session"),
        (lambda summary: summary["sessions"][0].update(error="boom"), "session"),
        (
            lambda summary: summary["sessions"][0].update(pipecat_publication_wait_s=None),
            "publication wait",
        ),
        (lambda summary: summary["sessions"][0].update(capture_shutdown=None), "shutdown"),
        (
            lambda summary: summary["sessions"][0]["capture_shutdown"].update(
                shutdown_error="boom"
            ),
            "shutdown",
        ),
        (
            lambda summary: summary["sessions"][0]["capture_shutdown"].update(
                fixture_marker_records=14
            ),
            "marker",
        ),
        (lambda summary: summary["fixtures"][0].update(response_ids=[]), "response-ID"),
        (lambda summary: summary["fixtures"][0].update(error="boom"), "error"),
        (
            lambda summary: summary["fixtures"][0]["marker_emission"]["complete"].update(
                status="failed", error="boom"
            ),
            "marker",
        ),
        (
            lambda summary: summary["fixtures"][6]["interruption"].update(send_wall_time_s=None),
            "interruption send",
        ),
        (
            lambda summary: summary["fixtures"][6]["interruption"].update(actual_offset_s=4.5),
            "offset",
        ),
        (
            lambda summary: summary["fixtures"][6]["interruption"].update(
                second_answer_gate_evidence=None
            ),
            "second-answer gate evidence",
        ),
        (
            lambda summary: summary["terminalization_errors"].append("exit failed"),
            "terminalization",
        ),
    ],
)
def test_completed_summary_cross_field_validation_is_fail_closed(
    tmp_path: Path, mutation, match: str
) -> None:
    summary = completed_summary(tmp_path)
    mutation(summary)
    with pytest.raises(ValueError, match=match):
        validate_fixture_summary(summary)


def test_completed_summary_passes_strict_cross_field_validation(tmp_path: Path) -> None:
    validate_fixture_summary(completed_summary(tmp_path))


@pytest.mark.parametrize("adjudicated_text", ["The answer is five.", "4"])
def test_completed_i1_accepts_consistent_false_second_answer_adjudication(
    tmp_path: Path, adjudicated_text: str
) -> None:
    summary = completed_summary(tmp_path)
    fixture = summary["fixtures"][6]
    replacement = fixture["response_attribution"]["replacement_response"]
    replacement["output_text"] = adjudicated_text
    replacement["rtvi_event_bracket"]["output_text_sha256"] = hashlib.sha256(
        adjudicated_text.encode()
    ).hexdigest()
    fixture["transcript_gate_evidence"] = driver.transcript_gate_evidence(
        fixture["transcript_evidence"], adjudicated_text
    )
    fixture["interruption"]["second_answer_gate_evidence"] = driver.second_answer_gate_evidence(
        fixture["interruption"]["second_answer_transcript_evidence"],
        adjudicated_text,
    )

    pipecat_path = Path(summary["sessions"][0]["pipecat_artifact"])
    records = read_playout_trace(pipecat_path)
    replacement_id = replacement["response_id"]
    for record in records:
        if (
            record.get("type") == "response.output_text.delta"
            and record.get("response_id") == replacement_id
        ):
            record["delta"] = adjudicated_text
    _write_jsonl(pipecat_path, records)
    summary["sessions"][0]["observed_playout_trace"] = validate_published_playout_artifact(
        pipecat_path
    )

    validate_fixture_summary(summary)


def test_completed_i1_accepts_empty_browser_transcript_with_recorded_adjudication(
    tmp_path: Path,
) -> None:
    summary = completed_summary(tmp_path)
    fixture = summary["fixtures"][6]
    replacement = fixture["response_attribution"]["replacement_response"]
    fixture["transcript_evidence"] = []
    replacement["transcript_evidence"] = []
    replacement["rtvi_event_bracket"]["transcript_event_indices"] = []
    replacement["rtvi_event_bracket"]["transcript_performance_ms"] = []
    fixture["transcript_gate_evidence"] = driver.transcript_gate_evidence(
        [], replacement["output_text"]
    )
    fixture["interruption"]["second_answer_transcript_evidence"] = []
    fixture["interruption"]["second_answer_gate_evidence"] = driver.second_answer_gate_evidence(
        [], replacement["output_text"]
    )

    validate_fixture_summary(summary)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.update(joined_browser_transcript="forged"),
        lambda evidence: evidence.update(response_owned_traced_text="forged"),
        lambda evidence: evidence.update(second_answer_adjudication=False),
        lambda evidence: evidence.update(second_answer_adjudication=1),
        lambda evidence: evidence.pop("second_answer_adjudication"),
    ],
)
def test_completed_summary_rejects_inconsistent_second_answer_gate_evidence(
    tmp_path: Path, mutation
) -> None:
    summary = completed_summary(tmp_path)
    mutation(summary["fixtures"][6]["interruption"]["second_answer_gate_evidence"])

    with pytest.raises(ValueError, match="second-answer gate evidence"):
        validate_fixture_summary(summary)


def test_completed_summary_accepts_present_consistent_transcript_mismatch_evidence(
    tmp_path: Path,
) -> None:
    summary = completed_summary(tmp_path)
    fixture = summary["fixtures"][0]
    response = fixture["response_attribution"]["primary_response"]
    fixture["transcript_evidence"] = []
    response["transcript_evidence"] = []
    response["rtvi_event_bracket"]["transcript_event_indices"] = []
    response["rtvi_event_bracket"]["transcript_performance_ms"] = []
    fixture["transcript_gate_evidence"] = driver.transcript_gate_evidence(
        [], response["output_text"]
    )

    validate_fixture_summary(summary)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda fixture: fixture.update(transcript_gate_evidence=None),
        lambda fixture: fixture["transcript_gate_evidence"].update(transcript_equal=False),
        lambda fixture: fixture["transcript_gate_evidence"].update(transcript_equal=1),
        lambda fixture: fixture["transcript_gate_evidence"].update(
            mismatch_reason="invented mismatch"
        ),
        lambda fixture: fixture["transcript_gate_evidence"].update(
            joined_browser_transcript="forged browser transcript"
        ),
        lambda fixture: fixture["transcript_gate_evidence"].update(
            response_owned_traced_text="forged traced text"
        ),
    ],
)
def test_completed_summary_rejects_missing_or_inconsistent_transcript_gate_evidence(
    tmp_path: Path, mutation
) -> None:
    summary = completed_summary(tmp_path)
    mutation(summary["fixtures"][0])

    with pytest.raises(ValueError, match="transcript gate evidence"):
        validate_fixture_summary(summary)


def test_retained_immediate_push_trace_passes_completed_summary_validation(
    tmp_path: Path,
) -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T060844Z/driver/pipecat-playout-0001.jsonl"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "acea2107db42d664fb0c9997d62fc8a42998f64bd6be9e2e8c29c6af5654b23b"
    )
    retained_records = read_playout_trace(artifact)
    retained_attribution = correlate_response_lifecycles(retained_records, interruption=False)
    assert retained_attribution is not None
    assert retained_attribution["primary_response"]["audio_delta_ordinals"] == list(range(1, 68))

    relevant_types = {
        "response.created",
        "response.output_text.delta",
        "response.output_audio.delta",
        "response.done",
        "input_audio_buffer.speech_started",
        "voicechat.playout.interruption",
        "voicechat.playout.release",
        "voicechat.playout.downstream_push",
    }
    retained_fixture_records = [
        copy.deepcopy(record) for record in retained_records if record.get("type") in relevant_types
    ]
    retained_start = float(retained_fixture_records[0]["client_received_monotonic_s"])
    clock_shift = retained_start - 0.9
    for record in retained_fixture_records:
        for key, value in record.items():
            if key.endswith("_monotonic_s") and isinstance(value, (int, float)):
                record[key] = float(value) - clock_shift
    summary_attribution = correlate_response_lifecycles(
        retained_fixture_records, interruption=False
    )
    assert summary_attribution is not None

    summary = completed_summary(tmp_path)
    fixture = summary["fixtures"][0]
    old_response = fixture["response_attribution"]["primary_response"]
    response = summary_attribution["primary_response"]
    transcript = response["output_text"]
    bracket = copy.deepcopy(old_response["rtvi_event_bracket"])
    bracket["response_id"] = response["response_id"]
    bracket["output_text_sha256"] = hashlib.sha256(transcript.encode()).hexdigest()
    response.update(
        speaking_start_wall_time_s=old_response["speaking_start_wall_time_s"],
        speaking_stop_wall_time_s=old_response["speaking_stop_wall_time_s"],
        transcript_evidence=[transcript],
        rtvi_event_bracket=bracket,
    )
    fixture["response_ids"] = summary_attribution["response_ids"]
    fixture["response_attribution"] = summary_attribution
    fixture["transcript_evidence"] = [transcript]
    fixture["transcript_gate_evidence"] = driver.transcript_gate_evidence([transcript], transcript)

    pipecat_path = Path(summary["sessions"][0]["pipecat_artifact"])
    session_records = read_playout_trace(pipecat_path)
    fixture_starts = [
        index
        for index, record in enumerate(session_records)
        if record.get("type") == "input_audio_buffer.speech_started"
        and record.get("source") == "typed"
        and record.get("response_id") is None
    ]
    session_records[fixture_starts[0] : fixture_starts[1]] = retained_fixture_records
    _write_jsonl(pipecat_path, session_records)
    summary["sessions"][0]["observed_playout_trace"] = validate_published_playout_artifact(
        pipecat_path
    )

    validate_fixture_summary(summary)


@pytest.mark.parametrize("quiet_window_ms", [0.0, 499.0, -1.0, float("nan")])
def test_completed_summary_rejects_invalid_quiescence_window(
    tmp_path: Path, quiet_window_ms: float
) -> None:
    summary = completed_summary(tmp_path)
    summary["fixtures"][0]["quiescence_barrier"]["quiet_window_ms"] = quiet_window_ms

    with pytest.raises(ValueError, match="RTVI quiescence window"):
        validate_fixture_summary(summary)


def test_completed_summary_rejects_wall_clock_frozen_quiescence_window(
    tmp_path: Path,
) -> None:
    summary = completed_summary(tmp_path)
    barrier = summary["fixtures"][0]["quiescence_barrier"]
    barrier["window_end_wall_time_s"] = barrier["window_start_wall_time_s"]

    with pytest.raises(ValueError, match="quiet-window evidence"):
        validate_fixture_summary(summary)


def test_completed_summary_rejects_wall_duration_shorter_than_clock_tolerance(
    tmp_path: Path,
) -> None:
    summary = completed_summary(tmp_path)
    barrier = summary["fixtures"][0]["quiescence_barrier"]
    barrier["window_end_wall_time_s"] = (
        barrier["window_start_wall_time_s"]
        + barrier["quiet_window_ms"] / 1000.0
        - RTVI_WALL_CLOCK_SKEW_TOLERANCE_S
        - 0.000001
    )

    with pytest.raises(ValueError, match="quiet-window evidence"):
        validate_fixture_summary(summary)


def test_completed_summary_allows_one_wall_clock_quantization_tick(tmp_path: Path) -> None:
    summary = completed_summary(tmp_path)
    barrier = summary["fixtures"][0]["quiescence_barrier"]
    barrier["window_end_wall_time_s"] = (
        barrier["window_start_wall_time_s"]
        + barrier["quiet_window_ms"] / 1000.0
        - RTVI_WALL_CLOCK_SKEW_TOLERANCE_S
    )

    validate_fixture_summary(summary)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda summary: summary["plan"]["fixtures"][0].update(
                typed_text=summary["plan"]["fixtures"][0]["typed_text"] + "!"
            ),
            "hash|canonical",
        ),
        (
            lambda summary: summary["fixtures"][0]["response_attribution"]["primary_response"][
                "terminal"
            ].update(monotonic_s=0.4),
            "terminal precedes",
        ),
        (
            lambda summary: summary["fixtures"][6]["interruption"].update(
                typed_text="Stop. What is two plus three?"
            ),
            "replacement bytes",
        ),
        (
            lambda summary: summary["fixtures"][1].update(send_attempt_wall_time_s=103.0),
            "overlap",
        ),
        (
            lambda summary: summary["sessions"][0].update(
                browser_artifact_validation={"valid": True, "fixture_marker_records": 15}
            ),
            "full checked browser",
        ),
    ],
)
def test_completed_summary_rejects_round2_accepted_mutations(
    tmp_path: Path, mutation, match: str
) -> None:
    summary = completed_summary(tmp_path)
    mutation(summary)
    with pytest.raises(ValueError, match=match):
        validate_fixture_summary(summary)


@pytest.mark.parametrize("field", ["top-level", "nested"])
def test_completed_summary_rejects_redundant_transcript_mutations(
    tmp_path: Path, field: str
) -> None:
    summary = completed_summary(tmp_path)
    if field == "top-level":
        summary["fixtures"][0]["transcript_evidence"] = ["forged top-level text"]
    else:
        summary["fixtures"][0]["response_attribution"]["primary_response"][
            "transcript_evidence"
        ] = ["forged nested text"]
    with pytest.raises(ValueError, match="transcript"):
        validate_fixture_summary(summary)


def test_completed_summary_rejects_campaign_response_id_reuse(tmp_path: Path) -> None:
    summary = completed_summary(tmp_path)
    reused = summary["fixtures"][0]["response_ids"][0]
    fixture = summary["fixtures"][1]
    fixture["response_ids"] = [reused]
    fixture["response_attribution"]["response_ids"] = [reused]
    fixture["response_attribution"]["primary_response"]["response_id"] = reused
    fixture["response_attribution"]["primary_response"]["rtvi_event_bracket"]["response_id"] = (
        reused
    )

    with pytest.raises(ValueError, match="reuse|campaign"):
        validate_fixture_summary(summary)


def test_completed_summary_rejects_invented_response_membership(tmp_path: Path) -> None:
    summary = completed_summary(tmp_path)
    response = summary["fixtures"][0]["response_attribution"]["primary_response"]
    response["audio_delta_ordinals"] = [10]
    response["release_reasons"] = ["invented"]
    response["release_membership"] = [
        {
            "reason": "invented",
            "ordinals": [10],
            "frames_released": 1,
            "frames_cleared": 0,
        }
    ]
    response["downstream_push_ordinals"] = [10]

    with pytest.raises(ValueError, match="membership|artifact"):
        validate_fixture_summary(summary)


def test_completed_summary_rejects_markers_outside_fixture_lifecycles(tmp_path: Path) -> None:
    summary = completed_summary(tmp_path)
    artifact = Path(summary["sessions"][0]["artifact"])
    records = [json.loads(line) for line in artifact.read_text().splitlines()]
    for record in records:
        if record.get("type") == "browser.fixture_boundary":
            record["browser_performance_ms"] = 1.0
            record["browser_wall_time_s"] = 1.0
            record["playwright_batch_received_wall_time_s"] = 1.0
    _write_jsonl(artifact, records)
    summary["sessions"][0]["browser_artifact_validation"] = validate_browser_artifact(
        records, capture_enabled=True
    )

    with pytest.raises(ValueError, match="marker|bracket"):
        validate_fixture_summary(summary)


def test_path_resolution_rejects_collision_symlink_existing_output_and_rerun(tmp_path) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    output = tmp_path / "session"
    on = output / "on.jsonl"
    off = output / "off.jsonl"
    summary = output / "summary.json"
    pipecat = output / "pipecat-{connection}.jsonl"

    resolved = resolve_step2_paths(
        microphone=microphone,
        capture_on=on,
        capture_off=off,
        summary=summary,
        pipecat_trace=pipecat,
        output_dir=output,
    )
    assert all(Path(path).is_absolute() for path in resolved.values())
    with pytest.raises(FileExistsError, match="reused"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=on,
            capture_off=off,
            summary=summary,
            pipecat_trace=pipecat,
            output_dir=output,
        )

    collision_output = tmp_path / "collision"
    with pytest.raises(ValueError, match="collide"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=collision_output / "same.jsonl",
            capture_off=collision_output / "off.jsonl",
            summary=collision_output / "same.jsonl",
            pipecat_trace=collision_output / "pipecat-{connection}.jsonl",
            output_dir=collision_output,
        )

    symlink_output = tmp_path / "symlink"
    symlink_output.mkdir()
    target = symlink_output / "target.jsonl"
    alias = symlink_output / "alias.jsonl"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=target,
            capture_off=symlink_output / "off.jsonl",
            summary=alias,
            pipecat_trace=symlink_output / "pipecat-{connection}.jsonl",
            output_dir=symlink_output,
        )

    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    existing_on = existing_output / "on.jsonl"
    existing_on.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=existing_on,
            capture_off=existing_output / "off.jsonl",
            summary=existing_output / "summary.json",
            pipecat_trace=existing_output / "pipecat-{connection}.jsonl",
            output_dir=existing_output,
        )


def test_path_resolution_rejects_symlinked_session_directory_before_mutation(tmp_path) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    target = tmp_path / "real-session"
    target.mkdir()
    session = tmp_path / "session-link"
    session.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=session / "on.jsonl",
            capture_off=session / "off.jsonl",
            summary=session / "summary.json",
            pipecat_trace=session / "pipecat-{connection}.jsonl",
            output_dir=session,
        )

    assert not (target / ".step2-live-fixture-session").exists()


def test_path_resolution_requires_connection_token_before_session_mutation(tmp_path) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    session = tmp_path / "missing-token-session"

    with pytest.raises(ValueError, match=r"literal \{connection\} token"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=session / "on.jsonl",
            capture_off=session / "off.jsonl",
            summary=session / "summary.json",
            pipecat_trace=session / "pipecat.jsonl",
            output_dir=session,
        )

    assert not session.exists()


@pytest.mark.parametrize("race", ["before-resolution", "before-reservation"])
def test_path_resolution_rejects_deterministic_session_symlink_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    session = tmp_path / "session"
    session.mkdir()
    displaced = tmp_path / "displaced-session"
    target = tmp_path / "race-target"
    target.mkdir()

    def swap() -> None:
        session.rename(displaced)
        session.symlink_to(target, target_is_directory=True)

    if race == "before-resolution":
        original_absolute = driver._absolute_unresolved
        calls = 0

        def racing_absolute(path: Path) -> Path:
            nonlocal calls
            calls += 1
            resolved = original_absolute(path)
            if calls == 6:
                swap()
            return resolved

        monkeypatch.setattr(driver, "_absolute_unresolved", racing_absolute)
    else:
        original_reserve = driver._reserve_session_directory

        def racing_reserve(path: Path) -> int:
            swap()
            return original_reserve(path)

        monkeypatch.setattr(driver, "_reserve_session_directory", racing_reserve)

    with pytest.raises(ValueError, match="symlink|real directory"):
        resolve_step2_paths(
            microphone=microphone,
            capture_on=session / "on.jsonl",
            capture_off=session / "off.jsonl",
            summary=session / "summary.json",
            pipecat_trace=session / "playout-{connection}.jsonl",
            output_dir=session,
        )
    assert not (target / ".step2-live-fixture-session").exists()


def test_path_reservation_retains_opened_directory_across_post_open_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    session = tmp_path / "session"
    session.mkdir()
    displaced = tmp_path / "displaced-session"
    target = tmp_path / "race-target"
    target.mkdir()
    original_reserve = driver._reserve_session_directory

    def racing_reserve(path: Path) -> int:
        descriptor = original_reserve(path)
        session.rename(displaced)
        session.symlink_to(target, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(driver, "_reserve_session_directory", racing_reserve)
    reservation = resolve_step2_paths(
        microphone=microphone,
        capture_on=session / "on.jsonl",
        capture_off=session / "off.jsonl",
        summary=session / "summary.json",
        pipecat_trace=session / "playout-{connection}.jsonl",
        output_dir=session,
    )
    try:
        assert (displaced / ".step2-live-fixture-session").is_file()
        assert list(target.iterdir()) == []
        with pytest.raises(ValueError, match="no longer names|identity changed"):
            read_playout_trace(
                Path(reservation["pipecat_playout_capture_on"]),
                reservation=reservation,
            )
    finally:
        reservation.close()


def test_two_session_connection_template_uses_real_writer_publication_lifecycle(
    tmp_path: Path,
) -> None:
    template = tmp_path / "playout-{connection}.jsonl"
    code = """
import json
import os
from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService

statuses = []
for _ in range(2):
    service = NemotronVoicechatLLMService()
    statuses.append(service._playout_trace.close(timeout=2))
print(json.dumps(statuses))
"""
    environment = os.environ.copy()
    environment["NEMOTRON_VOICECHAT_PLAYOUT_TRACE"] = str(template)
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    statuses = json.loads(process.stdout.splitlines()[-1])
    assert [status["artifact_published"] for status in statuses] == [True, True]
    expected = [connection_playout_trace_path(template, ordinal) for ordinal in (1, 2)]
    assert sorted(tmp_path.glob("playout-*.jsonl")) == expected
    assert validate_connection_playout_trace_set(template) == tuple(expected)
    validations = [
        asyncio.run(wait_for_playout_trace_publication(path, timeout_s=1)) for path in expected
    ]
    assert [validation["valid"] for validation in validations] == [True, True]
    assert [validation["terminal_status"]["dropped_records"] for validation in validations] == [
        0,
        0,
    ]


def test_publication_wait_covers_late_fake_writer_and_fails_closed_past_bound(
    tmp_path: Path,
) -> None:
    class DelayedPublishWriter:
        def __init__(self, path: Path, *, publish_after_s: float):
            self.path = path
            self.publish_after_s = publish_after_s
            self.now = 0.0
            self.sleeps: list[float] = []
            self.published = False
            self.pending = path.with_name(f".{path.name}.fake.pending")
            self.pending.write_text("staged\n", encoding="utf-8")

        async def sleep(self, delay_s: float) -> None:
            self.sleeps.append(delay_s)
            self.now += delay_s
            if not self.published and self.now >= self.publish_after_s:
                self.pending.unlink()
                _write_playout_artifact(self.path, self.now)
                self.published = True

    assert PLAYOUT_TRACE_PUBLICATION_TIMEOUT_S == 90.0
    assert PLAYOUT_TRACE_PUBLICATION_POLL_S == 1.0

    late_path = tmp_path / "late-pipecat.jsonl"
    late_writer = DelayedPublishWriter(late_path, publish_after_s=24.0)
    validation = asyncio.run(
        wait_for_playout_trace_publication(
            late_path,
            monotonic=lambda: late_writer.now,
            sleep=late_writer.sleep,
        )
    )

    assert validation["valid"] is True
    assert late_writer.now == 24.0
    assert late_writer.sleeps == [1.0] * 24

    too_late_path = tmp_path / "too-late-pipecat.jsonl"
    too_late_writer = DelayedPublishWriter(too_late_path, publish_after_s=91.0)
    with pytest.raises(
        TimeoutError,
        match=r"published=False pending_staging_records=1",
    ):
        asyncio.run(
            wait_for_playout_trace_publication(
                too_late_path,
                monotonic=lambda: too_late_writer.now,
                sleep=too_late_writer.sleep,
            )
        )

    assert too_late_writer.published is False
    assert too_late_writer.now == 90.0
    assert too_late_writer.sleeps == [1.0] * 90


def test_terminal_trace_set_rejects_real_live_connection_three(tmp_path: Path) -> None:
    template = tmp_path / "playout-{connection}.jsonl"
    code = """
import json
from pathlib import Path
from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService
from step2_live_fixture_driver import validate_connection_playout_trace_set

services = [NemotronVoicechatLLMService() for _ in range(3)]
services[0]._playout_trace.close(timeout=2)
services[1]._playout_trace.close(timeout=2)
rejected = False
try:
    validate_connection_playout_trace_set(Path(__import__('os').environ['NEMOTRON_VOICECHAT_PLAYOUT_TRACE']))
except ValueError:
    rejected = True
finally:
    services[2]._playout_trace.close(timeout=2)
print(json.dumps({'rejected': rejected}))
"""
    environment = os.environ.copy()
    environment["NEMOTRON_VOICECHAT_PLAYOUT_TRACE"] = str(template)
    environment["PYTHONPATH"] = str(QUALIFICATION_TOOLS)
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(process.stdout.splitlines()[-1]) == {"rejected": True}


@pytest.mark.parametrize("kind", ["published", "pending", "staged", "quarantined"])
def test_terminal_trace_set_rejects_every_third_connection_namespace(
    tmp_path: Path, kind: str
) -> None:
    template = tmp_path / "playout-{connection}.jsonl"
    for ordinal in (1, 2):
        _write_playout_artifact(connection_playout_trace_path(template, ordinal), float(ordinal))
    third = connection_playout_trace_path(template, 3)
    if kind == "published":
        path = third
    elif kind == "quarantined":
        quarantine = tmp_path / ".voicechat-playout-trace-quarantine"
        quarantine.mkdir()
        path = quarantine / f".{third.name}.123.token.pending.orphan"
    else:
        path = tmp_path / f".{third.name}.123.token.{kind}"
    path.write_text("trace", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the closed"):
        validate_connection_playout_trace_set(template)


def test_live_trace_reader_prefers_single_staging_artifact_and_ignores_partial_tail(
    tmp_path,
) -> None:
    published = tmp_path / "pipecat.jsonl"
    published.write_text('{"trace_schema":"old"}\n', encoding="utf-8")
    pending = tmp_path / ".pipecat.jsonl.123.token.pending"
    record = {
        "trace_schema": PLAYOUT_TRACE_SCHEMA,
        "type": "response.created",
        "client_received_monotonic_s": 1.0,
        "response_id": "response-1",
        "turn_id": None,
    }
    pending.write_text(json.dumps(record) + "\n{", encoding="utf-8")

    assert active_playout_trace_path(published) == pending
    assert read_playout_trace(published) == [record]

    second = tmp_path / ".pipecat.jsonl.456.token.pending"
    second.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        active_playout_trace_path(published)
