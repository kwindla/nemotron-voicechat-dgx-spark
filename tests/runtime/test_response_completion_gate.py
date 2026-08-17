from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

QUALIFICATION_TOOLS = Path(__file__).resolve().parents[2] / "tools/qualification"
sys.path.insert(0, str(QUALIFICATION_TOOLS))

from response_completion_gate import (
    RECOGNIZED_TRACE_EVENTS,
    load_jsonl,
    response_completion_gate,
)

MODEL_TRACE_ROOT = Path.home() / ".local/state/nemotron-voicechat/traces/model"
KNOWN_BAD_TRACE_ID = "2f88c304-b895-4b81-aa1d-adde6043d58c"
KNOWN_BAD_TRACE_SHA256 = "bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703"
REPAIRED_TRACE_IDS = (
    "5ca16bcb-9c4a-4074-a69e-951c173e591e",
    "5a09f1bd-c027-453e-814a-0e8716dad110",
    "60d0ecf8-6a5e-4ab4-ba21-6f163bcd6c16",
    "93d79622-9d69-4f88-8c6c-f5e5774d9245",
    "9549ed8a-22e6-4261-9bd0-21427e87ac3b",
    "1d188f45-a7bd-40e0-b039-0ef4d17c1c52",
    "793f01ab-42b3-4644-821b-0d12e882f683",
    "c203bca9-0f7c-4544-925a-f1d7cb9ab962",
)
ALL_RETAINED_TRACE_IDS = (KNOWN_BAD_TRACE_ID, *REPAIRED_TRACE_IDS)
SESSION_REPAIR_TRACE_ID = "c42c6701-f14d-4960-83fc-3c24bd269194"
SERVER_SOURCE_PATH = (
    Path(__file__).resolve().parents[2] / "src/nemotron_voicechat_runtime/server.py"
)


def retained_trace(trace_id: str) -> tuple[Path, list[dict]]:
    path = MODEL_TRACE_ROOT / trace_id / "events.jsonl"
    assert path.is_file(), f"retained model trace is missing: {path}"
    return path, load_jsonl(path)


def pinned_known_bad_trace() -> list[dict]:
    path, records = retained_trace(KNOWN_BAD_TRACE_ID)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == KNOWN_BAD_TRACE_SHA256
    return records


def audible_interior_model_steps(records: list[dict]) -> list[dict]:
    selected = []
    for record in records:
        if record.get("event") != "model_step":
            continue
        turn_state = record["turn_state"]
        if turn_state["response_boundary"]["event"] is not None:
            continue
        audio = record.get("model_output_audio") or record["output_audio"]
        if audio["rms_dbfs"] > turn_state["agent_silence_watchdog"]["threshold_dbfs"]:
            selected.append(record)
    assert len(selected) == 208
    return selected


def step(
    frame: int,
    *,
    boundary_event: str | None = None,
    reason: str | None = None,
    dbfs: float = -20.0,
    request_reason: str | None = None,
    text: str = "",
) -> dict:
    model_audio = {"rms_dbfs": dbfs}
    return {
        "trace": "voicechat",
        "event": "model_step",
        "session_id": "synthetic-session",
        "monotonic_s": float(frame),
        "frame": frame,
        "transport_frame": frame,
        "input_audio": {},
        "output_audio": model_audio,
        "audio_delivered": True,
        "inference_ms": 1.0,
        "server_step_ms": 1.0,
        "user_text": "",
        "user_text_changed": False,
        "assistant_delta": text,
        "function_delta": "",
        "function_text": "",
        "model_output_audio": model_audio,
        "preroll_replay": False,
        "step9_capture_fallback": {},
        "turn_state": {
            "response_boundary": {
                "event": boundary_event,
                "eos_reason": reason,
            },
            "agent_silence_watchdog": {
                "threshold_dbfs": -90.0,
                "request_pending": request_reason is not None,
                "request_reason": request_reason,
            },
        },
    }


def trace_record(event: str, **fields: object) -> dict:
    return {
        "trace": "voicechat",
        "event": event,
        "session_id": "synthetic-session",
        "monotonic_s": 0.0,
        **fields,
    }


def _trace_event_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "event"
    ]


def _serialized_trace_event_nodes(tree: ast.AST) -> list[ast.Constant]:
    """Find mapping values that overwrite an event envelope when expanded."""

    expanded_mapping_names = {
        keyword.value.slice.value
        for call in _trace_event_calls(tree)
        for keyword in call.keywords
        if keyword.arg is None
        and isinstance(keyword.value, ast.Subscript)
        and isinstance(keyword.value.slice, ast.Constant)
        and isinstance(keyword.value.slice.value, str)
    }
    assert expanded_mapping_names, "no serialized diagnostic mapping expansions found"

    resolved_mapping_names: set[str] = set()
    event_nodes: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        assigned_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        matched_names = assigned_names & expanded_mapping_names
        if not matched_names:
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "event":
                assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
                    "serialized diagnostic event values must be string literals"
                )
                event_nodes.append(value)
                resolved_mapping_names.update(matched_names)

    assert resolved_mapping_names == expanded_mapping_names, (
        "every diagnostic mapping expanded into SessionTrace.event() must declare "
        "a literal serialized event discriminator"
    )
    return event_nodes


def _source_trace_event_vocabulary(source: str) -> set[str]:
    tree = ast.parse(source)
    event_calls = _trace_event_calls(tree)
    nonliteral_lines = [
        call.lineno
        for call in event_calls
        if not call.args
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ]
    assert not nonliteral_lines, (
        "SessionTrace.event() discriminators must be string literals; "
        f"nonliteral calls at lines {nonliteral_lines}"
    )
    return {call.args[0].value for call in event_calls} | {
        node.value for node in _serialized_trace_event_nodes(tree)
    }


def test_completion_gate_allows_long_text_quiet_audible_response() -> None:
    records = [step(1, boundary_event="start", text="answer")]
    records.extend(step(frame) for frame in range(2, 43))
    records.append(
        step(
            43,
            boundary_event="end",
            reason="decoded_silence_watchdog",
            dbfs=-120.0,
        )
    )
    result = response_completion_gate(records)
    assert result["passed"] is True
    assert result["max_consecutive_text_quiet_but_audible_frames"] == 41
    assert result["watchdog_closures_while_audible"] == 0


def test_completion_gate_mutation_catches_audible_watchdog_truncation() -> None:
    records = [step(1, boundary_event="start", text="answer")]
    records.extend(step(frame) for frame in range(2, 32))
    records.append(step(32, request_reason="no_text_since_bos_watchdog"))
    records.append(
        step(
            33,
            boundary_event="end",
            reason="no_text_since_bos_watchdog",
            dbfs=-120.0,
        )
    )
    good = copy.deepcopy(records)
    good[-2]["turn_state"]["agent_silence_watchdog"].update(
        request_pending=False,
        request_reason=None,
    )
    good[-1]["turn_state"]["response_boundary"]["eos_reason"] = "decoded_silence_watchdog"
    assert response_completion_gate(good)["passed"] is True
    mutated = response_completion_gate(records)
    assert mutated["passed"] is False
    assert mutated["watchdog_closures_while_audible"] == 1

    missing_request_edge = copy.deepcopy(good)
    missing_request_edge[-1]["turn_state"]["response_boundary"]["eos_reason"] = (
        "no_text_since_bos_watchdog"
    )
    omitted_edge_result = response_completion_gate(missing_request_edge)
    assert omitted_edge_result["passed"] is False
    assert omitted_edge_result["watchdog_closures_while_audible"] == 1


def test_retained_known_bad_and_repaired_gate_evidence() -> None:
    evidence = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "reports/notext-watchdog/completion-gate-evidence-r2.json"
        ).read_text(encoding="utf-8")
    )
    bad_evidence = evidence["known_bad"]
    assert bad_evidence["trace_id"] == KNOWN_BAD_TRACE_ID
    bad_records = pinned_known_bad_trace()
    bad = response_completion_gate(bad_records)
    assert bad["input_schema_valid"] is True
    assert bad["passed"] is False
    assert bad["watchdog_closures_while_audible"] == 4
    assert bad["watchdog_closures_without_audible_audio"] == 1

    for expected in evidence["repaired"]:
        _, repaired_records = retained_trace(expected["session_id"])
        repaired = response_completion_gate(repaired_records)
        assert repaired["input_schema_valid"] is True
        assert repaired["passed"] is True
        assert repaired["total_responses"] == expected["total_responses"]
        assert repaired["watchdog_closures_while_audible"] == 0


def test_all_retained_first_party_traces_have_zero_schema_errors() -> None:
    """The pre-fix trace and all eight repaired traces remain parseable as emitted."""

    for trace_id in ALL_RETAINED_TRACE_IDS:
        _, records = retained_trace(trace_id)
        result = response_completion_gate(records)
        assert result["input_schema_valid"] is True, trace_id
        assert result["input_schema_errors"] == [], trace_id


def test_retained_known_bad_fails_when_acoustic_telemetry_is_deleted() -> None:
    records = pinned_known_bad_trace()
    for record in records:
        if record.get("event") == "model_step":
            record.pop("model_output_audio", None)
            record.pop("output_audio", None)
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert any(
        "missing_or_non_finite_output_rms_dbfs" in error for error in result["input_schema_errors"]
    )


def test_retained_known_bad_fails_when_threshold_is_nan() -> None:
    records = pinned_known_bad_trace()
    for record in records:
        if record.get("event") == "model_step":
            record["turn_state"]["agent_silence_watchdog"]["threshold_dbfs"] = float("nan")
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert any(
        "missing_or_non_finite_threshold_dbfs" in error for error in result["input_schema_errors"]
    )


def test_retained_known_bad_fails_when_audible_steps_omit_event() -> None:
    records = pinned_known_bad_trace()
    for record in audible_interior_model_steps(records):
        record.pop("event")
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert (
        sum("missing_or_invalid_event" in error for error in result["input_schema_errors"]) == 208
    )


def test_retained_known_bad_fails_when_audible_steps_misdeclare_event() -> None:
    records = pinned_known_bad_trace()
    for record in audible_interior_model_steps(records):
        record["event"] = "client_event"
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert (
        sum("event_schema_unexpected_fields" in error for error in result["input_schema_errors"])
        == 208
    )


def test_retained_r4_relabel_and_deleted_frame_evasion_is_schema_invalid() -> None:
    records = pinned_known_bad_trace()
    selected = audible_interior_model_steps(records)
    for record in selected:
        record["event"] = "client_event"
        record.pop("frame")
    assert len(selected) == 208

    result = response_completion_gate(records)

    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert (
        sum("event_schema_unexpected_fields" in error for error in result["input_schema_errors"])
        == 208
    )
    assert (
        sum(
            "event_schema_missing_required_fields" in error
            for error in result["input_schema_errors"]
        )
        == 208
    )


def test_retained_r4_nested_model_payload_evasion_is_schema_invalid() -> None:
    records = pinned_known_bad_trace()
    selected = audible_interior_model_steps(records)
    for record in selected:
        payload = dict(record)
        envelope = {key: record[key] for key in ("trace", "session_id", "monotonic_s")}
        record.clear()
        record.update(envelope, event="client_event", type="model.step", payload=payload)
    assert len(selected) == 208

    result = response_completion_gate(records)

    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert (
        sum("event_schema_unexpected_fields" in error for error in result["input_schema_errors"])
        == 208
    )


def test_retained_known_bad_fails_when_audible_steps_use_unknown_event() -> None:
    records = pinned_known_bad_trace()
    for record in audible_interior_model_steps(records):
        record["event"] = "unknown_model_step"
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert sum("unrecognized_event" in error for error in result["input_schema_errors"]) == 208


def test_completion_gate_allows_recognized_non_model_trace_records() -> None:
    records = [
        trace_record("session_trace_started", capture_directory=None),
        trace_record("client_event", type="session.update"),
        step(1, boundary_event="start"),
        step(
            2,
            boundary_event="end",
            reason="decoded_silence_watchdog",
            dbfs=-120.0,
        ),
        trace_record("session_trace_stopped", reason="test"),
    ]
    result = response_completion_gate(records)
    assert result["input_schema_valid"] is True
    assert result["passed"] is True


def test_gate_vocabulary_covers_every_source_and_serialized_producer_event() -> None:
    source = SERVER_SOURCE_PATH.read_text(encoding="utf-8")
    assert _source_trace_event_vocabulary(source) == RECOGNIZED_TRACE_EVENTS


def test_gate_vocabulary_fails_when_serialized_producer_event_is_renamed() -> None:
    source = SERVER_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    serialized_nodes = _serialized_trace_event_nodes(tree)
    assert len(serialized_nodes) == 1
    target_location = (serialized_nodes[0].lineno, serialized_nodes[0].col_offset)

    class RenameSerializedDiagnostic(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if (node.lineno, node.col_offset) == target_location:
                return ast.copy_location(ast.Constant(value="renamed_serialized_diagnostic"), node)
            return node

    renamed_source = ast.unparse(RenameSerializedDiagnostic().visit(tree))
    original_vocabulary = _source_trace_event_vocabulary(source)
    renamed_vocabulary = _source_trace_event_vocabulary(renamed_source)

    assert original_vocabulary == RECOGNIZED_TRACE_EVENTS
    assert renamed_vocabulary != original_vocabulary
    assert renamed_vocabulary != RECOGNIZED_TRACE_EVENTS
    assert "renamed_serialized_diagnostic" in renamed_vocabulary


def test_current_session_trace_diagnostic_serialization_passes_total_parse() -> None:
    import nemotron_voicechat_runtime.runtime_optimizations as optimizations
    from nemotron_voicechat_runtime.server import (
        SessionTrace,
        StepResult,
        VoiceChatEngine,
        send_step,
    )

    engine = VoiceChatEngine.__new__(VoiceChatEngine)
    engine.pipeline = SimpleNamespace(
        s2s_model=SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(text_pad_id=12))),
        _request_id_for_stream=lambda _stream_id: "request-5",
    )
    engine.stream_id = 5
    engine.pad_pair_idle_no_buffer_frames = 0
    engine.pad_pair_watchdog_emitted = False
    engine.pad_pair_trace_errors = 0
    context = SimpleNamespace(
        gen_text=np.asarray([[12]]),
        gen_function_text=np.asarray([[77]]),
    )
    turn_state = {
        "agent_control": "pad",
        "function_calling": {
            "active": False,
            "awaiting_response": False,
            "injecting_response": False,
            "forced_tokens": 0,
            "background_active": False,
        },
    }
    snapshot = {
        "enabled": True,
        "events": [],
        "last_effective_tokens": {
            "frame_idx": 0,
            "text": 12,
            "function": 77,
            "pad": 12,
            "both_pad": False,
        },
        "state": {"previous_effective_pad": False},
    }
    with (
        mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True),
        mock.patch.object(optimizations, "consume_pad_pair_trace", return_value=snapshot),
    ):
        trace_diagnostics = None
        for _ in range(25):
            trace_diagnostics = engine._pad_pair_trace_diagnostics(context, 0, turn_state)

    assert trace_diagnostics is not None
    assert trace_diagnostics["watchdog"] is not None

    class CapturingWebSocket:
        def __init__(self) -> None:
            self.events: list[dict] = []

        async def send_json(self, event: dict) -> None:
            self.events.append(event)

    class TraceOnlyProtocol:
        turn_id = None
        response_id = None

        def step_events(self, *_args, **_kwargs) -> list[dict]:
            return []

        def diagnostic_event(self, kind: str, **fields: object) -> dict:
            return {"type": kind, **fields}

    def step_result(model_step: dict, diagnostics: dict | None = None) -> StepResult:
        return StepResult(
            audio=np.empty(0, dtype=np.float32),
            user_text=model_step["user_text"],
            assistant_delta=model_step["assistant_delta"],
            assistant_text=model_step["assistant_delta"],
            inference_ms=model_step["inference_ms"],
            server_step_ms=model_step["server_step_ms"],
            frame_index=model_step["frame"],
            input_audio=model_step["input_audio"],
            output_audio=model_step["model_output_audio"],
            turn_state=model_step["turn_state"],
            function_delta=model_step["function_delta"],
            function_text=model_step["function_text"],
            trace_diagnostics=diagnostics,
        )

    with tempfile.TemporaryDirectory() as directory:
        trace = SessionTrace(Path(directory), "synthetic-session")
        websocket = CapturingWebSocket()
        protocol = TraceOnlyProtocol()
        asyncio.run(
            send_step(
                websocket,
                step_result(step(1, boundary_event="start"), trace_diagnostics),
                "",
                trace,
                1,
                protocol,
            )
        )
        asyncio.run(
            send_step(
                websocket,
                step_result(
                    step(
                        2,
                        boundary_event="end",
                        reason="decoded_silence_watchdog",
                        dbfs=-120.0,
                    )
                ),
                "",
                trace,
                2,
                protocol,
            )
        )
        trace.close("test")

        records = load_jsonl(Path(directory) / "synthetic-session/events.jsonl")

    serialized_events = {
        node.value
        for node in _serialized_trace_event_nodes(
            ast.parse(SERVER_SOURCE_PATH.read_text(encoding="utf-8"))
        )
    }
    assert len(serialized_events) == 1
    serialized_event = next(iter(serialized_events))
    assert trace_diagnostics["watchdog"]["event"] == serialized_event
    assert [record["event"] for record in records] == [
        "session_trace_started",
        "model_step",
        serialized_event,
        "model_step",
        "session_trace_stopped",
    ]
    assert {key: records[2][key] for key in trace_diagnostics["watchdog"]} == trace_diagnostics[
        "watchdog"
    ]
    result = response_completion_gate(records)
    assert result["input_schema_valid"] is True
    assert result["input_schema_errors"] == []
    assert result["passed"] is True


def test_retained_repaired_trace_fails_on_overlapping_response_start() -> None:
    repaired_trace_id = "5ca16bcb-9c4a-4074-a69e-951c173e591e"
    _, records = retained_trace(repaired_trace_id)
    start_index = next(
        index
        for index, record in enumerate(records)
        if record.get("event") == "model_step"
        and record["turn_state"]["response_boundary"]["event"] == "start"
    )
    next_step = next(
        record for record in records[start_index + 1 :] if record.get("event") == "model_step"
    )
    next_step["turn_state"]["response_boundary"]["event"] = "start"
    result = response_completion_gate(records)
    assert result["passed"] is False
    assert result["input_schema_valid"] is False
    assert any("overlapping_response_start" in error for error in result["input_schema_errors"])


def test_completion_gate_rejects_other_impossible_boundary_topologies() -> None:
    end_without_start = [
        step(1, boundary_event="end", reason="decoded_silence_watchdog", dbfs=-120.0)
    ]
    end_result = response_completion_gate(end_without_start)
    assert end_result["passed"] is False
    assert any("response_end_without_start" in error for error in end_result["input_schema_errors"])

    open_at_eof = [step(1, boundary_event="start")]
    open_result = response_completion_gate(open_at_eof)
    assert open_result["passed"] is False
    assert any("response_open_at_eof" in error for error in open_result["input_schema_errors"])


def test_completion_gate_records_open_response_terminated_by_session_lifecycle() -> None:
    terminated = [
        step(1, boundary_event="start", text="still speaking"),
        step(2, text=" when disconnected"),
        trace_record("terminal_session_claimed", reason="internal_error"),
        trace_record("session_trace_stopped", reason="internal_error"),
    ]

    result = response_completion_gate(terminated)

    assert result["input_schema_valid"] is True
    assert result["passed"] is True
    assert result["total_responses"] == 1
    assert result["completion_reasons"] == {"session_terminated": 1}
    assert result["session_terminated_open_response_count"] == 1
    termination = result["session_terminated_open_responses"][0]["termination"]
    assert termination == {
        "bos_frame": 1,
        "trace_stop_reason": "internal_error",
        "trace_stop_record_index": 3,
        "disposition": "session_terminated_open_response",
    }


def test_completion_gate_rejects_open_response_in_cleanly_completed_session() -> None:
    completed = [
        step(1, boundary_event="start", text="never closed"),
        trace_record("client_event", type="session.stop"),
        trace_record("session_trace_stopped", reason="client_stop"),
    ]

    result = response_completion_gate(completed)

    assert result["input_schema_valid"] is False
    assert result["passed"] is False
    assert any("response_open_at_eof" in error for error in result["input_schema_errors"])
    assert result["session_terminated_open_response_count"] == 0


def test_completion_gate_rejects_unknown_or_typo_stop_reason_without_claim() -> None:
    for reason in ("client_stpo", "test"):
        result = response_completion_gate(
            [
                step(1, boundary_event="start", text="never closed"),
                trace_record("session_trace_stopped", reason=reason),
            ]
        )

        assert result["input_schema_valid"] is False
        assert result["passed"] is False
        assert any("response_open_at_eof" in error for error in result["input_schema_errors"])
        assert result["session_terminated_open_response_count"] == 0


def test_completion_gate_accepts_narrow_unclaimed_transport_disconnect() -> None:
    result = response_completion_gate(
        [
            step(1, boundary_event="start", text="transport dropped"),
            trace_record("session_trace_stopped", reason="disconnect"),
        ]
    )

    assert result["input_schema_valid"] is True
    assert result["passed"] is True
    assert result["completion_reasons"] == {"session_terminated": 1}


def test_completion_gate_rejects_duplicate_session_trace_stop() -> None:
    result = response_completion_gate(
        [
            step(1, boundary_event="start"),
            trace_record("session_trace_stopped", reason="disconnect"),
            trace_record("session_trace_stopped", reason="disconnect"),
        ]
    )

    assert result["input_schema_valid"] is False
    assert result["passed"] is False
    assert any(
        "duplicate_session_trace_stopped" in error
        for error in result["input_schema_errors"]
    )


def test_completion_gate_requires_well_formed_terminal_lifecycle() -> None:
    missing_reason = [
        step(1, boundary_event="start"),
        trace_record("session_trace_stopped", reason=None),
    ]
    result = response_completion_gate(missing_reason)
    assert result["passed"] is False
    assert any(
        "missing_or_invalid_session_stop_reason" in error
        for error in result["input_schema_errors"]
    )

    record_after_stop = [
        step(1, boundary_event="start"),
        trace_record("session_trace_stopped", reason="internal_error"),
        trace_record("client_event", type="late"),
    ]
    result = response_completion_gate(record_after_stop)
    assert result["passed"] is False
    assert any(
        "record_after_session_trace_stopped" in error
        for error in result["input_schema_errors"]
    )


def test_session_repair_trace_records_legitimate_terminated_open_response() -> None:
    path, records = retained_trace(SESSION_REPAIR_TRACE_ID)

    result = response_completion_gate(records, source=str(path))

    assert result["input_schema_valid"] is True
    assert result["passed"] is True
    assert result["total_responses"] == 9
    assert result["session_terminated_open_response_count"] == 1
    assert result["session_terminated_open_responses"][0]["bos_frame"] == 975


def test_completion_gate_rejects_invalid_frame_session_and_output_dbfs() -> None:
    valid = [
        step(1, boundary_event="start"),
        step(2, boundary_event="end", reason="decoded_silence_watchdog", dbfs=-120.0),
    ]

    invalid_frame = copy.deepcopy(valid)
    invalid_frame[0].pop("frame")
    frame_result = response_completion_gate(invalid_frame)
    assert frame_result["passed"] is False
    assert any("missing_or_invalid_frame" in error for error in frame_result["input_schema_errors"])

    invalid_session = copy.deepcopy(valid)
    invalid_session[0]["session_id"] = ""
    session_result = response_completion_gate(invalid_session)
    assert session_result["passed"] is False
    assert any(
        "missing_or_invalid_session_id" in error for error in session_result["input_schema_errors"]
    )

    non_finite_output = copy.deepcopy(valid)
    non_finite_output[1]["model_output_audio"]["rms_dbfs"] = float("inf")
    output_result = response_completion_gate(non_finite_output)
    assert output_result["passed"] is False
    assert any(
        "missing_or_non_finite_output_rms_dbfs" in error
        for error in output_result["input_schema_errors"]
    )
