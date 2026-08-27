#!/usr/bin/env python3
"""Trace-derived response completion gate for no-text watchdog qualification.

Threat model: this gate detects silent runtime regressions, malformed or incomplete
telemetry, and adjudication absorption in first-party traces emitted by this
repository's ``SessionTrace`` producer.  Its mutation suite is robustness evidence;
it is not an integrity guarantee against an adversary with arbitrary write access to
the retained trace bytes.

Every input record is parsed totally against one discriminator-selected schema.
There is deliberately no ignored or unvalidated record class.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

WATCHDOG_REASONS = {
    "no_text_since_bos_watchdog",
    "no_audio_since_bos_watchdog",
}
COMPLETED_SESSION_REASONS = {"client_stop"}
UNCLAIMED_ABNORMAL_SESSION_REASONS = {"disconnect"}
AUDIBLE_WINDOW_FRAMES = 5
TRACE_ENVELOPE_FIELDS = frozenset({"trace", "event", "session_id", "monotonic_s"})


@dataclass(frozen=True)
class TraceEventSchema:
    """Closed top-level shape for one event emitted by ``SessionTrace``."""

    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional


def _schema(required: Iterable[str] = (), optional: Iterable[str] = ()) -> TraceEventSchema:
    return TraceEventSchema(
        required=TRACE_ENVELOPE_FIELDS | frozenset(required),
        optional=frozenset(optional),
    )


_FENCE_EVIDENCE_FIELDS = {
    "client_turn_id",
    "target_blank_frames",
    "armed_transport_frame",
    "armed_model_frame",
    "prearmed_blank_frames",
    "eligible_audio_frames",
}
_SETTLEMENT_EVIDENCE_FIELDS = {
    "target_blank_frames",
    "starting_blank_frames",
    "prearmed_blank_frames",
    "post_commit_blank_deficit",
    "prearm",
    "post_commit_steps",
    "ending_blank_frames",
    "max_model_steps",
    "model_steps",
    "real_audio_model_steps",
    "synthetic_model_steps",
    "real_audio_stopped_by_speech_gate",
    "eartts_reset_count_starting",
    "fence_reached",
    "within_bound",
    "pre_eou_clean",
    "steps",
}

# This is an exhaustive closed vocabulary for SessionTrace.event() call sites in
# src/nemotron_voicechat_runtime/server.py, including the serialized diagnostic
# discriminator which can replace the declared nano_pad_pair_watchdog kind through
# the producer's top-level **fields merge.  The source-enumeration and producer-level
# tests fail whenever a call site or serialized variant is missing here.
TRACE_EVENT_SCHEMAS: dict[str, TraceEventSchema] = {
    "client_event": _schema({"type"}),
    "client_input_deferred_for_function": _schema({"type", "call_id", "queued_bytes"}),
    "client_input_turn_commit_received": _schema(
        {
            "client_turn_id",
            "client_diagnostics",
            "client_queue_wait_ms",
            "client_send_to_server_receive_ms",
        }
    ),
    "client_input_turn_committed": _schema(
        {
            "client_turn_id",
            "receive_to_eou_settled_ms",
            "eou_settled_to_bos_ms",
            "receive_to_bos_ms",
        },
        {"fused_terminal_bos"},
    ),
    "client_input_turn_started": _schema({"client_turn_id", "preroll_frames"}),
    "client_turn_start_lost_function_publication_race": _schema(
        {"client_turn_id", "call_id", "queued_bytes"}
    ),
    "deferred_bos_superseded_by_client_turn": _schema(
        {"initiating_turn_id", "responding_turn_id", "settlement"}
    ),
    "deferred_bos_superseded_by_interrupt": _schema(
        {"initiating_turn_id", "responding_turn_id", "settlement"}
    ),
    "eartts_client_commit_arm": _schema({"client_turn_id", "evidence"}),
    "eartts_turn_start_prepare": _schema({"client_turn_id", "evidence"}),
    "function_call_loop_limit": _schema(
        {"call_id", "name", "arguments", "attempt", "limit", "turn_id", "response_id"}
    ),
    "function_call_output_received": _schema(
        {
            "call_id",
            "output_bytes",
            "output_sha256",
            "output_tokens",
            "model_output_supplied",
            "model_output_bytes",
            "model_output_sha256",
            "model_output_tokens",
            "injection_source",
            "injection_tokens",
        }
    ),
    "function_call_publication_suppressed": _schema({"call_id", "name", "turn_id", "reason"}),
    "function_call_requested": _schema({"call_id", "name", "arguments", "turn_id"}),
    "function_call_timeout": _schema({"call_id", "timeout_seconds"}),
    "function_call_timeout_failed_event_error": _schema({"call_id", "error"}),
    "idle_agent_without_pair_buffer": _schema(
        {
            "frame",
            "transport_frame",
            "consecutive_frames",
            "threshold_frames",
            "effective_tokens",
            "scheduler",
        }
    ),
    "idle_step": _schema({"transport_frame", "input_audio", "output_audio", "turn_state"}),
    "input_packet": _schema({"bytes", "source", "job_id", "model_input_provenance", "audio"}),
    "microphone_packet_gated": _schema({"bytes", "active_job_id"}),
    "microphone_partial_discarded_for_typed_input": _schema({"job_id", "samples"}),
    "model_abort_cleanup_error": _schema({"error"}),
    "model_session_empty_restart_completed": _schema(),
    "model_session_empty_restart_requested": _schema({"previous_stream_id"}),
    "model_session_empty_stop_completed": _schema(),
    "model_session_empty_stop_requested": _schema(),
    "model_session_start_completed": _schema({"reason", "start_evidence", "elapsed_ms"}),
    "model_session_start_failed": _schema(
        {"reason", "phase", "timings_ms", "elapsed_ms", "cleanup_error"}
    ),
    "model_session_start_requested": _schema({"reason"}),
    "model_step": _schema(
        {
            "frame",
            "transport_frame",
            "input_audio",
            "output_audio",
            "model_output_audio",
            "audio_delivered",
            "inference_ms",
            "server_step_ms",
            "user_text",
            "user_text_changed",
            "assistant_delta",
            "function_delta",
            "function_text",
            "turn_state",
            "preroll_replay",
            "step9_capture_fallback",
        },
        {"runtime_diagnostics"},
    ),
    "nano_pad_pair_watchdog": _schema(
        {
            "frame",
            "transport_frame",
            "consecutive_frames",
            "threshold_frames",
            "effective_tokens",
            "scheduler",
        }
    ),
    "pending_bos_expectation_terminated": _schema(
        {"code", "call_id", "epoch_invalidation_error", "settlement"}
    ),
    "pre_eou_activity_recovered": _schema(
        {"context", "response_open", "function_cycle_effective"},
        {
            "function_interrupt",
            "dangling_response_closed",
            "drained_frames",
            "escalated_to_cancel",
        },
    ),
    "pre_eou_activity_recovery": _schema({"context", "response_open", "function_cycle_effective"}),
    "session_position_limit": _schema({"model_frames", "max_model_frames"}),
    "session_tools_configured": _schema(
        {
            "tools",
            "rendered_prompt_bytes",
            "configuration_changed",
            "model_ready",
            "function_output_model_output",
        }
    ),
    "session_trace_started": _schema({"capture_directory"}),
    "session_trace_stopped": _schema({"reason"}),
    "session_update_model_input_lock_acquired": _schema({"wait_ms"}),
    "session_update_model_input_lock_requested": _schema(),
    "speech_gate_preroll_completed": _schema({"frames", "elapsed_ms"}),
    "speech_gate_preroll_selected": _schema(
        {
            "reason",
            "buffered_frames",
            "replayed_frames",
            "skipped_frames",
            "first_active_index",
            "onset_context_frames",
            "threshold_dbfs",
        }
    ),
    "speech_gate_preroll_started": _schema(
        {"frames", "session_started", "transport_frames", "sources", "job_ids", "input_active"}
    ),
    "terminal_cleanup_wait_skipped": _schema({"reason", "owner_reason", "task"}),
    "terminal_event_preparation_error": _schema({"error"}),
    "terminal_publication_error": _schema({"error"}),
    "terminal_session_claimed": _schema({"reason"}),
    "tool_bridge_cleanup_error": _schema({"error"}),
    "typed_input_accepted": _schema({"job_id", "text_bytes"}),
    "typed_input_error": _schema({"job_id", "error"}),
    "typed_input_injection_finished": _schema({"job_id", "disposition", "audio_started"}),
    "typed_input_injection_started": _schema(
        {"job_id", "generated_seconds", "synthesis_seconds", "time_to_first_audio_seconds"}
    ),
    "typed_input_tail_discarded": _schema({"job_id", "samples"}),
    "typed_task_cleanup_error": _schema({"error"}),
    "unpublished_function_cycle_interrupt": _schema(
        {"client_turn_id", "initiating_turn_id", "evidence"}
    ),
    "user_eou_fence_prearm_rejected": _schema(_FENCE_EVIDENCE_FIELDS | {"current_blank_frames"}),
    "user_eou_fence_prearm_revoked": _schema(_FENCE_EVIDENCE_FIELDS | {"reason", "audio_rms_dbfs"}),
    "user_eou_fence_prearmed": _schema(_FENCE_EVIDENCE_FIELDS),
    "user_eou_post_commit_frame": _schema(
        {
            "client_turn_id",
            "model_input_provenance",
            "real_samples",
            "padded_samples",
            "blank_frames_before",
            "blank_frames_after",
            "blank_frames_advanced",
            "model_frame",
        }
    ),
    "user_eou_settled": _schema(
        _SETTLEMENT_EVIDENCE_FIELDS | {"source", "job_id"},
        {
            "fused_terminal_bos",
            "separate_terminal_bos",
            "deferred_terminal_bos",
            "schema",
            "expected_client_turn_id",
            "eartts_transition_count_starting",
            "eartts_reuse_count_starting",
            "terminal_bos_epoch",
        },
    ),
    "user_eou_settlement_real_audio_blocked": _schema({"reason", "threshold_dbfs", "settlement"}),
}
RECOGNIZED_TRACE_EVENTS = frozenset(TRACE_EVENT_SCHEMAS)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _output_dbfs(step: dict[str, Any]) -> float | None:
    audio = step.get("model_output_audio")
    if audio is None:
        audio = step.get("output_audio")
    if not isinstance(audio, dict):
        return None
    return _finite_number(audio.get("rms_dbfs"))


def _parse_trace_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], set[str], dict[str, Any] | None]:
    """Totally parse every record, returning validated model steps for classification."""

    steps: list[dict[str, Any]] = []
    errors: list[str] = []
    session_ids: set[str] = set()
    active_start_frame: int | None = None
    active_start_record_index: int | None = None
    previous_frame: int | None = None
    terminal_claim: dict[str, Any] | None = None
    trace_stop: dict[str, Any] | None = None

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record[{record_index}]: non_object_record")
            continue

        if trace_stop is not None:
            errors.append(
                f"record[{record_index}]: record_after_session_trace_stopped"
            )

        event = record.get("event")
        if not isinstance(event, str) or not event:
            errors.append(f"record[{record_index}]: missing_or_invalid_event")
            continue

        schema = TRACE_EVENT_SCHEMAS.get(event)
        if schema is None:
            errors.append(f"record[{record_index}]: unrecognized_event ({event!r})")
            continue

        missing = sorted(schema.required - record.keys())
        if missing:
            errors.append(
                f"record[{record_index}]: event_schema_missing_required_fields "
                f"(event={event!r}, fields={missing!r})"
            )
        unexpected = sorted(record.keys() - schema.allowed)
        if unexpected:
            errors.append(
                f"record[{record_index}]: event_schema_unexpected_fields "
                f"(event={event!r}, fields={unexpected!r})"
            )

        trace_kind = record.get("trace")
        if trace_kind != "voicechat":
            errors.append(f"record[{record_index}]: missing_or_invalid_trace")

        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            errors.append(f"record[{record_index}]: missing_or_invalid_session_id")
        else:
            session_ids.add(session_id)

        monotonic_s = _finite_number(record.get("monotonic_s"))
        if monotonic_s is None or monotonic_s < 0:
            errors.append(f"record[{record_index}]: missing_or_invalid_monotonic_s")

        if event == "terminal_session_claimed":
            claim_reason = record.get("reason")
            if terminal_claim is not None:
                errors.append(f"record[{record_index}]: duplicate_terminal_session_claimed")
            if not isinstance(claim_reason, str) or not claim_reason.strip():
                errors.append(
                    f"record[{record_index}]: missing_or_invalid_terminal_claim_reason"
                )
            terminal_claim = {
                "record_index": record_index,
                "reason": claim_reason,
            }

        if event == "session_trace_stopped":
            if trace_stop is not None:
                errors.append(f"record[{record_index}]: duplicate_session_trace_stopped")
            stop_reason = record.get("reason")
            if not isinstance(stop_reason, str) or not stop_reason.strip():
                errors.append(
                    f"record[{record_index}]: missing_or_invalid_session_stop_reason"
                )
            trace_stop = {
                "record_index": record_index,
                "reason": stop_reason,
                "monotonic_s": monotonic_s,
            }
            if terminal_claim is not None and terminal_claim["reason"] != stop_reason:
                errors.append(
                    f"record[{record_index}]: terminal_claim_stop_reason_mismatch"
                )

        if event != "model_step":
            continue
        steps.append(record)
        label = f"model_step[{record_index}]"

        frame = record.get("frame")
        frame_valid = isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0
        if not frame_valid:
            errors.append(f"{label}: missing_or_invalid_frame")
        elif previous_frame is not None and frame <= previous_frame:
            errors.append(f"{label}: non_monotonic_frame")
        if frame_valid:
            previous_frame = frame

        turn_state = record.get("turn_state")
        if not isinstance(turn_state, dict):
            errors.append(f"{label}: missing_or_invalid_turn_state")
            continue

        watchdog = turn_state.get("agent_silence_watchdog")
        if not isinstance(watchdog, dict):
            errors.append(f"{label}: missing_or_invalid_agent_silence_watchdog")
        elif _finite_number(watchdog.get("threshold_dbfs")) is None:
            errors.append(f"{label}: missing_or_non_finite_threshold_dbfs")

        if _output_dbfs(record) is None:
            errors.append(f"{label}: missing_or_non_finite_output_rms_dbfs")

        boundary = turn_state.get("response_boundary")
        if not isinstance(boundary, dict):
            errors.append(f"{label}: missing_or_invalid_response_boundary")
            continue
        event = boundary.get("event")
        if event not in {None, "start", "end"}:
            errors.append(f"{label}: invalid_response_boundary_event")
        elif event == "start":
            if active_start_frame is not None:
                errors.append(
                    f"{label}: overlapping_response_start (open_bos_frame={active_start_frame})"
                )
            else:
                active_start_frame = frame if frame_valid else -1
                active_start_record_index = record_index
        elif event == "end":
            if active_start_frame is None:
                errors.append(f"{label}: response_end_without_start")
            else:
                active_start_frame = None
                active_start_record_index = None
            reason = boundary.get("eos_reason")
            if not isinstance(reason, str) or not reason:
                errors.append(f"{label}: missing_or_invalid_eos_reason")

    if not steps:
        errors.append("trace: no_model_steps")
    if len(session_ids) > 1:
        errors.append("trace: multiple_session_ids")
    terminated_open_response = None
    if active_start_frame is not None:
        claimed_abnormal_termination = bool(
            terminal_claim is not None
            and terminal_claim["record_index"] > active_start_record_index
            and terminal_claim["record_index"] < trace_stop["record_index"]
            and terminal_claim["reason"] == trace_stop["reason"]
        ) if trace_stop is not None and active_start_record_index is not None else False
        unclaimed_transport_disconnect = bool(
            trace_stop is not None
            and trace_stop["reason"] in UNCLAIMED_ABNORMAL_SESSION_REASONS
        )
        if (
            trace_stop is not None
            and active_start_record_index is not None
            and trace_stop["record_index"] > active_start_record_index
            and trace_stop["reason"] not in COMPLETED_SESSION_REASONS
            and (claimed_abnormal_termination or unclaimed_transport_disconnect)
        ):
            terminated_open_response = {
                "bos_frame": active_start_frame,
                "trace_stop_reason": trace_stop["reason"],
                "trace_stop_record_index": trace_stop["record_index"],
                "disposition": "session_terminated_open_response",
            }
        else:
            errors.append(f"trace: response_open_at_eof (bos_frame={active_start_frame})")
    return steps, errors, session_ids, terminated_open_response


def _schema_failure_result(
    *,
    source: str | None,
    errors: list[str],
    session_ids: set[str],
) -> dict[str, Any]:
    return {
        "schema": "nemotron_voicechat.response_completion_gate.v1",
        "source": source,
        "session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
        "input_schema_valid": False,
        "input_schema_errors": errors,
        "audible_window_frames": AUDIBLE_WINDOW_FRAMES,
        "threshold_dbfs": None,
        "total_responses": 0,
        "completion_reasons": {},
        "watchdog_closures_while_audible": 0,
        "watchdog_closures_without_audible_audio": 0,
        "decoded_silence_watchdog_closures": 0,
        "text_quiet_but_audible_frames": 0,
        "max_consecutive_text_quiet_but_audible_frames": 0,
        "audible_watchdog_defects": [],
        "session_terminated_open_response_count": 0,
        "session_terminated_open_responses": [],
        "incomplete_responses": errors,
        "responses": [],
        "passed": False,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            records.append(value)
    return records


def response_completion_gate(
    records: Iterable[dict[str, Any]],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    (
        model_steps,
        schema_errors,
        validated_session_ids,
        terminated_open_response,
    ) = _parse_trace_records(records)
    if schema_errors:
        return _schema_failure_result(
            source=source,
            errors=schema_errors,
            session_ids=validated_session_ids,
        )

    responses: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    history: deque[dict[str, Any]] = deque(maxlen=AUDIBLE_WINDOW_FRAMES)
    histogram: Counter[str] = Counter()
    audible_watchdog_defects: list[dict[str, Any]] = []
    dead_watchdog_rescues = 0
    decoded_silence_closures = 0
    text_quiet_audible_total = 0
    max_consecutive = 0
    current_consecutive = 0
    threshold_seen: float | None = None
    session_ids = validated_session_ids

    for step in model_steps:
        turn_state = step["turn_state"]
        boundary = turn_state["response_boundary"]
        watchdog = turn_state["agent_silence_watchdog"]
        threshold = float(watchdog["threshold_dbfs"])
        threshold_seen = threshold
        dbfs = _output_dbfs(step)
        assert dbfs is not None
        audible = dbfs > threshold
        frame = step["frame"]
        history.append({"frame": frame, "rms_dbfs": dbfs, "audible": audible})

        if boundary.get("event") == "start":
            assert active is None
            active = {
                "bos_frame": frame,
                "eos_frame": None,
                "completion_reason": None,
                "watchdog_request_frame": None,
                "watchdog_audible_window": [],
                "text_quiet_audible_frames": 0,
                "max_consecutive_text_quiet_audible_frames": 0,
            }
            current_consecutive = 0

        if active is None:
            continue
        text_delta = str(step.get("assistant_delta") or "")
        if audible and not text_delta:
            text_quiet_audible_total += 1
            active["text_quiet_audible_frames"] += 1
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
            active["max_consecutive_text_quiet_audible_frames"] = max(
                active["max_consecutive_text_quiet_audible_frames"], current_consecutive
            )
        else:
            current_consecutive = 0

        request_reason = watchdog.get("request_reason")
        if (
            active["watchdog_request_frame"] is None
            and watchdog.get("request_pending") is True
            and request_reason in WATCHDOG_REASONS
        ):
            active["watchdog_request_frame"] = frame
            active["watchdog_audible_window"] = list(history)

        if boundary.get("event") == "end":
            reason = boundary["eos_reason"]
            active["eos_frame"] = frame
            active["completion_reason"] = reason
            histogram[reason] += 1
            if reason in WATCHDOG_REASONS:
                # A retained request edge gives the most exact firing window.
                # Fall back to the closure window so a producer cannot evade the
                # gate merely by omitting request_pending from its trace.
                window = active["watchdog_audible_window"] or list(history)
                active["watchdog_audible_window"] = window
                if any(item["audible"] for item in window):
                    audible_watchdog_defects.append(
                        {
                            "bos_frame": active["bos_frame"],
                            "watchdog_request_frame": active["watchdog_request_frame"],
                            "eos_frame": frame,
                            "reason": reason,
                            "audible_window": window,
                        }
                    )
                else:
                    dead_watchdog_rescues += 1
            elif reason == "decoded_silence_watchdog":
                decoded_silence_closures += 1
            responses.append(active)
            active = None
            current_consecutive = 0

    incomplete: list[Any] = []
    terminated_responses: list[dict[str, Any]] = []
    if active is not None:
        assert terminated_open_response is not None
        active["completion_reason"] = "session_terminated"
        active["termination"] = terminated_open_response
        terminated_responses.append(active)
        responses.append(active)
        histogram["session_terminated"] += 1
        active = None
    passed = bool(responses) and not audible_watchdog_defects and not incomplete
    return {
        "schema": "nemotron_voicechat.response_completion_gate.v1",
        "source": source,
        "session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
        "input_schema_valid": True,
        "input_schema_errors": [],
        "audible_window_frames": AUDIBLE_WINDOW_FRAMES,
        "threshold_dbfs": threshold_seen,
        "total_responses": len(responses),
        "completion_reasons": dict(sorted(histogram.items())),
        "watchdog_closures_while_audible": len(audible_watchdog_defects),
        "watchdog_closures_without_audible_audio": dead_watchdog_rescues,
        "decoded_silence_watchdog_closures": decoded_silence_closures,
        "text_quiet_but_audible_frames": text_quiet_audible_total,
        "max_consecutive_text_quiet_but_audible_frames": max_consecutive,
        "audible_watchdog_defects": audible_watchdog_defects,
        "session_terminated_open_response_count": len(terminated_responses),
        "session_terminated_open_responses": terminated_responses,
        "incomplete_responses": incomplete,
        "responses": responses,
        "passed": passed,
    }


def gate_trace(path: Path) -> dict[str, Any]:
    return response_completion_gate(load_jsonl(path), source=str(path.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = gate_trace(args.trace)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
