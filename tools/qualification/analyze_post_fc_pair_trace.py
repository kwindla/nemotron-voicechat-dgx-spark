#!/usr/bin/env python3
"""Analyze opt-in Nano PAD-pair telemetry around function-call responses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fc_idle(step: dict[str, Any]) -> bool:
    function = (step.get("turn_state") or {}).get("function_calling") or {}
    return not any(
        function.get(key)
        for key in (
            "active",
            "awaiting_response",
            "injecting_response",
            "forced_tokens",
            "background_active",
        )
    )


def _buffered(step: dict[str, Any]) -> bool:
    events = ((step.get("runtime_diagnostics") or {}).get("pad_pair") or {}).get("events", [])
    return any(event.get("decision") == "buffered" for event in events)


def _ordinary_pad(step: dict[str, Any]) -> bool:
    diagnostics = step.get("runtime_diagnostics") or {}
    effective = diagnostics.get("effective_tokens") or {}
    return bool(
        _fc_idle(step)
        and (step.get("turn_state") or {}).get("agent_control") == "pad"
        and effective.get("both_pad") is True
    )


def _tool_response_groups(
    network_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    done: dict[str, float | int | None] = {}
    for event in network_events:
        response_id = event.get("response_id")
        if not isinstance(response_id, str):
            continue
        if event.get("type") == "response.function_call_arguments.done":
            calls[response_id].append(event)
        elif event.get("type") == "response.done":
            done[response_id] = event.get("client_received_monotonic_s")
    groups = []
    for response_id, response_calls in calls.items():
        groups.append(
            {
                "response_id": response_id,
                "turn_id": response_calls[-1].get("turn_id"),
                "call_ids": [event.get("call_id") for event in response_calls],
                "call_count": len(response_calls),
                "last_call_monotonic_s": response_calls[-1].get("client_received_monotonic_s"),
                "response_closed": response_id in done,
                "response_done_monotonic_s": done.get(response_id),
            }
        )
    return groups


def _integrity_failures(model_steps: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        "trace_errors": 0,
        "events_dropped": 0,
        "trace_record_errors": 0,
        "request_id_mismatches": 0,
        "effective_token_mismatches": 0,
        "invariant_violations": 0,
    }
    for step in model_steps:
        diagnostics = step.get("runtime_diagnostics") or {}
        if diagnostics.get("trace_error") is not None:
            result["trace_errors"] += 1
        if diagnostics.get("effective_tokens_match_finalize") is False:
            result["effective_token_mismatches"] += 1
        pad_pair = diagnostics.get("pad_pair") or {}
        result["events_dropped"] += int(pad_pair.get("events_dropped") or 0)
        result["trace_record_errors"] = max(
            result["trace_record_errors"], int(pad_pair.get("trace_record_errors") or 0)
        )
        for event in pad_pair.get("events", []):
            if event.get("request_id_match") is False:
                result["request_id_mismatches"] += 1
            if event.get("invariant_violation") is True:
                result["invariant_violations"] += 1
    return result


def analyze_session(
    network_events: list[dict[str, Any]], trace_events: list[dict[str, Any]]
) -> dict[str, Any]:
    model_steps = [event for event in trace_events if event.get("event") == "model_step"]
    watchdogs = [event for event in trace_events if event.get("event") == "nano_pad_pair_watchdog"]
    groups = _tool_response_groups(network_events)
    recoveries: list[dict[str, Any]] = []
    for group in groups:
        response_done = group.get("response_done_monotonic_s")
        after_close = [
            step
            for step in model_steps
            if isinstance(response_done, (int, float))
            and isinstance(step.get("monotonic_s"), (int, float))
            and step["monotonic_s"] >= response_done
        ]
        pad_index = next(
            (index for index, step in enumerate(after_close) if _ordinary_pad(step)), None
        )
        first_after_close_frame = after_close[0].get("frame") if after_close else None
        first_pad_frame = None
        first_buffer_frame = None
        status = "no_model_step_after_response_close"
        pad_return_frame_delta = None
        if after_close:
            status = "no_ordinary_pad_after_response_close"
        if pad_index is not None:
            first_pad_frame = after_close[pad_index].get("frame")
            if isinstance(first_after_close_frame, int) and isinstance(first_pad_frame, int):
                pad_return_frame_delta = first_pad_frame - first_after_close_frame
            for step in after_close[pad_index:]:
                if _buffered(step):
                    first_buffer_frame = step.get("frame")
                    break
            if first_buffer_frame is None:
                status = "ordinary_pad_without_pair_buffer"
            elif isinstance(first_pad_frame, int) and isinstance(first_buffer_frame, int):
                if pad_return_frame_delta is None or pad_return_frame_delta > 2:
                    status = "late_pad_return_after_response_close"
                elif first_buffer_frame - first_pad_frame > 2:
                    status = "late_pair_recovery"
                else:
                    status = "recovered"
        recoveries.append(
            {
                **group,
                "first_frame_after_response_close": first_after_close_frame,
                "first_ordinary_pad_frame": first_pad_frame,
                "pad_return_frame_delta": pad_return_frame_delta,
                "first_pair_buffer_frame": first_buffer_frame,
                "recovery_frame_delta": (
                    first_buffer_frame - first_pad_frame
                    if isinstance(first_pad_frame, int) and isinstance(first_buffer_frame, int)
                    else None
                ),
                "status": status,
            }
        )

    integrity = _integrity_failures(model_steps)
    passed = bool(
        len(groups) == 2
        and all(group["call_count"] == 2 for group in groups)
        and all(group["response_closed"] for group in groups)
        and all(recovery["status"] == "recovered" for recovery in recoveries)
        and not watchdogs
        and not any(integrity.values())
    )
    return {
        "passed": passed,
        "model_steps": len(model_steps),
        "tool_response_groups": groups,
        "recoveries": recoveries,
        "watchdogs": len(watchdogs),
        "integrity": integrity,
    }


def analyze_campaign(
    campaign_root: Path, *, session_glob: str, required_sessions: int
) -> dict[str, Any]:
    sessions = []
    for session_dir in sorted(campaign_root.glob(session_glob)):
        network_path = session_dir / "events.jsonl"
        if not network_path.is_file():
            continue
        network_events = read_jsonl(network_path)
        session_id = next(
            (
                event.get("session_id")
                for event in network_events
                if event.get("type") == "session.created"
            ),
            None,
        )
        if not isinstance(session_id, str):
            raise ValueError(f"{network_path} has no strict-v3 session.created event")
        trace_path = campaign_root / "model" / session_id / "events.jsonl"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        harness_report_path = session_dir / "report.json"
        harness_report = (
            json.loads(harness_report_path.read_text(encoding="utf-8"))
            if harness_report_path.is_file()
            else {}
        )
        fixture = {
            "typed_completed": harness_report.get("typed_completed"),
            "typed_answered": harness_report.get("typed_answered"),
            "responses": len(harness_report.get("responses") or []),
        }
        fixture["passed"] = fixture == {
            "typed_completed": 3,
            "typed_answered": 3,
            "responses": 3,
        }
        result = analyze_session(network_events, read_jsonl(trace_path))
        result["fixture"] = fixture
        result["passed"] = bool(result["passed"] and fixture["passed"])
        result.update(
            {
                "name": session_dir.name,
                "session_id": session_id,
                "network_events": str(network_path),
                "model_trace": str(trace_path),
            }
        )
        sessions.append(result)
    return {
        "passed": len(sessions) == required_sessions
        and all(session["passed"] for session in sessions),
        "session_glob": session_glob,
        "required_sessions": required_sessions,
        "analyzed_sessions": len(sessions),
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--session-glob", default="bounded-session-*")
    parser.add_argument("--required-sessions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_campaign(
        args.campaign_root.resolve(),
        session_glob=args.session_glob,
        required_sessions=args.required_sessions,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
