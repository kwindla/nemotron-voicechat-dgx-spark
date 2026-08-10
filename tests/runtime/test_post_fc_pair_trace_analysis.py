from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("tools/qualification/analyze_post_fc_pair_trace.py")
    spec = importlib.util.spec_from_file_location("analyze_post_fc_pair_trace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def network_tool_response(response_id: str, turn_id: str, at: float):
    return [
        {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "turn_id": turn_id,
            "call_id": f"{response_id}-1",
            "client_received_monotonic_s": at - 2,
        },
        {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "turn_id": turn_id,
            "call_id": f"{response_id}-2",
            "client_received_monotonic_s": at - 1,
        },
        {
            "type": "response.done",
            "response_id": response_id,
            "turn_id": turn_id,
            "client_received_monotonic_s": at,
        },
    ]


def model_step(
    frame: int,
    at: float,
    *,
    both_pad: bool,
    decision: str | None,
) -> dict:
    scheduler_event = []
    if decision:
        scheduler_event.append(
            {
                "decision": decision,
                "request_id_match": True,
                "invariant_violation": False,
            }
        )
    return {
        "event": "model_step",
        "frame": frame,
        "monotonic_s": at,
        "turn_state": {
            "agent_control": "pad",
            "function_calling": {
                "active": False,
                "awaiting_response": False,
                "injecting_response": False,
                "forced_tokens": 0,
                "background_active": False,
            },
        },
        "runtime_diagnostics": {
            "effective_tokens_match_finalize": True,
            "effective_tokens": {
                "both_pad": both_pad,
                "text": 12,
                "function": 12 if both_pad else 99,
                "pad": 12,
            },
            "pad_pair": {
                "events": scheduler_event,
                "events_dropped": 0,
                "trace_record_errors": 0,
            },
        },
    }


def test_two_double_call_responses_recover_at_the_post_close_fence() -> None:
    module = load_module()
    network = network_tool_response("response-1", "turn-1", 5.0)
    network += network_tool_response("response-2", "turn-2", 15.0)
    trace = [
        model_step(50, 5.1, both_pad=True, decision="sequential_conditional"),
        model_step(51, 5.2, both_pad=True, decision="buffered"),
        model_step(150, 15.1, both_pad=True, decision="pair_accepted"),
        model_step(151, 15.2, both_pad=True, decision="buffered"),
    ]

    report = module.analyze_session(network, trace)

    assert report["passed"] is True
    assert [item["status"] for item in report["recoveries"]] == [
        "recovered",
        "recovered",
    ]


def test_user_turn_that_eventually_restores_pad_cannot_false_green_a_stall() -> None:
    module = load_module()
    network = network_tool_response("response-1", "turn-1", 5.0)
    network += network_tool_response("response-2", "turn-2", 30.0)
    trace = [
        model_step(50, 5.1, both_pad=False, decision="sequential_conditional"),
        model_step(51, 5.2, both_pad=False, decision="sequential_conditional"),
        # A later user turn changes the upstream function channel back to PAD.
        model_step(100, 20.0, both_pad=True, decision="sequential_conditional"),
        model_step(101, 20.1, both_pad=True, decision="buffered"),
        model_step(200, 30.1, both_pad=True, decision="sequential_conditional"),
        model_step(201, 30.2, both_pad=True, decision="buffered"),
    ]

    report = module.analyze_session(network, trace)

    assert report["passed"] is False
    assert report["recoveries"][0]["status"] == "late_pad_return_after_response_close"
    assert report["recoveries"][0]["pad_return_frame_delta"] == 50


def test_missing_post_close_model_frame_fails_closed() -> None:
    module = load_module()
    network = network_tool_response("response-1", "turn-1", 5.0)
    network += network_tool_response("response-2", "turn-2", 15.0)
    trace = [model_step(10, 4.0, both_pad=True, decision="buffered")]

    report = module.analyze_session(network, trace)

    assert report["passed"] is False
    assert all(
        item["status"] == "no_model_step_after_response_close" for item in report["recoveries"]
    )


def test_campaign_requires_the_separate_non_tool_response(tmp_path: Path) -> None:
    module = load_module()
    session = tmp_path / "bounded-session-01"
    session.mkdir()
    network = [{"type": "session.created", "session_id": "session-1"}]
    network += network_tool_response("response-1", "turn-1", 5.0)
    network += network_tool_response("response-2", "turn-2", 15.0)
    (session / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in network), encoding="utf-8"
    )
    (session / "report.json").write_text(
        json.dumps(
            {
                "typed_completed": 3,
                "typed_answered": 2,
                "responses": [{}, {}],
            }
        ),
        encoding="utf-8",
    )
    trace_dir = tmp_path / "model/session-1"
    trace_dir.mkdir(parents=True)
    trace = [
        model_step(50, 5.1, both_pad=True, decision="sequential_conditional"),
        model_step(51, 5.2, both_pad=True, decision="buffered"),
        model_step(150, 15.1, both_pad=True, decision="sequential_conditional"),
        model_step(151, 15.2, both_pad=True, decision="buffered"),
    ]
    (trace_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in trace), encoding="utf-8"
    )

    report = module.analyze_campaign(
        tmp_path, session_glob="bounded-session-*", required_sessions=1
    )

    assert report["sessions"][0]["fixture"]["passed"] is False
    assert report["passed"] is False
