from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def load_module():
    path = Path("tools/qualification/sustained_strict_v3.py")
    spec = importlib.util.spec_from_file_location("sustained_strict_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_queue_model_passes_flat_realtime_and_fails_positive_drift() -> None:
    module = load_module()
    flat = module.queue_model([70.0] * 4_000)
    assert flat["passed"]
    drift = module.queue_model([70 + index * 0.01 for index in range(4_000)])
    assert not drift["passed"]
    assert drift["linear_slope_ms_per_minute"] > 1


def test_timing_stats_keeps_p95_separate_from_mean() -> None:
    module = load_module()
    result = module.timing_stats([60.0] * 95 + [100.0] * 5)
    assert result["mean_ms"] == 62.0
    assert result["p95_ms"] >= 60.0
    assert result["over_80ms"] == 5


def test_max_typed_prompts_preserves_a_silence_only_tail() -> None:
    module = load_module()
    assert module.typed_prompt_index(0, 10, 3) == 0
    assert module.typed_prompt_index(10, 10, 3) == 1
    assert module.typed_prompt_index(20, 10, 3) == 2
    assert module.typed_prompt_index(30, 10, 3) is None
    assert module.typed_prompt_index(31, 10, 3) is None
    assert module.typed_prompt_index(30, 10, None) == 3


def test_step4c_health_check_and_fixed_contract(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"status":"ready","active_client":false}'),
    )
    assert module.read_health("http://127.0.0.1:8786/health")["active_client"] is False
    assert module.QUALIFICATION_FIXTURE == "step4c-l1"
    assert (module.STEP4C_WARMUP_SECONDS, module.STEP4C_DURATION_SECONDS) == (10.0, 120.0)

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"status":"ready","active_client":true}'),
    )
    with pytest.raises(RuntimeError, match="single-client"):
        module.read_health("http://127.0.0.1:8786/health")


def test_tool_only_prompt_mode_repeats_real_tool_calls() -> None:
    module = load_module()

    assert module.typed_prompts("mixed") == module.DEFAULT_TYPED_PROMPTS
    prompts = module.typed_prompts("tool-only")
    assert len(prompts) == 4
    assert all("tool" in prompt.lower() for prompt in prompts)
    assert all(module.prompt_expects_tool("tool-only", prompt) for prompt in prompts)
    assert module.prompt_expects_tool("mixed", module.DEFAULT_TYPED_PROMPTS[-1])
    assert not module.prompt_expects_tool("mixed", module.DEFAULT_TYPED_PROMPTS[0])
    try:
        module.typed_prompts("unknown")
    except ValueError as exc:
        assert "unsupported typed prompt mode" in str(exc)
    else:
        raise AssertionError("unknown prompt mode was accepted")


def test_tool_definition_and_result_preserve_mixed_mode_isolation() -> None:
    module = load_module()

    mixed_definition = module.tool_definition("mixed")
    mixed_expected, mixed_semantic, mixed_output = module.tool_result("mixed", now="15:09 UTC")
    assert mixed_definition["description"] == "Get the current clock time in UTC."
    assert mixed_expected == "15:09 UTC"
    assert mixed_semantic is False
    assert mixed_output == {"timezone": "UTC", "spoken": "15:09 UTC"}

    focused_definition = module.tool_definition("tool-only")
    focused_expected, focused_semantic, focused_output = module.tool_result(
        "tool-only", verification_word="river", ordinal_word="first"
    )
    assert focused_definition["name"] == "get_fresh_verification_word"
    assert "fresh verification word" in focused_definition["description"]
    assert focused_expected == "river"
    assert focused_semantic is True
    assert focused_output == {
        "spoken": "River is the first fresh word, ready now.",
    }


def test_tool_response_channel_gate_requires_eotr_and_idle_function_head() -> None:
    module = load_module()
    response = {
        "turn_id": "turn-1",
        "response_id": "response-1",
        "expected_response_any": ["13:03 UTC"],
        "done_reason": "no_text_since_bos_watchdog",
        "text": "It is 13:03 UTC.",
        "audio_bytes": 1024,
        "expected_text_semantic_required": False,
    }
    observation = {
        "turn_id": "turn-1",
        "transport_frame": 100,
        "observed": True,
        "observed_frame": 70,
        "wait_steps": 1,
        "timeout": False,
        "eotr_token_id": 22,
        "first_post_drain_token_id": 22,
        "first_post_drain_token_frame": 70,
        "unexpected_token_id": None,
        "unexpected_token_frame": None,
        "feedback_token_id": 22,
        "feedback_frame": 70,
        "feedback_committed": True,
    }
    metric = {
        "response_id": "response-1",
        "transport_frame": 100,
        "turn_state": {"function_calling": {"effective_token_is_pad": True}},
    }
    audible_metric = {
        **metric,
        "transport_frame": 101,
        "output_audio": {"rms_dbfs": -35.0},
        "turn_state": {
            "function_calling": {"effective_token_is_pad": True},
            "agent_silence_watchdog": {"threshold_dbfs": -90.0},
        },
    }

    checks, passed = module.evaluate_tool_response_channels(
        [response], [metric, audible_metric], [observation]
    )
    assert passed
    assert checks[0]["eotr_valid"]
    assert checks[0]["function_channel_idle"]
    assert checks[0]["bounded_watchdog_close"]
    assert checks[0]["post_eotr_audible"]
    assert checks[0]["audible_post_eotr_frames"] == [101]

    semantic_response = {
        **response,
        "expected_response_any": ["cobalt"],
        "expected_text_semantic_required": True,
    }
    checks, passed = module.evaluate_tool_response_channels(
        [semantic_response], [metric, audible_metric], [observation]
    )
    assert not passed
    assert not checks[0]["text_semantic_match"]
    semantic_response["text"] = "Fresh verification code cobalt."
    checks, passed = module.evaluate_tool_response_channels(
        [semantic_response], [metric, audible_metric], [observation]
    )
    assert passed
    assert checks[0]["text_semantic_match"]

    silent_metric = {
        **audible_metric,
        "output_audio": {"rms_dbfs": -100.0},
    }
    checks, passed = module.evaluate_tool_response_channels(
        [response], [metric, silent_metric], [observation]
    )
    assert not passed
    assert not checks[0]["post_eotr_audible"]

    leaked_metric = {
        **metric,
        "transport_frame": 102,
        "turn_state": {"function_calling": {"effective_token_is_pad": False}},
    }
    checks, passed = module.evaluate_tool_response_channels(
        [response], [metric, audible_metric, leaked_metric], [observation]
    )
    assert not passed
    assert checks[0]["function_channel_nonpad_frames"] == [102]

    pre_eotr_nonpad = {**leaked_metric, "transport_frame": 99}
    checks, passed = module.evaluate_tool_response_channels(
        [response], [pre_eotr_nonpad, metric, audible_metric], [observation]
    )
    assert passed
    assert checks[0]["function_channel_nonpad_frames"] == []

    malformed = {**observation, "first_post_drain_token_id": 999}
    checks, passed = module.evaluate_tool_response_channels(
        [response], [metric, audible_metric], [malformed]
    )
    assert not passed
    assert not checks[0]["eotr_valid"]


def test_tool_cycle_cardinality_requires_every_boundary_and_forced_bos() -> None:
    module = load_module()
    checks = [{"turn_id": f"turn-{index}"} for index in range(4)]
    metrics = [
        {
            "turn_state": {
                "function_calling": {
                    "client_eou_sotc_committed_count": 4,
                    "post_fc_client_bos_forced_count": 4,
                    "eotr_observed_count": 4,
                }
            }
        },
        {
            "turn_state": {
                "function_calling": {"effective_token_is_pad": True},
                "agent_silence_watchdog": {"agent_open": False},
            }
        },
    ]

    result = module.evaluate_tool_cycle_cardinality(
        requested_tool_prompts=4,
        tool_response_checks=checks,
        metrics=metrics,
    )
    assert result["passed"]

    for requested, observed_checks, committed, forced, eotr_observed in (
        (4, checks[:3], 4, 4, 4),
        (4, checks, 3, 4, 4),
        (4, checks, 4, 3, 4),
        (4, checks, 4, 4, 3),
        (4, checks, None, None, None),
        (0, [], 0, 0, 0),
    ):
        result = module.evaluate_tool_cycle_cardinality(
            requested_tool_prompts=requested,
            tool_response_checks=observed_checks,
            metrics=[
                {
                    "turn_state": {
                        "function_calling": {
                            "client_eou_sotc_committed_count": committed,
                            "post_fc_client_bos_forced_count": forced,
                            "eotr_observed_count": eotr_observed,
                        }
                    }
                }
            ],
        )
        assert not result["passed"]


def test_source_completion_requires_the_declared_close_contract() -> None:
    module = load_module()
    assert module.source_completion_passes(
        sent_frames=12_000,
        total_frames=15_000,
        close_reason="session_position_limit",
        expect_limit=True,
    )
    assert not module.source_completion_passes(
        sent_frames=15_000,
        total_frames=15_000,
        close_reason="client_stop",
        expect_limit=True,
    )
    assert module.source_completion_passes(
        sent_frames=2_250,
        total_frames=2_250,
        close_reason="client_stop",
        expect_limit=False,
    )
    assert not module.source_completion_passes(
        sent_frames=2_250,
        total_frames=2_250,
        close_reason="client_requested",
        expect_limit=False,
    )


def test_response_job_attribution_falls_back_to_late_turn_mapping() -> None:
    module = load_module()
    response = {"turn_id": "turn-4", "job_id": None}
    assert module.resolve_response_job_id(response, {"turn-4": "qual-2250"}) == "qual-2250"
    response["job_id"] = "explicit-job"
    assert module.resolve_response_job_id(response, {"turn-4": "qual-2250"}) == "explicit-job"


def test_expected_sender_close_is_narrow() -> None:
    module = load_module()
    closed = {"reason": "session_position_limit"}
    assert module.expected_sender_close(
        receiver_done=True, session_closed=closed, expect_limit=True
    )
    assert not module.expected_sender_close(
        receiver_done=False, session_closed=closed, expect_limit=True
    )
    assert not module.expected_sender_close(
        receiver_done=True, session_closed=closed, expect_limit=False
    )
    assert not module.expected_sender_close(
        receiver_done=True,
        session_closed={"reason": "internal_error"},
        expect_limit=True,
    )


def test_protocol_error_classification_only_exempts_exact_expected_limit() -> None:
    module = load_module()
    limit = {
        "type": "error",
        "error": {
            "code": "session_position_limit",
            "fatal": True,
            "model_frames": 12_000,
            "max_model_frames": 12_000,
        },
    }
    closed = {"reason": "session_position_limit"}
    unexpected, count = module.classify_protocol_errors(
        [limit], expect_limit=True, session_closed=closed
    )
    assert unexpected == []
    assert count == 1

    variants = [
        ({**limit, "error": {**limit["error"], "fatal": False}}, True, closed),
        ({**limit, "error": {**limit["error"], "model_frames": 11_999}}, True, closed),
        ({**limit, "error": {**limit["error"], "max_model_frames": None}}, True, closed),
        ({**limit, "error": {**limit["error"], "code": "internal_error"}}, True, closed),
        (limit, False, closed),
        (limit, True, {"reason": "internal_error"}),
    ]
    for event, expect_limit, session_closed in variants:
        unexpected, count = module.classify_protocol_errors(
            [event], expect_limit=expect_limit, session_closed=session_closed
        )
        assert unexpected == [event]
        assert count == 0

    unexpected, count = module.classify_protocol_errors(
        [limit, limit], expect_limit=True, session_closed=closed
    )
    assert unexpected == [limit]
    assert count == 1


def test_exception_evidence_never_overwrites_an_assembled_report(tmp_path: Path) -> None:
    module = load_module()
    report = tmp_path / "report.json"
    report.write_text('{"passed": true}\n', encoding="utf-8")
    failure = {"passed": False, "errors": [{"type": "LateFailure"}]}

    module.write_exception_evidence(tmp_path, failure)

    assert report.read_text(encoding="utf-8") == '{"passed": true}\n'
    assert '"LateFailure"' in (tmp_path / "error.json").read_text(encoding="utf-8")

    empty = tmp_path / "empty"
    module.write_exception_evidence(empty, failure)
    assert (empty / "report.json").read_text() == (empty / "error.json").read_text()
