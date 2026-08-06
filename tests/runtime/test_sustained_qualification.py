from __future__ import annotations

import importlib.util
from pathlib import Path


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
