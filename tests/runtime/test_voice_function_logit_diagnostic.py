from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_module():
    qualification = Path("tools/qualification").resolve()
    sys.path.insert(0, str(qualification))
    try:
        path = qualification / "voice_function_logit_diagnostic.py"
        spec = importlib.util.spec_from_file_location("voice_function_logit_diagnostic", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(qualification))


def _voice_report(schema: int, *, marker: str = "same", audio_samples: int = 1_764):
    report = {
        "schema": schema,
        "kind": "voice_function_logit_probe",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        "voice_reproduction_valid": True,
        "passed": True,
        "response_audio_samples": audio_samples,
        "response_audio_path": "/run/response.wav",
        "marker": marker,
        "runtime_provenance": {
            "checkpoint": {
                "fixed_stage_optimizations": {
                    "cpu_codec": {
                        "warmup_worker_ms": 1.5,
                        "worker": {"pid": 123, "status": "ready"},
                    }
                }
            }
        },
    }
    if schema == 2:
        report["function_logit_trace"] = {"entries": [{"model_frame": 1}]}
    return report


def test_voice_neutrality_accepts_stable_discrete_outcome_and_audio_envelope() -> None:
    module = load_module()
    control_a = _voice_report(1, audio_samples=1_764)
    traced = _voice_report(2, audio_samples=3_528)
    control_b = _voice_report(1, audio_samples=5_292)

    evidence = module.voice_neutrality(control_a, traced, control_b)

    assert evidence["passed"] is True
    assert evidence["discrete_outcome_equal"] is True


def test_voice_neutrality_recursively_strips_nested_trace_evidence() -> None:
    module = load_module()
    control_a = _voice_report(1)
    traced = _voice_report(2)
    control_b = _voice_report(1)
    control_a["nested"] = {}
    traced["nested"] = {"function_logit_trace": {"entries": [{"model_frame": 2}]}}
    control_b["nested"] = {}

    evidence = module.voice_neutrality(control_a, traced, control_b)

    assert evidence["passed"] is True


def test_voice_neutrality_retains_but_does_not_gate_on_operational_timing() -> None:
    module = load_module()
    reports = [_voice_report(1), _voice_report(2), _voice_report(1)]
    for report, frames in zip(reports, (45, 53, 48), strict=True):
        report["response_frames"] = frames
        report["quiescence"] = [
            {"response_position": position, "quiescent": position == frames - 1}
            for position in range(frames)
        ]
        report["response_pacing"] = {
            "clock": "monotonic_deadline",
            "frame_seconds": 0.08,
            "budget_frames": module.VOICE_RESPONSE_FRAME_LIMIT,
            "observed_frames": frames,
            "paced_transitions": frames - 1,
        }

    evidence = module.voice_neutrality(*reports)

    assert evidence["passed"] is True
    assert evidence["operational_timing"] == {
        "control_a": {
            "response_frames": 45,
            "paced_transitions": 44,
            "quiescence_records": 45,
        },
        "traced": {
            "response_frames": 53,
            "paced_transitions": 52,
            "quiescence_records": 53,
        },
        "control_b": {
            "response_frames": 48,
            "paced_transitions": 47,
            "quiescence_records": 48,
        },
    }


def _add_shifted_position_evidence(report: dict, *, offset: int) -> None:
    report["eou_boundary"] = {
        "engine_frame": offset + 2,
        "response_boundary": {
            "event": "start",
            "bos_frame": offset + 2,
            "text_eos_frame": None,
        },
    }
    report["sotc_evidence"] = {
        "after_eou_boundary": {
            "client_eou_sotc_committed_count": 1,
            "client_eou_sotc_committed_frame": offset + 2,
            "post_fc_client_bos_forced_count": 1,
            "post_fc_client_bos_forced_frame": offset + 3,
        }
    }
    report["settlement"] = {
        "model_steps": 2,
        "ending_blank_frames": 10,
        "steps": [
            {
                "model_step": index + 1,
                "frame_index": offset + index,
                "audio_frame_index_after": offset + index + 1,
                "perception_frame_idx_after": offset + index + 1,
                "rnnt_blank_frames": 9 + index,
                "function_calling": {
                    "client_eou_sotc_committed_count": 0,
                    "client_eou_sotc_committed_frame": None,
                },
                "vllm_request_positions": {
                    "eartts": {
                        "generated_tokens": offset + index + 2,
                        "session_positions": offset + index + 1,
                    },
                    "nano": {
                        "generated_tokens": offset + index + 2,
                        "session_positions": offset + index + 1,
                    },
                    "request_id": "1",
                },
            }
            for index in range(2)
        ],
    }


def test_voice_neutrality_normalizes_shifted_absolute_model_positions() -> None:
    module = load_module()
    reports = [_voice_report(1), _voice_report(2), _voice_report(1)]
    for report, offset in zip(reports, (40, 47, 41), strict=True):
        _add_shifted_position_evidence(report, offset=offset)

    evidence = module.voice_neutrality(*reports)

    assert evidence["passed"] is True


def test_voice_neutrality_still_rejects_model_count_divergence() -> None:
    module = load_module()
    reports = [_voice_report(1), _voice_report(2), _voice_report(1)]
    for report, offset in zip(reports, (40, 47, 41), strict=True):
        _add_shifted_position_evidence(report, offset=offset)
    reports[1]["settlement"]["model_steps"] = 3

    evidence = module.voice_neutrality(*reports)

    assert evidence["passed"] is False
    assert any(
        violation["path"] == "settlement/model_steps"
        and violation["reason"] == "trace_changed_control_stable_field"
        for violation in evidence["violations"]
    )


def test_voice_neutrality_rejects_trace_change_to_control_stable_field() -> None:
    module = load_module()
    control_a = _voice_report(1)
    traced = _voice_report(2, marker="changed")
    control_b = _voice_report(1)

    evidence = module.voice_neutrality(control_a, traced, control_b)

    assert evidence["passed"] is False
    assert any(
        violation["reason"] == "trace_changed_control_stable_field"
        for violation in evidence["violations"]
    )


def test_voice_neutrality_retains_structured_error_run() -> None:
    module = load_module()
    control_a = _voice_report(1)
    traced = {
        "schema": 2,
        "kind": "voice_function_logit_probe",
        "passed": False,
        "voice_reproduction_valid": False,
        "error": {"type": "RuntimeError", "message": "boom"},
    }
    control_b = _voice_report(1)

    evidence = module.voice_neutrality(control_a, traced, control_b)

    assert evidence["passed"] is False
    assert evidence["error_runs"] == ["on"]
    assert evidence["violations"] == [
        {"path": "runs", "reason": "voice_probe_error", "runs": ["on"]}
    ]


def test_probe_command_uses_immutable_image_and_read_only_fixture(tmp_path: Path) -> None:
    module = load_module()
    args = argparse.Namespace(
        cache_root=tmp_path / "cache",
        corpus=tmp_path / "corpus.json",
        runtime_image="voicechat:test",
        speech_root=tmp_path / "Speech",
    )
    image_id = "sha256:" + "a" * 64
    fixture = tmp_path / "fixture"
    output = tmp_path / "output"

    command = module.probe_command(
        args,
        image_id=image_id,
        fixture=fixture,
        output=output,
        name="on",
        top_k=10,
    )

    assert image_id in command
    assert "voicechat:test" not in command[-20:]
    assert f"{fixture.resolve()}:/qualification-input/fixture:ro" in command
    assert "S2S_FUNCTION_LOGIT_TRACE_TOPK=10" in command
    assert "VOICECHAT_NANO_PAD_PAIR=0" in command
    assert (
        f"VOICECHAT_VOICE_FUNCTION_MAX_FRAMES={module.VOICE_RESPONSE_FRAME_LIMIT}" in command
    )


def test_margin_summary_compares_only_settlement_and_eou_boundary() -> None:
    module = load_module()
    watched = {"sotc": 10, "pad": 20}

    def entry(phase: str, *, rank: int, margin: float, token: int) -> dict:
        return {
            "phase": phase,
            "sotc_rank": rank,
            "raw_function_token_id": token,
            "watched": {
                "sotc": {"logit": margin},
                "pad": {"logit": 0.0},
            },
        }

    summary = module._margin_summary(
        [
            entry("voice_input", rank=99, margin=99.0, token=10),
            entry("settlement", rank=2, margin=-2.0, token=20),
            entry("eou_boundary", rank=1, margin=0.5, token=10),
            entry("post_bos", rank=100, margin=100.0, token=10),
        ],
        watched,
    )

    assert summary == {
        "phases": ["settlement", "eou_boundary"],
        "entry_count": 2,
        "min_sotc_rank": 1,
        "max_sotc_minus_pad_logit": 0.5,
        "raw_sotc_argmax_count": 1,
    }
