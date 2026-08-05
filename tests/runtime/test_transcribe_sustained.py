from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np


def load_module():
    path = Path("tools/qualification/transcribe_sustained.py")
    spec = importlib.util.spec_from_file_location("transcribe_sustained", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_word_error_rate_normalizes_spoken_numbers() -> None:
    module = load_module()
    assert module.word_error_rate("The answer is 20.", "the answer is twenty") == 0
    assert module.word_error_rate("one two", "one four") == 0.5


def test_semantic_candidate_accepts_literal_and_spoken_utc_time() -> None:
    module = load_module()
    assert module.semantic_candidate_match("The result is sapphire.", "sapphire")
    assert module.semantic_candidate_match(
        "The current UTC time is 22 hours and 6 minutes UTC.", "22:06 UTC"
    )
    assert module.semantic_candidate_match("The time is 22 hours 6 minutes UTC.", "22:06 UTC")
    assert not module.semantic_candidate_match(
        "The current UTC time is 22 hours and 7 minutes UTC.", "22:06 UTC"
    )
    assert not module.semantic_candidate_match("It is 22:06 local time.", "22:06 UTC")


def test_runtime_gate_verdict_is_recomputed_independently_of_prior_asr() -> None:
    module = load_module()
    report = {
        "passed": False,
        "source_frames": 15_000,
        "sent_frames": 13_000,
        "metric_sequence_complete": True,
        "queue": {"passed": True},
        "typed_jobs": {str(index): {} for index in range(18)},
        "typed_completed": 18,
        "typed_answered": 18,
        "typed_unanswered": 0,
        "allowed_unanswered_at_fp32_base_rate": 1,
        "responses": [{} for _ in range(18)],
        "unexpected_errors": [],
        "expected_session_limit_error_count": 1,
        "expected_session_position_limit": True,
        "session_closed": {"reason": "session_position_limit"},
        "external_asr": {"passed": False},
    }
    assert module.runtime_gate_passed(report)
    report["queue"]["passed"] = False
    assert not module.runtime_gate_passed(report)


def test_audio_level_and_resampling(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source.wav"
    tone = (np.sin(np.arange(22_050) * 2 * np.pi * 440 / 22_050) * 0.1 * 32767).astype("<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22_050)
        output.writeframes(tone.tobytes())
    values, source_rate = module.read_and_resample(source)
    assert source_rate == 22_050
    assert len(values) == 16_000
    assert -24 < module.dbfs(values) < -22


def test_container_audio_path_falls_back_to_mounted_run_directory(tmp_path: Path) -> None:
    module = load_module()
    mounted = tmp_path / "response-001.wav"
    mounted.touch()
    assert module.resolve_audio_path(tmp_path, "/host/elsewhere/response-001.wav") == mounted
