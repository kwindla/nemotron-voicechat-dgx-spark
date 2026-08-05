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
