from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/finalize_eartts_component_asr.py")
        spec = importlib.util.spec_from_file_location("finalize_eartts_component_asr", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def write_wav(path: Path, rate: int = 16_000) -> None:
    samples = (np.sin(np.arange(3200) * 2 * np.pi * 220 / rate) * 2000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


class FakeBackend:
    def transcribe(self, values):
        return {"transcript": "The answer is five.", "token_ids": [1, 2, 3]}

    def provenance(self):
        return {"backend": "fake", "evaluator_image_id": f"sha256:{'a' * 64}"}


def pending_report(root: Path, module) -> dict:
    records = {}
    for name in ("streaming", "offline"):
        path = root / f"{name}.wav"
        write_wav(path)
        records[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": module.sha256_file(path),
        }
    return {
        "passed": False,
        "pre_asr_passed": True,
        "pending_external_asr": True,
        "fixture": {"reference_text": "The answer is five."},
        "external_asr": {"status": "pending", "max_wer": 0.5},
        "offline_diagnostic": {"external_asr": {"status": "pending"}},
        "asr_inputs": records,
    }


def test_finalize_eartts_report_hash_checks_and_scores_both_wavs(tmp_path: Path) -> None:
    module = load_module()
    report = module.finalize_report(pending_report(tmp_path, module), tmp_path, FakeBackend())

    assert report["passed"] is True
    assert report["pending_external_asr"] is False
    assert report["external_asr"]["transcript"] == "The answer is five."
    assert report["offline_diagnostic"]["external_asr"]["token_ids"] == [1, 2, 3]


def test_finalize_eartts_report_rejects_mutated_wav(tmp_path: Path) -> None:
    module = load_module()
    report = pending_report(tmp_path, module)
    with (tmp_path / "streaming.wav").open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(RuntimeError, match="identity mismatch"):
        module.finalize_report(report, tmp_path, FakeBackend())


def test_finalize_eartts_report_rejects_noncanonical_audio(tmp_path: Path) -> None:
    module = load_module()
    report = pending_report(tmp_path, module)
    path = tmp_path / "streaming.wav"
    write_wav(path, rate=8000)
    report["asr_inputs"]["streaming"] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": module.sha256_file(path),
    }

    with pytest.raises(RuntimeError, match="not PCM16 mono 16 kHz"):
        module.finalize_report(report, tmp_path, FakeBackend())
