from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/recover_offline_asr.py")
        spec = importlib.util.spec_from_file_location("recover_offline_asr", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.sin(np.arange(1600) * 2 * np.pi * 220 / 16_000) * 4000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.tobytes())


def test_recovery_command_uses_immutable_image_and_read_only_source(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    output = tmp_path / "recovery/multiturn"
    args = argparse.Namespace(
        source_run=source,
        asr_evaluator_image="asr:mutable-tag",
        asr_evaluator_image_id=f"sha256:{'a' * 64}",
        runtime_image_id=f"sha256:{'a' * 64}",
    )

    command = module.evaluator_command(args, "multiturn", output, None)

    assert f"{(source / 'multiturn').resolve()}:/qualification-input:ro" in command
    assert f"{output.resolve()}:/qualification-output" in command
    assert f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID=sha256:{'a' * 64}" in command
    assert not any("asr.nemo" in item for item in command)
    assert command[command.index("--network") + 1] == "none"
    assert args.runtime_image_id in command
    assert args.asr_evaluator_image not in command
    assert command[command.index("--output-dir") + 1] == "/qualification-output"


def test_snapshot_verification_detects_retained_audio_mutation(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    audio = source / "response.wav"
    write_wav(audio)
    paths = {"audio": audio}
    snapshot = {"audio": module.file_record(audio, source)}

    with audio.open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(RuntimeError, match="changed during recovery"):
        module.verify_snapshot(paths, snapshot)


def test_inherited_evidence_rejects_stored_red_browser_gate(tmp_path: Path) -> None:
    module = load_module()
    for name in module.INHERITED_EVIDENCE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "browser-attempts.json":
            path.write_text(json.dumps({"passed": False}), encoding="utf-8")
        elif name == "restart-cycles/report.json":
            path.write_text(json.dumps({"passed": True}), encoding="utf-8")
        elif name.endswith(".json"):
            path.write_text(json.dumps({"passed": False}), encoding="utf-8")
        else:
            path.write_text("retained evidence\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="browser or restart verdict is red"):
        module.collect_inherited_evidence(tmp_path)


def test_consumed_input_rejects_audio_path_escape(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    outside = tmp_path / "outside.wav"
    write_wav(outside)
    run_dir = tmp_path / "source/multiturn"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "report_kind": "multiturn_strict_v3",
                "fixture_generator": "Pocket TTS",
                "manifest": str(tmp_path / "source/manifest.json"),
                "expected_turn_count": 3,
                "max_input_wer": 0.35,
                "responses": [
                    {
                        "audio_path": str(outside),
                        "audio_bytes": 100,
                        "text": "valid",
                        "input_wer": 0.0,
                        "semantic_match": True,
                    }
                    for _ in range(3)
                ],
                "session_closed": {"reason": "client_stop"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "QUALIFICATIONS", (("multiturn", "multiturn_strict_v3", None),))

    with pytest.raises(RuntimeError, match="audio_path escapes"):
        module.collect_consumed_inputs(tmp_path / "source")
