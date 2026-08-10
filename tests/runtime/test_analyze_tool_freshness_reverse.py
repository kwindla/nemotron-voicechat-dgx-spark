from __future__ import annotations

import copy
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from nemotron_voicechat_runtime.artifacts import sha256_file
from nemotron_voicechat_runtime.tool_freshness_contract import experiment_sha256
from nemotron_voicechat_runtime.tool_freshness_fixture import self_hash as fixture_self_hash


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        path = Path("tools/qualification/analyze_tool_freshness_reverse.py")
        spec = importlib.util.spec_from_file_location(
            "analyze_tool_freshness_reverse", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def load_comparator_tests():
    path = Path("tests/runtime/test_compare_tool_freshness_modalities.py")
    spec = importlib.util.spec_from_file_location("comparator_test_support", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reverse_fixture(tmp_path: Path, monkeypatch):
    analyzer = load_module()
    support = load_comparator_tests()
    forward_comparator, forward_typed, forward_speech, forward_runtime = (
        support.prepare_pair(tmp_path / "forward")
    )
    forward = forward_comparator.compare(
        forward_typed, forward_speech, forward_runtime
    )
    _comparator, typed_path, speech_path, runtime_path = support.prepare_pair(
        tmp_path / "reverse"
    )
    new_image = f"sha256:{'b' * 64}"
    heartbeat = {
        "fc_async_heartbeat": True,
        "sigusr1_all_thread_dump": True,
    }
    reverse_manifest = None
    reports = {}
    for modality, path in (("typed", typed_path), ("speech", speech_path)):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["runtime_provenance"]["runtime_image_id"] = new_image
        report["runtime_provenance"]["diagnostics"] = heartbeat
        report["health_runtime_provenance"] = copy.deepcopy(
            report["runtime_provenance"]
        )
        report["health_snapshot"]["checkpoint"]["runtime_provenance"] = copy.deepcopy(
            report["runtime_provenance"]
        )
        report["health_snapshot"]["diagnostics"] = heartbeat
        manifest = report["fixture_manifest"]
        manifest["provenance"]["runtime_image_id"] = new_image
        manifest["sha256"] = fixture_self_hash(manifest)
        report["experiment_sha256"] = experiment_sha256(
            scenario_sha256=report["scenario_sha256"],
            fixture_manifest_sha256=manifest["sha256"],
        )
        reverse_manifest = copy.deepcopy(manifest)
        raw = copy.deepcopy(report)
        raw.pop("external_asr")
        raw.pop("runtime_gate_passed")
        raw.pop("external_asr_source_report")
        for response in raw["responses"]:
            response.pop("external_asr")
            response.pop("audio_dbfs", None)
        raw_path = tmp_path / "reverse/runtime" / modality / "report.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        report["external_asr_source_report"] = {"sha256": sha256_file(raw_path)}
        path.write_text(json.dumps(report), encoding="utf-8")
        reports[modality] = report
    assert reverse_manifest is not None
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.update(
        {
            "kind": "tool_freshness_reverse_order_diagnostic_runtime",
            "execution_order": ["speech", "typed"],
            "promotion_eligible": False,
            "scenario_sha256": reports["typed"]["scenario_sha256"],
            "fixture_manifest_sha256": reverse_manifest["sha256"],
            "experiment_sha256": reports["typed"]["experiment_sha256"],
            "session_ids": [
                reports["speech"]["session_id"],
                reports["typed"]["session_id"],
            ],
            "runtime_provenance": reports["typed"]["runtime_provenance"],
        }
    )
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    fixture_path = tmp_path / "reverse/fixtures/manifest.json"
    fixture_path.parent.mkdir()
    fixture_path.write_text(json.dumps(reverse_manifest), encoding="utf-8")
    forward_manifest = copy.deepcopy(reports["typed"]["fixture_manifest"])
    forward_manifest["provenance"]["runtime_image_id"] = forward[
        "runtime_provenance"
    ]["runtime_image_id"]
    forward_manifest["sha256"] = fixture_self_hash(forward_manifest)
    # Cross-run identity intentionally ignores image/source provenance and binds
    # the deterministic synthesis block plus the ordered carrier hashes.
    monkeypatch.setattr(
        analyzer,
        "validate_frozen_forward",
        lambda *_a, **_k: {
            "comparison": {"path": "forward", "sha256": "1" * 64},
            "fixture": {"path": "fixture", "sha256": "2" * 64},
            "comparison_data": forward,
            "fixture_data": forward_manifest,
        },
    )
    monkeypatch.setattr(
        analyzer,
        "validate_tool_freshness_fixtures",
        lambda *_a, **_k: (reverse_manifest, (b"pcm",) * 4),
    )
    return (
        analyzer,
        typed_path,
        speech_path,
        runtime_path,
        fixture_path,
        reverse_manifest,
    )


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ((0.1, 0.1, 0.1, 0.1), "tie"),
        ((0.1, 0.5, 0.5, 0.1), "consistent_with_speech_modality"),
        ((0.1, 0.5, 0.1, 0.5), "consistent_with_second_position"),
        ((0.1, 0.5, 0.25, 0.1), "unstable"),
    ],
)
def test_response_asr_wer_pattern_table(cells, expected) -> None:
    module = load_module()
    names = (
        "typed_forward",
        "speech_forward",
        "speech_reverse",
        "typed_reverse",
    )
    assert module.classify_wer_pattern(dict(zip(names, cells, strict=True))) == expected


def test_response_asr_wer_pattern_rejects_schema_drift() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="schema"):
        module.classify_wer_pattern({"typed_forward": 0.1})


def write_wav(path: Path, samples: np.ndarray, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.astype("<i2").tobytes())


def test_audio_envelope_all_inactive_counts_every_frame_as_trailing(tmp_path) -> None:
    module = load_module()
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(16_640 + 1, dtype=np.int16))

    result = module.audio_envelope(path)

    assert result["first_active_frame"] is None
    assert result["last_active_frame"] is None
    assert result["trailing_frame_count"] == result["total_frame_count"]
    assert result["final_frame_sample_count"] == 1
    assert result["gating"] is False


def test_audio_envelope_includes_partial_frame_without_padding(tmp_path) -> None:
    module = load_module()
    path = tmp_path / "tail.wav"
    samples = np.zeros(1281, dtype=np.int16)
    samples[-1] = 32767
    write_wav(path, samples)

    result = module.audio_envelope(path)

    assert result["samples_per_full_frame"] == 1280
    assert result["total_frame_count"] == 2
    assert result["final_frame_sample_count"] == 1
    assert result["last_active_frame"] == 1
    assert result["trailing_frame_count"] == 0


def test_audio_envelope_rejects_empty_wav(tmp_path) -> None:
    module = load_module()
    path = tmp_path / "empty.wav"
    write_wav(path, np.zeros(0, dtype=np.int16))
    with pytest.raises(ValueError, match="contain"):
        module.audio_envelope(path)


def test_frozen_forward_requires_both_file_and_internal_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    comparison_path = tmp_path / "comparison.json"
    fixture_path = tmp_path / "manifest.json"
    comparison_path.write_text("{}", encoding="utf-8")
    fixture = {
        "sha256": module.FORWARD_FIXTURE_SELF_HASH,
        "provenance": {"runtime_image_id": f"sha256:{'a' * 64}"},
    }
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    stored = {
        "sha256": module.FORWARD_COMPARISON_SELF_HASH,
        "passed": False,
        "scenario_sha256": "3" * 64,
        "fixture_manifest_sha256": module.FORWARD_FIXTURE_SELF_HASH,
        "arms": {
            "typed": {"report": {"path": "/typed"}},
            "speech": {"report": {"path": "/speech"}},
        },
        "runtime_report": {"path": "/runtime"},
    }
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: (
            module.FORWARD_COMPARISON_SHA256
            if Path(path).name == "comparison.json"
            else module.FORWARD_FIXTURE_FILE_SHA256
        ),
    )
    monkeypatch.setattr(module, "read_report", lambda _path: stored)
    monkeypatch.setattr(module, "compare", lambda *_a: stored)
    monkeypatch.setattr(
        module,
        "validate_tool_freshness_fixtures",
        lambda *_a, **_k: (fixture, (b"pcm",) * 4),
    )

    result = module.validate_frozen_forward(comparison_path, fixture_path)
    assert result["comparison_data"] is stored

    monkeypatch.setattr(module, "sha256_file", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="comparison file hash"):
        module.validate_frozen_forward(comparison_path, fixture_path)

    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: (
            module.FORWARD_COMPARISON_SHA256
            if Path(path).name == "comparison.json"
            else module.FORWARD_FIXTURE_FILE_SHA256
        ),
    )
    stored["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="comparison self-hash"):
        module.validate_frozen_forward(comparison_path, fixture_path)


def test_full_reverse_analyzer_is_complete_but_permanently_red(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer, typed, speech, runtime, fixture, _manifest = reverse_fixture(
        tmp_path, monkeypatch
    )

    result = analyzer.analyze(
        forward_comparison=tmp_path / "unused-forward.json",
        forward_fixture=tmp_path / "unused-fixture.json",
        reverse_typed=typed,
        reverse_speech=speech,
        reverse_runtime=runtime,
        reverse_fixture=fixture,
    )

    assert result["diagnostic_complete"] is True
    assert result["passed"] is False
    assert result["promotion_eligible"] is False
    assert result["runtime_provenance_mismatch_confounds_exact_ab"] is True
    assert result["reverse_heartbeat_diagnostics"]["passed"] is True
    assert result["fixture_pair_identity"]["passed"] is True
    assert result["sha256"] == analyzer.self_hash(result)
    assert len(result["cases"]) == 4
    assert all(
        set(case["response_audio_envelope"])
        == {
            "typed_forward",
            "speech_forward",
            "speech_reverse",
            "typed_reverse",
        }
        for case in result["cases"]
    )
    assert all(
        envelope["source_wav"]["sha256"]
        for case in result["cases"]
        for envelope in case["response_audio_envelope"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime_order",
        "promotion_flag",
        "session_reuse",
        "source_sha",
        "heartbeat",
        "carrier",
    ],
)
def test_full_reverse_analyzer_rejects_identity_mutations(
    mutation: str, tmp_path: Path, monkeypatch
) -> None:
    analyzer, typed, speech, runtime, fixture, manifest = reverse_fixture(
        tmp_path, monkeypatch
    )
    if mutation in {"runtime_order", "promotion_flag", "session_reuse"}:
        value = json.loads(runtime.read_text(encoding="utf-8"))
        if mutation == "runtime_order":
            value["execution_order"] = ["typed", "speech"]
        elif mutation == "promotion_flag":
            value["promotion_eligible"] = True
        else:
            value["session_ids"][1] = value["session_ids"][0]
        runtime.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "source_sha":
        value = json.loads(typed.read_text(encoding="utf-8"))
        value["external_asr_source_report"]["sha256"] = "0" * 64
        typed.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "heartbeat":
        value = json.loads(typed.read_text(encoding="utf-8"))
        value["health_snapshot"].pop("diagnostics")
        raw_path = tmp_path / "reverse/runtime/typed/report.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["health_snapshot"].pop("diagnostics")
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        value["external_asr_source_report"]["sha256"] = sha256_file(raw_path)
        typed.write_text(json.dumps(value), encoding="utf-8")
    else:
        manifest["fixtures"][0]["pcm"]["sha256"] = "0" * 64

    expected_message = "heartbeat-enabled" if mutation == "heartbeat" else None
    with pytest.raises(ValueError, match=expected_message):
        analyzer.analyze(
            forward_comparison=tmp_path / "unused-forward.json",
            forward_fixture=tmp_path / "unused-fixture.json",
            reverse_typed=typed,
            reverse_speech=speech,
            reverse_runtime=runtime,
            reverse_fixture=fixture,
        )
