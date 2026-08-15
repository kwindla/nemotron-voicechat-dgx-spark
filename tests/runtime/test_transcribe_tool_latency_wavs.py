from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_driver(monkeypatch):
    sustained = ModuleType("transcribe_sustained")
    sustained.classify_audio = object()
    sustained.read_and_resample = object()
    sustained.word_error_rate = object()
    evaluator = ModuleType("nemotron_voicechat_asr_evaluator")
    evaluator.NemotronEnglishAsr = object
    monkeypatch.setitem(sys.modules, "transcribe_sustained", sustained)
    monkeypatch.setitem(sys.modules, "nemotron_voicechat_asr_evaluator", evaluator)
    return runpy.run_path(
        str(REPO_ROOT / "tools/qualification/transcribe_tool_latency_wavs.py")
    )


def test_expected_words_uses_explicit_safety_scenario_not_directory_suffix(
    monkeypatch,
):
    driver = _load_driver(monkeypatch)
    expected_words = driver["expected_words"]

    assert expected_words(
        "step2-final-safety-r2",
        "result_injection_interruption",
        {"payload": {}},
    ) == {"stopped"}
    assert expected_words(
        "step2-final-safety-r2",
        "chained_two_call",
        {"payload": {}},
    ) == set()
    assert expected_words(
        "step2-final-fast-n10",
        None,
        {"payload": {"verification_word": "River"}},
    ) == {"river"}


def test_required_word_normalizer_is_versioned_and_does_not_change_wer(monkeypatch):
    driver = _load_driver(monkeypatch)
    normalized_words = driver["normalized_words"]

    assert normalized_words("harbour", version="exact_v1") == {"harbour"}
    assert normalized_words("harbour", version="english_spelling_v2") == {"harbor"}
    assert normalized_words("harvest", version="english_spelling_v2") == {"harvest"}
    assert driver["REQUIRED_WORD_NORMALIZERS"]["english_spelling_v2"] == {
        "harbour": "harbor"
    }

    raw_arguments = []
    driver["evaluate_response"].__globals__["word_error_rate"] = (
        lambda reference, hypothesis: raw_arguments.append((reference, hypothesis)) or 0.2
    )
    wer, word_present, passed = driver["evaluate_response"](
        reference="The fresh word is Harbor.",
        hypothesis="The fresh word is harbour.",
        classification="speech",
        required=True,
        required_words={"harbor"},
        required_word_normalizer="english_spelling_v2",
    )
    assert raw_arguments == [
        ("The fresh word is Harbor.", "The fresh word is harbour.")
    ]
    assert (wer, word_present, passed) == (0.2, True, True)

    _, word_present, passed = driver["evaluate_response"](
        reference="The fresh word is Harbor.",
        hypothesis="The fresh word is harvest.",
        classification="speech",
        required=True,
        required_words={"harbor"},
        required_word_normalizer="english_spelling_v2",
    )
    assert (word_present, passed) == (False, False)


def test_versioned_normalizer_predecessor_is_exactly_bound(monkeypatch, tmp_path):
    driver = _load_driver(monkeypatch)
    report = {
        "schema": 1,
        "kind": "tool_call_latency_step2_asr",
        "passed": False,
    }
    report["self_sha256"] = driver["report_self_sha256"](report)
    path = tmp_path / "red.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    binding = driver["predecessor_record"](path)
    assert binding["self_sha256"] == report["self_sha256"]
    assert binding["file_sha256"] == driver["sha256_file"](path)

    report["passed"] = True
    report["self_sha256"] = driver["report_self_sha256"](report)
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        driver["predecessor_record"](path)
    except RuntimeError as error:
        assert "red verdict" in str(error)
    else:
        raise AssertionError("a green report cannot be used as the red predecessor")
