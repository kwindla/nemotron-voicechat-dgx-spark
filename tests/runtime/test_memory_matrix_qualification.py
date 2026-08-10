from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES


def load_module():
    path = Path("tools/qualification/memory_matrix_strict_v3.py")
    spec = importlib.util.spec_from_file_location("memory_matrix_strict_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def runtime_provenance(image: str = "a", source: str = "b") -> dict:
    return {
        "runtime_image_id": f"sha256:{image * 64}",
        "source_sha256": {
            name: source * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES
        },
    }


def matrix_manifest() -> dict:
    specs = (
        ("voice-to-voice", "voice", "voice", "sapphire"),
        ("voice-to-text", "voice", "text", "topaz"),
        ("text-to-voice", "text", "voice", "cobalt"),
        ("text-to-text", "text", "text", "amber"),
        ("system-to-voice", "system", "voice", "garnet"),
        ("system-to-text", "system", "text", "indigo"),
    )
    cases = []
    for case_id, seed, recall, secret in specs:
        case = {
            "case_id": case_id,
            "seed_modality": seed,
            "recall_modality": recall,
            "secret": secret,
            "seed_text": f"Remember that the codeword is {secret}.",
            "recall_text": "What codeword did I ask you to remember?",
        }
        if seed == "voice":
            case["seed_audio_path"] = f"/fixtures/{case_id}-seed.wav"
        if recall == "voice":
            case["recall_audio_path"] = "/fixtures/recall.wav"
        cases.append(case)
    return {
        "memory_matrix": {
            "cases": cases,
            "negative_control": {
                "case_id": "unseeded-to-text",
                "seed_modality": "none",
                "recall_modality": "text",
                "recall_text": "What codeword did I ask you to remember?",
            },
        }
    }


def test_memory_matrix_has_unique_secrets_full_surface_and_negative_control() -> None:
    module = load_module()
    cases = module.load_cases(matrix_manifest())
    seeded = [case for case in cases if case.secret]
    negative = [case for case in cases if not case.secret]

    assert {(case.seed, case.recall) for case in seeded} == {
        ("voice", "voice"),
        ("voice", "text"),
        ("text", "voice"),
        ("text", "text"),
        ("system", "voice"),
        ("system", "text"),
    }
    assert len({case.secret for case in seeded}) == 6
    assert [(case.seed, case.recall) for case in negative] == [("none", "text")]


def test_matrix_rejects_duplicate_secrets_and_prompt_leaks() -> None:
    module = load_module()
    manifest = matrix_manifest()
    manifest["memory_matrix"]["cases"][1]["secret"] = "sapphire"
    with pytest.raises(ValueError, match="unique"):
        module.load_cases(manifest)

    manifest = matrix_manifest()
    manifest["memory_matrix"]["cases"][0]["recall_text"] += " sapphire"
    with pytest.raises(ValueError, match="leaks"):
        module.load_cases(manifest)


def test_secret_matching_is_word_bounded_and_system_prompt_is_parameterized() -> None:
    module = load_module()

    assert module.contains_secret("The remembered value was SAPPHIRE.", "sapphire")
    assert not module.contains_secret("The remembered values were sapphires.", "sapphire")
    assert module.contains_secret(module.system_instructions("topaz"), "topaz")
    assert not module.contains_secret(module.DEFAULT_INSTRUCTIONS, "topaz")


def test_input_wer_detects_missing_or_substituted_words() -> None:
    module = load_module()

    assert module.word_error_rate("what codeword", "what codeword") == 0
    assert module.word_error_rate("what codeword", "what password") == 0.5


def test_flattened_responses_require_own_secret_and_exclude_others(
    tmp_path: Path,
) -> None:
    module = load_module()
    audio = tmp_path / "voice-to-text/response.wav"
    cases = [
        {
            "case_id": "voice-to-text",
            "secret": "topaz",
            "turns": [
                {"text": "Acknowledged.", "audio_path": str(audio), "audio_bytes": 10},
                {"text": "Topaz.", "audio_path": str(audio), "audio_bytes": 10},
            ],
        },
        {
            "case_id": "unseeded-to-text",
            "secret": None,
            "turns": [{"text": "No codeword.", "audio_path": str(audio), "audio_bytes": 10}],
        },
    ]

    responses = module.flatten_responses(cases, tmp_path, ("sapphire", "topaz"))

    assert [response["turn_role"] for response in responses] == [
        "seed",
        "recall",
        "recall",
    ]
    assert responses[0]["expected_response_any"] == []
    assert responses[1]["expected_response_any"] == ["topaz"]
    assert responses[1]["expected_response_none"] == ["sapphire"]
    assert responses[2]["expected_response_any"] == []
    assert responses[2]["expected_response_none"] == ["sapphire", "topaz"]
    assert responses[0]["audio_path"] == "voice-to-text/response.wav"


def test_full_runtime_gate_requires_negative_control_last_and_nonvacuous() -> None:
    module = load_module()
    cases = [{"passed": True, "negative_control": False} for _ in range(6)] + [
        {"passed": True, "negative_control": True}
    ]
    report = {
        "fresh_session_per_case": True,
        "runtime_provenance": runtime_provenance(),
        "runtime_provenance_consistent": True,
        "negative_control_ran_after_seeded": True,
        "expected_case_count": 7,
        "cases": cases,
    }

    assert module.runtime_gate_passed(report)
    report["cases"][-1]["passed"] = False
    assert not module.runtime_gate_passed(report)
    report["cases"][-1]["passed"] = True
    report["cases"][-1]["negative_control"] = False
    assert not module.runtime_gate_passed(report)


def test_runtime_gate_rejects_missing_or_mixed_provenance() -> None:
    module = load_module()
    cases = [{"passed": True, "negative_control": False} for _ in range(6)] + [
        {"passed": True, "negative_control": True}
    ]
    report = {
        "fresh_session_per_case": True,
        "negative_control_ran_after_seeded": True,
        "expected_case_count": 7,
        "cases": cases,
        "runtime_provenance": runtime_provenance(),
        "runtime_provenance_consistent": False,
    }

    assert not module.runtime_gate_passed(report)
    report["runtime_provenance_consistent"] = True
    assert module.runtime_gate_passed(report)
    report["runtime_provenance"] = {}
    assert not module.runtime_gate_passed(report)


def test_session_runtime_provenance_reads_protocol_session_envelope() -> None:
    module = load_module()
    provenance = runtime_provenance()
    created = {"session": {"checkpoint": {"runtime_provenance": provenance}}}

    assert module.session_runtime_provenance(created) == provenance
    assert module.session_runtime_provenance({"checkpoint": provenance}) is None


def test_runtime_provenance_requires_image_id_and_every_source_hash() -> None:
    module = load_module()
    provenance = runtime_provenance()

    assert module.valid_runtime_provenance(provenance)
    provenance["runtime_image_id"] = "mutable-tag"
    assert not module.valid_runtime_provenance(provenance)
    provenance = runtime_provenance()
    provenance["source_sha256"].pop(next(iter(RUNTIME_PROVENANCE_REQUIRED_SOURCES)))
    assert not module.valid_runtime_provenance(provenance)
