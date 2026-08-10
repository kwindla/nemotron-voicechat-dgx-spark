from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nemotron_voicechat_runtime.direct_semantic_probe import _transaction_checks
from nemotron_voicechat_runtime.semantic_corpus import (
    REQUIRED_CATEGORIES,
    contains_phrase,
    evaluate_pocket_carrier,
    evaluate_semantic_case,
    load_semantic_corpus,
)

CORPUS = Path("tools/qualification/direct_text_semantic_corpus.json")


def test_checked_in_corpus_covers_every_category_and_tool_schema() -> None:
    corpus = load_semantic_corpus(CORPUS)

    categories = {category for case in corpus["cases"] for category in case["categories"]}
    expected_tools = {
        case["predicate"]["tool"]["name"]
        for case in corpus["cases"]
        if "tool" in case["predicate"]
    }

    assert len(corpus["cases"]) == 20
    assert REQUIRED_CATEGORIES <= categories
    assert expected_tools == {tool["name"] for tool in corpus["tools"]}
    assert sum(case["predicate"]["safety_critical"] for case in corpus["cases"]) == 4


def test_phrase_matching_is_casefolded_word_bounded_and_punctuation_tolerant() -> None:
    assert contains_phrase("The result is AX-17/B.", "ax 17 b")
    assert contains_phrase("TWENTY-FIVE", "twenty five")
    assert not contains_phrase("Kestrels", "kestrel")


def test_punctuation_case_requires_the_literal_identifier() -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == "c003")

    assert evaluate_semantic_case(
        case, response_text="AX-17/B", tool_calls=[]
    )["passed"]
    assert not evaluate_semantic_case(
        case, response_text="AX 17 B", tool_calls=[]
    )["passed"]


@pytest.mark.parametrize(
    ("case_id", "transcript"),
    [
        ("c003", "repeat this identifier exactly AX seventeen B"),
        ("c005", "do not book anything just tell me the color in this sentence"),
        ("c006", "the test value is four to seventy two"),
    ],
)
def test_pocket_carrier_marks_lost_predicate_content_inconclusive(
    case_id: str, transcript: str
) -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == case_id)

    result = evaluate_pocket_carrier(case, transcript=transcript)

    assert not result["passed"]
    assert result["classification"] == "carrier_inconclusive"


@pytest.mark.parametrize("case_id", ["c004", "c010"])
def test_pocket_carrier_preserves_exact_reference_and_safety_prompts(case_id: str) -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == case_id)

    result = evaluate_pocket_carrier(case, transcript=case["prompt"])

    assert result["passed"]
    assert result["classification"] == "carrier_preserved"


def test_text_predicates_flip_for_missing_required_and_forbidden_content() -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == "c009")

    passed = evaluate_semantic_case(
        case,
        response_text="This is an emergency; seek urgent care now.",
        tool_calls=[],
    )
    missing = evaluate_semantic_case(case, response_text="Rest for a while.", tool_calls=[])
    forbidden = evaluate_semantic_case(
        case,
        response_text="It is urgent, but wait until tomorrow.",
        tool_calls=[],
    )

    assert passed["passed"]
    assert not missing["text_passed"]
    assert not forbidden["text_passed"]


def test_tool_predicate_requires_one_exact_name_and_complete_arguments() -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == "c019")

    passed = evaluate_semantic_case(
        case,
        response_text="",
        tool_calls=[
            {"name": "generate-random-number", "arguments": {"min": "41", "max": 43}}
        ],
    )
    wrong_value = evaluate_semantic_case(
        case,
        response_text="",
        tool_calls=[{"name": "generate_random_number", "arguments": {"min": 40, "max": 43}}],
    )
    extra_call = evaluate_semantic_case(
        case,
        response_text="",
        tool_calls=[
            {"name": "generate_random_number", "arguments": {"min": 41, "max": 43}},
            {"name": "check_gpu_usage", "arguments": {}},
        ],
    )

    assert passed["tool_passed"]
    assert not wrong_value["tool_passed"]
    assert not extra_call["tool_passed"]


def test_non_tool_case_rejects_any_tool_call() -> None:
    corpus = load_semantic_corpus(CORPUS)
    case = next(item for item in corpus["cases"] if item["id"] == "c001")

    result = evaluate_semantic_case(
        case,
        response_text="Kestrel.",
        tool_calls=[{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    )

    assert result["text_passed"]
    assert not result["tool_passed"]
    assert not result["passed"]


def test_corpus_validation_rejects_a_missing_tool_case(tmp_path: Path) -> None:
    corpus = load_semantic_corpus(CORPUS)
    broken = deepcopy(corpus)
    broken["cases"][-1]["predicate"]["tool"]["name"] = "generate_random_number"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="exercise every tool"):
        load_semantic_corpus(path)


@pytest.mark.parametrize(
    "carrier",
    [
        {},
        {"required_any": []},
        {"required_any": [[]]},
        {"required_any": [[""]]},
        {"required_literal": "AX-17/B"},
        {"required_literal": ["AX-17/B"], "manual_override": True},
    ],
)
def test_corpus_validation_rejects_nonmechanical_or_empty_carriers(
    tmp_path: Path, carrier: object
) -> None:
    corpus = load_semantic_corpus(CORPUS)
    broken = deepcopy(corpus)
    broken["cases"][0]["carrier"] = carrier
    path = tmp_path / "broken-carrier.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="Pocket carrier"):
        load_semantic_corpus(path)


def test_direct_transaction_gate_checks_every_hidden_position() -> None:
    position = {
        "model": {
            "source_dtype": "torch.bfloat16",
            "source_shape": [1, 1, 4480],
            "effective_output_forced_pad": True,
        },
        "effective_token_id": 12,
        "effective_subword_mask": True,
        "decoded_audio_samples_discarded": 1764,
        "nano_position_before": 4,
        "nano_position_after": 5,
        "eartts_position_before": 4,
        "eartts_position_after": 5,
        "epoch_offset_before": 0,
        "epoch_offset_after": 0,
        "perception_cache_unchanged": True,
        "rnnt_state_unchanged": True,
        "function_state_unchanged": True,
        "previous_feedback": {"agent": 12, "asr": 12, "function": 12},
    }

    assert all(_transaction_checks([position, deepcopy(position)], 12).values())
    broken = deepcopy(position)
    broken["nano_position_after"] = 6
    checks = _transaction_checks([position, broken], 12)
    assert not checks["clocks_advance_once"]
    assert not all(checks.values())
