"""Deterministic structured predicates for the direct-text semantic corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

REQUIRED_CATEGORIES = frozenset(
    {"names", "numbers", "punctuation", "reference_resolution", "interruption", "safety", "tool"}
)
_CARRIER_KEYS = frozenset({"required_any", "required_literal", "forbidden_any"})


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    haystack = normalize_words(text)
    needle = normalize_words(phrase)
    return bool(needle) and any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _validate_carrier(case: dict[str, Any]) -> None:
    carrier = case.get("carrier")
    if not isinstance(carrier, dict):
        raise ValueError("every semantic case requires a mechanical Pocket carrier predicate")
    if unknown := set(carrier) - _CARRIER_KEYS:
        raise ValueError(f"unsupported Pocket carrier keys: {sorted(unknown)}")
    required_any = carrier.get("required_any", [])
    required_literal = carrier.get("required_literal", [])
    forbidden_any = carrier.get("forbidden_any", [])
    if not isinstance(required_any, list) or any(
        not isinstance(group, list)
        or not group
        or any(not isinstance(value, str) or not value.strip() for value in group)
        for group in required_any
    ):
        raise ValueError("Pocket carrier required_any must contain non-empty string groups")
    if any(
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        for values in (required_literal, forbidden_any)
    ):
        raise ValueError("Pocket carrier literal/forbidden predicates must be string lists")
    if not required_any and not required_literal:
        raise ValueError("Pocket carrier requires predicate-bearing prompt content")


def load_semantic_corpus(path: Path) -> dict[str, Any]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("schema") != 2:
        raise ValueError("direct-text semantic corpus schema must be 2")
    tools = corpus.get("tools")
    cases = corpus.get("cases")
    if not isinstance(tools, list) or not isinstance(cases, list) or len(cases) < 20:
        raise ValueError("direct-text semantic corpus requires tools and at least 20 cases")
    tool_names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
    if len(tool_names) != len(set(tool_names)) or any(not name for name in tool_names):
        raise ValueError("direct-text semantic corpus tool names must be unique")
    case_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or len(case_ids) != len(set(case_ids)):
        raise ValueError("direct-text semantic corpus case IDs must be unique")
    categories = {
        category
        for case in cases
        for category in case.get("categories", [])
        if isinstance(category, str)
    }
    if not REQUIRED_CATEGORIES.issubset(categories):
        raise ValueError("direct-text semantic corpus does not cover every required category")
    expected_tools = [
        case.get("predicate", {}).get("tool", {}).get("name")
        for case in cases
        if isinstance(case.get("predicate", {}).get("tool"), dict)
    ]
    if len(expected_tools) != len(tool_names) or set(expected_tools) != set(tool_names):
        raise ValueError("direct-text semantic corpus must exercise every tool exactly")
    if not all(isinstance(case.get("prompt"), str) and case["prompt"].strip() for case in cases):
        raise ValueError("direct-text semantic corpus prompts must be non-empty")
    for case in cases:
        _validate_carrier(case)
    return corpus


def _equivalent_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str) and actual.casefold() in {"true", "false"}:
            actual = actual.casefold() == "true"
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value == float(expected)
    return str(actual).casefold() == str(expected).casefold()


def evaluate_semantic_case(
    case: dict[str, Any], *, response_text: str, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    predicate = case["predicate"]
    required = predicate.get("required_any") or []
    required_literal = predicate.get("required_literal") or []
    forbidden = predicate.get("forbidden_any") or []
    required_results = [
        any(contains_phrase(response_text, phrase) for phrase in alternatives)
        for alternatives in required
    ]
    forbidden_matches = [
        phrase for phrase in forbidden if contains_phrase(response_text, phrase)
    ]
    literal_results = [
        literal.casefold() in response_text.casefold() for literal in required_literal
    ]
    text_passed = all(required_results) and all(literal_results) and not forbidden_matches
    expected_tool = predicate.get("tool")
    if expected_tool is None:
        tool_passed = not tool_calls if predicate.get("forbid_tools") else True
    else:
        expected_name = expected_tool["name"].replace("-", "_")
        expected_arguments = expected_tool.get("arguments") or {}
        matching = [
            call
            for call in tool_calls
            if str(call.get("name", "")).replace("-", "_") == expected_name
        ]
        tool_passed = len(tool_calls) == 1 and len(matching) == 1
        if tool_passed:
            actual_arguments = matching[0].get("arguments")
            tool_passed = isinstance(actual_arguments, dict) and set(actual_arguments) == set(
                expected_arguments
            )
            if tool_passed:
                tool_passed = all(
                    _equivalent_scalar(actual_arguments[name], expected)
                    for name, expected in expected_arguments.items()
                )
    return {
        "case_id": case["id"],
        "categories": case["categories"],
        "safety_critical": bool(predicate.get("safety_critical")),
        "expected_tool": expected_tool,
        "response_text": response_text,
        "tool_calls": tool_calls,
        "required_predicates": required_results,
        "required_literal_predicates": literal_results,
        "forbidden_matches": forbidden_matches,
        "text_passed": text_passed,
        "tool_passed": tool_passed,
        "passed": text_passed and tool_passed,
    }


def evaluate_pocket_carrier(case: dict[str, Any], *, transcript: str) -> dict[str, Any]:
    """Classify whether Pocket+RNNT preserved predicate-relevant prompt content."""

    predicate = case["carrier"]
    required = predicate.get("required_any") or []
    required_literal = predicate.get("required_literal") or []
    forbidden = predicate.get("forbidden_any") or []
    required_results = [
        any(contains_phrase(transcript, phrase) for phrase in alternatives)
        for alternatives in required
    ]
    literal_results = [
        literal.casefold() in transcript.casefold() for literal in required_literal
    ]
    forbidden_matches = [phrase for phrase in forbidden if contains_phrase(transcript, phrase)]
    passed = all(required_results) and all(literal_results) and not forbidden_matches
    return {
        "case_id": case["id"],
        "transcript": transcript,
        "required_predicates": required_results,
        "required_literal_predicates": literal_results,
        "forbidden_matches": forbidden_matches,
        "passed": passed,
        "classification": "carrier_preserved" if passed else "carrier_inconclusive",
    }
