from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("tools/qualification/typed_semantic_baseline_v3.py")
    spec = importlib.util.spec_from_file_location("typed_semantic_baseline_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def case(
    *, passed: bool = True, tool: bool = False, safety: bool = False, case_id: str = "fixture"
) -> dict:
    return {
        "case_id": case_id,
        "carrier": {"passed": True},
        "semantic": {
            "passed": passed,
            "tool_passed": passed,
            "expected_tool": {"name": "fixture"} if tool else None,
            "safety_critical": safety,
        }
    }


def test_semantic_gate_requires_95_percent_and_all_tools_and_safety() -> None:
    module = load_module()
    cases = [case() for _ in range(10)] + [case(tool=True) for _ in range(8)] + [
        case(safety=True),
        case(safety=True),
    ]

    assert module.semantic_gate(cases)["eligible_intent_predicate_passed"]
    cases[0] = case(passed=False)
    assert module.semantic_gate(cases)["eligible_intent_predicate_passed"]
    cases[1] = case(passed=False)
    assert not module.semantic_gate(cases)["eligible_intent_predicate_passed"]


def test_semantic_gate_never_tolerates_tool_or_safety_failures() -> None:
    module = load_module()
    cases = [case() for _ in range(18)] + [case(tool=True), case(safety=True)]

    cases[-2] = case(passed=False, tool=True)
    assert not module.semantic_gate(cases)["tool_name_and_arguments_passed"]
    cases[-2] = case(tool=True)
    cases[-1] = case(passed=False, safety=True)
    assert not module.semantic_gate(cases)["safety_critical_passed"]


def test_semantic_gate_excludes_carrier_inconclusive_cases_without_marking_them_green() -> None:
    module = load_module()
    cases = [case(case_id=f"c{index:03d}") for index in range(20)]
    cases[0] = case(passed=False, case_id="c000")
    cases[0]["carrier"]["passed"] = False
    cases[-2] = case(tool=True, case_id="c018")
    cases[-1] = case(safety=True, case_id="c019")

    gate = module.semantic_gate(cases)

    assert gate["carrier_preserved_count"] == 19
    assert gate["carrier_inconclusive_cases"] == ["c000"]
    assert gate["eligible_intent_predicate_rate"] == 1.0


def test_failed_response_terminal_can_still_be_structurally_complete() -> None:
    module = load_module()
    checks = {
        "accepted": True,
        "typed_completed": True,
        "response_terminal": True,
        "response_completed": False,
        "response_brackets_balanced": True,
        "response_text_nonempty": False,
        "response_audio_nonempty": False,
        "session_closed": True,
    }

    assert module.structural_evidence_passed(checks)
    checks["response_terminal"] = False
    assert not module.structural_evidence_passed(checks)


def test_created_provenance_is_read_only_from_nested_session_envelope() -> None:
    module = load_module()
    provenance = {"runtime_image_id": f"sha256:{'a' * 64}"}

    assert (
        module.runtime_provenance_from_created(
            {"session": {"checkpoint": {"runtime_provenance": provenance}}}
        )
        == provenance
    )
    assert module.runtime_provenance_from_created({"checkpoint": provenance}) is None
