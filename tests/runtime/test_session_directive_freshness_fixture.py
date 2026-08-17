from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tools/qualification/session_directive_freshness_fixture.py"
)


def load_fixture():
    spec = importlib.util.spec_from_file_location(
        "session_directive_freshness_fixture", FIXTURE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def turns(module, *, stale: bool = False):
    texts = [
        module.UTC_RESULT,
        module.TIMEZONE_RESULT,
        module.PACIFIC_RESULT,
        module.UTC_RESULT if stale else module.PACIFIC_RESULT,
    ]
    return [
        {
            "ordinal": ordinal,
            "utterance": module.PROMPTS[ordinal - 1],
            "assistant_text": texts[ordinal - 1],
            "tool_calls": (
                []
                if expected is None
                else [
                    {
                        "name": expected,
                        "arguments": (
                            {"timezone": "Pacific"}
                            if expected == "set_timezone"
                            else {}
                        ),
                    }
                ]
            ),
            "output_applied": expected is not None,
            "response_status": "completed",
        }
        for ordinal, expected in enumerate(module.EXPECTED_CALLS, 1)
    ]


def test_directive_fixture_gates_stale_utc_after_pacific_result() -> None:
    module = load_fixture()

    fresh = module.evaluate_replicate(turns(module), [])
    stale = module.evaluate_replicate(turns(module, stale=True), [])

    assert fresh["sequence_elicited"] is True
    assert fresh["eligible_responses"] == 2
    assert fresh["stale_rate"] == 0.0
    assert fresh["freshness_gate_passed"] is True
    assert fresh["passed"] is True
    assert stale["stale_rate"] == 0.5
    assert stale["freshness_gate_passed"] is False
    assert stale["passed"] is False


def test_directive_fixture_rejects_omitted_or_incomplete_fresh_result() -> None:
    module = load_fixture()

    omitted = turns(module)
    omitted[2]["assistant_text"] = ""
    incomplete = turns(module)
    incomplete[3]["assistant_text"] = "It is currently one fifty five."

    for mutated in (omitted, incomplete):
        result = module.evaluate_replicate(mutated, [])
        assert result["stale_responses"] == []
        assert result["freshness_gate_passed"] is False
        assert result["passed"] is False
        assert len(result["missing_fresh_result_responses"]) == 1


def test_directive_comparison_does_not_claim_fix_when_rate_is_unchanged(
    tmp_path: Path,
) -> None:
    module = load_fixture()
    pre_path = tmp_path / "pre.json"
    post_path = tmp_path / "post.json"
    output = tmp_path / "comparison.json"
    pre_path.write_text(
        json.dumps({"arm": "pre", "stale_rate": 0.0, "sequence_elicitation_rate": 1.0})
    )
    post_path.write_text(
        json.dumps(
            {
                "arm": "post",
                "stale_rate": 0.0,
                "sequence_elicitation_rate": 1.0,
                "freshness_gate_passed": True,
            }
        )
    )

    result = module.compare_reports(pre_path, post_path, output)

    assert result["post_freshness_gate_passed"] is True
    assert result["rate_changed"] is False
    assert result["i4_fix_supported"] is False
    assert "do not claim I4 fixed" in result["finding"]
    assert json.loads(output.read_text()) == result
