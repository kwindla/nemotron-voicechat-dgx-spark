from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("tools/qualification/compare_candidate_reports.py")
    spec = importlib.util.spec_from_file_location("compare_candidate_reports", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_reports(
    root: Path,
    *,
    token_agreement: float = 1.0,
    live: bool = True,
    browser_attempts: int = 1,
) -> None:
    values = {
        "components/nano/report.json": {
            "passed": token_agreement == 1.0,
            "calls": 2,
            "token_agreement": token_agreement,
            "first_token_divergence": None,
            "control_tokens": {"order_agreement": 1.0},
            "comparisons": [{"teacher_token": 12, "candidate_token": 12}],
            "mean_call_ms": 99,
        },
        "components/eartts-eager/report.json": {
            "passed": True,
            "engine": {"enforce_eager": True},
            "fixture": {"tokens": [1, 2]},
            "codes": {"shape": [1, 2, 8], "dtype": "torch.int64", "min": 1, "max": 2},
            "audio": {"samples": 100, "sample_rate": 22050, "chunk_samples": [100]},
            "external_asr": {"transcript": "five", "wer": 0.0},
            "timing": {"startup_seconds": 999},
        },
        "components/eartts-graph/report.json": {
            "passed": True,
            "engine": {"enforce_eager": False},
            "fixture": {"tokens": [1, 2]},
            "codes": {"shape": [1, 2, 8], "dtype": "torch.int64", "min": 1, "max": 2},
            "audio": {"samples": 100, "sample_rate": 22050, "chunk_samples": [100]},
            "external_asr": {"transcript": "five", "wer": 0.0},
        },
        "report.json": {
            "passed": live,
            "browser": {
                "passed": live,
                "attempts": [
                    {"attempt": attempt + 1, "passed": live and attempt + 1 == browser_attempts}
                    for attempt in range(browser_attempts)
                ],
            },
            "multiturn": {"passed": live},
            "sustained": {"passed": live},
        },
    }
    for relative, value in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


def test_identical_component_outputs_and_verdicts_pass(tmp_path: Path) -> None:
    module = load_module()
    left, right = tmp_path / "downloaded", tmp_path / "converted"
    write_reports(left)
    write_reports(right)
    assert module.compare(left, right)["passed"] is True


def test_browser_attempt_count_does_not_change_live_verdict(tmp_path: Path) -> None:
    module = load_module()
    left, right = tmp_path / "downloaded", tmp_path / "converted"
    write_reports(left, browser_attempts=1)
    write_reports(right, browser_attempts=2)
    report = module.compare(left, right)
    assert report["passed"] is True
    assert report["live_verdicts"]["downloaded"]["browser_passed"] is True


def test_component_difference_fails_even_when_live_verdicts_match(tmp_path: Path) -> None:
    module = load_module()
    left, right = tmp_path / "downloaded", tmp_path / "converted"
    write_reports(left)
    write_reports(right, token_agreement=0.9)
    report = module.compare(left, right)
    assert report["passed"] is False
    assert report["components"]["nano"]["passed"] is False


def test_explicit_staged_sustained_report_is_transparent_and_fail_closed(
    tmp_path: Path,
) -> None:
    module = load_module()
    left, right = tmp_path / "downloaded", tmp_path / "converted"
    write_reports(left)
    write_reports(right)
    (right / "report.json").write_text(
        json.dumps({"passed": False, "error": "integrated sustained failed"}),
        encoding="utf-8",
    )
    (right / "browser-attempts.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (right / "multiturn").mkdir(exist_ok=True)
    (right / "multiturn/report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    sustained = tmp_path / "staged-sustained.json"
    sustained.write_text(json.dumps({"passed": True}), encoding="utf-8")

    report = module.compare(left, right, converted_sustained_report=sustained)
    assert report["passed"] is True
    assert report["live_verdicts"]["evidence"]["converted"]["sustained"] == str(
        sustained.resolve()
    )

    sustained.write_text(json.dumps({"passed": False}), encoding="utf-8")
    assert module.compare(left, right, converted_sustained_report=sustained)["passed"] is False
