#!/usr/bin/env python3
"""Compare component outputs and qualification verdicts for two artifact sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_report(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required qualification report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def component_identity(kind: str, report: dict[str, Any]) -> dict[str, Any]:
    if kind == "nano":
        return {
            "passed": report.get("passed"),
            "calls": report.get("calls"),
            "token_agreement": report.get("token_agreement"),
            "first_token_divergence": report.get("first_token_divergence"),
            "control_tokens": report.get("control_tokens"),
            "comparisons": report.get("comparisons"),
        }
    if kind.startswith("eartts"):
        return {
            "passed": report.get("passed"),
            "engine_enforce_eager": report.get("engine", {}).get("enforce_eager"),
            "fixture_tokens": report.get("fixture", {}).get("tokens"),
            "codes": {
                key: report.get("codes", {}).get(key) for key in ("shape", "dtype", "min", "max")
            },
            "audio": {
                key: report.get("audio", {}).get(key)
                for key in ("samples", "sample_rate", "chunk_samples")
            },
            "external_asr": {
                key: report.get("external_asr", {}).get(key) for key in ("transcript", "wer")
            },
        }
    raise ValueError(f"unknown component kind: {kind}")


def _report_passed(value: Any) -> bool:
    return bool(value.get("passed")) if isinstance(value, dict) else value == "passed"


def live_verdict_from_evidence(
    root: Path, *, sustained_report: Path | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a coherent report or explicit leaf reports from retained staged gates."""

    top_path = root / "report.json"
    top = read_report(root, "report.json")
    browser_path = top_path
    browser = top.get("browser")
    if not isinstance(browser, dict):
        browser_path = root / "browser-attempts.json"
        browser = json.loads(browser_path.read_text(encoding="utf-8"))
    multiturn_path = top_path
    multiturn = top.get("multiturn")
    if not isinstance(multiturn, dict):
        multiturn_path = root / "multiturn/report.json"
        multiturn = json.loads(multiturn_path.read_text(encoding="utf-8"))
    sustained_path = sustained_report or top_path
    sustained = (
        json.loads(sustained_report.read_text(encoding="utf-8"))
        if sustained_report is not None
        else top.get("sustained")
    )
    if not isinstance(sustained, dict):
        sustained_path = root / "sustained/report.json"
        sustained = json.loads(sustained_path.read_text(encoding="utf-8"))
    verdict = {
        "browser_passed": _report_passed(browser),
        "multiturn_passed": _report_passed(multiturn),
        "sustained_passed": _report_passed(sustained),
    }
    verdict["passed"] = all(verdict.values())
    evidence = {
        "browser": str(browser_path.resolve()),
        "multiturn": str(multiturn_path.resolve()),
        "sustained": str(sustained_path.resolve()),
    }
    return verdict, evidence


def compare(
    left: Path, right: Path, *, converted_sustained_report: Path | None = None
) -> dict[str, Any]:
    paths = {
        "nano": "components/nano/report.json",
        "eartts_eager": "components/eartts-eager/report.json",
        "eartts_graph": "components/eartts-graph/report.json",
    }
    component_comparisons = {}
    for kind, relative in paths.items():
        left_identity = component_identity(kind, read_report(left, relative))
        right_identity = component_identity(kind, read_report(right, relative))
        component_comparisons[kind] = {
            "passed": left_identity == right_identity,
            "downloaded": left_identity,
            "converted": right_identity,
        }
    left_live, left_live_evidence = live_verdict_from_evidence(left)
    right_live, right_live_evidence = live_verdict_from_evidence(
        right, sustained_report=converted_sustained_report
    )
    report = {
        "passed": all(item["passed"] for item in component_comparisons.values())
        and left_live == right_live,
        "downloaded_root": str(left.resolve()),
        "converted_root": str(right.resolve()),
        "components": component_comparisons,
        "live_verdicts": {
            "passed": left_live == right_live,
            "downloaded": left_live,
            "converted": right_live,
            "evidence": {
                "downloaded": left_live_evidence,
                "converted": right_live_evidence,
            },
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument(
        "--converted-sustained-report",
        type=Path,
        help="Explicit retained sustained report when converted gates ran in stages.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(
            args.downloaded,
            args.converted,
            converted_sustained_report=args.converted_sustained_report,
        )
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
