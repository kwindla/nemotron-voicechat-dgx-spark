#!/usr/bin/env python3
"""Transcribe and finalize a generated EarTTS component report."""

from __future__ import annotations

import argparse
import copy
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
from transcribe_sustained import normalize, word_error_rate

from nemotron_voicechat_asr_evaluator import NemotronEnglishAsr, sha256_file
from nemotron_voicechat_runtime.artifacts import atomic_json

TARGET_RATE = 16_000


def validate_input(root: Path, record: dict[str, Any]) -> tuple[Path, np.ndarray]:
    path = root / str(record.get("path") or "")
    if path.parent.resolve() != root.resolve() or not path.is_file():
        raise RuntimeError("EarTTS ASR input is missing or escapes its component directory")
    observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    expected = {"bytes": record.get("bytes"), "sha256": record.get("sha256")}
    if observed != expected:
        raise RuntimeError(f"EarTTS ASR input identity mismatch: {path.name}")
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != TARGET_RATE
        ):
            raise RuntimeError(f"EarTTS ASR input is not PCM16 mono 16 kHz: {path.name}")
        values = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return path, values.astype(np.float32) / 32768.0


def finalize_report(
    report: dict[str, Any], root: Path, backend: NemotronEnglishAsr
) -> dict[str, Any]:
    if report.get("pre_asr_passed") is not True or report.get("pending_external_asr") is not True:
        raise RuntimeError("EarTTS report is not awaiting independent ASR")
    inputs = report.get("asr_inputs") or {}
    if set(inputs) != {"streaming", "offline"}:
        raise RuntimeError("EarTTS report has an invalid ASR input manifest")
    results = {}
    for name in ("streaming", "offline"):
        path, values = validate_input(root, inputs[name])
        results[name] = {
            **backend.transcribe(values),
            "input_path": path.name,
            "input_bytes": path.stat().st_size,
            "input_sha256": sha256_file(path),
        }

    reference = str((report.get("fixture") or {}).get("reference_text") or "")
    max_wer = (report.get("external_asr") or {}).get("max_wer")
    if not reference or not isinstance(max_wer, (int, float)):
        raise RuntimeError("EarTTS report has no valid ASR scoring contract")
    streaming_wer = word_error_rate(reference, results["streaming"]["transcript"])
    offline_wer = word_error_rate(reference, results["offline"]["transcript"])
    passed = bool(
        results["streaming"]["transcript"].strip()
        and streaming_wer <= max_wer
        and "5" in normalize(results["streaming"]["transcript"])
    )
    report["external_asr"] = {
        **backend.provenance(),
        **results["streaming"],
        "wer": streaming_wer,
        "max_wer": max_wer,
        "is_required_gate": True,
        "passed": passed,
    }
    report["offline_diagnostic"]["external_asr"] = {
        **results["offline"],
        "wer": offline_wer,
        "is_required_gate": False,
    }
    report["pending_external_asr"] = False
    report["passed"] = passed
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-out",
        type=Path,
        help="write a finalized copy here, preserving the phase-one report",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    source_report_path = root / "report.json"
    report_path = args.report_out.resolve() if args.report_out else source_report_path
    if args.report_out and report_path.exists():
        raise RuntimeError(
            f"external ASR output already exists without a stage marker: {report_path}"
        )
    report = copy.deepcopy(json.loads(source_report_path.read_text(encoding="utf-8")))
    report["pre_asr_report_sha256"] = sha256_file(source_report_path)
    backend = NemotronEnglishAsr(device=args.device)
    report = finalize_report(report, root, backend)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, report)
    print(json.dumps(report["external_asr"], indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
