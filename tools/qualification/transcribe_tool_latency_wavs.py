#!/usr/bin/env python3
"""Independently transcribe retained tool-latency response WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transcribe_sustained import classify_audio, read_and_resample, word_error_rate

from nemotron_voicechat_asr_evaluator import NemotronEnglishAsr

SUITE_PROFILES = {
    "step1": {
        "step1-final-fast-n10": 10,
        "step1-final-slow": 1,
        "step1-final-always-ack": 1,
        "step1-final-safety": 12,
    },
    "step2": {
        "step2-primary": 40,
        "step2-final-fast-n10": 10,
        "step2-final-slow": 1,
        "step2-final-safety-r2": 12,
    },
}
MAX_WER = 0.5
REQUIRED_WORD_NORMALIZERS = {
    "exact_v1": {},
    "english_spelling_v2": {"harbour": "harbor"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_words(text: str, *, version: str = "exact_v1") -> set[str]:
    aliases = REQUIRED_WORD_NORMALIZERS[version]
    words = {word.lower().strip(".,!?;:'\"") for word in text.split() if word}
    return {aliases.get(word, word) for word in words}


def report_self_sha256(report: dict[str, Any]) -> str:
    payload = dict(report)
    payload.pop("self_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def predecessor_record(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != 1 or report.get("kind") != "tool_call_latency_step2_asr":
        raise RuntimeError("required-word v2 predecessor must be a step2 ASR schema-1 report")
    if report.get("passed") is not False:
        raise RuntimeError("required-word v2 predecessor must retain the observed red verdict")
    if report.get("self_sha256") != report_self_sha256(report):
        raise RuntimeError("required-word v2 predecessor self-hash mismatch")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "self_sha256": report["self_sha256"],
    }


def expected_words(
    suite: str, scenario: str | None, report: dict[str, Any]
) -> set[str]:
    if scenario is None:
        payload = report.get("payload") or {}
        word = payload.get("verification_word")
        if not isinstance(word, str) or not word.strip():
            raise RuntimeError(f"{suite}: required row lacks a verification word")
        return {word.casefold()}
    if scenario in {"call_emission_interruption", "result_injection_interruption"}:
        return {"stopped"}
    return set()


def evaluate_response(
    *,
    reference: str,
    hypothesis: str,
    classification: str,
    required: bool,
    required_words: set[str],
    required_word_normalizer: str,
) -> tuple[float | None, bool, bool]:
    # WER deliberately sees the raw transcript. Spelling equivalences are scoped
    # only to the independent required-word semantic-presence conjunct.
    wer = word_error_rate(reference, hypothesis) if reference else None
    words_present = required_words.issubset(
        normalized_words(hypothesis, version=required_word_normalizer)
    )
    passed = bool(
        classification == "speech"
        and hypothesis.strip()
        and (not required or (wer is not None and wer <= MAX_WER and words_present))
    )
    return wer, words_present, passed


def transcribe(
    root: Path,
    image_id: str,
    profile: str,
    *,
    required_word_normalizer: str = "exact_v1",
    predecessor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suites = SUITE_PROFILES[profile]
    backend = NemotronEnglishAsr(device="cuda:0")
    provenance = backend.provenance()
    if provenance["evaluator_image_id"] != image_id:
        raise RuntimeError("ASR backend reported a different evaluator image")

    records: list[dict[str, Any]] = []
    for suite, expected_count in suites.items():
        suite_root = root / suite
        top_path = suite_root / "report.json"
        top = json.loads(top_path.read_text(encoding="utf-8"))
        cells = top.get("cells")
        if not isinstance(cells, list) or len(cells) != expected_count:
            raise RuntimeError(f"{suite}: expected exactly {expected_count} cells")
        for ordinal, cell in enumerate(cells, 1):
            report_path = suite_root / f"cell-{ordinal:03d}" / "report.json"
            wav_path = report_path.with_name("response.wav")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reference = report.get("text")
            if not isinstance(reference, str):
                raise RuntimeError(f"{report_path}: missing model text")
            scenario = cell.get("scenario")
            known_chained_red = scenario == "chained_two_call"
            required = not known_chained_red
            if required and not reference.strip():
                raise RuntimeError(f"{report_path}: required row has empty model text")
            audio, source_rate = read_and_resample(wav_path)
            classification, audio_dbfs = classify_audio(audio)
            result = backend.transcribe(audio)
            hypothesis = result["transcript"]
            required_words = expected_words(suite, scenario, report)
            wer, words_present, passed = evaluate_response(
                reference=reference,
                hypothesis=hypothesis,
                classification=classification,
                required=required,
                required_words=required_words,
                required_word_normalizer=required_word_normalizer,
            )
            records.append(
                {
                    "suite": suite,
                    "ordinal": ordinal,
                    "scenario": scenario,
                    "required": required,
                    "non_gating_reason": "known_chained_call_red" if known_chained_red else None,
                    "source_report": str(report_path.relative_to(root)),
                    "source_report_sha256": sha256_file(report_path),
                    "source_wav": str(wav_path.relative_to(root)),
                    "source_wav_sha256": sha256_file(wav_path),
                    "source_rate": source_rate,
                    "classification": classification,
                    "audio_dbfs": audio_dbfs,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "token_ids": result["token_ids"],
                    "audio_samples": result["audio_samples"],
                    "audio_seconds": result["audio_seconds"],
                    "wer": round(wer, 4) if wer is not None else None,
                    "required_words": sorted(required_words),
                    "required_words_present": words_present,
                    "passed": passed,
                }
            )

    report: dict[str, Any] = {
        "schema": 2 if required_word_normalizer != "exact_v1" else 1,
        "kind": f"tool_call_latency_{profile}_asr",
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator": provenance,
        "max_wer": MAX_WER,
        "suite_contract": suites,
        "records": records,
        "required_records": sum(record["required"] for record in records),
        "required_passed": sum(record["required"] and record["passed"] for record in records),
        "non_gating_records": sum(not record["required"] for record in records),
        "non_gating_passed": sum(not record["required"] and record["passed"] for record in records),
    }
    if required_word_normalizer != "exact_v1":
        if predecessor is None:
            raise RuntimeError("required-word v2 requires an immutable red predecessor")
        report["required_word_normalizer"] = {
            "version": required_word_normalizer,
            "scope": "required_word_presence_only",
            "equivalences": REQUIRED_WORD_NORMALIZERS[required_word_normalizer],
            "wer_uses_equivalences": False,
            "classification_uses_equivalences": False,
        }
        report["predecessor"] = predecessor
    report["passed"] = bool(
        report["required_records"] > 0
        and report["required_records"] == report["required_passed"]
    )
    report["self_sha256"] = report_self_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-image-id", required=True)
    parser.add_argument("--profile", choices=tuple(SUITE_PROFILES), default="step1")
    parser.add_argument(
        "--required-word-normalizer",
        choices=tuple(REQUIRED_WORD_NORMALIZERS),
        default="exact_v1",
    )
    parser.add_argument("--predecessor-report", type=Path)
    args = parser.parse_args()
    predecessor = (
        predecessor_record(args.predecessor_report)
        if args.predecessor_report is not None
        else None
    )
    if args.required_word_normalizer != "exact_v1" and predecessor is None:
        parser.error("--predecessor-report is required for a versioned normalizer")
    report = transcribe(
        args.root,
        args.evaluator_image_id,
        args.profile,
        required_word_normalizer=args.required_word_normalizer,
        predecessor=predecessor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    summary_keys = (
        "passed",
        "required_records",
        "required_passed",
        "non_gating_records",
        "non_gating_passed",
        "self_sha256",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
