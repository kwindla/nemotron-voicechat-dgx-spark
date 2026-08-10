#!/usr/bin/env python3
"""Independently transcribe every direct-position probe response WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transcribe_sustained import (
    contains_token_sequence,
    dbfs,
    read_and_resample,
    resolve_audio_path,
    word_error_rate,
    write_pcm16,
)

from nemotron_voicechat_runtime.artifacts import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wer", type=float, default=0.50)
    parser.add_argument("--silence-dbfs", type=float, default=-55.0)
    parser.add_argument(
        "--required-phrase",
        action="append",
        default=[],
        help="Phrase that must occur in every independent transcript; repeatable",
    )
    parser.add_argument(
        "--excluded-phrase",
        action="append",
        default=[],
        help="Phrase that must not occur in any independent transcript; repeatable",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases") or []
    prepared: list[tuple[dict[str, Any], Path, Path, float, Any]] = []
    for case in cases:
        recorded = str((case.get("response") or {}).get("wav") or "")
        if not recorded:
            raise ValueError(f"probe case {case.get('token_form')!r} has no response WAV")
        source = resolve_audio_path(run_dir, recorded)
        if not source.is_file():
            raise FileNotFoundError(source)
        values, _source_rate = read_and_resample(source)
        level = round(dbfs(values), 3)
        destination = run_dir / f"{case['token_form']}-16k.wav"
        write_pcm16(destination, values)
        canonical_values, canonical_rate = read_and_resample(destination)
        if canonical_rate != 16_000:
            raise RuntimeError("canonical ASR WAV was not written at 16 kHz")
        prepared.append((case, source, destination, level, canonical_values))

    from nemotron_voicechat_asr_evaluator import (
        DEFAULT_MANIFEST,
        DEFAULT_MODEL,
        NemotronEnglishAsr,
    )

    backend = NemotronEnglishAsr(
        model_root=(args.model or DEFAULT_MODEL),
        manifest_path=(args.model_manifest or DEFAULT_MANIFEST),
        device=args.device,
    )
    hypotheses = [backend.transcribe(values) for *_, values in prepared]
    asr_provenance = backend.provenance()

    results: list[dict[str, Any]] = []
    for (case, source, destination, level, _values), raw in zip(
        prepared, hypotheses, strict=True
    ):
        hypothesis = raw["transcript"]
        reference = str(case.get("assistant_text") or "")
        wer = word_error_rate(reference, hypothesis)
        required_match = all(
            contains_token_sequence(hypothesis, phrase)
            for phrase in args.required_phrase
        )
        excluded_match = not any(
            contains_token_sequence(hypothesis, phrase)
            for phrase in args.excluded_phrase
        )
        passed = bool(
            reference
            and hypothesis
            and level > args.silence_dbfs
            and wer <= args.max_wer
            and required_match
            and excluded_match
        )
        results.append(
            {
                "token_form": case.get("token_form"),
                "model_text": reference,
                "transcript": hypothesis,
                "token_ids": raw["token_ids"],
                "text_channel_wer": round(wer, 4),
                "audio_dbfs": level,
                "source_wav": str(source),
                "source_wav_sha256": sha256_file(source),
                "wav_16k": str(destination),
                "canonical_wav_sha256": sha256_file(destination),
                "required_phrases": args.required_phrase,
                "required_phrases_match": required_match,
                "excluded_phrases": args.excluded_phrase,
                "excluded_phrases_absent": excluded_match,
                "passed": passed,
            }
        )

    artifact = {
        "schema": 1,
        "kind": "direct_user_source_position_external_asr",
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "runtime_gate_passed": bool(
            report.get("passed")
            and cases
            and all(case.get("passed") is True for case in cases)
        ),
        "external_asr": asr_provenance,
        "max_wer": args.max_wer,
        "silence_dbfs": args.silence_dbfs,
        "cases": results,
        "passed": bool(results) and all(result["passed"] for result in results),
    }
    artifact["passed"] = artifact["runtime_gate_passed"] and artifact["passed"]
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "external-asr.json"
    )
    atomic_json(output_path, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    raise SystemExit(0 if artifact["passed"] else 1)


if __name__ == "__main__":
    main()
