#!/usr/bin/env python3
"""Transcribe pinned positive-control WAVs for an ASR judge-swap calibration."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
from transcribe_sustained import word_error_rate

from nemotron_voicechat_asr_evaluator import NemotronEnglishAsr, sha256_file
from nemotron_voicechat_runtime.artifacts import atomic_json


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16_000
            or source.getnframes() <= 0
        ):
            raise RuntimeError(f"control is not nonempty PCM16 mono 16 kHz: {path}")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
            np.float32
        ) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("LABEL", "WAV", "REFERENCE"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = NemotronEnglishAsr()
    cases = []
    for label, raw_path, reference in args.case:
        path = Path(raw_path).resolve()
        result = backend.transcribe(read_pcm16(path))
        wer = word_error_rate(reference, result["transcript"])
        cases.append(
            {
                "label": label,
                "wav": str(path),
                "wav_bytes": path.stat().st_size,
                "wav_sha256": sha256_file(path),
                "reference": reference,
                **result,
                "wer": wer,
                "passed": bool(result["transcript"].strip() and wer <= 0.5),
            }
        )
    report = {
        "schema": 1,
        "kind": "asr_judge_swap_positive_controls",
        "external_asr": backend.provenance(),
        "cases": cases,
        "passed": bool(cases) and all(case["passed"] for case in cases),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
