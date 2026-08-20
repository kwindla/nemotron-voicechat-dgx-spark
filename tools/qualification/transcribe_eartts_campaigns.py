#!/usr/bin/env python3
"""Transcribe EarTTS A/B campaign WAVs with the pinned offline ASR backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

TARGET_RATE = 16_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def read_and_resample(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        payload = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only PCM16 WAVs are supported")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE and values.size:
        count = round(values.size * TARGET_RATE / rate)
        values = np.interp(
            np.arange(count, dtype=np.float64) * rate / TARGET_RATE,
            np.arange(values.size, dtype=np.float64),
            values,
        ).astype(np.float32)
    return values, rate


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_roots", type=Path, nargs="+")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-output", type=Path, required=True)
    args = parser.parse_args()

    from nemotron_voicechat_asr_evaluator import (
        DEFAULT_MANIFEST,
        DEFAULT_MODEL,
        NemotronEnglishAsr,
    )

    backend = NemotronEnglishAsr(
        model_root=args.model or DEFAULT_MODEL,
        manifest_path=args.model_manifest or DEFAULT_MANIFEST,
        device=args.device,
    )
    total = 0
    sources: list[dict[str, Any]] = []
    for campaign_root in args.campaign_roots:
        campaign = json.loads((campaign_root / "campaign.json").read_text())
        attempts = campaign.get("attempts", [])
        if not attempts:
            raise ValueError(f"{campaign_root}: expected at least one attempt")
        for attempt_number in range(1, len(attempts) + 1):
            attempt_dir = campaign_root / f"attempt-{attempt_number:02d}"
            report = json.loads((attempt_dir / "report.json").read_text())
            responses = report.get("responses", [])
            if not responses:
                raise ValueError(f"{attempt_dir}: expected at least one response")
            output = attempt_dir / "transcripts.json"
            if output.exists():
                transcripts = json.loads(output.read_text())
                for response in responses:
                    response_number = int(response["ordinal"])
                    wav_path = attempt_dir / f"response_{response_number:03d}.wav"
                    retained = transcripts.get(wav_path.name) or {}
                    if retained.get("source_audio_sha256") != sha256_file(wav_path):
                        raise ValueError(f"{output}: source audio hash mismatch")
                total += len(responses)
                continue
            transcripts: dict[str, Any] = {}
            for response in responses:
                response_number = int(response["ordinal"])
                wav_path = attempt_dir / f"response_{response_number:03d}.wav"
                values, _ = read_and_resample(wav_path)
                result = backend.transcribe(values)
                transcripts[wav_path.name] = {
                    "transcript": str(result["transcript"]),
                    "seconds": round(wav_seconds(wav_path), 3),
                    "source_audio_sha256": sha256_file(wav_path),
                }
                total += 1
            write_json(output, transcripts)
        sources.append(
            {
                "campaign_root": str(campaign_root),
                "campaign_sha256": sha256_file(campaign_root / "campaign.json"),
            }
        )

    write_json(
        args.execution_output,
        {
            "schema": "nemotron_voicechat.eartts_quantization_ab_asr.v1",
            "network": "none",
            "response_count": total,
            "campaigns": sources,
            "backend_provenance": backend.provenance(),
        },
    )


if __name__ == "__main__":
    main()
