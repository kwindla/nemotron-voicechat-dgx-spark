#!/usr/bin/env python3
"""Independently transcribe strict-v3 response WAVs and close the audio/text gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np

TARGET_RATE = 16_000
NUMBER_EQUIVALENTS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize(text: str) -> list[str]:
    return [
        NUMBER_EQUIVALENTS.get(token, token)
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
    ]


def contains_token_sequence(text: str, candidate: str) -> bool:
    haystack = normalize(text)
    needle = normalize(candidate)
    return bool(needle) and any(
        haystack[offset : offset + len(needle)] == needle
        for offset in range(len(haystack) - len(needle) + 1)
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize(reference)
    actual = normalize(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for index, expected_token in enumerate(expected, 1):
        current = [index]
        for offset, actual_token in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (expected_token != actual_token),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def extract_text(result: Any) -> str:
    while isinstance(result, (list, tuple)) and result:
        result = result[0]
    return str(result.text) if hasattr(result, "text") else str(result or "")


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


def dbfs(values: np.ndarray) -> float:
    if not values.size:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    return max(-120.0, 20 * math.log10(max(rms, 1e-6)))


def write_pcm16(path: Path, values: np.ndarray) -> None:
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(TARGET_RATE)
        output.writeframes(pcm.tobytes())


def resolve_audio_path(run_dir: Path, recorded: str) -> Path:
    """Map a host-recorded response path into the mounted qualification directory."""

    source = Path(recorded)
    return source if source.is_file() else run_dir / source.name


def configure_decoding(model: Any) -> None:
    from omegaconf import OmegaConf

    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wer", type=float, default=0.50)
    parser.add_argument("--silence-dbfs", type=float, default=-55.0)
    parser.add_argument(
        "--max-silent-rate",
        type=float,
        default=1 / 15,
        help="qualified FP32 public-model near-silent base rate",
    )
    args = parser.parse_args()

    report_path = args.run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prepared: list[tuple[dict[str, Any], Path, float]] = []
    silent: list[dict[str, Any]] = []
    for index, response in enumerate(report.get("responses", []), 1):
        source = resolve_audio_path(args.run_dir, response["audio_path"])
        values, _source_rate = read_and_resample(source)
        level = round(dbfs(values), 3)
        response["audio_dbfs"] = level
        if level <= args.silence_dbfs:
            response["external_asr"] = {"classification": "near_silent", "audio_dbfs": level}
            silent.append(response)
            continue
        destination = args.run_dir / f"response-{index:03d}-16k.wav"
        write_pcm16(destination, values)
        prepared.append((response, destination, level))

    import nemo.collections.asr as nemo_asr
    import torch

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = nemo_asr.models.ASRModel.restore_from(str(args.model), map_location=device)
    configure_decoding(model)
    model = model.to(device).eval()
    paths = [str(path) for _, path, _ in prepared]
    with torch.inference_mode():
        hypotheses = model.transcribe(paths, batch_size=1) if paths else []

    passed_audio = 0
    for (response, path, level), raw in zip(prepared, hypotheses, strict=True):
        hypothesis = extract_text(raw)
        reference = str(response.get("text") or "")
        wer = word_error_rate(reference, hypothesis)
        expected = response.get("expected_response_any") or []
        semantic_match = not expected or any(
            contains_token_sequence(hypothesis, candidate) for candidate in expected
        )
        passed = bool(hypothesis) and bool(reference) and wer <= args.max_wer and semantic_match
        response["external_asr"] = {
            "classification": "speech",
            "audio_dbfs": level,
            "wav_16k": str(path),
            "transcript": hypothesis,
            "text_channel_wer": round(wer, 4),
            "expected_response_any": expected,
            "semantic_match": semantic_match,
            "passed": passed,
        }
        passed_audio += int(passed)

    response_count = len(report.get("responses", []))
    silent_rate = len(silent) / response_count if response_count else 1.0
    asr_gate = {
        "model": str(args.model.resolve()),
        "device": str(device),
        "response_count": response_count,
        "speech_responses": len(prepared),
        "passing_speech_responses": passed_audio,
        "near_silent_responses": len(silent),
        "near_silent_rate": round(silent_rate, 4),
        "qualified_fp32_base_rate": round(args.max_silent_rate, 4),
        "max_wer": args.max_wer,
        "passed": (
            bool(prepared) and passed_audio == len(prepared) and silent_rate <= args.max_silent_rate
        ),
    }
    report["external_asr"] = asr_gate
    report["passed"] = bool(report.get("passed")) and asr_gate["passed"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asr_gate, indent=2, sort_keys=True))
    raise SystemExit(0 if asr_gate["passed"] else 1)


if __name__ == "__main__":
    main()
