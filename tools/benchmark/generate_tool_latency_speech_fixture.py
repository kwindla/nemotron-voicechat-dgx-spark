#!/usr/bin/env python3
"""Generate one deterministic Pocket carrier for tool-latency diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager


async def generate(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=False)
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        language="english_2026-04",
        voice="alba",
        threads=4,
        seed=0,
        cpus=(7, 8, 9, 15),
        startup_timeout=180,
    )
    try:
        await manager.start()
        health = dict(manager.health)
        result = await manager.synthesize(
            args.text,
            job_id="tool-latency-speech-carrier",
            cancel_event=asyncio.Event(),
        )
    finally:
        await manager.stop()
    if result.sample_rate != 16_000 or not result.pcm16 or len(result.pcm16) % 2:
        raise RuntimeError("Pocket returned an invalid PCM16 carrier")
    pcm = args.output / "carrier.pcm"
    wav = args.output / "carrier.wav"
    pcm.write_bytes(result.pcm16)
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(result.pcm16)
    report: dict[str, object] = {
        "schema": 1,
        "kind": "tool_call_latency_speech_fixture",
        "text": args.text,
        "pcm_bytes": len(result.pcm16),
        "pcm_sha256": hashlib.sha256(result.pcm16).hexdigest(),
        "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "generated_seconds": result.generated_seconds,
        "wall_seconds": result.wall_seconds,
        "time_to_first_audio_seconds": result.time_to_first_audio_seconds,
        "pocket_health": health,
        "runtime_image_id": args.runtime_image_id,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(generate(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
