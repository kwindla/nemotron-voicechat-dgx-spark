#!/usr/bin/env python3
"""Generate the sealed attnW8-GPTQ tool carriers with qualified Pocket TTS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager

RATE = 16_000
FRAME_SAMPLES = 1280


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm)


async def generate(args: argparse.Namespace) -> dict[str, object]:
    spec = json.loads(args.scenarios.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"fixture output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        language="english_2026-04",
        voice="alba",
        threads=4,
        seed=0,
        cpus=(7, 8, 9, 15),
        startup_timeout=180,
    )
    records: list[dict[str, object]] = []
    synthesis_health: dict[str, object] = {}
    try:
        await manager.start()
        synthesis_health = dict(manager.health)
        for partition in ("a2", "c2"):
            for expected in ("positive", "no_call"):
                for scenario_id, prompt in spec[partition][expected]:
                    result = await manager.synthesize(
                        prompt,
                        job_id=f"attnw8-gptq-{scenario_id}",
                        cancel_event=asyncio.Event(),
                    )
                    pcm_path = args.output / f"{scenario_id}.pcm"
                    wav_path = args.output / f"{scenario_id}.wav"
                    pcm_path.write_bytes(result.pcm16)
                    write_wav(wav_path, result.pcm16)
                    records.append(
                        {
                            "scenario_id": scenario_id,
                            "partition": partition,
                            "expected": expected,
                            "prompt": prompt,
                            "expected_transcript": prompt.casefold().rstrip(".?!"),
                            "pcm": {
                                "path": pcm_path.name,
                                "bytes": len(result.pcm16),
                                "sha256": digest(result.pcm16),
                                "sample_rate": RATE,
                            },
                            "wav": {
                                "path": wav_path.name,
                                "bytes": wav_path.stat().st_size,
                                "sha256": digest(wav_path.read_bytes()),
                            },
                        }
                    )
        c1 = spec["c1"]
        result = await manager.synthesize(
            c1["prompt"],
            job_id="attnw8-gptq-c1-precedent",
            cancel_event=asyncio.Event(),
        )
        c1_pcm = result.pcm16 + bytes(
            int(c1["trailing_silence_frames_80ms"]) * FRAME_SAMPLES * 2
        )
        if digest(c1_pcm) != c1["pcm_sha256"]:
            raise RuntimeError("deterministic C1 carrier did not reproduce its pinned SHA-256")
        c1_pcm_path = args.output / "c1-precedent-carrier.pcm"
        c1_wav_path = args.output / "c1-precedent-carrier.wav"
        c1_pcm_path.write_bytes(c1_pcm)
        write_wav(c1_wav_path, c1_pcm)
        records.append(
            {
                "scenario_id": c1["scenario_id"],
                "partition": "c1",
                "expected": "positive",
                "prompt": c1["prompt"],
                "expected_transcript": c1["expected_transcript"],
                "pcm": {
                    "path": c1_pcm_path.name,
                    "bytes": len(c1_pcm),
                    "sha256": digest(c1_pcm),
                    "sample_rate": RATE,
                },
                "wav": {
                    "path": c1_wav_path.name,
                    "bytes": c1_wav_path.stat().st_size,
                    "sha256": digest(c1_wav_path.read_bytes()),
                },
            }
        )
    finally:
        await manager.stop()
    payload = {
        "schema": 1,
        "kind": "attnw8_gptq_tool_fixture_manifest",
        "scenario_source": str(args.scenarios.resolve()),
        "scenario_source_sha256": digest(args.scenarios.read_bytes()),
        "synthesis": {
            **synthesis_health,
            "language": "english_2026-04",
            "voice": "alba",
            "base_seed": 0,
            "seed_scheme": "sha256-text-plus-base-v1",
            "threads": 4,
            "cpus": [7, 8, 9, 15],
        },
        "records": records,
    }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(generate(args))
    print(json.dumps({"records": len(payload["records"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
