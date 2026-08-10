#!/usr/bin/env python3
"""Generate independent conversational PCM fixtures with qualified Pocket TTS."""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager

INPUT_RATE = 16_000
RECALL_TEXT = "What codeword did I ask you to remember?"
MEMORY_CASES = (
    ("voice-to-voice", "voice", "voice", "sapphire"),
    ("voice-to-text", "voice", "text", "topaz"),
    ("text-to-voice", "text", "voice", "cobalt"),
    ("text-to-text", "text", "text", "amber"),
    ("system-to-voice", "system", "voice", "garnet"),
    ("system-to-text", "system", "text", "indigo"),
)


def memory_seed_text(secret: str) -> str:
    return f"Remember that the codeword is {secret} and acknowledge it briefly."


TURNS = (
    {
        "id": "remember-codeword",
        "text": memory_seed_text("sapphire"),
        "expected_response_any": ["sapphire", "noted", "acknowledged", "got it"],
    },
    {
        "id": "simple-math",
        "text": "What is two plus three?",
        "expected_response_any": ["five", "5"],
    },
    {
        "id": "recall-codeword",
        "text": RECALL_TEXT,
        "expected_response_any": ["sapphire"],
    },
)


def write_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(INPUT_RATE)
        output.writeframes(pcm16)


async def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=False)
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        cpus=tuple(args.cpus),
        threads=args.threads,
        startup_timeout=args.startup_timeout,
    )
    fixtures = []
    first_turn_pcm = b""
    memory_cases = []
    try:
        await manager.start()
        for index, fixture in enumerate(TURNS, 1):
            result = await manager.synthesize(
                fixture["text"],
                job_id=f"fixture-{index}",
                cancel_event=asyncio.Event(),
            )
            path = args.output / f"turn-{index:02d}.wav"
            write_wav(path, result.pcm16)
            if index == 1:
                first_turn_pcm = result.pcm16
            fixtures.append({**fixture, "audio_path": str(path)})

        for case_id, seed, recall, secret in MEMORY_CASES:
            case = {
                "case_id": case_id,
                "seed_modality": seed,
                "recall_modality": recall,
                "secret": secret,
                "seed_text": memory_seed_text(secret),
                "recall_text": RECALL_TEXT,
            }
            if seed == "voice":
                if secret == "sapphire":
                    seed_path = Path(fixtures[0]["audio_path"])
                else:
                    result = await manager.synthesize(
                        case["seed_text"],
                        job_id=f"memory-{case_id}-seed",
                        cancel_event=asyncio.Event(),
                    )
                    seed_path = args.output / f"memory-{case_id}-seed.wav"
                    write_wav(seed_path, result.pcm16)
                case["seed_audio_path"] = str(seed_path)
            if recall == "voice":
                case["recall_audio_path"] = fixtures[2]["audio_path"]
            memory_cases.append(case)
    finally:
        await manager.stop()

    # The browser first completes a typed turn, then receives real microphone
    # speech over SmallWebRTC. Leading silence keeps those two assertions
    # causally separate without browser-specific message injection.
    browser_path = args.output / "browser-microphone.wav"
    browser_pcm = bytes(round(args.browser_leading_silence_seconds * INPUT_RATE) * 2)
    browser_pcm += first_turn_pcm
    browser_pcm += bytes(5 * INPUT_RATE * 2)
    write_wav(browser_path, browser_pcm)

    manifest = {
        "schema": 2,
        "generator": "Pocket TTS 2.1.0 CPU worker, qualified alba voice",
        "turns": fixtures,
        "memory_matrix": {
            "cases": memory_cases,
            "negative_control": {
                "case_id": "unseeded-to-text",
                "seed_modality": "none",
                "recall_modality": "text",
                "recall_text": RECALL_TEXT,
            },
        },
        "browser_microphone_wav": str(browser_path),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", default=[7, 8, 9, 15])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--browser-leading-silence-seconds", type=float, default=12)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "error.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if "turns" in report else 1)


if __name__ == "__main__":
    main()
