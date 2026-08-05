#!/usr/bin/env python3
"""Drive strict-v3 with independently synthesized multi-turn audio fixtures."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_SAMPLES = 1_280
FRAME_SECONDS = FRAME_SAMPLES / INPUT_RATE
NUMBER_EQUIVALENTS = {
    "five": "5",
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
    expected, actual = normalize(reference), normalize(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_token in enumerate(expected, 1):
        current = [row]
        for column, actual_token in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_token != actual_token),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def read_fixture(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        values = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    if width != 2:
        raise ValueError(f"{path}: expected PCM16 WAV")
    samples = values.astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_RATE:
        count = round(samples.size * INPUT_RATE / rate)
        samples = np.interp(
            np.arange(count, dtype=np.float64) * rate / INPUT_RATE,
            np.arange(samples.size, dtype=np.float64),
            samples,
        ).astype(np.float32)
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(OUTPUT_RATE)
        output.writeframes(payload)


async def send_audio(
    websocket: Any, payload: bytes, event_prefix: str, trailing_frames: int
) -> None:
    frame_bytes = FRAME_SAMPLES * 2
    padded = payload + bytes((-len(payload)) % frame_bytes)
    frames = [
        padded[offset : offset + frame_bytes] for offset in range(0, len(padded), frame_bytes)
    ]
    frames.extend([bytes(frame_bytes)] * trailing_frames)
    started = time.monotonic()
    for index, frame in enumerate(frames):
        deadline = started + index * FRAME_SECONDS
        if (delay := deadline - time.monotonic()) > 0:
            await asyncio.sleep(delay)
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{event_prefix}-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": base64.b64encode(frame).decode(),
                }
            )
        )


async def send_silence_until_stopped(
    websocket: Any, event_prefix: str, stop: asyncio.Event
) -> None:
    """Keep the frame-driven model advancing until its response closes."""

    frame = bytes(FRAME_SAMPLES * 2)
    started = time.monotonic()
    index = 0
    while not stop.is_set():
        deadline = started + (index + 1) * FRAME_SECONDS
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=max(0.0, deadline - time.monotonic())
            )
            break
        except TimeoutError:
            pass
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{event_prefix}-continuation-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": base64.b64encode(frame).decode(),
                }
            )
        )
        index += 1


async def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    event_stream = (args.output / "events.jsonl").open("w", encoding="utf-8")
    results: list[dict[str, Any]] = []

    async with websockets.connect(
        args.url,
        open_timeout=30,
        ping_interval=None,
        max_size=16 * 1024 * 1024,
    ) as websocket:
        created = json.loads(await websocket.recv())
        event_stream.write(json.dumps(created, sort_keys=True) + "\n")
        if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
            raise RuntimeError(f"unexpected protocol: {created.get('protocol')}")
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "protocol_version": 3,
                        "instructions": "Answer briefly and preserve conversational memory.",
                        "tools": [],
                    },
                }
            )
        )
        while True:
            event = json.loads(await websocket.recv())
            event_stream.write(json.dumps(event, sort_keys=True) + "\n")
            if event.get("type") == "session.updated":
                break

        for turn_index, fixture in enumerate(manifest["turns"], 1):
            active: dict[str, Any] | None = None
            input_text = ""
            payload = read_fixture(Path(fixture["audio_path"]))
            await send_audio(websocket, payload, fixture["id"], args.trailing_silence_frames)
            continuation_stop = asyncio.Event()
            continuation = asyncio.create_task(
                send_silence_until_stopped(websocket, fixture["id"], continuation_stop)
            )
            try:
                deadline = time.monotonic() + args.turn_timeout_seconds
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=deadline - time.monotonic()
                    )
                    event = json.loads(raw)
                    event_stream.write(json.dumps(event, sort_keys=True) + "\n")
                    kind = event.get("type")
                    if kind == "conversation.item.input_audio_transcription.delta":
                        input_text += str(event.get("delta") or "")
                    elif kind == "response.created":
                        active = {
                            "response_id": event.get("response_id"),
                            "text": "",
                            "audio": [],
                        }
                    elif kind == "response.output_text.delta" and active is not None:
                        active["text"] += str(event.get("delta") or "")
                    elif kind == "response.output_audio.delta" and active is not None:
                        active["audio"].append(base64.b64decode(event["delta"]))
                    elif kind == "error":
                        raise RuntimeError(f"server error: {event}")
                    elif kind == "response.done" and active is not None:
                        audio = b"".join(active.pop("audio"))
                        audio_path = args.output / f"response-{turn_index:03d}.wav"
                        write_wav(audio_path, audio)
                        results.append(
                            {
                                "id": fixture["id"],
                                "expected_input_text": fixture["text"],
                                "displayed_input_text": input_text,
                                "expected_response_any": fixture.get(
                                    "expected_response_any", []
                                ),
                                "response_id": active["response_id"],
                                "text": active["text"],
                                "audio_path": str(audio_path),
                                "audio_bytes": len(audio),
                            }
                        )
                        result = results[-1]
                        result["input_wer"] = round(
                            word_error_rate(
                                result["expected_input_text"],
                                result["displayed_input_text"],
                            ),
                            4,
                        )
                        expected = result["expected_response_any"]
                        result["semantic_match"] = not expected or any(
                            contains_token_sequence(result["text"], candidate)
                            for candidate in expected
                        )
                        break
                else:
                    raise RuntimeError(f"turn timed out: {fixture['id']}")
            finally:
                continuation_stop.set()
                await continuation

        await websocket.send(json.dumps({"type": "session.stop"}))
        session_closed = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
            event_stream.write(json.dumps(event, sort_keys=True) + "\n")
            if event.get("type") == "session.closed":
                session_closed = event
                break

    event_stream.close()
    report = {
        "passed": (
            len(results) == len(manifest["turns"])
            and all(
                result["audio_bytes"] > 0
                and result["text"]
                and result["input_wer"] <= args.max_input_wer
                and result["semantic_match"]
                for result in results
            )
            and session_closed is not None
        ),
        "fixture_generator": manifest.get("generator"),
        "manifest": str(args.manifest.resolve()),
        "responses": results,
        "session_closed": session_closed,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trailing-silence-frames", type=int, default=30)
    parser.add_argument("--turn-timeout-seconds", type=float, default=60)
    parser.add_argument("--max-input-wer", type=float, default=0.35)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
