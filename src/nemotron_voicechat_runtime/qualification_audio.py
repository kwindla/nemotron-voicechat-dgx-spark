"""Shared strict-v3 PCM fixture and client-owned turn helpers."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

INPUT_RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_SECONDS = FRAME_SAMPLES / INPUT_RATE


def read_fixture(path: Path) -> bytes:
    """Read a PCM16 WAV and return mono 16 kHz PCM16 bytes."""

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


async def send_audio(
    websocket: Any,
    payload: bytes,
    event_prefix: str,
    trailing_frames: int,
    client_turn_id: int,
) -> None:
    """Pace one fixture through the explicit client Smart-Turn lifecycle."""

    frame_bytes = FRAME_SAMPLES * 2
    padded = payload + bytes((-len(payload)) % frame_bytes)
    frames = [
        padded[offset : offset + frame_bytes]
        for offset in range(0, len(padded), frame_bytes)
    ]
    frames.extend([bytes(frame_bytes)] * trailing_frames)
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.turn_start",
                "event_id": f"{event_prefix}-start",
                "client_turn_id": client_turn_id,
            }
        )
    )
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
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.commit",
                "event_id": f"{event_prefix}-commit",
                "client_turn_id": client_turn_id,
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
