from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from nemotron_voicechat_runtime import qualification_audio


def load_module():
    path = Path("tools/qualification/multiturn_strict_v3.py")
    spec = importlib.util.spec_from_file_location("multiturn_strict_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_memory_fixture_accepts_acknowledgement_but_recall_requires_secret() -> None:
    path = Path("tools/qualification/generate_multiturn_fixtures.py")
    spec = importlib.util.spec_from_file_location("generate_multiturn_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    acknowledgement, math, recall = module.TURNS
    assert acknowledgement["text"] == (
        "Remember that the codeword is sapphire and acknowledge it briefly."
    )
    assert "noted" in acknowledgement["expected_response_any"]
    assert math["text"] == "What is two plus three?"
    assert recall["expected_response_any"] == ["sapphire"]
    assert "sapphire" not in recall["text"].casefold()


def test_fixture_reader_resamples_and_mixes_to_mono(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "stereo.wav"
    mono = (np.sin(np.arange(24_000) * 2 * np.pi * 220 / 24_000) * 8_000).astype("<i2")
    stereo = np.column_stack((mono, mono)).reshape(-1).astype("<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(stereo.tobytes())
    payload = module.read_fixture(source)
    assert len(payload) == 16_000 * 2


@pytest.mark.asyncio
async def test_continuation_silence_is_paced_until_response_stop() -> None:
    module = load_module()
    stop = asyncio.Event()

    class RecordingWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))
            if len(self.messages) == 3:
                stop.set()

    websocket = RecordingWebSocket()
    await module.send_silence_until_stopped(websocket, "turn-1", stop)

    assert len(websocket.messages) == 3
    assert [message["event_id"] for message in websocket.messages] == [
        "turn-1-continuation-0",
        "turn-1-continuation-1",
        "turn-1-continuation-2",
    ]
    assert all(message["type"] == "input_audio_buffer.append" for message in websocket.messages)
    assert all(message["sample_rate"] == 16_000 for message in websocket.messages)
    assert all(
        base64.b64decode(str(message["audio"])) == bytes(module.FRAME_SAMPLES * 2)
        for message in websocket.messages
    )


@pytest.mark.parametrize("trailing_frames", [0, 2])
@pytest.mark.asyncio
async def test_audio_turn_helper_preserves_exact_barriers_pcm_and_padding(
    monkeypatch: pytest.MonkeyPatch, trailing_frames: int
) -> None:
    module = load_module()
    assert module.send_audio is qualification_audio.send_audio

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(qualification_audio.asyncio, "sleep", no_sleep)

    class RecordingWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

    payload = b"\x01\x02" * (module.FRAME_SAMPLES + 5)
    websocket = RecordingWebSocket()
    await module.send_audio(
        websocket,
        payload,
        "fixture",
        trailing_frames,
        client_turn_id=7,
    )

    assert [message["type"] for message in websocket.messages] == [
        "input_audio_buffer.turn_start",
        *(["input_audio_buffer.append"] * (2 + trailing_frames)),
        "input_audio_buffer.commit",
    ]
    assert websocket.messages[0]["client_turn_id"] == 7
    assert websocket.messages[-1]["client_turn_id"] == 7
    assert [message["event_id"] for message in websocket.messages] == [
        "fixture-start",
        *[f"fixture-{index}" for index in range(2 + trailing_frames)],
        "fixture-commit",
    ]
    appended = websocket.messages[1:-1]
    assert all(
        message["encoding"] == "pcm16"
        and message["sample_rate"] == 16_000
        and message["channels"] == 1
        for message in appended
    )
    decoded = [base64.b64decode(str(message["audio"])) for message in appended]
    frame_bytes = module.FRAME_SAMPLES * 2
    expected = payload + bytes(2 * frame_bytes - len(payload))
    assert b"".join(decoded[:2]) == expected
    assert decoded[2:] == [bytes(frame_bytes)] * trailing_frames
