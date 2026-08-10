"""Network-level service test against a deterministic strict-v3 server."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from websockets.asyncio.server import serve

from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService


class NetworkRecordingService(NemotronVoicechatLLMService):
    def __init__(self, url: str):
        super().__init__(base_url=url, liveness_timeout_secs=60)
        self.frames = []
        self.errors = []

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)

    async def push_error(self, error_msg, **kwargs):
        self.errors.append((error_msg, kwargs))

    def create_task(self, coroutine, *args, **kwargs):
        return asyncio.create_task(coroutine)

    async def cancel_task(self, task, timeout=None):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def stop_ttfb_metrics(self):
        pass

    async def stop_all_metrics(self):
        pass


@pytest.mark.asyncio
async def test_input_queue_overflow_reports_once_and_disables_input(monkeypatch):
    service = NetworkRecordingService("ws://unused")

    class Resampler:
        async def resample(self, audio, _source_rate, _target_rate):
            return audio

    async def overflow(_awaitable, timeout):
        del timeout
        _awaitable.close()
        raise TimeoutError

    disconnects = 0

    async def disconnect():
        nonlocal disconnects
        disconnects += 1

    service._input_resampler = Resampler()
    service._session_ready.set()
    service._context_ready.set()
    service._audio_send_task = object()
    monkeypatch.setattr(asyncio, "wait_for", overflow)
    monkeypatch.setattr(service, "_disconnect", disconnect)
    frame = InputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16_000, num_channels=1)

    await service._queue_audio(frame)
    await service._queue_audio(frame)

    assert [message for message, _ in service.errors] == ["Voicechat input audio queue overflow"]
    assert disconnects == 1
    assert service._input_failed is True


@pytest.mark.asyncio
async def test_strict_handshake_audio_response_and_graceful_close():
    observed = []
    server_finished = asyncio.Event()

    async def handler(websocket):
        await websocket.send(
            json.dumps(
                {
                    "type": "session.created",
                    "event_id": "evt_1",
                    "protocol": {"name": "voicechat.realtime", "version": 3},
                    "session": {"id": "session_test"},
                    "capabilities": {
                        "input_turn_detection": "client_smart_turn_v1"
                    },
                }
            )
        )
        update = json.loads(await websocket.recv())
        observed.append(update)
        await websocket.send(
            json.dumps(
                {
                    "type": "session.updated",
                    "event_id": "evt_2",
                    "session": {"id": "session_test", "protocol_version": 3},
                }
            )
        )
        append = json.loads(await websocket.recv())
        observed.append(append)
        response_id = "response_test"
        await websocket.send(json.dumps({"type": "response.created", "response_id": response_id}))
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "encoding": "pcm16",
                    "sample_rate": 22_050,
                    "channels": 1,
                    "delta": base64.b64encode(b"\x01\x00" * 80).decode(),
                }
            )
        )
        await websocket.send(json.dumps({"type": "response.done", "response_id": response_id}))
        stop = json.loads(await websocket.recv())
        observed.append(stop)
        await websocket.send(
            json.dumps(
                {
                    "type": "session.closed",
                    "event_id": "evt_9",
                    "status": "completed",
                    "reason": "client_requested",
                }
            )
        )
        server_finished.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = NetworkRecordingService(f"ws://127.0.0.1:{port}")
        await service._connect()
        await service._handle_context(LLMContext(messages=[]))
        await service._audio_queue.put(b"\x00\x00" * 160)

        for _ in range(100):
            if any(isinstance(frame, LLMFullResponseEndFrame) for frame in service.frames):
                break
            await asyncio.sleep(0.01)
        await service._graceful_disconnect()
        await asyncio.wait_for(server_finished.wait(), timeout=1)

    assert service.errors == []
    assert observed[0]["session"]["protocol_version"] == 3
    assert observed[1]["type"] == "input_audio_buffer.append"
    assert observed[2]["type"] == "session.stop"
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
