"""Frame ordering and state-machine tests for the realtime service."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    AggregationType,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InputTextRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from nemotron_voicechat_pipecat import events
from nemotron_voicechat_pipecat import llm as llm_module
from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService
from nemotron_voicechat_pipecat.text_input import VoicechatTypedInputFrame
from nemotron_voicechat_pipecat.tool_results import VoicechatLLMContext
from nemotron_voicechat_runtime.protocol import (
    FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS,
    FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
)


class RecordingService(NemotronVoicechatLLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames = []
        self.broadcasts = []
        self.errors = []
        self.sent = []
        self.calls = []
        self.interruptions = 0

    async def push_frame(self, frame, direction=None):
        self.frames.append(frame)

    async def broadcast_frame(self, frame_class, *args, **kwargs):
        self.broadcasts.append(frame_class)

    async def broadcast_interruption(self, *args, **kwargs):
        self.interruptions += 1

    async def push_error(self, error_msg, **kwargs):
        self.errors.append((error_msg, kwargs))

    async def start_ttfb_metrics(self):
        pass

    async def stop_ttfb_metrics(self):
        pass

    async def start_processing_metrics(self):
        pass

    async def stop_all_metrics(self):
        pass

    async def run_function_calls(self, calls):
        self.calls.extend(calls)

    async def _send(self, event):
        self.sent.append(event)


def mark_live_input_ready(service: RecordingService) -> None:
    service._session_ready.set()
    service._context_ready.set()
    service._audio_send_task = object()


def test_function_output_ack_deadline_uses_shared_protocol_contract():
    service = RecordingService()

    assert service._function_output_ack_timeout_secs == FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "value",
    [True, "0.4", float("nan"), float("inf"), float("-inf"), 0, -0.1, 1.01],
)
def test_held_unassigned_preroll_rejects_unbounded_or_invalid_config(value):
    with pytest.raises(ValueError, match="at most one second"):
        RecordingService(held_unassigned_preroll_seconds=value)


@pytest.mark.asyncio
async def test_connect_uses_separate_model_ready_timeout(monkeypatch):
    service = RecordingService()
    service._connect_timeout_secs = 2.0
    service._handshake_timeout_secs = 3.0
    service._model_ready_timeout_secs = 47.0
    observed_timeouts = []

    async def fake_websocket_connect(**_kwargs):
        return object()

    async def fake_wait_for(awaitable, *, timeout):
        observed_timeouts.append(timeout)
        if len(observed_timeouts) == 1:
            return await awaitable
        awaitable.close()
        return None

    async def fake_send_session_update():
        return None

    def fake_create_task(coro):
        coro.close()
        return object()

    monkeypatch.setattr(llm_module, "websocket_connect", fake_websocket_connect)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(service, "_send_session_update", fake_send_session_update)
    monkeypatch.setattr(service, "create_task", fake_create_task)

    await service._connect()

    assert observed_timeouts == [2.0, 3.0, 47.0]


@pytest.mark.asyncio
async def test_server_microphone_turn_events_are_telemetry_only_and_transcribe():
    service = RecordingService()

    await service._handle_server_event(
        {"type": "input_audio_buffer.speech_started", "turn_id": "turn_1"}
    )
    await service._handle_server_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "turn_id": "turn_1",
            "transcript": "hello",
        }
    )
    await service._handle_server_event(
        {"type": "input_audio_buffer.speech_stopped", "turn_id": "turn_1"}
    )
    await service._handle_server_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "turn_id": "turn_1",
            "transcript": "hello there",
        }
    )

    assert service.broadcasts == []
    assert service.interruptions == 0
    assert [type(frame) for frame in service.frames] == [
        InterimTranscriptionFrame,
        TranscriptionFrame,
    ]
    assert service.frames[-1].text == "hello there"


@pytest.mark.asyncio
async def test_text_audio_and_response_brackets_have_pipecat_order():
    service = RecordingService()
    pcm = b"\x01\x00" * 160

    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "Hello",
        }
    )
    await service._handle_server_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": 22_050,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    await service._handle_server_event({"type": "response.done", "response_id": "response_1"})

    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSTextFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    assert service.frames[1].append_to_context is False
    assert service.frames[4].includes_inter_frame_spaces is False
    assert service.frames[4].aggregated_by == AggregationType.SENTENCE
    assert service.frames[3].audio == pcm


@pytest.mark.asyncio
async def test_response_playout_prebuffer_fills_then_drains_in_order():
    service = RecordingService(prebuffer_ms=160)
    pcm = b"\x01\x00" * int(events.OUTPUT_SAMPLE_RATE * 0.08)
    await service._response_created({"response_id": "response_1"})

    await service._audio_delta(
        {
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": events.OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    assert [type(frame) for frame in service.frames] == [LLMFullResponseStartFrame]

    await service._audio_delta(
        {
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": events.OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSAudioRawFrame,
    ]
    assert service.frames[2].audio == pcm
    assert service.frames[3].audio == pcm


@pytest.mark.asyncio
async def test_short_response_drains_prebuffer_on_response_end():
    service = RecordingService(prebuffer_ms=160)
    pcm = b"\x02\x00" * int(events.OUTPUT_SAMPLE_RATE * 0.08)
    await service._response_created({"response_id": "response_1"})
    await service._audio_delta(
        {
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": events.OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    await service._response_done({"response_id": "response_1"})

    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_interruption_discards_unplayed_prebuffer():
    service = RecordingService(prebuffer_ms=160)
    pcm = b"\x03\x00" * int(events.OUTPUT_SAMPLE_RATE * 0.08)
    await service._response_created({"response_id": "response_1"})
    await service._audio_delta(
        {
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": events.OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )

    await service._reset_response_after_interruption()

    assert [type(frame) for frame in service.frames] == [LLMFullResponseStartFrame]
    assert service._playout_buffer == []
    assert service._playout_buffer_ms == 0.0


@pytest.mark.asyncio
async def test_response_playout_prebuffer_can_be_disabled():
    service = RecordingService(prebuffer_ms=0)
    pcm = b"\x04\x00" * 80
    await service._response_created({"response_id": "response_1"})
    await service._audio_delta(
        {
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": events.OUTPUT_SAMPLE_RATE,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )

    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
    ]


def test_response_playout_prebuffer_env_overrides_constructor(monkeypatch):
    monkeypatch.setenv("NEMOTRON_VOICECHAT_PREBUFFER_MS", "0")
    service = RecordingService(prebuffer_ms=999)
    assert service._prebuffer_ms == 0


@pytest.mark.asyncio
async def test_text_deltas_are_aggregated_into_sentence_frames_with_lookahead():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})

    for delta in ("I", " can", " do", " math", ".", " Also", "!"):
        await service._handle_server_event(
            {
                "type": "response.output_text.delta",
                "response_id": "response_1",
                "delta": delta,
            }
        )

    tts_frames = [frame for frame in service.frames if isinstance(frame, TTSTextFrame)]
    assert [frame.text for frame in tts_frames] == ["I can do math."]

    await service._handle_server_event({"type": "response.done", "response_id": "response_1"})

    llm_frames = [frame for frame in service.frames if isinstance(frame, LLMTextFrame)]
    tts_frames = [frame for frame in service.frames if isinstance(frame, TTSTextFrame)]
    assert [frame.text for frame in llm_frames] == [
        "I",
        " can",
        " do",
        " math",
        ".",
        " Also",
        "!",
    ]
    assert all(frame.append_to_context is False for frame in llm_frames)
    assert [frame.text for frame in tts_frames] == ["I can do math.", "Also!"]
    assert all(frame.aggregated_by == AggregationType.SENTENCE for frame in tts_frames)
    assert all(frame.includes_inter_frame_spaces is False for frame in tts_frames)
    assert " ".join(frame.text for frame in tts_frames) == "I can do math. Also!"


@pytest.mark.asyncio
async def test_response_done_flushes_pending_text_fragment_before_end():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "  pending  fragment  ",
        }
    )
    await service._handle_server_event({"type": "response.done", "response_id": "response_1"})

    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert service.frames[2].text == "pending  fragment"


@pytest.mark.asyncio
async def test_interruption_frame_discards_pending_fragment_before_tts_stop():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "Interrupted fragment",
        }
    )
    service._tts_open = True

    interruption = InterruptionFrame()
    await service.process_frame(interruption, FrameDirection.DOWNSTREAM)

    assert [type(frame) for frame in service.frames[-2:]] == [
        TTSStoppedFrame,
        InterruptionFrame,
    ]
    assert not any(
        isinstance(frame, TTSTextFrame) and frame.text == "Interrupted fragment"
        for frame in service.frames
    )
    assert service._response_id == "response_1"
    assert service._response_cancelling


@pytest.mark.asyncio
async def test_typed_speech_start_discards_incomplete_tail_and_keeps_terminal_scope():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "Speech-start fragment!",
        }
    )
    service._tts_open = True

    await service._handle_server_event(
        {
            "type": "input_audio_buffer.speech_started",
            "turn_id": "turn_2",
            "source": "typed",
        }
    )

    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSStoppedFrame,
    ]
    assert service._response_open is True
    assert service._response_id == "response_1"
    assert service._response_cancelling is True
    assert service.broadcasts == [UserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_server_cancelled_terminal_discards_incomplete_tail_without_local_interrupt():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_text.delta",
            "response_id": "response_1",
            "delta": "fragment that must not be spoken",
        }
    )

    await service._handle_server_event(
        {"type": "response.done", "response_id": "response_1", "status": "cancelled"}
    )

    assert not any(isinstance(frame, TTSTextFrame) for frame in service.frames)
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_silent_response_has_balanced_llm_brackets_only():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event({"type": "response.done", "response_id": "response_1"})
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_local_barge_in_discards_stale_deltas_but_consumes_terminal():
    service = RecordingService()
    pcm = b"\x01\x00" * 80
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": 22_050,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    await service._reset_response_after_interruption()
    await service._handle_server_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "response_1",
            "encoding": "pcm16",
            "sample_rate": 22_050,
            "channels": 1,
            "delta": base64.b64encode(pcm).decode(),
        }
    )
    await service._handle_server_event(
        {"type": "response.done", "response_id": "response_1", "status": "cancelled"}
    )

    assert service.interruptions == 0
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_matching_initial_context_is_accepted_after_audio_started():
    service = RecordingService()
    service._settings.system_instruction = "Be concise."
    service._settings.tools = []
    service._audio_sent = True

    await service._handle_context(LLMContext(messages=[]))

    assert service.errors == []


@pytest.mark.asyncio
async def test_late_context_configuration_change_is_fatal():
    service = RecordingService()
    service._settings.system_instruction = "Original"
    service._audio_sent = True

    await service._handle_context(LLMContext(messages=[{"role": "system", "content": "Changed"}]))

    assert service.errors[-1][1]["fatal"] is True
    assert "changed instructions or tools" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_server_function_timeout_closes_local_function_state():
    service = RecordingService()
    cancelled = []

    async def cancel(call_id):
        cancelled.append(call_id)

    service._cancel_function_calls_by_tool_call_id = cancel
    await service._handle_server_event(
        {
            "type": "response.function_call.failed",
            "call_id": "call_1",
            "reason": "function_call_output_timeout",
        }
    )

    assert cancelled == ["call_1"]
    assert "call_1" in service._completed_tool_calls


@pytest.mark.asyncio
async def test_local_interruption_closes_tts_without_llm_end():
    service = RecordingService()
    service._response_open = True
    service._response_id = "response_1"
    service._tts_open = True

    await service._reset_response_after_interruption()

    assert [type(frame) for frame in service.frames] == [TTSStoppedFrame]
    assert service._response_open is True
    assert service._response_id == "response_1"
    assert service._response_cancelling is True


@pytest.mark.asyncio
async def test_local_interruption_discards_incomplete_sentence_tail():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "r1"})
    await service._handle_server_event(
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "unfinished"}
    )

    await service._reset_response_after_interruption()
    await service._handle_server_event(
        {"type": "response.done", "response_id": "r1", "status": "cancelled"}
    )

    assert not any(isinstance(frame, TTSTextFrame) for frame in service.frames)


@pytest.mark.asyncio
async def test_failed_server_terminal_discards_incomplete_sentence_tail():
    service = RecordingService()
    await service._handle_server_event({"type": "response.created", "response_id": "r1"})
    await service._handle_server_event(
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "unfinished"}
    )

    await service._handle_server_event(
        {
            "type": "response.done",
            "response_id": "r1",
            "status": "failed",
            "reason": "function_call_loop_limit",
        }
    )

    assert not any(isinstance(frame, TTSTextFrame) for frame in service.frames)
    assert any(isinstance(frame, LLMFullResponseEndFrame) for frame in service.frames)


@pytest.mark.asyncio
async def test_liveness_is_not_armed_until_audio_is_sent():
    service = RecordingService()
    service._liveness_timeout_secs = 0.03
    task = asyncio.create_task(service._liveness_loop())
    await asyncio.sleep(0.06)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert service.errors == []


@pytest.mark.asyncio
async def test_pre_ready_audio_is_dropped_before_resampling_without_queue_overflow():
    service = RecordingService()

    class RejectingResampler:
        async def resample(self, *_args, **_kwargs):
            raise AssertionError("pre-ready audio must not reach the resampler")

    service._input_resampler = RejectingResampler()
    frame = InputAudioRawFrame(
        audio=b"\x01\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )
    for _ in range(129):
        await service._queue_audio(frame)

    assert service._audio_queue.empty()
    assert service._audio_queue_bytes == 0
    assert service._pre_ready_audio_chunks_dropped == 129
    assert service._pre_ready_audio_bytes_dropped == 129 * len(frame.audio)
    assert service.errors == []


@pytest.mark.asyncio
async def test_session_ack_without_context_or_sender_still_drops_audio():
    service = RecordingService()
    service._session_ready.set()
    frame = InputAudioRawFrame(
        audio=b"\x02\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )

    await service._queue_audio(frame)
    service._context_ready.set()
    await service._queue_audio(frame)

    assert service._audio_queue.empty()
    assert service._pre_ready_audio_chunks_dropped == 2


@pytest.mark.asyncio
async def test_pre_ready_turn_stays_dropped_when_readiness_arrives_mid_turn():
    service = RecordingService()
    metric_starts = []

    async def record_metric_start():
        metric_starts.append(True)

    service.start_ttfb_metrics = record_metric_start
    service.start_processing_metrics = record_metric_start
    frame = InputAudioRawFrame(
        audio=b"\x03\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )

    await service._queue_turn_start()
    dropped_turn_id = service._pre_ready_dropped_turn_id
    await service._queue_audio(frame)
    mark_live_input_ready(service)
    await service._queue_audio(frame)
    await service._queue_turn_commit()

    assert dropped_turn_id == 1
    assert service._pre_ready_dropped_turn_id is None
    assert service._audio_queue.empty()
    assert service._turn_barrier_event_ids == {}
    assert service._turn_barrier_timeout_tasks == {}
    assert metric_starts == []
    assert service.errors == []


@pytest.mark.asyncio
async def test_first_ready_turn_after_dropped_turn_is_byte_exact_and_normal():
    service = RecordingService()

    class IdentityResampler:
        async def resample(self, audio, _source_rate, _target_rate):
            return audio

    service._input_resampler = IdentityResampler()
    frame = InputAudioRawFrame(
        audio=b"\x04\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )
    await service._queue_turn_start()
    await service._queue_audio(frame)
    await service._queue_turn_commit()

    mark_live_input_ready(service)
    await service._queue_turn_start()
    await service._queue_audio(frame)
    await service._queue_turn_commit()

    queued = [await service._audio_queue.get() for _ in range(3)]
    assert queued[0]["type"] == "input_audio_buffer.turn_start"
    assert queued[0]["client_turn_id"] == 2
    assert queued[1] == frame.audio
    assert queued[2]["type"] == "input_audio_buffer.commit"
    assert queued[2]["client_turn_id"] == 2
    assert service._pre_ready_drop_summary_emitted is True


@pytest.mark.asyncio
async def test_drop_summary_waits_for_mid_ready_turn_commit(monkeypatch):
    service = RecordingService()
    service._settings.system_instruction = "Be concise."
    service._session_ready.set()
    service._audio_send_task = object()
    summaries = []
    monkeypatch.setattr(llm_module.logger, "warning", summaries.append)
    frame = InputAudioRawFrame(
        audio=b"\x06\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )

    await service._queue_turn_start()
    await service._queue_audio(frame)
    await service._handle_context(LLMContext(messages=[]))
    await service._queue_audio(frame)

    assert summaries == []
    await service._queue_turn_commit()

    assert len(summaries) == 1
    assert "chunks=2" in summaries[0]
    assert f"bytes={2 * len(frame.audio)}" in summaries[0]
    assert "turns=1" in summaries[0]


@pytest.mark.asyncio
async def test_typed_input_waits_for_startup_dropped_turn_commit():
    service = RecordingService()
    await service._queue_turn_start()
    mark_live_input_ready(service)

    await service._request_typed_input(
        VoicechatTypedInputFrame(text="after microphone", job_id="job-after")
    )

    assert len(service._pending_typed_requests) == 1
    assert service._audio_queue.empty()
    await service._queue_turn_commit()

    request = await service._audio_queue.get()
    assert request["type"] == "input_text.request"
    assert request["job_id"] == "job-after"
    assert request["text"] == "after microphone"


@pytest.mark.asyncio
async def test_pre_context_typed_input_queues_without_blocking_context_frame():
    """An overtaking typed frame cannot blockade the initial context behind itself."""
    service = RecordingService(system_instruction="Be concise.")
    service._session_ready.set()
    service._audio_send_task = object()

    await asyncio.wait_for(
        service._request_typed_input(
            VoicechatTypedInputFrame(text="early typed", job_id="job-early")
        ),
        timeout=0.05,
    )

    assert [item.job_id for item in service._pending_typed_requests] == ["job-early"]
    assert service._audio_queue.empty()
    await service._handle_context(LLMContext(messages=[]))

    request = await service._audio_queue.get()
    assert request["type"] == "input_text.request"
    assert request["job_id"] == "job-early"
    assert request["text"] == "early typed"
    assert service._pending_typed_requests == []


@pytest.mark.asyncio
async def test_pending_typed_flush_rechecks_readiness_for_every_item(monkeypatch):
    service = RecordingService()
    mark_live_input_ready(service)
    service._pending_typed_requests = [
        VoicechatTypedInputFrame(text="first", job_id="job-first"),
        VoicechatTypedInputFrame(text="second", job_id="job-second"),
    ]
    admitted = []

    async def admit(frame):
        admitted.append(frame.job_id)
        service._function_epoch_active = True

    monkeypatch.setattr(service, "_admit_typed_input", admit)
    await service._flush_pending_typed_inputs()

    assert admitted == ["job-first"]
    assert [item.job_id for item in service._pending_typed_requests] == ["job-second"]


@pytest.mark.asyncio
async def test_disconnect_reports_and_discards_pending_typed_input(monkeypatch):
    service = RecordingService()
    summaries = []
    monkeypatch.setattr(llm_module.logger, "warning", summaries.append)
    await service._request_typed_input(
        VoicechatTypedInputFrame(text="never admitted", job_id="job-dropped")
    )

    await service._disconnect()

    assert service._pending_typed_requests == []
    assert len(summaries) == 1
    assert "reason=disconnect" in summaries[0]
    assert "typed_requests=1" in summaries[0]


@pytest.mark.asyncio
async def test_session_updates_are_serialized_through_distinct_ready_acks(monkeypatch):
    service = RecordingService()
    sends = []

    async def fake_send_session_update():
        service._session_ready.clear()
        sends.append(len(sends) + 1)

    monkeypatch.setattr(service, "_send_session_update", fake_send_session_update)
    first = asyncio.create_task(service._update_session_and_wait())
    second = asyncio.create_task(service._update_session_and_wait())
    while sends != [1]:
        await asyncio.sleep(0)

    service._session_ready.set()
    await first
    while sends != [1, 2]:
        await asyncio.sleep(0)
    assert second.done() is False

    service._session_ready.set()
    await second


@pytest.mark.asyncio
async def test_control_only_open_turn_rejects_context_and_tool_changes():
    context_service = RecordingService()
    context_service._settings.system_instruction = "Original"
    mark_live_input_ready(context_service)
    await context_service._queue_turn_start()

    await context_service._handle_context(
        LLMContext(messages=[{"role": "system", "content": "Changed"}])
    )

    assert context_service.errors[-1][1]["fatal"] is True
    assert "after a microphone turn began" in context_service.errors[-1][0]
    assert context_service._settings.system_instruction == "Original"

    tool_service = RecordingService()
    mark_live_input_ready(tool_service)
    await tool_service._queue_turn_start()
    await tool_service._handle_tools([{"type": "function", "function": {"name": "x"}}])

    assert "after a microphone turn begins" in tool_service.errors[-1][0]
    assert tool_service._settings.tools == []


@pytest.mark.asyncio
async def test_unmatched_pre_ready_commit_remains_fatal():
    service = RecordingService()

    await service._queue_turn_commit()

    assert service.errors[-1][1]["fatal"] is True
    assert "without an open local turn" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_disconnect_clears_pre_ready_drop_state():
    service = RecordingService()
    frame = InputAudioRawFrame(
        audio=b"\x05\x00" * 160,
        sample_rate=16_000,
        num_channels=1,
    )
    await service._queue_turn_start()
    await service._queue_audio(frame)

    await service._disconnect()

    assert service._pre_ready_dropped_turn_id is None
    assert service._pre_ready_audio_chunks_dropped == 0
    assert service._pre_ready_audio_bytes_dropped == 0
    assert service._pre_ready_turns_dropped == 0
    assert service._pre_ready_drop_summary_emitted is False


@pytest.mark.asyncio
async def test_session_updated_must_confirm_protocol_v3():
    service = RecordingService()
    with pytest.raises(events.ProtocolError, match="confirm protocol v3"):
        await service._handle_server_event(
            {"type": "session.updated", "session": {"protocol_version": 1}}
        )


@pytest.mark.asyncio
async def test_concise_tool_result_capability_is_advertised_then_confirmed():
    service = RecordingService()
    await service._handle_server_event(
        {
            "type": "session.created",
            "protocol": {"name": events.PROTOCOL_NAME, "version": events.PROTOCOL_VERSION},
            "capabilities": {
                "input_turn_detection": "client_smart_turn_v1",
                "function_output_model_output": {
                    "mode": FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
                },
            },
            "session": {"id": "session-1"},
        }
    )
    assert service._function_output_model_output_advertised is True

    await service._handle_server_event(
        {
            "type": "session.updated",
            "session": {
                "protocol_version": events.PROTOCOL_VERSION,
                "capabilities": {
                    "function_output_model_output": FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
                },
            },
        }
    )
    assert service._function_output_model_output_negotiated is True


@pytest.mark.asyncio
async def test_rejected_text_input_does_not_leak_downstream():
    service = RecordingService()

    await service.process_frame(InputTextRawFrame(text="inject this"), FrameDirection.DOWNSTREAM)

    assert service.frames == []
    assert "does not support text" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_typed_input_is_committed_byte_exactly_once_when_injection_starts():
    service = RecordingService()
    service._context = LLMContext(messages=[])
    mark_live_input_ready(service)
    text = "  Keep these spaces.  "
    await service._request_typed_input(VoicechatTypedInputFrame(text=text, job_id="job-1"))
    request = await service._audio_queue.get()
    assert request["type"] == "input_text.request"
    assert request["text"] == text

    await service._handle_server_event({"type": "input_text.accepted", "job_id": "job-1"})
    assert service._context.get_messages() == []
    await service._handle_server_event({"type": "input_text.injection_started", "job_id": "job-1"})
    assert service._context.get_messages() == [{"role": "user", "content": text}]
    assert [frame.text for frame in service.frames if isinstance(frame, TranscriptionFrame)] == [
        text
    ]
    with pytest.raises(events.ProtocolError, match="more than once"):
        await service._handle_server_event(
            {"type": "input_text.injection_started", "job_id": "job-1"}
        )


@pytest.mark.asyncio
async def test_typed_request_waits_behind_open_microphone_commit_in_one_fifo():
    service = RecordingService()
    mark_live_input_ready(service)
    await service._queue_turn_start()
    await service._request_typed_input(VoicechatTypedInputFrame(text="typed", job_id="job-typed"))

    assert "job-typed" not in service._typed_jobs
    assert len(service._pending_typed_requests) == 1
    await service._queue_turn_commit()

    queued = [await service._audio_queue.get() for _ in range(3)]
    assert [event["type"] for event in queued] == [
        "input_audio_buffer.turn_start",
        "input_audio_buffer.commit",
        "input_text.request",
    ]
    assert service._typed_jobs == {"job-typed": "typed"}


@pytest.mark.asyncio
async def test_typed_request_waits_for_function_applied_before_held_microphone():
    service = RecordingService()
    mark_live_input_ready(service)
    service._function_epoch_active = True
    service._function_epoch_call_id = "call-1"
    await service._request_typed_input(VoicechatTypedInputFrame(text="typed", job_id="job-typed"))
    await service._hold_audio(b"mic\x00")

    await service._finish_function_epoch("call-1")

    request = await service._audio_queue.get()
    assert request["type"] == "input_text.request"
    assert service._held_input[0].payload == b"mic\x00"


@pytest.mark.asyncio
async def test_typed_rnnt_transcript_is_not_double_committed_or_emitted():
    service = RecordingService()
    service._context = LLMContext(messages=[{"role": "user", "content": "Exact"}])
    await service._handle_server_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "source": "typed",
            "job_id": "job-2",
            "transcript": "RNNT approximation",
        }
    )
    assert service.frames == []
    assert service._context.get_messages() == [{"role": "user", "content": "Exact"}]


@pytest.mark.asyncio
async def test_microphone_turn_is_held_and_released_fifo_during_typed_epoch():
    service = RecordingService()
    mark_live_input_ready(service)

    class Resampler:
        async def resample(self, audio, _source_rate, _target_rate):
            return audio

    service._input_resampler = Resampler()
    service._typed_jobs["job-1"] = "typed"
    await service._queue_audio(
        InputAudioRawFrame(
            audio=b"\x01\x00" * 160,
            sample_rate=16_000,
            num_channels=1,
        )
    )
    await service._queue_turn_start()
    await service._queue_turn_commit()

    assert service._audio_queue.empty()
    assert [
        item.payload["type"] if isinstance(item.payload, dict) else "audio"
        for item in service._held_input
    ] == [
        "audio",
        "input_audio_buffer.turn_start",
        "input_audio_buffer.commit",
    ]

    await service._handle_server_event(
        {
            "type": "input_text.injection_finished",
            "job_id": "job-1",
            "disposition": "completed",
        }
    )
    released = [await service._audio_queue.get() for _ in range(3)]
    assert isinstance(released[0], bytes)
    assert [item["type"] for item in released[1:]] == [
        "input_audio_buffer.turn_start",
        "input_audio_buffer.commit",
    ]


@pytest.mark.asyncio
async def test_held_overflow_drops_only_new_turn_and_preserves_earlier_commit():
    service = RecordingService()
    mark_live_input_ready(service)
    service._typed_jobs["job-1"] = "typed"
    service._held_input_max_bytes = 10

    await service._hold_audio(b"123456")
    await service._queue_turn_start()
    await service._queue_turn_commit()
    await service._queue_turn_start()
    overflowing_turn = service._active_client_turn_id
    await service._hold_audio(b"abcdef")
    await service._queue_turn_commit()

    assert [item.turn_id for item in service._held_input] == [1, 1, 1]
    assert service._held_input_bytes == 6
    assert any(str(overflowing_turn) in message for message, _ in service.errors)


@pytest.mark.asyncio
async def test_turn_ack_timeout_starts_after_barrier_send_and_is_fatal():
    service = RecordingService()
    event = events.audio_turn_start(1)
    service._turn_barrier_event_ids[("start", 1)] = event["event_id"]
    service._turn_barrier_timeout_secs = 0.001

    service._arm_turn_barrier_timeout(event)
    await asyncio.sleep(0.01)

    assert service.errors[-1][1]["fatal"] is True
    assert "acknowledgement timed out" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_committed_turn_without_response_start_is_fatal():
    service = RecordingService()
    event = events.audio_commit(1)
    service._turn_barrier_event_ids[("commit", 1)] = event["event_id"]
    service._response_start_timeout_secs = 0.001

    service._validate_turn_ack(
        {
            "type": "input_audio_buffer.committed",
            "client_turn_id": 1,
            "client_event_id": event["event_id"],
            "turn_id": "turn-1",
        },
        "commit",
    )
    await asyncio.sleep(0.01)

    assert service.errors[-1][1]["fatal"] is True
    assert "did not start" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_function_call_is_native_pipecat_call():
    service = RecordingService()
    service._context = LLMContext(messages=[])
    await service._handle_server_event({"type": "response.created", "response_id": "response_1"})
    await service._handle_server_event(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "response_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Austin"}',
        }
    )

    assert len(service.calls) == 1
    call = service.calls[0]
    assert isinstance(call, FunctionCallFromLLM)
    assert call.function_name == "get_weather"
    assert call.arguments == {"city": "Austin"}

    await service._handle_server_event(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "response_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Austin"}',
        }
    )
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_function_epoch_holds_microphone_until_correlated_applied_ack():
    service = RecordingService()
    mark_live_input_ready(service)
    service._context = LLMContext(messages=[])
    await service._handle_server_event({"type": "response.created", "response_id": "r1"})
    await service._handle_server_event(
        {
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": "{}",
        }
    )
    await service._hold_audio(b"audio\x00")
    await service._queue_turn_start()
    await service._queue_turn_commit()

    assert service._audio_queue.empty()
    await service._handle_server_event(
        {
            "type": "conversation.item.function_call_output.applied",
            "call_id": "call-1",
        }
    )

    released = [await service._audio_queue.get() for _ in range(3)]
    assert released[0] == b"audio\x00"
    assert [item["type"] for item in released[1:]] == [
        "input_audio_buffer.turn_start",
        "input_audio_buffer.commit",
    ]


@pytest.mark.asyncio
async def test_long_function_epoch_bounds_unassigned_audio_before_next_turn():
    """A long tool call cannot turn continuous idle PCM into seconds of debt."""

    service = RecordingService(held_unassigned_preroll_seconds=0.4)
    mark_live_input_ready(service)
    service._function_epoch_active = True
    service._function_epoch_call_id = "call-long"
    class IdentityResampler:
        async def resample(self, audio, _source_rate, _target_rate):
            return audio

    service._input_resampler = IdentityResampler()
    chunk_bytes = 640  # 20 ms of mono 16 kHz PCM16.
    chunks = [index.to_bytes(2, "little") * (chunk_bytes // 2) for index in range(500)]

    for chunk in chunks:  # Ten seconds with no VAD-owned turn.
        await service._queue_audio(
            InputAudioRawFrame(
                audio=chunk,
                sample_rate=16_000,
                num_channels=1,
            )
        )

    retained = [item.payload for item in service._held_input]
    assert retained == chunks[-20:]
    assert service._held_input_bytes == 12_800
    assert service._held_unassigned_evicted_chunks == 480
    assert service._held_unassigned_evicted_bytes == 480 * chunk_bytes

    # VAD retroactively owns only the bounded onset suffix. Audio after start
    # is real turn content and must remain byte-exact through the commit.
    await service._queue_turn_start()
    speech = [b"a\x00" * 320, b"b\x00" * 320]
    for chunk in speech:
        await service._hold_audio(chunk)
    await service._queue_turn_commit()

    assert all(item.turn_id == 1 for item in service._held_input)
    await service._finish_function_epoch("call-long")
    await service._audio_queue.put(None)
    await service._audio_send_loop()
    assert [event["type"] for event in service.sent] == (
        ["input_audio_buffer.append"] * 20
        + ["input_audio_buffer.turn_start"]
        + ["input_audio_buffer.append"] * 2
        + ["input_audio_buffer.commit"]
    )
    assert [base64.b64decode(event["audio"]) for event in service.sent[:20]] == chunks[-20:]
    assert [base64.b64decode(event["audio"]) for event in service.sent[21:23]] == speech
    assert service._last_held_input_release == {
        "preroll_cap_bytes": 12_800,
        "retained_unassigned_chunks": 20,
        "retained_unassigned_bytes": 12_800,
        "evicted_unassigned_chunks": 480,
        "evicted_unassigned_bytes": 307_200,
        "released_chunks": 24,
        "released_audio_bytes": 14_080,
    }
    assert service._held_unassigned_evicted_chunks == 0
    assert service._held_unassigned_evicted_bytes == 0
    for task in service._turn_barrier_timeout_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_typed_epoch_uses_same_bounded_unassigned_preroll_policy():
    service = RecordingService(held_unassigned_preroll_seconds=0.04)
    mark_live_input_ready(service)

    class IdentityResampler:
        async def resample(self, audio, _source_rate, _target_rate):
            return audio

    service._input_resampler = IdentityResampler()
    service._typed_jobs["job-long"] = "typed"
    chunks = [bytes([index, 0]) * 320 for index in range(5)]

    for chunk in chunks:
        await service._queue_audio(
            InputAudioRawFrame(audio=chunk, sample_rate=16_000, num_channels=1)
        )

    assert [item.payload for item in service._held_input] == chunks[-2:]
    assert service._held_input_bytes == 1_280
    assert service._held_unassigned_evicted_chunks == 3
    assert service._held_unassigned_evicted_bytes == 1_920
    await service._handle_server_event(
        {
            "type": "input_text.injection_finished",
            "job_id": "job-long",
            "disposition": "completed",
        }
    )
    assert [await service._audio_queue.get(), await service._audio_queue.get()] == chunks[-2:]
    assert service._last_held_input_release["retained_unassigned_bytes"] == 1_280


@pytest.mark.asyncio
async def test_held_audio_enforces_pcm16_alignment_and_trims_oversized_suffix():
    service = RecordingService(held_unassigned_preroll_seconds=0.00025)
    with pytest.raises(ValueError, match="PCM16"):
        await service._hold_audio(b"\x00")

    payload = bytes(range(20))
    await service._hold_audio(payload)

    assert service._held_input[0].payload == payload[-8:]
    assert service._held_input_bytes == 8
    assert service._held_unassigned_evicted_chunks == 1
    assert service._held_unassigned_evicted_bytes == 12


@pytest.mark.asyncio
async def test_global_budget_drop_is_included_in_unassigned_eviction_accounting():
    service = RecordingService()
    mark_live_input_ready(service)
    service._typed_jobs["job-long"] = "typed"
    service._held_input_max_bytes = 10
    await service._hold_audio(b"123456")
    await service._queue_turn_start()
    await service._queue_turn_commit()

    await service._hold_audio(b"abcdef")

    assert [item.payload for item in service._held_input if isinstance(item.payload, bytes)] == [
        b"123456"
    ]
    assert service._held_unassigned_evicted_chunks == 1
    assert service._held_unassigned_evicted_bytes == 6


@pytest.mark.asyncio
async def test_chained_function_calls_are_distinct_ordered_epochs():
    service = RecordingService()
    mark_live_input_ready(service)
    service._context = LLMContext(messages=[])
    await service._handle_server_event({"type": "response.created", "response_id": "r1"})

    released_turns = []
    for index in (1, 2):
        call_id = f"call-{index}"
        await service._handle_server_event(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "r1",
                "call_id": call_id,
                "name": f"lookup_{index}",
                "arguments": "{}",
            }
        )
        await service._hold_audio(f"audio-{index}\x00".encode())
        await service._queue_turn_start()
        await service._queue_turn_commit()

        assert service._audio_queue.empty()
        assert service._function_epoch_call_id == call_id
        await service._handle_server_event(
            {
                "type": "conversation.item.function_call_output.applied",
                "call_id": call_id,
            }
        )
        released_turns.append([await service._audio_queue.get() for _ in range(3)])

    assert [call.tool_call_id for call in service.calls] == ["call-1", "call-2"]
    assert [turn[0] for turn in released_turns] == [b"audio-1\x00", b"audio-2\x00"]
    assert [[event["type"] for event in turn[1:]] for turn in released_turns] == [
        ["input_audio_buffer.turn_start", "input_audio_buffer.commit"],
        ["input_audio_buffer.turn_start", "input_audio_buffer.commit"],
    ]
    assert service._function_epoch_active is False
    assert service._function_epoch_call_id is None


@pytest.mark.asyncio
async def test_tool_result_sent_once_from_context_update():
    service = RecordingService()
    service._seen_function_calls.add("call_1")
    service._function_epoch_active = True
    service._function_epoch_call_id = "call_1"
    service._context = LLMContext(messages=[])
    context = LLMContext(
        messages=[{"role": "tool", "tool_call_id": "call_1", "content": '{"value":18}'}]
    )

    service._context = context
    await service._process_completed_function_calls(send_new_results=True)
    await service._process_completed_function_calls(send_new_results=True)

    assert len(service.sent) == 1
    assert service.sent[0]["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"value":18}',
    }
    await service._handle_server_event(
        {
            "type": "conversation.item.function_call_output.applied",
            "call_id": "call_1",
        }
    )


@pytest.mark.asyncio
async def test_negotiated_concise_tool_result_is_sent_once_then_discarded():
    service = RecordingService()
    service._seen_function_calls.add("call_1")
    service._function_epoch_active = True
    service._function_epoch_call_id = "call_1"
    service._function_output_model_output_negotiated = True
    context = VoicechatLLMContext(
        messages=[
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"timezone":"UTC","iso_utc":"full"}',
            }
        ]
    )
    context.set_voicechat_model_output("call_1", "It is currently four o'clock.")
    service._context = context

    await service._process_completed_function_calls(send_new_results=True)

    assert service.sent[0]["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"timezone":"UTC","iso_utc":"full"}',
        "model_output": "It is currently four o'clock.",
    }
    assert context.voicechat_model_output("call_1") is None


@pytest.mark.asyncio
async def test_unadvertised_peer_falls_back_to_full_tool_result():
    service = RecordingService()
    service._function_epoch_active = True
    service._function_epoch_call_id = "call_1"
    context = VoicechatLLMContext(messages=[])
    context.set_voicechat_model_output("call_1", "concise")
    service._context = context

    await service._send_tool_result("call_1", "full", model_output="concise")

    assert service.sent[0]["item"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "full",
    }
    assert context.voicechat_model_output("call_1") is None


@pytest.mark.asyncio
async def test_unknown_tool_result_is_not_sent():
    service = RecordingService()
    service._context = LLMContext(
        messages=[{"role": "tool", "tool_call_id": "unknown", "content": "result"}]
    )

    await service._process_completed_function_calls(send_new_results=True)

    assert service.sent == []
    assert "unknown Voicechat call_id=unknown" in service.errors[-1][0]


def test_stereo_downmix_is_pcm16_mean():
    stereo = (1000).to_bytes(2, "little", signed=True) + (-1000).to_bytes(2, "little", signed=True)
    assert RecordingService._downmix(stereo, 2) == b"\x00\x00"


@pytest.mark.asyncio
async def test_protocol_error_is_reported_fatal_by_receive_loop_boundary():
    service = RecordingService()
    await service._handle_server_event(
        {
            "type": "error",
            "error": {"code": "internal_error", "message": "boom", "fatal": True},
        }
    )
    assert service.errors == [("Voicechat error internal_error: boom", {"fatal": True})]


def test_function_result_json_serialization_is_single_layer():
    output = {"value": 18}
    serialized = json.dumps(output)
    assert json.loads(serialized) == output
