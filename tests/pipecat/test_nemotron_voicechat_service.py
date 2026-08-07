"""Frame ordering and state-machine tests for the realtime service."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from pipecat.frames.frames import (
    FunctionCallFromLLM,
    InputTextRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from nemotron_voicechat_pipecat import events
from nemotron_voicechat_pipecat.llm import NemotronVoicechatLLMService
from nemotron_voicechat_pipecat.text_input import VoicechatTypedInputFrame


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


@pytest.mark.asyncio
async def test_server_turn_events_broadcast_exactly_once_and_transcribe():
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

    assert service.broadcasts == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
    assert service.interruptions == 1
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
        TTSTextFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    assert service.frames[1].append_to_context is False
    assert service.frames[2].includes_inter_frame_spaces is True
    assert service.frames[4].audio == pcm


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
async def test_barge_in_discards_stale_response_events():
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
    await service._handle_server_event(
        {"type": "input_audio_buffer.speech_started", "turn_id": "turn_2"}
    )
    await service._handle_server_event({"type": "response.done", "response_id": "response_1"})

    assert service.interruptions == 1
    assert [type(frame) for frame in service.frames] == [
        LLMFullResponseStartFrame,
    ]


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
async def test_interruption_flushes_unplayed_prebuffer():
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
async def test_session_updated_must_confirm_protocol_v3():
    service = RecordingService()
    with pytest.raises(events.ProtocolError, match="confirm protocol v3"):
        await service._handle_server_event(
            {"type": "session.updated", "session": {"protocol_version": 1}}
        )


@pytest.mark.asyncio
async def test_rejected_text_input_does_not_leak_downstream():
    service = RecordingService()

    await service.process_frame(InputTextRawFrame(text="inject this"), FrameDirection.DOWNSTREAM)

    assert service.frames == []
    assert "does not support text" in service.errors[-1][0]


@pytest.mark.asyncio
async def test_typed_input_is_committed_byte_exactly_once_after_server_acceptance():
    service = RecordingService()
    service._context = LLMContext(messages=[])
    service._context_ready.set()
    text = "  Keep these spaces.  "
    await service._request_typed_input(VoicechatTypedInputFrame(text=text, job_id="job-1"))
    assert service.sent[-1]["type"] == "input_text.request"
    assert service.sent[-1]["text"] == text

    await service._handle_server_event({"type": "input_text.accepted", "job_id": "job-1"})
    assert service._context.get_messages() == [{"role": "user", "content": text}]
    assert [frame.text for frame in service.frames if isinstance(frame, TranscriptionFrame)] == [
        text
    ]
    with pytest.raises(events.ProtocolError, match="more than once"):
        await service._handle_server_event({"type": "input_text.accepted", "job_id": "job-1"})


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
async def test_tool_result_sent_once_from_context_update():
    service = RecordingService()
    service._seen_function_calls.add("call_1")
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
