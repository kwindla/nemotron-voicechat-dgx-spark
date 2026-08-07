"""Pipecat realtime service for NVIDIA Nemotron Voicechat.

The service is intentionally a strict adapter for ``voicechat.realtime`` v3:
PCM16 audio, server-owned RNNT turns, native assistant audio/text, and native
function calls.  A disconnected model session is never transparently resumed.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    InputTextRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, assert_given
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect as websocket_connect

from . import events
from .adapter import NemotronVoicechatLLMAdapter
from .text_input import VoicechatTypedInputFrame


@dataclass
class NemotronVoicechatLLMSettings(LLMSettings):
    """Settings fixed during the pre-audio Voicechat handshake."""

    tools: ToolsSchema | list[dict[str, Any]] | None = field(default_factory=list)


class NemotronVoicechatLLMService(LLMService[NemotronVoicechatLLMAdapter]):
    """Continuous speech-to-speech Pipecat service for Voicechat v3."""

    Settings = NemotronVoicechatLLMSettings
    adapter_class = NemotronVoicechatLLMAdapter

    def __init__(
        self,
        *,
        base_url: str = "ws://127.0.0.1:8786/v1/realtime",
        system_instruction: str | None = None,
        tools: ToolsSchema | list[dict[str, Any]] | None = None,
        settings: NemotronVoicechatLLMSettings | None = None,
        connect_timeout_secs: float = 15.0,
        handshake_timeout_secs: float = 15.0,
        graceful_close_timeout_secs: float = 8.0,
        liveness_timeout_secs: float = 45.0,
        audio_queue_max_chunks: int = 128,
        prebuffer_ms: int = 160,
        **kwargs,
    ):
        defaults = self.Settings(
            model="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            system_instruction=system_instruction,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            tools=tools or [],
            extra={},
        )
        if settings is not None:
            defaults.apply_update(settings)
        super().__init__(
            settings=defaults,
            function_call_timeout_secs=kwargs.pop("function_call_timeout_secs", 20.0),
            **kwargs,
        )

        self.base_url = base_url
        self._connect_timeout_secs = connect_timeout_secs
        self._handshake_timeout_secs = handshake_timeout_secs
        self._graceful_close_timeout_secs = graceful_close_timeout_secs
        self._liveness_timeout_secs = liveness_timeout_secs

        configured_prebuffer_ms = os.environ.get("NEMOTRON_VOICECHAT_PREBUFFER_MS")
        if configured_prebuffer_ms is not None:
            try:
                prebuffer_ms = int(configured_prebuffer_ms)
            except ValueError as exc:
                raise ValueError(
                    "NEMOTRON_VOICECHAT_PREBUFFER_MS must be a non-negative integer"
                ) from exc
        if prebuffer_ms < 0:
            raise ValueError("prebuffer_ms must be a non-negative integer")
        self._prebuffer_ms = prebuffer_ms
        self._playout_buffer: list[TTSAudioRawFrame] = []
        self._playout_buffer_ms = 0.0
        self._playout_started = False

        self._websocket = None
        self._receive_task: asyncio.Task | None = None
        self._audio_send_task: asyncio.Task | None = None
        self._liveness_task: asyncio.Task | None = None
        self._session_created = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._context_ready = asyncio.Event()
        self._session_closed = asyncio.Event()
        self._disconnecting = False
        self._last_server_event_at = 0.0

        self._input_resampler = create_stream_resampler()
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=audio_queue_max_chunks
        )
        self._audio_sent = False
        self._input_failed = False
        self._context: LLMContext | None = None
        self._completed_tool_calls: set[str] = set()
        self._seen_function_calls: set[str] = set()
        self._typed_jobs: dict[str, str] = {}
        self._typed_context_commits: set[str] = set()

        self._response_id: str | None = None
        self._response_open = False
        self._tts_open = False
        self._user_id = ""
        self._session_id: str | None = None

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Advertise native server turn decisions to Pipecat aggregators."""
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            user_turn_strategies=ExternalUserTurnStrategies(),
        )

    def can_generate_metrics(self) -> bool:
        return True

    def _service_tools(self):
        return self._settings.tools

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._graceful_disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def cleanup(self):
        await self._disconnect()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, LLMSetToolsFrame):
            await self._handle_tools(frame.tools)
        elif isinstance(frame, InputAudioRawFrame):
            await self._queue_audio(frame)
        elif isinstance(frame, VoicechatTypedInputFrame):
            await self._request_typed_input(frame)
            return
        elif isinstance(frame, InterruptionFrame):
            await self._reset_response_after_interruption()
        elif isinstance(frame, (InputTextRawFrame, LLMMessagesAppendFrame)):
            await self.push_error(
                error_msg=(
                    "Voicechat is a continuous audio session and does not support "
                    "text or conversation-history injection"
                )
            )
            return

        await self.push_frame(frame, direction)

    async def _connect(self):
        if self._websocket is not None:
            return
        self._input_failed = False
        try:
            self._websocket = await asyncio.wait_for(
                websocket_connect(
                    uri=self.base_url,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=self._connect_timeout_secs,
            )
            self._last_server_event_at = time.monotonic()
            self._receive_task = self.create_task(self._receive_loop())
            await asyncio.wait_for(
                self._session_created.wait(), timeout=self._handshake_timeout_secs
            )
            await self._send_session_update()
            await asyncio.wait_for(self._session_ready.wait(), timeout=self._handshake_timeout_secs)
            self._audio_send_task = self.create_task(self._audio_send_loop())
            self._liveness_task = self.create_task(self._liveness_loop())
        except Exception as exc:  # noqa: BLE001 - surface transport/library failures
            await self.push_error(
                error_msg=f"Voicechat connection failed: {exc}",
                exception=exc,
                fatal=True,
            )
            await self._disconnect()

    async def _disconnect(self):
        if self._disconnecting:
            return
        self._disconnecting = True
        self._input_failed = True
        websocket, self._websocket = self._websocket, None
        for task_name in ("_liveness_task", "_audio_send_task", "_receive_task"):
            task = getattr(self, task_name)
            if task is not None and task is not asyncio.current_task():
                await self.cancel_task(task, timeout=1.0)
            setattr(self, task_name, None)
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logger.debug(f"{self}: websocket close during teardown failed: {exc}")
        self._disconnecting = False

    async def _graceful_disconnect(self):
        if self._websocket is None:
            return
        deadline = time.monotonic() + self._graceful_close_timeout_secs
        silence = b"\x00\x00" * int(events.INPUT_SAMPLE_RATE * 0.08)
        while self._response_open and time.monotonic() < deadline:
            try:
                await self._send(events.audio_append(silence))
            except Exception as exc:  # noqa: BLE001 - remote may close while draining
                logger.debug(f"{self}: response drain ended during remote close: {exc}")
                break
            try:
                await asyncio.wait_for(self._session_closed.wait(), timeout=0.08)
            except TimeoutError:
                pass
        try:
            await self._send(events.session_stop())
        except Exception as exc:  # noqa: BLE001 - graceful stop is best-effort
            logger.debug(f"{self}: session.stop skipped after remote close: {exc}")
            await self._disconnect()
            return
        try:
            await asyncio.wait_for(
                self._session_closed.wait(), timeout=self._graceful_close_timeout_secs
            )
        except TimeoutError:
            logger.warning(f"{self}: Voicechat graceful close timed out")
        await self._disconnect()

    async def _send(self, event: dict[str, Any]):
        websocket = self._websocket
        if websocket is None or self._disconnecting:
            raise RuntimeError("Voicechat websocket is not connected")
        await websocket.send(json.dumps(event, separators=(",", ":")))

    async def _send_session_update(self):
        instruction = assert_given(self._settings.system_instruction)
        tools = self.get_llm_adapter().from_standard_tools(self._settings.tools)
        if not isinstance(tools, list):
            tools = []
        await self._send(events.session_update(instructions=instruction, tools=tools))

    async def _request_typed_input(self, frame: VoicechatTypedInputFrame) -> None:
        await self._context_ready.wait()
        if frame.job_id in self._typed_jobs or frame.job_id in self._typed_context_commits:
            await self.push_error(error_msg=f"Duplicate Voicechat typed job ID: {frame.job_id}")
            return
        self._typed_jobs[frame.job_id] = frame.text
        try:
            await self._send(events.typed_input_request(text=frame.text, job_id=frame.job_id))
        except Exception:
            self._typed_jobs.pop(frame.job_id, None)
            raise

    async def _handle_context(self, context: LLMContext):
        first_context = self._context is None
        self._context = context
        self._sync_registered_tool_handlers(context.tools)
        if first_context:
            params = self.get_llm_adapter().get_llm_invocation_params(
                context,
                system_instruction=assert_given(self._settings.system_instruction),
            )
            await self._process_completed_function_calls(send_new_results=False)
            current_tools = self.get_llm_adapter().from_standard_tools(self._settings.tools)
            current_tools = current_tools if isinstance(current_tools, list) else []
            settings_changed = (
                params["system_instruction"] != assert_given(self._settings.system_instruction)
                or params["tools"] != current_tools
            )
            if self._audio_sent and settings_changed:
                await self.push_error(
                    error_msg=(
                        "Initial Voicechat context changed instructions or tools after "
                        "audio streaming began"
                    ),
                    fatal=True,
                )
                return
            self._settings.system_instruction = params["system_instruction"]
            self._settings.tools = params["tools"]
            if self._session_ready.is_set() and not self._audio_sent and settings_changed:
                await self._send_session_update()
            self._context_ready.set()
        else:
            await self._process_completed_function_calls(send_new_results=True)

    async def _handle_tools(self, tools):
        if self._audio_sent:
            await self.push_error(
                error_msg="Voicechat tools are immutable after audio streaming begins"
            )
            return
        self._sync_registered_tool_handlers(tools)
        converted = self.get_llm_adapter().from_standard_tools(tools)
        self._settings.tools = converted if isinstance(converted, list) else []
        if self._session_ready.is_set():
            await self._send_session_update()

    @staticmethod
    def _downmix(audio: bytes, channels: int) -> bytes:
        if channels == 1:
            return audio
        if channels < 1:
            raise ValueError("audio channel count must be positive")
        samples = np.frombuffer(audio, dtype=np.int16)
        usable = len(samples) - (len(samples) % channels)
        if usable == 0:
            return b""
        mono = samples[:usable].reshape(-1, channels).astype(np.int32).mean(axis=1)
        return np.clip(mono, -32768, 32767).astype(np.int16).tobytes()

    async def _queue_audio(self, frame: InputAudioRawFrame):
        if self._input_failed:
            return
        mono = self._downmix(frame.audio, frame.num_channels)
        pcm = await self._input_resampler.resample(
            mono, frame.sample_rate, events.INPUT_SAMPLE_RATE
        )
        if not pcm:
            return
        try:
            await asyncio.wait_for(self._audio_queue.put(pcm), timeout=0.5)
        except TimeoutError:
            self._input_failed = True
            await self.push_error(error_msg="Voicechat input audio queue overflow", fatal=True)
            await self._disconnect()

    async def _audio_send_loop(self):
        next_send = time.monotonic()
        try:
            await self._context_ready.wait()
            while True:
                pcm = await self._audio_queue.get()
                if pcm is None:
                    return
                delay = next_send - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await self._send(events.audio_append(pcm))
                self._audio_sent = True
                next_send = max(next_send, time.monotonic()) + len(pcm) / (
                    2 * events.INPUT_SAMPLE_RATE
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface transport failures
            self._input_failed = True
            await self.push_error(
                error_msg=f"Voicechat audio send failed: {exc}",
                exception=exc,
                fatal=True,
            )
            await self._disconnect()

    async def _liveness_loop(self):
        while True:
            await asyncio.sleep(min(5.0, self._liveness_timeout_secs / 3))
            if not self._audio_sent:
                continue
            if time.monotonic() - self._last_server_event_at > self._liveness_timeout_secs:
                await self.push_error(
                    error_msg="Voicechat server event liveness timeout", fatal=True
                )
                await self._disconnect()
                return

    async def _receive_loop(self):
        websocket = self._websocket
        try:
            async for message in websocket:
                self._last_server_event_at = time.monotonic()
                event = events.parse_server_event(message)
                await self._handle_server_event(event)
            if not self._disconnecting and not self._session_closed.is_set():
                await self.push_error(
                    error_msg="Voicechat websocket closed without session.closed",
                    fatal=True,
                )
                await self._disconnect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface transport/protocol failures
            if not self._disconnecting:
                await self.push_error(
                    error_msg=f"Voicechat receive failed: {exc}",
                    exception=exc,
                    fatal=True,
                )

    async def _handle_server_event(self, event: dict[str, Any]):
        event_type = event["type"]
        if event_type == "session.created":
            protocol = event.get("protocol") or {}
            if (
                protocol.get("name") != events.PROTOCOL_NAME
                or protocol.get("version") != events.PROTOCOL_VERSION
            ):
                raise events.ProtocolError("server does not support voicechat.realtime v3")
            self._session_id = (event.get("session") or {}).get("id")
            self._session_created.set()
        elif event_type == "session.updated":
            session = event.get("session") or {}
            if session.get("protocol_version") != events.PROTOCOL_VERSION:
                raise events.ProtocolError("session.updated did not confirm protocol v3")
            self._session_ready.set()
        elif event_type == "input_text.accepted":
            await self._typed_input_accepted(event)
        elif event_type == "input_text.rejected":
            await self._typed_input_rejected(event)
        elif event_type == "input_text.injection_started":
            return
        elif event_type == "input_text.injection_finished":
            await self._typed_input_finished(event)
        elif event_type == "input_audio_buffer.speech_started":
            await self._speech_started()
        elif event_type == "input_audio_buffer.speech_stopped":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(UserStoppedSpeakingFrame)
        elif event_type == "conversation.item.input_audio_transcription.delta":
            if event.get("source") == "typed":
                return
            text = event.get("transcript") or ""
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text, user_id=self._user_id, timestamp=time_now_iso8601()
                    )
                )
        elif event_type == "conversation.item.input_audio_transcription.completed":
            if event.get("source") == "typed":
                return
            text = event.get("transcript") or ""
            if text.strip():
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        finalized=True,
                    )
                )
        elif event_type == "response.created":
            await self._response_created(event)
        elif event_type == "response.output_text.delta":
            await self._text_delta(event)
        elif event_type == "response.output_audio.delta":
            await self._audio_delta(event)
        elif event_type == "response.function_call_arguments.done":
            await self._function_call(event)
        elif event_type == "response.function_call.failed":
            call_id = event.get("call_id")
            if isinstance(call_id, str):
                self._completed_tool_calls.add(call_id)
                await self._cancel_function_calls_by_tool_call_id(call_id)
            await self.push_error(
                error_msg=f"Voicechat function call {call_id} failed: {event.get('reason')}"
            )
        elif event_type == "response.done":
            await self._response_done(event)
        elif event_type == "error":
            detail = event.get("error") or {}
            await self.push_error(
                error_msg=f"Voicechat error {detail.get('code')}: {detail.get('message')}",
                fatal=bool(detail.get("fatal")),
            )
            if detail.get("fatal"):
                await self._disconnect()
        elif event_type == "session.closed":
            logger.info(
                f"{self}: Voicechat session closed status={event.get('status')} "
                f"reason={event.get('reason')}"
            )
            self._session_closed.set()
        elif event_type in {
            "response.output_text.done",
            "response.output_audio.done",
            "voicechat.metrics",
        }:
            return
        else:
            logger.debug(f"{self}: ignoring additive server event {event_type}")

    async def _typed_input_accepted(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or job_id not in self._typed_jobs:
            raise events.ProtocolError("input_text.accepted references an unknown job")
        if job_id in self._typed_context_commits:
            raise events.ProtocolError("typed input was accepted more than once")
        if self._context is None:
            raise events.ProtocolError("typed input was accepted before context initialization")
        text = self._typed_jobs[job_id]
        self._context.add_message({"role": "user", "content": text})
        self._typed_context_commits.add(job_id)
        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                finalized=True,
            )
        )

    async def _typed_input_rejected(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if isinstance(job_id, str):
            self._typed_jobs.pop(job_id, None)
        await self.push_error(
            error_msg=f"Voicechat typed input rejected: {event.get('reason') or 'unknown'}"
        )

    async def _typed_input_finished(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or job_id not in self._typed_jobs:
            raise events.ProtocolError("input_text.injection_finished references an unknown job")
        disposition = event.get("disposition")
        self._typed_jobs.pop(job_id, None)
        if disposition == "error":
            await self.push_error(error_msg=f"Voicechat typed input failed: {job_id}")

    async def _speech_started(self):
        self._clear_playout_buffer()
        if self._tts_open:
            await self.push_frame(TTSStoppedFrame())
        self._response_open = False
        self._tts_open = False
        self._response_id = None
        await self.stop_all_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        await self.broadcast_interruption()

    async def _reset_response_after_interruption(self):
        await self.stop_all_metrics()
        self._clear_playout_buffer()
        if self._tts_open:
            await self.push_frame(TTSStoppedFrame())
        self._response_open = False
        self._tts_open = False
        self._response_id = None

    async def _response_created(self, event: dict[str, Any]):
        response_id = event.get("response_id")
        if not isinstance(response_id, str):
            raise events.ProtocolError("response.created requires response_id")
        if self._response_open and self._response_id != response_id:
            raise events.ProtocolError("overlapping Voicechat responses")
        if not self._response_open:
            self._response_open = True
            self._response_id = response_id
            self._tts_open = False
            self._clear_playout_buffer()
            await self.push_frame(LLMFullResponseStartFrame())

    def _is_current_response(self, event: dict[str, Any]) -> bool:
        response_id = event.get("response_id")
        return isinstance(response_id, str) and response_id == self._response_id

    async def _text_delta(self, event: dict[str, Any]):
        if not self._is_current_response(event):
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        llm_text = LLMTextFrame(delta)
        llm_text.append_to_context = False
        await self.push_frame(llm_text)
        tts_text = TTSTextFrame(delta, aggregated_by=AggregationType.SENTENCE)
        tts_text.includes_inter_frame_spaces = True
        await self.push_frame(tts_text)

    async def _audio_delta(self, event: dict[str, Any]):
        if not self._is_current_response(event):
            return
        audio = events.decode_audio_delta(event)
        if not audio:
            return
        await self.stop_ttfb_metrics()
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=events.OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        if self._playout_started or self._prebuffer_ms == 0:
            await self._start_playout()
            await self.push_frame(frame)
            return

        self._playout_buffer.append(frame)
        self._playout_buffer_ms += (
            len(frame.audio) * 1000.0 / (2 * frame.sample_rate * frame.num_channels)
        )
        if self._playout_buffer_ms >= self._prebuffer_ms:
            await self._release_playout_buffer()

    async def _start_playout(self) -> None:
        if self._playout_started:
            return
        self._playout_started = True
        self._tts_open = True
        await self.push_frame(TTSStartedFrame())

    async def _release_playout_buffer(self) -> None:
        if not self._playout_buffer:
            return
        buffered, self._playout_buffer = self._playout_buffer, []
        self._playout_buffer_ms = 0.0
        await self._start_playout()
        for frame in buffered:
            await self.push_frame(frame)

    def _clear_playout_buffer(self) -> None:
        self._playout_buffer.clear()
        self._playout_buffer_ms = 0.0
        self._playout_started = False

    async def _response_done(self, event: dict[str, Any]):
        if not self._is_current_response(event):
            return
        await self._release_playout_buffer()
        if self._tts_open:
            await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()
        self._response_open = False
        self._tts_open = False
        self._response_id = None
        self._clear_playout_buffer()

    async def _function_call(self, event: dict[str, Any]):
        if not self._is_current_response(event):
            return
        if self._context is None:
            await self.push_error(error_msg="Voicechat function call arrived before context")
            return
        call_id = event.get("call_id")
        name = event.get("name")
        arguments = event.get("arguments")
        if not all(isinstance(value, str) for value in (call_id, name, arguments)):
            raise events.ProtocolError("malformed Voicechat function call")
        if call_id in self._seen_function_calls:
            logger.debug(f"{self}: ignoring duplicate function call event {call_id}")
            return
        self._seen_function_calls.add(call_id)
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            await self.push_error(error_msg=f"Voicechat function arguments are invalid JSON: {exc}")
            return
        if not isinstance(parsed_arguments, dict):
            await self.push_error(error_msg="Voicechat function arguments must be an object")
            return
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=name,
                    tool_call_id=call_id,
                    arguments=parsed_arguments,
                    context=self._context,
                )
            ]
        )

    async def _process_completed_function_calls(self, *, send_new_results: bool):
        if self._context is None:
            return
        for message in self._context.get_messages():
            if isinstance(message, LLMSpecificMessage):
                continue
            async_payload = async_tool_messages.parse_message(message)
            if async_payload is not None:
                if async_payload.tool_call_id in self._completed_tool_calls:
                    continue
                if async_payload.kind == "intermediate":
                    await self.push_error(
                        error_msg="Voicechat does not support intermediate async tool results"
                    )
                    continue
                if async_payload.kind != "final":
                    continue
                if send_new_results:
                    await self._send_tool_result(async_payload.tool_call_id, async_payload.result)
                self._completed_tool_calls.add(async_payload.tool_call_id)
                continue
            if not isinstance(message, dict) or message.get("content") == "IN_PROGRESS":
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id in self._completed_tool_calls:
                continue
            if send_new_results:
                if call_id not in self._seen_function_calls:
                    await self.push_error(
                        error_msg=f"Ignoring result for unknown Voicechat call_id={call_id}"
                    )
                else:
                    await self._send_tool_result(call_id, message.get("content"))
            self._completed_tool_calls.add(call_id)

    async def _send_tool_result(self, call_id: str, result: Any):
        output = result if isinstance(result, str) else json.dumps(result)
        await self._send(events.function_call_output(call_id, output))
