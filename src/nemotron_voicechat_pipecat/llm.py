"""Pipecat realtime service for NVIDIA Nemotron Voicechat.

The service is intentionally a strict adapter for ``voicechat.realtime`` v3:
PCM16 audio, server-owned RNNT turns, native assistant audio/text, and native
function calls.  A disconnected model session is never transparently resumed.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
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
from pipecat.utils.time import time_now_iso8601
from websockets.asyncio.client import connect as websocket_connect

from nemotron_voicechat_runtime.protocol import (
    FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS,
    FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
)

from . import events
from .adapter import NemotronVoicechatLLMAdapter
from .text_input import VoicechatTypedInputFrame
from .tool_results import VoicechatLLMContext

PLAYOUT_TRACE_ENV = "NEMOTRON_VOICECHAT_PLAYOUT_TRACE"
PLAYOUT_TRACE_SCHEMA = "nemotron_voicechat.playout.v1"


class _PlayoutTraceWriter:
    """Bound trace memory and isolate every writer failure from media."""

    _STOP = object()
    _QUARANTINE_DIRECTORY = ".voicechat-playout-trace-quarantine"

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        queue_records: int = 256,
        batch_records: int = 32,
        stream: Any | None = None,
    ):
        if queue_records <= 0 or batch_records <= 0:
            raise ValueError("playout trace queue and batch sizes must be positive")
        trace_path = Path(path).expanduser()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_path = trace_path
        self._staging_path: Path | None = None
        self._staging_lease_fd: int | None = None
        if stream is None:
            self._stream = self._open_staging_stream(trace_path)
        else:
            self._stream = stream
        self._clock = clock
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=queue_records
        )
        self._batch_records = batch_records
        self._closed = False
        self._error: BaseException | None = None
        self._invalid_reason: str | None = None
        self._dropped_records = 0
        self._written_records = 0
        self._published = False
        self._thread = threading.Thread(
            target=self._write_loop,
            name="voicechat-playout-trace",
            daemon=True,
        )
        self._thread.start()

    def record(
        self,
        event_type: str,
        *,
        timestamp_field: str | None = None,
        timestamp: float | None = None,
        **fields: Any,
    ) -> bool:
        if self._closed:
            self._invalidate("record-after-close")
            return False
        if self._error is not None:
            self._dropped_records += 1
            return False
        record = {
            "trace_schema": PLAYOUT_TRACE_SCHEMA,
            "type": event_type,
            **fields,
        }
        if timestamp_field is not None:
            record[timestamp_field] = self._clock() if timestamp is None else timestamp
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_records += 1
            self._invalidate("queue-overflow")
            return False
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "valid": self._invalid_reason is None and self._error is None,
            "invalid_reason": self._invalid_reason,
            "error": repr(self._error) if self._error is not None else None,
            "dropped_records": self._dropped_records,
            "written_records": self._written_records,
            "queued_records": self._queue.qsize(),
            "closed": self._closed,
            "writer_alive": self._thread.is_alive(),
            "artifact_published": self._published,
        }

    def close(self, *, timeout: float = 2.0) -> dict[str, Any]:
        if self._closed:
            return self.status()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("playout trace close timeout must be finite and positive")
        self._closed = True
        deadline = time.monotonic() + timeout
        terminal = {
            "trace_schema": PLAYOUT_TRACE_SCHEMA,
            "type": "voicechat.playout.trace_status",
            "trace_closed_monotonic_s": self._clock(),
            "valid": True,
            "invalid_reason": None,
            "error": None,
            "dropped_records": 0,
        }
        if not self._put_control(terminal, deadline):
            self._invalidate("close-queue-timeout")
        if not self._put_control(self._STOP, deadline):
            self._invalidate("close-queue-timeout")
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            self._invalidate("close-timeout")
        elif self._invalid_reason is None and self._error is None:
            if self._staging_path is not None:
                try:
                    self._publish_staging()
                except BaseException as exc:  # noqa: BLE001 - evidence failure is status only
                    self._invalidate("publish-error", exc)
                else:
                    self._published = True
            else:
                self._published = True
        self._cleanup_staging()
        return self.status()

    def _open_staging_stream(self, trace_path: Path):
        lock_path = trace_path.with_name(f".{trace_path.name}.pending.lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._quarantine_orphaned_staging(trace_path)
            if trace_path.exists():
                raise FileExistsError(
                    f"playout trace artifact already exists; refusing stale evidence: {trace_path}"
                )
            staging_path = trace_path.with_name(
                f".{trace_path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending"
            )
            lease_fd = os.open(staging_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream_fd = os.dup(lease_fd)
            except BaseException:
                os.close(lease_fd)
                staging_path.unlink(missing_ok=True)
                raise
            self._staging_path = staging_path
            self._staging_lease_fd = lease_fd
            return os.fdopen(stream_fd, "w", encoding="utf-8")
        finally:
            os.close(lock_fd)

    def _quarantine_orphaned_staging(self, trace_path: Path) -> None:
        pattern = f".{trace_path.name}.*.pending"
        for pending_path in trace_path.parent.glob(pattern):
            try:
                pending_fd = os.open(pending_path, os.O_RDWR)
            except FileNotFoundError:
                continue
            try:
                try:
                    fcntl.flock(pending_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                quarantine = trace_path.parent / self._QUARANTINE_DIRECTORY
                quarantine.mkdir(mode=0o700, exist_ok=True)
                orphan_path = quarantine / f"{pending_path.name}.orphan"
                pending_path.rename(orphan_path)
                self._fsync_directory(trace_path.parent)
                self._fsync_directory(quarantine)
            finally:
                os.close(pending_fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _publish_staging(self) -> None:
        assert self._staging_path is not None
        os.link(self._staging_path, self._artifact_path)
        try:
            self._fsync_directory(self._artifact_path.parent)
            self._staging_path.unlink()
            self._fsync_directory(self._staging_path.parent)
        except BaseException:
            self._artifact_path.unlink(missing_ok=True)
            self._fsync_directory(self._artifact_path.parent)
            raise

    def _cleanup_staging(self) -> None:
        if self._staging_path is not None and self._staging_path.exists():
            try:
                self._staging_path.unlink()
                self._fsync_directory(self._staging_path.parent)
            except BaseException as exc:  # noqa: BLE001 - evidence failure is status only
                self._invalidate("staging-cleanup-error", exc)
        if self._staging_lease_fd is not None:
            os.close(self._staging_lease_fd)
            self._staging_lease_fd = None

    def _put_control(self, item: dict[str, Any] | object, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            self._queue.put(item, timeout=remaining)
        except queue.Full:
            return False
        return True

    def _invalidate(self, reason: str, error: BaseException | None = None) -> None:
        if self._invalid_reason is None:
            self._invalid_reason = reason
        if error is not None and self._error is None:
            self._error = error

    def _write_loop(self) -> None:
        try:
            stop = False
            while not stop:
                item = self._queue.get()
                batch: list[dict[str, Any]] = []
                if item is self._STOP:
                    break
                batch.append(item)
                while len(batch) < self._batch_records:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is self._STOP:
                        stop = True
                        break
                    batch.append(item)
                payload = "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in batch
                )
                self._stream.write(payload)
                self._stream.flush()
                self._written_records += len(batch)
            if self._staging_path is not None:
                os.fsync(self._stream.fileno())
        except BaseException as exc:  # noqa: BLE001 - evidence failure is status only
            self._invalidate("writer-error", exc)
        finally:
            try:
                self._stream.close()
            except BaseException as exc:  # noqa: BLE001 - evidence failure is status only
                self._invalidate("close-error", exc)


class _SentenceBuffer:
    """Collect streaming text until sentence punctuation has whitespace lookahead."""

    _BOUNDARY = re.compile(r"[.!?][\"'\)\]\}\u00bb\u2019\u201d]*\s+")

    def __init__(self) -> None:
        self._text = ""

    def add(self, delta: str) -> list[str]:
        self._text += delta
        sentences: list[str] = []
        while match := self._BOUNDARY.search(self._text):
            sentence = self._text[: match.end()].strip()
            self._text = self._text[match.end() :]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        remainder = self._text.strip()
        self._text = ""
        return remainder or None

    def reset(self) -> None:
        self._text = ""


@dataclass
class _HeldInput:
    payload: bytes | dict[str, Any]
    audio_bytes: int = 0
    turn_id: int | None = None
    barrier: str | None = None


@dataclass
class NemotronVoicechatLLMSettings(LLMSettings):
    """Settings fixed during the pre-audio Voicechat handshake."""

    tools: ToolsSchema | list[dict[str, Any]] | None = field(default_factory=list)


class NemotronVoicechatLLMService(LLMService[NemotronVoicechatLLMAdapter]):
    """Continuous speech-to-speech Pipecat service for Voicechat v3."""

    Settings = NemotronVoicechatLLMSettings
    adapter_class = NemotronVoicechatLLMAdapter

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls is NemotronVoicechatLLMService and os.environ.get(PLAYOUT_TRACE_ENV, "").strip():
            return super().__new__(_TracedNemotronVoicechatLLMService)
        return super().__new__(cls)

    def __copy__(self):
        raise TypeError("NemotronVoicechatLLMService instances cannot be shallow-copied")

    def __deepcopy__(self, _memo: dict[int, object]):
        raise TypeError("NemotronVoicechatLLMService instances cannot be deep-copied")

    def __reduce__(self):
        raise TypeError("NemotronVoicechatLLMService instances cannot be pickled")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("NemotronVoicechatLLMService instances cannot be pickled")

    def __init__(
        self,
        *,
        base_url: str = "ws://127.0.0.1:8786/v1/realtime",
        system_instruction: str | None = None,
        tools: ToolsSchema | list[dict[str, Any]] | None = None,
        settings: NemotronVoicechatLLMSettings | None = None,
        connect_timeout_secs: float = 15.0,
        handshake_timeout_secs: float = 15.0,
        model_ready_timeout_secs: float = 45.0,
        graceful_close_timeout_secs: float = 8.0,
        liveness_timeout_secs: float = 45.0,
        audio_queue_max_chunks: int = 128,
        prebuffer_ms: int = 160,
        audio_queue_max_bytes: int = 256 * 1024,
        held_unassigned_preroll_seconds: float = 0.4,
        turn_barrier_ack_timeout_secs: float = 15.0,
        function_output_ack_timeout_secs: float = FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS,
        **kwargs,
    ):
        function_call_timeout_secs = kwargs.pop("function_call_timeout_secs", 20.0)
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
            function_call_timeout_secs=function_call_timeout_secs,
            **kwargs,
        )

        self.base_url = base_url
        self._connect_timeout_secs = connect_timeout_secs
        self._handshake_timeout_secs = handshake_timeout_secs
        self._model_ready_timeout_secs = model_ready_timeout_secs
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
        self._live_input_ready_event = asyncio.Event()
        self._session_update_lock = asyncio.Lock()
        self._context_ready = asyncio.Event()
        self._session_closed = asyncio.Event()
        self._disconnecting = False
        self._last_server_event_at = 0.0

        self._input_resampler = create_stream_resampler()
        self._audio_queue: asyncio.Queue[bytes | dict[str, Any] | None] = asyncio.Queue(
            maxsize=audio_queue_max_chunks
        )
        self._audio_queue_max_bytes = audio_queue_max_bytes
        self._audio_queue_bytes = 0
        self._audio_capacity_available = asyncio.Event()
        self._audio_capacity_available.set()
        self._audio_sent = False
        self._input_failed = False
        self._context: LLMContext | None = None
        self._completed_tool_calls: set[str] = set()
        self._seen_function_calls: set[str] = set()
        self._typed_jobs: dict[str, str] = {}
        self._typed_context_commits: set[str] = set()
        self._pending_typed_requests: list[VoicechatTypedInputFrame] = []
        self._pending_typed_flush_lock = asyncio.Lock()

        self._response_id: str | None = None
        self._response_open = False
        self._tts_open = False
        self._sentence_buffer = _SentenceBuffer()
        self._user_id = ""
        self._session_id: str | None = None
        self._function_output_model_output_advertised = False
        self._function_output_model_output_negotiated = False
        self._client_turn_sequence = 0
        self._active_client_turn_id: int | None = None
        self._pre_ready_dropped_turn_id: int | None = None
        self._pre_ready_audio_chunks_dropped = 0
        self._pre_ready_audio_bytes_dropped = 0
        self._pre_ready_turns_dropped = 0
        self._pre_ready_typed_requests_dropped = 0
        self._pre_ready_drop_summary_emitted = False
        self._turn_barrier_event_ids: dict[tuple[str, int], str] = {}
        self._turn_barrier_timeout_secs = turn_barrier_ack_timeout_secs
        self._turn_barrier_timeout_tasks: dict[tuple[str, int], asyncio.Task] = {}
        self._response_cancelling = False
        self._held_input: list[_HeldInput] = []
        self._held_input_bytes = 0
        self._held_input_max_bytes = 1024 * 1024
        if (
            isinstance(held_unassigned_preroll_seconds, bool)
            or not isinstance(held_unassigned_preroll_seconds, Real)
            or not math.isfinite(float(held_unassigned_preroll_seconds))
            or not 0 < float(held_unassigned_preroll_seconds) <= 1.0
        ):
            raise ValueError(
                "held unassigned microphone pre-roll must be a finite duration "
                "greater than zero and at most one second"
            )
        self._held_unassigned_preroll_max_bytes = round(
            events.INPUT_SAMPLE_RATE * 2 * held_unassigned_preroll_seconds
        )
        if self._held_unassigned_preroll_max_bytes < 2:
            raise ValueError("held unassigned microphone pre-roll must be positive")
        # PCM16 trimming must never split a sample.
        self._held_unassigned_preroll_max_bytes -= (
            self._held_unassigned_preroll_max_bytes % 2
        )
        self._held_unassigned_evicted_chunks = 0
        self._held_unassigned_evicted_bytes = 0
        self._held_unassigned_bound_chunks = 0
        self._held_unassigned_bound_bytes = 0
        self._last_held_input_release: dict[str, int] | None = None
        self._dropped_held_turns: set[int] = set()
        self._function_epoch_active = False
        self._function_epoch_call_id: str | None = None
        self._function_output_ack_timeout_secs = function_output_ack_timeout_secs
        self._function_output_ack_timeout_task: asyncio.Task | None = None
        self._response_start_timeout_secs = 15.0
        self._response_start_timeout_tasks: dict[str, asyncio.Task] = {}

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Keep realtime aggregation while application turn strategies win."""
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            user_turn_strategies=None,
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
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._queue_turn_start()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._queue_turn_commit()
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
            await self._update_session_and_wait()
            self._audio_send_task = self.create_task(self._audio_send_loop())
            await self._activate_live_input_if_ready()
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
        self._live_input_ready_event.clear()
        self._pre_ready_typed_requests_dropped += len(self._pending_typed_requests)
        self._pending_typed_requests.clear()
        self._emit_pre_ready_drop_summary(reason="disconnect")
        self._reset_pre_ready_drop_state()
        for task in self._turn_barrier_timeout_tasks.values():
            if task is not asyncio.current_task():
                task.cancel()
        self._turn_barrier_timeout_tasks.clear()
        for task in self._response_start_timeout_tasks.values():
            if task is not asyncio.current_task():
                task.cancel()
        self._response_start_timeout_tasks.clear()
        if (
            self._function_output_ack_timeout_task is not None
            and self._function_output_ack_timeout_task is not asyncio.current_task()
        ):
            self._function_output_ack_timeout_task.cancel()
        self._function_output_ack_timeout_task = None
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
        self._session_ready.clear()
        await self._send(
            events.session_update(
                instructions=instruction,
                tools=tools,
                function_output_model_output=(
                    FUNCTION_OUTPUT_MODEL_OUTPUT_MODE
                    if self._function_output_model_output_advertised
                    else None
                ),
            )
        )

    async def _wait_for_model_ready(self) -> None:
        await asyncio.wait_for(
            self._session_ready.wait(), timeout=self._model_ready_timeout_secs
        )

    async def wait_for_live_input_ready(self) -> None:
        """Wait until the model handshake, context, and send loop are all ready."""
        await asyncio.wait_for(
            self._live_input_ready_event.wait(), timeout=self._model_ready_timeout_secs
        )

    def live_input_readiness_status(self) -> dict[str, bool]:
        """Return bounded readiness evidence safe for client-facing diagnostics."""
        return {
            "session_ready": self._session_ready.is_set(),
            "context_ready": self._context_ready.is_set(),
            "audio_send_task": self._audio_send_task is not None,
        }

    async def abort_live_input_startup(self) -> None:
        """Close a model session whose browser readiness gate failed."""
        await self._disconnect()

    async def _activate_live_input_if_ready(self) -> None:
        if not self._live_input_ready():
            return
        self._live_input_ready_event.set()
        self._emit_pre_ready_drop_summary(reason="model_ready")
        await self._flush_pending_typed_inputs()

    async def _update_session_and_wait(self) -> None:
        async with self._session_update_lock:
            await self._send_session_update()
            await self._wait_for_model_ready()

    async def _request_typed_input(self, frame: VoicechatTypedInputFrame) -> None:
        if (
            frame.job_id in self._typed_jobs
            or frame.job_id in self._typed_context_commits
            or any(item.job_id == frame.job_id for item in self._pending_typed_requests)
        ):
            await self.push_error(error_msg=f"Duplicate Voicechat typed job ID: {frame.job_id}")
            return
        if self._input_failed:
            return
        if (
            not self._live_input_ready()
            or self._active_client_turn_id is not None
            or self._pre_ready_dropped_turn_id is not None
            or self._function_epoch_active
        ):
            self._pending_typed_requests.append(frame)
            return
        await self._admit_typed_input(frame)

    async def _admit_typed_input(self, frame: VoicechatTypedInputFrame) -> None:
        self._typed_jobs[frame.job_id] = frame.text
        admitted = await self._enqueue_live_input(
            events.typed_input_request(text=frame.text, job_id=frame.job_id)
        )
        if not admitted:
            self._typed_jobs.pop(frame.job_id, None)

    async def _flush_pending_typed_inputs(self) -> None:
        async with self._pending_typed_flush_lock:
            while self._pending_typed_requests:
                if (
                    self._input_failed
                    or not self._live_input_ready()
                    or self._active_client_turn_id is not None
                    or self._pre_ready_dropped_turn_id is not None
                    or self._function_epoch_active
                ):
                    return
                frame = self._pending_typed_requests.pop(0)
                await self._admit_typed_input(frame)

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
            if settings_changed and (
                self._active_client_turn_id is not None
                or self._pre_ready_dropped_turn_id is not None
            ):
                await self.push_error(
                    error_msg=(
                        "Initial Voicechat context changed instructions or tools after "
                        "a microphone turn began"
                    ),
                    fatal=True,
                )
                return
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
                await self._update_session_and_wait()
            self._context_ready.set()
            await self._activate_live_input_if_ready()
        else:
            await self._process_completed_function_calls(send_new_results=True)

    async def _handle_tools(self, tools):
        if (
            self._audio_sent
            or self._active_client_turn_id is not None
            or self._pre_ready_dropped_turn_id is not None
        ):
            await self.push_error(
                error_msg="Voicechat tools are immutable after a microphone turn begins"
            )
            return
        self._sync_registered_tool_handlers(tools)
        converted = self.get_llm_adapter().from_standard_tools(tools)
        self._settings.tools = converted if isinstance(converted, list) else []
        if self._session_ready.is_set():
            await self._update_session_and_wait()

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
        if not self._live_input_ready() or self._pre_ready_dropped_turn_id is not None:
            self._pre_ready_audio_chunks_dropped += 1
            self._pre_ready_audio_bytes_dropped += len(frame.audio)
            return
        self._emit_pre_ready_drop_summary(reason="model_ready")
        mono = self._downmix(frame.audio, frame.num_channels)
        pcm = await self._input_resampler.resample(
            mono, frame.sample_rate, events.INPUT_SAMPLE_RATE
        )
        if not pcm:
            return
        if self._input_epoch_active():
            await self._hold_audio(pcm)
            return
        await self._enqueue_live_input(pcm, audio_bytes=len(pcm))

    async def _enqueue_live_input(
        self, payload: bytes | dict[str, Any], *, audio_bytes: int = 0
    ) -> bool:
        if audio_bytes > self._audio_queue_max_bytes:
            await self._fail_input("Voicechat input audio packet exceeds byte budget")
            return False
        deadline = time.monotonic() + 0.5
        while audio_bytes and self._audio_queue_bytes + audio_bytes > self._audio_queue_max_bytes:
            self._audio_capacity_available.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._fail_input("Voicechat input audio byte budget exceeded")
                return False
            try:
                await asyncio.wait_for(self._audio_capacity_available.wait(), timeout=remaining)
            except TimeoutError:
                await self._fail_input("Voicechat input audio byte budget exceeded")
                return False
        try:
            await asyncio.wait_for(self._audio_queue.put(payload), timeout=0.5)
        except TimeoutError:
            kind = "audio " if audio_bytes else "control "
            await self._fail_input(f"Voicechat input {kind}queue overflow")
            return False
        self._audio_queue_bytes += audio_bytes
        return True

    async def _fail_input(self, message: str) -> None:
        self._input_failed = True
        await self.push_error(error_msg=message, fatal=True)
        await self._disconnect()

    async def _audio_send_loop(self):
        try:
            await self._context_ready.wait()
            while True:
                item = await self._audio_queue.get()
                if item is None:
                    return
                if isinstance(item, bytes):
                    self._audio_queue_bytes -= len(item)
                    self._audio_capacity_available.set()
                    await self._send(events.audio_append(item))
                    self._audio_sent = True
                else:
                    if item.get("type") == "input_audio_buffer.commit":
                        diagnostics = item.get("diagnostics")
                        if isinstance(diagnostics, dict):
                            diagnostics["client_send_monotonic_s"] = time.monotonic()
                            diagnostics["queue_chunks_after_get"] = self._audio_queue.qsize()
                            diagnostics["queue_audio_bytes_after_get"] = self._audio_queue_bytes
                    await self._send(item)
                    self._arm_turn_barrier_timeout(item)
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

    async def _queue_control(self, event: dict[str, Any]) -> None:
        if self._input_epoch_active():
            turn_id = event.get("client_turn_id")
            barrier = "start" if event["type"].endswith("turn_start") else "commit"
            if barrier == "start":
                bound_chunks = 0
                bound_bytes = 0
                for item in reversed(self._held_input):
                    if item.barrier == "commit":
                        break
                    if item.turn_id is None:
                        if item.barrier is None:
                            bound_chunks += 1
                            bound_bytes += item.audio_bytes
                        item.turn_id = turn_id
                self._held_unassigned_bound_chunks += bound_chunks
                self._held_unassigned_bound_bytes += bound_bytes
            if turn_id not in self._dropped_held_turns:
                self._held_input.append(_HeldInput(event, turn_id=turn_id, barrier=barrier))
            if barrier == "commit":
                if turn_id in self._dropped_held_turns:
                    self._turn_barrier_event_ids.pop(("start", turn_id), None)
                    self._turn_barrier_event_ids.pop(("commit", turn_id), None)
                    await self.stop_all_metrics()
                self._dropped_held_turns.discard(turn_id)
            return
        await self._enqueue_live_input(event)

    def _arm_turn_barrier_timeout(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "input_audio_buffer.turn_start":
            barrier = "start"
        elif event_type == "input_audio_buffer.commit":
            barrier = "commit"
        else:
            return
        turn_id = event.get("client_turn_id")
        if not isinstance(turn_id, int):
            return
        key = (barrier, turn_id)
        old_task = self._turn_barrier_timeout_tasks.pop(key, None)
        if old_task is not None:
            old_task.cancel()
        self._turn_barrier_timeout_tasks[key] = asyncio.create_task(
            self._turn_barrier_ack_timeout(key),
            name=f"voicechat-{barrier}-ack-{turn_id}",
        )

    async def _turn_barrier_ack_timeout(self, key: tuple[str, int]) -> None:
        try:
            await asyncio.sleep(self._turn_barrier_timeout_secs)
            if key not in self._turn_barrier_event_ids:
                return
            await self._fail_input(
                f"Voicechat client turn {key[0]} acknowledgement timed out: {key[1]}"
            )
        finally:
            self._turn_barrier_timeout_tasks.pop(key, None)

    def _input_epoch_active(self) -> bool:
        return bool(self._typed_jobs) or self._function_epoch_active

    async def _hold_audio(self, pcm: bytes) -> None:
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("held microphone audio must be non-empty PCM16 bytes")
        turn_id = self._active_client_turn_id
        if turn_id in self._dropped_held_turns:
            return
        if turn_id is None:
            # While a typed/function epoch owns the model, continuous microphone
            # PCM with no VAD-owned turn is only onset pre-roll. Keep a short
            # rolling suffix on every append; replaying an entire long epoch
            # would put later turn barriers seconds behind audio the client has
            # already sent. Once VAD starts a turn, _queue_control() binds this
            # suffix to that turn and subsequent speech remains lossless.
            if len(pcm) > self._held_unassigned_preroll_max_bytes:
                discarded = len(pcm) - self._held_unassigned_preroll_max_bytes
                pcm = pcm[-self._held_unassigned_preroll_max_bytes :]
                self._held_unassigned_evicted_chunks += 1
                self._held_unassigned_evicted_bytes += discarded
            unassigned_bytes = sum(
                item.audio_bytes
                for item in self._held_input
                if item.turn_id is None and item.barrier is None
            )
            while (
                unassigned_bytes + len(pcm)
                > self._held_unassigned_preroll_max_bytes
            ):
                index = next(
                    (
                        index
                        for index, item in enumerate(self._held_input)
                        if item.turn_id is None and item.barrier is None
                    ),
                    None,
                )
                if index is None:
                    break
                item = self._held_input.pop(index)
                self._held_input_bytes -= item.audio_bytes
                unassigned_bytes -= item.audio_bytes
                self._held_unassigned_evicted_chunks += 1
                self._held_unassigned_evicted_bytes += item.audio_bytes
        if self._held_input_bytes + len(pcm) > self._held_input_max_bytes:
            if turn_id is None:
                # Assigned/committed turns are immutable. If they consume the
                # whole epoch budget, dropping optional unassigned pre-roll is
                # safer than corrupting an already accepted turn.
                self._held_unassigned_evicted_chunks += 1
                self._held_unassigned_evicted_bytes += len(pcm)
                return
            retained: list[_HeldInput] = []
            for item in self._held_input:
                if item.turn_id == turn_id:
                    self._held_input_bytes -= item.audio_bytes
                else:
                    retained.append(item)
            self._held_input = retained
            self._dropped_held_turns.add(turn_id)
            await self.push_error(
                error_msg=(
                    f"voicechat.input_turn_dropped reason=arbiter_capacity client_turn_id={turn_id}"
                )
            )
            return
        self._held_input.append(_HeldInput(pcm, audio_bytes=len(pcm), turn_id=turn_id))
        self._held_input_bytes += len(pcm)

    async def _release_held_input(self) -> None:
        if self._input_epoch_active():
            return
        items, self._held_input = self._held_input, []
        self._held_input_bytes = 0
        current_unassigned = [
            item for item in items if item.turn_id is None and item.barrier is None
        ]
        summary = {
            "preroll_cap_bytes": self._held_unassigned_preroll_max_bytes,
            "retained_unassigned_chunks": self._held_unassigned_bound_chunks
            + len(current_unassigned),
            "retained_unassigned_bytes": self._held_unassigned_bound_bytes
            + sum(item.audio_bytes for item in current_unassigned),
            "evicted_unassigned_chunks": self._held_unassigned_evicted_chunks,
            "evicted_unassigned_bytes": self._held_unassigned_evicted_bytes,
            "released_chunks": len(items),
            "released_audio_bytes": sum(item.audio_bytes for item in items),
        }
        self._last_held_input_release = summary
        logger.info(
            "voicechat.held_input_release "
            + " ".join(f"{key}={value}" for key, value in summary.items())
        )
        self._held_unassigned_evicted_chunks = 0
        self._held_unassigned_evicted_bytes = 0
        self._held_unassigned_bound_chunks = 0
        self._held_unassigned_bound_bytes = 0
        for item in items:
            if not await self._enqueue_live_input(item.payload, audio_bytes=item.audio_bytes):
                return

    async def _queue_turn_start(self) -> None:
        if (
            self._active_client_turn_id is not None
            or self._pre_ready_dropped_turn_id is not None
        ):
            return
        self._client_turn_sequence += 1
        turn_id = self._client_turn_sequence
        if not self._live_input_ready():
            self._pre_ready_dropped_turn_id = turn_id
            self._pre_ready_turns_dropped += 1
            return
        self._emit_pre_ready_drop_summary(reason="model_ready")
        self._active_client_turn_id = turn_id
        event = events.audio_turn_start(turn_id)
        self._turn_barrier_event_ids[("start", turn_id)] = event["event_id"]
        await self._queue_control(event)

    async def _queue_turn_commit(self) -> None:
        if self._pre_ready_dropped_turn_id is not None:
            self._pre_ready_dropped_turn_id = None
            if self._live_input_ready():
                self._emit_pre_ready_drop_summary(reason="model_ready")
            await self._flush_pending_typed_inputs()
            return
        turn_id = self._active_client_turn_id
        if turn_id is None:
            await self.push_error(
                error_msg="Voicechat Smart Turn stop arrived without an open local turn",
                fatal=True,
            )
            await self._disconnect()
            return
        event = events.audio_commit(
            turn_id,
            diagnostics={
                "client_enqueue_monotonic_s": time.monotonic(),
                "queue_chunks_at_enqueue": self._audio_queue.qsize(),
                "queue_audio_bytes_at_enqueue": self._audio_queue_bytes,
            },
        )
        self._turn_barrier_event_ids[("commit", turn_id)] = event["event_id"]
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self._queue_control(event)
        self._active_client_turn_id = None
        await self._flush_pending_typed_inputs()

    def _live_input_ready(self) -> bool:
        return bool(
            self._session_ready.is_set()
            and self._context_ready.is_set()
            and self._audio_send_task is not None
        )

    def _emit_pre_ready_drop_summary(self, *, reason: str) -> None:
        if self._pre_ready_drop_summary_emitted or not (
            self._pre_ready_audio_chunks_dropped
            or self._pre_ready_turns_dropped
            or self._pre_ready_typed_requests_dropped
        ):
            return
        if self._pre_ready_dropped_turn_id is not None and reason != "disconnect":
            return
        logger.warning(
            "voicechat.pre_ready_input_dropped "
            f"reason={reason} chunks={self._pre_ready_audio_chunks_dropped} "
            f"bytes={self._pre_ready_audio_bytes_dropped} "
            f"turns={self._pre_ready_turns_dropped} "
            f"typed_requests={self._pre_ready_typed_requests_dropped}"
        )
        self._pre_ready_drop_summary_emitted = True

    def _reset_pre_ready_drop_state(self) -> None:
        self._pre_ready_dropped_turn_id = None
        self._pre_ready_audio_chunks_dropped = 0
        self._pre_ready_audio_bytes_dropped = 0
        self._pre_ready_turns_dropped = 0
        self._pre_ready_typed_requests_dropped = 0
        self._pre_ready_drop_summary_emitted = False

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
            capabilities = event.get("capabilities") or {}
            if capabilities.get("input_turn_detection") != "client_smart_turn_v1":
                raise events.ProtocolError("server does not support client Smart Turn v1")
            model_output_capability = capabilities.get("function_output_model_output")
            self._function_output_model_output_advertised = bool(
                isinstance(model_output_capability, dict)
                and model_output_capability.get("mode")
                == FUNCTION_OUTPUT_MODEL_OUTPUT_MODE
            )
            self._session_created.set()
        elif event_type == "session.updated":
            session = event.get("session") or {}
            if session.get("protocol_version") != events.PROTOCOL_VERSION:
                raise events.ProtocolError("session.updated did not confirm protocol v3")
            negotiated = (session.get("capabilities") or {}).get(
                "function_output_model_output"
            )
            if self._function_output_model_output_advertised:
                if negotiated != FUNCTION_OUTPUT_MODEL_OUTPUT_MODE:
                    raise events.ProtocolError(
                        "session.updated did not negotiate concise tool results"
                    )
                self._function_output_model_output_negotiated = True
            elif negotiated is not None:
                raise events.ProtocolError(
                    "session.updated negotiated an unadvertised concise tool-result capability"
                )
            self._session_ready.set()
        elif event_type == "input_text.accepted":
            await self._typed_input_accepted(event)
        elif event_type == "input_text.rejected":
            await self._typed_input_rejected(event)
        elif event_type == "input_text.injection_started":
            await self._typed_input_started(event)
        elif event_type == "input_text.injection_finished":
            await self._typed_input_finished(event)
        elif event_type == "input_audio_buffer.speech_started":
            if event.get("source") == "typed":
                await self._speech_started()
        elif event_type == "input_audio_buffer.speech_stopped":
            if event.get("source") == "typed":
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                await self.broadcast_frame(UserStoppedSpeakingFrame)
        elif event_type == "input_audio_buffer.turn_started":
            self._validate_turn_ack(event, "start")
        elif event_type == "input_audio_buffer.committed":
            self._validate_turn_ack(event, "commit")
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
                if call_id == self._function_epoch_call_id:
                    await self._finish_function_epoch(call_id)
            await self.push_error(
                error_msg=f"Voicechat function call {call_id} failed: {event.get('reason')}"
            )
        elif event_type == "conversation.item.function_call_output.applied":
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or call_id != self._function_epoch_call_id:
                raise events.ProtocolError("uncorrelated function output applied acknowledgement")
            await self._finish_function_epoch(call_id)
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
        # Admission is not model mutation. Context is committed only when the
        # server reports the first actual injection step.

    async def _typed_input_started(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or job_id not in self._typed_jobs:
            raise events.ProtocolError("input_text.injection_started references an unknown job")
        if job_id in self._typed_context_commits:
            raise events.ProtocolError("typed input injection started more than once")
        if self._context is None:
            raise events.ProtocolError("typed input started before context initialization")
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

    def _validate_turn_ack(self, event: dict[str, Any], barrier: str) -> None:
        turn_id = event.get("client_turn_id")
        expected = self._turn_barrier_event_ids.get((barrier, turn_id))
        if expected is None or event.get("client_event_id") != expected:
            raise events.ProtocolError(f"uncorrelated client turn {barrier} acknowledgement")
        self._turn_barrier_event_ids.pop((barrier, turn_id), None)
        task = self._turn_barrier_timeout_tasks.pop((barrier, turn_id), None)
        if task is not None:
            task.cancel()
        if barrier == "commit":
            server_turn_id = event.get("turn_id")
            if not isinstance(server_turn_id, str) or not server_turn_id:
                raise events.ProtocolError("commit acknowledgement requires server turn_id")
            self._response_start_timeout_tasks[server_turn_id] = asyncio.create_task(
                self._response_start_timeout(server_turn_id),
                name=f"voicechat-response-start-{server_turn_id}",
            )

    async def _response_start_timeout(self, server_turn_id: str) -> None:
        try:
            await asyncio.sleep(self._response_start_timeout_secs)
            if server_turn_id not in self._response_start_timeout_tasks:
                return
            await self.stop_all_metrics()
            await self._fail_input(
                f"Voicechat response did not start for committed turn: {server_turn_id}"
            )
        finally:
            self._response_start_timeout_tasks.pop(server_turn_id, None)

    async def _typed_input_rejected(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if isinstance(job_id, str):
            self._typed_jobs.pop(job_id, None)
        await self._release_held_input()
        await self.push_error(
            error_msg=f"Voicechat typed input rejected: {event.get('reason') or 'unknown'}"
        )

    async def _typed_input_finished(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or job_id not in self._typed_jobs:
            raise events.ProtocolError("input_text.injection_finished references an unknown job")
        disposition = event.get("disposition")
        self._typed_jobs.pop(job_id, None)
        await self._release_held_input()
        if disposition == "error":
            await self.push_error(error_msg=f"Voicechat typed input failed: {job_id}")

    async def _speech_started(self):
        await self._reset_response_after_interruption()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        await self.broadcast_interruption()

    async def _reset_response_after_interruption(self):
        self._sentence_buffer.reset()
        self._clear_playout_buffer()
        tts_was_open = self._tts_open
        self._tts_open = False
        self._response_cancelling = self._response_open
        await self.stop_all_metrics()
        if tts_was_open:
            await self.push_frame(TTSStoppedFrame())

    async def _response_created(self, event: dict[str, Any]):
        response_id = event.get("response_id")
        if not isinstance(response_id, str):
            raise events.ProtocolError("response.created requires response_id")
        server_turn_id = event.get("turn_id")
        if isinstance(server_turn_id, str):
            task = self._response_start_timeout_tasks.pop(server_turn_id, None)
            if task is not None:
                task.cancel()
        if self._response_open and self._response_id != response_id:
            raise events.ProtocolError("overlapping Voicechat responses")
        if not self._response_open:
            self._sentence_buffer.reset()
            self._clear_playout_buffer()
            self._response_open = True
            self._response_id = response_id
            self._tts_open = False
            self._response_cancelling = False
            await self.push_frame(LLMFullResponseStartFrame())

    def _is_current_response(self, event: dict[str, Any]) -> bool:
        response_id = event.get("response_id")
        return isinstance(response_id, str) and response_id == self._response_id

    async def _text_delta(self, event: dict[str, Any]):
        if not self._is_current_response(event) or self._response_cancelling:
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        sentences = self._sentence_buffer.add(delta)
        llm_text = LLMTextFrame(delta)
        llm_text.append_to_context = False
        await self.push_frame(llm_text)
        for sentence in sentences:
            await self.push_frame(TTSTextFrame(sentence, aggregated_by=AggregationType.SENTENCE))

    async def _audio_delta(self, event: dict[str, Any]):
        if not self._is_current_response(event) or self._response_cancelling:
            return
        audio = events.decode_audio_delta(event)
        if not audio:
            return
        await self.stop_ttfb_metrics()
        audio_frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=events.OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        await self.process_ttfa_metrics(audio_frame)
        if self._playout_started or self._prebuffer_ms == 0:
            await self._start_playout()
            await self.push_frame(audio_frame)
            return

        self._playout_buffer.append(audio_frame)
        self._playout_buffer_ms += (
            len(audio_frame.audio)
            * 1000.0
            / (2 * audio_frame.sample_rate * audio_frame.num_channels)
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
        unsuccessful = self._response_cancelling or event.get("status") not in {
            None,
            "completed",
        }
        if unsuccessful:
            self._sentence_buffer.reset()
            remainder = None
        else:
            remainder = self._sentence_buffer.flush()
        await self._release_playout_buffer()
        if remainder is not None:
            await self.push_frame(TTSTextFrame(remainder, aggregated_by=AggregationType.SENTENCE))
        if self._tts_open:
            await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()
        self._response_open = False
        self._tts_open = False
        self._response_id = None
        self._response_cancelling = False
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
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            await self.push_error(error_msg=f"Voicechat function arguments are invalid JSON: {exc}")
            return
        if not isinstance(parsed_arguments, dict):
            await self.push_error(error_msg="Voicechat function arguments must be an object")
            return
        if self._function_epoch_active:
            raise events.ProtocolError("overlapping Voicechat function calls")
        self._seen_function_calls.add(call_id)
        self._function_epoch_active = True
        self._function_epoch_call_id = call_id
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
                    await self._send_tool_result(
                        async_payload.tool_call_id,
                        async_payload.result,
                        model_output=self._model_output_for_call(async_payload.tool_call_id),
                    )
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
                    await self._send_tool_result(
                        call_id,
                        message.get("content"),
                        model_output=self._model_output_for_call(call_id),
                    )
            self._completed_tool_calls.add(call_id)

    def _model_output_for_call(self, call_id: str) -> str | None:
        if isinstance(self._context, VoicechatLLMContext):
            return self._context.voicechat_model_output(call_id)
        return None

    async def _send_tool_result(
        self, call_id: str, result: Any, *, model_output: str | None = None
    ):
        if call_id != self._function_epoch_call_id:
            raise events.ProtocolError("tool result does not match the active function epoch")
        output = result if isinstance(result, str) else json.dumps(result)
        await self._send(
            events.function_call_output(
                call_id,
                output,
                model_output=(
                    model_output
                    if self._function_output_model_output_negotiated
                    else None
                ),
            )
        )
        if isinstance(self._context, VoicechatLLMContext):
            self._context.discard_voicechat_model_output(call_id)
        if self._function_output_ack_timeout_task is not None:
            self._function_output_ack_timeout_task.cancel()
        self._function_output_ack_timeout_task = asyncio.create_task(
            self._function_output_ack_timeout(call_id),
            name=f"voicechat-function-output-ack-{call_id}",
        )

    async def _function_output_ack_timeout(self, call_id: str) -> None:
        try:
            await asyncio.sleep(self._function_output_ack_timeout_secs)
            if call_id != self._function_epoch_call_id:
                return
            await self._fail_input(
                f"Voicechat function output applied acknowledgement timed out: {call_id}"
            )
        finally:
            if self._function_output_ack_timeout_task is asyncio.current_task():
                self._function_output_ack_timeout_task = None

    async def _finish_function_epoch(self, call_id: str) -> None:
        if call_id != self._function_epoch_call_id:
            raise events.ProtocolError("function epoch terminal does not match active call")
        if self._function_output_ack_timeout_task is not None:
            self._function_output_ack_timeout_task.cancel()
            self._function_output_ack_timeout_task = None
        self._function_epoch_active = False
        self._function_epoch_call_id = None
        await self._flush_pending_typed_inputs()
        await self._release_held_input()


class _TracedNemotronVoicechatLLMService(NemotronVoicechatLLMService):
    """Qualification-only service variant with observational playout tracing."""

    def __init__(self, *args: Any, **kwargs: Any):
        path = os.environ.get(PLAYOUT_TRACE_ENV, "").strip()
        if not path:
            raise RuntimeError("traced Voicechat service requires a trace path")
        try:
            close_timeout = float(
                os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE_CLOSE_TIMEOUT", "2")
            )
        except ValueError as exc:
            raise ValueError(
                "NEMOTRON_VOICECHAT_PLAYOUT_TRACE_CLOSE_TIMEOUT must be finite and positive"
            ) from exc
        if not math.isfinite(close_timeout) or close_timeout <= 0:
            raise ValueError(
                "NEMOTRON_VOICECHAT_PLAYOUT_TRACE_CLOSE_TIMEOUT must be finite and positive"
            )
        super().__init__(*args, **kwargs)
        self._playout_trace = _PlayoutTraceWriter(
            path,
            queue_records=int(os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE_QUEUE", "256")),
            batch_records=int(os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE_BATCH", "32")),
        )
        self._playout_trace_close_timeout = close_timeout
        self._playout_trace_ordinals: dict[str, int] = {}
        self._playout_trace_frame_ordinals: dict[int, int] = {}
        self._playout_trace_interruption_observed = False
        self._playout_trace.record(
            "voicechat.playout.config",
            timestamp_field="client_configured_monotonic_s",
            configured_prebuffer_ms=self._prebuffer_ms,
            sample_rate_hz=events.OUTPUT_SAMPLE_RATE,
            channels=1,
        )

    async def cleanup(self):
        try:
            await super().cleanup()
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._playout_trace.close,
                        timeout=self._playout_trace_close_timeout,
                    ),
                    timeout=self._playout_trace_close_timeout + 0.25,
                )
            except TimeoutError:
                self._playout_trace._invalidate("async-cleanup-timeout")

    def live_input_readiness_status(self) -> dict[str, Any]:
        return {
            **super().live_input_readiness_status(),
            "playout_trace": self._playout_trace.status(),
        }

    async def _receive_loop(self):
        websocket = self._websocket
        try:
            async for message in websocket:
                received_monotonic_s = time.monotonic()
                self._last_server_event_at = received_monotonic_s
                event = events.parse_server_event(message)
                event["_voicechat_received_monotonic_s"] = received_monotonic_s
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

    def _receipt_time(self, event: dict[str, Any]) -> float:
        timestamp = event.get("_voicechat_received_monotonic_s")
        return float(timestamp) if isinstance(timestamp, Real) else time.monotonic()

    async def _handle_server_event(self, event: dict[str, Any]):
        event_type = event.get("type")
        received = self._receipt_time(event)
        response_id = event.get("response_id")
        if event_type == "input_audio_buffer.speech_started" and event.get(
            "source"
        ) == "typed":
            self._playout_trace_interruption_observed = self._response_open
            self._playout_trace.record(
                event_type,
                timestamp_field="client_received_monotonic_s",
                timestamp=received,
                source="typed",
                response_id=self._response_id,
            )
        accepted_done = event_type == "response.done" and self._is_current_response(event)
        if accepted_done:
            self._playout_trace.record(
                event_type,
                timestamp_field="client_received_monotonic_s",
                timestamp=received,
                response_id=response_id,
                status=event.get("status"),
            )
        await super()._handle_server_event(event)
        if event_type == "response.created" and isinstance(response_id, str):
            self._playout_trace_ordinals[response_id] = 0
            self._playout_trace.record(
                event_type,
                timestamp_field="client_received_monotonic_s",
                timestamp=received,
                response_id=response_id,
                turn_id=event.get("turn_id"),
            )
        elif accepted_done:
            if isinstance(response_id, str):
                self._playout_trace_ordinals.pop(response_id, None)
        elif event_type == "voicechat.metrics":
            self._playout_trace.record(
                event_type,
                timestamp_field="client_received_monotonic_s",
                timestamp=received,
                **{
                    name: value
                    for name, value in event.items()
                    if name not in {"type", "_voicechat_received_monotonic_s"}
                },
            )

    async def _audio_delta(self, event: dict[str, Any]):
        if not self._is_current_response(event) or self._response_cancelling:
            return
        audio = events.decode_audio_delta(event)
        if not audio:
            return
        response_id = self._response_id
        if not isinstance(response_id, str):
            raise events.ProtocolError("accepted audio delta lacks current response")
        ordinal = self._playout_trace_ordinals.get(response_id, 0) + 1
        self._playout_trace_ordinals[response_id] = ordinal
        self._playout_trace.record(
            "response.output_audio.delta",
            timestamp_field="client_received_monotonic_s",
            timestamp=self._receipt_time(event),
            response_id=response_id,
            ordinal=ordinal,
            sample_count=len(audio) // 2,
            encoding=event.get("encoding"),
            sample_rate=event.get("sample_rate"),
            sample_rate_hz=event.get("sample_rate"),
            channels=event.get("channels"),
            delta=event.get("delta"),
        )
        await self.stop_ttfb_metrics()
        audio_frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=events.OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        await self.process_ttfa_metrics(audio_frame)
        self._playout_trace_frame_ordinals[id(audio_frame)] = ordinal
        if self._playout_started or self._prebuffer_ms == 0:
            if not self._playout_started:
                self._trace_playout_release("threshold", [audio_frame])
            await self._start_playout()
            await self._push_playout_audio(audio_frame)
            return

        self._playout_buffer.append(audio_frame)
        self._playout_buffer_ms += (
            len(audio_frame.audio)
            * 1000.0
            / (2 * audio_frame.sample_rate * audio_frame.num_channels)
        )
        if self._playout_buffer_ms >= self._prebuffer_ms:
            await self._release_playout_buffer(reason="threshold")

    async def _release_playout_buffer(self, *, reason: str = "done") -> None:
        if not self._playout_buffer:
            return
        buffered, self._playout_buffer = self._playout_buffer, []
        self._playout_buffer_ms = 0.0
        self._trace_playout_release(reason, buffered)
        await self._start_playout()
        for frame in buffered:
            await self._push_playout_audio(frame)

    async def _push_playout_audio(self, frame: TTSAudioRawFrame) -> None:
        ordinal = self._playout_trace_frame_ordinals.pop(id(frame))
        self._playout_trace.record(
            "voicechat.playout.downstream_push",
            timestamp_field="client_downstream_push_monotonic_s",
            response_id=self._response_id,
            ordinal=ordinal,
            sample_count=len(frame.audio) // (2 * frame.num_channels),
            sample_rate_hz=frame.sample_rate,
            channels=frame.num_channels,
        )
        await self.push_frame(frame)

    def _trace_playout_release(
        self,
        reason: str,
        frames: list[TTSAudioRawFrame],
        *,
        frames_cleared: int = 0,
    ) -> None:
        ordinals = [self._playout_trace_frame_ordinals[id(frame)] for frame in frames]
        self._playout_trace.record(
            "voicechat.playout.release",
            timestamp_field="client_release_monotonic_s",
            response_id=self._response_id,
            reason=reason,
            ordinals=ordinals,
            frames_released=len(frames),
            frames_cleared=frames_cleared,
            sample_count=sum(
                len(frame.audio) // (2 * frame.num_channels) for frame in frames
            ),
        )

    def _clear_playout_buffer(self, *, reason: str | None = None) -> None:
        buffered = list(self._playout_buffer)
        if reason is not None:
            ordinals = [
                self._playout_trace_frame_ordinals[id(frame)] for frame in buffered
            ]
            self._playout_trace.record(
                "voicechat.playout.release",
                timestamp_field="client_release_monotonic_s",
                response_id=self._response_id,
                reason=reason,
                ordinals=ordinals,
                frames_released=0,
                frames_cleared=len(buffered),
                sample_count=0,
            )
        for frame in buffered:
            self._playout_trace_frame_ordinals.pop(id(frame), None)
        super()._clear_playout_buffer()

    async def _reset_response_after_interruption(self):
        self._sentence_buffer.reset()
        if self._response_open and not self._playout_trace_interruption_observed:
            self._playout_trace.record(
                "voicechat.playout.interruption",
                timestamp_field="client_interruption_monotonic_s",
                response_id=self._response_id,
            )
        self._clear_playout_buffer(
            reason="interruption-clear" if self._response_open else None
        )
        self._playout_trace_interruption_observed = False
        tts_was_open = self._tts_open
        self._tts_open = False
        self._response_cancelling = self._response_open
        await self.stop_all_metrics()
        if tts_was_open:
            await self.push_frame(TTSStoppedFrame())
