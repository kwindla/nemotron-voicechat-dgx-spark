from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from nemotron_voicechat_runtime.pocket_controller import PocketSynthesis, PocketSynthesisCancelled
from nemotron_voicechat_runtime.server import (
    FRAME_SAMPLES,
    INPUT_SAMPLE_RATE,
    REALTIME_RECEIVE_QUEUE_MESSAGES,
    StepResult,
    audio_stats,
    create_app,
)


class FakePocketWorker:
    def __init__(self) -> None:
        self.ready = True
        self.requests = []

    @property
    def health(self):
        return {"ready": self.ready, "backend": "fake-pocket"}

    async def start(self):
        self.ready = True

    async def stop(self):
        self.ready = False

    async def synthesize(self, text, *, job_id, cancel_event):
        self.requests.append((job_id, text))
        pcm = b"\x01\x10" * (INPUT_SAMPLE_RATE // 5)
        return PocketSynthesis(
            pcm16=pcm,
            sample_rate=INPUT_SAMPLE_RATE,
            generated_seconds=0.2,
            wall_seconds=0.01,
            time_to_first_audio_seconds=0.005,
        )


class ReplacingFakePocketWorker(FakePocketWorker):
    async def synthesize(self, text, *, job_id, cancel_event):
        self.requests.append((job_id, text))
        if job_id == "typed-old":
            await cancel_event.wait()
            raise PocketSynthesisCancelled
        return await super().synthesize(text, job_id=job_id, cancel_event=cancel_event)


class FakeEngine:
    def __init__(self) -> None:
        self.eartts = SimpleNamespace(_reset_on_bos_count=0)
        self.wrapper = SimpleNamespace(
            model=SimpleNamespace(tts_model=self.eartts),
            _external_user_eou_blank_fence_last=None,
        )
        self.pipeline = SimpleNamespace(
            checkpoint_provenance={"kind": "test"},
            s2s_model=self.wrapper,
        )
        self.frame_index = 0
        self.started = False
        self.system_prompt = ""
        self.tools = {}
        self.abort_count = 0
        self.start_count = 0
        self.process_calls = []
        self.lifecycle = []
        self.user_eou_requests = 0
        self.blank_fence_eou_requests = 0
        self.blank_fence_eou_cancellations = 0
        self.blank_fence_target = None
        self.user_transcript_resets = 0
        self.agent_eos_requests = 0
        self.response_boundary = SimpleNamespace(max_tail_frames=25)
        self.emit_user_text = True
        self.emit_response_end = True
        self.simulate_eou_settlement = False
        self.settlement_target = 1
        self.settlement_blank_count = 1
        self.settlement_stuck = False
        self.stall_after_blank_fence_arm = False
        self.pre_eou_function_activity_after_silence = False
        self.pre_eou_response_after_silence = False
        self._settled_response_started = False
        self.start_failure: Exception | None = None
        self.block_process = False
        self.process_started = Event()
        self.process_release = Event()

    def configure_external_tools(self, tools) -> None:
        self.tools = tools

    def reset_user_transcript(self) -> None:
        self.user_transcript_resets += 1

    def start(self, system_prompt=""):
        if self.start_failure is not None:
            raise self.start_failure
        self.started = True
        self.system_prompt = system_prompt
        self.start_count += 1
        self.lifecycle.append("start")
        return {
            "schema": 1,
            "timings_ms": {
                "reset": 0.1,
                "open_session": 0.2,
                "prefill": 0.3,
                "finalize": 0.1,
                "total": 0.7,
            },
        }

    def process(self, samples, *, is_last=False, input_active=None):
        if self.block_process:
            self.process_started.set()
            if not self.process_release.wait(timeout=5.0):
                raise TimeoutError("test did not release blocked model process")
        self.process_calls.append(
            {
                "samples": samples.copy(),
                "input_active": input_active,
                "is_last": is_last,
            }
        )
        self.lifecycle.append("process")
        index = self.frame_index
        self.frame_index += 1
        audio = np.zeros(1764, dtype=np.float32)
        user_text = ""
        assistant_delta = ""
        rnnt = {
            "speech_confirmed": False,
            "blank_count": self.settlement_blank_count,
        }
        control = "pad"
        boundary = {"event": None, "phase": "idle"}
        function_calling = {"active": False, "effective_token_is_pad": True}
        blank_fence_marker = None
        if self.simulate_eou_settlement:
            if np.max(np.abs(samples)) > 0:
                self.settlement_blank_count = 0
                user_text = "hello" if self.emit_user_text else ""
                rnnt["speech_confirmed"] = True
            else:
                if not self.settlement_stuck:
                    self.settlement_blank_count += 1
                user_text = "hello" if self.emit_user_text else ""
                if self.pre_eou_function_activity_after_silence:
                    function_calling["active"] = True
                if self.pre_eou_response_after_silence and not self.user_eou_requests:
                    assistant_delta = "Too early."
                    control = "agent_bos"
                    boundary = {"event": "start", "phase": "responding"}
                    audio[:] = 0.05
                if (
                    self.blank_fence_target is not None
                    and self.settlement_blank_count >= self.blank_fence_target
                ):
                    blank_fence_marker = {
                        "frame": index,
                        "blank_count_at_check": self.settlement_blank_count,
                        "target_blank_frames": self.blank_fence_target,
                        "chosen_token_id": 1,
                    }
                    self.wrapper._external_user_eou_blank_fence_last = dict(
                        blank_fence_marker
                    )
                    self.blank_fence_target = None
                    self.settlement_blank_count = 0
                    rnnt["blank_count"] = 0
                    control = "agent_bos"
                    boundary = {"event": "start", "phase": "responding"}
                    self._settled_response_started = True
                    self.eartts._reset_on_bos_count += 1
            rnnt["blank_count"] = self.settlement_blank_count
            if self.user_eou_requests and not self._settled_response_started:
                self._settled_response_started = True
                assistant_delta = "Hello."
                control = "agent_bos"
                boundary = {"event": "start", "phase": "responding"}
                audio[:] = 0.05
        elif getattr(self, "external_user_eou_mode", False):
            user_text = "hello" if self.emit_user_text else ""
            if self.user_eou_requests and not self._settled_response_started:
                self._settled_response_started = True
                assistant_delta = "Hello."
                control = "agent_bos"
                boundary = {"event": "start", "phase": "responding"}
                audio[:] = 0.05
            elif self._settled_response_started and self.emit_response_end:
                control = "agent_eos"
                boundary = {
                    "event": "end",
                    "phase": "idle",
                    "boundary_reason": "post_eos_audio_silence",
                }
        elif index == 0:
            user_text = "hello" if self.emit_user_text else ""
            rnnt["speech_confirmed"] = True
        elif index == 1:
            user_text = "hello" if self.emit_user_text else ""
            assistant_delta = "Hello."
            control = "agent_bos"
            boundary = {"event": "start", "phase": "responding"}
            audio[:] = 0.05
        elif index == 2 and self.emit_response_end:
            user_text = "hello" if self.emit_user_text else ""
            control = "agent_eos"
            boundary = {
                "event": "end",
                "phase": "idle",
                "boundary_reason": "post_eos_audio_silence",
            }
        if is_last:
            self.started = False
        return StepResult(
            audio=audio,
            user_text=user_text,
            assistant_delta=assistant_delta,
            assistant_text=assistant_delta,
            inference_ms=10.0,
            server_step_ms=11.0,
            frame_index=index,
            input_audio=audio_stats(samples),
            output_audio=audio_stats(audio),
            turn_state={
                "rnnt": rnnt,
                "agent_control": control,
                "response_boundary": boundary,
                "function_calling": function_calling,
                "control_ids": {"agent_bos": 1},
                **(
                    {"client_eou_blank_fence": blank_fence_marker}
                    if blank_fence_marker is not None
                    else {}
                ),
            },
        )

    def abort(self) -> None:
        self.abort_count += 1
        self.started = False
        self.lifecycle.append("abort")

    def request_user_eou(self) -> None:
        self.user_eou_requests += 1

    def request_user_eou_at_blank_fence(self, target_blank_frames: int) -> None:
        self.blank_fence_eou_requests += 1
        self.blank_fence_target = target_blank_frames
        self.wrapper._external_user_eou_blank_fence_last = None
        if self.stall_after_blank_fence_arm:
            self.settlement_stuck = True

    def cancel_user_eou_at_blank_fence(self) -> None:
        self.blank_fence_eou_cancellations += 1
        self.blank_fence_target = None
        self.wrapper._external_user_eou_blank_fence_last = None

    def user_eou_settlement_blank_frames(self) -> int:
        # Most endpoint tests isolate protocol ordering rather than the live
        # model's acoustic settlement contract.
        return self.settlement_target

    def rnnt_blank_count(self, result) -> int:
        return self.settlement_blank_count if result is not None else 0

    def request_agent_eos(self) -> None:
        self.agent_eos_requests += 1


class PreparedEpochFakeEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.stream_id = 1
        self.pipeline._request_id_for_stream = lambda _stream_id: "1"
        self.backend_request_id = "1__eartts_generation_1"
        self.eartts.tts_model = SimpleNamespace(
            engine=SimpleNamespace(
                requests={
                    "1": SimpleNamespace(
                        backend_request_id=self.backend_request_id,
                        backend_generation=1,
                    )
                }
            )
        )
        self.epoch_status = {
            "schema": 2,
            "enabled": True,
            "state": "invalid",
            "reason": "not_started",
            "session_epoch": 1,
            "prepare_epoch": 0,
            "request_id": None,
            "backend_request_id": None,
            "backend_generation": None,
            "generated_tokens": None,
            "armed_client_turn_id": None,
            "prepare_count": 0,
            "prepare_failure_count": 0,
            "prepared_reuse_count": 0,
            "bos_transition_count": 0,
            "last_bos_event": None,
        }
        self.wrapper._eartts_prepared_epoch_status = lambda: dict(self.epoch_status)

    def start(self, system_prompt=""):
        evidence = super().start(system_prompt)
        self.epoch_status.update(
            state="clean_unarmed",
            reason="session_prefill_attested",
            prepare_epoch=1,
            request_id="1",
            backend_request_id=self.backend_request_id,
            backend_generation=1,
            generated_tokens=1,
        )
        return evidence

    def prepare_eartts_epoch(self):
        self.lifecycle.append("prepare")
        return {
            "passed": True,
            **self.epoch_status,
            "timing": {"site": "turn_start_prepare"},
        }

    def arm_eartts_epoch(self, client_turn_id):
        self.lifecycle.append("arm")
        self.epoch_status.update(
            state="clean_armed",
            reason="client_commit_armed",
            armed_client_turn_id=client_turn_id,
        )
        return {"passed": True, **self.epoch_status}

    def process(self, samples, *, is_last=False, input_active=None):
        result = super().process(samples, is_last=is_last, input_active=input_active)
        if result.turn_state.get("agent_control") == "agent_bos":
            turn_id = self.epoch_status["armed_client_turn_id"]
            self.eartts._reset_on_bos_count -= 1
            self.epoch_status.update(
                state="dirty",
                reason="prepared_bos_completed",
                request_id=None,
                backend_request_id=self.backend_request_id,
                backend_generation=1,
                generated_tokens=None,
                armed_client_turn_id=None,
                prepared_reuse_count=self.epoch_status["prepared_reuse_count"] + 1,
                bos_transition_count=self.epoch_status["bos_transition_count"] + 1,
            )
            self.epoch_status["last_bos_event"] = {
                "schema": 2,
                "mode": "prepared_reuse",
                "request_id": "1",
                "backend_request_id": self.backend_request_id,
                "backend_generation": 1,
                "armed_client_turn_id": turn_id,
                "session_epoch": 1,
                "prepare_epoch": 1,
                "reset_count": self.eartts._reset_on_bos_count,
                "transition_count": self.epoch_status["bos_transition_count"],
                "reuse_count": self.epoch_status["prepared_reuse_count"],
                "fallback_reason": None,
            }
        return result


class DeferredToolTimeoutFakeEngine(PreparedEpochFakeEngine):
    def __init__(self, *, invalidation_failure: bool = False) -> None:
        super().__init__()
        self.simulate_eou_settlement = True
        self.settlement_target = 2
        self.settlement_blank_count = 0
        self.invalidation_failure = invalidation_failure
        self.invalidations: list[str] = []
        self.tool_results: list[str] = []
        self._tool_started = False
        self.publication_delay = 0.01
        self.unpublished_interrupts = 0
        self.function_worker_finished = False
        self.function_worker_registered = Event()
        self.abort_join_timed_out = False
        self.cleanup_completed = Event()
        self.pipeline._fc_async_bg = {}
        self.wrapper._build_fc_response_tokens = lambda value: list(value.encode())

    def interrupt_unpublished_function_cycle(self, _timeout_seconds=5.0):
        self.unpublished_interrupts += 1
        return {
            "schema": 1,
            "requested": True,
            "reason": "client_turn_start",
            "quit_to_normal": True,
            "async_steps": 1,
        }

    def function_cycle_quiescent(self):
        return self.unpublished_interrupts > 0 and self.function_worker_finished

    def invalidate_eartts_epoch(self, reason: str) -> dict:
        self.invalidations.append(reason)
        if self.invalidation_failure:
            raise RuntimeError("invalidation failed")
        self.epoch_status.update(
            state="invalid",
            reason=reason,
            request_id=None,
            generated_tokens=None,
            armed_client_turn_id=None,
        )
        return dict(self.epoch_status)

    def process(self, samples, *, is_last=False, input_active=None):
        start_tool = bool(
            self.user_eou_requests and self.tools and not self._tool_started
        )
        if start_tool:
            # Defer agent BOS while the external function cycle owns recovery.
            self._settled_response_started = True
        result = super().process(
            samples,
            is_last=is_last,
            input_active=input_active,
        )
        if not start_tool:
            return result
        self._tool_started = True
        result.turn_state["agent_control"] = "pad"
        result.turn_state["response_boundary"] = {"event": None, "phase": "idle"}
        result.turn_state["function_calling"] = {
            "active": True,
            "effective_token_is_pad": False,
        }
        handler = next(iter(self.tools.values()))
        worker: Thread

        def invoke_after_publication() -> None:
            time.sleep(self.publication_delay)
            self.pipeline._fc_async_bg[self.stream_id] = worker
            self.function_worker_registered.set()
            try:
                self.tool_results.append(handler({}))
            finally:
                self.pipeline._fc_async_bg.pop(self.stream_id, None)
                self.function_worker_finished = True

        worker = Thread(target=invoke_after_publication, daemon=True)
        worker.start()
        return result

    def abort(self) -> None:
        worker = self.pipeline._fc_async_bg.get(self.stream_id)
        if isinstance(worker, Thread):
            worker.join(timeout=5.0)
            self.abort_join_timed_out = worker.is_alive()
        super().abort()
        self.cleanup_completed.set()


class BlockingAbortFakeEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.abort_entered = Event()
        self.abort_release = Event()
        self.abort_completed = Event()
        self.abort_wait_timed_out = False

    def abort(self) -> None:
        self.abort_entered.set()
        if not self.abort_release.wait(timeout=2.0):
            self.abort_wait_timed_out = True
        super().abort()
        self.abort_completed.set()


class RecordingAsyncWebSocket:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.inbound: asyncio.Queue[dict] = asyncio.Queue()
        self.sent: list[dict] = []
        self.fatal_abort_completed: list[bool] = []
        self.closed: list[tuple[int, str | None, bool]] = []
        self.accepted = False

    def queue_json(self, body: dict) -> None:
        self.inbound.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(body)}
        )

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        return await self.inbound.get()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)
        if data.get("type") == "error" and data.get("error", {}).get("fatal"):
            completed = getattr(self.engine, "abort_completed", None)
            self.fatal_abort_completed.append(
                bool(completed.is_set()) if isinstance(completed, Event) else True
            )

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        completed = getattr(self.engine, "abort_completed", None)
        self.closed.append(
            (
                code,
                reason,
                bool(completed.is_set()) if isinstance(completed, Event) else True,
            )
        )


def audio_event(
    event_id: str = "audio-1",
    *,
    amplitude: float = 0.1,
    samples: int = FRAME_SAMPLES,
) -> dict:
    pcm = np.full(samples, amplitude, dtype="<f4")
    pcm16 = np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()
    return {
        "type": "input_audio_buffer.append",
        "event_id": event_id,
        "encoding": "pcm16",
        "sample_rate": INPUT_SAMPLE_RATE,
        "channels": 1,
        "audio": base64.b64encode(pcm16).decode("ascii"),
    }


def receive_until(websocket, event_type: str, limit: int = 20):
    events = []
    for _ in range(limit):
        event = websocket.receive_json()
        events.append(event)
        if event["type"] == event_type:
            return event, events
    raise AssertionError(f"did not receive {event_type}; got {[e['type'] for e in events]}")


class RealtimeWebSocketEndpointTest(unittest.TestCase):
    def make_client(
        self, *, typed_input=False, pocket_worker=None, engine=None, **app_kwargs
    ):
        temporary = tempfile.TemporaryDirectory()
        engine = engine or FakeEngine()
        pocket = pocket_worker or (FakePocketWorker() if typed_input else None)
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            pocket_worker=pocket,
            **app_kwargs,
        )
        return temporary, engine, TestClient(app)

    def trace_events(self, temporary) -> list[dict]:
        event_paths = list(Path(temporary.name).glob("*/events.jsonl"))
        self.assertEqual(len(event_paths), 1)
        return [json.loads(line) for line in event_paths[0].read_text().splitlines()]

    def wait_for_trace_event(self, temporary, event_name: str) -> dict:
        for _ in range(100):
            for event in self.trace_events(temporary):
                if event["event"] == event_name:
                    return event
            time.sleep(0.01)
        self.fail(f"trace did not emit {event_name}")

    def configure(self, websocket, *, tools=None, model_output=False):
        session = {
            "protocol_version": 3,
            "instructions": "Be brief.",
            "tools": [] if tools is None else tools,
        }
        if model_output:
            session["capabilities"] = {
                "function_output_model_output": "client_authored_v1"
            }
        websocket.send_json(
            {
                "type": "session.update",
                "event_id": "config-1",
                "session": session,
            }
        )
        updated = websocket.receive_json()
        self.assertEqual(updated["type"], "session.updated")
        return updated

    @staticmethod
    async def wait_until(predicate, *, timeout: float = 1.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.001)
        return True

    @staticmethod
    def realtime_endpoint(app):
        return next(route.endpoint for route in app.routes if route.path == "/v1/realtime")

    @staticmethod
    def health_endpoint(app):
        return next(route.endpoint for route in app.routes if route.path == "/health")

    @staticmethod
    def queue_configuration(websocket: RecordingAsyncWebSocket) -> None:
        websocket.queue_json(
            {
                "type": "session.update",
                "event_id": "config-1",
                "session": {
                    "protocol_version": 3,
                    "instructions": "Be brief.",
                    "tools": [],
                },
            }
        )

    def test_terminal_owner_drains_abort_when_directly_cancelled(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = BlockingAbortFakeEngine()
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def reproduce() -> tuple[bool, bool, bool]:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "fatal-commit",
                    "client_turn_id": 1,
                }
            )
            endpoint_task = asyncio.create_task(
                self.realtime_endpoint(app)(websocket),
                name="cancelled-owner-endpoint",
            )
            self.assertTrue(
                await self.wait_until(engine.abort_entered.is_set, timeout=1.0)
            )
            cleanup_task = next(
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith("voicechat-terminal-cleanup-")
            )
            cleanup_task.cancel()
            await asyncio.sleep(0.02)
            fatal_before_abort = bool(websocket.fatal_abort_completed)
            close_before_abort = bool(websocket.closed)
            trace_stopped_before_abort = any(
                event["event"] == "session_trace_stopped"
                for event in self.trace_events(temporary)
            )
            health_while_abort_blocked = await self.health_endpoint(app)()
            engine.abort_release.set()
            self.assertTrue(await self.wait_until(endpoint_task.done, timeout=1.0))
            await asyncio.gather(endpoint_task, return_exceptions=True)
            self.assertTrue(cleanup_task.done())
            self.assertFalse(cleanup_task.cancelled())
            self.assertEqual(cleanup_task.exception(), None)
            self.assertTrue(engine.abort_completed.is_set())
            self.assertFalse(engine.abort_wait_timed_out)
            self.assertEqual(engine.abort_count, 1)
            self.assertEqual(websocket.fatal_abort_completed, [True])
            self.assertEqual(websocket.closed, [(1002, "input_turn_conflict", True)])
            self.assertTrue(health_while_abort_blocked["active_client"])
            self.assertFalse((await self.health_endpoint(app)())["active_client"])
            return fatal_before_abort, close_before_abort, trace_stopped_before_abort

        try:
            premature = asyncio.run(reproduce())
        finally:
            engine.abort_release.set()
        self.assertEqual(premature, (False, False, False))
        self.wait_for_trace_event(temporary, "session_trace_stopped")

    def test_terminal_owner_drains_abort_during_all_task_shutdown_cancellation(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = BlockingAbortFakeEngine()
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def reproduce() -> tuple[bool, bool, bool]:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "shutdown-fatal-commit",
                    "client_turn_id": 1,
                }
            )
            endpoint_task = asyncio.create_task(
                self.realtime_endpoint(app)(websocket),
                name="shutdown-cancelled-endpoint",
            )
            self.assertTrue(
                await self.wait_until(engine.abort_entered.is_set, timeout=1.0)
            )
            cleanup_task = next(
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith("voicechat-terminal-cleanup-")
            )
            current_task = asyncio.current_task()
            self.assertIsNotNone(current_task)
            for task in list(asyncio.all_tasks()):
                task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                while current_task is not None and current_task.cancelling():
                    current_task.uncancel()
            await asyncio.sleep(0.02)
            fatal_before_abort = bool(websocket.fatal_abort_completed)
            close_before_abort = bool(websocket.closed)
            trace_stopped_before_abort = any(
                event["event"] == "session_trace_stopped"
                for event in self.trace_events(temporary)
            )
            self.assertTrue((await self.health_endpoint(app)())["active_client"])
            engine.abort_release.set()
            self.assertTrue(await self.wait_until(endpoint_task.done, timeout=1.0))
            await asyncio.gather(endpoint_task, return_exceptions=True)
            self.assertTrue(cleanup_task.done())
            self.assertFalse(cleanup_task.cancelled())
            self.assertEqual(cleanup_task.exception(), None)
            self.assertTrue(engine.abort_completed.is_set())
            self.assertFalse(engine.abort_wait_timed_out)
            self.assertEqual(engine.abort_count, 1)
            self.assertEqual(websocket.fatal_abort_completed, [True])
            self.assertEqual(websocket.closed, [(1002, "input_turn_conflict", True)])
            self.assertFalse((await self.health_endpoint(app)())["active_client"])
            return fatal_before_abort, close_before_abort, trace_stopped_before_abort

        try:
            premature = asyncio.run(reproduce())
        finally:
            engine.abort_release.set()
        self.assertEqual(premature, (False, False, False))
        self.wait_for_trace_event(temporary, "session_trace_stopped")

    def test_racing_endpoint_and_typed_fatals_complete_without_wait_cycle(
        self,
    ) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.block_process = True
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            pocket_worker=FakePocketWorker(),
            client_turn_detection=True,
            max_session_model_frames=1,
        )

        async def reproduce() -> tuple[float, bool]:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            endpoint_task = asyncio.create_task(
                self.realtime_endpoint(app)(websocket),
                name="two-fatal-branch-endpoint",
            )
            self.assertTrue(
                await self.wait_until(
                    lambda: any(
                        event.get("type") == "session.updated"
                        for event in websocket.sent
                    )
                )
            )
            websocket.queue_json(
                {
                    "type": "input_text.request",
                    "job_id": "typed-fatal-race",
                    "text": "Race the terminal owner",
                }
            )
            self.assertTrue(
                await self.wait_until(engine.process_started.is_set, timeout=1.0)
            )
            typed_task = next(
                task
                for task in asyncio.all_tasks()
                if task.get_name() == "typed-input-typed-fatal-race"
            )
            started_at = time.monotonic()
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "conflicting-turn",
                    "client_turn_id": 1,
                }
            )
            self.assertTrue(
                await self.wait_until(
                    lambda: any(
                        task.get_name().startswith("voicechat-terminal-cleanup-")
                        for task in asyncio.all_tasks()
                    )
                )
            )
            cleanup_task = next(
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith("voicechat-terminal-cleanup-")
            )
            engine.process_release.set()
            bounded = await self.wait_until(
                lambda: bool(websocket.fatal_abort_completed), timeout=1.0
            )
            elapsed = time.monotonic() - started_at
            if not bounded:
                cleanup_task.cancel()
            self.assertTrue(await self.wait_until(endpoint_task.done, timeout=1.0))
            await asyncio.gather(endpoint_task, return_exceptions=True)
            self.assertTrue(typed_task.done())
            self.assertTrue(cleanup_task.done())
            self.assertEqual(engine.abort_count, 1)
            self.assertEqual(websocket.fatal_abort_completed, [True])
            self.assertEqual(websocket.closed, [(1002, "input_turn_conflict", True)])
            self.assertFalse((await self.health_endpoint(app)())["active_client"])
            return elapsed, bounded

        try:
            elapsed, bounded = asyncio.run(reproduce())
        finally:
            engine.process_release.set()
        self.assertTrue(bounded)
        self.assertLess(elapsed, 1.0)
        events = self.trace_events(temporary)
        self.assertEqual(
            [event["reason"] for event in events if event["event"] == "terminal_session_claimed"],
            ["input_turn_conflict"],
        )
        self.assertEqual(
            [
                event["reason"]
                for event in events
                if event["event"] == "terminal_cleanup_wait_skipped"
            ],
            ["session_position_limit"],
        )
        self.assertTrue(any(event["event"] == "session_trace_stopped" for event in events))

    def test_strict_handshake_and_permanent_stop(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            created = websocket.receive_json()
            self.assertEqual(created["type"], "session.created")
            self.assertEqual(created["protocol"]["version"], 3)
            self.assertEqual(created["session"]["input_audio_sample_rate"], 16_000)
            self.assertEqual(created["session"]["output_audio_sample_rate"], 22_050)
            updated = self.configure(websocket)
            self.assertEqual(updated["session"]["protocol_version"], 3)
            websocket.send_json({"type": "session.stop", "event_id": "stop-1"})
            closed, events = receive_until(websocket, "session.closed")
            self.assertEqual(closed["reason"], "client_stop")
            self.assertNotIn("session.created", [event["type"] for event in events])
        self.assertGreaterEqual(engine.abort_count, 1)

    def test_session_updated_is_model_ready_without_consuming_audio(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            self.assertTrue(engine.started)
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.process_calls, [])
            websocket.send_json({"type": "session.stop", "event_id": "ready-empty-stop"})
            receive_until(websocket, "session.closed")
        self.wait_for_trace_event(temporary, "model_session_empty_stop_completed")
        self.assertEqual(engine.process_calls, [])
        self.assertGreaterEqual(engine.abort_count, 1)

        events = self.trace_events(temporary)
        names = [event["event"] for event in events]
        self.assertLess(
            names.index("model_session_start_completed"),
            names.index("session_tools_configured"),
        )
        completed = next(
            event for event in events if event["event"] == "model_session_start_completed"
        )
        self.assertEqual(completed["reason"], "session_update_initial")
        self.assertEqual(completed["start_evidence"]["schema"], 1)
        self.assertEqual(completed["start_evidence"]["timings_ms"]["total"], 0.7)

    def test_unnegotiated_model_output_is_rejected_before_tool_submission(self) -> None:
        temporary, _engine, client = self.make_client(
            function_template_path=Path("/unused/test-template.jinja")
        )
        self.addCleanup(temporary.cleanup)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with (
            mock.patch(
                "nemotron_voicechat_runtime.server.render_tool_system_prompt",
                return_value="tool-aware prompt",
            ),
            client.websocket_connect("/v1/realtime") as websocket,
        ):
            websocket.receive_json()
            self.configure(websocket, tools=tools)
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "event_id": "unnegotiated-model-output",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call-unknown",
                        "output": "full",
                        "model_output": "concise",
                    },
                }
            )
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "invalid_function_call_output")
            self.assertIn("not negotiated", error["error"]["message"])

    def test_identical_pre_frame_update_reacks_without_restart(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "same-config",
                    "session": {
                        "protocol_version": 3,
                        "instructions": "Be brief.",
                        "tools": [],
                    },
                }
            )
            updated = websocket.receive_json()
            self.assertEqual(updated["type"], "session.updated")
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.abort_count, 0)

    def test_changed_pre_frame_update_aborts_empty_session_and_restarts(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "changed-config",
                    "session": {"instructions": "Answer precisely.", "tools": []},
                }
            )
            updated = websocket.receive_json()
            self.assertEqual(updated["type"], "session.updated")
            self.assertEqual(updated["session"]["instructions"], "Answer precisely.")
            self.assertEqual(engine.start_count, 2)
            self.assertEqual(engine.abort_count, 1)
            self.assertEqual(engine.system_prompt, "Answer precisely.")
            self.assertEqual(engine.process_calls, [])
            self.assertEqual(engine.lifecycle[:3], ["start", "abort", "start"])

    def test_invalid_pre_frame_update_does_not_mutate_or_restart(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            prompt = engine.system_prompt
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "invalid-config",
                    "session": {
                        "instructions": "must not stick",
                        "tools": [{"name": "bad name", "parameters": {}}],
                    },
                }
            )
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "invalid_tools")
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.abort_count, 0)
            self.assertEqual(engine.system_prompt, prompt)

    def test_update_is_locked_while_typed_synthesis_is_in_flight(self) -> None:
        pocket = ReplacingFakePocketWorker()
        temporary, engine, client = self.make_client(
            pocket_worker=pocket,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-old", "text": "Old"}
            )
            self.assertEqual(websocket.receive_json()["type"], "input_text.accepted")
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "update-during-synthesis",
                    "session": {"instructions": "Must be rejected", "tools": []},
                }
            )
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "settings_locked")
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.abort_count, 0)
            self.assertEqual(engine.system_prompt, "Be brief.")

    def test_update_is_locked_while_empty_client_turn_is_open(self) -> None:
        temporary, engine, client = self.make_client(client_turn_detection=True)
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "open-empty-turn",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "update-during-empty-turn",
                    "session": {"instructions": "Must be rejected", "tools": []},
                }
            )
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "settings_locked")
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.abort_count, 0)
            self.assertEqual(engine.system_prompt, "Be brief.")

    def test_update_is_locked_while_typed_model_step_is_in_flight(self) -> None:
        temporary, engine, client = self.make_client(typed_input=True)
        self.addCleanup(temporary.cleanup)
        engine.block_process = True
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-race", "text": "Hello"}
            )
            self.assertEqual(websocket.receive_json()["type"], "input_text.accepted")
            self.assertTrue(engine.process_started.wait(timeout=2.0))
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "racing-config",
                    "session": {"instructions": "Must be rejected", "tools": []},
                }
            )
            events = []
            for _ in range(100):
                event = websocket.receive_json()
                events.append(event)
                if event.get("type") == "error" and event["error"]["code"] == "settings_locked":
                    break
            engine.process_release.set()
            self.assertEqual(events[-1]["error"]["code"], "settings_locked")
            self.assertEqual(engine.start_count, 1)
            self.assertEqual(engine.abort_count, 0)
            self.assertEqual(engine.system_prompt, "Be brief.")

    def test_no_audio_stop_aborts_empty_prefill_without_model_step(self) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json({"type": "session.stop", "event_id": "empty-stop"})
            closed, events = receive_until(websocket, "session.closed")
            self.assertEqual(closed["status"], "completed")
            self.assertNotIn("response.created", [event["type"] for event in events])
        self.assertEqual(engine.process_calls, [])
        self.assertGreaterEqual(engine.abort_count, 1)

    def test_eager_start_failure_is_fatal_before_session_updated(self) -> None:
        engine = FakeEngine()
        engine.start_failure = RuntimeError("prefill failed")
        temporary, engine, client = self.make_client(engine=engine)
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "failing-config",
                    "session": {
                        "protocol_version": 3,
                        "instructions": "Be brief.",
                        "tools": [],
                    },
                }
            )
            error = websocket.receive_json()
            closed = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "internal_error")
            self.assertTrue(error["error"]["fatal"])
            self.assertEqual(closed["status"], "failed")
            with self.assertRaises(WebSocketDisconnect):
                websocket.receive_json()
        self.assertEqual(engine.start_count, 0)
        self.assertGreaterEqual(engine.abort_count, 1)

    def test_stale_raw_websocket_browser_is_not_served(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(client.get("/").status_code, 404)

    def test_audio_before_configuration_is_rejected_without_starting_model(
        self,
    ) -> None:
        temporary, engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            websocket.send_json(audio_event())
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "session_not_configured")
            self.assertFalse(error["error"]["fatal"])
            self.assertFalse(engine.started)

    def test_missing_protocol_version_fails_handshake(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "session.update", "event_id": "bad", "session": {}})
            error = websocket.receive_json()
            closed = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "unsupported_protocol_version")
            self.assertTrue(error["error"]["fatal"])
            self.assertEqual(closed["type"], "session.closed")
            with self.assertRaises(WebSocketDisconnect) as raised:
                websocket.receive_json()
            self.assertEqual(raised.exception.code, 1002)

    def test_binary_audio_and_malformed_pcm_are_recoverable(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_bytes(b"\x00\x00")
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "binary_audio_unsupported")

            malformed = audio_event("odd")
            malformed["audio"] = base64.b64encode(b"\x00").decode("ascii")
            websocket.send_json(malformed)
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "invalid_audio")
            self.assertEqual(error["error"]["client_event_id"], "odd")

            oversized = audio_event("oversized")
            oversized["audio"] = base64.b64encode(b"\x00\x00" * 25_000).decode("ascii")
            websocket.send_json(oversized)
            error = websocket.receive_json()
            self.assertEqual(error["error"]["code"], "audio_packet_too_large")

    def test_settings_lock_null_restore_and_version_immutability(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)

            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "restore",
                    "session": {"instructions": None},
                }
            )
            restored = websocket.receive_json()
            self.assertEqual(restored["type"], "session.updated")
            self.assertEqual(restored["session"]["instructions"], "")

            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "change-version",
                    "session": {"protocol_version": 2},
                }
            )
            version_error = websocket.receive_json()
            self.assertEqual(version_error["error"]["code"], "unsupported_protocol_version")
            self.assertFalse(version_error["error"]["fatal"])

            websocket.send_json(audio_event())
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "session.update",
                    "event_id": "late-settings",
                    "session": {"instructions": "too late"},
                }
            )
            locked = websocket.receive_json()
            self.assertEqual(locked["error"]["code"], "settings_locked")
            self.assertEqual(locked["error"]["client_event_id"], "late-settings")

    def test_commit_and_session_close_are_recoverable_unsupported_events(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for event_type in ("input_audio_buffer.commit", "session.close"):
                websocket.send_json({"type": event_type, "event_id": event_type})
                error = websocket.receive_json()
                self.assertEqual(error["error"]["code"], "unsupported_event")
                self.assertFalse(error["error"]["fatal"])

    def test_client_smart_turn_start_and_commit_drive_one_response(self) -> None:
        temporary, engine, client = self.make_client(client_turn_detection=True)
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            created = websocket.receive_json()
            self.assertEqual(
                created["capabilities"]["input_turn_detection"],
                "client_smart_turn_v1",
            )
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-1",
                    "client_turn_id": 1,
                }
            )
            started = websocket.receive_json()
            self.assertEqual(started["type"], "input_audio_buffer.speech_started")
            start_ack = websocket.receive_json()
            self.assertEqual(start_ack["type"], "input_audio_buffer.turn_started")
            self.assertEqual(start_ack["client_event_id"], "start-1")

            websocket.send_json(audio_event("audio-smart"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-1",
                    "client_turn_id": 1,
                }
            )
            committed = websocket.receive_json()
            self.assertEqual(committed["type"], "input_audio_buffer.committed")
            self.assertEqual(committed["client_event_id"], "commit-1")
            _created, response_events = receive_until(websocket, "response.created")
            self.assertIn(
                "input_audio_buffer.speech_stopped",
                [event["type"] for event in response_events],
            )
            self.assertEqual(engine.user_eou_requests, 1)

            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-retry",
                    "client_turn_id": 1,
                }
            )
            duplicate, _events = receive_until(
                websocket, "input_audio_buffer.committed"
            )
            self.assertEqual(duplicate["type"], "input_audio_buffer.committed")
            self.assertEqual(engine.user_eou_requests, 1)

    def test_prepared_epoch_is_prepared_at_start_and_armed_only_after_pcm_flush(
        self,
    ) -> None:
        engine = PreparedEpochFakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 2
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "prepared-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            self.assertEqual(engine.lifecycle, ["start", "prepare"])

            websocket.send_json(audio_event("prepared-audio"))
            receive_until(websocket, "voicechat.metrics")
            self.assertEqual(engine.lifecycle, ["start", "prepare", "process"])
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "prepared-commit",
                    "client_turn_id": 1,
                }
            )
            committed, _events = receive_until(
                websocket, "input_audio_buffer.committed"
            )
            self.assertEqual(committed["client_turn_id"], 1)

        self.assertEqual(engine.lifecycle[:4], ["start", "prepare", "process", "arm"])
        self.assertEqual(engine.epoch_status["prepared_reuse_count"], 1)
        self.assertEqual(engine.epoch_status["bos_transition_count"], 1)
        self.assertEqual(engine.eartts._reset_on_bos_count, 0)
        self.assertIsNone(engine.epoch_status["armed_client_turn_id"])

    def test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails(
        self,
    ) -> None:
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        for invalidation_failure in (False, True):
            with self.subTest(invalidation_failure=invalidation_failure):
                engine = DeferredToolTimeoutFakeEngine(
                    invalidation_failure=invalidation_failure
                )
                fatal_abort_counts: list[int] = []
                endpoint_tasks: list[asyncio.Task] = []
                original_send_json = WebSocket.send_json

                async def observe_fatal_send(websocket, data, *args, **kwargs):
                    current_task = asyncio.current_task()
                    if not endpoint_tasks and current_task is not None:
                        endpoint_tasks.append(current_task)
                    if (
                        data.get("type") == "error"
                        and data.get("error", {}).get("code")
                        == "function_call_output_timeout"
                    ):
                        fatal_abort_counts.append(engine.abort_count)
                        result = await original_send_json(
                            websocket, data, *args, **kwargs
                        )
                        endpoint_tasks[0].cancel()
                        return result
                    return await original_send_json(websocket, data, *args, **kwargs)

                temporary, engine, client = self.make_client(
                    engine=engine,
                    client_turn_detection=True,
                    function_template_path=Path("/unused/test-template.jinja"),
                )
                self.addCleanup(temporary.cleanup)
                with (
                    mock.patch(
                        "nemotron_voicechat_runtime.server.render_tool_system_prompt",
                        return_value="tool-aware prompt",
                    ),
                    mock.patch(
                        "nemotron_voicechat_runtime.server.FUNCTION_CALL_TIMEOUT_SECONDS",
                        0.03,
                    ),
                    mock.patch.object(WebSocket, "send_json", new=observe_fatal_send),
                    client.websocket_connect("/v1/realtime") as websocket,
                ):
                    websocket.receive_json()
                    self.configure(websocket, tools=tools)
                    websocket.send_json(
                        {
                            "type": "input_audio_buffer.turn_start",
                            "event_id": "deferred-timeout-start",
                            "client_turn_id": 1,
                        }
                    )
                    receive_until(websocket, "input_audio_buffer.turn_started")
                    websocket.send_json(audio_event("deferred-timeout-audio"))
                    receive_until(websocket, "voicechat.metrics")
                    fatal_started_at = time.monotonic()
                    websocket.send_json(
                        {
                            "type": "input_audio_buffer.commit",
                            "event_id": "deferred-timeout-commit",
                            "client_turn_id": 1,
                        }
                    )
                    error, events = receive_until(websocket, "error", limit=40)
                    self.assertEqual(
                        error["error"]["code"], "function_call_output_timeout"
                    )
                    self.assertTrue(error["error"]["fatal"])
                    self.assertEqual(
                        error["error"]["epoch_invalidation_error"],
                        "RuntimeError" if invalidation_failure else None,
                    )
                    self.assertIn(
                        "response.function_call.failed",
                        [event["type"] for event in events],
                    )
                    with self.assertRaises(WebSocketDisconnect) as closed:
                        websocket.receive_json()
                    fatal_boundary_seconds = time.monotonic() - fatal_started_at
                    self.assertEqual(closed.exception.code, 1011)

                self.assertEqual(engine.invalidations, ["function_call_output_timeout"])
                self.assertEqual(fatal_abort_counts, [1])
                self.assertGreaterEqual(engine.abort_count, 1)
                self.assertLess(fatal_boundary_seconds, 1.0)
                self.assertTrue(engine.cleanup_completed.wait(timeout=0.5))
                self.assertTrue(engine.function_worker_registered.is_set())
                self.assertTrue(engine.function_worker_finished)
                self.assertFalse(engine.abort_join_timed_out)
                self.assertEqual(engine.pipeline._fc_async_bg, {})
                self.wait_for_trace_event(temporary, "session_trace_stopped")
                terminated = [
                    event
                    for event in self.trace_events(temporary)
                    if event["event"] == "pending_bos_expectation_terminated"
                ]
                self.assertEqual(len(terminated), 1)
                self.assertEqual(
                    terminated[0]["epoch_invalidation_error"],
                    "RuntimeError" if invalidation_failure else None,
                )

    def test_deferred_tool_closes_turn_and_barge_cancels_before_publication(self) -> None:
        engine = DeferredToolTimeoutFakeEngine()
        engine.publication_delay = 0.2
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
            function_template_path=Path("/unused/test-template.jinja"),
        )
        self.addCleanup(temporary.cleanup)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with (
            mock.patch(
                "nemotron_voicechat_runtime.server.render_tool_system_prompt",
                return_value="tool-aware prompt",
            ),
            client.websocket_connect("/v1/realtime") as websocket,
        ):
            websocket.receive_json()
            self.configure(websocket, tools=tools)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "first-start",
                    "client_turn_id": 1,
                }
            )
            first_started, _events = receive_until(
                websocket, "input_audio_buffer.turn_started"
            )
            first_turn_id = first_started["turn_id"]
            websocket.send_json(audio_event("first-audio"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "first-commit",
                    "client_turn_id": 1,
                }
            )
            committed, commit_events = receive_until(
                websocket, "input_audio_buffer.committed"
            )
            self.assertEqual(committed["turn_id"], first_turn_id)

            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "barge-start",
                    "client_turn_id": 2,
                }
            )
            second_started, turn_events = receive_until(
                websocket, "input_audio_buffer.turn_started", limit=30
            )
            self.assertNotEqual(second_started["turn_id"], first_turn_id)
            lifecycle = commit_events + turn_events
            self.assertIn(
                "conversation.item.input_audio_transcription.completed",
                [event["type"] for event in lifecycle],
            )

            websocket.send_json(audio_event("barge-audio"))
            receive_until(websocket, "voicechat.metrics")
            self.assertEqual(engine.unpublished_interrupts, 1)
            time.sleep(engine.publication_delay + 0.05)
            websocket.send_json(audio_event("barge-reconcile"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop"})
            receive_until(websocket, "session.closed")

        trace_events = self.trace_events(temporary)
        self.assertEqual(
            len(
                [
                    event
                    for event in trace_events
                    if event["event"] == "unpublished_function_cycle_interrupt"
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                [
                    event
                    for event in trace_events
                    if event["event"] == "function_call_publication_suppressed"
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                [
                    event
                    for event in trace_events
                    if event["event"] == "deferred_bos_superseded_by_client_turn"
                ]
            ),
            1,
        )

    def test_typed_finished_is_immediate_readiness_barrier_during_deferred_tool(self) -> None:
        engine = DeferredToolTimeoutFakeEngine()
        engine.publication_delay = 0.2
        temporary, engine, client = self.make_client(
            engine=engine,
            typed_input=True,
            client_turn_detection=True,
            function_template_path=Path("/unused/test-template.jinja"),
        )
        self.addCleanup(temporary.cleanup)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with (
            mock.patch(
                "nemotron_voicechat_runtime.server.render_tool_system_prompt",
                return_value="tool-aware prompt",
            ),
            client.websocket_connect("/v1/realtime") as websocket,
        ):
            websocket.receive_json()
            self.configure(websocket, tools=tools)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-tool", "text": "Use it"}
            )
            receive_until(websocket, "input_text.injection_finished", limit=100)

            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "immediate-after-typed",
                    "client_turn_id": 2,
                }
            )
            started, events = receive_until(
                websocket, "input_audio_buffer.turn_started", limit=30
            )
            self.assertEqual(started["client_event_id"], "immediate-after-typed")
            self.assertEqual(started["client_turn_id"], 2)
            self.assertNotIn("error", [event["type"] for event in events])

            websocket.send_json({"type": "session.stop", "event_id": "stop"})
            receive_until(websocket, "session.closed")

        self.assertEqual(engine.unpublished_interrupts, 1)

    def test_deferred_tool_recovery_timeout_clears_epoch_and_closes(self) -> None:
        engine = DeferredToolTimeoutFakeEngine()
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
            function_template_path=Path("/unused/test-template.jinja"),
        )
        self.addCleanup(temporary.cleanup)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with (
            mock.patch(
                "nemotron_voicechat_runtime.server.render_tool_system_prompt",
                return_value="tool-aware prompt",
            ),
            mock.patch(
                "nemotron_voicechat_runtime.server.FUNCTION_CALL_TIMEOUT_SECONDS",
                1.0,
            ),
            mock.patch(
                "nemotron_voicechat_runtime.server.advance_function_output_recovery",
                new=mock.AsyncMock(return_value=False),
            ),
            client.websocket_connect("/v1/realtime") as websocket,
        ):
            websocket.receive_json()
            updated = self.configure(websocket, tools=tools, model_output=True)
            self.assertEqual(
                updated["session"]["capabilities"]["function_output_model_output"],
                "client_authored_v1",
            )
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "recovery-timeout-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("recovery-timeout-audio"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "recovery-timeout-commit",
                    "client_turn_id": 1,
                }
            )
            function_call, _events = receive_until(
                websocket, "response.function_call_arguments.done", limit=30
            )
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "event_id": "recovery-timeout-output",
                    "item": {
                        "type": "function_call_output",
                        "call_id": function_call["call_id"],
                        "output": '{"spoken":"ordinary words"}',
                        "model_output": "Short answer.",
                    },
                }
            )
            error, _events = receive_until(websocket, "error", limit=20)
            self.assertEqual(error["error"]["code"], "function_output_apply_timeout")
            self.assertTrue(error["error"]["fatal"])

        self.assertEqual(engine.invalidations, ["function_output_apply_timeout"])
        self.assertEqual(engine.tool_results, ["Short answer."])
        received = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "function_call_output_received"
        )
        self.assertEqual(received["injection_source"], "model_output")
        self.assertEqual(received["model_output_tokens"], received["injection_tokens"])
        self.assertEqual(
            received["output_sha256"],
            hashlib.sha256(b'{"spoken":"ordinary words"}').hexdigest(),
        )
        self.assertEqual(
            received["model_output_sha256"],
            hashlib.sha256(b"Short answer.").hexdigest(),
        )
        terminated = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "pending_bos_expectation_terminated"
        ]
        self.assertEqual(len(terminated), 1)
        self.assertEqual(terminated[0]["code"], "function_output_apply_timeout")

    def test_tool_free_client_commit_fuses_final_blank_with_one_bos(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-settle",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-settle"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-settle",
                    "client_turn_id": 1,
                    "diagnostics": {
                        "schema": 1,
                        "client_enqueue_monotonic_s": 1.0,
                        "client_send_monotonic_s": 1.25,
                        "queue_chunks_at_enqueue": 4,
                    },
                }
            )
            _created, events = receive_until(websocket, "response.created")

        event_types = [event["type"] for event in events]
        self.assertLess(
            event_types.index("input_audio_buffer.committed"),
            event_types.index("response.created"),
        )
        # One real audio position and exactly three settlement positions.  The
        # third proves the fence and owns BOS, so there is no fifth EOU frame.
        self.assertEqual(len(engine.process_calls), 4)
        self.assertEqual(engine.user_eou_requests, 0)
        self.assertEqual(engine.blank_fence_eou_requests, 1)
        self.assertEqual(engine.blank_fence_eou_cancellations, 0)
        settled = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        ]
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]["model_steps"], 3)
        self.assertEqual(settled[0]["ending_blank_frames"], 3)
        self.assertTrue(settled[0]["fused_terminal_bos"])
        self.assertTrue(settled[0]["steps"][-1]["terminal_fused_bos"])
        self.assertEqual(
            settled[0]["steps"][-1]["client_eou_blank_fence"][
                "blank_count_at_check"
            ],
            3,
        )
        trace_events = self.trace_events(temporary)
        received = [
            event
            for event in trace_events
            if event["event"] == "client_input_turn_commit_received"
        ]
        committed = [
            event
            for event in trace_events
            if event["event"] == "client_input_turn_committed"
        ]
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["client_queue_wait_ms"], 250.0)
        self.assertEqual(received[0]["client_diagnostics"]["queue_chunks_at_enqueue"], 4)
        self.assertEqual(len(committed), 1)
        self.assertTrue(committed[0]["fused_terminal_bos"])
        self.assertGreaterEqual(committed[0]["receive_to_bos_ms"], 0)

    def test_precommit_quiet_frames_overlap_settlement_fence(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "prearm-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("prearm-speech"))
            receive_until(websocket, "voicechat.metrics")
            for index in range(3):
                websocket.send_json(
                    audio_event(f"prearm-quiet-{index}", amplitude=0.0)
                )
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "prearm-commit",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "response.created")

        settled = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["prearmed_blank_frames"], 3)
        self.assertEqual(settled["post_commit_blank_deficit"], 0)
        self.assertEqual(settled["model_steps"], 0)
        self.assertEqual(settled["real_audio_model_steps"], 0)
        self.assertEqual(settled["synthetic_model_steps"], 0)
        self.assertEqual(engine.user_eou_requests, 1)

    def test_precommit_arm_is_revoked_at_and_above_the_speech_gate(self) -> None:
        gate_payload = base64.b64decode(audio_event("gate", amplitude=0.01)["audio"])
        gate_samples = np.frombuffer(gate_payload, dtype="<i2").astype(np.float32) / 32768.0
        exact_gate_dbfs = float(audio_stats(gate_samples)["rms_dbfs"])
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
            speech_gate_dbfs=exact_gate_dbfs,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "revoke-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            for event in (
                audio_event("revoke-speech", amplitude=0.1),
                audio_event("revoke-quiet-exact", amplitude=0.0),
                audio_event("revoke-exact", amplitude=0.01),
                audio_event("revoke-quiet-above", amplitude=0.0),
                audio_event("revoke-above", amplitude=0.1),
            ):
                websocket.send_json(event)
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "revoke-commit",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "response.created")

        trace_events = self.trace_events(temporary)
        revoked = [
            event
            for event in trace_events
            if event["event"] == "user_eou_fence_prearm_revoked"
        ]
        self.assertEqual(len(revoked), 2)
        self.assertEqual(revoked[0]["audio_rms_dbfs"], exact_gate_dbfs)
        self.assertGreater(revoked[1]["audio_rms_dbfs"], exact_gate_dbfs)
        settled = next(
            event for event in trace_events if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["prearmed_blank_frames"], 0)
        self.assertEqual(settled["post_commit_blank_deficit"], 3)
        self.assertEqual(settled["synthetic_model_steps"], 3)

    def test_commit_tail_padding_cannot_arm_after_commit_receipt(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def exercise() -> None:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "partial-before-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("partial-before-speech"))
            websocket.queue_json(
                audio_event(
                    "partial-before-quiet",
                    amplitude=0.0,
                    samples=FRAME_SAMPLES // 2,
                )
            )
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "partial-before-commit",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json({"type": "session.stop"})
            await self.realtime_endpoint(app)(websocket)

        asyncio.run(exercise())
        trace_events = self.trace_events(temporary)
        commit_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "client_input_turn_commit_received"
        )
        padded_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "user_eou_post_commit_frame"
        )
        self.assertLess(commit_index, padded_index)
        self.assertFalse(
            any(
                event["event"] == "user_eou_fence_prearmed"
                for event in trace_events[commit_index + 1 :]
            )
        )
        settled = next(
            event for event in trace_events if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["prearmed_blank_frames"], 0)
        self.assertEqual(settled["starting_blank_frames"], 1)
        self.assertEqual(settled["post_commit_blank_deficit"], 2)
        self.assertEqual(settled["model_steps"], 2)
        self.assertEqual(settled["real_audio_model_steps"], 0)
        self.assertEqual(settled["synthetic_model_steps"], 2)
        self.assertEqual(
            settled["post_commit_steps"],
            [
                {
                    "model_input_provenance": "commit_tail_padding",
                    "real_samples": FRAME_SAMPLES // 2,
                    "padded_samples": FRAME_SAMPLES // 2,
                    "blank_frames_before": 0,
                    "blank_frames_after": 1,
                    "blank_frames_advanced": 1,
                    "model_frame": 1,
                }
            ],
        )

    def test_partial_audio_after_commit_remains_on_ordinary_path(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def exercise() -> None:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "partial-after-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("partial-after-speech"))
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "partial-after-commit",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(
                audio_event(
                    "partial-after-audio",
                    amplitude=0.2,
                    samples=FRAME_SAMPLES // 2,
                )
            )
            websocket.queue_json({"type": "session.stop"})
            await self.realtime_endpoint(app)(websocket)

        asyncio.run(exercise())
        settled = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["prearmed_blank_frames"], 0)
        self.assertEqual(settled["post_commit_blank_deficit"], 3)
        self.assertEqual(settled["real_audio_model_steps"], 0)
        self.assertEqual(settled["synthetic_model_steps"], 3)
        self.assertEqual(settled["post_commit_steps"], [])
        self.assertTrue(engine.process_calls[-1]["is_last"])
        self.assertEqual(
            int(np.count_nonzero(engine.process_calls[-1]["samples"])),
            FRAME_SAMPLES // 2,
        )

    def test_postcommit_quiet_queue_supplies_settlement_frames(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def exercise() -> None:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "queued-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("queued-speech"))
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "queued-commit",
                    "client_turn_id": 1,
                }
            )
            for index in range(3):
                websocket.queue_json(
                    audio_event(f"queued-quiet-{index}", amplitude=0.0)
                )
            websocket.queue_json({"type": "session.stop"})
            await self.realtime_endpoint(app)(websocket)

        asyncio.run(exercise())
        settled = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["prearmed_blank_frames"], 0)
        self.assertEqual(settled["post_commit_blank_deficit"], 3)
        self.assertEqual(settled["real_audio_model_steps"], 3)
        self.assertEqual(settled["synthetic_model_steps"], 0)
        self.assertEqual(
            [step["settlement_input_source"] for step in settled["steps"]],
            ["queued_microphone_audio"] * 3,
        )

    def test_settlement_does_not_cross_non_audio_event_in_receive_fifo(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def exercise() -> None:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "fifo-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("fifo-speech"))
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "fifo-commit",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("fifo-quiet", amplitude=0.0))
            websocket.queue_json(
                {
                    "type": "session.update",
                    "event_id": "fifo-boundary",
                    "session": {"protocol_version": 3},
                }
            )
            websocket.queue_json(audio_event("fifo-later-audio", amplitude=0.2))
            websocket.queue_json({"type": "session.stop"})
            await self.realtime_endpoint(app)(websocket)

        asyncio.run(exercise())
        trace_events = self.trace_events(temporary)
        settled_index, settled = next(
            (index, event)
            for index, event in enumerate(trace_events)
            if event["event"] == "user_eou_settled"
        )
        boundary_index = [
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "client_event" and event.get("type") == "session.update"
        ][-1]
        later_audio_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "input_packet"
            and event["audio"]["rms_dbfs"] > -20.0
        )
        self.assertEqual(settled["real_audio_model_steps"], 1)
        self.assertEqual(settled["synthetic_model_steps"], 2)
        self.assertLess(settled_index, boundary_index)
        self.assertLess(boundary_index, later_audio_index)
        later_calls = [
            call
            for call in engine.process_calls
            if np.isclose(float(call["samples"].mean()), 0.2, atol=1 / 32768)
        ]
        self.assertEqual(len(later_calls), 1)

    def test_postcommit_speech_is_not_fence_eligible(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
            speech_gate_dbfs=-40.0,
        )

        async def exercise() -> None:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "speech-gate-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("speech-gate-first"))
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "speech-gate-commit",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("speech-gate-barge", amplitude=0.2))
            websocket.queue_json({"type": "session.stop"})
            await self.realtime_endpoint(app)(websocket)

        asyncio.run(exercise())
        settled = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        )
        self.assertEqual(settled["real_audio_model_steps"], 0)
        self.assertEqual(settled["synthetic_model_steps"], 3)
        self.assertTrue(settled["real_audio_stopped_by_speech_gate"])
        blocked = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settlement_real_audio_blocked"
        ]
        self.assertEqual(len(blocked), 1)
        barge_calls = [
            call
            for call in engine.process_calls
            if np.isclose(float(call["samples"].mean()), 0.2, atol=1 / 32768)
        ]
        self.assertEqual(len(barge_calls), 1)
        trace_events = self.trace_events(temporary)
        settled_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "user_eou_settled"
        )
        barge_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "input_packet"
            and event["audio"]["rms_dbfs"] > -20.0
        )
        stop_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "client_event" and event.get("type") == "session.stop"
        )
        self.assertLess(settled_index, barge_index)
        self.assertLess(barge_index, stop_index)

    def test_bounded_receive_pump_releases_and_shutdown_makes_progress(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine()
        engine.block_process = True
        app = create_app(
            engine,
            trace_dir=Path(temporary.name),
            client_turn_detection=True,
        )

        async def exercise() -> RecordingAsyncWebSocket:
            websocket = RecordingAsyncWebSocket(engine)
            self.queue_configuration(websocket)
            websocket.queue_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "pump-start",
                    "client_turn_id": 1,
                }
            )
            websocket.queue_json(audio_event("pump-blocked-audio"))
            endpoint_task = asyncio.create_task(self.realtime_endpoint(app)(websocket))
            try:
                self.assertTrue(
                    await self.wait_until(engine.process_started.is_set),
                    "model step did not block",
                )
                for index in range(REALTIME_RECEIVE_QUEUE_MESSAGES + 1):
                    websocket.queue_json(
                        {
                            "type": "session.update",
                            "event_id": f"pump-queued-{index}",
                            "session": {"protocol_version": 3},
                        }
                    )
                websocket.queue_json({"type": "session.stop"})
                self.assertTrue(
                    await self.wait_until(
                        lambda: websocket.inbound.qsize() <= 2,
                        timeout=2.0,
                    ),
                    "receive pump did not fill its bounded queue",
                )
                self.assertFalse(endpoint_task.done())
                engine.process_release.set()
                await asyncio.wait_for(endpoint_task, timeout=5.0)
            finally:
                engine.process_release.set()
                if not endpoint_task.done():
                    endpoint_task.cancel()
                    await asyncio.gather(endpoint_task, return_exceptions=True)
            return websocket

        websocket = asyncio.run(exercise())
        self.assertTrue(websocket.closed)
        self.assertIn("abort", engine.lifecycle)
        trace_events = self.trace_events(temporary)
        self.assertTrue(
            any(
                event["event"] == "client_event" and event.get("type") == "session.stop"
                for event in trace_events
            )
        )

    def test_tool_session_keeps_separate_settlement_and_legacy_eou_position(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
            function_template_path=Path("/unused/test-template.jinja"),
        )
        self.addCleanup(temporary.cleanup)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with mock.patch(
            "nemotron_voicechat_runtime.server.render_tool_system_prompt",
            return_value="tool-aware prompt",
        ):
            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()
                self.configure(websocket, tools=tools)
                websocket.send_json(
                    {
                        "type": "input_audio_buffer.turn_start",
                        "event_id": "start-tool-settle",
                        "client_turn_id": 1,
                    }
                )
                receive_until(websocket, "input_audio_buffer.turn_started")
                websocket.send_json(audio_event("audio-tool-settle"))
                receive_until(websocket, "voicechat.metrics")
                websocket.send_json(
                    {
                        "type": "input_audio_buffer.commit",
                        "event_id": "commit-tool-settle",
                        "client_turn_id": 1,
                    }
                )
                _created, events = receive_until(websocket, "response.created")

        event_types = [event["type"] for event in events]
        self.assertLess(
            event_types.index("input_audio_buffer.committed"),
            event_types.index("response.created"),
        )
        self.assertEqual(len(engine.process_calls), 5)
        self.assertEqual(engine.user_eou_requests, 1)
        self.assertEqual(engine.blank_fence_eou_requests, 0)
        settled = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "user_eou_settled"
        ]
        self.assertEqual(len(settled), 1)
        self.assertNotIn("fused_terminal_bos", settled[0])
        self.assertTrue(all(step["agent_control"] == "pad" for step in settled[0]["steps"]))
        committed = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "client_input_turn_committed"
        ]
        self.assertEqual(len(committed), 1)
        self.assertNotIn("fused_terminal_bos", committed[0])
        trace_events = self.trace_events(temporary)
        commit_index = next(
            index
            for index, event in enumerate(trace_events)
            if event["event"] == "client_input_turn_commit_received"
        )
        self.assertFalse(
            any(
                event["event"] == "user_eou_fence_prearmed"
                for event in trace_events[commit_index + 1 :]
            )
        )
        input_provenance = [
            event["model_input_provenance"]
            for event in trace_events[commit_index + 1 :]
            if event["event"] == "input_packet"
        ]
        self.assertEqual(input_provenance[-1], "synthetic_terminal_bos")

    def test_client_commit_settlement_has_fatal_twice_fence_bound(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        engine.settlement_stuck = True
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-stuck-settle",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-stuck-settle"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-stuck-settle",
                    "client_turn_id": 1,
                }
            )
            error, events = receive_until(websocket, "error", limit=30)

        self.assertEqual(error["error"]["code"], "input_turn_settle_timeout")
        self.assertTrue(error["error"]["fatal"])
        self.assertNotIn(
            "input_audio_buffer.committed", [event["type"] for event in events]
        )
        self.assertEqual(len(engine.process_calls), 7)
        self.assertEqual(engine.user_eou_requests, 0)

    def test_client_commit_cancels_armed_fused_latch_on_settlement_timeout(
        self,
    ) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        engine.stall_after_blank_fence_arm = True
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-armed-timeout",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-armed-timeout"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-armed-timeout",
                    "client_turn_id": 1,
                }
            )
            error, events = receive_until(websocket, "error", limit=30)

        event_types = [event["type"] for event in events]
        self.assertEqual(error["error"]["code"], "input_turn_settle_timeout")
        self.assertTrue(error["error"]["fatal"])
        self.assertEqual(engine.blank_fence_eou_requests, 1)
        self.assertEqual(engine.blank_fence_eou_cancellations, 1)
        self.assertIsNone(engine.blank_fence_target)
        self.assertNotIn("input_audio_buffer.committed", event_types)
        self.assertNotIn("response.created", event_types)
        self.assertEqual(engine.user_eou_requests, 0)

    def test_client_commit_fails_closed_if_function_state_activates_before_eou(
        self,
    ) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        engine.pre_eou_function_activity_after_silence = True
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-pre-eou-function",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-pre-eou-function"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-pre-eou-function",
                    "client_turn_id": 1,
                }
            )
            error, events = receive_until(websocket, "error")

        self.assertEqual(error["error"]["code"], "pre_eou_function_activity")
        self.assertTrue(error["error"]["fatal"])
        self.assertNotIn(
            "input_audio_buffer.committed", [event["type"] for event in events]
        )
        self.assertEqual(len(engine.process_calls), 2)
        self.assertEqual(engine.user_eou_requests, 0)

    def test_client_commit_rejects_pre_eou_response_before_publication(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        engine.pre_eou_response_after_silence = True
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-pre-eou-response",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-pre-eou-response"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-pre-eou-response",
                    "client_turn_id": 1,
                }
            )
            error, events = receive_until(websocket, "error", limit=20)

        self.assertEqual(error["error"]["code"], "pre_eou_response_activity")
        self.assertTrue(error["error"]["fatal"])
        self.assertNotIn("response.created", [event["type"] for event in events])
        self.assertNotIn("response.output_text.delta", [event["type"] for event in events])
        self.assertNotIn("response.output_audio.delta", [event["type"] for event in events])
        self.assertEqual(engine.user_eou_requests, 0)

    def test_client_commit_with_no_rnnt_tokens_still_starts_response(self) -> None:
        temporary, engine, client = self.make_client(client_turn_detection=True)
        engine.emit_user_text = False
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-empty-rnnt",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-empty-rnnt"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-empty-rnnt",
                    "client_turn_id": 1,
                }
            )
            committed, commit_events = receive_until(
                websocket, "input_audio_buffer.committed"
            )
            created, response_events = receive_until(websocket, "response.created")

        self.assertEqual(committed["client_event_id"], "commit-empty-rnnt")
        self.assertEqual(created["turn_id"], committed["turn_id"])
        self.assertNotIn(
            "conversation.item.input_audio_transcription.delta",
            [event["type"] for event in commit_events + response_events],
        )
        self.assertEqual(engine.user_eou_requests, 1)

    def test_client_turn_start_cancels_open_response_to_terminal(self) -> None:
        temporary, engine, client = self.make_client(client_turn_detection=True)
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-1",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-1"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-1",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "response.created")

            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-2",
                    "client_turn_id": 2,
                }
            )
            done, events = receive_until(websocket, "response.done")
            self.assertEqual(done["status"], "cancelled")
            self.assertEqual(done["reason"], "client_vad")
            _ack, remainder = receive_until(
                websocket, "input_audio_buffer.turn_started"
            )
            all_types = [event["type"] for event in events + remainder]
            self.assertLess(
                all_types.index("response.done"),
                all_types.index("input_audio_buffer.speech_started"),
            )
            self.assertEqual(engine.agent_eos_requests, 1)

    def test_client_turn_cancellation_tail_timeout_is_fatal(self) -> None:
        temporary, engine, client = self.make_client(client_turn_detection=True)
        engine.emit_response_end = False
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-open",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-open"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-open",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "response.created")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-timeout",
                    "client_turn_id": 2,
                }
            )
            error, _events = receive_until(websocket, "error", limit=100)
            self.assertEqual(error["error"]["code"], "response_cancel_timeout")
            self.assertTrue(error["error"]["fatal"])
            with self.assertRaises(WebSocketDisconnect) as raised:
                websocket.receive_json()
            self.assertEqual(raised.exception.code, 1011)

    def test_position_limit_applies_to_synthetic_cancellation_steps(self) -> None:
        temporary, engine, client = self.make_client(
            client_turn_detection=True,
            max_session_model_frames=2,
        )
        engine.emit_response_end = False
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-limited",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-limited"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-limited",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "response.created")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-over-limit",
                    "client_turn_id": 2,
                }
            )
            error, _events = receive_until(websocket, "error")
            self.assertEqual(error["error"]["code"], "session_position_limit")
            self.assertTrue(error["error"]["fatal"])

        self.assertEqual(len(engine.process_calls), 2)

    def test_position_limit_applies_mid_client_commit_settlement(self) -> None:
        engine = FakeEngine()
        engine.simulate_eou_settlement = True
        engine.settlement_target = 3
        engine.settlement_blank_count = 0
        temporary, engine, client = self.make_client(
            engine=engine,
            client_turn_detection=True,
            max_session_model_frames=3,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-settlement-limit",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(audio_event("audio-settlement-limit"))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "commit-settlement-limit",
                    "client_turn_id": 1,
                }
            )
            error, events = receive_until(websocket, "error", limit=20)

        self.assertEqual(error["error"]["code"], "session_position_limit")
        self.assertTrue(error["error"]["fatal"])
        self.assertNotIn(
            "input_audio_buffer.committed", [event["type"] for event in events]
        )
        self.assertEqual(len(engine.process_calls), 3)
        self.assertEqual(engine.user_eou_requests, 0)

    def test_representative_turn_and_response_lifecycle(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            observed = []
            for index in range(3):
                websocket.send_json(audio_event(f"audio-{index}"))
                _metrics, events = receive_until(websocket, "voicechat.metrics")
                observed.extend(event["type"] for event in events)
            expected = [
                "input_audio_buffer.speech_started",
                "conversation.item.input_audio_transcription.delta",
                "input_audio_buffer.speech_stopped",
                "conversation.item.input_audio_transcription.completed",
                "response.created",
                "response.output_text.delta",
                "response.output_audio.delta",
                "response.output_text.done",
                "response.output_audio.done",
                "response.done",
            ]
            positions = [observed.index(event_type) for event_type in expected]
            self.assertEqual(positions, sorted(positions))

    def test_gate_preroll_replays_fifo_before_trigger_with_recorded_state(self) -> None:
        temporary, engine, client = self.make_client(
            speech_gate_dbfs=-30.0,
            speech_gate_min_frames=2,
            speech_gate_preroll_frames=3,
        )
        self.addCleanup(temporary.cleanup)
        amplitudes = [0.001, 0.002, 0.1, 0.2, 0.3]
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index, amplitude in enumerate(amplitudes):
                websocket.send_json(
                    audio_event(f"preroll-{index}", amplitude=amplitude)
                )
                expected_metrics = 4 if index == 3 else 1
                for _ in range(expected_metrics):
                    receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop-preroll"})
            receive_until(websocket, "session.closed")

        model_calls = engine.process_calls[:5]
        self.assertEqual(len(model_calls), 5)
        observed_amplitudes = [float(call["samples"].mean()) for call in model_calls]
        np.testing.assert_allclose(observed_amplitudes, amplitudes, atol=1 / 32768)
        self.assertEqual(
            [call["input_active"] for call in model_calls],
            [False, False, False, True, True],
        )
        self.assertEqual(engine.start_count, 1)
        self.assertEqual(engine.lifecycle[:2], ["start", "process"])

        model_steps = [
            event for event in self.trace_events(temporary) if event["event"] == "model_step"
        ]
        self.assertEqual(
            [event["transport_frame"] for event in model_steps[:5]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [event["preroll_replay"] for event in model_steps[:5]],
            [True, True, True, False, False],
        )
        self.assertEqual(
            [event["turn_state"]["transport"]["input_active"] for event in model_steps[:4]],
            [False, False, False, True],
        )

    def test_gate_preroll_skips_old_quiet_frames_but_preserves_onset(self) -> None:
        temporary, engine, client = self.make_client(
            speech_gate_dbfs=-40.0,
            speech_gate_min_frames=3,
            speech_gate_preroll_frames=12,
            speech_gate_onset_context_frames=2,
        )
        self.addCleanup(temporary.cleanup)
        amplitudes = [0.001] * 10 + [0.1, 0.1, 0.1]
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index, amplitude in enumerate(amplitudes):
                websocket.send_json(
                    audio_event(f"trimmed-preroll-{index}", amplitude=amplitude)
                )
                expected_metrics = 5 if index == len(amplitudes) - 1 else 1
                for _ in range(expected_metrics):
                    receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop-trimmed"})
            receive_until(websocket, "session.closed")

        self.assertEqual(len(engine.process_calls), 6)
        np.testing.assert_allclose(
            [float(call["samples"].mean()) for call in engine.process_calls[:5]],
            [0.001, 0.001, 0.1, 0.1, 0.1],
            atol=1 / 32768,
        )
        selected = [
            event
            for event in self.trace_events(temporary)
            if event["event"] == "speech_gate_preroll_selected"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["reason"], "transport_gate")
        self.assertEqual(selected[0]["buffered_frames"], 12)
        self.assertEqual(selected[0]["replayed_frames"], 4)
        self.assertEqual(selected[0]["skipped_frames"], 8)
        self.assertEqual(selected[0]["first_active_index"], 10)
        model_steps = [
            event for event in self.trace_events(temporary) if event["event"] == "model_step"
        ]
        self.assertEqual(
            [event["transport_frame"] for event in model_steps[:5]],
            [8, 9, 10, 11, 12],
        )

    def test_client_turn_start_replays_selected_entries_with_original_metadata(self) -> None:
        temporary, engine, client = self.make_client(
            client_turn_detection=True,
            speech_gate_dbfs=-40.0,
            speech_gate_min_frames=10,
            speech_gate_preroll_frames=12,
            speech_gate_onset_context_frames=2,
        )
        self.addCleanup(temporary.cleanup)
        amplitudes = [0.001] * 10 + [0.1, 0.1]
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index, amplitude in enumerate(amplitudes):
                websocket.send_json(
                    audio_event(f"barrier-preroll-{index}", amplitude=amplitude)
                )
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "barrier-preroll-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            for _ in range(4):
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop-barrier"})
            receive_until(websocket, "session.closed")

        np.testing.assert_allclose(
            [float(call["samples"].mean()) for call in engine.process_calls[:4]],
            [0.001, 0.001, 0.1, 0.1],
            atol=1 / 32768,
        )
        self.assertEqual(
            [call["input_active"] for call in engine.process_calls[:4]],
            [False, False, False, False],
        )
        events = self.trace_events(temporary)
        selected = next(
            event
            for event in events
            if event["event"] == "speech_gate_preroll_selected"
        )
        self.assertEqual(selected["reason"], "client_turn_start")
        self.assertEqual(selected["buffered_frames"], 12)
        self.assertEqual(selected["replayed_frames"], 4)
        self.assertEqual(selected["skipped_frames"], 8)
        self.assertEqual(selected["first_active_index"], 10)
        started = next(
            event for event in events if event["event"] == "speech_gate_preroll_started"
        )
        self.assertEqual(started["transport_frames"], [8, 9, 10, 11])
        self.assertEqual(started["sources"], ["microphone"] * 4)
        self.assertEqual(started["job_ids"], [None] * 4)
        self.assertEqual(started["input_active"], [False] * 4)
        model_steps = [event for event in events if event["event"] == "model_step"]
        self.assertEqual(
            [event["transport_frame"] for event in model_steps[:4]],
            [8, 9, 10, 11],
        )

    def test_client_turn_start_all_quiet_preroll_falls_back_to_full_replay(self) -> None:
        temporary, engine, client = self.make_client(
            client_turn_detection=True,
            speech_gate_dbfs=-40.0,
            speech_gate_min_frames=3,
            speech_gate_preroll_frames=3,
            speech_gate_onset_context_frames=2,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index in range(3):
                websocket.send_json(audio_event(f"quiet-barrier-{index}", amplitude=0.001))
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "quiet-barrier-start",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            for _ in range(3):
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop-quiet-barrier"})
            receive_until(websocket, "session.closed")

        self.assertEqual(len(engine.process_calls[:3]), 3)
        selected = next(
            event
            for event in self.trace_events(temporary)
            if event["event"] == "speech_gate_preroll_selected"
        )
        self.assertEqual(selected["reason"], "client_turn_start")
        self.assertIsNone(selected["first_active_index"])
        self.assertEqual(selected["skipped_frames"], 0)
        self.assertEqual(selected["replayed_frames"], 3)

    def test_typed_input_acceptance_clears_microphone_preroll(self) -> None:
        temporary, engine, client = self.make_client(
            typed_input=True,
            speech_gate_dbfs=-30.0,
            speech_gate_min_frames=1,
            speech_gate_preroll_frames=12,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(audio_event("quiet-mic", amplitude=0.001))
            receive_until(websocket, "voicechat.metrics")
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-clear", "text": "Hello"}
            )
            accepted = websocket.receive_json()
            self.assertEqual(accepted["type"], "input_text.accepted")
            receive_until(websocket, "input_text.injection_finished", limit=100)
            websocket.send_json({"type": "session.stop", "event_id": "stop-typed-clear"})
            receive_until(websocket, "session.closed")

        self.assertTrue(engine.process_calls)
        first_model_amplitude = float(engine.process_calls[0]["samples"].mean())
        self.assertGreater(first_model_amplitude, 0.1)

    def test_zero_gate_preroll_preserves_gate_drop_behavior(self) -> None:
        temporary, engine, client = self.make_client(
            speech_gate_dbfs=-30.0,
            speech_gate_min_frames=2,
            speech_gate_preroll_frames=0,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index, amplitude in enumerate((0.001, 0.1, 0.2)):
                websocket.send_json(audio_event(f"disabled-{index}", amplitude=amplitude))
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json({"type": "session.stop", "event_id": "stop-disabled"})
            receive_until(websocket, "session.closed")

        self.assertEqual(len(engine.process_calls), 2)
        np.testing.assert_allclose(
            engine.process_calls[0]["samples"].mean(), 0.2, atol=1 / 32768
        )
        self.assertTrue(engine.process_calls[0]["input_active"])
        model_steps = [
            event for event in self.trace_events(temporary) if event["event"] == "model_step"
        ]
        self.assertEqual(model_steps[0]["transport_frame"], 2)
        self.assertFalse(model_steps[0]["preroll_replay"])

    def test_session_position_limit_stops_mid_preroll_drain(self) -> None:
        temporary, engine, client = self.make_client(
            speech_gate_dbfs=-30.0,
            speech_gate_min_frames=1,
            speech_gate_preroll_frames=2,
            max_session_model_frames=1,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            for index, amplitude in enumerate((0.001, 0.002)):
                websocket.send_json(audio_event(f"limited-{index}", amplitude=amplitude))
                receive_until(websocket, "voicechat.metrics")
            websocket.send_json(audio_event("limited-trigger", amplitude=0.1))
            error, events = receive_until(websocket, "error")
            self.assertEqual(error["error"]["code"], "session_position_limit")
            closed, more_events = receive_until(websocket, "session.closed")
            self.assertEqual(closed["reason"], "session_position_limit")
            self.assertEqual(
                [event["type"] for event in events + more_events].count("voicechat.metrics"),
                1,
            )

        self.assertEqual(len(engine.process_calls), 1)
        np.testing.assert_allclose(
            engine.process_calls[0]["samples"].mean(), 0.001, atol=1 / 32768
        )

    def test_typed_input_uses_server_owned_audio_path_and_tags_transcript(self) -> None:
        temporary, engine, client = self.make_client(typed_input=True)
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            created = websocket.receive_json()
            self.assertTrue(created["capabilities"]["typed_input"])
            self.configure(websocket)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-1", "text": "Exact text"}
            )
            accepted = websocket.receive_json()
            self.assertEqual(accepted["type"], "input_text.accepted")
            observed = []
            for _ in range(100):
                event = websocket.receive_json()
                observed.append(event)
                if event["type"] == "input_text.injection_finished":
                    break
            self.assertEqual(observed[-1]["disposition"], "completed")
            transcripts = [
                event
                for event in observed
                if event["type"].startswith("conversation.item.input_audio_transcription")
            ]
            self.assertTrue(transcripts)
            self.assertTrue(all(event["source"] == "typed" for event in transcripts))
            self.assertTrue(all(event["job_id"] == "typed-1" for event in transcripts))
            self.assertGreater(engine.frame_index, 0)
            self.assertEqual(engine.start_count, 1)

    def test_typed_input_uses_external_eou_without_legacy_closure_silence(self) -> None:
        temporary, engine, client = self.make_client(
            typed_input=True,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            created = websocket.receive_json()
            self.assertEqual(
                created["capabilities"]["input_turn_detection"],
                "client_smart_turn_v1",
            )
            self.configure(websocket)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-eou", "text": "Exact"}
            )
            accepted = websocket.receive_json()
            self.assertEqual(accepted["type"], "input_text.accepted")
            observed = []
            for _ in range(100):
                event = websocket.receive_json()
                observed.append(event)
                if event["type"] == "input_text.injection_finished":
                    break

        event_types = [event["type"] for event in observed]
        self.assertEqual(observed[-1]["disposition"], "completed")
        self.assertLess(
            event_types.index("input_text.injection_started"),
            event_types.index("input_audio_buffer.speech_started"),
        )
        self.assertLess(
            event_types.index("input_audio_buffer.speech_started"),
            event_types.index("voicechat.metrics"),
        )
        self.assertIn("input_audio_buffer.speech_stopped", event_types)
        self.assertEqual(engine.user_eou_requests, 1)
        self.assertEqual(len(engine.process_calls), 4)

    def test_typed_replacement_before_first_mutation_never_injects_old_audio(self) -> None:
        pocket = ReplacingFakePocketWorker()
        temporary, engine, client = self.make_client(
            pocket_worker=pocket,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-old", "text": "Old"}
            )
            self.assertEqual(websocket.receive_json()["type"], "input_text.accepted")
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-new", "text": "New"}
            )
            observed = []
            for _ in range(120):
                event = websocket.receive_json()
                observed.append(event)
                if (
                    event["type"] == "input_text.injection_finished"
                    and event["job_id"] == "typed-new"
                ):
                    break

        old_events = [event for event in observed if event.get("job_id") == "typed-old"]
        new_events = [event for event in observed if event.get("job_id") == "typed-new"]
        self.assertEqual(old_events[-1]["disposition"], "replaced")
        self.assertNotIn(
            "input_text.injection_started",
            [event["type"] for event in old_events],
        )
        self.assertIn("input_text.injection_started", [event["type"] for event in new_events])
        self.assertTrue(engine.process_calls)

    def test_buggy_typed_request_during_open_microphone_turn_is_rejected(self) -> None:
        temporary, engine, client = self.make_client(
            typed_input=True,
            client_turn_detection=True,
        )
        self.addCleanup(temporary.cleanup)
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            self.configure(websocket)
            websocket.send_json(
                {
                    "type": "input_audio_buffer.turn_start",
                    "event_id": "start-before-typed",
                    "client_turn_id": 1,
                }
            )
            receive_until(websocket, "input_audio_buffer.turn_started")
            websocket.send_json(
                {"type": "input_text.request", "job_id": "typed-race", "text": "Race"}
            )
            rejected = websocket.receive_json()

        self.assertEqual(rejected["type"], "input_text.rejected")
        self.assertEqual(rejected["reason"], "input_turn_active")
        self.assertFalse(engine.process_calls)

    def test_local_mode_has_no_auth_and_busy_is_explicit(self) -> None:
        temporary, _engine, client = self.make_client()
        self.addCleanup(temporary.cleanup)
        self.assertFalse(client.get("/health").json()["auth_required"])
        with client.websocket_connect("/v1/realtime") as first:
            first.receive_json()
            with client.websocket_connect("/v1/realtime") as second:
                busy = second.receive_json()
                self.assertEqual(busy["error"]["code"], "server_busy")
                with self.assertRaises(WebSocketDisconnect) as raised:
                    second.receive_json()
                self.assertEqual(raised.exception.code, 1013)


if __name__ == "__main__":
    unittest.main()
