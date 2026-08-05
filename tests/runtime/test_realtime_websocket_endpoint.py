from __future__ import annotations

import base64
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nemotron_voicechat_runtime.pocket_controller import PocketSynthesis
from nemotron_voicechat_runtime.server import (
    FRAME_SAMPLES,
    INPUT_SAMPLE_RATE,
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


class FakeEngine:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(checkpoint_provenance={"kind": "test"})
        self.frame_index = 0
        self.started = False
        self.system_prompt = ""
        self.tools = {}
        self.abort_count = 0

    def configure_external_tools(self, tools) -> None:
        self.tools = tools

    def start(self, system_prompt="") -> None:
        self.started = True
        self.system_prompt = system_prompt

    def process(self, samples, *, is_last=False, input_active=None):
        index = self.frame_index
        self.frame_index += 1
        audio = np.zeros(1764, dtype=np.float32)
        user_text = ""
        assistant_delta = ""
        rnnt = {"speech_confirmed": False}
        control = "pad"
        boundary = {"event": None, "phase": "idle"}
        if index == 0:
            user_text = "hello"
            rnnt["speech_confirmed"] = True
        elif index == 1:
            user_text = "hello"
            assistant_delta = "Hello."
            control = "agent_bos"
            boundary = {"event": "start", "phase": "responding"}
            audio[:] = 0.05
        elif index == 2:
            user_text = "hello"
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
            },
        )

    def abort(self) -> None:
        self.abort_count += 1
        self.started = False


def audio_event(event_id: str = "audio-1") -> dict:
    pcm = np.full(FRAME_SAMPLES, 0.1, dtype="<f4")
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
    def make_client(self, *, typed_input=False):
        temporary = tempfile.TemporaryDirectory()
        engine = FakeEngine()
        pocket = FakePocketWorker() if typed_input else None
        app = create_app(engine, pocket_worker=pocket)
        return temporary, engine, TestClient(app)

    def configure(self, websocket):
        websocket.send_json(
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
        updated = websocket.receive_json()
        self.assertEqual(updated["type"], "session.updated")
        return updated

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
