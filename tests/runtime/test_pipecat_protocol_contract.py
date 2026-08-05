from __future__ import annotations

import unittest
from types import SimpleNamespace

from nemotron_voicechat_runtime.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RealtimeProtocolSession,
    protocol_capabilities,
)


def result(
    *,
    user_text: str = "",
    assistant_delta: str = "",
    speech_confirmed: bool = False,
    control: str | None = None,
    boundary_event: str | None = None,
    boundary_reason: str | None = None,
    decoded_token_count: int = 0,
):
    return SimpleNamespace(
        user_text=user_text,
        assistant_delta=assistant_delta,
        turn_state={
            "rnnt": {
                "speech_confirmed": speech_confirmed,
                "decoded_token_count": decoded_token_count,
            },
            "agent_control": control,
            "response_boundary": {
                "event": boundary_event,
                "boundary_reason": boundary_reason,
            },
        },
    )


class RealtimeProtocolContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = RealtimeProtocolSession(
            "session-test",
            protocol_capabilities(
                function_call_timeout_seconds=30.0,
                max_session_model_frames=12_000,
            ),
        )

    def events(self, value, audio: bytes = b""):
        return self.protocol.step_events(
            value,
            output_pcm=audio,
            audio_delivered=bool(audio),
            output_sample_rate=22_050,
        )

    def test_handshake_is_fixed_version_three(self) -> None:
        created = self.protocol.session_created({"input_audio_format": "pcm16"})
        self.assertEqual(created["protocol"], {"name": PROTOCOL_NAME, "version": 3})
        self.assertEqual(PROTOCOL_VERSION, 3)
        self.assertEqual(created["session"]["id"], "session-test")
        self.assertFalse(created["capabilities"]["binary_audio"])

        updated = self.protocol.session_updated({"instructions": "brief", "tools": []})
        self.assertEqual(updated["session"]["protocol_version"], 3)
        self.assertEqual(updated["session"]["id"], "session-test")

    def test_full_turn_lifecycle_is_ordered_and_scoped(self) -> None:
        first = self.events(result(user_text="hello", speech_confirmed=True))
        self.assertEqual(
            [event["type"] for event in first],
            [
                "input_audio_buffer.speech_started",
                "conversation.item.input_audio_transcription.delta",
            ],
        )
        turn_id = first[0]["turn_id"]
        self.assertEqual(first[1]["delta"], "hello")

        opened = self.events(
            result(
                user_text="hello there",
                control="agent_bos",
                boundary_event="start",
            )
        )
        self.assertEqual(
            [event["type"] for event in opened],
            [
                "conversation.item.input_audio_transcription.delta",
                "input_audio_buffer.speech_stopped",
                "conversation.item.input_audio_transcription.completed",
                "response.created",
            ],
        )
        self.assertEqual(opened[0]["delta"], " there")
        self.assertEqual(opened[2]["transcript"], "hello there")
        self.assertEqual(opened[3]["turn_id"], turn_id)
        response_id = opened[3]["response_id"]

        content = self.events(result(assistant_delta="Hi!"), b"\x01\x00" * 10)
        self.assertEqual(
            [event["type"] for event in content],
            ["response.output_text.delta", "response.output_audio.delta"],
        )
        self.assertTrue(all(event["response_id"] == response_id for event in content))

        done = self.events(result(boundary_event="end", boundary_reason="post_eos_audio_silence"))
        self.assertEqual(
            [event["type"] for event in done],
            [
                "response.output_text.done",
                "response.output_audio.done",
                "response.done",
            ],
        )
        self.assertEqual(done[0]["text"], "Hi!")
        self.assertFalse(done[1]["empty"])
        self.assertEqual(done[2]["reason"], "post_eos_audio_silence")
        self.assertEqual(len({event["event_id"] for event in first + opened + content + done}), 11)

    def test_typed_input_event_family_is_bounded_and_correlated(self) -> None:
        accepted = self.protocol.typed_input_accepted("job-1")
        started = self.protocol.typed_input_started("job-1")
        finished = self.protocol.typed_input_finished("job-1", "completed")
        self.assertEqual(
            [accepted["type"], started["type"], finished["type"]],
            [
                "input_text.accepted",
                "input_text.injection_started",
                "input_text.injection_finished",
            ],
        )
        self.assertEqual(finished["disposition"], "completed")
        with self.assertRaisesRegex(ValueError, "invalid typed-input disposition"):
            self.protocol.typed_input_finished("job-1", "unknown")

    def test_second_turn_transcript_is_not_session_cumulative(self) -> None:
        self.events(result(user_text="first", speech_confirmed=True))
        self.events(result(user_text="first", control="agent_bos", boundary_event="start"))
        self.events(result(boundary_event="end"))

        second = self.events(result(user_text="second", speech_confirmed=True))
        self.assertEqual(second[1]["delta"], "second")
        self.assertEqual(second[1]["transcript"], "second")

    def test_repeated_prefix_is_not_stripped_from_new_turn(self) -> None:
        self.events(result(user_text="yes", speech_confirmed=True, decoded_token_count=1))
        self.events(result(user_text="yes", control="agent_bos", boundary_event="start"))
        self.events(result(boundary_event="end"))

        second = self.events(
            result(
                user_text="yes please",
                speech_confirmed=True,
                decoded_token_count=2,
            )
        )
        self.assertEqual(second[1]["transcript"], "yes please")

    def test_response_done_keeps_owning_turn_id_during_barge_in(self) -> None:
        first = self.events(result(user_text="first", speech_confirmed=True, decoded_token_count=1))
        first_turn_id = first[0]["turn_id"]
        opened = self.events(result(user_text="first", control="agent_bos", boundary_event="start"))
        response_id = opened[-1]["response_id"]

        barge_in = self.events(
            result(user_text="interrupt", speech_confirmed=True, decoded_token_count=1)
        )
        second_turn_id = barge_in[0]["turn_id"]
        self.assertNotEqual(first_turn_id, second_turn_id)

        done = self.events(result(boundary_event="end"))
        self.assertTrue(all(event["turn_id"] == first_turn_id for event in done))
        self.assertTrue(all(event["response_id"] == response_id for event in done))

    def test_silent_response_has_outer_bracket_without_audio(self) -> None:
        opened = self.events(result(control="agent_bos", boundary_event="start"))
        self.assertEqual([event["type"] for event in opened], ["response.created"])
        done = self.events(result(boundary_event="end"))
        self.assertTrue(done[0]["empty"])
        self.assertTrue(done[1]["empty"])

    def test_function_call_is_bracketed_and_has_stable_ids(self) -> None:
        events = self.protocol.function_call_events(
            call_id="call-1",
            name="lookup",
            arguments='{"q":"x"}',
        )
        self.assertEqual(
            [event["type"] for event in events],
            ["response.created", "response.function_call_arguments.done"],
        )
        self.assertEqual(events[0]["response_id"], events[1]["response_id"])
        self.assertEqual(events[1]["call_id"], "call-1")

        failed = self.protocol.function_call_failed_event(
            call_id="call-1", reason="function_call_output_timeout"
        )
        self.assertEqual(failed["type"], "response.function_call.failed")
        self.assertEqual(failed["response_id"], events[0]["response_id"])
        self.assertEqual(failed["reason"], "function_call_output_timeout")

    def test_error_is_typed_and_correlated(self) -> None:
        event = self.protocol.error(
            "invalid_audio",
            "odd byte count",
            client_event_id="client-3",
        )
        self.assertEqual(event["error"]["code"], "invalid_audio")
        self.assertFalse(event["error"]["fatal"])
        self.assertEqual(event["error"]["client_event_id"], "client-3")


if __name__ == "__main__":
    unittest.main()
