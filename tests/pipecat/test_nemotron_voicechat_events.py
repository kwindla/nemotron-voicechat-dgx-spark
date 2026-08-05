"""Strict wire-contract tests for the Pipecat client."""

import base64
import json

import pytest

from nemotron_voicechat_pipecat import events


def test_session_update_selects_only_protocol_v3():
    update = events.session_update(instructions="Be useful", tools=[])
    assert update["type"] == "session.update"
    assert update["session"] == {
        "protocol_version": 3,
        "instructions": "Be useful",
        "tools": [],
    }


def test_audio_append_is_explicit_pcm16_mono_16khz():
    message = events.audio_append(b"\x01\x02")
    assert message["encoding"] == "pcm16"
    assert message["sample_rate"] == 16_000
    assert message["channels"] == 1
    assert base64.b64decode(message["audio"]) == b"\x01\x02"


def test_typed_input_request_preserves_exact_text_and_client_job_id():
    message = events.typed_input_request(text="  exact  ", job_id="job-1")
    assert message["type"] == "input_text.request"
    assert message["text"] == "  exact  "
    assert message["job_id"] == "job-1"


def test_binary_server_event_is_rejected():
    with pytest.raises(events.ProtocolError, match="JSON text"):
        events.parse_server_event(b"{}")


def test_malformed_output_audio_is_rejected():
    with pytest.raises(events.ProtocolError, match="22050"):
        events.decode_audio_delta(
            {
                "encoding": "pcm16",
                "sample_rate": 24_000,
                "channels": 1,
                "delta": "AA==",
            }
        )


def test_function_result_is_not_double_json_encoded():
    message = events.function_call_output("call_1", '{"value":18}')
    encoded = json.dumps(message)
    decoded = json.loads(encoded)
    assert decoded["item"]["output"] == '{"value":18}'


def test_tool_names_that_collide_after_server_normalization_are_rejected():
    tools = [
        {"name": "get-weather", "parameters": {}},
        {"name": "get_weather", "parameters": {}},
    ]
    with pytest.raises(events.ProtocolError, match="after normalization"):
        events.session_update(instructions=None, tools=tools)


def test_more_than_32_tools_are_rejected_before_handshake():
    tools = [{"name": f"tool_{index}", "parameters": {}} for index in range(33)]
    with pytest.raises(events.ProtocolError, match="at most 32"):
        events.session_update(instructions=None, tools=tools)
