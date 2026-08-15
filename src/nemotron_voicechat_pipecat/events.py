"""Wire helpers for Voicechat realtime protocol version 3.

The server owns all turn and response decisions.  This module deliberately
contains no compatibility aliases for the historical browser protocol.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from typing import Any

PROTOCOL_NAME = "voicechat.realtime"
PROTOCOL_VERSION = 3
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")


class ProtocolError(ValueError):
    """Raised when a server event violates the negotiated contract."""


def event_id() -> str:
    """Return a unique client event ID."""
    return f"client_evt_{uuid.uuid4().hex}"


def session_update(
    *,
    instructions: str | None,
    tools: list[dict[str, Any]],
    function_output_model_output: str | None = None,
) -> dict[str, Any]:
    """Build the one required pre-audio session update."""
    validate_tools(tools)
    session: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "instructions": instructions,
        "tools": tools,
    }
    if function_output_model_output is not None:
        session["capabilities"] = {
            "function_output_model_output": function_output_model_output
        }
    return {
        "type": "session.update",
        "event_id": event_id(),
        "session": session,
    }


def validate_tools(tools: list[dict[str, Any]]) -> None:
    """Fail before handshake on server-invalid or colliding tool schemas."""
    if len(tools) > 32:
        raise ProtocolError("Voicechat supports at most 32 functions")
    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ProtocolError(f"tool {index} must be an object")
        function = tool.get("function", tool)
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise ProtocolError(f"tool {index} has an invalid function name")
        normalized = name.replace("-", "_")
        if normalized in seen:
            raise ProtocolError(f"duplicate tool name after normalization: {name}")
        seen.add(normalized)


def audio_append(audio: bytes) -> dict[str, Any]:
    """Build one PCM16 mono 16 kHz input event."""
    return {
        "type": "input_audio_buffer.append",
        "event_id": event_id(),
        "encoding": "pcm16",
        "sample_rate": INPUT_SAMPLE_RATE,
        "channels": 1,
        "audio": base64.b64encode(audio).decode("ascii"),
    }


def audio_turn_start(client_turn_id: int) -> dict[str, Any]:
    """Open one client-detected microphone turn."""
    if client_turn_id < 1:
        raise ValueError("client_turn_id must be positive")
    return {
        "type": "input_audio_buffer.turn_start",
        "event_id": event_id(),
        "client_turn_id": client_turn_id,
    }


def audio_commit(
    client_turn_id: int, *, diagnostics: dict[str, int | float] | None = None
) -> dict[str, Any]:
    """Commit one client-detected microphone turn."""
    if client_turn_id < 1:
        raise ValueError("client_turn_id must be positive")
    event: dict[str, Any] = {
        "type": "input_audio_buffer.commit",
        "event_id": event_id(),
        "client_turn_id": client_turn_id,
    }
    if diagnostics is not None:
        event["diagnostics"] = {"schema": 1, **diagnostics}
    return event


def typed_input_request(*, text: str, job_id: str) -> dict[str, Any]:
    """Build one server-owned typed-input request."""

    return {
        "type": "input_text.request",
        "event_id": event_id(),
        "job_id": job_id,
        "text": text,
    }


def function_call_output(
    call_id: str, output: str, *, model_output: str | None = None
) -> dict[str, Any]:
    """Build a terminal function result event."""
    item = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }
    if model_output is not None:
        item["model_output"] = model_output
    return {
        "type": "conversation.item.create",
        "event_id": event_id(),
        "item": item,
    }


def session_stop() -> dict[str, Any]:
    """Build the permanent graceful-close event."""
    return {"type": "session.stop", "event_id": event_id()}


def parse_server_event(message: str | bytes) -> dict[str, Any]:
    """Parse and minimally validate one server event."""
    if isinstance(message, bytes):
        raise ProtocolError("Voicechat v3 server events must be JSON text")
    try:
        event = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid server JSON: {exc}") from exc
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise ProtocolError("server event requires a string type")
    return event


def decode_audio_delta(event: dict[str, Any]) -> bytes:
    """Validate and decode one output-audio delta."""
    if event.get("encoding") != "pcm16":
        raise ProtocolError("response audio encoding must be pcm16")
    if event.get("sample_rate") != OUTPUT_SAMPLE_RATE or event.get("channels") != 1:
        raise ProtocolError("response audio must be mono PCM16 at 22050 Hz")
    payload = event.get("delta")
    if not isinstance(payload, str):
        raise ProtocolError("response audio delta must be base64 text")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("response audio delta is invalid base64") from exc
