#!/usr/bin/env python3
"""Run the preregistered direct-WebSocket tool-call latency cells."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Callable

import websockets

FRAME_SECONDS = 0.08
INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_BYTES = round(INPUT_RATE * FRAME_SECONDS) * 2
SPEECH_DBFS = -40.0
PRIMARY_ORDER = (
    (20, 48, 72, 120),
    (48, 72, 120, 20),
    (72, 120, 20, 48),
    (120, 20, 48, 72),
    (20, 48, 72, 120),
    (120, 72, 48, 20),
    (20, 120, 72, 48),
    (48, 20, 120, 72),
    (72, 48, 20, 120),
    (120, 72, 48, 20),
)

SHORT_TOOL = {
    "name": "get_benchmark_word",
    "description": "Return the fresh benchmark word.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
LONG_TOOL = {
    "name": "retrieve_the_fresh_benchmark_verification_word_for_the_current_request",
    "description": (
        "Return the single fresh ordinary English verification word for this request. "
        "Use this tool whenever the user asks for the benchmark verification word."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_kind": {
                "type": "string",
                "description": "The kind of fresh verification word requested by the user.",
            }
        },
        "required": [],
    },
}

FIRST_CHAIN_TOOL = {
    "name": "get_first_benchmark_word",
    "description": "Return the first fresh benchmark word.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
SECOND_CHAIN_TOOL = {
    "name": "get_second_benchmark_word",
    "description": "Return the second fresh benchmark word.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
SAFETY_SCENARIOS = (
    "no_tool",
    "chained_two_call",
    "call_emission_interruption",
    "result_injection_interruption",
)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_dbfs(payload: bytes) -> float:
    if not payload:
        return -120.0
    samples = memoryview(payload).cast("h")
    if not samples:
        return -120.0
    mean_square = sum(float(sample) * float(sample) for sample in samples) / len(samples)
    if mean_square <= 0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(math.sqrt(mean_square) / 32768.0))


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(OUTPUT_RATE)
        stream.writeframes(payload)


def phase_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    fc = (event.get("turn_state") or {}).get("function_calling") or {}
    rows = fc.get("async_benchmark_phases") or []
    return [row for row in rows if isinstance(row, dict)]


def tool_wait_row(event: dict[str, Any]) -> dict[str, Any] | None:
    fc = (event.get("turn_state") or {}).get("function_calling") or {}
    value = fc.get("tool_wait_evidence")
    return dict(value) if isinstance(value, dict) else None


def evaluate_tool_wait_gate(
    cell: dict[str, Any],
    *,
    expected_policy: str,
    expect_reminder: bool,
    require_reminder_transport: bool = True,
) -> dict[str, Any]:
    """Re-derive the reminder/fast-path gate from retained cell evidence."""
    evidence = cell.get("tool_wait_evidence") or {}
    witness = cell.get("reminder_transport_witness")
    reminder_rendered = bool(
        evidence.get("reminder_samples", 0) > 0
        and evidence.get("reminder_started_monotonic_s") is not None
        and evidence.get("reminder_ended_monotonic_s") is not None
    )
    reminder_transported = bool(
        isinstance(witness, dict)
        and witness.get("generated_samples", 0) > 0
        and witness.get("received_samples", 0) > 0
        and witness.get("audible_chunks", 0) > 0
    )
    fast_delay = evidence.get("result_available_to_phase2_ms")
    fast_delay_valid = bool(
        expect_reminder
        or (
            not isinstance(fast_delay, bool)
            and isinstance(fast_delay, (int, float))
            and math.isfinite(float(fast_delay))
            and float(fast_delay) >= 0.0
        )
    )
    gate = {
        "expected_policy": expected_policy,
        "observed_policy": evidence.get("policy"),
        "policy_matched": evidence.get("policy") == expected_policy,
        "expected_reminder": expect_reminder,
        "reminder_rendered": reminder_rendered,
        "reminder_rendering_matched": reminder_rendered is expect_reminder,
        "require_reminder_transport": require_reminder_transport,
        "reminder_transported": reminder_transported,
        "reminder_transport_matched": bool(
            not require_reminder_transport or reminder_transported is expect_reminder
        ),
        "fast_result_delay_valid": fast_delay_valid,
    }
    gate["passed"] = bool(
        gate["policy_matched"]
        and gate["reminder_rendering_matched"]
        and gate["reminder_transport_matched"]
        and gate["fast_result_delay_valid"]
    )
    return gate


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        raise ValueError("values must be nonempty and percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def session_model_output_capability(use_model_output: bool) -> dict[str, str] | None:
    return (
        {"function_output_model_output": "client_authored_v1"}
        if use_model_output
        else None
    )


def function_output_item(
    *, call_id: str, payload: dict[str, Any], use_model_output: bool
) -> dict[str, Any]:
    item = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": payload["output"],
    }
    if use_model_output:
        model_output = payload.get("model_output")
        if not isinstance(model_output, str) or not model_output.strip():
            raise ValueError("payload lacks a nonempty model_output")
        item["model_output"] = model_output
    return item


def expected_injection_tokens(payload: dict[str, Any], use_model_output: bool) -> int:
    key = "model_output_tokens" if use_model_output else "realized_tokens"
    value = payload.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"payload lacks valid {key}")
    return value


def tool_response_position_contract(
    phases: list[dict[str, Any]],
    payload: dict[str, Any],
    use_model_output: bool,
) -> tuple[int, bool]:
    """Require one model-owned position after every injected result token."""

    expected = expected_injection_tokens(payload, use_model_output) + 1
    response_phase = next(
        (row for row in phases if row.get("phase") == "tool_response"), None
    )
    return expected, bool(
        response_phase is not None and response_phase.get("positions_count") == expected
    )


async def send_delayed_tool_output(
    websocket: Any,
    *,
    ordinal: int,
    call_id: str,
    payload: dict[str, Any],
    use_model_output: bool,
    delay_ms: float,
) -> float:
    """Send a tool result after a delay without blocking the receive loop."""
    if not math.isfinite(delay_ms) or not 0.0 <= delay_ms <= 30000.0:
        raise ValueError("delay_ms must be between zero and 30000")
    if delay_ms > 0.0:
        await asyncio.sleep(delay_ms / 1000.0)
    sent = time.monotonic()
    await websocket.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "event_id": f"benchmark-output-{ordinal}",
                "item": function_output_item(
                    call_id=call_id,
                    payload=payload,
                    use_model_output=use_model_output,
                ),
            }
        )
    )
    return sent


async def silence_clock(websocket: Any, stop: asyncio.Event, prefix: str) -> None:
    started = time.monotonic()
    index = 0
    silence = base64.b64encode(bytes(FRAME_BYTES)).decode()
    while not stop.is_set():
        deadline = started + index * FRAME_SECONDS
        delay = deadline - time.monotonic()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                break
            except TimeoutError:
                pass
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{prefix}-silence-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": silence,
                }
            )
        )
        index += 1


async def send_speech_carrier(
    websocket: Any,
    payload: bytes,
    *,
    prefix: str,
    client_turn_id: int,
    real_time: bool,
    on_turn_start_sent: Callable[[float], None] | None = None,
) -> float:
    if not payload or len(payload) % 2:
        raise ValueError("speech carrier must be nonempty PCM16")
    padded = payload + bytes((-len(payload)) % FRAME_BYTES)
    frames = [
        padded[offset : offset + FRAME_BYTES]
        for offset in range(0, len(padded), FRAME_BYTES)
    ]
    frames.extend([bytes(FRAME_BYTES)] * 3)
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.turn_start",
                "event_id": f"{prefix}-start",
                "client_turn_id": client_turn_id,
            }
        )
    )
    turn_start_sent = time.monotonic()
    if on_turn_start_sent is not None:
        on_turn_start_sent(turn_start_sent)
    started = time.monotonic()
    for index, frame in enumerate(frames):
        if real_time:
            delay = started + index * FRAME_SECONDS - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{prefix}-audio-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": base64.b64encode(frame).decode(),
                }
            )
        )
    commit_monotonic = time.monotonic()
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.commit",
                "event_id": f"{prefix}-commit",
                "client_turn_id": client_turn_id,
            }
        )
    )
    return commit_monotonic


async def send_speech_carrier_with_continuation_clock(
    websocket: Any,
    payload: bytes,
    *,
    prefix: str,
    client_turn_id: int,
    real_time: bool,
    stop: asyncio.Event,
    carrier_done: asyncio.Event,
    on_turn_start_sent: Callable[[float], None] | None = None,
) -> float:
    """Send one committed speech turn, then keep advancing the live model clock."""
    commit_monotonic = await send_speech_carrier(
        websocket,
        payload,
        prefix=prefix,
        client_turn_id=client_turn_id,
        real_time=real_time,
        on_turn_start_sent=on_turn_start_sent,
    )
    carrier_done.set()
    await silence_clock(websocket, stop, f"{prefix}-continuation")
    return commit_monotonic


async def run_call(
    *,
    url: str,
    output_dir: Path,
    ordinal: int,
    block: int,
    schema_name: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    speech_pcm: bytes | None = None,
    speech_text: str | None = None,
    real_time_input: bool = True,
    tool_delay_ms: float = 0.0,
    use_model_output: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(tool_delay_ms) or not 0.0 <= tool_delay_ms <= 30000.0:
        raise ValueError("tool_delay_ms must be between zero and 30000")
    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "events.jsonl"
    tool = SHORT_TOOL if schema_name == "short" else LONG_TOOL
    prompt = f"Use {tool['name']} now, then state the exact fresh word from its result."
    session_id = None
    response_id = None
    turn_id = None
    call_id = None
    text = ""
    audio_parts: list[bytes] = []
    phases: list[dict[str, Any]] = []
    tool_wait_rows: list[dict[str, Any]] = []
    audio_observations: list[dict[str, Any]] = []
    function_output_tokens = None
    client_commit_monotonic = None
    function_received_monotonic = None
    output_sent_monotonic = None
    first_text_monotonic = None
    first_audio_monotonic = None
    first_audible_monotonic = None
    response_done_monotonic = None
    errors: list[dict[str, Any]] = []
    input_transcript = None
    stop_clock = asyncio.Event()
    clock: asyncio.Task[None] | None = None
    tool_output_task: asyncio.Task[None] | None = None

    async def send_tool_output(websocket: Any, current_call_id: str) -> None:
        nonlocal output_sent_monotonic
        output_sent_monotonic = await send_delayed_tool_output(
            websocket,
            ordinal=ordinal,
            call_id=current_call_id,
            payload=payload,
            use_model_output=use_model_output,
            delay_ms=tool_delay_ms,
        )
    with events_path.open("w", encoding="utf-8") as events:
        async with websockets.connect(
            url,
            open_timeout=30,
            ping_interval=None,
            max_size=16 * 1024 * 1024,
            write_limit=64 * 1024 * 1024,
        ) as websocket:
            created = json.loads(await websocket.recv())
            created["client_received_monotonic_s"] = time.monotonic()
            events.write(json.dumps(created, sort_keys=True) + "\n")
            session_id = (created.get("session") or {}).get("id")
            capabilities = created.get("capabilities") or {}
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError("server did not negotiate strict protocol v3")
            if capabilities.get("typed_input") is not True:
                raise RuntimeError("server did not advertise typed input")
            if use_model_output:
                model_output_capability = capabilities.get(
                    "function_output_model_output"
                )
                if not (
                    isinstance(model_output_capability, dict)
                    and model_output_capability.get("mode") == "client_authored_v1"
                ):
                    raise RuntimeError("server did not advertise concise tool results")
            session = {
                "protocol_version": 3,
                "instructions": (
                    "Use the requested tool. Then answer in one short sentence."
                ),
                "tools": [tool],
            }
            capabilities_request = session_model_output_capability(use_model_output)
            if capabilities_request is not None:
                session["capabilities"] = capabilities_request
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": f"benchmark-config-{ordinal}",
                        "session": session,
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                event["client_received_monotonic_s"] = time.monotonic()
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session update failed: {event}")
            job_id = f"benchmark-{block}-{ordinal}-{uuid.uuid4().hex[:8]}"
            if speech_pcm is None:
                client_commit_monotonic = time.monotonic()
                await websocket.send(
                    json.dumps(
                        {"type": "input_text.request", "job_id": job_id, "text": prompt}
                    )
                )
            else:
                client_commit_monotonic = await send_speech_carrier(
                    websocket,
                    speech_pcm,
                    prefix=job_id,
                    client_turn_id=ordinal,
                    real_time=real_time_input,
                )
                clock = asyncio.create_task(silence_clock(websocket, stop_clock, job_id))
            deadline = time.monotonic() + timeout_seconds
            try:
                while time.monotonic() < deadline:
                    event = json.loads(
                        await asyncio.wait_for(websocket.recv(), deadline - time.monotonic())
                    )
                    now = time.monotonic()
                    event["client_received_monotonic_s"] = now
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    kind = event.get("type")
                    if kind == "voicechat.metrics":
                        phases.extend(phase_rows(event))
                        wait_row = tool_wait_row(event)
                        if wait_row is not None:
                            tool_wait_rows.append(wait_row)
                    elif kind == "input_text.injection_finished" and event.get("job_id") == job_id:
                        if clock is None:
                            clock = asyncio.create_task(
                                silence_clock(websocket, stop_clock, job_id)
                            )
                    elif kind == "conversation.item.input_audio_transcription.completed":
                        turn_id = event.get("turn_id")
                        input_transcript = event.get("transcript")
                    elif kind == "response.created":
                        response_id = event.get("response_id")
                        turn_id = event.get("turn_id")
                    elif kind == "response.function_call_arguments.done":
                        function_received_monotonic = now
                        call_id = event.get("call_id")
                        if not isinstance(call_id, str) or tool_output_task is not None:
                            raise RuntimeError("benchmark received an invalid duplicate tool call")
                        tool_output_task = asyncio.create_task(
                            send_tool_output(websocket, call_id),
                            name=f"benchmark-tool-output-{ordinal}",
                        )
                    elif kind == "conversation.item.function_call_output.applied":
                        function_output_tokens = event.get("injection_tokens")
                    elif kind == "response.output_text.delta":
                        if first_text_monotonic is None and event.get("delta"):
                            first_text_monotonic = now
                        text += str(event.get("delta") or "")
                    elif kind == "response.output_audio.delta":
                        chunk = base64.b64decode(event.get("delta") or "")
                        audio_observations.append(
                            {
                                "received_monotonic_s": now,
                                "bytes": len(chunk),
                                "samples": len(chunk) // 2,
                                "dbfs": pcm_dbfs(chunk),
                            }
                        )
                        if first_audio_monotonic is None and chunk:
                            first_audio_monotonic = now
                        if first_audible_monotonic is None and pcm_dbfs(chunk) > SPEECH_DBFS:
                            first_audible_monotonic = now
                        audio_parts.append(chunk)
                    elif kind == "error":
                        errors.append(event)
                    elif kind == "response.done":
                        response_done_monotonic = now
                        break
                else:
                    raise TimeoutError("tool latency call timed out")
            finally:
                stop_clock.set()
                if clock is not None:
                    await clock
                if tool_output_task is not None:
                    await tool_output_task
            await websocket.send(json.dumps({"type": "session.stop"}))
            while True:
                event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                event["client_received_monotonic_s"] = time.monotonic()
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.closed":
                    break
                if event.get("type") == "voicechat.metrics":
                    phases.extend(phase_rows(event))
                if event.get("type") == "error":
                    errors.append(event)
    audio = b"".join(audio_parts)
    write_wav(output_dir / "response.wav", audio)
    unique_phases = {
        (row.get("cycle_sotc_frame"), row.get("phase"), row.get("invocation")): row
        for row in phases
    }
    ordered_phases = sorted(
        unique_phases.values(), key=lambda row: (row.get("phase", 0), row.get("invocation", 0))
    )
    tool_wait = max(
        tool_wait_rows,
        key=lambda row: float(row.get("phase2_started_monotonic_s", 0.0)),
        default=None,
    )
    reminder_transport_witness = None
    if tool_wait is not None and tool_wait.get("reminder_started_monotonic_s") is not None:
        start = float(tool_wait["reminder_started_monotonic_s"])
        end = float(tool_wait["phase2_started_monotonic_s"]) + 1.0
        observed = [
            row for row in audio_observations if start <= row["received_monotonic_s"] <= end
        ]
        reminder_transport_witness = {
            "window_start_monotonic_s": start,
            "window_end_monotonic_s": end,
            "generated_samples": tool_wait["reminder_samples"],
            "received_chunks": len(observed),
            "received_bytes": sum(row["bytes"] for row in observed),
            "received_samples": sum(row["samples"] for row in observed),
            "audible_chunks": sum(row["dbfs"] > SPEECH_DBFS for row in observed),
            "peak_dbfs": max((row["dbfs"] for row in observed), default=-120.0),
        }
    expected_response_positions, response_positions_match = tool_response_position_contract(
        ordered_phases, payload, use_model_output
    )
    passed = bool(
        not errors
        and isinstance(call_id, str)
        and len(ordered_phases) == 2
        and {row.get("phase") for row in ordered_phases} == {"tool_call", "tool_response"}
        and all(row.get("positions_truncated") is False for row in ordered_phases)
        and response_positions_match
        and payload["verification_word"].casefold() in text.casefold()
        and first_audible_monotonic is not None
    )
    report = {
        "schema": 1,
        "kind": "tool_call_latency_cell",
        "passed": passed,
        "block": block,
        "ordinal": ordinal,
        "schema_name": schema_name,
        "input_modality": "typed" if speech_pcm is None else "speech",
        "input_real_time": real_time_input if speech_pcm is not None else None,
        "expected_input_text": speech_text if speech_pcm is not None else prompt,
        "input_transcript": input_transcript,
        "tool": tool,
        "prompt": prompt,
        "payload": payload,
        "tool_delay_ms": tool_delay_ms,
        "tool_wait_evidence": tool_wait,
        "reminder_transport_witness": reminder_transport_witness,
        "session_id": session_id,
        "turn_id": turn_id,
        "response_id": response_id,
        "call_id": call_id,
        "text": text,
        "audio_bytes": len(audio),
        "function_output_tokens_ack": function_output_tokens,
        "expected_tool_response_positions": expected_response_positions,
        "tool_response_positions_match": response_positions_match,
        "model_output_enabled": use_model_output,
        "phases": ordered_phases,
        "client_commit_monotonic_s": client_commit_monotonic,
        "function_received_monotonic_s": function_received_monotonic,
        "output_sent_monotonic_s": output_sent_monotonic,
        "first_text_monotonic_s": first_text_monotonic,
        "first_audio_monotonic_s": first_audio_monotonic,
        "first_audible_monotonic_s": first_audible_monotonic,
        "response_done_monotonic_s": response_done_monotonic,
        "errors": errors,
    }
    atomic_json(output_dir / "report.json", report)
    return report


async def run_same_session(
    *,
    url: str,
    output_dir: Path,
    payload: dict[str, Any],
    timeout_seconds: float,
    use_model_output: bool,
) -> list[dict[str, Any]]:
    """Run ten serialized calls in one session for context-drift evidence."""

    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "events.jsonl"
    reports: list[dict[str, Any]] = []
    stop_clock = asyncio.Event()
    clock: asyncio.Task[None] | None = None
    session_id = None
    with events_path.open("w", encoding="utf-8") as events:
        async with websockets.connect(
            url,
            open_timeout=30,
            ping_interval=None,
            max_size=16 * 1024 * 1024,
            write_limit=64 * 1024 * 1024,
        ) as websocket:
            created = json.loads(await websocket.recv())
            created["client_received_monotonic_s"] = time.monotonic()
            events.write(json.dumps(created, sort_keys=True) + "\n")
            session_id = (created.get("session") or {}).get("id")
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError("server did not negotiate strict protocol v3")
            if (created.get("capabilities") or {}).get("typed_input") is not True:
                raise RuntimeError("server did not advertise typed input")
            session = {
                "protocol_version": 3,
                "instructions": (
                    "Use the requested tool. Then answer in one short sentence."
                ),
                "tools": [SHORT_TOOL],
            }
            capabilities_request = session_model_output_capability(use_model_output)
            if capabilities_request is not None:
                session["capabilities"] = capabilities_request
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "benchmark-same-session-config",
                        "session": session,
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                event["client_received_monotonic_s"] = time.monotonic()
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session update failed: {event}")

            for ordinal in range(1, 11):
                prompt = (
                    f"Use {SHORT_TOOL['name']} now, then state the exact fresh word "
                    "from its result."
                )
                job_id = f"benchmark-same-{ordinal}-{uuid.uuid4().hex[:8]}"
                response_id = None
                turn_id = None
                call_id = None
                text = ""
                audio_parts: list[bytes] = []
                phases: list[dict[str, Any]] = []
                function_output_tokens = None
                client_commit_monotonic = time.monotonic()
                function_received_monotonic = None
                output_sent_monotonic = None
                first_text_monotonic = None
                first_audio_monotonic = None
                first_audible_monotonic = None
                response_done_monotonic = None
                errors: list[dict[str, Any]] = []
                await websocket.send(
                    json.dumps({"type": "input_text.request", "job_id": job_id, "text": prompt})
                )
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    event = json.loads(
                        await asyncio.wait_for(websocket.recv(), deadline - time.monotonic())
                    )
                    now = time.monotonic()
                    event["client_received_monotonic_s"] = now
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    kind = event.get("type")
                    if kind == "voicechat.metrics":
                        phases.extend(phase_rows(event))
                    elif kind == "input_text.injection_finished" and event.get("job_id") == job_id:
                        if clock is None:
                            clock = asyncio.create_task(
                                silence_clock(websocket, stop_clock, "benchmark-same")
                            )
                    elif kind == "conversation.item.input_audio_transcription.completed":
                        turn_id = event.get("turn_id")
                    elif kind == "response.created":
                        response_id = event.get("response_id")
                        turn_id = event.get("turn_id")
                    elif kind == "response.function_call_arguments.done":
                        function_received_monotonic = now
                        call_id = event.get("call_id")
                        output_sent_monotonic = time.monotonic()
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "event_id": f"benchmark-same-output-{ordinal}",
                                    "item": function_output_item(
                                        call_id=call_id,
                                        payload=payload,
                                        use_model_output=use_model_output,
                                    ),
                                }
                            )
                        )
                    elif kind == "conversation.item.function_call_output.applied":
                        function_output_tokens = event.get("injection_tokens")
                    elif kind == "response.output_text.delta":
                        if first_text_monotonic is None and event.get("delta"):
                            first_text_monotonic = now
                        text += str(event.get("delta") or "")
                    elif kind == "response.output_audio.delta":
                        chunk = base64.b64decode(event.get("delta") or "")
                        if first_audio_monotonic is None and chunk:
                            first_audio_monotonic = now
                        if first_audible_monotonic is None and pcm_dbfs(chunk) > SPEECH_DBFS:
                            first_audible_monotonic = now
                        audio_parts.append(chunk)
                    elif kind == "error":
                        errors.append(event)
                    elif kind == "response.done":
                        response_done_monotonic = now
                        break
                else:
                    raise TimeoutError(f"same-session call {ordinal} timed out")

                audio = b"".join(audio_parts)
                cell_dir = output_dir / f"cell-{ordinal:03d}"
                cell_dir.mkdir()
                write_wav(cell_dir / "response.wav", audio)
                unique_phases = {
                    (row.get("cycle_sotc_frame"), row.get("phase"), row.get("invocation")): row
                    for row in phases
                }
                cycle_frames = {
                    key[0] for key in unique_phases if isinstance(key[0], int)
                }
                current_cycle = max(cycle_frames) if cycle_frames else None
                ordered_phases = sorted(
                    (
                        row
                        for key, row in unique_phases.items()
                        if key[0] == current_cycle
                    ),
                    key=lambda row: (row.get("phase", ""), row.get("invocation", 0)),
                )
                expected_response_positions, response_positions_match = (
                    tool_response_position_contract(
                        ordered_phases, payload, use_model_output
                    )
                )
                passed = bool(
                    not errors
                    and isinstance(call_id, str)
                    and len(ordered_phases) == 2
                    and {row.get("phase") for row in ordered_phases}
                    == {"tool_call", "tool_response"}
                    and all(row.get("positions_truncated") is False for row in ordered_phases)
                    and response_positions_match
                    and payload["verification_word"].casefold() in text.casefold()
                    and first_audible_monotonic is not None
                )
                report = {
                    "schema": 1,
                    "kind": "tool_call_latency_cell",
                    "passed": passed,
                    "block": 1,
                    "ordinal": ordinal,
                    "schema_name": "short",
                    "same_session": True,
                    "tool": SHORT_TOOL,
                    "prompt": prompt,
                    "payload": payload,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "response_id": response_id,
                    "call_id": call_id,
                    "text": text,
                    "audio_bytes": len(audio),
                    "function_output_tokens_ack": function_output_tokens,
                    "expected_tool_response_positions": expected_response_positions,
                    "tool_response_positions_match": response_positions_match,
                    "model_output_enabled": use_model_output,
                    "phases": ordered_phases,
                    "client_commit_monotonic_s": client_commit_monotonic,
                    "function_received_monotonic_s": function_received_monotonic,
                    "output_sent_monotonic_s": output_sent_monotonic,
                    "first_text_monotonic_s": first_text_monotonic,
                    "first_audio_monotonic_s": first_audio_monotonic,
                    "first_audible_monotonic_s": first_audible_monotonic,
                    "response_done_monotonic_s": response_done_monotonic,
                    "errors": errors,
                }
                atomic_json(cell_dir / "report.json", report)
                reports.append(report)

            stop_clock.set()
            if clock is not None:
                await clock
            await websocket.send(json.dumps({"type": "session.stop"}))
            while True:
                event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                event["client_received_monotonic_s"] = time.monotonic()
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.closed":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session close failed: {event}")
    return reports


def _function_state(event: dict[str, Any]) -> dict[str, Any]:
    return (event.get("turn_state") or {}).get("function_calling") or {}


def _heartbeat_phase(function_state: dict[str, Any]) -> str | None:
    heartbeat = function_state.get("async_heartbeat") or {}
    loop = heartbeat.get("loop") or {}
    phase = loop.get("phase")
    return phase if isinstance(phase, str) else None


def _runtime_provenance(created: dict[str, Any]) -> dict[str, Any]:
    checkpoint = ((created.get("session") or {}).get("checkpoint") or {})
    provenance = checkpoint.get("runtime_provenance") or {}
    return provenance if isinstance(provenance, dict) else {}


async def require_safety_diagnostics(url: str) -> dict[str, Any]:
    """Fail before measured cells unless observed-state diagnostics are enabled."""

    async with websockets.connect(
        url,
        open_timeout=30,
        ping_interval=None,
        max_size=16 * 1024 * 1024,
    ) as websocket:
        created = json.loads(await websocket.recv())
    diagnostics = _runtime_provenance(created).get("diagnostics") or {}
    if diagnostics.get("fc_async_heartbeat") is not True:
        raise RuntimeError("safety baseline requires fc_async_heartbeat diagnostics")
    return _runtime_provenance(created)


async def wait_for_server_idle(url: str, timeout_seconds: float = 30.0) -> None:
    """Wait until prior fresh-session cleanup releases the single-client slot."""

    parsed = urllib.parse.urlsplit(url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    health_url = urllib.parse.urlunsplit((scheme, parsed.netloc, "/health", "", ""))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            def read_health() -> dict[str, Any]:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    return json.load(response)

            health = await asyncio.to_thread(read_health)
            if health.get("status") == "ready" and health.get("active_client") is False:
                return
        except (OSError, ValueError):
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("server did not become idle between safety cells")


def safety_cell_passed(report: dict[str, Any]) -> bool:
    """Independently rederive one frozen safety-cell verdict."""

    common = bool(
        not report.get("errors")
        and report.get("session_closed") is True
        and report.get("first_audible_monotonic_s") is not None
        and report.get("response_done_count", 0) >= 1
    )
    scenario = report.get("scenario")
    calls = report.get("function_calls") or []
    applied = report.get("function_outputs_applied") or []
    if scenario == "no_tool":
        return common and not calls and not applied and report.get("barge_trigger") is None
    if scenario == "chained_two_call":
        return bool(
            common
            and [call.get("name") for call in calls]
            == [FIRST_CHAIN_TOOL["name"], SECOND_CHAIN_TOOL["name"]]
            and len(applied) == 2
            and {item.get("call_id") for item in applied}
            == {item.get("call_id") for item in calls}
            and "river" in str(report.get("text", "")).casefold()
            and "meadow" in str(report.get("text", "")).casefold()
        )
    trigger = report.get("barge_trigger") or {}
    expected_reason = (
        "client_send_during_output_recovery"
        if scenario == "result_injection_interruption"
        else "client_send_during_call_emission"
    )
    trigger_reason = trigger.get("reason")
    trigger_attribution_valid = trigger_reason is None or bool(
        trigger_reason == expected_reason
        and isinstance(trigger.get("turn_start_sent_monotonic_s"), (int, float))
        and math.isfinite(float(trigger["turn_start_sent_monotonic_s"]))
    )
    barge_common = bool(
        common
        and trigger_attribution_valid
        and trigger.get("sent") is True
        and trigger.get("completed") is True
        and trigger.get("turn_started_ack") is True
        and trigger.get("committed_ack") is True
        and trigger.get("input_transcript_completed") is True
        and not trigger.get("turn_id_mismatch")
        and isinstance(trigger.get("turn_id"), str)
        and trigger.get("turn_id") in report.get("response_done_turn_ids", [])
    )
    if scenario == "call_emission_interruption":
        # Current qualified behavior permits sustained speech to cancel the
        # background call before its side effect is published.
        return barge_common and trigger.get("phase") == "tool_call" and not calls
    if scenario == "result_injection_interruption":
        # Strict v3 serializes input received during result recovery. The
        # already-published call/result must complete, then the queued barge turn
        # must receive its own response without a stale function cycle.
        return bool(
            barge_common
            and trigger.get("phase") == "tool_response"
            and len(calls) == 1
            and len(applied) == 1
            and applied[0].get("call_id") == calls[0].get("call_id")
            and report.get("response_done_count", 0) >= 2
        )
    return False


async def run_safety_cell(
    *,
    url: str,
    output_dir: Path,
    ordinal: int,
    replicate: int,
    scenario: str,
    payloads: dict[int, dict[str, Any]],
    timeout_seconds: float,
    barge_pcm: bytes,
    barge_text: str,
    use_model_output: bool,
) -> dict[str, Any]:
    """Run one fresh-session, observed-state-triggered safety cell."""

    if scenario not in SAFETY_SCENARIOS:
        raise ValueError(f"unknown safety scenario: {scenario}")
    if scenario.endswith("interruption") and (not barge_pcm or len(barge_pcm) % 2):
        raise ValueError("interruption safety cells require nonempty PCM16 barge audio")
    output_dir.mkdir(parents=True, exist_ok=False)
    events_path = output_dir / "events.jsonl"
    tools: list[dict[str, Any]]
    if scenario == "no_tool":
        tools = []
        prompt = "Answer briefly without a tool. Say the benchmark safety check is ready."
    elif scenario == "chained_two_call":
        tools = [FIRST_CHAIN_TOOL, SECOND_CHAIN_TOOL]
        prompt = (
            "Use the first benchmark word tool, then use the second benchmark word "
            "tool, then state both fresh words in one short sentence."
        )
    else:
        tools = [SHORT_TOOL]
        prompt = "Use get_benchmark_word now, then state its exact fresh word."

    errors: list[dict[str, Any]] = []
    function_calls: list[dict[str, Any]] = []
    outputs_applied: list[dict[str, Any]] = []
    response_ids: list[str] = []
    response_done_count = 0
    response_done_turn_ids: list[str] = []
    text = ""
    audio_parts: list[bytes] = []
    first_audible = None
    session_id = None
    session_closed = False
    stop_clock = asyncio.Event()
    barge_clock_stop = asyncio.Event()
    clock: asyncio.Task[None] | None = None
    barge_task: asyncio.Task[float] | None = None
    barge_carrier_done = asyncio.Event()
    trigger: dict[str, Any] | None = None
    typed_input_finished = False
    pending_call_trigger_frame: Any = None
    barge_client_turn_id = 2
    barge_start_event_id = f"safety-{ordinal}-barge-start"
    barge_commit_event_id = f"safety-{ordinal}-barge-commit"

    async def start_barge(websocket: Any, phase: str, frame: Any) -> None:
        nonlocal barge_task, trigger
        if barge_task is not None:
            return
        stop_clock.set()
        if clock is not None:
            await clock
        trigger_reason = (
            "client_send_during_output_recovery"
            if phase == "tool_response"
            else "client_send_during_call_emission"
        )
        trigger = {
            "phase": phase,
            "reason": trigger_reason,
            "transport_frame": frame,
            "observed_monotonic_s": time.monotonic(),
            "turn_start_sent_monotonic_s": None,
            "sent": False,
            "completed": False,
            "turn_started_ack": False,
            "committed_ack": False,
            "input_transcript_completed": False,
        }
        if phase == "tool_response" and function_calls:
            trigger["call_id"] = function_calls[-1]["call_id"]
            trigger["function_output_event_id"] = function_calls[-1].get(
                "output_event_id"
            )
            trigger["function_output_sent_monotonic_s"] = function_calls[-1].get(
                "output_sent_monotonic_s"
            )
        current_trigger = trigger

        def record_turn_start_sent(sent: float) -> None:
            current_trigger["turn_start_sent_monotonic_s"] = sent
            current_trigger["sent"] = True

        barge_task = asyncio.create_task(
            send_speech_carrier_with_continuation_clock(
                websocket,
                barge_pcm,
                prefix=f"safety-{ordinal}-barge",
                client_turn_id=barge_client_turn_id,
                real_time=True,
                stop=barge_clock_stop,
                carrier_done=barge_carrier_done,
                on_turn_start_sent=record_turn_start_sent,
            ),
            name=f"safety-barge-{ordinal}",
        )

    with events_path.open("w", encoding="utf-8") as events:
        async with websockets.connect(
            url,
            open_timeout=30,
            ping_interval=None,
            max_size=16 * 1024 * 1024,
            write_limit=64 * 1024 * 1024,
        ) as websocket:
            created = json.loads(await websocket.recv())
            created["client_received_monotonic_s"] = time.monotonic()
            events.write(json.dumps(created, sort_keys=True) + "\n")
            session_id = (created.get("session") or {}).get("id")
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError("server did not negotiate strict protocol v3")
            session = {
                "protocol_version": 3,
                "instructions": (
                    "Follow the request exactly. Use tools only when requested. "
                    "After a tool result, answer in one short sentence."
                ),
                "tools": tools,
            }
            capabilities_request = session_model_output_capability(use_model_output)
            if capabilities_request is not None:
                session["capabilities"] = capabilities_request
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": f"safety-config-{ordinal}",
                        "session": session,
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                event["client_received_monotonic_s"] = time.monotonic()
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session update failed: {event}")

            job_id = f"safety-{scenario}-{replicate}-{uuid.uuid4().hex[:8]}"
            await websocket.send(
                json.dumps({"type": "input_text.request", "job_id": job_id, "text": prompt})
            )
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                try:
                    raw_event = await asyncio.wait_for(
                        websocket.recv(),
                        max(0.001, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    errors.append({"type": "benchmark.timeout", "scenario": scenario})
                    break
                except Exception as exc:  # noqa: BLE001 - retain a red cell
                    errors.append(
                        {
                            "type": "benchmark.receive_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    break
                event = json.loads(raw_event)
                now = time.monotonic()
                event["client_received_monotonic_s"] = now
                events.write(json.dumps(event, sort_keys=True) + "\n")
                kind = event.get("type")
                if kind == "input_text.injection_finished" and event.get("job_id") == job_id:
                    typed_input_finished = True
                    if clock is None:
                        clock = asyncio.create_task(
                            silence_clock(websocket, stop_clock, f"safety-{ordinal}")
                        )
                    if (
                        scenario == "call_emission_interruption"
                        and trigger is None
                        and pending_call_trigger_frame is not None
                    ):
                        await start_barge(
                            websocket,
                            "tool_call",
                            pending_call_trigger_frame,
                        )
                elif kind == "voicechat.metrics" and scenario.endswith("interruption"):
                    fc = _function_state(event)
                    phase = _heartbeat_phase(fc)
                    if (
                        scenario == "call_emission_interruption"
                        and trigger is None
                        and fc.get("background_active") is True
                        and fc.get("awaiting_response") is False
                        and phase == "tool_call"
                    ):
                        if typed_input_finished:
                            await start_barge(
                                websocket, "tool_call", event.get("transport_frame")
                            )
                        else:
                            pending_call_trigger_frame = event.get("transport_frame")
                    elif (
                        scenario == "result_injection_interruption"
                        and trigger is None
                        and fc.get("background_active") is True
                        and (fc.get("injecting_response") is True or phase == "tool_response")
                    ):
                        await start_barge(
                            websocket, "tool_response", event.get("transport_frame")
                        )
                elif kind == "response.created":
                    response_id = event.get("response_id")
                    if isinstance(response_id, str):
                        response_ids.append(response_id)
                elif kind == "response.function_call_arguments.done":
                    call = {
                        "call_id": event.get("call_id"),
                        "name": event.get("name"),
                        "arguments": event.get("arguments"),
                        "response_id": event.get("response_id"),
                        "turn_id": event.get("turn_id"),
                        "received_monotonic_s": now,
                    }
                    function_calls.append(call)
                    name = call["name"]
                    if scenario == "chained_two_call" and name == FIRST_CHAIN_TOOL["name"]:
                        output_payload = payloads[20]
                    elif scenario == "chained_two_call" and name == SECOND_CHAIN_TOOL["name"]:
                        output_payload = payloads[48]
                    else:
                        # Cleanup an unexpectedly published call in the
                        # call-emission interruption cell so the retained
                        # session can still close deterministically.
                        output_payload = (
                            payloads[72]
                            if scenario == "result_injection_interruption"
                            else payloads[20]
                        )
                    output_event_id = f"safety-output-{ordinal}-{len(function_calls)}"
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "event_id": output_event_id,
                                "item": function_output_item(
                                    call_id=call["call_id"],
                                    payload=output_payload,
                                    use_model_output=use_model_output,
                                ),
                            }
                        )
                    )
                    call["output_event_id"] = output_event_id
                    call["output_sent_monotonic_s"] = time.monotonic()
                elif kind == "conversation.item.function_call_output.applied":
                    outputs_applied.append(
                        {"call_id": event.get("call_id"), "received_monotonic_s": now}
                    )
                elif kind == "input_audio_buffer.turn_started" and trigger is not None:
                    if (
                        event.get("client_event_id") == barge_start_event_id
                        and event.get("client_turn_id") == barge_client_turn_id
                    ):
                        trigger["turn_started_ack"] = True
                        trigger["turn_id"] = event.get("turn_id")
                        trigger["turn_started_ack_received_monotonic_s"] = now
                elif kind == "input_audio_buffer.committed" and trigger is not None:
                    if (
                        event.get("client_event_id") == barge_commit_event_id
                        and event.get("client_turn_id") == barge_client_turn_id
                    ):
                        trigger["committed_ack"] = True
                        trigger["committed_ack_received_monotonic_s"] = now
                        if trigger.get("turn_id") != event.get("turn_id"):
                            trigger["turn_id_mismatch"] = True
                elif (
                    kind == "conversation.item.input_audio_transcription.completed"
                    and trigger is not None
                    and event.get("source") == "microphone"
                    and event.get("turn_id") == trigger.get("turn_id")
                ):
                    trigger["input_transcript_completed"] = True
                    trigger["input_transcript"] = event.get("transcript")
                elif kind == "response.output_text.delta":
                    text += str(event.get("delta") or "")
                elif kind == "response.output_audio.delta":
                    chunk = base64.b64decode(event.get("delta") or "")
                    audio_parts.append(chunk)
                    if first_audible is None and pcm_dbfs(chunk) > SPEECH_DBFS:
                        first_audible = now
                elif kind == "error":
                    errors.append(event)
                elif kind == "response.done":
                    response_done_count += 1
                    done_turn_id = event.get("turn_id")
                    if isinstance(done_turn_id, str):
                        response_done_turn_ids.append(done_turn_id)

                if barge_task is not None and trigger is not None:
                    if barge_carrier_done.is_set():
                        trigger["completed"] = True
                    elif barge_task.done():
                        trigger["completed"] = False
                        error = barge_task.exception()
                        if error is not None:
                            errors.append(
                                {
                                    "type": "benchmark.barge_error",
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            break
                expected_done = 2 if scenario == "result_injection_interruption" else 1
                if response_done_count >= expected_done:
                    if barge_task is None or barge_carrier_done.is_set():
                        break
            else:
                errors.append({"type": "benchmark.timeout", "scenario": scenario})

            stop_clock.set()
            barge_clock_stop.set()
            if clock is not None:
                await clock
            if barge_task is not None:
                try:
                    await barge_task
                    if trigger is not None:
                        trigger["completed"] = True
                except Exception as exc:
                    if trigger is not None:
                        trigger["completed"] = False
                        trigger["error"] = f"{type(exc).__name__}: {exc}"
            try:
                await websocket.send(json.dumps({"type": "session.stop"}))
                while True:
                    event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                    event["client_received_monotonic_s"] = time.monotonic()
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    if event.get("type") == "session.closed":
                        session_closed = True
                        break
                    if event.get("type") == "error":
                        errors.append(event)
            except Exception as exc:
                errors.append(
                    {"type": "benchmark.close_error", "error": f"{type(exc).__name__}: {exc}"}
                )

    if trigger is not None and outputs_applied:
        applied_at = max(float(item["received_monotonic_s"]) for item in outputs_applied)
        started_at = trigger.get("turn_started_ack_received_monotonic_s")
        trigger["turn_started_after_function_output_applied"] = (
            isinstance(started_at, (int, float)) and float(started_at) > applied_at
        )

    audio = b"".join(audio_parts)
    write_wav(output_dir / "response.wav", audio)
    report = {
        "schema": 1,
        "kind": "tool_call_latency_safety_cell",
        "scenario": scenario,
        "replicate": replicate,
        "ordinal": ordinal,
        "session_id": session_id,
        "tools": tools,
        "prompt": prompt,
        "barge_text": barge_text if scenario.endswith("interruption") else None,
        "barge_trigger": trigger,
        "function_calls": function_calls,
        "function_outputs_applied": outputs_applied,
        "model_output_enabled": use_model_output,
        "response_ids": response_ids,
        "response_done_count": response_done_count,
        "response_done_turn_ids": response_done_turn_ids,
        "text": text,
        "audio_bytes": len(audio),
        "first_audible_monotonic_s": first_audible,
        "session_closed": session_closed,
        "errors": errors,
    }
    report["passed"] = safety_cell_passed(report)
    atomic_json(output_dir / "report.json", report)
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.payload_manifest.read_text())
    payload_by_target = {row["target_tokens"]: row for row in manifest["payloads"]}
    if args.mode == "safety":
        if args.speech_pcm is None or not args.speech_text:
            raise ValueError("safety mode requires --speech-pcm and --speech-text")
        prior_safety_reports = []
        for prior_path in args.prior_safety_report:
            prior = json.loads(prior_path.read_text())
            if (
                prior.get("kind") != "tool_call_latency_benchmark"
                or prior.get("mode") != "safety"
                or prior.get("schema") != 1
            ):
                raise ValueError(f"invalid prior safety report: {prior_path}")
            prior_safety_reports.append(
                {
                    "report": str(prior_path.resolve()),
                    "sha256": sha256_file(prior_path),
                    "passed": prior.get("passed"),
                }
            )
        barge_pcm = args.speech_pcm.read_bytes()
        runtime_provenance = await require_safety_diagnostics(args.url)
        reports = []
        started = time.monotonic()
        for scenario_index, scenario in enumerate(SAFETY_SCENARIOS):
            for replicate in range(1, 4):
                ordinal = scenario_index * 3 + replicate
                cell_dir = args.output / f"cell-{ordinal:03d}"
                try:
                    await wait_for_server_idle(args.url)
                    cell_report = await run_safety_cell(
                        url=args.url,
                        output_dir=cell_dir,
                        ordinal=ordinal,
                        replicate=replicate,
                        scenario=scenario,
                        payloads=payload_by_target,
                        timeout_seconds=args.timeout_seconds,
                        barge_pcm=barge_pcm,
                        barge_text=args.speech_text,
                        use_model_output=args.model_output,
                    )
                except Exception as exc:
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    cell_report = {
                        "schema": 1,
                        "kind": "tool_call_latency_safety_cell",
                        "scenario": scenario,
                        "replicate": replicate,
                        "ordinal": ordinal,
                        "passed": False,
                        "session_closed": False,
                        "errors": [
                            {
                                "type": "benchmark.cell_exception",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                    }
                    atomic_json(cell_dir / "report.json", cell_report)
                reports.append(cell_report)
        report = {
            "schema": 1,
            "kind": "tool_call_latency_benchmark",
            "mode": "safety",
            "model_output_enabled": args.model_output,
            "passed": all(row["passed"] for row in reports),
            "scenario_order": list(SAFETY_SCENARIOS),
            "replicates_per_scenario": 3,
            "payload_manifest": str(args.payload_manifest.resolve()),
            "payload_manifest_sha256": sha256_file(args.payload_manifest),
            "barge_pcm": str(args.speech_pcm.resolve()),
            "barge_pcm_sha256": sha256_file(args.speech_pcm),
            "barge_text": args.speech_text,
            "runner_sha256": sha256_file(Path(__file__)),
            "safety_predicate": "safety_cell_passed:v1",
            "prior_safety_reports": prior_safety_reports,
            "runtime_provenance": runtime_provenance,
            "elapsed_seconds": time.monotonic() - started,
            "cells": [
                {
                    "report": str(
                        (args.output / f"cell-{row['ordinal']:03d}" / "report.json").resolve()
                    ),
                    "scenario": row["scenario"],
                    "replicate": row["replicate"],
                    "passed": row["passed"],
                }
                for row in reports
            ],
        }
        atomic_json(args.output / "report.json", report)
        return report
    if args.mode == "reminder":
        if not 1 <= args.replicates <= 20:
            raise ValueError("--replicates must be between one and twenty")
        runtime_provenance = await require_safety_diagnostics(args.url)
        await wait_for_server_idle(args.url)
        started = time.monotonic()
        expects_reminder = args.expect_reminder == "yes"
        require_transport = args.reminder_transport_gate == "required"
        cells = []
        for ordinal in range(1, args.replicates + 1):
            if ordinal > 1:
                await wait_for_server_idle(args.url)
            cell = await run_call(
                url=args.url,
                output_dir=args.output / f"cell-{ordinal:03d}",
                ordinal=ordinal,
                block=ordinal,
                schema_name="short",
                payload=payload_by_target[20],
                timeout_seconds=args.timeout_seconds,
                tool_delay_ms=args.tool_delay_ms,
                use_model_output=args.model_output,
            )
            cell["base_passed"] = cell["passed"]
            gate = evaluate_tool_wait_gate(
                cell,
                expected_policy=args.expected_tool_wait_policy,
                expect_reminder=expects_reminder,
                require_reminder_transport=require_transport,
            )
            cell["tool_wait_gate"] = gate
            cell["passed"] = bool(cell["base_passed"] and gate["passed"])
            atomic_json(args.output / f"cell-{ordinal:03d}" / "report.json", cell)
            cells.append(cell)
        fast_delays = [
            float(cell["tool_wait_evidence"]["result_available_to_phase2_ms"])
            for cell in cells
            if not expects_reminder and cell.get("tool_wait_evidence")
        ]
        fast_p95_ms = nearest_rank_percentile(fast_delays, 0.95) if fast_delays else None
        fast_p95_passed = bool(
            expects_reminder or (fast_p95_ms is not None and fast_p95_ms <= 250.0)
        )
        report = {
            "schema": 1,
            "kind": "tool_call_latency_benchmark",
            "mode": "reminder",
            "model_output_enabled": args.model_output,
            "passed": bool(all(cell["passed"] for cell in cells) and fast_p95_passed),
            "payload_manifest": str(args.payload_manifest.resolve()),
            "payload_manifest_sha256": sha256_file(args.payload_manifest),
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime_provenance": runtime_provenance,
            "tool_delay_ms": args.tool_delay_ms,
            "expected_tool_wait_policy": args.expected_tool_wait_policy,
            "expect_reminder": args.expect_reminder,
            "reminder_transport_gate": args.reminder_transport_gate,
            "replicates": args.replicates,
            "fast_result_to_phase2_p95_ms": fast_p95_ms,
            "fast_result_to_phase2_p95_passed": fast_p95_passed,
            "elapsed_seconds": time.monotonic() - started,
            "cells": [
                {
                    "report": str(
                        (args.output / f"cell-{ordinal:03d}" / "report.json").resolve()
                    ),
                    "passed": cell["passed"],
                }
                for ordinal, cell in enumerate(cells, 1)
            ],
        }
        atomic_json(args.output / "report.json", report)
        return report
    if args.mode == "same-session":
        started = time.monotonic()
        reports = await run_same_session(
            url=args.url,
            output_dir=args.output / "session",
            payload=payload_by_target[72],
            timeout_seconds=args.timeout_seconds,
            use_model_output=args.model_output,
        )
        report = {
            "schema": 1,
            "kind": "tool_call_latency_benchmark",
            "mode": args.mode,
            "model_output_enabled": args.model_output,
            "passed": all(row["passed"] for row in reports),
            "payload_manifest": str(args.payload_manifest.resolve()),
            "payload_manifest_sha256": sha256_file(args.payload_manifest),
            "elapsed_seconds": time.monotonic() - started,
            "cells": [
                {
                    "report": str(
                        (
                            args.output
                            / "session"
                            / f"cell-{row['ordinal']:03d}"
                            / "report.json"
                        ).resolve()
                    ),
                    "passed": row["passed"],
                }
                for row in reports
            ],
        }
        atomic_json(args.output / "report.json", report)
        return report
    speech_pcm = args.speech_pcm.read_bytes() if args.speech_pcm is not None else None
    if args.mode in {"speech", "carrier"} and speech_pcm is None:
        raise ValueError(f"--speech-pcm is required for {args.mode} mode")
    if args.mode == "canary":
        cells = [(1, 1, "short", 72)]
    elif args.mode == "primary":
        cells = [
            (block, (block - 1) * 4 + within, "short", target)
            for block, targets in enumerate(PRIMARY_ORDER, 1)
            for within, target in enumerate(targets, 1)
        ]
    elif args.mode == "schema":
        cells = [(block, block, "long" if block % 2 else "short", 72) for block in range(1, 11)]
    elif args.mode == "schema-long":
        cells = [(block, block, "long", 72) for block in range(1, 6)]
    elif args.mode == "speech":
        cells = [(block, block, "short", 72) for block in range(1, 11)]
    elif args.mode == "carrier":
        cells = [(block, block, "short", 72) for block in range(1, 7)]
    else:
        raise AssertionError(f"unhandled benchmark mode: {args.mode}")
    reports = []
    started = time.monotonic()
    for block, ordinal, schema_name, target in cells:
        reports.append(
            await run_call(
                url=args.url,
                output_dir=args.output / f"cell-{ordinal:03d}",
                ordinal=ordinal,
                block=block,
                schema_name=schema_name,
                payload=payload_by_target[target],
                timeout_seconds=args.timeout_seconds,
                speech_pcm=speech_pcm if args.mode in {"speech", "carrier"} else None,
                speech_text=args.speech_text,
                real_time_input=args.mode != "carrier" or block % 2 == 1,
                use_model_output=args.model_output,
            )
        )
    report = {
        "schema": 1,
        "kind": "tool_call_latency_benchmark",
        "mode": args.mode,
        "model_output_enabled": args.model_output,
        "passed": all(row["passed"] for row in reports),
        "payload_manifest": str(args.payload_manifest.resolve()),
        "payload_manifest_sha256": sha256_file(args.payload_manifest),
        "elapsed_seconds": time.monotonic() - started,
        "cells": [
            {
                "report": str(
                    (args.output / f"cell-{row['ordinal']:03d}" / "report.json").resolve()
                ),
                "passed": row["passed"],
            }
            for row in reports
        ],
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "canary",
            "primary",
            "schema",
            "schema-long",
            "same-session",
            "speech",
            "carrier",
            "safety",
            "reminder",
        ),
        default="canary",
    )
    parser.add_argument("--speech-pcm", type=Path)
    parser.add_argument("--speech-text")
    parser.add_argument("--prior-safety-report", action="append", type=Path, default=[])
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--tool-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--model-output",
        action="store_true",
        help="Negotiate and inject each payload's concise client-authored model_output",
    )
    parser.add_argument(
        "--expected-tool-wait-policy",
        choices=("disabled", "fast_grace", "always_acknowledge"),
        default="disabled",
    )
    parser.add_argument("--expect-reminder", choices=("yes", "no"), default="no")
    parser.add_argument(
        "--reminder-transport-gate",
        choices=("required", "non-gating"),
        default="required",
    )
    parser.add_argument("--replicates", type=int, default=1)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": 1,
            "kind": "tool_call_latency_benchmark",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
