#!/usr/bin/env python3
"""Fail-closed rapid-turn replay for user-EOU settlement qualification.

The fixture deliberately keeps streaming real microphone silence after each
commit. That is the production queue shape which the settlement path may use,
subject to the server speech gate, instead of adding synthetic silence ahead of
the queued audio.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import math
import time
from pathlib import Path
from typing import Any

import websockets
from response_completion_gate import gate_trace
from tool_call_behavior_suite import (
    BENCHMARK_TOOL,
    INJECTED_OUTPUT,
    PRODUCTION_INSTRUCTION,
)

from nemotron_voicechat_runtime.provenance import valid_runtime_provenance

INPUT_RATE = 16_000
FRAME_SECONDS = 0.08
FRAME_BYTES = round(INPUT_RATE * FRAME_SECONDS) * 2


def pcm_dbfs(payload: bytes) -> float:
    rms = audioop.rms(payload, 2)
    return 20.0 * math.log10(rms / 32768.0) if rms else -120.0


async def run(args: argparse.Namespace) -> dict[str, Any]:
    speech = args.speech_pcm.read_bytes()
    if not speech or len(speech) % 2:
        raise ValueError("speech fixture must be non-empty whole-sample PCM16")
    speech += bytes((-len(speech)) % FRAME_BYTES)
    speech_frames = [
        speech[offset : offset + FRAME_BYTES] for offset in range(0, len(speech), FRAME_BYTES)
    ]
    if not any(pcm_dbfs(frame) >= args.speech_gate_dbfs for frame in speech_frames):
        raise ValueError("speech fixture has no frame at or above the speech gate")

    args.output.mkdir(parents=True, exist_ok=True)
    events_path = args.output / "events.jsonl"
    responses: list[dict[str, Any]] = []
    response_by_id: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    user_transcripts: list[str] = []
    commits: list[float] = []
    response_done = asyncio.Condition()
    stop_reader = asyncio.Event()
    session_closed = asyncio.Event()
    tool_calls = 0
    created_session_id: str | None = None
    runtime_provenance: dict[str, Any] | None = None
    production_value_coverage: dict[str, Any] | None = None

    with events_path.open("w", encoding="utf-8") as event_file:
        async with websockets.connect(
            args.url,
            max_size=64 * 1024 * 1024,
            open_timeout=30,
        ) as websocket:

            async def receive_logged() -> dict[str, Any]:
                event = json.loads(await websocket.recv())
                event["client_received_monotonic_s"] = time.monotonic()
                event_file.write(json.dumps(event, sort_keys=True) + "\n")
                event_file.flush()
                return event

            created = await receive_logged()
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError(f"protocol negotiation failed: {created}")
            created_session_id = created.get("session", {}).get("id")
            runtime_provenance = created.get("session", {}).get("checkpoint", {}).get(
                "runtime_provenance"
            )
            semantic = (
                runtime_provenance.get("semantic_environment", {})
                if isinstance(runtime_provenance, dict)
                else {}
            )
            actual_values = {
                name: semantic.get(name)
                for name in (
                    "VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES",
                    "VOICECHAT_WEB_AGENT_NO_AUDIO_FRAMES",
                    "VOICECHAT_WEB_AGENT_SILENCE_EOS_FRAMES",
                    "VOICECHAT_WEB_AGENT_SILENCE_EOS_DBFS",
                    "VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC",
                )
            }
            expected_values = {
                "VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES": "30",
                "VOICECHAT_WEB_AGENT_NO_AUDIO_FRAMES": "30",
                "VOICECHAT_WEB_AGENT_SILENCE_EOS_FRAMES": "20",
                "VOICECHAT_WEB_AGENT_SILENCE_EOS_DBFS": "-90",
                "VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC": "30",
            }
            production_value_coverage = {
                "qualification_override_used": False,
                "expected": expected_values,
                "actual": actual_values,
                "passed": actual_values == expected_values,
            }
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "rapid-config",
                        "session": {
                            "protocol_version": 3,
                            "instructions": PRODUCTION_INSTRUCTION,
                            "tools": [BENCHMARK_TOOL] if args.with_tool else [],
                            "capabilities": {"function_output_model_output": "client_authored_v1"},
                        },
                    }
                )
            )
            while True:
                event = await receive_logged()
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session configuration failed: {event}")

            async def reader() -> None:
                nonlocal tool_calls
                while not stop_reader.is_set():
                    try:
                        event = await asyncio.wait_for(receive_logged(), timeout=1.0)
                    except TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        return
                    event_type = event.get("type")
                    received_at = float(event["client_received_monotonic_s"])
                    if event_type == "response.created":
                        response_id = event.get("response_id")
                        if not isinstance(response_id, str):
                            errors.append({"reason": "missing_response_id", "event": event})
                            continue
                        response = {
                            "response_id": response_id,
                            "turn_id": event.get("turn_id"),
                            "created_monotonic_s": received_at,
                            "first_audio_monotonic_s": None,
                            "text": "",
                            "audio_chunks": 0,
                            "audible_audio_chunks": 0,
                            "status": None,
                        }
                        responses.append(response)
                        response_by_id[response_id] = response
                    elif event_type == "response.output_text.delta":
                        response = response_by_id.get(event.get("response_id"))
                        if response is not None:
                            response["text"] += str(event.get("delta") or "")
                    elif event_type == "response.output_audio.delta":
                        response = response_by_id.get(event.get("response_id"))
                        if response is not None:
                            if response["first_audio_monotonic_s"] is None:
                                response["first_audio_monotonic_s"] = received_at
                            response["audio_chunks"] += 1
                            try:
                                audio = base64.b64decode(event.get("delta", ""), validate=True)
                            except (ValueError, TypeError):
                                audio = b""
                            if audio and pcm_dbfs(audio) >= args.audible_dbfs:
                                response["audible_audio_chunks"] += 1
                    elif event_type == "response.function_call_arguments.done":
                        tool_calls += 1
                        output, model_output = INJECTED_OUTPUT["benchmark_word"]
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "event_id": f"rapid-tool-{tool_calls}",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": event.get("call_id"),
                                        "output": json.dumps(output),
                                        "model_output": model_output,
                                    },
                                }
                            )
                        )
                    elif event_type == "response.done":
                        response = response_by_id.get(event.get("response_id"))
                        if response is not None:
                            response["status"] = event.get("status")
                        async with response_done:
                            response_done.notify_all()
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        user_transcripts.append(str(event.get("transcript") or ""))
                    elif event_type == "error":
                        errors.append(event)
                    elif event_type == "session.closed":
                        session_closed.set()

            reader_task = asyncio.create_task(reader(), name="rapid-turn-reader")
            silence = b"\x00" * FRAME_BYTES
            for turn_index in range(args.turns):
                turn_id = turn_index + 1
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.turn_start",
                            "event_id": f"rapid-start-{turn_id}",
                            "client_turn_id": turn_id,
                        }
                    )
                )
                for frame_index, frame in enumerate(speech_frames):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"rapid-audio-{turn_id}-{frame_index}",
                                "encoding": "pcm16",
                                "sample_rate": INPUT_RATE,
                                "channels": 1,
                                "audio": base64.b64encode(frame).decode("ascii"),
                            }
                        )
                    )
                    await asyncio.sleep(FRAME_SECONDS)
                commit_at = time.monotonic()
                commits.append(commit_at)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.commit",
                            "event_id": f"rapid-commit-{turn_id}",
                            "client_turn_id": turn_id,
                        }
                    )
                )

                async def stream_trailing_silence() -> None:
                    silence_index = 0
                    while len([r for r in responses if r["status"] is not None]) < turn_id:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "event_id": f"rapid-tail-{turn_id}-{silence_index}",
                                    "encoding": "pcm16",
                                    "sample_rate": INPUT_RATE,
                                    "channels": 1,
                                    "audio": base64.b64encode(silence).decode("ascii"),
                                }
                            )
                        )
                        silence_index += 1
                        await asyncio.sleep(FRAME_SECONDS)

                silence_task = asyncio.create_task(stream_trailing_silence())
                deadline = time.monotonic() + args.response_timeout_s
                async with response_done:
                    while len([r for r in responses if r["status"] is not None]) < turn_id:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            await asyncio.wait_for(response_done.wait(), timeout=remaining)
                        except TimeoutError:
                            break
                silence_task.cancel()
                await asyncio.gather(silence_task, return_exceptions=True)
                if len([r for r in responses if r["status"] is not None]) < turn_id:
                    errors.append({"reason": "response_timeout", "turn": turn_id})
                    break
                await asyncio.sleep(args.inter_turn_s)

            await websocket.send(json.dumps({"type": "session.stop"}))
            try:
                await asyncio.wait_for(
                    session_closed.wait(),
                    timeout=args.session_close_timeout_s,
                )
            except TimeoutError:
                errors.append({"reason": "session_close_timeout"})
            stop_reader.set()
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)

    for index, response in enumerate(responses):
        commit_at = commits[index] if index < len(commits) else None
        first_audio = response["first_audio_monotonic_s"]
        response["commit_to_first_audio_ms"] = (
            None
            if commit_at is None or first_audio is None
            else round((float(first_audio) - commit_at) * 1000.0, 3)
        )
        response["empty"] = not response["text"].strip() and not response["audible_audio_chunks"]

    fatal_errors = [
        error
        for error in errors
        if bool((error.get("error") or {}).get("fatal"))
        or error.get("reason")
        in {"response_timeout", "missing_response_id", "session_close_timeout"}
    ]
    empty_responses = [response["response_id"] for response in responses if response["empty"]]
    nonempty_text = [bool(response["text"].strip()) for response in responses]
    ttfb_ms = [response["commit_to_first_audio_ms"] for response in responses]
    completion_gate: dict[str, Any]
    trace_path = (
        args.model_trace_root / str(created_session_id) / "events.jsonl"
        if created_session_id
        else args.model_trace_root / "missing-session-id"
    )
    try:
        completion_gate = gate_trace(trace_path)
    except Exception as exc:
        completion_gate = {
            "schema": "nemotron_voicechat.response_completion_gate.v1",
            "source": str(trace_path),
            "passed": False,
            "error": repr(exc),
        }
    provenance_passed = bool(
        isinstance(runtime_provenance, dict)
        and valid_runtime_provenance(runtime_provenance)
        and isinstance(runtime_provenance.get("runtime_image"), str)
        and runtime_provenance.get("runtime_image")
        and runtime_provenance.get("runtime_contract")
        == "production-hotfix-notext-watchdog-v1"
    )
    result = {
        "kind": "rapid_turn_settlement_v1",
        "scenario": args.scenario,
        "with_tool": args.with_tool,
        "turns_requested": args.turns,
        "responses": responses,
        "responses_completed": len([r for r in responses if r["status"] is not None]),
        "user_transcripts": user_transcripts,
        "all_response_transcripts_nonempty": all(nonempty_text)
        and len(nonempty_text) == args.turns,
        "empty_response_ids": empty_responses,
        "empty_response_count": len(empty_responses),
        "fatal_errors": fatal_errors,
        "errors": errors,
        "tool_call_count": tool_calls,
        "session_id": created_session_id,
        "runtime_provenance": runtime_provenance,
        "runtime_provenance_passed": provenance_passed,
        "production_value_coverage": production_value_coverage,
        "response_completion_gate": completion_gate,
        "ttfb_ms": ttfb_ms,
        "six_turn_ttfb_ms": ttfb_ms[:6],
        "passed": (
            len(responses) == args.turns
            and len([r for r in responses if r["status"] is not None]) == args.turns
            and not empty_responses
            and not fatal_errors
            and all(nonempty_text)
            and provenance_passed
            and production_value_coverage is not None
            and production_value_coverage["passed"]
            and completion_gate.get("passed") is True
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--speech-pcm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--with-tool", action="store_true")
    parser.add_argument("--speech-gate-dbfs", type=float, default=-40.0)
    parser.add_argument("--audible-dbfs", type=float, default=-60.0)
    parser.add_argument("--inter-turn-s", type=float, default=0.08)
    parser.add_argument("--response-timeout-s", type=float, default=20.0)
    parser.add_argument("--session-close-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--model-trace-root",
        type=Path,
        default=Path.home() / ".local/state/nemotron-voicechat/traces/model",
    )
    args = parser.parse_args()
    if args.turns < 8:
        parser.error("--turns must be at least 8")
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
