#!/usr/bin/env python3
"""Tool-call behavioral qualification suite.

Replays retained conversation audio against a live service over the direct
WebSocket, with the PRODUCTION session configuration (demo system
instruction, demo tool schema), injecting tool outputs exactly as the
production bot does, and measures degenerate tool-call dynamics:

- tool_calls: every function_call with timestamp and arguments
- max_call_run: longest run of consecutive tool calls with no intervening
  user speech onset and less than ``--run-text-chars`` of new assistant text
- repetition_ratio: fraction of repeated assistant-text 4-grams
- completion: whether the session reached quiescence without error

Diagnostic by design: it records verdict-relevant metrics rather than
enforcing pass/fail, so the same run can baseline pre-existing behavior on
one artifact and compare another.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import websockets

FRAME_SECONDS = 0.08
INPUT_RATE = 16_000
FRAME_BYTES = round(INPUT_RATE * FRAME_SECONDS) * 2

PRODUCTION_INSTRUCTION = (
    "You are a helpful voice assistant in a live conversation.\n"
    "Keep answers concise, natural, and easy to understand when spoken aloud.\n"
    "Use a tool only when the user's request matches that tool's purpose. Never use an\n"
    "available tool as a substitute for an unrelated or unavailable capability. After a\n"
    "tool returns, state its result accurately and continue the conversation normally."
)

UTC_TIME_TOOL = {
    "name": "get_current_utc_time",
    "description": "Get the current clock time in UTC.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BENCHMARK_TOOL = {
    "name": "get_benchmark_word",
    "description": "Return the fresh benchmark word.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

TOOLS = {"utc_time": UTC_TIME_TOOL, "benchmark_word": BENCHMARK_TOOL}
INJECTED_OUTPUT = {
    "utc_time": (
        {"timezone": "UTC", "iso_utc": "2026-08-14T16:59:00+00:00", "spoken": "16:59 UTC"},
        "It is currently 16:59 UTC.",
    ),
    "benchmark_word": (
        {"word": "Harbor"},
        "Harbor is the fresh verification word for this request.",
    ),
}


def segment_turns(
    payload: bytes, gate_dbfs: float = -40.0, max_silence_frames: int = 15
) -> list[tuple[int, int]]:
    """Energy-segment PCM16 into speech turns (frame indices, half-open)."""
    import audioop

    frames = [
        payload[o : o + FRAME_BYTES]
        for o in range(0, len(payload) - FRAME_BYTES + 1, FRAME_BYTES)
    ]
    speech = []
    for f in frames:
        rms = audioop.rms(f, 2)
        import math

        dbfs = 20 * math.log10(rms / 32768) if rms else -120.0
        speech.append(dbfs > gate_dbfs)
    turns: list[tuple[int, int]] = []
    start = None
    silence = 0
    for i, is_speech in enumerate(speech):
        if is_speech:
            if start is None:
                start = max(0, i - 2)  # preroll
            silence = 0
        elif start is not None:
            silence += 1
            if silence > max_silence_frames:
                turns.append((start, i - silence + 3))
                start, silence = None, 0
    if start is not None:
        turns.append((start, len(frames)))
    return [(a, b) for a, b in turns if b - a >= 4]


def repetition_ratio(text: str) -> float:
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


async def run_scenario(args: argparse.Namespace) -> dict[str, Any]:
    payload = args.speech_pcm.read_bytes()
    tool = TOOLS[args.tool]
    output_obj, model_output = INJECTED_OUTPUT[args.tool]
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    tool_calls: list[dict[str, Any]] = []
    injections = 0
    assistant_text: list[str] = []
    call_runs: list[int] = [0]
    text_since_last_call = 0
    user_speech_onsets = 0
    errors: list[dict[str, Any]] = []
    last_bot_activity = {"t": 0.0}
    response_done_count = {"n": 0}

    events = events_path.open("w", encoding="utf-8")
    async with websockets.connect(
        args.url, max_size=64 * 1024 * 1024, open_timeout=30
    ) as websocket:

        async def recv_logged() -> dict[str, Any]:
            event = json.loads(await websocket.recv())
            event["client_received_monotonic_s"] = time.monotonic()
            events.write(json.dumps(event, sort_keys=True) + "\n")
            return event

        created = await recv_logged()
        if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
            raise RuntimeError(f"protocol negotiation failed: {created.get('type')}")
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "behavior-config",
                    "session": {
                        "protocol_version": 3,
                        "instructions": PRODUCTION_INSTRUCTION,
                        "tools": [tool],
                        "capabilities": {
                            "function_output_model_output": "client_authored_v1"
                        },
                    },
                }
            )
        )
        while True:
            event = await recv_logged()
            if event.get("type") == "session.updated":
                break
            if event.get("type") == "error":
                raise RuntimeError(f"session update failed: {event}")

        stop_reading = asyncio.Event()

        async def reader() -> None:
            nonlocal injections, text_since_last_call, user_speech_onsets
            while not stop_reading.is_set():
                try:
                    event = await asyncio.wait_for(recv_logged(), timeout=1.0)
                except TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    stop_reading.set()
                    return
                etype = event.get("type")
                if etype == "input_audio_buffer.speech_started":
                    user_speech_onsets += 1
                    call_runs.append(0)
                elif etype in ("response.output_audio.delta", "response.created"):
                    last_bot_activity["t"] = time.monotonic()
                elif etype == "response.done":
                    response_done_count["n"] += 1
                elif etype == "response.output_text.delta":
                    last_bot_activity["t"] = time.monotonic()
                    delta = event.get("delta") or ""
                    assistant_text.append(delta)
                    text_since_last_call += len(delta)
                elif etype == "response.function_call_arguments.done":
                    if text_since_last_call < args.run_text_chars:
                        call_runs[-1] += 1
                    else:
                        call_runs.append(1)
                    text_since_last_call = 0
                    tool_calls.append(
                        {
                            "monotonic_s": event["client_received_monotonic_s"],
                            "name": event.get("name"),
                            "arguments": event.get("arguments"),
                            "call_id": event.get("call_id"),
                        }
                    )
                    if injections < args.max_injections:
                        injections += 1
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "event_id": f"inject-{injections}",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": event.get("call_id"),
                                        "output": json.dumps(output_obj),
                                        "model_output": model_output,
                                    },
                                }
                            )
                        )
                elif etype == "error":
                    errors.append(event)

        reader_task = asyncio.create_task(reader())
        session_fatal = None

        # replay on the ORIGINAL timeline, bracketing energy-segmented turns
        # with turn_start/commit per the negotiated client_smart_turn mode
        padded = payload + bytes((-len(payload)) % FRAME_BYTES)
        frames = [
            padded[offset : offset + FRAME_BYTES]
            for offset in range(0, len(padded), FRAME_BYTES)
        ]
        turns = segment_turns(padded, max_silence_frames=args.max_silence_frames)
        if not turns:
            raise RuntimeError("no speech turns detected in replay audio")
        started = time.monotonic()
        try:
            for turn_index, (begin, end) in enumerate(turns):
                delay = started + begin * FRAME_SECONDS - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                # protocol-aware pacing (recorded deviation from original
                # overlap timing): each committed turn produces one response;
                # wait for its response.done before opening the next turn, plus a
                # short quiesce gap, capped so a nonterminating response cannot
                # stall the replay forever
                done_deadline = time.monotonic() + args.turn_quiesce_cap_s
                while time.monotonic() < done_deadline:
                    if response_done_count["n"] >= turn_index:
                        break
                    await asyncio.sleep(0.1)
                quiesce_deadline = time.monotonic() + args.turn_quiesce_cap_s
                while time.monotonic() < quiesce_deadline:
                    if time.monotonic() - last_bot_activity["t"] >= args.turn_quiesce_s:
                        break
                    await asyncio.sleep(0.1)
                turn_started = time.monotonic()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.turn_start",
                            "event_id": f"turn-{turn_index}-start",
                            "client_turn_id": turn_index + 1,
                        }
                    )
                )
                for index in range(begin, end):
                    delay = turn_started + (index - begin) * FRAME_SECONDS - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"audio-{turn_index}-{index}",
                                "encoding": "pcm16",
                                "sample_rate": INPUT_RATE,
                                "channels": 1,
                                "audio": base64.b64encode(frames[index]).decode(),
                            }
                        )
                    )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.commit",
                            "event_id": f"turn-{turn_index}-commit",
                            "client_turn_id": turn_index + 1,
                        }
                    )
                )
        except websockets.ConnectionClosed as exc:
            session_fatal = str(exc)
        # post-audio observation window for late loops
        await asyncio.sleep(args.observe_s)
        stop_reading.set()
        await reader_task

    events.close()
    text = "".join(assistant_text)
    report = {
        "kind": "tool_call_behavior_scenario",
        "schema": 1,
        "scenario": args.scenario,
        "url": args.url,
        "speech_pcm": str(args.speech_pcm),
        "speech_seconds": len(payload) / (INPUT_RATE * 2),
        "tool": args.tool,
        "instructions": "production",
        "max_injections": args.max_injections,
        "tool_call_count": len(tool_calls),
        "injections_sent": injections,
        "max_call_run": max(call_runs),
        "call_runs": [r for r in call_runs if r],
        "user_speech_onsets": user_speech_onsets,
        "replayed_turns": len(turns),
        "assistant_text": text,
        "assistant_text_chars": len(text),
        "repetition_ratio_4gram": round(repetition_ratio(text), 4),
        "errors": errors,
        "session_fatal": session_fatal,
        "tool_calls_head": tool_calls[:10],
        "loop_flag": max(call_runs) > 3,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--speech-pcm", type=Path, required=True)
    parser.add_argument("--tool", choices=sorted(TOOLS), default="utc_time")
    parser.add_argument("--max-injections", type=int, default=8)
    parser.add_argument("--run-text-chars", type=int, default=20)
    parser.add_argument("--trailing-silence-s", type=float, default=8.0)
    parser.add_argument("--turn-quiesce-s", type=float, default=1.0)
    parser.add_argument("--max-silence-frames", type=int, default=15)
    parser.add_argument("--turn-quiesce-cap-s", type=float, default=25.0)
    parser.add_argument("--observe-s", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run_scenario(args))
    condensed = {
        k: report[k]
        for k in (
            "scenario",
            "tool_call_count",
            "max_call_run",
            "repetition_ratio_4gram",
            "loop_flag",
            "assistant_text_chars",
        )
    }
    condensed["errors"] = len(report["errors"])
    print(json.dumps(condensed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
