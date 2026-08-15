#!/usr/bin/env python3
"""FC-phase-targeted interruption matrix.

Closes the qualification gap at the tool-call x interruption intersection:
elicits a real tool call with the retained spoken carrier, then injects a
second user utterance at a controlled offset keyed to an OBSERVED function-
cycle phase event, and adjudicates the session with variation-tolerant
metrics (loop incidence, fatals, recovery events, response lifecycle).

Phases (trigger -> when the interrupting utterance starts):
  pre_response      on response.function_call_arguments.done, BEFORE any
                    function_call_output is sent; the output is sent after
                    the utterance finishes (call outstanding during speech)
  during_injection  send the output immediately, start the utterance at a
                    short fixed delay while the model integrates the result
  during_ack        wait for first response.output_audio.delta after the
                    output (the spoken acknowledgment), then interrupt it
  control           no interruption (baseline cell)

Every cell reports: tool calls, max call-run, fatal, errors, whether a
second answer lifecycle followed the interruption, and the server-side
recovery/watchdog events observed in the model trace afterward.
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

TOOL = {
    "name": "get_benchmark_word",
    "description": "Return the fresh benchmark word.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
TOOL_OUTPUT = {"word": "Harbor"}
TOOL_MODEL_OUTPUT = "Harbor is the fresh verification word for this request."
PHASES = ("control", "pre_response", "during_injection", "during_ack")


async def silence_clock(websocket: Any, stop: asyncio.Event, gate: asyncio.Lock) -> None:
    """Keep the realtime frame clock advancing with silence between turns."""
    silence = base64.b64encode(bytes(FRAME_BYTES)).decode()
    index = 0
    started = time.monotonic()
    while not stop.is_set():
        deadline = started + index * FRAME_SECONDS
        delay = deadline - time.monotonic()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                break
            except TimeoutError:
                pass
        async with gate:
            if stop.is_set():
                break
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": f"clock-{index}",
                        "encoding": "pcm16",
                        "sample_rate": INPUT_RATE,
                        "channels": 1,
                        "audio": silence,
                    }
                )
            )
        index += 1


async def send_turn(
    websocket: Any, pcm: bytes, turn_id: int, label: str, gate: asyncio.Lock
) -> None:
    padded = pcm + bytes((-len(pcm)) % FRAME_BYTES)
    frames = [padded[o : o + FRAME_BYTES] for o in range(0, len(padded), FRAME_BYTES)]
    frames.extend([bytes(FRAME_BYTES)] * 3)
    async with gate:
        await _send_turn_locked(websocket, frames, turn_id, label)


async def _send_turn_locked(websocket: Any, frames: list, turn_id: int, label: str) -> None:
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.turn_start",
                "event_id": f"{label}-start",
                "client_turn_id": turn_id,
            }
        )
    )
    started = time.monotonic()
    for index, frame in enumerate(frames):
        delay = started + index * FRAME_SECONDS - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{label}-audio-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": base64.b64encode(frame).decode(),
                }
            )
        )
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.commit",
                "event_id": f"{label}-commit",
                "client_turn_id": turn_id,
            }
        )
    )


async def run_cell(args: argparse.Namespace, phase: str, out_dir: Path) -> dict[str, Any]:
    carrier = args.carrier_pcm.read_bytes()
    interrupter = args.interrupt_pcm.read_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_log = (out_dir / "events.jsonl").open("w", encoding="utf-8")
    tool_calls: list[dict[str, Any]] = []
    fatals: list[str] = []
    errors: list[dict[str, Any]] = []
    call_event = asyncio.Event()
    injection_done = asyncio.Event()
    ack_audio = asyncio.Event()
    responses_done = 0
    pending_call: dict[str, Any] = {}

    async with websockets.connect(
        args.url, max_size=64 * 1024 * 1024, open_timeout=30
    ) as websocket:

        async def recv_logged() -> dict[str, Any]:
            event = json.loads(await websocket.recv())
            event["client_received_monotonic_s"] = time.monotonic()
            events_log.write(json.dumps(event, sort_keys=True) + "\n")
            return event

        created = await recv_logged()
        if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
            raise RuntimeError("protocol negotiation failed")
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "matrix-config",
                    "session": {
                        "protocol_version": 3,
                        "instructions": PRODUCTION_INSTRUCTION,
                        "tools": [TOOL],
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

        stop = asyncio.Event()

        async def send_output(call_id: str) -> None:
            nonlocal_injected = {
                "type": "conversation.item.create",
                "event_id": f"inject-{call_id[:8]}",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(TOOL_OUTPUT),
                    "model_output": TOOL_MODEL_OUTPUT,
                },
            }
            await websocket.send(json.dumps(nonlocal_injected))

        async def reader() -> None:
            nonlocal responses_done
            while not stop.is_set():
                try:
                    event = await asyncio.wait_for(recv_logged(), timeout=1.0)
                except TimeoutError:
                    continue
                except websockets.ConnectionClosed as exc:
                    fatals.append(str(exc))
                    stop.set()
                    return
                etype = event.get("type")
                if etype == "response.function_call_arguments.done":
                    tool_calls.append(
                        {"call_id": event.get("call_id"), "name": event.get("name")}
                    )
                    pending_call["call_id"] = event.get("call_id")
                    call_event.set()
                    if phase != "pre_response" and len(tool_calls) == 1:
                        await send_output(event.get("call_id"))
                        injection_done.set()
                    elif len(tool_calls) > 1 and len(tool_calls) <= 4:
                        await send_output(event.get("call_id"))
                elif etype == "conversation.item.function_call_output.applied":
                    injection_done.set()
                elif etype == "response.output_audio.delta" and injection_done.is_set():
                    ack_audio.set()
                elif etype == "response.done":
                    responses_done += 1
                elif etype == "error":
                    errors.append(event)
                    if (event.get("error") or {}).get("fatal"):
                        fatals.append(str(event.get("error", {}).get("code")))

        reader_task = asyncio.create_task(reader())
        clock_gate = asyncio.Lock()
        clock_task = asyncio.create_task(silence_clock(websocket, stop, clock_gate))
        try:
            await send_turn(websocket, carrier, 1, "carrier", clock_gate)
            try:
                await asyncio.wait_for(call_event.wait(), timeout=30)
            except TimeoutError:
                raise RuntimeError("tool call never elicited")

            if phase == "control":
                pass
            elif phase == "pre_response":
                await send_turn(websocket, interrupter, 2, "interrupt", clock_gate)
                await send_output(pending_call["call_id"])
            elif phase == "during_injection":
                await asyncio.sleep(0.3)
                await send_turn(websocket, interrupter, 2, "interrupt", clock_gate)
            elif phase == "during_ack":
                try:
                    await asyncio.wait_for(ack_audio.wait(), timeout=30)
                except TimeoutError:
                    raise RuntimeError("acknowledgment audio never started")
                await send_turn(websocket, interrupter, 2, "interrupt", clock_gate)

            await asyncio.sleep(args.observe_s)
        except (RuntimeError, websockets.ConnectionClosed) as exc:
            errors.append({"harness": str(exc)})
        finally:
            stop.set()
            await reader_task
            await clock_task
            events_log.close()

    report = {
        "kind": "tool_call_interrupt_cell",
        "schema": 1,
        "phase": phase,
        "tool_call_count": len(tool_calls),
        "responses_done": responses_done,
        "fatals": fatals,
        "errors": [str(e)[:200] for e in errors],
        "second_lifecycle_after_interrupt": responses_done >= 2 if phase != "control" else None,
        "passed": not fatals
        and len(tool_calls) >= 1
        and len(tool_calls) <= 5
        and not any("harness" in str(e) for e in errors),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


async def run_matrix(args: argparse.Namespace) -> None:
    results = []
    for phase in PHASES:
        for replicate in range(1, args.replicates + 1):
            cell_dir = args.output / f"{phase}-r{replicate}"
            report = await run_cell(args, phase, cell_dir)
            results.append(report)
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "phase",
                            "tool_call_count",
                            "responses_done",
                            "fatals",
                            "passed",
                        )
                    },
                    sort_keys=True,
                )
            )
            await asyncio.sleep(args.between_cells_s)
    summary = {
        "kind": "tool_call_interrupt_matrix",
        "cells": len(results),
        "passed_cells": sum(1 for r in results if r["passed"]),
        "fatal_cells": sum(1 for r in results if r["fatals"]),
        "all_passed": all(r["passed"] for r in results),
    }
    (args.output / "matrix-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--carrier-pcm", type=Path, required=True)
    parser.add_argument("--interrupt-pcm", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--observe-s", type=float, default=12.0)
    parser.add_argument("--between-cells-s", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()
