#!/usr/bin/env python3
"""Replay retained session turns directly through the realtime WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
SILENCE_FRAME = bytes(1_280 * 2)
SYSTEM_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state the MOST RECENT result accurately and continue the conversation
normally. Never restate an older tool result after a newer result supersedes it."""
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current clock time in the time zone this conversation is using.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_timezone",
        "description": (
            "Set the time zone used when reporting the current time. Accepts an IANA "
            "name such as America/Los_Angeles or a common spoken name such as Pacific."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA time zone name (America/Los_Angeles) or common spoken "
                        "name (Pacific, Eastern, UTC, Tokyo)."
                    ),
                }
            },
            "required": ["timezone"],
        },
    },
]


@dataclass(frozen=True)
class Packet:
    monotonic_s: float
    payload: bytes


@dataclass(frozen=True)
class SourceTurn:
    client_turn_id: int
    start_s: float
    commit_s: float


def load_source(trace_dir: Path) -> tuple[list[Packet], list[SourceTurn], str]:
    pcm_path = trace_dir / "input-16000-mono-s16le.pcm"
    events_path = trace_dir / "events.jsonl"
    pcm = pcm_path.read_bytes()
    packets: list[Packet] = []
    starts: dict[int, float] = {}
    commits: dict[int, float] = {}
    session_id = trace_dir.name
    offset = 0
    with events_path.open(encoding="utf-8") as stream:
        for raw in stream:
            event = json.loads(raw)
            kind = event.get("event")
            if kind == "input_packet":
                size = int(event["bytes"])
                packets.append(
                    Packet(float(event["monotonic_s"]), pcm[offset : offset + size])
                )
                offset += size
            elif kind == "client_input_turn_started":
                starts[int(event["client_turn_id"])] = float(event["monotonic_s"])
            elif kind == "client_input_turn_commit_received":
                commits[int(event["client_turn_id"])] = float(event["monotonic_s"])
    if offset != len(pcm):
        raise ValueError(f"input packet map covers {offset} of {len(pcm)} PCM bytes")
    if starts.keys() != commits.keys():
        raise ValueError("source trace has unmatched client turn boundaries")
    turns = [
        SourceTurn(turn_id, starts[turn_id], commits[turn_id])
        for turn_id in sorted(starts)
    ]
    return packets, turns, session_id


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(OUTPUT_RATE)
        output.writeframes(payload)


async def send_json(websocket: Any, value: dict[str, Any]) -> None:
    await websocket.send(json.dumps(value, separators=(",", ":")))


class EventReceiver:
    """Own WebSocket reads and resolve diagnostic per-input completion barriers."""

    def __init__(self, websocket: Any, event_stream: Any):
        self.websocket = websocket
        self.event_stream = event_stream
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                event = json.loads(await self.websocket.recv())
                self.event_stream.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "voicechat.client_event.processed":
                    event_id = event.get("client_event_id")
                    future = self.pending.get(str(event_id))
                    if future is not None and not future.done():
                        future.set_result(event)
                await self.events.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)

    async def send_and_wait(
        self,
        value: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        event_id = str(value["event_id"])
        if event_id in self.pending:
            raise RuntimeError(f"duplicate diagnostic event ID: {event_id}")
        future = asyncio.get_running_loop().create_future()
        self.pending[event_id] = future
        try:
            await send_json(self.websocket, value)
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        finally:
            self.pending.pop(event_id, None)

    async def receive(self, *, timeout_seconds: float) -> dict[str, Any]:
        return await asyncio.wait_for(self.events.get(), timeout=timeout_seconds)

    async def close(self) -> None:
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


async def replay_source_turn(
    websocket: Any,
    *,
    packets: list[Packet],
    turn: SourceTurn,
    prefix_seconds: float,
    event_prefix: str,
    pacing: str,
    receiver: EventReceiver,
    step_timeout_seconds: float,
) -> dict[str, Any]:
    window_start = packets[0].monotonic_s if turn.client_turn_id == 1 else (
        turn.start_s - prefix_seconds
    )
    selected = [
        packet
        for packet in packets
        if window_start <= packet.monotonic_s <= turn.commit_s
    ]
    if not selected:
        raise RuntimeError(f"source turn {turn.client_turn_id} selected no PCM packets")
    actions: list[tuple[float, str, bytes | None]] = [
        (packet.monotonic_s, "audio", packet.payload) for packet in selected
    ]
    actions.extend(
        [
            (turn.start_s, "turn_start", None),
            (turn.commit_s, "commit", None),
        ]
    )
    actions.sort(key=lambda item: (item[0], {"turn_start": 0, "audio": 1, "commit": 2}[item[1]]))
    origin = actions[0][0]
    started = time.monotonic()
    audio_bytes = 0
    audio_packets = 0
    for index, (source_time, kind, payload) in enumerate(actions):
        if pacing == "wall":
            deadline = started + (source_time - origin)
            if (delay := deadline - time.monotonic()) > 0:
                await asyncio.sleep(delay)
        event_id = f"{event_prefix}-{index:05d}-{kind}"
        if kind == "audio":
            assert payload is not None
            audio_bytes += len(payload)
            audio_packets += 1
            message = {
                "type": "input_audio_buffer.append",
                "event_id": event_id,
                "encoding": "pcm16",
                "sample_rate": INPUT_RATE,
                "channels": 1,
                "audio": base64.b64encode(payload).decode(),
            }
        else:
            message = {
                "type": f"input_audio_buffer.{kind}",
                "event_id": event_id,
                "client_turn_id": turn.client_turn_id,
            }
        if pacing == "lock-step":
            await receiver.send_and_wait(
                message,
                timeout_seconds=step_timeout_seconds,
            )
        else:
            await send_json(websocket, message)
    return {
        "source_client_turn_id": turn.client_turn_id,
        "source_window_start_s": window_start,
        "source_turn_start_s": turn.start_s,
        "source_turn_commit_s": turn.commit_s,
        "audio_packets": audio_packets,
        "audio_bytes": audio_bytes,
        "audio_sha256": hashlib.sha256(b"".join(packet.payload for packet in selected)).hexdigest(),
    }


async def send_silence(
    websocket: Any,
    stop: asyncio.Event,
    event_prefix: str,
    *,
    pacing: str,
    receiver: EventReceiver,
    step_timeout_seconds: float,
) -> None:
    started = time.monotonic()
    index = 0
    while not stop.is_set():
        if pacing == "wall":
            deadline = started + (index + 1) * 0.08
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=max(0.0, deadline - time.monotonic())
                )
                return
            except TimeoutError:
                pass
        elif stop.is_set():
            return
        message = {
            "type": "input_audio_buffer.append",
            "event_id": f"{event_prefix}-silence-{index:05d}",
            "encoding": "pcm16",
            "sample_rate": INPUT_RATE,
            "channels": 1,
            "audio": base64.b64encode(SILENCE_FRAME).decode(),
        }
        if pacing == "lock-step":
            await receiver.send_and_wait(
                message,
                timeout_seconds=step_timeout_seconds,
            )
        else:
            await send_json(websocket, message)
        index += 1


async def run_attempt(
    *,
    url: str,
    attempt_dir: Path,
    packets: list[Packet],
    turns: list[SourceTurn],
    source_turn_ids: list[int],
    prefix_seconds: float,
    turn_timeout_seconds: float,
    pacing: str,
    step_timeout_seconds: float,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    events_path = attempt_dir / "events.jsonl"
    responses: list[dict[str, Any]] = []
    source_by_id = {turn.client_turn_id: turn for turn in turns}
    with events_path.open("x", encoding="utf-8") as event_stream:
        async with websockets.connect(
            url,
            open_timeout=30,
            ping_interval=None,
            max_size=32 * 1024 * 1024,
        ) as websocket:
            created = json.loads(await websocket.recv())
            event_stream.write(json.dumps(created, sort_keys=True) + "\n")
            await send_json(
                websocket,
                {
                    "type": "session.update",
                    "event_id": "eartts-diagnostic-session-update",
                    "session": {
                        "protocol_version": 3,
                        "instructions": SYSTEM_INSTRUCTION,
                        "tools": TOOLS,
                        "capabilities": {
                            "function_output_model_output": "client_authored_v1"
                        },
                    },
                },
            )
            while True:
                event = json.loads(await websocket.recv())
                event_stream.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session update failed: {event}")

            receiver = EventReceiver(websocket, event_stream)
            try:
                for ordinal, source_turn_id in enumerate(source_turn_ids, 1):
                    source = await replay_source_turn(
                        websocket,
                        packets=packets,
                        turn=source_by_id[source_turn_id],
                        prefix_seconds=prefix_seconds,
                        event_prefix=f"attempt-turn-{ordinal}",
                        pacing=pacing,
                        receiver=receiver,
                        step_timeout_seconds=step_timeout_seconds,
                    )
                    stop = asyncio.Event()
                    clock = asyncio.create_task(
                        send_silence(
                            websocket,
                            stop,
                            f"attempt-turn-{ordinal}",
                            pacing=pacing,
                            receiver=receiver,
                            step_timeout_seconds=step_timeout_seconds,
                        )
                    )
                    active: dict[str, Any] | None = None
                    input_text = ""
                    deadline = time.monotonic() + turn_timeout_seconds
                    try:
                        while time.monotonic() < deadline:
                            event = await receiver.receive(
                                timeout_seconds=deadline - time.monotonic()
                            )
                            kind = event.get("type")
                            if kind == "conversation.item.input_audio_transcription.delta":
                                input_text += str(event.get("delta") or "")
                            elif (
                                kind
                                == "conversation.item.input_audio_transcription.completed"
                            ):
                                input_text = str(event.get("transcript") or input_text)
                            elif kind == "response.created":
                                active = {
                                    "response_id": event.get("response_id"),
                                    "turn_id": event.get("turn_id"),
                                    "text": "",
                                    "audio": [],
                                }
                            elif (
                                kind == "response.output_text.delta"
                                and active is not None
                            ):
                                active["text"] += str(event.get("delta") or "")
                            elif (
                                kind == "response.output_audio.delta"
                                and active is not None
                            ):
                                active["audio"].append(base64.b64decode(event["delta"]))
                            elif kind == "response.function_call_arguments.done":
                                await send_json(
                                    websocket,
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": event.get("call_id"),
                                            "output": json.dumps(
                                                {"diagnostic": "not exercised"}
                                            ),
                                            "model_output": (
                                                "The diagnostic tool result is unavailable."
                                            ),
                                        },
                                    },
                                )
                            elif kind == "error":
                                raise RuntimeError(f"server error: {event}")
                            elif kind == "response.done" and active is not None:
                                audio = b"".join(active.pop("audio"))
                                audio_path = attempt_dir / f"response_{ordinal:03d}.wav"
                                write_wav(audio_path, audio)
                                responses.append(
                                    {
                                        **source,
                                        **active,
                                        "ordinal": ordinal,
                                        "recognized_input": input_text,
                                        "status": event.get("status"),
                                        "reason": event.get("reason"),
                                        "audio_path": str(audio_path),
                                        "audio_bytes": len(audio),
                                        "audio_sha256": hashlib.sha256(audio).hexdigest(),
                                    }
                                )
                                break
                        else:
                            raise TimeoutError(f"turn {ordinal} timed out")
                    finally:
                        stop.set()
                        await clock

                await send_json(websocket, {"type": "session.stop"})
                closed = None
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    event = await receiver.receive(
                        timeout_seconds=deadline - time.monotonic()
                    )
                    if event.get("type") == "session.closed":
                        closed = event
                        break
            finally:
                await receiver.close()
    report = {
        "schema": "eartts-truncation-replay-attempt-v1",
        "url": url,
        "session_id": created.get("session", {}).get("id"),
        "source_turn_ids": source_turn_ids,
        "responses": responses,
        "session_closed": closed,
    }
    (attempt_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = args.source_trace_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    packets, turns, source_session_id = load_source(trace_dir)
    source_turn_ids = [int(value) for value in args.source_turns.split(",")]
    reports = []
    for attempt in range(1, args.attempts + 1):
        reports.append(
            await run_attempt(
                url=args.url,
                attempt_dir=output / f"attempt-{attempt:02d}",
                packets=packets,
                turns=turns,
                source_turn_ids=source_turn_ids,
                prefix_seconds=args.prefix_seconds,
                turn_timeout_seconds=args.turn_timeout_seconds,
                pacing=args.pacing,
                step_timeout_seconds=args.step_timeout_seconds,
            )
        )
        if attempt < args.attempts and args.attempt_gap_seconds > 0:
            await asyncio.sleep(args.attempt_gap_seconds)
    campaign = {
        "schema": "eartts-truncation-replay-campaign-v1",
        "source_session_id": source_session_id,
        "source_trace_dir": str(trace_dir),
        "source_pcm_sha256": hashlib.sha256(
            (trace_dir / "input-16000-mono-s16le.pcm").read_bytes()
        ).hexdigest(),
        "pacing": args.pacing,
        "attempts": reports,
    }
    (output / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8796/v1/realtime")
    parser.add_argument("--source-trace-dir", type=Path, required=True)
    parser.add_argument("--source-turns", default="1,2,3")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--turn-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--step-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--pacing", choices=("wall", "lock-step"), default="wall")
    parser.add_argument(
        "--attempt-gap-seconds",
        type=float,
        default=0.0,
        help="wait for server session cleanup before opening the next attempt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.attempt_gap_seconds < 0:
        raise SystemExit("--attempt-gap-seconds must be non-negative")
    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
