#!/usr/bin/env python3
"""Turn-gated Phase 6 campaign driver for the local strict-v3 Spark stack.

The NIM and Spark drivers intentionally share this per-session JSONL schema:

* ``{"kind":"meta", "arm","session_id","scenario","fixtures","tools",
  "started_unix"}``
* ``{"kind":"event", "t":..., "event": <server event>}`` (audio payload compacted)
* ``{"kind":"sent", "t":..., "type": <client event type>, "detail": {...}}``
* ``{"kind":"turn", "index","fixture","audio_seconds","gated_on",
  "wait_seconds","outcome"}``
* ``{"kind":"response_audio", "response_index","wav","rate"}``
* ``{"kind":"summary", ...}``

Spark-specific protocol translation is limited to strict-v3's explicit
``turn_start``/``commit`` lifecycle and client-authored ``model_output`` for function
results. Fixtures are PCM16 mono 16 kHz WAVs streamed in 80 ms frames at wall-clock
pace. Silence keeps the frame-driven model advancing until the response correlated to
that turn reaches ``response.done`` (60 s default), followed by a 1 s settle. Each
scenario then receives the same fixed trailing-silence window as the NIM arm.

This is qualification tooling only. It does not import or modify production code.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import websockets

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_SAMPLES = 1_280
FRAME_SECONDS = FRAME_SAMPLES / INPUT_RATE
FRAME_BYTES = FRAME_SAMPLES * 2
DEFAULT_SETTLE_SECONDS = 1.0
DEFAULT_GATE_TIMEOUT_SECONDS = 60.0
DEFAULT_CLOSE_TIMEOUT_SECONDS = 30.0

SYSTEM_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state the MOST RECENT result accurately and continue the conversation
normally. Never restate an older tool result after a newer result supersedes it."""

# Match the NIM arm exactly. The only documented prompt confound is each arm's
# as-shipped system instruction; tool wording must not introduce another one.
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Returns the current local time",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "set_timezone",
        "description": "Sets the timezone used for time answers",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
]

SCENARIOS = {
    "s1": (("t1_greeting.wav", "t3_story.wav"), False, 20.0),
    "s2t": (("t1_greeting.wav", "t4_time.wav"), True, 20.0),
    "s2z": (("t1_greeting.wav", "t5_timezone.wav"), True, 20.0),
    "s3": (("t1_greeting.wav",), False, 30.0),
}


def normalize_uri(server: str) -> str:
    uri = server.rstrip("/")
    if not uri.endswith("/v1/realtime"):
        uri += "/v1/realtime"
    return uri


def read_wav(path: Path) -> tuple[bytes, int]:
    """Read one campaign fixture, rejecting any arm-changing conversion."""

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        compression = source.getcomptype()
        payload = source.readframes(source.getnframes())
    if (channels, width, rate, compression) != (1, 2, INPUT_RATE, "NONE"):
        raise ValueError(
            f"{path}: expected PCM16 mono {INPUT_RATE} Hz WAV, got "
            f"channels={channels} width={width} rate={rate} compression={compression}"
        )
    if not payload or len(payload) % 2:
        raise ValueError(f"{path}: expected non-empty whole-sample PCM16")
    return payload, rate


def write_wav(path: Path, payload: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(payload)


def deterministic_tool_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "get_current_time":
        return {"time": "3:47 PM"}
    if name == "set_timezone":
        return {"status": "ok", "timezone": arguments.get("timezone", "?")}
    return {"error": f"unknown tool {name}"}


def deterministic_model_output(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    """Give strict-v3 the spoken continuation its negotiated capability requires."""

    if name == "get_current_time":
        return "The current time is 3:47 PM."
    if name == "set_timezone":
        return f"The time zone is set to {arguments.get('timezone', '?')}."
    return str(result["error"])


class EvidenceLog:
    def __init__(self, stream: Any):
        self.stream = stream
        self.t0 = 0.0

    def start_clock(self) -> None:
        self.t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.t0 if self.t0 else 0.0

    def emit(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()

    def received(self, event: dict[str, Any], *, audio_pcm_bytes: int | None = None) -> None:
        if audio_pcm_bytes is None:
            compact = event
        else:
            # Match the NIM driver's compact event shape. Its historical field name
            # says b64 even though both drivers record decoded PCM byte length.
            compact = {
                "type": "response.output_audio.delta",
                "audio_b64_bytes": audio_pcm_bytes,
            }
        self.emit({"kind": "event", "t": round(self.now(), 3), "event": compact})

    def sent(self, event_type: str, detail: dict[str, Any] | None = None) -> None:
        self.emit(
            {
                "kind": "sent",
                "t": round(self.now(), 3),
                "type": event_type,
                "detail": detail or {},
            }
        )


class SessionState:
    """Single WebSocket reader, response capture, and correlated turn barriers."""

    def __init__(self, evidence: EvidenceLog):
        self.evidence = evidence
        self.websocket: Any = None
        self.send_lock = asyncio.Lock()
        self.changed = asyncio.Condition()
        self.response_index = -1
        self.response_audio: dict[int, bytearray] = {}
        self.response_records: dict[str, dict[str, Any]] = {}
        self.done_records: list[dict[str, Any]] = []
        self.turn_ids: dict[int, str] = {}
        self.function_calls: list[dict[str, Any]] = []
        self.counts = {"responses": 0, "response_done": 0, "errors": 0}
        self.session_closed = asyncio.Event()
        self.reader_error: str | None = None
        self.output_rate = OUTPUT_RATE

    async def send(self, payload: dict[str, Any], detail: dict[str, Any] | None = None) -> None:
        async with self.send_lock:
            await self.websocket.send(json.dumps(payload, separators=(",", ":")))
            self.evidence.sent(str(payload.get("type") or "?"), detail)

    def _decode_audio(self, event: dict[str, Any]) -> bytes:
        encoded = event.get("delta") or event.get("audio") or ""
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            self.counts["errors"] += 1
            return b""

    async def record_received(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        audio = self._decode_audio(event) if kind == "response.output_audio.delta" else None
        self.evidence.received(event, audio_pcm_bytes=len(audio) if audio is not None else None)
        await self._apply(event, audio)

    async def _apply(self, event: dict[str, Any], audio: bytes | None) -> None:
        kind = event.get("type")
        if kind in {"input_audio_buffer.turn_started", "input_audio_buffer.committed"}:
            client_turn_id = event.get("client_turn_id")
            turn_id = event.get("turn_id")
            if isinstance(client_turn_id, int) and isinstance(turn_id, str) and turn_id:
                self.turn_ids[client_turn_id] = turn_id
            async with self.changed:
                self.changed.notify_all()
        elif kind == "response.created":
            self.response_index += 1
            self.counts["responses"] += 1
            response_id = event.get("response_id")
            if isinstance(response_id, str) and response_id:
                self.response_records[response_id] = {
                    "response_index": self.response_index,
                    "response_id": response_id,
                    "turn_id": event.get("turn_id"),
                    "text": "",
                    "status": None,
                    "reason": None,
                }
            self.response_audio.setdefault(self.response_index, bytearray())
            async with self.changed:
                self.changed.notify_all()
        elif kind == "response.output_text.delta":
            record = self.response_records.get(str(event.get("response_id")))
            if record is not None:
                record["text"] += str(event.get("delta") or "")
        elif kind == "response.output_audio.delta":
            record = self.response_records.get(str(event.get("response_id")))
            index = record["response_index"] if record is not None else self.response_index
            if index >= 0 and audio:
                self.response_audio.setdefault(index, bytearray()).extend(audio)
        elif kind == "response.function_call_arguments.done":
            await self._answer_function_call(event)
        elif kind == "response.done":
            response_id = str(event.get("response_id") or "")
            record = self.response_records.get(response_id)
            turn_id = event.get("turn_id") or (record or {}).get("turn_id")
            done = {
                "response_id": response_id or None,
                "response_index": (record or {}).get("response_index"),
                "turn_id": turn_id,
                "status": event.get("status"),
                "reason": event.get("reason"),
                "received_monotonic": time.monotonic(),
            }
            if record is not None:
                record["status"] = event.get("status")
                record["reason"] = event.get("reason")
            self.done_records.append(done)
            self.counts["response_done"] += 1
            async with self.changed:
                self.changed.notify_all()
        elif kind == "session.updated":
            output = ((event.get("session") or {}).get("audio") or {}).get("output") or {}
            output_format = output.get("format") if isinstance(output, dict) else None
            if isinstance(output_format, dict) and isinstance(output_format.get("rate"), int):
                self.output_rate = output_format["rate"]
        elif kind == "error":
            self.counts["errors"] += 1
            async with self.changed:
                self.changed.notify_all()
        elif kind == "session.closed":
            self.session_closed.set()

    async def _answer_function_call(self, event: dict[str, Any]) -> None:
        name = str(event.get("name") or "")
        raw_arguments = event.get("arguments") or "{}"
        arguments_well_formed = True
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                arguments_well_formed = False
                arguments = {"_non_object": arguments}
        except (json.JSONDecodeError, TypeError):
            arguments_well_formed = False
            arguments = {}
        result = deterministic_tool_result(name, arguments)
        call = {
            "name": name,
            "args": arguments,
            "arguments_raw": raw_arguments,
            "arguments_well_formed": arguments_well_formed,
            "call_id": event.get("call_id"),
            "response_id": event.get("response_id"),
            "turn_id": event.get("turn_id"),
            "result": result,
        }
        self.function_calls.append(call)
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":"))
        await self.send(
            {
                "type": "conversation.item.create",
                "event_id": f"phase6-tool-{len(self.function_calls)}",
                "item": {
                    "type": "function_call_output",
                    "call_id": event.get("call_id"),
                    "output": serialized,
                    "model_output": deterministic_model_output(name, arguments, result),
                },
            },
            {"tool": name, "result": result},
        )

    async def reader(self) -> None:
        try:
            async for raw in self.websocket:
                await self.record_received(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as exc:
            self.reader_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - preserved in campaign evidence
            self.reader_error = f"{type(exc).__name__}: {exc}"
            self.counts["errors"] += 1
        finally:
            async with self.changed:
                self.changed.notify_all()

    async def wait_for_turn_done(
        self,
        client_turn_id: int,
        *,
        baseline: int,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        async with self.changed:
            while True:
                server_turn_id = self.turn_ids.get(client_turn_id)
                if server_turn_id is not None:
                    for done in self.done_records[baseline:]:
                        if done.get("turn_id") == server_turn_id:
                            return done
                if self.reader_error is not None:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=remaining)
                except TimeoutError:
                    return None


class SilenceStreamer:
    def __init__(self, state: SessionState, prefix: str):
        self.state = state
        self.prefix = prefix
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name=f"{self.prefix}-silence")

    async def _run(self) -> None:
        silence = bytes(FRAME_BYTES)
        encoded = base64.b64encode(silence).decode("ascii")
        started = time.monotonic()
        index = 0
        while not self.stop_event.is_set():
            deadline = started + (index + 1) * FRAME_SECONDS
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=max(0.0, deadline - time.monotonic())
                )
                return
            except TimeoutError:
                pass
            await self.state.send(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{self.prefix}-silence-{index:05d}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": encoded,
                },
                {
                    "event_id": f"{self.prefix}-silence-{index:05d}",
                    "audio_pcm_bytes": len(silence),
                    "silence": True,
                },
            )
            index += 1

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is not None:
            await self.task


async def stream_fixture(
    state: SessionState,
    *,
    payload: bytes,
    fixture: Path,
    client_turn_id: int,
    event_prefix: str,
) -> None:
    await state.send(
        {
            "type": "input_audio_buffer.turn_start",
            "event_id": f"{event_prefix}-start",
            "client_turn_id": client_turn_id,
        },
        {"fixture": fixture.name, "client_turn_id": client_turn_id},
    )
    padded = payload + bytes((-len(payload)) % FRAME_BYTES)
    frames = [
        padded[offset : offset + FRAME_BYTES] for offset in range(0, len(padded), FRAME_BYTES)
    ]
    started = time.monotonic()
    for index, frame in enumerate(frames):
        deadline = started + index * FRAME_SECONDS
        if (delay := deadline - time.monotonic()) > 0:
            await asyncio.sleep(delay)
        event_id = f"{event_prefix}-audio-{index:05d}"
        await state.send(
            {
                "type": "input_audio_buffer.append",
                "event_id": event_id,
                "encoding": "pcm16",
                "sample_rate": INPUT_RATE,
                "channels": 1,
                "audio": base64.b64encode(frame).decode("ascii"),
            },
            {
                "event_id": event_id,
                "audio_pcm_bytes": len(frame),
                "fixture": fixture.name,
                "frame_index": index,
            },
        )
    await state.send(
        {
            "type": "input_audio_buffer.commit",
            "event_id": f"{event_prefix}-commit",
            "client_turn_id": client_turn_id,
        },
        {"fixture": fixture.name, "client_turn_id": client_turn_id},
    )


class SparkSession:
    def __init__(
        self,
        *,
        uri: str,
        output_dir: Path,
        arm: str,
        scenario: str,
        fixtures: list[Path],
        use_tools: bool,
        trailing_silence_seconds: float,
        gate_timeout_seconds: float,
        settle_seconds: float,
        close_timeout_seconds: float,
    ):
        self.uri = uri
        self.output_dir = output_dir
        self.scenario = scenario
        self.fixtures = fixtures
        self.use_tools = use_tools
        self.trailing_silence_seconds = trailing_silence_seconds
        self.gate_timeout_seconds = gate_timeout_seconds
        self.settle_seconds = settle_seconds
        self.close_timeout_seconds = close_timeout_seconds
        self.evidence_session_id = uuid.uuid4().hex[:12]
        self.meta = {
            "kind": "meta",
            "arm": arm,
            "session_id": self.evidence_session_id,
            "scenario": scenario,
            "fixtures": [fixture.name for fixture in fixtures],
            "tools": use_tools,
            "started_unix": time.time(),
        }

    async def _initial_event(self, websocket: Any, evidence: EvidenceLog) -> dict[str, Any]:
        event = json.loads(await websocket.recv())
        evidence.received(event)
        return event

    async def _configure(
        self, websocket: Any, evidence: EvidenceLog, state: SessionState
    ) -> dict[str, Any]:
        created = await self._initial_event(websocket, evidence)
        if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
            raise RuntimeError(f"server did not negotiate strict protocol v3: {created}")
        if created.get("capabilities", {}).get("input_turn_detection") != "client_smart_turn_v1":
            raise RuntimeError("server did not advertise client Smart Turn v1")
        await state.send(
            {
                "type": "session.update",
                "event_id": f"phase6-{self.evidence_session_id}-config",
                "session": {
                    "protocol_version": 3,
                    "instructions": SYSTEM_INSTRUCTION,
                    "tools": TOOLS if self.use_tools else [],
                    "capabilities": {"function_output_model_output": "client_authored_v1"},
                },
            },
            {"tools": self.use_tools},
        )
        while True:
            event = await self._initial_event(websocket, evidence)
            await state._apply(event, None)
            if event.get("type") == "session.updated":
                return created
            if event.get("type") == "error":
                raise RuntimeError(f"session update failed: {event}")

    async def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.scenario}-{self.evidence_session_id}"
        log_path = self.output_dir / f"{stem}.jsonl"
        turns: list[dict[str, Any]] = []
        session_close_timeout = False
        created: dict[str, Any] = {}
        with log_path.open("x", encoding="utf-8") as stream:
            evidence = EvidenceLog(stream)
            evidence.emit(self.meta)
            async with websockets.connect(
                self.uri,
                open_timeout=30,
                ping_interval=None,
                compression=None,
                max_size=64 * 1024 * 1024,
            ) as websocket:
                evidence.start_clock()
                state = SessionState(evidence)
                state.websocket = websocket
                created = await self._configure(websocket, evidence, state)
                reader_task = asyncio.create_task(state.reader(), name="phase6-spark-reader")
                try:
                    for index, fixture in enumerate(self.fixtures):
                        payload, rate = read_wav(fixture)
                        baseline = len(state.done_records)
                        client_turn_id = index + 1
                        prefix = f"phase6-turn-{client_turn_id}"
                        await stream_fixture(
                            state,
                            payload=payload,
                            fixture=fixture,
                            client_turn_id=client_turn_id,
                            event_prefix=prefix,
                        )
                        silence = SilenceStreamer(state, prefix)
                        silence.start()
                        started_wait = time.monotonic()
                        try:
                            done = await state.wait_for_turn_done(
                                client_turn_id,
                                baseline=baseline,
                                timeout_seconds=self.gate_timeout_seconds,
                            )
                            outcome = "response.done" if done is not None else "timeout"
                            if done is not None:
                                await asyncio.sleep(self.settle_seconds)
                        finally:
                            await silence.stop()
                        turn = {
                            "kind": "turn",
                            "index": index,
                            "fixture": fixture.name,
                            "audio_seconds": round(len(payload) / (2 * rate), 2),
                            "gated_on": "response.done",
                            "wait_seconds": round(time.monotonic() - started_wait, 2),
                            "outcome": outcome,
                            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                            "pcm_sha256": hashlib.sha256(payload).hexdigest(),
                            "client_turn_id": client_turn_id,
                            "server_turn_id": state.turn_ids.get(client_turn_id),
                            "response_id": (done or {}).get("response_id"),
                        }
                        turns.append(turn)
                        evidence.emit(turn)

                    tail = SilenceStreamer(state, "phase6-session-tail")
                    tail.start()
                    try:
                        await asyncio.sleep(self.trailing_silence_seconds)
                    finally:
                        await tail.stop()

                    await state.send({"type": "session.stop"})
                    try:
                        await asyncio.wait_for(
                            state.session_closed.wait(), timeout=self.close_timeout_seconds
                        )
                    except TimeoutError:
                        session_close_timeout = True
                finally:
                    reader_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader_task

            response_audio_records = []
            for response_index in range(state.response_index + 1):
                audio = bytes(state.response_audio.get(response_index, b""))
                wav_name = f"{stem}-r{response_index:02d}.wav"
                write_wav(self.output_dir / wav_name, audio, state.output_rate)
                record = {
                    "kind": "response_audio",
                    "response_index": response_index,
                    "wav": wav_name,
                    "rate": state.output_rate,
                }
                response_audio_records.append(record)
                evidence.emit(record)

            runtime = ((created.get("session") or {}).get("checkpoint") or {}).get(
                "runtime_provenance"
            )
            summary = {
                "kind": "summary",
                **state.counts,
                "function_calls": state.function_calls,
                "turns": len(turns),
                "turn_timeouts": sum(turn["outcome"] == "timeout" for turn in turns),
                "session_closed": state.session_closed.is_set(),
                "session_close_timeout": session_close_timeout,
                "reader_error": state.reader_error,
                "server_session_id": (created.get("session") or {}).get("id"),
                "runtime_provenance": runtime,
                "response_audio_files": len(response_audio_records),
                "trailing_silence_seconds": self.trailing_silence_seconds,
                "gate_timeout_seconds": self.gate_timeout_seconds,
                "settle_seconds": self.settle_seconds,
            }
            evidence.emit(summary)
        return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_names, use_tools, scenario_tail = SCENARIOS[args.scenario]
    fixture_root = args.fixtures_dir.expanduser().resolve()
    fixtures = [fixture_root / name for name in fixture_names]
    for fixture in fixtures:
        if not fixture.is_file():
            raise FileNotFoundError(fixture)
        read_wav(fixture)
    tail = scenario_tail if args.trailing_silence_seconds is None else args.trailing_silence_seconds
    session = SparkSession(
        uri=normalize_uri(args.server),
        output_dir=args.out.expanduser().resolve(),
        arm=args.arm,
        scenario=args.scenario,
        fixtures=fixtures,
        use_tools=use_tools,
        trailing_silence_seconds=tail,
        gate_timeout_seconds=args.gate_timeout_seconds,
        settle_seconds=args.settle_seconds,
        close_timeout_seconds=args.close_timeout_seconds,
    )
    return await session.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--arm", default="spark")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path.home() / "src/nvidia-voice-chat/turns",
    )
    parser.add_argument("--gate-timeout-seconds", type=float, default=DEFAULT_GATE_TIMEOUT_SECONDS)
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument(
        "--close-timeout-seconds", type=float, default=DEFAULT_CLOSE_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--trailing-silence-seconds",
        type=float,
        default=None,
        help="test-only override; campaign defaults are 20 s (S1/S2) and 30 s (S3)",
    )
    args = parser.parse_args()
    for name in (
        "gate_timeout_seconds",
        "settle_seconds",
        "close_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.trailing_silence_seconds is not None and args.trailing_silence_seconds < 0:
        parser.error("--trailing-silence-seconds must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
