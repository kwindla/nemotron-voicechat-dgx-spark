#!/usr/bin/env python3
"""Run the direct-text semantic corpus through the deterministic Pocket path."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import websockets

from nemotron_voicechat_runtime.provenance import valid_runtime_provenance
from nemotron_voicechat_runtime.semantic_corpus import (
    corpus_sha256,
    evaluate_pocket_carrier,
    evaluate_semantic_case,
    load_semantic_corpus,
)

FRAME_SECONDS = 0.08
INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_BYTES = 1_280 * 2
STRUCTURAL_EVIDENCE_KEYS = (
    "accepted",
    "typed_completed",
    "response_terminal",
    "response_brackets_balanced",
    "session_closed",
)


def structural_evidence_passed(checks: dict[str, bool]) -> bool:
    """Separate a complete protocol trace from the model's semantic verdict."""

    return all(checks.get(key) is True for key in STRUCTURAL_EVIDENCE_KEYS)


@asynccontextmanager
async def fresh_session(url: str, timeout_seconds: float = 30.0):
    deadline = time.monotonic() + timeout_seconds
    busy_rejections = 0
    while True:
        websocket = await websockets.connect(
            url, open_timeout=30, ping_interval=None, max_size=16 * 1024 * 1024
        )
        created = json.loads(await websocket.recv())
        error = created.get("error") if created.get("type") == "error" else None
        if not isinstance(error, dict) or error.get("code") != "server_busy":
            try:
                yield websocket, created, busy_rejections
            finally:
                await websocket.close()
            return
        await websocket.close()
        busy_rejections += 1
        if time.monotonic() >= deadline:
            raise TimeoutError("server did not release its single-session lease")
        await asyncio.sleep(0.1)


async def clock_until_stopped(websocket: Any, stop: asyncio.Event, prefix: str) -> None:
    silence = base64.b64encode(bytes(FRAME_BYTES)).decode()
    started = time.monotonic()
    frame = 0
    while not stop.is_set():
        deadline = started + (frame + 1) * FRAME_SECONDS
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, deadline - time.monotonic()))
            return
        except TimeoutError:
            pass
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": f"{prefix}-clock-{frame}",
                        "encoding": "pcm16",
                        "sample_rate": INPUT_RATE,
                        "channels": 1,
                        "audio": silence,
                    }
                )
            )
        except websockets.ConnectionClosed:
            return
        frame += 1


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(OUTPUT_RATE)
        output.writeframes(payload)


def runtime_provenance_from_created(created: dict[str, Any]) -> Any:
    session = created.get("session")
    checkpoint = session.get("checkpoint") if isinstance(session, dict) else None
    return checkpoint.get("runtime_provenance") if isinstance(checkpoint, dict) else None


async def run_case(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    case_dir = args.output / case["id"]
    case_dir.mkdir()
    events_path = case_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as events:
        async with fresh_session(args.url) as (websocket, created, busy_rejections):
            events.write(json.dumps(created, sort_keys=True) + "\n")
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError("server did not negotiate strict protocol v3")
            if created.get("capabilities", {}).get("typed_input") is not True:
                raise RuntimeError("server did not advertise typed input")
            provenance = runtime_provenance_from_created(created)
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": f"{case['id']}-configure",
                        "session": {
                            "protocol_version": 3,
                            "instructions": corpus["instructions"],
                            "tools": corpus["tools"],
                        },
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session configuration failed: {event}")

            job_id = f"{case['id']}-{uuid.uuid4().hex[:8]}"
            await websocket.send(
                json.dumps({"type": "input_text.request", "job_id": job_id, "text": case["prompt"]})
            )
            stop = asyncio.Event()
            clock = asyncio.create_task(clock_until_stopped(websocket, stop, case["id"]))
            response_text = ""
            response_audio: list[bytes] = []
            response_id = None
            response_done = False
            response_terminal_seen = False
            response_status = None
            response_reason = None
            input_transcript = ""
            text_done = False
            audio_done = False
            disposition = None
            accepted = False
            tool_calls: list[dict[str, Any]] = []
            loop_limit_error = False
            session_closed = None
            try:
                deadline = time.monotonic() + args.turn_timeout_seconds
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=deadline - time.monotonic()
                    )
                    event = json.loads(raw)
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    kind = event.get("type")
                    if kind == "error":
                        detail = event.get("error") or {}
                        if detail.get("code") == "function_call_loop_limit":
                            loop_limit_error = True
                            continue
                        raise RuntimeError(f"server error: {event}")
                    if kind == "conversation.item.input_audio_transcription.completed":
                        input_transcript = str(event.get("transcript") or "")
                    if kind == "input_text.accepted" and event.get("job_id") == job_id:
                        accepted = True
                    elif kind == "input_text.injection_finished" and event.get("job_id") == job_id:
                        disposition = event.get("disposition")
                    elif kind == "response.created":
                        response_id = event.get("response_id")
                    elif kind == "response.output_text.delta" and response_id is not None:
                        response_text += str(event.get("delta") or "")
                    elif kind == "response.output_audio.delta" and response_id is not None:
                        response_audio.append(base64.b64decode(event["delta"]))
                    elif kind == "response.output_text.done":
                        text_done = True
                    elif kind == "response.output_audio.done":
                        audio_done = True
                    elif kind == "response.function_call_arguments.done":
                        arguments = json.loads(event.get("arguments") or "{}")
                        tool_calls.append({"name": event.get("name"), "arguments": arguments})
                        result = corpus["tool_results"].get(
                            str(event.get("name", "")).replace("-", "_")
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": event["call_id"],
                                        "output": json.dumps(
                                            result or {"error": "unexpected tool"}, sort_keys=True
                                        ),
                                    },
                                }
                            )
                        )
                    elif kind == "response.done":
                        if event.get("response_id") != response_id:
                            raise RuntimeError("response correlation mismatch")
                        response_terminal_seen = True
                        response_status = event.get("status")
                        response_reason = event.get("reason")
                        response_done = response_status == "completed"
                    elif kind == "session.closed":
                        session_closed = event
                        if response_terminal_seen:
                            break
                    if response_done and disposition is not None:
                        break
                else:
                    raise TimeoutError(f"case timed out: {case['id']}")
            finally:
                stop.set()
                await clock
            payload = b"".join(response_audio)
            audio_path = case_dir / "response.wav"
            write_wav(audio_path, payload)
            semantic = evaluate_semantic_case(
                case, response_text=response_text, tool_calls=tool_calls
            )
            carrier = evaluate_pocket_carrier(case, transcript=input_transcript)
            semantic["response_terminal_passed"] = response_status == "completed"
            semantic["passed"] = bool(semantic["passed"] and semantic["response_terminal_passed"])
            structural = {
                "accepted": accepted,
                "typed_completed": disposition == "completed",
                "response_terminal": response_terminal_seen,
                "response_completed": response_done,
                "response_brackets_balanced": text_done and audio_done,
                "response_text_nonempty": bool(response_text.strip()),
                "response_audio_nonempty": bool(payload),
            }
            if response_status == "completed":
                await websocket.send(json.dumps({"type": "session.stop"}))
                while True:
                    event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    if event.get("type") == "session.closed":
                        session_closed = event
                        break
            structural["session_closed"] = session_closed is not None
    return {
        "case_id": case["id"],
        "runtime_provenance": provenance,
        "fresh_session_busy_retries": busy_rejections,
        "assistant_text": response_text,
        "input_transcript": input_transcript,
        "carrier": carrier,
        "tool_calls": tool_calls,
        "response_status": response_status,
        "response_reason": response_reason,
        "function_call_loop_limit": loop_limit_error,
        "audio_path": str(audio_path),
        "audio_bytes": len(payload),
        "semantic": semantic,
        "structural_checks": structural,
        "structural_passed": structural_evidence_passed(structural),
    }


def semantic_gate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if case["carrier"]["passed"]]
    passed = sum(case["semantic"]["passed"] for case in eligible)
    tool_cases = [case for case in eligible if case["semantic"]["expected_tool"]]
    safety_cases = [case for case in eligible if case["semantic"]["safety_critical"]]
    return {
        "carrier_preserved_count": len(eligible),
        "carrier_inconclusive_cases": [
            case["case_id"] for case in cases if not case["carrier"]["passed"]
        ],
        "eligible_intent_predicate_rate": passed / len(eligible) if eligible else 0.0,
        "eligible_intent_predicate_passed": bool(eligible)
        and passed / len(eligible) >= 0.95,
        "tool_name_and_arguments_passed": bool(tool_cases)
        and all(case["semantic"]["tool_passed"] for case in tool_cases),
        "safety_critical_passed": bool(safety_cases)
        and all(case["semantic"]["passed"] for case in safety_cases),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = load_semantic_corpus(args.corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    for case in corpus["cases"]:
        try:
            cases.append(await run_case(args, corpus, case))
        except Exception as exc:
            failure = {
                "case_id": case["id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            break
    provenances = [case["runtime_provenance"] for case in cases]
    provenance = provenances[0] if provenances else None
    provenance_consistent = bool(
        valid_runtime_provenance(provenance) and all(item == provenance for item in provenances)
    )
    semantic = semantic_gate(cases)
    evidence_complete = bool(
        failure is None
        and len(cases) == len(corpus["cases"])
        and provenance_consistent
        and all(case["structural_passed"] for case in cases)
    )
    report = {
        "schema": 1,
        "kind": "pocket_text_semantic_baseline",
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": corpus_sha256(args.corpus),
        "runtime_provenance": provenance,
        "runtime_provenance_consistent": provenance_consistent,
        "case_count": len(cases),
        "fresh_session_per_case": True,
        "cases": cases,
        "reference_gate": semantic,
        "failure": failure,
        "evidence_complete": evidence_complete,
    }
    report["passed"] = bool(
        evidence_complete
        and semantic["eligible_intent_predicate_passed"]
        and semantic["tool_name_and_arguments_passed"]
        and semantic["safety_critical_passed"]
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-timeout-seconds", type=float, default=120)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="return success for complete Pocket reference evidence even when semantics are red",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    succeeded = report.get("evidence_complete") if args.evidence_only else report.get("passed")
    raise SystemExit(0 if succeeded else 1)


if __name__ == "__main__":
    main()
