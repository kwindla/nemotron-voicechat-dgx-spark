#!/usr/bin/env python3
"""Paced strict-v3 sustained gate with interleaved server-side typed turns."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from step4c_contract import (
    DRAIN_SECONDS as STEP4C_DRAIN_SECONDS,
)
from step4c_contract import (
    DURATION_SECONDS as STEP4C_DURATION_SECONDS,
)
from step4c_contract import (
    L1_SHA256,
    L1_TEXT,
    QUALIFICATION_FIXTURE,
    SYSTEM_INSTRUCTION,
)
from step4c_contract import (
    WARMUP_SECONDS as STEP4C_WARMUP_SECONDS,
)

from nemotron_voicechat_runtime.tool_freshness_contract import (
    evaluate_tool_cycle_cardinality,
    evaluate_tool_response_channels,
    tool_definition,
    tool_result,
)
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    TOOL_ONLY_PROMPTS as TOOL_ONLY_TYPED_PROMPTS,
)
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    TOOL_RESULT_ORDINALS,
    TOOL_VERIFICATION_WORDS,
)

FRAME_SECONDS = 0.08
INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_BYTES = int(INPUT_RATE * FRAME_SECONDS) * 2
SETTLE_FRAMES = 1_500
BLOCK_FRAMES = 750

DEFAULT_TYPED_PROMPTS = (
    "What is two plus three? Answer in one short sentence.",
    "Remember that the codeword is sapphire and acknowledge it briefly.",
    "What codeword did I ask you to remember?",
    "Use the get_current_utc_time tool and state its exact result.",
)


def typed_prompts(mode: str) -> tuple[str, ...]:
    if mode == "mixed":
        return DEFAULT_TYPED_PROMPTS
    if mode == "tool-only":
        return TOOL_ONLY_TYPED_PROMPTS
    raise ValueError(f"unsupported typed prompt mode: {mode}")


def prompt_expects_tool(mode: str, prompt: str) -> bool:
    return mode == "tool-only" or "get_current_utc_time" in prompt


def read_health(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - fixed local URL
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("active_client") is not False:
        raise RuntimeError("single-client lock is occupied or health response is invalid")
    return payload


def typed_prompt_index(
    frame_index: int, prompt_interval: int, max_typed_prompts: int | None
) -> int | None:
    """Return the scheduled prompt number while allowing silence-only tail frames."""
    if frame_index % prompt_interval:
        return None
    prompt_index = frame_index // prompt_interval
    if max_typed_prompts is not None and prompt_index >= max_typed_prompts:
        return None
    return prompt_index


def percentile(values: list[float], quantile: float) -> float | None:
    return None if not values else round(float(np.percentile(values, quantile)), 3)


def timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"frames": 0}
    return {
        "frames": len(values),
        "mean_ms": round(float(np.mean(values)), 3),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": round(max(values), 3),
        "over_80ms": sum(value > 80 for value in values),
    }


def queue_model(values: list[float]) -> dict[str, Any]:
    """Model backlog implied by serial step time relative to the 80 ms source cadence."""

    settled = values[min(SETTLE_FRAMES, len(values)) :]
    if not settled:
        return {"passed": False, "reason": "insufficient settled frames"}
    block = min(BLOCK_FRAMES, len(settled))
    first = float(np.mean(settled[:block]))
    last = float(np.mean(settled[-block:]))
    x = np.arange(len(settled), dtype=np.float64) * FRAME_SECONDS / 60
    slope = float(np.polyfit(x, np.asarray(settled), 1)[0]) if len(settled) > 1 else math.inf
    debt = 0.0
    peak_debt = 0.0
    for duration in settled:
        debt = max(0.0, debt + duration - FRAME_SECONDS * 1_000)
        peak_debt = max(peak_debt, debt)
    return {
        "passed": last <= 80.0 and slope <= 1.0 and peak_debt <= 1_000.0,
        "first_block_mean_ms": round(first, 3),
        "last_block_mean_ms": round(last, 3),
        "linear_slope_ms_per_minute": round(slope, 3),
        "peak_implied_queue_debt_ms": round(peak_debt, 3),
        "criterion": "late mean <=80ms, slope <=1ms/min, implied serial debt <=1s",
    }


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(OUTPUT_RATE)
        stream.writeframes(payload)


def resolve_response_job_id(
    response: dict[str, Any], typed_job_by_turn: dict[str, str]
) -> str | None:
    job_id = response.get("job_id")
    if isinstance(job_id, str):
        return job_id
    turn_id = response.get("turn_id")
    return typed_job_by_turn.get(turn_id) if isinstance(turn_id, str) else None


def source_completion_passes(
    *, sent_frames: int, total_frames: int, close_reason: str | None, expect_limit: bool
) -> bool:
    if expect_limit:
        return sent_frames < total_frames and close_reason == "session_position_limit"
    return sent_frames == total_frames and close_reason == "client_stop"


def expected_sender_close(
    *, receiver_done: bool, session_closed: dict[str, Any] | None, expect_limit: bool
) -> bool:
    return bool(
        receiver_done
        and expect_limit
        and (session_closed or {}).get("reason") == "session_position_limit"
    )


def classify_protocol_errors(
    events: list[dict[str, Any]],
    *,
    expect_limit: bool,
    session_closed: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    unexpected: list[dict[str, Any]] = []
    expected_limit_count = 0
    matching_close = (session_closed or {}).get("reason") == "session_position_limit"
    for event in events:
        error = event.get("error") if isinstance(event, dict) else None
        is_expected_limit = bool(
            expect_limit
            and matching_close
            and isinstance(error, dict)
            and error.get("code") == "session_position_limit"
            and error.get("fatal") is True
            and isinstance(error.get("model_frames"), int)
            and error.get("model_frames") == error.get("max_model_frames")
        )
        if is_expected_limit and expected_limit_count == 0:
            expected_limit_count += 1
        else:
            unexpected.append(event)
    return unexpected, expected_limit_count


def write_exception_evidence(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output / "error.json").write_text(encoded, encoding="utf-8")
    report_path = output / "report.json"
    if not report_path.exists():
        report_path.write_text(encoded, encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    health = read_health(getattr(args, "health_url", "http://127.0.0.1:8786/health"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "health-immediately-before.json").write_text(
        json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    events_file = (args.output / "events.jsonl").open("w", encoding="utf-8")
    silence = bytes(FRAME_BYTES)
    total_frames = round(args.duration_seconds / FRAME_SECONDS)
    prompt_interval = max(1, round(args.typed_interval_seconds / FRAME_SECONDS))
    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    typed: dict[str, dict[str, Any]] = {}
    typed_job_by_turn: dict[str, str] = {}
    expected_tool_result_by_turn: dict[str, str] = {}
    expected_text_semantic_by_turn: dict[str, bool] = {}
    responses: list[dict[str, Any]] = []
    eotr_observations: list[dict[str, Any]] = []
    last_eotr_observed_count = 0
    active_response: dict[str, Any] | None = None
    receiver_done = asyncio.Event()
    session_closed: dict[str, Any] | None = None
    typed_answer_events: dict[str, asyncio.Event] = {}
    serialized_typed_error: dict[str, str] | None = None
    qualification_fixture = getattr(args, "qualification_fixture", None)
    step4c = qualification_fixture == QUALIFICATION_FIXTURE
    prompt_mode = str(getattr(args, "typed_prompt_mode", "mixed"))
    prompt_sequence = (L1_TEXT,) if step4c else typed_prompts(prompt_mode)

    async with websockets.connect(
        args.url,
        open_timeout=30,
        ping_interval=None,
        max_size=16 * 1024 * 1024,
        write_limit=64 * 1024 * 1024,
    ) as websocket:
        created = json.loads(await websocket.recv())
        events_file.write(json.dumps(created, sort_keys=True) + "\n")
        if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
            raise RuntimeError(f"unexpected protocol: {created.get('protocol')}")
        capabilities = created.get("capabilities", {})
        if capabilities.get("input_turn_detection") != "client_smart_turn_v1":
            raise RuntimeError("server did not advertise client Smart Turn v1")
        if not capabilities.get("typed_input"):
            raise RuntimeError("server did not advertise typed input")
        tools = [] if step4c else [tool_definition(prompt_mode)]
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "qualification-session-update",
                    "session": {
                        "protocol_version": 3,
                        "instructions": (
                            SYSTEM_INSTRUCTION
                            if step4c
                            else "Answer briefly. Use tools only when requested."
                        ),
                        "tools": tools,
                    },
                }
            )
        )
        while True:
            event = json.loads(await websocket.recv())
            events_file.write(json.dumps(event, sort_keys=True) + "\n")
            if event.get("type") == "session.updated":
                break

        async def receiver() -> None:
            nonlocal active_response, last_eotr_observed_count, session_closed
            try:
                async for raw in websocket:
                    event = json.loads(raw)
                    event["client_received_monotonic_s"] = round(time.monotonic(), 6)
                    events_file.write(json.dumps(event, sort_keys=True) + "\n")
                    kind = event.get("type")
                    if kind == "voicechat.metrics":
                        metrics.append(event)
                        fc = (event.get("turn_state") or {}).get("function_calling") or {}
                        observed_count = int(fc.get("eotr_observed_count") or 0)
                        if (
                            observed_count > last_eotr_observed_count
                            and fc.get("eotr_feedback_committed") is True
                        ):
                            eotr_observations.append(
                                {
                                    "turn_id": event.get("turn_id"),
                                    "transport_frame": event.get("transport_frame"),
                                    "observed_count": observed_count,
                                    "observed": fc.get("eotr_observed"),
                                    "observed_frame": fc.get("eotr_observed_frame"),
                                    "wait_steps": fc.get("eotr_wait_steps"),
                                    "timeout": fc.get("eotr_timeout"),
                                    "eotr_token_id": fc.get("eotr_token_id"),
                                    "first_post_drain_token_id": fc.get(
                                        "eotr_first_post_drain_token_id"
                                    ),
                                    "first_post_drain_token_frame": fc.get(
                                        "eotr_first_post_drain_token_frame"
                                    ),
                                    "unexpected_token_id": fc.get("eotr_unexpected_token_id"),
                                    "unexpected_token_frame": fc.get("eotr_unexpected_token_frame"),
                                    "feedback_token_id": fc.get("eotr_feedback_token_id"),
                                    "feedback_frame": fc.get("eotr_feedback_frame"),
                                    "feedback_committed": fc.get("eotr_feedback_committed"),
                                }
                            )
                            last_eotr_observed_count = observed_count
                    elif kind == "input_text.accepted":
                        typed.setdefault(event["job_id"], {})["accepted"] = True
                    elif kind == "input_text.injection_started":
                        typed.setdefault(event["job_id"], {})["started"] = True
                    elif kind == "input_text.injection_finished":
                        typed.setdefault(event["job_id"], {})["disposition"] = event.get(
                            "disposition"
                        )
                    elif (
                        kind == "conversation.item.input_audio_transcription.completed"
                        and event.get("source") == "typed"
                        and isinstance(event.get("job_id"), str)
                        and isinstance(event.get("turn_id"), str)
                    ):
                        job_id = event["job_id"]
                        turn_id = event["turn_id"]
                        typed_job_by_turn[turn_id] = job_id
                        typed.setdefault(job_id, {})["turn_id"] = turn_id
                    elif kind == "response.created":
                        turn_id = event.get("turn_id")
                        active_response = {
                            "response_id": event.get("response_id"),
                            "turn_id": turn_id,
                            "job_id": typed_job_by_turn.get(turn_id),
                            "text": "",
                            "audio": [],
                        }
                    elif kind == "response.output_text.delta" and active_response is not None:
                        active_response["text"] += str(event.get("delta") or "")
                    elif kind == "response.output_audio.delta" and active_response is not None:
                        active_response["audio"].append(base64.b64decode(event["delta"]))
                    elif kind == "response.function_call_arguments.done":
                        if prompt_mode == "tool-only":
                            output_index = len(expected_tool_result_by_turn) % len(
                                TOOL_VERIFICATION_WORDS
                            )
                            expected_result, semantic_required, output = tool_result(
                                prompt_mode,
                                verification_word=TOOL_VERIFICATION_WORDS[output_index],
                                ordinal_word=TOOL_RESULT_ORDINALS[output_index],
                            )
                        else:
                            expected_result, semantic_required, output = tool_result(
                                prompt_mode,
                                now=time.strftime("%H:%M UTC", time.gmtime()),
                            )
                        if isinstance(event.get("turn_id"), str):
                            expected_tool_result_by_turn[event["turn_id"]] = expected_result
                            expected_text_semantic_by_turn[event["turn_id"]] = semantic_required
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": event["call_id"],
                                        "output": json.dumps(output),
                                    },
                                }
                            )
                        )
                    elif kind == "response.done" and active_response is not None:
                        active_response["done_reason"] = event.get("reason")
                        active_response["job_id"] = resolve_response_job_id(
                            active_response, typed_job_by_turn
                        )
                        payload = b"".join(active_response.pop("audio"))
                        index = len(responses) + 1
                        path = args.output / f"response-{index:03d}.wav"
                        write_wav(path, payload)
                        active_response["audio_path"] = str(path)
                        active_response["audio_bytes"] = len(payload)
                        expected_tool_result = expected_tool_result_by_turn.get(
                            active_response.get("turn_id")
                        )
                        active_response["expected_response_any"] = (
                            [expected_tool_result] if expected_tool_result else []
                        )
                        active_response["expected_text_semantic_required"] = bool(
                            expected_text_semantic_by_turn.get(
                                active_response.get("turn_id"), False
                            )
                        )
                        if active_response.get("job_id") in typed:
                            response_job_id = active_response["job_id"]
                            typed[response_job_id]["answered"] = True
                            typed[response_job_id]["response_id"] = active_response["response_id"]
                            answer_event = typed_answer_events.get(response_job_id)
                            if answer_event is not None:
                                answer_event.set()
                        responses.append(active_response)
                        active_response = None
                    elif kind == "input_text.rejected":
                        typed.setdefault(event.get("job_id", "unknown"), {})["rejected"] = (
                            event.get("reason")
                        )
                    elif kind == "error":
                        errors.append(event)
                    elif kind == "session.closed":
                        session_closed = event
                        return
            finally:
                receiver_done.set()

        receiver_task = asyncio.create_task(receiver())
        started = time.monotonic() + 0.25
        sent_frames = 0
        max_typed_prompts = getattr(args, "max_typed_prompts", None)
        serialize_typed_turns = bool(getattr(args, "serialize_typed_turns", False))
        warmup_frames = round(float(getattr(args, "warmup_seconds", 0.0)) / FRAME_SECONDS)

        async def send_serialized_typed_turns() -> None:
            nonlocal serialized_typed_error
            if max_typed_prompts is None:
                serialized_typed_error = {
                    "type": "ConfigurationError",
                    "message": "serialized typed turns require max_typed_prompts",
                }
                return
            try:
                for prompt_index in range(max_typed_prompts):
                    job_id = f"qual-serialized-{prompt_index}-{uuid.uuid4().hex[:8]}"
                    answer_event = asyncio.Event()
                    typed_answer_events[job_id] = answer_event
                    prompt = prompt_sequence[prompt_index % len(prompt_sequence)]
                    typed[job_id] = {
                        "text": prompt,
                        "expected_tool_call": prompt_expects_tool(prompt_mode, prompt),
                        "requested_frame": sent_frames,
                    }
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_text.request",
                                "job_id": job_id,
                                "text": typed[job_id]["text"],
                            }
                        )
                    )
                    await asyncio.wait_for(
                        answer_event.wait(), timeout=args.typed_response_timeout_seconds
                    )
                    if prompt_index + 1 < max_typed_prompts:
                        await asyncio.sleep(args.typed_settle_seconds)
            except Exception as exc:
                serialized_typed_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        typed_sender_task = (
            asyncio.create_task(send_serialized_typed_turns()) if serialize_typed_turns else None
        )
        try:
            for index in range(total_frames):
                if receiver_done.is_set():
                    break
                deadline = started + index * FRAME_SECONDS
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    relative_index = index - warmup_frames
                    prompt_index = (
                        None
                        if serialize_typed_turns or relative_index < 0
                        else typed_prompt_index(relative_index, prompt_interval, max_typed_prompts)
                    )
                    if prompt_index is not None:
                        prompt = prompt_sequence[prompt_index % len(prompt_sequence)]
                        job_id = f"qual-{index}-{uuid.uuid4().hex[:8]}"
                        typed[job_id] = {
                            "text": prompt,
                            "expected_tool_call": prompt_expects_tool(prompt_mode, prompt),
                            "requested_frame": index,
                        }
                        await websocket.send(
                            json.dumps(
                                {"type": "input_text.request", "job_id": job_id, "text": prompt}
                            )
                        )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"audio-{index}",
                                "encoding": "pcm16",
                                "sample_rate": INPUT_RATE,
                                "channels": 1,
                                "audio": base64.b64encode(silence).decode(),
                            }
                        )
                    )
                except websockets.ConnectionClosedOK:
                    try:
                        await asyncio.wait_for(receiver_done.wait(), timeout=1)
                    except TimeoutError:
                        pass
                    if expected_sender_close(
                        receiver_done=receiver_done.is_set(),
                        session_closed=session_closed,
                        expect_limit=args.expect_session_limit,
                    ):
                        break
                    raise
                sent_frames += 1
                await asyncio.sleep(0)
            if not receiver_done.is_set():
                await asyncio.sleep(args.drain_seconds)
                await websocket.send(json.dumps({"type": "session.stop"}))
                await asyncio.wait_for(receiver_done.wait(), timeout=30)
        finally:
            if typed_sender_task is not None and not typed_sender_task.done():
                typed_sender_task.cancel()
            if typed_sender_task is not None:
                await asyncio.gather(typed_sender_task, return_exceptions=True)
            if not receiver_task.done():
                receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
            events_file.flush()
            events_file.close()

    server_steps = [
        float(event.get("server_step_ms", 0))
        for event in metrics
        if float(event.get("server_step_ms", 0)) > 0
    ]
    queue = queue_model(server_steps)
    metric_transport_frames = [
        int(event["transport_frame"])
        for event in metrics
        if isinstance(event.get("transport_frame"), int)
    ]
    metric_sequence_complete = bool(metric_transport_frames) and metric_transport_frames == list(
        range(metric_transport_frames[0], metric_transport_frames[-1] + 1)
    )
    typed_completed = [value for value in typed.values() if value.get("disposition") == "completed"]
    answered_jobs = [value for value in typed_completed if value.get("answered")]
    unanswered_jobs = len(typed_completed) - len(answered_jobs)
    allowed_unanswered = len(typed_completed) // 15
    limit_expected = bool(args.expect_session_limit)
    close_reason = (session_closed or {}).get("reason")
    unexpected_errors, expected_limit_error_count = classify_protocol_errors(
        errors,
        expect_limit=limit_expected,
        session_closed=session_closed,
    )
    source_completion_ok = source_completion_passes(
        sent_frames=sent_frames,
        total_frames=total_frames,
        close_reason=close_reason,
        expect_limit=limit_expected,
    )
    tool_response_checks, observed_tool_response_gate = evaluate_tool_response_channels(
        responses, metrics, eotr_observations
    )
    requested_tool_prompts = sum(item.get("expected_tool_call") is True for item in typed.values())
    tool_prompt_requested = requested_tool_prompts > 0
    tool_cycle_cardinality = evaluate_tool_cycle_cardinality(
        requested_tool_prompts=requested_tool_prompts,
        tool_response_checks=tool_response_checks,
        metrics=metrics,
    )
    tool_response_channel_gate = (
        observed_tool_response_gate and tool_cycle_cardinality["passed"]
        if tool_prompt_requested
        else True
    )
    report = {
        "report_kind": "sustained_strict_v3",
        "passed": (
            not unexpected_errors
            and source_completion_ok
            and (not limit_expected or expected_limit_error_count == 1)
            and len(typed_completed) >= max(1, len(typed) - 1)
            and unanswered_jobs <= allowed_unanswered
            and len(responses) >= len(answered_jobs)
            and (not args.require_queue or queue.get("passed") is True)
            and metric_sequence_complete
            and session_closed is not None
            and serialized_typed_error is None
            and tool_response_channel_gate
        ),
        "duration_seconds": args.duration_seconds,
        "warmup_seconds": float(getattr(args, "warmup_seconds", 0.0)),
        "drain_seconds": args.drain_seconds,
        "source_frames": total_frames,
        "sent_frames": sent_frames,
        "metric_frames": len(metrics),
        "metric_sequence_complete": metric_sequence_complete,
        "server_timing": timing_stats(server_steps),
        "queue": queue,
        "queue_required": args.require_queue,
        "typed_jobs": typed,
        "typed_prompt_mode": prompt_mode,
        "typed_prompt_sequence": list(prompt_sequence),
        "qualification_fixture": qualification_fixture,
        "system_instruction": (
            SYSTEM_INSTRUCTION if step4c else "Answer briefly. Use tools only when requested."
        ),
        "prompt_sha256": L1_SHA256 if step4c else None,
        "session_id": created.get("session_id"),
        "typed_completed": len(typed_completed),
        "typed_answered": len(answered_jobs),
        "typed_unanswered": unanswered_jobs,
        "allowed_unanswered_at_fp32_base_rate": allowed_unanswered,
        "responses": responses,
        "eotr_observations": eotr_observations,
        "tool_response_checks": tool_response_checks,
        "tool_prompt_requested": tool_prompt_requested,
        "tool_cycle_cardinality": tool_cycle_cardinality,
        "tool_response_channel_gate": tool_response_channel_gate,
        "errors": errors,
        "unexpected_errors": unexpected_errors,
        "expected_session_limit_error_count": expected_limit_error_count,
        "session_closed": session_closed,
        "serialized_typed_turns": serialize_typed_turns,
        "serialized_typed_error": serialized_typed_error,
        "expected_session_position_limit": limit_expected,
        "two_frame_p95_target": {
            "target_ms": 80.0,
            "status": "documented_near_miss_not_redefined",
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--health-url", default="http://127.0.0.1:8786/health")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=1200)
    parser.add_argument("--typed-interval-seconds", type=float, default=60)
    parser.add_argument(
        "--max-typed-prompts",
        type=int,
        help="stop injecting typed turns after N prompts while paced audio continues",
    )
    parser.add_argument("--serialize-typed-turns", action="store_true")
    parser.add_argument("--typed-prompt-mode", choices=("mixed", "tool-only"), default="mixed")
    parser.add_argument("--typed-settle-seconds", type=float, default=2.0)
    parser.add_argument("--typed-response-timeout-seconds", type=float, default=40.0)
    parser.add_argument("--drain-seconds", type=float, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--qualification-fixture", choices=(QUALIFICATION_FIXTURE,))
    parser.add_argument(
        "--require-queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the sustained queue-health gate (disable only for focused structural probes)",
    )
    parser.add_argument(
        "--expect-session-limit", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.qualification_fixture == QUALIFICATION_FIXTURE:
        if (
            args.duration_seconds != STEP4C_DURATION_SECONDS
            or args.warmup_seconds != STEP4C_WARMUP_SECONDS
            or args.drain_seconds != STEP4C_DRAIN_SECONDS
            or args.max_typed_prompts != 1
            or args.serialize_typed_turns
            or args.expect_session_limit
            or args.require_queue
        ):
            parser.error(
                "Step 4c fixes 10/120/15 seconds, one paced prompt, no session limit, "
                "and no sustained queue gate"
            )
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        result = {
            "report_kind": "sustained_strict_v3",
            "passed": False,
            "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        }
        write_exception_evidence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
