#!/usr/bin/env python3
"""Gate tool-result freshness with a directive-phrased, deterministic fixture.

The retained conversational replay remains the end-to-end acceptance test, but
whether its casual requests elicit tools is a model-propensity observation.  This
fixture makes the tool affordance explicit, injects fixed UTC and Pacific results,
and gates the invariant that no assistant text after the Pacific result reports the
superseded UTC value.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import websockets

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nemotron_voicechat_runtime.qualification_audio import (  # noqa: E402
    send_silence_until_stopped,
)

PRE_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state its result accurately and continue the conversation normally."""

POST_INSTRUCTION = """You are a helpful voice assistant in a live conversation.
Keep answers concise, natural, and easy to understand when spoken aloud.
Use a tool only when the user's request matches that tool's purpose. Never use an
available tool as a substitute for an unrelated or unavailable capability. After a
tool returns, state the MOST RECENT result accurately and continue the conversation
normally. Never restate an older tool result after a newer result supersedes it."""

TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            "Get the current clock time in the time zone this conversation is using."
        ),
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
                    "description": "IANA time zone name or common spoken name.",
                }
            },
            "required": ["timezone"],
        },
    },
]

PROMPTS = (
    "Use the current time tool now, then state the exact fresh time from its result.",
    "Use the time zone tool now to set Pacific time, then state the exact result.",
    "Invoke the current time tool for this turn, even though it was used earlier, "
    "and state the exact fresh time it returns.",
    "State the most recent current time result again without calling a tool.",
)
EXPECTED_CALLS: tuple[str | None, ...] = (
    "get_current_time",
    "set_timezone",
    "get_current_time",
    None,
)
UTC_RESULT = "It is currently 8:55 PM UTC."
TIMEZONE_RESULT = "The time zone is now America/Los_Angeles."
PACIFIC_RESULT = "It is currently 1:55 PM PDT."
TOOL_OUTPUTS = (
    ({"timezone": "UTC", "spoken": "8:55 PM UTC"}, UTC_RESULT),
    ({"ok": True, "timezone": "America/Los_Angeles"}, TIMEZONE_RESULT),
    (
        {"timezone": "America/Los_Angeles", "spoken": "1:55 PM PDT"},
        PACIFIC_RESULT,
    ),
)


def atomic_json(path: Path, value: Any) -> None:
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def normalized_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (value or "").casefold())


def contains_phrase(text: str, phrase: str) -> bool:
    haystack = normalized_tokens(text)
    needle = normalized_tokens(phrase)
    return bool(needle) and any(
        haystack[offset : offset + len(needle)] == needle
        for offset in range(len(haystack) - len(needle) + 1)
    )


def reports_utc_result(text: str) -> bool:
    return contains_phrase(text, "8:55 PM UTC") or contains_phrase(
        text, "eight fifty five PM UTC"
    )


def reports_pacific_result(text: str) -> bool:
    return any(
        contains_phrase(text, variant)
        for variant in (
            "1:55 PM PDT",
            "one fifty five PM PDT",
            "1:55 PM Pacific Daylight Time",
            "one fifty five PM Pacific Daylight Time",
        )
    )


def expected_call_matches(call: dict[str, Any], expected_name: str) -> bool:
    if call.get("name") != expected_name:
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    if expected_name == "get_current_time":
        return arguments == {}
    timezone = arguments.get("timezone")
    return isinstance(timezone, str) and timezone.casefold() in {
        "pacific",
        "pacific time",
        "america/los_angeles",
    }


def evaluate_replicate(turns: list[dict[str, Any]], errors: list[Any]) -> dict[str, Any]:
    observed = [
        call["name"]
        for turn in turns
        for call in turn.get("tool_calls", [])
    ]
    expected = [name for name in EXPECTED_CALLS if name is not None]
    turn_calls_match = len(turns) == len(EXPECTED_CALLS) and all(
        (
            not turn.get("tool_calls")
            if expected_name is None
            else len(turn.get("tool_calls", [])) == 1
            and expected_call_matches(turn["tool_calls"][0], expected_name)
        )
        for turn, expected_name in zip(turns, EXPECTED_CALLS, strict=True)
    )
    eligible = turns[2:] if len(turns) >= 4 and turns[2].get("output_applied") else []
    stale = [
        {"ordinal": turn["ordinal"], "assistant_text": turn.get("assistant_text", "")}
        for turn in eligible
        if reports_utc_result(turn.get("assistant_text", ""))
    ]
    fresh = [
        turn["ordinal"]
        for turn in eligible
        if reports_pacific_result(turn.get("assistant_text", ""))
    ]
    required_fresh_ordinals = {3, 4}
    missing_fresh = [
        {"ordinal": turn["ordinal"], "assistant_text": turn.get("assistant_text", "")}
        for turn in eligible
        if turn["ordinal"] in required_fresh_ordinals
        and turn["ordinal"] not in fresh
    ]
    lifecycle_complete = len(turns) == len(PROMPTS) and all(
        turn.get("response_status") == "completed" for turn in turns
    )
    freshness_gate_passed = (
        len(eligible) == len(required_fresh_ordinals)
        and not stale
        and required_fresh_ordinals.issubset(fresh)
    )
    return {
        "expected_sequence": expected,
        "observed_sequence": observed,
        "sequence_elicited": observed == expected and turn_calls_match,
        "per_turn": [
            {
                "ordinal": turn["ordinal"],
                "utterance": turn["utterance"],
                "call_requested": bool(turn.get("tool_calls")),
                "calls": turn.get("tool_calls", []),
                "assistant_text": turn.get("assistant_text", ""),
            }
            for turn in turns
        ],
        "eligible_responses": len(eligible),
        "stale_responses": stale,
        "fresh_response_ordinals": fresh,
        "missing_fresh_result_responses": missing_fresh,
        "stale_rate": round(len(stale) / len(eligible), 4) if eligible else None,
        "freshness_gate_passed": freshness_gate_passed,
        "lifecycle_complete": lifecycle_complete,
        "errors_absent": not errors,
        "passed": bool(
            observed == expected
            and turn_calls_match
            and lifecycle_complete
            and not errors
            and freshness_gate_passed
        ),
    }


async def run_turn(
    websocket: Any,
    events: Any,
    *,
    ordinal: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = PROMPTS[ordinal - 1]
    job_id = f"directive-{ordinal}-{uuid.uuid4().hex[:8]}"
    await websocket.send(
        json.dumps({"type": "input_text.request", "job_id": job_id, "text": prompt})
    )
    turn: dict[str, Any] = {
        "ordinal": ordinal,
        "utterance": prompt,
        "job_id": job_id,
        "assistant_text": "",
        "tool_calls": [],
        "output_applied": False,
        "responses": [],
    }
    response_ids: list[str] = []
    response_turn_ids: dict[str, str | None] = {}
    clock = None
    clock_stop = asyncio.Event()
    deadline = time.monotonic() + timeout_seconds
    no_call_deadline: float | None = None
    try:
        while time.monotonic() < deadline:
            receive_deadline = min(
                deadline,
                no_call_deadline if no_call_deadline is not None else deadline,
            )
            try:
                event = json.loads(
                    await asyncio.wait_for(
                        websocket.recv(), timeout=receive_deadline - time.monotonic()
                    )
                )
            except TimeoutError:
                if no_call_deadline is not None:
                    turn["response_status"] = turn["responses"][-1]["status"]
                    return turn
                raise
            events.write(json.dumps(event, sort_keys=True) + "\n")
            kind = event.get("type")
            if kind == "input_text.injection_finished" and event.get("job_id") == job_id:
                turn["input_disposition"] = event.get("disposition")
                if clock is None:
                    clock = asyncio.create_task(
                        send_silence_until_stopped(websocket, f"directive-{ordinal}", clock_stop)
                    )
            elif kind == "conversation.item.input_audio_transcription.completed":
                if event.get("job_id") == job_id:
                    turn["recognized_utterance"] = event.get("transcript")
            elif kind == "response.created":
                response_id = event.get("response_id")
                if isinstance(response_id, str):
                    response_ids.append(response_id)
                    response_turn_ids[response_id] = event.get("turn_id")
            elif kind == "response.output_text.delta" and event.get(
                "response_id"
            ) in set(response_ids):
                turn["assistant_text"] += str(event.get("delta") or "")
            elif kind == "response.function_call_arguments.done":
                call_id = event.get("call_id")
                try:
                    arguments = json.loads(event.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_invalid_json": event.get("arguments")}
                turn["tool_calls"].append(
                    {"name": event.get("name"), "arguments": arguments, "call_id": call_id}
                )
                output_index = min(len(turn["tool_calls"]) - 1, len(TOOL_OUTPUTS) - 1)
                if ordinal <= len(TOOL_OUTPUTS):
                    output_index = ordinal - 1
                payload, model_output = TOOL_OUTPUTS[output_index]
                await websocket.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(payload, sort_keys=True),
                                "model_output": model_output,
                            },
                        }
                    )
                )
            elif kind == "conversation.item.function_call_output.applied":
                if event.get("call_id") in {
                    call.get("call_id") for call in turn["tool_calls"]
                }:
                    turn["output_applied"] = True
            elif kind == "response.done" and event.get("response_id") in set(response_ids):
                turn["responses"].append(
                    {
                        "response_id": event.get("response_id"),
                        "turn_id": response_turn_ids.get(event.get("response_id")),
                        "status": event.get("status"),
                        "reason": event.get("reason"),
                        "after_output_applied": turn["output_applied"],
                    }
                )
                if EXPECTED_CALLS[ordinal - 1] is not None and turn["output_applied"]:
                    turn["response_status"] = event.get("status")
                    return turn
                if EXPECTED_CALLS[ordinal - 1] is None:
                    no_call_deadline = time.monotonic() + 3.0
            elif kind == "error":
                raise RuntimeError(f"server error during directive turn: {event}")
        raise TimeoutError(f"directive turn {ordinal} timed out")
    finally:
        clock_stop.set()
        if clock is not None:
            await clock


async def run_replicate(args: argparse.Namespace, replicate: int) -> dict[str, Any]:
    replicate_dir = args.output / f"replicate-{replicate:02d}"
    replicate_dir.mkdir()
    events_path = replicate_dir / "events.jsonl"
    instructions = POST_INSTRUCTION if args.arm == "post" else PRE_INSTRUCTION
    turns: list[dict[str, Any]] = []
    errors: list[str] = []
    created: dict[str, Any] = {}
    session_closed = False
    with events_path.open("x", encoding="utf-8") as events:
        try:
            async with websockets.connect(
                args.url,
                open_timeout=30,
                ping_interval=None,
                max_size=64 * 1024 * 1024,
            ) as websocket:
                created = json.loads(await websocket.recv())
                events.write(json.dumps(created, sort_keys=True) + "\n")
                if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                    raise RuntimeError("server did not negotiate protocol v3")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "event_id": f"directive-{args.arm}-{replicate}",
                            "session": {
                                "protocol_version": 3,
                                "instructions": instructions,
                                "tools": TOOLS,
                                "capabilities": {
                                    "function_output_model_output": "client_authored_v1"
                                },
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
                        raise RuntimeError(f"session update failed: {event}")
                for ordinal in range(1, len(PROMPTS) + 1):
                    turns.append(
                        await run_turn(
                            websocket,
                            events,
                            ordinal=ordinal,
                            timeout_seconds=args.turn_timeout_seconds,
                        )
                    )
                    await asyncio.sleep(args.settle_seconds)
                await websocket.send(json.dumps({"type": "session.stop"}))
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    event = json.loads(
                        await asyncio.wait_for(
                            websocket.recv(), timeout=deadline - time.monotonic()
                        )
                    )
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                    if event.get("type") == "session.closed":
                        session_closed = True
                        break
                    if event.get("type") == "error":
                        errors.append(json.dumps(event, sort_keys=True))
        except Exception as exc:  # noqa: BLE001 - retained as fixture evidence
            errors.append(repr(exc))
    evaluation = evaluate_replicate(turns, errors)
    report = {
        "schema": "nemotron_voicechat.session_directive_freshness_replicate.v1",
        "arm": args.arm,
        "replicate": replicate,
        "instructions": instructions,
        "runtime": {
            "session_id": (created.get("session") or {}).get("id"),
            "checkpoint": (created.get("session") or {}).get("checkpoint"),
        },
        "turns": turns,
        "session_closed": session_closed,
        "errors": errors,
        "evaluation": evaluation,
        "passed": bool(evaluation["passed"] and session_closed),
    }
    atomic_json(replicate_dir / "report.json", report)
    return report


async def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing stale directive output: {args.output}")
    args.output.mkdir(parents=True)
    reports = []
    for replicate in range(1, args.replicates + 1):
        reports.append(await run_replicate(args, replicate))
        if replicate < args.replicates:
            await asyncio.sleep(args.between_replicates_seconds)
    eligible = sum(r["evaluation"]["eligible_responses"] for r in reports)
    stale = sum(len(r["evaluation"]["stale_responses"]) for r in reports)
    passed_replicates = sum(1 for r in reports if r["passed"])
    summary = {
        "schema": "nemotron_voicechat.session_directive_freshness_arm.v1",
        "arm": args.arm,
        "classification": "hard_i4_freshness_fixture",
        "fixture_sha256": hashlib.sha256(
            json.dumps(
                {"prompts": PROMPTS, "tools": TOOLS, "outputs": TOOL_OUTPUTS},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "replicates": len(reports),
        "passed_replicates": passed_replicates,
        "sequence_elicitation_rate": round(
            sum(1 for r in reports if r["evaluation"]["sequence_elicited"])
            / len(reports),
            4,
        ),
        "eligible_responses": eligible,
        "stale_responses": stale,
        "stale_rate": round(stale / eligible, 4) if eligible else None,
        "freshness_gate_passed": bool(
            eligible and all(r["evaluation"]["freshness_gate_passed"] for r in reports)
        ),
        "passed": passed_replicates == len(reports),
        "replicate_reports": [
            str((args.output / f"replicate-{index:02d}" / "report.json").resolve())
            for index in range(1, len(reports) + 1)
        ],
    }
    atomic_json(args.output / "summary.json", summary)
    return summary


def compare_reports(pre_path: Path, post_path: Path, output: Path) -> dict[str, Any]:
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))
    if pre.get("arm") != "pre" or post.get("arm") != "post":
        raise RuntimeError("directive comparison requires pre and post arm reports")
    pre_rate = pre.get("stale_rate")
    post_rate = post.get("stale_rate")
    comparable = (
        isinstance(pre_rate, (int, float))
        and not isinstance(pre_rate, bool)
        and isinstance(post_rate, (int, float))
        and not isinstance(post_rate, bool)
    )
    changed = bool(comparable and pre_rate != post_rate)
    improved = bool(comparable and post_rate < pre_rate)
    result = {
        "schema": "nemotron_voicechat.session_directive_freshness_comparison.v1",
        "pre": {
            "report": str(pre_path.resolve()),
            "stale_rate": pre_rate,
            "sequence_elicitation_rate": pre.get("sequence_elicitation_rate"),
        },
        "post": {
            "report": str(post_path.resolve()),
            "stale_rate": post_rate,
            "sequence_elicitation_rate": post.get("sequence_elicitation_rate"),
        },
        "rate_changed": changed,
        "rate_improved": improved,
        "post_freshness_gate_passed": post.get("freshness_gate_passed") is True,
        "i4_fix_supported": bool(improved and post.get("freshness_gate_passed") is True),
        "finding": (
            "post instruction reduced the stale-result rate"
            if improved
            else "stale-result rate was unchanged; do not claim I4 fixed"
            if comparable and not changed
            else "post instruction did not reduce the stale-result rate"
            if comparable
            else "pre/post stale rate was not measurable"
        ),
    }
    atomic_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--arm", choices=("pre", "post"), required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    arm.add_argument("--replicates", type=int, default=3)
    arm.add_argument("--turn-timeout-seconds", type=float, default=60.0)
    arm.add_argument("--settle-seconds", type=float, default=1.0)
    arm.add_argument("--between-replicates-seconds", type=float, default=2.0)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--pre-report", type=Path, required=True)
    compare.add_argument("--post-report", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "arm":
        if args.replicates <= 0:
            raise ValueError("replicates must be positive")
        report = asyncio.run(run_arm(args))
    else:
        report = compare_reports(args.pre_report, args.post_report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.command == "arm" and args.arm == "post" and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
