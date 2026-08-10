#!/usr/bin/env python3
"""Run paired typed-Pocket and client-speech tool-freshness arms."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import websockets
from generate_tool_freshness_fixtures import scenario_document

from nemotron_voicechat_runtime.artifacts import atomic_json
from nemotron_voicechat_runtime.provenance import valid_runtime_provenance
from nemotron_voicechat_runtime.qualification_audio import (
    send_audio,
    send_silence_until_stopped,
)
from nemotron_voicechat_runtime.tool_freshness_contract import (
    INPUT_EVENT_TYPES,
    PARITY_MARGIN_POLICY,
    RESPONSE_SINGLETON_EVENT_TYPES,
    SESSION_INSTRUCTIONS,
    STRUCTURAL_CONJUNCTS,
    TRAILING_SILENCE_FRAMES,
    InputModality,
    derive_structural_conjuncts,
    evaluate_tool_cycle_cardinality,
    evaluate_tool_response_channels,
    experiment_sha256,
    fixture_synthesis_identity,
    health_synthesis_identity,
)
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    canonical_sha256,
    validate_fixture_source_provenance,
    validate_tool_freshness_fixtures,
)

OUTPUT_RATE = 22_050
__all__ = ("STRUCTURAL_CONJUNCTS", "derive_structural_conjuncts")


def normalize_words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", (text or "").lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = normalize_words(reference), normalize_words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_token in enumerate(expected, 1):
        current = [row]
        for column, actual_token in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_token != actual_token),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(OUTPUT_RATE)
        output.writeframes(payload)


def record_eotr_observation(event: dict[str, Any]) -> dict[str, Any] | None:
    fc = (event.get("turn_state") or {}).get("function_calling") or {}
    if fc.get("eotr_observed") is not True or fc.get("eotr_feedback_committed") is not True:
        return None
    return {
        "turn_id": event.get("turn_id"),
        "transport_frame": event.get("transport_frame"),
        "observed_count": fc.get("eotr_observed_count"),
        "observed": fc.get("eotr_observed"),
        "observed_frame": fc.get("eotr_observed_frame"),
        "wait_steps": fc.get("eotr_wait_steps"),
        "timeout": fc.get("eotr_timeout"),
        "eotr_token_id": fc.get("eotr_token_id"),
        "first_post_drain_token_id": fc.get("eotr_first_post_drain_token_id"),
        "first_post_drain_token_frame": fc.get("eotr_first_post_drain_token_frame"),
        "unexpected_token_id": fc.get("eotr_unexpected_token_id"),
        "unexpected_token_frame": fc.get("eotr_unexpected_token_frame"),
        "feedback_token_id": fc.get("eotr_feedback_token_id"),
        "feedback_frame": fc.get("eotr_feedback_frame"),
        "feedback_committed": fc.get("eotr_feedback_committed"),
    }


async def run_arm(
    args: argparse.Namespace,
    *,
    modality: InputModality,
    scenario: dict[str, Any],
    scenario_sha256: str,
    manifest: dict[str, Any],
    fixture_pcm: tuple[bytes, ...],
    health_snapshot: dict[str, Any],
) -> dict[str, Any]:
    arm_dir = args.output / modality
    arm_dir.mkdir()
    events_path = arm_dir / "events.jsonl"
    inputs: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    eotr_observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    session_closed = None
    runtime_provenance = None
    session_id = None
    session_configuration = None
    with events_path.open("w", encoding="utf-8") as events:
        async with websockets.connect(
            args.url, open_timeout=30, ping_interval=None, max_size=16 * 1024 * 1024
        ) as websocket:
            created = json.loads(await websocket.recv())
            events.write(json.dumps(created, sort_keys=True) + "\n")
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError("server did not negotiate strict protocol v3")
            capabilities = created.get("capabilities") or {}
            if capabilities.get("input_turn_detection") != "client_smart_turn_v1":
                raise RuntimeError("server did not advertise client Smart Turn v1")
            if capabilities.get("typed_input") is not True:
                raise RuntimeError("server did not advertise typed input")
            runtime_provenance = ((created.get("session") or {}).get("checkpoint") or {}).get(
                "runtime_provenance"
            )
            if not valid_runtime_provenance(runtime_provenance):
                raise RuntimeError("server runtime provenance is invalid")
            session_id = (created.get("session") or {}).get("id")
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("server session identity is invalid")
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": f"{modality}-session-update",
                        "session": {
                            "protocol_version": 3,
                            "instructions": scenario["instructions"],
                            "tools": [scenario["tool"]],
                        },
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    session_configuration = event.get("session") or {}
                    if (
                        session_configuration.get("id") != session_id
                        or session_configuration.get("instructions") != scenario["instructions"]
                        or session_configuration.get("tools") != [scenario["tool"]]
                    ):
                        raise RuntimeError("server session configuration diverged")
                    break
                if event.get("type") == "error":
                    raise RuntimeError(f"session configuration failed: {event}")

            last_eotr_count = 0
            for ordinal, (prompt, tool_output, pcm) in enumerate(
                zip(
                    scenario["prompts"],
                    scenario["tool_outputs"],
                    fixture_pcm,
                    strict=True,
                ),
                1,
            ):
                case_id = f"tool-freshness-{ordinal}"
                job_id = f"{case_id}-{uuid.uuid4().hex[:8]}" if modality == "typed" else None
                input_record: dict[str, Any] = {
                    "case_id": case_id,
                    "ordinal": ordinal,
                    "input_modality": modality,
                    "expected_input_text": prompt,
                    "requested_text": prompt if modality == "typed" else None,
                    "client_turn_id": ordinal if modality == "speech" else None,
                    "job_id": job_id,
                    "displayed_input_text": "",
                    "event_counts": {},
                    "event_correlations": [],
                }
                if modality == "typed":
                    await websocket.send(
                        json.dumps({"type": "input_text.request", "job_id": job_id, "text": prompt})
                    )
                    clock = None
                else:
                    await send_audio(
                        websocket,
                        pcm,
                        case_id,
                        TRAILING_SILENCE_FRAMES,
                        ordinal,
                    )
                    clock_stop = asyncio.Event()
                    clock = asyncio.create_task(
                        send_silence_until_stopped(websocket, case_id, clock_stop)
                    )
                active: dict[str, Any] | None = None
                tool_calls: list[dict[str, Any]] = []
                tool_call_ids: list[str] = []
                tool_call_correlations: list[bool] = []
                applied_call_ids: list[str] = []
                text_done = False
                audio_done = False
                response_done = False
                response_event_counts: dict[str, int] = {}
                response_event_correlations: list[bool] = []
                function_outputs_sent: list[str] = []
                try:
                    deadline = time.monotonic() + args.turn_timeout_seconds
                    while time.monotonic() < deadline:
                        event = json.loads(
                            await asyncio.wait_for(
                                websocket.recv(), timeout=deadline - time.monotonic()
                            )
                        )
                        events.write(json.dumps(event, sort_keys=True) + "\n")
                        kind = event.get("type")
                        if kind in INPUT_EVENT_TYPES:
                            input_record["event_counts"][kind] = (
                                input_record["event_counts"].get(kind, 0) + 1
                            )
                        if kind in RESPONSE_SINGLETON_EVENT_TYPES:
                            response_event_counts[kind] = response_event_counts.get(kind, 0) + 1
                        if kind == "voicechat.metrics":
                            metrics.append(event)
                            observation = record_eotr_observation(event)
                            if (
                                observation
                                and isinstance(observation.get("observed_count"), int)
                                and observation["observed_count"] > last_eotr_count
                            ):
                                eotr_observations.append(observation)
                                last_eotr_count = observation["observed_count"]
                        elif kind == "error":
                            errors.append(event)
                        elif kind == "input_text.accepted":
                            correlated = modality == "typed" and event.get("job_id") == job_id
                            input_record["event_correlations"].append(correlated)
                            input_record["accepted"] = correlated
                        elif kind == "input_text.injection_started":
                            correlated = modality == "typed" and event.get("job_id") == job_id
                            input_record["event_correlations"].append(correlated)
                            input_record["started"] = correlated
                        elif kind == "input_text.injection_finished":
                            correlated = modality == "typed" and event.get("job_id") == job_id
                            input_record["event_correlations"].append(correlated)
                            if correlated:
                                input_record["disposition"] = event.get("disposition")
                            if clock is None:
                                clock_stop = asyncio.Event()
                                clock = asyncio.create_task(
                                    send_silence_until_stopped(websocket, case_id, clock_stop)
                                )
                        elif kind == "input_audio_buffer.speech_started":
                            input_record["speech_source"] = event.get("source")
                            input_record["speech_started"] = event.get("source") == (
                                "typed" if modality == "typed" else "microphone"
                            ) and (modality != "typed" or event.get("job_id") == job_id)
                            input_record["turn_id"] = event.get("turn_id")
                            input_record["event_correlations"].append(
                                bool(input_record["speech_started"])
                            )
                        elif kind == "input_audio_buffer.turn_started":
                            input_record["turn_started_ack"] = bool(
                                event.get("client_turn_id") == ordinal
                                and event.get("client_event_id") == f"{case_id}-start"
                            )
                            input_record["turn_started_turn_id"] = event.get("turn_id")
                            input_record["event_correlations"].append(
                                bool(input_record["turn_started_ack"])
                            )
                        elif kind == "input_audio_buffer.committed":
                            input_record["committed_ack"] = bool(
                                event.get("client_turn_id") == ordinal
                                and event.get("client_event_id") == f"{case_id}-commit"
                            )
                            input_record["committed_turn_id"] = event.get("turn_id")
                            input_record["event_correlations"].append(
                                bool(input_record["committed_ack"])
                            )
                        elif kind == "conversation.item.input_audio_transcription.delta":
                            input_record["event_correlations"].append(
                                bool(
                                    event.get("source")
                                    == ("typed" if modality == "typed" else "microphone")
                                    and event.get("turn_id") == input_record.get("turn_id")
                                    and (modality != "typed" or event.get("job_id") == job_id)
                                )
                            )
                            input_record["displayed_input_text"] += str(event.get("delta") or "")
                        elif kind == "conversation.item.input_audio_transcription.completed":
                            input_record["displayed_input_text"] = str(
                                event.get("transcript") or input_record["displayed_input_text"]
                            )
                            input_record["transcription_source"] = event.get("source")
                            input_record["transcription_turn_id"] = event.get("turn_id")
                            input_record["transcription_job_id"] = event.get("job_id")
                            input_record["event_correlations"].append(
                                bool(
                                    event.get("source")
                                    == ("typed" if modality == "typed" else "microphone")
                                    and event.get("turn_id") == input_record.get("turn_id")
                                    and (modality != "typed" or event.get("job_id") == job_id)
                                )
                            )
                        elif kind == "response.created":
                            correlated = bool(
                                active is None
                                and event.get("turn_id") == input_record.get("turn_id")
                                and isinstance(event.get("response_id"), str)
                                and event.get("response_id")
                            )
                            response_event_correlations.append(correlated)
                            if active is None:
                                active = {
                                    "case_id": case_id,
                                    "ordinal": ordinal,
                                    "response_id": event.get("response_id"),
                                    "turn_id": event.get("turn_id"),
                                    "text": "",
                                    "audio": [],
                                    "text_done_correlated": False,
                                    "audio_done_correlated": False,
                                    "response_done_correlated": False,
                                }
                        elif kind == "response.output_text.delta":
                            response_event_correlations.append(
                                bool(
                                    active is not None
                                    and event.get("response_id") == active.get("response_id")
                                    and event.get("turn_id") == active.get("turn_id")
                                )
                            )
                            if active is not None:
                                active["text"] += str(event.get("delta") or "")
                        elif kind == "response.output_audio.delta":
                            response_event_correlations.append(
                                bool(
                                    active is not None
                                    and event.get("response_id") == active.get("response_id")
                                    and event.get("turn_id") == active.get("turn_id")
                                )
                            )
                            if active is not None:
                                active["audio"].append(base64.b64decode(event["delta"]))
                        elif kind == "response.output_text.done":
                            text_done = True
                            if active is not None:
                                active["text_done_correlated"] = bool(
                                    event.get("response_id") == active.get("response_id")
                                    and event.get("turn_id") == active.get("turn_id")
                                )
                                response_event_correlations.append(active["text_done_correlated"])
                        elif kind == "response.output_audio.done":
                            audio_done = True
                            if active is not None:
                                active["audio_done_correlated"] = bool(
                                    event.get("response_id") == active.get("response_id")
                                    and event.get("turn_id") == active.get("turn_id")
                                )
                                response_event_correlations.append(active["audio_done_correlated"])
                        elif kind == "response.function_call_arguments.done":
                            arguments = json.loads(event.get("arguments") or "{}")
                            call_id = event.get("call_id")
                            tool_calls.append({"name": event.get("name"), "arguments": arguments})
                            if isinstance(call_id, str) and call_id:
                                tool_call_ids.append(call_id)
                            tool_call_correlations.append(
                                bool(
                                    active is not None
                                    and event.get("response_id") == active.get("response_id")
                                    and event.get("turn_id") == active.get("turn_id")
                                    and isinstance(call_id, str)
                                    and call_id
                                )
                            )
                            response_event_correlations.append(tool_call_correlations[-1])
                            serialized_function_output = json.dumps(
                                tool_output["function_output"],
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            function_outputs_sent.append(serialized_function_output)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": serialized_function_output,
                                        },
                                    }
                                )
                            )
                        elif kind == "conversation.item.function_call_output.applied":
                            call_id = event.get("call_id")
                            if isinstance(call_id, str) and call_id:
                                applied_call_ids.append(call_id)
                            response_event_correlations.append(bool(call_id in tool_call_ids))
                        elif kind == "response.done" and active is not None:
                            active["done_reason"] = event.get("reason")
                            active["done_status"] = event.get("status")
                            active["response_done_correlated"] = bool(
                                event.get("response_id") == active.get("response_id")
                                and event.get("turn_id") == active.get("turn_id")
                            )
                            response_event_correlations.append(active["response_done_correlated"])
                            response_done = True
                        if response_done and (
                            (modality == "speech" and input_record.get("committed_ack") is True)
                            or (modality == "typed" and input_record.get("disposition") is not None)
                        ):
                            break
                    else:
                        raise TimeoutError(f"tool freshness turn timed out: {case_id}")
                finally:
                    if clock is not None:
                        clock_stop.set()
                        await clock
                if active is None:
                    raise RuntimeError(f"tool freshness turn had no response: {case_id}")
                payload = b"".join(active.pop("audio"))
                audio_path = arm_dir / f"response-{ordinal:03d}.wav"
                write_wav(audio_path, payload)
                active.update(
                    {
                        "audio_path": str(audio_path),
                        "audio_bytes": len(payload),
                        "tool_calls": tool_calls,
                        "tool_call_ids": tool_call_ids,
                        "tool_call_correlated": bool(
                            len(tool_call_ids) == 1
                            and len(set(tool_call_ids)) == 1
                            and tool_call_correlations == [True]
                        ),
                        "function_output_applied": applied_call_ids == tool_call_ids,
                        "function_output_sent": (
                            function_outputs_sent[0]
                            if len(function_outputs_sent) == 1
                            else function_outputs_sent
                        ),
                        "expected_response_any": [tool_output["expected_response"]],
                        "expected_response_none": tool_output.get("excluded_responses", []),
                        "expected_text_semantic_required": True,
                        "response_brackets_balanced": text_done and audio_done,
                        "protocol_correlation": bool(
                            active.get("text_done_correlated") is True
                            and active.get("audio_done_correlated") is True
                            and active.get("response_done_correlated") is True
                            and active.get("done_status") == "completed"
                        ),
                        "event_counts": response_event_counts,
                        "event_correlations": response_event_correlations,
                    }
                )
                input_record["input_wer"] = (
                    round(word_error_rate(prompt, input_record["displayed_input_text"]), 4)
                    if modality == "speech"
                    else None
                )
                inputs.append(input_record)
                responses.append(active)
                if ordinal < 4:
                    await asyncio.sleep(args.settle_seconds)

            await websocket.send(json.dumps({"type": "session.stop"}))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                events.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.closed":
                    session_closed = event
                    break
                if event.get("type") == "error":
                    errors.append(event)

    checks, _semantic_coupled_gate = evaluate_tool_response_channels(
        responses, metrics, eotr_observations
    )
    cardinality = evaluate_tool_cycle_cardinality(
        requested_tool_prompts=4,
        tool_response_checks=checks,
        metrics=metrics,
    )
    runtime_fixture_identity = (
        fixture_synthesis_identity(manifest)
        == health_synthesis_identity(health_snapshot.get("typed_input") or {})
        and manifest["provenance"]["runtime_image_id"] == args.runtime_image_id
        and manifest["provenance"]["runtime_image_id"] == runtime_provenance.get("runtime_image_id")
        and runtime_provenance
        == ((health_snapshot.get("checkpoint") or {}).get("runtime_provenance"))
    )
    conjuncts = derive_structural_conjuncts(
        modality=modality,
        inputs=inputs,
        responses=responses,
        checks=checks,
        cardinality=cardinality,
        errors=errors,
        session_closed=session_closed,
        scenario=scenario,
        fixtures=manifest["fixtures"],
        scenario_identity=scenario_sha256 == manifest["scenario_sha256"],
        runtime_fixture_identity=runtime_fixture_identity,
    )
    report = {
        "report_kind": "tool_freshness_parity_arm_v1",
        "schema": 1,
        "input_modality": modality,
        "model_input_path": (
            "server_typed_pocket" if modality == "typed" else "client_pocket_fixture_pcm"
        ),
        "scenario": scenario,
        "scenario_sha256": scenario_sha256,
        "experiment_sha256": experiment_sha256(
            scenario_sha256=scenario_sha256,
            fixture_manifest_sha256=manifest["sha256"],
        ),
        "fixture_manifest_path": str(args.fixture_manifest.resolve()),
        "fixture_manifest": manifest,
        "health_snapshot": health_snapshot,
        "live_typed_input_health": health_snapshot["typed_input"],
        "health_runtime_provenance": (
            (health_snapshot.get("checkpoint") or {}).get("runtime_provenance")
        ),
        "runtime_provenance": runtime_provenance,
        "session_id": session_id,
        "session_configuration": session_configuration,
        "inputs": inputs,
        "responses": responses,
        "metrics": metrics,
        "eotr_observations": eotr_observations,
        "tool_response_checks": checks,
        "tool_cycle_cardinality": cardinality,
        "structural_conjuncts": conjuncts,
        "structural_passed": all(conjuncts.values()),
        "errors": errors,
        "session_closed": session_closed,
    }
    report["passed"] = report["structural_passed"]
    atomic_json(arm_dir / "report.json", report)
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    scenario = scenario_document()
    if scenario["instructions"] != SESSION_INSTRUCTIONS:
        raise RuntimeError("tool freshness session instructions drifted")
    scenario_sha256 = canonical_sha256(scenario)
    fixture_dir = args.fixture_manifest.resolve().parent
    manifest, fixture_pcm = validate_tool_freshness_fixtures(
        fixture_dir,
        expected_runtime_image_id=args.runtime_image_id,
        expected_scenario_sha256=scenario_sha256,
    )
    validate_fixture_source_provenance(manifest, project_root=Path(__file__).resolve().parents[2])

    def fetch_health() -> dict[str, Any]:
        with urllib.request.urlopen(args.health_url, timeout=10) as response:
            return json.load(response)

    async def wait_for_idle() -> dict[str, Any]:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            health = await asyncio.to_thread(fetch_health)
            if health.get("active_client") is False:
                return health
            await asyncio.sleep(0.2)
        raise TimeoutError("server did not become idle between parity arms")

    execution_order = (
        ("speech", "typed") if args.nonpromotable_reverse_order else ("typed", "speech")
    )
    reports = {}
    for modality in execution_order:
        health = await wait_for_idle()
        live_typed_input = health.get("typed_input") or {}
        if live_typed_input.get("ready") is not True:
            raise RuntimeError("live typed-input worker is not ready")
        reports[modality] = await run_arm(
            args,
            modality=modality,
            scenario=scenario,
            scenario_sha256=scenario_sha256,
            manifest=manifest,
            fixture_pcm=fixture_pcm,
            health_snapshot=health,
        )
    if reports["typed"]["session_id"] == reports["speech"]["session_id"]:
        raise RuntimeError("parity arms did not use distinct fresh sessions")
    if reports["typed"]["runtime_provenance"] != reports["speech"]["runtime_provenance"]:
        raise RuntimeError("parity arms used different runtime provenance")
    experiment_identity = experiment_sha256(
        scenario_sha256=scenario_sha256,
        fixture_manifest_sha256=manifest["sha256"],
    )
    if any(report.get("experiment_sha256") != experiment_identity for report in reports.values()):
        raise RuntimeError("parity arm experiment identity diverged")
    result = {
        "schema": 1,
        "kind": (
            "tool_freshness_reverse_order_diagnostic_runtime"
            if args.nonpromotable_reverse_order
            else "tool_freshness_parity_runtime"
        ),
        "comparison_policy": PARITY_MARGIN_POLICY,
        "execution_order": list(execution_order),
        "scenario": scenario,
        "scenario_sha256": scenario_sha256,
        "fixture_manifest_sha256": manifest["sha256"],
        "experiment_sha256": experiment_identity,
        "session_ids": [reports[modality]["session_id"] for modality in execution_order],
        "runtime_provenance": reports["typed"]["runtime_provenance"],
        "arms": {
            modality: {
                "report": str((args.output / modality / "report.json").resolve()),
                "structural_passed": report["structural_passed"],
            }
            for modality, report in reports.items()
        },
        "passed": all(report["structural_passed"] for report in reports.values()),
    }
    if args.nonpromotable_reverse_order:
        result.pop("comparison_policy")
        result["promotion_eligible"] = False
    atomic_json(args.output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--health-url", default="http://127.0.0.1:8786/health")
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-timeout-seconds", type=float, default=60)
    parser.add_argument("--settle-seconds", type=float, default=2)
    parser.add_argument("--nonpromotable-reverse-order", action="store_true")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": 1,
            "kind": (
                "tool_freshness_reverse_order_diagnostic_runtime"
                if args.nonpromotable_reverse_order
                else "tool_freshness_parity_runtime"
            ),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.nonpromotable_reverse_order:
            report["promotion_eligible"] = False
        atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
