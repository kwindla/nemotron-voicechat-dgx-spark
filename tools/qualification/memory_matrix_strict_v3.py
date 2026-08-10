#!/usr/bin/env python3
"""Qualify conversational memory across every supported prompt-input path."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import websockets

from nemotron_voicechat_runtime.provenance import valid_runtime_provenance

INPUT_RATE = 16_000
OUTPUT_RATE = 22_050
FRAME_SAMPLES = 1_280
FRAME_SECONDS = FRAME_SAMPLES / INPUT_RATE
DEFAULT_INSTRUCTIONS = "Answer briefly and preserve conversational memory."

InputModality = Literal["voice", "text"]
SeedModality = Literal["voice", "text", "system", "none"]


class MemoryCase(NamedTuple):
    case_id: str
    seed: SeedModality
    recall: InputModality
    secret: str | None
    seed_text: str | None
    recall_text: str
    seed_audio_path: Path | None
    recall_audio_path: Path | None


@asynccontextmanager
async def acquire_fresh_session(url: str, *, timeout_seconds: float = 30.0):
    """Acquire the server's single session lease with a bounded handoff retry."""
    deadline = time.monotonic() + timeout_seconds
    busy_rejections = 0
    while True:
        websocket = await websockets.connect(
            url,
            open_timeout=min(30.0, max(1.0, deadline - time.monotonic())),
            ping_interval=None,
            max_size=16 * 1024 * 1024,
        )
        created = json.loads(await websocket.recv())
        error = created.get("error") if created.get("type") == "error" else None
        if not isinstance(error, dict) or error.get("code") != "server_busy":
            try:
                yield websocket, created, busy_rejections
            finally:
                await websocket.close()
            return
        busy_rejections += 1
        await websocket.close()
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "server did not release its single-session lease within "
                f"{timeout_seconds:.1f}s"
            )
        await asyncio.sleep(0.1)


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def contains_secret(text: str, secret: str) -> bool:
    return secret.casefold() in normalize(text)


def session_runtime_provenance(created: dict[str, Any]) -> Any:
    session = created.get("session")
    checkpoint = session.get("checkpoint") if isinstance(session, dict) else None
    return checkpoint.get("runtime_provenance") if isinstance(checkpoint, dict) else None


def system_instructions(secret: str | None) -> str:
    if secret is None:
        return DEFAULT_INSTRUCTIONS
    return f"{DEFAULT_INSTRUCTIONS} The codeword supplied by the system is {secret}."


def load_cases(manifest: dict[str, Any]) -> tuple[MemoryCase, ...]:
    matrix = manifest.get("memory_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("fixture manifest has no parameterized memory matrix")
    raw_cases = matrix.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("memory matrix cases must be a list")
    expected_pairs = {
        (seed, recall) for seed in ("voice", "text", "system") for recall in ("voice", "text")
    }
    observed_pairs = {
        (str(case.get("seed_modality")), str(case.get("recall_modality"))) for case in raw_cases
    }
    if observed_pairs != expected_pairs or len(raw_cases) != 6:
        raise ValueError("memory matrix must cover each seeded modality pair exactly once")
    secrets = [str(case.get("secret") or "").casefold() for case in raw_cases]
    if any(not secret or len(normalize(secret)) != 1 for secret in secrets):
        raise ValueError("each memory case secret must be one non-empty word")
    if len(set(secrets)) != len(secrets):
        raise ValueError("memory matrix secrets must be unique per case")

    cases: list[MemoryCase] = []
    for raw, secret in zip(raw_cases, secrets, strict=True):
        seed = str(raw["seed_modality"])
        recall = str(raw["recall_modality"])
        seed_audio = raw.get("seed_audio_path")
        recall_audio = raw.get("recall_audio_path")
        if seed == "voice" and not seed_audio:
            raise ValueError(f"{raw['case_id']}: voice seed has no audio fixture")
        if recall == "voice" and not recall_audio:
            raise ValueError(f"{raw['case_id']}: voice recall has no audio fixture")
        recall_text = str(raw.get("recall_text") or "")
        if any(contains_secret(recall_text, candidate) for candidate in secrets):
            raise ValueError(f"{raw['case_id']}: recall prompt leaks a matrix secret")
        cases.append(
            MemoryCase(
                case_id=str(raw["case_id"]),
                seed=seed,
                recall=recall,
                secret=secret,
                seed_text=str(raw.get("seed_text") or ""),
                recall_text=recall_text,
                seed_audio_path=Path(seed_audio) if seed_audio else None,
                recall_audio_path=Path(recall_audio) if recall_audio else None,
            )
        )

    negative = matrix.get("negative_control")
    if not isinstance(negative, dict):
        raise ValueError("memory matrix has no unseeded negative control")
    negative_recall = str(negative.get("recall_text") or "")
    if any(contains_secret(negative_recall, secret) for secret in secrets):
        raise ValueError("negative-control recall prompt leaks a matrix secret")
    cases.append(
        MemoryCase(
            case_id=str(negative.get("case_id") or "unseeded-to-text"),
            seed="none",
            recall=str(negative.get("recall_modality") or "text"),
            secret=None,
            seed_text=None,
            recall_text=negative_recall,
            seed_audio_path=None,
            recall_audio_path=(
                Path(negative["recall_audio_path"]) if negative.get("recall_audio_path") else None
            ),
        )
    )
    return tuple(cases)


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = normalize(reference), normalize(hypothesis)
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


def read_fixture(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        values = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    if width != 2:
        raise ValueError(f"{path}: expected PCM16 WAV")
    samples = values.astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_RATE:
        count = round(samples.size * INPUT_RATE / rate)
        samples = np.interp(
            np.arange(count, dtype=np.float64) * rate / INPUT_RATE,
            np.arange(samples.size, dtype=np.float64),
            samples,
        ).astype(np.float32)
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def write_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(OUTPUT_RATE)
        output.writeframes(payload)


async def send_voice_turn(
    websocket: Any,
    *,
    payload: bytes,
    event_prefix: str,
    client_turn_id: int,
) -> None:
    frame_bytes = FRAME_SAMPLES * 2
    padded = payload + bytes((-len(payload)) % frame_bytes)
    frames = [
        padded[offset : offset + frame_bytes] for offset in range(0, len(padded), frame_bytes)
    ]
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.turn_start",
                "event_id": f"{event_prefix}-start",
                "client_turn_id": client_turn_id,
            }
        )
    )
    for index, frame in enumerate(frames):
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{event_prefix}-audio-{index}",
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
                "event_id": f"{event_prefix}-commit",
                "client_turn_id": client_turn_id,
            }
        )
    )


async def advance_until_stopped(websocket: Any, *, event_prefix: str, stop: asyncio.Event) -> None:
    """Supply model clock positions; the server owns framing and EOU settlement."""

    frame = bytes(FRAME_SAMPLES * 2)
    started = time.monotonic()
    index = 0
    while not stop.is_set():
        deadline = started + (index + 1) * FRAME_SECONDS
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, deadline - time.monotonic()))
            return
        except TimeoutError:
            pass
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"{event_prefix}-clock-{index}",
                    "encoding": "pcm16",
                    "sample_rate": INPUT_RATE,
                    "channels": 1,
                    "audio": base64.b64encode(frame).decode(),
                }
            )
        )
        index += 1


async def receive_turn(
    websocket: Any,
    *,
    events_file: Any,
    event_prefix: str,
    expected_source: InputModality,
    expected_job_id: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    stop = asyncio.Event()
    clock = asyncio.create_task(
        advance_until_stopped(websocket, event_prefix=event_prefix, stop=stop)
    )
    response: dict[str, Any] | None = None
    response_done = False
    output_text_done = False
    output_audio_done = False
    response_done_reason: str | None = None
    transcript = ""
    typed_disposition: str | None = None
    accepted = expected_source == "voice"
    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(websocket.recv(), timeout=deadline - time.monotonic())
            event = json.loads(raw)
            events_file.write(json.dumps(event, sort_keys=True) + "\n")
            kind = event.get("type")
            if kind == "error":
                raise RuntimeError(f"server error: {event}")
            if kind == "input_text.accepted" and event.get("job_id") == expected_job_id:
                accepted = True
            elif kind == "input_text.injection_finished" and event.get("job_id") == expected_job_id:
                typed_disposition = str(event.get("disposition"))
            elif kind == "conversation.item.input_audio_transcription.completed":
                source = "text" if event.get("source") == "typed" else "voice"
                if source == expected_source and (
                    expected_job_id is None or event.get("job_id") == expected_job_id
                ):
                    transcript = str(event.get("transcript") or "")
            elif kind == "response.created":
                if response is not None:
                    raise RuntimeError(f"overlapping responses in {event_prefix}")
                response = {
                    "response_id": event.get("response_id"),
                    "turn_id": event.get("turn_id"),
                    "text": "",
                    "audio": [],
                }
            elif kind == "response.output_text.delta" and response is not None:
                response["text"] += str(event.get("delta") or "")
            elif kind == "response.output_audio.delta" and response is not None:
                response["audio"].append(base64.b64decode(event["delta"]))
            elif kind == "response.output_text.done" and response is not None:
                output_text_done = True
            elif kind == "response.output_audio.done" and response is not None:
                output_audio_done = True
            elif kind == "response.done" and response is not None:
                if event.get("response_id") != response["response_id"]:
                    raise RuntimeError(f"response correlation mismatch in {event_prefix}")
                if event.get("status") != "completed":
                    raise RuntimeError(f"response did not complete in {event_prefix}: {event}")
                response_done_reason = str(event.get("reason") or "")
                response_done = True
            if (
                response_done
                and response is not None
                and (expected_source == "voice" or typed_disposition is not None)
            ):
                audio = b"".join(response.pop("audio"))
                return {
                    **response,
                    "audio": audio,
                    "transcript": transcript,
                    "input_accepted": accepted,
                    "typed_disposition": typed_disposition,
                    "response_done_reason": response_done_reason,
                    "response_brackets_balanced": output_text_done and output_audio_done,
                }
        raise TimeoutError(f"turn timed out: {event_prefix}")
    finally:
        stop.set()
        await clock


async def drive_turn(
    websocket: Any,
    *,
    modality: InputModality,
    text: str,
    audio: bytes,
    case_id: str,
    turn_number: int,
    events_file: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    event_prefix = f"{case_id}-turn-{turn_number}"
    job_id: str | None = None
    if modality == "voice":
        await send_voice_turn(
            websocket,
            payload=audio,
            event_prefix=event_prefix,
            client_turn_id=turn_number,
        )
    else:
        job_id = f"{event_prefix}-{uuid.uuid4().hex[:8]}"
        await websocket.send(
            json.dumps({"type": "input_text.request", "job_id": job_id, "text": text})
        )
    result = await receive_turn(
        websocket,
        events_file=events_file,
        event_prefix=event_prefix,
        expected_source=modality,
        expected_job_id=job_id,
        timeout_seconds=timeout_seconds,
    )
    audio_path = Path(events_file.name).parent / f"{event_prefix}-response.wav"
    response_audio = result.pop("audio")
    write_wav(audio_path, response_audio)
    result["audio_path"] = str(audio_path)
    result["audio_bytes"] = len(response_audio)
    result["input_modality"] = modality
    result["model_input_path"] = "microphone_pcm" if modality == "voice" else "typed_pocket"
    result["expected_input_text"] = text
    result["input_wer"] = (
        round(word_error_rate(text, result["transcript"]), 4) if modality == "voice" else None
    )
    return result


async def run_case(
    args: argparse.Namespace,
    case: MemoryCase,
    *,
    matrix_secrets: tuple[str, ...],
) -> dict[str, Any]:
    case_dir = args.output / case.case_id
    case_dir.mkdir()
    events_path = case_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as events_file:
        async with acquire_fresh_session(args.url) as (
            websocket,
            created,
            busy_rejections,
        ):
            events_file.write(json.dumps(created, sort_keys=True) + "\n")
            if created.get("protocol") != {"name": "voicechat.realtime", "version": 3}:
                raise RuntimeError(f"unexpected protocol: {created.get('protocol')}")
            capabilities = created.get("capabilities", {})
            if capabilities.get("input_turn_detection") != "client_smart_turn_v1":
                raise RuntimeError("server did not advertise client Smart Turn v1")
            if capabilities.get("typed_input") is not True:
                raise RuntimeError("server did not advertise typed input")
            runtime_provenance = session_runtime_provenance(created)
            instructions = system_instructions(case.secret if case.seed == "system" else None)
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": f"{case.case_id}-session-update",
                        "session": {
                            "protocol_version": 3,
                            "instructions": instructions,
                            "tools": [],
                        },
                    }
                )
            )
            while True:
                event = json.loads(await websocket.recv())
                events_file.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.updated":
                    break

            turns: list[dict[str, Any]] = []
            turn_number = 1
            if case.seed in ("voice", "text"):
                if case.seed_text is None:
                    raise ValueError(f"{case.case_id}: seeded case has no seed text")
                turns.append(
                    await drive_turn(
                        websocket,
                        modality=case.seed,
                        text=case.seed_text,
                        audio=(
                            read_fixture(case.seed_audio_path)
                            if case.seed_audio_path is not None
                            else b""
                        ),
                        case_id=case.case_id,
                        turn_number=turn_number,
                        events_file=events_file,
                        timeout_seconds=args.turn_timeout_seconds,
                    )
                )
                turn_number += 1
            recall = await drive_turn(
                websocket,
                modality=case.recall,
                text=case.recall_text,
                audio=(
                    read_fixture(case.recall_audio_path)
                    if case.recall_audio_path is not None
                    else b""
                ),
                case_id=case.case_id,
                turn_number=turn_number,
                events_file=events_file,
                timeout_seconds=args.turn_timeout_seconds,
            )
            turns.append(recall)
            await websocket.send(json.dumps({"type": "session.stop"}))
            session_closed = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                events_file.write(json.dumps(event, sort_keys=True) + "\n")
                if event.get("type") == "session.closed":
                    session_closed = event
                    break

    input_quality = all(
        turn["input_accepted"]
        and (turn["input_wer"] is None or turn["input_wer"] <= args.max_input_wer)
        and (turn["input_modality"] == "voice" or turn["typed_disposition"] == "completed")
        for turn in turns
    )
    response_quality = all(
        turn["text"] and turn["audio_bytes"] > 0 and turn["response_brackets_balanced"]
        for turn in turns
    )
    recalled_secrets = tuple(
        secret for secret in matrix_secrets if contains_secret(recall["text"], secret)
    )
    semantic_match = (
        not recalled_secrets if case.secret is None else recalled_secrets == (case.secret,)
    )
    failure_class = None
    seed_turns = turns[:-1]
    if not input_quality:
        failure_class = "input_quality"
    elif any(not turn["text"] or turn["audio_bytes"] <= 0 for turn in seed_turns):
        failure_class = "seed_empty_response"
    elif any(not turn["response_brackets_balanced"] for turn in turns):
        failure_class = "unbalanced_response"
    elif not recall["text"] or recall["audio_bytes"] <= 0:
        failure_class = "recall_empty_response"
    elif case.secret is None and recalled_secrets:
        failure_class = "negative_control_leak"
    elif case.secret is not None and recalled_secrets != (case.secret,):
        failure_class = "recall_wrong_response"
    result = {
        "case_id": case.case_id,
        "seed_modality": case.seed,
        "recall_modality": case.recall,
        "secret": case.secret,
        "negative_control": case.secret is None,
        "instructions_contain_secret": bool(
            case.secret and contains_secret(instructions, case.secret)
        ),
        "recall_prompt_contains_any_secret": any(
            contains_secret(case.recall_text, secret) for secret in matrix_secrets
        ),
        "turns": turns,
        "recall_text": recall["text"],
        "recalled_secrets": recalled_secrets,
        "semantic_match": semantic_match,
        "failure_class": failure_class,
        "session_closed": session_closed,
        "fresh_session_busy_retries": busy_rejections,
        "runtime_provenance": runtime_provenance,
    }
    result["passed"] = bool(
        input_quality
        and response_quality
        and result["semantic_match"]
        and not result["recall_prompt_contains_any_secret"]
        and session_closed is not None
    )
    (case_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def flatten_responses(
    cases: list[dict[str, Any]], output: Path, matrix_secrets: tuple[str, ...]
) -> list[dict[str, Any]]:
    responses = []
    for case in cases:
        for index, turn in enumerate(case["turns"]):
            audio_path = Path(turn["audio_path"])
            try:
                recorded_path = str(audio_path.relative_to(output))
            except ValueError:
                recorded_path = str(audio_path)
            recall = index == len(case["turns"]) - 1
            expected = [case["secret"]] if recall and case["secret"] else []
            excluded = (
                [secret for secret in matrix_secrets if secret != case["secret"]] if recall else []
            )
            responses.append(
                {
                    "case_id": case["case_id"],
                    "turn_role": "recall" if recall else "seed",
                    "text": turn["text"],
                    "audio_path": recorded_path,
                    "audio_bytes": turn["audio_bytes"],
                    "expected_response_any": expected,
                    "expected_response_none": excluded,
                }
            )
    return responses


def runtime_gate_passed(report: dict[str, Any]) -> bool:
    cases = report.get("cases") or []
    expected = report.get("expected_case_count")
    full_matrix = expected == 7
    return bool(
        report.get("fresh_session_per_case")
        and report.get("runtime_provenance_consistent") is True
        and valid_runtime_provenance(report.get("runtime_provenance"))
        and isinstance(expected, int)
        and len(cases) == expected
        and all(case.get("passed") is True for case in cases)
        and (
            not full_matrix
            or (
                report.get("negative_control_ran_after_seeded") is True
                and [case.get("negative_control") for case in cases]
                == [False, False, False, False, False, False, True]
            )
        )
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    configured_cases = load_cases(manifest)
    matrix_secrets = tuple(case.secret for case in configured_cases if case.secret)
    args.output.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    selected_cases = tuple(case for case in configured_cases if args.case in (None, case.case_id))
    if not selected_cases:
        raise ValueError(f"unknown memory matrix case: {args.case}")
    for case in selected_cases:
        cases.append(
            await run_case(
                args,
                case,
                matrix_secrets=matrix_secrets,
            )
        )
    runtime_provenances = [case.get("runtime_provenance") for case in cases]
    runtime_provenance = runtime_provenances[0] if runtime_provenances else None
    runtime_provenance_consistent = bool(
        valid_runtime_provenance(runtime_provenance)
        and all(item == runtime_provenance for item in runtime_provenances)
    )
    report = {
        "report_kind": "memory_matrix_strict_v3",
        "passed": len(cases) == len(selected_cases) and all(case["passed"] for case in cases),
        "matrix": {
            "seed_modalities": ["voice", "text", "system"],
            "recall_modalities": ["voice", "text"],
            "seeded_case_count": 6,
            "negative_control_count": 1,
            "secrets_unique_per_case": True,
        },
        "fresh_session_per_case": True,
        "runtime_provenance": runtime_provenance,
        "runtime_provenance_consistent": runtime_provenance_consistent,
        "negative_control_ran_after_seeded": bool(
            len(cases) == 7
            and all(not case["negative_control"] for case in cases[:-1])
            and cases[-1]["negative_control"]
        ),
        "expected_case_count": len(selected_cases),
        "manifest": str(args.manifest.resolve()),
        "cases": cases,
    }
    report["responses"] = flatten_responses(cases, args.output, matrix_secrets)
    report["runtime_gate_passed"] = runtime_gate_passed(report)
    report["passed"] = report["runtime_gate_passed"]
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-timeout-seconds", type=float, default=90)
    parser.add_argument("--max-input-wer", type=float, default=0.35)
    parser.add_argument("--case")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "report_kind": "memory_matrix_strict_v3",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
