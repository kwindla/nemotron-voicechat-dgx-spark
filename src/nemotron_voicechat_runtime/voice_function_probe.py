"""Diagnostic-only voice positive control for direct-text function logits."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .direct_semantic_probe import (
    FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES,
    _append_function_logit_trace,
    _sotc_evidence,
)
from .protocol import (
    FUNCTION_OUTPUT_RECOVERY_FRAME_SECONDS,
    FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES,
)
from .semantic_corpus import evaluate_pocket_carrier, evaluate_semantic_case
from .voice_function_fixture import (
    FIXTURE_CASE_ID,
    FIXTURE_MAX_SAMPLES,
    FIXTURE_PACKET_SAMPLES,
    validate_fixture_source_provenance,
    validate_voice_function_fixture,
)

VOICE_RESPONSE_FRAME_LIMIT = FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES
VOICE_RESPONSE_FRAME_SECONDS = FUNCTION_OUTPUT_RECOVERY_FRAME_SECONDS
VOICE_QUIESCENCE_OBSERVATIONS = 2


class _MonotonicResponseFramePacer:
    """Synchronous mirror of the production function-recovery frame clock."""

    def __init__(
        self,
        *,
        frame_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if frame_seconds <= 0:
            raise ValueError("response frame duration must be positive")
        self.frame_seconds = frame_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_frame_at: float | None = None
        self.observed_frames = 0
        self.paced_transitions = 0

    def positions(self, max_frames: int) -> Iterator[int]:
        if max_frames <= 0:
            raise ValueError("response frame bound must be positive")
        self._next_frame_at = self._monotonic()
        for position in range(max_frames):
            if position:
                self._next_frame_at += self.frame_seconds
                self._sleep(max(0.0, self._next_frame_at - self._monotonic()))
                self.paced_transitions += 1
            self.observed_frames = position + 1
            yield position


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    values = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(values.tobytes())


def _trace_capture(
    result: Any,
    *,
    enabled: bool,
    phase: str,
    phase_position: int,
    foreground_position: int,
    last_model_frame: int | None,
    destination: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
) -> tuple[int | None, int]:
    """Record one fresh model position or an explicit background-owned gap."""

    value = result.turn_state.get("function_logit_trace")
    background = bool((result.turn_state.get("function_calling") or {}).get("background_active"))
    record: dict[str, Any] = {
        "phase": phase,
        "phase_position": phase_position,
        "engine_frame": int(result.frame_index),
        "background_active": background,
    }
    if not enabled:
        record["status"] = "trace_disabled"
        coverage.append(record)
        return last_model_frame, foreground_position
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        model_frame = value[0].get("model_frame")
        if isinstance(model_frame, int) and not isinstance(model_frame, bool):
            if (last_model_frame is None or model_frame > last_model_frame) and (
                not background or model_frame == int(result.frame_index)
            ):
                _append_function_logit_trace(
                    destination,
                    value,
                    phase=phase,
                    phase_position=foreground_position,
                    expected_model_frame=model_frame,
                )
                record.update(
                    {
                        "status": "fresh_model_position",
                        "model_frame": model_frame,
                        "foreground_position": foreground_position,
                    }
                )
                coverage.append(record)
                return model_frame, foreground_position + 1
            if background and last_model_frame is not None:
                record.update(
                    {
                        "status": "no_new_model_position",
                        "reason": "fc_background_owned_advancement",
                        "stale_model_frame_observed": model_frame,
                    }
                )
                coverage.append(record)
                return last_model_frame, foreground_position
    if background and value is None:
        record.update(
            {
                "status": "no_new_model_position",
                "reason": "fc_background_owned_advancement",
                "stale_model_frame_observed": None,
            }
        )
        coverage.append(record)
        return last_model_frame, foreground_position
    raise RuntimeError(
        f"voice function trace has missing, stale, or ambiguous evidence at "
        f"{phase}[{phase_position}]"
    )


def _validate_trace_coverage(
    trace: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    *,
    enabled: bool,
) -> None:
    if not coverage:
        raise RuntimeError("voice function trace coverage is empty")
    if not enabled:
        if trace or any(record.get("status") != "trace_disabled" for record in coverage):
            raise RuntimeError("untraced voice function report contains trace evidence")
        return
    fresh = [record for record in coverage if record.get("status") == "fresh_model_position"]
    gaps = [record for record in coverage if record.get("status") == "no_new_model_position"]
    if len(fresh) != len(trace):
        raise RuntimeError("voice function trace coverage does not match trace entries")
    if any(
        record.get("reason") != "fc_background_owned_advancement"
        or record.get("background_active") is not True
        or record.get("phase") != "post_bos"
        for record in gaps
    ):
        raise RuntimeError("voice function trace has an unjustified coverage gap")
    if [record["model_frame"] for record in fresh] != [entry["model_frame"] for entry in trace]:
        raise RuntimeError("voice function trace coverage model frames disagree")
    if len({entry["model_frame"] for entry in trace}) != len(trace):
        raise RuntimeError("voice function trace repeats a model frame")
    for phase in ("voice_input", "settlement", "eou_boundary", "post_bos"):
        entries = [entry for entry in trace if entry["phase"] == phase]
        if [entry["phase_position"] for entry in entries] != list(range(len(entries))):
            raise RuntimeError(f"voice function trace {phase} positions are not contiguous")
        model_frames = [entry["model_frame"] for entry in entries]
        if any(right != left + 1 for left, right in zip(model_frames, model_frames[1:])):
            raise RuntimeError(f"voice function trace {phase} model frames are not contiguous")


def _function_quiescence(
    result: Any,
    engine: Any,
    recorded_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    fc = result.turn_state.get("function_calling") or {}
    background_absent = engine.stream_id not in getattr(engine.pipeline, "_fc_async_bg", {})
    expected_call = recorded_calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    expected_eotr_id = getattr(engine.pipeline.s2s_model, "_fc_eotr_id", None)
    quiescent = bool(
        expected_call
        and background_absent
        and fc.get("active") is False
        and fc.get("awaiting_response") is False
        and fc.get("injecting_response") is False
        and int(fc.get("forced_tokens") or 0) == 0
        and int(fc.get("completed_calls") or 0) == 1
        and fc.get("awaiting_eotr") is False
        and fc.get("eotr_observed") is True
        and int(fc.get("eotr_observed_count") or 0) == 1
        and isinstance(fc.get("eotr_observed_frame"), int)
        and isinstance(expected_eotr_id, int)
        and fc.get("eotr_first_post_drain_token_id") == expected_eotr_id
        and fc.get("eotr_first_post_drain_token_frame") == fc.get("eotr_observed_frame")
        and int(fc.get("eotr_wait_steps") or 0) == 1
        and fc.get("eotr_timeout") is False
        and fc.get("eotr_unexpected_token_id") is None
        and fc.get("eotr_unexpected_token_frame") is None
        and fc.get("eotr_feedback_token_id") == expected_eotr_id
        and fc.get("eotr_feedback_frame") == fc.get("eotr_observed_frame")
        and fc.get("eotr_feedback_committed") is True
    )
    return {
        "expected_call_recorded": expected_call,
        "background_absent": background_absent,
        "active": fc.get("active"),
        "awaiting_response": fc.get("awaiting_response"),
        "injecting_response": fc.get("injecting_response"),
        "forced_tokens": fc.get("forced_tokens"),
        "completed_calls": fc.get("completed_calls"),
        "expected_eotr_token_id": expected_eotr_id,
        "awaiting_eotr": fc.get("awaiting_eotr"),
        "eotr_observed": fc.get("eotr_observed"),
        "eotr_observed_count": fc.get("eotr_observed_count"),
        "eotr_observed_frame": fc.get("eotr_observed_frame"),
        "eotr_wait_steps": fc.get("eotr_wait_steps"),
        "eotr_timeout": fc.get("eotr_timeout"),
        "eotr_unexpected_token_id": fc.get("eotr_unexpected_token_id"),
        "eotr_unexpected_token_frame": fc.get("eotr_unexpected_token_frame"),
        "eotr_first_post_drain_token_id": fc.get("eotr_first_post_drain_token_id"),
        "eotr_first_post_drain_token_frame": fc.get("eotr_first_post_drain_token_frame"),
        "eotr_feedback_token_id": fc.get("eotr_feedback_token_id"),
        "eotr_feedback_frame": fc.get("eotr_feedback_frame"),
        "eotr_feedback_committed": fc.get("eotr_feedback_committed"),
        "quiescent": quiescent,
    }


def _trace_summary(entries: list[dict[str, Any]], watched: dict[str, int]) -> dict[str, Any]:
    sotc_id = watched["sotc"]
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("voice_input", "settlement", "eou_boundary", "post_bos"):
        values = [entry for entry in entries if entry["phase"] == phase]
        sotc_ranks = [entry["sotc_rank"] for entry in values if entry["sotc_rank"] is not None]
        margins = [
            entry["watched"]["sotc"]["logit"] - entry["watched"]["pad"]["logit"]
            for entry in values
        ]
        raw_sotc = [
            {
                "phase_position": entry["phase_position"],
                "model_frame": entry["model_frame"],
            }
            for entry in values
            if entry["raw_function_token_id"] == sotc_id
        ]
        phases[phase] = {
            "entry_count": len(values),
            "min_sotc_rank": min(sotc_ranks) if sotc_ranks else None,
            "max_sotc_minus_pad_logit": max(margins) if margins else None,
            "raw_sotc_argmax": raw_sotc,
        }
    return {"phases": phases}


def run_voice_function_probe(
    engine: Any,
    *,
    fixture_dir: Path,
    corpus_path: Path,
    output_dir: Path,
    function_template_path: Path,
    precision_label: str,
    runtime_image_id: str,
    speech_gate_dbfs: float,
    speech_gate_min_frames: int,
    continuous_after_speech: bool,
    max_response_frames: int = VOICE_RESPONSE_FRAME_LIMIT,
) -> dict[str, Any]:
    """Replay one known-working acoustic tool call and observe function logits."""

    from .server import (
        FRAME_SAMPLES,
        FRAME_SECONDS,
        OUTPUT_SAMPLE_RATE,
        PcmFrameBuffer,
        TransportModelGate,
        UserEouSettlement,
        _direct_probe_runtime_provenance,
        audio_stats,
        clean_display_text,
        coerce_tool_arguments,
        normalize_tool_definitions,
        render_tool_system_prompt,
        response_audio_is_deliverable,
    )

    if max_response_frames != VOICE_RESPONSE_FRAME_LIMIT:
        raise ValueError(f"voice function response bound must be {VOICE_RESPONSE_FRAME_LIMIT}")
    if FRAME_SECONDS != VOICE_RESPONSE_FRAME_SECONDS:
        raise RuntimeError("voice probe and production recovery frame clocks disagree")
    if os.environ.get("VOICECHAT_NANO_PAD_PAIR", "0").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("voice function probe requires VOICECHAT_NANO_PAD_PAIR=0")
    fixture, pcm = validate_voice_function_fixture(
        fixture_dir,
        corpus_path=corpus_path,
        expected_runtime_image_id=runtime_image_id,
    )
    validate_fixture_source_provenance(
        fixture,
        project_root=Path("/workspace/project"),
        installed_worker=Path(
            "/opt/pocket-tts/lib/python3.12/site-packages/"
            "nemotron_voicechat_runtime/pocket_worker.py"
        ),
    )
    from .pocket_worker import validate_assets

    if validate_assets() != fixture["synthesis"]["assets"]:
        raise RuntimeError("voice function fixture assets differ at replay time")
    if len(pcm) // 2 > FIXTURE_MAX_SAMPLES:
        raise RuntimeError("voice function fixture exceeds the carrier bound")
    from .semantic_corpus import load_semantic_corpus

    corpus = load_semantic_corpus(corpus_path)
    case = next(case for case in corpus["cases"] if case["id"] == FIXTURE_CASE_ID)
    tools = normalize_tool_definitions(corpus["tools"])
    rendered_prompt = render_tool_system_prompt(
        function_template_path, corpus["instructions"], tools
    )
    tool_schemas = {tool["name"]: tool["parameters"] for tool in tools}
    calls: list[dict[str, Any]] = []
    calls_lock = threading.Lock()

    def make_handler(name: str):
        def handler(arguments: Any) -> str:
            coerced = coerce_tool_arguments(arguments, tool_schemas[name])
            with calls_lock:
                calls.append({"name": name, "arguments": coerced})
            return json.dumps(corpus["tool_results"][name], sort_keys=True)

        return handler

    handlers = {tool["name"].replace("-", "_"): make_handler(tool["name"]) for tool in tools}
    wrapper = engine.pipeline.s2s_model
    trace_top_k = int(getattr(wrapper.model_llm_interface, "_function_logit_trace_topk", 0))
    trace_enabled = trace_top_k > 0
    watched = dict(getattr(wrapper.model_llm_interface, "_function_logit_trace_ids", {}) or {})
    if trace_enabled and set(watched) != {"pad", "sotc", "eotc", "eotr", "text_bos"}:
        raise RuntimeError("voice function probe has invalid watched token IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine.external_user_eou_mode = True
    engine.configure_external_tools(handlers)
    engine.start(system_prompt=rendered_prompt)
    sotc_evidence = {"starting": _sotc_evidence(wrapper)}
    trace: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    delivered_chunks: list[np.ndarray] = []
    last_result = None
    last_model_frame: int | None = None
    foreground_positions = {
        "voice_input": 0,
        "settlement": 0,
        "eou_boundary": 0,
        "post_bos": 0,
    }
    response_started = False
    response_complete = False
    terminal_after_quiescence = False
    quiescence_streak = 0
    quiescence_records: list[dict[str, Any]] = []
    try:
        buffer = PcmFrameBuffer()
        gate = TransportModelGate(
            threshold_dbfs=speech_gate_dbfs,
            continuous_after_speech=continuous_after_speech,
            min_active_frames=speech_gate_min_frames,
        )
        gate.force_open()

        def process_input_frame(frame: np.ndarray, packet_index: int, *, padded: bool) -> None:
            nonlocal last_result, last_model_frame
            stats = audio_stats(frame)
            advanced = gate.should_advance(float(stats["rms_dbfs"]))
            if not advanced:
                raise RuntimeError("explicit voice turn unexpectedly failed to advance")
            last_result = engine.process(frame, input_active=gate.input_active)
            position = len(input_records)
            input_records.append(
                {
                    "position": position,
                    "packet_index": packet_index,
                    "padded_tail": padded,
                    "input_active": gate.input_active,
                    "audio": stats,
                }
            )
            last_model_frame, foreground_positions["voice_input"] = _trace_capture(
                last_result,
                enabled=trace_enabled,
                phase="voice_input",
                phase_position=position,
                foreground_position=foreground_positions["voice_input"],
                last_model_frame=last_model_frame,
                destination=trace,
                coverage=coverage,
            )

        packet_bytes = FIXTURE_PACKET_SAMPLES * 2
        packet_index = 0
        for offset in range(0, len(pcm), packet_bytes):
            packet = pcm[offset : offset + packet_bytes]
            for frame in buffer.push(packet):
                process_input_frame(frame, packet_index, padded=False)
            packet_index += 1
        original_pending_samples = buffer.pending_samples
        if original_pending_samples:
            process_input_frame(buffer.flush(), packet_index, padded=True)
        sotc_evidence["after_voice_input"] = _sotc_evidence(wrapper)

        settlement = UserEouSettlement.begin(engine, last_result)
        while settlement.needs_step(engine, last_result):
            silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            if not gate.should_advance(float(audio_stats(silence)["rms_dbfs"])):
                raise RuntimeError("settlement silence unexpectedly failed to advance")
            last_result = engine.process(silence, input_active=gate.input_active)
            phase_position = settlement.model_steps
            last_model_frame, foreground_positions["settlement"] = _trace_capture(
                last_result,
                enabled=trace_enabled,
                phase="settlement",
                phase_position=phase_position,
                foreground_position=foreground_positions["settlement"],
                last_model_frame=last_model_frame,
                destination=trace,
                coverage=coverage,
            )
            settlement.observe(engine, last_result)
        sotc_evidence["after_settlement"] = _sotc_evidence(wrapper)

        engine.request_user_eou()
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        if not gate.should_advance(float(audio_stats(silence)["rms_dbfs"])):
            raise RuntimeError("EOU boundary silence unexpectedly failed to advance")
        boundary_step = engine.process(silence, input_active=gate.input_active)
        last_result = boundary_step
        eou_boundary_evidence = {
            "performed": True,
            "engine_frame": int(boundary_step.frame_index),
            "agent_control": boundary_step.turn_state.get("agent_control"),
            "response_boundary": boundary_step.turn_state.get("response_boundary"),
        }
        last_model_frame, foreground_positions["eou_boundary"] = _trace_capture(
            boundary_step,
            enabled=trace_enabled,
            phase="eou_boundary",
            phase_position=0,
            foreground_position=foreground_positions["eou_boundary"],
            last_model_frame=last_model_frame,
            destination=trace,
            coverage=coverage,
        )
        sotc_evidence["after_eou_boundary"] = _sotc_evidence(wrapper)

        assistant_text = ""
        function_text = ""
        response_pacer = _MonotonicResponseFramePacer(frame_seconds=FRAME_SECONDS)
        for response_position in response_pacer.positions(max_response_frames):
            step = boundary_step
            if response_position:
                if not gate.should_advance(float(audio_stats(silence)["rms_dbfs"])):
                    raise RuntimeError("response silence unexpectedly failed to advance")
                step = engine.process(silence, input_active=gate.input_active)
                last_result = step
                if response_position <= FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES:
                    last_model_frame, foreground_positions["post_bos"] = _trace_capture(
                        step,
                        enabled=trace_enabled,
                        phase="post_bos",
                        phase_position=response_position - 1,
                        foreground_position=foreground_positions["post_bos"],
                        last_model_frame=last_model_frame,
                        destination=trace,
                        coverage=coverage,
                    )
            assistant_text = clean_display_text(step.assistant_text)
            function_text = step.function_text
            if response_audio_is_deliverable(step.turn_state) and step.audio.size:
                delivered_chunks.append(step.audio.copy())
            boundary = step.turn_state.get("response_boundary") or {}
            response_started |= boundary.get("event") == "start"
            with calls_lock:
                calls_snapshot = list(calls)
            quiescence = _function_quiescence(step, engine, calls_snapshot)
            if quiescence["quiescent"]:
                quiescence_streak += 1
            else:
                quiescence_streak = 0
            if boundary.get("event") == "end" and quiescence["quiescent"]:
                terminal_after_quiescence = True
                response_complete = True
            quiescence_records.append(
                {
                    "response_position": response_position,
                    "engine_frame": int(step.frame_index),
                    "boundary_event": boundary.get("event"),
                    "boundary_phase": boundary.get("phase"),
                    "quiescence_streak": quiescence_streak,
                    **quiescence,
                }
            )
            if terminal_after_quiescence and quiescence_streak >= VOICE_QUIESCENCE_OBSERVATIONS:
                break

        with calls_lock:
            recorded_calls = list(calls)
        sotc_evidence["final"] = _sotc_evidence(wrapper)
        final_quiescence = (
            quiescence_records[-1]
            if quiescence_records
            else {"quiescent": False, "quiescence_streak": 0}
        )
        voice_reproduction_valid = bool(
            recorded_calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
            and response_started
            and response_complete
            and terminal_after_quiescence
            and final_quiescence.get("quiescent") is True
            and int(final_quiescence.get("quiescence_streak") or 0)
            >= VOICE_QUIESCENCE_OBSERVATIONS
        )
        failure = None
        if not voice_reproduction_valid:
            failure = {
                "code": "voice_reproduction_invalid",
                "message": (
                    "Frozen voice reference did not reach the expected tool call, "
                    "function quiescence, and post-quiescence acoustic terminal "
                    f"within {max_response_frames} response frames"
                ),
            }
        _validate_trace_coverage(trace, coverage, enabled=trace_enabled)
        delivered = (
            np.concatenate(delivered_chunks).astype(np.float32, copy=False)
            if delivered_chunks
            else np.empty(0, dtype=np.float32)
        )
        audio_path = output_dir / "response.wav"
        _write_wav(audio_path, delivered, OUTPUT_SAMPLE_RATE)
        semantic = evaluate_semantic_case(
            case, response_text=assistant_text, tool_calls=recorded_calls
        )
        carrier = evaluate_pocket_carrier(case, transcript=engine.user_text)
        structural = {
            "fixture_revalidated": True,
            "fixture_within_bound": fixture["pcm"]["sample_count"] <= FIXTURE_MAX_SAMPLES,
            "input_frames_nonempty": bool(input_records),
            "input_frame_count_exact": len(input_records)
            == math.ceil(fixture["pcm"]["sample_count"] / FRAME_SAMPLES),
            "input_padding_count_exact": len(input_records) * FRAME_SAMPLES
            - fixture["pcm"]["sample_count"]
            == ((FRAME_SAMPLES - original_pending_samples) % FRAME_SAMPLES),
            "tail_padding_exact": sum(record["padded_tail"] for record in input_records)
            == int(original_pending_samples > 0),
            "settlement_clean": settlement.evidence()["pre_eou_clean"],
            "single_eou_boundary": eou_boundary_evidence["performed"] is True,
            "boundary_sotc_committed_while_idle": bool(
                eou_boundary_evidence["agent_control"] == "pad"
                and (eou_boundary_evidence["response_boundary"] or {}).get("event") is None
                and sotc_evidence["after_eou_boundary"]["client_eou_sotc_committed_count"]
                == sotc_evidence["after_settlement"]["client_eou_sotc_committed_count"] + 1
                and sotc_evidence["after_eou_boundary"]["post_fc_client_bos_pending"]
                and sotc_evidence["after_eou_boundary"]["client_eou_sotc_committed_frame"]
                == sotc_evidence["after_eou_boundary"]["post_fc_client_bos_requested_frame"]
            ),
            "single_post_fc_client_bos": bool(
                not sotc_evidence["final"]["post_fc_client_bos_pending"]
                and sotc_evidence["final"]["post_fc_client_bos_forced_count"]
                == sotc_evidence["after_settlement"]["post_fc_client_bos_forced_count"] + 1
                and isinstance(sotc_evidence["final"]["post_fc_client_bos_forced_frame"], int)
                and isinstance(sotc_evidence["final"]["client_eou_sotc_committed_frame"], int)
                and sotc_evidence["final"]["post_fc_client_bos_forced_frame"]
                > sotc_evidence["final"]["client_eou_sotc_committed_frame"]
            ),
            "response_started": response_started,
            "response_frame_clock_exact": bool(
                response_pacer.observed_frames == len(quiescence_records)
                and response_pacer.paced_transitions
                == max(0, len(quiescence_records) - 1)
                and response_pacer.frame_seconds == VOICE_RESPONSE_FRAME_SECONDS
            ),
            "response_complete_after_function_quiescence": response_complete,
            "function_quiescent_twice": int(final_quiescence.get("quiescence_streak") or 0)
            >= VOICE_QUIESCENCE_OBSERVATIONS,
            "expected_tool_call": recorded_calls
            == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
            "carrier_preserved": carrier["passed"],
            "response_audio_nonempty": delivered.size > 0,
            "trace_coverage_complete": True,
        }
        report: dict[str, Any] = {
            "schema": 2 if trace_enabled else 1,
            "kind": "voice_function_logit_probe",
            "precision": precision_label,
            "case_id": FIXTURE_CASE_ID,
            "runtime_provenance": _direct_probe_runtime_provenance(engine),
            "fixture": fixture,
            "input": {
                "packet_samples": FIXTURE_PACKET_SAMPLES,
                "model_frame_samples": FRAME_SAMPLES,
                "original_samples": fixture["pcm"]["sample_count"],
                "original_pending_samples": original_pending_samples,
                "padded_samples": len(input_records) * FRAME_SAMPLES,
                "records": input_records,
            },
            "settlement": settlement.evidence(),
            "eou_boundary": eou_boundary_evidence,
            "sotc_evidence": sotc_evidence,
            "assistant_text": assistant_text,
            "function_text": function_text,
            "tool_calls": recorded_calls,
            "user_transcript": engine.user_text,
            "carrier": carrier,
            "semantic": semantic,
            "response_audio_path": str(audio_path),
            "response_audio_samples": int(delivered.size),
            "response_frames": len(quiescence_records),
            "response_pacing": {
                "clock": "monotonic_deadline",
                "frame_seconds": response_pacer.frame_seconds,
                "budget_frames": max_response_frames,
                "observed_frames": response_pacer.observed_frames,
                "paced_transitions": response_pacer.paced_transitions,
            },
            "quiescence_required": VOICE_QUIESCENCE_OBSERVATIONS,
            "quiescence": quiescence_records,
            "voice_reproduction_valid": voice_reproduction_valid,
            "failure": failure,
            "structural_checks": structural,
            "structural_passed": all(structural.values()),
            "passed": bool(voice_reproduction_valid and all(structural.values())),
        }
        if trace_enabled:
            report["function_logit_trace"] = {
                "top_k": trace_top_k,
                "post_bos_frames": FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES,
                "phases": ["voice_input", "settlement", "eou_boundary", "post_bos"],
                "watched_token_ids": watched,
                "entries": trace,
                "coverage": coverage,
                "summary": _trace_summary(trace, watched),
            }
        return report
    finally:
        if engine.started:
            engine.abort()
