"""Diagnostic-only full-corpus runner for direct user-source positions."""

from __future__ import annotations

import json
import os
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_corpus import corpus_sha256, evaluate_semantic_case, load_semantic_corpus

FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES = 8


def _append_function_logit_trace(
    destination: list[dict[str, Any]],
    value: Any,
    *,
    phase: str,
    phase_position: int,
    expected_model_frame: int,
) -> None:
    """Attach a caller-owned phase to one model-owned position summary."""
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError(
            f"function-logit tracing expected one entry for {phase}[{phase_position}]"
        )
    entry = dict(value[0])
    if entry.get("model_frame") != expected_model_frame:
        raise RuntimeError(
            f"function-logit trace frame mismatch for {phase}[{phase_position}]: "
            f"{entry.get('model_frame')} != {expected_model_frame}"
        )
    entry["phase"] = phase
    entry["phase_position"] = phase_position
    destination.append(entry)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    values = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(values.tobytes())


def _transaction_checks(positions: list[dict[str, Any]], pad_id: int) -> dict[str, bool]:
    return {
        "nonempty": bool(positions),
        "bf16_source_1x1x4480": bool(positions)
        and all(
            position.get("model", {}).get("source_dtype") == "torch.bfloat16"
            and position.get("model", {}).get("source_shape") == [1, 1, 4480]
            for position in positions
        ),
        "effective_outputs_forced_pad": bool(positions)
        and all(
            position.get("model", {}).get("effective_output_forced_pad") is True
            and int(position.get("effective_token_id", -1)) == pad_id
            and position.get("effective_subword_mask") is True
            for position in positions
        ),
        "hidden_audio_discarded": bool(positions)
        and all(
            int(position.get("decoded_audio_samples_discarded", 0)) > 0 for position in positions
        ),
        "clocks_advance_once": bool(positions)
        and all(
            int(position["nano_position_after"]) == int(position["nano_position_before"]) + 1
            and int(position["eartts_position_after"])
            == int(position["eartts_position_before"]) + 1
            and int(position["epoch_offset_after"]) == int(position["epoch_offset_before"])
            for position in positions
        ),
        "non_source_state_held": bool(positions)
        and all(
            position.get("perception_cache_unchanged") is True
            and position.get("rnnt_state_unchanged") is True
            and position.get("function_state_unchanged") is True
            for position in positions
        ),
        "feedback_is_exact_pad_chain": bool(positions)
        and all(
            all(value in {None, pad_id} for value in position["previous_feedback"].values())
            for position in positions
        ),
    }


def _sotc_evidence(wrapper: Any) -> dict[str, Any]:
    """Capture boundary-SOTC counters without interpreting semantic intent."""

    return {
        "pre_eou_suppressed_tokens": int(
            getattr(wrapper, "_pre_eou_function_suppression_count", 0)
        ),
        "pre_eou_last_raw_token_id": getattr(wrapper, "_pre_eou_function_last_raw_token_id", None),
        "pre_eou_suppressed_frame": getattr(wrapper, "_pre_eou_function_suppressed_frame", None),
        "client_eou_sotc_committed_count": int(
            getattr(wrapper, "_client_eou_sotc_committed_count", 0)
        ),
        "client_eou_sotc_committed_frame": getattr(
            wrapper, "_client_eou_sotc_committed_frame", None
        ),
        "post_fc_client_bos_pending": bool(
            getattr(wrapper, "_post_fc_client_bos_requested", False)
        ),
        "post_fc_client_bos_requested_frame": getattr(
            wrapper, "_post_fc_client_bos_requested_frame", None
        ),
        "post_fc_client_bos_forced_count": int(
            getattr(wrapper, "_post_fc_client_bos_forced_count", 0)
        ),
        "post_fc_client_bos_forced_frame": getattr(
            wrapper, "_post_fc_client_bos_forced_frame", None
        ),
    }


def run_direct_semantic_probe(
    engine: Any,
    *,
    corpus_path: Path,
    output_dir: Path,
    function_template_path: Path,
    token_forms: list[str],
    precision_label: str,
    max_response_frames: int = 240,
) -> dict[str, Any]:
    """Run every corpus prompt through each direct token form on a real graph."""
    from .server import (
        FRAME_SAMPLES,
        OUTPUT_SAMPLE_RATE,
        UserEouSettlement,
        UserEouSettlementFailure,
        _direct_probe_runtime_provenance,
        _direct_probe_token_ids,
        clean_display_text,
        coerce_tool_arguments,
        normalize_tool_definitions,
        render_tool_system_prompt,
        response_audio_is_deliverable,
    )

    if max_response_frames < 1:
        raise ValueError("direct semantic probe max response frames must be positive")
    if os.environ.get("VOICECHAT_NANO_PAD_PAIR", "0").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("direct semantic probe requires VOICECHAT_NANO_PAD_PAIR=0")
    if not token_forms or set(token_forms) - {"text", "user_bos_text_user_eos"}:
        raise ValueError("direct semantic probe requires supported explicit token forms")
    corpus = load_semantic_corpus(corpus_path)
    tools = normalize_tool_definitions(corpus["tools"])
    rendered_prompt = render_tool_system_prompt(
        function_template_path, corpus["instructions"], tools
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = engine.pipeline.s2s_model
    trace_top_k = int(getattr(wrapper.model_llm_interface, "_function_logit_trace_topk", 0))
    pad_id = int(wrapper.model.stt_model.text_pad_id)
    tool_results = corpus["tool_results"]
    tool_schemas = {tool["name"]: tool["parameters"] for tool in tools}
    lanes: list[dict[str, Any]] = []

    for token_form in token_forms:
        results: list[dict[str, Any]] = []
        lane_dir = output_dir / token_form
        lane_dir.mkdir()
        for case in corpus["cases"]:
            calls: list[dict[str, Any]] = []
            calls_lock = threading.Lock()

            def make_handler(name: str):
                def handler(arguments: Any) -> str:
                    coerced = coerce_tool_arguments(arguments, tool_schemas[name])
                    with calls_lock:
                        calls.append({"name": name, "arguments": coerced})
                    return json.dumps(tool_results[name], sort_keys=True)

                return handler

            handlers = {
                tool["name"].replace("-", "_"): make_handler(tool["name"]) for tool in tools
            }
            engine.external_user_eou_mode = True
            engine.configure_external_tools(handlers)
            engine.start(system_prompt=rendered_prompt)
            positions: list[dict[str, Any]] = []
            function_logit_trace: list[dict[str, Any]] = []
            delivered_chunks: list[np.ndarray] = []
            response_started = False
            response_complete = False
            assistant_text = ""
            function_text = ""
            reset_controller = wrapper.model.tts_model
            reset_before = int(getattr(reset_controller, "_reset_on_bos_count", 0))
            try:
                token_ids = _direct_probe_token_ids(engine.pipeline, case["prompt"], token_form)
                perception_before = int(
                    engine.pipeline.context_manager.slot_contexts[
                        engine.pipeline.context_manager.streamidx2slotidx[engine.stream_id]
                    ].perception_frame_idx
                )
                for token_position, token_id in enumerate(token_ids):
                    position = engine.inject_user_source_position(token_id)
                    positions.append(position)
                    if trace_top_k:
                        _append_function_logit_trace(
                            function_logit_trace,
                            position.get("model", {}).get("function_logit_trace"),
                            phase="direct_position",
                            phase_position=token_position,
                            expected_model_frame=int(position["frame_idx_before"]),
                        )
                context = engine.pipeline.context_manager.slot_contexts[
                    engine.pipeline.context_manager.streamidx2slotidx[engine.stream_id]
                ]
                perception_after_direct = int(context.perception_frame_idx)
                audio_frame_after_direct = int(engine.audio_frame_index)
                reset_after_direct = int(getattr(reset_controller, "_reset_on_bos_count", 0))
                settlement = UserEouSettlement.begin(engine, None)
                settlement_failure: dict[str, Any] | None = None
                last_settlement_step = None
                try:
                    while settlement.needs_step(engine, last_settlement_step):
                        last_settlement_step = engine.process(
                            np.zeros(FRAME_SAMPLES, dtype=np.float32),
                            input_active=False,
                        )
                        if trace_top_k:
                            _append_function_logit_trace(
                                function_logit_trace,
                                last_settlement_step.turn_state.get("function_logit_trace"),
                                phase="settlement",
                                phase_position=settlement.model_steps,
                                expected_model_frame=int(last_settlement_step.frame_index),
                            )
                        settlement.observe(engine, last_settlement_step)
                except UserEouSettlementFailure as exc:
                    settlement_failure = {"code": exc.code, "message": str(exc)}
                settlement_evidence = settlement.evidence()
                first_settlement_pcm = (
                    settlement_evidence["steps"][0] if settlement_evidence["steps"] else None
                )
                reset_after_settlement = int(getattr(reset_controller, "_reset_on_bos_count", 0))
                boundary_evidence: dict[str, Any] = {
                    "performed": False,
                    "audio_frame_index_before": int(engine.audio_frame_index),
                    "audio_frame_index_after": None,
                    "perception_frame_idx_before": int(context.perception_frame_idx),
                    "perception_frame_idx_after": None,
                    "frame_index": None,
                    "eartts_reset_count_after": None,
                    "agent_control": None,
                    "response_boundary": None,
                    "vllm_request_positions": None,
                    "sotc_before": _sotc_evidence(wrapper),
                    "sotc_after": None,
                    "sotc_final": None,
                }
                if settlement_failure is None:
                    engine.request_user_eou()
                    boundary_step = engine.process(
                        np.zeros(FRAME_SAMPLES, dtype=np.float32), input_active=False
                    )
                    boundary_evidence.update(
                        {
                            "performed": True,
                            "audio_frame_index_after": int(engine.audio_frame_index),
                            "perception_frame_idx_after": int(context.perception_frame_idx),
                            "frame_index": int(boundary_step.frame_index),
                            "eartts_reset_count_after": int(
                                getattr(reset_controller, "_reset_on_bos_count", 0)
                            ),
                            "agent_control": boundary_step.turn_state.get("agent_control"),
                            "response_boundary": boundary_step.turn_state.get("response_boundary"),
                            "vllm_request_positions": boundary_step.turn_state.get(
                                "vllm_request_positions"
                            ),
                            "sotc_after": _sotc_evidence(wrapper),
                        }
                    )
                    if trace_top_k:
                        _append_function_logit_trace(
                            function_logit_trace,
                            boundary_step.turn_state.get("function_logit_trace"),
                            phase="eou_boundary",
                            phase_position=0,
                            expected_model_frame=int(boundary_step.frame_index),
                        )
                    for frame_index in range(max_response_frames):
                        step = (
                            boundary_step
                            if frame_index == 0
                            else engine.process(
                                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                                input_active=False,
                            )
                        )
                        if (
                            trace_top_k
                            and frame_index > 0
                            and frame_index <= FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES
                        ):
                            _append_function_logit_trace(
                                function_logit_trace,
                                step.turn_state.get("function_logit_trace"),
                                phase="post_bos",
                                phase_position=frame_index - 1,
                                expected_model_frame=int(step.frame_index),
                            )
                        assistant_text = clean_display_text(step.assistant_text)
                        function_text = step.function_text
                        if response_audio_is_deliverable(step.turn_state) and step.audio.size:
                            delivered_chunks.append(step.audio.copy())
                        boundary = step.turn_state.get("response_boundary") or {}
                        response_started |= boundary.get("event") == "start"
                        if boundary.get("event") == "end":
                            response_complete = True
                            break
                    boundary_evidence["sotc_final"] = _sotc_evidence(wrapper)
                delivered = (
                    np.concatenate(delivered_chunks).astype(np.float32, copy=False)
                    if delivered_chunks
                    else np.empty(0, dtype=np.float32)
                )
                audio_path = lane_dir / f"{case['id']}.wav"
                _write_wav(audio_path, delivered, OUTPUT_SAMPLE_RATE)
                with calls_lock:
                    recorded_calls = list(calls)
                semantic = evaluate_semantic_case(
                    case, response_text=assistant_text, tool_calls=recorded_calls
                )
                transaction = _transaction_checks(positions, pad_id)
                structural = {
                    **transaction,
                    "perception_held_during_direct": perception_after_direct == perception_before,
                    "audio_clock_held_during_direct": audio_frame_after_direct == 0,
                    "eartts_held_during_direct": reset_after_direct == reset_before,
                    "first_settlement_pcm_initialized_perception": bool(
                        first_settlement_pcm
                        and first_settlement_pcm["audio_frame_index_after"]
                        == audio_frame_after_direct + 1
                        and first_settlement_pcm["perception_frame_idx_after"]
                        == perception_after_direct + 1
                    ),
                    "settlement_reached_blank_fence": bool(settlement_evidence["fence_reached"]),
                    "settlement_within_bound": bool(settlement_evidence["within_bound"]),
                    "settlement_pre_eou_clean": bool(
                        settlement_evidence["pre_eou_clean"] and settlement_failure is None
                    ),
                    "eartts_held_during_settlement": reset_after_settlement == reset_after_direct,
                    "eou_boundary_position_exactly_once": bool(
                        boundary_evidence["performed"]
                        and boundary_evidence["audio_frame_index_after"]
                        == boundary_evidence["audio_frame_index_before"] + 1
                        and boundary_evidence["perception_frame_idx_after"]
                        == boundary_evidence["perception_frame_idx_before"] + 1
                    ),
                    "eartts_bos_reset_exactly_once": bool(
                        boundary_evidence["performed"]
                        and boundary_evidence["eartts_reset_count_after"] == reset_after_direct + 1
                    ),
                    "response_started": response_started,
                    "response_complete": response_complete,
                    "response_audio_nonempty": delivered.size > 0,
                }
                case_result = {
                    "case_id": case["id"],
                    "categories": case["categories"],
                    "prompt": case["prompt"],
                    "token_ids": token_ids,
                    "direct_positions": positions,
                    "direct_state": {
                        "perception_frame_idx_before": perception_before,
                        "perception_frame_idx_after": perception_after_direct,
                        "audio_frame_index_after": audio_frame_after_direct,
                        "eartts_reset_count_before": reset_before,
                        "eartts_reset_count_after": reset_after_direct,
                    },
                    "settlement": settlement_evidence,
                    "settlement_failure": settlement_failure,
                    "first_settlement_pcm": first_settlement_pcm,
                    "eou_boundary": boundary_evidence,
                    "assistant_text": assistant_text,
                    "function_text": function_text,
                    "tool_calls": recorded_calls,
                    "audio_path": str(audio_path),
                    "audio_samples": int(delivered.size),
                    "semantic": semantic,
                    "structural_checks": structural,
                    "structural_passed": all(structural.values()),
                }
                if trace_top_k:
                    case_result["function_logit_trace"] = function_logit_trace
                results.append(case_result)
            finally:
                if engine.started:
                    engine.abort()
        lane = {
            "token_form": token_form,
            "case_count": len(results),
            "cases": results,
            "structural_passed": len(results) == len(corpus["cases"])
            and all(case["structural_passed"] for case in results),
            "semantic_pass_count": sum(case["semantic"]["passed"] for case in results),
        }
        lanes.append(lane)

    report = {
        "schema": 3 if trace_top_k else 2,
        "kind": "direct_text_semantic_probe",
        "precision": precision_label,
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256(corpus_path),
        "runtime_provenance": _direct_probe_runtime_provenance(engine),
        "pad_token_id": pad_id,
        "token_forms": token_forms,
        "lanes": lanes,
        "structural_passed": all(lane["structural_passed"] for lane in lanes),
        "passed": all(lane["structural_passed"] for lane in lanes),
    }
    if trace_top_k:
        stt = wrapper.model.stt_model
        report["function_logit_trace"] = {
            "top_k": trace_top_k,
            "post_bos_frames": FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES,
            "phases": ["direct_position", "settlement", "eou_boundary", "post_bos"],
            "watched_token_ids": {
                "pad": int(stt.text_pad_id),
                "sotc": int(wrapper._fc_sotc_id),
                "eotc": int(wrapper._fc_eotc_id),
                "eotr": int(wrapper._fc_eotr_id),
                "text_bos": int(stt.text_bos_id),
            },
        }
    return report
