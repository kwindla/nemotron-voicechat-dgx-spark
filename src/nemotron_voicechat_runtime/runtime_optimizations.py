#!/usr/bin/env python3
"""Opt-in fixed-stage optimizations for the exact public VoiceChat runtime."""

from __future__ import annotations

import inspect
import logging
import os
from functools import update_wrapper
from pathlib import Path
from typing import Any


def enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


class RealtimeCyclicGCGuard:
    """Suspend stop-the-world cyclic GC while realtime streams are active."""

    def __init__(self, gc_module: Any = None):
        if gc_module is None:
            import gc as gc_module

        self.gc = gc_module
        self.active_streams: set[object] = set()
        self.restore_enabled = False

    def start(self, stream_id: object) -> bool:
        if stream_id in self.active_streams:
            return False
        changed = False
        if not self.active_streams:
            self.restore_enabled = bool(self.gc.isenabled())
            if self.restore_enabled:
                self.gc.disable()
                changed = True
        self.active_streams.add(stream_id)
        return changed

    def finish(self, stream_id: object) -> bool:
        self.active_streams.discard(stream_id)
        if self.active_streams or not self.restore_enabled:
            return False
        self.gc.enable()
        self.restore_enabled = False
        return True


def perception_uses_captured_graph(manager: Any) -> bool:
    graph_state = getattr(manager, "cudagraph_state", None)
    return bool(
        getattr(manager, "use_cudagraph", False)
        and graph_state is not None
        and graph_state.is_captured()
    )


def install_perception_precision_override() -> None:
    """Run the qualified perception stage in BF16 without altering its weights."""
    requested = os.environ.get("EA_PERCEPTION_DTYPE", "").strip().lower()
    if not requested or requested in {"fp32", "float32"}:
        return
    if requested not in {"bf16", "bfloat16"}:
        raise ValueError("EA_PERCEPTION_DTYPE must be one of bf16, bfloat16, fp32, float32")

    import torch
    from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
        PerceptionCacheManager,
    )

    if getattr(PerceptionCacheManager, "_voicechat_bf16_installed", False):
        return
    original_setup = PerceptionCacheManager.setup
    original_step = PerceptionCacheManager.step

    def bf16_setup(manager):
        manager.model.stt_model.perception.to(dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return original_setup(manager)

    def bf16_step(manager, *args, **kwargs):
        if perception_uses_captured_graph(manager):
            return original_step(manager, *args, **kwargs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return original_step(manager, *args, **kwargs)

    update_wrapper(bf16_setup, original_setup)
    update_wrapper(bf16_step, original_step)
    PerceptionCacheManager.setup = bf16_setup
    PerceptionCacheManager.step = bf16_step
    PerceptionCacheManager._voicechat_bf16_installed = True


def pad_pair_fc_in_progress(fc_state: dict[str, Any] | None) -> bool:
    """Return whether reactive tool calling requires sequential Nano steps."""
    return bool(
        fc_state
        and (
            fc_state.get("active", False)
            or fc_state.get("forced_function_tokens")
            or fc_state.get("injecting_response", False)
            or fc_state.get("awaiting_response", False)
        )
    )


def pad_pair_should_draft(previous_effective_pad: bool | None) -> bool:
    """Draft only inside a confirmed PAD stretch under the conditional policy."""
    return not enabled("VOICECHAT_NANO_PAD_PAIR_CONDITIONAL") or previous_effective_pad is True


def _state_scalar(state: dict[str, Any], name: str, default: Any) -> Any:
    """Read a batch-one RNNT state value without moving tensors between devices."""
    value = state.get(name)
    if value is None:
        return default
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return value


def pad_pair_precontrol_reason(wrapper: Any, arguments: dict[str, Any]) -> str | None:
    """Predict forced controls early enough to keep a pair from crossing them.

    NVIDIA updates RNNT state after Nano inference, then may overwrite Nano's
    token with BOS/EOS in the same frame. The pre-step counters identify the
    last frame before that threshold. A false positive only costs one safe
    sequential step; a false negative can cross the control boundary.
    """
    if getattr(wrapper, "_external_agent_eos_requested", False):
        return "external_eos"
    if getattr(wrapper, "_redirect_tokens_queue", None):
        return "redirect"

    rnnt = arguments.get("rnnt_partial_hypotheses")
    if not rnnt or getattr(wrapper, "_turn_taking_source", None) != "rnnt":
        return None
    cfg = getattr(wrapper, "model_cfg", {})
    if not cfg.get("force_turn_taking", False):
        return None

    agent_speaking = bool(_state_scalar(rnnt, "agent_speaking", False))
    if agent_speaking:
        nonblank = int(_state_scalar(rnnt, "nonblank_consec", 0))
        bou_frames = int(
            cfg.get(
                "user_bos_frames",
                cfg.get("asr_bou", cfg.get("rnnt_bou_frames", 4)),
            )
        )
        if nonblank >= max(0, bou_frames - 1):
            return "rnnt_bou"
        return None

    blank_count = int(_state_scalar(rnnt, "blank_count", 0))
    speech_confirmed = bool(_state_scalar(rnnt, "speech_confirmed", False))
    user_first_turn = bool(
        _state_scalar(
            rnnt,
            "user_first_turn",
            _state_scalar(rnnt, "first_turn", True),
        )
    )
    eou_frames = int(cfg.get("rnnt_eou_frames", cfg.get("asr_eou", 15)))
    if user_first_turn:
        eou_frames = int(cfg.get("asr_eou_first_turn", eou_frames))
    one_frame_before_eou = blank_count >= max(0, eou_frames - 1)
    transport_speech = bool(
        getattr(wrapper, "_transport_input_activity_available", False)
        and getattr(wrapper, "_transport_user_speech_seen", False)
    )
    transport_silent = int(getattr(wrapper, "_transport_input_silent_frames", 0)) >= max(
        0, eou_frames - 1
    )
    if one_frame_before_eou and (speech_confirmed or (transport_speech and transport_silent)):
        return "rnnt_eou"
    return None


def eartts_pad_should_decode_silence(
    *,
    current_token: int,
    pad_token: int,
    agent_idle: bool,
    prior_content_tokens: int,
    prior_pad_tokens: int,
    tail_ratio: float,
) -> bool:
    """Mirror the existing PAD-tail gate before a code reaches the codec."""
    tail_done = (
        tail_ratio > 0 and not agent_idle and prior_pad_tokens > tail_ratio * prior_content_tokens
    )
    return current_token == pad_token and (agent_idle or tail_done)


def slice_position_outputs(
    custom_outputs: dict[str, Any] | None, position: int
) -> dict[str, Any] | None:
    """Select one accepted position while preserving a length-one batch axis."""
    if not custom_outputs:
        return custom_outputs
    selected: dict[str, Any] = {}
    for name, value in custom_outputs.items():
        if not hasattr(value, "shape") or not value.shape:
            selected[name] = value
            continue
        if position < 0 or position >= int(value.shape[0]):
            raise RuntimeError(
                f"PAD-pair output {name} has {value.shape[0]} rows, "
                f"cannot select position {position}"
            )
        selected[name] = value[position : position + 1]
    return selected


def correct_add_fusion_pending(
    pending: Any,
    *,
    embed_tokens: Any,
    assumed_text: int,
    effective_text: int,
    assumed_function: int,
    effective_function: int,
    agent_text_weight: float,
    function_weight: float,
) -> Any:
    """Correct a delayed fused row after its predecessor output becomes known."""
    import torch

    token_ids = torch.tensor(
        [assumed_text, effective_text, assumed_function, effective_function],
        device=pending.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        embeddings = embed_tokens(token_ids).to(device=pending.device, dtype=pending.dtype)
        text_delta = embeddings[1] - embeddings[0]
        function_delta = embeddings[3] - embeddings[2]
        return (
            pending
            + text_delta.unsqueeze(0) * float(agent_text_weight)
            + function_delta.unsqueeze(0) * float(function_weight)
        )


def _pad_pair_token_id(engine: Any) -> int | None:
    """Read and cache the config-gated public Nano PAD proposal."""
    if hasattr(engine, "_voicechat_pad_pair_token_id"):
        return engine._voicechat_pad_pair_token_id
    config_path = Path(engine.model_path) / "config.json"
    config = {}
    if config_path.is_file():
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
    value = config.get("voicechat_pad_pair_token_id")
    engine._voicechat_pad_pair_token_id = None if value is None else int(value)
    return engine._voicechat_pad_pair_token_id


def _pad_pair_engine_eligible(engine: Any, input_tensors: list[Any]) -> bool:
    """Fail closed unless this is the exact one-input public Nano contract."""
    if not enabled("VOICECHAT_NANO_PAD_PAIR"):
        return False
    specs = getattr(engine, "custom_input_specs", ())
    names = tuple(spec.get("name") if isinstance(spec, dict) else spec.name for spec in specs)
    return (
        names == ("combined_embeds",)
        and len(input_tensors) == 1
        and int(input_tensors[0].shape[0]) == 1
        and _pad_pair_token_id(engine) is not None
    )


async def _generate_packed_pad_pair(
    engine: Any,
    input_tensor: Any,
    request_id: str,
) -> tuple[Any, int]:
    """Run an initialized two-position continuation and select its last accept."""
    import torch
    from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import (
        GenerationResult,
        StreamStatus,
    )

    request_state = engine.requests[request_id]
    if request_state.generation_iterator is None or not request_state.generated_tokens:
        raise RuntimeError("PAD pair requires an initialized continuation request")
    spec = engine.custom_input_specs[0]
    input_dtype = spec.get("dtype") if isinstance(spec, dict) else spec.dtype
    input_dtype = input_dtype or "float32"
    spec_name = spec.get("name") if isinstance(spec, dict) else spec.name
    custom_inputs = {spec_name: input_tensor.to(dtype=getattr(torch, input_dtype)).cpu()}
    await engine.engine.append_request(
        request_id=request_id,
        custom_inputs=custom_inputs,
    )
    output = await request_state.generation_iterator.__anext__()
    completion = output.outputs[0]
    current_tokens = completion.token_ids
    new_tokens = current_tokens[len(request_state.generated_tokens) :]
    if len(new_tokens) not in (1, 2):
        raise RuntimeError(
            f"PAD-pair continuation expected one or two accepted tokens, got {len(new_tokens)}"
        )
    request_state.generated_tokens.extend(new_tokens)
    accepted = len(new_tokens)
    if output.finished:
        request_state.status = StreamStatus.FINISHED
    custom_outputs = slice_position_outputs(
        getattr(completion, "custom_outputs", None), accepted - 1
    )
    return (
        GenerationResult(
            token_id=int(new_tokens[-1]),
            custom_outputs=custom_outputs,
            is_finished=bool(output.finished),
            finish_reason=(completion.finish_reason if output.finished else None),
            total_tokens=len(request_state.generated_tokens),
        ),
        accepted,
    )


def _pad_pair_state(engine: Any, request_id: str) -> dict[str, Any]:
    states = getattr(engine, "_voicechat_pad_pair_states", None)
    if states is None:
        states = {}
        engine._voicechat_pad_pair_states = states
    return states.setdefault(
        request_id,
        {
            "pending": None,
            "needs_correction": False,
            "assumed": None,
            "buffered": 0,
            "accepted": 0,
            "rejected": 0,
            "sequential_bypass": 0,
            "sequential_pending_drains": 0,
            "sequential_mode": False,
            "conditional_singles": 0,
            "previous_effective_pad": None,
            "control_barriers": 0,
            "control_pending_drains": 0,
            "precontrol_barriers": 0,
            "precontrol_pending_drains": 0,
            "precontrol_reasons": {},
        },
    )


def _install_public_pad_pair_engine() -> None:
    """Install the opt-in two-position continuation policy on Nano only."""
    from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import (
        GenerationResult,
        LLMStreamingEngine,
    )
    from nemo.utils import logging as nemo_logging

    if hasattr(LLMStreamingEngine, "_voicechat_original_generate_next_token"):
        return
    original = LLMStreamingEngine.generate_next_token

    async def pad_pair_generate_next_token(
        engine,
        input_tensors,
        prompt_token_ids=None,
        request_id="speech_stream",
    ):
        import torch

        if not _pad_pair_engine_eligible(engine, input_tensors):
            return await original(
                engine,
                input_tensors,
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
            )
        request_state = engine.requests.get(request_id)
        if request_state is None:
            raise RuntimeError(f"Request {request_id} not found. Call start_generation() first.")
        state = _pad_pair_state(engine, request_id)
        if request_state.generation_iterator is None:
            state.update(
                pending=None,
                needs_correction=False,
                assumed=None,
                sequential_mode=False,
                previous_effective_pad=None,
            )
            return await original(
                engine,
                input_tensors,
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
            )

        current = input_tensors[0].detach().clone()
        context = getattr(engine, "_voicechat_pad_pair_call_context", {})
        # NVIDIA's non-blocking function loop calls model_llm_interface
        # directly, outside infer_one_step.  Keep a request-scoped latch so
        # those calls remain sequential even though no fresh wrapper context
        # is installed by that loop.
        bypass = bool(context.get("bypass", False) or state["sequential_mode"])
        assumed = context.get("assumed")
        pending = state["pending"]
        should_draft = pad_pair_should_draft(state["previous_effective_pad"])

        if pending is None:
            if bypass or not should_draft:
                if bypass:
                    state["sequential_bypass"] += 1
                else:
                    state["conditional_singles"] += 1
                return await original(
                    engine,
                    [current],
                    prompt_token_ids=prompt_token_ids,
                    request_id=request_id,
                )
            state["pending"] = current
            state["buffered"] += 1
            pad_id = _pad_pair_token_id(engine)
            nemo_logging.info(
                "Nano PAD-pair buffered request=%s buffered=%d",
                request_id,
                state["buffered"],
            )
            return GenerationResult(
                token_id=pad_id,
                custom_outputs={
                    "function_tokens": current.new_tensor([pad_id], dtype=torch.long).cpu()
                },
                is_finished=False,
                total_tokens=len(request_state.generated_tokens),
            )

        if bypass:
            # A function SOTC can be the accepted first position of a rejected
            # PAD draft, leaving the next source embedding delayed.  The async
            # FC loop supplies replaceable silence, so execute the delayed row
            # exactly once, discard that synthetic current row, and remove the
            # one-position lag before continuing sequentially.
            result = await original(
                engine,
                [pending],
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
            )
            state["pending"] = None
            state["needs_correction"] = False
            state["assumed"] = None
            state["sequential_bypass"] += 1
            state["sequential_pending_drains"] += 1
            return result

        if not should_draft:
            result = await original(
                engine,
                [pending],
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
            )
            state["pending"] = current
            state["needs_correction"] = result is not None
            state["assumed"] = assumed
            state["conditional_singles"] += 1
            return result

        packed = torch.cat((pending, current), dim=0)
        result, accepted = await _generate_packed_pad_pair(engine, packed, request_id)
        if accepted == 2:
            state["pending"] = None
            state["needs_correction"] = False
            state["assumed"] = None
            state["accepted"] += 1
        else:
            state["pending"] = current
            state["needs_correction"] = True
            state["assumed"] = assumed
            state["rejected"] += 1
        return result

    LLMStreamingEngine._voicechat_original_generate_next_token = original
    LLMStreamingEngine.generate_next_token = pad_pair_generate_next_token


def _prepare_pad_pair_call(wrapper: Any, arguments: dict[str, Any]) -> Any | None:
    """Attach post-state-machine correction context before NVIDIA's frame loop."""
    if not enabled("VOICECHAT_NANO_PAD_PAIR"):
        return None
    model = getattr(wrapper, "model_llm_interface", None)
    engine = getattr(model, "engine", None)
    if engine is None or _pad_pair_token_id(engine) is None:
        return None
    if getattr(wrapper, "_has_asr_head", False):
        raise RuntimeError("Nano PAD-pair does not support a recurrent ASR head")
    if type(wrapper.fusion_module).__name__ != "AddFusion":
        raise RuntimeError("Nano PAD-pair requires NVIDIA AddFusion")
    num_frames = int(arguments.get("num_frames_per_chunk", 1))
    if num_frames != 1:
        raise RuntimeError("Nano PAD-pair requires one source frame per call")
    frame_idx = int(arguments["frame_idx"])
    gen_text = arguments["gen_text"]
    gen_function = arguments.get("gen_function_text")
    if gen_function is None:
        raise RuntimeError("Nano PAD-pair requires the always-on function channel")
    request_id = arguments.get("request_id") or wrapper.request_id
    assumed = None
    if frame_idx > 0:
        assumed = (
            int(gen_text[:, frame_idx - 1].item()),
            int(gen_function[:, frame_idx - 1].item()),
        )
    fc_in_progress = pad_pair_fc_in_progress(arguments.get("fc_state"))
    state = _pad_pair_state(engine, request_id)
    if fc_in_progress:
        state["sequential_mode"] = True
    elif state["sequential_mode"] and state["pending"] is None:
        # The first ordinary frame after the async function cycle may resume
        # the conditional policy.  previous_effective_pad still prevents an
        # immediate draft unless the last effective outputs were both PAD.
        state["sequential_mode"] = False
    if enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"):
        precontrol_reason = pad_pair_precontrol_reason(wrapper, arguments)
        if precontrol_reason is not None:
            state["sequential_mode"] = True
            state["precontrol_barriers"] += 1
            reasons = state["precontrol_reasons"]
            reasons[precontrol_reason] = reasons.get(precontrol_reason, 0) + 1
            if state["pending"] is not None:
                state["precontrol_pending_drains"] += 1
    engine._voicechat_pad_pair_call_context = {
        "request_id": request_id,
        "frame_idx": frame_idx,
        "assumed": assumed,
        "bypass": fc_in_progress,
    }
    return engine


def _finalize_pad_pair_call(wrapper: Any, engine: Any | None, arguments: dict[str, Any]) -> None:
    """Apply effective post-FC text/function tokens to a delayed fused row."""
    if engine is None:
        return
    context = getattr(engine, "_voicechat_pad_pair_call_context", {})
    request_id = context.get("request_id")
    state = _pad_pair_state(engine, request_id)
    frame_idx = int(arguments["frame_idx"])
    gen_text = arguments["gen_text"]
    gen_function = arguments["gen_function_text"]
    effective_text = int(gen_text[:, frame_idx].item())
    effective_function = int(gen_function[:, frame_idx].item())
    pad_id = _pad_pair_token_id(engine)
    state["previous_effective_pad"] = effective_text == pad_id and effective_function == pad_id
    if enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"):
        stt = wrapper.model.stt_model
        control_tokens = {int(stt.text_bos_id), int(stt.text_eos_id)}
        if effective_text in control_tokens:
            state["control_barriers"] += 1
            if state["pending"] is not None:
                # RNNT/transport control injection happens after Nano returns.
                # Reuse the proven request-scoped SOTC drain on the next call
                # so a delayed row cannot carry the pre-control PAD context
                # across the turn boundary.
                state["sequential_mode"] = True
                state["control_pending_drains"] += 1
    if pad_pair_fc_in_progress(arguments.get("fc_state")):
        # This post-state check catches the SOTC transition itself.  NVIDIA's
        # async loop starts immediately afterward and does not re-enter this
        # wrapper hook, so the engine-side latch must already be armed.
        state["sequential_mode"] = True
    if not state["needs_correction"]:
        return
    assumed = state.get("assumed")
    if assumed is None or state["pending"] is None:
        raise RuntimeError("PAD-pair delayed row lacks its assumed predecessor tokens")
    fusion = wrapper.fusion_module
    state["pending"] = correct_add_fusion_pending(
        state["pending"],
        embed_tokens=wrapper.model.stt_model.embed_tokens,
        assumed_text=assumed[0],
        effective_text=effective_text,
        assumed_function=assumed[1],
        effective_function=effective_function,
        agent_text_weight=fusion.agent_text_weight,
        function_weight=fusion.function_weight,
    )
    state["needs_correction"] = False
    state["assumed"] = None


def public_pad_pair_status(wrapper: Any) -> dict[str, Any]:
    """Expose bounded live policy/counter state without tensor payloads."""
    model = getattr(wrapper, "model_llm_interface", None)
    engine = getattr(model, "engine", None)
    if engine is None:
        return {"enabled": False, "eligible": False, "requests": {}}
    pad_id = _pad_pair_token_id(engine)
    states = getattr(engine, "_voicechat_pad_pair_states", {})
    requests = {
        request_id: {
            key: value
            for key, value in state.items()
            if key
            in {
                "buffered",
                "accepted",
                "rejected",
                "sequential_bypass",
                "sequential_pending_drains",
                "conditional_singles",
                "control_barriers",
                "control_pending_drains",
                "precontrol_barriers",
                "precontrol_pending_drains",
                "precontrol_reasons",
            }
        }
        | {
            "pending": state.get("pending") is not None,
            "sequential_mode": bool(state.get("sequential_mode")),
        }
        for request_id, state in states.items()
    }
    return {
        "enabled": enabled("VOICECHAT_NANO_PAD_PAIR"),
        "eligible": pad_id is not None,
        "pad_token_id": pad_id,
        "conditional": enabled("VOICECHAT_NANO_PAD_PAIR_CONDITIONAL"),
        "control_barrier": enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"),
        "requests": requests,
    }


def _public_vllm_models(wrapper: Any) -> list[Any]:
    models = [getattr(wrapper, "model_llm_interface", None)]
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    models.append(getattr(tts_model, "tts_model", None))
    return [
        model
        for model in models
        if model is not None
        and hasattr(model, "_run_async")
        and getattr(getattr(model, "engine", None), "engine", None) is not None
    ]


async def generate_next_token_delta(
    stream_engine: Any,
    input_tensors: list[Any],
    prompt_token_ids: list[int] | None = None,
    request_id: str = "speech_stream",
) -> Any:
    """Speech streaming step adapted to vLLM's one-token DELTA outputs."""
    import torch
    from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import (
        GenerationResult,
        StreamStatus,
    )
    from nemo.utils import logging as nemo_logging

    if request_id not in stream_engine.requests:
        raise RuntimeError(f"Request {request_id} not found. Call start_generation() first.")
    request_state = stream_engine.requests[request_id]
    if request_state.status != StreamStatus.ACTIVE:
        return None
    if len(input_tensors) != len(stream_engine.custom_input_specs):
        raise AssertionError(
            f"Expected {len(stream_engine.custom_input_specs)} input tensors, "
            f"got {len(input_tensors)}"
        )

    custom_inputs: dict[str, Any] = {}
    max_length = 1
    for tensor, spec in zip(input_tensors, stream_engine.custom_input_specs):
        input_dtype = spec.dtype or "float32"
        if spec.dim is not None and spec.dim != tensor.shape[-1]:
            raise ValueError(
                f"Input tensor dimension mismatch for {spec.name}: "
                f"expected {spec.dim}, got {tensor.shape[-1]}"
            )
        custom_inputs[spec.name] = tensor.to(dtype=getattr(torch, input_dtype)).cpu()
        max_length = max(max_length, tensor.shape[0])

    try:
        if request_state.generation_iterator is None:
            prompt_tokens = (
                prompt_token_ids
                if prompt_token_ids is not None
                else stream_engine._get_safe_prompt_tokens(max_length)
            )
            if len(prompt_tokens) != max_length:
                raise AssertionError(
                    f"Prompt tokens length {len(prompt_tokens)} does not match "
                    f"input length {max_length}"
                )
            request_state.generation_iterator = stream_engine.engine.generate(
                {
                    "prompt_token_ids": prompt_tokens,
                    "custom_inputs": custom_inputs,
                },
                stream_engine.sampling_params,
                request_id=request_id,
            )
        elif request_state.generated_tokens:
            try:
                await stream_engine.engine.append_request(
                    request_id=request_id,
                    custom_inputs=custom_inputs,
                )
            except ValueError as exc:
                if "not found" not in str(exc):
                    raise
                request_state.status = StreamStatus.FINISHED
                return None

        output = await request_state.generation_iterator.__anext__()
        delta_tokens = output.outputs[0].token_ids
        if not delta_tokens:
            if output.finished:
                request_state.status = StreamStatus.FINISHED
            return None
        if len(delta_tokens) != 1:
            raise AssertionError(f"Expected exactly one DELTA token, got {len(delta_tokens)}")
        latest_token = delta_tokens[0]
        request_state.generated_tokens.append(latest_token)
        finished = bool(output.finished)
        if finished:
            request_state.status = StreamStatus.FINISHED
        completion = output.outputs[0]
        return GenerationResult(
            token_id=latest_token,
            custom_outputs=getattr(completion, "custom_outputs", None),
            is_finished=finished,
            finish_reason=(getattr(completion, "finish_reason", None) if finished else None),
            total_tokens=len(request_state.generated_tokens),
        )
    except StopAsyncIteration:
        request_state.status = StreamStatus.FINISHED
        return None
    except Exception as exc:
        nemo_logging.error(
            "Error in DELTA generate_next_token for request %s: %s",
            request_id,
            exc,
        )
        request_state.status = StreamStatus.FINISHED
        return None


def install_vllm_delta_output(wrapper: Any, target: str = "eartts") -> list[str]:
    """Opt selected streaming engines into non-cumulative vLLM outputs."""
    from vllm.sampling_params import RequestOutputKind

    installed: list[str] = []
    for model in _public_vllm_models(wrapper):
        stream_engine = model.engine
        label = os.path.basename(os.path.normpath(stream_engine.model_path))
        if target not in label.lower():
            continue
        stream_engine.sampling_params.output_kind = RequestOutputKind.DELTA

        async def delta_generate(*args, __engine=stream_engine, **kwargs):
            return await generate_next_token_delta(__engine, *args, **kwargs)

        stream_engine.generate_next_token = delta_generate
        installed.append(label)
    return installed


def install_eartts_decode_pad_silence(wrapper: Any) -> bool:
    """Apply the existing PAD-tail policy before codes enter live decoding."""
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    original = getattr(tts_model, "infer_codes_one_step", None)
    if original is None or getattr(tts_model, "_decode_pad_silence_installed", False):
        return False

    def guarded_infer_codes_one_step(*args, **kwargs):
        code, past_key_values = original(*args, **kwargs)
        current = kwargs.get("current_subword_id")
        if current is None or current.numel() != 1:
            return code, past_key_values
        ratio = float(os.environ.get("S2S_TTS_PAD_TAIL_RATIO", "3"))
        if eartts_pad_should_decode_silence(
            current_token=int(current.item()),
            pad_token=int(tts_model.text_pad_id),
            agent_idle=bool(wrapper._get_agent_idle()),
            prior_content_tokens=int(getattr(wrapper, "_tts_in_turn_content", 0)),
            prior_pad_tokens=int(getattr(wrapper, "_tts_in_turn_pads", 0)),
            tail_ratio=ratio,
        ):
            silence = tts_model.codec_silence_tokens.view(1, 1, -1).expand_as(code)
            code = silence.clone()
        return code, past_key_values

    tts_model.infer_codes_one_step = guarded_infer_codes_one_step
    tts_model._decode_pad_silence_installed = True
    return True


def install_eartts_reset_on_bos(wrapper: Any) -> bool:
    """Reset only the EarTTS vLLM request immediately before each spoken turn."""
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    backend = getattr(tts_model, "tts_model", None)
    original = getattr(tts_model, "infer_codes_one_step", None)
    if original is None or backend is None or getattr(tts_model, "_reset_on_bos_installed", False):
        return False

    def reset_on_bos_infer_codes_one_step(*args, **kwargs):
        current = kwargs.get("current_subword_id")
        request_id = kwargs.get("request_id")
        if (
            current is not None
            and current.numel() == 1
            and int(current.item()) == int(tts_model.text_bos_id)
            and request_id is not None
        ):
            backend.abort_request(request_id)
            backend(
                wrapper.tts_init_inputs,
                request_id=request_id,
                prompt_token_ids=wrapper.tts_prompt_token_ids,
            )
            initial_code = wrapper.first_tts_code_input
            if initial_code is None:
                raise RuntimeError("EarTTS BOS reset requires first_tts_code_input")
            kwargs["prev_audio_tokens"] = initial_code.detach().clone()
            logging.info("Reset and speaker-prefilled EarTTS request %s at BOS", request_id)
        return original(*args, **kwargs)

    tts_model.infer_codes_one_step = reset_on_bos_infer_codes_one_step
    tts_model._reset_on_bos_installed = True
    return True


def install_public_wrapper_hooks() -> None:
    """Install pre-construction BF16 and per-stream runtime hooks.

    This must run before S2SPipelineBuilder constructs the public wrapper so
    perception graph capture occurs using the requested precision.
    """
    from nemo.collections.speechlm2.inference.model_wrappers import (
        nemotron_voicechat_inference_wrapper as wrapper_module,
    )

    NemotronVoicechatInferenceWrapper = wrapper_module.NemotronVoicechatInferenceWrapper
    from nemo.utils import logging

    install_perception_precision_override()
    _install_public_pad_pair_engine()
    if getattr(NemotronVoicechatInferenceWrapper, "_public_fixed_stage_hooks", False):
        return
    original = NemotronVoicechatInferenceWrapper.infer_one_step
    original_signature = inspect.signature(original)

    def optimized_infer_one_step(wrapper, *args, **kwargs):
        if enabled("EA_DISABLE_CYCLIC_GC"):
            guard = getattr(wrapper, "_ea_realtime_gc_guard", None)
            if guard is None:
                guard = RealtimeCyclicGCGuard()
                wrapper._ea_realtime_gc_guard = guard
            stream_id = kwargs.get("stream_id", 0)
            if guard.start(0 if stream_id is None else stream_id):
                logging.info("Disabled cyclic GC while realtime streams are active")

        if enabled("EA_CPU_CODEC_ASYNC") and not hasattr(wrapper, "_ea_async_codec_pipeline"):
            if not enabled("EA_CPU_CODEC_PROCESS"):
                raise RuntimeError("The public CPU codec path requires EA_CPU_CODEC_PROCESS=1")
            from .cpu_codec_offload import OrderedProcessCodecDecode

            codec = wrapper.model.tts_model.audio_codec
            if not hasattr(codec, "_ea_cpu_bridged_decode"):
                raise RuntimeError("EA_CPU_CODEC_ASYNC=1 requires the public codec to be prepared")
            pipeline = OrderedProcessCodecDecode(
                codec, samples_per_frame=wrapper._samples_per_audio_output_frame()
            )
            wrapper._ea_async_codec_pipeline = pipeline
            object.__setattr__(codec, "decode", pipeline.decode)
            logging.info(
                "Enabled public ordered two-core CPU codec with one-frame buffering: %s",
                pipeline.worker_details,
            )
        bound_arguments = None
        pad_pair_engine = None
        if enabled("VOICECHAT_NANO_PAD_PAIR"):
            bound_arguments = dict(
                original_signature.bind_partial(wrapper, *args, **kwargs).arguments
            )
            bound_arguments.pop("self", None)
            pad_pair_engine = _prepare_pad_pair_call(wrapper, bound_arguments)
        result = original(wrapper, *args, **kwargs)
        if bound_arguments is not None:
            _finalize_pad_pair_call(wrapper, pad_pair_engine, bound_arguments)
        return result

    update_wrapper(optimized_infer_one_step, original)
    NemotronVoicechatInferenceWrapper.infer_one_step = optimized_infer_one_step
    NemotronVoicechatInferenceWrapper._public_fixed_stage_hooks = True


def prepare_public_fixed_stages(pipeline: Any, checkpoint_root: Path) -> dict[str, Any]:
    """Prepare post-construction stages whose weights already reside in the model."""
    details: dict[str, Any] = {
        "perception_dtype": os.environ.get("EA_PERCEPTION_DTYPE", "fp32") or "fp32",
        "cyclic_gc_guard": enabled("EA_DISABLE_CYCLIC_GC"),
        "cpu_codec": False,
        "vllm_delta_output": [],
        "eartts_decode_pad_silence": False,
        "eartts_reset_on_bos": False,
        "nano_pad_pair_control_barrier": enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"),
    }
    if enabled("VOICECHAT_EARTTS_DECODE_PAD_SILENCE"):
        details["eartts_decode_pad_silence"] = install_eartts_decode_pad_silence(pipeline.s2s_model)
    if enabled("VOICECHAT_EARTTS_RESET_ON_BOS"):
        details["eartts_reset_on_bos"] = install_eartts_reset_on_bos(pipeline.s2s_model)
    if enabled("VOICECHAT_VLLM_DELTA_OUTPUT"):
        details["vllm_delta_output"] = install_vllm_delta_output(
            pipeline.s2s_model,
            os.environ.get("VOICECHAT_VLLM_DELTA_OUTPUT_TARGET", "eartts"),
        )
    if enabled("EA_CPU_CODEC_PROCESS"):
        if not enabled("EA_CPU_CODEC") or not enabled("EA_CPU_CODEC_ASYNC"):
            raise RuntimeError(
                "Public process codec requires EA_CPU_CODEC=1 and EA_CPU_CODEC_ASYNC=1"
            )
        from .cpu_codec_offload import OrderedProcessCodecDecode, prepare_process_codec

        codec_details = prepare_process_codec(
            pipeline.s2s_model.model, checkpoint_root=checkpoint_root
        )
        details["cpu_codec"] = codec_details
        details["cpu_codec"]["pointwise_linear"] = enabled("EA_CPU_CODEC_POINTWISE_LINEAR")
        details["cpu_codec"]["persistent"] = enabled("EA_CPU_CODEC_PERSISTENT")
        if enabled("EA_CPU_CODEC_PERSISTENT"):
            import torch

            wrapper = pipeline.s2s_model
            codec = wrapper.model.tts_model.audio_codec
            codec_pipeline = OrderedProcessCodecDecode(
                codec, samples_per_frame=wrapper._samples_per_audio_output_frame()
            )
            silence = wrapper.model.tts_model.codec_silence_tokens.detach()
            warmup_codes = silence.reshape(1, 1, -1)
            warmup_lengths = torch.ones(
                (warmup_codes.shape[0],), dtype=torch.long, device=warmup_codes.device
            )
            warmup = codec_pipeline.warmup(warmup_codes, warmup_lengths)
            wrapper._ea_async_codec_pipeline = codec_pipeline
            object.__setattr__(codec, "decode", codec_pipeline.decode)
            details["cpu_codec"]["worker"] = codec_pipeline.worker_details
            details["cpu_codec"]["warmup_worker_ms"] = warmup["worker_ms"]
            logging.info(
                "Prewarmed persistent public CPU codec worker: worker=%s warmup_ms=%.3f",
                codec_pipeline.worker_details,
                warmup["worker_ms"],
            )
    return details


def finalize_public_fixed_stages(wrapper: Any, stream_id: object, state: Any) -> dict[str, Any]:
    """Drain buffered codec work and restore stream-scoped cyclic GC."""
    details: dict[str, Any] = {}
    async_codec = getattr(wrapper, "_ea_async_codec_pipeline", None)
    if async_codec is not None:
        state.audio_buffer = async_codec.finalize_audio(
            state.audio_buffer, output_device=wrapper.device
        )
        persistent = enabled("EA_CPU_CODEC_PERSISTENT")
        details["codec_submitted"] = (
            async_codec.finish_session() if persistent else async_codec.submitted
        )
        details["codec_worker"] = async_codec.worker_details
        details["codec_persistent"] = persistent
        if not persistent:
            codec = wrapper.model.tts_model.audio_codec
            object.__setattr__(codec, "decode", codec._ea_cpu_bridged_decode)
            async_codec.close()
            delattr(wrapper, "_ea_async_codec_pipeline")
    guard = getattr(wrapper, "_ea_realtime_gc_guard", None)
    if guard is not None:
        details["cyclic_gc_restored"] = guard.finish(0 if stream_id is None else stream_id)
    return details
