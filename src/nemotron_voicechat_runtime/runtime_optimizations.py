#!/usr/bin/env python3
"""Opt-in fixed-stage optimizations for the exact public VoiceChat runtime."""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from collections import deque
from functools import update_wrapper
from pathlib import Path
from typing import Any


def _runtime_tensor_identity(value: Any) -> Any:
    """Return a synchronization-free identity for immutable runtime inputs."""

    if hasattr(value, "data_ptr") and hasattr(value, "shape"):
        return (
            "tensor",
            id(value),
            int(value.data_ptr()),
            int(getattr(value, "_version", 0)),
            tuple(int(part) for part in value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (str(key), _runtime_tensor_identity(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_runtime_tensor_identity(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"unsupported prepared-epoch input type: {type(value).__name__}")


class EarTTSPreparedEpochController:
    """Own one request-bound, single-use EarTTS speaker-prefill epoch."""

    def __init__(self, wrapper: Any, tts_model: Any, backend: Any):
        self.wrapper = wrapper
        self.tts_model = tts_model
        self.backend = backend
        self.lock = threading.RLock()
        self.session_epoch = 0
        self.prepare_epoch = 0
        self.state = "invalid"
        self.reason = "not_attested"
        self.prepared: dict[str, Any] | None = None
        self.armed_client_turn_id: int | None = None
        self.prepare_count = 0
        self.prepare_failure_count = 0
        self.prepared_reuse_count = 0
        self.bos_transition_count = 0
        self.last_prepare: dict[str, Any] | None = None
        self.last_bos_event: dict[str, Any] | None = None
        stream_engine = getattr(backend, "engine", None)
        if not bool(getattr(stream_engine, "_unique_backend_request_ids", False)):
            raise RuntimeError(
                "EarTTS prepared epoch requires generation-unique backend request IDs"
            )
        if not callable(getattr(stream_engine, "resolve_backend_request_id", None)):
            raise RuntimeError(
                "EarTTS prepared epoch requires a backend request-ID resolver"
            )

    def _request_snapshot(self, request_id: str) -> dict[str, Any]:
        engine = getattr(self.backend, "engine", None)
        requests = getattr(engine, "requests", None)
        request = requests.get(request_id) if hasattr(requests, "get") else None
        status = getattr(getattr(request, "status", None), "value", None)
        generated = getattr(request, "generated_tokens", None)
        iterator = getattr(request, "generation_iterator", None)
        backend_request_id = getattr(request, "backend_request_id", None)
        backend_generation = getattr(request, "backend_generation", None)
        if request is None or status != "active" or generated is None or iterator is None:
            raise RuntimeError("EarTTS prepared epoch requires an active initialized request")
        if (
            not isinstance(backend_request_id, str)
            or not backend_request_id
            or backend_request_id == request_id
            or isinstance(backend_generation, bool)
            or not isinstance(backend_generation, int)
            or backend_generation < 1
        ):
            raise RuntimeError(
                "EarTTS prepared epoch requires a unique physical backend request"
            )
        resolved = engine.resolve_backend_request_id(request_id)
        if resolved != backend_request_id:
            raise RuntimeError("EarTTS backend request-ID resolver is inconsistent")
        return {
            "request_id": request_id,
            "backend_request_id": backend_request_id,
            "backend_generation": backend_generation,
            "request_state_id": id(request),
            "iterator_id": id(iterator),
            "generated_tokens": len(generated),
            "speaker_identity": _runtime_tensor_identity(self.wrapper.tts_init_inputs),
            "prompt_token_ids": tuple(int(token) for token in self.wrapper.tts_prompt_token_ids),
            "initial_code_identity": _runtime_tensor_identity(
                self.wrapper.first_tts_code_input
            ),
        }

    def _matches(self, request_id: str) -> tuple[bool, str]:
        if self.prepared is None:
            return False, "missing_prepared_snapshot"
        if self.prepared.get("session_epoch") != self.session_epoch:
            return False, "session_epoch_mismatch"
        if self.prepared.get("prepare_epoch") != self.prepare_epoch:
            return False, "prepare_epoch_mismatch"
        try:
            current = self._request_snapshot(request_id)
        except Exception:
            return False, "request_inactive"
        for key in (
            "request_id",
            "backend_request_id",
            "backend_generation",
            "request_state_id",
            "iterator_id",
            "generated_tokens",
            "speaker_identity",
            "prompt_token_ids",
            "initial_code_identity",
        ):
            if current.get(key) != self.prepared.get(key):
                return False, f"{key}_mismatch"
        return True, "exact_match"

    def _publish_clean(self, request_id: str, *, reason: str) -> None:
        snapshot = self._request_snapshot(request_id)
        self.prepare_epoch += 1
        snapshot.update(
            session_epoch=self.session_epoch,
            prepare_epoch=self.prepare_epoch,
        )
        self.prepared = snapshot
        self.armed_client_turn_id = None
        self.state = "clean_unarmed"
        self.reason = reason
        # A BOS event is evidence for the epoch that produced it. Publishing a
        # new clean epoch makes the preceding event stale by construction; the
        # next successful BOS will publish a new event before its frame can be
        # exposed by the server.
        self.last_bos_event = None

    def invalidate(self, reason: str) -> None:
        with self.lock:
            self.state = "invalid"
            self.reason = reason
            self.prepared = None
            self.armed_client_turn_id = None

    def reset_session(self, reason: str = "session_reset") -> None:
        with self.lock:
            self.session_epoch += 1
            self.prepare_count = 0
            self.prepare_failure_count = 0
            self.prepared_reuse_count = 0
            self.bos_transition_count = 0
            self.state = "invalid"
            self.reason = reason
            self.prepared = None
            self.armed_client_turn_id = None
            self.last_prepare = None
            self.last_bos_event = None
            self.tts_model._reset_on_bos_count = 0
            self.tts_model._reset_on_bos_last_request_id = None
            self.tts_model._reset_on_bos_last_timing_ms = None

    def attest_existing_prefill(self, request_id: str) -> dict[str, Any]:
        with self.lock:
            self._publish_clean(str(request_id), reason="session_prefill_attested")
            return self.status()

    def prepare(
        self,
        request_id: str,
        *,
        quiescent_check: Any,
        site: str = "turn_start_prepare",
    ) -> dict[str, Any]:
        request_id = str(request_id)
        with self.lock:
            started = time.perf_counter()
            if not callable(quiescent_check) or not bool(quiescent_check()):
                self.invalidate("function_cycle_not_quiescent")
                self.prepare_failure_count += 1
                return {"passed": False, **self.status(), "timing": None}
            matched, _ = self._matches(request_id)
            if self.state == "clean_unarmed" and matched:
                self.last_prepare = {
                    "schema": 1,
                    "site": site,
                    "outcome": "already_clean",
                    "abort_call_ms": 0.0,
                    "prefill_call_ms": 0.0,
                    "total_ms": (time.perf_counter() - started) * 1000.0,
                }
                return {"passed": True, **self.status(), "timing": dict(self.last_prepare)}

            self.invalidate("preparation_started")
            abort_finished = started
            prefill_finished = None
            try:
                requests = getattr(getattr(self.backend, "engine", None), "requests", None)
                old_request = requests.get(request_id) if hasattr(requests, "get") else None
                if not bool(self.backend.abort_request(request_id)):
                    raise RuntimeError("EarTTS prepare could not abort the active request")
                abort_finished = time.perf_counter()
                if hasattr(requests, "get") and requests.get(request_id) is old_request:
                    raise RuntimeError("EarTTS prepare retained the aborted request state")
                self.backend(
                    self.wrapper.tts_init_inputs,
                    request_id=request_id,
                    prompt_token_ids=self.wrapper.tts_prompt_token_ids,
                )
                prefill_finished = time.perf_counter()
                if not bool(quiescent_check()):
                    raise RuntimeError("function cycle became active during EarTTS prepare")
                if hasattr(requests, "get") and requests.get(request_id) is old_request:
                    raise RuntimeError("EarTTS prepare reused the prior request state")
                self._publish_clean(request_id, reason="turn_start_prepared")
                self.prepare_count += 1
                self.last_prepare = {
                    "schema": 1,
                    "site": site,
                    "outcome": "prepared",
                    "abort_call_ms": (abort_finished - started) * 1000.0,
                    "prefill_call_ms": (prefill_finished - abort_finished) * 1000.0,
                    "total_ms": (prefill_finished - started) * 1000.0,
                }
                return {"passed": True, **self.status(), "timing": dict(self.last_prepare)}
            except Exception:
                self.prepare_failure_count += 1
                self.invalidate("preparation_failed")
                self.last_prepare = {
                    "schema": 1,
                    "site": site,
                    "outcome": "failed",
                    "abort_call_ms": max(0.0, (abort_finished - started) * 1000.0),
                    "prefill_call_ms": (
                        None
                        if prefill_finished is None
                        else (prefill_finished - abort_finished) * 1000.0
                    ),
                    "total_ms": (time.perf_counter() - started) * 1000.0,
                }
                return {"passed": False, **self.status(), "timing": dict(self.last_prepare)}

    def arm(
        self,
        request_id: str,
        client_turn_id: int,
        *,
        quiescent_check: Any,
    ) -> dict[str, Any]:
        with self.lock:
            if (
                not isinstance(client_turn_id, int)
                or isinstance(client_turn_id, bool)
                or client_turn_id < 1
            ):
                raise ValueError("prepared epoch requires a positive client_turn_id")
            if self.state == "clean_armed":
                return {"passed": False, **self.status(), "reason": "already_armed"}
            if not callable(quiescent_check) or not bool(quiescent_check()):
                self.invalidate("function_cycle_not_quiescent")
                return {"passed": False, **self.status()}
            matched, mismatch = self._matches(str(request_id))
            if self.state != "clean_unarmed" or not matched:
                self.invalidate(mismatch if not matched else "not_clean_unarmed")
                return {"passed": False, **self.status()}
            self.state = "clean_armed"
            self.reason = "client_commit_armed"
            self.armed_client_turn_id = client_turn_id
            return {"passed": True, **self.status()}

    def consume_prepared_bos(self, request_id: str) -> tuple[bool, str, int | None]:
        """Consume a matching arm before BOS inference; failures stay invalid."""

        matched, mismatch = self._matches(str(request_id))
        armed_turn = self.armed_client_turn_id
        if self.state != "clean_armed" or armed_turn is None or not matched:
            reason = mismatch if not matched else "prepared_epoch_not_armed"
            self.invalidate(reason)
            return False, reason, None
        self.state = "dirty"
        self.reason = "prepared_bos_consumed"
        self.armed_client_turn_id = None
        return True, "exact_prepared_epoch", armed_turn

    def mark_real_call(self, reason: str) -> None:
        self.state = "dirty"
        self.reason = reason
        self.prepared = None
        self.armed_client_turn_id = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            identity = self.prepared or self.last_bos_event
            return {
                "schema": 2,
                "enabled": True,
                "state": self.state,
                "reason": self.reason,
                "session_epoch": self.session_epoch,
                "prepare_epoch": self.prepare_epoch,
                "request_id": (
                    None
                    if self.prepared is None
                    else str(self.prepared.get("request_id"))
                ),
                "backend_request_id": (
                    None
                    if identity is None
                    else str(identity.get("backend_request_id"))
                ),
                "backend_generation": (
                    None
                    if identity is None
                    else int(identity.get("backend_generation", 0))
                ),
                "generated_tokens": (
                    None
                    if self.prepared is None
                    else int(self.prepared.get("generated_tokens", 0))
                ),
                "armed_client_turn_id": self.armed_client_turn_id,
                "prepare_count": self.prepare_count,
                "prepare_failure_count": self.prepare_failure_count,
                "prepared_reuse_count": self.prepared_reuse_count,
                "bos_transition_count": self.bos_transition_count,
                "last_bos_event": (
                    None if self.last_bos_event is None else dict(self.last_bos_event)
                ),
            }


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
    if getattr(wrapper, "_external_user_eou_requested", False):
        return "external_user_eou"
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


def eartts_idle_pad_bypass_eligible(
    *,
    enabled: bool,
    decode_audio: bool,
    use_vllm_eartts: bool,
    ordinary_pcm: bool,
    source_embeddings_supplied: bool,
    force_output_pad: bool,
    current_token: int,
    pad_token: int,
    agent_idle: bool,
    fc_state: dict[str, Any] | None,
    acoustic_capture_enabled: bool,
) -> bool:
    """Return whether one PAD position may skip EarTTS generation.

    This is deliberately narrower than the PAD-to-decoder-silence policy.  It
    applies only to ordinary PCM positions while the agent and the complete
    function-call state machine are idle.  Direct text source positions still
    advance EarTTS as part of their qualified hidden-position transaction.
    """
    fc_quiescent = not (
        fc_state
        and (
            fc_state.get("active", False)
            or fc_state.get("awaiting_response", False)
            or fc_state.get("awaiting_eotr", False)
            or fc_state.get("injecting_response", False)
            or fc_state.get("forced_function_tokens")
        )
    )
    decoder_silence_eligible = eartts_pad_should_decode_silence(
        current_token=current_token,
        pad_token=pad_token,
        agent_idle=agent_idle,
        prior_content_tokens=0,
        prior_pad_tokens=0,
        tail_ratio=0.0,
    )
    return bool(
        enabled
        and decode_audio
        and use_vllm_eartts
        and ordinary_pcm
        and not source_embeddings_supplied
        and not force_output_pad
        and decoder_silence_eligible
        and fc_quiescent
        and not acoustic_capture_enabled
    )


def eartts_idle_pad_bypass_transition(
    *, recurrent_code: Any, past_key_values: Any, codec_silence_tokens: Any
) -> tuple[Any, Any, Any]:
    """Create decoder silence without mutating EarTTS recurrent state."""
    if not hasattr(recurrent_code, "shape") or len(recurrent_code.shape) != 3:
        raise RuntimeError("EarTTS recurrent code must have shape [B, T, Q]")
    if int(recurrent_code.shape[1]) != 1:
        raise RuntimeError("EarTTS idle-PAD bypass requires one recurrent position")
    if not hasattr(codec_silence_tokens, "numel"):
        raise RuntimeError("EarTTS codec silence tokens are unavailable")
    if int(codec_silence_tokens.numel()) != int(recurrent_code.shape[2]):
        raise RuntimeError("EarTTS codec silence shape does not match recurrent code")
    decoder_code = (
        codec_silence_tokens.view(1, 1, -1).expand_as(recurrent_code).clone()
    )
    return decoder_code, recurrent_code, past_key_values


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


def _pad_pair_buffered_custom_outputs(current: Any, pad_id: int) -> dict[str, Any]:
    """Build the synthetic PAD result, stamping it only for logit diagnostics."""
    import torch

    outputs = {
        "function_tokens": current.new_tensor([pad_id], dtype=torch.long).cpu()
    }
    try:
        agent_logit_trace_topk = int(
            os.environ.get("S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK", "0").strip()
        )
    except ValueError:
        agent_logit_trace_topk = 0
    if agent_logit_trace_topk > 0:
        outputs["voicechat_pad_pair_buffered"] = True
    return outputs


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
    resolve_backend_request_id = getattr(engine, "resolve_backend_request_id", None)
    backend_request_id = (
        str(resolve_backend_request_id(request_id))
        if callable(resolve_backend_request_id)
        else request_id
    )
    if request_state.generation_iterator is None or not request_state.generated_tokens:
        raise RuntimeError("PAD pair requires an initialized continuation request")
    spec = engine.custom_input_specs[0]
    input_dtype = spec.get("dtype") if isinstance(spec, dict) else spec.dtype
    input_dtype = input_dtype or "float32"
    spec_name = spec.get("name") if isinstance(spec, dict) else spec.name
    custom_inputs = {spec_name: input_tensor.to(dtype=getattr(torch, input_dtype)).cpu()}
    await engine.engine.append_request(
        request_id=backend_request_id,
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
            # Diagnostic-only state. These records contain scalars and booleans
            # only, so observing the scheduler never synchronizes CUDA tensors.
            "trace_sequence": 0,
            "trace_events": deque(),
            "trace_events_dropped": 0,
            "trace_record_errors": 0,
            "last_effective_tokens": None,
        },
    )


def _record_pad_pair_decision(
    engine: Any,
    request_id: str,
    state: dict[str, Any],
    **fields: Any,
) -> None:
    """Record diagnostics without ever invalidating a computed model result."""
    if not enabled("VOICECHAT_NANO_PAD_PAIR_TRACE"):
        return
    try:
        _record_pad_pair_decision_unchecked(engine, request_id, state, **fields)
    except Exception:
        # Observability is never allowed to change inference behavior. Keep this
        # fallback defensive as it executes while handling a diagnostic failure.
        try:
            state["trace_record_errors"] = int(state.get("trace_record_errors", 0)) + 1
            logging.exception("Dropped Nano PAD-pair trace event")
        except Exception:
            pass


def _record_pad_pair_decision_unchecked(
    engine: Any,
    request_id: str,
    state: dict[str, Any],
    *,
    decision: str,
    pending_before: bool,
    bypass: bool,
    should_draft: bool,
    generated_before: int,
    generated_after: int,
    accepted: int | None = None,
) -> None:
    """Retain a bounded, scalar-only account of each Nano scheduling decision."""
    context = getattr(engine, "_voicechat_pad_pair_call_context", {})
    has_prepare_context = context.get("request_id") is not None
    context_calls = int(context.get("engine_calls", 0))
    context["engine_calls"] = context_calls + 1
    context_request_id = context.get("request_id")
    disabled_reasons: list[str] = []
    if bool(context.get("bypass", False)):
        disabled_reasons.append("fc_in_progress")
    if bool(state.get("sequential_mode", False)):
        disabled_reasons.append("sequential_mode")
    if context.get("precontrol_reason") is not None:
        disabled_reasons.append(f"precontrol:{context['precontrol_reason']}")
    if pending_before:
        disabled_reasons.append("pending_row")
    if not should_draft:
        disabled_reasons.append("previous_effective_not_pad")

    pair_eligible = not pending_before and not bypass and should_draft
    state["trace_sequence"] += 1
    event = {
        "sequence": int(state["trace_sequence"]),
        "decision": decision,
        "request_id": str(request_id),
        "context_request_id": (
            None if context_request_id is None else str(context_request_id)
        ),
        "request_id_match": (
            None
            if context_request_id is None
            else str(context_request_id) == str(request_id)
        ),
        "context_frame_idx": context.get("frame_idx"),
        "context_engine_call": context_calls,
        # This is deliberately positional, not an identity claim: NVIDIA's
        # asynchronous function loop can interleave after prepare.
        "call_origin": (
            "no_prepare_context"
            if not has_prepare_context
            else "first_call_since_prepare"
            if context_calls == 0
            else "additional_call_since_prepare"
        ),
        "pending_before": pending_before,
        "pending_after": state.get("pending") is not None,
        "needs_correction_after": bool(state.get("needs_correction", False)),
        "sequential_mode": bool(state.get("sequential_mode", False)),
        "previous_effective_pad": state.get("previous_effective_pad"),
        "bypass": bypass,
        "should_draft": should_draft,
        "pair_eligible": pair_eligible,
        "invariant_violation": pair_eligible and decision != "buffered",
        "disabled_reasons": disabled_reasons,
        "generated_before": generated_before,
        "generated_after": generated_after,
        "accepted": accepted,
    }
    events = state["trace_events"]
    if len(events) >= 128:
        events.popleft()
        state["trace_events_dropped"] += 1
    events.append(event)


def consume_pad_pair_trace(wrapper: Any, request_id: str) -> dict[str, Any]:
    """Drain bounded per-call diagnostics for one request into a model-step trace."""
    model = getattr(wrapper, "model_llm_interface", None)
    engine = getattr(model, "engine", None)
    if engine is None:
        return {"enabled": False, "request_id": str(request_id), "events": []}
    states = getattr(engine, "_voicechat_pad_pair_states", {})
    state = states.get(request_id)
    if state is None:
        return {
            "enabled": enabled("VOICECHAT_NANO_PAD_PAIR"),
            "request_id": str(request_id),
            "state_found": False,
            "events": [],
        }
    events = state.get("trace_events")
    drained: list[dict[str, Any]] = []
    if events is not None:
        while events:
            drained.append(events.popleft())
    return {
        "enabled": enabled("VOICECHAT_NANO_PAD_PAIR"),
        "request_id": str(request_id),
        "state_found": True,
        "events": drained,
        "events_dropped": int(state.get("trace_events_dropped", 0)),
        "trace_record_errors": int(state.get("trace_record_errors", 0)),
        "last_effective_tokens": state.get("last_effective_tokens"),
        "state": {
            "pending": state.get("pending") is not None,
            "needs_correction": bool(state.get("needs_correction", False)),
            "assumed": state.get("assumed"),
            "sequential_mode": bool(state.get("sequential_mode", False)),
            "previous_effective_pad": state.get("previous_effective_pad"),
            "buffered": int(state.get("buffered", 0)),
            "accepted": int(state.get("accepted", 0)),
            "rejected": int(state.get("rejected", 0)),
            "sequential_bypass": int(state.get("sequential_bypass", 0)),
            "sequential_pending_drains": int(
                state.get("sequential_pending_drains", 0)
            ),
            "conditional_singles": int(state.get("conditional_singles", 0)),
            "trace_record_errors": int(state.get("trace_record_errors", 0)),
        },
    }


def _install_public_pad_pair_engine(
    streaming_engine_module: Any | None = None,
    nemo_logging: Any | None = None,
) -> None:
    """Install the opt-in two-position continuation policy on Nano only."""
    if streaming_engine_module is None:
        from nemo.collections.speechlm2.inference.vllm import streaming_llm_engine

        streaming_engine_module = streaming_llm_engine
    if nemo_logging is None:
        from nemo.utils import logging as imported_nemo_logging

        nemo_logging = imported_nemo_logging
    GenerationResult = streaming_engine_module.GenerationResult
    LLMStreamingEngine = streaming_engine_module.LLMStreamingEngine

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
            generated_before = len(request_state.generated_tokens)
            result = await original(
                engine,
                input_tensors,
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
            )
            _record_pad_pair_decision(
                engine,
                request_id,
                state,
                decision="initial",
                pending_before=False,
                bypass=False,
                should_draft=False,
                generated_before=generated_before,
                generated_after=len(request_state.generated_tokens),
            )
            return result

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
        generated_before = len(request_state.generated_tokens)

        if pending is None:
            if bypass or not should_draft:
                if bypass:
                    state["sequential_bypass"] += 1
                else:
                    state["conditional_singles"] += 1
                result = await original(
                    engine,
                    [current],
                    prompt_token_ids=prompt_token_ids,
                    request_id=request_id,
                )
                _record_pad_pair_decision(
                    engine,
                    request_id,
                    state,
                    decision="sequential_bypass" if bypass else "sequential_conditional",
                    pending_before=False,
                    bypass=bypass,
                    should_draft=should_draft,
                    generated_before=generated_before,
                    generated_after=len(request_state.generated_tokens),
                )
                return result
            state["pending"] = current
            state["buffered"] += 1
            pad_id = _pad_pair_token_id(engine)
            nemo_logging.info(
                "Nano PAD-pair buffered request=%s buffered=%d",
                request_id,
                state["buffered"],
            )
            result = GenerationResult(
                token_id=pad_id,
                custom_outputs=_pad_pair_buffered_custom_outputs(current, pad_id),
                is_finished=False,
                total_tokens=len(request_state.generated_tokens),
            )
            _record_pad_pair_decision(
                engine,
                request_id,
                state,
                decision="buffered",
                pending_before=False,
                bypass=bypass,
                should_draft=should_draft,
                generated_before=generated_before,
                generated_after=len(request_state.generated_tokens),
            )
            return result

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
            _record_pad_pair_decision(
                engine,
                request_id,
                state,
                decision="sequential_pending_drain",
                pending_before=True,
                bypass=bypass,
                should_draft=should_draft,
                generated_before=generated_before,
                generated_after=len(request_state.generated_tokens),
            )
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
            _record_pad_pair_decision(
                engine,
                request_id,
                state,
                decision="sequential_pending_roll",
                pending_before=True,
                bypass=bypass,
                should_draft=should_draft,
                generated_before=generated_before,
                generated_after=len(request_state.generated_tokens),
            )
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
        _record_pad_pair_decision(
            engine,
            request_id,
            state,
            decision="pair_accepted" if accepted == 2 else "pair_rejected",
            pending_before=True,
            bypass=bypass,
            should_draft=should_draft,
            generated_before=generated_before,
            generated_after=len(request_state.generated_tokens),
            accepted=accepted,
        )
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
    precontrol_reason = None
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
        "fc_in_progress": fc_in_progress,
        "precontrol_reason": precontrol_reason,
        "engine_calls": 0,
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
    if enabled("VOICECHAT_NANO_PAD_PAIR_TRACE"):
        state["last_effective_tokens"] = {
            "frame_idx": frame_idx,
            "request_id": None if request_id is None else str(request_id),
            "text": effective_text,
            "function": effective_function,
            "pad": pad_id,
            "text_is_pad": effective_text == pad_id,
            "function_is_pad": effective_function == pad_id,
            "both_pad": state["previous_effective_pad"],
            "fc_in_progress": pad_pair_fc_in_progress(arguments.get("fc_state")),
            "precontrol_reason": context.get("precontrol_reason"),
        }
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
        "full_graph": enabled("VOICECHAT_NANO_PAIR_FULL_GRAPH"),
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
    resolve_backend_request_id = getattr(
        stream_engine, "resolve_backend_request_id", None
    )
    backend_request_id = (
        str(resolve_backend_request_id(request_id))
        if callable(resolve_backend_request_id)
        else request_id
    )
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
                request_id=backend_request_id,
            )
        elif request_state.generated_tokens:
            try:
                await stream_engine.engine.append_request(
                    request_id=backend_request_id,
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


def install_eartts_idle_pad_bypass(wrapper: Any) -> bool:
    """Install the narrow ordinary-PCM idle-PAD EarTTS bypass contract."""
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    if getattr(wrapper, "_eartts_idle_pad_bypass_installed", False):
        return False
    if not getattr(wrapper, "use_vllm_eartts", False):
        raise RuntimeError("EarTTS idle-PAD bypass requires the vLLM EarTTS backend")
    if not getattr(tts_model, "_decode_pad_silence_installed", False):
        raise RuntimeError(
            "EarTTS idle-PAD bypass requires the qualified PAD decoder-silence policy"
        )
    silence = getattr(tts_model, "codec_silence_tokens", None)
    if silence is None:
        raise RuntimeError("EarTTS idle-PAD bypass requires codec silence tokens")
    initial_code = getattr(wrapper, "first_tts_code_input", None)
    if initial_code is None:
        raise RuntimeError("EarTTS idle-PAD bypass requires the initial recurrent code")
    # Validate the fixed acoustic dimensions during startup, before serving can
    # select the bypass. Runtime positions use this same recurrent-code shape.
    eartts_idle_pad_bypass_transition(
        recurrent_code=initial_code,
        past_key_values=None,
        codec_silence_tokens=silence,
    )
    wrapper._eartts_idle_pad_bypass_eligible = eartts_idle_pad_bypass_eligible
    wrapper._eartts_idle_pad_bypass_transition = eartts_idle_pad_bypass_transition
    wrapper._eartts_idle_pad_bypass_installed = True
    return True


def install_eartts_reset_on_bos(wrapper: Any, *, prepared_epoch: bool = False) -> bool:
    """Reset only the EarTTS vLLM request immediately before each spoken turn."""
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    backend = getattr(tts_model, "tts_model", None)
    original = getattr(tts_model, "infer_codes_one_step", None)
    if original is None or backend is None or getattr(tts_model, "_reset_on_bos_installed", False):
        return False

    controller = (
        EarTTSPreparedEpochController(wrapper, tts_model, backend)
        if prepared_epoch
        else None
    )

    def call_infer_codes_one_step(*args, **kwargs):
        current = kwargs.get("current_subword_id")
        request_id = kwargs.get("request_id")
        # This diagnostic always describes the current call. A non-BOS call or
        # a failing BOS call must not expose the preceding reset's timing.
        tts_model._reset_on_bos_last_timing_ms = None
        completed_reset_timing = None
        is_bos = bool(
            current is not None
            and current.numel() == 1
            and int(current.item()) == int(tts_model.text_bos_id)
            and request_id is not None
        )
        prepared_reuse = False
        prepared_reason = "prepared_epoch_disabled"
        armed_client_turn_id = None
        prepared_session_epoch = None
        prepared_prepare_epoch = None
        if controller is not None and is_bos:
            prepared_reuse, prepared_reason, armed_client_turn_id = (
                controller.consume_prepared_bos(str(request_id))
            )
            if prepared_reuse and controller.prepared is not None:
                prepared_session_epoch = int(controller.prepared["session_epoch"])
                prepared_prepare_epoch = int(controller.prepared["prepare_epoch"])
        elif controller is not None:
            controller.mark_real_call("real_non_bos_call")

        if is_bos and not prepared_reuse:
            reset_started = time.perf_counter()
            requests = getattr(getattr(backend, "engine", None), "requests", None)
            existing_request = (
                requests.get(request_id) if hasattr(requests, "get") else None
            )
            if existing_request is not None and not bool(
                backend.abort_request(request_id)
            ):
                raise RuntimeError("EarTTS BOS reset could not abort the active request")
            abort_finished = time.perf_counter()
            backend(
                wrapper.tts_init_inputs,
                request_id=request_id,
                prompt_token_ids=wrapper.tts_prompt_token_ids,
            )
            prefill_finished = time.perf_counter()
            tts_model._reset_on_bos_count += 1
            tts_model._reset_on_bos_last_request_id = str(request_id)
            completed_reset_timing = {
                "schema": 1,
                "reset_count": int(tts_model._reset_on_bos_count),
                "request_id": str(request_id),
                "abort_call_ms": (abort_finished - reset_started) * 1000.0,
                "prefill_call_ms": (prefill_finished - abort_finished) * 1000.0,
                "reset_total_ms": (prefill_finished - reset_started) * 1000.0,
            }
            initial_code = wrapper.first_tts_code_input
            if initial_code is None:
                raise RuntimeError("EarTTS BOS reset requires first_tts_code_input")
            kwargs["prev_audio_tokens"] = initial_code.detach().clone()
            logging.info("Reset and speaker-prefilled EarTTS request %s at BOS", request_id)
        elif is_bos:
            initial_code = wrapper.first_tts_code_input
            if initial_code is None:
                if controller is not None:
                    controller.invalidate("initial_code_missing_at_prepared_bos")
                raise RuntimeError("EarTTS prepared BOS requires first_tts_code_input")
            kwargs["prev_audio_tokens"] = initial_code.detach().clone()

        try:
            result = original(*args, **kwargs)
        except Exception:
            if controller is not None:
                controller.invalidate(
                    "prepared_bos_inference_failed"
                    if prepared_reuse
                    else "real_eartts_call_failed"
                )
            raise
        if completed_reset_timing is not None:
            tts_model._reset_on_bos_last_timing_ms = completed_reset_timing
        if controller is not None and is_bos:
            current_request = controller._request_snapshot(str(request_id))
            controller.bos_transition_count += 1
            if prepared_reuse:
                controller.prepared_reuse_count += 1
            controller.state = "dirty"
            controller.reason = (
                "prepared_bos_completed" if prepared_reuse else "legacy_bos_completed"
            )
            controller.prepared = None
            controller.armed_client_turn_id = None
            controller.last_bos_event = {
                "schema": 2,
                "mode": "prepared_reuse" if prepared_reuse else "legacy_fallback",
                "request_id": str(request_id),
                "backend_request_id": current_request["backend_request_id"],
                "backend_generation": current_request["backend_generation"],
                "armed_client_turn_id": (
                    armed_client_turn_id if prepared_reuse else None
                ),
                "session_epoch": (
                    prepared_session_epoch
                    if prepared_reuse
                    else controller.session_epoch
                ),
                "prepare_epoch": (
                    prepared_prepare_epoch
                    if prepared_reuse
                    else controller.prepare_epoch
                ),
                "reset_count": int(tts_model._reset_on_bos_count),
                "transition_count": controller.bos_transition_count,
                "reuse_count": controller.prepared_reuse_count,
                "fallback_reason": None if prepared_reuse else prepared_reason,
            }
        return result

    def reset_on_bos_infer_codes_one_step(*args, **kwargs):
        if controller is None:
            return call_infer_codes_one_step(*args, **kwargs)
        with controller.lock:
            return call_infer_codes_one_step(*args, **kwargs)

    tts_model.infer_codes_one_step = reset_on_bos_infer_codes_one_step
    tts_model._reset_on_bos_count = 0
    tts_model._reset_on_bos_last_request_id = None
    tts_model._reset_on_bos_last_timing_ms = None
    tts_model._reset_on_bos_installed = True
    if controller is not None:
        wrapper._eartts_prepared_epoch_controller = controller
        wrapper._attest_eartts_prepared_epoch = controller.attest_existing_prefill
        wrapper._prepare_eartts_prepared_epoch = controller.prepare
        wrapper._arm_eartts_prepared_epoch = controller.arm
        wrapper._reset_eartts_prepared_epoch = controller.reset_session
        wrapper._eartts_prepared_epoch_status = controller.status
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
        "eartts_idle_pad_bypass": False,
        "eartts_prepared_epoch": False,
        "eartts_unique_backend_request_ids": False,
        "eartts_reset_on_bos": False,
        "nano_pad_pair_control_barrier": enabled("VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"),
    }
    if enabled("VOICECHAT_EARTTS_DECODE_PAD_SILENCE"):
        details["eartts_decode_pad_silence"] = install_eartts_decode_pad_silence(pipeline.s2s_model)
    if enabled("VOICECHAT_EARTTS_IDLE_PAD_BYPASS"):
        details["eartts_idle_pad_bypass"] = install_eartts_idle_pad_bypass(
            pipeline.s2s_model
        )
    if enabled("VOICECHAT_EARTTS_RESET_ON_BOS"):
        prepared_epoch = enabled("VOICECHAT_EARTTS_PREPARED_EPOCH")
        if prepared_epoch and not details["eartts_idle_pad_bypass"]:
            raise RuntimeError("EarTTS prepared epoch requires idle-PAD bypass")
        details["eartts_reset_on_bos"] = install_eartts_reset_on_bos(
            pipeline.s2s_model,
            prepared_epoch=prepared_epoch,
        )
        details["eartts_prepared_epoch"] = prepared_epoch
        if prepared_epoch:
            stream_engine = getattr(
                getattr(
                    getattr(pipeline.s2s_model.model, "tts_model", None),
                    "tts_model",
                    None,
                ),
                "engine",
                None,
            )
            details["eartts_unique_backend_request_ids"] = bool(
                getattr(stream_engine, "_unique_backend_request_ids", False)
            )
            if not details["eartts_unique_backend_request_ids"]:
                raise RuntimeError(
                    "EarTTS prepared epoch requires unique backend request IDs"
                )
    elif enabled("VOICECHAT_EARTTS_PREPARED_EPOCH"):
        raise RuntimeError("EarTTS prepared epoch requires reset-on-BOS")
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
