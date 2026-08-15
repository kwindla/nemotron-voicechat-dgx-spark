#!/usr/bin/env python3
"""Minimal persistent WebSocket server for NVIDIA's VoiceChat source pipeline.

This implements the strict version-3 protocol used by the project-owned Pipecat
service. One active stream is supported because the qualified DGX Spark
configuration has batch size one and one set of recurrent model state. Only the
public combined Hugging Face checkpoint is supported.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import faulthandler
import functools
import gc
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .pocket_controller import (
    PocketSynthesisCancelled,
    PocketWorkerError,
    PocketWorkerManager,
)
from .protocol import (
    FUNCTION_OUTPUT_MAX_BYTES,
    FUNCTION_OUTPUT_MAX_TOKENS,
    FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
    FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES,
    MAX_FUNCTION_CALLS_PER_RESPONSE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RealtimeProtocolSession,
    protocol_capabilities,
)
from .provenance import (
    NANO_PAD_PAIR_PRODUCTION_POLICY,
    PRODUCTION_ENVIRONMENT,
    QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV,
    QUALIFICATION_MODE_ENV,
    RUNTIME_SOURCE_PATHS,
    ValidatedQualificationMode,
    is_validated_qualification_mode,
    validate_nano_pad_pair_production_policy,
    validate_qualification_watchdog_production_policy,
    validated_qualification_mode_fields,
)

PINNED_COMMIT = "911ec674ab40f04302ef33672be4179f45a7310f"
PRODUCTION_ENGINE_TYPE = "vllm_llm_vllm_eartts"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
FRAME_SECONDS = 0.08
FRAME_SAMPLES = int(INPUT_SAMPLE_RATE * FRAME_SECONDS)
MAX_INPUT_MESSAGE_BYTES = FRAME_SAMPLES * 2 * 16
MAX_TYPED_INPUT_CHARS = 1_000
FUNCTION_CALL_TIMEOUT_SECONDS = 30.0
FUNCTION_OUTPUT_APPLY_MAX_FRAMES = FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
TOOL_GROUNDING_INSTRUCTIONS = (
    "Call a tool only when the user's request needs a capability that an advertised "
    "tool provides. If no advertised tool matches the request, answer directly in "
    "words and call no tool. "
    "Never invent or imply a tool result without calling the tool. If you say you "
    "will use a tool, emit that tool call before ending the turn. After a tool "
    "response arrives, answer the user and explicitly include the exact returned "
    "result."
)
PUBLIC_VLLM_MANIFEST_KINDS = frozenset(
    {
        "exact_public_vllm_quantized_nano_eartts",
        "nemotron_voicechat_dgx_spark_hf_release",
    }
)
DIAGNOSTIC_FP32_VLLM_MANIFEST_KIND = "exact_public_vllm_extraction"
LOGGER = logging.getLogger(__name__)
STARTUP_READY_SENTINEL = Path("/tmp/voicechat-step9-ready")
STEP9_FALLBACK_COUNTER = Path("/tmp/voicechat-step9-fallbacks.jsonl")


def startup_event(event: str, **details: Any) -> None:
    """Emit a machine-readable startup boundary without changing execution."""

    payload = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_ns": time.time_ns(),
        **details,
    }
    print(f"VOICECHAT_STARTUP {json.dumps(payload, sort_keys=True)}", flush=True)


startup_event("server_module_imported")


def step9_capture_fallback_status(path: Path = STEP9_FALLBACK_COUNTER) -> dict[str, Any]:
    """Return the container-lifetime graphless-fallback count and last shape."""

    try:
        records = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except FileNotFoundError:
        records = []
    last_descriptor = None
    if records:
        try:
            last_descriptor = json.loads(records[-1])
        except json.JSONDecodeError:
            last_descriptor = {"malformed_record": records[-1]}
    return {"count": len(records), "last_descriptor": last_descriptor}


PRODUCTION_NANO_CUSTOM_OUTPUTS = ["function_tokens", "function_logits"]
DIAGNOSTIC_NANO_CUSTOM_OUTPUTS = ["text_logits", *PRODUCTION_NANO_CUSTOM_OUTPUTS]
SKIP_TEXT_LOGITS_CONTRACT = "temperature=0, top_p=1, repetition_penalty=1"
FC_ASYNC_HEARTBEAT_ENV = "S2S_FC_ASYNC_HEARTBEAT"
FC_ASYNC_BENCHMARK_ENV = "S2S_FC_ASYNC_BENCHMARK"
FC_FAST_TOOL_GRACE_MS_ENV = "VOICECHAT_FC_FAST_TOOL_GRACE_MS"
FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV = "VOICECHAT_FC_ALWAYS_ACKNOWLEDGE_TOOLS"
FC_ASYNC_HEARTBEAT_KEYS = frozenset(
    {
        "schema",
        "cycle_sotc_frame",
        "phase",
        "invocation",
        "async_step",
        "model_frame",
        "stage",
        "state",
        "stage_started_monotonic",
        "last_completed_stage",
        "last_completed_monotonic",
        "last_failed_stage",
        "thread_ident",
        "live_audio_queue_depth",
        "perception_queue_depth",
        "tts_audio_queue_depth",
        "rnnt_text_queue_depth",
    }
)
FC_ASYNC_HEARTBEAT_STAGES = frozenset(
    {
        "entry",
        "audio_wait",
        "perception_encode",
        "perception_sync",
        "embedding_publish",
        "prepare",
        "perception_poll",
        "nano",
        "eartts",
        "codec_decode",
        "codec_cpu_copy",
        "rnnt",
        "between_stages",
        "stage_failed",
        "complete",
        "failed",
        "aborted",
    }
)
FC_ASYNC_HEARTBEAT_STATES = frozenset({"started", "idle", "complete", "failed", "aborted"})


def accepted_vllm_manifest_kinds() -> frozenset[str]:
    diagnostic_fp32 = bool(
        os.environ.get("VOICECHAT_DIRECT_SEMANTIC_PROBE_DIR", "").strip()
        and os.environ.get("VOICECHAT_DIRECT_SEMANTIC_PRECISION", "").casefold() == "fp32"
    )
    return PUBLIC_VLLM_MANIFEST_KINDS | (
        {DIAGNOSTIC_FP32_VLLM_MANIFEST_KIND} if diagnostic_fp32 else set()
    )


def typed_input_trailing_silence_seconds(
    *, rnnt_eou_frames: int, margin: float, injection_frame_ms: int = 20
) -> float:
    """Return EOU-derived silence rounded to an 80 ms model-frame boundary."""

    raw = rnnt_eou_frames * FRAME_SECONDS * margin + injection_frame_ms / 1_000
    return math.ceil(raw / FRAME_SECONDS) * FRAME_SECONDS


class RealtimeProtocolError(Exception):
    """A client-visible protocol error with explicit connection semantics."""

    def __init__(self, code: str, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.code = code
        self.fatal = fatal


class SessionTerminated(Exception):
    """Internal control flow after a terminal session event was emitted."""


class VoiceChatEngineStartFailure(RuntimeError):
    """A transactionally cleaned failure while preparing one model session."""

    def __init__(self, phase: str, timings_ms: dict[str, float]):
        super().__init__(f"VoiceChat engine start failed during {phase}")
        self.phase = phase
        self.timings_ms = timings_ms


@dataclass
class FunctionCallBudget:
    """Bound model-selected tool calls to one user-turn response scope."""

    limit: int = MAX_FUNCTION_CALLS_PER_RESPONSE
    turn_id: str | None = None
    count: int = 0
    initialized: bool = False

    def consume(self, turn_id: str | None) -> tuple[bool, int]:
        if not self.initialized or turn_id != self.turn_id:
            self.turn_id = turn_id
            self.count = 0
            self.initialized = True
        self.count += 1
        return self.count <= self.limit, self.count


def function_call_loop_terminal_events(
    protocol: RealtimeProtocolSession,
    *,
    call_id: str,
    attempt: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Fail the response and session without fabricating assistant output."""

    reason = "function_call_loop_limit"
    events = protocol.fail_active_response(reason)
    events.append(
        protocol.error(
            reason,
            f"Model exceeded {limit} function calls in one response",
            fatal=True,
            call_id=call_id,
            attempt=attempt,
            limit=limit,
        )
    )
    events.append(protocol.session_closed(reason, status="failed"))
    return events


def function_cycle_active(result: Any) -> bool:
    """Return whether a model step still reports an open function cycle."""

    fc = (result.turn_state.get("function_calling") if result else None) or {}
    return any(
        (
            fc.get("active"),
            fc.get("awaiting_response"),
            fc.get("injecting_response"),
            fc.get("awaiting_eotr"),
            fc.get("forced_tokens"),
            fc.get("background_active"),
        )
    )


class UserEouSettlementFailure(RuntimeError):
    """A fail-closed violation of the explicit-EOU settlement contract."""

    def __init__(self, code: str, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.code = code
        self.evidence = evidence


def _eartts_reset_count(engine: Any) -> int:
    pipeline = getattr(engine, "pipeline", None)
    wrapper = getattr(pipeline, "s2s_model", None)
    model = getattr(wrapper, "model", None)
    tts_model = getattr(model, "tts_model", None)
    return int(getattr(tts_model, "_reset_on_bos_count", 0))


def _eartts_epoch_status(engine: Any) -> dict[str, Any] | None:
    pipeline = getattr(engine, "pipeline", None)
    wrapper = getattr(pipeline, "s2s_model", None)
    status_fn = getattr(wrapper, "_eartts_prepared_epoch_status", None)
    if not callable(status_fn):
        return None
    status = status_fn()
    return dict(status) if isinstance(status, dict) else None


def _eartts_request_id(engine: Any) -> str | None:
    request_id_fn = getattr(engine, "_stream_request_id", None)
    if callable(request_id_fn):
        return request_id_fn()
    pipeline = getattr(engine, "pipeline", None)
    mapping = getattr(pipeline, "_request_id_for_stream", None)
    stream_id = getattr(engine, "stream_id", None)
    if callable(mapping) and isinstance(stream_id, int):
        request_id = mapping(stream_id)
        return request_id if isinstance(request_id, str) and request_id else None
    return None


def _eartts_backend_request_identity(engine: Any) -> dict[str, Any] | None:
    """Return the active logical/physical EarTTS request identity."""
    logical_request_id = _eartts_request_id(engine)
    pipeline = getattr(engine, "pipeline", None)
    wrapper = getattr(pipeline, "s2s_model", None)
    tts_model = getattr(getattr(wrapper, "model", None), "tts_model", None)
    interface = getattr(tts_model, "tts_model", None)
    stream_engine = getattr(interface, "engine", None)
    requests = getattr(stream_engine, "requests", None)
    state = (
        requests.get(logical_request_id)
        if logical_request_id is not None and hasattr(requests, "get")
        else None
    )
    backend_request_id = getattr(state, "backend_request_id", None)
    backend_generation = getattr(state, "backend_generation", None)
    if (
        state is None
        or logical_request_id is None
        or not isinstance(backend_request_id, str)
        or not backend_request_id
        or isinstance(backend_generation, bool)
        or not isinstance(backend_generation, int)
        or backend_generation < 1
    ):
        return None
    return {
        "request_id": logical_request_id,
        "backend_request_id": backend_request_id,
        "backend_generation": backend_generation,
    }


def _eartts_epoch_step_snapshot(engine: Any) -> dict[str, Any] | None:
    """Capture synchronization-free EarTTS epoch state around one model step."""

    pipeline = getattr(engine, "pipeline", None)
    wrapper = getattr(pipeline, "s2s_model", None)
    controller = getattr(wrapper, "_eartts_prepared_epoch_controller", None)
    lock = getattr(controller, "lock", None)
    if controller is None or lock is None:
        return None
    with lock:
        status = _eartts_epoch_status(engine)
        if status is None:
            return None
        identity = _eartts_backend_request_identity(engine)
        if identity is None:
            raise RuntimeError("EarTTS prepared epoch has no active backend identity")
        values: dict[str, int] = {}
        for key in (
            "session_epoch",
            "prepare_epoch",
            "bos_transition_count",
            "prepared_reuse_count",
        ):
            value = status.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"EarTTS prepared epoch {key} is invalid")
            values[key] = value
        return {
            "reset_count": _eartts_reset_count(engine),
            "transition_count": values["bos_transition_count"],
            "reuse_count": values["prepared_reuse_count"],
            "session_epoch": values["session_epoch"],
            "prepare_epoch": values["prepare_epoch"],
            **identity,
        }


@dataclass
class UserEouSettlement:
    """Transport-independent controller for bounded pre-EOU model settlement.

    Callers own the transport-specific model step. This controller owns the
    blank-fence bound, per-step response/function guards, and evidence shape so
    WebSocket commits and diagnostic probes cannot silently diverge.
    """

    target_blank_frames: int
    starting_blank_frames: int
    ending_blank_frames: int
    assistant_text_baseline: str
    function_text_baseline: str
    eartts_reset_count_starting: int
    expected_client_turn_id: int | None = None
    eartts_transition_count_starting: int | None = None
    eartts_reuse_count_starting: int | None = None
    model_steps: int = 0
    steps: list[dict[str, Any]] | None = None
    fused_terminal_bos: bool = False
    separate_terminal_bos: bool = False
    deferred_terminal_bos: bool = False
    terminal_bos_epoch: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.target_blank_frames < 1:
            raise ValueError("user EOU settlement target must be positive")
        if self.steps is None:
            self.steps = []

    @classmethod
    def begin(
        cls,
        engine: Any,
        result: Any,
        *,
        expected_client_turn_id: int | None = None,
    ) -> UserEouSettlement:
        target = int(engine.user_eou_settlement_blank_frames())
        starting = int(engine.rnnt_blank_count(result))
        epoch = _eartts_epoch_status(engine)
        settlement = cls(
            target_blank_frames=target,
            starting_blank_frames=starting,
            ending_blank_frames=starting,
            assistant_text_baseline=str(getattr(result, "assistant_text", "") or ""),
            function_text_baseline=str(getattr(result, "function_text", "") or ""),
            eartts_reset_count_starting=_eartts_reset_count(engine),
            expected_client_turn_id=expected_client_turn_id,
            eartts_transition_count_starting=(
                None if epoch is None else int(epoch["bos_transition_count"])
            ),
            eartts_reuse_count_starting=(
                None if epoch is None else int(epoch["prepared_reuse_count"])
            ),
        )
        if result is not None:
            settlement._reject_initial_activity(result)
        return settlement

    @property
    def max_model_steps(self) -> int:
        return self.target_blank_frames * 2

    def needs_step(self, engine: Any, result: Any) -> bool:
        if self.fused_terminal_bos:
            return False
        self.ending_blank_frames = int(engine.rnnt_blank_count(result))
        if self.ending_blank_frames >= self.target_blank_frames:
            return False
        if self.model_steps >= self.max_model_steps:
            raise UserEouSettlementFailure(
                "input_turn_settle_timeout",
                "User EOU settlement did not reach the RNNT blank fence",
                self.evidence(),
            )
        return True

    def observe(self, engine: Any, result: Any) -> None:
        """Record and validate one model step before the caller publishes it."""

        self.model_steps += 1
        self.ending_blank_frames = int(engine.rnnt_blank_count(result))
        turn_state = result.turn_state or {}
        boundary = turn_state.get("response_boundary") or {}
        function_state = turn_state.get("function_calling") or {}
        deliverable = response_audio_is_deliverable(turn_state)
        violations: list[str] = []
        if boundary.get("event") is not None or boundary.get("phase") not in {None, "idle"}:
            violations.append("response_boundary_open")
        if turn_state.get("agent_control") != "pad":
            violations.append("agent_output_effective")
        if deliverable:
            violations.append("response_audio_deliverable")
        if (
            bool(getattr(result, "assistant_delta", ""))
            or str(getattr(result, "assistant_text", "") or "") != self.assistant_text_baseline
        ):
            violations.append("assistant_text_mutated")
        if (
            bool(getattr(result, "function_delta", ""))
            or str(getattr(result, "function_text", "") or "") != self.function_text_baseline
        ):
            violations.append("function_text_mutated")
        effective_function = function_cycle_active(result)
        if effective_function:
            violations.append("function_cycle_effective")

        perception_frame = None
        perception_frame_fn = getattr(engine, "perception_frame_index", None)
        if callable(perception_frame_fn):
            perception_frame = int(perception_frame_fn())
        step = {
            "model_step": self.model_steps,
            "frame_index": int(getattr(result, "frame_index", -1)),
            "audio_frame_index_after": int(getattr(engine, "audio_frame_index", -1)),
            "perception_frame_idx_after": perception_frame,
            "eartts_reset_count_after": _eartts_reset_count(engine),
            "rnnt_blank_frames": self.ending_blank_frames,
            "agent_control": turn_state.get("agent_control"),
            "response_boundary": {
                "event": boundary.get("event"),
                "phase": boundary.get("phase"),
            },
            "response_audio_deliverable": deliverable,
            "assistant_delta": str(getattr(result, "assistant_delta", "") or ""),
            "assistant_text_unchanged": str(getattr(result, "assistant_text", "") or "")
            == self.assistant_text_baseline,
            "function_delta": str(getattr(result, "function_delta", "") or ""),
            "function_text_unchanged": str(getattr(result, "function_text", "") or "")
            == self.function_text_baseline,
            "function_cycle_effective": effective_function,
            "function_calling": {
                key: function_state.get(key)
                for key in (
                    "active",
                    "awaiting_response",
                    "injecting_response",
                    "awaiting_eotr",
                    "forced_tokens",
                    "background_active",
                    "pre_eou_suppressed_tokens",
                    "pre_eou_last_raw_token_id",
                    "pre_eou_suppressed_frame",
                    "client_eou_sotc_committed_count",
                    "client_eou_sotc_committed_frame",
                    "post_fc_client_bos_pending",
                    "post_fc_client_bos_requested_frame",
                    "post_fc_client_bos_forced_count",
                    "post_fc_client_bos_forced_frame",
                )
            },
            "vllm_request_positions": turn_state.get("vllm_request_positions"),
            "violations": violations,
        }
        assert self.steps is not None
        self.steps.append(step)
        if violations:
            code = (
                "pre_eou_function_activity" if effective_function else "pre_eou_response_activity"
            )
            raise UserEouSettlementFailure(
                code,
                "Model activity became effective before the explicit user EOU",
                self.evidence(),
            )

    def observe_fused_terminal(self, engine: Any, result: Any) -> None:
        """Validate the one no-tools position that proves the fence and opens BOS."""

        self.model_steps += 1
        turn_state = result.turn_state or {}
        marker = turn_state.get("client_eou_blank_fence") or {}
        boundary = turn_state.get("response_boundary") or {}
        function_state = turn_state.get("function_calling") or {}
        effective_function = function_cycle_active(result)
        marker_blank = marker.get("blank_count_at_check")
        marker_target = marker.get("target_blank_frames")
        marker_frame = marker.get("frame")
        control_ids = turn_state.get("control_ids") or {}
        violations: list[str] = []
        if marker_blank != self.target_blank_frames or marker_target != self.target_blank_frames:
            violations.append("blank_fence_not_proven")
        if marker_frame != int(getattr(result, "frame_index", -1)):
            violations.append("blank_fence_frame_mismatch")
        if marker.get("chosen_token_id") != control_ids.get("agent_bos"):
            violations.append("blank_fence_token_mismatch")
        if turn_state.get("agent_control") != "agent_bos":
            violations.append("terminal_bos_missing")
        if boundary.get("event") != "start" or boundary.get("phase") != "responding":
            violations.append("response_boundary_not_started")
        if not response_audio_is_deliverable(turn_state):
            violations.append("terminal_audio_not_deliverable")
        if float((getattr(result, "output_audio", None) or {}).get("rms_dbfs", 0.0)) > -90.0:
            violations.append("terminal_audio_not_silent")
        if (
            bool(getattr(result, "assistant_delta", ""))
            or str(getattr(result, "assistant_text", "") or "") != self.assistant_text_baseline
        ):
            violations.append("assistant_text_mutated")
        if (
            bool(getattr(result, "function_delta", ""))
            or str(getattr(result, "function_text", "") or "") != self.function_text_baseline
        ):
            violations.append("function_text_mutated")
        if effective_function:
            violations.append("function_cycle_effective")
        if function_state and not function_state.get("effective_token_is_pad", False):
            violations.append("function_token_effective")
        violations.extend(self._validate_terminal_bos_epoch(engine, result))

        self.ending_blank_frames = int(marker_blank or 0)
        perception_frame = None
        perception_frame_fn = getattr(engine, "perception_frame_index", None)
        if callable(perception_frame_fn):
            perception_frame = int(perception_frame_fn())
        step = {
            "model_step": self.model_steps,
            "frame_index": int(getattr(result, "frame_index", -1)),
            "audio_frame_index_after": int(getattr(engine, "audio_frame_index", -1)),
            "perception_frame_idx_after": perception_frame,
            "eartts_reset_count_after": _eartts_reset_count(engine),
            "rnnt_blank_frames": self.ending_blank_frames,
            "agent_control": turn_state.get("agent_control"),
            "response_boundary": {
                "event": boundary.get("event"),
                "phase": boundary.get("phase"),
            },
            "response_audio_deliverable": response_audio_is_deliverable(turn_state),
            "assistant_delta": str(getattr(result, "assistant_delta", "") or ""),
            "assistant_text_unchanged": str(getattr(result, "assistant_text", "") or "")
            == self.assistant_text_baseline,
            "function_delta": str(getattr(result, "function_delta", "") or ""),
            "function_text_unchanged": str(getattr(result, "function_text", "") or "")
            == self.function_text_baseline,
            "function_cycle_effective": effective_function,
            "function_calling": {
                key: function_state.get(key)
                for key in (
                    "active",
                    "awaiting_response",
                    "injecting_response",
                    "awaiting_eotr",
                    "forced_tokens",
                    "background_active",
                    "pre_eou_suppressed_tokens",
                    "pre_eou_last_raw_token_id",
                    "pre_eou_suppressed_frame",
                    "client_eou_sotc_committed_count",
                    "client_eou_sotc_committed_frame",
                    "post_fc_client_bos_pending",
                    "post_fc_client_bos_requested_frame",
                    "post_fc_client_bos_forced_count",
                    "post_fc_client_bos_forced_frame",
                )
            },
            "vllm_request_positions": turn_state.get("vllm_request_positions"),
            "terminal_fused_bos": True,
            "client_eou_blank_fence": dict(marker),
            "violations": violations,
        }
        assert self.steps is not None
        self.steps.append(step)
        if violations:
            raise UserEouSettlementFailure(
                "fused_eou_contract_failure",
                "Fused blank-fence/BOS position violated its terminal contract",
                self.evidence(),
            )
        self.fused_terminal_bos = True

    def _validate_terminal_bos_epoch(self, engine: Any, result: Any) -> list[str]:
        """Validate one exclusive legacy-reset or prepared-reuse BOS transition."""

        status = _eartts_epoch_status(engine)
        reset_after = _eartts_reset_count(engine)
        if status is None:
            if reset_after != self.eartts_reset_count_starting + 1:
                return ["eartts_bos_reset_count"]
            return []
        if (
            self.eartts_transition_count_starting is None
            or self.eartts_reuse_count_starting is None
        ):
            return ["eartts_bos_epoch_start_missing"]
        transition_after = int(status.get("bos_transition_count", -1))
        reuse_after = int(status.get("prepared_reuse_count", -1))
        reset_start = self.eartts_reset_count_starting
        transition_start = self.eartts_transition_count_starting
        reuse_start = self.eartts_reuse_count_starting
        terminal_step_start = None
        step = (getattr(result, "turn_state", None) or {}).get("eartts_bos_epoch_step")
        violations: list[str] = []
        if self.deferred_terminal_bos:
            required_step = {
                "schema",
                "reset_count_before",
                "reset_count_after",
                "transition_count_before",
                "transition_count_after",
                "reuse_count_before",
                "reuse_count_after",
                "session_epoch_before",
                "session_epoch_after",
                "prepare_epoch_before",
                "prepare_epoch_after",
                "request_id_before",
                "request_id_after",
                "backend_request_id_before",
                "backend_request_id_after",
                "backend_generation_before",
                "backend_generation_after",
            }
            if not isinstance(step, dict) or set(step) != required_step or step.get("schema") != 1:
                violations.append("eartts_bos_step_evidence_missing")
            else:
                integer_keys = required_step - {
                    "schema",
                    "request_id_before",
                    "request_id_after",
                    "backend_request_id_before",
                    "backend_request_id_after",
                }
                if any(
                    isinstance(step.get(key), bool)
                    or not isinstance(step.get(key), int)
                    or int(step[key]) < 0
                    for key in integer_keys
                ):
                    violations.append("eartts_bos_step_evidence_invalid")
                elif any(
                    not isinstance(step.get(key), str) or not step[key]
                    for key in (
                        "request_id_before",
                        "request_id_after",
                        "backend_request_id_before",
                        "backend_request_id_after",
                    )
                ):
                    violations.append("eartts_bos_step_evidence_invalid")
                else:
                    terminal_step_start = dict(step)
                    reset_start = int(step["reset_count_before"])
                    transition_start = int(step["transition_count_before"])
                    reuse_start = int(step["reuse_count_before"])
                    if (
                        int(step["reset_count_after"]) != reset_after
                        or int(step["transition_count_after"]) != transition_after
                        or int(step["reuse_count_after"]) != reuse_after
                    ):
                        violations.append("eartts_bos_step_after_stale")
                    if (
                        step["session_epoch_before"] != step["session_epoch_after"]
                        or step["session_epoch_after"] != status.get("session_epoch")
                        or step["prepare_epoch_before"] != step["prepare_epoch_after"]
                        or step["prepare_epoch_after"] != status.get("prepare_epoch")
                    ):
                        violations.append("eartts_bos_step_epoch_mismatch")
                    if (
                        step["backend_generation_before"] < 1
                        or step["backend_generation_after"] < 1
                        or step["backend_request_id_before"] == step["request_id_before"]
                        or step["backend_request_id_after"] == step["request_id_after"]
                    ):
                        violations.append("eartts_bos_step_physical_identity_invalid")
        reset_delta = reset_after - reset_start
        transition_delta = transition_after - transition_start
        reuse_delta = reuse_after - reuse_start
        event = status.get("last_bos_event")
        expected_request_id = _eartts_request_id(engine)
        expected_backend_identity = _eartts_backend_request_identity(engine)
        if transition_delta != 1:
            violations.append("eartts_bos_transition_count")
        if transition_delta != reset_delta + reuse_delta:
            violations.append("eartts_bos_transition_accounting")
        if (reset_delta, reuse_delta) not in {(1, 0), (0, 1)}:
            violations.append("eartts_bos_transition_mode")
        if self.deferred_terminal_bos and (reset_delta, reuse_delta) != (1, 0):
            violations.append("eartts_deferred_bos_not_legacy")
        whole_reset_delta = reset_after - self.eartts_reset_count_starting
        whole_transition_delta = transition_after - self.eartts_transition_count_starting
        whole_reuse_delta = reuse_after - self.eartts_reuse_count_starting
        if (
            min(whole_reset_delta, whole_transition_delta, whole_reuse_delta) < 0
            or whole_transition_delta != whole_reset_delta + whole_reuse_delta
        ):
            violations.append("eartts_bos_whole_turn_accounting")
        if terminal_step_start is not None:
            preterminal_reset_delta = (
                terminal_step_start["reset_count_before"] - self.eartts_reset_count_starting
            )
            preterminal_transition_delta = (
                terminal_step_start["transition_count_before"]
                - self.eartts_transition_count_starting
            )
            preterminal_reuse_delta = (
                terminal_step_start["reuse_count_before"] - self.eartts_reuse_count_starting
            )
            if (
                min(
                    preterminal_reset_delta,
                    preterminal_transition_delta,
                    preterminal_reuse_delta,
                )
                < 0
                or preterminal_reuse_delta != 0
                or preterminal_transition_delta != preterminal_reset_delta
            ):
                violations.append("eartts_preterminal_fc_accounting")
            if (
                terminal_step_start["request_id_before"] != terminal_step_start["request_id_after"]
                or terminal_step_start["request_id_after"] != expected_request_id
            ):
                violations.append("eartts_bos_step_request_mismatch")
            if expected_backend_identity is None or (
                terminal_step_start["backend_request_id_after"]
                != expected_backend_identity["backend_request_id"]
                or terminal_step_start["backend_generation_after"]
                != expected_backend_identity["backend_generation"]
            ):
                violations.append("eartts_bos_step_after_identity_stale")
            if reset_delta == 1 and (
                terminal_step_start["backend_request_id_before"]
                == terminal_step_start["backend_request_id_after"]
                or terminal_step_start["backend_generation_after"]
                != terminal_step_start["backend_generation_before"] + 1
            ):
                violations.append("eartts_bos_step_backend_rotation")
            if reuse_delta == 1 and (
                terminal_step_start["backend_request_id_before"]
                != terminal_step_start["backend_request_id_after"]
                or terminal_step_start["backend_generation_before"]
                != terminal_step_start["backend_generation_after"]
            ):
                violations.append("eartts_bos_step_backend_reuse")
        if not isinstance(event, dict):
            violations.append("eartts_bos_event_missing")
        else:
            if event.get("transition_count") != transition_after:
                violations.append("eartts_bos_event_transition_stale")
            if event.get("reset_count") != reset_after:
                violations.append("eartts_bos_event_reset_stale")
            if event.get("reuse_count") != reuse_after:
                violations.append("eartts_bos_event_reuse_stale")
            if expected_request_id is None or event.get("request_id") != expected_request_id:
                violations.append("eartts_bos_event_request_mismatch")
            if expected_backend_identity is None:
                violations.append("eartts_bos_event_backend_identity_missing")
            elif (
                event.get("backend_request_id") != expected_backend_identity["backend_request_id"]
                or event.get("backend_generation")
                != expected_backend_identity["backend_generation"]
            ):
                violations.append("eartts_bos_event_backend_identity_mismatch")
            if event.get("session_epoch") != status.get("session_epoch"):
                violations.append("eartts_bos_event_session_epoch_mismatch")
            if event.get("prepare_epoch") != status.get("prepare_epoch"):
                violations.append("eartts_bos_event_prepare_epoch_mismatch")
            mode = event.get("mode")
            if reset_delta == 1 and (
                mode != "legacy_fallback" or event.get("armed_client_turn_id") is not None
            ):
                violations.append("eartts_bos_legacy_event_invalid")
            if reuse_delta == 1 and (
                mode != "prepared_reuse"
                or self.expected_client_turn_id is None
                or event.get("armed_client_turn_id") != self.expected_client_turn_id
            ):
                violations.append("eartts_bos_reuse_event_invalid")
        if status.get("state") not in {"dirty", "invalid"}:
            violations.append("eartts_bos_epoch_not_consumed")
        if status.get("armed_client_turn_id") is not None:
            violations.append("eartts_bos_epoch_still_armed")
        self.terminal_bos_epoch = {
            "schema": 1,
            "expected_client_turn_id": self.expected_client_turn_id,
            "reset_count_starting": self.eartts_reset_count_starting,
            "reset_count_after": reset_after,
            "transition_count_starting": self.eartts_transition_count_starting,
            "transition_count_after": transition_after,
            "reuse_count_starting": self.eartts_reuse_count_starting,
            "reuse_count_after": reuse_after,
            "terminal_step_start": terminal_step_start,
            "terminal_step_deltas": {
                "reset": reset_delta,
                "transition": transition_delta,
                "reuse": reuse_delta,
            },
            "preterminal_fc_deltas": (
                None
                if terminal_step_start is None
                else {
                    "reset": terminal_step_start["reset_count_before"]
                    - self.eartts_reset_count_starting,
                    "transition": terminal_step_start["transition_count_before"]
                    - self.eartts_transition_count_starting,
                    "reuse": terminal_step_start["reuse_count_before"]
                    - self.eartts_reuse_count_starting,
                }
            ),
            "last_bos_event": dict(event) if isinstance(event, dict) else None,
        }
        return violations

    def observe_separate_terminal(self, engine: Any, result: Any) -> None:
        """Validate a legacy/tool BOS before its model step is published."""

        turn_state = result.turn_state or {}
        boundary = turn_state.get("response_boundary") or {}
        violations: list[str] = []
        if turn_state.get("agent_control") != "agent_bos":
            violations.append("terminal_bos_missing")
        if boundary.get("event") != "start" or boundary.get("phase") != "responding":
            violations.append("response_boundary_not_started")
        if not response_audio_is_deliverable(turn_state):
            violations.append("terminal_audio_not_deliverable")
        if function_cycle_active(result):
            violations.append("function_cycle_effective_at_bos")
        violations.extend(self._validate_terminal_bos_epoch(engine, result))
        if violations:
            raise UserEouSettlementFailure(
                "separate_eou_contract_failure",
                "Separate EOU/BOS position violated its terminal contract",
                {**self.evidence(), "terminal_violations": violations},
            )
        self.separate_terminal_bos = True

    def _reject_initial_activity(self, result: Any) -> None:
        turn_state = result.turn_state or {}
        boundary = turn_state.get("response_boundary") or {}
        response_open = (
            boundary.get("event") is not None
            or boundary.get("phase") not in {None, "idle"}
            or turn_state.get("agent_control") != "pad"
            or response_audio_is_deliverable(turn_state)
        )
        effective_function = function_cycle_active(result)
        if not response_open and not effective_function:
            return
        code = "pre_eou_function_activity" if effective_function else "pre_eou_response_activity"
        raise UserEouSettlementFailure(
            code,
            "Model activity was already effective before explicit user EOU settlement",
            self.evidence(),
        )

    def evidence(self) -> dict[str, Any]:
        steps = list(self.steps or [])
        evidence = {
            "target_blank_frames": self.target_blank_frames,
            "starting_blank_frames": self.starting_blank_frames,
            "ending_blank_frames": self.ending_blank_frames,
            "max_model_steps": self.max_model_steps,
            "model_steps": self.model_steps,
            "eartts_reset_count_starting": self.eartts_reset_count_starting,
            "fence_reached": self.ending_blank_frames >= self.target_blank_frames,
            "within_bound": self.model_steps <= self.max_model_steps,
            "pre_eou_clean": not any(step["violations"] for step in steps),
            "steps": steps,
        }
        if self.fused_terminal_bos:
            evidence["fused_terminal_bos"] = True
        if self.separate_terminal_bos:
            evidence["separate_terminal_bos"] = True
        if self.deferred_terminal_bos:
            evidence["deferred_terminal_bos"] = True
        if self.eartts_transition_count_starting is not None:
            evidence.update(
                schema=2,
                expected_client_turn_id=self.expected_client_turn_id,
                eartts_transition_count_starting=self.eartts_transition_count_starting,
                eartts_reuse_count_starting=self.eartts_reuse_count_starting,
                terminal_bos_epoch=self.terminal_bos_epoch,
            )
        return evidence


@dataclass(frozen=True)
class ValidatedFunctionOutput:
    """Canonical full result plus the exact client-authored injection value."""

    output: str
    output_bytes: int
    output_tokens: int
    model_output: str | None
    model_output_bytes: int | None
    model_output_tokens: int | None
    injection_output: str
    injection_tokens: int


def _function_output_token_count(engine: Any, output: str) -> int:
    """Size one value in the exact wrapper consumed by NVIDIA Speech."""

    wrapper = engine.pipeline.s2s_model
    build_tokens = getattr(wrapper, "_build_fc_response_tokens", None)
    if not callable(build_tokens):
        raise ValueError("model cannot size function_call_output injection")
    wrapped = f"<TOOL_RESPONSE>[{output}]</TOOL_RESPONSE>"
    if bool(getattr(wrapper, "_fc_convert_num_to_text", False)):
        convert_numbers = getattr(wrapper, "_convert_tool_response_nums_to_text", None)
        if not callable(convert_numbers):
            raise ValueError("model cannot normalize function_call_output injection")
        wrapped = convert_numbers(wrapped)
    return len(build_tokens(wrapped))


def validate_function_output(
    engine: Any,
    output: Any,
    *,
    model_output: Any = None,
    model_output_negotiated: bool = False,
) -> ValidatedFunctionOutput:
    """Validate full and model-visible tool results without deriving either one."""

    if not isinstance(output, str):
        raise ValueError("function_call_output output must be a string")
    output_bytes = len(output.encode("utf-8"))
    if output_bytes > FUNCTION_OUTPUT_MAX_BYTES:
        raise ValueError(f"function_call_output exceeds {FUNCTION_OUTPUT_MAX_BYTES} UTF-8 bytes")
    if model_output is not None and not model_output_negotiated:
        raise ValueError("function_call_output model_output was not negotiated")
    output_tokens = _function_output_token_count(engine, output)

    if model_output is not None:
        if not isinstance(model_output, str) or not model_output.strip():
            raise ValueError("function_call_output model_output must be a non-empty string")
        model_output_bytes = len(model_output.encode("utf-8"))
        if model_output_bytes > FUNCTION_OUTPUT_MAX_BYTES:
            raise ValueError(
                "function_call_output model_output exceeds "
                f"{FUNCTION_OUTPUT_MAX_BYTES} UTF-8 bytes"
            )
        model_output_tokens = _function_output_token_count(engine, model_output)
        injection_output = model_output
        injection_tokens = model_output_tokens
    else:
        model_output_bytes = None
        model_output_tokens = None
        injection_output = output
        injection_tokens = output_tokens

    if injection_tokens > FUNCTION_OUTPUT_MAX_TOKENS:
        raise ValueError(
            "function_call_output expands to "
            f"{injection_tokens} tokens; maximum is {FUNCTION_OUTPUT_MAX_TOKENS}"
        )
    return ValidatedFunctionOutput(
        output=output,
        output_bytes=output_bytes,
        output_tokens=output_tokens,
        model_output=model_output,
        model_output_bytes=model_output_bytes,
        model_output_tokens=model_output_tokens,
        injection_output=injection_output,
        injection_tokens=injection_tokens,
    )


async def advance_function_output_recovery(
    step: Callable[[], Awaitable[Any]],
    *,
    max_frames: int = FUNCTION_OUTPUT_APPLY_MAX_FRAMES,
    frame_seconds: float = FRAME_SECONDS,
    completion: Callable[[Any], bool] | None = None,
) -> bool:
    """Advance post-tool recovery on its real-time model-frame clock."""

    next_frame_at = time.monotonic()
    for frame_index in range(max_frames):
        result = await step()
        if not function_cycle_active(result) and (completion is None or bool(completion(result))):
            return True
        if frame_index + 1 < max_frames:
            next_frame_at += frame_seconds
            await asyncio.sleep(max(0.0, next_frame_at - time.monotonic()))
    return False


@dataclass
class ActiveTypedInput:
    job_id: str
    text: str
    cancel_event: asyncio.Event
    done: asyncio.Event
    audio_started: bool = False
    replacement_requested: bool = False


def parse_nvidia_driver_version(version_text: str) -> str | None:
    """Extract the host NVIDIA driver version from /proc's NVRM banner."""
    match = re.search(r"\b(\d{3,4}(?:\.\d+){1,3})\s+Release Build\b", version_text)
    return match.group(1) if match else None


def host_runtime_provenance() -> dict[str, str | None]:
    """Return host facts that materially affect CUDA runtime reproducibility."""
    driver_version = None
    nvrm_version = Path("/proc/driver/nvidia/version")
    try:
        driver_version = parse_nvidia_driver_version(nvrm_version.read_text(encoding="utf-8"))
    except OSError:
        pass
    if driver_version is None:
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            driver_version = next(
                (line.strip() for line in output.splitlines() if line.strip()), None
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "driver_version": driver_version,
        "kernel_version": platform.release(),
    }


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment switch with an explicit default."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be a boolean, got {value!r}")


def fc_async_heartbeat_enabled() -> bool:
    """Return whether the diagnostic heartbeat and thread dump are enabled."""
    return env_bool(FC_ASYNC_HEARTBEAT_ENV, False)


def fc_async_benchmark_enabled() -> bool:
    """Return whether the opt-in FC latency ledger is enabled."""
    return env_bool(FC_ASYNC_BENCHMARK_ENV, False)


def fc_fast_tool_grace_ms() -> float:
    """Return the bounded opt-in grace before an on-hold utterance."""
    raw = os.environ.get(FC_FAST_TOOL_GRACE_MS_ENV, "0")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{FC_FAST_TOOL_GRACE_MS_ENV} must be a finite number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1000.0:
        raise SystemExit(f"{FC_FAST_TOOL_GRACE_MS_ENV} must be between zero and 1000")
    return value


def fc_always_acknowledge_tools() -> list[str]:
    """Return explicitly named tools that always retain layer-one audio."""
    raw = os.environ.get(FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV, "")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if len(names) != len(set(names)):
        raise SystemExit(f"{FC_ALWAYS_ACKNOWLEDGE_TOOLS_ENV} contains duplicate tool names")
    return names


def _finite_number(value: Any, *, nonnegative: bool = True) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (not nonnegative or float(value) >= 0)
    )


def _validated_tool_wait_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item = dict(value)
    expected = {
        "schema",
        "tool_name",
        "policy",
        "grace_ms",
        "tool_started_monotonic_s",
        "tool_completed_monotonic_s",
        "grace_expired_monotonic_s",
        "tool_completed_before_reminder",
        "reminder_text",
        "reminder_variant_sha256",
        "reminder_tokens",
        "reminder_started_monotonic_s",
        "reminder_ended_monotonic_s",
        "reminder_samples",
        "phase2_started_monotonic_s",
        "result_available_to_phase2_ms",
        "fast_path",
    }
    if set(item) != expected or item["schema"] != 1:
        return None
    if not isinstance(item["tool_name"], str):
        return None
    if item["policy"] not in {"disabled", "fast_grace", "always_acknowledge"}:
        return None
    if not _finite_number(item["grace_ms"]) or float(item["grace_ms"]) > 1000.0:
        return None
    for key in (
        "tool_started_monotonic_s",
        "tool_completed_monotonic_s",
        "phase2_started_monotonic_s",
        "result_available_to_phase2_ms",
    ):
        if not _finite_number(item[key]):
            return None
    if not isinstance(item["tool_completed_before_reminder"], bool) or not isinstance(
        item["fast_path"], bool
    ):
        return None
    for key in ("reminder_tokens", "reminder_samples"):
        if not isinstance(item[key], int) or isinstance(item[key], bool) or item[key] < 0:
            return None
    optional_numbers = (
        "grace_expired_monotonic_s",
        "reminder_started_monotonic_s",
        "reminder_ended_monotonic_s",
    )
    if any(item[key] is not None and not _finite_number(item[key]) for key in optional_numbers):
        return None
    for key in ("reminder_text", "reminder_variant_sha256"):
        if item[key] is not None and not isinstance(item[key], str):
            return None
    start = float(item["tool_started_monotonic_s"])
    complete = float(item["tool_completed_monotonic_s"])
    phase2 = float(item["phase2_started_monotonic_s"])
    if not start <= complete <= phase2:
        return None
    expected_delay = (phase2 - complete) * 1000.0
    if not math.isclose(
        float(item["result_available_to_phase2_ms"]),
        expected_delay,
        rel_tol=0.0,
        abs_tol=0.05,
    ):
        return None
    reminder_fields = (
        item["reminder_text"],
        item["reminder_variant_sha256"],
        item["reminder_started_monotonic_s"],
        item["reminder_ended_monotonic_s"],
    )
    has_reminder = any(field is not None for field in reminder_fields)
    if has_reminder and any(field is None for field in reminder_fields):
        return None
    if has_reminder:
        if not (
            start
            <= float(item["reminder_started_monotonic_s"])
            <= float(item["reminder_ended_monotonic_s"])
            <= phase2
        ):
            return None
        if item["reminder_tokens"] <= 0 or item["reminder_samples"] <= 0:
            return None
        digest = item["reminder_variant_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            return None
    elif item["reminder_samples"] != 0:
        return None
    elif item["tool_completed_before_reminder"]:
        return None
    if item["fast_path"]:
        if item["policy"] != "fast_grace" or has_reminder:
            return None
        if item["grace_expired_monotonic_s"] is not None:
            return None
    elif item["policy"] == "fast_grace" and item["grace_expired_monotonic_s"] is None:
        return None
    grace_expired = item["grace_expired_monotonic_s"]
    if grace_expired is not None:
        if not start <= float(grace_expired) <= phase2:
            return None
        if has_reminder and float(grace_expired) > float(item["reminder_started_monotonic_s"]):
            return None
    if item["policy"] == "disabled" and float(item["grace_ms"]) != 0.0:
        return None
    if item["policy"] == "fast_grace" and float(item["grace_ms"]) <= 0.0:
        return None
    return item


def _validated_nano_engine_benchmark(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item = dict(value)
    expected = {
        "schema",
        "request_id",
        "backend_request_id",
        "backend_generation",
        "position",
        "action",
        "input_copy_ms",
        "submit_ms",
        "output_wait_ms",
        "parse_ms",
        "named_sum_ms",
        "total_ms",
        "residual_ms",
    }
    if set(item) != expected or item["schema"] != 1:
        return None
    if not all(
        isinstance(item[key], str) and item[key] for key in ("request_id", "backend_request_id")
    ):
        return None
    if item["action"] not in {"initial_submit", "append"}:
        return None
    for key in ("backend_generation", "position"):
        if not isinstance(item[key], int) or isinstance(item[key], bool) or item[key] < 0:
            return None
    for key in (
        "input_copy_ms",
        "submit_ms",
        "output_wait_ms",
        "parse_ms",
        "named_sum_ms",
        "total_ms",
    ):
        if not _finite_number(item[key]):
            return None
    if not _finite_number(item["residual_ms"], nonnegative=False):
        return None
    expected_sum = sum(
        float(item[key]) for key in ("input_copy_ms", "submit_ms", "output_wait_ms", "parse_ms")
    )
    if not math.isclose(float(item["named_sum_ms"]), expected_sum, rel_tol=0.0, abs_tol=0.05):
        return None
    if not math.isclose(
        float(item["total_ms"]),
        expected_sum + float(item["residual_ms"]),
        rel_tol=0.0,
        abs_tol=0.05,
    ):
        return None
    return item


def _validated_fc_async_benchmark_phase(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item = dict(value)
    expected = {
        "schema",
        "cycle_sotc_frame",
        "phase",
        "invocation",
        "request_id",
        "phase_start_monotonic_s",
        "phase_end_monotonic_s",
        "wall_ms",
        "positions_count",
        "positions_retained",
        "positions_truncated",
        "nano_position_start",
        "nano_position_end",
        "real_audio_frames",
        "perception_calls",
        "rnnt_steps",
        "tts_calls",
        "eotr_wait_steps",
        "stage_totals_ms",
        "named_sum_ms",
        "residual_ms",
        "positions",
    }
    if set(item) != expected or item["schema"] != 1:
        return None
    if item["phase"] not in {"tool_call", "tool_response"} or item["invocation"] not in {1, 2}:
        return None
    if (item["phase"], item["invocation"]) not in {("tool_call", 1), ("tool_response", 2)}:
        return None
    if not isinstance(item["request_id"], str) or not item["request_id"]:
        return None
    integer_keys = (
        "cycle_sotc_frame",
        "positions_count",
        "positions_retained",
        "nano_position_start",
        "nano_position_end",
        "real_audio_frames",
        "perception_calls",
        "rnnt_steps",
        "tts_calls",
        "eotr_wait_steps",
    )
    if any(
        not isinstance(item[key], int) or isinstance(item[key], bool) or item[key] < 0
        for key in integer_keys
    ):
        return None
    if not isinstance(item["positions_truncated"], bool):
        return None
    if any(
        not _finite_number(item[key], nonnegative=key != "residual_ms")
        for key in (
            "phase_start_monotonic_s",
            "phase_end_monotonic_s",
            "wall_ms",
            "named_sum_ms",
            "residual_ms",
        )
    ):
        return None
    if float(item["phase_end_monotonic_s"]) < float(item["phase_start_monotonic_s"]):
        return None
    stage_keys = {
        "embed_ms",
        "perception_poll_ms",
        "fusion_ms",
        "nano_interface_ms",
        "eartts_infer_ms",
        "codec_decode_ms",
        "codec_cpu_copy_ms",
        "rnnt_ms",
        "perception_join_ms",
    }
    stages = item["stage_totals_ms"]
    if (
        not isinstance(stages, dict)
        or set(stages) != stage_keys
        or any(not _finite_number(value) for value in stages.values())
    ):
        return None
    positions = item["positions"]
    if (
        not isinstance(positions, list)
        or len(positions) != item["positions_retained"]
        or len(positions) > 512
    ):
        return None
    position_keys = {
        "schema",
        "async_step",
        "nano_position",
        "monotonic_s",
        "embed_ms",
        "perception_poll_ms",
        "fusion_ms",
        "nano_interface_ms",
        "nano_engine",
        "eartts_infer_ms",
        "codec_decode_ms",
        "codec_cpu_copy_ms",
        "rnnt_ms",
        "named_sum_ms",
        "total_ms",
        "residual_ms",
    }
    copied_positions = []
    for position in positions:
        if (
            not isinstance(position, dict)
            or set(position) != position_keys
            or position["schema"] != 1
        ):
            return None
        for key in ("async_step", "nano_position"):
            if (
                not isinstance(position[key], int)
                or isinstance(position[key], bool)
                or position[key] < 0
            ):
                return None
        number_keys = position_keys - {
            "schema",
            "async_step",
            "nano_position",
            "nano_engine",
            "residual_ms",
        }
        if any(not _finite_number(position[key]) for key in number_keys) or not _finite_number(
            position["residual_ms"], nonnegative=False
        ):
            return None
        engine = _validated_nano_engine_benchmark(position["nano_engine"])
        if engine is None:
            return None
        position_named_keys = (
            "embed_ms",
            "perception_poll_ms",
            "fusion_ms",
            "nano_interface_ms",
            "eartts_infer_ms",
            "codec_decode_ms",
            "codec_cpu_copy_ms",
            "rnnt_ms",
        )
        position_named_sum = sum(float(position[key]) for key in position_named_keys)
        if not math.isclose(
            float(position["named_sum_ms"]),
            position_named_sum,
            rel_tol=0.0,
            abs_tol=0.05,
        ):
            return None
        if not math.isclose(
            float(position["total_ms"]),
            position_named_sum + float(position["residual_ms"]),
            rel_tol=0.0,
            abs_tol=0.05,
        ):
            return None
        copied = dict(position)
        copied["nano_engine"] = engine
        copied_positions.append(copied)
    if not item["positions_truncated"] and item["positions_count"] != len(copied_positions):
        return None
    named_sum = sum(float(value) for value in stages.values())
    if not math.isclose(float(item["named_sum_ms"]), named_sum, rel_tol=0.0, abs_tol=0.05):
        return None
    if not math.isclose(
        float(item["wall_ms"]), named_sum + float(item["residual_ms"]), rel_tol=0.0, abs_tol=0.05
    ):
        return None
    item["positions"] = copied_positions
    return item


def configure_fc_async_thread_dump() -> bool:
    """Install SIGUSR1 stack dumping only for an explicitly instrumented run."""
    enabled = fc_async_heartbeat_enabled()
    if enabled:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    return enabled


def _valid_optional_nonnegative_int(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _validated_fc_async_heartbeat(value: Any) -> dict[str, Any] | None:
    """Copy and validate one immutable scalar heartbeat for public telemetry."""
    if not isinstance(value, dict):
        return None
    snapshot = dict(value)
    if frozenset(snapshot) != FC_ASYNC_HEARTBEAT_KEYS:
        return None
    if snapshot["schema"] != 1:
        return None
    if (
        not isinstance(snapshot["cycle_sotc_frame"], int)
        or isinstance(snapshot["cycle_sotc_frame"], bool)
        or snapshot["cycle_sotc_frame"] < 0
    ):
        return None
    if snapshot["phase"] not in {"tool_call", "tool_response"}:
        return None
    if snapshot["invocation"] not in {1, 2}:
        return None
    if (snapshot["phase"], snapshot["invocation"]) not in {
        ("tool_call", 1),
        ("tool_response", 2),
    }:
        return None
    if not _valid_optional_nonnegative_int(snapshot["async_step"]):
        return None
    if not _valid_optional_nonnegative_int(snapshot["model_frame"]):
        return None
    if snapshot["stage"] not in FC_ASYNC_HEARTBEAT_STAGES:
        return None
    if snapshot["state"] not in FC_ASYNC_HEARTBEAT_STATES:
        return None
    if (
        (snapshot["stage"] == "between_stages") != (snapshot["state"] == "idle")
        or (snapshot["stage"] == "complete") != (snapshot["state"] == "complete")
        or (snapshot["stage"] == "aborted") != (snapshot["state"] == "aborted")
        or (snapshot["stage"] in {"stage_failed", "failed"}) != (snapshot["state"] == "failed")
    ):
        return None
    for key in ("stage_started_monotonic", "last_completed_monotonic"):
        item = snapshot[key]
        if item is not None and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
        ):
            return None
    for key in ("last_completed_stage", "last_failed_stage"):
        if snapshot[key] is not None and snapshot[key] not in FC_ASYNC_HEARTBEAT_STAGES:
            return None
    if (snapshot["last_completed_stage"] is None) != (snapshot["last_completed_monotonic"] is None):
        return None
    if (
        not isinstance(snapshot["thread_ident"], int)
        or isinstance(snapshot["thread_ident"], bool)
        or snapshot["thread_ident"] <= 0
    ):
        return None
    for key in (
        "live_audio_queue_depth",
        "perception_queue_depth",
        "tts_audio_queue_depth",
        "rnnt_text_queue_depth",
    ):
        item = snapshot[key]
        if item is not None and (
            not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 1_000_000
        ):
            return None
    return snapshot


def fc_async_heartbeat_snapshot(fc_state: Any) -> dict[str, Any] | None:
    """Return validated loop/perception heartbeats without exposing shared dicts."""
    if not isinstance(fc_state, dict):
        return None
    result: dict[str, Any] = {}
    for source_key, public_key in (
        ("async_heartbeat", "loop"),
        ("async_perception_heartbeat", "perception"),
    ):
        heartbeat = _validated_fc_async_heartbeat(fc_state.get(source_key))
        if heartbeat is not None:
            result[public_key] = heartbeat
    return result or None


def websocket_ping_interval() -> float | None:
    """Return the Uvicorn server-ping interval, allowing an explicit disable."""
    value = os.environ.get("VOICECHAT_WEB_WS_PING_INTERVAL", "20").strip().lower()
    if value in {"none", "off", "disabled"}:
        return None
    interval = float(value)
    if interval <= 0:
        raise SystemExit("VOICECHAT_WEB_WS_PING_INTERVAL must be positive or 'none'")
    return interval


def websocket_ping_timeout() -> float | None:
    """Disable ping timeouts whenever server-originated pings are disabled."""
    if websocket_ping_interval() is None:
        return None
    value = os.environ.get("VOICECHAT_WEB_WS_PING_TIMEOUT", "20").strip().lower()
    if value in {"none", "off", "disabled"}:
        return None
    timeout = float(value)
    if timeout <= 0:
        raise SystemExit("VOICECHAT_WEB_WS_PING_TIMEOUT must be positive or 'none'")
    return timeout


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifested_model(root: Path, component: dict) -> dict:
    """Validate a monolithic or sharded vLLM model declared by its manifest."""
    declared_path = component.get("path")
    if declared_path and Path(declared_path).resolve() != root.resolve():
        raise SystemExit(f"Manifest component path mismatch: {declared_path} != {root}")

    validated_files: dict[str, dict[str, Any]] = {}
    for name, expected in sorted((component.get("files") or {}).items()):
        if not name or Path(name).name != name:
            raise SystemExit(f"Unsafe manifested artifact filename: {name!r}")
        path = root / name
        if not path.is_file():
            raise SystemExit(f"Missing manifested artifact file: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != {
            "bytes": expected.get("bytes"),
            "sha256": expected.get("sha256"),
        }:
            raise SystemExit(f"Manifest mismatch for artifact file: {path}")
        validated_files[name] = actual

    model = component.get("model") or {}
    declared_files = model.get("files")
    if declared_files:
        digest = hashlib.sha256()
        total_bytes = 0
        for item in sorted(declared_files, key=lambda value: value["name"]):
            name = item.get("name")
            if not name or Path(name).name != name:
                raise SystemExit(f"Unsafe manifested model filename: {name!r}")
            path = root / name
            if not path.is_file():
                raise SystemExit(f"Missing manifested model shard: {path}")
            cached = validated_files.get(name)
            actual_bytes = cached["bytes"] if cached else path.stat().st_size
            actual_sha = cached["sha256"] if cached else sha256_file(path)
            if actual_bytes != item.get("bytes") or actual_sha != item.get("sha256"):
                raise SystemExit(f"Manifest mismatch for model shard: {path}")
            total_bytes += actual_bytes
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(actual_bytes).encode("ascii"))
            digest.update(b"\0")
            digest.update(actual_sha.encode("ascii"))
            digest.update(b"\n")
        actual_sha = digest.hexdigest()
        if total_bytes != model.get("bytes") or actual_sha != model.get("sha256"):
            raise SystemExit(f"Composite model manifest mismatch: {root}")
        return {
            "bytes": total_bytes,
            "sha256": actual_sha,
            "files": declared_files,
            "artifact_files": validated_files or None,
        }

    model_file = root / "model.safetensors"
    expected_sha = model.get("sha256")
    if not expected_sha or not model_file.is_file():
        raise SystemExit(f"Incomplete public vLLM artifact: {root}")
    cached = validated_files.get(model_file.name)
    actual_sha = cached["sha256"] if cached else sha256_file(model_file)
    if actual_sha != expected_sha:
        raise SystemExit(f"Public vLLM SHA mismatch: {actual_sha} != {expected_sha}")
    return {
        "bytes": model_file.stat().st_size,
        "sha256": actual_sha,
        "artifact_files": validated_files or None,
    }


def prepare_agent_logit_diagnostic_nano(
    source: Path, scratch: Path
) -> tuple[Path, dict[str, Any] | None]:
    """Create a weight-identical Nano view exposing text logits only on request."""

    raw_top_k = os.environ.get("S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK", "0").strip()
    try:
        top_k = int(raw_top_k)
    except ValueError as exc:
        raise SystemExit(
            "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK must be an integer from 0 through 20"
        ) from exc
    if not 0 <= top_k <= 20:
        raise SystemExit("S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK must be an integer from 0 through 20")
    if not top_k:
        return source, None
    try:
        trace_frames = int(os.environ.get("S2S_POST_FC_TOKEN_TRACE_FRAMES", "0"))
    except ValueError as exc:
        raise SystemExit(
            "agent-logit tracing requires S2S_POST_FC_TOKEN_TRACE_FRAMES from 1 through 32"
        ) from exc
    if not 1 <= trace_frames <= 32:
        raise SystemExit(
            "agent-logit tracing requires S2S_POST_FC_TOKEN_TRACE_FRAMES from 1 through 32"
        )

    source = source.expanduser().resolve()
    source_config_path = source / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    skip_contract = source_config.get("voicechat_skip_custom_text_logits")
    if (
        source_config.get("custom_outputs") != PRODUCTION_NANO_CUSTOM_OUTPUTS
        or not isinstance(skip_contract, dict)
        or skip_contract.get("enabled") is not True
        or skip_contract.get("contract") != SKIP_TEXT_LOGITS_CONTRACT
        or not isinstance(skip_contract.get("source_config_sha256"), str)
        or len(skip_contract["source_config_sha256"]) != 64
    ):
        raise SystemExit(
            "agent-logit tracing requires the qualified production Nano text-logit skip contract"
        )

    diagnostic_config = dict(source_config)
    diagnostic_config["custom_outputs"] = DIAGNOSTIC_NANO_CUSTOM_OUTPUTS
    diagnostic_config.pop("voicechat_skip_custom_text_logits")
    diagnostic_bytes = (json.dumps(diagnostic_config, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        if not entry.is_file():
            raise SystemExit(f"unexpected non-file in Nano artifact: {entry}")
        if entry.name != "config.json":
            (scratch / entry.name).symlink_to(entry)
    (scratch / "config.json").write_bytes(diagnostic_bytes)
    provenance = {
        "mode": "temporary-text-logit-overlay",
        "top_k": top_k,
        "trace_frames": trace_frames,
        "source_config_sha256": sha256_file(source_config_path),
        "diagnostic_config_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
        "source_custom_outputs": PRODUCTION_NANO_CUSTOM_OUTPUTS,
        "diagnostic_custom_outputs": DIAGNOSTIC_NANO_CUSTOM_OUTPUTS,
        "sampling_contract": skip_contract["contract"],
        "weights_changed": False,
        "performance_metrics_valid": False,
        "performance_metrics_note": (
            "The diagnostic text-head output changes execution cost; latency and "
            "throughput from this run are not qualification measurements."
        ),
    }
    return scratch.resolve(), provenance


def release_component_manifest(release: dict[str, Any], name: str) -> dict[str, Any]:
    """Adapt the signed Hugging Face release inventory to component validation."""

    prefix = f"{name}/"
    files = {
        item["path"].removeprefix(prefix): {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in release.get("files", [])
        if item.get("path", "").startswith(prefix)
    }
    shards = [
        {"name": filename, **details}
        for filename, details in files.items()
        if filename.startswith("model-") and filename.endswith(".safetensors")
    ]
    composite = release.get(f"{name}_composite_sha256")
    if not files or not shards or not composite:
        raise SystemExit(f"Signed release has no complete {name} component")
    return {
        "files": files,
        "model": {
            "files": shards,
            "bytes": sum(item["bytes"] for item in shards),
            "sha256": composite,
        },
        "reproducibility": {"runs": 2, "byte_identical": True},
        "quantization": {"release": release.get("candidate")},
    }


def validate_nano_runtime_optimization(nano_root: Path, model_cfg: Any) -> dict | None:
    """Fail closed when a Nano artifact omits the custom text-logit head.

    vLLM still evaluates its ordinary language-model head to choose the token.
    Omitting the second, custom copy is equivalent only while the surrounding
    NVIDIA wrapper accepts that greedy token without applying another sampler.
    """
    config_path = nano_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    optimization = config.get("voicechat_skip_custom_text_logits") or {}
    if not optimization.get("enabled"):
        return None

    custom_outputs = list(config.get("custom_outputs") or [])
    if "text_logits" in custom_outputs:
        raise SystemExit("Nano skip-custom-text-logits artifact still declares text_logits")
    required_outputs = {"function_tokens", "function_logits"}
    if not required_outputs <= set(custom_outputs):
        raise SystemExit("Nano skip-custom-text-logits artifact must retain both function outputs")

    actual = {
        "temperature": float(model_cfg.get("temperature", 1.0)),
        "top_p": float(model_cfg.get("top_p", 1.0)),
        "repetition_penalty": float(model_cfg.get("repetition_penalty", 1.0)),
    }
    expected = {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}
    if actual != expected:
        raise SystemExit(
            "Nano skip-custom-text-logits requires greedy wrapper sampling: "
            f"expected {expected}, got {actual}"
        )
    return {
        "skip_custom_text_logits": True,
        "sampling_contract": actual,
        "retained_custom_outputs": custom_outputs,
    }


def validate_nano_pad_pair_runtime(
    nano_root: Path, manifest: dict[str, Any], requested: bool
) -> dict[str, Any]:
    """Fail closed unless the live artifact declares the qualified pair contract."""
    config = json.loads((nano_root / "config.json").read_text(encoding="utf-8"))
    contract = config.get("voicechat_pad_pair_contract") or {}
    result = {
        "enabled": bool(requested),
        "conditional_previous_effective_pad": env_bool(
            "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL", False
        ),
        "pad_token_id": config.get("voicechat_pad_pair_token_id"),
        "mamba_chunk_size": config.get("chunk_size"),
        "vllm_max_model_len": int(os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")),
        "streaming_max_len": int(os.environ.get("VOICECHAT_STREAMING_MAX_LEN", "12288")),
        "max_session_model_frames": int(
            os.environ.get("VOICECHAT_WEB_MAX_SESSION_MODEL_FRAMES", "12000")
        ),
        "contract": contract,
    }
    if not requested:
        return result
    required = {
        "scope": "production live runtime",
        "max_positions": 2,
        "production_mamba_rollback": True,
        "function_channel_rejection": True,
        "post_fc_fusion_correction": True,
        "mamba_chunk_size": 1,
    }
    actual = {key: contract.get(key) for key in required}
    if actual != required or config.get("voicechat_pad_pair_token_id") is None:
        raise SystemExit(f"Nano PAD-pair production contract mismatch: {actual} != {required}")
    if manifest.get("kind") == "nemotron_voicechat_dgx_spark_hf_release":
        if manifest.get("candidate") != "production-candidate-1":
            raise SystemExit("Unsupported signed VoiceChat artifact candidate")
        release_contract = manifest.get("runtime_contract") or {}
        expected_release_contract = {
            "nano_pad_pair": True,
            "nano_pad_pair_conditional": True,
            "nano_pad_pair_control_barrier": True,
            "vllm_max_model_len": 12288,
            "streaming_max_len": 12288,
            "max_session_model_frames": 12000,
            "agent_no_text_frames": 30,
            "agent_no_audio_frames": 30,
            "agent_silence_eos_frames": 20,
        }
        actual_release_contract = {
            key: release_contract.get(key) for key in expected_release_contract
        }
        if actual_release_contract != expected_release_contract:
            raise SystemExit("Signed release runtime contract mismatch")
        if not result["conditional_previous_effective_pad"]:
            raise SystemExit("Production Candidate 1 requires conditional PAD drafting")
        for key in ("vllm_max_model_len", "streaming_max_len", "max_session_model_frames"):
            if result[key] != expected_release_contract[key]:
                raise SystemExit(f"Production Candidate 1 {key} mismatch")
        result["production_candidate"] = {
            "number": 1,
            "status": "signed_release",
            "release_sha256": manifest.get("release_sha256"),
        }
        return result
    manifest_contract = manifest.get("runtime_optimization") or {}
    required_manifest = {
        "pad_pair_production": True,
        "production_mamba_rollback": True,
        "function_channel_rejection": True,
        "post_fc_fusion_correction": True,
    }
    if any(manifest_contract.get(key) is not value for key, value in required_manifest.items()):
        raise SystemExit("Nano PAD-pair manifest lacks the production contract")
    candidate = manifest.get("production_candidate")
    if candidate is not None:
        if candidate.get("number") != 1 or candidate.get("status") != "frozen":
            raise SystemExit("Unsupported or unfrozen VoiceChat production candidate")
        if not result["conditional_previous_effective_pad"]:
            raise SystemExit("Production Candidate 1 requires conditional PAD drafting")
        conditional_contract = {
            "conditional_previous_effective_pad": True,
            "function_async_sequential_latch": True,
            "function_async_pending_drain": True,
            "agent_no_text_frames": 30,
            "agent_no_audio_frames": 30,
            "agent_decoded_silence_frames": 20,
            "vllm_max_model_len": 12288,
            "streaming_max_len": 12288,
            "max_session_model_frames": 12000,
            "production_candidate": 1,
        }
        actual_conditional = {key: contract.get(key) for key in conditional_contract}
        if actual_conditional != conditional_contract:
            raise SystemExit(
                "Production Candidate 1 config contract mismatch: "
                f"{actual_conditional} != {conditional_contract}"
            )
        if result["vllm_max_model_len"] != conditional_contract["vllm_max_model_len"]:
            raise SystemExit(
                "Production Candidate 1 max-model-length mismatch: "
                f"{result['vllm_max_model_len']} != "
                f"{conditional_contract['vllm_max_model_len']}"
            )
        if result["streaming_max_len"] != conditional_contract["streaming_max_len"]:
            raise SystemExit(
                "Production Candidate 1 streaming-buffer-length mismatch: "
                f"{result['streaming_max_len']} != "
                f"{conditional_contract['streaming_max_len']}"
            )
        if result["max_session_model_frames"] != conditional_contract["max_session_model_frames"]:
            raise SystemExit(
                "Production Candidate 1 session-frame guard mismatch: "
                f"{result['max_session_model_frames']} != "
                f"{conditional_contract['max_session_model_frames']}"
            )
        if any(manifest_contract.get(key) != value for key, value in conditional_contract.items()):
            raise SystemExit("Production Candidate 1 manifest contract mismatch")
        source_root = Path(__file__).resolve().parents[2]
        runtime_sources = candidate.get("runtime_sources") or {}
        required_sources = set(RUNTIME_SOURCE_PATHS)
        if set(runtime_sources) != required_sources:
            raise SystemExit("Production Candidate 1 runtime source set mismatch")
        for name, expected in runtime_sources.items():
            path = source_root / name
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                raise SystemExit(f"Production Candidate 1 source size mismatch: {name}")
            actual_sha = sha256_file(path)
            if actual_sha != expected["sha256"]:
                raise SystemExit(
                    f"Production Candidate 1 source SHA mismatch for {name}: "
                    f"{actual_sha} != {expected['sha256']}"
                )
        result["production_candidate"] = {
            "number": 1,
            "status": "frozen",
            "runtime_image": candidate.get("runtime_image"),
            "runtime_image_id": candidate.get("runtime_image_id"),
            "runtime_sources": runtime_sources,
        }
    return result


def pcm16_bytes_to_float32(payload: bytes) -> np.ndarray:
    """Decode little-endian mono PCM16 to normalized float32."""
    if len(payload) % 2:
        raise ValueError("PCM16 payload has an odd byte count")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Encode normalized audio as clipped little-endian PCM16."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


def clean_display_text(value: str) -> str:
    """Remove model control/timestamp markers without changing human text."""
    value = re.sub(r"</?s>|<SPECIAL_12>|<\$[\d.]+\$>|<\|[\d.]+\|>", " ", value or "")
    return " ".join(value.lstrip("^").replace("\u0120", " ").split())


def normalize_tool_definitions(value: Any) -> list[dict[str, Any]]:
    """Validate OpenAI- or NVIDIA-style function schemas for one session."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("session.tools must be an array")
    if len(value) > 32:
        raise ValueError("session.tools supports at most 32 functions")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"session.tools[{index}] must be an object")
        function = item.get("function", item)
        if not isinstance(function, dict):
            raise ValueError(f"session.tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not TOOL_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"session.tools[{index}] has an invalid function name")
        registry_name = name.replace("-", "_")
        if registry_name in seen:
            raise ValueError(f"duplicate function name after normalization: {name}")
        seen.add(registry_name)
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ValueError(f"session.tools[{index}].parameters must be an object")
        normalized.append(
            {
                "name": name,
                "description": str(function.get("description", "")),
                "parameters": parameters,
            }
        )
    return normalized


def render_tool_system_prompt(
    template_path: Path, system_message: str, tools: list[dict[str, Any]]
) -> str:
    """Render NVIDIA's function prompt plus the live grounding requirements."""
    if not tools:
        return system_message
    from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
        render_fc_system_prompt,
    )

    rendered = render_fc_system_prompt(str(template_path), system_message, tools)
    return f"{rendered}\n\n{TOOL_GROUNDING_INSTRUCTIONS}"


def coerce_tool_arguments(arguments: Any, schema: Any) -> Any:
    """Conservatively restore JSON scalar types declared by a tool schema.

    NVIDIA's public function head sometimes spells numeric arguments as JSON
    strings.  The external transport must still honor the schema advertised by
    the client.  Only exact, lossless scalar spellings are coerced; all other
    values are preserved for the tool implementation to validate.
    """
    if not isinstance(schema, dict):
        return arguments
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(arguments, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return arguments
        return {
            key: coerce_tool_arguments(value, properties.get(key))
            for key, value in arguments.items()
        }
    if schema_type == "array" and isinstance(arguments, list):
        item_schema = schema.get("items")
        return [coerce_tool_arguments(value, item_schema) for value in arguments]
    if not isinstance(arguments, str):
        return arguments
    stripped = arguments.strip()
    if schema_type == "integer" and re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", stripped):
        return int(stripped)
    if schema_type == "number" and re.fullmatch(
        r"[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        stripped,
    ):
        value = float(stripped)
        return value if math.isfinite(value) else arguments
    if schema_type == "boolean" and stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    return arguments


@dataclass
class PendingExternalToolCall:
    call_id: str
    name: str
    arguments: str
    completed: threading.Event
    output: str | None = None
    model_output: str | None = None
    injection_output: str | None = None


class ExternalToolBridge:
    """Bridge NVIDIA's blocking tool hook to two-phase WebSocket events."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        emit: Any,
        *,
        emit_timeout: Any | None = None,
        timeout_seconds: float = FUNCTION_CALL_TIMEOUT_SECONDS,
    ):
        self.loop = loop
        self.emit = emit
        self.emit_timeout = emit_timeout
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, PendingExternalToolCall] = {}
        self._closed = False

    @staticmethod
    def _arguments_json(arguments: Any) -> str:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"value": arguments}
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        return json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"))

    def handlers(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        handlers: dict[str, Any] = {}
        for tool in tools:
            advertised_name = tool["name"]
            parameters = tool.get("parameters", {})

            def handler(
                arguments: Any,
                *,
                _name: str = advertised_name,
                _parameters: Any = parameters,
            ) -> str:
                return self._invoke(_name, coerce_tool_arguments(arguments, _parameters))

            handlers[advertised_name.replace("-", "_")] = handler
        return handlers

    def _invoke(self, name: str, arguments: Any) -> str:
        call = PendingExternalToolCall(
            call_id=f"call_{uuid.uuid4().hex}",
            name=name,
            arguments=self._arguments_json(arguments),
            completed=threading.Event(),
        )
        with self._lock:
            if self._closed:
                return json.dumps({"error": "session_closed"})
            self._pending[call.call_id] = call
        future = asyncio.run_coroutine_threadsafe(self.emit(call), self.loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:
            with self._lock:
                self._pending.pop(call.call_id, None)
            return json.dumps({"error": f"function_call_event_failed: {exc}"})
        if not call.completed.wait(self.timeout_seconds):
            with self._lock:
                self._pending.pop(call.call_id, None)
            if self.emit_timeout is not None:
                timeout_future = asyncio.run_coroutine_threadsafe(
                    self.emit_timeout(call), self.loop
                )
                try:
                    timeout_future.result(timeout=5.0)
                except Exception:
                    pass
            return json.dumps({"error": "function_call_output_timeout"})
        with self._lock:
            self._pending.pop(call.call_id, None)
        return (
            call.injection_output
            if call.injection_output is not None
            else json.dumps({"error": "empty_output"})
        )

    def submit(self, call_id: str, output: ValidatedFunctionOutput) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("function_call_output requires call_id")
        with self._lock:
            call = self._pending.get(call_id)
            if call is None:
                raise ValueError(f"unknown or completed call_id: {call_id}")
            if call.completed.is_set():
                raise ValueError(f"duplicate function output for call_id: {call_id}")
            if not isinstance(output, ValidatedFunctionOutput):
                raise TypeError("function output must be validated before submission")
            call.output = output.output
            call.model_output = output.model_output
            call.injection_output = output.injection_output
            call.completed.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for call in self._pending.values():
                if call.completed.is_set():
                    continue
                call.output = json.dumps({"error": "session_closed"})
                call.injection_output = call.output
                call.completed.set()


def audio_stats(samples: np.ndarray) -> dict[str, float | int]:
    """Return compact signal diagnostics for one PCM payload or model frame."""
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not values.size:
        return {"samples": 0, "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0}
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    return {
        "samples": int(values.size),
        "rms": round(rms, 8),
        "rms_dbfs": round(max(-120.0, 20.0 * math.log10(max(rms, 1e-6))), 3),
        "peak": round(float(np.max(np.abs(values))), 8),
    }


def response_audio_is_deliverable(turn_state: dict[str, Any]) -> bool:
    """Expose decoded audio only while a tokenized assistant turn is open."""
    boundary = turn_state.get("response_boundary")
    if not isinstance(boundary, dict):
        return False
    return (
        boundary.get("phase") in {"responding", "tail_draining"} or boundary.get("event") == "start"
    )


class SessionTrace:
    """Lossless application-payload capture plus structured streaming events."""

    def __init__(self, root: Path | None, session_id: str):
        self.session_id = session_id
        self.directory: Path | None = None
        self._events = None
        self._input_pcm = None
        self._output_pcm = None
        if root is not None:
            self.directory = root / session_id
            self.directory.mkdir(parents=True, exist_ok=False)
            self._events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
            self._input_pcm = (self.directory / "input-16000-mono-s16le.pcm").open("wb")
            self._output_pcm = (self.directory / "output-22050-mono-s16le.pcm").open("wb")
        self.event(
            "session_trace_started",
            capture_directory=str(self.directory) if self.directory else None,
        )

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "trace": "voicechat",
            "event": kind,
            "session_id": self.session_id,
            "monotonic_s": round(time.monotonic(), 6),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        if self._events is not None:
            self._events.write(line + "\n")
            self._events.flush()

    def input_pcm(self, payload: bytes) -> None:
        if self._input_pcm is not None:
            self._input_pcm.write(payload)
            self._input_pcm.flush()

    def output_pcm(self, payload: bytes) -> None:
        if self._output_pcm is not None:
            self._output_pcm.write(payload)
            self._output_pcm.flush()

    def close(self, reason: str) -> None:
        self.event("session_trace_stopped", reason=reason)
        for stream in (self._events, self._input_pcm, self._output_pcm):
            if stream is not None:
                stream.close()
        self._events = None
        self._input_pcm = None
        self._output_pcm = None


class PcmFrameBuffer:
    """Reassemble arbitrary PCM packets into exact 80 ms model frames."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES):
        self.frame_samples = frame_samples
        self._pending = np.empty(0, dtype=np.float32)

    def push(self, payload: bytes) -> list[np.ndarray]:
        decoded = pcm16_bytes_to_float32(payload)
        if decoded.size:
            self._pending = np.concatenate((self._pending, decoded))
        frames: list[np.ndarray] = []
        while self._pending.size >= self.frame_samples:
            frames.append(self._pending[: self.frame_samples].copy())
            self._pending = self._pending[self.frame_samples :]
        return frames

    def flush(self) -> np.ndarray:
        """Return one zero-padded final frame, including for an empty buffer."""
        frame = np.zeros(self.frame_samples, dtype=np.float32)
        if self._pending.size:
            frame[: self._pending.size] = self._pending
        self._pending = np.empty(0, dtype=np.float32)
        return frame

    @property
    def pending_samples(self) -> int:
        return int(self._pending.size)

    def discard(self) -> int:
        """Discard a partial packet boundary and return its sample count."""

        samples = int(self._pending.size)
        self._pending = np.empty(0, dtype=np.float32)
        return samples

    def take_pending(self) -> np.ndarray:
        """Detach a partial packet so synthetic control steps cannot consume it."""

        pending = self._pending
        self._pending = np.empty(0, dtype=np.float32)
        return pending


@dataclass
class DeferredClientInputQueue:
    """Bound raw client input while an atomic function cycle is in flight."""

    max_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        self._items: deque[tuple[dict[str, Any], int]] = deque()
        self.bytes = 0

    def append(self, event: dict[str, Any], encoded_bytes: int) -> bool:
        if encoded_bytes < 0 or self.bytes + encoded_bytes > self.max_bytes:
            return False
        self._items.append((event, encoded_bytes))
        self.bytes += encoded_bytes
        return True

    def popleft(self) -> dict[str, Any]:
        event, encoded_bytes = self._items.popleft()
        self.bytes -= encoded_bytes
        return event

    def __bool__(self) -> bool:
        return bool(self._items)


@dataclass
class StepResult:
    audio: np.ndarray
    user_text: str
    assistant_delta: str
    assistant_text: str
    inference_ms: float
    server_step_ms: float
    frame_index: int
    input_audio: dict[str, float | int]
    output_audio: dict[str, float | int]
    turn_state: dict[str, Any]
    function_delta: str = ""
    function_text: str = ""
    trace_diagnostics: dict[str, Any] | None = None


@dataclass
class AgentSilenceEosWatchdog:
    """Request model-state EOS when an opened answer stops making progress."""

    threshold_dbfs: float = -90.0
    required_frames: int = 12
    no_text_required_frames: int = 30
    no_audio_required_frames: int = 30
    function_call_emission_limit: int = 4
    function_call_emissions: int = 0
    agent_open: bool = False
    text_seen: bool = False
    function_seen: bool = False
    audible_seen: bool = False
    silent_frames: int = 0
    no_text_frames: int = 0
    no_audio_frames: int = 0
    request_pending: bool = False
    request_reason: str | None = None

    def reset(self) -> None:
        self.function_call_emissions = 0
        self.agent_open = False
        self.text_seen = False
        self.function_seen = False
        self.audible_seen = False
        self.silent_frames = 0
        self.no_text_frames = 0
        self.no_audio_frames = 0
        self.request_pending = False
        self.request_reason = None

    def observe(
        self,
        control: str | None,
        text_delta: str,
        output_dbfs: float,
        *,
        function_delta: str = "",
    ) -> bool:
        if control == "agent_bos" and not self.agent_open:
            self.agent_open = True
            self.text_seen = False
            self.function_seen = False
            self.audible_seen = False
            self.silent_frames = 0
            self.no_text_frames = 0
            self.no_audio_frames = 0
            self.function_call_emissions = 0
            self.request_pending = False
            self.request_reason = None

        if not self.agent_open:
            return False

        if text_delta:
            self.text_seen = True
        if function_delta:
            self.function_seen = True
            self.function_call_emissions += function_delta.count("<TOOLCALL>")
        if output_dbfs > self.threshold_dbfs:
            self.audible_seen = True
        if function_delta or text_delta:
            self.no_text_frames = 0
        else:
            self.no_text_frames += 1
        if function_delta or output_dbfs > self.threshold_dbfs:
            self.no_audio_frames = 0
        else:
            self.no_audio_frames += 1
        if self.audible_seen and not text_delta and output_dbfs <= self.threshold_dbfs:
            self.silent_frames += 1
        else:
            self.silent_frames = 0

        no_text_stalled = (
            self.no_text_required_frames > 0 and self.no_text_frames >= self.no_text_required_frames
        )
        no_audio_stalled = (
            self.no_audio_required_frames > 0
            and self.no_audio_frames >= self.no_audio_required_frames
        )
        decoded_silence = self.required_frames > 0 and self.silent_frames >= self.required_frames
        function_repetition = (
            self.function_call_emission_limit > 0
            and self.function_call_emissions > self.function_call_emission_limit
        )
        should_request = (
            no_text_stalled or no_audio_stalled or decoded_silence or function_repetition
        ) and not self.request_pending
        if should_request:
            self.request_pending = True
            if function_repetition:
                self.request_reason = "function_repetition_watchdog"
            elif no_text_stalled:
                self.request_reason = "no_text_since_bos_watchdog"
            elif no_audio_stalled:
                self.request_reason = "no_audio_since_bos_watchdog"
            else:
                self.request_reason = "decoded_silence_watchdog"

        if control == "agent_eos":
            self.agent_open = False
            self.text_seen = False
            self.function_seen = False
            self.audible_seen = False
            self.silent_frames = 0
            self.no_text_frames = 0
            self.no_audio_frames = 0
            self.function_call_emissions = 0
            self.request_pending = False
            self.request_reason = None
            return False
        return should_request

    def snapshot(self) -> dict[str, Any]:
        return {
            "threshold_dbfs": self.threshold_dbfs,
            "required_frames": self.required_frames,
            "no_text_required_frames": self.no_text_required_frames,
            "no_audio_required_frames": self.no_audio_required_frames,
            "function_call_emission_limit": self.function_call_emission_limit,
            "function_call_emissions": self.function_call_emissions,
            "agent_open": self.agent_open,
            "text_seen": self.text_seen,
            "function_seen": self.function_seen,
            "audible_seen": self.audible_seen,
            "silent_frames": self.silent_frames,
            "no_text_frames": self.no_text_frames,
            "no_audio_frames": self.no_audio_frames,
            "request_pending": self.request_pending,
            "request_reason": self.request_reason,
        }


@dataclass(frozen=True, slots=True)
class _QualificationModeIdentity:
    """Engine-owned copy of every field sealed by the validator."""

    label: str
    candidate: str
    no_text_watchdog_override_frames: int
    environment_sha256: str


@dataclass
class QualificationAgentSilenceEosWatchdog(AgentSilenceEosWatchdog):
    """Qualification variant selected once; production snapshot stays unchanged."""

    qualification_mode: _QualificationModeIdentity | None = None

    def snapshot(self) -> dict[str, Any]:
        if self.qualification_mode is None:
            raise RuntimeError("qualification watchdog lacks validated identity")
        return {
            **super().snapshot(),
            "qualification_no_text_watchdog_override_frames": (
                self.qualification_mode.no_text_watchdog_override_frames
            ),
            "qualification_mode": self.qualification_mode.label,
            "qualification_candidate": self.qualification_mode.candidate,
        }


@dataclass
class TransportModelGate:
    """Gate only leading silence; preserve NVIDIA's continuous stream thereafter."""

    threshold_dbfs: float = -50.0
    continuous_after_speech: bool = True
    min_active_frames: int = 1
    speech_started: bool = False
    consecutive_active_frames: int = 0
    input_active: bool = False

    def reset(self) -> None:
        self.speech_started = False
        self.consecutive_active_frames = 0
        self.input_active = False

    def force_open(self) -> None:
        """Release the onset gate for an explicit client-owned turn."""
        self.speech_started = True
        self.consecutive_active_frames = max(1, self.min_active_frames)
        self.input_active = True

    def should_advance(self, input_dbfs: float) -> bool:
        above_threshold = input_dbfs >= self.threshold_dbfs
        if above_threshold:
            self.consecutive_active_frames += 1
        else:
            self.consecutive_active_frames = 0
        self.input_active = self.consecutive_active_frames >= max(1, self.min_active_frames)
        if self.input_active:
            self.speech_started = True
        return self.speech_started

    def observe_response_end(self) -> None:
        """Rearm only after the decoded-audio boundary has fully closed."""
        if not self.continuous_after_speech:
            self.speech_started = False

    def snapshot(self, input_dbfs: float, advanced: bool) -> dict[str, Any]:
        return {
            "continuous_after_speech": self.continuous_after_speech,
            "speech_started": self.speech_started,
            "input_active": self.input_active,
            "above_threshold": input_dbfs >= self.threshold_dbfs,
            "consecutive_active_frames": self.consecutive_active_frames,
            "min_active_frames": self.min_active_frames,
            "model_advanced": advanced,
            "speech_gate_dbfs": self.threshold_dbfs,
        }


class GatePrerollEntry(NamedTuple):
    frame_f32: np.ndarray
    frame_stats: dict[str, float | int]
    input_active: bool
    transport_state_snapshot: dict[str, Any]
    source: str
    job_id: str | None
    transport_frame: int


class GatePrerollSelection(NamedTuple):
    entries: list[GatePrerollEntry]
    buffered_frames: int
    skipped_frames: int
    first_active_index: int | None


class GatePrerollBuffer:
    """Retain recent gate-suppressed frames from one input source context."""

    def __init__(self, preroll_frames: int):
        self.maxlen = max(0, int(preroll_frames))
        self._entries: deque[GatePrerollEntry] = deque(maxlen=self.maxlen)

    def append(
        self,
        frame_f32: np.ndarray,
        frame_stats: dict[str, float | int],
        input_active: bool,
        transport_state_snapshot: dict[str, Any],
        source: str,
        job_id: str | None,
        transport_frame: int,
    ) -> None:
        if self._entries:
            previous = self._entries[-1]
            if (previous.source, previous.job_id) != (source, job_id):
                self.clear()
        self._entries.append(
            GatePrerollEntry(
                frame_f32,
                frame_stats,
                input_active,
                transport_state_snapshot,
                source,
                job_id,
                transport_frame,
            )
        )

    def drain(self) -> list[GatePrerollEntry]:
        entries = list(self._entries)
        self._entries.clear()
        return entries

    def drain_onset_suffix(
        self,
        *,
        threshold_dbfs: float,
        context_frames: int,
    ) -> GatePrerollSelection:
        """Retain active onset and bounded context without old silence."""

        entries = list(self._entries)
        self._entries.clear()
        first_active_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if float(entry.frame_stats["rms_dbfs"]) >= float(threshold_dbfs)
            ),
            None,
        )
        if first_active_index is None:
            start = 0
        else:
            start = max(0, first_active_index - max(0, int(context_frames)))
        return GatePrerollSelection(
            entries=entries[start:],
            buffered_frames=len(entries),
            skipped_frames=start,
            first_active_index=first_active_index,
        )

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class AcousticResponseBoundary:
    """Close assistant turns on decoded-audio silence after text EOS."""

    threshold_dbfs: float = -90.0
    required_silent_frames: int = 3
    max_tail_frames: int = 25
    delivery_silence_frames: int = 12
    open: bool = False
    text_seen: bool = False
    acoustic_complete: bool = False
    bos_frame: int | None = None
    eos_frame: int | None = None
    eos_reason: str | None = None
    silent_frames: int = 0
    post_eos_frames: int = 0

    def reset(self) -> None:
        self.open = False
        self.text_seen = False
        self.acoustic_complete = False
        self.bos_frame = None
        self.eos_frame = None
        self.eos_reason = None
        self.silent_frames = 0
        self.post_eos_frames = 0

    def observe(
        self,
        control: str | None,
        output_dbfs: float,
        frame_index: int,
        *,
        text_delta: str = "",
        watchdog_eos_pending: bool = False,
        watchdog_eos_reason: str | None = None,
        max_response_frames: int = 0,
    ) -> dict[str, Any]:
        event: str | None = None
        boundary_reason: str | None = None
        tail_timed_out = False

        if control == "agent_bos" and not self.open:
            self.open = True
            self.text_seen = False
            self.acoustic_complete = False
            self.bos_frame = frame_index
            self.eos_frame = None
            self.eos_reason = None
            self.silent_frames = 0
            self.post_eos_frames = 0
            event = "start"

        if self.open:
            if text_delta:
                self.text_seen = True
            if output_dbfs <= self.threshold_dbfs:
                self.silent_frames += 1
            else:
                self.silent_frames = 0
                # Delivery silence is a reversible acoustic observation, not a
                # model-state boundary. Resume delivery if speech returns after
                # a long prosodic pause before text EOS.
                if self.eos_frame is None:
                    self.acoustic_complete = False

            if (
                self.eos_frame is None
                and self.text_seen
                and self.delivery_silence_frames > 0
                and self.silent_frames >= self.delivery_silence_frames
            ):
                self.acoustic_complete = True

            if control == "agent_eos" and self.eos_frame is None:
                self.eos_frame = frame_index
                self.post_eos_frames = 0
                if watchdog_eos_pending:
                    self.eos_reason = watchdog_eos_reason or "decoded_silence_watchdog"
                elif (
                    self.bos_frame is not None
                    and max_response_frames > 0
                    and frame_index - self.bos_frame >= max_response_frames
                ):
                    self.eos_reason = "max_response_cap"
                else:
                    self.eos_reason = "model_or_turn_taking"
            elif self.eos_frame is not None and frame_index > self.eos_frame:
                self.post_eos_frames += 1

            enough_silence = (
                self.eos_frame is not None
                and self.required_silent_frames > 0
                and self.silent_frames >= self.required_silent_frames
            )
            tail_deadline = (
                self.eos_frame is not None
                and self.max_tail_frames > 0
                and self.post_eos_frames >= self.max_tail_frames
            )
            if enough_silence or tail_deadline:
                event = "end"
                tail_timed_out = tail_deadline and not enough_silence
                boundary_reason = (
                    "post_eos_audio_silence" if enough_silence else "post_eos_tail_deadline"
                )
                snapshot = {
                    "event": event,
                    "phase": "complete",
                    "bos_frame": self.bos_frame,
                    "text_eos_frame": self.eos_frame,
                    "eos_reason": self.eos_reason,
                    "boundary_reason": boundary_reason,
                    "silent_frames": self.silent_frames,
                    "post_eos_frames": self.post_eos_frames,
                    "tail_timed_out": tail_timed_out,
                    "acoustic_complete": self.acoustic_complete,
                    "delivery_silence_frames": self.delivery_silence_frames,
                    "threshold_dbfs": self.threshold_dbfs,
                    "required_silent_frames": self.required_silent_frames,
                    "max_tail_frames": self.max_tail_frames,
                }
                self.reset()
                return snapshot

        phase = "idle"
        if self.open:
            if self.eos_frame is not None:
                phase = "tail_draining"
            elif self.acoustic_complete:
                phase = "internal_drain"
            else:
                phase = "responding"
        return {
            "event": event,
            "phase": phase,
            "bos_frame": self.bos_frame,
            "text_eos_frame": self.eos_frame,
            "eos_reason": self.eos_reason,
            "boundary_reason": boundary_reason,
            "silent_frames": self.silent_frames,
            "post_eos_frames": self.post_eos_frames,
            "tail_timed_out": tail_timed_out,
            "acoustic_complete": self.acoustic_complete,
            "delivery_silence_frames": self.delivery_silence_frames,
            "threshold_dbfs": self.threshold_dbfs,
            "required_silent_frames": self.required_silent_frames,
            "max_tail_frames": self.max_tail_frames,
        }


def clear_stale_agent_eos_latch(pipeline: Any, stream_id: int) -> bool:
    """Consume an EOS edge before an unrelated future FC worker can see it.

    NVIDIA's non-blocking function loop consumes this boolean only while a
    worker is active. Without this guard an ordinary turn ending remains
    latched and kills the next worker immediately after SOTC.
    """
    active = getattr(pipeline, "_fc_async_bg", {})
    if stream_id in active:
        return False
    wrapper = pipeline.s2s_model
    if not getattr(wrapper, "_agent_eos_just_fired", False):
        return False
    wrapper._agent_eos_just_fired = False
    return True


class VoiceChatEngine:
    """Small stateful adapter around NVIDIA's incremental pipeline API."""

    def __init__(
        self,
        pipeline: Any,
        *,
        agent_silence_eos_dbfs: float = -90.0,
        agent_silence_eos_frames: int = 12,
        agent_no_text_frames: int = 30,
        agent_no_audio_frames: int = 30,
        qualification_mode: ValidatedQualificationMode | None = None,
        response_tail_dbfs: float = -60.0,
        response_tail_silence_frames: int = 3,
        response_tail_max_frames: int = 25,
        delivery_silence_frames: int = 12,
    ):
        self.pipeline = pipeline
        self.stream_id = 0
        self.frame_index = 0
        self.audio_frame_index = 0
        self.started = False
        self.system_prompt = ""
        self.assistant_position = 0
        self.function_position = 0
        self.user_text = ""
        self.user_text_prefix = ""
        self.user_text_segment = ""
        self.last_rnnt_decoded_count = 0
        self.vllm_request_position_baseline: dict[str, int] = {}
        qualification_identity: _QualificationModeIdentity | None = None
        if qualification_mode is not None:
            qualification_identity = _QualificationModeIdentity(
                *validated_qualification_mode_fields(qualification_mode)
            )
        self.qualification_mode = qualification_identity
        watchdog_type = (
            QualificationAgentSilenceEosWatchdog
            if qualification_identity is not None
            else AgentSilenceEosWatchdog
        )
        watchdog_kwargs: dict[str, Any] = {
            "threshold_dbfs": agent_silence_eos_dbfs,
            "required_frames": agent_silence_eos_frames,
            "no_text_required_frames": (
                qualification_identity.no_text_watchdog_override_frames
                if qualification_identity is not None
                else agent_no_text_frames
            ),
            "no_audio_required_frames": agent_no_audio_frames,
        }
        if qualification_identity is not None:
            watchdog_kwargs["qualification_mode"] = qualification_identity
        self.agent_silence_watchdog = watchdog_type(**watchdog_kwargs)
        self.response_boundary = AcousticResponseBoundary(
            threshold_dbfs=response_tail_dbfs,
            required_silent_frames=response_tail_silence_frames,
            max_tail_frames=response_tail_max_frames,
            delivery_silence_frames=delivery_silence_frames,
        )
        self.pad_pair_idle_no_buffer_frames = 0
        self.pad_pair_watchdog_emitted = False
        self.pad_pair_trace_errors = 0
        self.external_user_eou_mode = False
        self.cleanup_failure: str | None = None

    def _reset_wrapper_session_state(self) -> None:
        reset_fn = getattr(self.pipeline.s2s_model, "reset_transport_session_state", None)
        if not callable(reset_fn):
            raise RuntimeError(
                "inference wrapper has no reset_transport_session_state(); refusing "
                "to risk carrying wrapper-global turn state across sessions"
            )
        reset_fn()
        reset_prepared = getattr(self.pipeline.s2s_model, "_reset_eartts_prepared_epoch", None)
        if callable(reset_prepared):
            reset_prepared("transport_session_reset")
        tts_model = getattr(getattr(self.pipeline.s2s_model, "model", None), "tts_model", None)
        if hasattr(tts_model, "_reset_on_bos_last_timing_ms"):
            tts_model._reset_on_bos_last_timing_ms = None

    def _stream_request_id(self) -> str:
        request_id_for_stream = getattr(self.pipeline, "_request_id_for_stream", None)
        if not callable(request_id_for_stream):
            raise RuntimeError("pipeline has no request-id mapping seam")
        request_id = request_id_for_stream(self.stream_id)
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("pipeline returned an invalid request ID")
        return request_id

    @staticmethod
    def _first_scalar(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().reshape(-1)
            return value[0].item() if value.numel() else None
        if isinstance(value, (list, tuple)):
            return VoiceChatEngine._first_scalar(value[0]) if value else None
        if isinstance(value, (bool, int, float, str)):
            return value
        return None

    @staticmethod
    def _validated_model_stage_timings(value: Any, frame_position: int) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("model stage timings must be a dictionary")
        required = {
            "schema",
            "measurement",
            "frame_start",
            "frame_end",
            "positions",
            "perception_ms",
            "nano_interface_ms",
            "eartts_total_ms",
            "eartts_reset",
            "eartts_idle_pad_bypass",
            "codec_call_ms",
            "codec_pipeline",
            "named_stage_sum_ms",
            "wrapper_residual_ms",
            "wrapper_total_ms",
        }
        if set(value) != required:
            raise RuntimeError("model stage timings have an unexpected schema")
        if value["schema"] != 2:
            raise RuntimeError("unsupported model stage timing schema")
        if value["measurement"] != "host_intervals_between_existing_cuda_syncs":
            raise RuntimeError("unexpected model stage timing semantics")
        if value["codec_pipeline"] != "waits_for_previous_frame_and_submits_current":
            raise RuntimeError("unexpected model codec timing semantics")
        integer_keys = ("frame_start", "frame_end", "positions")
        integers: dict[str, int] = {}
        for key in integer_keys:
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, int):
                raise RuntimeError(f"model stage timing {key} must be an integer")
            integers[key] = item
        if integers["frame_end"] != frame_position:
            return None
        if (
            integers["positions"] <= 0
            or integers["frame_start"] > integers["frame_end"]
            or integers["positions"] != integers["frame_end"] - integers["frame_start"] + 1
        ):
            raise RuntimeError("model stage timing frame range is inconsistent")
        timing_keys = required - {
            "schema",
            "measurement",
            *integer_keys,
            "eartts_reset",
            "eartts_idle_pad_bypass",
            "codec_pipeline",
        }
        timings: dict[str, float] = {}
        for key in timing_keys:
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise RuntimeError(f"model stage timing {key} must be numeric")
            number = float(item)
            if not math.isfinite(number) or number < 0:
                raise RuntimeError(f"model stage timing {key} must be finite and nonnegative")
            timings[key] = number
        expected_named_sum = (
            timings["perception_ms"]
            + timings["nano_interface_ms"]
            + timings["eartts_total_ms"]
            + timings["codec_call_ms"]
        )
        if not math.isclose(
            timings["named_stage_sum_ms"],
            expected_named_sum,
            rel_tol=0.0,
            abs_tol=0.05,
        ):
            raise RuntimeError("model stage timing named sum is inconsistent")
        if timings["named_stage_sum_ms"] > timings["wrapper_total_ms"] + 0.05:
            raise RuntimeError("model stage timings exceed the wrapper total")
        expected_total = timings["named_stage_sum_ms"] + timings["wrapper_residual_ms"]
        if not math.isclose(
            timings["wrapper_total_ms"],
            expected_total,
            rel_tol=0.0,
            abs_tol=0.05,
        ):
            raise RuntimeError("model stage timing total is inconsistent")
        reset = value["eartts_reset"]
        if reset is not None:
            reset_required = {
                "schema",
                "request_id",
                "reset_count",
                "abort_call_ms",
                "prefill_call_ms",
                "reset_total_ms",
                "eartts_non_reset_remainder_ms",
            }
            if not isinstance(reset, dict) or set(reset) != reset_required:
                raise RuntimeError("EarTTS reset timing has an unexpected schema")
            if reset["schema"] != 1:
                raise RuntimeError("unsupported EarTTS reset timing schema")
            if not isinstance(reset["request_id"], str) or not reset["request_id"]:
                raise RuntimeError("EarTTS reset timing request ID is invalid")
            if (
                isinstance(reset["reset_count"], bool)
                or not isinstance(reset["reset_count"], int)
                or reset["reset_count"] <= 0
            ):
                raise RuntimeError("EarTTS reset timing count is invalid")
            reset_timings: dict[str, float] = {}
            for key in reset_required - {"schema", "request_id", "reset_count"}:
                item = reset[key]
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise RuntimeError(f"EarTTS reset timing {key} must be numeric")
                number = float(item)
                if not math.isfinite(number) or number < 0:
                    raise RuntimeError(f"EarTTS reset timing {key} must be finite and nonnegative")
                reset_timings[key] = number
            if not math.isclose(
                reset_timings["reset_total_ms"],
                reset_timings["abort_call_ms"] + reset_timings["prefill_call_ms"],
                rel_tol=0.0,
                abs_tol=0.05,
            ):
                raise RuntimeError("EarTTS reset timing arithmetic is inconsistent")
            if not math.isclose(
                timings["eartts_total_ms"],
                reset_timings["reset_total_ms"] + reset_timings["eartts_non_reset_remainder_ms"],
                rel_tol=0.0,
                abs_tol=0.05,
            ):
                raise RuntimeError("EarTTS reset timing exceeds the enclosing call")
        bypass = value["eartts_idle_pad_bypass"]
        bypass_required = {
            "schema",
            "enabled",
            "mode",
            "reason",
            "skipped_positions",
            "skipped_frames",
            "cumulative_skipped_positions",
            "recurrent_state",
            "decoder_code",
        }
        if not isinstance(bypass, dict) or set(bypass) != bypass_required:
            raise RuntimeError("EarTTS idle-PAD bypass has an unexpected schema")
        if bypass["schema"] != 1 or not isinstance(bypass["enabled"], bool):
            raise RuntimeError("EarTTS idle-PAD bypass identity is invalid")
        if bypass["mode"] not in {"disabled", "enabled_no_skip", "bypassed"}:
            raise RuntimeError("EarTTS idle-PAD bypass mode is invalid")
        for key in ("skipped_positions", "cumulative_skipped_positions"):
            item = bypass[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise RuntimeError(f"EarTTS idle-PAD bypass {key} is invalid")
        frames = bypass["skipped_frames"]
        if (
            not isinstance(frames, list)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not integers["frame_start"] <= item <= integers["frame_end"]
                for item in frames
            )
            or frames != sorted(set(frames))
            or len(frames) != bypass["skipped_positions"]
            or bypass["skipped_positions"] > integers["positions"]
            or bypass["cumulative_skipped_positions"] < bypass["skipped_positions"]
        ):
            raise RuntimeError("EarTTS idle-PAD bypass frame evidence is invalid")
        skipped = bool(frames)
        expected_mode = (
            "bypassed" if skipped else ("enabled_no_skip" if bypass["enabled"] else "disabled")
        )
        if bypass["mode"] != expected_mode:
            raise RuntimeError("EarTTS idle-PAD bypass mode is inconsistent")
        expected_reason = "ordinary_pcm_agent_idle_effective_pad_fc_quiescent" if skipped else None
        if bypass["reason"] != expected_reason:
            raise RuntimeError("EarTTS idle-PAD bypass reason is inconsistent")
        expected_recurrent = "preserved_input_identity" if skipped else None
        expected_decoder = "codec_silence_tokens_clone" if skipped else None
        if bypass["recurrent_state"] != expected_recurrent:
            raise RuntimeError("EarTTS idle-PAD recurrent semantics are inconsistent")
        if bypass["decoder_code"] != expected_decoder:
            raise RuntimeError("EarTTS idle-PAD decoder semantics are inconsistent")
        if skipped and not bypass["enabled"]:
            raise RuntimeError("EarTTS idle-PAD bypass ran while disabled")
        return dict(value)

    @staticmethod
    def _validated_eartts_prepared_epoch(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        required = {
            "schema",
            "enabled",
            "state",
            "reason",
            "session_epoch",
            "prepare_epoch",
            "request_id",
            "backend_request_id",
            "backend_generation",
            "generated_tokens",
            "armed_client_turn_id",
            "prepare_count",
            "prepare_failure_count",
            "prepared_reuse_count",
            "bos_transition_count",
            "last_bos_event",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise RuntimeError("EarTTS prepared epoch has an unexpected schema")
        if value["schema"] != 2 or value["enabled"] is not True:
            raise RuntimeError("EarTTS prepared epoch identity is invalid")
        if value["state"] not in {"invalid", "clean_unarmed", "clean_armed", "dirty"}:
            raise RuntimeError("EarTTS prepared epoch state is invalid")
        if not isinstance(value["reason"], str) or not value["reason"]:
            raise RuntimeError("EarTTS prepared epoch reason is invalid")
        for key in (
            "session_epoch",
            "prepare_epoch",
            "prepare_count",
            "prepare_failure_count",
            "prepared_reuse_count",
            "bos_transition_count",
        ):
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise RuntimeError(f"EarTTS prepared epoch {key} is invalid")
        for key in ("generated_tokens", "armed_client_turn_id"):
            item = value[key]
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int) or item < 1
            ):
                raise RuntimeError(f"EarTTS prepared epoch {key} is invalid")
        if value["request_id"] is not None and (
            not isinstance(value["request_id"], str) or not value["request_id"]
        ):
            raise RuntimeError("EarTTS prepared epoch request ID is invalid")
        if value["backend_request_id"] is not None and (
            not isinstance(value["backend_request_id"], str) or not value["backend_request_id"]
        ):
            raise RuntimeError("EarTTS prepared epoch backend request ID is invalid")
        backend_generation = value["backend_generation"]
        if backend_generation is not None and (
            isinstance(backend_generation, bool)
            or not isinstance(backend_generation, int)
            or backend_generation < 1
        ):
            raise RuntimeError("EarTTS prepared epoch backend generation is invalid")
        if (value["backend_request_id"] is None) != (backend_generation is None):
            raise RuntimeError("EarTTS prepared epoch backend identity is incomplete")
        armed = value["armed_client_turn_id"]
        if (value["state"] == "clean_armed") != (armed is not None):
            raise RuntimeError("EarTTS prepared epoch arm is inconsistent")
        event = value["last_bos_event"]
        if event is not None:
            event_required = {
                "schema",
                "mode",
                "request_id",
                "backend_request_id",
                "backend_generation",
                "armed_client_turn_id",
                "session_epoch",
                "prepare_epoch",
                "reset_count",
                "transition_count",
                "reuse_count",
                "fallback_reason",
            }
            if not isinstance(event, dict) or set(event) != event_required:
                raise RuntimeError("EarTTS BOS epoch event has an unexpected schema")
            if event["schema"] != 2 or event["mode"] not in {
                "legacy_fallback",
                "prepared_reuse",
            }:
                raise RuntimeError("EarTTS BOS epoch event identity is invalid")
            if not isinstance(event["request_id"], str) or not event["request_id"]:
                raise RuntimeError("EarTTS BOS epoch event request ID is invalid")
            if (
                not isinstance(event["backend_request_id"], str)
                or not event["backend_request_id"]
                or event["backend_request_id"] == event["request_id"]
            ):
                raise RuntimeError("EarTTS BOS epoch backend request ID is invalid")
            if (
                isinstance(event["backend_generation"], bool)
                or not isinstance(event["backend_generation"], int)
                or event["backend_generation"] < 1
            ):
                raise RuntimeError("EarTTS BOS epoch backend generation is invalid")
            for key in (
                "session_epoch",
                "prepare_epoch",
                "reset_count",
                "transition_count",
                "reuse_count",
            ):
                item = event[key]
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise RuntimeError(f"EarTTS BOS epoch event {key} is invalid")
            if event["transition_count"] != value["bos_transition_count"]:
                raise RuntimeError("EarTTS BOS epoch event transition count is stale")
            if event["reuse_count"] != value["prepared_reuse_count"]:
                raise RuntimeError("EarTTS BOS epoch event reuse count is stale")
            if event["session_epoch"] != value["session_epoch"]:
                raise RuntimeError("EarTTS BOS epoch event session epoch is stale")
            if event["prepare_epoch"] != value["prepare_epoch"]:
                raise RuntimeError("EarTTS BOS epoch event prepare epoch is stale")
            if event["mode"] == "prepared_reuse":
                if (
                    isinstance(event["armed_client_turn_id"], bool)
                    or not isinstance(event["armed_client_turn_id"], int)
                    or event["armed_client_turn_id"] < 1
                    or event["fallback_reason"] is not None
                ):
                    raise RuntimeError("EarTTS prepared-reuse event is invalid")
            elif (
                event["armed_client_turn_id"] is not None
                or not isinstance(event["fallback_reason"], str)
                or not event["fallback_reason"]
            ):
                raise RuntimeError("EarTTS legacy-fallback event is invalid")
        return dict(value)

    def _turn_state(self, context: Any, frame_position: int) -> dict[str, Any]:
        if context is None or frame_position < 0:
            return {}
        stt = self.pipeline.s2s_model.model.stt_model
        agent_token_id = int(context.gen_text[0, frame_position].item())
        asr_token_id = int(context.gen_asr_text[0, frame_position].item())
        control_ids = {
            "pad": int(stt.text_pad_id),
            "agent_bos": int(stt.text_bos_id),
            "agent_eos": int(stt.text_eos_id),
            "user_bos": int(stt.user_bos_id),
        }

        def label(token_id: int, *, asr: bool = False) -> str | None:
            if token_id == control_ids["pad"]:
                return "pad"
            if asr and token_id == control_ids["user_bos"]:
                return "user_bos"
            if token_id == control_ids["agent_bos"]:
                return "agent_bos"
            if token_id == control_ids["agent_eos"]:
                return "agent_eos"
            return None

        result: dict[str, Any] = {
            "agent_token_id": agent_token_id,
            "agent_control": label(agent_token_id),
            "asr_token_id": asr_token_id,
            "asr_control": label(asr_token_id, asr=True),
            "control_ids": control_ids,
        }
        stage_timings = self._validated_model_stage_timings(
            getattr(context, "last_step_stage_timings", None), frame_position
        )
        if stage_timings is not None:
            reset_timing = stage_timings.get("eartts_reset")
            if isinstance(reset_timing, dict):
                expected_request_id = str(self.pipeline._request_id_for_stream(self.stream_id))
                if reset_timing["request_id"] != expected_request_id:
                    raise RuntimeError("EarTTS reset stage timing belongs to another request")
            result["model_stage_timings_ms"] = stage_timings
        rnnt = getattr(context, "rnnt_partial_hypotheses", None)
        if isinstance(rnnt, dict):
            result["rnnt"] = {
                key: self._first_scalar(rnnt.get(key))
                for key in (
                    "blank_count",
                    "nonblank_consec",
                    "nonblank_total",
                    "speech_confirmed",
                    "agent_speaking",
                    "first_turn",
                    "user_first_turn",
                    "rolling_density",
                    "forced_bos",
                    "post_eos_fired",
                )
                if key in rnnt
            }
            result["rnnt"]["decoded_token_count"] = len(rnnt.get("y_sequence", []))
            for source_key, result_key in (
                ("_turn_text_tokens", "turn_text_tokens"),
                ("_agent_talking_frames", "agent_talking_frames"),
            ):
                counter = rnnt.get(source_key)
                if isinstance(counter, dict):
                    result["rnnt"][result_key] = self._first_scalar(counter.get(0))
        wrapper = self.pipeline.s2s_model
        fused_eou = getattr(wrapper, "_external_user_eou_blank_fence_last", None)
        if isinstance(fused_eou, dict) and fused_eou.get("frame") == frame_position:
            result["client_eou_blank_fence"] = dict(fused_eou)
        fc_state = getattr(context, "fc_state", None)
        if isinstance(fc_state, dict):
            drain_post_fc_trace = getattr(wrapper, "drain_post_fc_token_trace", None)
            post_fc_token_trace = drain_post_fc_trace() if callable(drain_post_fc_trace) else []
            suppressed_frame = getattr(wrapper, "_pre_eou_function_suppressed_frame", None)
            function_token_id = (
                int(context.gen_function_text[0, frame_position].item())
                if getattr(context, "gen_function_text", None) is not None
                else None
            )
            result["function_calling"] = {
                "active": bool(fc_state.get("active", False)),
                "awaiting_response": bool(fc_state.get("awaiting_response", False)),
                "injecting_response": bool(fc_state.get("injecting_response", False)),
                "forced_tokens": len(fc_state.get("forced_function_tokens") or []),
                "completed_calls": len(fc_state.get("completed_calls") or []),
                "effective_token_id": function_token_id,
                "effective_token_is_pad": function_token_id == control_ids["pad"],
                "eotr_token_id": getattr(wrapper, "_fc_eotr_id", None),
                "awaiting_eotr": bool(fc_state.get("awaiting_eotr", False)),
                "eotr_observed": bool(fc_state.get("eotr_observed", False)),
                "eotr_observed_count": int(fc_state.get("eotr_observed_count", 0)),
                "eotr_observed_frame": fc_state.get("eotr_observed_frame"),
                "eotr_wait_steps": int(fc_state.get("eotr_wait_steps", 0)),
                "eotr_timeout": bool(fc_state.get("eotr_timeout", False)),
                "eotr_unexpected_token_id": fc_state.get("eotr_unexpected_token_id"),
                "eotr_unexpected_token_frame": fc_state.get("eotr_unexpected_token_frame"),
                "eotr_first_post_drain_token_id": fc_state.get("eotr_first_post_drain_token_id"),
                "eotr_first_post_drain_token_frame": fc_state.get(
                    "eotr_first_post_drain_token_frame"
                ),
                "eotr_feedback_token_id": fc_state.get("eotr_feedback_token_id"),
                "eotr_feedback_frame": fc_state.get("eotr_feedback_frame"),
                "eotr_feedback_committed": bool(fc_state.get("eotr_feedback_committed", False)),
                "background_active": self.stream_id in getattr(self.pipeline, "_fc_async_bg", {}),
                "agent_eos_edge_pending": bool(getattr(wrapper, "_agent_eos_just_fired", False)),
                "pre_eou_suppressed_tokens": int(
                    getattr(wrapper, "_pre_eou_function_suppression_count", 0)
                ),
                "pre_eou_last_raw_token_id": getattr(
                    wrapper, "_pre_eou_function_last_raw_token_id", None
                ),
                "pre_eou_suppressed_frame": suppressed_frame,
                "agent_open_suppressed_tokens": int(
                    getattr(wrapper, "_agent_open_function_suppression_count", 0)
                ),
                "agent_open_last_raw_token_id": getattr(
                    wrapper, "_agent_open_function_last_raw_token_id", None
                ),
                "agent_open_suppressed_frame": getattr(
                    wrapper, "_agent_open_function_suppressed_frame", None
                ),
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
            async_heartbeat = fc_async_heartbeat_snapshot(fc_state)
            if async_heartbeat is not None:
                result["function_calling"]["async_heartbeat"] = async_heartbeat
            raw_tool_wait = fc_state.get("tool_wait_evidence")
            if raw_tool_wait is not None:
                tool_wait = _validated_tool_wait_evidence(raw_tool_wait)
                if tool_wait is None:
                    raise RuntimeError("FC async emitted invalid tool-wait evidence")
                result["function_calling"]["tool_wait_evidence"] = tool_wait
            drain_benchmark = getattr(wrapper, "drain_fc_async_benchmark", None)
            benchmark_records = drain_benchmark() if callable(drain_benchmark) else []
            if benchmark_records:
                validated_records = [
                    _validated_fc_async_benchmark_phase(record) for record in benchmark_records
                ]
                if any(record is None for record in validated_records):
                    raise RuntimeError("FC async benchmark emitted an invalid phase record")
                result["function_calling"]["async_benchmark_phases"] = validated_records
            if post_fc_token_trace:
                result["function_calling"]["post_fc_token_trace"] = post_fc_token_trace
        function_logit_trace = getattr(context, "function_logit_trace", None)
        if function_logit_trace is not None:
            result["function_logit_trace"] = function_logit_trace
        perception_cache = getattr(context, "perception_cache", None)
        if perception_cache is not None:
            cache_channel = getattr(perception_cache, "cache_last_channel", None)
            cache_time = getattr(perception_cache, "cache_last_time", None)
            cache_len = getattr(perception_cache, "cache_last_channel_len", None)
            initialized = getattr(perception_cache, "is_initialized", None)
            result["perception_cache"] = {
                "initialized": bool(initialized()) if callable(initialized) else None,
                "cache_last_channel_shape": (
                    list(cache_channel.shape) if cache_channel is not None else None
                ),
                "cache_last_time_shape": (
                    list(cache_time.shape) if cache_time is not None else None
                ),
                "cache_last_channel_len": self._first_scalar(cache_len),
            }
        return result

    def _vllm_request_positions(self) -> dict[str, Any]:
        """Read live request lengths without synchronizing or touching tensors."""
        request_id_fn = getattr(self.pipeline, "_request_id_for_stream", None)
        request_id = (
            str(request_id_fn(self.stream_id)) if callable(request_id_fn) else str(self.stream_id)
        )
        wrapper = self.pipeline.s2s_model
        interfaces = {
            "nano": getattr(wrapper, "model_llm_interface", None),
            "eartts": getattr(
                getattr(getattr(wrapper, "model", None), "tts_model", None),
                "tts_model",
                None,
            ),
        }
        result: dict[str, Any] = {"request_id": request_id}
        for name, interface in interfaces.items():
            engine = getattr(interface, "engine", None)
            requests = getattr(engine, "requests", None)
            state = requests.get(request_id) if hasattr(requests, "get") else None
            generated = getattr(state, "generated_tokens", None)
            count = len(generated) if generated is not None else None
            baseline = self.vllm_request_position_baseline.get(name)
            position = {
                "generated_tokens": count,
                "prefill_baseline": baseline,
                "session_positions": (
                    count - baseline if count is not None and baseline is not None else None
                ),
            }
            backend_generation = int(getattr(state, "backend_generation", 0))
            if backend_generation > 0:
                position.update(
                    backend_request_id=getattr(state, "backend_request_id", None),
                    backend_generation=backend_generation,
                )
            result[name] = position
        return result

    def inject_user_source_position(self, source_token_id: int) -> dict[str, Any]:
        """Advance one hidden direct-text position without perception or delivery.

        The pipeline owns the forward-only model transaction. This adapter
        verifies the server's Nano/EarTTS epoch offset before calling it, then
        advances the transport frame index only after both clocks have moved
        exactly once and preserved that offset. EarTTS intentionally starts a
        new acoustic request epoch on assistant BOS, so absolute prefill-relative
        equality is valid only before the first BOS. Any exception after the
        pipeline call begins is session-fatal to the caller; vLLM continuation
        state cannot be rolled back.
        """
        if not self.started:
            raise RuntimeError("direct source position requires an active session")
        inject = getattr(self.pipeline, "inject_user_source_position", None)
        if not callable(inject):
            raise RuntimeError("streaming pipeline has no direct source-position seam")

        # The ordinary PCM path consumes this one-frame EOS notification before
        # its next model step so it cannot kill an unrelated future FC worker.
        # Direct injection is also a next model step and must perform the same
        # server-owned cleanup. If FC is active the helper deliberately retains
        # the edge and pipeline preflight rejects the direct transaction.
        clear_stale_agent_eos_latch(self.pipeline, self.stream_id)
        before = self._vllm_request_positions()
        before_nano = before["nano"]["session_positions"]
        before_tts = before["eartts"]["session_positions"]
        if before_nano is None or before_tts is None:
            raise RuntimeError("direct source position cannot read model request clocks")
        epoch_offset_before = before_nano - before_tts

        diagnostics = inject(self.stream_id, source_token_id)
        after = self._vllm_request_positions()
        after_nano = after["nano"]["session_positions"]
        after_tts = after["eartts"]["session_positions"]
        if after_nano != before_nano + 1 or after_tts != before_tts + 1:
            raise RuntimeError(
                "direct source position did not advance both model clocks exactly once"
            )
        epoch_offset_after = after_nano - after_tts
        if epoch_offset_after != epoch_offset_before:
            raise RuntimeError(
                "direct source position changed the Nano/EarTTS epoch offset: "
                f"before={epoch_offset_before} after={epoch_offset_after}"
            )
        self.frame_index += 1
        return {
            **diagnostics,
            "nano_session_position_before": before_nano,
            "nano_session_position_after": after_nano,
            "eartts_session_position_before": before_tts,
            "eartts_session_position_after": after_tts,
            "epoch_offset_before": epoch_offset_before,
            "epoch_offset_after": epoch_offset_after,
            # Backward-compatible aliases used by the first-epoch probe.
            "session_position_before": before_nano,
            "session_position_after": after_nano,
            "transport_frame_index_after": self.frame_index,
        }

    def _pad_pair_trace_diagnostics(
        self,
        context: Any,
        frame_position: int,
        turn_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Collect opt-in evidence without allowing diagnostics to break inference."""
        from .runtime_optimizations import enabled

        if not enabled("VOICECHAT_NANO_PAD_PAIR_TRACE"):
            return None
        try:
            return self._collect_pad_pair_trace_diagnostics(context, frame_position, turn_state)
        except Exception as exc:
            self.pad_pair_trace_errors += 1
            LOGGER.exception("Dropped Nano PAD-pair model-step diagnostics")
            return {
                "trace_error": type(exc).__name__,
                "trace_record_errors": self.pad_pair_trace_errors,
            }

    def _collect_pad_pair_trace_diagnostics(
        self,
        context: Any,
        frame_position: int,
        turn_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the enabled scalar snapshot; caller owns exception isolation."""
        from .runtime_optimizations import consume_pad_pair_trace

        request_id_fn = getattr(self.pipeline, "_request_id_for_stream", None)
        request_id = (
            str(request_id_fn(self.stream_id)) if callable(request_id_fn) else str(self.stream_id)
        )
        snapshot = consume_pad_pair_trace(self.pipeline.s2s_model, request_id)
        finalized = snapshot.get("last_effective_tokens")
        effective: dict[str, Any] | None = None
        if context is not None and frame_position >= 0:
            gen_function = getattr(context, "gen_function_text", None)
            if gen_function is not None:
                text_token = int(context.gen_text[0, frame_position].item())
                function_token = int(gen_function[0, frame_position].item())
                scheduler_pad = (finalized or {}).get("pad")
                pad_token = (
                    int(scheduler_pad)
                    if scheduler_pad is not None
                    else int(self.pipeline.s2s_model.model.stt_model.text_pad_id)
                )
                effective = {
                    "frame_idx": frame_position,
                    "text": text_token,
                    "function": function_token,
                    "pad": pad_token,
                    "text_is_pad": text_token == pad_token,
                    "function_is_pad": function_token == pad_token,
                    "both_pad": text_token == pad_token and function_token == pad_token,
                }

        fc = turn_state.get("function_calling") or {}
        fc_idle = not any(
            (
                fc.get("active"),
                fc.get("awaiting_response"),
                fc.get("injecting_response"),
                fc.get("awaiting_eotr"),
                fc.get("forced_tokens"),
                fc.get("background_active"),
            )
        )
        buffered = any(event.get("decision") == "buffered" for event in snapshot["events"])
        watching = bool(
            snapshot.get("enabled") and fc_idle and turn_state.get("agent_control") == "pad"
        )
        watchdog = None
        if watching and not buffered:
            self.pad_pair_idle_no_buffer_frames += 1
            if self.pad_pair_idle_no_buffer_frames >= 25 and not self.pad_pair_watchdog_emitted:
                self.pad_pair_watchdog_emitted = True
                watchdog = {
                    "event": "idle_agent_without_pair_buffer",
                    "consecutive_frames": self.pad_pair_idle_no_buffer_frames,
                    "threshold_frames": 25,
                    "effective_tokens": effective,
                    "scheduler": snapshot.get("state"),
                }
        else:
            self.pad_pair_idle_no_buffer_frames = 0
            self.pad_pair_watchdog_emitted = False
        effective_matches_finalize = None
        if effective is not None and finalized is not None:
            effective_matches_finalize = all(
                effective.get(key) == finalized.get(key)
                for key in ("frame_idx", "text", "function", "pad", "both_pad")
            )
        return {
            "effective_tokens": effective,
            "effective_tokens_match_finalize": effective_matches_finalize,
            "pad_pair": snapshot,
            "idle_no_buffer_frames": self.pad_pair_idle_no_buffer_frames,
            "watchdog": watchdog,
        }

    def start(self, system_prompt: str = "") -> dict[str, Any]:
        if self.cleanup_failure is not None:
            raise RuntimeError(
                f"engine is unavailable after failed session cleanup: {self.cleanup_failure}"
            )
        if self.started:
            raise RuntimeError("session already started")
        transaction_started_at = time.perf_counter()
        timings_ms: dict[str, float] = {}
        phase = "reset"
        cleanup_required = False
        phase_started_at = transaction_started_at
        try:
            self._reset_wrapper_session_state()
            mode_fn = getattr(self.pipeline.s2s_model, "set_external_user_eou_mode", None)
            if self.external_user_eou_mode and not callable(mode_fn):
                raise RuntimeError("inference wrapper has no set_external_user_eou_mode()")
            self.stream_id += 1
            self.frame_index = 0
            self.audio_frame_index = 0
            self.assistant_position = 0
            self.function_position = 0
            self.user_text = ""
            self.user_text_prefix = ""
            self.user_text_segment = ""
            self.last_rnnt_decoded_count = 0
            self.vllm_request_position_baseline = {}
            self.agent_silence_watchdog.reset()
            self.response_boundary.reset()
            self.pad_pair_idle_no_buffer_frames = 0
            self.pad_pair_watchdog_emitted = False
            self.pad_pair_trace_errors = 0
            timings_ms["reset"] = round((time.perf_counter() - phase_started_at) * 1000.0, 3)

            phase = "open_session"
            phase_started_at = time.perf_counter()
            cleanup_required = True
            self.pipeline.open_session()
            timings_ms["open_session"] = round((time.perf_counter() - phase_started_at) * 1000.0, 3)
            self.system_prompt = system_prompt.strip()

            # Prime the two vLLM request streams without consuming an audio
            # frame. EarTTS must receive its prepared speaker/prompt inputs
            # before infer_one_step appends the first text/acoustic frame.
            phase = "prefill"
            phase_started_at = time.perf_counter()
            prefill = getattr(self.pipeline, "prefill_for_new_stream", None)
            if not callable(prefill):
                raise RuntimeError(
                    "streaming pipeline lacks prefill_for_new_stream(); refusing "
                    "to start an unprimed EarTTS request"
                )
            prefill(self.stream_id, self.system_prompt)
            prepared_attest = getattr(
                self.pipeline.s2s_model, "_attest_eartts_prepared_epoch", None
            )
            if callable(prepared_attest):
                prepared_attest(self._stream_request_id())
            timings_ms["prefill"] = round((time.perf_counter() - phase_started_at) * 1000.0, 3)

            # Prefill resets RNNT turn-taking state after creating the context.
            # Activate external EOU only after that reset.
            phase = "finalize"
            phase_started_at = time.perf_counter()
            if self.external_user_eou_mode:
                mode_fn(True)
            prefill_positions = self._vllm_request_positions()
            for name in ("nano", "eartts"):
                count = prefill_positions[name]["generated_tokens"]
                if count is not None:
                    self.vllm_request_position_baseline[name] = count
            self.started = True
            timings_ms["finalize"] = round((time.perf_counter() - phase_started_at) * 1000.0, 3)
            timings_ms["total"] = round((time.perf_counter() - transaction_started_at) * 1000.0, 3)
            return {"schema": 1, "timings_ms": timings_ms}
        except Exception as exc:
            timings_ms[phase] = round((time.perf_counter() - phase_started_at) * 1000.0, 3)
            timings_ms["total"] = round((time.perf_counter() - transaction_started_at) * 1000.0, 3)
            if cleanup_required:
                try:
                    self.pipeline.reset_session()
                except Exception:
                    pass
            self._reset_wrapper_session_state()
            self.started = False
            self.system_prompt = ""
            self.user_text = ""
            self.user_text_prefix = ""
            self.user_text_segment = ""
            self.vllm_request_position_baseline = {}
            self.agent_silence_watchdog.reset()
            self.response_boundary.reset()
            self.last_rnnt_decoded_count = 0
            raise VoiceChatEngineStartFailure(phase, timings_ms) from exc

    def _eartts_function_cycle_quiescent(self) -> bool:
        background = getattr(self.pipeline, "_fc_async_bg", None)
        if isinstance(background, dict) and background:
            return False
        manager = self.pipeline.context_manager
        slot = manager.streamidx2slotidx.get(self.stream_id)
        if slot is None:
            return False
        context = manager.slot_contexts[slot]
        if context is None:
            return False
        fc = getattr(context, "fc_state", None) or {}
        if any(
            (
                fc.get("active"),
                fc.get("awaiting_response"),
                fc.get("injecting_response"),
                fc.get("awaiting_eotr"),
                fc.get("forced_function_tokens"),
            )
        ):
            return False
        if getattr(context, "tool_response_text", None):
            return False
        if getattr(context, "tool_response_queue", None):
            return False
        return True

    def function_cycle_quiescent(self) -> bool:
        """Expose the exact FC ownership predicate used by prepared EarTTS."""

        return self._eartts_function_cycle_quiescent()

    def interrupt_unpublished_function_cycle(self, timeout_seconds: float = 5.0) -> dict[str, Any]:
        """Stop a pre-publication FC worker at a committed client BOU edge."""

        interrupt = getattr(self.pipeline, "interrupt_fc_async", None)
        if not callable(interrupt):
            raise RuntimeError("pipeline has no bounded FC async interrupt hook")
        return interrupt(self.stream_id, timeout_seconds)

    def prepare_eartts_epoch(self) -> dict[str, Any]:
        if not self.started:
            return {"passed": False, "reason": "session_not_started"}
        prepare = getattr(self.pipeline.s2s_model, "_prepare_eartts_prepared_epoch", None)
        if not callable(prepare):
            return {"passed": False, "reason": "prepared_epoch_disabled"}
        return prepare(
            self._stream_request_id(),
            quiescent_check=self._eartts_function_cycle_quiescent,
            site="turn_start_prepare",
        )

    def arm_eartts_epoch(self, client_turn_id: int) -> dict[str, Any]:
        if not self.started:
            return {"passed": False, "reason": "session_not_started"}
        arm = getattr(self.pipeline.s2s_model, "_arm_eartts_prepared_epoch", None)
        if not callable(arm):
            return {"passed": False, "reason": "prepared_epoch_disabled"}
        return arm(
            self._stream_request_id(),
            client_turn_id,
            quiescent_check=self._eartts_function_cycle_quiescent,
        )

    def invalidate_eartts_epoch(self, reason: str) -> None:
        controller = getattr(self.pipeline.s2s_model, "_eartts_prepared_epoch_controller", None)
        invalidate = getattr(controller, "invalidate", None)
        if callable(invalidate):
            invalidate(reason)

    def _eartts_epoch_before_model_step(self) -> dict[str, Any] | None:
        """Snapshot a quiescent host step without racing the FC worker."""

        controller = getattr(self.pipeline.s2s_model, "_eartts_prepared_epoch_controller", None)
        if controller is None:
            return None
        background = getattr(self.pipeline, "_fc_async_bg", None)
        if isinstance(background, dict) and self.stream_id in background:
            return None
        return _eartts_epoch_step_snapshot(self)

    def process(
        self,
        samples: np.ndarray,
        *,
        is_last: bool = False,
        input_active: bool | None = None,
    ) -> StepResult:
        if not self.started:
            self.start()
        if samples.shape != (FRAME_SAMPLES,):
            raise ValueError(f"expected {FRAME_SAMPLES} samples, received {samples.shape}")

        import torch
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import (
            S2SRequestOptions,
        )

        if input_active is not None:
            observe_activity = getattr(
                self.pipeline.s2s_model, "observe_transport_user_activity", None
            )
            if not callable(observe_activity):
                raise RuntimeError(
                    "inference wrapper has no observe_transport_user_activity(); "
                    "refusing to run without the short-utterance EOU backstop"
                )
            observe_activity(input_active, stream_id=self.stream_id)

        # Direct source positions advance the shared model timeline without
        # consuming audio. Keep an independent audio clock so the first real
        # PCM frame still initializes the bufferer, perception cache, RNNT, and
        # first-frame pipeline metadata correctly.
        first_frame = self.audio_frame_index == 0
        frame = Frame(
            samples=torch.from_numpy(np.asarray(samples, dtype=np.float32)),
            stream_id=self.stream_id,
            is_first=first_frame,
            is_last=is_last,
            options=(S2SRequestOptions(system_prompt=self.system_prompt) if first_frame else None),
        )
        eartts_epoch_before = self._eartts_epoch_before_model_step()
        started_at = time.perf_counter()
        clear_stale_agent_eos_latch(self.pipeline, self.stream_id)
        self.pipeline.generate_step([frame])
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        state = self.pipeline.get_or_create_state(self.stream_id)
        if is_last:
            # Exact-public fixed-stage optimizations are opt-in. Draining here
            # is required by the WebSocket step API, which does not call the
            # file-oriented StreamingS2SPipeline finalizer.
            from .runtime_optimizations import finalize_public_fixed_stages

            finalize_public_fixed_stages(self.pipeline.s2s_model, self.stream_id, state)
        audio_tensor = state.audio_buffer.detach().to(torch.float32).cpu()
        audio = audio_tensor.reshape(-1).numpy().copy()
        assistant_text = clean_display_text(state.get_output_text())
        function_text = state.get_output_function_text()
        rnnt_display = clean_display_text(state.get_output_asr_text())
        # Keep ASR-head decoding as a fallback for checkpoints with no RNNT.
        token_data = state.get_token_tensors() if is_last else None
        context = None
        manager = self.pipeline.context_manager
        slot = manager.streamidx2slotidx.get(self.stream_id)
        if slot is not None:
            context = manager.slot_contexts[slot]
        rnnt_decoded_count: int | None = None
        if context is not None:
            rnnt_hypotheses = getattr(context, "rnnt_partial_hypotheses", None)
            if isinstance(rnnt_hypotheses, dict):
                y_sequence = rnnt_hypotheses.get("y_sequence")
                if isinstance(y_sequence, list):
                    rnnt_decoded_count = len(y_sequence)
        fused_eou = getattr(
            self.pipeline.s2s_model,
            "_external_user_eou_blank_fence_last",
            None,
        )
        fused_phrase_reset = bool(
            isinstance(fused_eou, dict) and fused_eou.get("frame") == self.frame_index
        )
        if fused_phrase_reset and rnnt_decoded_count != 0:
            raise RuntimeError("fused client EOU marker did not clear the RNNT hypothesis")
        self._update_user_transcript(
            rnnt_display,
            rnnt_decoded_count,
            suppress_stale_reset_display=fused_phrase_reset,
        )
        if token_data is not None:
            _, asr_tokens, total_frames, _ = token_data
        else:
            asr_tokens = None
            total_frames = 0
            if context is not None:
                asr_tokens = context.gen_asr_text
                total_frames = context.frame_idx
        if (
            getattr(self.pipeline, "ui_asr_source", "asr_head") != "rnnt"
            and not rnnt_display
            and asr_tokens is not None
            and total_frames > 0
        ):
            from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str

            lengths = torch.tensor([total_frames], dtype=torch.long)
            decoded = tokens_to_str(
                asr_tokens[:, :total_frames],
                lengths,
                tokenizer=self.pipeline.s2s_model.tokenizer,
                pad_id=self.pipeline.s2s_model.model.stt_model.text_pad_id,
                eval_text_turn_taking=False,
            )[0]
            displayed = clean_display_text(decoded)
            if displayed:
                self.user_text = displayed
        user_text = self.user_text
        assistant_delta = assistant_text[self.assistant_position :]
        self.assistant_position = len(assistant_text)
        function_delta = function_text[self.function_position :]
        self.function_position = len(function_text)
        state.cleanup_after_response()
        server_step_ms = (time.perf_counter() - started_at) * 1000.0

        turn_state = self._turn_state(context, total_frames - 1)
        prepared_epoch = self._validated_eartts_prepared_epoch(_eartts_epoch_status(self))
        if prepared_epoch is not None:
            turn_state["eartts_prepared_epoch"] = prepared_epoch
        if turn_state.get("agent_control") == "agent_bos" and prepared_epoch is not None:
            if eartts_epoch_before is None:
                raise RuntimeError("EarTTS BOS has no step-local epoch evidence")
            eartts_epoch_after = _eartts_epoch_step_snapshot(self)
            if eartts_epoch_after is None:
                raise RuntimeError("EarTTS BOS has no step-local epoch evidence")
            turn_state["eartts_bos_epoch_step"] = {
                "schema": 1,
                **{
                    f"{key}_before": eartts_epoch_before[key]
                    for key in (
                        "reset_count",
                        "transition_count",
                        "reuse_count",
                        "session_epoch",
                        "prepare_epoch",
                        "request_id",
                        "backend_request_id",
                        "backend_generation",
                    )
                },
                **{
                    f"{key}_after": eartts_epoch_after[key]
                    for key in (
                        "reset_count",
                        "transition_count",
                        "reuse_count",
                        "session_epoch",
                        "prepare_epoch",
                        "request_id",
                        "backend_request_id",
                        "backend_generation",
                    )
                },
            }
        turn_state["vllm_request_positions"] = self._vllm_request_positions()
        trace_diagnostics = self._pad_pair_trace_diagnostics(context, total_frames - 1, turn_state)
        rnnt_state = turn_state.get("rnnt")
        if isinstance(rnnt_state, dict):
            decoded_count = int(rnnt_state.get("decoded_token_count") or 0)
            rnnt_state["emitted_this_frame"] = max(0, decoded_count - self.last_rnnt_decoded_count)
            self.last_rnnt_decoded_count = decoded_count
        result = StepResult(
            audio=audio,
            user_text=user_text,
            assistant_delta=assistant_delta,
            assistant_text=assistant_text,
            inference_ms=inference_ms,
            server_step_ms=server_step_ms,
            frame_index=self.frame_index,
            input_audio=audio_stats(samples),
            output_audio=audio_stats(audio),
            turn_state=turn_state,
            function_delta=function_delta,
            function_text=function_text,
            trace_diagnostics=trace_diagnostics,
        )
        control = result.turn_state.get("agent_control")
        rnnt_metrics = result.turn_state.get("rnnt", {})
        text_tokens = int(rnnt_metrics.get("turn_text_tokens") or 0)
        talking_frames = int(rnnt_metrics.get("agent_talking_frames") or 0)
        tts_ratio_cap = float(
            self.pipeline.s2s_model.model_cfg.get("tts_text_token_ratio_cap", 16.0)
        )
        tts_token_min = int(self.pipeline.s2s_model.model_cfg.get("tts_text_token_min", 5))
        if isinstance(rnnt_metrics, dict):
            rnnt_metrics["tts_ratio_cap"] = tts_ratio_cap
            rnnt_metrics["tts_token_min"] = tts_token_min
            rnnt_metrics["tts_ratio_guard_armed"] = bool(
                tts_ratio_cap > 0 and text_tokens >= tts_token_min
            )
            rnnt_metrics["tts_ratio_remaining_frames"] = (
                max(0.0, tts_ratio_cap * text_tokens - talking_frames)
                if tts_ratio_cap > 0 and text_tokens >= tts_token_min
                else None
            )
        watchdog_eos_pending = self.agent_silence_watchdog.request_pending
        watchdog_eos_reason = self.agent_silence_watchdog.request_reason
        request_eos = self.agent_silence_watchdog.observe(
            control,
            result.assistant_delta,
            float(result.output_audio["rms_dbfs"]),
            function_delta=result.function_delta,
        )
        result.turn_state["agent_silence_watchdog"] = self.agent_silence_watchdog.snapshot()
        result.turn_state["response_boundary"] = self.response_boundary.observe(
            control,
            float(result.output_audio["rms_dbfs"]),
            self.frame_index,
            text_delta=result.assistant_delta,
            watchdog_eos_pending=watchdog_eos_pending,
            watchdog_eos_reason=watchdog_eos_reason,
            max_response_frames=int(
                getattr(self.pipeline.s2s_model, "_max_agent_response_frames", 0)
            ),
        )
        if request_eos:
            request_fn = getattr(self.pipeline.s2s_model, "request_agent_eos", None)
            if not callable(request_fn):
                raise RuntimeError(
                    "agent silence watchdog fired but the inference wrapper has no "
                    "request_agent_eos()"
                )
            request_fn()
        self.frame_index += 1
        self.audio_frame_index += 1
        if is_last:
            self.pipeline.delete_state(self.stream_id)
            self._reset_wrapper_session_state()
            self.started = False
        return result

    def abort(self) -> None:
        cleanup_error: Exception | None = None
        try:
            if self.started:
                background = getattr(self.pipeline, "_fc_async_bg", None)
                if isinstance(background, dict) and self.stream_id in background:
                    # A background worker owns the same Nano/EarTTS requests.
                    # Reset_session first aborts and joins it, then releases the
                    # request/context state without a racy synthetic model tick.
                    self.pipeline.reset_session()
                elif self.frame_index == 0:
                    # Eager prefill created request state but no model position,
                    # codec work, or tool decision exists to drain. Reset it
                    # directly; a synthetic is_last position could trigger an
                    # old-prompt response during reconfiguration.
                    self.pipeline.reset_session()
                else:
                    # Drive the normal end-of-stream path so ordered local codec
                    # pipelines drain instead of retaining one pending frame.
                    self.process(
                        np.zeros(FRAME_SAMPLES, dtype=np.float32),
                        is_last=True,
                        input_active=False,
                    )
        except Exception:
            try:
                self.pipeline.reset_session()
            except Exception as exc:
                cleanup_error = exc
        finally:
            self._reset_wrapper_session_state()
            self.started = False
            self.system_prompt = ""
            self.user_text = ""
            self.user_text_prefix = ""
            self.user_text_segment = ""
            self.vllm_request_position_baseline = {}
            self.agent_silence_watchdog.reset()
            self.response_boundary.reset()
            self.last_rnnt_decoded_count = 0
            if cleanup_error is not None:
                self.cleanup_failure = f"{type(cleanup_error).__name__}: {cleanup_error}"

    def request_user_eou(self) -> None:
        request_fn = getattr(self.pipeline.s2s_model, "request_user_eou", None)
        if not callable(request_fn):
            raise RuntimeError("inference wrapper has no request_user_eou()")
        request_fn()

    def request_user_eou_at_blank_fence(self, target_blank_frames: int) -> None:
        """Arm the post-RNNT fused BOS path used only by tool-free sessions."""

        request_fn = getattr(self.pipeline.s2s_model, "request_user_eou_at_blank_fence", None)
        if not callable(request_fn):
            raise RuntimeError("inference wrapper has no request_user_eou_at_blank_fence()")
        request_fn(target_blank_frames)

    def cancel_user_eou_at_blank_fence(self) -> None:
        cancel_fn = getattr(self.pipeline.s2s_model, "cancel_user_eou_at_blank_fence", None)
        if not callable(cancel_fn):
            raise RuntimeError("inference wrapper has no cancel_user_eou_at_blank_fence()")
        cancel_fn()

    def user_eou_settlement_blank_frames(self) -> int:
        """Return the acoustic blank fence required before an explicit EOU.

        Nano consumes the fused streaming positions incrementally. The pinned
        Speech pipeline uses this same fence before it clears the current RNNT
        phrase/predictor state; live qualification shows that forcing BOS
        earlier can open a formally valid but empty response.
        """
        value = int(self.pipeline.s2s_model.model_cfg.get("nonblank_reset_after_silence", 10))
        if value < 1:
            raise RuntimeError("user EOU settlement blank frames must be positive")
        return value

    @staticmethod
    def rnnt_blank_count(result: StepResult | None) -> int:
        if result is None:
            return 0
        rnnt = result.turn_state.get("rnnt") or {}
        return max(0, int(rnnt.get("blank_count") or 0))

    def perception_frame_index(self) -> int:
        """Return the active stream's independent acoustic position."""

        manager = self.pipeline.context_manager
        slot = manager.streamidx2slotidx.get(self.stream_id)
        if slot is None:
            raise RuntimeError("active stream has no perception context")
        return int(manager.slot_contexts[slot].perception_frame_idx)

    def request_agent_eos(self) -> None:
        request_fn = getattr(self.pipeline.s2s_model, "request_agent_eos", None)
        if not callable(request_fn):
            raise RuntimeError("inference wrapper has no request_agent_eos()")
        request_fn()

    def reset_user_transcript(self) -> None:
        self.user_text = ""
        self.user_text_prefix = ""
        self.user_text_segment = ""
        self.last_rnnt_decoded_count = 0
        state = self.pipeline.get_or_create_state(self.stream_id)
        state.output_asr_text_str = ""
        if hasattr(state, "_last_sent_asr_text"):
            state._last_sent_asr_text = None

    @staticmethod
    def _join_user_transcript(prefix: str, segment: str) -> str:
        prefix = prefix.strip()
        segment = segment.strip()
        if not prefix:
            return segment
        if not segment:
            return prefix
        separator = "" if segment[0] in ".,?!;:" else " "
        return f"{prefix}{separator}{segment}"

    def _update_user_transcript(
        self,
        rnnt_display: str,
        decoded_count: int | None,
        *,
        suppress_stale_reset_display: bool = False,
    ) -> None:
        """Keep one monotonic UI transcript across RNNT hypothesis clears.

        The pinned pipeline clears ``y_sequence`` after 800 ms of blanks so
        its predictor can start a fresh phrase. Client-side Smart Turn can
        intentionally keep the same user turn open across a longer natural
        pause, so that predictor reset is a segment boundary, not a UI turn
        boundary. Preserve the completed segment and let the next hypothesis
        extend it rather than replacing already-emitted words.
        """
        if not getattr(self, "external_user_eou_mode", False):
            # Legacy RNNT endpointing closes a turn after the pipeline's
            # phrase reset. Retain its original per-hypothesis replacement
            # behavior so phrases cannot accumulate across legacy turns.
            if rnnt_display:
                self.user_text = rnnt_display
            return
        if decoded_count == 0 and self.user_text_segment:
            self.user_text_prefix = self._join_user_transcript(
                self.user_text_prefix, self.user_text_segment
            )
            self.user_text_segment = ""
        # A fused blank-fence/BOS position clears y_sequence inside the wrapper,
        # but the streaming state's display string still describes the phrase
        # from immediately before that same-position reset. Promote the segment
        # once and ignore only that marker-bound stale value.
        if rnnt_display and not suppress_stale_reset_display:
            self.user_text_segment = rnnt_display
        self.user_text = self._join_user_transcript(self.user_text_prefix, self.user_text_segment)

    def configure_external_tools(self, handlers: dict[str, Any]) -> None:
        """Authorize exactly the functions advertised by the active client."""
        if self.started:
            raise RuntimeError("tools must be configured before session start")
        self.pipeline.tool_registry = dict(handlers)


def checked_speech_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    entrypoint = root / "examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py"
    if not entrypoint.is_file():
        raise SystemExit(f"Missing upstream streaming entrypoint: {entrypoint}")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"Speech checkout is {commit}; expected {PINNED_COMMIT}")
    reviewed_paths = [
        ("nemo/collections/speechlm2/inference/model_wrappers/model_factory.py"),
        (
            "nemo/collections/speechlm2/inference/model_wrappers/"
            "nemotron_voicechat_inference_wrapper.py"
        ),
        "nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py",
        ("nemo/collections/speechlm2/inference/streaming/state/s2s_context_manager.py"),
        "nemo/collections/speechlm2/inference/vllm/streaming_llm_engine.py",
    ]
    retained_patch = (
        Path(__file__).resolve().parent / "patches/nemotron-voicechat-rnnt-turn-taking.patch"
    )
    if not retained_patch.is_file():
        raise SystemExit(f"Missing retained Speech source patch: {retained_patch}")
    actual_patch = subprocess.check_output(["git", "-C", str(root), "diff", "--", *reviewed_paths])
    expected_patch = retained_patch.read_bytes()
    if actual_patch != expected_patch:
        raise SystemExit(
            "Speech runtime working-tree diff does not match the retained, "
            f"reviewed patch: {retained_patch}"
        )
    return root


def attach_separate_rnnt_runtime(pipeline: Any, checkpoint: Path) -> dict[str, Any]:
    """Attach decoder/joint omitted by the old separate-RNNT source path.

    The pinned source restores the separate ASR checkpoint to initialize the
    perception encoder, but deliberately discards its decoder and joint. The
    realtime wrapper looks only for underscore-prefixed aliases, so RNNT turn
    detection is otherwise silently disabled even when configured as `rnnt`.
    """
    wrapper = pipeline.s2s_model
    stt = wrapper.model.stt_model
    if getattr(stt, "_rnnt_decoder", None) is not None:
        return {"attached": False, "reason": "already_available"}
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Separate RNNT checkpoint is missing: {checkpoint}")

    import nemo.collections.asr as nemo_asr
    import torch

    started_at = time.perf_counter()
    restored = nemo_asr.models.ASRModel.restore_from(str(checkpoint), map_location="cpu")
    decoder = restored.decoder.to(device=wrapper.device, dtype=wrapper.dtype).eval()
    joint = restored.joint.to(device=wrapper.device, dtype=wrapper.dtype).eval()
    tokenizer = restored.tokenizer
    vocabulary = list(getattr(joint, "vocabulary", None) or [])
    blank_id = getattr(getattr(restored, "decoding", None), "blank_id", None)
    if blank_id is None:
        blank_id = getattr(decoder, "blank_idx", len(vocabulary))

    stt.rnnt_decoder = decoder
    stt.rnnt_joint = joint
    stt.rnnt_tokenizer = tokenizer
    object.__setattr__(stt, "_rnnt_decoder", decoder)
    object.__setattr__(stt, "_rnnt_joint", joint)
    object.__setattr__(stt, "_rnnt_blank_id", int(blank_id))

    wrapper._rnnt_is_start_tokens = [token.startswith("\u2581") for token in vocabulary]
    wrapper._rnnt_unk_id = next(
        (index for index, token in enumerate(vocabulary) if token in ("<unk>", "\u2047")),
        None,
    )
    wrapper._rnnt_punct_ids = []
    wrapper._rnnt_punct_ids_set = set()
    wrapper._rnnt_punct_bias_increments = {}
    increments = wrapper.model_cfg.get("rnnt_punct_bias_increments", {}) or {}
    if wrapper.model_cfg.get("rnnt_punct_bias_enabled", False):
        for punctuation in wrapper.model_cfg.get("rnnt_punct_bias_tokens", [".", ",", "?", "!"]):
            for candidate in (punctuation, "\u2581" + punctuation):
                if candidate in vocabulary:
                    token_id = vocabulary.index(candidate)
                    if token_id not in wrapper._rnnt_punct_ids_set:
                        wrapper._rnnt_punct_ids.append(token_id)
                        wrapper._rnnt_punct_ids_set.add(token_id)
                        wrapper._rnnt_punct_bias_increments[token_id] = float(
                            increments.get(punctuation, 1.0)
                        )

    wrapper.model_cfg["rnnt_self_play_suppression"] = True
    wrapper.model_cfg["force_turn_taking"] = True
    pipeline.ui_asr_source = "rnnt"
    del restored
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "attached": True,
        "checkpoint": str(checkpoint),
        "blank_id": int(blank_id),
        "vocabulary_size": len(vocabulary),
        "load_seconds": round(time.perf_counter() - started_at, 3),
    }
    print(f"RNNT runtime attached: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def _install_local_nano_skeleton(skeleton: Path) -> None:
    """Route architecture/tokenizer construction to an exact local skeleton.

    NVIDIA's combined checkpoint deliberately contains all trained VoiceChat
    weights but its config still names the Nano architecture repository. This
    narrow hook changes only that construction-time path and keeps Hub access
    disabled; it does not replace or merge any model weights.
    """
    required = (
        skeleton / "config.json",
        skeleton / "configuration_nemotron_h.py",
        skeleton / "modeling_nemotron_h.py",
        skeleton / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Incomplete local Nano skeleton: {missing}")

    from nemo.collections.speechlm2.inference.model_wrappers import (
        nemotron_voicechat_inference_wrapper as wrapper_module,
    )

    wrapper_class = wrapper_module.NemotronVoicechatInferenceWrapper
    if getattr(wrapper_class, "_voicechat_local_skeleton_installed", False):
        return
    upstream_loader = wrapper_class._load_and_merge_configs

    def load_with_local_skeleton(wrapper: Any) -> Any:
        cfg = upstream_loader(wrapper)
        cfg.model.stt.model.pretrained_llm = str(skeleton)
        cfg.model.speech_generation.model.pretrained_lm_name = str(skeleton)
        return cfg

    wrapper_class._load_and_merge_configs = load_with_local_skeleton
    wrapper_class._voicechat_local_skeleton_installed = True


def _install_step9_capture_sizes() -> None:
    """Inject the qualified capture inventory without mutating audited Speech."""

    raw = os.environ.get("VOICECHAT_STEP9_CAPTURE_SIZES", "").strip()
    if not raw:
        return
    eartts_raw = os.environ.get("VOICECHAT_STEP9_EARTTS_CAPTURE_SIZES", raw).strip()

    def parse_sizes(value: str, name: str) -> list[int]:
        parsed = [int(item) for item in value.split(",")]
        if (
            any(item <= 0 for item in parsed)
            or len(parsed) != len(set(parsed))
            or parsed != sorted(parsed, reverse=True)
        ):
            raise RuntimeError(f"{name} must be unique positive integers in descending order")
        return parsed

    sizes = parse_sizes(raw, "VOICECHAT_STEP9_CAPTURE_SIZES")
    eartts_sizes = parse_sizes(eartts_raw, "VOICECHAT_STEP9_EARTTS_CAPTURE_SIZES")
    from vllm.engine.arg_utils import AsyncEngineArgs

    installed_contract = (sizes, eartts_sizes)
    if getattr(AsyncEngineArgs, "_voicechat_step9_capture_sizes", None) == installed_contract:
        return
    if hasattr(AsyncEngineArgs, "_voicechat_step9_original_init"):
        raise RuntimeError("Step-9 capture sizes were already installed with another inventory")
    original = AsyncEngineArgs.__init__

    @functools.wraps(original)
    def init_with_capture_sizes(instance: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("compilation_config") is not None:
            raise RuntimeError("Step-9 refuses to overwrite an existing compilation_config")
        model = kwargs.get("model", args[0] if args else "")
        component_sizes = eartts_sizes if Path(str(model)).name == "eartts" else sizes
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": component_sizes}
        original(instance, *args, **kwargs)

    AsyncEngineArgs._voicechat_step9_original_init = original
    AsyncEngineArgs._voicechat_step9_capture_sizes = installed_contract
    AsyncEngineArgs.__init__ = init_with_capture_sizes
    print(
        "VOICECHAT_STEP9_CAPTURE_SIZES "
        + json.dumps({"nano": sizes, "eartts": eartts_sizes}, sort_keys=True),
        flush=True,
    )


def build_public_pipeline(args: argparse.Namespace) -> Any:
    """Build NVIDIA's streamer from the exact public combined weights."""
    validate_nano_pad_pair_production_policy(os.environ)
    qualification_mode = validate_qualification_watchdog_production_policy(os.environ)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

    from omegaconf import OmegaConf

    from .audio_compat import install_torchaudio_soundfile_io

    speech_root = checked_speech_root(args.speech_root)
    checkpoint = Path(args.checkpoint_root).expanduser().resolve()
    skeleton = Path(args.hf_skeleton).expanduser().resolve()
    required = (
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "rnnt_tokenizer" / "tokenizer.model",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Incomplete public VoiceChat checkpoint: {missing}")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    rnnt_info = config.get("_rnnt_merge_info") or {}
    if not rnnt_info.get("decoder_config") or not rnnt_info.get("joint_config"):
        raise SystemExit("Public checkpoint lacks embedded RNNT decoder/joint metadata")

    sys.path.insert(0, str(speech_root))
    install_torchaudio_soundfile_io()
    _install_step9_capture_sizes()
    _install_local_nano_skeleton(skeleton)
    from .runtime_optimizations import install_public_wrapper_hooks

    install_public_wrapper_hooks()

    config_path = (
        speech_root / "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml"
    )
    cfg = OmegaConf.load(config_path)
    manifest_path = Path(args.vllm_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in accepted_vllm_manifest_kinds():
        raise SystemExit(f"Unexpected public vLLM manifest kind: {manifest_path}")
    is_hf_release = manifest.get("kind") == "nemotron_voicechat_dgx_spark_hf_release"
    source = manifest.get("parent") if is_hf_release else manifest.get("source")
    source = source or {}
    if (
        source.get("repository") != "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
        or source.get("revision") != args.checkpoint_revision
        or (
            source.get("model_sha256")
            if is_hf_release
            else (source.get("model") or {}).get("sha256")
        )
        != args.checkpoint_sha256
        or manifest.get("derived_from_ea") is not False
    ):
        raise SystemExit("Public vLLM manifest does not match the active checkpoint")
    resolved_artifacts = {}
    for name, requested in (
        ("nano", args.nano_vllm_path),
        ("eartts", args.eartts_vllm_path),
    ):
        root = Path(requested).expanduser().resolve()
        component = (
            release_component_manifest(manifest, name)
            if is_hf_release
            else (manifest.get("components") or {}).get(name) or {}
        )
        validated = validate_manifested_model(root, component)
        if not (component.get("reproducibility") or {}).get("byte_identical"):
            raise SystemExit(f"Public {name} vLLM artifact lacks reproducibility proof")
        resolved_artifacts[name] = {
            "path": str(root),
            "bytes": validated["bytes"],
            "sha256": validated["sha256"],
            "files": validated.get("files"),
            "artifact_files": validated.get("artifact_files"),
            "tensor_count": (component.get("model") or {}).get("inventory", {}).get("tensor_count"),
            "reproducibility_runs": component["reproducibility"].get("runs"),
            "quantization": component.get("quantization"),
            "runtime_optimization": component.get("runtime_optimization"),
        }
    artifact_provenance = {
        "manifest": str(manifest_path),
        "derived_from_ea": False,
        "runtime": source.get("runtime"),
        "runtime_optimization": manifest.get("runtime_optimization"),
        "components": resolved_artifacts,
    }
    production_nano_root = Path(args.nano_vllm_path).expanduser().resolve()
    nano_engine_root, agent_logit_diagnostic = prepare_agent_logit_diagnostic_nano(
        production_nano_root,
        Path("/tmp/voicechat-nano-agent-logit-diagnostic"),
    )
    if agent_logit_diagnostic is not None:
        artifact_provenance["nano_agent_logit_diagnostic"] = agent_logit_diagnostic

    updates = {
        "audio_file": "/dev/null",
        "output_dir": "/tmp/voicechat-web",
        "s2s.model_path": str(checkpoint),
        "s2s.llm_checkpoint_path": str(checkpoint),
        "s2s.speaker_reference": None,
        "s2s.speaker_name": args.speaker_name,
        "s2s.engine_type": PRODUCTION_ENGINE_TYPE,
        "s2s.use_separate_rnnt_ckpt": False,
        "s2s.pretrained_rnnt_asr": None,
        "s2s.turn_taking_source": "rnnt",
        "s2s.ui_asr_source": "rnnt",
        # NVIDIA's public configuration enables this. The environment switch is
        # diagnostic-only and defaults to the published behavior so a run can
        # isolate perception graph replay without changing EarTTS graph mode.
        "s2s.use_perception_cudagraph": env_bool("VOICECHAT_USE_PERCEPTION_CUDAGRAPH", True),
        "s2s.rnnt_eou_frames": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.rnnt_bou_frames": int(os.environ.get("VOICECHAT_RNNT_BOU_FRAMES", "3")),
        "s2s.rnnt_fc_interrupt_ms": int(os.environ.get("VOICECHAT_RNNT_FC_INTERRUPT_MS", "240")),
        "s2s.asr_eou_first_turn": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.transport_vad_eou_frames": int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "10")),
        "s2s.transport_vad_min_rnnt_tokens": int(
            os.environ.get("VOICECHAT_TRANSPORT_VAD_MIN_RNNT_TOKENS", "1")
        ),
        "s2s.system_prompt": "",
        "s2s.max_agent_response_sec": float(
            os.environ.get("VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC", "0")
        ),
        "s2s.tts_text_token_ratio_cap": float(os.environ.get("S2S_TTS_TEXT_TOKEN_RATIO_CAP", "16")),
        "s2s.tts_text_token_min": int(os.environ.get("S2S_TTS_TEXT_TOKEN_MIN", "5")),
        "s2s.fc_fast_tool_grace_ms": fc_fast_tool_grace_ms(),
        "s2s.fc_always_acknowledge_tools": fc_always_acknowledge_tools(),
        "streaming.chunk_size_in_secs": FRAME_SECONDS,
        "streaming.max_len": int(os.environ.get("VOICECHAT_STREAMING_MAX_LEN", "12288")),
    }
    updates.update(
        {
            "s2s.vllm_llm_config.model_path": str(nano_engine_root),
            "s2s.vllm_llm_config.engine_path": str(nano_engine_root),
            "s2s.vllm_llm_config.gpu_memory_utilization": float(
                os.environ.get("VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION", "0.42")
            ),
            "s2s.vllm_llm_config.dtype": "bfloat16",
            "s2s.vllm_llm_config.max_model_len": int(
                os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")
            ),
        }
    )
    updates.update(
        {
            "s2s.vllm_tts_config.model_path": str(
                Path(args.eartts_vllm_path).expanduser().resolve()
            ),
            "s2s.vllm_tts_config.engine_path": str(
                Path(args.eartts_vllm_path).expanduser().resolve()
            ),
            "s2s.vllm_tts_config.gpu_memory_utilization": float(
                os.environ.get("VOICECHAT_VLLM_EARTTS_MEMORY_UTILIZATION", "0.10")
            ),
            "s2s.vllm_tts_config.dtype": "float32",
            "s2s.vllm_tts_config.max_model_len": int(
                os.environ.get("VOICECHAT_VLLM_MAX_MODEL_LEN", "12288")
            ),
            "s2s.vllm_tts_config.enforce_eager": os.environ.get(
                "VOICECHAT_VLLM_EARTTS_ENFORCE_EAGER", "0"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
        }
    )
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.resolve(cfg)

    nano_runtime_optimization = validate_nano_runtime_optimization(production_nano_root, cfg.s2s)
    nano_pad_pair_runtime = validate_nano_pad_pair_runtime(
        production_nano_root,
        manifest,
        env_bool("VOICECHAT_NANO_PAD_PAIR", False),
    )

    import torch
    from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import (
        S2SPipelineBuilder,
    )

    torch.set_float32_matmul_precision(str(cfg.get("matmul_precision", "high")))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = S2SPipelineBuilder.build_pipeline(cfg)
    if nano_runtime_optimization is not None:
        llm_interface = pipeline.s2s_model.model_llm_interface
        interface_sampling = {
            "temperature": float(llm_interface.temperature),
            "top_p": float(llm_interface.top_p),
            "repetition_penalty": float(llm_interface.repetition_penalty),
        }
        if interface_sampling != nano_runtime_optimization["sampling_contract"]:
            raise RuntimeError(
                "Nano wrapper sampling changed after the runtime-optimization gate: "
                f"{interface_sampling}"
            )
        engine_temperature = float(llm_interface.engine.sampling_params.temperature)
        if engine_temperature != 0.0:
            raise RuntimeError(
                "Nano vLLM engine must remain greedy when custom text logits are omitted"
            )
        nano_runtime_optimization["vllm_temperature"] = engine_temperature
    from .runtime_optimizations import prepare_public_fixed_stages

    fixed_stage_optimizations = prepare_public_fixed_stages(pipeline, checkpoint)
    # NVIDIA's wrapper currently overwrites this setting to False during
    # initialization. Restore the requested public-runtime policy afterwards so
    # native BOS cannot self-trigger a new answer during post-turn silence.
    pipeline.s2s_model.model_cfg["rnnt_self_play_suppression"] = True
    stt = pipeline.s2s_model.model.stt_model
    if getattr(stt, "_rnnt_decoder", None) is None or getattr(stt, "_rnnt_joint", None) is None:
        raise RuntimeError("Exact public checkpoint did not initialize its embedded RNNT")
    pipeline.checkpoint_provenance = {
        "kind": "public_checkpoint_production_candidate_1",
        "repository": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        "revision": args.checkpoint_revision,
        "model_sha256": args.checkpoint_sha256,
        "checkpoint": str(checkpoint),
        "nano_skeleton": str(skeleton),
        "speaker": args.speaker_name,
        "turn_taking_source": "rnnt",
        "rnnt_self_play_suppression": True,
        "rnnt_eou_frames": int(pipeline.s2s_model.model_cfg["rnnt_eou_frames"]),
        "rnnt_bou_frames": int(pipeline.s2s_model.model_cfg["rnnt_bou_frames"]),
        "rnnt_fc_interrupt_ms": int(pipeline.s2s_model.model_cfg["rnnt_fc_interrupt_ms"]),
        "ui_asr_source": "rnnt",
        "rnnt_source": "embedded_combined_checkpoint",
        "use_perception_cudagraph": bool(pipeline.s2s_model.model_cfg["use_perception_cudagraph"]),
        "engine_type": PRODUCTION_ENGINE_TYPE,
        "vllm_artifacts": artifact_provenance,
        "nano_runtime_optimization": nano_runtime_optimization,
        "nano_pad_pair_runtime": nano_pad_pair_runtime,
        "fixed_stage_optimizations": fixed_stage_optimizations,
        "host_runtime": host_runtime_provenance(),
    }
    pipeline.qualification_mode = qualification_mode
    print(
        f"Checkpoint provenance: {json.dumps(pipeline.checkpoint_provenance, sort_keys=True)}",
        flush=True,
    )
    return pipeline


def build_pipeline(args: argparse.Namespace) -> Any:
    """Build only the qualified public combined-checkpoint pipeline."""
    return build_public_pipeline(args)


def warm_pipeline(pipeline: Any, warmup_wav: str | None, *, system_prompt: str = "") -> None:
    if not warmup_wav:
        exact_prompt = env_bool("VOICECHAT_STEP9_EXACT_PROMPT_WARMUP", False)
        pipeline.warmup(system_prompt=system_prompt if exact_prompt else None)
        return
    path = Path(warmup_wav).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Warmup WAV does not exist: {path}")
    from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import (
        S2SRequestOptions,
    )

    print(f"Running full streaming warmup with {path}", flush=True)
    started_at = time.perf_counter()
    pipeline.run([str(path)], options=[S2SRequestOptions(system_prompt="")])
    print(
        f"Full streaming warmup finished in {time.perf_counter() - started_at:.2f}s",
        flush=True,
    )


def warm_realtime_engine(
    engine: Any, frame_count: int = 4, system_prompt: str = ""
) -> dict[str, Any]:
    """Exercise the exact WebSocket continuation shape before health-ready.

    Pipeline prefill warmup does not reach Nano's paired continuation path. On
    DGX Spark that path pays a one-time multi-second initialization on its third
    model frame, which must not be charged to the first client session.
    """
    if frame_count < 3:
        raise ValueError("realtime warmup requires at least three model frames")
    started_at = time.perf_counter()
    frame_ms: list[float] = []
    engine.start(system_prompt=system_prompt)
    try:
        for frame_index in range(frame_count):
            result = engine.process(
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                is_last=frame_index == frame_count - 1,
                input_active=True,
            )
            frame_ms.append(float(result.inference_ms))
    finally:
        if engine.started:
            engine.abort()
    details = {
        "frames": frame_count,
        "frame_ms": frame_ms,
        "system_prompt": system_prompt,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    print(f"Realtime continuation warmup: {json.dumps(details)}", flush=True)
    return details


def _probe_tensor_fingerprint(value: Any) -> dict[str, Any] | None:
    """Return a small deterministic fingerprint for diagnostic tensor state."""
    if value is None or not hasattr(value, "detach"):
        return None
    tensor = value.detach().cpu().contiguous()
    # NumPy has no native bfloat16 dtype. Hash an FP32 materialization while
    # preserving the original dtype in the diagnostic metadata.
    payload = tensor.float().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _direct_probe_token_ids(pipeline: Any, text: str, token_form: str) -> list[int]:
    """Build one explicitly named experimental user-token form."""
    tokenizer = pipeline.s2s_model.tokenizer
    text_ids = [int(token) for token in tokenizer.text_to_ids(text)]
    stt = pipeline.s2s_model.model.stt_model
    forms = {
        "text": text_ids,
        "user_bos_text": [int(stt.user_bos_id), *text_ids],
        "text_user_eos": [*text_ids, int(stt.user_eos_id)],
        "user_bos_text_user_eos": [
            int(stt.user_bos_id),
            *text_ids,
            int(stt.user_eos_id),
        ],
    }
    if token_form not in forms:
        raise ValueError(f"unsupported direct probe token form: {token_form}")
    token_ids = forms[token_form]
    if not token_ids:
        raise ValueError("direct probe text produced no source tokens")
    return token_ids


def _write_probe_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(OUTPUT_SAMPLE_RATE)
        wav_file.writeframes(float32_to_pcm16_bytes(samples))


def _probe_state_value(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Materialize bounded diagnostic state for exact before/after comparison."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    tensor = _probe_tensor_fingerprint(value)
    if tensor is not None:
        return {"kind": "tensor", **tensor}
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"kind": "reference", "id": identity}
    seen.add(identity)
    if depth >= 4:
        return {"kind": type(value).__name__, "id": identity}
    if isinstance(value, dict):
        return {
            str(key): _probe_state_value(item, depth=depth + 1, seen=seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            return {"kind": type(value).__name__, "id": identity, "length": len(value)}
        return [_probe_state_value(item, depth=depth + 1, seen=seen) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            "kind": type(value).__name__,
            "id": identity,
            "attributes": {
                str(key): _probe_state_value(item, depth=depth + 1, seen=seen)
                for key, item in sorted(attributes.items())
                if not callable(item)
            },
        }
    return {"kind": type(value).__name__, "id": identity}


def _direct_probe_owner_snapshot(engine: VoiceChatEngine) -> dict[str, Any]:
    """Snapshot the live direct-transaction owners without mutating them."""
    pipeline = engine.pipeline
    wrapper = pipeline.s2s_model
    manager = pipeline.context_manager
    slot = manager.streamidx2slotidx[engine.stream_id]
    context = manager.slot_contexts[slot]
    state = pipeline._state_pool.get(engine.stream_id)
    request_id = pipeline._request_id_for_stream(engine.stream_id)
    nano = pipeline._direct_request_state(wrapper.model_llm_interface, request_id)
    eartts = pipeline._direct_request_state(wrapper.model.tts_model.tts_model, request_id)
    nano_engine = getattr(wrapper.model_llm_interface, "engine", None)
    pad_states = getattr(nano_engine, "_voicechat_pad_pair_states", {})
    pad_state = pad_states.get(request_id) if hasattr(pad_states, "get") else None
    if state is not None and hasattr(state, "is_agent_idle"):
        agent_idle = bool(state.is_agent_idle())
    elif state is not None and hasattr(state, "agent_idle"):
        agent_idle = bool(state.agent_idle)
    else:
        agent_idle = bool(getattr(wrapper, "_agent_idle", True))
    return {
        "engine": {
            "frame_index": engine.frame_index,
            "audio_frame_index": engine.audio_frame_index,
            "assistant_position": engine.assistant_position,
            "function_position": engine.function_position,
            "user_text": engine.user_text,
            "last_rnnt_decoded_count": engine.last_rnnt_decoded_count,
            "positions": engine._vllm_request_positions(),
        },
        "manager": {
            "streamidx2slotidx": dict(manager.streamidx2slotidx),
            "slotidx2streamidx": dict(manager.slotidx2streamidx),
            "slot": slot,
            "context_id": id(context),
        },
        "requests": {"nano": nano, "eartts": eartts},
        "context": {
            "frame_idx": int(context.frame_idx),
            "perception_frame_idx": int(context.perception_frame_idx),
            "gen_text": _probe_state_value(context.gen_text),
            "gen_asr_text": _probe_state_value(context.gen_asr_text),
            "gen_function_text": _probe_state_value(context.gen_function_text),
            "audio_toks_buffer": _probe_state_value(context.audio_toks_buffer),
            "input_embeds_history": _probe_state_value(context.input_embeds_history),
            "dynamic_cache": _probe_state_value(context.dynamic_cache),
            "past_key_values": _probe_state_value(context.past_key_values),
            "code": _probe_state_value(context.code),
            "subword_mask": _probe_state_value(context.subword_mask),
            "perception_cache": _probe_state_value(context.perception_cache),
            "codec_cache": _probe_state_value(context.codec_cache),
            "rnnt_partial_hypotheses": _probe_state_value(context.rnnt_partial_hypotheses),
            "fc_state": _probe_state_value(context.fc_state),
            "tool_response_text": context.tool_response_text,
            "tool_response_queue": _probe_state_value(context.tool_response_queue),
        },
        "wrapper": {
            "last_user_audio_emb": _probe_state_value(
                getattr(wrapper, "_last_user_audio_emb", None)
            ),
            "external_user_eou_requested": bool(
                getattr(wrapper, "_external_user_eou_requested", False)
            ),
            "client_eou_sotc_committed_frame": getattr(
                wrapper, "_client_eou_sotc_committed_frame", None
            ),
            "client_eou_sotc_committed_count": int(
                getattr(wrapper, "_client_eou_sotc_committed_count", 0)
            ),
            "agent_open_function_suppression_count": int(
                getattr(wrapper, "_agent_open_function_suppression_count", 0)
            ),
            "agent_open_function_last_raw_token_id": getattr(
                wrapper, "_agent_open_function_last_raw_token_id", None
            ),
            "agent_open_function_suppressed_frame": getattr(
                wrapper, "_agent_open_function_suppressed_frame", None
            ),
            "post_fc_client_bos_requested": bool(
                getattr(wrapper, "_post_fc_client_bos_requested", False)
            ),
            "post_fc_client_bos_requested_frame": getattr(
                wrapper, "_post_fc_client_bos_requested_frame", None
            ),
            "post_fc_client_bos_forced_frame": getattr(
                wrapper, "_post_fc_client_bos_forced_frame", None
            ),
            "post_fc_client_bos_forced_count": int(
                getattr(wrapper, "_post_fc_client_bos_forced_count", 0)
            ),
            "external_agent_eos_requested": bool(
                getattr(wrapper, "_external_agent_eos_requested", False)
            ),
            "agent_eos_just_fired": bool(getattr(wrapper, "_agent_eos_just_fired", False)),
            "post_tc_bos_exempt": bool(getattr(wrapper, "_post_tc_bos_exempt", False)),
            "redirect_tokens": _probe_state_value(getattr(wrapper, "_redirect_tokens_queue", None)),
            "tts_in_turn_content": int(getattr(wrapper, "_tts_in_turn_content", 0)),
            "tts_in_turn_pads": int(getattr(wrapper, "_tts_in_turn_pads", 0)),
            "agent_idle": agent_idle,
        },
        "state": {
            "id": id(state),
            "output_text_str": getattr(state, "output_text_str", None),
            "output_text_tokens": _probe_state_value(getattr(state, "output_text_tokens", None)),
            "output_asr_text_str": getattr(state, "output_asr_text_str", None),
            "output_asr_text_tokens": _probe_state_value(
                getattr(state, "output_asr_text_tokens", None)
            ),
            "output_function_text_str": getattr(state, "output_function_text_str", None),
            "output_words": _probe_state_value(getattr(state, "output_words", None)),
            "audio_buffer": _probe_state_value(getattr(state, "audio_buffer", None)),
        },
        "pad_state": _probe_state_value(pad_state),
    }


def _probe_content_state(value: Any) -> Any:
    """Fingerprint state content while ignoring per-session object identities."""
    snapshot = _probe_state_value(value)

    def strip_identity(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: strip_identity(child) for key, child in item.items() if key != "id"}
        if isinstance(item, list):
            return [strip_identity(child) for child in item]
        return item

    return strip_identity(snapshot)


def _runtime_source_provenance(engine: VoiceChatEngine) -> dict[str, Any]:
    """Identify the exact image, source, and semantic environment in use."""
    package_root = Path(__file__).resolve().parent
    source_paths = {
        f"src/nemotron_voicechat_runtime/{name}": package_root / name
        for name in (
            "direct_semantic_probe.py",
            "pocket_controller.py",
            "pocket_worker.py",
            "protocol.py",
            "provenance.py",
            "runtime_optimizations.py",
            "semantic_corpus.py",
            "server.py",
            "voice_function_fixture.py",
            "voice_function_probe.py",
        )
    }
    source_paths[
        "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
    ] = package_root / "patches/nemotron-voicechat-rnnt-turn-taking.patch"
    pocket_worker_install = Path(
        "/opt/pocket-tts/lib/python3.12/site-packages/nemotron_voicechat_runtime/pocket_worker.py"
    )
    if pocket_worker_install.is_file():
        source_paths["pocket_environment/pocket_worker.py"] = pocket_worker_install
    wrapper = getattr(engine.pipeline, "s2s_model", None)
    provenance = {
        "runtime_image": os.environ.get("VOICECHAT_RUNTIME_IMAGE"),
        "runtime_image_id": os.environ.get("VOICECHAT_RUNTIME_IMAGE_ID"),
        "speech_commit": PINNED_COMMIT,
        "compute_dtype": str(getattr(wrapper, "dtype", None)),
        "source_sha256": {name: sha256_file(path) for name, path in sorted(source_paths.items())},
        "semantic_environment": {
            name: os.environ.get(name) for name in sorted(PRODUCTION_ENVIRONMENT)
        },
        "nano_pad_pair_policy": {
            name: os.environ.get(name, "0") for name in sorted(NANO_PAD_PAIR_PRODUCTION_POLICY)
        },
        "tool_wait_policy": {
            "fast_tool_grace_ms": fc_fast_tool_grace_ms(),
            "always_acknowledge_tools": fc_always_acknowledge_tools(),
        },
    }
    diagnostics: dict[str, bool] = {}
    if fc_async_heartbeat_enabled():
        diagnostics.update(
            fc_async_heartbeat=True,
            sigusr1_all_thread_dump=True,
        )
    if fc_async_benchmark_enabled():
        diagnostics["fc_async_benchmark"] = True
    if diagnostics:
        provenance["diagnostics"] = diagnostics
    return provenance


def _direct_probe_runtime_provenance(engine: VoiceChatEngine) -> dict[str, Any]:
    """Pin checkpoint identity alongside the reusable runtime provenance."""
    return {
        "checkpoint": dict(getattr(engine.pipeline, "checkpoint_provenance", {}) or {}),
        **_runtime_source_provenance(engine),
    }


def run_direct_position_recurrence_probe(
    engine: VoiceChatEngine,
    *,
    output_dir: Path,
    system_prompt: str,
    positions: int = 4,
) -> dict[str, Any]:
    """Compare direct hidden PAD recurrence with ordinary sequential idle PAD."""
    if positions < 1:
        raise ValueError("recurrence probe positions must be positive")
    if env_bool("VOICECHAT_NANO_PAD_PAIR", False):
        raise RuntimeError("recurrence probe requires sequential Nano PAD mode")
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.external_user_eou_mode = True
    pad_id = int(engine.pipeline.s2s_model.model.stt_model.text_pad_id)

    def live_context() -> Any:
        manager = engine.pipeline.context_manager
        return manager.slot_contexts[manager.streamidx2slotidx[engine.stream_id]]

    direct_records: list[dict[str, Any]] = []
    engine.start(system_prompt=system_prompt)
    try:
        for index in range(positions):
            transaction = engine.inject_user_source_position(pad_id)
            context = live_context()
            direct_records.append(
                {
                    "index": index,
                    "effective_token_id": transaction["effective_token_id"],
                    "code": _probe_tensor_fingerprint(context.code),
                    "codec_cache": _probe_content_state(context.codec_cache),
                    "decoded_audio": transaction["decoded_audio_fingerprint"],
                }
            )
    finally:
        if engine.started:
            engine.abort()

    control_records: list[dict[str, Any]] = []
    engine.start(system_prompt=system_prompt)
    try:
        for index in range(positions):
            result = engine.process(np.zeros(FRAME_SAMPLES, dtype=np.float32), input_active=False)
            context = live_context()
            control_records.append(
                {
                    "index": index,
                    "agent_control": result.turn_state.get("agent_control"),
                    "effective_token_id": int(
                        context.gen_text[0, int(context.frame_idx) - 1].item()
                    ),
                    "code": _probe_tensor_fingerprint(context.code),
                    "codec_cache": _probe_content_state(context.codec_cache),
                    "decoded_audio": _probe_state_value(result.audio),
                }
            )
    finally:
        if engine.started:
            engine.abort()

    comparisons: list[dict[str, Any]] = []
    for direct, control in zip(direct_records, control_records, strict=True):
        direct_audio_sha = (direct["decoded_audio"] or {}).get("sha256")
        control_audio_sha = (control["decoded_audio"] or {}).get("sha256")
        comparison = {
            "index": direct["index"],
            "control_is_effective_pad": (
                control["agent_control"] == "pad" and control["effective_token_id"] == pad_id
            ),
            "recurrent_code_equal": direct["code"] == control["code"],
            "codec_cache_equal": direct["codec_cache"] == control["codec_cache"],
            "decoded_audio_equal": bool(
                direct_audio_sha and control_audio_sha and direct_audio_sha == control_audio_sha
            ),
            "direct": direct,
            "control": control,
        }
        comparison["passed"] = all(
            comparison[key]
            for key in (
                "control_is_effective_pad",
                "recurrent_code_equal",
                "codec_cache_equal",
                "decoded_audio_equal",
            )
        )
        comparisons.append(comparison)
    return {
        "schema": 1,
        "kind": "direct_user_source_position_recurrence_probe",
        "runtime_provenance": _direct_probe_runtime_provenance(engine),
        "positions": positions,
        "comparisons": comparisons,
        "passed": bool(comparisons) and all(item["passed"] for item in comparisons),
    }


def run_direct_position_safety_probe(
    engine: VoiceChatEngine, *, output_dir: Path, system_prompt: str
) -> dict[str, Any]:
    """Exercise real-graph preflight purity and post-Nano fatality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.external_user_eou_mode = True
    wrapper = engine.pipeline.s2s_model
    pad_pair_name = "VOICECHAT_NANO_PAD_PAIR"
    fault_name = "S2S_DIRECT_POSITION_DIAGNOSTIC_FAULT_AFTER_NANO"
    original_pad_pair = os.environ.get(pad_pair_name)
    original_fault = os.environ.get(fault_name)
    cases: list[dict[str, Any]] = []

    def restore_environment() -> None:
        for name, value in (
            (pad_pair_name, original_pad_pair),
            (fault_name, original_fault),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def configure_pending_pad_draft() -> None:
        request_id = engine.pipeline._request_id_for_stream(engine.stream_id)
        nano_engine = wrapper.model_llm_interface.engine
        states = getattr(nano_engine, "_voicechat_pad_pair_states", None)
        if states is None:
            states = {}
            nano_engine._voicechat_pad_pair_states = states
        states[request_id] = {
            "pending": {"diagnostic": True},
            "needs_correction": False,
            "assumed": None,
        }

    reject_cases = (
        (
            "boolean_token",
            lambda: True,
            None,
            TypeError,
            "source_token_id must be an integer",
        ),
        (
            "out_of_vocabulary_token",
            lambda: int(wrapper.model.stt_model.embed_tokens.weight.shape[0]),
            None,
            ValueError,
            "source_token_id outside vocabulary",
        ),
        (
            "pending_user_eou",
            lambda: 12,
            lambda: setattr(wrapper, "_external_user_eou_requested", True),
            RuntimeError,
            "direct position rejected with pending wrapper edge",
        ),
        (
            "agent_response_open",
            lambda: 12,
            lambda: wrapper._set_agent_idle(False, engine.stream_id),
            RuntimeError,
            "direct position rejected while the agent response is open",
        ),
        (
            "pad_pair_enabled",
            lambda: 12,
            lambda: os.environ.__setitem__(pad_pair_name, "1"),
            RuntimeError,
            "direct position prototype requires Nano PAD-pair disabled",
        ),
        (
            "pad_pair_pending_draft",
            lambda: 12,
            configure_pending_pad_draft,
            RuntimeError,
            "direct position rejected with a buffered Nano PAD draft",
        ),
    )
    try:
        for (
            name,
            token_factory,
            configure,
            expected_type,
            expected_error,
        ) in reject_cases:
            restore_environment()
            engine.start(system_prompt=system_prompt)
            try:
                if configure is not None:
                    configure()
                before = _direct_probe_owner_snapshot(engine)
                caught: BaseException | None = None
                try:
                    engine.inject_user_source_position(token_factory())
                except BaseException as exc:  # qualification records exact class/message
                    caught = exc
                after = _direct_probe_owner_snapshot(engine)
                cases.append(
                    {
                        "name": name,
                        "error_type": type(caught).__name__ if caught else None,
                        "error": str(caught) if caught else None,
                        "expected_error_type": expected_type.__name__,
                        "expected_error": expected_error,
                        "rejected_as_expected": bool(
                            isinstance(caught, expected_type) and expected_error in str(caught)
                        ),
                        "owners_unchanged": before == after,
                        "before": before,
                        "after": after,
                        "passed": bool(
                            isinstance(caught, expected_type)
                            and expected_error in str(caught)
                            and before == after
                        ),
                    }
                )
            finally:
                restore_environment()
                if engine.started:
                    if name == "pending_user_eou":
                        wrapper._external_user_eou_requested = False
                    if name == "agent_response_open":
                        wrapper._set_agent_idle(True, engine.stream_id)
                    if name == "pad_pair_pending_draft":
                        request_id = engine.pipeline._request_id_for_stream(engine.stream_id)
                        pad_states = getattr(
                            wrapper.model_llm_interface.engine,
                            "_voicechat_pad_pair_states",
                            {},
                        )
                        pad_states.pop(request_id, None)
                    engine.abort()

        restore_environment()
        engine.start(system_prompt=system_prompt)
        request_id = engine.pipeline._request_id_for_stream(engine.stream_id)
        before = _direct_probe_owner_snapshot(engine)
        os.environ[fault_name] = "1"
        caught = None
        try:
            engine.inject_user_source_position(12)
        except BaseException as exc:
            caught = exc
        after_fault = _direct_probe_owner_snapshot(engine)
        os.environ.pop(fault_name, None)
        nano_delta = (
            after_fault["requests"]["nano"]["generated_tokens"]
            - before["requests"]["nano"]["generated_tokens"]
        )
        eartts_delta = (
            after_fault["requests"]["eartts"]["generated_tokens"]
            - before["requests"]["eartts"]["generated_tokens"]
        )
        engine.abort()
        nano_requests = getattr(wrapper.model_llm_interface.engine, "requests", {})
        eartts_requests = getattr(wrapper.model.tts_model.tts_model.engine, "requests", {})
        requests_removed = request_id not in nano_requests and request_id not in eartts_requests
        fatal_case = {
            "name": "fault_after_nano",
            "error_type": type(caught).__name__ if caught else None,
            "error": str(caught) if caught else None,
            "nano_position_delta": nano_delta,
            "eartts_position_delta": eartts_delta,
            "engine_frame_delta": (
                after_fault["engine"]["frame_index"] - before["engine"]["frame_index"]
            ),
            "owners_changed": before != after_fault,
            "session_aborted": not engine.started,
            "requests_removed": requests_removed,
            "before": before,
            "after_fault": after_fault,
        }
        fatal_case["passed"] = bool(
            caught
            and "fault after Nano" in str(caught)
            and nano_delta == 1
            and eartts_delta == 0
            and fatal_case["engine_frame_delta"] == 0
            and fatal_case["owners_changed"]
            and fatal_case["session_aborted"]
            and requests_removed
        )
        cases.append(fatal_case)
    finally:
        restore_environment()
        if engine.started:
            engine.abort()

    return {
        "schema": 1,
        "kind": "direct_user_source_position_safety_probe",
        "runtime_provenance": _direct_probe_runtime_provenance(engine),
        "cases": cases,
        "passed": bool(cases) and all(case["passed"] for case in cases),
    }


def run_direct_position_probe(
    engine: VoiceChatEngine,
    *,
    text: str,
    token_forms: list[str],
    output_dir: Path,
    system_prompt: str,
    max_response_frames: int = 240,
) -> dict[str, Any]:
    """Run diagnostic-only direct-position sessions on the real model graph.

    This deliberately has no protocol entry point. It is a startup qualification
    mode used before the direct seam is made reachable by a client.
    """
    if os.environ.get("VOICECHAT_NANO_PAD_PAIR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("direct-position probe requires VOICECHAT_NANO_PAD_PAIR=0")
    if max_response_frames < 1:
        raise ValueError("direct-position probe max_response_frames must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = engine.pipeline.s2s_model
    pad_id = int(wrapper.model.stt_model.text_pad_id)
    eartts_resets_on_bos = bool(
        getattr(engine.pipeline, "checkpoint_provenance", {})
        .get("fixed_stage_optimizations", {})
        .get("eartts_reset_on_bos", False)
    )
    cases: list[dict[str, Any]] = []

    for token_form in token_forms:
        token_ids = _direct_probe_token_ids(engine.pipeline, text, token_form)
        engine.external_user_eou_mode = True
        engine.start(system_prompt=system_prompt)
        case: dict[str, Any] = {
            "token_form": token_form,
            "text": text,
            "token_ids": token_ids,
            "prefill_positions": engine._vllm_request_positions(),
            "direct_positions": [],
            "response_frames": [],
        }
        delivered_chunks: list[np.ndarray] = []
        try:
            last_user_audio_before = _probe_tensor_fingerprint(
                getattr(wrapper, "_last_user_audio_emb", None)
            )
            for token_id in token_ids:
                position = engine.inject_user_source_position(token_id)
                case["direct_positions"].append(position)

            direct_context = engine.pipeline.context_manager.slot_contexts[
                engine.pipeline.context_manager.streamidx2slotidx[engine.stream_id]
            ]
            actual_last_source = getattr(wrapper, "_last_user_audio_emb", None)
            import torch

            expected_last_source = (
                wrapper.model.stt_model.embed_tokens(
                    torch.tensor([token_ids[-1]], device=wrapper.device, dtype=torch.long)
                )
                .to(device=wrapper.device, dtype=wrapper.dtype)
                .unsqueeze(1)
            )
            case["last_user_audio_embedding"] = {
                "before": last_user_audio_before,
                "after": _probe_tensor_fingerprint(actual_last_source),
                "expected_last_token": _probe_tensor_fingerprint(expected_last_source),
                "matches_last_token": bool(
                    actual_last_source is not None
                    and torch.equal(actual_last_source, expected_last_source)
                ),
            }
            case["after_direct"] = {
                "engine_frame_idx": engine.frame_index,
                "engine_audio_frame_idx": engine.audio_frame_index,
                "context_frame_idx": int(direct_context.frame_idx),
                "perception_frame_idx": int(direct_context.perception_frame_idx),
                "rnnt_initialized": direct_context.rnnt_partial_hypotheses is not None,
                "positions": engine._vllm_request_positions(),
            }

            eou_positions_before = engine._vllm_request_positions()
            eartts_controller = wrapper.model.tts_model
            eartts_reset_count_before = int(getattr(eartts_controller, "_reset_on_bos_count", 0))
            engine.request_user_eou()
            response_started = False
            response_complete = False
            first_pcm_positions: dict[str, Any] | None = None
            last_assistant_text = ""
            for response_index in range(max_response_frames):
                result = engine.process(
                    np.zeros(FRAME_SAMPLES, dtype=np.float32),
                    input_active=False,
                )
                boundary = result.turn_state.get("response_boundary") or {}
                last_assistant_text = result.assistant_text
                if response_index == 0:
                    context = engine.pipeline.context_manager.slot_contexts[
                        engine.pipeline.context_manager.streamidx2slotidx[engine.stream_id]
                    ]
                    first_pcm_positions = {
                        "positions_before_eou_request": eou_positions_before,
                        "positions_after_first_pcm": engine._vllm_request_positions(),
                        "engine_audio_frame_idx_after": engine.audio_frame_index,
                        "perception_frame_idx_after": int(context.perception_frame_idx),
                        "rnnt_initialized": context.rnnt_partial_hypotheses is not None,
                        "eartts_reset_count_after": int(
                            getattr(eartts_controller, "_reset_on_bos_count", 0)
                        ),
                    }
                deliverable = response_audio_is_deliverable(result.turn_state)
                if deliverable and result.audio.size:
                    delivered_chunks.append(result.audio.copy())
                if boundary.get("event") == "start":
                    response_started = True
                if boundary.get("event") == "end":
                    response_complete = True
                case["response_frames"].append(
                    {
                        "index": response_index,
                        "model_frame": result.frame_index,
                        "agent_control": result.turn_state.get("agent_control"),
                        "assistant_delta": result.assistant_delta,
                        "output_audio": result.output_audio,
                        "audio_deliverable": deliverable,
                        "response_boundary": boundary,
                        "positions": result.turn_state.get("vllm_request_positions"),
                    }
                )
                if response_complete:
                    break

            delivered = (
                np.concatenate(delivered_chunks).astype(np.float32, copy=False)
                if delivered_chunks
                else np.empty(0, dtype=np.float32)
            )
            wav_path = output_dir / f"{token_form}.wav"
            _write_probe_wav(wav_path, delivered)
            case["first_pcm"] = first_pcm_positions
            case["assistant_text"] = clean_display_text(last_assistant_text)
            case["response"] = {
                "started": response_started,
                "complete": response_complete,
                "frames": len(case["response_frames"]),
                "delivered_audio_samples": int(delivered.size),
                "delivered_audio": audio_stats(delivered),
                "wav": str(wav_path),
            }
            eartts_reset_count_after_response = int(
                getattr(eartts_controller, "_reset_on_bos_count", 0)
            )
            case["eartts_bos_reset"] = {
                "enabled": eartts_resets_on_bos,
                "count_before_eou": eartts_reset_count_before,
                "count_after_first_pcm": (
                    first_pcm_positions["eartts_reset_count_after"] if first_pcm_positions else None
                ),
                "count_after_response": eartts_reset_count_after_response,
                "last_request_id": getattr(
                    eartts_controller, "_reset_on_bos_last_request_id", None
                ),
            }
            mid_session_feedback: dict[str, Any] | None = None
            if response_complete:
                context = engine.pipeline.context_manager.slot_contexts[
                    engine.pipeline.context_manager.streamidx2slotidx[engine.stream_id]
                ]
                prior_index = int(context.frame_idx) - 1
                expected_previous = {
                    "agent": int(context.gen_text[0, prior_index].item()),
                    "asr": int(context.gen_asr_text[0, prior_index].item()),
                    "function": (
                        int(context.gen_function_text[0, prior_index].item())
                        if context.gen_function_text is not None
                        else None
                    ),
                }
                followup_ids = _direct_probe_token_ids(engine.pipeline, "Continue.", "text")
                position = engine.inject_user_source_position(followup_ids[0])
                mid_session_feedback = {
                    "source_token_id": followup_ids[0],
                    "expected_previous": expected_previous,
                    "observed_previous": position["previous_feedback"],
                    "agent_previous_is_non_pad": expected_previous["agent"] != pad_id,
                    "exact_previous_feedback": (position["previous_feedback"] == expected_previous),
                    "epoch_offset_before": position["epoch_offset_before"],
                    "epoch_offset_after": position["epoch_offset_after"],
                    "epoch_offset_preserved": (
                        position["epoch_offset_before"] == position["epoch_offset_after"]
                    ),
                    "position": position,
                }
            case["mid_session_feedback"] = mid_session_feedback
            direct = case["direct_positions"]
            case["checks"] = {
                "all_effective_pad": all(
                    int(position["effective_token_id"]) == pad_id for position in direct
                ),
                "all_subword_masks_true": all(
                    position.get("effective_subword_mask") is True for position in direct
                ),
                "all_hidden_audio_discarded": all(
                    int(position["decoded_audio_samples_discarded"]) > 0 for position in direct
                ),
                "all_hidden_clocks_advance_once": all(
                    int(position["nano_position_after"])
                    == int(position["nano_position_before"]) + 1
                    and int(position["eartts_position_after"])
                    == int(position["eartts_position_before"]) + 1
                    for position in direct
                ),
                "all_hidden_clocks_aligned": all(
                    int(position["session_position_before"]) == index
                    and int(position["session_position_after"]) == index + 1
                    for index, position in enumerate(direct)
                ),
                "first_feedback_is_pad": all(
                    value in {None, pad_id} for value in direct[0]["previous_feedback"].values()
                ),
                "perception_held_during_direct": (
                    case["after_direct"]["perception_frame_idx"] == 0
                ),
                "audio_clock_held_during_direct": (
                    case["after_direct"]["engine_audio_frame_idx"] == 0
                ),
                "last_source_embedding_replaced": case["last_user_audio_embedding"][
                    "matches_last_token"
                ],
                "first_pcm_initialized_perception": bool(
                    first_pcm_positions
                    and first_pcm_positions["perception_frame_idx_after"] == 1
                    and first_pcm_positions["engine_audio_frame_idx_after"] == 1
                    and first_pcm_positions["rnnt_initialized"]
                ),
                # Nano remains in the conversational request epoch. The
                # qualified EarTTS runtime deliberately starts a fresh acoustic
                # request epoch on BOS, re-prefills the speaker latent, then
                # consumes BOS as position one. Without that optimization it
                # advances in the shared epoch like Nano.
                "eou_clock_transition_valid": bool(
                    first_pcm_positions
                    and first_pcm_positions["positions_after_first_pcm"]["nano"][
                        "session_positions"
                    ]
                    == len(token_ids) + 1
                    and first_pcm_positions["positions_after_first_pcm"]["eartts"][
                        "session_positions"
                    ]
                    == (1 if eartts_resets_on_bos else len(token_ids) + 1)
                ),
                "eartts_bos_reset_exactly_once": bool(
                    (
                        first_pcm_positions
                        and first_pcm_positions["eartts_reset_count_after"]
                        - eartts_reset_count_before
                        == 1
                        and eartts_reset_count_after_response - eartts_reset_count_before == 1
                    )
                    if eartts_resets_on_bos
                    else eartts_reset_count_after_response == eartts_reset_count_before
                ),
                "response_started": response_started,
                "response_complete": response_complete,
                "assistant_text_nonempty": bool(case["assistant_text"].strip()),
                "response_audio_nonempty": delivered.size > 0,
                "mid_session_feedback_exact": bool(
                    mid_session_feedback and mid_session_feedback["exact_previous_feedback"]
                ),
                "mid_session_feedback_agent_non_pad": bool(
                    mid_session_feedback and mid_session_feedback["agent_previous_is_non_pad"]
                ),
                "mid_session_epoch_offset_preserved": bool(
                    mid_session_feedback and mid_session_feedback["epoch_offset_preserved"]
                ),
            }
            case["passed"] = all(case["checks"].values())
        finally:
            if engine.started:
                engine.abort()
        cases.append(case)

    return {
        "schema": 1,
        "kind": "direct_user_source_position_probe",
        "runtime_provenance": _direct_probe_runtime_provenance(engine),
        "text": text,
        "system_prompt": system_prompt,
        "pad_pair_enabled": False,
        "eartts_resets_on_bos": eartts_resets_on_bos,
        "cases": cases,
        "passed": bool(cases) and all(case.get("passed", False) for case in cases),
    }


async def send_step(
    websocket: Any,
    result: StepResult,
    last_user_text: str,
    trace: SessionTrace,
    transport_frame: int,
    protocol: RealtimeProtocolSession,
    *,
    preroll_replay: bool = False,
    synthetic_control: bool = False,
    suppress_response_output: bool = False,
) -> str:
    step9_fallback = step9_capture_fallback_status()
    output_bytes = b""
    audio_delivered = (
        response_audio_is_deliverable(result.turn_state) and not suppress_response_output
    )
    delivered_output_audio: dict[str, float | int]
    if audio_delivered:
        delivered_output_audio = result.output_audio
    else:
        delivered_output_audio = {
            "samples": 0,
            "rms": 0.0,
            "rms_dbfs": -120.0,
            "peak": 0.0,
        }
    user_text_changed = result.user_text != last_user_text
    if result.audio.size and audio_delivered:
        output_bytes = float32_to_pcm16_bytes(result.audio)
        trace.output_pcm(output_bytes)
    for event in protocol.step_events(
        result,
        output_pcm=output_bytes,
        audio_delivered=audio_delivered,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
        synthetic_control=synthetic_control,
        suppress_response_output=suppress_response_output,
    ):
        await websocket.send_json(event)
    if result.user_text != last_user_text:
        last_user_text = result.user_text
    if result.function_delta and not suppress_response_output:
        await websocket.send_json(
            protocol.diagnostic_event(
                "voicechat.function_text.delta",
                delta=result.function_delta,
                text=result.function_text,
                turn_id=protocol.turn_id,
                response_id=protocol.response_id,
            )
        )
    await websocket.send_json(
        protocol.diagnostic_event(
            "voicechat.metrics",
            frame=result.frame_index,
            transport_frame=transport_frame,
            inference_ms=round(result.inference_ms, 3),
            server_step_ms=round(result.server_step_ms, 3),
            budget_ms=FRAME_SECONDS * 1000,
            over_budget=result.server_step_ms > FRAME_SECONDS * 1000,
            input_audio=result.input_audio,
            output_audio=delivered_output_audio,
            model_output_audio=result.output_audio,
            audio_delivered=audio_delivered,
            turn_state=result.turn_state,
            turn_id=protocol.turn_id,
            response_id=protocol.response_id,
            step9_capture_fallback=step9_fallback,
        )
    )
    trace_fields = {
        "frame": result.frame_index,
        "transport_frame": transport_frame,
        "input_audio": result.input_audio,
        "output_audio": {**delivered_output_audio, "bytes": len(output_bytes)},
        "model_output_audio": result.output_audio,
        "audio_delivered": audio_delivered,
        "inference_ms": round(result.inference_ms, 3),
        "server_step_ms": round(result.server_step_ms, 3),
        "user_text": result.user_text,
        "user_text_changed": user_text_changed,
        "assistant_delta": result.assistant_delta,
        "function_delta": result.function_delta,
        "function_text": result.function_text,
        "turn_state": result.turn_state,
        "preroll_replay": preroll_replay,
        "step9_capture_fallback": step9_fallback,
    }
    if result.trace_diagnostics is not None:
        trace_fields["runtime_diagnostics"] = result.trace_diagnostics
    trace.event(
        "model_step",
        **trace_fields,
    )
    if result.trace_diagnostics and result.trace_diagnostics.get("watchdog"):
        trace.event(
            "nano_pad_pair_watchdog",
            frame=result.frame_index,
            transport_frame=transport_frame,
            **result.trace_diagnostics["watchdog"],
        )
    return last_user_text


async def send_idle_step(
    websocket: Any,
    trace: SessionTrace,
    transport_frame: int,
    input_audio: dict[str, float | int],
    gate_dbfs: float,
    transport_state: dict[str, Any] | None = None,
    protocol: RealtimeProtocolSession | None = None,
) -> None:
    step9_fallback = step9_capture_fallback_status()
    output = {"samples": 0, "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0}
    turn_state = {
        "gate": "leading_silence",
        "speech_gate_dbfs": gate_dbfs,
        "transport": transport_state or {},
    }
    event_fields = {
        "frame": None,
        "transport_frame": transport_frame,
        "inference_ms": 0.0,
        "server_step_ms": 0.0,
        "budget_ms": FRAME_SECONDS * 1000,
        "over_budget": False,
        "input_audio": input_audio,
        "output_audio": output,
        "turn_state": turn_state,
        "step9_capture_fallback": step9_fallback,
    }
    await websocket.send_json(
        protocol.diagnostic_event("voicechat.metrics", **event_fields)
        if protocol is not None
        else {"type": "voicechat.metrics", **event_fields}
    )
    trace.event(
        "idle_step",
        transport_frame=transport_frame,
        input_audio=input_audio,
        output_audio=output,
        turn_state=turn_state,
    )


def session_position_limit_event(model_frames: int, max_model_frames: int) -> dict[str, Any] | None:
    if max_model_frames <= 0 or model_frames < max_model_frames:
        return None
    return {
        "type": "error",
        "error": {
            "code": "session_position_limit",
            "message": (
                f"Continuous model session reached its supported {max_model_frames}-frame limit"
            ),
            "model_frames": model_frames,
            "max_model_frames": max_model_frames,
        },
    }


def create_app(
    engine: VoiceChatEngine,
    trace_dir: Path | None = None,
    speech_gate_dbfs: float = -50.0,
    speech_gate_min_frames: int = 1,
    continuous_after_speech: bool = True,
    default_system_prompt: str = "",
    function_template_path: Path | None = None,
    max_session_model_frames: int = 12_000,
    pocket_worker: PocketWorkerManager | None = None,
    speech_gate_preroll_frames: int = 12,
    speech_gate_onset_context_frames: int = 2,
    client_turn_detection: bool = False,
) -> Any:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    engine.external_user_eou_mode = client_turn_detection

    # With postponed annotations, FastAPI resolves route parameter types from
    # module globals. Keep the heavyweight optional import lazy for unit tests,
    # but publish WebSocket so the upgrade route is recognized correctly.
    globals()["WebSocket"] = WebSocket

    @asynccontextmanager
    async def lifespan(_app):
        if pocket_worker is not None:
            # Loading and one full prewarm synthesis are readiness requirements.
            await pocket_worker.start()
        STARTUP_READY_SENTINEL.write_text("ready\n", encoding="utf-8")
        startup_event("application_ready")
        try:
            yield
        finally:
            STARTUP_READY_SENTINEL.unlink(missing_ok=True)
            if pocket_worker is not None:
                await pocket_worker.stop()

    app = FastAPI(title="Nemotron VoiceChat DGX Spark server", lifespan=lifespan)
    active_client = asyncio.Lock()
    model_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="voicechat-model"
    )
    loaded_at = time.time()
    checkpoint_provenance = dict(getattr(engine.pipeline, "checkpoint_provenance", {}) or {})
    host_provenance = host_runtime_provenance()
    checkpoint_provenance.setdefault("host_runtime", host_provenance)
    checkpoint_provenance["runtime_provenance"] = _runtime_source_provenance(engine)

    async def model_call(function: Any, *args: Any, **kwargs: Any) -> Any:
        """Run synchronous NVIDIA wrappers away from Uvicorn's asyncio loop."""
        loop = asyncio.get_running_loop()
        call = functools.partial(function, *args, **kwargs)
        return await loop.run_in_executor(model_executor, call)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        nano_pad_pair = {"enabled": False, "eligible": False, "requests": {}}
        try:
            from .runtime_optimizations import public_pad_pair_status

            nano_pad_pair = public_pad_pair_status(engine.pipeline.s2s_model)
        except (AttributeError, ImportError):
            pass
        typed_input_health = pocket_worker.health if pocket_worker is not None else {"ready": False}
        cleanup_failure = getattr(engine, "cleanup_failure", None)
        model_ready = cleanup_failure is None
        result = {
            "status": (
                "ready"
                if model_ready and (pocket_worker is None or typed_input_health["ready"])
                else "degraded"
            ),
            "model": "ready" if model_ready else "cleanup_failed",
            "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            "auth_required": False,
            "checkpoint": checkpoint_provenance,
            "host_runtime": host_provenance,
            "driver_version": host_provenance["driver_version"],
            "kernel_version": host_provenance["kernel_version"],
            "nano_pad_pair": nano_pad_pair,
            "step9_capture_fallback": step9_capture_fallback_status(),
            "typed_input": typed_input_health,
            "active_client": active_client.locked(),
            "uptime_seconds": round(time.time() - loaded_at, 1),
        }
        qualification_mode = getattr(engine, "qualification_mode", None)
        if type(qualification_mode) is _QualificationModeIdentity:
            result["qualification_no_text_watchdog_override"] = {
                "environment": QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV,
                "frames": qualification_mode.no_text_watchdog_override_frames,
                "mode_environment": QUALIFICATION_MODE_ENV,
                "mode": qualification_mode.label,
                "candidate": qualification_mode.candidate,
                "environment_sha256": qualification_mode.environment_sha256,
            }
        if cleanup_failure is not None:
            result["model_cleanup_failure"] = cleanup_failure
        diagnostics: dict[str, bool] = {}
        if fc_async_heartbeat_enabled():
            diagnostics.update(
                fc_async_heartbeat=True,
                sigusr1_all_thread_dump=True,
            )
        if fc_async_benchmark_enabled():
            diagnostics["fc_async_benchmark"] = True
        if diagnostics:
            result["diagnostics"] = diagnostics
        return result

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        if getattr(engine, "cleanup_failure", None) is not None:
            failed_id = str(uuid.uuid4())
            failed_protocol = RealtimeProtocolSession(
                failed_id,
                protocol_capabilities(
                    function_call_timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
                    max_session_model_frames=max_session_model_frames,
                ),
            )
            await websocket.send_json(failed_protocol.session_created())
            await websocket.send_json(
                failed_protocol.error(
                    "model_cleanup_failed",
                    "The model session could not be restored after a prior failure",
                    fatal=True,
                )
            )
            await websocket.close(code=1011, reason="model_cleanup_failed")
            return
        if active_client.locked():
            busy_id = str(uuid.uuid4())
            busy_protocol = RealtimeProtocolSession(
                busy_id,
                protocol_capabilities(
                    function_call_timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
                    max_session_model_frames=max_session_model_frames,
                    typed_input=pocket_worker is not None and pocket_worker.ready,
                    client_turn_detection=client_turn_detection,
                ),
            )
            await websocket.send_json(
                busy_protocol.error(
                    "server_busy",
                    "Only one active client is supported",
                    fatal=True,
                )
            )
            await websocket.close(code=1013)
            return

        await active_client.acquire()
        session_id = str(uuid.uuid4())
        capabilities = protocol_capabilities(
            function_call_timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
            max_session_model_frames=max_session_model_frames,
            typed_input=pocket_worker is not None and pocket_worker.ready,
            client_turn_detection=client_turn_detection,
        )
        protocol = RealtimeProtocolSession(session_id, capabilities)
        trace = SessionTrace(trace_dir, session_id)
        pcm_buffer = PcmFrameBuffer()
        gate_preroll = GatePrerollBuffer(speech_gate_preroll_frames)
        last_user_text = ""
        configured = False
        function_output_model_output_negotiated = False
        session_started = False
        close_reason = "disconnect"
        transport_frame = -1
        system_message = default_system_prompt
        system_prompt = default_system_prompt
        session_tools: list[dict[str, Any]] = []
        transport_gate = TransportModelGate(
            threshold_dbfs=speech_gate_dbfs,
            continuous_after_speech=continuous_after_speech,
            min_active_frames=speech_gate_min_frames,
        )
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        model_input_lock = asyncio.Lock()
        active_typed: ActiveTypedInput | None = None
        typed_task: asyncio.Task | None = None
        seen_typed_job_ids: set[str] = set()
        open_client_turn_id: int | None = None
        started_client_turns: dict[int, str | None] = {}
        completed_client_turns: dict[int, str | None] = {}
        function_cycle_call_id: str | None = None
        pending_bos_settlement: UserEouSettlement | None = None
        pending_function_turn_id: str | None = None
        cancelling_unpublished_function_turn_id: str | None = None
        function_cycle_finished = asyncio.Event()
        function_cycle_finished.set()
        terminal_session = asyncio.Event()
        function_call_budget = FunctionCallBudget()
        deferred_client_input = DeferredClientInputQueue()
        last_model_result: StepResult | None = None
        terminal_cleanup_task: asyncio.Task[None] | None = None
        terminal_origin_task: asyncio.Task[Any] | None = None

        def clear_current_task_cancellation() -> None:
            current = asyncio.current_task()
            if current is not None:
                while current.cancelling():
                    current.uncancel()

        async def drain_cancellation_safe(future: asyncio.Future[Any]) -> Any:
            """Drain protected work even when this task is directly cancelled."""

            clear_current_task_cancellation()
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    clear_current_task_cancellation()
            return future.result()

        async def run_session_cleanup(
            *,
            terminal_events: list[dict[str, Any]] | None = None,
            terminal_close_code: int = 1011,
            prepare_terminal_events: (
                Callable[[], Awaitable[list[dict[str, Any]]]] | None
            ) = None,
        ) -> None:
            """Own model teardown and terminal publication independently of callers."""

            nonlocal active_typed, typed_task
            try:
                if active_typed is not None:
                    active_typed.cancel_event.set()
                try:
                    tool_bridge.close()
                except Exception as exc:
                    trace.event(
                        "tool_bridge_cleanup_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if (
                    typed_task is not None
                    and typed_task is not terminal_origin_task
                    and not typed_task.done()
                ):
                    try:
                        await asyncio.wait_for(asyncio.shield(typed_task), timeout=3.0)
                    except TimeoutError:
                        typed_task.cancel()
                        await asyncio.gather(typed_task, return_exceptions=True)
                    except (Exception, asyncio.CancelledError) as exc:
                        trace.event(
                            "typed_task_cleanup_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                if prepare_terminal_events is not None:
                    try:
                        terminal_events = await prepare_terminal_events()
                    except (Exception, asyncio.CancelledError) as exc:
                        trace.event(
                            "terminal_event_preparation_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                abort_call = functools.partial(engine.abort)
                abort_future = loop.run_in_executor(model_executor, abort_call)
                try:
                    await drain_cancellation_safe(abort_future)
                except Exception as exc:
                    trace.event(
                        "model_abort_cleanup_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if terminal_events is not None:
                    try:
                        async with send_lock:
                            for event in terminal_events:
                                await websocket.send_json(event)
                    except (Exception, asyncio.CancelledError) as exc:
                        trace.event(
                            "terminal_publication_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    finally:
                        try:
                            await websocket.close(
                                code=terminal_close_code,
                                reason=close_reason,
                            )
                        except Exception:
                            pass
            finally:
                try:
                    trace.close(close_reason)
                finally:
                    active_client.release()

        def start_terminal_cleanup(
            reason: str,
            *,
            events: list[dict[str, Any]] | None = None,
            close_code: int = 1011,
            prepare_events: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        ) -> asyncio.Task[None]:
            """End input ownership and transfer teardown to one uncancelled task."""

            nonlocal close_reason, terminal_cleanup_task, terminal_origin_task
            if terminal_cleanup_task is not None:
                return terminal_cleanup_task
            close_reason = reason
            terminal_origin_task = asyncio.current_task()
            terminal_session.set()
            trace.event("terminal_session_claimed", reason=reason)
            terminal_cleanup_task = asyncio.create_task(
                run_session_cleanup(
                    terminal_events=events,
                    terminal_close_code=close_code,
                    prepare_terminal_events=prepare_events,
                ),
                name=f"voicechat-terminal-cleanup-{session_id}",
            )
            return terminal_cleanup_task

        async def terminate_model_session(
            reason: str,
            *,
            events: list[dict[str, Any]] | None = None,
            close_code: int = 1011,
            prepare_events: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        ) -> None:
            """Publish a fatal boundary only after cancellation-safe model abort."""

            task = start_terminal_cleanup(
                reason,
                events=events,
                close_code=close_code,
                prepare_events=prepare_events,
            )
            current = asyncio.current_task()
            if current is typed_task and terminal_origin_task is not current:
                trace.event(
                    "terminal_cleanup_wait_skipped",
                    reason=reason,
                    owner_reason=close_reason,
                    task="typed_input",
                )
                return
            await asyncio.shield(task)

        async def await_cleanup_cancellation_safe(task: asyncio.Task[None]) -> None:
            """Wait through endpoint cancellation until transferred cleanup finishes."""

            await drain_cancellation_safe(task)

        def select_gate_preroll(*, reason: str) -> list[GatePrerollEntry]:
            selection = gate_preroll.drain_onset_suffix(
                threshold_dbfs=transport_gate.threshold_dbfs,
                context_frames=speech_gate_onset_context_frames,
            )
            trace.event(
                "speech_gate_preroll_selected",
                reason=reason,
                buffered_frames=selection.buffered_frames,
                replayed_frames=len(selection.entries),
                skipped_frames=selection.skipped_frames,
                first_active_index=selection.first_active_index,
                onset_context_frames=max(0, int(speech_gate_onset_context_frames)),
                threshold_dbfs=transport_gate.threshold_dbfs,
            )
            return selection.entries

        async def start_model_session(prompt: str, *, reason: str) -> None:
            """Prepare one real session before acknowledging model readiness."""

            nonlocal session_started
            if session_started:
                raise RuntimeError("model session already started")
            started_at = time.monotonic()
            trace.event("model_session_start_requested", reason=reason)
            try:
                raw_evidence = await model_call(engine.start, prompt)
            except Exception as exc:
                failure_phase = getattr(exc, "phase", None)
                failure_timings = getattr(exc, "timings_ms", None)
                cleanup_error: str | None = None
                try:
                    await model_call(engine.abort)
                except Exception as cleanup_exc:
                    cleanup_error = type(cleanup_exc).__name__
                trace.event(
                    "model_session_start_failed",
                    reason=reason,
                    phase=failure_phase if isinstance(failure_phase, str) else "unknown",
                    timings_ms=(failure_timings if isinstance(failure_timings, dict) else {}),
                    elapsed_ms=round((time.monotonic() - started_at) * 1000, 3),
                    cleanup_error=cleanup_error,
                )
                raise
            evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
            session_started = True
            trace.event(
                "model_session_start_completed",
                reason=reason,
                start_evidence=evidence,
                elapsed_ms=round((time.monotonic() - started_at) * 1000, 3),
            )

        async def replay_gate_preroll(
            entries: list[GatePrerollEntry],
            *,
            before_send: Callable[[StepResult], Awaitable[None] | None] | None = None,
            synthetic_control: bool = False,
            suppress_response_output: bool = False,
        ) -> StepResult | None:
            """Replay retained entries without changing their transport identity."""

            nonlocal close_reason, last_model_result, last_user_text
            if not model_input_lock.locked():
                raise RuntimeError("gate preroll replay requires the model-input lock")
            last_result: StepResult | None = None
            replay_started_at = time.monotonic()
            trace.event(
                "speech_gate_preroll_started",
                frames=len(entries),
                session_started=session_started,
                transport_frames=[entry.transport_frame for entry in entries],
                sources=[entry.source for entry in entries],
                job_ids=[entry.job_id for entry in entries],
                input_active=[entry.input_active for entry in entries],
            )
            for entry in entries:
                protocol.set_input_source(entry.source, entry.job_id)
                if not session_started:
                    await start_model_session(
                        system_prompt,
                        reason="speech_gate_preroll_fallback",
                    )
                limit_event = session_position_limit_event(
                    engine.frame_index, max_session_model_frames
                )
                if limit_event is not None:
                    trace.event(
                        "session_position_limit",
                        model_frames=engine.frame_index,
                        max_model_frames=max_session_model_frames,
                    )
                    await terminate_model_session(
                        "session_position_limit",
                        events=[
                            protocol.error(
                                "session_position_limit",
                                limit_event["error"]["message"],
                                fatal=True,
                                model_frames=engine.frame_index,
                                max_model_frames=max_session_model_frames,
                            ),
                            *protocol.close_events(
                                "session_position_limit",
                                cumulative_user_text=last_user_text,
                                status="failed",
                            ),
                        ],
                        close_code=1000,
                    )
                    raise SessionTerminated
                result = await model_call(
                    engine.process,
                    entry.frame_f32,
                    input_active=entry.input_active,
                )
                last_result = result
                last_model_result = result
                result.turn_state["transport"] = entry.transport_state_snapshot
                if before_send is not None:
                    before_send(result)
                async with send_lock:
                    last_user_text = await send_step(
                        websocket,
                        result,
                        last_user_text,
                        trace,
                        entry.transport_frame,
                        protocol,
                        preroll_replay=True,
                        synthetic_control=synthetic_control,
                        suppress_response_output=suppress_response_output,
                    )
                boundary = result.turn_state.get("response_boundary") or {}
                if boundary.get("event") == "end":
                    transport_gate.observe_response_end()
            trace.event(
                "speech_gate_preroll_completed",
                frames=len(entries),
                elapsed_ms=round((time.monotonic() - replay_started_at) * 1000, 3),
            )
            return last_result

        def remember_client_turn(
            records: dict[int, str | None], client_turn_id: int, turn_id: str | None
        ) -> None:
            records[client_turn_id] = turn_id
            while len(records) > 16:
                records.pop(next(iter(records)))

        async def process_pcm(
            payload: bytes,
            *,
            source: str,
            job_id: str | None,
            before_first_model_frame: Callable[[], Awaitable[None]] | None = None,
            before_send: Callable[[StepResult], None] | None = None,
            synthetic_control: bool = False,
            suppress_response_output: bool = False,
        ) -> StepResult | None:
            """Feed PCM through the single RNNT-owned model input path."""

            nonlocal transport_frame, session_started, last_user_text, close_reason
            nonlocal last_model_result, pending_bos_settlement, pending_function_turn_id
            nonlocal cancelling_unpublished_function_turn_id
            decoded_packet = pcm16_bytes_to_float32(payload)
            last_result: StepResult | None = None
            trace.input_pcm(payload)
            trace.event(
                "input_packet",
                bytes=len(payload),
                source=source,
                job_id=job_id,
                audio=audio_stats(decoded_packet),
            )
            async with model_input_lock:
                for frame in pcm_buffer.push(payload):
                    if before_first_model_frame is not None:
                        callback, before_first_model_frame = before_first_model_frame, None
                        await callback()
                    protocol.set_input_source(source, job_id)
                    transport_frame += 1
                    frame_stats = audio_stats(frame)
                    advance_model = transport_gate.should_advance(float(frame_stats["rms_dbfs"]))
                    if not advance_model:
                        transport_state = transport_gate.snapshot(
                            float(frame_stats["rms_dbfs"]), False
                        )
                        gate_preroll.append(
                            frame,
                            frame_stats,
                            transport_gate.input_active,
                            transport_state,
                            source,
                            job_id,
                            transport_frame,
                        )
                        async with send_lock:
                            await send_idle_step(
                                websocket,
                                trace,
                                transport_frame,
                                frame_stats,
                                speech_gate_dbfs,
                                transport_state,
                                protocol,
                            )
                        continue
                    if gate_preroll:
                        preroll_entries = select_gate_preroll(reason="transport_gate")
                        replay_result = await replay_gate_preroll(
                            preroll_entries,
                            before_send=before_send,
                            synthetic_control=synthetic_control,
                            suppress_response_output=suppress_response_output,
                        )
                        if replay_result is not None:
                            last_result = replay_result
                        protocol.set_input_source(source, job_id)
                    if not session_started:
                        await start_model_session(
                            system_prompt,
                            reason="model_frame_fallback",
                        )
                    limit_event = session_position_limit_event(
                        engine.frame_index, max_session_model_frames
                    )
                    if limit_event is not None:
                        trace.event(
                            "session_position_limit",
                            model_frames=engine.frame_index,
                            max_model_frames=max_session_model_frames,
                        )
                        await terminate_model_session(
                            "session_position_limit",
                            events=[
                                protocol.error(
                                    "session_position_limit",
                                    limit_event["error"]["message"],
                                    fatal=True,
                                    model_frames=engine.frame_index,
                                    max_model_frames=max_session_model_frames,
                                ),
                                *protocol.close_events(
                                    "session_position_limit",
                                    cumulative_user_text=last_user_text,
                                    status="failed",
                                ),
                            ],
                            close_code=1000,
                        )
                        raise SessionTerminated
                    result = await model_call(
                        engine.process,
                        frame,
                        input_active=transport_gate.input_active,
                    )
                    last_result = result
                    last_model_result = result
                    if (
                        cancelling_unpublished_function_turn_id is not None
                        and function_cycle_call_id is None
                        and engine.function_cycle_quiescent()
                    ):
                        trace.event(
                            "deferred_bos_superseded_by_client_turn",
                            initiating_turn_id=cancelling_unpublished_function_turn_id,
                            responding_turn_id=protocol.turn_id,
                            settlement=(
                                None
                                if pending_bos_settlement is None
                                else pending_bos_settlement.evidence()
                            ),
                        )
                        pending_bos_settlement = None
                        pending_function_turn_id = None
                        cancelling_unpublished_function_turn_id = None
                    if (
                        pending_bos_settlement is not None
                        and function_cycle_call_id is None
                        and (result.turn_state or {}).get("agent_control") == "agent_bos"
                        and protocol.turn_id != pending_function_turn_id
                    ):
                        trace.event(
                            "deferred_bos_superseded_by_interrupt",
                            initiating_turn_id=pending_function_turn_id,
                            responding_turn_id=protocol.turn_id,
                            settlement=pending_bos_settlement.evidence(),
                        )
                        pending_bos_settlement = None
                        pending_function_turn_id = None
                    result.turn_state["transport"] = transport_gate.snapshot(
                        float(frame_stats["rms_dbfs"]), True
                    )
                    if before_send is not None:
                        callback_result = before_send(result)
                        if callback_result is not None:
                            await callback_result
                    async with send_lock:
                        last_user_text = await send_step(
                            websocket,
                            result,
                            last_user_text,
                            trace,
                            transport_frame,
                            protocol,
                            synthetic_control=synthetic_control,
                            suppress_response_output=suppress_response_output,
                        )
                    boundary = result.turn_state.get("response_boundary") or {}
                    if boundary.get("event") == "end":
                        transport_gate.observe_response_end()
            return last_result

        async def settle_user_eou(
            *,
            source: str,
            job_id: str | None,
            fuse_final_bos: bool,
            expected_client_turn_id: int | None = None,
            on_fused_terminal: Callable[[], Awaitable[None]] | None = None,
        ) -> UserEouSettlement:
            """Advance the minimum bounded blank fence before forcing BOS."""

            nonlocal close_reason
            settlement = UserEouSettlement.begin(
                engine,
                last_model_result,
                expected_client_turn_id=expected_client_turn_id,
            )
            latch_armed = False

            async def observe_settlement(result: StepResult) -> None:
                marker = (result.turn_state or {}).get("client_eou_blank_fence")
                if marker is not None:
                    settlement.observe_fused_terminal(engine, result)
                    if on_fused_terminal is not None:
                        await on_fused_terminal()
                else:
                    settlement.observe(engine, result)

            try:
                while settlement.needs_step(engine, last_model_result):
                    if (
                        fuse_final_bos
                        and not latch_armed
                        and settlement.ending_blank_frames == settlement.target_blank_frames - 1
                    ):
                        engine.request_user_eou_at_blank_fence(settlement.target_blank_frames)
                        latch_armed = True
                    await process_pcm(
                        b"\x00\x00" * FRAME_SAMPLES,
                        source=source,
                        job_id=job_id,
                        before_send=observe_settlement,
                    )
            except UserEouSettlementFailure as exc:
                failure_code = exc.code
                failure_message = str(exc)
                failure_evidence = exc.evidence
                cancel_blank_fence = latch_armed and not settlement.fused_terminal_bos
                latch_armed = False

                async def prepare_failure_event() -> list[dict[str, Any]]:
                    if cancel_blank_fence:
                        engine.cancel_user_eou_at_blank_fence()
                    invalidate_epoch = getattr(engine, "invalidate_eartts_epoch", None)
                    if callable(invalidate_epoch):
                        try:
                            await model_call(invalidate_epoch, "settlement_failed")
                        except (Exception, asyncio.CancelledError):
                            pass
                    return [
                        protocol.error(
                            failure_code,
                            failure_message,
                            fatal=True,
                            source=source,
                            job_id=job_id,
                            settlement=failure_evidence,
                        )
                    ]

                await terminate_model_session(
                    failure_code,
                    prepare_events=prepare_failure_event,
                )
                raise SessionTerminated from exc
            finally:
                if latch_armed and not settlement.fused_terminal_bos:
                    engine.cancel_user_eou_at_blank_fence()
            evidence = settlement.evidence()
            trace.event(
                "user_eou_settled",
                source=source,
                job_id=job_id,
                **evidence,
            )
            return settlement

        async def observe_separate_bos_or_defer(
            settlement: UserEouSettlement, result: StepResult
        ) -> None:
            """Validate an immediate BOS or retain the expectation across FC."""

            nonlocal pending_bos_settlement, pending_function_turn_id
            if (result.turn_state or {}).get("agent_control") == "agent_bos":
                if settlement.eartts_transition_count_starting is not None:
                    settlement.observe_separate_terminal(engine, result)
                pending_bos_settlement = None
                return
            if function_cycle_active(result):
                if pending_bos_settlement is not None:
                    raise UserEouSettlementFailure(
                        "overlapping_bos_expectation",
                        "A second terminal BOS expectation overlapped function recovery",
                        settlement.evidence(),
                    )
                settlement.deferred_terminal_bos = True
                pending_bos_settlement = settlement
                pending_function_turn_id = protocol.turn_id
                rnnt = ((result.turn_state or {}).get("rnnt") or {})
                lifecycle_events = protocol.finish_committed_input_turn(
                    result.user_text,
                    decoded_token_count=int(rnnt.get("decoded_token_count") or 0),
                )
                if not lifecycle_events or pending_function_turn_id is None:
                    raise UserEouSettlementFailure(
                        "deferred_input_lifecycle_failure",
                        "Deferred function cycle did not close its committed input turn",
                        settlement.evidence(),
                    )
                async with send_lock:
                    for lifecycle_event in lifecycle_events:
                        await websocket.send_json(lifecycle_event)
                return
            if settlement.eartts_transition_count_starting is None:
                return
            raise UserEouSettlementFailure(
                "missing_terminal_bos",
                "Explicit EOU produced neither BOS nor a function cycle",
                settlement.evidence(),
            )

        async def terminate_pending_bos_expectation(
            *,
            code: str,
            message: str,
            call_id: str,
            client_event_id: str | None = None,
            wait_for_cleanup: bool = True,
        ) -> bool:
            """Transfer a deferred-BOS failure to the common terminal owner."""

            nonlocal pending_bos_settlement
            pending = pending_bos_settlement
            pending_bos_settlement = None
            if pending is None:
                return False
            settlement_evidence = pending.evidence()

            async def prepare_failure_event() -> list[dict[str, Any]]:
                invalidation_error: str | None = None
                invalidate_epoch = getattr(engine, "invalidate_eartts_epoch", None)
                if callable(invalidate_epoch):
                    try:
                        await model_call(invalidate_epoch, code)
                    except (Exception, asyncio.CancelledError) as exc:
                        invalidation_error = type(exc).__name__
                trace.event(
                    "pending_bos_expectation_terminated",
                    code=code,
                    call_id=call_id,
                    epoch_invalidation_error=invalidation_error,
                    settlement=settlement_evidence,
                )
                return [
                    protocol.error(
                        code,
                        message,
                        fatal=True,
                        client_event_id=client_event_id,
                        call_id=call_id,
                        settlement=settlement_evidence,
                        epoch_invalidation_error=invalidation_error,
                    )
                ]

            task = start_terminal_cleanup(
                code,
                prepare_events=prepare_failure_event,
            )
            if wait_for_cleanup:
                await asyncio.shield(task)
            return True

        async def cancel_response_to_terminal(reason: str) -> None:
            """Drive model and acoustic response state to one terminal edge."""
            nonlocal close_reason
            if not protocol.response_open:
                return
            protocol.cancel_active_response(reason)
            engine.request_agent_eos()
            max_steps = int(engine.response_boundary.max_tail_frames) + 2
            for _ in range(max_steps):
                await process_pcm(
                    b"\x00\x00" * FRAME_SAMPLES,
                    source="microphone",
                    job_id=None,
                    synthetic_control=True,
                    suppress_response_output=True,
                )
                if not protocol.response_open:
                    return
            await terminate_model_session(
                "response_cancel_timeout",
                events=[
                    protocol.error(
                        "response_cancel_timeout",
                        "Cancelled response did not reach its acoustic terminal",
                        fatal=True,
                    )
                ],
            )
            raise SessionTerminated

        async def recover_pre_eou_activity(context: str) -> dict[str, Any] | None:
            """Settle model self-start before an explicit client EOU recoverably.

            The strict settlement contract treats effective model activity at
            the client's commit edge as unattributable and fatal. A live model
            may legitimately self-start inside an open client turn, so drive
            that activity to its terminal with the existing bounded machinery
            first; the settlement's fail-closed rejection remains the backstop
            if recovery leaves activity effective.
            """
            result = last_model_result
            active_function = function_cycle_active(result) if result is not None else False
            response_open_flag = protocol.response_open
            if not active_function and not response_open_flag:
                return None
            evidence: dict[str, Any] = {
                "context": context,
                "response_open": response_open_flag,
                "function_cycle_effective": active_function,
            }
            trace.event("pre_eou_activity_recovery", **evidence)
            if active_function:
                interrupt = getattr(engine, "interrupt_unpublished_function_cycle", None)
                if callable(interrupt):
                    evidence["function_interrupt"] = await model_call(interrupt)
            turn_state = (result.turn_state or {}) if result is not None else {}
            model_quiescent = (
                turn_state.get("agent_control") == "pad"
                and not response_audio_is_deliverable(turn_state)
                and not function_cycle_active(result)
            )
            if response_open_flag and model_quiescent:
                # The model self-started and then self-interrupted back to
                # listening; only the protocol response is dangling. Close it
                # administratively without fabricating output and let the
                # explicit user EOU settle normally.
                closure_events = protocol.fail_active_response("pre_eou_selfstart_reset")
                async with send_lock:
                    for closure_event in closure_events:
                        await websocket.send_json(closure_event)
                evidence["dangling_response_closed"] = True
                trace.event("pre_eou_activity_recovered", **evidence)
                return evidence
            # Drain the self-started response to its natural terminal first: a
            # freshly opened response cannot reach the cancelled-acoustic
            # terminal, and finishing the utterance is the correct full-duplex
            # behavior. Escalate to the bounded cancel only if the drain
            # exhausts its frame budget.
            drain_budget = int(engine.response_boundary.max_tail_frames) + 400
            drained_frames = 0
            while protocol.response_open and drained_frames < drain_budget:
                await process_pcm(
                    b"\x00\x00" * FRAME_SAMPLES,
                    source="microphone",
                    job_id=None,
                    synthetic_control=True,
                )
                drained_frames += 1
            evidence["drained_frames"] = drained_frames
            if protocol.response_open:
                await cancel_response_to_terminal(f"pre_eou_activity_recovery:{context}")
                evidence["escalated_to_cancel"] = True
            trace.event("pre_eou_activity_recovered", **evidence)
            return evidence

        async def inject_trailing_silence(job: ActiveTypedInput) -> None:
            rnnt_eou_frames = int(os.environ.get("VOICECHAT_RNNT_EOU_FRAMES", "20"))
            margin = float(os.environ.get("VOICECHAT_TYPED_INPUT_EOU_MARGIN", "1.25"))
            seconds = typed_input_trailing_silence_seconds(
                rnnt_eou_frames=rnnt_eou_frames,
                margin=margin,
            )
            frame = b"\x00\x00" * (INPUT_SAMPLE_RATE // 50)
            for _ in range(math.ceil(seconds / 0.02)):
                await process_pcm(frame, source="typed", job_id=job.job_id)

        async def run_typed_job(job: ActiveTypedInput) -> None:
            nonlocal active_typed, close_reason
            disposition = "completed"
            fatal_after_mutation = False
            try:
                if pocket_worker is None or not pocket_worker.ready:
                    raise PocketWorkerError("Pocket TTS worker is unavailable")
                synthesis = await pocket_worker.synthesize(
                    job.text,
                    job_id=job.job_id,
                    cancel_event=job.cancel_event,
                )
                if synthesis.sample_rate != INPUT_SAMPLE_RATE:
                    raise PocketWorkerError(f"Pocket worker returned {synthesis.sample_rate} Hz")
                if job.cancel_event.is_set():
                    disposition = "replaced" if job.replacement_requested else "cancelled"
                else:
                    await cancel_response_to_terminal("typed_input")

                    async def begin_mutation() -> None:
                        if job.cancel_event.is_set():
                            raise PocketSynthesisCancelled
                        async with send_lock:
                            await websocket.send_json(protocol.typed_input_started(job.job_id))
                            if client_turn_detection:
                                await websocket.send_json(
                                    protocol.begin_input_turn(
                                        "typed", job.job_id, job_id=job.job_id
                                    )
                                )
                        if client_turn_detection:
                            transport_gate.force_open()
                        job.audio_started = True
                        trace.event(
                            "typed_input_injection_started",
                            job_id=job.job_id,
                            generated_seconds=synthesis.generated_seconds,
                            synthesis_seconds=synthesis.wall_seconds,
                            time_to_first_audio_seconds=synthesis.time_to_first_audio_seconds,
                        )

                    bytes_per_frame = INPUT_SAMPLE_RATE * 20 // 1_000 * 2
                    for offset in range(0, len(synthesis.pcm16), bytes_per_frame):
                        chunk = synthesis.pcm16[offset : offset + bytes_per_frame]
                        if not chunk:
                            break
                        await process_pcm(
                            chunk,
                            source="typed",
                            job_id=job.job_id,
                            before_first_model_frame=(
                                begin_mutation if not job.audio_started else None
                            ),
                        )
                    if not job.audio_started and pcm_buffer.pending_samples:
                        tail = pcm_buffer.flush()
                        await process_pcm(
                            float32_to_pcm16_bytes(tail),
                            source="typed",
                            job_id=job.job_id,
                            before_first_model_frame=begin_mutation,
                        )
            except PocketSynthesisCancelled:
                disposition = "replaced" if job.replacement_requested else "cancelled"
            except SessionTerminated:
                disposition = "error"
                fatal_after_mutation = job.audio_started
            except Exception as exc:  # noqa: BLE001 - contain typed-input failure
                disposition = "error"
                fatal_after_mutation = job.audio_started
                trace.event(
                    "typed_input_error",
                    job_id=job.job_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if fatal_after_mutation:
                    start_terminal_cleanup(
                        "typed_input_failed_after_mutation",
                        events=[
                            protocol.error(
                                "typed_input_failed",
                                str(exc),
                                fatal=True,
                                client_event_id=job.job_id,
                            )
                        ],
                    )
                else:
                    try:
                        async with send_lock:
                            await websocket.send_json(
                                protocol.error(
                                    "typed_input_failed",
                                    str(exc),
                                    fatal=False,
                                    client_event_id=job.job_id,
                                )
                            )
                    except Exception:
                        pass
            finally:
                # Once model mutation begins the typed turn is non-cancellable.
                if (
                    job.audio_started
                    and not fatal_after_mutation
                    and close_reason != "session_position_limit"
                ):
                    try:
                        if client_turn_detection:
                            if pcm_buffer.pending_samples:
                                tail = pcm_buffer.flush()
                                await process_pcm(
                                    float32_to_pcm16_bytes(tail),
                                    source="typed",
                                    job_id=job.job_id,
                                )
                            # No recovery here: the typed path owns deliberate
                            # cancel-then-replace interruption semantics; recovery
                            # would drain a response the client intends to cancel.
                            typed_settlement = await settle_user_eou(
                                source="typed",
                                job_id=job.job_id,
                                fuse_final_bos=not session_tools,
                            )
                            if not typed_settlement.fused_terminal_bos:
                                engine.request_user_eou()

                                async def validate_typed_terminal(result: StepResult) -> None:
                                    await observe_separate_bos_or_defer(
                                        typed_settlement, result
                                    )

                                await process_pcm(
                                    b"\x00\x00" * FRAME_SAMPLES,
                                    source="typed",
                                    job_id=job.job_id,
                                    before_send=validate_typed_terminal,
                                )
                            engine.reset_user_transcript()
                        else:
                            await inject_trailing_silence(job)
                    except Exception:
                        disposition = "error"
                protocol.set_input_source("microphone")
                async with model_input_lock:
                    discarded = pcm_buffer.discard()
                    gate_preroll.clear()
                if discarded:
                    trace.event(
                        "typed_input_tail_discarded",
                        job_id=job.job_id,
                        samples=discarded,
                    )
                # The public finished event is a readiness barrier. Clear the
                # internal owner before publishing it so an immediately
                # following client turn cannot observe stale typed ownership.
                if active_typed is job:
                    active_typed = None
                if not fatal_after_mutation:
                    try:
                        async with send_lock:
                            await websocket.send_json(
                                protocol.typed_input_finished(job.job_id, disposition)
                            )
                    except Exception:
                        pass
                trace.event(
                    "typed_input_injection_finished",
                    job_id=job.job_id,
                    disposition=disposition,
                    audio_started=job.audio_started,
                )
                job.done.set()

        async def emit_external_tool_call(call: PendingExternalToolCall) -> None:
            nonlocal function_cycle_call_id, pending_function_turn_id, close_reason
            function_turn_id = pending_function_turn_id or protocol.turn_id
            if cancelling_unpublished_function_turn_id == function_turn_id:
                call.output = json.dumps({"error": "cancelled_by_new_client_turn"})
                call.completed.set()
                trace.event(
                    "function_call_publication_suppressed",
                    call_id=call.call_id,
                    name=call.name,
                    turn_id=function_turn_id,
                    reason="client_turn_start",
                )
                return
            allowed, attempt = function_call_budget.consume(function_turn_id)
            if not allowed:
                trace.event(
                    "function_call_loop_limit",
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    attempt=attempt,
                    limit=function_call_budget.limit,
                    turn_id=function_turn_id,
                    response_id=protocol.response_id,
                )
                start_terminal_cleanup(
                    "function_call_loop_limit",
                    events=function_call_loop_terminal_events(
                        protocol,
                        call_id=call.call_id,
                        attempt=attempt,
                        limit=function_call_budget.limit,
                    ),
                )
                raise SessionTerminated
            if client_turn_detection:
                if function_cycle_call_id is not None:
                    raise RuntimeError("overlapping external function cycles")
                function_cycle_call_id = call.call_id
                function_cycle_finished.clear()
            trace.event(
                "function_call_requested",
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                turn_id=function_turn_id,
            )
            async with send_lock:
                for event in protocol.function_call_events(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    turn_id=function_turn_id,
                ):
                    await websocket.send_json(event)
            pending_function_turn_id = None

        async def emit_external_tool_timeout(call: PendingExternalToolCall) -> None:
            nonlocal function_cycle_call_id, pending_bos_settlement, close_reason
            trace.event(
                "function_call_timeout",
                call_id=call.call_id,
                timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
            )
            matched_cycle = function_cycle_call_id == call.call_id
            if matched_cycle:
                function_cycle_call_id = None
            failed_event_error: str | None = None
            try:
                async with send_lock:
                    await websocket.send_json(
                        protocol.function_call_failed_event(
                            call_id=call.call_id,
                            reason="function_call_output_timeout",
                        )
                    )
            except (Exception, asyncio.CancelledError) as exc:
                failed_event_error = type(exc).__name__
                trace.event(
                    "function_call_timeout_failed_event_error",
                    call_id=call.call_id,
                    error=failed_event_error,
                )
            if pending_bos_settlement is not None:
                await terminate_pending_bos_expectation(
                    code="function_call_output_timeout",
                    message="Function cycle timed out before its terminal BOS",
                    call_id=call.call_id,
                    wait_for_cleanup=False,
                )
                return
            if matched_cycle:
                function_cycle_finished.set()

        tool_bridge = ExternalToolBridge(
            loop,
            emit_external_tool_call,
            emit_timeout=emit_external_tool_timeout,
            timeout_seconds=FUNCTION_CALL_TIMEOUT_SECONDS,
        )
        engine.configure_external_tools({})

        async def receive_or_terminal() -> dict[str, Any]:
            receive_task = asyncio.create_task(websocket.receive())
            terminal_task = asyncio.create_task(terminal_session.wait())
            done, _ = await asyncio.wait(
                {receive_task, terminal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if terminal_task in done:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                raise SessionTerminated
            terminal_task.cancel()
            await asyncio.gather(terminal_task, return_exceptions=True)
            return receive_task.result()

        try:
            await websocket.send_json(
                protocol.session_created(
                    {
                        "input_audio_format": "pcm16",
                        "input_audio_sample_rate": INPUT_SAMPLE_RATE,
                        "output_audio_format": "pcm16",
                        "output_audio_sample_rate": OUTPUT_SAMPLE_RATE,
                        "frame_duration_ms": int(FRAME_SECONDS * 1000),
                        "checkpoint": checkpoint_provenance,
                    }
                )
            )
            while True:
                if deferred_client_input and function_cycle_call_id is None:
                    body = deferred_client_input.popleft()
                    message_bytes = 0
                else:
                    if deferred_client_input and function_cycle_call_id is not None:
                        receive_task = asyncio.create_task(receive_or_terminal())
                        cycle_task = asyncio.create_task(function_cycle_finished.wait())
                        done, _ = await asyncio.wait(
                            {receive_task, cycle_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if cycle_task in done and receive_task not in done:
                            receive_task.cancel()
                            await asyncio.gather(receive_task, return_exceptions=True)
                            continue
                        cycle_task.cancel()
                        await asyncio.gather(cycle_task, return_exceptions=True)
                        message = receive_task.result()
                    else:
                        message = await receive_or_terminal()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("bytes") is not None:
                        await websocket.send_json(
                            protocol.error(
                                "binary_audio_unsupported",
                                "Send JSON input_audio_buffer.append with base64 PCM16 audio",
                            )
                        )
                        continue
                    text = message.get("text")
                    if text is None:
                        continue
                    message_bytes = len(text.encode("utf-8"))
                    try:
                        body = json.loads(text)
                    except json.JSONDecodeError:
                        await websocket.send_json(
                            protocol.error("invalid_json", "Message is not valid JSON")
                        )
                        continue
                    if not isinstance(body, dict):
                        await websocket.send_json(
                            protocol.error("invalid_event", "Event must be a JSON object")
                        )
                        continue
                message_type = body.get("type", "")
                client_event_id = body.get("event_id")

                if function_cycle_call_id is not None and message_type in {
                    "input_audio_buffer.turn_start",
                    "input_audio_buffer.append",
                    "input_audio_buffer.commit",
                }:
                    if not deferred_client_input.append(body, message_bytes):
                        await terminate_model_session(
                            "deferred_input_overflow",
                            events=[protocol.error(
                                "deferred_input_overflow",
                                "Client sent too much input during a function cycle",
                                fatal=True,
                                client_event_id=client_event_id,
                                call_id=function_cycle_call_id,
                            )],
                            close_code=1002,
                        )
                        return
                    trace.event(
                        "client_input_deferred_for_function",
                        type=message_type,
                        call_id=function_cycle_call_id,
                        queued_bytes=deferred_client_input.bytes,
                    )
                    continue

                if message_type == "session.update":
                    trace.event("client_event", type=message_type)
                    if (
                        last_model_result is not None
                        or active_typed is not None
                        or open_client_turn_id is not None
                    ):
                        await websocket.send_json(
                            protocol.error(
                                "settings_locked",
                                "session.update must precede the first input turn",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    session_update = body.get("session", {})
                    if not isinstance(session_update, dict):
                        await websocket.send_json(
                            protocol.error(
                                "invalid_session_update",
                                "session.update.session must be an object",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    requested_version = session_update.get("protocol_version")
                    if not configured and requested_version != PROTOCOL_VERSION:
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_protocol_version",
                                f"{PROTOCOL_NAME} version {PROTOCOL_VERSION} is required",
                                fatal=True,
                                client_event_id=client_event_id,
                                requested_version=requested_version,
                            )
                        )
                        await websocket.send_json(
                            protocol.session_closed("unsupported_protocol_version", status="failed")
                        )
                        close_reason = "unsupported_protocol_version"
                        await websocket.close(code=1002, reason=close_reason)
                        return
                    if configured and requested_version not in (None, PROTOCOL_VERSION):
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_protocol_version",
                                f"protocol version cannot change from {PROTOCOL_VERSION}",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    requested_capabilities = session_update.get("capabilities", {})
                    if not isinstance(requested_capabilities, dict):
                        await websocket.send_json(
                            protocol.error(
                                "invalid_session_capabilities",
                                "session.update capabilities must be an object",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    requested_model_output_mode = requested_capabilities.get(
                        "function_output_model_output"
                    )
                    if requested_model_output_mode not in (
                        None,
                        FUNCTION_OUTPUT_MODEL_OUTPUT_MODE,
                    ):
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_session_capability",
                                "Unsupported function_output_model_output capability",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    candidate_model_output_negotiated = (
                        requested_model_output_mode == FUNCTION_OUTPUT_MODEL_OUTPUT_MODE
                    )
                    if (
                        configured
                        and candidate_model_output_negotiated
                        != function_output_model_output_negotiated
                    ):
                        await websocket.send_json(
                            protocol.error(
                                "settings_locked",
                                "Negotiated session capabilities cannot change",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    candidate_system_message = system_message
                    candidate_tools = session_tools
                    if "instructions" in session_update:
                        instructions = session_update.get("instructions")
                        candidate_system_message = (
                            default_system_prompt if instructions is None else str(instructions)
                        )
                    if "tools" in session_update:
                        try:
                            candidate_tools = normalize_tool_definitions(session_update["tools"])
                        except (TypeError, ValueError) as exc:
                            await websocket.send_json(
                                protocol.error(
                                    "invalid_tools",
                                    str(exc),
                                    client_event_id=client_event_id,
                                )
                            )
                            continue
                    if candidate_tools:
                        if function_template_path is None:
                            await terminate_model_session(
                                "function_calling_unavailable",
                                events=[protocol.error(
                                    "function_calling_unavailable",
                                    "Function-calling template is unavailable",
                                    fatal=True,
                                    client_event_id=client_event_id,
                                )],
                            )
                            return
                        candidate_prompt = render_tool_system_prompt(
                            function_template_path,
                            candidate_system_message,
                            candidate_tools,
                        )
                    else:
                        candidate_prompt = candidate_system_message
                    settings_locked_after_validation = False
                    configuration_lock_requested_at = time.monotonic()
                    trace.event("session_update_model_input_lock_requested")
                    async with model_input_lock:
                        trace.event(
                            "session_update_model_input_lock_acquired",
                            wait_ms=round(
                                (time.monotonic() - configuration_lock_requested_at) * 1000,
                                3,
                            ),
                        )
                        # Typed synthesis advances model input on its own task.
                        # Recheck under the same lock used by process_pcm so a
                        # model position cannot race the empty-session restart.
                        if (
                            last_model_result is not None
                            or active_typed is not None
                            or open_client_turn_id is not None
                        ):
                            settings_locked_after_validation = True
                            configuration_changed = False
                        else:
                            configuration_changed = bool(
                                not configured
                                or candidate_system_message != system_message
                                or candidate_tools != session_tools
                                or candidate_prompt != system_prompt
                            )
                            if configuration_changed:
                                if session_started:
                                    trace.event(
                                        "model_session_empty_restart_requested",
                                        previous_stream_id=getattr(engine, "stream_id", None),
                                    )
                                    await model_call(engine.abort)
                                    session_started = False
                                    trace.event("model_session_empty_restart_completed")
                                engine.configure_external_tools(
                                    tool_bridge.handlers(candidate_tools)
                                )
                                await start_model_session(
                                    candidate_prompt,
                                    reason=(
                                        "session_update_initial"
                                        if not configured
                                        else "session_update_changed"
                                    ),
                                )
                                system_message = candidate_system_message
                                session_tools = candidate_tools
                                system_prompt = candidate_prompt
                            configured = True
                            function_output_model_output_negotiated = (
                                candidate_model_output_negotiated
                            )
                    if settings_locked_after_validation:
                        await websocket.send_json(
                            protocol.error(
                                "settings_locked",
                                "session.update must precede the first input turn",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    trace.event(
                        "session_tools_configured",
                        tools=[tool["name"] for tool in session_tools],
                        rendered_prompt_bytes=len(system_prompt.encode("utf-8")),
                        configuration_changed=configuration_changed,
                        model_ready=session_started,
                        function_output_model_output=(
                            FUNCTION_OUTPUT_MODEL_OUTPUT_MODE
                            if function_output_model_output_negotiated
                            else None
                        ),
                    )
                    await websocket.send_json(
                        protocol.session_updated(
                            {
                                "instructions": system_message,
                                "tools": session_tools,
                                "input_audio_format": "pcm16",
                                "input_audio_sample_rate": INPUT_SAMPLE_RATE,
                                "output_audio_format": "pcm16",
                                "output_audio_sample_rate": OUTPUT_SAMPLE_RATE,
                                "capabilities": {
                                    "function_output_model_output": (
                                        FUNCTION_OUTPUT_MODEL_OUTPUT_MODE
                                        if function_output_model_output_negotiated
                                        else None
                                    )
                                },
                            }
                        )
                    )
                    continue
                if message_type == "conversation.item.create":
                    if not configured:
                        await websocket.send_json(
                            protocol.error(
                                "session_not_configured",
                                "Complete session.update before sending conversation items",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    item = body.get("item", {})
                    if not isinstance(item, dict) or item.get("type") != "function_call_output":
                        await websocket.send_json(
                            protocol.error(
                                "invalid_conversation_item",
                                "conversation.item.create requires a function_call_output item",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if not session_tools:
                        await websocket.send_json(
                            protocol.error(
                                "tools_not_set",
                                "No tools were configured in session.update",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    call_id = item.get("call_id")
                    try:
                        function_output = validate_function_output(
                            engine,
                            item.get("output", ""),
                            model_output=item.get("model_output"),
                            model_output_negotiated=(
                                function_output_model_output_negotiated
                            ),
                        )
                        tool_bridge.submit(call_id, function_output)
                    except ValueError as exc:
                        await websocket.send_json(
                            protocol.error(
                                "invalid_function_call_output",
                                str(exc),
                                client_event_id=client_event_id,
                                call_id=call_id,
                            )
                        )
                        continue
                    protocol.function_call_completed(call_id)
                    trace.event(
                        "function_call_output_received",
                        call_id=call_id,
                        output_bytes=function_output.output_bytes,
                        output_sha256=hashlib.sha256(
                            function_output.output.encode("utf-8")
                        ).hexdigest(),
                        output_tokens=function_output.output_tokens,
                        model_output_supplied=function_output.model_output is not None,
                        model_output_bytes=function_output.model_output_bytes,
                        model_output_sha256=(
                            hashlib.sha256(
                                function_output.model_output.encode("utf-8")
                            ).hexdigest()
                            if function_output.model_output is not None
                            else None
                        ),
                        model_output_tokens=function_output.model_output_tokens,
                        injection_source=(
                            "model_output"
                            if function_output.model_output is not None
                            else "output"
                        ),
                        injection_tokens=function_output.injection_tokens,
                    )
                    if not client_turn_detection:
                        continue

                    async def advance_recovery_frame() -> Any:
                        def validate_pending_terminal(result: StepResult) -> None:
                            nonlocal pending_bos_settlement
                            settlement = pending_bos_settlement
                            if settlement is None:
                                return
                            turn_state = result.turn_state or {}
                            if turn_state.get("agent_control") == "agent_bos":
                                settlement.observe_separate_terminal(engine, result)
                                pending_bos_settlement = None
                                return
                            if function_cycle_active(result):
                                return
                            fc = turn_state.get("function_calling") or {}
                            if fc.get("post_fc_client_bos_pending"):
                                return
                            raise UserEouSettlementFailure(
                                "missing_post_fc_terminal_bos",
                                "Function recovery completed without its terminal BOS",
                                settlement.evidence(),
                            )

                        return await process_pcm(
                            b"\x00\x00" * FRAME_SAMPLES,
                            source="microphone",
                            job_id=None,
                            before_send=validate_pending_terminal,
                        )

                    applied = await advance_function_output_recovery(
                        advance_recovery_frame,
                        completion=lambda _result: pending_bos_settlement is None,
                    )
                    if not applied:
                        terminated = await terminate_pending_bos_expectation(
                            code="function_output_apply_timeout",
                            message="Function output did not leave post-FC recovery",
                            call_id=call_id,
                            client_event_id=client_event_id,
                        )
                        if not terminated:
                            await terminate_model_session(
                                "function_output_apply_timeout",
                                events=[protocol.error(
                                        "function_output_apply_timeout",
                                        "Function output did not leave post-FC recovery",
                                        fatal=True,
                                        client_event_id=client_event_id,
                                        call_id=call_id,
                                    )],
                            )
                        return
                    if function_cycle_call_id != call_id:
                        await terminate_model_session(
                            "function_cycle_mismatch",
                            events=[protocol.error(
                                "function_cycle_mismatch",
                                "Applied function output did not match active cycle",
                                fatal=True,
                                call_id=call_id,
                                active_call_id=function_cycle_call_id,
                            )],
                        )
                        return
                    await websocket.send_json(protocol.function_call_output_applied(call_id))
                    function_cycle_call_id = None
                    function_cycle_finished.set()
                    continue
                if message_type == "input_audio_buffer.turn_start":
                    if not client_turn_detection:
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_event",
                                "Client turn detection was not negotiated",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if not configured:
                        await websocket.send_json(
                            protocol.error(
                                "session_not_configured",
                                "Complete session.update before starting a turn",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if not isinstance(client_event_id, str) or not client_event_id:
                        await terminate_model_session(
                            "invalid_event_id",
                            events=[protocol.error(
                                "invalid_event_id",
                                "Client turn barriers require a non-empty event_id",
                                fatal=True,
                            )],
                            close_code=1002,
                        )
                        return
                    client_turn_id = body.get("client_turn_id")
                    if not isinstance(client_turn_id, int) or client_turn_id < 1:
                        await terminate_model_session(
                            "invalid_client_turn_id",
                            events=[protocol.error(
                                "invalid_client_turn_id",
                                "client_turn_id must be a positive integer",
                                fatal=True,
                                client_event_id=client_event_id,
                            )],
                            close_code=1002,
                        )
                        return
                    if client_turn_id in started_client_turns:
                        await websocket.send_json(
                            protocol.input_turn_started_ack(
                                client_event_id,
                                client_turn_id,
                                turn_id=started_client_turns[client_turn_id],
                            )
                        )
                        continue
                    if open_client_turn_id is not None or active_typed is not None:
                        await terminate_model_session(
                            "input_turn_conflict",
                            events=[protocol.error(
                                "input_turn_conflict",
                                "A different input turn is already active",
                                fatal=True,
                                client_event_id=client_event_id,
                                client_turn_id=client_turn_id,
                            )],
                            close_code=1002,
                        )
                        return
                    # Freeze real onset audio before synthetic cancellation
                    # steps can advance or clear RNNT evidence.  A function
                    # call that publishes while this handler waits for the
                    # model lock wins the race and keeps the established
                    # post-publication deferred-input contract.
                    publication_race = False
                    interrupt_evidence: dict[str, Any] | None = None
                    async with model_input_lock:
                        if function_cycle_call_id is not None:
                            publication_race = True
                            preroll_entries = []
                            partial_onset = np.array([], dtype=np.float32)
                        else:
                            preroll_entries = select_gate_preroll(reason="client_turn_start")
                            partial_onset = pcm_buffer.take_pending()
                            if pending_bos_settlement is not None:
                                if pending_function_turn_id is None:
                                    raise RuntimeError(
                                        "deferred FC settlement has no initiating turn"
                                    )
                                cancelling_unpublished_function_turn_id = (
                                    pending_function_turn_id
                                )
                                interrupt_evidence = await model_call(
                                    engine.interrupt_unpublished_function_cycle
                                )
                    if publication_race:
                        if not deferred_client_input.append(body, message_bytes):
                            raise RuntimeError(
                                "function publication race overflowed deferred input"
                            )
                        trace.event(
                            "client_turn_start_lost_function_publication_race",
                            client_turn_id=client_turn_id,
                            call_id=function_cycle_call_id,
                            queued_bytes=deferred_client_input.bytes,
                        )
                        continue
                    if interrupt_evidence is not None:
                        trace.event(
                            "unpublished_function_cycle_interrupt",
                            client_turn_id=client_turn_id,
                            initiating_turn_id=cancelling_unpublished_function_turn_id,
                            evidence=interrupt_evidence,
                        )
                    await cancel_response_to_terminal("client_vad")
                    prepare_evidence: dict[str, Any] = {
                        "passed": False,
                        "reason": "server_preconditions_not_met",
                    }
                    if (
                        function_cycle_call_id is None
                        and function_cycle_finished.is_set()
                        and not protocol.response_open
                        and session_started
                    ):
                        prepare_fn = getattr(engine, "prepare_eartts_epoch", None)
                        if callable(prepare_fn):
                            async with model_input_lock:
                                prepare_evidence = await model_call(prepare_fn)
                    trace.event(
                        "eartts_turn_start_prepare",
                        client_turn_id=client_turn_id,
                        evidence=prepare_evidence,
                    )
                    open_client_turn_id = client_turn_id
                    await websocket.send_json(
                        protocol.begin_input_turn("microphone", client_turn_id)
                    )
                    remember_client_turn(started_client_turns, client_turn_id, protocol.turn_id)
                    await websocket.send_json(
                        protocol.input_turn_started_ack(client_event_id, client_turn_id)
                    )
                    async with model_input_lock:
                        transport_gate.force_open()
                        await replay_gate_preroll(preroll_entries)
                    if partial_onset.size:
                        await process_pcm(
                            float32_to_pcm16_bytes(partial_onset),
                            source="microphone",
                            job_id=None,
                        )
                    trace.event(
                        "client_input_turn_started",
                        client_turn_id=client_turn_id,
                        preroll_frames=len(preroll_entries),
                    )
                    continue
                if message_type == "input_audio_buffer.commit":
                    commit_received_at = time.monotonic()
                    if not client_turn_detection:
                        await websocket.send_json(
                            protocol.error(
                                "unsupported_event",
                                "Client turn detection was not negotiated",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if not isinstance(client_event_id, str) or not client_event_id:
                        await terminate_model_session(
                            "invalid_event_id",
                            events=[protocol.error(
                                "invalid_event_id",
                                "Client turn barriers require a non-empty event_id",
                                fatal=True,
                            )],
                            close_code=1002,
                        )
                        return
                    client_turn_id = body.get("client_turn_id")
                    if not isinstance(client_turn_id, int) or client_turn_id < 1:
                        await terminate_model_session(
                            "invalid_client_turn_id",
                            events=[protocol.error(
                                "invalid_client_turn_id",
                                "client_turn_id must be a positive integer",
                                fatal=True,
                                client_event_id=client_event_id,
                            )],
                            close_code=1002,
                        )
                        return
                    if client_turn_id in completed_client_turns:
                        await websocket.send_json(
                            protocol.input_turn_committed_ack(
                                client_event_id,
                                client_turn_id,
                                turn_id=completed_client_turns[client_turn_id],
                            )
                        )
                        continue
                    if client_turn_id != open_client_turn_id:
                        await terminate_model_session(
                            "input_turn_conflict",
                            events=[protocol.error(
                                "input_turn_conflict",
                                "Commit does not match the open client turn",
                                fatal=True,
                                client_event_id=client_event_id,
                                client_turn_id=client_turn_id,
                                open_client_turn_id=open_client_turn_id,
                            )],
                            close_code=1002,
                        )
                        return
                    raw_diagnostics = body.get("diagnostics")
                    client_diagnostics = {
                        key: value
                        for key, value in (
                            raw_diagnostics.items() if isinstance(raw_diagnostics, dict) else ()
                        )
                        if key
                        in {
                            "schema",
                            "client_enqueue_monotonic_s",
                            "client_send_monotonic_s",
                            "queue_chunks_at_enqueue",
                            "queue_audio_bytes_at_enqueue",
                            "queue_chunks_after_get",
                            "queue_audio_bytes_after_get",
                        }
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    }
                    client_send_at = client_diagnostics.get("client_send_monotonic_s")
                    client_enqueue_at = client_diagnostics.get("client_enqueue_monotonic_s")
                    trace.event(
                        "client_input_turn_commit_received",
                        client_turn_id=client_turn_id,
                        client_diagnostics=client_diagnostics,
                        client_queue_wait_ms=(
                            round((float(client_send_at) - float(client_enqueue_at)) * 1000, 3)
                            if client_send_at is not None and client_enqueue_at is not None
                            else None
                        ),
                        client_send_to_server_receive_ms=(
                            round((commit_received_at - float(client_send_at)) * 1000, 3)
                            if client_send_at is not None
                            else None
                        ),
                    )
                    if pcm_buffer.pending_samples:
                        tail = pcm_buffer.flush()
                        await process_pcm(
                            float32_to_pcm16_bytes(tail),
                            source="microphone",
                            job_id=None,
                        )
                    arm_fn = getattr(engine, "arm_eartts_epoch", None)
                    if callable(arm_fn):
                        async with model_input_lock:
                            arm_evidence = await model_call(arm_fn, client_turn_id)
                    else:
                        arm_evidence = {
                            "passed": False,
                            "reason": "prepared_epoch_disabled",
                        }
                    trace.event(
                        "eartts_client_commit_arm",
                        client_turn_id=client_turn_id,
                        evidence=arm_evidence,
                    )
                    commit_ack_sent = False
                    fused_bos_at: float | None = None

                    async def acknowledge_fused_terminal() -> None:
                        nonlocal commit_ack_sent, fused_bos_at
                        fused_bos_at = time.monotonic()
                        remember_client_turn(
                            completed_client_turns,
                            client_turn_id,
                            protocol.turn_id,
                        )
                        async with send_lock:
                            await websocket.send_json(
                                protocol.input_turn_committed_ack(client_event_id, client_turn_id)
                            )
                        commit_ack_sent = True

                    await recover_pre_eou_activity("microphone_commit")
                    microphone_settlement = await settle_user_eou(
                        source="microphone",
                        job_id=None,
                        fuse_final_bos=not session_tools,
                        expected_client_turn_id=client_turn_id,
                        on_fused_terminal=acknowledge_fused_terminal,
                    )
                    fused_terminal = microphone_settlement.fused_terminal_bos
                    eou_settled_at = fused_bos_at or time.monotonic()
                    if not commit_ack_sent:
                        remember_client_turn(
                            completed_client_turns,
                            client_turn_id,
                            protocol.turn_id,
                        )
                        await websocket.send_json(
                            protocol.input_turn_committed_ack(client_event_id, client_turn_id)
                        )
                    if not fused_terminal:
                        engine.request_user_eou()

                        async def validate_microphone_terminal(result: StepResult) -> None:
                            await observe_separate_bos_or_defer(
                                microphone_settlement, result
                            )

                        await process_pcm(
                            b"\x00\x00" * FRAME_SAMPLES,
                            source="microphone",
                            job_id=None,
                            before_send=validate_microphone_terminal,
                        )
                    engine.reset_user_transcript()
                    open_client_turn_id = None
                    bos_at = fused_bos_at or time.monotonic()
                    trace.event(
                        "client_input_turn_committed",
                        client_turn_id=client_turn_id,
                        receive_to_eou_settled_ms=round(
                            (eou_settled_at - commit_received_at) * 1000,
                            3,
                        ),
                        eou_settled_to_bos_ms=round(
                            (bos_at - eou_settled_at) * 1000,
                            3,
                        ),
                        receive_to_bos_ms=round(
                            (bos_at - commit_received_at) * 1000,
                            3,
                        ),
                        **({"fused_terminal_bos": True} if fused_terminal else {}),
                    )
                    continue
                if message_type == "input_text.request":
                    if not configured:
                        await websocket.send_json(
                            protocol.typed_input_rejected(
                                body.get("job_id"), "session_not_configured"
                            )
                        )
                        continue
                    job_id = body.get("job_id")
                    typed_text = body.get("text")
                    rejection = None
                    if pocket_worker is None or not pocket_worker.ready:
                        rejection = "typed_input_unavailable"
                    elif open_client_turn_id is not None:
                        rejection = "input_turn_active"
                    elif function_cycle_call_id is not None:
                        rejection = "function_cycle_active"
                    elif not isinstance(job_id, str) or not job_id or len(job_id) > 128:
                        rejection = "invalid_job_id"
                    elif job_id in seen_typed_job_ids:
                        rejection = "duplicate_job_id"
                    elif not isinstance(typed_text, str) or not typed_text.strip():
                        rejection = "empty_text"
                    elif len(typed_text) > MAX_TYPED_INPUT_CHARS:
                        rejection = "text_too_long"
                    if rejection is not None:
                        await websocket.send_json(
                            protocol.typed_input_rejected(
                                job_id if isinstance(job_id, str) else None,
                                rejection,
                            )
                        )
                        continue
                    if active_typed is not None:
                        if not active_typed.audio_started:
                            active_typed.replacement_requested = True
                            active_typed.cancel_event.set()
                        await active_typed.done.wait()
                        if typed_task is not None:
                            await asyncio.gather(typed_task, return_exceptions=True)
                    seen_typed_job_ids.add(job_id)
                    async with model_input_lock:
                        discarded = pcm_buffer.discard()
                        gate_preroll.clear()
                    if discarded:
                        trace.event(
                            "microphone_partial_discarded_for_typed_input",
                            job_id=job_id,
                            samples=discarded,
                        )
                    active_typed = ActiveTypedInput(
                        job_id=job_id,
                        text=typed_text,
                        cancel_event=asyncio.Event(),
                        done=asyncio.Event(),
                    )
                    trace.event(
                        "typed_input_accepted",
                        job_id=job_id,
                        text_bytes=len(typed_text.encode("utf-8")),
                    )
                    await websocket.send_json(protocol.typed_input_accepted(job_id))
                    typed_task = asyncio.create_task(
                        run_typed_job(active_typed),
                        name=f"typed-input-{job_id}",
                    )
                    continue
                if message_type == "input_audio_buffer.append":
                    if not configured:
                        await websocket.send_json(
                            protocol.error(
                                "session_not_configured",
                                "Complete session.update before sending audio",
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    try:
                        if body.get("encoding") != "pcm16":
                            raise RealtimeProtocolError(
                                "unsupported_audio_encoding",
                                "Only pcm16 input is supported",
                            )
                        if int(body.get("sample_rate", 0)) != INPUT_SAMPLE_RATE:
                            raise RealtimeProtocolError(
                                "unsupported_sample_rate",
                                f"Input must be {INPUT_SAMPLE_RATE} Hz",
                            )
                        if int(body.get("channels", 0)) != 1:
                            raise RealtimeProtocolError(
                                "unsupported_channel_count", "Input must be mono"
                            )
                        payload = base64.b64decode(body.get("audio", ""), validate=True)
                        if not payload:
                            raise RealtimeProtocolError(
                                "empty_audio", "Audio payload must not be empty"
                            )
                        if len(payload) % 2:
                            raise RealtimeProtocolError(
                                "invalid_audio",
                                "PCM16 audio must contain whole samples",
                            )
                        if len(payload) > MAX_INPUT_MESSAGE_BYTES:
                            raise RealtimeProtocolError(
                                "audio_packet_too_large", "Audio packet is too large"
                            )
                    except (ValueError, TypeError, base64.binascii.Error) as exc:
                        await websocket.send_json(
                            protocol.error(
                                "invalid_audio",
                                str(exc),
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    except RealtimeProtocolError as exc:
                        await websocket.send_json(
                            protocol.error(
                                exc.code,
                                str(exc),
                                fatal=exc.fatal,
                                client_event_id=client_event_id,
                            )
                        )
                        continue
                    if active_typed is not None:
                        trace.event(
                            "microphone_packet_gated",
                            bytes=len(payload),
                            active_job_id=active_typed.job_id,
                        )
                        continue
                    await process_pcm(payload, source="microphone", job_id=None)
                    continue
                if message_type == "session.stop":
                    trace.event("client_event", type=message_type)
                    if active_typed is not None:
                        active_typed.cancel_event.set()
                        try:
                            await asyncio.wait_for(active_typed.done.wait(), timeout=10.0)
                        except TimeoutError:
                            pass
                    async with model_input_lock:
                        gate_preroll.clear()
                    if session_started:
                        if last_model_result is None:
                            trace.event("model_session_empty_stop_requested")
                            await model_call(engine.abort)
                            trace.event("model_session_empty_stop_completed")
                        else:
                            transport_frame += 1
                            final_frame = pcm_buffer.flush()
                            final_stats = audio_stats(final_frame)
                            result = await model_call(
                                engine.process,
                                final_frame,
                                is_last=True,
                                input_active=False,
                            )
                            result.turn_state["transport"] = transport_gate.snapshot(
                                float(final_stats["rms_dbfs"]), True
                            )
                            async with send_lock:
                                last_user_text = await send_step(
                                    websocket,
                                    result,
                                    last_user_text,
                                    trace,
                                    transport_frame,
                                    protocol,
                                )
                        session_started = False
                        transport_gate.reset()
                    for event in protocol.close_events(
                        "client_stop", cumulative_user_text=last_user_text
                    ):
                        await websocket.send_json(event)
                    close_reason = "client_stop"
                    await websocket.close(code=1000, reason=close_reason)
                    return
                await websocket.send_json(
                    protocol.error(
                        "unsupported_event",
                        f"Unsupported event: {message_type}",
                        client_event_id=client_event_id,
                    )
                )
        except SessionTerminated:
            pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await terminate_model_session(
                "internal_error",
                events=[
                    protocol.error(
                        "internal_error",
                        f"{type(exc).__name__}: {exc}",
                        fatal=True,
                    ),
                    protocol.session_closed("internal_error", status="failed"),
                ],
            )
        finally:
            if terminal_cleanup_task is None:
                if active_typed is not None:
                    active_typed.cancel_event.set()
                if typed_task is not None and not typed_task.done():
                    try:
                        await asyncio.wait_for(typed_task, timeout=3.0)
                    except TimeoutError:
                        typed_task.cancel()
                        await asyncio.gather(typed_task, return_exceptions=True)
                tool_bridge.close()
                await model_call(engine.abort)
                trace.close(close_reason)
                active_client.release()
            else:
                await await_cleanup_cancellation_safe(terminal_cleanup_task)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--hf-skeleton", required=True)
    parser.add_argument(
        "--checkpoint-revision",
        default="fb0f94eaf4d03ddc430f39565229393fa1b50c26",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default="d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b",
    )
    parser.add_argument("--speaker-name", default="Aria")
    parser.add_argument("--nano-vllm-path", required=True)
    parser.add_argument("--eartts-vllm-path", required=True)
    parser.add_argument("--vllm-manifest", required=True)
    parser.add_argument("--warmup-wav")
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("VOICECHAT_WEB_SYSTEM_PROMPT", ""),
        help="Default prompt used when session.start does not supply one",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--trace-dir",
        help="Write per-session JSONL plus exact input/output PCM payloads here",
    )
    parser.add_argument(
        "--speech-gate-dbfs",
        type=float,
        default=float(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_DBFS", "-50")),
        help="RMS threshold for transport speech evidence",
    )
    parser.add_argument(
        "--speech-gate-min-frames",
        type=int,
        default=int(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_MIN_FRAMES", "1")),
        help="Consecutive 80 ms frames required for transport speech evidence",
    )
    parser.add_argument(
        "--speech-gate-preroll-frames",
        type=int,
        default=int(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_PREROLL_FRAMES", "12")),
        help="Recent gated 80 ms frames replayed when transport speech opens",
    )
    parser.add_argument(
        "--speech-gate-onset-context-frames",
        type=int,
        default=int(os.environ.get("VOICECHAT_WEB_SPEECH_GATE_ONSET_CONTEXT_FRAMES", "2")),
        help="Quiet 80 ms context frames retained before buffered speech onset",
    )
    parser.add_argument(
        "--continuous-after-speech",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VOICECHAT_WEB_CONTINUOUS_AFTER_SPEECH", "1")
        not in {"0", "false", "False"},
        help="After first speech, advance every received frame including silence",
    )
    return parser.parse_args()


def main() -> None:
    STARTUP_READY_SENTINEL.unlink(missing_ok=True)
    startup_event("main_entered")
    args = parse_args()
    if os.environ.get("VOICECHAT_STEP9_VALIDATE_BAKED_CACHE", "0") == "1":
        from .step9_cache_guard import validate_baked_vllm_cache

        startup_event("baked_cache_validation_started")
        validate_baked_vllm_cache()
        startup_event("baked_cache_validation_finished")
    # Install the all-thread dump only for an explicitly instrumented run. The
    # ordinary server retains its prior signal dispositions and runtime path.
    configure_fc_async_thread_dump()
    function_template = (
        Path(args.speech_root).expanduser().resolve()
        / "examples/speechlm2/function_calling/template.jinja"
    )
    if not function_template.is_file():
        raise SystemExit(f"Missing NVIDIA function-calling template: {function_template}")
    startup_event("pipeline_build_started")
    pipeline = build_pipeline(args)
    startup_event("pipeline_build_finished")
    direct_probe_dir = os.environ.get("VOICECHAT_DIRECT_POSITION_PROBE_DIR", "").strip()
    direct_safety_probe_dir = os.environ.get(
        "VOICECHAT_DIRECT_POSITION_SAFETY_PROBE_DIR", ""
    ).strip()
    direct_recurrence_probe_dir = os.environ.get(
        "VOICECHAT_DIRECT_POSITION_RECURRENCE_PROBE_DIR", ""
    ).strip()
    direct_semantic_probe_dir = os.environ.get("VOICECHAT_DIRECT_SEMANTIC_PROBE_DIR", "").strip()
    voice_function_probe_dir = os.environ.get("VOICECHAT_VOICE_FUNCTION_PROBE_DIR", "").strip()
    diagnostic_probe_dir = (
        direct_probe_dir
        or direct_safety_probe_dir
        or direct_recurrence_probe_dir
        or direct_semantic_probe_dir
        or voice_function_probe_dir
    )
    direct_probe_skip_warmups = bool(
        diagnostic_probe_dir and env_bool("VOICECHAT_DIRECT_POSITION_PROBE_SKIP_WARMUPS", False)
    )
    if not direct_probe_skip_warmups:
        startup_event("pipeline_warmup_started")
        warm_pipeline(pipeline, args.warmup_wav, system_prompt=args.system_prompt)
        startup_event("pipeline_warmup_finished")
    agent_no_text_frames = int(os.environ.get("VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES", "30"))
    qualification_mode = getattr(pipeline, "qualification_mode", None)
    if qualification_mode is not None and not is_validated_qualification_mode(
        qualification_mode
    ):
        raise RuntimeError("pipeline retained an invalid qualification identity")
    agent_no_audio_frames = int(os.environ.get("VOICECHAT_WEB_AGENT_NO_AUDIO_FRAMES", "30"))
    agent_decoded_silence_frames = int(
        os.environ.get("VOICECHAT_WEB_AGENT_SILENCE_EOS_FRAMES", "20")
    )
    candidate_runtime = pipeline.checkpoint_provenance.get("nano_pad_pair_runtime", {}).get(
        "production_candidate"
    )
    if candidate_runtime is not None:
        candidate_contract = pipeline.checkpoint_provenance["nano_pad_pair_runtime"]["contract"]
        expected_no_text_frames = int(candidate_contract["agent_no_text_frames"])
        expected_no_audio_frames = int(candidate_contract["agent_no_audio_frames"])
        expected_decoded_silence_frames = int(candidate_contract["agent_decoded_silence_frames"])
        if (
            agent_no_text_frames != expected_no_text_frames
            or agent_no_audio_frames != expected_no_audio_frames
            or agent_decoded_silence_frames != expected_decoded_silence_frames
        ):
            raise RuntimeError(
                "Production candidate BOS progress deadline mismatch: "
                f"text={agent_no_text_frames}/{expected_no_text_frames}, "
                f"audio={agent_no_audio_frames}/{expected_no_audio_frames}, "
                "decoded_silence="
                f"{agent_decoded_silence_frames}/{expected_decoded_silence_frames}"
            )
    startup_event("realtime_engine_create_started")
    engine = VoiceChatEngine(
        pipeline,
        agent_silence_eos_dbfs=float(os.environ.get("VOICECHAT_WEB_AGENT_SILENCE_EOS_DBFS", "-90")),
        agent_silence_eos_frames=agent_decoded_silence_frames,
        agent_no_text_frames=agent_no_text_frames,
        agent_no_audio_frames=agent_no_audio_frames,
        qualification_mode=qualification_mode,
        response_tail_dbfs=float(os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_DBFS", "-60")),
        response_tail_silence_frames=int(
            os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_SILENCE_FRAMES", "3")
        ),
        response_tail_max_frames=int(
            os.environ.get("VOICECHAT_WEB_RESPONSE_TAIL_MAX_FRAMES", "25")
        ),
        delivery_silence_frames=int(os.environ.get("VOICECHAT_WEB_DELIVERY_SILENCE_FRAMES", "12")),
    )
    startup_event("realtime_engine_create_finished")
    if (
        os.environ.get("VOICECHAT_WEB_REALTIME_WARMUP", "1") == "1"
        and not direct_probe_skip_warmups
    ):
        startup_event("realtime_warmup_started")
        realtime_warmup = warm_realtime_engine(engine, system_prompt=args.system_prompt)
        pipeline.checkpoint_provenance["realtime_continuation_warmup"] = realtime_warmup
        startup_event("realtime_warmup_finished")
    if voice_function_probe_dir:
        from .voice_function_probe import VOICE_RESPONSE_FRAME_LIMIT, run_voice_function_probe

        output_dir = Path(voice_function_probe_dir).expanduser().resolve()
        report_path = output_dir / "report.json"
        try:
            corpus_value = os.environ.get("VOICECHAT_VOICE_FUNCTION_CORPUS", "").strip()
            fixture_value = os.environ.get("VOICECHAT_VOICE_FUNCTION_FIXTURE", "").strip()
            runtime_image_id = os.environ.get("VOICECHAT_RUNTIME_IMAGE_ID", "").strip()
            if not corpus_value or not fixture_value or not runtime_image_id:
                raise RuntimeError(
                    "voice function probe requires corpus, fixture, and runtime image ID"
                )
            report = run_voice_function_probe(
                engine,
                fixture_dir=Path(fixture_value).expanduser().resolve(),
                corpus_path=Path(corpus_value).expanduser().resolve(),
                output_dir=output_dir,
                function_template_path=function_template,
                precision_label="w8",
                runtime_image_id=runtime_image_id,
                speech_gate_dbfs=args.speech_gate_dbfs,
                speech_gate_min_frames=args.speech_gate_min_frames,
                continuous_after_speech=args.continuous_after_speech,
                max_response_frames=int(
                    os.environ.get(
                        "VOICECHAT_VOICE_FUNCTION_MAX_FRAMES",
                        str(VOICE_RESPONSE_FRAME_LIMIT),
                    )
                ),
            )
        except BaseException as exc:
            trace_enabled = bool(int(os.environ.get("S2S_FUNCTION_LOGIT_TRACE_TOPK", "0") or "0"))
            report = {
                "schema": 2 if trace_enabled else 1,
                "kind": "voice_function_logit_probe",
                "precision": "w8",
                "case_id": "c014",
                "passed": False,
                "voice_reproduction_valid": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Voice function probe report: {report_path}", flush=True)
        if not report.get("passed", False):
            raise RuntimeError(f"voice function probe failed; inspect {report_path}")
        return
    if direct_safety_probe_dir:
        output_dir = Path(direct_safety_probe_dir).expanduser().resolve()
        report_path = output_dir / "report.json"
        try:
            report = run_direct_position_safety_probe(
                engine,
                output_dir=output_dir,
                system_prompt=args.system_prompt,
            )
        except BaseException as exc:
            report = {
                "schema": 1,
                "kind": "direct_user_source_position_safety_probe",
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Direct-position safety probe report: {report_path}", flush=True)
        if not report.get("passed", False):
            raise RuntimeError(f"direct-position safety probe failed; inspect {report_path}")
        if (
            not direct_probe_dir
            and not direct_recurrence_probe_dir
            and not direct_semantic_probe_dir
        ):
            return
    if direct_recurrence_probe_dir:
        output_dir = Path(direct_recurrence_probe_dir).expanduser().resolve()
        report_path = output_dir / "report.json"
        try:
            report = run_direct_position_recurrence_probe(
                engine,
                output_dir=output_dir,
                system_prompt=args.system_prompt,
                positions=int(
                    os.environ.get("VOICECHAT_DIRECT_POSITION_RECURRENCE_POSITIONS", "4")
                ),
            )
        except BaseException as exc:
            report = {
                "schema": 1,
                "kind": "direct_user_source_position_recurrence_probe",
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Direct-position recurrence probe report: {report_path}", flush=True)
        if not report.get("passed", False):
            raise RuntimeError(f"direct-position recurrence probe failed; inspect {report_path}")
        if not direct_probe_dir:
            if not direct_semantic_probe_dir:
                return
    if direct_semantic_probe_dir:
        from .direct_semantic_probe import run_direct_semantic_probe

        output_dir = Path(direct_semantic_probe_dir).expanduser().resolve()
        report_path = output_dir / "report.json"
        try:
            corpus_value = os.environ.get("VOICECHAT_DIRECT_SEMANTIC_CORPUS", "").strip()
            if not corpus_value:
                raise RuntimeError("VOICECHAT_DIRECT_SEMANTIC_CORPUS is required")
            report = run_direct_semantic_probe(
                engine,
                corpus_path=Path(corpus_value).expanduser().resolve(),
                output_dir=output_dir,
                function_template_path=function_template,
                token_forms=[
                    value.strip()
                    for value in os.environ.get(
                        "VOICECHAT_DIRECT_SEMANTIC_FORMS",
                        "text,user_bos_text_user_eos",
                    ).split(",")
                    if value.strip()
                ],
                precision_label=os.environ.get(
                    "VOICECHAT_DIRECT_SEMANTIC_PRECISION", "unspecified"
                ),
                max_response_frames=int(
                    os.environ.get("VOICECHAT_DIRECT_SEMANTIC_MAX_FRAMES", "240")
                ),
            )
        except BaseException as exc:
            report = {
                "schema": 2,
                "kind": "direct_text_semantic_probe",
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Direct semantic probe report: {report_path}", flush=True)
        if not report.get("passed", False):
            raise RuntimeError(f"direct semantic probe failed; inspect {report_path}")
        if not direct_probe_dir:
            return
    if direct_probe_dir:
        output_dir = Path(direct_probe_dir).expanduser().resolve()
        report_path = output_dir / "report.json"
        try:
            report = run_direct_position_probe(
                engine,
                text=os.environ.get(
                    "VOICECHAT_DIRECT_POSITION_PROBE_TEXT",
                    "Reply with exactly the words direct path works.",
                ),
                token_forms=[
                    value.strip()
                    for value in os.environ.get(
                        "VOICECHAT_DIRECT_POSITION_PROBE_FORMS", "text"
                    ).split(",")
                    if value.strip()
                ],
                output_dir=output_dir,
                system_prompt=args.system_prompt,
                max_response_frames=int(
                    os.environ.get("VOICECHAT_DIRECT_POSITION_PROBE_MAX_FRAMES", "240")
                ),
            )
        except BaseException as exc:
            report = {
                "schema": 1,
                "kind": "direct_user_source_position_probe",
                "passed": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Direct-position probe report: {report_path}", flush=True)
        if not report.get("passed", False):
            raise RuntimeError(f"direct-position probe failed; inspect {report_path}")
        return
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"WebSocket traces: {trace_dir}", flush=True)
    startup_event("application_create_started")
    app = create_app(
        engine,
        trace_dir,
        args.speech_gate_dbfs,
        args.speech_gate_min_frames,
        args.continuous_after_speech,
        args.system_prompt,
        function_template,
        max_session_model_frames=int(
            os.environ.get("VOICECHAT_WEB_MAX_SESSION_MODEL_FRAMES", "12000")
        ),
        pocket_worker=PocketWorkerManager(
            executable=os.environ.get("VOICECHAT_POCKET_PYTHON", "/opt/pocket-tts/bin/python"),
            socket_path=Path(
                os.environ.get("VOICECHAT_POCKET_SOCKET", "/tmp/voicechat-pocket.sock")
            ),
            language=os.environ.get("VOICECHAT_TYPED_INPUT_LANGUAGE", "english_2026-04"),
            voice=os.environ.get("VOICECHAT_TYPED_INPUT_VOICE", "alba"),
            threads=int(os.environ.get("VOICECHAT_TYPED_INPUT_TORCH_THREADS", "4")),
            seed=int(os.environ.get("VOICECHAT_TYPED_INPUT_SEED", "0")),
            cpus=tuple(
                int(item)
                for item in os.environ.get("VOICECHAT_TYPED_INPUT_CPU_CORES", "7,8,9,15").split(",")
            ),
        ),
        speech_gate_preroll_frames=args.speech_gate_preroll_frames,
        speech_gate_onset_context_frames=args.speech_gate_onset_context_frames,
        client_turn_detection=True,
    )
    startup_event("application_create_finished")

    import uvicorn

    startup_event("uvicorn_run_entered")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=websocket_ping_interval(),
        ws_ping_timeout=websocket_ping_timeout(),
    )


if __name__ == "__main__":
    main()
