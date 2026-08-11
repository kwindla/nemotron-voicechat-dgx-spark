from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "src" / "nemotron_voicechat_runtime"

from nemotron_voicechat_runtime.protocol import RealtimeProtocolSession, protocol_capabilities
from nemotron_voicechat_runtime.provenance import RUNTIME_SOURCE_PATHS
from nemotron_voicechat_runtime.server import (
    PINNED_COMMIT,
    PUBLIC_VLLM_MANIFEST_KINDS,
    AcousticResponseBoundary,
    AgentSilenceEosWatchdog,
    DeferredClientInputQueue,
    FunctionCallBudget,
    GatePrerollBuffer,
    PcmFrameBuffer,
    SessionTrace,
    TransportModelGate,
    UserEouSettlement,
    UserEouSettlementFailure,
    VoiceChatEngine,
    VoiceChatEngineStartFailure,
    _direct_probe_token_ids,
    accepted_vllm_manifest_kinds,
    advance_function_output_recovery,
    audio_stats,
    clean_display_text,
    clear_stale_agent_eos_latch,
    configure_fc_async_thread_dump,
    create_app,
    env_bool,
    fc_async_heartbeat_snapshot,
    float32_to_pcm16_bytes,
    function_call_loop_terminal_events,
    function_cycle_active,
    host_runtime_provenance,
    parse_nvidia_driver_version,
    pcm16_bytes_to_float32,
    prepare_agent_logit_diagnostic_nano,
    release_component_manifest,
    render_tool_system_prompt,
    response_audio_is_deliverable,
    session_position_limit_event,
    step9_capture_fallback_status,
    validate_function_output,
    validate_manifested_model,
    validate_nano_pad_pair_runtime,
    warm_realtime_engine,
    websocket_ping_interval,
    websocket_ping_timeout,
)


class RealtimeWebServerTest(unittest.TestCase):
    def test_step9_fallback_counter_is_persistent_and_reports_last_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallbacks.jsonl"
            self.assertEqual(
                step9_capture_fallback_status(path),
                {"count": 0, "last_descriptor": None},
            )
            path.write_text(
                '{"num_tokens":320,"uniform_decode":false,"query_len":0}\n'
                '{"num_tokens":900,"uniform_decode":false,"query_len":0}\n'
            )
            self.assertEqual(step9_capture_fallback_status(path)["count"], 2)
            self.assertEqual(
                step9_capture_fallback_status(path)["last_descriptor"]["num_tokens"],
                900,
            )

    @staticmethod
    def _settlement_result(
        blank_count: int,
        *,
        control: str = "pad",
        boundary: dict | None = None,
        function_calling: dict | None = None,
        assistant_delta: str = "",
        function_delta: str = "",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            frame_index=blank_count,
            assistant_text=assistant_delta,
            assistant_delta=assistant_delta,
            function_text=function_delta,
            function_delta=function_delta,
            turn_state={
                "rnnt": {"blank_count": blank_count},
                "agent_control": control,
                "response_boundary": boundary or {"event": None, "phase": "idle"},
                "function_calling": function_calling
                or {
                    "active": False,
                    "pre_eou_suppressed_tokens": 0,
                    "client_eou_sotc_committed_count": 0,
                    "post_fc_client_bos_forced_count": 0,
                },
                "vllm_request_positions": {"nano": {"session_positions": blank_count}},
            },
        )

    @staticmethod
    def _fused_settlement_result(
        *,
        blank_count: int = 3,
        marker_blank_count: int | None = None,
        function_calling: dict | None = None,
        output_dbfs: float = -120.0,
    ) -> SimpleNamespace:
        marker_blank_count = (
            blank_count if marker_blank_count is None else marker_blank_count
        )
        return SimpleNamespace(
            frame_index=17,
            assistant_text="",
            assistant_delta="",
            function_text="",
            function_delta="",
            output_audio={"rms_dbfs": output_dbfs},
            turn_state={
                "rnnt": {"blank_count": 0},
                "agent_control": "agent_bos",
                "control_ids": {"pad": 0, "agent_bos": 1},
                "client_eou_blank_fence": {
                    "frame": 17,
                    "blank_count_at_check": marker_blank_count,
                    "target_blank_frames": blank_count,
                    "chosen_token_id": 1,
                },
                "response_boundary": {"event": "start", "phase": "responding"},
                "function_calling": function_calling
                or {
                    "active": False,
                    "effective_token_is_pad": True,
                    "pre_eou_suppressed_tokens": 0,
                    "client_eou_sotc_committed_count": 0,
                    "post_fc_client_bos_forced_count": 0,
                },
                "vllm_request_positions": {"nano": {"session_positions": 17}},
            },
        )

    @staticmethod
    def _heartbeat(**overrides) -> dict:
        value = {
            "schema": 1,
            "cycle_sotc_frame": 74,
            "phase": "tool_call",
            "invocation": 1,
            "async_step": 7,
            "model_frame": 82,
            "stage": "nano",
            "state": "started",
            "stage_started_monotonic": 123.5,
            "last_completed_stage": "perception_poll",
            "last_completed_monotonic": 123.4,
            "last_failed_stage": None,
            "thread_ident": 12345,
            "live_audio_queue_depth": 0,
            "perception_queue_depth": 1,
            "tts_audio_queue_depth": 2,
            "rnnt_text_queue_depth": None,
        }
        value.update(overrides)
        return value

    def test_fc_async_heartbeat_snapshot_copies_and_validates_exact_schema(self) -> None:
        raw = self._heartbeat()
        snapshot = fc_async_heartbeat_snapshot({"async_heartbeat": raw})

        self.assertEqual(snapshot, {"loop": raw})
        self.assertIsNot(snapshot["loop"], raw)
        raw["stage"] = "changed-after-publication"
        self.assertEqual(snapshot["loop"]["stage"], "nano")

        for mutation in (
            {"stage": "unknown"},
            {"state": "unknown"},
            {"phase": "tool_response"},
            {"cycle_sotc_frame": None},
            {"stage": "complete", "state": "started"},
            {"async_step": -1},
            {"thread_ident": True},
            {"live_audio_queue_depth": 1_000_001},
            {"stage_started_monotonic": float("nan")},
            {"extra": "field"},
        ):
            malformed = self._heartbeat(**mutation)
            self.assertIsNone(
                fc_async_heartbeat_snapshot({"async_heartbeat": malformed}),
                mutation,
            )

    def test_fc_async_thread_dump_handler_is_strictly_opt_in(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "nemotron_voicechat_runtime.server.faulthandler.register"
        ) as register:
            self.assertFalse(configure_fc_async_thread_dump())
            register.assert_not_called()

        with mock.patch.dict(
            os.environ, {"S2S_FC_ASYNC_HEARTBEAT": "1"}, clear=True
        ), mock.patch(
            "nemotron_voicechat_runtime.server.faulthandler.register"
        ) as register:
            self.assertTrue(configure_fc_async_thread_dump())
            register.assert_called_once_with(signal.SIGUSR1, all_threads=True)

    def test_user_eou_settlement_core_advances_only_to_fence_with_evidence(self) -> None:
        engine = SimpleNamespace(
            audio_frame_index=0,
            user_eou_settlement_blank_frames=lambda: 3,
            rnnt_blank_count=lambda result: (
                int(result.turn_state["rnnt"]["blank_count"]) if result else 0
            ),
            perception_frame_index=lambda: engine.audio_frame_index,
        )
        settlement = UserEouSettlement.begin(engine, None)
        result = None
        for blank_count in (1, 2, 3):
            self.assertTrue(settlement.needs_step(engine, result))
            engine.audio_frame_index += 1
            result = self._settlement_result(blank_count)
            settlement.observe(engine, result)

        self.assertFalse(settlement.needs_step(engine, result))
        evidence = settlement.evidence()
        self.assertEqual(evidence["starting_blank_frames"], 0)
        self.assertEqual(evidence["ending_blank_frames"], 3)
        self.assertEqual(evidence["model_steps"], 3)
        self.assertEqual(evidence["max_model_steps"], 6)
        self.assertTrue(evidence["fence_reached"])
        self.assertTrue(evidence["pre_eou_clean"])
        self.assertEqual(
            [step["audio_frame_index_after"] for step in evidence["steps"]],
            [1, 2, 3],
        )

    def test_user_eou_settlement_core_has_exact_twice_fence_bound(self) -> None:
        engine = SimpleNamespace(
            audio_frame_index=0,
            user_eou_settlement_blank_frames=lambda: 2,
            rnnt_blank_count=lambda result: 0,
        )
        settlement = UserEouSettlement.begin(engine, None)
        stuck = self._settlement_result(0)
        for _ in range(4):
            self.assertTrue(settlement.needs_step(engine, stuck))
            settlement.observe(engine, stuck)
        with self.assertRaisesRegex(
            UserEouSettlementFailure, "did not reach the RNNT blank fence"
        ) as raised:
            settlement.needs_step(engine, stuck)
        self.assertEqual(raised.exception.code, "input_turn_settle_timeout")
        self.assertEqual(raised.exception.evidence["model_steps"], 4)

    def test_user_eou_settlement_core_allows_only_suppressed_raw_sotc(self) -> None:
        engine = SimpleNamespace(
            audio_frame_index=1,
            user_eou_settlement_blank_frames=lambda: 1,
            rnnt_blank_count=lambda result: (
                int(result.turn_state["rnnt"]["blank_count"]) if result else 0
            ),
        )
        settlement = UserEouSettlement.begin(engine, None)
        suppressed = self._settlement_result(
            1,
            function_calling={
                "active": False,
                "pre_eou_suppressed_tokens": 1,
                "pre_eou_last_raw_token_id": 20,
                "pre_eou_suppressed_frame": 7,
                "post_fc_client_bos_pending": False,
                "client_eou_sotc_committed_count": 0,
                "post_fc_client_bos_forced_count": 0,
            },
        )
        settlement.observe(engine, suppressed)
        self.assertTrue(settlement.evidence()["pre_eou_clean"])
        self.assertEqual(
            settlement.evidence()["steps"][0]["function_calling"][
                "pre_eou_last_raw_token_id"
            ],
            20,
        )

        effective = self._settlement_result(1, function_calling={"active": True})
        settlement = UserEouSettlement.begin(engine, None)
        with self.assertRaises(UserEouSettlementFailure) as raised:
            settlement.observe(engine, effective)
        self.assertEqual(raised.exception.code, "pre_eou_function_activity")

    def test_user_eou_settlement_core_rejects_response_before_publication(self) -> None:
        engine = SimpleNamespace(
            audio_frame_index=1,
            user_eou_settlement_blank_frames=lambda: 1,
            rnnt_blank_count=lambda result: (
                int(result.turn_state["rnnt"]["blank_count"]) if result else 0
            ),
        )
        settlement = UserEouSettlement.begin(engine, None)
        response = self._settlement_result(
            1,
            control="agent_bos",
            boundary={"event": "start", "phase": "responding"},
            assistant_delta="Hello",
        )
        with self.assertRaises(UserEouSettlementFailure) as raised:
            settlement.observe(engine, response)
        self.assertEqual(raised.exception.code, "pre_eou_response_activity")
        self.assertIn(
            "assistant_text_mutated",
            raised.exception.evidence["steps"][0]["violations"],
        )

    def test_user_eou_settlement_accepts_one_exact_fused_terminal_bos(self) -> None:
        tts_model = SimpleNamespace(_reset_on_bos_count=0)
        engine = SimpleNamespace(
            audio_frame_index=17,
            pipeline=SimpleNamespace(
                s2s_model=SimpleNamespace(
                    model=SimpleNamespace(tts_model=tts_model)
                )
            ),
            user_eou_settlement_blank_frames=lambda: 3,
            rnnt_blank_count=lambda result: 0,
            perception_frame_index=lambda: 17,
        )
        settlement = UserEouSettlement.begin(engine, None)
        tts_model._reset_on_bos_count = 1

        settlement.observe_fused_terminal(
            engine, self._fused_settlement_result(blank_count=3)
        )

        evidence = settlement.evidence()
        self.assertTrue(evidence["fused_terminal_bos"])
        self.assertTrue(evidence["fence_reached"])
        self.assertTrue(evidence["pre_eou_clean"])
        self.assertEqual(evidence["model_steps"], 1)
        self.assertEqual(evidence["steps"][0]["rnnt_blank_frames"], 3)
        self.assertTrue(evidence["steps"][0]["terminal_fused_bos"])
        self.assertFalse(settlement.needs_step(engine, None))

    def test_user_eou_settlement_accepts_one_prepared_reuse_bos_epoch(self) -> None:
        backend_request_id = "1__eartts_generation_5"
        tts_model = SimpleNamespace(
            _reset_on_bos_count=4,
            tts_model=SimpleNamespace(
                engine=SimpleNamespace(
                    requests={
                        "1": SimpleNamespace(
                            backend_request_id=backend_request_id,
                            backend_generation=5,
                        )
                    }
                )
            ),
        )
        status = {
            "bos_transition_count": 8,
            "prepared_reuse_count": 2,
            "session_epoch": 2,
            "prepare_epoch": 5,
            "state": "clean_armed",
            "armed_client_turn_id": 17,
            "last_bos_event": None,
        }
        wrapper = SimpleNamespace(
            model=SimpleNamespace(tts_model=tts_model),
            _eartts_prepared_epoch_status=lambda: dict(status),
        )
        engine = SimpleNamespace(
            audio_frame_index=17,
            stream_id=1,
            pipeline=SimpleNamespace(
                s2s_model=wrapper,
                _request_id_for_stream=lambda _stream_id: "1",
            ),
            user_eou_settlement_blank_frames=lambda: 3,
            rnnt_blank_count=lambda result: 0,
        )
        settlement = UserEouSettlement.begin(
            engine, None, expected_client_turn_id=17
        )
        status.update(
            bos_transition_count=9,
            prepared_reuse_count=3,
            state="dirty",
            armed_client_turn_id=None,
            last_bos_event={
                "schema": 2,
                "mode": "prepared_reuse",
                "request_id": "1",
                "backend_request_id": backend_request_id,
                "backend_generation": 5,
                "armed_client_turn_id": 17,
                "session_epoch": 2,
                "prepare_epoch": 5,
                "reset_count": 4,
                "transition_count": 9,
                "reuse_count": 3,
                "fallback_reason": None,
            },
        )

        settlement.observe_fused_terminal(
            engine, self._fused_settlement_result(blank_count=3)
        )

        evidence = settlement.evidence()
        self.assertEqual(evidence["schema"], 2)
        self.assertEqual(evidence["terminal_bos_epoch"]["reuse_count_after"], 3)
        self.assertEqual(
            evidence["terminal_bos_epoch"]["last_bos_event"]["armed_client_turn_id"],
            17,
        )

    def test_deferred_tool_bos_uses_step_local_epoch_after_internal_resets(self) -> None:
        request = SimpleNamespace(
            backend_request_id="1__eartts_generation_7",
            backend_generation=7,
        )
        tts_model = SimpleNamespace(
            _reset_on_bos_count=0,
            tts_model=SimpleNamespace(
                engine=SimpleNamespace(requests={"1": request})
            ),
        )
        status = {
            "bos_transition_count": 3,
            "prepared_reuse_count": 3,
            "session_epoch": 2,
            "prepare_epoch": 5,
            "state": "dirty",
            "armed_client_turn_id": None,
            "last_bos_event": None,
        }
        wrapper = SimpleNamespace(
            model=SimpleNamespace(tts_model=tts_model),
            _eartts_prepared_epoch_status=lambda: dict(status),
        )
        engine = SimpleNamespace(
            audio_frame_index=17,
            stream_id=1,
            pipeline=SimpleNamespace(
                s2s_model=wrapper,
                _request_id_for_stream=lambda _stream_id: "1",
            ),
            user_eou_settlement_blank_frames=lambda: 3,
            rnnt_blank_count=lambda result: 3 if result else 0,
        )
        settlement = UserEouSettlement.begin(engine, None)
        settlement.deferred_terminal_bos = True

        # Function acknowledgement/reminder audio legitimately rotates EarTTS
        # twice before the single post-FC terminal BOS model step.
        tts_model._reset_on_bos_count = 3
        request.backend_request_id = "1__eartts_generation_10"
        request.backend_generation = 10
        status.update(
            bos_transition_count=6,
            prepared_reuse_count=3,
            state="dirty",
            last_bos_event={
                "schema": 2,
                "mode": "legacy_fallback",
                "request_id": "1",
                "backend_request_id": request.backend_request_id,
                "backend_generation": request.backend_generation,
                "armed_client_turn_id": None,
                "session_epoch": 2,
                "prepare_epoch": 5,
                "reset_count": 3,
                "transition_count": 6,
                "reuse_count": 3,
                "fallback_reason": "prepared_epoch_not_armed",
            },
        )
        result = self._settlement_result(
            3,
            control="agent_bos",
            boundary={"event": "start", "phase": "responding"},
        )
        result.turn_state["eartts_bos_epoch_step"] = {
            "schema": 1,
            "reset_count_before": 2,
            "reset_count_after": 3,
            "transition_count_before": 5,
            "transition_count_after": 6,
            "reuse_count_before": 3,
            "reuse_count_after": 3,
            "session_epoch_before": 2,
            "session_epoch_after": 2,
            "prepare_epoch_before": 5,
            "prepare_epoch_after": 5,
            "request_id_before": "1",
            "request_id_after": "1",
            "backend_request_id_before": "1__eartts_generation_9",
            "backend_request_id_after": "1__eartts_generation_10",
            "backend_generation_before": 9,
            "backend_generation_after": 10,
        }

        settlement.observe_separate_terminal(engine, result)

        evidence = settlement.evidence()["terminal_bos_epoch"]
        self.assertEqual(
            evidence["preterminal_fc_deltas"],
            {"reset": 2, "transition": 2, "reuse": 0},
        )
        self.assertEqual(
            evidence["terminal_step_deltas"],
            {"reset": 1, "transition": 1, "reuse": 0},
        )
        self.assertEqual(
            evidence["terminal_step_start"]["backend_request_id_before"],
            "1__eartts_generation_9",
        )

        mutations = {
            "missing_transition": (
                {"transition_count_before": 6},
                "eartts_bos_transition_count",
            ),
            "internal_reuse": (
                {"reuse_count_before": 4, "reuse_count_after": 3},
                "eartts_preterminal_fc_accounting",
            ),
            "wrong_physical_after": (
                {
                    "backend_request_id_after": "1__eartts_generation_11",
                    "backend_generation_after": 11,
                },
                "eartts_bos_step_after_identity_stale",
            ),
            "invalid_physical_before": (
                {
                    "backend_request_id_before": "1",
                    "backend_generation_before": 0,
                },
                "eartts_bos_step_physical_identity_invalid",
            ),
        }
        for name, (mutation, violation) in mutations.items():
            with self.subTest(name=name):
                candidate = UserEouSettlement.begin(engine, None)
                # Recreate the original pre-FC counts after begin observed the
                # current terminal state.
                candidate.eartts_reset_count_starting = 0
                candidate.eartts_transition_count_starting = 3
                candidate.eartts_reuse_count_starting = 3
                candidate.deferred_terminal_bos = True
                bad = self._settlement_result(
                    3,
                    control="agent_bos",
                    boundary={"event": "start", "phase": "responding"},
                )
                bad_step = dict(result.turn_state["eartts_bos_epoch_step"])
                bad_step.update(mutation)
                bad.turn_state["eartts_bos_epoch_step"] = bad_step
                with self.assertRaises(UserEouSettlementFailure) as raised:
                    candidate.observe_separate_terminal(engine, bad)
                self.assertIn(
                    violation,
                    raised.exception.evidence["terminal_violations"],
                )

        # A prepared reuse can be valid for an immediate/fused turn, but is
        # impossible after FC audio has dirtied the epoch. Keep that distinction
        # explicit even if every other count and identity is forged coherently.
        tts_model._reset_on_bos_count = 2
        request.backend_request_id = "1__eartts_generation_9"
        request.backend_generation = 9
        status.update(
            bos_transition_count=6,
            prepared_reuse_count=4,
            state="dirty",
            last_bos_event={
                "schema": 2,
                "mode": "prepared_reuse",
                "request_id": "1",
                "backend_request_id": request.backend_request_id,
                "backend_generation": request.backend_generation,
                "armed_client_turn_id": 17,
                "session_epoch": 2,
                "prepare_epoch": 5,
                "reset_count": 2,
                "transition_count": 6,
                "reuse_count": 4,
                "fallback_reason": None,
            },
        )
        forged = UserEouSettlement.begin(engine, None, expected_client_turn_id=17)
        forged.eartts_reset_count_starting = 0
        forged.eartts_transition_count_starting = 3
        forged.eartts_reuse_count_starting = 3
        forged.deferred_terminal_bos = True
        forged_result = self._settlement_result(
            3,
            control="agent_bos",
            boundary={"event": "start", "phase": "responding"},
        )
        forged_result.turn_state["eartts_bos_epoch_step"] = {
            "schema": 1,
            "reset_count_before": 2,
            "reset_count_after": 2,
            "transition_count_before": 5,
            "transition_count_after": 6,
            "reuse_count_before": 3,
            "reuse_count_after": 4,
            "session_epoch_before": 2,
            "session_epoch_after": 2,
            "prepare_epoch_before": 5,
            "prepare_epoch_after": 5,
            "request_id_before": "1",
            "request_id_after": "1",
            "backend_request_id_before": "1__eartts_generation_9",
            "backend_request_id_after": "1__eartts_generation_9",
            "backend_generation_before": 9,
            "backend_generation_after": 9,
        }
        with self.assertRaises(UserEouSettlementFailure) as raised:
            forged.observe_separate_terminal(engine, forged_result)
        self.assertIn(
            "eartts_deferred_bos_not_legacy",
            raised.exception.evidence["terminal_violations"],
        )

    def test_user_eou_settlement_rejects_invalid_bos_epoch_accounting(self) -> None:
        mutations = {
            "missing_transition": (0, 0, 0, "eartts_bos_transition_count"),
            "mixed_reset_and_reuse": (1, 1, 1, "eartts_bos_transition_accounting"),
            "wrong_turn": (1, 0, 1, "eartts_bos_reuse_event_invalid"),
        }
        for name, (transition_delta, reset_delta, reuse_delta, violation) in mutations.items():
            with self.subTest(name=name):
                tts_model = SimpleNamespace(_reset_on_bos_count=2)
                status = {
                    "bos_transition_count": 4,
                    "prepared_reuse_count": 1,
                    "state": "clean_armed",
                    "armed_client_turn_id": 7,
                    "last_bos_event": None,
                }
                wrapper = SimpleNamespace(
                    model=SimpleNamespace(tts_model=tts_model),
                    _eartts_prepared_epoch_status=lambda: dict(status),
                )
                engine = SimpleNamespace(
                    audio_frame_index=17,
                    stream_id=1,
                    pipeline=SimpleNamespace(
                        s2s_model=wrapper,
                        _request_id_for_stream=lambda _stream_id: "1",
                    ),
                    user_eou_settlement_blank_frames=lambda: 3,
                    rnnt_blank_count=lambda result: 0,
                )
                settlement = UserEouSettlement.begin(
                    engine, None, expected_client_turn_id=7
                )
                tts_model._reset_on_bos_count += reset_delta
                status.update(
                    bos_transition_count=4 + transition_delta,
                    prepared_reuse_count=1 + reuse_delta,
                    state="dirty",
                    armed_client_turn_id=None,
                    last_bos_event={
                        "schema": 1,
                        "mode": "prepared_reuse" if reuse_delta else "legacy_fallback",
                        "request_id": "1",
                        "armed_client_turn_id": 99 if name == "wrong_turn" else (
                            7 if reuse_delta else None
                        ),
                        "session_epoch": 2,
                        "prepare_epoch": 5,
                        "reset_count": tts_model._reset_on_bos_count,
                        "transition_count": 4 + transition_delta,
                        "reuse_count": 1 + reuse_delta,
                        "fallback_reason": None if reuse_delta else "not_armed",
                    },
                )
                with self.assertRaises(UserEouSettlementFailure) as raised:
                    settlement.observe_fused_terminal(
                        engine, self._fused_settlement_result(blank_count=3)
                    )
                self.assertIn(
                    violation,
                    raised.exception.evidence["steps"][0]["violations"],
                )

    def test_user_eou_settlement_binds_bos_event_request_and_epochs(self) -> None:
        mutations = {
            "request_id": ("wrong", "eartts_bos_event_request_mismatch"),
            "session_epoch": (3, "eartts_bos_event_session_epoch_mismatch"),
            "prepare_epoch": (6, "eartts_bos_event_prepare_epoch_mismatch"),
        }
        for key, (bad_value, violation) in mutations.items():
            with self.subTest(key=key):
                tts_model = SimpleNamespace(_reset_on_bos_count=4)
                status = {
                    "bos_transition_count": 8,
                    "prepared_reuse_count": 2,
                    "session_epoch": 2,
                    "prepare_epoch": 5,
                    "state": "clean_armed",
                    "armed_client_turn_id": 17,
                    "last_bos_event": None,
                }
                wrapper = SimpleNamespace(
                    model=SimpleNamespace(tts_model=tts_model),
                    _eartts_prepared_epoch_status=lambda: dict(status),
                )
                engine = SimpleNamespace(
                    audio_frame_index=17,
                    stream_id=1,
                    pipeline=SimpleNamespace(
                        s2s_model=wrapper,
                        _request_id_for_stream=lambda _stream_id: "1",
                    ),
                    user_eou_settlement_blank_frames=lambda: 3,
                    rnnt_blank_count=lambda result: 0,
                )
                settlement = UserEouSettlement.begin(
                    engine, None, expected_client_turn_id=17
                )
                event = {
                    "schema": 1,
                    "mode": "prepared_reuse",
                    "request_id": "1",
                    "armed_client_turn_id": 17,
                    "session_epoch": 2,
                    "prepare_epoch": 5,
                    "reset_count": 4,
                    "transition_count": 9,
                    "reuse_count": 3,
                    "fallback_reason": None,
                }
                event[key] = bad_value
                status.update(
                    bos_transition_count=9,
                    prepared_reuse_count=3,
                    state="dirty",
                    armed_client_turn_id=None,
                    last_bos_event=event,
                )
                with self.assertRaises(UserEouSettlementFailure) as raised:
                    settlement.observe_fused_terminal(
                        engine, self._fused_settlement_result(blank_count=3)
                    )
                self.assertIn(
                    violation,
                    raised.exception.evidence["steps"][0]["violations"],
                )

    def test_user_eou_settlement_rejects_mutated_fused_terminal_contract(self) -> None:
        cases = {
            "blank_fence_not_proven": {"marker_blank_count": 2},
            "function_cycle_effective": {
                "function_calling": {
                    "active": True,
                    "effective_token_is_pad": False,
                }
            },
            "terminal_audio_not_silent": {"output_dbfs": -20.0},
            "eartts_bos_reset_count": {"reset_count": 0},
        }
        for expected_violation, mutation in cases.items():
            with self.subTest(expected_violation=expected_violation):
                tts_model = SimpleNamespace(_reset_on_bos_count=0)
                engine = SimpleNamespace(
                    audio_frame_index=17,
                    pipeline=SimpleNamespace(
                        s2s_model=SimpleNamespace(
                            model=SimpleNamespace(tts_model=tts_model)
                        )
                    ),
                    user_eou_settlement_blank_frames=lambda: 3,
                    rnnt_blank_count=lambda result: 0,
                )
                settlement = UserEouSettlement.begin(engine, None)
                tts_model._reset_on_bos_count = mutation.pop("reset_count", 1)
                with self.assertRaises(UserEouSettlementFailure) as raised:
                    settlement.observe_fused_terminal(
                        engine,
                        self._fused_settlement_result(
                            blank_count=3,
                            **mutation,
                        ),
                    )
                self.assertEqual(raised.exception.code, "fused_eou_contract_failure")
                self.assertIn(
                    expected_violation,
                    raised.exception.evidence["steps"][0]["violations"],
                )

    def test_function_output_is_capped_before_model_injection(self) -> None:
        seen = []

        def build_tokens(text):
            seen.append(text)
            return [1] * (129 if "too-many" in text else 37)

        engine = SimpleNamespace(
            pipeline=SimpleNamespace(
                s2s_model=SimpleNamespace(
                    _fc_convert_num_to_text=True,
                    _convert_tool_response_nums_to_text=lambda text: text.replace("18", "eighteen"),
                    _build_fc_response_tokens=build_tokens,
                )
            )
        )

        validated = validate_function_output(engine, '{"value":18}')
        self.assertEqual(validated.output, '{"value":18}')
        self.assertEqual(validated.injection_tokens, 37)
        self.assertEqual(validated.injection_output, '{"value":18}')
        self.assertEqual(seen[-1], '<TOOL_RESPONSE>[{"value":eighteen}]</TOOL_RESPONSE>')
        with self.assertRaisesRegex(ValueError, "must be a string"):
            validate_function_output(engine, {"value": 18})
        with self.assertRaisesRegex(ValueError, "16384 UTF-8 bytes"):
            validate_function_output(engine, "x" * 16_385)
        with self.assertRaisesRegex(ValueError, "129 tokens; maximum is 128"):
            validate_function_output(engine, "too-many")

        concise = validate_function_output(
            engine,
            "full-output-too-many",
            model_output="short",
            model_output_negotiated=True,
        )
        self.assertEqual(concise.output_tokens, 129)
        self.assertEqual(concise.model_output_tokens, 37)
        self.assertEqual(concise.injection_output, "short")
        with self.assertRaisesRegex(ValueError, "was not negotiated"):
            validate_function_output(engine, "full", model_output="short")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_function_output(
                engine,
                "full",
                model_output="  ",
                model_output_negotiated=True,
            )

    def test_function_output_recovery_is_paced_until_background_clears(self) -> None:
        busy = SimpleNamespace(turn_state={"function_calling": {"background_active": True}})
        clear = SimpleNamespace(turn_state={"function_calling": {}})
        results = iter((busy, busy, clear))
        calls = 0
        delays = []

        async def step():
            nonlocal calls
            calls += 1
            return next(results)

        async def record_sleep(delay):
            delays.append(delay)

        with mock.patch(
            "nemotron_voicechat_runtime.server.asyncio.sleep",
            side_effect=record_sleep,
        ):
            applied = asyncio.run(
                advance_function_output_recovery(
                    step,
                    max_frames=3,
                    frame_seconds=0.08,
                )
            )

        self.assertTrue(applied)
        self.assertEqual(calls, 3)
        self.assertEqual(len(delays), 2)
        self.assertTrue(all(delay > 0.0 for delay in delays))
        self.assertTrue(function_cycle_active(busy))
        self.assertTrue(
            function_cycle_active(
                SimpleNamespace(turn_state={"function_calling": {"awaiting_eotr": True}})
            )
        )
        self.assertFalse(function_cycle_active(clear))

    def test_function_output_recovery_fails_after_bounded_frames(self) -> None:
        calls = 0

        async def step():
            nonlocal calls
            calls += 1
            return SimpleNamespace(turn_state={"function_calling": {"injecting_response": True}})

        applied = asyncio.run(
            advance_function_output_recovery(step, max_frames=4, frame_seconds=0.0)
        )

        self.assertFalse(applied)
        self.assertEqual(calls, 4)

    def test_function_output_recovery_waits_for_deferred_terminal_completion(self) -> None:
        clear = SimpleNamespace(turn_state={"function_calling": {}})
        results = iter((clear, clear))
        calls = 0

        async def step():
            nonlocal calls
            calls += 1
            return next(results)

        applied = asyncio.run(
            advance_function_output_recovery(
                step,
                max_frames=2,
                frame_seconds=0.0,
                completion=lambda _result: calls == 2,
            )
        )
        self.assertTrue(applied)
        self.assertEqual(calls, 2)

    def test_prepared_epoch_uses_pipeline_request_id_mapping(self) -> None:
        prepared_ids = []
        armed = []
        wrapper = SimpleNamespace(
            _prepare_eartts_prepared_epoch=lambda request_id, **_kwargs: (
                prepared_ids.append(request_id) or {"passed": True}
            ),
            _arm_eartts_prepared_epoch=lambda request_id, turn_id, **_kwargs: (
                armed.append((request_id, turn_id)) or {"passed": True}
            ),
        )
        context = SimpleNamespace(
            fc_state={}, tool_response_text=None, tool_response_queue=[]
        )
        pipeline = SimpleNamespace(
            s2s_model=wrapper,
            _fc_async_bg={},
            _request_id_for_stream=lambda stream_id: f"mapped-request-{stream_id}",
            context_manager=SimpleNamespace(
                streamidx2slotidx={7: 0}, slot_contexts=[context]
            ),
        )
        engine = VoiceChatEngine(pipeline)
        engine.started = True
        engine.stream_id = 7

        self.assertTrue(engine.prepare_eartts_epoch()["passed"])
        self.assertTrue(engine.arm_eartts_epoch(19)["passed"])
        self.assertEqual(prepared_ids, ["mapped-request-7"])
        self.assertEqual(armed, [("mapped-request-7", 19)])

    def test_direct_probe_token_forms_are_explicit(self) -> None:
        pipeline = SimpleNamespace(
            s2s_model=SimpleNamespace(
                tokenizer=SimpleNamespace(text_to_ids=lambda text: [101, 102]),
                model=SimpleNamespace(stt_model=SimpleNamespace(user_bos_id=13, user_eos_id=14)),
            )
        )

        self.assertEqual(_direct_probe_token_ids(pipeline, "hello", "text"), [101, 102])
        self.assertEqual(
            _direct_probe_token_ids(pipeline, "hello", "user_bos_text"),
            [13, 101, 102],
        )
        self.assertEqual(
            _direct_probe_token_ids(pipeline, "hello", "text_user_eos"),
            [101, 102, 14],
        )
        self.assertEqual(
            _direct_probe_token_ids(pipeline, "hello", "user_bos_text_user_eos"),
            [13, 101, 102, 14],
        )
        with self.assertRaisesRegex(ValueError, "unsupported direct probe token form"):
            _direct_probe_token_ids(pipeline, "hello", "implicit")

    def test_user_eou_settlement_uses_pinned_rnnt_phrase_fence(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(
            s2s_model=SimpleNamespace(model_cfg={"nonblank_reset_after_silence": 10})
        )
        result = SimpleNamespace(turn_state={"rnnt": {"blank_count": 6}})

        self.assertEqual(engine.user_eou_settlement_blank_frames(), 10)
        self.assertEqual(engine.rnnt_blank_count(result), 6)
        self.assertEqual(engine.rnnt_blank_count(None), 0)

    def test_user_eou_settlement_rejects_disabled_phrase_fence(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(
            s2s_model=SimpleNamespace(model_cfg={"nonblank_reset_after_silence": 0})
        )

        with self.assertRaisesRegex(RuntimeError, "must be positive"):
            engine.user_eou_settlement_blank_frames()

    def test_user_transcript_survives_mid_turn_rnnt_hypothesis_clear(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.user_text = ""
        engine.user_text_prefix = ""
        engine.user_text_segment = ""
        engine.external_user_eou_mode = True

        engine._update_user_transcript("what is two plus three?", 6)
        self.assertEqual(engine.user_text, "what is two plus three?")

        # The pinned pipeline clears y_sequence after 800 ms of blanks. Smart
        # Turn still owns the open user turn, so this is only a phrase boundary.
        engine._update_user_transcript("", 0)
        engine._update_user_transcript("answer in one short sent", 5)
        engine._update_user_transcript("answer in one short sentence", 6)

        self.assertEqual(
            engine.user_text,
            "what is two plus three? answer in one short sentence",
        )

    def test_fused_eou_ignores_same_position_stale_rnnt_display(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.user_text = ""
        engine.user_text_prefix = ""
        engine.user_text_segment = ""
        engine.external_user_eou_mode = True

        phrase = "remember that the code word is sapphire"
        engine._update_user_transcript(phrase, 8)
        engine._update_user_transcript(
            phrase,
            0,
            suppress_stale_reset_display=True,
        )

        self.assertEqual(engine.user_text, phrase)
        self.assertEqual(engine.user_text_prefix, phrase)
        self.assertEqual(engine.user_text_segment, "")

    def test_legacy_user_transcript_does_not_accumulate_across_phrase_clears(
        self,
    ) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.user_text = ""
        engine.user_text_prefix = ""
        engine.user_text_segment = ""
        engine.external_user_eou_mode = False

        engine._update_user_transcript("yes", 1)
        engine._update_user_transcript("", 0)
        engine._update_user_transcript("yes", 1)

        self.assertEqual(engine.user_text, "yes")
        self.assertEqual(engine.user_text_prefix, "")
        self.assertEqual(engine.user_text_segment, "")

    def test_user_transcript_reset_clears_segment_accumulator(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.user_text = "old text"
        engine.user_text_prefix = "old"
        engine.user_text_segment = "text"
        engine.last_rnnt_decoded_count = 2
        state = SimpleNamespace(output_asr_text_str="old text", _last_sent_asr_text="old")
        engine.pipeline = SimpleNamespace(get_or_create_state=lambda _stream_id: state)
        engine.stream_id = 4

        engine.reset_user_transcript()

        self.assertEqual(engine.user_text, "")
        self.assertEqual(engine.user_text_prefix, "")
        self.assertEqual(engine.user_text_segment, "")
        self.assertEqual(engine.last_rnnt_decoded_count, 0)
        self.assertEqual(state.output_asr_text_str, "")
        self.assertIsNone(state._last_sent_asr_text)

    def test_defensive_function_input_queue_is_fifo_and_byte_bounded(self) -> None:
        queue = DeferredClientInputQueue(max_bytes=10)
        first = {"type": "input_audio_buffer.turn_start"}
        second = {"type": "input_audio_buffer.append"}

        self.assertTrue(queue.append(first, 4))
        self.assertTrue(queue.append(second, 6))
        self.assertFalse(queue.append({"type": "input_audio_buffer.commit"}, 1))
        self.assertEqual(queue.bytes, 10)
        self.assertEqual(queue.popleft(), first)
        self.assertEqual(queue.popleft(), second)
        self.assertEqual(queue.bytes, 0)

    def test_pad_pair_trace_failure_is_observability_only(self) -> None:
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pad_pair_trace_errors = 0

        with (
            mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True),
            mock.patch.object(
                engine,
                "_collect_pad_pair_trace_diagnostics",
                side_effect=RuntimeError("diagnostic failure"),
            ),
        ):
            diagnostics = engine._pad_pair_trace_diagnostics(None, -1, {})

        self.assertEqual(diagnostics["trace_error"], "RuntimeError")
        self.assertEqual(diagnostics["trace_record_errors"], 1)
        self.assertEqual(engine.pad_pair_trace_errors, 1)

    def test_pad_pair_trace_watchdog_captures_nonpad_function_channel(self) -> None:
        import nemotron_voicechat_runtime.runtime_optimizations as optimizations

        wrapper = SimpleNamespace(model=SimpleNamespace(stt_model=SimpleNamespace(text_pad_id=12)))
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(
            s2s_model=wrapper,
            _request_id_for_stream=lambda _stream_id: "request-5",
        )
        engine.stream_id = 5
        engine.pad_pair_idle_no_buffer_frames = 0
        engine.pad_pair_watchdog_emitted = False
        context = SimpleNamespace(
            gen_text=np.asarray([[12]]),
            gen_function_text=np.asarray([[77]]),
        )
        turn_state = {
            "agent_control": "pad",
            "function_calling": {
                "active": False,
                "awaiting_response": False,
                "injecting_response": False,
                "forced_tokens": 0,
                "background_active": False,
            },
        }

        def snapshot(*_args):
            return {
                "enabled": True,
                "events": [],
                "last_effective_tokens": {
                    "frame_idx": 0,
                    "text": 12,
                    "function": 77,
                    "pad": 12,
                    "both_pad": False,
                },
                "state": {"previous_effective_pad": False},
            }

        with (
            mock.patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_TRACE": "1"}, clear=True),
            mock.patch.object(optimizations, "consume_pad_pair_trace", side_effect=snapshot),
        ):
            diagnostics = None
            for _ in range(25):
                diagnostics = engine._pad_pair_trace_diagnostics(context, 0, turn_state)

        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics["idle_no_buffer_frames"], 25)
        self.assertFalse(diagnostics["effective_tokens"]["function_is_pad"])
        self.assertTrue(diagnostics["effective_tokens_match_finalize"])
        self.assertEqual(diagnostics["watchdog"]["event"], "idle_agent_without_pair_buffer")

    def test_turn_state_exposes_current_pre_eou_function_suppression(self) -> None:
        pending_trace = [
            {
                "sequence": 1,
                "phase": "async_eotr",
                "model_frame": 7,
                "raw_agent_token_id": 1,
                "effective_agent_token_id": 12,
            }
        ]

        def drain_trace():
            records = list(pending_trace)
            pending_trace.clear()
            return records

        wrapper = SimpleNamespace(
            model=SimpleNamespace(
                stt_model=SimpleNamespace(
                    text_pad_id=12,
                    text_bos_id=13,
                    text_eos_id=14,
                    user_bos_id=15,
                )
            ),
            _pre_eou_function_suppression_count=2,
            _pre_eou_function_last_raw_token_id=20,
            _pre_eou_function_suppressed_frame=0,
            _agent_open_function_suppression_count=3,
            _agent_open_function_last_raw_token_id=12068,
            _agent_open_function_suppressed_frame=4,
            _client_eou_sotc_committed_frame=0,
            _client_eou_sotc_committed_count=1,
            _post_fc_client_bos_requested=True,
            _post_fc_client_bos_requested_frame=0,
            _post_fc_client_bos_forced_frame=None,
            _post_fc_client_bos_forced_count=0,
            _agent_eos_just_fired=False,
            drain_post_fc_token_trace=drain_trace,
        )
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(s2s_model=wrapper, _fc_async_bg={})
        engine.stream_id = 0
        context = SimpleNamespace(
            frame_idx=1,
            gen_text=np.asarray([[12]]),
            gen_asr_text=np.asarray([[12]]),
            rnnt_partial_hypotheses={
                "agent_speaking": np.asarray([False]),
                "blank_count": np.asarray([10]),
                "y_sequence": [],
            },
            fc_state={
                "active": False,
                "awaiting_response": False,
                "injecting_response": False,
                "forced_function_tokens": [],
                "completed_calls": [],
            },
            perception_cache=None,
        )

        state = engine._turn_state(context, 0)

        self.assertEqual(state["function_calling"]["pre_eou_suppressed_tokens"], 2)
        self.assertEqual(state["function_calling"]["pre_eou_last_raw_token_id"], 20)
        self.assertEqual(state["function_calling"]["pre_eou_suppressed_frame"], 0)
        self.assertNotIn("pre_eou_suppressed_this_frame", state["function_calling"])
        self.assertEqual(state["function_calling"]["agent_open_suppressed_tokens"], 3)
        self.assertEqual(
            state["function_calling"]["agent_open_last_raw_token_id"], 12068
        )
        self.assertEqual(state["function_calling"]["agent_open_suppressed_frame"], 4)
        self.assertTrue(state["function_calling"]["post_fc_client_bos_pending"])
        self.assertEqual(state["function_calling"]["client_eou_sotc_committed_count"], 1)
        self.assertEqual(state["function_calling"]["client_eou_sotc_committed_frame"], 0)
        self.assertEqual(state["function_calling"]["post_fc_client_bos_requested_frame"], 0)
        self.assertEqual(state["function_calling"]["post_fc_client_bos_forced_count"], 0)
        self.assertIsNone(state["function_calling"]["post_fc_client_bos_forced_frame"])
        self.assertEqual(
            state["function_calling"]["post_fc_token_trace"][0]["phase"],
            "async_eotr",
        )
        second_state = engine._turn_state(context, 0)
        self.assertNotIn("post_fc_token_trace", second_state["function_calling"])
        self.assertFalse(function_cycle_active(SimpleNamespace(turn_state=state)))

    def test_turn_state_stage_timing_is_frame_bound_and_additive(self) -> None:
        wrapper = SimpleNamespace(
            model=SimpleNamespace(
                stt_model=SimpleNamespace(
                    text_pad_id=12,
                    text_bos_id=1,
                    text_eos_id=2,
                    user_bos_id=1094,
                )
            )
        )
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(
            s2s_model=wrapper,
            _fc_async_bg={},
            _request_id_for_stream=lambda _stream_id: "2",
        )
        engine.stream_id = 0
        context = SimpleNamespace(
            frame_idx=1,
            gen_text=np.asarray([[12]]),
            gen_asr_text=np.asarray([[12]]),
            rnnt_partial_hypotheses=None,
            fc_state=None,
            perception_cache=None,
            last_step_stage_timings=None,
        )
        baseline = engine._turn_state(context, 0)
        timings = {
            "schema": 2,
            "measurement": "host_intervals_between_existing_cuda_syncs",
            "frame_start": 0,
            "frame_end": 0,
            "positions": 1,
            "perception_ms": 2.0,
            "nano_interface_ms": 5.0,
            "eartts_total_ms": 7.0,
            "eartts_reset": {
                "schema": 1,
                "request_id": "2",
                "reset_count": 1,
                "abort_call_ms": 1.0,
                "prefill_call_ms": 2.0,
                "reset_total_ms": 3.0,
                "eartts_non_reset_remainder_ms": 4.0,
            },
            "eartts_idle_pad_bypass": {
                "schema": 1,
                "enabled": True,
                "mode": "enabled_no_skip",
                "reason": None,
                "skipped_positions": 0,
                "skipped_frames": [],
                "cumulative_skipped_positions": 4,
                "recurrent_state": None,
                "decoder_code": None,
            },
            "codec_call_ms": 3.0,
            "codec_pipeline": "waits_for_previous_frame_and_submits_current",
            "named_stage_sum_ms": 17.0,
            "wrapper_residual_ms": 3.0,
            "wrapper_total_ms": 20.0,
        }
        context.last_step_stage_timings = timings
        instrumented = engine._turn_state(context, 0)
        self.assertEqual(instrumented["model_stage_timings_ms"], timings)
        self.assertEqual(
            {key: value for key, value in instrumented.items() if key != "model_stage_timings_ms"},
            baseline,
        )

        context.last_step_stage_timings = {**timings, "frame_start": 1, "frame_end": 1}
        self.assertNotIn("model_stage_timings_ms", engine._turn_state(context, 0))

    def test_model_stage_timing_validation_rejects_bad_values(self) -> None:
        valid = {
            "schema": 2,
            "measurement": "host_intervals_between_existing_cuda_syncs",
            "frame_start": 4,
            "frame_end": 5,
            "positions": 2,
            "perception_ms": 2.0,
            "nano_interface_ms": 5.0,
            "eartts_total_ms": 7.0,
            "eartts_reset": None,
            "eartts_idle_pad_bypass": {
                "schema": 1,
                "enabled": True,
                "mode": "bypassed",
                "reason": "ordinary_pcm_agent_idle_effective_pad_fc_quiescent",
                "skipped_positions": 1,
                "skipped_frames": [4],
                "cumulative_skipped_positions": 9,
                "recurrent_state": "preserved_input_identity",
                "decoder_code": "codec_silence_tokens_clone",
            },
            "codec_call_ms": 3.0,
            "codec_pipeline": "waits_for_previous_frame_and_submits_current",
            "named_stage_sum_ms": 17.0,
            "wrapper_residual_ms": 3.0,
            "wrapper_total_ms": 20.0,
        }
        for mutation in (
            {"perception_ms": float("nan")},
            {"nano_interface_ms": -1.0},
            {"named_stage_sum_ms": 18.0},
            {"wrapper_total_ms": 16.0},
            {"codec_pipeline": "not-pipelined"},
            {
                "eartts_idle_pad_bypass": {
                    **valid["eartts_idle_pad_bypass"],
                    "skipped_frames": [6],
                }
            },
            {
                "eartts_idle_pad_bypass": {
                    **valid["eartts_idle_pad_bypass"],
                    "mode": "enabled_no_skip",
                }
            },
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(RuntimeError):
                    VoiceChatEngine._validated_model_stage_timings(
                        {**valid, **mutation}, 5
                    )

    def test_wrapper_session_reset_clears_stage_reset_timing(self) -> None:
        tts_model = SimpleNamespace(_reset_on_bos_last_timing_ms={"stale": True})
        reset = mock.Mock()
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = SimpleNamespace(
            s2s_model=SimpleNamespace(
                reset_transport_session_state=reset,
                model=SimpleNamespace(tts_model=tts_model),
            )
        )

        engine._reset_wrapper_session_state()

        reset.assert_called_once_with()
        self.assertIsNone(tts_model._reset_on_bos_last_timing_ms)

    def test_speech_patch_idle_pad_bypass_preserves_recurrent_state(self) -> None:
        patch = (
            Path(__file__).resolve().parents[2]
            / "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()
        branch_start = patch.index("if bypass_idle_pad:")
        branch_end = patch.index("else:", branch_start)
        branch = patch[branch_start:branch_end]
        self.assertIn("decoder_code, preserved_code, preserved_past_key_values", branch)
        self.assertIn("preserved_code is not code", branch)
        self.assertIn("preserved_past_key_values is not past_key_values", branch)
        self.assertNotIn("infer_codes_one_step", branch)
        self.assertNotIn("torch.cuda.synchronize", branch)
        self.assertIn("source_embeddings_supplied=source_embeddings is not None", patch)
        self.assertIn("force_output_pad=bool(force_output_pad)", patch)
        self.assertIn("fc_state=fc_state", patch)
        self.assertIn("new_codes_for_decode.append(decoder_code.clone())", patch)
        self.assertIn("[audio_toks_buffer[:, 1:], decoder_code]", patch)
        self.assertIn("self._eartts_idle_pad_bypass_total = 0", patch)
        self.assertIn("'schema': 2", patch)

    def test_realtime_warmup_exercises_third_frame_and_closes_session(self) -> None:
        from types import SimpleNamespace

        class Engine:
            started = False

            def __init__(self) -> None:
                self.calls = []

            def start(self, system_prompt="") -> None:
                self.started = True
                self.system_prompt = system_prompt

            def process(self, samples, *, is_last, input_active):
                self.calls.append((samples.copy(), is_last, input_active))
                if is_last:
                    self.started = False
                return SimpleNamespace(inference_ms=10.0 + len(self.calls))

            def abort(self) -> None:
                raise AssertionError("clean warmup must close through is_last")

        engine = Engine()
        result = warm_realtime_engine(engine, frame_count=4, system_prompt="production prompt")

        self.assertEqual(result["frame_ms"], [11.0, 12.0, 13.0, 14.0])
        self.assertEqual(result["system_prompt"], "production prompt")
        self.assertEqual(engine.system_prompt, "production prompt")
        self.assertEqual(len(engine.calls), 4)
        self.assertEqual([call[1] for call in engine.calls], [False, False, False, True])
        self.assertTrue(all(call[2] for call in engine.calls))
        self.assertFalse(engine.started)

    def test_websocket_server_ping_can_be_disabled_for_browser_equivalent_runs(
        self,
    ) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {"VOICECHAT_WEB_WS_PING_INTERVAL": "none"}):
            self.assertIsNone(websocket_ping_interval())
            self.assertIsNone(websocket_ping_timeout())

    def test_vllm_request_positions_report_prefill_relative_counts(self) -> None:
        from types import SimpleNamespace

        request_id = "stream-3"
        nano = SimpleNamespace(
            engine=SimpleNamespace(
                requests={request_id: SimpleNamespace(generated_tokens=[1, 2, 3, 4])}
            )
        )
        eartts = SimpleNamespace(
            engine=SimpleNamespace(
                requests={request_id: SimpleNamespace(generated_tokens=list(range(42)))}
            )
        )
        pipeline = SimpleNamespace(
            _request_id_for_stream=lambda stream_id: f"stream-{stream_id}",
            s2s_model=SimpleNamespace(
                model_llm_interface=nano,
                model=SimpleNamespace(tts_model=SimpleNamespace(tts_model=eartts)),
            ),
        )
        engine = VoiceChatEngine(pipeline)
        engine.stream_id = 3
        engine.vllm_request_position_baseline = {"nano": 2, "eartts": 37}

        positions = engine._vllm_request_positions()

        self.assertEqual(positions["request_id"], request_id)
        self.assertEqual(positions["nano"]["session_positions"], 2)
        self.assertEqual(positions["eartts"]["session_positions"], 5)

    def test_direct_source_position_advances_aligned_model_clocks_once(self) -> None:
        request_id = "stream-3"
        nano_request = SimpleNamespace(generated_tokens=[1, 2])
        tts_request = SimpleNamespace(generated_tokens=list(range(37)))
        nano = SimpleNamespace(engine=SimpleNamespace(requests={request_id: nano_request}))
        eartts = SimpleNamespace(engine=SimpleNamespace(requests={request_id: tts_request}))

        def inject(stream_id: int, token_id: int) -> dict:
            self.assertEqual((stream_id, token_id), (3, 42))
            nano_request.generated_tokens.append(12)
            tts_request.generated_tokens.append(0)
            return {"effective_token_id": 12}

        pipeline = SimpleNamespace(
            _request_id_for_stream=lambda stream_id: f"stream-{stream_id}",
            inject_user_source_position=inject,
            _fc_async_bg={},
            s2s_model=SimpleNamespace(
                _agent_eos_just_fired=True,
                model_llm_interface=nano,
                model=SimpleNamespace(tts_model=SimpleNamespace(tts_model=eartts)),
            ),
        )
        engine = VoiceChatEngine(pipeline)
        engine.started = True
        engine.stream_id = 3
        engine.frame_index = 0
        engine.vllm_request_position_baseline = {"nano": 2, "eartts": 37}

        result = engine.inject_user_source_position(42)

        self.assertEqual(result["session_position_before"], 0)
        self.assertEqual(result["session_position_after"], 1)
        self.assertEqual(result["transport_frame_index_after"], 1)
        self.assertEqual(engine.frame_index, 1)
        self.assertEqual(engine.audio_frame_index, 0)

    def test_direct_source_position_preserves_post_bos_epoch_offset(self) -> None:
        request_id = "stream-3"
        nano = SimpleNamespace(
            engine=SimpleNamespace(
                requests={request_id: SimpleNamespace(generated_tokens=[1, 2, 3])}
            )
        )
        eartts = SimpleNamespace(
            engine=SimpleNamespace(
                requests={request_id: SimpleNamespace(generated_tokens=list(range(37)))}
            )
        )

        def inject(stream_id: int, token_id: int) -> dict:
            self.assertEqual((stream_id, token_id), (3, 42))
            nano.engine.requests[request_id].generated_tokens.append(12)
            eartts.engine.requests[request_id].generated_tokens.append(0)
            return {"effective_token_id": 12}

        pipeline = SimpleNamespace(
            _request_id_for_stream=lambda stream_id: f"stream-{stream_id}",
            inject_user_source_position=inject,
            _fc_async_bg={},
            s2s_model=SimpleNamespace(
                _agent_eos_just_fired=True,
                model_llm_interface=nano,
                model=SimpleNamespace(tts_model=SimpleNamespace(tts_model=eartts)),
            ),
        )
        engine = VoiceChatEngine(pipeline)
        engine.started = True
        engine.stream_id = 3
        engine.frame_index = 9
        engine.vllm_request_position_baseline = {"nano": 2, "eartts": 37}

        result = engine.inject_user_source_position(42)

        self.assertEqual(result["nano_session_position_before"], 1)
        self.assertEqual(result["eartts_session_position_before"], 0)
        self.assertEqual(result["epoch_offset_before"], 1)
        self.assertEqual(result["epoch_offset_after"], 1)
        self.assertFalse(pipeline.s2s_model._agent_eos_just_fired)
        self.assertEqual(engine.frame_index, 10)

    def test_direct_source_position_rejects_changed_epoch_offset(self) -> None:
        request_id = "stream-3"
        nano_request = SimpleNamespace(generated_tokens=[1, 2, 3])
        eartts_request = SimpleNamespace(generated_tokens=list(range(37)))
        nano = SimpleNamespace(engine=SimpleNamespace(requests={request_id: nano_request}))
        eartts = SimpleNamespace(engine=SimpleNamespace(requests={request_id: eartts_request}))

        def inject(_stream_id: int, _token_id: int) -> dict:
            nano_request.generated_tokens.extend([12, 12])
            eartts_request.generated_tokens.append(0)
            return {"effective_token_id": 12}

        pipeline = SimpleNamespace(
            _request_id_for_stream=lambda stream_id: f"stream-{stream_id}",
            inject_user_source_position=inject,
            s2s_model=SimpleNamespace(
                model_llm_interface=nano,
                model=SimpleNamespace(tts_model=SimpleNamespace(tts_model=eartts)),
            ),
        )
        engine = VoiceChatEngine(pipeline)
        engine.started = True
        engine.stream_id = 3
        engine.frame_index = 9
        engine.vllm_request_position_baseline = {"nano": 2, "eartts": 37}

        with self.assertRaisesRegex(RuntimeError, "advance both model clocks exactly once"):
            engine.inject_user_source_position(42)

        self.assertEqual(engine.frame_index, 9)

    def test_session_position_limit_is_explicit_and_graceful(self) -> None:
        self.assertIsNone(session_position_limit_event(11_999, 12_000))
        event = session_position_limit_event(12_000, 12_000)
        self.assertEqual(event["type"], "error")
        self.assertEqual(event["error"]["code"], "session_position_limit")
        self.assertEqual(event["error"]["model_frames"], 12_000)
        self.assertEqual(event["error"]["max_model_frames"], 12_000)

    def test_server_accepts_composed_quantized_public_manifest_kind(self) -> None:
        self.assertIn(
            "exact_public_vllm_quantized_nano_eartts",
            PUBLIC_VLLM_MANIFEST_KINDS,
        )

    def test_fp32_manifest_is_accepted_only_in_explicit_semantic_diagnostic(self) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            self.assertNotIn("exact_public_vllm_extraction", accepted_vllm_manifest_kinds())
        with patch.dict(
            "os.environ",
            {
                "VOICECHAT_DIRECT_SEMANTIC_PROBE_DIR": "/qualification",
                "VOICECHAT_DIRECT_SEMANTIC_PRECISION": "fp32",
            },
            clear=True,
        ):
            self.assertIn("exact_public_vllm_extraction", accepted_vllm_manifest_kinds())
        with patch.dict(
            "os.environ",
            {
                "VOICECHAT_DIRECT_SEMANTIC_PROBE_DIR": "/qualification",
                "VOICECHAT_DIRECT_SEMANTIC_PRECISION": "w8",
            },
            clear=True,
        ):
            self.assertNotIn("exact_public_vllm_extraction", accepted_vllm_manifest_kinds())

    def test_host_runtime_provenance_records_driver_and_kernel(self) -> None:
        banner = (
            "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  "
            "580.142  Release Build  (builder)"
        )
        self.assertEqual(parse_nvidia_driver_version(banner), "580.142")
        provenance = host_runtime_provenance()
        self.assertIn("driver_version", provenance)
        self.assertTrue(provenance["kernel_version"])

    def test_health_exposes_explicit_driver_and_kernel_versions(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        engine = SimpleNamespace(pipeline=SimpleNamespace(checkpoint_provenance={"kind": "test"}))
        expected = {
            "driver_version": "580.142",
            "kernel_version": "6.17.0-1014-nvidia",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "nemotron_voicechat_runtime.server.host_runtime_provenance",
                return_value=expected,
            ),
        ):
            html = Path(directory) / "client.html"
            html.write_text("ok")
            app = create_app(engine)
            endpoint = next(route.endpoint for route in app.routes if route.path == "/health")
            result = asyncio.run(endpoint())
        self.assertEqual(result["driver_version"], "580.142")
        self.assertEqual(result["kernel_version"], "6.17.0-1014-nvidia")
        self.assertEqual(result["host_runtime"], expected)
        self.assertEqual(result["checkpoint"]["host_runtime"], expected)
        self.assertEqual(
            result["step9_capture_fallback"],
            {"count": 0, "last_descriptor": None},
        )
        runtime = result["checkpoint"]["runtime_provenance"]
        self.assertEqual(runtime["speech_commit"], PINNED_COMMIT)
        self.assertIn("src/nemotron_voicechat_runtime/server.py", runtime["source_sha256"])

    def test_env_bool_uses_default_and_validates_explicit_values(self) -> None:
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(env_bool("TEST_SWITCH", True))
            self.assertFalse(env_bool("TEST_SWITCH", False))
        with patch.dict("os.environ", {"TEST_SWITCH": "off"}, clear=True):
            self.assertFalse(env_bool("TEST_SWITCH", True))
        with patch.dict("os.environ", {"TEST_SWITCH": "YES"}, clear=True):
            self.assertTrue(env_bool("TEST_SWITCH", False))
        with patch.dict("os.environ", {"TEST_SWITCH": "maybe"}, clear=True):
            with self.assertRaises(SystemExit):
                env_bool("TEST_SWITCH", True)

    def test_tool_prompt_requires_exact_post_call_grounding(self) -> None:
        from unittest.mock import patch

        module_name = "nemo.collections.speechlm2.inference.utils.offline_voicechat"
        module = ModuleType(module_name)
        module.render_fc_system_prompt = lambda *_args: "NVIDIA tool template"
        with patch.dict(sys.modules, {module_name: module}):
            rendered = render_tool_system_prompt(
                Path("/unused/template.j2"),
                "system",
                [{"name": "get_random_number"}],
            )
        self.assertIn("NVIDIA tool template", rendered)
        self.assertIn("explicitly include the exact returned result", rendered)
        self.assertIn("If no advertised tool matches the request", rendered)
        self.assertNotIn("emergency services", rendered)

    def test_function_call_budget_is_per_turn_and_fails_without_canned_text(self) -> None:
        budget = FunctionCallBudget(limit=2)
        self.assertEqual(budget.consume("turn-1"), (True, 1))
        self.assertEqual(budget.consume("turn-1"), (True, 2))
        self.assertEqual(budget.consume("turn-1"), (False, 3))
        self.assertEqual(budget.consume("turn-2"), (True, 1))

        protocol = RealtimeProtocolSession(
            "loop-test",
            protocol_capabilities(
                function_call_timeout_seconds=30.0,
                max_session_model_frames=12_000,
            ),
        )
        protocol.function_call_events(call_id="call-1", name="lookup", arguments='{"q":"x"}')
        events = function_call_loop_terminal_events(
            protocol,
            call_id="call-3",
            attempt=3,
            limit=2,
        )
        self.assertEqual(
            [event["type"] for event in events],
            [
                "response.output_text.done",
                "response.output_audio.done",
                "response.done",
                "error",
                "session.closed",
            ],
        )
        self.assertTrue(events[0]["empty"])
        self.assertTrue(events[1]["empty"])
        self.assertEqual(events[2]["status"], "failed")
        self.assertEqual(events[2]["reason"], "function_call_loop_limit")
        self.assertEqual(events[3]["error"]["attempt"], 3)
        self.assertTrue(events[3]["error"]["fatal"])
        self.assertEqual(events[4]["status"], "failed")

    def test_nano_pad_pair_production_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "voicechat_pad_pair_token_id": 12,
                "chunk_size": 1,
                "voicechat_pad_pair_contract": {
                    "scope": "production live runtime",
                    "max_positions": 2,
                    "production_mamba_rollback": True,
                    "function_channel_rejection": True,
                    "post_fc_fusion_correction": True,
                    "mamba_chunk_size": 1,
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            manifest = {
                "runtime_optimization": {
                    "pad_pair_production": True,
                    "production_mamba_rollback": True,
                    "function_channel_rejection": True,
                    "post_fc_fusion_correction": True,
                }
            }
            result = validate_nano_pad_pair_runtime(root, manifest, True)
            self.assertTrue(result["enabled"])
            self.assertEqual(result["pad_token_id"], 12)
            config["voicechat_pad_pair_contract"]["mamba_chunk_size"] = 128
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                validate_nano_pad_pair_runtime(root, manifest, True)

    def test_candidate_one_pins_conditional_function_and_runtime_sources(self) -> None:
        from unittest.mock import patch

        source_names = set(RUNTIME_SOURCE_PATHS)
        sources = {
            name: {
                "bytes": (REPO_ROOT / name).stat().st_size,
                "sha256": hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest(),
            }
            for name in source_names
        }
        conditional = {
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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1"}, clear=True),
        ):
            root = Path(directory)
            config = {
                "voicechat_pad_pair_token_id": 12,
                "chunk_size": 1,
                "voicechat_pad_pair_contract": {
                    "scope": "production live runtime",
                    "max_positions": 2,
                    "production_mamba_rollback": True,
                    "function_channel_rejection": True,
                    "post_fc_fusion_correction": True,
                    "mamba_chunk_size": 1,
                    **conditional,
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            manifest = {
                "runtime_optimization": {
                    "pad_pair_production": True,
                    "production_mamba_rollback": True,
                    "function_channel_rejection": True,
                    "post_fc_fusion_correction": True,
                    **conditional,
                },
                "production_candidate": {
                    "number": 1,
                    "status": "frozen",
                    "runtime_sources": sources,
                    "runtime_image": "candidate:1",
                    "runtime_image_id": "sha256:image",
                },
            }
            result = validate_nano_pad_pair_runtime(root, manifest, True)
            self.assertEqual(result["production_candidate"]["number"], 1)
            first_source = next(iter(source_names))
            manifest["production_candidate"]["runtime_sources"][first_source]["sha256"] = "0" * 64
            with self.assertRaises(SystemExit):
                validate_nano_pad_pair_runtime(root, manifest, True)

    def test_sharded_model_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "model-00001.safetensors": b"one",
                "model-00002.safetensors": b"two",
            }
            files = []
            composite = hashlib.sha256()
            for name, payload in sorted(payloads.items()):
                (root / name).write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                files.append({"name": name, "bytes": len(payload), "sha256": digest})
                composite.update(name.encode("utf-8"))
                composite.update(b"\0")
                composite.update(str(len(payload)).encode("ascii"))
                composite.update(b"\0")
                composite.update(digest.encode("ascii"))
                composite.update(b"\n")
            component = {
                "model": {
                    "bytes": sum(map(len, payloads.values())),
                    "sha256": composite.hexdigest(),
                    "files": files,
                }
            }
            result = validate_manifested_model(root, component)
            self.assertEqual(result["bytes"], 6)
            self.assertEqual(result["sha256"], composite.hexdigest())

    def test_signed_release_component_adapts_to_sharded_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "model-00001-of-00002.safetensors": b"one",
                "model-00002-of-00002.safetensors": b"two",
                "config.json": b"{}",
            }
            files = []
            composite = hashlib.sha256()
            for name, payload in sorted(payloads.items()):
                (root / name).write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                files.append(
                    {
                        "path": f"nano/{name}",
                        "bytes": len(payload),
                        "sha256": digest,
                    }
                )
                if name.endswith(".safetensors"):
                    composite.update(name.encode("utf-8"))
                    composite.update(b"\0")
                    composite.update(str(len(payload)).encode("ascii"))
                    composite.update(b"\0")
                    composite.update(digest.encode("ascii"))
                    composite.update(b"\n")
            release = {
                "candidate": "production-candidate-1",
                "nano_composite_sha256": composite.hexdigest(),
                "files": files,
            }

            component = release_component_manifest(release, "nano")
            result = validate_manifested_model(root, component)

            self.assertEqual(result["bytes"], 6)
            self.assertEqual(result["sha256"], composite.hexdigest())
            self.assertEqual(result["artifact_files"]["config.json"]["bytes"], 2)

    def test_signed_release_pad_pair_contract_is_fail_closed(self) -> None:
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "1"}, clear=True),
        ):
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "voicechat_pad_pair_token_id": 12,
                        "chunk_size": 1,
                        "voicechat_pad_pair_contract": {
                            "scope": "production live runtime",
                            "max_positions": 2,
                            "production_mamba_rollback": True,
                            "function_channel_rejection": True,
                            "post_fc_fusion_correction": True,
                            "mamba_chunk_size": 1,
                        },
                    }
                )
            )
            runtime_contract = {
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
            release = {
                "kind": "nemotron_voicechat_dgx_spark_hf_release",
                "candidate": "production-candidate-1",
                "release_sha256": "f" * 64,
                "runtime_contract": runtime_contract,
            }

            result = validate_nano_pad_pair_runtime(root, release, True)
            self.assertEqual(result["production_candidate"]["status"], "signed_release")
            release["runtime_contract"]["nano_pad_pair_control_barrier"] = False
            with self.assertRaisesRegex(SystemExit, "runtime contract mismatch"):
                validate_nano_pad_pair_runtime(root, release, True)

    def test_sharded_model_manifest_rejects_modified_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model-00001.safetensors").write_bytes(b"changed")
            component = {
                "model": {
                    "bytes": 3,
                    "sha256": "bad",
                    "files": [
                        {
                            "name": "model-00001.safetensors",
                            "bytes": 3,
                            "sha256": hashlib.sha256(b"old").hexdigest(),
                        }
                    ],
                }
            }
            with self.assertRaises(SystemExit):
                validate_manifested_model(root, component)

    def test_manifest_validation_checks_declared_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.safetensors"
            config = root / "config.json"
            model.write_bytes(b"weights")
            config.write_bytes(b'{"custom_outputs": []}')
            component = {
                "path": str(root),
                "files": {
                    "model.safetensors": {
                        "bytes": model.stat().st_size,
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    },
                    "config.json": {
                        "bytes": config.stat().st_size,
                        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    },
                },
                "model": {
                    "bytes": model.stat().st_size,
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                },
            }
            result = validate_manifested_model(root, component)
            self.assertEqual(
                set(result["artifact_files"]),
                {"config.json", "model.safetensors"},
            )

            config.write_bytes(b"tampered")
            with self.assertRaisesRegex(SystemExit, "Manifest mismatch for artifact file"):
                validate_manifested_model(root, component)

    def test_agent_logit_diagnostic_nano_is_opt_in_and_weight_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "nano"
            scratch = root / "diagnostic"
            source.mkdir()
            weights = source / "model.safetensors"
            weights.write_bytes(b"weights")
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "custom_outputs": ["function_tokens", "function_logits"],
                        "voicechat_skip_custom_text_logits": {
                            "enabled": True,
                            "contract": "temperature=0, top_p=1, repetition_penalty=1",
                            "source_config_sha256": "a" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                selected, provenance = prepare_agent_logit_diagnostic_nano(
                    source, scratch
                )
            self.assertEqual(selected, source.resolve())
            self.assertIsNone(provenance)
            self.assertFalse(scratch.exists())

            environment = {
                "S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "20",
                "S2S_POST_FC_TOKEN_TRACE_FRAMES": "9",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                selected, provenance = prepare_agent_logit_diagnostic_nano(
                    source, scratch
                )

            config = json.loads((selected / "config.json").read_text())
            self.assertEqual(
                config["custom_outputs"],
                ["text_logits", "function_tokens", "function_logits"],
            )
            self.assertNotIn("voicechat_skip_custom_text_logits", config)
            self.assertTrue((selected / weights.name).is_symlink())
            self.assertEqual((selected / weights.name).read_bytes(), b"weights")
            self.assertEqual(provenance["top_k"], 20)
            self.assertFalse(provenance["weights_changed"])
            self.assertFalse(provenance["performance_metrics_valid"])

    def test_agent_logit_diagnostic_requires_bounded_token_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "nano"
            source.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK": "20"},
                clear=True,
            ):
                with self.assertRaisesRegex(SystemExit, "requires.*TRACE_FRAMES"):
                    prepare_agent_logit_diagnostic_nano(
                        source, root / "diagnostic"
                    )

    def test_source_patch_counts_a_new_turn_despite_recent_prior_eos(self) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        self.assertIn("+            if agent_speaking and current_tok != eos_id:", patch)
        self.assertIn("+                    and current_tok != eos_id):", patch)
        self.assertIn("-                    and not (agent_window == eos_id).any()):", patch)
        self.assertNotIn(
            "+                    and not (agent_window == eos_id).any()):",
            patch,
        )

    def test_source_patch_resets_turn_text_count_at_forced_response_cap(self) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        self.assertIn("+                    if '_turn_text_tokens' in rnnt_state:", patch)
        self.assertIn("+                        rnnt_state['_turn_text_tokens'][b] = 0", patch)

    def test_source_patch_rejects_energy_only_transport_turns(self) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        self.assertIn('self.model_cfg.get("transport_vad_min_rnnt_tokens", 1)', patch)
        self.assertIn(
            "+                and nonblank_total >= max(1, transport_min_rnnt_tokens)",
            patch,
        )

    def test_source_patch_does_not_arm_transport_fallback_during_agent_audio(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        self.assertIn(
            "+    def observe_transport_user_activity(self, active: bool, stream_id=None)",
            patch,
        )
        self.assertIn("+        if not self._get_agent_idle(stream_id):", patch)
        self.assertIn("+                    self._transport_user_speech_seen = False", patch)

    def test_source_patch_makes_client_commit_the_only_user_eou(self) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        self.assertIn("+    def request_user_eou(self) -> None:", patch)
        self.assertIn("+    def set_external_user_eou_mode(self, enabled: bool) -> None:", patch)
        self.assertIn("+                self._external_user_eou_requested = False", patch)
        self.assertIn(
            "+            if (not getattr(self, '_external_user_eou_mode', False)",
            patch,
        )

    def test_source_patch_fuses_no_tools_bos_only_after_rnnt_blank_fence(self) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()
        suppression = patch.index(
            "+            # Client-owned turn mode has one authoritative"
        )
        state_machine = patch.index("             # FC state machine", suppression)
        pre_rnnt_function_guard = patch[suppression:state_machine]
        rnnt_step = patch.index(
            "+                    rnnt_partial_hypotheses, _rnnt_is_blank = self._rnnt_step("
        )
        rnnt_turn_taking = patch.index(
            "                     self._apply_rnnt_turn_taking(", rnnt_step
        )
        fused_block = patch.index(
            "+            # Tool-free client sessions can arm a separate blank-fence latch."
        )
        post_fc_block = patch.index(
            "+            # A boundary SOTC consumes the client EOU", fused_block
        )
        fused_block_text = patch[fused_block:post_fc_block]

        self.assertNotIn("_external_user_eou_blank_fence", pre_rnnt_function_guard)
        self.assertLess(rnnt_step, rnnt_turn_taking)
        self.assertIn("if blank_cnt >= int(_blank_fence_target):", fused_block_text)
        self.assertIn("if blank_cnt != int(_blank_fence_target):", fused_block_text)
        self.assertIn("gen_text[b, t] = bos_id", fused_block_text)
        self.assertIn("rnnt_state['forced_bos'][b] = True", fused_block_text)
        self.assertIn("self._external_user_eou_blank_fence_target = None", fused_block_text)
        self.assertIn("'chosen_token_id': int(bos_id)", fused_block_text)
        self.assertIn("+    def request_user_eou_at_blank_fence(", patch)
        self.assertIn("+    def cancel_user_eou_at_blank_fence(", patch)
        self.assertGreaterEqual(
            patch.count("self._external_user_eou_blank_fence_target = None"),
            6,
        )

    def test_source_patch_commits_boundary_sotc_and_forces_one_post_fc_bos(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        suppression = patch.index("+            # Client-owned turn mode has one authoritative")
        state_machine = patch.index("             # FC state machine", suppression)
        boundary = patch[suppression:state_machine]
        post_fc = patch.index("+            # A boundary SOTC consumes the client EOU")
        post_fc_end = patch.index("             # Self-play suppression", post_fc)
        post_fc_block = patch[post_fc:post_fc_end]

        self.assertIn("and getattr(self, '_external_user_eou_mode', False)", boundary)
        self.assertIn("if not _agent_open:", boundary)
        self.assertIn("function_predicted_tokens[:, frame_offset] = pad_id", boundary)
        self.assertIn("gen_function_text[:, current_frame_idx] = pad_id", boundary)
        self.assertIn("_pre_eou_function_last_raw_token_id", boundary)
        self.assertIn("getattr(self, '_external_user_eou_requested', False)", boundary)
        self.assertIn("_raw_function_token == self._fc_sotc_id", boundary)
        self.assertIn("_boundary_eou_sotc = bool(", boundary)
        self.assertIn("self._external_user_eou_requested = False", boundary)
        self.assertIn("self._post_fc_client_bos_requested = True", boundary)
        self.assertIn('_function_effective_reason = "boundary_sotc_commit"', boundary)
        self.assertIn("else:", boundary)
        self.assertIn("_fc_guard_active = bool(", boundary)
        self.assertIn("_opens_fc_cycle = bool(", boundary)
        self.assertIn("_raw_function_token == self._fc_sotc_id", boundary)
        self.assertIn("and not _opens_fc_cycle", boundary)
        self.assertIn('_function_effective_reason = "agent_open_guard"', boundary)
        self.assertIn("_agent_open_function_suppression_count += 1", boundary)
        self.assertGreaterEqual(
            boundary.count("function_predicted_tokens[:, frame_offset] = pad_id"), 2
        )
        self.assertGreaterEqual(
            boundary.count("gen_function_text[:, current_frame_idx] = pad_id"), 2
        )
        self.assertIn("if getattr(self, '_post_fc_client_bos_requested', False):", post_fc_block)
        self.assertIn("gen_text[b, t] = bos_id", post_fc_block)
        self.assertIn("rnnt_state['agent_speaking'][b] = True", post_fc_block)
        self.assertIn("self._post_fc_client_bos_requested = False", post_fc_block)
        self.assertIn("self._post_fc_client_bos_forced_count += 1", post_fc_block)
        self.assertLess(suppression, state_machine)

    def test_source_patch_resets_and_preflight_rejects_pending_post_fc_bos(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()

        self.assertGreaterEqual(
            patch.count("+        self._post_fc_client_bos_requested = False"), 1
        )
        self.assertIn(
            '+\t\t\t"post_fc_client_bos": bool(getattr(wrapper, '
            '"_post_fc_client_bos_requested", False)),',
            patch,
        )
        self.assertIn('bg_result["fc_complete"] = bool(', patch)
        self.assertIn('bg_result["failed"] = True', patch)
        self.assertIn("elif was_aborted or fc_failed:", patch)
        self.assertNotIn("_deferred_eou_sotc", patch)

    def test_source_patch_binds_unique_eartts_backend_generation(self) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        self.assertIn("backend_request_id: Optional[str] = None", patch)
        self.assertIn("def resolve_backend_request_id", patch)
        self.assertIn("Could not retire backend generation", patch)
        self.assertIn("await generation_iterator.aclose()", patch)
        self.assertIn("request_id=backend_request_id", patch)
        self.assertIn(
            'current["backend_request_id"] != prior["backend_request_id"]', patch
        )
        self.assertIn(
            'current["backend_generation"] != prior["backend_generation"]', patch
        )

    def test_source_patch_requires_immediate_model_owned_eotr_before_fc_complete(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()

        self.assertIn('self.model_cfg.get("fc_eotr_wait_max_steps", 1)', patch)
        self.assertIn('fc_state["awaiting_eotr"] = True', patch)
        self.assertIn('awaiting_eotr_at_step = bool(fc_state.get("awaiting_eotr", False))', patch)
        self.assertIn('fc_state["eotr_observed"] = True', patch)
        self.assertIn('fc_state["eotr_first_post_drain_token_id"] = func_tok_val', patch)
        self.assertIn('fc_state["eotr_unexpected_token_id"] = func_tok_val', patch)
        self.assertIn('fc_state["eotr_timeout"] = True', patch)
        self.assertIn('not_awaiting_eotr = not fc_state.get("awaiting_eotr", False)', patch)
        self.assertIn('and fc_state.get("eotr_observed") is True', patch)
        self.assertIn('and not fc_state.get("eotr_timeout", False)', patch)
        self.assertIn('bg_result["terminal_function_token_id"]', patch)
        self.assertIn('context.fc_state["eotr_feedback_committed"] = True', patch)
        self.assertIn(
            '"completed FC background result has invalid terminal EOTR feedback"',
            patch,
        )

    def test_source_patch_post_fc_trace_is_bounded_and_phase2_safe(self) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        async_start = patch.index("+            _trace_enabled = bool(")
        forced_branch = patch.index("             if forced:", async_start)
        async_trace = patch[async_start:forced_branch]
        self.assertIn("_trace_candidate = bool(", async_trace)
        self.assertIn("if _trace_candidate:", async_trace)
        self.assertIn("awaiting_eotr_at_step = False", async_trace)
        self.assertLess(
            async_trace.index("awaiting_eotr_at_step = False"),
            len(async_trace),
        )
        self.assertIn("def drain_post_fc_token_trace", patch)
        self.assertIn("with self._post_fc_token_trace_lock:", patch)
        self.assertIn("self._reset_post_fc_token_trace()", patch)
        self.assertIn("t + self._post_fc_token_trace_frames - 1", patch)

    def test_source_patch_fc_async_heartbeat_is_opt_in_immutable_and_stage_bounded(
        self,
    ) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        self.assertIn("S2S_FC_ASYNC_HEARTBEAT", patch)
        self.assertIn("if not self._fc_async_heartbeat_enabled:", patch)
        self.assertIn("fc_state[key] = {", patch)
        self.assertNotIn("fc_state[key].update(", patch)
        self.assertIn(
            "self._perception_live_worker_guarded\n"
            "+                    if self._fc_async_heartbeat_enabled\n"
            "+                    else self._perception_live_worker",
            patch,
        )
        self.assertIn('previous["invocation"] > invocation', patch)
        publisher_start = patch.index("+    def _publish_fc_async_heartbeat(")
        publisher_end = patch.index("+    def _append_post_fc_token_trace(", publisher_start)
        publisher = patch[publisher_start:publisher_end]
        disabled_guard = publisher.index("if not self._fc_async_heartbeat_enabled:")
        disabled_return = publisher.index("return", disabled_guard)
        self.assertLess(disabled_return, publisher.index("time.monotonic()"))
        self.assertLess(disabled_return, publisher.index("_bounded_queue_depth("))
        for stage in (
            "prepare",
            "nano",
            "eartts",
            "codec_decode",
            "codec_cpu_copy",
            "rnnt",
            "stage_failed",
            "complete",
            "failed",
        ):
            self.assertIn(f'stage="{stage}"', patch)
        self.assertIn('terminal_stage = "aborted"', patch)
        self.assertGreaterEqual(patch.count("cycle_sotc_frame=sotc_frame"), 4)
        self.assertIn("failed_stage=_heartbeat_tts_stage", patch)

    def test_source_patch_agent_logit_trace_is_opt_in_and_candidate_bounded(
        self,
    ) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        self.assertIn("S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK", patch)
        self.assertIn("_validate_agent_logit_trace_artifact", patch)
        self.assertIn('"text_logits" not in custom_outputs', patch)
        self.assertIn("capture_agent_logit_trace=bool(", patch)
        self.assertIn('"_agent_logit_trace_topk",', patch)
        self.assertIn('trace_record["agent_topk"] =', patch)
        self.assertIn("does not match recorded raw agent token", patch)
        self.assertIn("no_model_position", patch)
        self.assertIn("does not match recorded raw PAD", patch)
        self.assertIn('"raw_agent_source"', patch)
        self.assertNotIn("capture_agent_logit_trace=_trace_candidate", patch)

    def test_source_patch_resets_agent_idle_fallback_at_session_boundary(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        reset = patch.index("     def _reset_rnnt_turn_taking_state(self):")
        request_agent_eos = patch.index("+    def request_agent_eos(self) -> None:", reset)
        reset_block = patch[reset:request_agent_eos]

        self.assertIn("self._agent_idle = True", reset_block)
        self.assertLess(
            reset_block.index("self._agent_idle = True"),
            reset_block.index("self._agent_eos_just_fired = False"),
        )

    def test_source_patch_stage_timing_is_current_and_nonperturbing(self) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        self.assertIn("time.perf_counter()", patch)
        self.assertIn("host_intervals_between_existing_cuda_syncs", patch)
        self.assertIn("nano_interface_ms", patch)
        self.assertIn("waits_for_previous_frame_and_submits_current", patch)
        self.assertIn("model stage timings exceed the wrapper total", patch)
        self.assertIn("last_step_stage_timings: Optional[Dict[str, Any]] = None", patch)
        self.assertIn(
            'stage_timings = step_result.get("stage_timings_ms")', patch
        )
        self.assertIn(
            "dict(stage_timings) if isinstance(stage_timings, dict) else None",
            patch,
        )
        clear = patch.index("context.last_step_stage_timings = None")
        fc_early_branch = patch.index("if stream_id in self._fc_async_bg:", clear)
        restore = patch.index(
            'stage_timings = step_result.get("stage_timings_ms")', fc_early_branch
        )
        self.assertLess(clear, fc_early_branch)
        self.assertLess(fc_early_branch, restore)
        self.assertNotIn("torch.cuda.synchronize()  # stage timing", patch)

    def test_source_patch_ends_completed_function_cycle_before_normal_inference(
        self,
    ) -> None:
        patch = (RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch").read_text()
        completion = patch.index(
            ' \t\t\t\t\tlogging.info("[FC Async NB] Background thread done, '
            'resuming normal flow for stream %d", stream_id)'
        )
        next_branch = patch.index(" \t\t\telse:", completion)
        boundary = patch[completion:next_branch]

        self.assertIn("function-cycle boundary", boundary)
        self.assertIn("+\t\t\t\t\treturn", boundary)
        self.assertNotIn("+\t\t\t\t\tself.infer_one_step", boundary)

    def test_source_patch_bounds_committed_mode_fc_interrupt_and_reset(self) -> None:
        patch = (
            RUNTIME_ROOT / "patches" / "nemotron-voicechat-rnnt-turn-taking.patch"
        ).read_text()

        interrupt = patch.index("def interrupt_fc_async(")
        reset_helper = patch.index("def _terminate_fc_async_for_reset(", interrupt)
        reset = patch.index("def reset_session(", reset_helper)
        block = patch[interrupt:reset]
        self.assertIn('bg.get("quit_async_event")', block)
        self.assertIn("thread.join(timeout)", block)
        self.assertIn("thread.is_alive()", block)
        self.assertIn('result.get("quit_to_normal") is not True', block)
        self.assertIn('bg.get("abort_event")', block)
        self.assertLess(reset_helper, reset)
        self.assertIn("self._terminate_fc_async_for_reset()", patch[reset : reset + 300])

    def test_audio_delivery_is_bounded_by_tokenized_response_and_tail(self) -> None:
        self.assertFalse(response_audio_is_deliverable({}))
        self.assertFalse(
            response_audio_is_deliverable({"response_boundary": {"phase": "idle", "event": None}})
        )
        self.assertTrue(
            response_audio_is_deliverable(
                {"response_boundary": {"phase": "responding", "event": None}}
            )
        )
        self.assertTrue(
            response_audio_is_deliverable(
                {"response_boundary": {"phase": "tail_draining", "event": None}}
            )
        )
        self.assertFalse(
            response_audio_is_deliverable(
                {"response_boundary": {"phase": "complete", "event": "end"}}
            )
        )
        self.assertFalse(
            response_audio_is_deliverable(
                {"response_boundary": {"phase": "internal_drain", "event": None}}
            )
        )

    def test_transport_gate_only_suppresses_leading_silence(self) -> None:
        gate = TransportModelGate(threshold_dbfs=-50.0, continuous_after_speech=True)

        self.assertFalse(gate.should_advance(-120.0))
        self.assertTrue(gate.should_advance(-20.0))
        self.assertTrue(gate.should_advance(-120.0))
        gate.observe_response_end()
        self.assertTrue(gate.should_advance(-120.0))

    def test_transport_gate_requires_consecutive_active_frames(self) -> None:
        gate = TransportModelGate(
            threshold_dbfs=-40.0,
            continuous_after_speech=True,
            min_active_frames=3,
        )

        self.assertFalse(gate.should_advance(-35.0))
        self.assertFalse(gate.input_active)
        self.assertFalse(gate.should_advance(-35.0))
        self.assertFalse(gate.input_active)
        self.assertFalse(gate.should_advance(-60.0))
        self.assertEqual(gate.consecutive_active_frames, 0)
        self.assertFalse(gate.should_advance(-35.0))
        self.assertFalse(gate.should_advance(-35.0))
        self.assertTrue(gate.should_advance(-35.0))
        self.assertTrue(gate.input_active)
        self.assertTrue(gate.should_advance(-60.0))
        self.assertFalse(gate.input_active)

    def test_gate_preroll_retains_newest_entries_and_drains_fifo_once(self) -> None:
        buffer = GatePrerollBuffer(2)
        for index in range(3):
            buffer.append(
                np.asarray([index], dtype=np.float32),
                {"index": index},
                index == 2,
                {"transport": index},
                "microphone",
                None,
                index,
            )

        self.assertEqual(len(buffer), 2)
        entries = buffer.drain()
        self.assertEqual([entry.transport_frame for entry in entries], [1, 2])
        self.assertEqual([entry.frame_stats["index"] for entry in entries], [1, 2])
        self.assertEqual(buffer.drain(), [])
        self.assertEqual(len(buffer), 0)

    def test_gate_preroll_clear_discards_all_entries(self) -> None:
        buffer = GatePrerollBuffer(3)
        buffer.append(np.zeros(1, dtype=np.float32), {}, False, {}, "microphone", None, 0)

        buffer.clear()

        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.drain(), [])

    def test_gate_preroll_selects_active_onset_with_bounded_context(self) -> None:
        buffer = GatePrerollBuffer(12)
        levels = [-120.0] * 8 + [-62.0, -45.0, -13.0, -12.0]
        originals = []
        for index, level in enumerate(levels):
            samples = np.full(4, index, dtype=np.float32)
            buffer.append(
                samples,
                {"rms_dbfs": level},
                False,
                {"above_threshold": level >= -40.0},
                "microphone",
                None,
                index,
            )
            originals.append(buffer._entries[-1])

        selection = buffer.drain_onset_suffix(
            threshold_dbfs=-40.0,
            context_frames=2,
        )

        self.assertEqual(selection.buffered_frames, 12)
        self.assertEqual(selection.first_active_index, 10)
        self.assertEqual(selection.skipped_frames, 8)
        self.assertEqual(
            [entry.transport_frame for entry in selection.entries],
            [8, 9, 10, 11],
        )
        self.assertTrue(
            all(
                selected is original
                for selected, original in zip(selection.entries, originals[8:])
            )
        )
        np.testing.assert_array_equal(selection.entries[0].frame_f32, np.full(4, 8))
        self.assertEqual(len(buffer), 0)

    def test_gate_preroll_all_quiet_falls_back_to_full_fifo_replay(self) -> None:
        buffer = GatePrerollBuffer(3)
        for index in range(3):
            buffer.append(
                np.asarray([index], dtype=np.float32),
                {"rms_dbfs": -80.0},
                False,
                {"above_threshold": False},
                "microphone",
                None,
                index,
            )

        selection = buffer.drain_onset_suffix(
            threshold_dbfs=-40.0,
            context_frames=2,
        )

        self.assertIsNone(selection.first_active_index)
        self.assertEqual(selection.skipped_frames, 0)
        self.assertEqual(
            [entry.transport_frame for entry in selection.entries],
            [0, 1, 2],
        )

    def test_gate_preroll_source_or_job_mismatch_clears_stale_entries(self) -> None:
        buffer = GatePrerollBuffer(4)
        buffer.append(np.zeros(1, dtype=np.float32), {}, False, {}, "microphone", None, 0)
        buffer.append(np.ones(1, dtype=np.float32), {}, False, {}, "typed", "job-1", 1)
        self.assertEqual(
            [(entry.source, entry.job_id) for entry in buffer.drain()],
            [("typed", "job-1")],
        )

        buffer.append(np.ones(1, dtype=np.float32), {}, False, {}, "typed", "job-1", 2)
        buffer.append(np.ones(1, dtype=np.float32), {}, False, {}, "typed", "job-2", 3)
        self.assertEqual(
            [(entry.source, entry.job_id) for entry in buffer.drain()],
            [("typed", "job-2")],
        )

    def test_gate_preroll_nonpositive_lengths_disable_buffering(self) -> None:
        for configured_length in (0, -1, -100):
            with self.subTest(configured_length=configured_length):
                buffer = GatePrerollBuffer(configured_length)
                buffer.append(
                    np.ones(1, dtype=np.float32),
                    {},
                    False,
                    {},
                    "microphone",
                    None,
                    0,
                )
                self.assertEqual(buffer.maxlen, 0)
                self.assertEqual(len(buffer), 0)
                self.assertEqual(buffer.drain(), [])

    def test_stale_agent_eos_edge_is_cleared_before_unrelated_step(self) -> None:
        from types import SimpleNamespace

        wrapper = SimpleNamespace(_agent_eos_just_fired=True)
        pipeline = SimpleNamespace(s2s_model=wrapper, _fc_async_bg={})
        self.assertTrue(clear_stale_agent_eos_latch(pipeline, 7))
        self.assertFalse(wrapper._agent_eos_just_fired)
        self.assertFalse(clear_stale_agent_eos_latch(pipeline, 7))

    def test_active_function_worker_retains_agent_eos_edge(self) -> None:
        from types import SimpleNamespace

        wrapper = SimpleNamespace(_agent_eos_just_fired=True)
        pipeline = SimpleNamespace(s2s_model=wrapper, _fc_async_bg={7: object()})
        self.assertFalse(clear_stale_agent_eos_latch(pipeline, 7))
        self.assertTrue(wrapper._agent_eos_just_fired)

    def test_engine_interrupts_unpublished_function_through_pipeline_hook(self) -> None:
        pipeline = SimpleNamespace(
            interrupt_fc_async=mock.Mock(
                return_value={
                    "schema": 1,
                    "requested": True,
                    "reason": "client_turn_start",
                    "quit_to_normal": True,
                    "async_steps": 3,
                }
            )
        )
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = pipeline
        engine.stream_id = 9

        evidence = engine.interrupt_unpublished_function_cycle(0.75)

        pipeline.interrupt_fc_async.assert_called_once_with(9, 0.75)
        self.assertTrue(evidence["requested"])
        self.assertTrue(evidence["quit_to_normal"])

    def test_engine_abort_resets_background_without_synthetic_model_tick(self) -> None:
        class Wrapper:
            def reset_transport_session_state(self) -> None:
                pass

        class Pipeline:
            def __init__(self) -> None:
                self.s2s_model = Wrapper()
                self._fc_async_bg = {4: object()}
                self.reset_calls = 0

            def reset_session(self) -> None:
                self.reset_calls += 1
                self._fc_async_bg.clear()

        pipeline = Pipeline()
        engine = VoiceChatEngine(pipeline)
        engine.stream_id = 4
        engine.started = True
        engine.frame_index = 10
        with mock.patch.object(
            engine, "process", side_effect=AssertionError("synthetic model tick")
        ) as process:
            engine.abort()

        process.assert_not_called()
        self.assertEqual(pipeline.reset_calls, 1)
        self.assertFalse(engine.started)
        self.assertIsNone(engine.cleanup_failure)

    def test_eartts_step_snapshot_waits_for_completed_function_worker_boundary(
        self,
    ) -> None:
        controller = SimpleNamespace()
        pipeline = SimpleNamespace(
            s2s_model=SimpleNamespace(
                _eartts_prepared_epoch_controller=controller
            ),
            _fc_async_bg={7: SimpleNamespace(done=True)},
        )
        engine = VoiceChatEngine.__new__(VoiceChatEngine)
        engine.pipeline = pipeline
        engine.stream_id = 7
        expected = {"schema": 1, "transition_count": 9}
        with mock.patch(
            "nemotron_voicechat_runtime.server._eartts_epoch_step_snapshot",
            side_effect=RuntimeError("transient request identity"),
        ) as snapshot:
            self.assertIsNone(engine._eartts_epoch_before_model_step())
            snapshot.assert_not_called()

        # The completed-cycle boundary still owns the dictionary entry and is
        # intentionally skipped. The next ordinary tick, after cleanup removes
        # the entry, captures the exact pre-BOS state.
        pipeline._fc_async_bg.clear()
        with mock.patch(
            "nemotron_voicechat_runtime.server._eartts_epoch_step_snapshot",
            return_value=expected,
        ) as snapshot:
            self.assertIs(engine._eartts_epoch_before_model_step(), expected)
            snapshot.assert_called_once_with(engine)

    def test_legacy_transport_gate_rearms_after_agent_eos(self) -> None:
        gate = TransportModelGate(threshold_dbfs=-50.0, continuous_after_speech=False)

        self.assertTrue(gate.should_advance(-20.0))
        gate.observe_response_end()
        self.assertFalse(gate.should_advance(-120.0))

    def test_transport_gate_does_not_pause_before_acoustic_boundary_end(self) -> None:
        gate = TransportModelGate(threshold_dbfs=-50.0, continuous_after_speech=False)

        self.assertTrue(gate.should_advance(-20.0))
        # Text EOS alone does not rearm the transport gate; decoded tail frames
        # must still advance through the model.
        self.assertTrue(gate.should_advance(-120.0))
        gate.observe_response_end()
        self.assertFalse(gate.should_advance(-120.0))

    def test_acoustic_boundary_waits_for_decoded_silence_after_text_eos(self) -> None:
        boundary = AcousticResponseBoundary(
            threshold_dbfs=-90.0, required_silent_frames=3, max_tail_frames=10
        )

        self.assertEqual(boundary.observe("agent_bos", -30.0, 10)["event"], "start")
        eos = boundary.observe("agent_eos", -25.0, 20, max_response_frames=10)
        self.assertIsNone(eos["event"])
        self.assertEqual(eos["phase"], "tail_draining")
        self.assertEqual(eos["eos_reason"], "max_response_cap")
        self.assertIsNone(boundary.observe("pad", -120.0, 21)["event"])
        self.assertIsNone(boundary.observe("pad", -120.0, 22)["event"])
        complete = boundary.observe("pad", -120.0, 23)
        self.assertEqual(complete["event"], "end")
        self.assertEqual(complete["boundary_reason"], "post_eos_audio_silence")
        self.assertEqual(complete["post_eos_frames"], 3)
        self.assertFalse(complete["tail_timed_out"])

    def test_acoustic_boundary_marks_watchdog_and_tail_deadline(self) -> None:
        boundary = AcousticResponseBoundary(
            threshold_dbfs=-90.0, required_silent_frames=3, max_tail_frames=2
        )

        boundary.observe("agent_bos", -30.0, 1)
        eos = boundary.observe("agent_eos", -30.0, 2, watchdog_eos_pending=True)
        self.assertEqual(eos["eos_reason"], "decoded_silence_watchdog")
        boundary.observe("pad", -20.0, 3)
        complete = boundary.observe("pad", -20.0, 4)
        self.assertEqual(complete["boundary_reason"], "post_eos_tail_deadline")
        self.assertTrue(complete["tail_timed_out"])

    def test_acoustic_boundary_stops_delivery_without_forcing_model_eos(self) -> None:
        boundary = AcousticResponseBoundary(
            threshold_dbfs=-90.0,
            required_silent_frames=3,
            max_tail_frames=10,
            delivery_silence_frames=2,
        )

        boundary.observe("agent_bos", -30.0, 1)
        boundary.observe(None, -20.0, 2, text_delta="answer")
        self.assertEqual(boundary.observe("pad", -120.0, 3)["phase"], "responding")
        draining = boundary.observe("pad", -120.0, 4)
        self.assertEqual(draining["phase"], "internal_drain")
        self.assertTrue(draining["acoustic_complete"])
        self.assertIsNone(draining["event"])
        complete = boundary.observe("agent_eos", -120.0, 5)
        self.assertEqual(complete["event"], "end")
        self.assertEqual(complete["boundary_reason"], "post_eos_audio_silence")

    def test_acoustic_delivery_reopens_after_a_long_prosodic_pause(self) -> None:
        boundary = AcousticResponseBoundary(
            threshold_dbfs=-90.0,
            required_silent_frames=3,
            max_tail_frames=10,
            delivery_silence_frames=2,
        )

        boundary.observe("agent_bos", -30.0, 1, text_delta="first")
        boundary.observe("pad", -120.0, 2)
        draining = boundary.observe("pad", -120.0, 3)
        self.assertEqual(draining["phase"], "internal_drain")
        reopened = boundary.observe("pad", -25.0, 4)
        self.assertEqual(reopened["phase"], "responding")
        self.assertFalse(reopened["acoustic_complete"])

    def test_engine_resets_wrapper_global_state_across_sessions(self) -> None:
        class Wrapper:
            def __init__(self) -> None:
                self.pending = False
                self.redirect_queue = [1, 2]
                self.reset_calls = 0

            def reset_transport_session_state(self) -> None:
                self.pending = False
                self.redirect_queue = []
                self.reset_calls += 1

        class Pipeline:
            def __init__(self) -> None:
                self.s2s_model = Wrapper()
                self.reset_calls = 0
                self.generate_calls = 0
                self.prefill_calls = []

            def open_session(self) -> None:
                pass

            def generate_step(self, _frames) -> None:
                self.generate_calls += 1

            def prefill_for_new_stream(self, stream_id, system_prompt) -> None:
                self.prefill_calls.append((stream_id, system_prompt))

            def reset_session(self) -> None:
                self.reset_calls += 1

        pipeline = Pipeline()
        engine = VoiceChatEngine(pipeline)
        engine.start("brief system prompt")
        self.assertEqual(engine.system_prompt, "brief system prompt")
        self.assertEqual(pipeline.prefill_calls, [(1, "brief system prompt")])
        self.assertEqual(pipeline.generate_calls, 0)
        pipeline.s2s_model.pending = True
        pipeline.s2s_model.redirect_queue = [3]
        engine.abort()
        self.assertEqual(pipeline.generate_calls, 0)
        self.assertFalse(pipeline.s2s_model.pending)
        self.assertEqual(pipeline.s2s_model.redirect_queue, [])
        engine.user_text_prefix = "stale prefix"
        engine.user_text_segment = "stale segment"
        engine.start()
        self.assertEqual(
            pipeline.prefill_calls,
            [(1, "brief system prompt"), (2, "")],
        )
        self.assertFalse(pipeline.s2s_model.pending)
        self.assertEqual(pipeline.s2s_model.reset_calls, 3)
        self.assertEqual(pipeline.reset_calls, 1)
        self.assertEqual(engine.user_text_prefix, "")
        self.assertEqual(engine.user_text_segment, "")

    def test_engine_start_failure_is_transactional_and_restartable(self) -> None:
        class Wrapper:
            def __init__(self, pipeline) -> None:
                self.pipeline = pipeline
                self.reset_calls = 0

            def reset_transport_session_state(self) -> None:
                self.reset_calls += 1

            def set_external_user_eou_mode(self, _enabled: bool) -> None:
                if self.pipeline.fail_phase == "finalize":
                    raise RuntimeError("finalize failed")

        class Pipeline:
            def __init__(self, fail_phase: str) -> None:
                self.fail_phase = fail_phase
                self.reset_calls = 0
                self.open_calls = 0
                self.prefill_calls = 0
                self.s2s_model = Wrapper(self)

            def open_session(self) -> None:
                self.open_calls += 1
                if self.fail_phase == "open_session":
                    raise RuntimeError("open failed")

            def prefill_for_new_stream(self, _stream_id, _system_prompt) -> None:
                self.prefill_calls += 1
                if self.fail_phase == "prefill":
                    raise RuntimeError("prefill failed")

            def reset_session(self) -> None:
                self.reset_calls += 1

        for phase in ("open_session", "prefill", "finalize"):
            with self.subTest(phase=phase):
                pipeline = Pipeline(phase)
                engine = VoiceChatEngine(pipeline)
                engine.external_user_eou_mode = phase == "finalize"
                with self.assertRaises(VoiceChatEngineStartFailure) as raised:
                    engine.start("first prompt")
                self.assertEqual(raised.exception.phase, phase)
                self.assertIn("total", raised.exception.timings_ms)
                self.assertFalse(engine.started)
                self.assertEqual(engine.system_prompt, "")
                self.assertEqual(pipeline.reset_calls, 1)

                pipeline.fail_phase = ""
                evidence = engine.start("second prompt")
                self.assertTrue(engine.started)
                self.assertEqual(engine.system_prompt, "second prompt")
                self.assertEqual(evidence["schema"], 1)
                self.assertEqual(
                    set(evidence["timings_ms"]),
                    {"reset", "open_session", "prefill", "finalize", "total"},
                )

    def test_engine_enables_external_eou_after_upstream_prefill_reset(self) -> None:
        class Wrapper:
            def __init__(self) -> None:
                self.external_eou = False

            def reset_transport_session_state(self) -> None:
                self.external_eou = False

            def set_external_user_eou_mode(self, enabled: bool) -> None:
                self.external_eou = enabled

        class Pipeline:
            def __init__(self) -> None:
                self.s2s_model = Wrapper()

            def open_session(self) -> None:
                pass

            def prefill_for_new_stream(self, _stream_id, _system_prompt) -> None:
                # Mirrors pinned NVIDIA Speech: prefill resets RNNT state.
                self.s2s_model.external_eou = False

        pipeline = Pipeline()
        engine = VoiceChatEngine(pipeline)
        engine.external_user_eou_mode = True
        engine.start()

        self.assertTrue(pipeline.s2s_model.external_eou)

    def test_agent_silence_watchdog_waits_for_text_and_resets_on_audio(self) -> None:
        watchdog = AgentSilenceEosWatchdog(threshold_dbfs=-90.0, required_frames=3)

        self.assertFalse(watchdog.observe("agent_bos", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "answer", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -40.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("agent_eos", "", -120.0))
        self.assertFalse(watchdog.agent_open)

    def test_agent_silence_watchdog_does_not_count_tts_startup_silence(self) -> None:
        watchdog = AgentSilenceEosWatchdog(threshold_dbfs=-90.0, required_frames=3)

        self.assertFalse(watchdog.observe("agent_bos", "answer", -120.0))
        for _ in range(6):
            self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.audible_seen)
        self.assertEqual(watchdog.silent_frames, 0)

        self.assertFalse(watchdog.observe("pad", "", -40.0))
        self.assertTrue(watchdog.audible_seen)
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))

    def test_agent_silence_watchdog_terminates_zero_text_turn_from_bos(self) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=3,
            no_text_required_frames=3,
            no_audio_required_frames=100,
        )

        self.assertFalse(watchdog.observe("agent_bos", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -40.0))
        self.assertTrue(watchdog.observe("pad", "", -40.0))
        self.assertEqual(watchdog.request_reason, "no_text_since_bos_watchdog")
        self.assertEqual(watchdog.no_text_frames, 3)
        self.assertFalse(watchdog.observe("pad", "", -120.0))

    def test_agent_silence_watchdog_covers_short_text_without_audio(self) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=3,
            no_text_required_frames=100,
            no_audio_required_frames=3,
        )

        self.assertFalse(watchdog.observe("agent_bos", "brief", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))
        self.assertEqual(watchdog.request_reason, "no_audio_since_bos_watchdog")

    def test_decoded_silence_watchdog_arms_without_text_seen(self) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=3,
            no_text_required_frames=100,
            no_audio_required_frames=100,
        )

        self.assertFalse(watchdog.observe("agent_bos", "", -40.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.text_seen)
        self.assertEqual(watchdog.request_reason, "decoded_silence_watchdog")

    def test_agent_startup_watchdog_function_activity_resets_but_does_not_disable_deadline(
        self,
    ) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=3,
            no_text_required_frames=3,
            no_audio_required_frames=3,
        )

        self.assertFalse(
            watchdog.observe("agent_bos", "", -120.0, function_delta='{"name":"tool"}')
        )
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.function_seen)
        self.assertEqual(watchdog.no_text_frames, 3)
        self.assertEqual(watchdog.no_audio_frames, 3)
        self.assertEqual(watchdog.request_reason, "no_text_since_bos_watchdog")

    def test_agent_startup_watchdog_continuous_text_cannot_hide_silent_audio(self) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=100,
            no_text_required_frames=100,
            no_audio_required_frames=3,
        )

        self.assertFalse(watchdog.observe("agent_bos", "one", -120.0))
        self.assertFalse(watchdog.observe("pad", "two", -120.0))
        self.assertTrue(watchdog.observe("pad", "three", -120.0))
        self.assertEqual(watchdog.no_text_frames, 0)
        self.assertEqual(watchdog.request_reason, "no_audio_since_bos_watchdog")

    def test_acoustic_boundary_records_bos_startup_watchdog_reason(self) -> None:
        boundary = AcousticResponseBoundary(
            threshold_dbfs=-90.0, required_silent_frames=3, max_tail_frames=2
        )

        boundary.observe("agent_bos", -120.0, 10)
        eos = boundary.observe(
            "agent_eos",
            -120.0,
            11,
            watchdog_eos_pending=True,
            watchdog_eos_reason="no_text_since_bos_watchdog",
        )
        self.assertEqual(eos["eos_reason"], "no_text_since_bos_watchdog")

    def test_agent_silence_watchdog_ignores_duplicate_bos_inside_open_turn(
        self,
    ) -> None:
        watchdog = AgentSilenceEosWatchdog(threshold_dbfs=-90.0, required_frames=3)

        self.assertFalse(watchdog.observe("agent_bos", "answer", -40.0))
        self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertFalse(watchdog.observe("agent_bos", "", -120.0))
        self.assertTrue(watchdog.observe("pad", "", -120.0))
        self.assertEqual(watchdog.silent_frames, 3)

    def test_pcm16_round_trip_and_clipping(self) -> None:
        source = np.array([-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0], dtype=np.float32)
        decoded = pcm16_bytes_to_float32(float32_to_pcm16_bytes(source))
        expected = np.clip(source, -1.0, 1.0)
        np.testing.assert_allclose(decoded, expected, atol=1 / 32768)

    def test_frame_buffer_preserves_arbitrary_packet_boundaries(self) -> None:
        source = np.linspace(-0.8, 0.8, 11, dtype=np.float32)
        encoded = float32_to_pcm16_bytes(source)
        buffer = PcmFrameBuffer(frame_samples=4)

        frames = []
        frames.extend(buffer.push(encoded[:6]))
        frames.extend(buffer.push(encoded[6:16]))
        frames.extend(buffer.push(encoded[16:]))

        self.assertEqual([frame.shape for frame in frames], [(4,), (4,)])
        reconstructed = np.concatenate([*frames, buffer.flush()])
        decoded = pcm16_bytes_to_float32(encoded)
        np.testing.assert_array_equal(reconstructed[: decoded.size], decoded)
        np.testing.assert_array_equal(reconstructed[decoded.size :], np.zeros(1))

    def test_rejects_odd_pcm_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd byte count"):
            pcm16_bytes_to_float32(b"\x00")

    def test_frame_buffer_discard_prevents_cross_source_packet_merge(self) -> None:
        buffer = PcmFrameBuffer(frame_samples=4)
        assert buffer.push(b"\x01\x00" * 3) == []
        assert buffer.discard() == 3
        frames = buffer.push(b"\x02\x00" * 4)
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], np.full(4, 2 / 32768))

    def test_display_text_removes_only_stream_markers(self) -> None:
        value = "^<s>Hello,   world!</s> <$1.2$> <SPECIAL_12>"
        self.assertEqual(clean_display_text(value), "Hello, world!")

    def test_audio_stats_reports_silence_and_signal(self) -> None:
        self.assertEqual(
            audio_stats(np.zeros(4, dtype=np.float32)),
            {"samples": 4, "rms": 0.0, "rms_dbfs": -120.0, "peak": 0.0},
        )
        stats = audio_stats(np.array([-0.5, 0.5], dtype=np.float32))
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["rms"], 0.5)
        self.assertEqual(stats["rms_dbfs"], -6.021)
        self.assertEqual(stats["peak"], 0.5)

    def test_session_trace_preserves_exact_pcm_and_json_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = SessionTrace(Path(directory), "test-session")
            trace.input_pcm(b"\x01\x02\x03\x04")
            trace.output_pcm(b"\x05\x06")
            trace.event("model_step", frame=7)
            trace.close("test")

            root = Path(directory) / "test-session"
            self.assertEqual(
                (root / "input-16000-mono-s16le.pcm").read_bytes(),
                b"\x01\x02\x03\x04",
            )
            self.assertEqual((root / "output-22050-mono-s16le.pcm").read_bytes(), b"\x05\x06")
            events = [
                __import__("json").loads(line)
                for line in (root / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["session_trace_started", "model_step", "session_trace_stopped"],
            )


if __name__ == "__main__":
    unittest.main()
