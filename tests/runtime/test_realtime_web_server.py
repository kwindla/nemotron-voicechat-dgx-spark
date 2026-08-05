from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "src" / "nemotron_voicechat_runtime"

from nemotron_voicechat_runtime.provenance import RUNTIME_SOURCE_PATHS
from nemotron_voicechat_runtime.server import (
    PUBLIC_VLLM_MANIFEST_KINDS,
    AcousticResponseBoundary,
    AgentSilenceEosWatchdog,
    PcmFrameBuffer,
    SessionTrace,
    TransportModelGate,
    VoiceChatEngine,
    audio_stats,
    clean_display_text,
    clear_stale_agent_eos_latch,
    create_app,
    env_bool,
    float32_to_pcm16_bytes,
    host_runtime_provenance,
    parse_nvidia_driver_version,
    pcm16_bytes_to_float32,
    release_component_manifest,
    render_tool_system_prompt,
    response_audio_is_deliverable,
    session_position_limit_event,
    validate_manifested_model,
    validate_nano_pad_pair_runtime,
    warm_realtime_engine,
    websocket_ping_interval,
    websocket_ping_timeout,
)


class RealtimeWebServerTest(unittest.TestCase):
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
        self.assertFalse(pipeline.s2s_model.pending)
        self.assertEqual(pipeline.s2s_model.redirect_queue, [])
        engine.start()
        self.assertEqual(
            pipeline.prefill_calls,
            [(1, "brief system prompt"), (2, "")],
        )
        self.assertFalse(pipeline.s2s_model.pending)
        self.assertEqual(pipeline.s2s_model.reset_calls, 3)
        self.assertEqual(pipeline.reset_calls, 1)

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

    def test_agent_startup_watchdog_does_not_terminate_function_activity(self) -> None:
        watchdog = AgentSilenceEosWatchdog(
            threshold_dbfs=-90.0,
            required_frames=3,
            no_text_required_frames=3,
            no_audio_required_frames=3,
        )

        self.assertFalse(
            watchdog.observe("agent_bos", "", -120.0, function_delta='{"name":"tool"}')
        )
        for _ in range(5):
            self.assertFalse(watchdog.observe("pad", "", -120.0))
        self.assertTrue(watchdog.function_seen)
        self.assertEqual(watchdog.no_text_frames, 0)
        self.assertEqual(watchdog.no_audio_frames, 0)
        self.assertIsNone(watchdog.request_reason)

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
