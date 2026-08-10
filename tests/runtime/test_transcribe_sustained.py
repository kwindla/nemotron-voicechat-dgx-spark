from __future__ import annotations

import copy
import importlib.util
import json
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_module():
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    path = Path("tools/qualification/transcribe_sustained.py")
    try:
        spec = importlib.util.spec_from_file_location("transcribe_sustained", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification)


def test_word_error_rate_normalizes_spoken_numbers() -> None:
    module = load_module()
    assert module.word_error_rate("The answer is 20.", "the answer is twenty") == 0
    assert module.word_error_rate("one two", "one four") == 0.5


def test_semantic_candidate_accepts_literal_and_spoken_utc_time() -> None:
    module = load_module()
    assert module.semantic_candidate_match("The result is sapphire.", "sapphire")
    assert module.semantic_candidate_match(
        "The current UTC time is 22 hours and 6 minutes UTC.", "22:06 UTC"
    )
    assert module.semantic_candidate_match("The time is 22 hours 6 minutes UTC.", "22:06 UTC")
    assert not module.semantic_candidate_match(
        "The current UTC time is 22 hours and 7 minutes UTC.", "22:06 UTC"
    )
    assert not module.semantic_candidate_match("It is 22:06 local time.", "22:06 UTC")


def test_semantic_gate_requires_own_secret_and_excludes_other_case_secrets() -> None:
    module = load_module()

    assert module.response_semantic_matches(
        "The codeword is topaz.", ["topaz"], ["sapphire", "amber"]
    ) == (True, True)
    assert module.response_semantic_matches(
        "The codeword is sapphire.", ["topaz"], ["sapphire", "amber"]
    ) == (False, False)
    assert module.response_semantic_matches(
        "No codeword was supplied.", [], ["sapphire", "topaz"]
    ) == (True, True)


def test_runtime_gate_verdict_is_recomputed_independently_of_prior_asr() -> None:
    module = load_module()
    report = {
        "passed": False,
        "source_frames": 15_000,
        "sent_frames": 13_000,
        "metric_sequence_complete": True,
        "queue": {"passed": True},
        "typed_jobs": {str(index): {} for index in range(18)},
        "typed_completed": 18,
        "typed_answered": 18,
        "typed_unanswered": 0,
        "allowed_unanswered_at_fp32_base_rate": 1,
        "responses": [{} for _ in range(18)],
        "unexpected_errors": [],
        "expected_session_limit_error_count": 1,
        "expected_session_position_limit": True,
        "session_closed": {"reason": "session_position_limit"},
        "external_asr": {"passed": False},
    }
    assert module.runtime_gate_passed(report)
    report["queue"]["passed"] = False
    assert not module.runtime_gate_passed(report)
    report["queue_required"] = False
    assert module.runtime_gate_passed(report)

    report["tool_prompt_requested"] = True
    report["tool_response_channel_gate"] = False
    assert not module.runtime_gate_passed(report)
    report["tool_response_channel_gate"] = True
    for index in range(4):
        report["typed_jobs"][str(index)]["expected_tool_call"] = True
    report["tool_response_checks"] = [
        {
            "eotr_valid": True,
            "function_channel_idle": True,
            "post_eotr_audible": True,
            "bounded_watchdog_close": True,
            "text_nonempty": True,
            "text_semantic_match": True,
            "audio_nonempty": True,
        }
        for _ in range(4)
    ]
    report["eotr_observations"] = [{"observed_count": index} for index in range(1, 5)]
    report["tool_cycle_cardinality"] = {
        "requested_tool_prompts": 4,
        "validated_tool_responses": 4,
        "eotr_observed_count": 4,
        "client_eou_sotc_committed_count": 4,
        "post_fc_client_bos_forced_count": 4,
    }
    assert module.runtime_gate_passed(report)
    report["tool_cycle_cardinality"]["eotr_observed_count"] = 3
    assert not module.runtime_gate_passed(report)
    report["tool_cycle_cardinality"]["eotr_observed_count"] = 4
    report["eotr_observations"][-1]["observed_count"] = 3
    assert not module.runtime_gate_passed(report)
    report["eotr_observations"][-1]["observed_count"] = 4
    report["tool_response_checks"][0]["post_eotr_audible"] = False
    assert not module.runtime_gate_passed(report)

    report["serialized_typed_error"] = {"type": "TimeoutError"}
    assert not module.runtime_gate_passed(report)


def complete_multiturn_report() -> dict:
    return {
        "report_kind": "multiturn_strict_v3",
        "passed": False,
        "fixture_generator": "Pocket TTS",
        "manifest": "/retained/manifest.json",
        "expected_turn_count": 3,
        "max_input_wer": 0.35,
        "responses": [
            {
                "audio_bytes": 1024,
                "text": "A valid response.",
                "input_wer": 0.0,
                "semantic_match": True,
            }
            for _ in range(3)
        ],
        "session_closed": {"reason": "client_stop"},
        "external_asr": {"passed": False},
    }


def test_multiturn_runtime_gate_recomputes_original_runtime_invariants() -> None:
    module = load_module()
    report = complete_multiturn_report()

    assert module.runtime_gate_passed(report)
    for mutate in (
        lambda value: value.update({"expected_turn_count": 2}),
        lambda value: value.update({"max_input_wer": -0.1}),
        lambda value: value.update({"session_closed": None}),
        lambda value: value["responses"][0].update({"audio_bytes": 0}),
        lambda value: value["responses"][0].update({"text": ""}),
        lambda value: value["responses"][0].update({"input_wer": 0.36}),
        lambda value: value["responses"][0].update({"semantic_match": False}),
    ):
        candidate = copy.deepcopy(report)
        mutate(candidate)
        assert not module.runtime_gate_passed(candidate)


def test_legacy_multiturn_report_is_recognized_without_trusting_prior_verdict() -> None:
    module = load_module()
    report = complete_multiturn_report()
    del report["report_kind"]
    del report["expected_turn_count"]
    del report["max_input_wer"]

    assert module.report_kind(report) == module.MULTITURN_REPORT_KIND
    assert module.runtime_gate_passed(report)


def test_report_kind_dispatch_fails_closed_for_unknown_and_ambiguous_legacy_shapes() -> None:
    module = load_module()
    unknown = complete_multiturn_report()
    unknown["report_kind"] = "future_report"
    assert module.report_kind(unknown) is None
    assert not module.runtime_gate_passed(unknown)

    ambiguous = complete_multiturn_report()
    del ambiguous["report_kind"]
    ambiguous.update(
        {
            "source_frames": 100,
            "sent_frames": 100,
            "typed_jobs": {},
            "typed_completed": 1,
            "typed_answered": 1,
            "typed_unanswered": 0,
            "queue": {"passed": True},
            "metric_sequence_complete": True,
        }
    )
    assert module.report_kind(ambiguous) is None
    assert not module.runtime_gate_passed(ambiguous)


def test_memory_matrix_runtime_gate_is_recomputed_from_all_cases() -> None:
    module = load_module()
    from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES

    report = {
        "passed": False,
        "matrix": {"seed_modalities": ["voice", "text", "system"]},
        "fresh_session_per_case": True,
        "runtime_provenance": {
            "runtime_image_id": f"sha256:{'a' * 64}",
            "source_sha256": {name: "b" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
        },
        "runtime_provenance_consistent": True,
        "negative_control_ran_after_seeded": True,
        "expected_case_count": 7,
        "cases": [
            *[{"passed": True, "negative_control": False} for _ in range(6)],
            {"passed": True, "negative_control": True},
        ],
    }

    assert module.runtime_gate_passed(report)
    report["cases"][1]["passed"] = False
    assert not module.runtime_gate_passed(report)


def test_memory_matrix_asr_gate_rejects_partial_case_report() -> None:
    module = load_module()
    from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES

    report = {
        "matrix": {"seed_modalities": ["voice", "text", "system"]},
        "fresh_session_per_case": True,
        "runtime_provenance": {
            "runtime_image_id": f"sha256:{'a' * 64}",
            "source_sha256": {name: "b" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
        },
        "runtime_provenance_consistent": True,
        "negative_control_ran_after_seeded": False,
        "expected_case_count": 1,
        "cases": [{"passed": True, "negative_control": False}],
    }

    assert not module.runtime_gate_passed(report)


def test_full_memory_matrix_gate_rejects_negative_control_not_run_last() -> None:
    module = load_module()
    from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES

    cases = [{"passed": True, "negative_control": False} for _ in range(6)] + [
        {"passed": True, "negative_control": True}
    ]
    report = {
        "matrix": {},
        "fresh_session_per_case": True,
        "runtime_provenance": {
            "runtime_image_id": f"sha256:{'a' * 64}",
            "source_sha256": {name: "b" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
        },
        "runtime_provenance_consistent": True,
        "negative_control_ran_after_seeded": True,
        "expected_case_count": 7,
        "cases": cases,
    }

    assert module.runtime_gate_passed(report)
    report["cases"] = report["cases"][-1:] + report["cases"][:-1]
    assert not module.runtime_gate_passed(report)


def test_external_asr_negative_matcher_goes_red_on_synthetic_leak() -> None:
    module = load_module()

    assert module.response_semantic_matches(
        "No codeword was supplied.", [], ["sapphire", "topaz"]
    ) == (True, True)
    assert module.response_semantic_matches(
        "No codeword was supplied, but perhaps sapphire.",
        [],
        ["sapphire", "topaz"],
    ) == (True, False)


def test_parity_asr_preserves_positive_semantics_as_measurement_only() -> None:
    module = load_module()
    arguments = {
        "hypothesis": "The fresh word is ready now.",
        "reference": "River is the first fresh word, ready now.",
        "expected": ["river"],
        "excluded": [],
        "max_wer": 0.5,
    }

    parity = module.response_asr_evaluation(**arguments, parity_arm=True)
    legacy = module.response_asr_evaluation(**arguments, parity_arm=False)

    assert parity["semantic_match"] is False
    assert parity["audio_integrity_passed"] is True
    assert parity["passed"] is True
    assert legacy["passed"] is False
    assert "audio_integrity_passed" not in legacy


def complete_tool_freshness_report() -> dict:
    qualification = str(Path("tools/qualification").resolve())
    sys.path.insert(0, qualification)
    try:
        from generate_tool_freshness_fixtures import scenario_document
        from sustained_strict_v3 import (
            evaluate_tool_cycle_cardinality,
            evaluate_tool_response_channels,
        )
        from tool_freshness_parity import (
            RESPONSE_SINGLETON_EVENT_TYPES,
            derive_structural_conjuncts,
            experiment_sha256,
        )
    finally:
        sys.path.remove(qualification)
    from nemotron_voicechat_runtime.pocket_worker import EXPECTED_ASSETS
    from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES
    from nemotron_voicechat_runtime.tool_freshness_fixture import (
        FIXTURE_INSTALLED_WORKER_SOURCE,
        FIXTURE_PROJECT_SOURCE_PATHS,
    )

    scenario = scenario_document()
    synthesis = {
        "language": "english_2026-04",
        "voice": "alba",
        "quantize": True,
        "threads": 4,
        "cpus": [7, 8, 9, 15],
        "base_seed": 0,
        "seed_scheme": "sha256-text-plus-base-v1",
        "package_version": "2.1.0",
        "assets": {
            name: {"bytes": details[0], "sha256": details[1]}
            for name, details in EXPECTED_ASSETS.items()
        },
    }
    health = {
        "ready": True,
        "language": synthesis["language"],
        "voice": synthesis["voice"],
        "threads": synthesis["threads"],
        "cpus": synthesis["cpus"],
        "seed": synthesis["base_seed"],
        "seed_scheme": synthesis["seed_scheme"],
        "package_version": synthesis["package_version"],
        "assets": synthesis["assets"],
    }
    runtime_image_id = f"sha256:{'a' * 64}"
    fixture_rows = [
        {
            "case_id": f"tool-freshness-{ordinal}",
            "ordinal": ordinal,
            "text": prompt,
            "pcm": {
                "path": f"tool-freshness-{ordinal}.pcm",
                "sample_rate": 16_000,
                "sample_count": 128,
                "bytes": 256,
                "sha256": format(ordinal, "x") * 64,
            },
            "wav": {
                "path": f"tool-freshness-{ordinal}.wav",
                "sample_rate": 16_000,
                "channels": 1,
                "sample_width": 2,
                "samples": 128,
                "bytes": 300,
                "sha256": format(ordinal + 4, "x") * 64,
            },
        }
        for ordinal, prompt in enumerate(scenario["prompts"], 1)
    ]
    inputs = []
    responses = []
    metrics = []
    observations = []
    for ordinal, prompt in enumerate(scenario["prompts"], 1):
        turn_id = f"turn-{ordinal}"
        job_id = f"job-{ordinal}"
        inputs.append(
            {
                "ordinal": ordinal,
                "case_id": f"tool-freshness-{ordinal}",
                "input_modality": "typed",
                "turn_id": turn_id,
                "job_id": job_id,
                "expected_input_text": prompt,
                "requested_text": prompt,
                "accepted": True,
                "started": True,
                "disposition": "completed",
                "speech_started": True,
                "transcription_source": "typed",
                "transcription_job_id": job_id,
                "transcription_turn_id": turn_id,
                "event_counts": {
                    "input_text.accepted": 1,
                    "input_text.injection_started": 1,
                    "input_text.injection_finished": 1,
                    "input_audio_buffer.speech_started": 1,
                    "conversation.item.input_audio_transcription.completed": 1,
                },
                "event_correlations": [True] * 5,
            }
        )
        responses.append(
            {
                "turn_id": turn_id,
                "case_id": f"tool-freshness-{ordinal}",
                "ordinal": ordinal,
                "response_id": f"response-{ordinal}",
                "text": "The fresh verification word is available.",
                "audio_bytes": 1024,
                "done_reason": "decoded_silence_watchdog",
                "expected_response_any": [
                    scenario["tool_outputs"][ordinal - 1]["expected_response"]
                ],
                "expected_response_none": scenario["tool_outputs"][ordinal - 1][
                    "excluded_responses"
                ],
                "expected_text_semantic_required": True,
                "tool_calls": [{"name": scenario["tool"]["name"], "arguments": {}}],
                "tool_call_correlated": True,
                "function_output_applied": True,
                "response_brackets_balanced": True,
                "protocol_correlation": True,
                "function_output_sent": json.dumps(
                    scenario["tool_outputs"][ordinal - 1]["function_output"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "event_counts": {kind: 1 for kind in RESPONSE_SINGLETON_EVENT_TYPES},
                "event_correlations": [True] * 6,
            }
        )
        observations.append(
            {
                "turn_id": turn_id,
                "transport_frame": ordinal * 10,
                "observed": True,
                "observed_frame": ordinal * 100,
                "wait_steps": 1,
                "timeout": False,
                "eotr_token_id": 5,
                "first_post_drain_token_id": 5,
                "first_post_drain_token_frame": ordinal * 100,
                "unexpected_token_id": None,
                "unexpected_token_frame": None,
                "feedback_token_id": 5,
                "feedback_frame": ordinal * 100,
                "feedback_committed": True,
            }
        )
        metrics.append(
            {
                "response_id": f"response-{ordinal}",
                "transport_frame": ordinal * 10 + 1,
                "output_audio": {"rms_dbfs": -20.0},
                "turn_state": {
                    "function_calling": {
                        "effective_token_is_pad": True,
                        "client_eou_sotc_committed_count": ordinal,
                        "post_fc_client_bos_forced_count": ordinal,
                        "eotr_observed_count": ordinal,
                    },
                    "agent_silence_watchdog": {"threshold_dbfs": -90.0},
                },
            }
        )
    checks, _semantic_coupled_gate = evaluate_tool_response_channels(
        responses, metrics, observations
    )
    cardinality = evaluate_tool_cycle_cardinality(
        requested_tool_prompts=4,
        tool_response_checks=checks,
        metrics=metrics,
    )
    conjuncts = derive_structural_conjuncts(
        modality="typed",
        inputs=inputs,
        responses=responses,
        checks=checks,
        cardinality=cardinality,
        errors=[],
        session_closed={"reason": "client_stop", "status": "completed"},
        scenario=scenario,
        fixtures=fixture_rows,
        scenario_identity=True,
        runtime_fixture_identity=True,
    )
    module = load_module()
    scenario_sha256 = module.canonical_json_sha256(scenario)
    from nemotron_voicechat_runtime.tool_freshness_fixture import (
        FIXTURE_KIND,
        FIXTURE_SCHEMA,
    )

    manifest = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "scenario_sha256": scenario_sha256,
        "fixtures": fixture_rows,
        "synthesis": synthesis,
        "provenance": {
            "runtime_image": "runtime:test",
            "runtime_image_id": runtime_image_id,
            "source_sha256": {
                name: "d" * 64
                for name in (
                    *FIXTURE_PROJECT_SOURCE_PATHS,
                    FIXTURE_INSTALLED_WORKER_SOURCE,
                )
            },
        },
    }
    from nemotron_voicechat_runtime.tool_freshness_fixture import self_hash

    manifest["sha256"] = self_hash(manifest)
    runtime_provenance = {
        "runtime_image_id": runtime_image_id,
        "source_sha256": {name: "c" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
    }
    return {
        "report_kind": "tool_freshness_parity_arm_v1",
        "schema": 1,
        "input_modality": "typed",
        "model_input_path": "server_typed_pocket",
        "scenario": scenario,
        "scenario_sha256": scenario_sha256,
        "experiment_sha256": experiment_sha256(
            scenario_sha256=scenario_sha256,
            fixture_manifest_sha256=manifest["sha256"],
        ),
        "fixture_manifest": manifest,
        "live_typed_input_health": health,
        "health_snapshot": {
            "typed_input": health,
            "checkpoint": {"runtime_provenance": runtime_provenance},
        },
        "health_runtime_provenance": runtime_provenance,
        "runtime_provenance": runtime_provenance,
        "session_id": "session-typed",
        "session_configuration": {
            "id": "session-typed",
            "instructions": scenario["instructions"],
            "tools": [scenario["tool"]],
        },
        "inputs": inputs,
        "responses": responses,
        "metrics": metrics,
        "eotr_observations": observations,
        "tool_response_checks": checks,
        "tool_cycle_cardinality": cardinality,
        "errors": [],
        "session_closed": {"reason": "client_stop", "status": "completed"},
        "structural_conjuncts": conjuncts,
        "structural_passed": True,
    }


def test_tool_freshness_runtime_gate_rederives_structure_without_semantics() -> None:
    module = load_module()
    report = complete_tool_freshness_report()

    assert report["tool_response_checks"][0]["text_semantic_match"] is False
    assert module.runtime_gate_passed(report)

    for mutation in (
        lambda value: value["scenario"].update({"instructions": "changed"}),
        lambda value: value["live_typed_input_health"].update({"threads": 3}),
        lambda value: value["runtime_provenance"].update({"source_sha256": {}}),
        lambda value: value.update({"model_input_path": "client_pocket_fixture_pcm"}),
        lambda value: value["structural_conjuncts"].update({"function_channel_idle": False}),
        lambda value: value["metrics"][0]["turn_state"]["function_calling"].update(
            {"effective_token_is_pad": False}
        ),
        lambda value: value["inputs"][0].update({"transcription_job_id": "wrong"}),
    ):
        candidate = copy.deepcopy(report)
        mutation(candidate)
        assert not module.runtime_gate_passed(candidate)

    from nemotron_voicechat_runtime.tool_freshness_fixture import self_hash

    for manifest_mutation in (
        lambda value: value["fixtures"][0].update({"case_id": "wrong"}),
        lambda value: value["fixtures"][0].pop("pcm"),
        lambda value: value.update({"schema": 1}),
        lambda value: value.update({"kind": "wrong-kind"}),
        lambda value: value["synthesis"].update({"assets": {}}),
        lambda value: value["fixtures"][0]["pcm"].update({"sha256": "z" * 64}),
    ):
        resigned_fixture_mutation = copy.deepcopy(report)
        manifest_mutation(resigned_fixture_mutation["fixture_manifest"])
        resigned_fixture_mutation["fixture_manifest"]["sha256"] = self_hash(
            resigned_fixture_mutation["fixture_manifest"]
        )
        resigned_fixture_mutation["experiment_sha256"] = module.canonical_json_sha256(
            {
                "scenario_sha256": resigned_fixture_mutation["scenario_sha256"],
                "fixture_manifest_sha256": resigned_fixture_mutation["fixture_manifest"]["sha256"],
            }
        )
        assert not module.runtime_gate_passed(resigned_fixture_mutation)

    wrong_session_configuration = copy.deepcopy(report)
    wrong_session_configuration["session_configuration"]["instructions"] = "changed"
    assert not module.runtime_gate_passed(wrong_session_configuration)


def test_tool_freshness_recompute_imports_without_websocket_client_dependency() -> None:
    report = complete_tool_freshness_report()
    original_import = __import__

    def import_without_websockets(name, *args, **kwargs):
        if name == "websockets" or name.startswith("websockets."):
            raise ModuleNotFoundError("simulated evaluator image: no websockets")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_websockets):
        module = load_module()
        assert not module.tool_freshness_runtime_gate_passed({"input_modality": "typed"})
        assert module.tool_freshness_runtime_gate_passed(report)


def test_audio_level_and_resampling(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source.wav"
    tone = (np.sin(np.arange(22_050) * 2 * np.pi * 440 / 22_050) * 0.1 * 32767).astype("<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22_050)
        output.writeframes(tone.tobytes())
    values, source_rate = module.read_and_resample(source)
    assert source_rate == 22_050
    assert len(values) == 16_000
    assert -24 < module.dbfs(values) < -22


def test_container_audio_path_falls_back_to_mounted_run_directory(tmp_path: Path) -> None:
    module = load_module()
    mounted = tmp_path / "response-001.wav"
    mounted.touch()
    assert module.resolve_audio_path(tmp_path, "/host/elsewhere/response-001.wav") == mounted


def test_container_audio_path_preserves_matrix_subdirectory(tmp_path: Path) -> None:
    module = load_module()
    mounted = tmp_path / "voice-to-text/response.wav"
    mounted.parent.mkdir()
    mounted.touch()
    assert module.resolve_audio_path(tmp_path, "voice-to-text/response.wav") == mounted
