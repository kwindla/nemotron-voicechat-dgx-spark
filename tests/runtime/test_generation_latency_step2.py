from __future__ import annotations

import asyncio
import copy
import dataclasses
import dis
import pickle
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron_voicechat_runtime.artifacts import load_config
from nemotron_voicechat_runtime.provenance import (
    QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV,
    QUALIFICATION_MODE_ENV,
    ValidatedQualificationMode,
    checked_in_candidate_environments,
    is_validated_qualification_mode,
    validate_qualification_watchdog_production_policy,
)
from nemotron_voicechat_runtime.server import (
    AgentSilenceEosWatchdog,
    QualificationAgentSilenceEosWatchdog,
    VoiceChatEngine,
    create_app,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def candidate_environment(config_name: str = "production-candidate-1.toml") -> dict[str, str]:
    return dict(load_config(REPO_ROOT / "config" / config_name)["runtime"]["environment"])


def validated_mode(
    *, frames: int = 125, config_name: str = "production-candidate-1.toml"
) -> ValidatedQualificationMode:
    environment = candidate_environment(config_name)
    environment.update(
        {
            "VOICECHAT_WEB_SYSTEM_PROMPT": "Qualification latency fixture",
            QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV: str(frames),
            QUALIFICATION_MODE_ENV: "step2-latency",
        }
    )
    result = validate_qualification_watchdog_production_policy(environment)
    assert result is not None
    return result


def test_checked_candidate_loader_and_true_intersection_match_raw_tomls() -> None:
    loaded = checked_in_candidate_environments()
    raw: dict[str, dict[str, str]] = {}
    for path in sorted((REPO_ROOT / "config").glob("production-candidate-*.toml")):
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        raw[config["candidate"]] = config["runtime"]["environment"]

    assert loaded == raw
    environments = list(raw.values())
    intersection = {
        name: value
        for name, value in environments[0].items()
        if all(environment.get(name) == value for environment in environments[1:])
    }
    assert len(environments[0]) == 61
    assert len(environments[1]) == 67
    assert len(intersection) == 57
    assert {
        "VOICECHAT_TYPED_INPUT_CPU_CORES",
        "VOICECHAT_TYPED_INPUT_EOU_MARGIN",
        "VOICECHAT_TYPED_INPUT_FRAME_MS",
        "VOICECHAT_TYPED_INPUT_LANGUAGE",
        "VOICECHAT_TYPED_INPUT_MAX_TEXT_CHARS",
        "VOICECHAT_TYPED_INPUT_PREWARM",
        "VOICECHAT_TYPED_INPUT_QUANTIZE",
        "VOICECHAT_TYPED_INPUT_TORCH_THREADS",
        "VOICECHAT_TYPED_INPUT_VOICE",
    } <= intersection.keys()


@pytest.mark.parametrize(
    "config_name", ["production-candidate-1.toml", "production-candidate-2.toml"]
)
def test_positive_qualification_identity_accepts_only_allowlisted_deltas(
    config_name: str,
) -> None:
    mode = validated_mode(config_name=config_name)

    assert mode.label == "step2-latency"
    assert mode.no_text_watchdog_override_frames == 125
    assert len(mode.environment_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        mode.label = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HF_HUB_OFFLINE", "0"),
        ("EA_CPU_CODEC_THREADS", "3"),
        ("VOICECHAT_TYPED_INPUT_PREWARM", "0"),
        ("VOICECHAT_UNDECLARED_QUALIFICATION_DRIFT", "1"),
    ],
)
def test_one_key_near_production_attacks_are_rejected_with_key_named(
    name: str, value: str
) -> None:
    environment = candidate_environment()
    environment.update(
        {
            QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV: "120",
            QUALIFICATION_MODE_ENV: "step2-latency",
        }
    )
    environment[name] = value

    with pytest.raises(RuntimeError, match=rf"offending keys:.*{name}"):
        validate_qualification_watchdog_production_policy(environment)


def test_qualification_mode_is_required_and_override_values_are_positive() -> None:
    environment = candidate_environment()
    environment[QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV] = "120"
    with pytest.raises(RuntimeError, match=QUALIFICATION_MODE_ENV):
        validate_qualification_watchdog_production_policy(environment)

    for value in ("-1", "not-an-integer"):
        with pytest.raises(RuntimeError, match="non-negative integer"):
            validate_qualification_watchdog_production_policy(
                {QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV: value}
            )


@pytest.mark.parametrize(
    "label",
    [
        "x",
        "UPPERCASE",
        "has_underscore",
        "has.period",
        "-leading",
        "trailing-" * 9,
        " step2-latency",
        "step2-latency ",
        "step2-latency\t",
        "production",
    ],
)
def test_qualification_mode_label_contract_is_fail_closed(label: str) -> None:
    environment = candidate_environment()
    environment.update(
        {
            QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV: "120",
            QUALIFICATION_MODE_ENV: label,
        }
    )
    with pytest.raises(RuntimeError, match="matching|explicitly refused"):
        validate_qualification_watchdog_production_policy(environment)


def test_engine_boundary_rejects_all_copied_and_replaced_proofs() -> None:
    validated = validated_mode()
    for forged in (
        dataclasses.replace(validated, label="forged-mode"),
        dataclasses.replace(validated, candidate="production-candidate-2"),
        dataclasses.replace(validated, no_text_watchdog_override_frames=999999),
        dataclasses.replace(validated, environment_sha256="0" * 64),
        copy.copy(validated),
        copy.deepcopy(validated),
    ):
        with pytest.raises(TypeError, match="validated qualification identity"):
            VoiceChatEngine(SimpleNamespace(), qualification_mode=forged)

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(validated)


def test_engine_boundary_rejects_direct_and_subclass_forgeries() -> None:
    validated = validated_mode()
    direct = ValidatedQualificationMode(
        label=validated.label,
        candidate=validated.candidate,
        no_text_watchdog_override_frames=validated.no_text_watchdog_override_frames,
        environment_sha256=validated.environment_sha256,
    )
    with pytest.raises(TypeError, match="validated qualification identity"):
        VoiceChatEngine(SimpleNamespace(), qualification_mode=direct)

    class ForgedQualificationMode(ValidatedQualificationMode):
        def __getattribute__(self, name: str):
            if name == "label":
                return "production"
            if name == "candidate":
                return "forged-candidate"
            if name == "no_text_watchdog_override_frames":
                return 777777
            return super().__getattribute__(name)

    subclass = object.__new__(ForgedQualificationMode)
    for field in dataclasses.fields(ValidatedQualificationMode):
        object.__setattr__(
            subclass,
            field.name,
            object.__getattribute__(validated, field.name),
        )
    with pytest.raises(TypeError, match="validated qualification identity"):
        VoiceChatEngine(SimpleNamespace(), qualification_mode=subclass)


def test_engine_recovers_registry_values_after_reachable_result_rewrite() -> None:
    validated = validated_mode(frames=125)
    assert not hasattr(validated, "_proof")
    object.__setattr__(validated, "label", "production")
    object.__setattr__(validated, "candidate", "forged-candidate")
    object.__setattr__(validated, "no_text_watchdog_override_frames", 999999)
    object.__setattr__(validated, "environment_sha256", "0" * 64)
    assert is_validated_qualification_mode(validated)

    engine = VoiceChatEngine(SimpleNamespace(), qualification_mode=validated)

    assert engine.qualification_mode.label == "step2-latency"
    assert engine.qualification_mode.candidate == "production-candidate-1"
    assert engine.qualification_mode.no_text_watchdog_override_frames == 125
    assert engine.qualification_mode.environment_sha256 != "0" * 64
    assert not is_validated_qualification_mode(validated)


def test_registry_authority_is_invocation_bound_and_consumed_once() -> None:
    first = validated_mode(frames=125)
    second = validated_mode(frames=125)
    assert first is not second
    for field in dataclasses.fields(ValidatedQualificationMode):
        object.__setattr__(second, field.name, object.__getattribute__(first, field.name))

    first_engine = VoiceChatEngine(SimpleNamespace(), qualification_mode=first)
    second_engine = VoiceChatEngine(SimpleNamespace(), qualification_mode=second)
    assert first_engine.qualification_mode == second_engine.qualification_mode
    with pytest.raises(TypeError, match="validated qualification identity"):
        VoiceChatEngine(SimpleNamespace(), qualification_mode=first)


def test_engine_boundary_requires_validated_identity_and_selects_watchdog_variant() -> None:
    with pytest.raises(TypeError, match="validated qualification identity"):
        VoiceChatEngine(SimpleNamespace(), qualification_mode=125)  # type: ignore[arg-type]

    baseline = VoiceChatEngine(SimpleNamespace()).agent_silence_watchdog
    qualified = VoiceChatEngine(
        SimpleNamespace(), qualification_mode=validated_mode()
    ).agent_silence_watchdog
    assert type(baseline) is AgentSilenceEosWatchdog
    assert type(qualified) is QualificationAgentSilenceEosWatchdog
    instructions = tuple(dis.get_instructions(AgentSilenceEosWatchdog.snapshot))
    assert not any("qualification" in str(instruction.argval) for instruction in instructions)
    assert not any(instruction.opname.startswith("POP_JUMP") for instruction in instructions)


def test_override_changes_only_no_text_threshold_and_snapshot_identity() -> None:
    baseline = VoiceChatEngine(
        SimpleNamespace(),
        agent_silence_eos_dbfs=-87,
        agent_silence_eos_frames=19,
        agent_no_text_frames=30,
        agent_no_audio_frames=31,
    ).agent_silence_watchdog
    overridden = VoiceChatEngine(
        SimpleNamespace(),
        agent_silence_eos_dbfs=-87,
        agent_silence_eos_frames=19,
        agent_no_text_frames=30,
        agent_no_audio_frames=31,
        qualification_mode=validated_mode(frames=125),
    ).agent_silence_watchdog

    baseline_state = baseline.snapshot()
    overridden_state = overridden.snapshot()
    assert overridden_state.pop("qualification_no_text_watchdog_override_frames") == 125
    assert overridden_state.pop("qualification_mode") == "step2-latency"
    assert overridden_state.pop("qualification_candidate") == "production-candidate-1"
    assert overridden_state.pop("no_text_required_frames") == 125
    assert baseline_state.pop("no_text_required_frames") == 30
    assert overridden_state == baseline_state


def test_dynamic_watchdog_thresholds_resets_and_reason_precedence() -> None:
    watchdog = VoiceChatEngine(
        SimpleNamespace(),
        agent_silence_eos_frames=3,
        agent_no_text_frames=3,
        agent_no_audio_frames=100,
        qualification_mode=validated_mode(frames=5),
    ).agent_silence_watchdog
    assert not watchdog.observe("agent_bos", "", -120.0)
    for _ in range(3):
        assert not watchdog.observe("pad", "", -120.0)
    assert watchdog.observe("pad", "", -120.0)
    assert watchdog.request_reason == "no_text_since_bos_watchdog"

    watchdog.reset()
    watchdog.no_audio_required_frames = 4
    assert not watchdog.observe("agent_bos", "text", -120.0)
    assert not watchdog.observe("pad", "", -120.0)
    assert not watchdog.observe("pad", "", -120.0)
    assert watchdog.observe("pad", "", -120.0)
    assert watchdog.request_reason == "no_audio_since_bos_watchdog"

    watchdog.reset()
    assert not watchdog.observe("agent_bos", "text", -40.0)
    assert not watchdog.observe("pad", "", -120.0)
    assert not watchdog.observe("pad", "", -120.0)
    assert watchdog.observe("pad", "", -120.0)
    assert watchdog.request_reason == "decoded_silence_watchdog"

    watchdog.reset()
    assert not watchdog.observe("agent_bos", "", -120.0)
    assert not watchdog.observe("pad", "", -120.0, function_delta="{")
    assert watchdog.no_text_frames == 0
    assert watchdog.no_audio_frames == 0
    assert not watchdog.observe("pad", "text", -120.0)
    assert watchdog.no_text_frames == 0
    assert watchdog.no_audio_frames == 1

    coincident = QualificationAgentSilenceEosWatchdog(
        required_frames=2,
        no_text_required_frames=2,
        no_audio_required_frames=2,
        qualification_mode=validated_mode(frames=2),
    )
    assert not coincident.observe("agent_bos", "", -120.0)
    assert coincident.observe("pad", "", -120.0)
    assert coincident.request_reason == "no_text_since_bos_watchdog"


def test_health_exposes_retained_qualification_identity() -> None:
    pipeline = SimpleNamespace(checkpoint_provenance={"kind": "test"})
    engine = VoiceChatEngine(pipeline, qualification_mode=validated_mode(frames=120))
    health_endpoint = next(
        route.endpoint for route in create_app(engine).routes if route.path == "/health"
    )

    health = asyncio.run(health_endpoint())
    identity = health["qualification_no_text_watchdog_override"]
    assert identity["environment"] == QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV
    assert identity["mode_environment"] == QUALIFICATION_MODE_ENV
    assert identity["mode"] == "step2-latency"
    assert identity["candidate"] == "production-candidate-1"
    assert identity["frames"] == 120
    assert len(identity["environment_sha256"]) == 64


def test_health_and_snapshot_omit_inactive_qualification_fields() -> None:
    pipeline = SimpleNamespace(checkpoint_provenance={"kind": "test"})
    engine = VoiceChatEngine(pipeline)
    health_endpoint = next(
        route.endpoint for route in create_app(engine).routes if route.path == "/health"
    )

    assert "qualification_no_text_watchdog_override" not in asyncio.run(health_endpoint())
    assert all("qualification" not in key for key in engine.agent_silence_watchdog.snapshot())
