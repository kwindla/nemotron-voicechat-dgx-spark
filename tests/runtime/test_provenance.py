from __future__ import annotations

from pathlib import Path

import pytest

from nemotron_voicechat_runtime.artifacts import default_layout, load_config
from nemotron_voicechat_runtime.cli import model_container_command
from nemotron_voicechat_runtime.provenance import (
    NANO_PAD_PAIR_PRODUCTION_POLICY,
    PRODUCTION_ENVIRONMENT,
    RUNTIME_SOURCE_PATHS,
    validate_nano_pad_pair_production_policy,
)


def test_deployment_preserves_canonical_nano_skeleton_name():
    config = load_config()
    command = model_container_command(config, default_layout(), 8786)
    target = "/models/NVIDIA-Nemotron-Nano-9B-v2"
    assert any(value.endswith(f":{target}:ro") for value in command)
    assert command[command.index("--hf-skeleton") + 1] == target


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_source_allowlist_is_exact_and_present():
    assert tuple(sorted(RUNTIME_SOURCE_PATHS)) == RUNTIME_SOURCE_PATHS
    assert len(RUNTIME_SOURCE_PATHS) == len(set(RUNTIME_SOURCE_PATHS))
    assert all((REPO_ROOT / name).is_file() for name in RUNTIME_SOURCE_PATHS)
    assert not any("client" in name for name in RUNTIME_SOURCE_PATHS)


def test_checked_in_toml_preserves_production_contract():
    values = load_config()["runtime"]["environment"]
    assert {name: values.get(name) for name in PRODUCTION_ENVIRONMENT} == PRODUCTION_ENVIRONMENT


def test_production_retires_nano_pad_pair_speculation() -> None:
    assert NANO_PAD_PAIR_PRODUCTION_POLICY == {
        "VOICECHAT_NANO_PAD_PAIR": "0",
        "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "0",
        "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "0",
        "VOICECHAT_NANO_PAIR_FULL_GRAPH": "0",
    }
    validate_nano_pad_pair_production_policy(dict(NANO_PAD_PAIR_PRODUCTION_POLICY))

    for name in NANO_PAD_PAIR_PRODUCTION_POLICY:
        contradictory = dict(NANO_PAD_PAIR_PRODUCTION_POLICY)
        contradictory[name] = "1"
        with pytest.raises(RuntimeError, match="retired contract"):
            validate_nano_pad_pair_production_policy(contradictory)


def test_model_command_passes_retired_nano_pair_policy() -> None:
    config = load_config()
    command = model_container_command(config, default_layout(), 8786)
    for name, value in NANO_PAD_PAIR_PRODUCTION_POLICY.items():
        if name == "VOICECHAT_NANO_PAIR_FULL_GRAPH":
            assert f"{name}=1" not in command
            continue
        assert f"{name}={value}" in command

    candidate_two = load_config(REPO_ROOT / "config/production-candidate-2.toml")
    candidate_two_command = model_container_command(candidate_two, default_layout(), 8786)
    for name, value in NANO_PAD_PAIR_PRODUCTION_POLICY.items():
        assert f"{name}={value}" in candidate_two_command
