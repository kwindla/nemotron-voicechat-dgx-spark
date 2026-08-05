from __future__ import annotations

from pathlib import Path

from nemotron_voicechat_runtime.artifacts import default_layout, load_config
from nemotron_voicechat_runtime.cli import model_container_command
from nemotron_voicechat_runtime.provenance import (
    PRODUCTION_ENVIRONMENT,
    RUNTIME_SOURCE_PATHS,
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
