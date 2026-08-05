"""Checked-in configuration and content-addressed cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import PRODUCTION_ENVIRONMENT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/production-candidate-1.toml"


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("schema") != 1 or config.get("candidate") != "production-candidate-1":
        raise ValueError(f"unsupported Voicechat configuration: {path}")
    configured = config.get("runtime", {}).get("environment", {})
    if configured != PRODUCTION_ENVIRONMENT | {
        "VOICECHAT_TYPED_INPUT_LANGUAGE": "english_2026-04",
        "VOICECHAT_TYPED_INPUT_VOICE": "alba",
        "VOICECHAT_TYPED_INPUT_MAX_TEXT_CHARS": "1000",
        "VOICECHAT_TYPED_INPUT_FRAME_MS": "20",
        "VOICECHAT_TYPED_INPUT_EOU_MARGIN": "1.25",
        "VOICECHAT_TYPED_INPUT_CPU_CORES": "7,8,9,15",
        "VOICECHAT_TYPED_INPUT_TORCH_THREADS": "4",
        "VOICECHAT_TYPED_INPUT_QUANTIZE": "1",
        "VOICECHAT_TYPED_INPUT_PREWARM": "1",
    }:
        raise ValueError("TOML runtime environment differs from the frozen candidate")
    affinity = config.get("runtime", {}).get("cpu_affinity", {})
    codec = set(affinity.get("codec", []))
    pocket = set(affinity.get("pocket_tts", []))
    if not codec or not pocket or codec & pocket:
        raise ValueError("codec and Pocket TTS CPU affinities must be non-empty and disjoint")
    if configured["EA_CPU_CODEC_CORES"] != ",".join(map(str, affinity["codec"])):
        raise ValueError("codec CPU affinity differs between TOML sections")
    if configured["VOICECHAT_TYPED_INPUT_CPU_CORES"] != ",".join(map(str, affinity["pocket_tts"])):
        raise ValueError("Pocket TTS CPU affinity differs between TOML sections")
    return config


@dataclass(frozen=True)
class Layout:
    cache: Path
    traces: Path

    @property
    def artifacts(self) -> Path:
        return self.cache / "artifacts"

    @property
    def release(self) -> Path:
        return self.artifacts / "release"

    @property
    def parent(self) -> Path:
        return self.artifacts / "parent"

    @property
    def nano_skeleton(self) -> Path:
        return self.artifacts / "nano-skeleton"

    @property
    def hf_home(self) -> Path:
        return self.cache / "huggingface"

    @property
    def state_file(self) -> Path:
        return self.cache / "bootstrap-state.json"

    @property
    def runtime_state_file(self) -> Path:
        return self.cache / "runtime-state.json"


def default_layout(cache_root: str | None = None, trace_root: str | None = None) -> Layout:
    cache = (
        Path(
            cache_root
            or os.environ.get("VOICECHAT_CACHE_ROOT", "")
            or Path.home() / ".cache/nemotron-voicechat"
        )
        .expanduser()
        .resolve()
    )
    traces = (
        Path(
            trace_root
            or os.environ.get("VOICECHAT_TRACE_ROOT", "")
            or Path.home() / ".local/state/nemotron-voicechat/traces"
        )
        .expanduser()
        .resolve()
    )
    return Layout(cache=cache, traces=traces)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def require_free_space(path: Path, gib: int, operation: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(path).free
    minimum = gib * 1024**3
    if available < minimum:
        raise RuntimeError(
            f"{operation} requires at least {gib} GiB free beneath {path}; "
            f"only {available / 1024**3:.1f} GiB is available"
        )


def host_facts() -> dict[str, str]:
    return {
        "architecture": platform.machine(),
        "kernel_version": platform.release(),
        "python": platform.python_version(),
    }
