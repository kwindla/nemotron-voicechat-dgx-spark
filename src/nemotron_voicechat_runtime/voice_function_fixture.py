"""Frozen Pocket carrier fixture for the voice function-logit control."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any

from .pocket_worker import EXPECTED_ASSETS
from .semantic_corpus import corpus_sha256, load_semantic_corpus

FIXTURE_SCHEMA = 1
FIXTURE_KIND = "voice_function_reference_fixture"
FIXTURE_CASE_ID = "c014"
FIXTURE_PCM_NAME = "c014.pcm"
FIXTURE_WAV_NAME = "c014.wav"
FIXTURE_MANIFEST_NAME = "manifest.json"
FIXTURE_SAMPLE_RATE = 16_000
FIXTURE_MAX_SECONDS = 30
FIXTURE_MAX_SAMPLES = FIXTURE_SAMPLE_RATE * FIXTURE_MAX_SECONDS
FIXTURE_PACKET_SAMPLES = FIXTURE_SAMPLE_RATE // 50
FIXTURE_LANGUAGE = "english_2026-04"
FIXTURE_VOICE = "alba"
FIXTURE_BASE_SEED = 0
FIXTURE_SEED_SCHEME = "sha256-text-plus-base-v1"
FIXTURE_PROJECT_SOURCE_PATHS = (
    "src/nemotron_voicechat_runtime/pocket_controller.py",
    "src/nemotron_voicechat_runtime/pocket_worker.py",
    "src/nemotron_voicechat_runtime/voice_function_fixture.py",
    "tools/qualification/generate_voice_function_fixture.py",
)
FIXTURE_INSTALLED_WORKER_SOURCE = "pocket_environment/pocket_worker.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fixture_case(corpus_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = load_semantic_corpus(corpus_path)
    matches = [case for case in corpus["cases"] if case["id"] == FIXTURE_CASE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"semantic corpus must contain exactly one {FIXTURE_CASE_ID} case")
    return corpus, matches[0]


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"{label} has an invalid schema")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} has an invalid SHA256")
    return value


def _validate_source_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError("fixture source hashes are missing")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise RuntimeError("fixture source hash name is invalid")
        result[name] = _validate_sha256(digest, f"fixture source {name}")
    return result


def validate_fixture_source_provenance(
    manifest: dict[str, Any],
    *,
    project_root: Path,
    installed_worker: Path | None = None,
) -> None:
    """Rehash every source visible to this host or diagnostic container."""

    provenance = manifest.get("provenance")
    sources = provenance.get("source_sha256") if isinstance(provenance, dict) else None
    expected_keys = {*FIXTURE_PROJECT_SOURCE_PATHS, FIXTURE_INSTALLED_WORKER_SOURCE}
    if not isinstance(sources, dict) or set(sources) != expected_keys:
        raise RuntimeError("voice function fixture source inventory is incomplete")
    for name in FIXTURE_PROJECT_SOURCE_PATHS:
        path = project_root / name
        if not path.is_file() or sha256_path(path) != sources[name]:
            raise RuntimeError(f"voice function fixture source hash mismatch: {name}")
    if installed_worker is not None:
        if (
            not installed_worker.is_file()
            or sha256_path(installed_worker) != sources[FIXTURE_INSTALLED_WORKER_SOURCE]
        ):
            raise RuntimeError("voice function fixture installed worker hash mismatch")


def validate_voice_function_fixture(
    fixture_dir: Path,
    *,
    corpus_path: Path,
    expected_runtime_image_id: str,
) -> tuple[dict[str, Any], bytes]:
    """Validate and return one immutable c014 PCM carrier."""

    manifest_path = fixture_dir / FIXTURE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("voice function fixture manifest is unreadable") from exc
    _require_exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "case_id",
            "corpus_sha256",
            "prompt",
            "pcm",
            "wav",
            "synthesis",
            "provenance",
        },
        "voice function fixture manifest",
    )
    if (
        manifest["schema"] != FIXTURE_SCHEMA
        or manifest["kind"] != FIXTURE_KIND
        or manifest["case_id"] != FIXTURE_CASE_ID
    ):
        raise RuntimeError("voice function fixture has an invalid identity")

    _, case = fixture_case(corpus_path)
    if manifest["corpus_sha256"] != corpus_sha256(corpus_path):
        raise RuntimeError("voice function fixture corpus hash mismatch")
    if manifest["prompt"] != case["prompt"]:
        raise RuntimeError("voice function fixture prompt mismatch")

    pcm = _require_exact_keys(
        manifest["pcm"],
        {"path", "sample_rate", "sample_count", "bytes", "sha256"},
        "voice function fixture PCM",
    )
    if pcm["path"] != FIXTURE_PCM_NAME or pcm["sample_rate"] != FIXTURE_SAMPLE_RATE:
        raise RuntimeError("voice function fixture PCM format mismatch")
    if (
        not isinstance(pcm["sample_count"], int)
        or isinstance(pcm["sample_count"], bool)
        or not 0 < pcm["sample_count"] <= FIXTURE_MAX_SAMPLES
        or pcm["bytes"] != pcm["sample_count"] * 2
    ):
        raise RuntimeError("voice function fixture PCM length is invalid")
    pcm_sha256 = _validate_sha256(pcm["sha256"], "voice function fixture PCM")
    pcm_path = fixture_dir / FIXTURE_PCM_NAME
    try:
        pcm_bytes = pcm_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("voice function fixture PCM is unreadable") from exc
    if len(pcm_bytes) != pcm["bytes"] or sha256_bytes(pcm_bytes) != pcm_sha256:
        raise RuntimeError("voice function fixture PCM bytes do not match the manifest")

    wav = _require_exact_keys(
        manifest["wav"], {"path", "bytes", "sha256"}, "voice function fixture WAV"
    )
    if wav["path"] != FIXTURE_WAV_NAME or not isinstance(wav["bytes"], int):
        raise RuntimeError("voice function fixture WAV metadata is invalid")
    wav_sha256 = _validate_sha256(wav["sha256"], "voice function fixture WAV")
    wav_path = fixture_dir / FIXTURE_WAV_NAME
    try:
        wav_bytes = wav_path.read_bytes()
        with wave.open(str(wav_path), "rb") as stream:
            wav_format = (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getnframes(),
            )
            wav_pcm = stream.readframes(stream.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError("voice function fixture WAV is unreadable") from exc
    if (
        len(wav_bytes) != wav["bytes"]
        or sha256_bytes(wav_bytes) != wav_sha256
        or wav_format != (1, 2, FIXTURE_SAMPLE_RATE, pcm["sample_count"])
        or wav_pcm != pcm_bytes
    ):
        raise RuntimeError("voice function fixture WAV does not match the PCM")

    synthesis = _require_exact_keys(
        manifest["synthesis"],
        {
            "language",
            "voice",
            "quantize",
            "threads",
            "cpus",
            "base_seed",
            "seed_scheme",
            "package_version",
            "assets",
        },
        "voice function fixture synthesis",
    )
    if (
        synthesis["language"] != FIXTURE_LANGUAGE
        or synthesis["voice"] != FIXTURE_VOICE
        or synthesis["quantize"] is not True
        or synthesis["base_seed"] != FIXTURE_BASE_SEED
        or synthesis["seed_scheme"] != FIXTURE_SEED_SCHEME
        or not isinstance(synthesis["threads"], int)
        or synthesis["threads"] < 1
        or not isinstance(synthesis["cpus"], list)
        or not synthesis["cpus"]
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in synthesis["cpus"]
        )
        or not isinstance(synthesis["package_version"], str)
        or not synthesis["package_version"]
    ):
        raise RuntimeError("voice function fixture synthesis contract mismatch")
    assets = synthesis["assets"]
    expected_assets = {
        name: {"bytes": details[0], "sha256": details[1]}
        for name, details in EXPECTED_ASSETS.items()
    }
    if assets != expected_assets:
        raise RuntimeError("voice function fixture asset provenance is incomplete")
    for name, details in assets.items():
        _require_exact_keys(details, {"bytes", "sha256"}, f"voice function asset {name}")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(details["bytes"], int)
            or details["bytes"] < 1
        ):
            raise RuntimeError("voice function fixture asset provenance is invalid")
        _validate_sha256(details["sha256"], f"voice function asset {name}")

    provenance = _require_exact_keys(
        manifest["provenance"],
        {"runtime_image", "runtime_image_id", "source_sha256"},
        "voice function fixture provenance",
    )
    if provenance["runtime_image_id"] != expected_runtime_image_id:
        raise RuntimeError("voice function fixture runtime image ID mismatch")
    if not isinstance(provenance["runtime_image"], str) or not provenance["runtime_image"]:
        raise RuntimeError("voice function fixture runtime image name is invalid")
    _validate_source_hashes(provenance["source_sha256"])
    return manifest, pcm_bytes
