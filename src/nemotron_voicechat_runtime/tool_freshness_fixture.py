"""Validated Pocket fixtures for typed-versus-speech tool freshness parity."""

from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Any

from .pocket_worker import EXPECTED_ASSETS

FIXTURE_SCHEMA_V1 = 1
FIXTURE_KIND_V1 = "tool_freshness_pocket_fixture"
FIXTURE_SCHEMA = 2
FIXTURE_KIND = "tool_freshness_ordinary_english_pocket_fixture_v2"
FIXTURE_MANIFEST_NAME = "manifest.json"
FIXTURE_SAMPLE_RATE = 16_000
FIXTURE_MAX_SECONDS = 30
FIXTURE_MAX_SAMPLES = FIXTURE_SAMPLE_RATE * FIXTURE_MAX_SECONDS
FIXTURE_LANGUAGE = "english_2026-04"
FIXTURE_VOICE = "alba"
FIXTURE_BASE_SEED = 0
FIXTURE_SEED_SCHEME = "sha256-text-plus-base-v1"
TOOL_ONLY_PROMPTS_V1 = (
    "Call the available UTC time tool now and state its exact time and fresh verification code.",
    "Call the UTC time tool again now and state its new time and verification code.",
    "Invoke the UTC time tool for this turn, even though it was used earlier, "
    "and state the new verification code it returns.",
    "Make one new call to the UTC time tool now and answer with its fresh verification code.",
)
TOOL_VERIFICATION_CODES_V1 = ("cobalt", "topaz", "amber", "violet")
TOOL_RESULT_TIMES_V1 = ("15:01 UTC", "15:02 UTC", "15:03 UTC", "15:04 UTC")
TOOL_ONLY_PROMPTS = (
    "Call the available verification tool now and state the fresh word it returns.",
    "Call the verification tool again and state the new word it returns.",
    "Use the verification tool for this turn, even though it was used earlier, "
    "and state the new word.",
    "Make one new call to the verification tool now and answer with its fresh word.",
)
TOOL_VERIFICATION_WORDS = ("river", "candle", "garden", "winter")
TOOL_RESULT_ORDINALS = ("first", "second", "third", "fourth")
SCENARIO_SHA256_V1 = "77efc92480f0754eabcd6f4c8173337b2dfbdc918fe291a67f44375de68b80cf"
# Updated only when the preregistered canonical v2 scenario intentionally changes.
SCENARIO_SHA256_V2 = "2e95857362d4e74c3bb7cd4b2d8edcdfecfc87fdf2bf90a913b078132d700f28"
FIXTURE_PROJECT_SOURCE_PATHS = (
    "src/nemotron_voicechat_runtime/__init__.py",
    "src/nemotron_voicechat_runtime/pocket_controller.py",
    "src/nemotron_voicechat_runtime/pocket_ipc.py",
    "src/nemotron_voicechat_runtime/pocket_worker.py",
    "src/nemotron_voicechat_runtime/protocol.py",
    "src/nemotron_voicechat_runtime/qualification_audio.py",
    "src/nemotron_voicechat_runtime/tool_freshness_contract.py",
    "src/nemotron_voicechat_runtime/tool_freshness_fixture.py",
    "tools/qualification/generate_tool_freshness_fixtures.py",
    "tools/qualification/sustained_strict_v3.py",
)
FIXTURE_INSTALLED_WORKER_SOURCE = "pocket_environment/pocket_worker.py"
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def self_hash(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("sha256", None)
    return canonical_sha256(unsigned)


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


def _validate_sources(value: Any) -> dict[str, str]:
    expected = {*FIXTURE_PROJECT_SOURCE_PATHS, FIXTURE_INSTALLED_WORKER_SOURCE}
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("tool freshness fixture source inventory is incomplete")
    for name, digest in value.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("tool freshness fixture source path is invalid")
        _validate_sha256(digest, f"tool freshness fixture source {name}")
    return value


def fixture_contract_for_scenario_sha256(
    scenario_sha256: str,
) -> tuple[int, str, tuple[str, ...]]:
    """Resolve the only two admitted fixture contracts by exact scenario identity."""

    if scenario_sha256 == SCENARIO_SHA256_V1:
        return FIXTURE_SCHEMA_V1, FIXTURE_KIND_V1, TOOL_ONLY_PROMPTS_V1
    if scenario_sha256 == SCENARIO_SHA256_V2:
        return FIXTURE_SCHEMA, FIXTURE_KIND, TOOL_ONLY_PROMPTS
    raise RuntimeError("unknown tool freshness scenario hash")


def validate_fixture_source_provenance(
    manifest: dict[str, Any],
    *,
    project_root: Path,
    installed_worker: Path | None = None,
) -> None:
    provenance = manifest.get("provenance")
    sources = provenance.get("source_sha256") if isinstance(provenance, dict) else None
    sources = _validate_sources(sources)
    for name in FIXTURE_PROJECT_SOURCE_PATHS:
        path = project_root / name
        if not path.is_file() or sha256_path(path) != sources[name]:
            raise RuntimeError(f"tool freshness fixture source hash mismatch: {name}")
    if installed_worker is not None and (
        not installed_worker.is_file()
        or sha256_path(installed_worker) != sources[FIXTURE_INSTALLED_WORKER_SOURCE]
    ):
        raise RuntimeError("tool freshness fixture installed worker hash mismatch")


def validate_tool_freshness_manifest_document(
    manifest: Any,
    *,
    expected_runtime_image_id: str,
    expected_scenario_sha256: str,
) -> dict[str, Any]:
    """Validate retained fixture metadata without requiring carrier files."""

    manifest = _require_exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "scenario_sha256",
            "fixtures",
            "synthesis",
            "provenance",
            "sha256",
        },
        "tool freshness fixture manifest",
    )
    expected_schema, expected_kind, expected_prompts = fixture_contract_for_scenario_sha256(
        expected_scenario_sha256
    )
    if manifest["schema"] != expected_schema or manifest["kind"] != expected_kind:
        raise RuntimeError("tool freshness fixture identity is invalid")
    if manifest["sha256"] != self_hash(manifest):
        raise RuntimeError("tool freshness fixture self-hash mismatch")
    if _validate_sha256(expected_scenario_sha256, "expected scenario") != _validate_sha256(
        manifest["scenario_sha256"], "fixture scenario"
    ):
        raise RuntimeError("tool freshness fixture scenario hash mismatch")
    fixtures = manifest["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != len(expected_prompts):
        raise RuntimeError("tool freshness fixture cardinality is invalid")
    for ordinal, (fixture, prompt) in enumerate(zip(fixtures, expected_prompts, strict=True), 1):
        fixture = _require_exact_keys(
            fixture,
            {"case_id", "ordinal", "text", "pcm", "wav"},
            f"tool freshness fixture {ordinal}",
        )
        case_id = f"tool-freshness-{ordinal}"
        if (
            fixture["case_id"] != case_id
            or fixture["ordinal"] != ordinal
            or fixture["text"] != prompt
        ):
            raise RuntimeError("tool freshness fixture prompt identity mismatch")
        pcm = _require_exact_keys(
            fixture["pcm"],
            {"path", "sample_rate", "sample_count", "bytes", "sha256"},
            f"tool freshness fixture {ordinal} PCM",
        )
        if (
            pcm["path"] != f"{case_id}.pcm"
            or pcm["sample_rate"] != FIXTURE_SAMPLE_RATE
            or not isinstance(pcm["sample_count"], int)
            or isinstance(pcm["sample_count"], bool)
            or not 0 < pcm["sample_count"] <= FIXTURE_MAX_SAMPLES
            or pcm["bytes"] != pcm["sample_count"] * 2
        ):
            raise RuntimeError("tool freshness fixture PCM metadata is invalid")
        _validate_sha256(pcm["sha256"], "fixture PCM")
        wav = _require_exact_keys(
            fixture["wav"],
            {
                "path",
                "sample_rate",
                "channels",
                "sample_width",
                "samples",
                "bytes",
                "sha256",
            },
            f"tool freshness fixture {ordinal} WAV",
        )
        if (
            wav["path"] != f"{case_id}.wav"
            or wav["sample_rate"] != FIXTURE_SAMPLE_RATE
            or wav["channels"] != 1
            or wav["sample_width"] != 2
            or wav["samples"] != pcm["sample_count"]
            or not isinstance(wav["bytes"], int)
            or isinstance(wav["bytes"], bool)
            or wav["bytes"] <= 44
        ):
            raise RuntimeError("tool freshness fixture WAV metadata is invalid")
        _validate_sha256(wav["sha256"], "fixture WAV")

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
        "tool freshness fixture synthesis",
    )
    expected_assets = {
        name: {"bytes": details[0], "sha256": details[1]}
        for name, details in EXPECTED_ASSETS.items()
    }
    if (
        synthesis["language"] != FIXTURE_LANGUAGE
        or synthesis["voice"] != FIXTURE_VOICE
        or synthesis["quantize"] is not True
        or synthesis["base_seed"] != FIXTURE_BASE_SEED
        or synthesis["seed_scheme"] != FIXTURE_SEED_SCHEME
        or not isinstance(synthesis["threads"], int)
        or isinstance(synthesis["threads"], bool)
        or synthesis["threads"] < 1
        or not isinstance(synthesis["cpus"], list)
        or not synthesis["cpus"]
        or any(type(cpu) is not int or cpu < 0 for cpu in synthesis["cpus"])
        or not isinstance(synthesis["package_version"], str)
        or not synthesis["package_version"]
        or synthesis["assets"] != expected_assets
    ):
        raise RuntimeError("tool freshness fixture synthesis contract mismatch")
    provenance = _require_exact_keys(
        manifest["provenance"],
        {"runtime_image", "runtime_image_id", "source_sha256"},
        "tool freshness fixture provenance",
    )
    if (
        not IMAGE_ID_PATTERN.fullmatch(str(expected_runtime_image_id))
        or provenance["runtime_image_id"] != expected_runtime_image_id
        or not isinstance(provenance["runtime_image"], str)
        or not provenance["runtime_image"]
    ):
        raise RuntimeError("tool freshness fixture runtime image provenance mismatch")
    _validate_sources(provenance["source_sha256"])
    return manifest


def validate_tool_freshness_fixtures(
    fixture_dir: Path,
    *,
    expected_runtime_image_id: str,
    expected_scenario_sha256: str,
) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """Validate and return all four immutable Pocket PCM carriers."""

    manifest_path = fixture_dir / FIXTURE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("tool freshness fixture manifest is unreadable") from exc
    validate_tool_freshness_manifest_document(
        manifest,
        expected_runtime_image_id=expected_runtime_image_id,
        expected_scenario_sha256=expected_scenario_sha256,
    )
    _require_exact_keys(
        manifest,
        {
            "schema",
            "kind",
            "scenario_sha256",
            "fixtures",
            "synthesis",
            "provenance",
            "sha256",
        },
        "tool freshness fixture manifest",
    )
    expected_schema, expected_kind, expected_prompts = fixture_contract_for_scenario_sha256(
        expected_scenario_sha256
    )
    if manifest["schema"] != expected_schema or manifest["kind"] != expected_kind:
        raise RuntimeError("tool freshness fixture identity is invalid")
    if manifest["sha256"] != self_hash(manifest):
        raise RuntimeError("tool freshness fixture self-hash mismatch")
    if _validate_sha256(expected_scenario_sha256, "expected scenario") != _validate_sha256(
        manifest["scenario_sha256"], "fixture scenario"
    ):
        raise RuntimeError("tool freshness fixture scenario hash mismatch")
    fixtures = manifest["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != len(expected_prompts):
        raise RuntimeError("tool freshness fixture cardinality is invalid")

    fixture_pcm: list[bytes] = []
    for ordinal, (fixture, prompt) in enumerate(zip(fixtures, expected_prompts, strict=True), 1):
        _require_exact_keys(
            fixture,
            {"case_id", "ordinal", "text", "pcm", "wav"},
            f"tool freshness fixture {ordinal}",
        )
        case_id = f"tool-freshness-{ordinal}"
        if (
            fixture["case_id"] != case_id
            or fixture["ordinal"] != ordinal
            or fixture["text"] != prompt
        ):
            raise RuntimeError("tool freshness fixture prompt identity mismatch")
        pcm = _require_exact_keys(
            fixture["pcm"],
            {"path", "sample_rate", "sample_count", "bytes", "sha256"},
            f"tool freshness fixture {ordinal} PCM",
        )
        if pcm["path"] != f"{case_id}.pcm" or pcm["sample_rate"] != FIXTURE_SAMPLE_RATE:
            raise RuntimeError("tool freshness fixture PCM format mismatch")
        if (
            not isinstance(pcm["sample_count"], int)
            or isinstance(pcm["sample_count"], bool)
            or not 0 < pcm["sample_count"] <= FIXTURE_MAX_SAMPLES
            or pcm["bytes"] != pcm["sample_count"] * 2
        ):
            raise RuntimeError("tool freshness fixture PCM length is invalid")
        pcm_path = fixture_dir / pcm["path"]
        try:
            payload = pcm_path.read_bytes()
        except OSError as exc:
            raise RuntimeError("tool freshness fixture PCM is unreadable") from exc
        if len(payload) != pcm["bytes"] or sha256_bytes(payload) != _validate_sha256(
            pcm["sha256"], "fixture PCM"
        ):
            raise RuntimeError("tool freshness fixture PCM hash mismatch")
        wav = _require_exact_keys(
            fixture["wav"],
            {"path", "sample_rate", "channels", "sample_width", "samples", "bytes", "sha256"},
            f"tool freshness fixture {ordinal} WAV",
        )
        wav_path = fixture_dir / wav["path"]
        try:
            wav_bytes = wav_path.read_bytes()
            with wave.open(str(wav_path), "rb") as stream:
                wav_format = (
                    stream.getframerate(),
                    stream.getnchannels(),
                    stream.getsampwidth(),
                    stream.getnframes(),
                )
                wav_pcm = stream.readframes(stream.getnframes())
        except (OSError, EOFError, wave.Error) as exc:
            raise RuntimeError("tool freshness fixture WAV is unreadable") from exc
        if (
            wav["path"] != f"{case_id}.wav"
            or wav_format
            != (
                wav["sample_rate"],
                wav["channels"],
                wav["sample_width"],
                wav["samples"],
            )
            or wav_format != (FIXTURE_SAMPLE_RATE, 1, 2, pcm["sample_count"])
            or wav["bytes"] != len(wav_bytes)
            or sha256_bytes(wav_bytes) != _validate_sha256(wav["sha256"], "fixture WAV")
            or wav_pcm != payload
        ):
            raise RuntimeError("tool freshness fixture WAV does not match its PCM")
        fixture_pcm.append(payload)

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
        "tool freshness fixture synthesis",
    )
    expected_assets = {
        name: {"bytes": details[0], "sha256": details[1]}
        for name, details in EXPECTED_ASSETS.items()
    }
    if (
        synthesis["language"] != FIXTURE_LANGUAGE
        or synthesis["voice"] != FIXTURE_VOICE
        or synthesis["quantize"] is not True
        or synthesis["base_seed"] != FIXTURE_BASE_SEED
        or synthesis["seed_scheme"] != FIXTURE_SEED_SCHEME
        or not isinstance(synthesis["threads"], int)
        or isinstance(synthesis["threads"], bool)
        or synthesis["threads"] < 1
        or not isinstance(synthesis["cpus"], list)
        or not synthesis["cpus"]
        or any(type(cpu) is not int or cpu < 0 for cpu in synthesis["cpus"])
        or not isinstance(synthesis["package_version"], str)
        or not synthesis["package_version"]
        or synthesis["assets"] != expected_assets
    ):
        raise RuntimeError("tool freshness fixture synthesis contract mismatch")

    provenance = _require_exact_keys(
        manifest["provenance"],
        {"runtime_image", "runtime_image_id", "source_sha256"},
        "tool freshness fixture provenance",
    )
    if (
        not IMAGE_ID_PATTERN.fullmatch(str(expected_runtime_image_id))
        or provenance["runtime_image_id"] != expected_runtime_image_id
        or not isinstance(provenance["runtime_image"], str)
        or not provenance["runtime_image"]
    ):
        raise RuntimeError("tool freshness fixture runtime image provenance mismatch")
    _validate_sources(provenance["source_sha256"])
    return manifest, tuple(fixture_pcm)
