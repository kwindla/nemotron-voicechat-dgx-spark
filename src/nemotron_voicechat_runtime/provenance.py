"""Exact source and environment contract for Production Candidate 1."""

from __future__ import annotations

import re
import threading
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

RUNTIME_PROVENANCE_REQUIRED_SOURCES = frozenset(
    {
        "src/nemotron_voicechat_runtime/pocket_controller.py",
        "src/nemotron_voicechat_runtime/pocket_worker.py",
        "src/nemotron_voicechat_runtime/direct_semantic_probe.py",
        "src/nemotron_voicechat_runtime/protocol.py",
        "src/nemotron_voicechat_runtime/provenance.py",
        "src/nemotron_voicechat_runtime/runtime_optimizations.py",
        "src/nemotron_voicechat_runtime/semantic_corpus.py",
        "src/nemotron_voicechat_runtime/server.py",
        (
            "src/nemotron_voicechat_runtime/patches/"
            "nemotron-voicechat-rnnt-turn-taking.patch"
        ),
    }
)


def valid_runtime_image_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def valid_runtime_provenance(value: object) -> bool:
    """Require immutable image identity and every direct-path source hash."""
    if not isinstance(value, dict) or not valid_runtime_image_id(value.get("runtime_image_id")):
        return False
    sources = value.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not RUNTIME_PROVENANCE_REQUIRED_SOURCES.issubset(sources)
    ):
        return False
    return all(
        isinstance(sources[name], str)
        and re.fullmatch(r"[0-9a-f]{64}", sources[name]) is not None
        for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES
    )

RUNTIME_SOURCE_PATHS = (
    "container/Dockerfile.runtime-overlay",
    "src/nemotron_voicechat_runtime/audio_compat.py",
    "src/nemotron_voicechat_runtime/cpu_codec_offload.py",
    "src/nemotron_voicechat_runtime/cpu_codec_worker.py",
    "src/nemotron_voicechat_runtime/ea_bf16_head.py",
    "src/nemotron_voicechat_runtime/ea_w8a32.py",
    "src/nemotron_voicechat_runtime/patch_ea_w8a32.py",
    "src/nemotron_voicechat_runtime/patch_public_pad_pair_scheduler.py",
    "src/nemotron_voicechat_runtime/patch_public_skip_custom_text_logits.py",
    "src/nemotron_voicechat_runtime/patch_voicechat_marlin_g128_padding.py",
    "src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch",
    "src/nemotron_voicechat_runtime/pocket_controller.py",
    "src/nemotron_voicechat_runtime/pocket_worker.py",
    "src/nemotron_voicechat_runtime/protocol.py",
    "src/nemotron_voicechat_runtime/provenance.py",
    "src/nemotron_voicechat_runtime/runtime_optimizations.py",
    "src/nemotron_voicechat_runtime/server.py",
)

NANO_PAD_PAIR_PRODUCTION_POLICY = {
    "VOICECHAT_NANO_PAD_PAIR": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "0",
    "VOICECHAT_NANO_PAIR_FULL_GRAPH": "0",
}

QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV = (
    "VOICECHAT_QUAL_NO_TEXT_WATCHDOG_OVERRIDE_FRAMES"
)
QUALIFICATION_MODE_ENV = "VOICECHAT_QUALIFICATION_MODE"
QUALIFICATION_ENVIRONMENT_DELTAS = frozenset(
    {
        "VOICECHAT_WEB_SYSTEM_PROMPT",
        QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV,
        QUALIFICATION_MODE_ENV,
    }
)
_QUALIFICATION_MANAGED_PREFIXES = ("EA_", "S2S_", "VOICECHAT_")
_QUALIFICATION_MANAGED_NAMES = frozenset({"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"})


@dataclass(frozen=True, slots=True, eq=False)
class ValidatedQualificationMode:
    """Caller-visible identity for one validator-owned qualification authority."""

    label: str
    candidate: str
    no_text_watchdog_override_frames: int
    environment_sha256: str

    def __copy__(self) -> ValidatedQualificationMode:
        return type(self)(
            self.label,
            self.candidate,
            self.no_text_watchdog_override_frames,
            self.environment_sha256,
        )

    def __deepcopy__(self, _memo: dict[int, object]) -> ValidatedQualificationMode:
        return self.__copy__()

    def __reduce__(self) -> object:
        raise TypeError("validated qualification identities cannot be pickled")


@dataclass(frozen=True, slots=True)
class _QualificationAuthority:
    """Values retained only by the validator until one engine consumes them."""

    label: str
    candidate: str
    no_text_watchdog_override_frames: int
    environment_sha256: str


_QUALIFICATION_AUTHORITIES: dict[
    ValidatedQualificationMode, _QualificationAuthority
] = {}
_QUALIFICATION_AUTHORITIES_LOCK = threading.Lock()


def is_validated_qualification_mode(value: object) -> bool:
    """Return whether an exact validator result has unconsumed authority."""

    if type(value) is not ValidatedQualificationMode:
        return False
    with _QUALIFICATION_AUTHORITIES_LOCK:
        return value in _QUALIFICATION_AUTHORITIES


def validated_qualification_mode_fields(
    value: object,
) -> tuple[str, str, int, str]:
    """Consume and return validator-owned fields for one exact result identity."""

    if type(value) is not ValidatedQualificationMode:
        raise TypeError("qualification_mode must be a validated qualification identity")
    with _QUALIFICATION_AUTHORITIES_LOCK:
        authority = _QUALIFICATION_AUTHORITIES.pop(value, None)
    if authority is None:
        raise TypeError("qualification_mode must be a validated qualification identity")
    return (
        authority.label,
        authority.candidate,
        authority.no_text_watchdog_override_frames,
        authority.environment_sha256,
    )


def validate_nano_pad_pair_production_policy(environment: Mapping[str, str]) -> None:
    """Reject serving configurations that re-enable retired PAD speculation."""
    actual = {
        name: environment.get(name, "0") for name in NANO_PAD_PAIR_PRODUCTION_POLICY
    }
    if actual != NANO_PAD_PAIR_PRODUCTION_POLICY:
        raise RuntimeError(
            "production Nano PAD-pair policy differs from the retired contract: "
            f"expected {NANO_PAD_PAIR_PRODUCTION_POLICY}, got {actual}"
        )

# These values affect model semantics, response boundaries, or latency. The
# The foreground launcher materializes this precise qualified set from TOML as
# explicit Docker environment values.
PRODUCTION_ENVIRONMENT = {
    "EA_CPU_CODEC_CORES": "5,6",
    "EA_CPU_CODEC": "1",
    "EA_CPU_CODEC_ASYNC": "1",
    "EA_CPU_CODEC_PERSISTENT": "1",
    "EA_CPU_CODEC_POINTWISE_LINEAR": "1",
    "EA_CPU_CODEC_PROCESS": "1",
    "EA_CPU_CODEC_THREADS": "2",
    "EA_DISABLE_CYCLIC_GC": "1",
    "EA_PERCEPTION_DTYPE": "bf16",
    "HF_HUB_OFFLINE": "1",
    "S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD": "false",
    "S2S_TTS_TEXT_TOKEN_MIN": "5",
    "S2S_TTS_TEXT_TOKEN_RATIO_CAP": "16",
    "TRANSFORMERS_OFFLINE": "1",
    "VOICECHAT_EARTTS_DECODE_PAD_SILENCE": "1",
    "VOICECHAT_EARTTS_IDLE_PAD_BYPASS": "1",
    "VOICECHAT_EARTTS_PREPARED_EPOCH": "1",
    "VOICECHAT_EARTTS_RESET_ON_BOS": "1",
    "VOICECHAT_FC_ALWAYS_ACKNOWLEDGE_TOOLS": "",
    "VOICECHAT_FC_FAST_TOOL_GRACE_MS": "100",
    "VOICECHAT_NANO_PAD_PAIR": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONDITIONAL": "0",
    "VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER": "0",
    "VOICECHAT_RNNT_BOU_FRAMES": "3",
    "VOICECHAT_RNNT_EOU_FRAMES": "20",
    "VOICECHAT_RNNT_FC_INTERRUPT_MS": "240",
    "VOICECHAT_STREAMING_MAX_LEN": "12288",
    "VOICECHAT_TRANSPORT_VAD_MIN_RNNT_TOKENS": "2",
    "VOICECHAT_TYPED_INPUT_SEED": "0",
    "VOICECHAT_USE_PERCEPTION_CUDAGRAPH": "1",
    "VOICECHAT_VLLM_DELTA_OUTPUT": "1",
    "VOICECHAT_VLLM_DELTA_OUTPUT_TARGET": "eartts",
    "VOICECHAT_VLLM_EARTTS_ENFORCE_EAGER": "0",
    "VOICECHAT_VLLM_MAX_MODEL_LEN": "12288",
    "VOICECHAT_WEB_AGENT_NO_AUDIO_FRAMES": "30",
    "VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES": "30",
    "VOICECHAT_WEB_AGENT_SILENCE_EOS_DBFS": "-90",
    "VOICECHAT_WEB_AGENT_SILENCE_EOS_FRAMES": "20",
    "VOICECHAT_WEB_CONTINUOUS_AFTER_SPEECH": "1",
    "VOICECHAT_WEB_DELIVERY_SILENCE_FRAMES": "12",
    "VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC": "0",
    "VOICECHAT_WEB_MAX_SESSION_MODEL_FRAMES": "12000",
    "VOICECHAT_WEB_REALTIME_WARMUP": "1",
    "VOICECHAT_WEB_RESPONSE_TAIL_DBFS": "-60",
    "VOICECHAT_WEB_RESPONSE_TAIL_MAX_FRAMES": "25",
    "VOICECHAT_WEB_RESPONSE_TAIL_SILENCE_FRAMES": "3",
    "VOICECHAT_WEB_SPEECH_GATE_DBFS": "-40",
    "VOICECHAT_WEB_SPEECH_GATE_MIN_FRAMES": "3",
    "VOICECHAT_WEB_SPEECH_GATE_ONSET_CONTEXT_FRAMES": "2",
    "VOICECHAT_WEB_SYSTEM_PROMPT": ("Respond concisely in one sentence. Do not repeat yourself."),
    "VOICECHAT_WEB_WS_PING_INTERVAL": "none",
    "VOICECHAT_WEB_WS_PING_TIMEOUT": "none",
}

def checked_in_candidate_environments() -> dict[str, dict[str, str]]:
    """Load the serving identities from the checked-in candidate TOMLs."""

    config_root = Path(__file__).resolve().parents[2] / "config"
    candidates: dict[str, dict[str, str]] = {}
    for path in sorted(config_root.glob("production-candidate-*.toml")):
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        candidate = config.get("candidate")
        environment = config.get("runtime", {}).get("environment")
        if not isinstance(candidate, str) or not isinstance(environment, dict):
            raise RuntimeError(f"invalid checked-in candidate environment: {path}")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise RuntimeError(f"non-string checked-in candidate environment: {path}")
        candidates[candidate] = dict(environment)
    if not candidates:
        raise RuntimeError(f"no checked-in candidate environments found beneath {config_root}")
    return candidates


def qualification_no_text_watchdog_override_frames(
    environment: Mapping[str, str],
) -> int:
    """Parse the qualification-only no-text watchdog threshold override."""

    raw = environment.get(QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV, "0").strip()
    if not raw:
        return 0
    try:
        frames = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV} must be a non-negative integer"
        ) from exc
    if frames < 0:
        raise RuntimeError(
            f"{QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV} must be a non-negative integer"
        )
    return frames


def validate_qualification_watchdog_production_policy(
    environment: Mapping[str, str],
) -> ValidatedQualificationMode | None:
    """Return positive qualification identity or refuse every unmanaged drift."""

    frames = qualification_no_text_watchdog_override_frames(environment)
    if not frames:
        return None
    label = environment.get(QUALIFICATION_MODE_ENV, "")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", label) is None:
        raise RuntimeError(
            f"{QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV} requires explicit "
            f"{QUALIFICATION_MODE_ENV}=<label> matching "
            "^[a-z0-9][a-z0-9-]{2,63}$"
        )
    if label == "production":
        raise RuntimeError(f"{QUALIFICATION_MODE_ENV}=production is explicitly refused")

    candidates = checked_in_candidate_environments()
    actual_managed = {
        name: value
        for name, value in environment.items()
        if name in _QUALIFICATION_MANAGED_NAMES
        or name.startswith(_QUALIFICATION_MANAGED_PREFIXES)
    }
    comparisons: list[tuple[int, str, list[str], dict[str, str]]] = []
    for candidate, expected in candidates.items():
        managed_names = (set(actual_managed) | set(expected)) - QUALIFICATION_ENVIRONMENT_DELTAS
        offending = sorted(
            name for name in managed_names if actual_managed.get(name) != expected.get(name)
        )
        comparisons.append((len(offending), candidate, offending, expected))
    _, candidate, offending, expected = min(comparisons, key=lambda item: (item[0], item[1]))
    if offending:
        raise RuntimeError(
            "qualification environment does not match a checked-in candidate; "
            f"closest={candidate}; offending keys: {', '.join(offending)}"
        )

    identity_environment = {
        **expected,
        "VOICECHAT_WEB_SYSTEM_PROMPT": environment.get("VOICECHAT_WEB_SYSTEM_PROMPT", ""),
        QUAL_NO_TEXT_WATCHDOG_OVERRIDE_ENV: str(frames),
        QUALIFICATION_MODE_ENV: label,
    }
    digest_input = "\n".join(
        f"{name}={identity_environment[name]}" for name in sorted(identity_environment)
    ).encode()
    authority = _QualificationAuthority(
        label=label,
        candidate=candidate,
        no_text_watchdog_override_frames=frames,
        environment_sha256=sha256(digest_input).hexdigest(),
    )
    result = ValidatedQualificationMode(
        label=authority.label,
        candidate=authority.candidate,
        no_text_watchdog_override_frames=authority.no_text_watchdog_override_frames,
        environment_sha256=authority.environment_sha256,
    )
    with _QUALIFICATION_AUTHORITIES_LOCK:
        _QUALIFICATION_AUTHORITIES[result] = authority
    return result
