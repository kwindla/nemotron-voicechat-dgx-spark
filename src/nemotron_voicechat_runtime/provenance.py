"""Exact source and environment contract for Production Candidate 1."""

from __future__ import annotations

import re
from collections.abc import Mapping

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
