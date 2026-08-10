#!/usr/bin/env python3
"""Analyze a non-promotable speech-then-typed tool-freshness diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from compare_tool_freshness_modalities import (
    compare,
    read_report,
    resolve_recorded_path,
    self_hash,
    validate_arm,
    validate_runtime_report,
)

from nemotron_voicechat_runtime.artifacts import atomic_json, sha256_file
from nemotron_voicechat_runtime.tool_freshness_contract import (
    fixture_synthesis_identity,
)
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    validate_tool_freshness_fixtures,
)

SCHEMA = 1
KIND = "tool_freshness_reverse_order_diagnostic_v1"
FORWARD_COMPARISON_SHA256 = (
    "4889acc9305b691466ef80911280061a5d6eafd894a967c42d2a0d8381eae129"
)
FORWARD_COMPARISON_SELF_HASH = (
    "4916ec737007d14d96058229392a98a960778adc80ff842214054cfedd26274b"
)
FORWARD_FIXTURE_FILE_SHA256 = (
    "107bcf5086b1c08981976700fa44a7ee3434b7f672150f0b5b0e0ea7c6a3080c"
)
FORWARD_FIXTURE_SELF_HASH = (
    "a6f37db9e2fb8d63d24fb7febaa63d9a71d0ac42bd9c39db8e6c87fde22d9b93"
)
FRAME_MILLISECONDS = 80
ACTIVE_THRESHOLD_DBFS = -60.0
HEARTBEAT_DIAGNOSTICS = {
    "fc_async_heartbeat": True,
    "sigusr1_all_thread_dump": True,
}


def validate_frozen_forward(
    comparison_path: Path, fixture_manifest_path: Path
) -> dict[str, Any]:
    """Revalidate byte-pinned attempt 4 without trusting its stored verdict."""

    comparison_path = comparison_path.resolve()
    fixture_manifest_path = fixture_manifest_path.resolve()
    if sha256_file(comparison_path) != FORWARD_COMPARISON_SHA256:
        raise ValueError("frozen attempt-4 comparison file hash drifted")
    stored = read_report(comparison_path)
    if stored.get("sha256") != FORWARD_COMPARISON_SELF_HASH:
        raise ValueError("frozen attempt-4 comparison self-hash drifted")
    regenerated = compare(
        Path(stored["arms"]["typed"]["report"]["path"]),
        Path(stored["arms"]["speech"]["report"]["path"]),
        Path(stored["runtime_report"]["path"]),
    )
    if regenerated != stored or regenerated.get("passed") is not False:
        raise ValueError("frozen attempt-4 comparison does not exactly rederive red")
    if sha256_file(fixture_manifest_path) != FORWARD_FIXTURE_FILE_SHA256:
        raise ValueError("frozen attempt-4 fixture file hash drifted")
    fixture = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if fixture.get("sha256") != FORWARD_FIXTURE_SELF_HASH:
        raise ValueError("frozen attempt-4 fixture self-hash drifted")
    fixture, _pcm = validate_tool_freshness_fixtures(
        fixture_manifest_path.parent,
        expected_runtime_image_id=fixture["provenance"]["runtime_image_id"],
        expected_scenario_sha256=stored["scenario_sha256"],
    )
    if fixture["sha256"] != stored["fixture_manifest_sha256"]:
        raise ValueError("frozen comparison is not bound to the frozen fixture")
    return {
        "comparison": {
            "path": str(comparison_path),
            "sha256": FORWARD_COMPARISON_SHA256,
            "self_hash": FORWARD_COMPARISON_SELF_HASH,
        },
        "fixture": {
            "path": str(fixture_manifest_path),
            "sha256": FORWARD_FIXTURE_FILE_SHA256,
            "self_hash": FORWARD_FIXTURE_SELF_HASH,
        },
        "comparison_data": regenerated,
        "fixture_data": fixture,
    }


def audio_envelope(path: Path) -> dict[str, Any]:
    """Return deterministic, non-gating 80 ms activity diagnostics."""

    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("diagnostic source WAV must be mono PCM16")
        sample_rate = stream.getframerate()
        sample_count = stream.getnframes()
        pcm = stream.readframes(sample_count)
    if sample_count <= 0 or len(pcm) != sample_count * 2:
        raise ValueError("diagnostic source WAV must contain complete PCM16 samples")
    frame_numerator = sample_rate * FRAME_MILLISECONDS
    if frame_numerator % 1000:
        raise ValueError("diagnostic WAV rate has a non-integral 80 ms frame")
    samples_per_frame = frame_numerator // 1000
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    frame_dbfs: list[float] = []
    active: list[bool] = []
    for start in range(0, sample_count, samples_per_frame):
        frame = samples[start : start + samples_per_frame]
        rms = math.sqrt(float(np.mean(np.square(frame))))
        level = 20.0 * math.log10(max(rms, 1e-12))
        frame_dbfs.append(round(level, 4))
        active.append(level > ACTIVE_THRESHOLD_DBFS)
    active_indices = [index for index, is_active in enumerate(active) if is_active]
    first = active_indices[0] if active_indices else None
    last = active_indices[-1] if active_indices else None
    trailing = len(active) - last - 1 if last is not None else len(active)
    return {
        "source_wav": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "frame_milliseconds": FRAME_MILLISECONDS,
        "samples_per_full_frame": samples_per_frame,
        "total_frame_count": len(active),
        "final_frame_sample_count": sample_count % samples_per_frame
        or samples_per_frame,
        "active_threshold_dbfs": ACTIVE_THRESHOLD_DBFS,
        "first_active_frame": first,
        "last_active_frame": last,
        "trailing_frame_count": trailing,
        "frame_dbfs": frame_dbfs,
        "gating": False,
    }


def classify_wer_pattern(cells: dict[str, float]) -> str:
    """Apply the preregistered response-ASR-WER-only four-cell classifier."""

    if set(cells) != {
        "typed_forward",
        "speech_forward",
        "speech_reverse",
        "typed_reverse",
    }:
        raise ValueError("WER classifier cell schema drifted")
    typed_forward = cells["typed_forward"]
    speech_forward = cells["speech_forward"]
    speech_reverse = cells["speech_reverse"]
    typed_reverse = cells["typed_reverse"]
    if len(set(cells.values())) == 1:
        return "tie"
    if typed_forward == typed_reverse and speech_forward == speech_reverse:
        return "consistent_with_speech_modality"
    if typed_forward == speech_reverse and speech_forward == typed_reverse:
        return "consistent_with_second_position"
    return "unstable"


def _fixture_pair_identity(forward: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    synthesis_equal = fixture_synthesis_identity(forward) == fixture_synthesis_identity(
        reverse
    )
    carriers = []
    for forward_case, reverse_case in zip(
        forward["fixtures"], reverse["fixtures"], strict=True
    ):
        carriers.append(
            {
                "case_id": forward_case["case_id"],
                "case_identity_equal": (
                    forward_case["case_id"] == reverse_case["case_id"]
                    and forward_case["ordinal"] == reverse_case["ordinal"]
                    and forward_case["text"] == reverse_case["text"]
                ),
                "pcm_sha256_equal": (
                    forward_case["pcm"]["sha256"]
                    == reverse_case["pcm"]["sha256"]
                ),
                "wav_sha256_equal": (
                    forward_case["wav"]["sha256"]
                    == reverse_case["wav"]["sha256"]
                ),
            }
        )
    passed = synthesis_equal and all(
        all(value for key, value in item.items() if key != "case_id")
        for item in carriers
    )
    if not passed:
        raise ValueError("reverse run did not use the exact frozen Pocket carriers")
    return {
        "synthesis_identity_equal": synthesis_equal,
        "ordered_carriers": carriers,
        "passed": passed,
    }


def analyze(
    *,
    forward_comparison: Path,
    forward_fixture: Path,
    reverse_typed: Path,
    reverse_speech: Path,
    reverse_runtime: Path,
    reverse_fixture: Path,
) -> dict[str, Any]:
    frozen = validate_frozen_forward(forward_comparison, forward_fixture)
    forward = frozen["comparison_data"]
    typed = validate_arm(reverse_typed, modality="typed")
    speech = validate_arm(reverse_speech, modality="speech")
    if any(
        typed[field] != speech[field]
        for field in (
            "experiment_sha256",
            "scenario_sha256",
            "fixture_manifest_sha256",
            "runtime_provenance",
            "evaluator_provenance",
        )
    ):
        raise ValueError("reverse diagnostic arm identities differ")
    if typed["session_id"] == speech["session_id"]:
        raise ValueError("reverse diagnostic reused one session")
    runtime = validate_runtime_report(
        reverse_runtime,
        typed=typed,
        speech=speech,
        expected_kind="tool_freshness_reverse_order_diagnostic_runtime",
        expected_order=("speech", "typed"),
        require_nonpromotable=True,
    )
    reverse_runtime_data = read_report(reverse_runtime)
    reverse_manifest_data = json.loads(reverse_fixture.read_text(encoding="utf-8"))
    reverse_manifest, _pcm = validate_tool_freshness_fixtures(
        reverse_fixture.parent,
        expected_runtime_image_id=reverse_manifest_data["provenance"]["runtime_image_id"],
        expected_scenario_sha256=typed["scenario_sha256"],
    )
    if reverse_manifest["sha256"] != typed["fixture_manifest_sha256"]:
        raise ValueError("reverse arms are not bound to the reverse fixture")
    fixture_pair = _fixture_pair_identity(frozen["fixture_data"], reverse_manifest)
    forward_runtime = forward["runtime_provenance"]
    reverse_provenance = typed["runtime_provenance"]
    if (
        forward_runtime == reverse_provenance
        or forward_runtime.get("runtime_image_id")
        == reverse_provenance.get("runtime_image_id")
    ):
        raise ValueError("reverse diagnostic unexpectedly reused pre-heartbeat image")
    heartbeat_evidence = {
        "runtime_provenance": reverse_provenance.get("diagnostics"),
        "typed_health": (typed["report_data"].get("health_snapshot") or {}).get(
            "diagnostics"
        ),
        "speech_health": (speech["report_data"].get("health_snapshot") or {}).get(
            "diagnostics"
        ),
    }
    if any(value != HEARTBEAT_DIAGNOSTICS for value in heartbeat_evidence.values()):
        raise ValueError("reverse diagnostic did not prove heartbeat-enabled runtime")

    forward_cases = {
        modality: {case["case_id"]: case for case in forward["arms"][modality]["cases"]}
        for modality in ("typed", "speech")
    }
    reverse_cases = {
        "typed": {case["case_id"]: case for case in typed["cases"]},
        "speech": {case["case_id"]: case for case in speech["cases"]},
    }

    def retained_envelope(report_path: Path, case: dict[str, Any]) -> dict[str, Any]:
        expected_source = case["response_asr"]["source_wav"]
        envelope = audio_envelope(
            resolve_recorded_path(report_path, expected_source["path"])
        )
        if envelope["source_wav"]["sha256"] != expected_source["sha256"]:
            raise ValueError("audio envelope source differs from validated ASR source")
        return envelope

    cases = []
    for ordinal in range(1, 5):
        case_id = f"tool-freshness-{ordinal}"
        tf = forward_cases["typed"][case_id]
        sf = forward_cases["speech"][case_id]
        sr = reverse_cases["speech"][case_id]
        tr = reverse_cases["typed"][case_id]
        if not (
            tf["expected_response_any"]
            == sf["expected_response_any"]
            == sr["expected_response_any"]
            == tr["expected_response_any"]
        ):
            raise ValueError("cross-run case contract differs")
        cells = {
            "typed_forward": float(tf["response_asr"]["text_channel_wer"]),
            "speech_forward": float(sf["response_asr"]["text_channel_wer"]),
            "speech_reverse": float(sr["response_asr"]["text_channel_wer"]),
            "typed_reverse": float(tr["response_asr"]["text_channel_wer"]),
        }
        cases.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "response_asr_wer_cells": cells,
                "response_asr_wer_pattern": classify_wer_pattern(cells),
                "interpretation_is_causal": False,
                "model_text": {
                    "typed_forward": tf["model_text"],
                    "speech_forward": sf["model_text"],
                    "speech_reverse": sr["model_text"],
                    "typed_reverse": tr["model_text"],
                },
                "model_text_identical_to_forward": {
                    "typed": tr["model_text"] == tf["model_text"],
                    "speech": sr["model_text"] == sf["model_text"],
                },
                "response_asr_transcript": {
                    "typed_forward": tf["response_asr"]["transcript"],
                    "speech_forward": sf["response_asr"]["transcript"],
                    "speech_reverse": sr["response_asr"]["transcript"],
                    "typed_reverse": tr["response_asr"]["transcript"],
                },
                "semantics": {
                    "model_text": {
                        "typed_forward": tf["model_text_semantic_match"],
                        "speech_forward": sf["model_text_semantic_match"],
                        "speech_reverse": sr["model_text_semantic_match"],
                        "typed_reverse": tr["model_text_semantic_match"],
                    },
                    "response_audio": {
                        "typed_forward": tf["response_audio_semantic_match"],
                        "speech_forward": sf["response_audio_semantic_match"],
                        "speech_reverse": sr["response_audio_semantic_match"],
                        "typed_reverse": tr["response_audio_semantic_match"],
                    },
                },
                "response_audio_envelope": {
                    "typed_forward": retained_envelope(
                        Path(forward["arms"]["typed"]["report"]["path"]), tf
                    ),
                    "speech_forward": retained_envelope(
                        Path(forward["arms"]["speech"]["report"]["path"]), sf
                    ),
                    "speech_reverse": retained_envelope(reverse_speech, sr),
                    "typed_reverse": retained_envelope(reverse_typed, tr),
                },
            }
        )
    artifact = {
        "schema": SCHEMA,
        "kind": KIND,
        "passed": False,
        "promotion_eligible": False,
        "diagnostic_complete": True,
        "interpretation_scope": (
            "response_asr_wer_pattern is deterministic, descriptive, noncausal, "
            "and cannot override the frozen promotion failure"
        ),
        "runtime_provenance_mismatch_confounds_exact_ab": True,
        "forward_runtime_provenance": forward_runtime,
        "reverse_runtime_provenance": reverse_provenance,
        "reverse_heartbeat_diagnostics": {
            **heartbeat_evidence,
            "required": HEARTBEAT_DIAGNOSTICS,
            "passed": True,
        },
        "frozen_forward": {
            "comparison": frozen["comparison"],
            "fixture": frozen["fixture"],
        },
        "reverse": {
            "runtime_report": runtime,
            "runtime_report_data": reverse_runtime_data,
            "fixture_manifest": {
                "path": str(reverse_fixture.resolve()),
                "file_sha256": sha256_file(reverse_fixture),
                "self_hash": reverse_manifest["sha256"],
            },
            "arms": {
                "speech": speech["report"],
                "typed": typed["report"],
            },
        },
        "fixture_pair_identity": fixture_pair,
        "cases": cases,
    }
    artifact["sha256"] = self_hash(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-comparison", type=Path, required=True)
    parser.add_argument("--forward-fixture", type=Path, required=True)
    parser.add_argument("--reverse-typed-report", type=Path)
    parser.add_argument("--reverse-speech-report", type=Path)
    parser.add_argument("--reverse-runtime-report", type=Path)
    parser.add_argument("--reverse-fixture", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            frozen = validate_frozen_forward(
                args.forward_comparison, args.forward_fixture
            )
            artifact = {
                "schema": SCHEMA,
                "kind": "tool_freshness_reverse_order_preflight_v1",
                "passed": False,
                "promotion_eligible": False,
                "diagnostic_complete": True,
                "frozen_forward": {
                    "comparison": frozen["comparison"],
                    "fixture": frozen["fixture"],
                },
            }
        else:
            required = (
                args.reverse_typed_report,
                args.reverse_speech_report,
                args.reverse_runtime_report,
                args.reverse_fixture,
            )
            if any(path is None for path in required):
                raise ValueError("reverse diagnostic inputs are incomplete")
            artifact = analyze(
                forward_comparison=args.forward_comparison,
                forward_fixture=args.forward_fixture,
                reverse_typed=args.reverse_typed_report,
                reverse_speech=args.reverse_speech_report,
                reverse_runtime=args.reverse_runtime_report,
                reverse_fixture=args.reverse_fixture,
            )
    except Exception as exc:
        artifact = {
            "schema": SCHEMA,
            "kind": KIND,
            "passed": False,
            "promotion_eligible": False,
            "diagnostic_complete": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    artifact["sha256"] = self_hash(artifact)
    atomic_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    raise SystemExit(0 if artifact["diagnostic_complete"] else 1)


if __name__ == "__main__":
    main()
