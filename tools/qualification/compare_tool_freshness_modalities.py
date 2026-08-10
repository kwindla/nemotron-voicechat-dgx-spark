#!/usr/bin/env python3
"""Compare typed-input tool freshness against its exact Pocket speech twin."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
from transcribe_sustained import (
    TOOL_FRESHNESS_REPORT_KIND,
    classify_audio,
    normalize,
    read_and_resample,
    response_asr_evaluation,
    tool_freshness_runtime_gate_passed,
    word_error_rate,
)

from nemotron_voicechat_runtime.artifacts import atomic_json, sha256_file
from nemotron_voicechat_runtime.tool_freshness_contract import PARITY_MARGIN_POLICY

SCHEMA = 1
KIND = "tool_freshness_input_modality_parity_v1"
V2_SCHEMA = 2
V2_KIND = "tool_freshness_input_modality_parity_v2"
STRICT_MARGIN_V1 = "strict_margin_v1"
PARITY_MARGIN_V2 = PARITY_MARGIN_POLICY
MAX_WER = 0.5
MIN_WER_MARGIN = 0.1
ASR_POLICY = "audio_integrity_absolute_positive_semantic_paired"
PROMOTED_ASR_IMAGE_ID = (
    "sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d"
)
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
ASR_AGGREGATE_FIELDS = {
    "response_count",
    "speech_responses",
    "passing_speech_responses",
    "near_silent_responses",
    "near_silent_rate",
    "qualified_fp32_base_rate",
    "max_wer",
    "passed",
}
ASR_PLATFORM_CONTRACT = {
    "torch_version": "2.10.0a0+b4e4ee81d3.nv25.12",
    "cuda_version": "13.1",
    "python_version": "3.12.3",
    "machine": "aarch64",
    "device_name": "NVIDIA GB10",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def self_hash(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return canonical_sha256(unsigned)


def read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"parity arm report is unreadable: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"parity arm report is not an object: {path}")
    return report


def resolve_recorded_path(report_path: Path, recorded: Any) -> Path:
    path = Path(str(recorded or ""))
    if path.is_file():
        return path.resolve()
    candidate = report_path.parent / path.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"retained parity audio is missing: {recorded}")


def validate_wav(path: Path, *, sample_rate: int) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as stream:
            metadata = {
                "sample_rate": stream.getframerate(),
                "channels": stream.getnchannels(),
                "sample_width": stream.getsampwidth(),
                "samples": stream.getnframes(),
            }
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"retained parity WAV is invalid: {path}") from exc
    if metadata != {
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width": 2,
        "samples": metadata["samples"],
    } or metadata["samples"] <= 0:
        raise ValueError(f"retained parity WAV has the wrong format: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def expected_asr_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "config/nemotron-asr-en-0.6b.json"
    backend_path = root / "src/nemotron_voicechat_asr_evaluator/backend.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference = manifest["inference"]
    return {
        "backend": inference["backend"],
        "backend_path": "/opt/voicechat-asr/nemotron_voicechat_asr_evaluator/backend.py",
        "backend_sha256": sha256_file(backend_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_id": manifest["model_id"],
        "hf_revision": manifest["revision"],
        "snapshot_sha256": manifest["snapshot_sha256"],
        "snapshot_files": manifest["files"],
        "transformers_version": "5.14.1",
        "tokenizers_version": "0.22.2",
        "safetensors_version": "0.8.0",
        "dtype": inference["dtype"],
        "sampling_rate": inference["sampling_rate"],
        "language": inference["language"],
        "language_conditioning": inference["language_conditioning"],
        "decoder": {
            "strategy": inference["strategy"],
            "stopping": "encoder_exhaustion",
            "num_lookahead_tokens": inference["num_lookahead_tokens"],
            "attention_implementation": "sdpa",
            "streaming": False,
        },
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "float32_matmul_precision": "highest",
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "trust_remote_code": False,
        "network_required": False,
        "device": "cuda:0",
    }


def evaluator_provenance(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in gate.items()
        if key not in ASR_AGGREGATE_FIELDS and key != "evaluation_policy"
    }


def validate_evaluator_provenance(provenance: dict[str, Any]) -> None:
    image_id = provenance.get("evaluator_image_id")
    if (
        not isinstance(image_id, str)
        or IMAGE_ID_PATTERN.fullmatch(image_id) is None
        or image_id != PROMOTED_ASR_IMAGE_ID
    ):
        raise ValueError("ASR evaluator image is not the promoted immutable identity")
    expected = {**expected_asr_contract(), **ASR_PLATFORM_CONTRACT}
    if set(provenance) != {*expected, "evaluator_image_id"}:
        raise ValueError("ASR evaluator provenance schema drifted")
    mismatches = [
        key for key, value in expected.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"ASR evaluator provenance drifted: {mismatches}")


def validate_response_asr(
    report_path: Path,
    response: dict[str, Any],
) -> dict[str, Any]:
    external = response.get("external_asr")
    if not isinstance(external, dict) or external.get("classification") != "speech":
        raise ValueError("parity response was not independently classified as speech")
    source = resolve_recorded_path(report_path, response.get("audio_path"))
    source_record = validate_wav(source, sample_rate=22_050)
    if source_record["samples"] * 2 != response.get("audio_bytes"):
        raise ValueError("retained response WAV length differs from streamed audio bytes")
    values, _rate = read_and_resample(source)
    classification, level = classify_audio(values)
    if (
        classification != "speech"
        or external.get("classification") != classification
        or external.get("audio_dbfs") != level
        or external.get("source_audio_sha256") != source_record["sha256"]
    ):
        raise ValueError("response speech classification/source identity does not rederive")
    canonical = resolve_recorded_path(report_path, external.get("wav_16k"))
    canonical_record = validate_wav(canonical, sample_rate=16_000)
    expected_canonical_samples = round(source_record["samples"] * 16_000 / 22_050)
    if (
        external.get("canonical_wav_sha256") != canonical_record["sha256"]
        or canonical_record["samples"] != expected_canonical_samples
    ):
        raise ValueError("canonical ASR WAV identity does not rederive")
    expected_pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(canonical), "rb") as stream:
        canonical_pcm = stream.readframes(stream.getnframes())
    if canonical_pcm != expected_pcm:
        raise ValueError("canonical ASR WAV PCM does not derive from retained source")
    transcript = external.get("transcript")
    reference = response.get("text")
    token_ids = external.get("token_ids")
    if (
        not isinstance(transcript, str)
        or not transcript.strip()
        or not isinstance(reference, str)
        or not reference.strip()
        or not isinstance(token_ids, list)
        or not token_ids
        or any(type(token) is not int for token in token_ids)
    ):
        raise ValueError("response ASR text/token evidence is incomplete")
    expected = response.get("expected_response_any") or []
    excluded = external.get("expected_response_none") or []
    derived = response_asr_evaluation(
        hypothesis=transcript,
        reference=reference,
        expected=expected,
        excluded=excluded,
        max_wer=MAX_WER,
        parity_arm=True,
    )
    if any(external.get(key) != value for key, value in derived.items()):
        raise ValueError("response ASR verdict does not rederive")
    if derived["audio_integrity_passed"] is not True or derived["passed"] is not True:
        raise ValueError("response ASR audio-integrity gate failed")
    wer = float(derived["text_channel_wer"])
    return {
        "source_wav": source_record,
        "canonical_wav": canonical_record,
        "transcript": transcript,
        "token_ids": token_ids,
        **derived,
        "wer_margin": round(MAX_WER - wer, 4),
    }


def validate_arm(path: Path, *, modality: str) -> dict[str, Any]:
    report = read_report(path)
    expected_input_path = (
        "server_typed_pocket" if modality == "typed" else "client_pocket_fixture_pcm"
    )
    if (
        report.get("report_kind") != TOOL_FRESHNESS_REPORT_KIND
        or report.get("schema") != SCHEMA
        or report.get("input_modality") != modality
        or report.get("model_input_path") != expected_input_path
        or report.get("runtime_gate_passed") is not True
        or report.get("passed") is not True
        or not tool_freshness_runtime_gate_passed(report)
    ):
        raise ValueError(f"{modality} parity arm failed its independently recomputed gate")
    responses = report.get("responses")
    checks = report.get("tool_response_checks")
    inputs = report.get("inputs")
    if not all(
        isinstance(items, list) and len(items) == 4
        for items in (responses, checks, inputs)
    ):
        raise ValueError(f"{modality} parity arm cardinality is invalid")
    gate = report.get("external_asr")
    if not isinstance(gate, dict):
        raise ValueError(f"{modality} parity arm is missing external ASR")
    provenance = evaluator_provenance(gate)
    validate_evaluator_provenance(provenance)
    response_records = [validate_response_asr(path, response) for response in responses]
    derived_aggregate = {
        "response_count": 4,
        "speech_responses": 4,
        "passing_speech_responses": 4,
        "near_silent_responses": 0,
        "near_silent_rate": 0.0,
        "qualified_fp32_base_rate": 0.0,
        "max_wer": MAX_WER,
        "passed": True,
    }
    if (
        gate.get("evaluation_policy") != ASR_POLICY
        or any(gate.get(key) != value for key, value in derived_aggregate.items())
    ):
        raise ValueError(f"{modality} parity ASR aggregate does not rederive")
    cases = []
    for ordinal, (input_record, response, check, asr) in enumerate(
        zip(inputs, responses, checks, response_records, strict=True), 1
    ):
        if (
            input_record.get("ordinal") != ordinal
            or response.get("ordinal") != ordinal
            or input_record.get("case_id") != f"tool-freshness-{ordinal}"
            or response.get("case_id") != input_record.get("case_id")
            or response.get("turn_id") != input_record.get("turn_id")
            or check.get("turn_id") != input_record.get("turn_id")
        ):
            raise ValueError(f"{modality} parity case correlation drifted")
        input_wer = input_record.get("input_wer")
        if modality == "speech":
            derived_input_wer = round(
                word_error_rate(
                    str(input_record.get("expected_input_text") or ""),
                    str(input_record.get("displayed_input_text") or ""),
                ),
                4,
            )
            if input_wer != derived_input_wer or derived_input_wer > 0.35:
                raise ValueError("speech-input WER does not independently rederive")
        cases.append(
            {
                "case_id": input_record["case_id"],
                "ordinal": ordinal,
                "turn_id": input_record["turn_id"],
                "expected_response_any": response.get("expected_response_any"),
                "model_text": response.get("text"),
                "model_text_semantic_match": check.get("text_semantic_match") is True,
                "response_asr": asr,
                "response_audio_semantic_match": asr["semantic_match"] is True,
                "input_wer": input_wer,
            }
        )
    return {
        "report": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "report_data": report,
        "experiment_sha256": report.get("experiment_sha256"),
        "scenario_sha256": report.get("scenario_sha256"),
        "fixture_manifest_sha256": (report.get("fixture_manifest") or {}).get("sha256"),
        "session_id": report.get("session_id"),
        "runtime_provenance": report.get("runtime_provenance"),
        "evaluator_provenance": provenance,
        "asr_aggregate": derived_aggregate,
        "cases": cases,
        "source_report_sha256": (
            report.get("external_asr_source_report") or {}
        ).get("sha256"),
    }


def validate_runtime_report(
    path: Path,
    *,
    typed: dict[str, Any],
    speech: dict[str, Any],
    expected_kind: str = "tool_freshness_parity_runtime",
    expected_order: tuple[str, str] = ("typed", "speech"),
    require_nonpromotable: bool = False,
) -> dict[str, Any]:
    runtime = read_report(path)
    if (
        runtime.get("schema") != SCHEMA
        or runtime.get("kind") != expected_kind
        or runtime.get("passed") is not True
        or runtime.get("execution_order") != list(expected_order)
        or (not require_nonpromotable and "promotion_eligible" in runtime)
        or (
            require_nonpromotable
            and runtime.get("promotion_eligible") is not False
        )
        or runtime.get("experiment_sha256") != typed["experiment_sha256"]
        or runtime.get("scenario_sha256") != typed["scenario_sha256"]
        or runtime.get("fixture_manifest_sha256") != typed["fixture_manifest_sha256"]
        or runtime.get("runtime_provenance") != typed["runtime_provenance"]
        or runtime.get("session_ids")
        != [
            {"typed": typed, "speech": speech}[modality]["session_id"]
            for modality in expected_order
        ]
    ):
        raise ValueError("retained parity runtime report does not bind both arms")
    arms = runtime.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"typed", "speech"}:
        raise ValueError("retained parity runtime arm inventory is invalid")
    for modality, evaluated in (("typed", typed), ("speech", speech)):
        arm = arms.get(modality)
        if (
            not isinstance(arm, dict)
            or set(arm) != {"report", "structural_passed"}
            or arm.get("structural_passed") is not True
        ):
            raise ValueError("retained parity runtime arm declaration is invalid")
        raw_path = Path(str(arm.get("report") or ""))
        expected_raw_path = (path.resolve().parent / modality / "report.json").resolve()
        if not raw_path.is_absolute() or raw_path.resolve() != expected_raw_path:
            raise ValueError("retained parity runtime arm path is not canonical")
        if not raw_path.is_file():
            raise ValueError("retained parity runtime raw arm is missing")
        raw_sha256 = sha256_file(raw_path)
        if raw_sha256 != evaluated["source_report_sha256"]:
            raise ValueError("evaluated parity arm is not derived from its declared raw arm")
        raw_report = read_report(raw_path)
        evaluated_base = copy.deepcopy(evaluated["report_data"])
        evaluated_base.pop("external_asr_source_report", None)
        evaluated_base.pop("external_asr", None)
        evaluated_base.pop("runtime_gate_passed", None)
        for response in evaluated_base.get("responses") or []:
            response.pop("external_asr", None)
            response.pop("audio_dbfs", None)
        if evaluated_base != raw_report:
            raise ValueError("evaluated parity arm base differs from its raw source report")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "execution_order": runtime["execution_order"],
    }


def compare(
    typed_path: Path, speech_path: Path, runtime_report_path: Path
) -> dict[str, Any]:
    typed = validate_arm(typed_path, modality="typed")
    speech = validate_arm(speech_path, modality="speech")
    identity_fields = (
        "experiment_sha256",
        "scenario_sha256",
        "fixture_manifest_sha256",
        "runtime_provenance",
        "evaluator_provenance",
    )
    mismatched_identities = [
        field for field in identity_fields if typed[field] != speech[field]
    ]
    if mismatched_identities:
        raise ValueError(f"parity arm identities differ: {mismatched_identities}")
    if typed["session_id"] == speech["session_id"]:
        raise ValueError("parity arms reused one session")
    runtime_report = validate_runtime_report(
        runtime_report_path, typed=typed, speech=speech
    )

    comparisons = []
    blocking_reasons = []
    for typed_case, speech_case in zip(typed["cases"], speech["cases"], strict=True):
        if (
            typed_case["case_id"] != speech_case["case_id"]
            or typed_case["ordinal"] != speech_case["ordinal"]
            or typed_case["expected_response_any"]
            != speech_case["expected_response_any"]
        ):
            raise ValueError("paired tool-freshness case identity differs")
        typed_text = typed_case["model_text_semantic_match"]
        speech_text = speech_case["model_text_semantic_match"]
        typed_audio = typed_case["response_audio_semantic_match"]
        speech_audio = speech_case["response_audio_semantic_match"]
        text_typed_only_failure = not typed_text and speech_text
        audio_typed_only_failure = not typed_audio and speech_audio
        min_wer_margin = min(
            typed_case["response_asr"]["wer_margin"],
            speech_case["response_asr"]["wer_margin"],
        )
        if text_typed_only_failure:
            blocking_reasons.append(
                f"{typed_case['case_id']}:typed_only_model_text_semantic_failure"
            )
        if audio_typed_only_failure:
            blocking_reasons.append(
                f"{typed_case['case_id']}:typed_only_response_audio_semantic_failure"
            )
        if min_wer_margin < MIN_WER_MARGIN:
            blocking_reasons.append(
                f"{typed_case['case_id']}:response_asr_wer_margin_below_{MIN_WER_MARGIN}"
            )
        comparisons.append(
            {
                "case_id": typed_case["case_id"],
                "ordinal": typed_case["ordinal"],
                "expected_response_any": typed_case["expected_response_any"],
                "model_text_semantic": {
                    "typed": typed_text,
                    "speech": speech_text,
                    "typed_only_failure": text_typed_only_failure,
                    "shared_failure": not typed_text and not speech_text,
                    "speech_only_failure": typed_text and not speech_text,
                },
                "response_audio_semantic": {
                    "typed": typed_audio,
                    "speech": speech_audio,
                    "typed_only_failure": audio_typed_only_failure,
                    "shared_failure": not typed_audio and not speech_audio,
                    "speech_only_failure": typed_audio and not speech_audio,
                },
                "response_asr_wer": {
                    "typed": typed_case["response_asr"]["text_channel_wer"],
                    "speech": speech_case["response_asr"]["text_channel_wer"],
                    "minimum_margin": min_wer_margin,
                    "required_margin": MIN_WER_MARGIN,
                },
                "speech_input_wer": speech_case["input_wer"],
            }
        )
    artifact = {
        "schema": SCHEMA,
        "kind": KIND,
        "policy": {
            "absolute": [
                "both arm structural gates",
                "speech-input WER <= 0.35",
                "both response-audio integrity gates",
                "negative semantics",
                "zero near-silent responses",
                f"response ASR WER margin >= {MIN_WER_MARGIN}",
            ],
            "paired": {
                "model_text_positive_semantic": "typed >= speech per case",
                "response_audio_positive_semantic": "typed >= speech per case",
            },
        },
        "experiment_sha256": typed["experiment_sha256"],
        "scenario_sha256": typed["scenario_sha256"],
        "fixture_manifest_sha256": typed["fixture_manifest_sha256"],
        "runtime_provenance": typed["runtime_provenance"],
        "evaluator_provenance": typed["evaluator_provenance"],
        "runtime_report": runtime_report,
        "execution_order": runtime_report["execution_order"],
        "arms": {
            "typed": {
                "report": typed["report"],
                "session_id": typed["session_id"],
                "cases": typed["cases"],
            },
            "speech": {
                "report": speech["report"],
                "session_id": speech["session_id"],
                "cases": speech["cases"],
            },
        },
        "comparisons": comparisons,
        "blocking_reasons": blocking_reasons,
        "passed": not blocking_reasons,
    }
    artifact["sha256"] = self_hash(artifact)
    return artifact


def _margin_v2_disposition(
    *,
    typed_case: dict[str, Any],
    speech_case: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Classify one thin-margin case without weakening any absolute gate."""

    margin = float(comparison["response_asr_wer"]["minimum_margin"])
    if margin >= MIN_WER_MARGIN:
        return {"disposition": "clear"}
    typed_text = typed_case["model_text"]
    speech_text = speech_case["model_text"]
    normalized_word_count = len(normalize(typed_text))
    shared_model_negative = comparison["model_text_semantic"]["shared_failure"]
    shared_audio_negative = comparison["response_audio_semantic"]["shared_failure"]
    integrity_passed = all(
        case["response_asr"]["audio_integrity_passed"] is True
        for case in (typed_case, speech_case)
    )
    eligible_hold = bool(
        typed_text == speech_text
        and shared_model_negative
        and shared_audio_negative
        and integrity_passed
    )
    record = {
        "case_id": comparison["case_id"],
        "minimum_margin": margin,
        "required_margin": MIN_WER_MARGIN,
        "model_text_exactly_equal": typed_text == speech_text,
        "shared_model_text_negative": shared_model_negative,
        "shared_response_audio_negative": shared_audio_negative,
        "absolute_audio_integrity_passed": integrity_passed,
        "normalized_reference_word_count": normalized_word_count,
        "degenerate_reference": normalized_word_count <= 2,
    }
    if eligible_hold:
        return {
            "disposition": "review_hold",
            "reason": "shared_negative_equal_text_thin_margin",
            **record,
        }
    return {
        "disposition": "hard_block",
        "reason": "load_bearing_or_noncomparable_thin_margin",
        **record,
    }


def compare_v2(
    typed_path: Path, speech_path: Path, runtime_report_path: Path
) -> dict[str, Any]:
    """Apply prospective parity-margin v2 while retaining the exact v1 verdict."""

    prospective_runtime = read_report(runtime_report_path)
    if prospective_runtime.get("comparison_policy") != PARITY_MARGIN_V2:
        raise ValueError(
            "parity_margin_v2 requires a prospectively marked forward runtime report"
        )
    strict = compare(typed_path, speech_path, runtime_report_path)
    strict_margin_suffix = f"response_asr_wer_margin_below_{MIN_WER_MARGIN}"
    v2_blocking_reasons = [
        reason
        for reason in strict["blocking_reasons"]
        if not reason.endswith(strict_margin_suffix)
    ]
    held_cases: list[dict[str, Any]] = []
    margin_dispositions: list[dict[str, Any]] = []
    typed_cases = strict["arms"]["typed"]["cases"]
    speech_cases = strict["arms"]["speech"]["cases"]
    for typed_case, speech_case, comparison in zip(
        typed_cases, speech_cases, strict["comparisons"], strict=True
    ):
        disposition = _margin_v2_disposition(
            typed_case=typed_case,
            speech_case=speech_case,
            comparison=comparison,
        )
        if disposition["disposition"] == "clear":
            continue
        margin_dispositions.append(disposition)
        if disposition["disposition"] == "review_hold":
            held_cases.append(disposition)
        else:
            v2_blocking_reasons.append(
                f"{comparison['case_id']}:{strict_margin_suffix}"
            )
    core_passed = not v2_blocking_reasons
    unattended_promotion_eligible = bool(core_passed and not held_cases)
    artifact = copy.deepcopy(strict)
    artifact.update(
        {
            "schema": V2_SCHEMA,
            "kind": V2_KIND,
            "policy": {
                "selected": PARITY_MARGIN_V2,
                "absolute": [
                    item
                    for item in strict["policy"]["absolute"]
                    if not item.startswith("response ASR WER margin")
                ],
                "paired": strict["policy"]["paired"],
                "thin_margin": (
                    "hard block unless exact paired model text and shared-negative "
                    "model-text/audio semantics make the result eligible for "
                    "content-addressed attended review"
                ),
                "degenerate_reference": (
                    "normalized reference length <= 2 is recorded as a hold subtype "
                    "and never independently waives a margin"
                ),
            },
            "verdicts": {
                STRICT_MARGIN_V1: {
                    "blocking_reasons": strict["blocking_reasons"],
                    "passed": strict["passed"],
                },
                PARITY_MARGIN_V2: {
                    "blocking_reasons": v2_blocking_reasons,
                    "core_passed": core_passed,
                    "review_holds": held_cases,
                    "attended_review_required": bool(core_passed and held_cases),
                    "unattended_promotion_eligible": unattended_promotion_eligible,
                    "passed": unattended_promotion_eligible,
                },
            },
            "margin_dispositions": margin_dispositions,
            "blocking_reasons": v2_blocking_reasons,
            "core_passed": core_passed,
            "review_holds": held_cases,
            "attended_review_required": bool(core_passed and held_cases),
            "unattended_promotion_eligible": unattended_promotion_eligible,
            "passed": unattended_promotion_eligible,
        }
    )
    artifact["sha256"] = self_hash(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-report", type=Path, required=True)
    parser.add_argument("--speech-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=(STRICT_MARGIN_V1, PARITY_MARGIN_V2),
        default=PARITY_MARGIN_V2,
    )
    args = parser.parse_args()
    try:
        comparator = compare if args.policy == STRICT_MARGIN_V1 else compare_v2
        artifact = comparator(args.typed_report, args.speech_report, args.runtime_report)
    except Exception as exc:
        schema = SCHEMA if args.policy == STRICT_MARGIN_V1 else V2_SCHEMA
        kind = KIND if args.policy == STRICT_MARGIN_V1 else V2_KIND
        artifact = {
            "schema": schema,
            "kind": kind,
            "policy": {"selected": args.policy},
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        artifact["sha256"] = self_hash(artifact)
    atomic_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    raise SystemExit(0 if artifact["passed"] else 1)


if __name__ == "__main__":
    main()
