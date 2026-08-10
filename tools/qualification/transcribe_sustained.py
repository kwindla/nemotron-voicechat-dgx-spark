#!/usr/bin/env python3
"""Independently transcribe strict-v3 response WAVs and close the audio/text gate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np

from nemotron_voicechat_runtime.artifacts import atomic_json, sha256_file
from nemotron_voicechat_runtime.provenance import valid_runtime_provenance

TARGET_RATE = 16_000
MULTITURN_REPORT_KIND = "multiturn_strict_v3"
MEMORY_MATRIX_REPORT_KIND = "memory_matrix_strict_v3"
SUSTAINED_REPORT_KIND = "sustained_strict_v3"
TOOL_FRESHNESS_REPORT_KIND = "tool_freshness_parity_arm_v1"
REPORT_KINDS = {
    MULTITURN_REPORT_KIND,
    MEMORY_MATRIX_REPORT_KIND,
    SUSTAINED_REPORT_KIND,
    TOOL_FRESHNESS_REPORT_KIND,
}
NUMBER_EQUIVALENTS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize(text: str) -> list[str]:
    normalized = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        token = NUMBER_EQUIVALENTS.get(token, token)
        normalized.append(str(int(token)) if token.isdigit() else token)
    return normalized


def contains_token_sequence(text: str, candidate: str) -> bool:
    haystack = normalize(text)
    needle = normalize(candidate)
    return bool(needle) and any(
        haystack[offset : offset + len(needle)] == needle
        for offset in range(len(haystack) - len(needle) + 1)
    )


def semantic_candidate_match(text: str, candidate: str) -> bool:
    """Match literal candidates plus the spoken form of an exact UTC time."""

    if contains_token_sequence(text, candidate):
        return True
    expected_time = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s+UTC\s*", candidate, re.IGNORECASE)
    if expected_time is None:
        return False
    hour, minute = (str(int(value)) for value in expected_time.groups())
    spoken = " ".join(normalize(text))
    return bool(
        re.search(
            rf"\b{re.escape(hour)} hours? (?:and )?{re.escape(minute)} minutes? utc\b",
            spoken,
        )
    )


def response_semantic_matches(
    hypothesis: str, expected: list[str], excluded: list[str]
) -> tuple[bool, bool]:
    positive = not expected or any(
        semantic_candidate_match(hypothesis, candidate) for candidate in expected
    )
    negative = not any(semantic_candidate_match(hypothesis, candidate) for candidate in excluded)
    return positive, negative


def response_asr_evaluation(
    *,
    hypothesis: str,
    reference: str,
    expected: list[str],
    excluded: list[str],
    max_wer: float,
    parity_arm: bool,
) -> dict[str, Any]:
    """Evaluate one response without changing legacy semantic requirements."""

    wer = word_error_rate(reference, hypothesis)
    semantic_match, negative_semantic_match = response_semantic_matches(
        hypothesis, expected, excluded
    )
    audio_integrity_passed = bool(
        hypothesis and reference and wer <= max_wer and negative_semantic_match
    )
    result = {
        "text_channel_wer": round(wer, 4),
        "expected_response_any": expected,
        "expected_response_none": excluded,
        "semantic_match": semantic_match,
        "negative_semantic_match": negative_semantic_match,
        "passed": (
            audio_integrity_passed
            if parity_arm
            else bool(audio_integrity_passed and semantic_match)
        ),
    }
    if parity_arm:
        result["audio_integrity_passed"] = audio_integrity_passed
    return result


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize(reference)
    actual = normalize(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for index, expected_token in enumerate(expected, 1):
        current = [index]
        for offset, actual_token in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (expected_token != actual_token),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def extract_text(result: Any) -> str:
    while isinstance(result, (list, tuple)) and result:
        result = result[0]
    return str(result.text) if hasattr(result, "text") else str(result or "")


def read_and_resample(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        payload = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only PCM16 WAVs are supported")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE and values.size:
        count = round(values.size * TARGET_RATE / rate)
        values = np.interp(
            np.arange(count, dtype=np.float64) * rate / TARGET_RATE,
            np.arange(values.size, dtype=np.float64),
            values,
        ).astype(np.float32)
    return values, rate


def dbfs(values: np.ndarray) -> float:
    if not values.size:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    return max(-120.0, 20 * math.log10(max(rms, 1e-6)))


def classify_audio(values: np.ndarray, *, silence_dbfs: float = -55.0) -> tuple[str, float]:
    """Apply the shared pre-ASR speech/silence classifier."""

    level = round(dbfs(values), 3)
    return ("near_silent" if level <= silence_dbfs else "speech"), level


def write_pcm16(path: Path, values: np.ndarray) -> None:
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(TARGET_RATE)
        output.writeframes(pcm.tobytes())


def resolve_audio_path(run_dir: Path, recorded: str) -> Path:
    """Map a host-recorded response path into the mounted qualification directory."""

    source = Path(recorded)
    if source.is_file():
        return source
    relative = run_dir / source
    return relative if relative.is_file() else run_dir / source.name


def memory_matrix_runtime_gate_passed(report: dict[str, Any]) -> bool:
    cases = report.get("cases") or []
    expected = report.get("expected_case_count")
    return bool(
        expected == 7
        and report.get("fresh_session_per_case")
        and report.get("runtime_provenance_consistent") is True
        and valid_runtime_provenance(report.get("runtime_provenance"))
        and len(cases) == expected
        and all(case.get("passed") is True for case in cases)
        and report.get("negative_control_ran_after_seeded") is True
        and [case.get("negative_control") for case in cases]
        == [False, False, False, False, False, False, True]
    )


def multiturn_runtime_gate_passed(report: dict[str, Any]) -> bool:
    responses = report.get("responses")
    expected_turn_count = report.get("expected_turn_count", 3)
    max_input_wer = report.get("max_input_wer", 0.35)
    session_closed = report.get("session_closed")
    return bool(
        expected_turn_count == 3
        and isinstance(max_input_wer, (int, float))
        and not isinstance(max_input_wer, bool)
        and 0 <= max_input_wer <= 1
        and isinstance(responses, list)
        and len(responses) == expected_turn_count
        and all(
            isinstance(response, dict)
            and isinstance(response.get("audio_bytes"), int)
            and not isinstance(response.get("audio_bytes"), bool)
            and response["audio_bytes"] > 0
            and isinstance(response.get("text"), str)
            and bool(response["text"])
            and isinstance(response.get("input_wer"), (int, float))
            and not isinstance(response.get("input_wer"), bool)
            and 0 <= response["input_wer"] <= max_input_wer
            and response.get("semantic_match") is True
            for response in responses
        )
        and isinstance(session_closed, dict)
        and bool(session_closed)
    )


def sustained_runtime_gate_passed(report: dict[str, Any]) -> bool:
    """Recompute the sustained-stream runtime verdict independently of ASR state."""

    limit_expected = bool(report.get("expected_session_position_limit"))
    close_reason = (report.get("session_closed") or {}).get("reason")
    sent_frames = report.get("sent_frames")
    source_frames = report.get("source_frames")
    if not isinstance(sent_frames, int) or not isinstance(source_frames, int):
        return False
    source_completion_ok = (
        sent_frames < source_frames and close_reason == "session_position_limit"
        if limit_expected
        else sent_frames == source_frames and close_reason == "client_stop"
    )
    typed_jobs = report.get("typed_jobs") or {}
    typed_completed = report.get("typed_completed")
    typed_answered = report.get("typed_answered")
    typed_unanswered = report.get("typed_unanswered")
    allowed_unanswered = report.get("allowed_unanswered_at_fp32_base_rate")
    responses = report.get("responses") or []
    requested = sum(
        item.get("expected_tool_call") is True
        for item in typed_jobs.values()
        if isinstance(item, dict)
    )
    tool_requested = requested > 0
    cardinality = report.get("tool_cycle_cardinality") or {}
    checks = report.get("tool_response_checks") or []
    observations = report.get("eotr_observations") or []
    validated = len(checks)
    observed_values = [
        item.get("observed_count")
        for item in observations
        if isinstance(item, dict) and isinstance(item.get("observed_count"), int)
    ]
    observed = max(observed_values) if observed_values else None
    checks_passed = bool(checks) and all(
        isinstance(check, dict)
        and check.get("eotr_valid") is True
        and check.get("function_channel_idle") is True
        and check.get("post_eotr_audible") is True
        and check.get("bounded_watchdog_close") is True
        and check.get("text_nonempty") is True
        and check.get("text_semantic_match") is True
        and check.get("audio_nonempty") is True
        for check in checks
    )
    cardinality_rederived = bool(
        requested > 0
        and validated == requested
        and observed == requested
        and cardinality.get("requested_tool_prompts") == requested
        and cardinality.get("validated_tool_responses") == validated
        and cardinality.get("eotr_observed_count") == observed
        and cardinality.get("client_eou_sotc_committed_count") == requested
        and cardinality.get("post_fc_client_bos_forced_count") == requested
        and checks_passed
    )
    return bool(
        not report.get("unexpected_errors")
        and source_completion_ok
        and (not limit_expected or report.get("expected_session_limit_error_count") == 1)
        and isinstance(typed_completed, int)
        and typed_completed >= max(1, len(typed_jobs) - 1)
        and isinstance(typed_unanswered, int)
        and isinstance(allowed_unanswered, int)
        and typed_unanswered <= allowed_unanswered
        and isinstance(typed_answered, int)
        and len(responses) >= typed_answered
        and (
            report.get("queue_required", True) is False
            or (report.get("queue") or {}).get("passed") is True
        )
        and report.get("metric_sequence_complete") is True
        and report.get("session_closed") is not None
        and report.get("serialized_typed_error") is None
        and bool(report.get("tool_prompt_requested")) == tool_requested
        and (
            not tool_requested
            or (report.get("tool_response_channel_gate") is True and cardinality_rederived)
        )
    )


def canonical_json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tool_freshness_runtime_gate_passed(report: dict[str, Any]) -> bool:
    """Independently rederive the semantics-excluded parity-arm structure."""

    from nemotron_voicechat_runtime.tool_freshness_contract import (
        derive_structural_conjuncts,
        evaluate_tool_cycle_cardinality,
        evaluate_tool_response_channels,
        experiment_sha256,
        fixture_synthesis_identity,
        health_synthesis_identity,
        scenario_document_for_sha256,
    )
    from nemotron_voicechat_runtime.tool_freshness_fixture import (
        self_hash,
        validate_tool_freshness_manifest_document,
    )

    modality = report.get("input_modality")
    if modality not in {"typed", "speech"}:
        return False
    fixture = report.get("fixture_manifest") or {}
    scenario = report.get("scenario")
    scenario_sha256 = report.get("scenario_sha256")
    provenance = report.get("runtime_provenance") or {}
    try:
        validate_tool_freshness_manifest_document(
            fixture,
            expected_runtime_image_id=provenance.get("runtime_image_id"),
            expected_scenario_sha256=scenario_sha256,
        )
        expected_scenario = scenario_document_for_sha256(scenario_sha256)
        scenario_identity = bool(
            report.get("schema") == 1
            and isinstance(scenario, dict)
            and scenario == expected_scenario
            and report.get("model_input_path")
            == ("server_typed_pocket" if modality == "typed" else "client_pocket_fixture_pcm")
            and canonical_json_sha256(scenario) == scenario_sha256
            and fixture.get("scenario_sha256") == scenario_sha256
            and (report.get("session_configuration") or {}).get("id") == report.get("session_id")
            and (report.get("session_configuration") or {}).get("instructions")
            == scenario.get("instructions")
            and (report.get("session_configuration") or {}).get("tools") == [scenario.get("tool")]
        )
        health_snapshot = report.get("health_snapshot") or {}
        runtime_fixture_identity = bool(
            scenario_identity
            and fixture.get("sha256") == self_hash(fixture)
            and report.get("experiment_sha256")
            == experiment_sha256(
                scenario_sha256=scenario_sha256,
                fixture_manifest_sha256=fixture.get("sha256"),
            )
            and (report.get("live_typed_input_health") or {}).get("ready") is True
            and report.get("live_typed_input_health") == health_snapshot.get("typed_input")
            and valid_runtime_provenance(provenance)
            and fixture_synthesis_identity(fixture)
            == health_synthesis_identity(report.get("live_typed_input_health") or {})
            and fixture.get("provenance", {}).get("runtime_image_id")
            == provenance.get("runtime_image_id")
            and report.get("health_runtime_provenance") == provenance
            and ((health_snapshot.get("checkpoint") or {}).get("runtime_provenance")) == provenance
        )
        derived_checks, _semantic_coupled_gate = evaluate_tool_response_channels(
            report.get("responses") or [],
            report.get("metrics") or [],
            report.get("eotr_observations") or [],
        )
        derived_cardinality = evaluate_tool_cycle_cardinality(
            requested_tool_prompts=4,
            tool_response_checks=derived_checks,
            metrics=report.get("metrics") or [],
        )
        derived = derive_structural_conjuncts(
            modality=modality,
            inputs=report.get("inputs") or [],
            responses=report.get("responses") or [],
            checks=derived_checks,
            cardinality=derived_cardinality,
            errors=report.get("errors") or [],
            session_closed=report.get("session_closed"),
            scenario=scenario,
            fixtures=fixture.get("fixtures") or [],
            scenario_identity=scenario_identity,
            runtime_fixture_identity=runtime_fixture_identity,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return bool(
        report.get("tool_response_checks") == derived_checks
        and report.get("tool_cycle_cardinality") == derived_cardinality
        and report.get("structural_conjuncts") == derived
        and report.get("structural_passed") is all(derived.values())
        and report.get("structural_passed") is True
    )


def infer_legacy_report_kind(report: dict[str, Any]) -> str | None:
    """Recognize pre-kind reports only when exactly one strict shape matches."""

    candidates = []
    if {
        "fixture_generator",
        "manifest",
        "responses",
        "session_closed",
    }.issubset(report):
        candidates.append(MULTITURN_REPORT_KIND)
    if {
        "matrix",
        "cases",
        "expected_case_count",
        "fresh_session_per_case",
        "runtime_provenance",
        "runtime_provenance_consistent",
        "negative_control_ran_after_seeded",
    }.issubset(report):
        candidates.append(MEMORY_MATRIX_REPORT_KIND)
    if {
        "source_frames",
        "sent_frames",
        "typed_jobs",
        "typed_completed",
        "typed_answered",
        "typed_unanswered",
        "queue",
        "metric_sequence_complete",
        "session_closed",
        "responses",
    }.issubset(report):
        candidates.append(SUSTAINED_REPORT_KIND)
    return candidates[0] if len(candidates) == 1 else None


def report_kind(report: dict[str, Any]) -> str | None:
    explicit = report.get("report_kind")
    if explicit is not None:
        return explicit if explicit in REPORT_KINDS else None
    return infer_legacy_report_kind(report)


def runtime_gate_passed(report: dict[str, Any]) -> bool:
    """Recompute the pre-ASR verdict so evaluator reruns are idempotent."""

    kind = report_kind(report)
    if kind == MULTITURN_REPORT_KIND:
        return multiturn_runtime_gate_passed(report)
    if kind == MEMORY_MATRIX_REPORT_KIND:
        return memory_matrix_runtime_gate_passed(report)
    if kind == SUSTAINED_REPORT_KIND:
        return sustained_runtime_gate_passed(report)
    if kind == TOOL_FRESHNESS_REPORT_KIND:
        return tool_freshness_runtime_gate_passed(report)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write evaluated report/audio here, leaving --run-dir read-only",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-wer", type=float, default=0.50)
    parser.add_argument("--silence-dbfs", type=float, default=-55.0)
    parser.add_argument(
        "--max-silent-rate",
        type=float,
        default=1 / 15,
        help="qualified FP32 public-model near-silent base rate",
    )
    args = parser.parse_args()

    source_report_path = args.run_dir / "report.json"
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = copy.deepcopy(json.loads(source_report_path.read_text(encoding="utf-8")))
    kind = report_kind(report)
    parity_arm = kind == TOOL_FRESHNESS_REPORT_KIND
    if parity_arm:
        report["external_asr_source_report"] = {
            "sha256": sha256_file(source_report_path),
        }
    prepared: list[tuple[dict[str, Any], Path, Path, float, np.ndarray]] = []
    silent: list[dict[str, Any]] = []
    for index, response in enumerate(report.get("responses", []), 1):
        source = resolve_audio_path(args.run_dir, response["audio_path"])
        values, _source_rate = read_and_resample(source)
        classification, level = classify_audio(values, silence_dbfs=args.silence_dbfs)
        response["audio_dbfs"] = level
        if classification == "near_silent":
            response["external_asr"] = {
                "classification": classification,
                "audio_dbfs": level,
                "source_audio_sha256": sha256_file(source),
            }
            silent.append(response)
            continue
        destination = output_dir / f"response-{index:03d}-16k.wav"
        write_pcm16(destination, values)
        canonical_values, canonical_rate = read_and_resample(destination)
        if canonical_rate != TARGET_RATE:
            raise RuntimeError("canonical ASR WAV was not written at 16 kHz")
        prepared.append((response, source, destination, level, canonical_values))

    from nemotron_voicechat_asr_evaluator import (
        DEFAULT_MANIFEST,
        DEFAULT_MODEL,
        NemotronEnglishAsr,
    )

    backend = NemotronEnglishAsr(
        model_root=(args.model or DEFAULT_MODEL),
        manifest_path=(args.model_manifest or DEFAULT_MANIFEST),
        device=args.device,
    )
    hypotheses = [backend.transcribe(values) for *_, values in prepared]
    asr_provenance = backend.provenance()

    passed_audio = 0
    for (response, source, path, level, _values), raw in zip(prepared, hypotheses, strict=True):
        hypothesis = raw["transcript"]
        reference = str(response.get("text") or "")
        expected = response.get("expected_response_any") or []
        excluded = response.get("expected_response_none") or []
        evaluation = response_asr_evaluation(
            hypothesis=hypothesis,
            reference=reference,
            expected=expected,
            excluded=excluded,
            max_wer=args.max_wer,
            parity_arm=parity_arm,
        )
        response["external_asr"] = {
            "classification": "speech",
            "audio_dbfs": level,
            "wav_16k": str(path),
            "source_audio_sha256": sha256_file(source),
            "canonical_wav_sha256": sha256_file(path),
            "transcript": hypothesis,
            "token_ids": raw["token_ids"],
            **evaluation,
        }
        passed_audio += int(evaluation["passed"])

    response_count = len(report.get("responses", []))
    silent_rate = len(silent) / response_count if response_count else 1.0
    asr_gate = {
        **asr_provenance,
        "response_count": response_count,
        "speech_responses": len(prepared),
        "passing_speech_responses": passed_audio,
        "near_silent_responses": len(silent),
        "near_silent_rate": round(silent_rate, 4),
        "qualified_fp32_base_rate": round(args.max_silent_rate, 4),
        "max_wer": args.max_wer,
        "passed": (
            bool(prepared) and passed_audio == len(prepared) and silent_rate <= args.max_silent_rate
        ),
    }
    if parity_arm:
        asr_gate["evaluation_policy"] = "audio_integrity_absolute_positive_semantic_paired"
    report["external_asr"] = asr_gate
    report["runtime_gate_passed"] = runtime_gate_passed(report)
    report["passed"] = report["runtime_gate_passed"] and asr_gate["passed"]
    atomic_json(report_path, report)
    print(json.dumps(asr_gate, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
