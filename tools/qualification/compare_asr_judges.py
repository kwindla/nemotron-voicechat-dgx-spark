#!/usr/bin/env python3
"""Build and verify content-addressed evidence for an ASR judge replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from transcribe_sustained import (
    classify_audio,
    response_semantic_matches,
    runtime_gate_passed,
    word_error_rate,
)

from nemotron_voicechat_runtime.artifacts import atomic_json, sha256_file

if TYPE_CHECKING:
    from collections.abc import Sequence


DECISION_FIELDS = (
    "classification",
    "semantic_match",
    "negative_semantic_match",
    "passed",
)
MAX_WER = 0.5
GATE_RESULT_FIELDS = {
    "max_wer",
    "near_silent_rate",
    "near_silent_responses",
    "passed",
    "passing_speech_responses",
    "qualified_fp32_base_rate",
    "response_count",
    "speech_responses",
}
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
CONTROL_REQUIRED = ("five",)
CONTROL_EXCLUDED = ("six",)
CONTROL_LABELS = {
    "eager-streaming",
    "eager-offline",
    "graph-streaming",
    "graph-offline",
}
LEGACY_DECODER_GIT_BLOB = "8b8af889db779aeeaf660b4f57d676ebdcd9ed6f"
LEGACY_DECODER_SHA256 = "674a9ba6a97cd09186e16a69b355c59d67708d2773405b97d82b2a0f2d38cee3"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def self_hash(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return canonical_sha256(unsigned)


def verify_self_hash(payload: dict[str, Any]) -> None:
    recorded = payload.get("sha256")
    if not isinstance(recorded, str) or recorded != self_hash(payload):
        raise ValueError("ASR calibration artifact failed its self-hash")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return value


def _report_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _evaluator_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("evaluator provenance is missing")
    return {key: item for key, item in value.items() if key not in GATE_RESULT_FIELDS}


def _asr_record(response: dict[str, Any], *, include_tokens: bool) -> dict[str, Any]:
    asr = response.get("external_asr")
    if not isinstance(asr, dict):
        raise ValueError("response is missing external_asr evidence")
    classification = asr.get("classification")
    if classification == "near_silent":
        level = asr.get("audio_dbfs")
        if not isinstance(level, (int, float)) or isinstance(level, bool):
            raise ValueError("near-silent response is missing numeric dBFS")
        record = {
            "classification": classification,
            "audio_dbfs": level,
            "transcript": None,
            "wer": None,
            "recomputed_wer": None,
            "wer_within_threshold": None,
            "semantic_match": None,
            "negative_semantic_match": None,
            "passed": False,
        }
        if include_tokens:
            record["token_ids"] = []
        return record
    transcript = asr.get("transcript")
    reference = response.get("text")
    if not isinstance(transcript, str) or not isinstance(reference, str):
        raise ValueError("response is missing transcript/reference text")
    reported_wer = asr.get("text_channel_wer")
    if (
        not isinstance(reported_wer, (int, float))
        or isinstance(reported_wer, bool)
        or not 0 <= reported_wer
    ):
        raise ValueError("response has an invalid WER")
    recomputed_wer = word_error_rate(reference, transcript)
    if round(recomputed_wer, 4) != reported_wer:
        raise ValueError("response WER does not rederive from transcript/reference")
    expected = response.get("expected_response_any") or []
    excluded = asr.get("expected_response_none") or []
    semantic, negative = response_semantic_matches(transcript, expected, excluded)
    if classification != "speech":
        raise ValueError("response has an invalid audio classification")
    if asr.get("semantic_match") is not semantic or asr.get(
        "negative_semantic_match"
    ) is not negative:
        raise ValueError("response semantic verdict does not rederive")
    passed = bool(
        classification == "speech"
        and transcript
        and reference
        and recomputed_wer <= MAX_WER
        and semantic
        and negative
    )
    if asr.get("passed") is not passed:
        raise ValueError("response pass verdict does not rederive")
    record = {
        "transcript": asr.get("transcript"),
        "wer": reported_wer,
        "recomputed_wer": recomputed_wer,
        "wer_within_threshold": recomputed_wer <= MAX_WER,
        **{field: asr.get(field) for field in DECISION_FIELDS},
    }
    if include_tokens:
        tokens = asr.get("token_ids")
        if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
            raise ValueError("new evaluator response is missing integer token_ids")
        record["token_ids"] = tokens
    return record


def _wav_record(response: dict[str, Any], new_response: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(response.get("audio_path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"retained response WAV is missing: {path}")
    new_asr = new_response.get("external_asr", {})
    source_hash = sha256_file(path)
    if new_asr.get("source_audio_sha256") != source_hash:
        raise ValueError(f"new evaluator did not consume retained WAV bytes: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": source_hash,
        "canonical_wav_sha256": new_asr.get("canonical_wav_sha256"),
    }


def _masked_wer_flip(old: dict[str, Any], new: dict[str, Any]) -> bool:
    if old["wer_within_threshold"] == new["wer_within_threshold"]:
        return False
    if old["passed"] != new["passed"]:
        return False
    # WER is non-load-bearing only when another conjunct is identically false.
    return any(
        old[field] is False and new[field] is False
        for field in ("semantic_match", "negative_semantic_match")
    ) or (old["classification"] == new["classification"] != "speech")


def _positive_controls(
    old_paths: list[Path],
    new_path: Path,
    *,
    old_model_sha256: str,
    new_provenance: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    old_cases = []
    for path in old_paths:
        report = _read(path)
        reference = report.get("fixture", {}).get("reference_text")
        lane = path.parent.name.removeprefix("eartts-")
        if lane not in {"eager", "graph"}:
            raise ValueError(f"old positive-control lane is unknown: {path}")
        old_model_matches = (
            report.get("external_asr", {}).get("sha256") == old_model_sha256
        )
        variants = (
            ("streaming", report.get("audio", {}), report.get("external_asr", {})),
            (
                "offline",
                report.get("offline_diagnostic", {}),
                report.get("offline_diagnostic", {}).get("external_asr", {}),
            ),
        )
        for variant, audio, external in variants:
            transcript = external.get("transcript")
            recomputed_wer = (
                word_error_rate(reference, transcript)
                if isinstance(reference, str) and isinstance(transcript, str)
                else None
            )
            semantic, negative = response_semantic_matches(
                str(transcript or ""), CONTROL_REQUIRED, CONTROL_EXCLUDED
            )
            _, negative_failure_match = response_semantic_matches(
                str(transcript or ""), (), CONTROL_REQUIRED
            )
            source = path.parent / Path(str(audio.get("path", ""))).name
            canonical = source.with_name(f"{source.stem}-16k.wav")
            if not source.is_file() or not canonical.is_file():
                raise FileNotFoundError(f"old positive-control WAV is missing: {source}")
            old_cases.append(
                {
                    "label": f"{lane}-{variant}",
                    "report": _report_record(path),
                    "source_wav": {
                        "path": str(source.resolve()),
                        "bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                    },
                    "canonical_wav": {
                        "path": str(canonical.resolve()),
                        "bytes": canonical.stat().st_size,
                        "sha256": sha256_file(canonical),
                    },
                    "reference": reference,
                    "transcript": transcript,
                    "wer": external.get("wer"),
                    "recomputed_wer": recomputed_wer,
                    "required_semantic_match": semantic,
                    "excluded_semantic_match": negative,
                    "negative_failure_expected_none": list(CONTROL_REQUIRED),
                    "negative_failure_detected": not negative_failure_match,
                    "model_sha256_matches": old_model_matches,
                    "passed": report.get("passed") is True
                    and bool(transcript)
                    and isinstance(external.get("wer"), (int, float))
                    and external["wer"] == recomputed_wer
                    and external["wer"] <= MAX_WER
                    and semantic
                    and negative
                    and not negative_failure_match
                    and old_model_matches,
                }
            )
    new_report = _read(new_path)
    new_cases = new_report.get("cases")
    if not isinstance(new_cases, list) or not new_cases:
        raise ValueError("new positive-control report has no cases")
    if len(old_cases) != len({case["label"] for case in old_cases}):
        raise ValueError("old positive-control labels are duplicated")
    if {case["label"] for case in old_cases} != CONTROL_LABELS:
        raise ValueError("old positive controls do not cover the exact lane matrix")
    new_cases_rederived = []
    old_by_label = {case["label"]: case for case in old_cases}
    new_labels = [case.get("label") for case in new_cases]
    if len(new_labels) != len(set(new_labels)):
        raise ValueError("new positive-control labels are duplicated")
    for case in new_cases:
        label = case.get("label")
        if label not in old_by_label:
            raise ValueError(f"new positive control does not match an old lane: {label}")
        reference = case.get("reference")
        transcript = case.get("transcript")
        recomputed_wer = (
            word_error_rate(reference, transcript)
            if isinstance(reference, str) and isinstance(transcript, str)
            else None
        )
        semantic, negative = response_semantic_matches(
            str(transcript or ""), CONTROL_REQUIRED, CONTROL_EXCLUDED
        )
        _, negative_failure_match = response_semantic_matches(
            str(transcript or ""), (), CONTROL_REQUIRED
        )
        canonical_matches = (
            case.get("wav_bytes") == old_by_label[label]["canonical_wav"]["bytes"]
            and case.get("wav_sha256")
            == old_by_label[label]["canonical_wav"]["sha256"]
        )
        passed = bool(
            transcript
            and isinstance(case.get("wer"), (int, float))
            and case["wer"] == recomputed_wer
            and case["wer"] <= MAX_WER
            and semantic
            and negative
            and not negative_failure_match
            and canonical_matches
        )
        if case.get("passed") is not passed:
            raise ValueError("new positive-control verdict does not rederive")
        new_cases_rederived.append(
            {
                **case,
                "recomputed_wer": recomputed_wer,
                "required_semantic_match": semantic,
                "excluded_semantic_match": negative,
                "negative_failure_expected_none": list(CONTROL_REQUIRED),
                "negative_failure_detected": not negative_failure_match,
                "canonical_wav_matches_old_control": canonical_matches,
            }
        )
    if set(old_by_label) != set(new_labels):
        raise ValueError("positive-control lane coverage differs between judges")
    control_provenance = _evaluator_provenance(new_report.get("external_asr"))
    provenance_matches = control_provenance == new_provenance
    controls_pass = (
        bool(old_cases)
        and all(case["passed"] for case in old_cases)
        and provenance_matches
        and new_report.get("passed") is True
        and all(case["passed"] for case in new_cases_rederived)
    )
    return (
        {
            "old": old_cases,
            "new": {
                "report": _report_record(new_path),
                "evaluator": control_provenance,
                "evaluator_matches_candidate": provenance_matches,
                "cases": new_cases_rederived,
            },
        },
        controls_pass,
    )


def _silence_control() -> dict[str, Any]:
    pcm = bytes(16_000 * 2)
    sample_hash = hashlib.sha256(pcm).hexdigest()
    values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    classification, level = classify_audio(values)
    passed = classification == "near_silent"
    return {
        "construction": "one_second_pcm16_mono_16000_all_zero",
        "pcm_bytes": len(pcm),
        "pcm_sha256": sample_hash,
        "audio_dbfs": level,
        "silence_dbfs": -55.0,
        "classifier_owner": "shared_pre_asr_harness",
        "old": {"classification": classification, "asr_invoked": False},
        "new": {"classification": classification, "asr_invoked": False},
        "classification_identical": True,
        "passed": passed,
    }


def _rederive_asr_gate(
    report: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    gate = report.get("external_asr")
    if not isinstance(gate, dict):
        raise ValueError("report is missing external ASR aggregate")
    response_count = len(records)
    speech = [record for record in records if record["classification"] == "speech"]
    silent_count = response_count - len(speech)
    passing = sum(record["passed"] is True for record in speech)
    silent_rate = silent_count / response_count
    max_silent_rate = gate.get("qualified_fp32_base_rate")
    if not isinstance(max_silent_rate, (int, float)) or isinstance(max_silent_rate, bool):
        raise ValueError("report has an invalid qualified near-silent rate")
    max_wer = gate.get("max_wer")
    if (
        not isinstance(max_wer, (int, float))
        or isinstance(max_wer, bool)
        or max_wer != MAX_WER
    ):
        raise ValueError("report has an invalid max WER")
    derived = {
        "response_count": response_count,
        "speech_responses": len(speech),
        "passing_speech_responses": passing,
        "near_silent_responses": silent_count,
        "near_silent_rate": round(silent_rate, 4),
        "qualified_fp32_base_rate": max_silent_rate,
        "max_wer": max_wer,
        "passed": bool(speech)
        and passing == len(speech)
        and silent_rate <= max_silent_rate,
    }
    for field, value in derived.items():
        if gate.get(field) != value:
            raise ValueError(f"external ASR aggregate does not rederive: {field}")
    derived_runtime = runtime_gate_passed(report)
    if report.get("runtime_gate_passed") is not derived_runtime:
        raise ValueError("stored runtime gate does not rederive")
    derived_final = derived_runtime and derived["passed"]
    if report.get("passed") is not derived_final:
        raise ValueError("stored final report verdict does not rederive")
    return {
        **derived,
        "runtime_gate_passed": derived_runtime,
        "report_passed": derived_final,
    }


def _old_rerun_attestation(
    *,
    label: str,
    archived_path: Path,
    rerun_path: Path,
    log_path: Path,
    runtime_image_id: str,
    model_path: Path,
    model_sha256: str,
) -> dict[str, Any]:
    archived = _read(archived_path)
    rerun = _read(rerun_path)
    archived_responses = archived.get("responses")
    rerun_responses = rerun.get("responses")
    if not isinstance(archived_responses, list) or not isinstance(rerun_responses, list):
        raise ValueError(f"{label}: old rerun responses are malformed")
    if len(archived_responses) != len(rerun_responses) or not archived_responses:
        raise ValueError(f"{label}: old rerun cardinality differs")
    rows = []
    for index, (archived_response, rerun_response) in enumerate(
        zip(archived_responses, rerun_responses, strict=True), 1
    ):
        archived_asr = archived_response.get("external_asr", {})
        rerun_asr = rerun_response.get("external_asr", {})
        archived_wav = archived_path.parent / f"response-{index:03d}-16k.wav"
        rerun_wav = rerun_path.parent / f"response-{index:03d}-16k.wav"
        if not archived_wav.is_file() or not rerun_wav.is_file():
            raise FileNotFoundError(f"{label}: old canonical rerun WAV is missing")
        row = {
            "turn": index,
            "transcript_identical": archived_asr.get("transcript")
            == rerun_asr.get("transcript"),
            "wer_identical": archived_asr.get("text_channel_wer")
            == rerun_asr.get("text_channel_wer"),
            "canonical_wav_sha256": sha256_file(archived_wav),
            "rerun_canonical_wav_sha256": sha256_file(rerun_wav),
        }
        row["canonical_wav_identical"] = (
            row["canonical_wav_sha256"] == row["rerun_canonical_wav_sha256"]
        )
        if not all(
            row[field]
            for field in (
                "transcript_identical",
                "wer_identical",
                "canonical_wav_identical",
            )
        ):
            raise ValueError(f"{label}: old provenance rerun differs at turn {index}")
        rows.append(row)
    archived_gate = archived.get("external_asr", {})
    rerun_gate = rerun.get("external_asr", {})
    aggregate_fields_identical = all(
        archived_gate.get(field) == rerun_gate.get(field) for field in GATE_RESULT_FIELDS
    )
    if not aggregate_fields_identical:
        raise ValueError(f"{label}: old provenance rerun aggregate differs")
    log_text = log_path.read_text(encoding="utf-8")
    if "Transcribing:" not in log_text or '"model": "/models/asr.nemo"' not in log_text:
        raise ValueError(f"{label}: old rerun execution log is incomplete")
    try:
        execution_header = json.loads(log_text.splitlines()[0])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: old rerun execution header is malformed") from exc
    retained_decoder = rerun_path.parent / "legacy-transcribe-sustained.py"
    if not retained_decoder.is_file() or sha256_file(retained_decoder) != LEGACY_DECODER_SHA256:
        raise ValueError(f"{label}: retained legacy decoder does not match locked bytes")
    retained_source_report = rerun_path.parent / "source-report.json"
    archived_report_sha256 = sha256_file(archived_path)
    if (
        not retained_source_report.is_file()
        or sha256_file(retained_source_report) != archived_report_sha256
    ):
        raise ValueError(f"{label}: retained source report does not match archived bytes")
    docker_argv = execution_header.get("docker_argv")
    if not isinstance(docker_argv, list) or not all(
        isinstance(item, str) for item in docker_argv
    ):
        raise ValueError(f"{label}: old rerun Docker argv is malformed")
    expected_argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{rerun_path.parent.resolve()}:/work",
        "--volume",
        f"{model_path.resolve()}:/models/asr.nemo:ro",
        "--volume",
        (
            f"{retained_decoder.resolve()}:"
            "/workspace/project/tools/qualification/transcribe_sustained.py:ro"
        ),
        runtime_image_id,
        "python",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/work",
        "--model",
        "/models/asr.nemo",
        "--max-wer",
        "0.5",
        "--silence-dbfs",
        "-55",
        "--max-silent-rate",
        "0",
    ]
    expected_header_fields = {
        "kind": "legacy_asr_execution",
        "schema": 2,
        "runtime_image_id": runtime_image_id,
        "image_inspect_id": runtime_image_id,
        "model_mount": "/models/asr.nemo:ro",
        "source_run_dir": str(archived_path.parent.resolve()),
        "source_report_sha256": archived_report_sha256,
        "retained_source_report": str(retained_source_report.resolve()),
        "retained_source_report_sha256": archived_report_sha256,
        "model_sha256": model_sha256,
        "container_model_sha256": model_sha256,
        "decoder_script_git_path": "HEAD:tools/qualification/transcribe_sustained.py",
        "decoder_script_git_blob": LEGACY_DECODER_GIT_BLOB,
        "decoder_script_sha256": LEGACY_DECODER_SHA256,
        "container_decoder_script_sha256": LEGACY_DECODER_SHA256,
        "retained_decoder_script": str(retained_decoder.resolve()),
        "network": "none",
        "gpu": "all",
        "ipc": "host",
        "docker_argv": expected_argv,
        "docker_argv_sha256": canonical_sha256(expected_argv),
    }
    if any(
        execution_header.get(key) != expected
        for key, expected in expected_header_fields.items()
    ):
        raise ValueError(f"{label}: old rerun execution header does not match contract")
    repo_digests = execution_header.get("image_repo_digests")
    if not isinstance(repo_digests, list) or not all(
        isinstance(value, str) for value in repo_digests
    ):
        raise ValueError(f"{label}: old rerun image-inspect evidence is malformed")
    return {
        "runtime_image_id": runtime_image_id,
        "model_mount": "/models/asr.nemo:ro",
        "model_sha256": model_sha256,
        "network": "none",
        "decoder": "nemo_greedy_max_symbols_10_graph_disabled",
        "decoder_script": {
            "git_path": "HEAD:tools/qualification/transcribe_sustained.py",
            "git_blob": LEGACY_DECODER_GIT_BLOB,
            "sha256": LEGACY_DECODER_SHA256,
        },
        "execution_contract": (
            "docker run --rm --gpus all --ipc host --network none; exact immutable "
            "runtime image; exact legacy script mounted read-only; retained run copy "
            "mounted writable; exact .nemo mounted /models/asr.nemo:ro"
        ),
        "archived_report": _report_record(archived_path),
        "rerun_report": _report_record(rerun_path),
        "execution_log": _report_record(log_path),
        "execution_header": execution_header,
        "rows": rows,
        "aggregate_fields_identical": aggregate_fields_identical,
        "harness_verdicts": {
            "archived_runtime_gate_passed": archived.get("runtime_gate_passed"),
            "archived_report_passed": archived.get("passed"),
            "legacy_rerun_runtime_gate_passed": rerun.get("runtime_gate_passed"),
            "legacy_rerun_report_passed": rerun.get("passed"),
            "comparison_scope": "recorded_not_required_old_asr_judge_isolation",
        },
        "passed": True,
    }


def build_calibration(
    pairs: list[tuple[str, Path, Path, Path, str, int, Path, Path]],
    *,
    old_model_path: Path,
    old_positive_paths: list[Path],
    new_positive_path: Path,
    policy: str,
) -> dict[str, Any]:
    if policy not in {"strict_component_v1", "decision_concordance_v2"}:
        raise ValueError(f"unknown calibration policy: {policy}")
    if not pairs:
        raise ValueError("ASR calibration corpus is empty")
    labels = [item[0] for item in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("ASR calibration run labels are duplicated")
    old_model_sha256 = sha256_file(old_model_path)
    corpus = []
    report_records = []
    new_provenance = None
    aggregate_agreement = []
    rerun_order = [item[5] for item in pairs]
    if sorted(rerun_order) != list(range(1, len(pairs) + 1)):
        raise ValueError("rerun corpus order must be a complete one-based sequence")
    for (
        label,
        old_path,
        new_path,
        rerun_path,
        old_runtime_image_id,
        rerun_corpus_order,
        old_rerun_path,
        old_rerun_log,
    ) in pairs:
        if not IMAGE_ID_PATTERN.fullmatch(old_runtime_image_id):
            raise ValueError(f"{label}: old runtime image ID is not immutable")
        old_report, new_report, rerun_report = map(_read, (old_path, new_path, rerun_path))
        old_responses = old_report.get("responses")
        new_responses = new_report.get("responses")
        rerun_responses = rerun_report.get("responses")
        response_sets: Sequence[Any] = (old_responses, new_responses, rerun_responses)
        if not all(isinstance(value, list) for value in response_sets):
            raise ValueError(f"{label}: report responses are malformed")
        if not (len(old_responses) == len(new_responses) == len(rerun_responses)):
            raise ValueError(f"{label}: response cardinality changed between judges")
        if not old_responses:
            raise ValueError(f"{label}: response corpus is empty")
        current_provenance = _evaluator_provenance(new_report.get("external_asr"))
        rerun_provenance = _evaluator_provenance(rerun_report.get("external_asr"))
        if current_provenance != rerun_provenance:
            raise ValueError(f"{label}: evaluator provenance changed on rerun")
        if new_provenance is None:
            new_provenance = current_provenance
        elif current_provenance != new_provenance:
            raise ValueError(f"{label}: new evaluator provenance differs across corpora")
        report_records.append(
            {
                "label": label,
                "old": _report_record(old_path),
                "new": _report_record(new_path),
                "new_reverse_order_rerun": _report_record(rerun_path),
                "old_runtime_image_id": old_runtime_image_id,
                "rerun_corpus_order": rerun_corpus_order,
                "old_judge_rerun_attestation": _old_rerun_attestation(
                    label=label,
                    archived_path=old_path,
                    rerun_path=old_rerun_path,
                    log_path=old_rerun_log,
                    runtime_image_id=old_runtime_image_id,
                    model_path=old_model_path,
                    model_sha256=old_model_sha256,
                ),
            }
        )
        old_run_records = []
        new_run_records = []
        rerun_run_records = []
        for index, (old_response, new_response, rerun_response) in enumerate(
            zip(old_responses, new_responses, rerun_responses, strict=True), 1
        ):
            if old_response.get("text") != new_response.get("text"):
                raise ValueError(f"{label} turn {index}: reference text changed")
            if old_response.get("expected_response_any") != new_response.get(
                "expected_response_any"
            ) or old_response.get("external_asr", {}).get(
                "expected_response_none", []
            ) != new_response.get("external_asr", {}).get("expected_response_none", []):
                raise ValueError(f"{label} turn {index}: semantic contract changed")
            if rerun_response.get("text") != new_response.get("text"):
                raise ValueError(f"{label} turn {index}: rerun reference text changed")
            if rerun_response.get("expected_response_any") != new_response.get(
                "expected_response_any"
            ) or rerun_response.get("external_asr", {}).get(
                "expected_response_none", []
            ) != new_response.get("external_asr", {}).get("expected_response_none", []):
                raise ValueError(f"{label} turn {index}: rerun semantic contract changed")
            if rerun_response.get("audio_path") != new_response.get("audio_path"):
                raise ValueError(f"{label} turn {index}: rerun source WAV changed")
            old = _asr_record(old_response, include_tokens=False)
            new = _asr_record(new_response, include_tokens=True)
            rerun = _asr_record(rerun_response, include_tokens=True)
            old_run_records.append(old)
            new_run_records.append(new)
            rerun_run_records.append(rerun)
            if new["classification"] == "speech" and (
                rerun_response.get("external_asr", {}).get("source_audio_sha256")
                != new_response.get("external_asr", {}).get("source_audio_sha256")
                or rerun_response.get("external_asr", {}).get("canonical_wav_sha256")
                != new_response.get("external_asr", {}).get("canonical_wav_sha256")
            ):
                raise ValueError(f"{label} turn {index}: rerun WAV identity changed")
            if new["classification"] == "near_silent" and (
                rerun_response.get("external_asr", {}).get("source_audio_sha256")
                != new_response.get("external_asr", {}).get("source_audio_sha256")
            ):
                raise ValueError(f"{label} turn {index}: rerun source WAV identity changed")
            determinism = {
                "transcript_identical": rerun["transcript"] == new["transcript"],
                "token_ids_identical": rerun["token_ids"] == new["token_ids"],
                "rerun_corpus_order": rerun_corpus_order,
            }
            agreement = {
                field: old[field] == new[field]
                for field in (*DECISION_FIELDS, "wer_within_threshold")
            }
            row = {
                "run": label,
                "turn": index,
                "wav": _wav_record(old_response, new_response),
                "reference": old_response.get("text"),
                "expected_response_any": old_response.get("expected_response_any") or [],
                "expected_response_none": old_response.get("external_asr", {}).get(
                    "expected_response_none", []
                ),
                "old": old,
                "new": new,
                "new_reverse_order_rerun": {
                    "transcript": rerun["transcript"],
                    "token_ids": rerun["token_ids"],
                    **determinism,
                },
                "agreement": agreement,
                "wer_threshold_flipped": not agreement["wer_within_threshold"],
                "wer_flip_is_non_load_bearing": _masked_wer_flip(old, new),
            }
            corpus.append(row)
        old_aggregate = _rederive_asr_gate(old_report, old_run_records)
        new_aggregate = _rederive_asr_gate(new_report, new_run_records)
        rerun_aggregate = _rederive_asr_gate(rerun_report, rerun_run_records)
        if rerun_aggregate != new_aggregate:
            raise ValueError(f"{label}: new rerun aggregate changed")
        aggregate_agreement.append(
            {
                "run": label,
                "old": old_aggregate,
                "new": new_aggregate,
                "external_asr_passed_identical": old_aggregate["passed"]
                == new_aggregate["passed"],
                "runtime_gate_passed_identical": old_aggregate["runtime_gate_passed"]
                == new_aggregate["runtime_gate_passed"],
                "report_passed_identical": old_aggregate["report_passed"]
                == new_aggregate["report_passed"],
                "gate_config_identical": (
                    old_aggregate["max_wer"] == new_aggregate["max_wer"]
                    and old_aggregate["qualified_fp32_base_rate"]
                    == new_aggregate["qualified_fp32_base_rate"]
                ),
            }
        )

    if not isinstance(new_provenance, dict):
        raise ValueError("new evaluator provenance is missing")
    if not IMAGE_ID_PATTERN.fullmatch(str(new_provenance.get("evaluator_image_id", ""))):
        raise ValueError("new evaluator image ID is not immutable")
    positive_controls, controls_pass = _positive_controls(
        old_positive_paths,
        new_positive_path,
        old_model_sha256=old_model_sha256,
        new_provenance=new_provenance,
    )
    silence_control = _silence_control()
    decision_flips = [
        {"run": row["run"], "turn": row["turn"], "field": field}
        for row in corpus
        for field in DECISION_FIELDS
        if not row["agreement"][field]
    ]
    wer_flips = [
        {
            "run": row["run"],
            "turn": row["turn"],
            "old_wer": row["old"]["wer"],
            "new_wer": row["new"]["wer"],
            "non_load_bearing": row["wer_flip_is_non_load_bearing"],
        }
        for row in corpus
        if row["wer_threshold_flipped"]
    ]
    deterministic = all(
        row["new_reverse_order_rerun"]["transcript_identical"]
        and row["new_reverse_order_rerun"]["token_ids_identical"]
        for row in corpus
    )
    aggregate_identical = all(
        item["external_asr_passed_identical"]
        and item["runtime_gate_passed_identical"]
        and item["report_passed_identical"]
        and item["gate_config_identical"]
        for item in aggregate_agreement
    )
    strict_passed = (
        not decision_flips
        and not wer_flips
        and aggregate_identical
        and deterministic
        and controls_pass
        and silence_control["passed"]
    )
    v2_passed = (
        not decision_flips
        and aggregate_identical
        and deterministic
        and controls_pass
        and silence_control["passed"]
        and all(item["non_load_bearing"] for item in wer_flips)
    )
    passed = strict_passed if policy == "strict_component_v1" else v2_passed
    report = {
        "kind": "asr_judge_swap_calibration",
        "schema": 1,
        "policy": policy,
        "policy_contract": {
            "strict_component_v1": (
                "No classification, semantic inclusion/exclusion, per-response pass, "
                "aggregate pass, WER-threshold, silence, or negative-control outcome may change."
            ),
            "decision_concordance_v2": (
                "No classification, semantic inclusion/exclusion, per-response pass, or "
                "aggregate pass may change. A WER-threshold change is admissible only when "
                "another unchanged false conjunct makes it non-load-bearing; all changes are "
                "disclosed. Deterministic reruns and positive speech controls must pass."
            ),
        }[policy],
        "judges": {
            "old": {
                "backend": "nemo_offline_asrmodel",
                "model_path": str(old_model_path.resolve()),
                "model_bytes": old_model_path.stat().st_size,
                "model_sha256": old_model_sha256,
            },
            "new": new_provenance,
        },
        "source_reports": report_records,
        "corpus": corpus,
        "positive_controls": positive_controls,
        "silence_control": silence_control,
        "derived": {
            "response_count": len(corpus),
            "decision_flip_count": len(decision_flips),
            "decision_flips": decision_flips,
            "wer_threshold_flip_count": len(wer_flips),
            "wer_threshold_flips": wer_flips,
            "aggregate_agreement": aggregate_agreement,
            "aggregate_verdicts_identical": aggregate_identical,
            "new_reverse_order_deterministic": deterministic,
            "positive_controls_passed": controls_pass,
            "silence_control_passed": silence_control["passed"],
            "strict_component_v1_passed": strict_passed,
            "decision_concordance_v2_passed": v2_passed,
        },
        "passed": passed,
        "blocking_reasons": []
        if passed
        else [
            reason
            for condition, reason in (
                (bool(decision_flips), "decision_outcome_flips"),
                (not aggregate_identical, "aggregate_verdict_flips"),
                (not deterministic, "new_evaluator_nondeterminism"),
                (not controls_pass, "positive_control_failure"),
                (
                    policy == "strict_component_v1" and bool(wer_flips),
                    "wer_threshold_outcome_flips",
                ),
                (
                    policy == "decision_concordance_v2"
                    and any(not item["non_load_bearing"] for item in wer_flips),
                    "load_bearing_wer_threshold_flips",
                ),
            )
            if condition
        ],
    }
    report["sha256"] = self_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=8,
        action="append",
        metavar=(
            "LABEL",
            "OLD_REPORT",
            "NEW_REPORT",
            "NEW_RERUN",
            "OLD_RUNTIME_IMAGE_ID",
            "RERUN_CORPUS_ORDER",
            "OLD_PROVENANCE_RERUN_REPORT",
            "OLD_PROVENANCE_RERUN_LOG",
        ),
        required=True,
    )
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--old-positive-report", type=Path, action="append", required=True)
    parser.add_argument("--new-positive-report", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("strict_component_v1", "decision_concordance_v2"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_calibration(
        [
            (
                label,
                Path(old),
                Path(new),
                Path(rerun),
                runtime_image,
                int(rerun_order),
                Path(old_rerun),
                Path(old_log),
            )
            for (
                label,
                old,
                new,
                rerun,
                runtime_image,
                rerun_order,
                old_rerun,
                old_log,
            ) in args.pair
        ],
        old_model_path=args.old_model,
        old_positive_paths=args.old_positive_report,
        new_positive_path=args.new_positive_report,
        policy=args.policy,
    )
    verify_self_hash(report)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
