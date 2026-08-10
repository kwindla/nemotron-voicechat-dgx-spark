#!/usr/bin/env python3
"""Run the frozen c014 voice/direct function-logit positive control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import run_live_suite as live

from nemotron_voicechat_runtime.artifacts import atomic_json, load_config
from nemotron_voicechat_runtime.semantic_corpus import corpus_sha256
from nemotron_voicechat_runtime.voice_function_fixture import (
    FIXTURE_CASE_ID,
    FIXTURE_MANIFEST_NAME,
    validate_fixture_source_provenance,
    validate_voice_function_fixture,
)
from nemotron_voicechat_runtime.voice_function_probe import (
    VOICE_RESPONSE_FRAME_LIMIT,
    VOICE_RESPONSE_FRAME_SECONDS,
    _trace_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VOICE_TRACE_PHASES = ("voice_input", "settlement", "eou_boundary", "post_bos")
EXPECTED_CALL = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def fixture_command(args: argparse.Namespace, *, image_id: str, output: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
        "-v",
        f"{args.corpus.resolve()}:/qualification-input/corpus.json:ro",
        "-v",
        f"{output.resolve()}:/qualification",
        "-e",
        "PYTHONPATH=/workspace/project/src",
        "-e",
        "HF_HOME=/models/huggingface",
        "-e",
        "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        image_id,
        "python3",
        "/workspace/project/tools/qualification/generate_voice_function_fixture.py",
        "--corpus",
        "/qualification-input/corpus.json",
        "--output",
        "/qualification",
        "--runtime-image",
        args.runtime_image,
        "--runtime-image-id",
        image_id,
    ]


def probe_command(
    args: argparse.Namespace,
    *,
    image_id: str,
    fixture: Path,
    output: Path,
    name: str,
    top_k: int,
) -> list[str]:
    if not 0 <= top_k <= 20:
        raise ValueError("voice function trace top-k must be from 0 through 20")
    environment = load_config()["runtime"]["environment"] | {
        "HF_HOME": "/models/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/models/huggingface/hub",
        "HF_MODULES_CACHE": "/tmp/voicechat-hf-modules",
        "VOICECHAT_DIRECT_POSITION_PROBE_SKIP_WARMUPS": "1",
        "VOICECHAT_NANO_PAD_PAIR": "0",
        "VOICECHAT_RUNTIME_IMAGE": args.runtime_image,
        "VOICECHAT_RUNTIME_IMAGE_ID": image_id,
        "VOICECHAT_VOICE_FUNCTION_CORPUS": "/qualification-input/corpus.json",
        "VOICECHAT_VOICE_FUNCTION_FIXTURE": "/qualification-input/fixture",
        "VOICECHAT_VOICE_FUNCTION_MAX_FRAMES": str(VOICE_RESPONSE_FRAME_LIMIT),
        "VOICECHAT_VOICE_FUNCTION_PROBE_DIR": "/qualification",
        "VOICECHAT_WEB_REALTIME_WARMUP": "0",
    }
    if top_k:
        environment["S2S_FUNCTION_LOGIT_TRACE_TOPK"] = str(top_k)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"voicechat-voice-function-{name}",
        "--init",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--network",
        "none",
        "--security-opt",
        "label=disable",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'parent'}:/models/voicechat:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'nano-skeleton'}:"
        "/models/NVIDIA-Nemotron-Nano-9B-v2:ro",
        "-v",
        f"{args.cache_root / 'huggingface'}:/models/huggingface:ro",
        "-v",
        f"{args.cache_root / 'artifacts' / 'release'}:/models/derived:ro",
        "-v",
        f"{output.resolve()}:/qualification",
        "-v",
        f"{fixture.resolve()}:/qualification-input/fixture:ro",
        "-v",
        f"{args.corpus.resolve()}:/qualification-input/corpus.json:ro",
    ]
    if args.speech_root is not None:
        command.extend(
            [
                "-v",
                f"{args.speech_root.resolve()}:/opt/Speech:ro",
                "-e",
                "GIT_CONFIG_COUNT=1",
                "-e",
                "GIT_CONFIG_KEY_0=safe.directory",
                "-e",
                "GIT_CONFIG_VALUE_0=/opt/Speech",
                "-e",
                "PYTHONPATH=/workspace/project/src:/opt/Speech",
            ]
        )
    for key, value in sorted(environment.items()):
        command.extend(("-e", f"{key}={value}"))
    command.extend(
        [
            image_id,
            "python3",
            "-m",
            "nemotron_voicechat_runtime.server",
            "--speech-root",
            "/opt/Speech",
            "--checkpoint-root",
            "/models/voicechat",
            "--hf-skeleton",
            "/models/NVIDIA-Nemotron-Nano-9B-v2",
            "--vllm-manifest",
            "/models/derived/manifests/release.json",
            "--nano-vllm-path",
            "/models/derived/nano",
            "--eartts-vllm-path",
            "/models/derived/eartts",
            "--speaker-name",
            "Aria",
        ]
    )
    return command


def _validate_trace_entries(block: dict[str, Any], report: dict[str, Any], top_k: int) -> None:
    if set(block) != {
        "top_k",
        "post_bos_frames",
        "phases",
        "watched_token_ids",
        "entries",
        "coverage",
        "summary",
    }:
        raise RuntimeError("voice function trace block has an invalid schema")
    watched = block["watched_token_ids"]
    entries = block["entries"]
    coverage = block["coverage"]
    if (
        block["top_k"] != top_k
        or block["post_bos_frames"] != 8
        or block["phases"] != list(VOICE_TRACE_PHASES)
        or not isinstance(entries, list)
        or not isinstance(coverage, list)
        or not isinstance(watched, dict)
        or set(watched) != set(live.DIRECT_SEMANTIC_FUNCTION_TRACE_WATCHED)
    ):
        raise RuntimeError("voice function trace configuration is invalid")
    if any(
        not isinstance(entry, dict)
        or entry.get("phase") not in VOICE_TRACE_PHASES
        or entry.get("effective_token_reason") == "direct_hidden_pad"
        for entry in entries
    ):
        raise RuntimeError("voice function trace contains an invalid entry")
    if block["summary"] != _trace_summary(entries, watched):
        raise RuntimeError("voice function trace summary is contradictory")
    normalized = copy.deepcopy(entries)
    for entry in normalized:
        if entry["phase"] == "voice_input":
            entry["phase"] = "direct_position"
    fake_case = {
        "token_ids": [0]
        * sum(entry["phase"] == "direct_position" for entry in normalized),
        "settlement": {
            "model_steps": sum(entry["phase"] == "settlement" for entry in normalized)
        },
        "eou_boundary": {
            "performed": any(entry["phase"] == "eou_boundary" for entry in normalized)
        },
        "function_logit_trace": normalized,
    }
    live._validate_function_logit_trace(
        fake_case,
        "voice/c014",
        top_k=top_k,
        post_bos_frames=8,
        watched_token_ids=watched,
    )
    expected_coverage = (
        len(report["input"]["records"])
        + report["settlement"]["model_steps"]
        + 1
        + min(max(report["response_frames"] - 1, 0), 8)
    )
    if len(coverage) != expected_coverage:
        raise RuntimeError("voice function trace coverage length is inconsistent")
    fresh = [record for record in coverage if record.get("status") == "fresh_model_position"]
    gaps = [record for record in coverage if record.get("status") == "no_new_model_position"]
    if (
        len(fresh) != len(entries)
        or [record.get("model_frame") for record in fresh]
        != [entry["model_frame"] for entry in entries]
        or any(
            record.get("background_active") is not True
            or record.get("reason") != "fc_background_owned_advancement"
            or record.get("phase") != "post_bos"
            for record in gaps
        )
        or len(fresh) + len(gaps) != len(coverage)
    ):
        raise RuntimeError("voice function trace coverage is contradictory")


def validate_probe_report(
    report: dict[str, Any],
    *,
    image_id: str,
    corpus: Path,
    fixture: Path,
    top_k: int,
) -> None:
    expected_schema = 2 if top_k else 1
    if (
        report.get("schema") != expected_schema
        or report.get("kind") != "voice_function_logit_probe"
        or report.get("precision") != "w8"
        or report.get("case_id") != FIXTURE_CASE_ID
        or type(report.get("passed")) is not bool
        or type(report.get("voice_reproduction_valid")) is not bool
    ):
        raise RuntimeError("voice function probe report has an invalid identity")
    if report.get("error") is not None:
        if report["passed"] or report["voice_reproduction_valid"]:
            raise RuntimeError("voice function error report has a green verdict")
        return
    fixture_manifest, _ = validate_voice_function_fixture(
        fixture, corpus_path=corpus, expected_runtime_image_id=image_id
    )
    validate_fixture_source_provenance(fixture_manifest, project_root=REPO_ROOT)
    if report.get("fixture") != fixture_manifest:
        raise RuntimeError("voice function probe fixture provenance mismatch")
    provenance = report.get("runtime_provenance")
    if not isinstance(provenance, dict) or provenance.get("runtime_image_id") != image_id:
        raise RuntimeError("voice function probe runtime image ID mismatch")
    input_evidence = report.get("input")
    if (
        not isinstance(input_evidence, dict)
        or set(input_evidence)
        != {
            "packet_samples",
            "model_frame_samples",
            "original_samples",
            "original_pending_samples",
            "padded_samples",
            "records",
        }
        or input_evidence["packet_samples"] != 320
        or input_evidence["model_frame_samples"] != 1_280
        or input_evidence["original_samples"] != fixture_manifest["pcm"]["sample_count"]
        or input_evidence["original_pending_samples"]
        != fixture_manifest["pcm"]["sample_count"] % 1_280
        or input_evidence["padded_samples"]
        != math.ceil(fixture_manifest["pcm"]["sample_count"] / 1_280) * 1_280
        or not isinstance(input_evidence["records"], list)
        or len(input_evidence["records"]) != input_evidence["padded_samples"] // 1_280
    ):
        raise RuntimeError("voice function probe input replay evidence is contradictory")
    records = input_evidence["records"]
    if (
        [record.get("position") for record in records] != list(range(len(records)))
        or any(type(record.get("input_active")) is not bool for record in records)
        or any(not isinstance(record.get("audio"), dict) for record in records)
        or [index for index, record in enumerate(records) if record.get("padded_tail")]
        != ([len(records) - 1] if input_evidence["original_pending_samples"] else [])
    ):
        raise RuntimeError("voice function probe frame replay evidence is contradictory")
    checks = report.get("structural_checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(type(value) is not bool for value in checks.values())
    ):
        raise RuntimeError("voice function probe structural evidence is incomplete")
    sotc_evidence = report.get("sotc_evidence")
    if (
        not isinstance(sotc_evidence, dict)
        or set(sotc_evidence)
        != {"starting", "after_voice_input", "after_settlement", "after_eou_boundary", "final"}
        or not all(live._validate_sotc_evidence(value) for value in sotc_evidence.values())
    ):
        raise RuntimeError("voice function probe SOTC provenance is incomplete")
    for counter in (
        "pre_eou_suppressed_tokens",
        "client_eou_sotc_committed_count",
        "post_fc_client_bos_forced_count",
    ):
        values = [
            sotc_evidence[phase][counter]
            for phase in (
                "starting",
                "after_voice_input",
                "after_settlement",
                "after_eou_boundary",
                "final",
            )
        ]
        if values != sorted(values):
            raise RuntimeError("voice function probe SOTC counters are not monotonic")
    derived_structural = all(checks.values())
    if report.get("structural_passed") is not derived_structural:
        raise RuntimeError("voice function probe structural verdict is contradictory")
    if report.get("passed") is not (
        report["voice_reproduction_valid"] and derived_structural
    ):
        raise RuntimeError("voice function probe top-level verdict is contradictory")
    if (
        not isinstance(report.get("response_audio_samples"), int)
        or isinstance(report.get("response_audio_samples"), bool)
        or report["response_audio_samples"] < 0
        or not 1 <= report.get("response_frames", 0) <= VOICE_RESPONSE_FRAME_LIMIT
        or report.get("quiescence_required") != 2
    ):
        raise RuntimeError("voice function probe has invalid bounded response evidence")
    pacing = report.get("response_pacing")
    if (
        not isinstance(pacing, dict)
        or set(pacing)
        != {
            "clock",
            "frame_seconds",
            "budget_frames",
            "observed_frames",
            "paced_transitions",
        }
        or pacing["clock"] != "monotonic_deadline"
        or pacing["frame_seconds"] != VOICE_RESPONSE_FRAME_SECONDS
        or pacing["budget_frames"] != VOICE_RESPONSE_FRAME_LIMIT
        or pacing["observed_frames"] != report["response_frames"]
        or pacing["paced_transitions"] != max(0, report["response_frames"] - 1)
        or checks.get("response_frame_clock_exact") is not True
    ):
        raise RuntimeError("voice function probe response pacing evidence is contradictory")
    quiescence = report.get("quiescence")
    if not isinstance(quiescence, list) or len(quiescence) != report["response_frames"]:
        raise RuntimeError("voice function probe quiescence evidence is incomplete")
    final = quiescence[-1]
    if report["voice_reproduction_valid"]:
        if (
            report.get("tool_calls") != EXPECTED_CALL
            or report.get("failure") is not None
            or report.get("carrier", {}).get("passed") is not True
            or report.get("semantic", {}).get("passed") is not True
            or report["response_audio_samples"] <= 0
            or not (
                final.get("quiescent") is True
                and final.get("background_absent") is True
                and final.get("active") is False
                and final.get("awaiting_response") is False
                and final.get("injecting_response") is False
                and final.get("forced_tokens") == 0
                and final.get("completed_calls") == 1
                and final.get("quiescence_streak", 0) >= 2
                and any(
                    record.get("boundary_event") == "end"
                    and record.get("quiescent") is True
                    for record in quiescence
                )
            )
        ):
            raise RuntimeError("voice function probe accepted an incomplete function cycle")
    else:
        failure = report.get("failure")
        if (
            report["passed"]
            or not isinstance(failure, dict)
            or failure.get("code") != "voice_reproduction_invalid"
            or not isinstance(failure.get("message"), str)
        ):
            raise RuntimeError("red voice function probe lacks bounded failure evidence")
    block = report.get("function_logit_trace")
    if top_k:
        if not isinstance(block, dict):
            raise RuntimeError("traced voice function report has no logit evidence")
        _validate_trace_entries(block, report, top_k)
    elif block is not None:
        raise RuntimeError("untraced voice function report contains logit evidence")


def run_probe_stage(
    command: list[str],
    *,
    log: Path,
    report_path: Path,
    image_id: str,
    corpus: Path,
    fixture: Path,
    top_k: int,
) -> tuple[dict[str, Any], int]:
    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=3600,
        )
    if not report_path.is_file():
        raise RuntimeError(f"voice function probe exited {completed.returncode} without a report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_probe_report(
        report,
        image_id=image_id,
        corpus=corpus,
        fixture=fixture,
        top_k=top_k,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"voice function probe exited unexpectedly: {completed.returncode}")
    if completed.returncode == 0 and report["passed"] is not True:
        raise RuntimeError("voice function probe exited zero with a red report")
    if completed.returncode == 1 and report["passed"] is not False:
        raise RuntimeError("voice function probe exited nonzero with a green report")
    return report, completed.returncode


def _normalize_voice_report(report: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(report)
    schema = normalized.get("schema")
    if schema not in {1, 2}:
        raise RuntimeError("voice function neutrality requires schema 1/2")
    normalized["schema"] = 1
    trace_keys = 0

    def visit(value: Any) -> None:
        nonlocal trace_keys
        if isinstance(value, dict):
            if "function_logit_trace" in value:
                trace_keys += 1
                value.pop("function_logit_trace")
            for key, child in list(value.items()):
                if key == "response_audio_path" and isinstance(child, str):
                    value[key] = Path(child).name
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    if (schema == 2 and trace_keys < 1) or (schema == 1 and trace_keys):
        raise RuntimeError("voice function trace/schema discrimination failed")
    checkpoint = normalized.get("runtime_provenance", {}).get("checkpoint", {})
    fixed = checkpoint.get("fixed_stage_optimizations", {}) if isinstance(checkpoint, dict) else {}
    codec = fixed.get("cpu_codec", {}) if isinstance(fixed, dict) else {}
    if isinstance(codec, dict):
        codec.pop("warmup_worker_ms", None)
        worker = codec.get("worker")
        if isinstance(worker, dict):
            worker.pop("pid", None)
    # These fields are validated fail-closed within each report above, but their
    # values describe wall-clock scheduling and the nondeterministic observation
    # point at which FC background work becomes quiescent.  They are retained in
    # the source reports and summarized by voice_neutrality(), not deep-compared
    # as categorical model output.
    normalized.pop("quiescence", None)
    normalized.pop("response_frames", None)
    pacing = normalized.get("response_pacing")
    if isinstance(pacing, dict):
        pacing.pop("observed_frames", None)
        pacing.pop("paced_transitions", None)
    eou = normalized.get("eou_boundary")
    if isinstance(eou, dict):
        eou.pop("engine_frame", None)
        boundary = eou.get("response_boundary")
        if isinstance(boundary, dict):
            boundary.pop("bos_frame", None)
            boundary.pop("text_eos_frame", None)
    sotc_evidence = normalized.get("sotc_evidence")
    if isinstance(sotc_evidence, dict):
        for phase in sotc_evidence.values():
            if isinstance(phase, dict):
                for key in [name for name in phase if name.endswith("_frame")]:
                    phase.pop(key)
    settlement = normalized.get("settlement")
    steps = settlement.get("steps") if isinstance(settlement, dict) else None
    if isinstance(steps, list) and steps:
        relative_paths = (
            ("frame_index",),
            ("audio_frame_index_after",),
            ("perception_frame_idx_after",),
            ("vllm_request_positions", "eartts", "generated_tokens"),
            ("vllm_request_positions", "eartts", "session_positions"),
            ("vllm_request_positions", "nano", "generated_tokens"),
            ("vllm_request_positions", "nano", "session_positions"),
        )

        def nested_value(value: Any, path: tuple[str, ...]) -> Any:
            for key in path:
                if not isinstance(value, dict):
                    return None
                value = value.get(key)
            return value

        def set_nested(value: Any, path: tuple[str, ...], replacement: int) -> None:
            for key in path[:-1]:
                value = value[key]
            value[path[-1]] = replacement

        baselines = {path: nested_value(steps[0], path) for path in relative_paths}
        for step in steps:
            function_calling = step.get("function_calling") if isinstance(step, dict) else None
            if isinstance(function_calling, dict):
                for key in [name for name in function_calling if name.endswith("_frame")]:
                    function_calling.pop(key)
            for path, baseline in baselines.items():
                value = nested_value(step, path)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and isinstance(baseline, int)
                    and not isinstance(baseline, bool)
                ):
                    set_nested(step, path, value - baseline)
    return normalized


def voice_neutrality(
    control_a: dict[str, Any], traced: dict[str, Any], control_b: dict[str, Any]
) -> dict[str, Any]:
    if [control_a.get("schema"), traced.get("schema"), control_b.get("schema")] != [1, 2, 1]:
        raise RuntimeError("voice function neutrality requires schema 1/2/1 ABA")
    evidence: dict[str, Any] = {
        "kind": "voice_function_trace_neutrality",
        "method": "contemporaneous_off_on_off_observed_null",
        "stable_field_count": 0,
        "control_nondeterministic_field_count": 0,
        "control_nondeterministic_fields": [],
        "violations": [],
    }
    error_runs = [
        name
        for name, report in (("off-a", control_a), ("on", traced), ("off-b", control_b))
        if report.get("error") is not None
    ]
    if error_runs:
        evidence["discrete_outcome_equal"] = False
        evidence["error_runs"] = error_runs
        evidence["violations"].append(
            {"path": "runs", "reason": "voice_probe_error", "runs": error_runs}
        )
        evidence["passed"] = False
        return evidence
    normalized = [_normalize_voice_report(item) for item in (control_a, traced, control_b)]
    evidence["operational_timing"] = {
        name: {
            "response_frames": report.get("response_frames"),
            "paced_transitions": (report.get("response_pacing") or {}).get(
                "paced_transitions"
            ),
            "quiescence_records": len(report.get("quiescence") or []),
        }
        for name, report in (
            ("control_a", control_a),
            ("traced", traced),
            ("control_b", control_b),
        )
    }
    audio = [item.pop("response_audio_samples") for item in normalized]
    live._classify_trace_value(*normalized, path=(), evidence=evidence)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in audio
    ):
        evidence["violations"].append({"path": "response_audio_samples", "reason": "invalid"})
    elif any(value % live.EARTTS_FRAME_SAMPLES for value in audio):
        evidence["violations"].append(
            {"path": "response_audio_samples", "reason": "not_eartts_frame_quantized"}
        )
    elif not min(audio[0], audio[2]) <= audio[1] <= max(audio[0], audio[2]):
        evidence["violations"].append(
            {"path": "response_audio_samples", "reason": "outside_control_envelope"}
        )
    elif abs(audio[1] - audio[0]) > abs(audio[2] - audio[0]):
        evidence["violations"].append(
            {"path": "response_audio_samples", "reason": "trace_delta_exceeds_control_delta"}
        )
    evidence["audio_samples"] = {"control_a": audio[0], "traced": audio[1], "control_b": audio[2]}
    evidence["discrete_outcome_equal"] = all(
        item.get("tool_calls") == EXPECTED_CALL
        and item.get("voice_reproduction_valid") is True
        and item.get("passed") is True
        for item in (control_a, traced, control_b)
    )
    if not evidence["discrete_outcome_equal"]:
        evidence["violations"].append(
            {"path": "c014", "reason": "voice_positive_control_not_stable"}
        )
    evidence["passed"] = not evidence["violations"]
    return evidence


def _margin_summary(entries: list[dict[str, Any]], watched: dict[str, int]) -> dict[str, Any]:
    selected = [entry for entry in entries if entry["phase"] in {"settlement", "eou_boundary"}]
    return {
        "phases": ["settlement", "eou_boundary"],
        "entry_count": len(selected),
        "min_sotc_rank": min((entry["sotc_rank"] for entry in selected), default=None),
        "max_sotc_minus_pad_logit": max(
            (
                entry["watched"]["sotc"]["logit"]
                - entry["watched"]["pad"]["logit"]
                for entry in selected
            ),
            default=None,
        ),
        "raw_sotc_argmax_count": sum(
            entry["raw_function_token_id"] == watched["sotc"] for entry in selected
        ),
    }


def rederive_comparison(
    source_diagnostic_path: Path,
    *,
    direct_diagnostic_path: Path,
    corpus: Path,
    output: Path,
) -> dict[str, Any]:
    """Recompute only the derived voice/direct comparison from retained evidence."""

    source = json.loads(source_diagnostic_path.read_text(encoding="utf-8"))
    if (
        source.get("schema") != 1
        or source.get("kind") != "voice_function_logit_diagnostic"
        or source.get("passed") is not True
        or source.get("neutrality", {}).get("passed") is not True
        or source.get("returncodes") != {"off-a": 0, "on": 0, "off-b": 0}
        or set(source.get("reports", {})) != {"off-a", "on", "off-b"}
        or source.get("corpus_sha256") != corpus_sha256(corpus)
    ):
        raise RuntimeError("source voice diagnostic is not a valid retained GREEN run")
    image_id = source.get("runtime_image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeError("source voice diagnostic has no immutable runtime image ID")
    voice_path = Path(source["reports"]["on"])
    voice = json.loads(voice_path.read_text(encoding="utf-8"))
    fixture = Path(source["fixture_manifest"]).parent
    validate_probe_report(
        voice,
        image_id=image_id,
        corpus=corpus,
        fixture=fixture,
        top_k=source["top_k"],
    )
    direct = json.loads(direct_diagnostic_path.read_text(encoding="utf-8"))
    comparison = compare_direct_voice(direct, voice, corpus=corpus, image_id=image_id)
    result = {
        "schema": 1,
        "kind": "rederived_direct_voice_function_logit_comparison",
        "passed": comparison["passed"],
        "source_voice_diagnostic": str(source_diagnostic_path.resolve()),
        "source_voice_diagnostic_sha256": hashlib.sha256(
            source_diagnostic_path.read_bytes()
        ).hexdigest(),
        "source_voice_report": str(voice_path.resolve()),
        "source_voice_report_sha256": hashlib.sha256(voice_path.read_bytes()).hexdigest(),
        "source_direct_diagnostic": str(direct_diagnostic_path.resolve()),
        "source_direct_diagnostic_sha256": hashlib.sha256(
            direct_diagnostic_path.read_bytes()
        ).hexdigest(),
        "runtime_image_id": image_id,
        "corpus_sha256": corpus_sha256(corpus),
        "comparison": comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def compare_direct_voice(
    direct_diagnostic: dict[str, Any], voice_traced: dict[str, Any], *, corpus: Path, image_id: str
) -> dict[str, Any]:
    if (
        direct_diagnostic.get("kind") != "direct_function_logit_diagnostic"
        or direct_diagnostic.get("runtime_image_id") != image_id
        or direct_diagnostic.get("corpus_sha256") != corpus_sha256(corpus)
    ):
        raise RuntimeError("direct function diagnostic identity mismatch")
    direct_report_path = Path(direct_diagnostic["reports"]["on"])
    direct = json.loads(direct_report_path.read_text(encoding="utf-8"))
    live.validate_direct_semantic_probe_report(
        direct,
        precision="w8",
        runtime_image_id=image_id,
        corpus_path=corpus,
        expected_token_forms=("text",),
        expected_trace_top_k=direct_diagnostic["top_k"],
    )
    direct_case = next(
        case
        for lane in direct["lanes"]
        for case in lane["cases"]
        if lane["token_form"] == "text" and case["case_id"] == FIXTURE_CASE_ID
    )
    direct_watched = direct["function_logit_trace"]["watched_token_ids"]
    direct_summary = _margin_summary(direct_case["function_logit_trace"], direct_watched)
    voice_block = voice_traced["function_logit_trace"]
    voice_entries = voice_block["entries"]
    voice_watched = voice_block["watched_token_ids"]
    if voice_watched != direct_watched:
        raise RuntimeError("voice/direct watched function token IDs differ")
    # Compare the same decision surface on both paths. Voice-input and post-BOS
    # positions are useful trace context, but the direct probe has no acoustic
    # input phase and must not be contrasted against those extra positions.
    voice_summary = _margin_summary(voice_entries, voice_watched)
    direct_rank_two = bool(
        direct_summary["entry_count"]
        and direct_summary["min_sotc_rank"] == 2
        and direct_summary["max_sotc_minus_pad_logit"] < 0
        and direct_summary["raw_sotc_argmax_count"] == 0
        and direct_case["tool_calls"] == []
    )
    voice_positive = bool(
        voice_traced["tool_calls"] == EXPECTED_CALL
        and voice_summary["raw_sotc_argmax_count"] >= 1
        and voice_summary["max_sotc_minus_pad_logit"] >= 0
    )
    direct_sotc = direct_case["eou_boundary"]["sotc_final"]
    voice_sotc = voice_traced["sotc_evidence"]["final"]
    sotc_provenance = {
        "direct": {
            key: direct_sotc[key]
            for key in (
                "pre_eou_suppressed_tokens",
                "client_eou_sotc_committed_count",
                "post_fc_client_bos_forced_count",
            )
        },
        "voice": {
            key: voice_sotc[key]
            for key in (
                "pre_eou_suppressed_tokens",
                "client_eou_sotc_committed_count",
                "post_fc_client_bos_forced_count",
            )
        },
    }
    return {
        "kind": "direct_voice_function_logit_comparison",
        "direct": direct_summary,
        "voice": voice_summary,
        "sotc_provenance": sotc_provenance,
        "direct_rank_two_pad_argmax": direct_rank_two,
        "voice_sotc_argmax_and_expected_call": voice_positive,
        "conclusion": (
            "acoustic_conditioning_drives_sotc; direct_text_as_implemented_is_distribution_shifted"
            if direct_rank_two and voice_positive
            else "inconclusive"
        ),
        "passed": direct_rank_two and voice_positive,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_root = args.cache_root.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()
    args.direct_diagnostic = args.direct_diagnostic.expanduser().resolve()
    if args.speech_root is not None:
        args.speech_root = args.speech_root.expanduser().resolve()
    if not 1 <= args.top_k <= 20:
        raise ValueError("--top-k must be from 1 through 20")
    image_id = live.runtime_image_id(args.runtime_image)
    live.run_checked(
        live.direct_semantic_image_preflight_command(image_id),
        timeout=60,
        log=args.output / "image-preflight.log",
    )
    fixture = args.output / "fixture"
    fixture.mkdir()
    live.run_checked(
        fixture_command(args, image_id=image_id, output=fixture),
        timeout=600,
        log=args.output / "fixture.log",
    )
    fixture_manifest, _ = validate_voice_function_fixture(
        fixture, corpus_path=args.corpus, expected_runtime_image_id=image_id
    )
    validate_fixture_source_provenance(fixture_manifest, project_root=REPO_ROOT)

    reports: dict[str, dict[str, Any]] = {}
    returncodes: dict[str, int] = {}
    for name, top_k in (("off-a", 0), ("on", args.top_k), ("off-b", 0)):
        output = args.output / name
        output.mkdir()
        report, returncode = run_probe_stage(
            probe_command(
                args,
                image_id=image_id,
                fixture=fixture,
                output=output,
                name=name,
                top_k=top_k,
            ),
            log=args.output / f"{name}.log",
            report_path=output / "report.json",
            image_id=image_id,
            corpus=args.corpus,
            fixture=fixture,
            top_k=top_k,
        )
        reports[name] = report
        returncodes[name] = returncode
    neutrality = voice_neutrality(reports["off-a"], reports["on"], reports["off-b"])
    direct_diagnostic = json.loads(args.direct_diagnostic.read_text(encoding="utf-8"))
    if (
        reports["on"].get("voice_reproduction_valid") is True
        and isinstance(reports["on"].get("function_logit_trace"), dict)
    ):
        comparison = compare_direct_voice(
            direct_diagnostic, reports["on"], corpus=args.corpus, image_id=image_id
        )
    else:
        comparison = {
            "kind": "direct_voice_function_logit_comparison",
            "passed": False,
            "conclusion": "inconclusive",
            "reason": "voice_reproduction_invalid",
        }
    result = {
        "schema": 1,
        "kind": "voice_function_logit_diagnostic",
        "passed": neutrality["passed"] and comparison["passed"],
        "runtime_image_id": image_id,
        "corpus_sha256": corpus_sha256(args.corpus),
        "fixture_manifest": str((fixture / FIXTURE_MANIFEST_NAME).resolve()),
        "fixture_manifest_sha256": hashlib.sha256(
            (fixture / FIXTURE_MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "top_k": args.top_k,
        "returncodes": returncodes,
        "neutrality": neutrality,
        "comparison": comparison,
        "reports": {
            name: str((args.output / name / "report.json").resolve()) for name in reports
        },
    }
    atomic_json(args.output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image")
    parser.add_argument("--speech-root", type=Path)
    parser.add_argument("--direct-diagnostic", type=Path, required=True)
    parser.add_argument(
        "--rederive-from",
        type=Path,
        help="rederive a corrected comparison from a retained GREEN voice diagnostic",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "tools/qualification/direct_text_semantic_corpus.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.rederive_from is not None:
            report = rederive_comparison(
                args.rederive_from,
                direct_diagnostic_path=args.direct_diagnostic,
                corpus=args.corpus,
                output=args.output,
            )
        else:
            missing = [
                name
                for name in ("cache_root", "runtime_image", "speech_root")
                if getattr(args, name) is None
            ]
            if missing:
                parser.error(
                    "full diagnostic mode requires "
                    + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
                )
            report = run(args)
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        failure_path = (
            args.output if args.rederive_from is not None else args.output / "report.json"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(failure_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
