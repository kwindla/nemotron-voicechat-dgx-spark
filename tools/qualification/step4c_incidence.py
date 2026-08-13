#!/usr/bin/env python3
"""Analyze preregistered Step 4c blocks with run-block uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import stratified_latency_analyzer as qualified
from step4c_contract import (
    BROWSER_CONNECTED_WINDOW_TOLERANCE_SECONDS,
    DRAIN_SECONDS,
    DURATION_SECONDS,
    EXPECTED_SOURCE_FRAMES,
    L1_SHA256,
    L1_TEXT,
    QUALIFICATION_FIXTURE,
    WARMUP_SECONDS,
)

SCHEMA = "nemotron_voicechat.step4c_incidence.v1"
MANIFEST_SCHEMA = "nemotron_voicechat.step4c_campaign.v1"
SCHEDULE = (
    "direct",
    "pipecat",
    "browser",
    "pipecat",
    "browser",
    "direct",
    "browser",
    "direct",
    "pipecat",
)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_260_813

INVALID_REASON_CODES = frozenset(
    {
        "HEALTH_NOT_EXCLUSIVE",
        "TELEMETRY_MISSING",
        "FIXTURE_TIMING_MISMATCH",
        "FIXTURE_IDENTITY_MISMATCH",
        "CLIENT_REPORT_FAILED",
        "RESPONSE_CARDINALITY_INVALID",
        "RESPONSE_CANCELLED",
        "DELIVERED_AUDIO_EMPTY",
        "RESPONSE_TARGET_MISMATCH",
        "TEARDOWN_NOT_CLEAN",
        "TRANSPORT_FRAMES_NONCONTIGUOUS",
        "ACOUSTIC_INPUT_NOT_EXACT_ZERO",
        "PLAYOUT_TRACE_NOT_PUBLISHED",
        "PLAYOUT_TRACE_INVALID",
        "PLAYOUT_TRACE_IDENTITY_MISMATCH",
        "BROWSER_ARTIFACT_INVALID",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _container_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize Docker's nondeterministically ordered Mounts array."""

    normalized = dict(value)
    mounts = normalized.get("Mounts")
    if isinstance(mounts, list):
        normalized["Mounts"] = sorted(
            mounts,
            key=lambda item: (
                str(item.get("Destination", "")) if isinstance(item, dict) else "",
                str(item.get("Source", "")) if isinstance(item, dict) else "",
            ),
        )
    return normalized


def response_target_matches(output_text: str) -> bool:
    """Require L1 content while allowing punctuation and minor spoken connective drift."""

    words = re.findall(r"[a-z0-9]+", output_text.lower())
    target = re.findall(r"[a-z0-9]+", L1_TEXT.lower())
    tail = target[-8:]
    tail_cursor = iter(words)
    tail_present = all(any(word == candidate for candidate in tail_cursor) for word in tail)
    return tail_present and SequenceMatcher(a=target, b=words, autojunk=False).ratio() >= 0.90


def _membership(report: Mapping[str, Any], population: str) -> dict[str, Any]:
    try:
        membership = report["populations"][population]["membership"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"analyzer lacks {population} membership") from exc
    if not isinstance(membership, dict) or not isinstance(membership.get("members"), list):
        raise ValueError(f"analyzer {population} membership is malformed")
    return membership


def _member_key(member: Mapping[str, Any]) -> str:
    return json.dumps(member, sort_keys=True, separators=(",", ":"))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_interval(
    rates_by_transport: Mapping[str, Sequence[float]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample whole blocks within transport and average blocks equally."""

    if iterations <= 0 or any(not values for values in rates_by_transport.values()):
        raise ValueError("bootstrap requires positive iterations and nonempty transport blocks")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        resampled = [
            values[rng.randrange(len(values))]
            for values in rates_by_transport.values()
            for _ in range(len(values))
        ]
        samples.append(sum(resampled) / len(resampled))
    observed = [rate for values in rates_by_transport.values() for rate in values]
    return {
        "estimand": "equal valid-run-block mean EarTTS-high incidence",
        "mean": sum(observed) / len(observed),
        "interval_95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "run block, stratified by assigned transport",
    }


def _failure(failures: set[str], condition: bool, code: str) -> None:
    if not condition:
        failures.add(code)


def _has_expected_source_frames(report: Mapping[str, Any]) -> bool:
    source_frames = report.get("source_frames")
    return (
        isinstance(source_frames, int)
        and not isinstance(source_frames, bool)
        and source_frames == EXPECTED_SOURCE_FRAMES
    )


def _validate_legacy_direct_timing_evidence(
    block: Mapping[str, Any], report: Mapping[str, Any], session_id: str
) -> bool:
    """Validate block 01's explicit derived timing sidecar without inventing a default."""

    evidence_path = block.get("timing_evidence")
    client_trace_path = block.get("client_trace")
    if not isinstance(evidence_path, str) or not isinstance(client_trace_path, str):
        return False
    evidence = _read_json(Path(evidence_path))
    if evidence != {
        "schema": "nemotron_voicechat.step4c_timing_evidence.v1",
        "session_id": session_id,
        "warmup_seconds": WARMUP_SECONDS,
        "duration_seconds": DURATION_SECONDS,
        "drain_seconds": DRAIN_SECONDS,
        "derivation": "client first metric to completed session.close",
    }:
        return False
    events = qualified.read_jsonl(Path(client_trace_path))
    metrics = [event for event in events if event.get("type") == "voicechat.metrics"]
    closed = [event for event in events if event.get("type") == "session.closed"]
    typed = [event for event in events if event.get("type") == "input_audio_buffer.speech_started"]
    if not metrics or len(closed) != 1 or len(typed) != 1:
        return False
    first = metrics[0].get("client_received_monotonic_s")
    terminal = closed[0].get("client_received_monotonic_s")
    if not isinstance(first, (int, float)) or not isinstance(terminal, (int, float)):
        return False
    connected_seconds = float(terminal) - float(first)
    return (
        _has_expected_source_frames(report)
        and closed[0].get("session_id") == session_id
        and closed[0].get("status") == "completed"
        and DURATION_SECONDS + DRAIN_SECONDS <= connected_seconds < 136.0
    )


def _client_validity_failures(
    transport: str,
    report: Mapping[str, Any],
    session_id: str,
    block: Mapping[str, Any],
) -> set[str]:
    """Adjudicate transport evidence while keeping failed attempts readable."""

    failures: set[str] = set()
    if transport in {"direct", "pipecat"}:
        _failure(
            failures,
            report.get("qualification_fixture") == QUALIFICATION_FIXTURE
            and report.get("prompt_sha256") == L1_SHA256,
            "FIXTURE_IDENTITY_MISMATCH",
        )
        explicit_timing = (
            report.get("warmup_seconds") == WARMUP_SECONDS
            and report.get("duration_seconds") == DURATION_SECONDS
            and report.get("drain_seconds") == DRAIN_SECONDS
            and _has_expected_source_frames(report)
        )
        if (
            transport == "direct"
            and report.get("drain_seconds") is None
            and _validate_legacy_direct_timing_evidence(block, report, session_id)
        ):
            explicit_timing = True
        _failure(failures, explicit_timing, "FIXTURE_TIMING_MISMATCH")
        _failure(
            failures,
            report.get("passed") is True and report.get("session_id") == session_id,
            "CLIENT_REPORT_FAILED",
        )
        if transport == "direct":
            responses = report.get("responses")
            cardinality = (
                isinstance(responses, list)
                and len(responses) == 1
                and report.get("typed_completed") == 1
                and report.get("typed_answered") == 1
                and report.get("typed_unanswered") == 0
            )
            _failure(failures, cardinality, "RESPONSE_CARDINALITY_INVALID")
            if isinstance(responses, list) and len(responses) == 1:
                response = responses[0]
                _failure(
                    failures,
                    isinstance(response, dict)
                    and isinstance(response.get("audio_bytes"), int)
                    and response["audio_bytes"] > 0,
                    "DELIVERED_AUDIO_EMPTY",
                )
                _failure(
                    failures,
                    isinstance(response, dict)
                    and "cancel" not in str(response.get("done_reason", "")).lower(),
                    "RESPONSE_CANCELLED",
                )
            closed = report.get("session_closed")
            _failure(
                failures,
                isinstance(closed, dict)
                and closed.get("session_id") == session_id
                and closed.get("status") == "completed"
                and closed.get("reason") == "client_stop"
                and report.get("errors") == []
                and report.get("unexpected_errors") == [],
                "TEARDOWN_NOT_CLEAN",
            )
        else:
            records = report.get("job_records")
            cardinality = (
                report.get("prompts_sent") == 1
                and report.get("responses_done") == 1
                and isinstance(records, list)
                and len(records) == 1
                and isinstance(records[0], dict)
                and records[0].get("terminal_status") == "completed"
            )
            _failure(failures, cardinality, "RESPONSE_CARDINALITY_INVALID")
            _failure(
                failures,
                isinstance(report.get("audio_bytes"), int) and report["audio_bytes"] > 0,
                "DELIVERED_AUDIO_EMPTY",
            )
            _failure(
                failures,
                report.get("terminal_errors") == [],
                "TEARDOWN_NOT_CLEAN",
            )
            status = report.get("playout_trace_status")
            _failure(
                failures,
                report.get("playout_trace_enabled") is True
                and isinstance(status, dict)
                and status.get("enabled") is True
                and status.get("artifact_published") is True
                and status.get("closed") is True
                and status.get("valid") is True
                and status.get("dropped_records") == 0
                and status.get("queued_records") == 0
                and status.get("writer_alive") is False
                and status.get("error") is None
                and status.get("invalid_reason") is None,
                "PLAYOUT_TRACE_NOT_PUBLISHED",
            )
        return failures
    if transport != "browser":
        raise ValueError(f"unknown transport: {transport}")
    session = report.get("session")
    fixture = report.get("fixture")
    plan = report.get("plan")
    _failure(
        failures,
        isinstance(plan, dict)
        and plan.get("schema") == "nemotron_voicechat.step4c_fixture_plan.v1"
        and plan.get("warmup_seconds") == WARMUP_SECONDS
        and plan.get("duration_seconds") == DURATION_SECONDS
        and plan.get("drain_seconds") == DRAIN_SECONDS
        and isinstance(fixture, dict)
        and fixture.get("typed_text_sha256") == L1_SHA256,
        "FIXTURE_IDENTITY_MISMATCH",
    )
    _failure(
        failures,
        report.get("status") == "completed"
        and isinstance(session, dict)
        and session.get("status") == "completed"
        and session.get("session_id") == session_id
        and isinstance(fixture, dict)
        and fixture.get("completion_status") == "completed",
        "CLIENT_REPORT_FAILED",
    )
    connected_window = (
        session.get("connected_window_seconds") if isinstance(session, dict) else None
    )
    _failure(
        failures,
        isinstance(connected_window, (int, float))
        and not isinstance(connected_window, bool)
        and math.isfinite(connected_window)
        and connected_window + BROWSER_CONNECTED_WINDOW_TOLERANCE_SECONDS
        >= DURATION_SECONDS + DRAIN_SECONDS,
        "FIXTURE_TIMING_MISMATCH",
    )
    response_ids = fixture.get("response_ids") if isinstance(fixture, dict) else None
    attribution = fixture.get("response_attribution") if isinstance(fixture, dict) else None
    primary = attribution.get("primary_response") if isinstance(attribution, dict) else None
    _failure(
        failures,
        isinstance(response_ids, list)
        and len(response_ids) == 1
        and isinstance(primary, dict)
        and primary.get("response_id") == response_ids[0]
        and isinstance(primary.get("terminal"), dict)
        and primary["terminal"].get("status") == "completed",
        "RESPONSE_CARDINALITY_INVALID",
    )
    _failure(
        failures,
        isinstance(primary, dict)
        and isinstance(primary.get("audio_delta_ordinals"), list)
        and bool(primary["audio_delta_ordinals"]),
        "DELIVERED_AUDIO_EMPTY",
    )
    capture = session.get("capture_shutdown") if isinstance(session, dict) else None
    _failure(
        failures,
        isinstance(session, dict)
        and session.get("teardown_errors") == []
        and isinstance(capture, dict)
        and capture.get("shutdown_error") is None
        and capture.get("sink_close_error") is None
        and isinstance(capture.get("webrtc_disconnect"), dict)
        and capture["webrtc_disconnect"].get("completed") is True,
        "TEARDOWN_NOT_CLEAN",
    )
    browser_artifact = (
        session.get("browser_artifact_validation") if isinstance(session, dict) else None
    )
    _failure(
        failures,
        isinstance(browser_artifact, dict) and browser_artifact.get("valid") is True,
        "BROWSER_ARTIFACT_INVALID",
    )
    observed = session.get("observed_playout_trace") if isinstance(session, dict) else None
    terminal = observed.get("terminal_status") if isinstance(observed, dict) else None
    _failure(
        failures,
        isinstance(observed, dict)
        and observed.get("valid") is True
        and isinstance(terminal, dict)
        and terminal.get("valid") is True
        and terminal.get("dropped_records") == 0
        and terminal.get("error") is None
        and terminal.get("invalid_reason") is None,
        "PLAYOUT_TRACE_NOT_PUBLISHED",
    )
    return failures


def _raw_block(trace: Path, analyzer: Mapping[str, Any], session_id: str) -> dict[str, Any]:
    events = qualified.read_jsonl(trace)
    observed_ids = {
        event.get("session_id") for event in events if isinstance(event.get("session_id"), str)
    }
    if observed_ids != {session_id}:
        raise ValueError(f"trace session identity mismatch: {sorted(observed_ids)}")
    normalized = qualified._normalize_model_rows(events)
    populations = qualified.classify_populations(normalized)
    for name in ("delivered-nonBOS", "EarTTS-high"):
        expected = qualified._membership_manifest(populations[name])
        actual = _membership(analyzer, name)
        if expected != actual:
            raise ValueError(f"qualified analyzer {name} membership does not reproduce")
    high_keys = {_member_key(member) for member in _membership(analyzer, "EarTTS-high")["members"]}
    delivered_keys = {
        _member_key(member) for member in _membership(analyzer, "delivered-nonBOS")["members"]
    }
    if not high_keys <= delivered_keys:
        raise ValueError("EarTTS-high is not a subset of delivered-nonBOS")
    failures: set[str] = set()
    transport_frames = [
        event["transport_frame"]
        for event in events
        if (
            event.get("event") in {"idle_step", "model_step"}
            or event.get("type") == "voicechat.metrics"
        )
        and isinstance(event.get("transport_frame"), int)
    ]
    _failure(
        failures,
        bool(transport_frames)
        and transport_frames == list(range(transport_frames[0], transport_frames[-1] + 1)),
        "TRANSPORT_FRAMES_NONCONTIGUOUS",
    )
    microphone = [
        event
        for event in events
        if event.get("event") == "input_packet" and event.get("source") == "microphone"
    ]
    _failure(
        failures,
        bool(microphone)
        and all(
            isinstance(event.get("audio"), dict)
            and event["audio"].get("peak") == 0.0
            and event["audio"].get("rms") == 0.0
            for event in microphone
        ),
        "ACOUSTIC_INPUT_NOT_EXACT_ZERO",
    )
    configurations = [event for event in events if event.get("event") == "session_tools_configured"]
    _failure(
        failures,
        len(configurations) == 1 and configurations[0].get("tools") == [],
        "FIXTURE_IDENTITY_MISMATCH",
    )
    output_text = "".join(
        str(row["event"].get("assistant_delta") or "") for row in normalized if row["delivered"]
    )
    if not output_text.strip():
        raise ValueError("response target produced no delivered output text")
    return {
        "events": events,
        "normalized": normalized,
        "populations": populations,
        "output_text": output_text,
        "response_target_match": response_target_matches(output_text),
        "validity_failures": failures,
        "microphone_packets": len(microphone),
        "microphone_max_peak": max(
            (float(event["audio"]["peak"]) for event in microphone), default=None
        ),
    }


def _playout_validity_failures(
    block: Mapping[str, Any], transport: str, session_id: str, report: Mapping[str, Any]
) -> set[str]:
    if transport == "direct":
        return set()
    path_value = block.get("playout_trace")
    if not isinstance(path_value, str) or not path_value:
        return {"PLAYOUT_TRACE_NOT_PUBLISHED"}
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size == 0:
        return {"PLAYOUT_TRACE_NOT_PUBLISHED"}
    try:
        records = qualified.read_jsonl(path)
        validation = qualified.validate_observed_playout_trace(records)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"PLAYOUT_TRACE_INVALID"}
    failures: set[str] = set()
    _failure(failures, validation.get("valid") is True, "PLAYOUT_TRACE_INVALID")
    session_ids = {
        record.get("session_id")
        for record in records
        if isinstance(record.get("session_id"), str)
    }
    _failure(failures, session_ids == {session_id}, "PLAYOUT_TRACE_IDENTITY_MISMATCH")
    response_ids = {
        record.get("response_id")
        for record in records
        if isinstance(record.get("response_id"), str)
    }
    expected: set[str] = set()
    if transport == "pipecat":
        jobs = report.get("job_records")
        if isinstance(jobs, list):
            expected = {
                job.get("response_id")
                for job in jobs
                if isinstance(job, dict) and isinstance(job.get("response_id"), str)
            }
    else:
        fixture = report.get("fixture")
        ids = fixture.get("response_ids") if isinstance(fixture, dict) else None
        if isinstance(ids, list):
            expected = {value for value in ids if isinstance(value, str)}
    _failure(
        failures,
        len(expected) == 1 and expected <= response_ids,
        "PLAYOUT_TRACE_IDENTITY_MISMATCH",
    )
    return failures


def _select_matched_states(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for block in blocks:
        raw = block["_raw"]
        highs = [
            row
            for row in raw["populations"]["EarTTS-high"]
            if qualified._token_phase(row) == "PAD-tail"
        ]
        if not highs:
            continue
        high = highs[0]
        candidates = [
            row
            for row in raw["populations"]["delivered-nonBOS"]
            if qualified._token_phase(row) == "PAD-tail"
            and row.get("response_group") == high.get("response_group")
            and row not in highs
        ]
        if not candidates:
            continue
        low = min(
            candidates,
            key=lambda row: abs(int(row["event"].get("frame")) - int(high["event"].get("frame"))),
        )
        model_rows = raw["normalized"]
        by_ordinal = {row["event_ordinal"]: row["event"] for row in model_rows}

        def state(row: Mapping[str, Any]) -> dict[str, Any]:
            ordinal = int(row["event_ordinal"])
            return {
                "membership": qualified._membership_identifier(row),
                "phase": qualified._token_phase(row),
                "event": row["event"],
                "neighbor_model_rows": [
                    by_ordinal[index]
                    for index in range(ordinal - 1, ordinal + 2)
                    if index in by_ordinal
                ],
            }

        return {
            "status": "retained-for-nsight",
            "matching": "same session/response, PAD-tail, nearest model frame",
            "block": block["block"],
            "transport": block["transport"],
            "session_id": block["session_id"],
            "trace": block["trace"],
            "high": state(high),
            "low": state(low),
        }
    return {"status": "not-triggered", "reason": "no matchable PAD-tail EarTTS-high row"}


def analyze(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("campaign manifest has the wrong schema")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(SCHEDULE):
        raise ValueError("campaign must contain exactly the nine preregistered blocks")
    before = _read_json(Path(str(manifest["container_inspect_before"])))
    after = _read_json(Path(str(manifest["container_inspect_after"])))
    if _container_identity(before) != _container_identity(after):
        raise ValueError("model container identity/configuration changed during Step 4c")
    session_ids: set[str] = set()
    analyzed_blocks: list[dict[str, Any]] = []
    rates: dict[str, list[float]] = defaultdict(list)
    for index, (block, assigned) in enumerate(zip(blocks, SCHEDULE, strict=True), start=1):
        if (
            not isinstance(block, dict)
            or block.get("block") != index
            or block.get("transport") != assigned
        ):
            raise ValueError(f"block {index} differs from the preregistered schedule")
        session_id = block.get("session_id")
        if not isinstance(session_id, str) or not session_id or session_id in session_ids:
            raise ValueError(f"block {index} has a missing or repeated session ID")
        session_ids.add(session_id)
        failures: set[str] = set()
        health = _read_json(Path(str(block["health"])))
        _failure(failures, health.get("active_client") is False, "HEALTH_NOT_EXCLUSIVE")
        telemetry_path = Path(str(block["telemetry"]))
        _failure(
            failures,
            telemetry_path.is_file() and telemetry_path.stat().st_size > 0,
            "TELEMETRY_MISSING",
        )
        trace_path = Path(str(block["trace"]))
        analyzer_path = Path(str(block["analyzer"]))
        analyzer_document = _read_json(analyzer_path)
        analyzer_sources = analyzer_document.get("sources")
        if not isinstance(analyzer_sources, list) or len(analyzer_sources) != 1:
            raise ValueError(f"block {index} analyzer does not contain exactly one source")
        analyzer = analyzer_sources[0]
        if not isinstance(analyzer, dict):
            raise ValueError(f"block {index} analyzer source is malformed")
        source = analyzer.get("source")
        if not isinstance(source, dict) or source.get("sha256") != _sha256(trace_path):
            raise ValueError(f"block {index} analyzer/trace source identity mismatch")
        client_report = _read_json(Path(str(block["client_report"])))
        raw = _raw_block(trace_path, analyzer, session_id)
        failures.update(raw["validity_failures"])
        failures.update(_client_validity_failures(assigned, client_report, session_id, block))
        failures.update(_playout_validity_failures(block, assigned, session_id, client_report))
        if raw["response_target_match"] is not True:
            failures.add("RESPONSE_TARGET_MISMATCH")
        declared_valid = block.get("valid")
        invalid_reason = block.get("invalid_reason")
        if not isinstance(declared_valid, bool):
            raise ValueError(f"block {index} lacks an explicit validity decision")
        if declared_valid and invalid_reason is not None:
            raise ValueError(f"valid block {index} carries an invalid reason")
        if not declared_valid and (
            not isinstance(invalid_reason, str)
            or not invalid_reason
            or invalid_reason not in INVALID_REASON_CODES
        ):
            raise ValueError(f"invalid block {index} lacks an enumerated nonempty reason")
        if declared_valid and failures:
            raise ValueError(
                f"block {index} is declared valid but evidence fails: {sorted(failures)}"
            )
        if not declared_valid and invalid_reason not in failures:
            raise ValueError(
                f"block {index} invalid reason {invalid_reason!r} contradicts evidence: "
                f"{sorted(failures)}"
            )
        denominator = int(_membership(analyzer, "delivered-nonBOS")["n"])
        high = int(_membership(analyzer, "EarTTS-high")["n"])
        if denominator <= 0 or high > denominator:
            raise ValueError(f"block {index} has invalid qualified incidence counts")
        rate = high / denominator
        if declared_valid:
            rates[assigned].append(rate)
        analyzed_blocks.append(
            {
                "block": index,
                "transport": assigned,
                "session_id": session_id,
                "valid": declared_valid,
                "invalid_reason": invalid_reason,
                "validity_failures": sorted(failures),
                "response_target_match": raw["response_target_match"],
                "trace": str(trace_path),
                "trace_sha256": _sha256(trace_path),
                "delivered_nonbos": denominator,
                "eartts_high": high,
                "incidence": rate,
                "any_eartts_high": high > 0,
                "microphone_packets": raw["microphone_packets"],
                "microphone_max_peak": raw["microphone_max_peak"],
                "delivered_nonbos_membership_sha256": _membership(analyzer, "delivered-nonBOS")[
                    "sha256"
                ],
                "eartts_high_membership_sha256": _membership(analyzer, "EarTTS-high")["sha256"],
                "_raw": raw,
            }
        )
    transport_summaries = {}
    for transport in SCHEDULE:
        if transport in transport_summaries:
            continue
        values = rates[transport]
        transport_summaries[transport] = {
            "valid_blocks": len(values),
            "equal_block_mean_incidence": sum(values) / len(values) if values else None,
            "blocks_with_event": sum(value > 0 for value in values),
            "bootstrap": bootstrap_interval({transport: values}) if values else None,
        }
    valid_blocks = [block for block in analyzed_blocks if block["valid"]]
    total_high = sum(block["eartts_high"] for block in valid_blocks)
    total_delivered = sum(block["delivered_nonbos"] for block in valid_blocks)
    states = _select_matched_states(valid_blocks)
    public_blocks = [
        {key: value for key, value in block.items() if key != "_raw"} for block in analyzed_blocks
    ]
    result = {
        "schema": SCHEMA,
        "decision": "RECURRENT" if total_high else "INSUFFICIENT",
        "assignment_unit": "independently connected run block",
        "uncertainty_unit": "run block",
        "schedule": list(SCHEDULE),
        "exposure_cap_blocks": len(SCHEDULE),
        "attempted_blocks": len(analyzed_blocks),
        "valid_blocks": len(valid_blocks),
        "invalid_blocks": len(analyzed_blocks) - len(valid_blocks),
        "blocks": public_blocks,
        "transport_summaries": transport_summaries,
        "overall_equal_block": (
            bootstrap_interval(rates)
            if all(rates[transport] for transport in set(SCHEDULE))
            else None
        ),
        "descriptive_pooled_exposure": {
            "eartts_high": total_high,
            "delivered_nonbos": total_delivered,
            "incidence": total_high / total_delivered if total_delivered else None,
        },
        "matched_states": {
            key: value for key, value in states.items() if key not in {"high", "low"}
        },
    }
    return result, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    args = parser.parse_args()
    result, states = analyze(args.manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.states_output.write_text(
        json.dumps(states, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
