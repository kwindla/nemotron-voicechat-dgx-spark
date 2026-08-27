#!/usr/bin/env python3
"""Audit generation-frame latency from retained JSONL artifacts.

The analyzer understands the two retained formats used by the generation-frame
latency plan:

* client qualification streams use ``type == "voicechat.metrics"`` as model
  rows.  Their server clocks are embedded in those rows, while the normative
  queue clock and exact PCM payloads come from
  ``response.output_audio.delta`` events.
* server model traces use ``event == "model_step"``.  Canonical delivery uses
  ``audio_delivered`` when present; positive ``output_audio.bytes`` is only a
  labeled legacy fallback.  Their ``monotonic_s`` clock is diagnostic-only.

EarTTS BOS transitions and actual abort/prefill resets are separate strata.
Function-cycle pseudo-steps are reported separately and excluded from normative
``idle``; literal response-less rows remain available as ``response-null`` for
legacy-anchor reproduction.  High-mode predicates apply only to delivered,
non-BOS-transition rows.

Only the Python standard library is used.  Input artifacts are never modified.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

JsonObject = dict[str, Any]
DEFAULT_RESERVES_MS = tuple(range(0, 561, 80))
SAMPLE_RATE_HZ = 22_050
BOOTSTRAP_SEED = 20_260_812
BOOTSTRAP_ITERATIONS = 10_000
POPULATION_NAMES = (
    "delivered",
    "delivered-nonBOS",
    "BOS-transition",
    "abort-prefill-reset",
    "response-null",
    "idle",
    "function-cycle",
    "EarTTS-high",
    "residual-high",
)
STAGE_FIELDS = {
    "perception": "perception_ms",
    "nano_interface": "nano_interface_ms",
    "eartts_total": "eartts_total_ms",
    "codec_call": "codec_call_ms",
    "wrapper_residual": "wrapper_residual_ms",
    "wrapper_total": "wrapper_total_ms",
}
CLOCK_FIELDS = ("server_step_ms", "inference_ms")
WIRE_ENCODING = "pcm16"
WIRE_CHANNELS = 1
PAD_TOKEN_ID = 12


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read JSON objects from *path*, failing closed on malformed rows."""

    events: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ordinal}: invalid JSON: {exc.msg}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{ordinal}: event must be a JSON object")
            events.append(event)
    if not events:
        raise ValueError(f"{path}: no JSON events")
    return events


def source_identity(path: Path) -> JsonObject:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    """Return the nearest-rank percentile (ceil(p/100*n), one based)."""

    if not values:
        return None
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return ordered[index]


def distribution_stats(values: Sequence[float]) -> JsonObject:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "count_over_80_ms": 0,
        }
    numeric = [float(value) for value in values]
    return {
        "n": len(numeric),
        "mean": sum(numeric) / len(numeric),
        "p50": nearest_rank(numeric, 50),
        "p95": nearest_rank(numeric, 95),
        "p99": nearest_rank(numeric, 99),
        "max": max(numeric),
        "count_over_80_ms": sum(value > 80.0 for value in numeric),
    }


def _stage_timings(row: JsonObject) -> JsonObject:
    turn_state = row.get("turn_state")
    if not isinstance(turn_state, dict):
        return {}
    timings = turn_state.get("model_stage_timings_ms")
    return timings if isinstance(timings, dict) else {}


def is_bos_transition(row: JsonObject) -> bool:
    timings = _stage_timings(row)
    turn_state = row.get("turn_state")
    return bool(
        isinstance(timings.get("eartts_reset"), dict)
        or (isinstance(turn_state, dict) and turn_state.get("eartts_bos_epoch_step") is not None)
    )


def is_abort_prefill_reset(row: JsonObject) -> bool:
    return isinstance(_stage_timings(row).get("eartts_reset"), dict)


def _delivery_classification(row: JsonObject) -> tuple[bool, str]:
    if row.get("type") == "voicechat.metrics" or row.get("event") == "model_step":
        if "audio_delivered" in row:
            return row.get("audio_delivered") is True, "audio_delivered"
    if row.get("event") == "model_step":
        output_audio = row.get("output_audio")
        delivered = bool(
            isinstance(output_audio, dict) and (_number(output_audio.get("bytes")) or 0.0) > 0.0
        )
        return delivered, "legacy_output_audio_bytes"
    return False, "unavailable"


def is_delivered(row: JsonObject) -> bool:
    return _delivery_classification(row)[0]


def is_function_cycle(row: JsonObject) -> bool:
    """Recognize pseudo-steps in client metrics and legacy server schemas.

    Client metrics identify the skipped background function cycle directly.
    Legacy server traces expose the later response-injection/drain state instead;
    ordinary active function-token generation is not a pseudo-step there.
    """

    if row.get("event") in {"function_cycle", "function_cycle_step"} or row.get("type") in {
        "function_cycle",
        "function_cycle_step",
    }:
        return True
    if is_delivered(row) or row.get("response_id") is not None or _stage_timings(row):
        return False
    turn_state = row.get("turn_state")
    function = turn_state.get("function_calling") if isinstance(turn_state, dict) else None
    if not isinstance(function, dict):
        return False
    if row.get("type") == "voicechat.metrics":
        return bool(function.get("active") and function.get("background_active"))
    return bool(
        function.get("awaiting_response")
        or function.get("injecting_response")
        or function.get("awaiting_eotr")
        or (_number(function.get("forced_tokens")) or 0.0) > 0.0
    )


def _normalize_model_rows(events: Sequence[JsonObject]) -> list[JsonObject]:
    rows: list[JsonObject] = []
    generated_response = 0
    active_generated_response: str | None = None
    for event_ordinal, event in enumerate(events, 1):
        if event.get("type") != "voicechat.metrics" and event.get("event") != "model_step":
            continue
        delivered = is_delivered(event)
        bos_transition = is_bos_transition(event)
        abort_prefill_reset = is_abort_prefill_reset(event)
        _, delivery_mode = _delivery_classification(event)
        explicit_response = event.get("response_id")
        response_group = explicit_response if isinstance(explicit_response, str) else None
        if event.get("event") == "model_step" and delivered:
            if bos_transition or active_generated_response is None:
                generated_response += 1
                active_generated_response = f"inferred-response-{generated_response:04d}"
            response_group = active_generated_response
        rows.append(
            {
                "event": event,
                "event_ordinal": event_ordinal,
                "delivered": delivered,
                "bos_transition": bos_transition,
                "abort_prefill_reset": abort_prefill_reset,
                "delivery_mode": delivery_mode,
                "function_cycle": is_function_cycle(event),
                "response_group": response_group,
            }
        )
    if not rows:
        raise ValueError("artifact has neither voicechat.metrics nor model_step rows")
    return rows


def _membership_identifier(normalized: JsonObject) -> JsonObject:
    row = normalized["event"]
    timestamp_field = None
    timestamp = None
    for field in ("monotonic_s", "client_received_monotonic_s"):
        value = _number(row.get(field))
        if value is not None:
            timestamp_field = field
            timestamp = value
            break
    result: JsonObject = {
        "event_ordinal": normalized["event_ordinal"],
        "frame": row.get("frame"),
        "timestamp_field": timestamp_field,
        "monotonic_timestamp_s": timestamp,
    }
    if normalized.get("response_group") is not None:
        result["response_id"] = normalized["response_group"]
    return result


def _membership_manifest(rows: Sequence[JsonObject]) -> JsonObject:
    members = [_membership_identifier(row) for row in rows]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "n": len(members),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "members": members,
    }


def classify_populations(rows: Sequence[JsonObject]) -> dict[str, list[JsonObject]]:
    populations: dict[str, list[JsonObject]] = {name: [] for name in POPULATION_NAMES}
    for normalized in rows:
        row = normalized["event"]
        timings = _stage_timings(row)
        delivered = normalized["delivered"]
        bos_transition = normalized["bos_transition"]
        if delivered:
            populations["delivered"].append(normalized)
            populations["BOS-transition" if bos_transition else "delivered-nonBOS"].append(
                normalized
            )
            if normalized["abort_prefill_reset"]:
                populations["abort-prefill-reset"].append(normalized)
        response_null = row.get("response_id") is None
        if response_null:
            populations["response-null"].append(normalized)
            if not delivered and not normalized["function_cycle"]:
                populations["idle"].append(normalized)
        if normalized["function_cycle"]:
            populations["function-cycle"].append(normalized)
        if delivered and not bos_transition:
            if (_number(timings.get("eartts_total_ms")) or -math.inf) >= 18.0:
                populations["EarTTS-high"].append(normalized)
            if (_number(timings.get("wrapper_residual_ms")) or -math.inf) >= 10.0:
                populations["residual-high"].append(normalized)
    return populations


def _stage_means(rows: Sequence[JsonObject]) -> JsonObject:
    result: JsonObject = {}
    for label, field in STAGE_FIELDS.items():
        values = [
            value
            for normalized in rows
            if (value := _number(_stage_timings(normalized["event"]).get(field))) is not None
        ]
        result[label] = {
            "n": len(values),
            "mean_ms": sum(values) / len(values) if values else None,
        }
    return result


def _stage_distributions(rows: Sequence[JsonObject]) -> JsonObject:
    """Return auditable distributions for each stage present in *rows*."""

    result: JsonObject = {}
    for label, field in STAGE_FIELDS.items():
        values = [
            value
            for normalized in rows
            if (value := _number(_stage_timings(normalized["event"]).get(field))) is not None
        ]
        result[label] = distribution_stats(values)
    return result


def _token_phase(normalized: JsonObject) -> str:
    """Classify a row using the retained agent token, without latency inference."""

    if normalized["bos_transition"]:
        return "BOS"
    turn_state = normalized["event"].get("turn_state")
    token_id = turn_state.get("agent_token_id") if isinstance(turn_state, dict) else None
    agent_control = turn_state.get("agent_control") if isinstance(turn_state, dict) else None
    control_ids = turn_state.get("control_ids") if isinstance(turn_state, dict) else None
    pad_control_id = control_ids.get("pad") if isinstance(control_ids, dict) else None
    pad_control_match = (
        isinstance(pad_control_id, int)
        and not isinstance(pad_control_id, bool)
        and token_id == pad_control_id
    )
    if token_id == PAD_TOKEN_ID or agent_control == "pad" or pad_control_match:
        return "PAD-tail"
    if isinstance(agent_control, str) and agent_control:
        return "agent-control"
    if isinstance(control_ids, dict) and any(
        name != "pad" and token_id == control_id
        for name, control_id in control_ids.items()
        if isinstance(name, str)
        and isinstance(control_id, int)
        and not isinstance(control_id, bool)
    ):
        return "agent-control"
    if isinstance(token_id, int) and not isinstance(token_id, bool):
        return "text-emission"
    return "token-unavailable"


def _corpus_membership(rows: Sequence[JsonObject]) -> JsonObject:
    members = []
    for item in rows:
        member = _membership_identifier(item["normalized"])
        member["artifact"] = item["artifact"]
        member["source_sha256"] = item["source_sha256"]
        members.append(member)
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "n": len(members),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "members": members,
    }


def _corpus_stratum(rows: Sequence[JsonObject], definition: str) -> JsonObject:
    normalized = [item["normalized"] for item in rows]
    return {
        "definition": definition,
        "membership": _corpus_membership(rows),
        "clocks": {
            clock: distribution_stats(
                [
                    value
                    for row in normalized
                    if (value := _number(row["event"].get(clock))) is not None
                ]
            )
            for clock in CLOCK_FIELDS
        },
        "stage_timings": _stage_distributions(normalized),
    }


def _stage_timed(item: JsonObject) -> bool:
    return any(
        _number(_stage_timings(item["normalized"]["event"]).get(field)) is not None
        for field in STAGE_FIELDS.values()
    )


def _nano_session_position(normalized: JsonObject) -> int | None:
    turn_state = normalized["event"].get("turn_state")
    positions = turn_state.get("vllm_request_positions") if isinstance(turn_state, dict) else None
    nano = positions.get("nano") if isinstance(positions, dict) else None
    value = nano.get("session_positions") if isinstance(nano, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _pair_schedule(rows: Sequence[JsonObject]) -> JsonObject:
    """Classify Nano calls from retained request-position advancement.

    A zero-position advance is the scheduler's synthetic buffered row, a
    two-position advance is the following packed request, and a one-position
    advance is ordinary sequential execution.  No latency threshold is used.
    """

    previous_by_request: dict[str, int] = {}
    scheduled: list[JsonObject] = []
    for item in rows:
        normalized = item["normalized"]
        row = normalized["event"]
        turn_state = row.get("turn_state")
        positions = (
            turn_state.get("vllm_request_positions") if isinstance(turn_state, dict) else None
        )
        request_id = positions.get("request_id") if isinstance(positions, dict) else None
        current = _nano_session_position(normalized)
        delta = None
        if isinstance(request_id, str) and current is not None:
            previous = previous_by_request.get(request_id)
            if previous is not None and current >= previous:
                delta = current - previous
            previous_by_request[request_id] = current
        if not normalized["delivered"]:
            continue
        if delta == 0:
            schedule_class = "buffered"
        elif delta == 2:
            schedule_class = "packed-pair"
        elif delta == 1:
            schedule_class = "sequential"
        elif delta is None:
            schedule_class = "unavailable"
        else:
            schedule_class = "other-position-delta"
        scheduled.append({**item, "schedule_class": schedule_class, "nano_position_delta": delta})

    classes = ("buffered", "packed-pair", "sequential", "unavailable", "other-position-delta")
    class_reports = {
        name: _corpus_stratum(
            [item for item in scheduled if item["schedule_class"] == name],
            f"delivered rows with Nano request-position delta classified as {name}",
        )
        for name in classes
    }
    transitions: dict[str, int] = defaultdict(int)
    pair_rows = [
        item for item in scheduled if item["schedule_class"] in {"buffered", "packed-pair"}
    ]
    for previous, current in zip(pair_rows, pair_rows[1:]):
        transitions[f"{previous['schedule_class']}->{current['schedule_class']}"] += 1
    return {
        "classification_basis": "delta of turn_state.vllm_request_positions.nano.session_positions",
        "delta_mapping": {"0": "buffered", "1": "sequential", "2": "packed-pair"},
        "classes": class_reports,
        "pair_class_membership": _corpus_membership(pair_rows),
        "pair_class_row_count": len(pair_rows),
        "adjacent_pair_class_transition_count": max(0, len(pair_rows) - 1),
        "transitions": dict(sorted(transitions.items())),
        "rows": [
            {
                **_membership_identifier(item["normalized"]),
                "phase": _token_phase(item["normalized"]),
                "agent_token_id": (item["normalized"]["event"].get("turn_state") or {}).get(
                    "agent_token_id"
                ),
                "server_step_ms": _number(item["normalized"]["event"].get("server_step_ms")),
                "nano_position_delta": item["nano_position_delta"],
                "schedule_class": item["schedule_class"],
            }
            for item in scheduled
        ],
    }


def _row_context(
    item: JsonObject,
    artifact_rows: Sequence[JsonObject],
    response_positions: dict[int, tuple[int, int]],
) -> JsonObject:
    normalized = item["normalized"]
    row = normalized["event"]
    turn_state = row.get("turn_state")

    def summary(candidate: JsonObject | None) -> JsonObject | None:
        if candidate is None:
            return None
        event = candidate["event"]
        candidate_state = event.get("turn_state")
        return {
            **_membership_identifier(candidate),
            "delivered": candidate["delivered"],
            "phase": _token_phase(candidate),
            "agent_token_id": (
                candidate_state.get("agent_token_id") if isinstance(candidate_state, dict) else None
            ),
            "server_step_ms": _number(event.get("server_step_ms")),
        }

    index = artifact_rows.index(normalized)
    response_ordinal, response_count = response_positions.get(id(normalized), (None, None))
    return {
        "artifact": item["artifact"],
        "source_sha256": item["source_sha256"],
        **_membership_identifier(normalized),
        "phase": _token_phase(normalized),
        "agent_token_id": turn_state.get("agent_token_id")
        if isinstance(turn_state, dict)
        else None,
        "agent_control": turn_state.get("agent_control") if isinstance(turn_state, dict) else None,
        "response_delivered_ordinal": response_ordinal,
        "response_delivered_count": response_count,
        "server_step_ms": _number(row.get("server_step_ms")),
        "inference_ms": _number(row.get("inference_ms")),
        "stage_timings_ms": {
            label: _number(_stage_timings(row).get(field)) for label, field in STAGE_FIELDS.items()
        },
        "previous_model_row": summary(artifact_rows[index - 1] if index else None),
        "next_model_row": summary(
            artifact_rows[index + 1] if index + 1 < len(artifact_rows) else None
        ),
    }


def _inventory_patterns(rows: Sequence[JsonObject]) -> JsonObject:
    by_phase: dict[str, int] = defaultdict(int)
    by_artifact: dict[str, int] = defaultdict(int)
    response_ordinals: list[float] = []
    response_end_distances: list[float] = []
    response_gaps: list[float] = []
    frame_gaps: list[float] = []
    previous_frame_by_artifact: dict[str, float] = {}
    previous_ordinal_by_response: dict[tuple[str, str], float] = {}
    for row in rows:
        by_phase[str(row["phase"])] += 1
        by_artifact[str(row["artifact"])] += 1
        ordinal = _number(row.get("response_delivered_ordinal"))
        response_count = _number(row.get("response_delivered_count"))
        if ordinal is not None:
            response_ordinals.append(ordinal)
            if response_count is not None:
                response_end_distances.append(response_count - ordinal)
            response_key = (str(row["artifact"]), str(row.get("response_id")))
            if response_key in previous_ordinal_by_response:
                response_gaps.append(ordinal - previous_ordinal_by_response[response_key])
            previous_ordinal_by_response[response_key] = ordinal
        frame = _number(row.get("frame"))
        artifact = str(row["artifact"])
        if frame is not None and artifact in previous_frame_by_artifact:
            frame_gaps.append(frame - previous_frame_by_artifact[artifact])
        if frame is not None:
            previous_frame_by_artifact[artifact] = frame

    def numeric_stats(values: Sequence[float]) -> JsonObject:
        return {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            "p50": nearest_rank(values, 50),
            "p95": nearest_rank(values, 95),
            "max": max(values) if values else None,
        }

    return {
        "counts_by_phase": dict(sorted(by_phase.items())),
        "counts_by_artifact": dict(sorted(by_artifact.items())),
        "response_delivered_ordinal": numeric_stats(response_ordinals),
        "delivered_rows_from_response_end": numeric_stats(response_end_distances),
        "within_response_high_row_ordinal_gap": numeric_stats(response_gaps),
        "within_artifact_frame_gap": numeric_stats(frame_gaps),
    }


def _build_step1_details(paths: Sequence[Path], source_reports: Sequence[JsonObject]) -> JsonObject:
    corpus: list[JsonObject] = []
    artifact_rows: dict[str, list[JsonObject]] = {}
    source_by_path = {report["source"]["path"]: report for report in source_reports}
    for path in paths:
        resolved = str(path.expanduser().resolve(strict=True))
        source_report = source_by_path[resolved]
        artifact = Path(resolved).parent.name
        if artifact in artifact_rows:
            raise ValueError(f"duplicate Step 1 artifact label: {artifact}")
        normalized_rows = _normalize_model_rows(read_jsonl(Path(resolved)))
        artifact_rows[artifact] = normalized_rows
        corpus.extend(
            {
                "artifact": artifact,
                "source_sha256": source_report["source"]["sha256"],
                "normalized": normalized,
            }
            for normalized in normalized_rows
        )

    delivered_nonbos_stage = [
        item
        for item in corpus
        if item["normalized"]["delivered"]
        and not item["normalized"]["bos_transition"]
        and _stage_timed(item)
    ]
    bos_stage = [
        item
        for item in corpus
        if item["normalized"]["delivered"]
        and item["normalized"]["bos_transition"]
        and _stage_timed(item)
    ]
    strata_rows = {
        "delivered-nonBOS-stage-timed": delivered_nonbos_stage,
        "text-emission": [
            item
            for item in delivered_nonbos_stage
            if _token_phase(item["normalized"]) == "text-emission"
        ],
        "PAD-tail": [
            item
            for item in delivered_nonbos_stage
            if _token_phase(item["normalized"]) == "PAD-tail"
        ],
        "agent-control": [
            item
            for item in delivered_nonbos_stage
            if _token_phase(item["normalized"]) == "agent-control"
        ],
        "token-unavailable": [
            item
            for item in delivered_nonbos_stage
            if _token_phase(item["normalized"]) == "token-unavailable"
        ],
        "EarTTS-high": [
            item
            for item in delivered_nonbos_stage
            if (
                _number(_stage_timings(item["normalized"]["event"]).get("eartts_total_ms"))
                or -math.inf
            )
            >= 18.0
        ],
        "residual-high": [
            item
            for item in delivered_nonbos_stage
            if (
                _number(_stage_timings(item["normalized"]["event"]).get("wrapper_residual_ms"))
                or -math.inf
            )
            >= 10.0
        ],
        "BOS-transition": bos_stage,
        "abort-prefill-reset": [
            item for item in bos_stage if item["normalized"]["abort_prefill_reset"]
        ],
        "prepared-reuse-BOS": [
            item for item in bos_stage if not item["normalized"]["abort_prefill_reset"]
        ],
    }
    definitions = {
        "delivered-nonBOS-stage-timed": (
            "delivered-nonBOS rows with at least one named stage timing"
        ),
        "text-emission": (
            "stage-timed delivered-nonBOS rows with an integer agent_token_id that is "
            "neither PAD (12) nor a source-identified agent control"
        ),
        "PAD-tail": "stage-timed delivered-nonBOS rows with agent_token_id == PAD (12)",
        "agent-control": (
            "stage-timed delivered-nonBOS rows identified by agent_control or a "
            "matching non-PAD control_ids entry"
        ),
        "token-unavailable": (
            "stage-timed delivered-nonBOS rows without source control evidence or an "
            "integer agent_token_id"
        ),
        "EarTTS-high": "stage-timed delivered-nonBOS rows with eartts_total_ms >= 18",
        "residual-high": "stage-timed delivered-nonBOS rows with wrapper_residual_ms >= 10",
        "BOS-transition": "stage-timed delivered BOS-transition rows",
        "abort-prefill-reset": (
            "stage-timed delivered BOS-transition rows with an actual eartts_reset"
        ),
        "prepared-reuse-BOS": (
            "stage-timed delivered BOS-transition rows without an actual eartts_reset"
        ),
    }

    population_rows: dict[str, list[JsonObject]] = {name: [] for name in POPULATION_NAMES}
    for artifact, rows in artifact_rows.items():
        source_sha = next(
            report["source"]["sha256"]
            for report in source_reports
            if Path(report["source"]["path"]).parent.name == artifact
        )
        populations = classify_populations(rows)
        for name in POPULATION_NAMES:
            population_rows[name].extend(
                {"artifact": artifact, "source_sha256": source_sha, "normalized": row}
                for row in populations[name]
            )

    schedules = {
        artifact: _pair_schedule(
            [
                {
                    "artifact": artifact,
                    "source_sha256": next(
                        report["source"]["sha256"]
                        for report in source_reports
                        if Path(report["source"]["path"]).parent.name == artifact
                    ),
                    "normalized": row,
                }
                for row in rows
            ]
        )
        for artifact, rows in artifact_rows.items()
    }
    pair_candidates = [
        artifact
        for artifact, schedule in schedules.items()
        if schedule["classes"]["buffered"]["membership"]["n"]
        and schedule["classes"]["packed-pair"]["membership"]["n"]
    ]
    if len(pair_candidates) != 1:
        raise ValueError(
            "Step 1 details require exactly one artifact with retained buffered/packed evidence"
        )
    pair_artifact = pair_candidates[0]
    pair_vs_sequential: JsonObject = {
        "clock": "server_step_ms",
        "pair_artifact": pair_artifact,
        "sequential_artifacts": sorted(set(artifact_rows) - {pair_artifact}),
        "populations": {},
        "pair_schedule": schedules[pair_artifact],
    }
    for name in POPULATION_NAMES:
        pair_rows = [item for item in population_rows[name] if item["artifact"] == pair_artifact]
        sequential_rows = [
            item for item in population_rows[name] if item["artifact"] != pair_artifact
        ]
        pair_vs_sequential["populations"][name] = {
            "pair": _corpus_stratum(pair_rows, f"{name} rows from the pair artifact"),
            "sequential": _corpus_stratum(
                sequential_rows, f"{name} rows from all non-pair retained artifacts"
            ),
        }

    response_positions: dict[str, dict[int, tuple[int, int]]] = {}
    for artifact, rows in artifact_rows.items():
        grouped: dict[str, list[JsonObject]] = defaultdict(list)
        for normalized in rows:
            if normalized["delivered"]:
                grouped[str(normalized.get("response_group") or "unclustered")].append(normalized)
        response_positions[artifact] = {
            id(normalized): (ordinal, len(group))
            for group in grouped.values()
            for ordinal, normalized in enumerate(group, 1)
        }

    inventories: JsonObject = {}
    for name in ("EarTTS-high", "residual-high"):
        rows = population_rows[name]
        details = [
            _row_context(
                item, artifact_rows[item["artifact"]], response_positions[item["artifact"]]
            )
            for item in rows
        ]
        inventories[name] = {
            "definition": definitions[name],
            "membership": _corpus_membership(rows),
            "pattern_summary": _inventory_patterns(details),
            "rows": details,
        }

    return {
        "schema": "generation-frame-latency-step1-details-v1",
        "pad_token_id": PAD_TOKEN_ID,
        "artifact_count": len(artifact_rows),
        "artifact_ids": sorted(artifact_rows),
        "stage_timing_strata": {
            name: _corpus_stratum(rows, definitions[name]) for name, rows in strata_rows.items()
        },
        "pair_vs_sequential": pair_vs_sequential,
        "high_mode_inventories": inventories,
    }


def _population_report(rows: Sequence[JsonObject]) -> JsonObject:
    clocks: JsonObject = {}
    for clock in CLOCK_FIELDS:
        values = [
            value
            for normalized in rows
            if (value := _number(normalized["event"].get(clock))) is not None
        ]
        clocks[clock] = distribution_stats(values)
    return {
        "membership": _membership_manifest(rows),
        "clocks": clocks,
        "stage_timings": _stage_means(rows),
    }


def response_clustered_stats(
    rows: Sequence[JsonObject],
    clock_field: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonObject:
    clusters: dict[str, list[float]] = defaultdict(list)
    for normalized in rows:
        value = _number(normalized["event"].get(clock_field))
        response_id = normalized.get("response_group")
        if value is not None:
            clusters[str(response_id or "unclustered")].append(value)
    per_response = [
        {
            "response_id": response_id,
            "n": len(values),
            "mean_ms": sum(values) / len(values),
        }
        for response_id, values in sorted(clusters.items())
    ]
    all_values = [value for values in clusters.values() for value in values]
    if not all_values:
        return {
            "n_responses": 0,
            "n_rows": 0,
            "mean_ms": None,
            "per_response": [],
            "bootstrap_95_ci_ms": None,
            "bootstrap_seed": seed,
            "bootstrap_iterations": iterations,
        }
    cluster_values = list(clusters.values())
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(cluster_values) for _ in cluster_values]
        denominator = sum(len(values) for values in sample)
        bootstrapped.append(sum(sum(values) for values in sample) / denominator)
    return {
        "n_responses": len(clusters),
        "n_rows": len(all_values),
        "mean_ms": sum(all_values) / len(all_values),
        "per_response": per_response,
        "bootstrap_95_ci_ms": [
            nearest_rank(bootstrapped, 2.5),
            nearest_rank(bootstrapped, 97.5),
        ],
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
    }


def _decode_sample_count(event: JsonObject, ordinal: int) -> int:
    if event.get("encoding") != WIRE_ENCODING:
        raise ValueError(f"audio delta at event {ordinal} encoding is not pcm16")
    if event.get("sample_rate") != SAMPLE_RATE_HZ or event.get("channels") != WIRE_CHANNELS:
        raise ValueError(f"audio delta at event {ordinal} is not mono PCM16 at {SAMPLE_RATE_HZ} Hz")
    delta = event.get("delta")
    if not isinstance(delta, str):
        raise ValueError(f"audio delta at event {ordinal} has no base64 string")
    try:
        payload = base64.b64decode(delta, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"audio delta at event {ordinal} has invalid base64") from exc
    if len(payload) % 2:
        raise ValueError(f"audio delta at event {ordinal} has an odd PCM byte count")
    return len(payload) // 2


def extract_queue_responses(
    events: Sequence[JsonObject], rows: Sequence[JsonObject] | None = None
) -> tuple[str, dict[str, JsonObject], JsonObject]:
    """Return service-accepted queue observations and their clock label.

    Client events are filtered through the response lifecycle used by
    ``NemotronVoicechatLLMService``: deltas for a non-current or cancelling
    response are ignored before wire decoding.  Server-only traces are an
    explicitly diagnostic fallback and never invent a completion boundary.
    """

    responses: dict[str, JsonObject] = {}
    lifecycle_audit: JsonObject = {
        "wire_contract": "pcm16/22050Hz/mono",
        "accepted_delta_count": 0,
        "ignored_stale_delta_count": 0,
        "ignored_cancelling_delta_count": 0,
        "observed_interruption_clear_count": 0,
    }
    has_client_deltas = any(event.get("type") == "response.output_audio.delta" for event in events)
    if has_client_deltas:
        if not any(event.get("type") == "response.created" for event in events):
            raise ValueError("audio-delta artifact lacks response.created lifecycle evidence")
        current_response: str | None = None
        cancelling = False
        for ordinal, event in enumerate(events, 1):
            event_type = event.get("type")
            if event_type == "input_audio_buffer.speech_started" and event.get("source") == "typed":
                if current_response is not None:
                    cancelling = True
                    responses[current_response]["clear_events"].append(
                        {
                            "event_ordinal": ordinal,
                            "timestamp_s": _number(event.get("client_received_monotonic_s")),
                        }
                    )
                    lifecycle_audit["observed_interruption_clear_count"] += 1
                continue
            if event_type == "response.created":
                response_id = event.get("response_id")
                if not isinstance(response_id, str):
                    raise ValueError(f"response.created at event {ordinal} lacks response_id")
                if current_response is not None and current_response != response_id:
                    raise ValueError(f"overlapping responses at event {ordinal}")
                current_response = response_id
                cancelling = False
                responses.setdefault(
                    response_id,
                    {
                        "response_id": response_id,
                        "deltas": [],
                        "clear_events": [],
                        "done_time_s": None,
                        "done_observed": False,
                        "done_event_ordinal": None,
                        "clock_normative": True,
                    },
                )
                continue
            if event_type == "response.output_audio.delta":
                response_id = event.get("response_id")
                if response_id != current_response or cancelling:
                    # This is the runtime's intentional stale/cancelling-delta path.
                    key = (
                        "ignored_cancelling_delta_count"
                        if cancelling and response_id == current_response
                        else "ignored_stale_delta_count"
                    )
                    lifecycle_audit[key] += 1
                    continue
                timestamp = _number(event.get("client_received_monotonic_s"))
                if not isinstance(response_id, str) or timestamp is None:
                    raise ValueError(
                        f"audio delta at event {ordinal} lacks response_id or client receipt clock"
                    )
                response = responses[response_id]
                if response["deltas"] and timestamp < response["deltas"][-1]["timestamp_s"]:
                    raise ValueError(f"audio delta clock regresses at event {ordinal}")
                sample_count = _decode_sample_count(event, ordinal)
                if sample_count == 0:
                    continue
                lifecycle_audit["accepted_delta_count"] += 1
                response["deltas"].append(
                    {
                        "event_ordinal": ordinal,
                        "timestamp_s": timestamp,
                        "sample_count": sample_count,
                        "sample_rate_hz": SAMPLE_RATE_HZ,
                    }
                )
                continue
            if event_type == "response.done":
                response_id = event.get("response_id")
                if response_id != current_response:
                    continue
                timestamp = _number(event.get("client_received_monotonic_s"))
                if isinstance(response_id, str) and timestamp is not None:
                    response = responses[response_id]
                    response["done_time_s"] = timestamp
                    response["done_observed"] = True
                    response["done_event_ordinal"] = ordinal
                current_response = None
                cancelling = False
        clock = "client_received_monotonic_s (response.output_audio.delta)"
        lifecycle_audit["client_lifecycle_observed"] = True
    else:
        normalized_rows = list(rows) if rows is not None else _normalize_model_rows(events)
        for normalized in normalized_rows:
            if not normalized["delivered"]:
                continue
            event = normalized["event"]
            timestamp = _number(event.get("monotonic_s"))
            response_id = normalized.get("response_group")
            output_audio = event.get("output_audio")
            if (
                timestamp is None
                or not isinstance(response_id, str)
                or not isinstance(output_audio, dict)
            ):
                raise ValueError("delivered model_step lacks monotonic clock, response, or audio")
            samples = _number(output_audio.get("samples"))
            if samples is None:
                samples = (_number(output_audio.get("bytes")) or 0.0) / 2.0
            if samples <= 0 or not samples.is_integer():
                raise ValueError("delivered model_step has invalid PCM sample count")
            response = responses.setdefault(
                response_id,
                {
                    "response_id": response_id,
                    "deltas": [],
                    "clear_events": [],
                    "done_time_s": None,
                    "done_observed": False,
                    "done_event_ordinal": None,
                    "clock_normative": False,
                },
            )
            response["deltas"].append(
                {
                    "event_ordinal": normalized["event_ordinal"],
                    "timestamp_s": timestamp,
                    "sample_count": int(samples),
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                }
            )
            lifecycle_audit["accepted_delta_count"] += 1
        clock = "monotonic_s (delivered model_step fallback)"
        lifecycle_audit["client_lifecycle_observed"] = False
    return clock, dict(sorted(responses.items())), lifecycle_audit


def reconcile_audio_deltas(
    rows: Sequence[JsonObject], responses: dict[str, JsonObject]
) -> JsonObject:
    """Prove the one-to-one ordered mapping from delivered rows to deltas."""

    expected: list[tuple[str, int]] = []
    for normalized in rows:
        if not normalized["delivered"]:
            continue
        response_id = normalized.get("response_group")
        output_audio = normalized["event"].get("output_audio")
        samples = _number(output_audio.get("samples")) if isinstance(output_audio, dict) else None
        if not isinstance(response_id, str) or samples is None or not samples.is_integer():
            return {
                "provable": False,
                "reason": "delivered row lacks explicit response_id or integral sample count",
                "delivered_count": len(expected) + 1,
                "delta_count": sum(len(response["deltas"]) for response in responses.values()),
            }
        expected.append((response_id, int(samples)))
    observed_with_ordinals = [
        (int(delta["event_ordinal"]), response_id, int(delta["sample_count"]))
        for response_id, response in responses.items()
        for delta in response["deltas"]
    ]
    observed_with_ordinals.sort()
    observed = [(response_id, samples) for _, response_id, samples in observed_with_ordinals]
    reason = None
    if not expected:
        reason = "no delivered logical frames"
    elif len(expected) != len(observed):
        reason = "delivered/delta count mismatch"
    elif expected != observed:
        reason = "delivered/delta response order or sample-count mismatch"
    encoded = json.dumps(observed, separators=(",", ":")).encode("utf-8")
    return {
        "provable": reason is None,
        "reason": reason,
        "delivered_count": len(expected),
        "delta_count": len(observed),
        "ordered_membership_sha256": hashlib.sha256(encoded).hexdigest(),
    }


PLAYOUT_TRACE_SCHEMA = "nemotron_voicechat.playout.v1"
PLAYOUT_TRACE_TYPES = frozenset(
    {
        "voicechat.playout.config",
        "session.created.provenance",
        "response.created",
        "response.output_text.delta",
        "response.output_audio.delta",
        "response.done",
        "voicechat.metrics",
        "input_audio_buffer.speech_started",
        "voicechat.playout.interruption",
        "voicechat.playout.release",
        "voicechat.playout.downstream_push",
        "voicechat.playout.trace_status",
    }
)
PLAYOUT_TRACE_CLOCKS = (
    "client_configured_monotonic_s",
    "client_received_monotonic_s",
    "client_release_monotonic_s",
    "client_downstream_push_monotonic_s",
    "client_interruption_monotonic_s",
    "trace_closed_monotonic_s",
)
PLAYOUT_TRACE_CLOCK_BY_TYPE = {
    "voicechat.playout.config": "client_configured_monotonic_s",
    "session.created.provenance": "client_received_monotonic_s",
    "response.created": "client_received_monotonic_s",
    "response.output_text.delta": "client_received_monotonic_s",
    "response.output_audio.delta": "client_received_monotonic_s",
    "response.done": "client_received_monotonic_s",
    "voicechat.metrics": "client_received_monotonic_s",
    "input_audio_buffer.speech_started": "client_received_monotonic_s",
    "voicechat.playout.interruption": "client_interruption_monotonic_s",
    "voicechat.playout.release": "client_release_monotonic_s",
    "voicechat.playout.downstream_push": "client_downstream_push_monotonic_s",
    "voicechat.playout.trace_status": "trace_closed_monotonic_s",
}

_PLAYOUT_TRACE_FIELDS_BY_TYPE = {
    "voicechat.playout.config": frozenset(
        {
            "trace_schema",
            "type",
            "client_configured_monotonic_s",
            "configured_prebuffer_ms",
            "sample_rate_hz",
            "channels",
        }
    ),
    "session.created.provenance": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "session_id",
            "session_created_sha256",
            "runtime_provenance",
        }
    ),
    "response.created": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "response_id",
            "turn_id",
        }
    ),
    "response.output_text.delta": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "response_id",
            "ordinal",
            "delta",
        }
    ),
    "response.output_audio.delta": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "response_id",
            "ordinal",
            "sample_count",
            "encoding",
            "sample_rate",
            "sample_rate_hz",
            "channels",
            "delta",
        }
    ),
    "response.done": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "response_id",
            "status",
        }
    ),
    "input_audio_buffer.speech_started": frozenset(
        {
            "trace_schema",
            "type",
            "client_received_monotonic_s",
            "source",
            "response_id",
        }
    ),
    "voicechat.playout.interruption": frozenset(
        {
            "trace_schema",
            "type",
            "client_interruption_monotonic_s",
            "response_id",
        }
    ),
    "voicechat.playout.release": frozenset(
        {
            "trace_schema",
            "type",
            "client_release_monotonic_s",
            "response_id",
            "reason",
            "ordinals",
            "frames_released",
            "frames_cleared",
            "sample_count",
        }
    ),
    "voicechat.playout.downstream_push": frozenset(
        {
            "trace_schema",
            "type",
            "client_downstream_push_monotonic_s",
            "response_id",
            "ordinal",
            "sample_count",
            "sample_rate_hz",
            "channels",
        }
    ),
    "voicechat.playout.trace_status": frozenset(
        {
            "trace_schema",
            "type",
            "trace_closed_monotonic_s",
            "valid",
            "invalid_reason",
            "error",
            "dropped_records",
        }
    ),
}
_PLAYOUT_METRIC_COMMON_FIELDS = frozenset(
    {
        "trace_schema",
        "type",
        "client_received_monotonic_s",
        "event_id",
        "session_id",
        "frame",
        "transport_frame",
        "inference_ms",
        "server_step_ms",
        "budget_ms",
        "over_budget",
        "input_audio",
        "output_audio",
        "turn_state",
        "step9_capture_fallback",
    }
)
_PLAYOUT_ACTIVE_METRIC_FIELDS = _PLAYOUT_METRIC_COMMON_FIELDS | {
    "model_output_audio",
    "audio_delivered",
    "turn_id",
    "response_id",
}


def _playout_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_playout_record_schema(event: JsonObject, event_ordinal: int) -> None:
    event_type = event.get("type")
    if event_type == "voicechat.metrics":
        expected_fields = (
            _PLAYOUT_ACTIVE_METRIC_FIELDS
            if "audio_delivered" in event
            else _PLAYOUT_METRIC_COMMON_FIELDS
        )
    else:
        expected_fields = _PLAYOUT_TRACE_FIELDS_BY_TYPE.get(event_type)
    if expected_fields is None or set(event) != expected_fields:
        raise ValueError(f"playout trace field schema mismatch at event {event_ordinal}")

    if event_type == "session.created.provenance":
        provenance = event.get("runtime_provenance")
        source_hashes = provenance.get("source_sha256") if isinstance(provenance, dict) else None
        session_hash = event.get("session_created_sha256")
        image_id = provenance.get("runtime_image_id") if isinstance(provenance, dict) else None
        try:
            hash_fields_valid = (
                isinstance(session_hash, str)
                and len(session_hash) == 64
                and int(session_hash, 16) >= 0
                and isinstance(image_id, str)
                and image_id.startswith("sha256:")
                and len(image_id) == 71
                and int(image_id[7:], 16) >= 0
                and isinstance(source_hashes, dict)
                and bool(source_hashes)
                and all(
                    isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0
                    for value in source_hashes.values()
                )
            )
        except ValueError:
            hash_fields_valid = False
        if (
            not isinstance(event.get("session_id"), str)
            or not event["session_id"]
            or not isinstance(provenance, dict)
            or not isinstance(provenance.get("runtime_image"), str)
            or not provenance["runtime_image"]
            or not isinstance(provenance.get("runtime_contract"), str)
            or not provenance["runtime_contract"]
            or not isinstance(provenance.get("semantic_environment"), dict)
            or not hash_fields_valid
        ):
            raise ValueError(
                f"session.created provenance schema mismatch at event {event_ordinal}"
            )

    response_id = event.get("response_id")
    response_types = {
        "response.created",
        "response.output_text.delta",
        "response.output_audio.delta",
        "response.done",
        "voicechat.playout.interruption",
        "voicechat.playout.release",
        "voicechat.playout.downstream_push",
    }
    if event_type in response_types and (
        not isinstance(response_id, str) or not response_id
    ):
        raise ValueError(f"playout trace response_id schema mismatch at event {event_ordinal}")
    if event_type == "response.created" and not (
        event.get("turn_id") is None or isinstance(event.get("turn_id"), str)
    ):
        raise ValueError(f"response.created turn_id schema mismatch at event {event_ordinal}")
    if event_type == "response.output_text.delta" and (
        not _playout_int(event.get("ordinal"), minimum=1)
        or not isinstance(event.get("delta"), str)
        or not event["delta"]
    ):
        raise ValueError(f"text delta field schema mismatch at event {event_ordinal}")
    if event_type == "input_audio_buffer.speech_started" and (
        event.get("source") != "typed"
        or not (response_id is None or isinstance(response_id, str))
    ):
        raise ValueError(f"typed speech-start schema mismatch at event {event_ordinal}")
    if event_type == "response.done" and event.get("status") not in {
        None,
        "completed",
        "cancelled",
        "failed",
    }:
        raise ValueError(f"response.done status schema mismatch at event {event_ordinal}")
    if event_type == "response.output_audio.delta" and (
        not _playout_int(event.get("ordinal"), minimum=1)
        or not _playout_int(event.get("sample_count"), minimum=1)
        or event.get("encoding") != WIRE_ENCODING
        or event.get("sample_rate") != SAMPLE_RATE_HZ
        or event.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or event.get("channels") != WIRE_CHANNELS
        or not isinstance(event.get("delta"), str)
    ):
        raise ValueError(f"audio delta field schema mismatch at event {event_ordinal}")
    if event_type == "voicechat.playout.release":
        ordinals = event.get("ordinals")
        if (
            event.get("reason") not in {"threshold", "done", "interruption-clear"}
            or not isinstance(ordinals, list)
            or not all(_playout_int(ordinal, minimum=1) for ordinal in ordinals)
            or not _playout_int(event.get("frames_released"))
            or not _playout_int(event.get("frames_cleared"))
            or not _playout_int(event.get("sample_count"))
        ):
            raise ValueError(f"playout release field schema mismatch at event {event_ordinal}")
    if event_type == "voicechat.playout.downstream_push" and (
        not _playout_int(event.get("ordinal"), minimum=1)
        or not _playout_int(event.get("sample_count"), minimum=1)
        or event.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or event.get("channels") != WIRE_CHANNELS
    ):
        raise ValueError(f"downstream push field schema mismatch at event {event_ordinal}")
    if event_type == "voicechat.metrics":
        if (
            not isinstance(event.get("event_id"), str)
            or not isinstance(event.get("session_id"), str)
            or not _playout_int(event.get("transport_frame"))
            or _number(event.get("inference_ms")) is None
            or _number(event.get("server_step_ms")) is None
            or _number(event.get("budget_ms")) is None
            or type(event.get("over_budget")) is not bool
            or not isinstance(event.get("input_audio"), dict)
            or not isinstance(event.get("output_audio"), dict)
            or not isinstance(event.get("turn_state"), dict)
            or not isinstance(event.get("step9_capture_fallback"), dict)
        ):
            raise ValueError(f"voicechat.metrics field schema mismatch at event {event_ordinal}")
        if "audio_delivered" in event and (
            # Real sessions begin at frame 0 (first model frame of the
            # session); only negative frames are malformed.
            not _playout_int(event.get("frame"), minimum=0)
            or type(event.get("audio_delivered")) is not bool
            or not isinstance(event.get("model_output_audio"), dict)
            or not (
                event.get("turn_id") is None or isinstance(event.get("turn_id"), str)
            )
            or not (
                event.get("response_id") is None
                or isinstance(event.get("response_id"), str)
            )
        ):
            raise ValueError(f"active voicechat.metrics schema mismatch at event {event_ordinal}")
        if "audio_delivered" not in event and event.get("frame") is not None:
            raise ValueError(f"idle voicechat.metrics schema mismatch at event {event_ordinal}")
    if event_type == "voicechat.playout.trace_status" and (
        event.get("valid") is not True
        or event.get("invalid_reason") is not None
        or event.get("error") is not None
        or type(event.get("dropped_records")) is not int
        or event.get("dropped_records") != 0
    ):
        raise ValueError(f"terminal trace status is inconsistent at event {event_ordinal}")


def validate_observed_playout_trace(events: Sequence[JsonObject]) -> JsonObject:
    """Reconcile an observed Step-2 trace against the service state machine."""

    schema_observed = any(event.get("trace_schema") is not None for event in events)
    if not schema_observed:
        return {"observed": False, "valid": None, "schema": None}
    traced = [
        (event_ordinal, event)
        for event_ordinal, event in enumerate(events, 1)
        if event.get("type") in PLAYOUT_TRACE_TYPES or event.get("trace_schema") is not None
    ]
    for event_ordinal, event in traced:
        if event.get("trace_schema") != PLAYOUT_TRACE_SCHEMA:
            raise ValueError(f"playout trace schema mismatch at event {event_ordinal}")
        event_type = event.get("type")
        if event_type not in PLAYOUT_TRACE_CLOCK_BY_TYPE:
            raise ValueError(f"unknown playout trace record at event {event_ordinal}")
        _validate_playout_record_schema(event, event_ordinal)
        required_clock = PLAYOUT_TRACE_CLOCK_BY_TYPE[event_type]
        present_clocks = [field for field in PLAYOUT_TRACE_CLOCKS if field in event]
        if present_clocks != [required_clock]:
            raise ValueError(
                f"playout trace clock schema mismatch at event {event_ordinal}: "
                f"required={required_clock} present={present_clocks}"
            )
        if _number(event[required_clock]) is None:
            raise ValueError(f"playout trace clock is not finite at event {event_ordinal}")

    configurations = [
        (event_ordinal, event)
        for event_ordinal, event in traced
        if event.get("type") == "voicechat.playout.config"
    ]
    if len(configurations) != 1 or configurations[0] != traced[0]:
        raise ValueError("playout trace requires one initial playout config record")
    _, configuration = configurations[0]
    configured_prebuffer_ms = configuration.get("configured_prebuffer_ms")
    if (
        not isinstance(configured_prebuffer_ms, int)
        or isinstance(configured_prebuffer_ms, bool)
        or configured_prebuffer_ms < 0
        or configuration.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or configuration.get("channels") != WIRE_CHANNELS
    ):
        raise ValueError("playout trace has invalid configured prebuffer identity")

    statuses = [
        (event_ordinal, event)
        for event_ordinal, event in traced
        if event.get("type") == "voicechat.playout.trace_status"
    ]
    if len(statuses) != 1 or statuses[0] != traced[-1]:
        raise ValueError("playout trace requires one terminal trace_status record")
    _, terminal = statuses[0]
    responses: dict[str, JsonObject] = {}
    current_response: str | None = None
    last_clock: float | None = None
    pushes = 0
    releases = 0

    def state_for(response_id: object, event_ordinal: int) -> JsonObject:
        if not isinstance(response_id, str) or response_id != current_response:
            raise ValueError(f"orphaned playout record at event {event_ordinal}")
        return responses[response_id]

    def finish_response(response_id: str, event_ordinal: int) -> None:
        state = responses[response_id]
        if not state["done"]:
            raise ValueError(
                f"response {response_id} lacks response.done before event {event_ordinal}"
            )
        if state["held"]:
            raise ValueError(f"response {response_id} lacks required done flush")
        if state["pending_push"]:
            raise ValueError(f"response {response_id} lacks downstream pushes")
        if state["interruption_pending"]:
            raise ValueError(f"response {response_id} lacks interruption-clear release")
        accepted = set(state["samples"])
        accounted = state["pushed"] | state["cleared"]
        if accepted != accounted:
            raise ValueError(f"response {response_id} has unaccounted audio ordinals")

    for event_ordinal, event in traced:
        event_type = event.get("type")
        timestamp = _number(event[PLAYOUT_TRACE_CLOCK_BY_TYPE[event_type]])
        assert timestamp is not None
        if last_clock is not None and timestamp + 1e-12 < last_clock:
            raise ValueError(f"playout trace clock regresses at event {event_ordinal}")
        last_clock = timestamp

        if current_response is not None:
            active = responses[current_response]
            if active["pending_push"] and event_type != "voicechat.playout.downstream_push":
                raise ValueError(
                    f"playout release/push ordering is not atomic at event {event_ordinal}"
                )
            if active["interruption_pending"] and event_type != "voicechat.playout.release":
                raise ValueError(
                    f"playout interruption/clear ordering is not atomic at event {event_ordinal}"
                )
            if active["done"] and active["held"] and event_type != "voicechat.playout.release":
                raise ValueError(
                    f"playout done/release ordering is not atomic at event {event_ordinal}"
                )

        if event_type == "voicechat.playout.config":
            continue

        if event_type == "response.created":
            response_id = event.get("response_id")
            if not isinstance(response_id, str):
                raise ValueError(f"response.created at event {event_ordinal} lacks response_id")
            if current_response is not None:
                finish_response(current_response, event_ordinal)
            if response_id in responses:
                raise ValueError(f"duplicate response.created for {response_id}")
            current_response = response_id
            responses[response_id] = {
                "next_ordinal": 1,
                "next_text_ordinal": 1,
                "samples": {},
                "held": [],
                "pending_push": [],
                "pushed": set(),
                "cleared": set(),
                "threshold_released": False,
                "done": False,
                "done_release": False,
                "interruption_pending": False,
                "interruption_cleared": False,
                "cancelling": False,
            }
            continue
        if event_type == "response.output_text.delta":
            state = state_for(event.get("response_id"), event_ordinal)
            if state["done"] or state["cancelling"]:
                raise ValueError(f"text delta contradicts lifecycle at event {event_ordinal}")
            if event.get("ordinal") != state["next_text_ordinal"]:
                raise ValueError(f"text delta ordinal mismatch at event {event_ordinal}")
            state["next_text_ordinal"] += 1
            continue
        if event_type == "response.output_audio.delta":
            state = state_for(event.get("response_id"), event_ordinal)
            if state["done"] or state["cancelling"]:
                raise ValueError(f"audio delta contradicts lifecycle at event {event_ordinal}")
            ordinal = event.get("ordinal")
            if ordinal != state["next_ordinal"]:
                raise ValueError(f"audio delta ordinal mismatch at event {event_ordinal}")
            decoded_samples = _decode_sample_count(event, event_ordinal)
            if event.get("sample_count") != decoded_samples:
                raise ValueError(f"audio delta sample_count mismatch at event {event_ordinal}")
            if event.get("sample_rate_hz") != SAMPLE_RATE_HZ:
                raise ValueError(f"audio delta sample_rate_hz mismatch at event {event_ordinal}")
            state["next_ordinal"] += 1
            state["samples"][ordinal] = decoded_samples
            if state["threshold_released"]:
                state["pending_push"].append(ordinal)
            else:
                state["held"].append(ordinal)
            continue
        interruption = event_type == "voicechat.playout.interruption" or (
            event_type == "input_audio_buffer.speech_started"
            and event.get("source") == "typed"
        )
        if interruption:
            if current_response is None and event.get("response_id") is None:
                continue
            if event.get("response_id") is None and current_response is not None:
                if responses[current_response]["done"]:
                    finish_response(current_response, event_ordinal)
                    current_response = None
                    continue
            state = state_for(event.get("response_id"), event_ordinal)
            if state["done"]:
                raise ValueError(f"interruption follows response.done at event {event_ordinal}")
            if state["cancelling"]:
                raise ValueError(f"duplicate interruption at event {event_ordinal}")
            state["interruption_pending"] = True
            state["cancelling"] = True
            continue
        if event_type == "response.done":
            state = state_for(event.get("response_id"), event_ordinal)
            if state["done"]:
                raise ValueError(f"duplicate response.done at event {event_ordinal}")
            state["done"] = True
            continue
        if event_type == "voicechat.playout.release":
            state = state_for(event.get("response_id"), event_ordinal)
            reason = event.get("reason")
            if state["cancelling"] and reason != "interruption-clear":
                raise ValueError(f"playout release follows cancellation at event {event_ordinal}")
            if reason not in {"threshold", "done", "interruption-clear"}:
                raise ValueError(f"invalid playout release reason at event {event_ordinal}")
            ordinals = event.get("ordinals")
            if not isinstance(ordinals, list) or not all(
                isinstance(ordinal, int) for ordinal in ordinals
            ):
                raise ValueError(f"invalid release ordinals at event {event_ordinal}")
            if ordinals != state["held"]:
                raise ValueError(
                    f"release ordinals contradict held frames at event {event_ordinal}"
                )
            released_samples = sum(state["samples"][ordinal] for ordinal in ordinals)
            if reason == "interruption-clear":
                if not state["interruption_pending"] or state["interruption_cleared"]:
                    raise ValueError(f"orphaned interruption clear at event {event_ordinal}")
                if event.get("frames_released") != 0:
                    raise ValueError(f"interruption clear released frames at event {event_ordinal}")
                if event.get("frames_cleared") != len(ordinals):
                    raise ValueError(f"interruption clear count mismatch at event {event_ordinal}")
                if event.get("sample_count") != 0:
                    raise ValueError(
                        f"interruption clear sample count mismatch at event {event_ordinal}"
                    )
                state["cleared"].update(ordinals)
                state["interruption_pending"] = False
                state["interruption_cleared"] = True
            else:
                if event.get("frames_released") != len(ordinals):
                    raise ValueError(f"release frame count mismatch at event {event_ordinal}")
                if event.get("frames_cleared") != 0:
                    raise ValueError(f"release clear count mismatch at event {event_ordinal}")
                if event.get("sample_count") != released_samples:
                    raise ValueError(f"release sample count mismatch at event {event_ordinal}")
                if reason == "threshold":
                    if state["threshold_released"] or state["done"]:
                        raise ValueError(
                            f"contradictory threshold release at event {event_ordinal}"
                        )
                    state["threshold_released"] = True
                    if configured_prebuffer_ms == 0:
                        if len(ordinals) != 1:
                            raise ValueError(
                                "zero-prebuffer threshold release mismatch "
                                f"at event {event_ordinal}"
                            )
                    else:
                        released_ms = released_samples * 1000.0 / SAMPLE_RATE_HZ
                        prior_ms = (
                            sum(state["samples"][ordinal] for ordinal in ordinals[:-1])
                            * 1000.0
                            / SAMPLE_RATE_HZ
                        )
                        if (
                            released_ms + 1e-9 < configured_prebuffer_ms
                            or prior_ms + 1e-9 >= configured_prebuffer_ms
                        ):
                            raise ValueError(
                                f"threshold release contradicts configured prebuffer "
                                f"at event {event_ordinal}"
                            )
                elif not state["done"] or state["threshold_released"]:
                    raise ValueError(f"orphaned done release at event {event_ordinal}")
                else:
                    state["done_release"] = True
                state["pending_push"].extend(ordinals)
            state["held"] = []
            releases += 1
            continue
        if event_type == "voicechat.playout.downstream_push":
            state = state_for(event.get("response_id"), event_ordinal)
            ordinal = event.get("ordinal")
            if not state["pending_push"] or ordinal != state["pending_push"].pop(0):
                raise ValueError(f"orphaned or reordered downstream push at event {event_ordinal}")
            if ordinal in state["pushed"]:
                raise ValueError(f"duplicate downstream push at event {event_ordinal}")
            if event.get("sample_count") != state["samples"][ordinal]:
                raise ValueError(f"downstream push sample count mismatch at event {event_ordinal}")
            if event.get("sample_rate_hz") != SAMPLE_RATE_HZ or event.get("channels") != 1:
                raise ValueError(f"downstream push format mismatch at event {event_ordinal}")
            state["pushed"].add(ordinal)
            pushes += 1

    if current_response is not None:
        finish_response(current_response, len(events) + 1)
    return {
        "observed": True,
        "valid": True,
        "schema": PLAYOUT_TRACE_SCHEMA,
        "response_count": len(responses),
        "release_count": releases,
        "downstream_push_count": pushes,
        "terminal_status": terminal,
    }


def replay_response(
    deltas: Sequence[JsonObject],
    reserve_ms: float,
    *,
    done_time_s: float | None = None,
    done_observed: bool | None = None,
    done_event_ordinal: int | None = None,
    clear_events: Sequence[JsonObject] = (),
    interruption_point: str | None = None,
    interruption_ordinal: int | None = None,
) -> JsonObject:
    """Replay playout with explicit service-buffer/release ordering points."""

    if reserve_ms < 0:
        raise ValueError("reserve must be non-negative")
    if done_observed is None:
        done_observed = done_time_s is not None
    interruption_points = {
        None,
        "pre-arrival",
        "post-buffer/pre-release",
        "at-release/post-detach/pre-first-push",
        "post-release-frame",
    }
    if interruption_point not in interruption_points:
        raise ValueError("unknown interruption point")
    if (interruption_point is None) != (interruption_ordinal is None):
        raise ValueError("interruption point and ordinal must be supplied together")
    if interruption_ordinal is not None and not 0 <= interruption_ordinal < len(deltas):
        raise ValueError("interruption ordinal is outside the delta sequence")
    held_frames: list[tuple[int, float]] = []
    downstream_queue_ms = 0.0
    released = False
    release_reason: str | None = None
    release_time_s: float | None = None
    first_time_s: float | None = None
    previous_arrival_s: float | None = None
    starvation_ms = 0.0
    simulated_starvation_events = 0
    processed = 0
    interrupted = False
    released_downstream_ms = 0.0
    released_downstream_ms_at_interrupt: float | None = None
    downstream_queue_ms_at_interrupt: float | None = None
    release_burst_after_interrupt_ms = 0.0
    release_trigger_ordinal: int | None = None
    observed_clear_count = 0
    observed_clear_time_s: float | None = None
    observed_clear_event_ordinal: int | None = None
    cleared_service_held_media_ms = 0.0
    ordered_clears = sorted(clear_events, key=lambda event: int(event.get("event_ordinal", -1)))
    clear_index = 0

    def advance_queue_to(timestamp_s: float) -> None:
        nonlocal downstream_queue_ms, starvation_ms, simulated_starvation_events
        if released and previous_arrival_s is not None:
            elapsed_ms = (timestamp_s - previous_arrival_s) * 1000.0
            if elapsed_ms < -1e-9:
                raise ValueError("interruption clock precedes the previous audio delta")
            if elapsed_ms > downstream_queue_ms + 1e-9:
                starvation_ms += elapsed_ms - downstream_queue_ms
                downstream_queue_ms = 0.0
                simulated_starvation_events += 1
            else:
                downstream_queue_ms -= elapsed_ms

    def interrupt(*, observed_clear: JsonObject | None = None) -> None:
        nonlocal interrupted, held_frames, released_downstream_ms_at_interrupt
        nonlocal downstream_queue_ms_at_interrupt, observed_clear_count
        nonlocal observed_clear_time_s, observed_clear_event_ordinal
        nonlocal cleared_service_held_media_ms, release_reason
        if interrupted:
            return
        cleared_service_held_media_ms = sum(duration for _, duration in held_frames)
        interrupted = True
        held_frames = []
        released_downstream_ms_at_interrupt = released_downstream_ms
        downstream_queue_ms_at_interrupt = downstream_queue_ms
        if observed_clear is not None:
            observed_clear_count += 1
            observed_clear_time_s = _number(observed_clear.get("timestamp_s"))
            observed_clear_event_ordinal = int(observed_clear["event_ordinal"])
            if not released:
                release_reason = "interruption-clear"

    def apply_observed_clears_before(event_ordinal: int | None) -> bool:
        nonlocal clear_index
        while clear_index < len(ordered_clears):
            clear = ordered_clears[clear_index]
            clear_ordinal = int(clear.get("event_ordinal", -1))
            if event_ordinal is not None and clear_ordinal >= event_ordinal:
                break
            clear_time = _number(clear.get("timestamp_s"))
            if clear_time is not None:
                advance_queue_to(clear_time)
            clear_index += 1
            interrupt(observed_clear=clear)
            return True
        return False

    def push_release_burst(frames: Sequence[tuple[int, float]], trigger_ordinal: int) -> None:
        nonlocal downstream_queue_ms, released_downstream_ms
        nonlocal release_burst_after_interrupt_ms
        if (
            not interrupted
            and interruption_point == "at-release/post-detach/pre-first-push"
            and interruption_ordinal == trigger_ordinal
        ):
            # _release_playout_buffer() has already detached its local list.
            # The clear cannot retract or stop any frame in this burst.
            interrupt()
        for logical_ordinal, media_ms in frames:
            downstream_queue_ms += media_ms
            released_downstream_ms += media_ms
            if (
                not interrupted
                and interruption_point == "post-release-frame"
                and interruption_ordinal == logical_ordinal
            ):
                interrupt()
            elif interrupted:
                # _release_playout_buffer() owns a detached local list.  A
                # concurrent clear cannot retract or stop its remaining pushes.
                release_burst_after_interrupt_ms += media_ms

    for index, delta in enumerate(deltas):
        timestamp = _number(delta.get("timestamp_s"))
        samples = _number(delta.get("sample_count"))
        sample_rate = _number(delta.get("sample_rate_hz"))
        event_ordinal = delta.get("event_ordinal")
        if (
            timestamp is None
            or samples is None
            or samples <= 0
            or sample_rate is None
            or sample_rate <= 0
        ):
            raise ValueError(f"invalid delta at replay ordinal {index}")
        if previous_arrival_s is not None and timestamp < previous_arrival_s:
            raise ValueError("delta timestamps are not monotonic")
        if apply_observed_clears_before(
            int(event_ordinal) if isinstance(event_ordinal, int) else None
        ):
            break
        if interruption_point == "pre-arrival" and interruption_ordinal == index:
            advance_queue_to(timestamp)
            interrupt()
            break
        if first_time_s is None:
            first_time_s = timestamp
        advance_queue_to(timestamp)
        media_ms = samples / sample_rate * 1000.0
        previous_arrival_s = timestamp
        processed += 1
        if released:
            push_release_burst(((index, media_ms),), index)
            if interrupted:
                break
            continue
        held_frames.append((index, media_ms))
        if interruption_point == "post-buffer/pre-release" and interruption_ordinal == index:
            interrupt()
            break
        held_ms = sum(duration for _, duration in held_frames)
        if held_ms + 1e-9 >= reserve_ms:
            released = True
            release_reason = "threshold"
            release_time_s = timestamp
            release_trigger_ordinal = index
            burst, held_frames = held_frames, []
            push_release_burst(burst, index)
            if interrupted:
                break
    if not interrupted:
        apply_observed_clears_before(done_event_ordinal)
    if not interrupted and not released and deltas and done_observed:
        if done_time_s is None:
            raise ValueError("observed response.done lacks a receipt timestamp")
        last_timestamp = _number(deltas[-1].get("timestamp_s"))
        if last_timestamp is not None and done_time_s + 1e-12 < last_timestamp:
            raise ValueError("response.done precedes the final audio delta")
        released = True
        release_reason = "response.done"
        release_time_s = done_time_s
        release_trigger_ordinal = len(deltas) - 1
        burst, held_frames = held_frames, []
        push_release_burst(burst, release_trigger_ordinal)
    elif not interrupted and not released and deltas and not done_observed:
        release_reason = "done_flush_unknown"
    added_release_delay_ms = (
        (release_time_s - first_time_s) * 1000.0
        if release_time_s is not None and first_time_s is not None
        else None
    )
    return {
        "reserve_ms": float(reserve_ms),
        "delta_count": len(deltas),
        "processed_delta_count": processed,
        "released": released,
        "release_reason": release_reason,
        "release_time_s": release_time_s,
        "added_release_delay_ms": added_release_delay_ms,
        "simulated_starvation_events": simulated_starvation_events,
        "starvation_evidence": "simulation-only; not browser underrun evidence",
        "starvation_ms": starvation_ms,
        "starved": starvation_ms > 1e-9,
        "service_held_media_ms_at_stop": sum(duration for _, duration in held_frames),
        "downstream_queue_ms_at_stop": downstream_queue_ms,
        "released_downstream_ms": released_downstream_ms,
        "released_downstream_ms_at_interrupt": released_downstream_ms_at_interrupt,
        "downstream_queue_ms_at_interrupt": downstream_queue_ms_at_interrupt,
        "release_burst_after_interrupt_ms": release_burst_after_interrupt_ms,
        "release_trigger_ordinal": release_trigger_ordinal,
        "observed_clear_count": observed_clear_count,
        "observed_clear_time_s": observed_clear_time_s,
        "observed_clear_event_ordinal": observed_clear_event_ordinal,
        "cleared_service_held_media_ms": cleared_service_held_media_ms,
        "interrupted": interrupted,
        "interruption_point": interruption_point,
        "interruption_ordinal": interruption_ordinal,
    }


def max_prefix_debt(deltas: Sequence[JsonObject]) -> JsonObject:
    """Return informational zero-reserve maximum cumulative queue debt."""

    cumulative = 0.0
    maximum = 0.0
    maximum_after_ordinal: int | None = None
    for index in range(1, len(deltas)):
        previous = deltas[index - 1]
        current = deltas[index]
        previous_timestamp = _number(previous.get("timestamp_s"))
        timestamp = _number(current.get("timestamp_s"))
        samples = _number(previous.get("sample_count"))
        sample_rate = _number(previous.get("sample_rate_hz"))
        if (
            previous_timestamp is None
            or timestamp is None
            or samples is None
            or sample_rate is None
            or sample_rate <= 0
        ):
            raise ValueError("invalid delta in prefix-debt sequence")
        gap_ms = (timestamp - previous_timestamp) * 1000.0
        media_ms = samples / sample_rate * 1000.0
        cumulative += gap_ms - media_ms
        if cumulative > maximum:
            maximum = cumulative
            maximum_after_ordinal = index
    return {
        "max_prefix_debt_ms": maximum,
        "maximum_after_delta_ordinal": maximum_after_ordinal,
        "definition": "max prefix cumulative sum(gap_ms - preceding_chunk_media_ms)",
    }


def queue_diagnostics(responses: dict[str, JsonObject], reserves_ms: Sequence[float]) -> JsonObject:
    response_reports: list[JsonObject] = []
    for response_id, response in responses.items():
        deltas = response["deltas"]
        reserve_reports = []
        for reserve_ms in reserves_ms:
            replay = replay_response(
                deltas,
                reserve_ms,
                done_time_s=response.get("done_time_s"),
                done_observed=response.get("done_observed", False),
                done_event_ordinal=response.get("done_event_ordinal"),
                clear_events=response.get("clear_events", ()),
            )
            interruption_specs = [("pre-arrival", index) for index in range(len(deltas))]
            if reserve_ms > 0:
                held_ms = 0.0
                for index, delta in enumerate(deltas):
                    held_ms += delta["sample_count"] / delta["sample_rate_hz"] * 1000.0
                    interruption_specs.append(("post-buffer/pre-release", index))
                    if held_ms + 1e-9 >= reserve_ms:
                        break
            if replay["released"]:
                interruption_specs.append(
                    (
                        "at-release/post-detach/pre-first-push",
                        replay["release_trigger_ordinal"],
                    )
                )
                interruption_specs.extend(
                    ("post-release-frame", index) for index in range(len(deltas))
                )
            replay["interruption_sweep"] = [
                replay_response(
                    deltas,
                    reserve_ms,
                    done_time_s=response.get("done_time_s"),
                    done_observed=response.get("done_observed", False),
                    done_event_ordinal=response.get("done_event_ordinal"),
                    clear_events=response.get("clear_events", ()),
                    interruption_point=point,
                    interruption_ordinal=index,
                )
                for point, index in interruption_specs
            ]
            reserve_reports.append(replay)
        response_reports.append(
            {
                "response_id": response_id,
                "delta_count": len(deltas),
                "total_samples": sum(delta["sample_count"] for delta in deltas),
                "done_time_s": response.get("done_time_s"),
                "done_observed": response.get("done_observed", False),
                "done_event_ordinal": response.get("done_event_ordinal"),
                "clear_events": response.get("clear_events", []),
                "normative": response.get("clock_normative", False),
                "zero_reserve_prefix_debt": max_prefix_debt(deltas),
                "reserve_replays": reserve_reports,
            }
        )
    sweep = []
    for reserve_ms in reserves_ms:
        runs = [
            next(
                replay
                for replay in response["reserve_replays"]
                if replay["reserve_ms"] == float(reserve_ms)
            )
            for response in response_reports
        ]
        delays = [
            run["added_release_delay_ms"]
            for run in runs
            if run["added_release_delay_ms"] is not None
        ]
        sweep.append(
            {
                "reserve_ms": float(reserve_ms),
                "responses": len(runs),
                "responses_starved": sum(run["starved"] for run in runs),
                "simulated_starvation_events": sum(
                    run["simulated_starvation_events"] for run in runs
                ),
                "starvation_evidence": "simulation-only; not browser underrun evidence",
                "total_starvation_ms": sum(run["starvation_ms"] for run in runs),
                "mean_added_release_delay_ms": (sum(delays) / len(delays) if delays else None),
                "responses_not_released": sum(not run["released"] for run in runs),
            }
        )
    maximum_debt = max(
        (
            response["zero_reserve_prefix_debt"]["max_prefix_debt_ms"]
            for response in response_reports
        ),
        default=0.0,
    )
    return {
        "normative": bool(response_reports)
        and all(response["normative"] for response in response_reports),
        "response_count": len(response_reports),
        "responses": response_reports,
        "reserve_sweep": sweep,
        "zero_reserve_max_prefix_debt_ms": maximum_debt,
    }


def structural_gate(
    timestamp_groups: dict[str, Sequence[float]],
    *,
    max_gap_ms: float = 160.0,
    normative: bool = True,
) -> JsonObject:
    """Evaluate the literal K=12 rule within each response."""

    if max_gap_ms < 0:
        raise ValueError("max-gap-ms must be non-negative")
    windows: list[JsonObject] = []
    gaps: list[JsonObject] = []
    for response_id, raw_timestamps in sorted(timestamp_groups.items()):
        timestamps = [float(timestamp) for timestamp in raw_timestamps]
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"timestamps for {response_id} are not monotonic")
        for index in range(len(timestamps) - 1):
            gaps.append(
                {
                    "response_id": response_id,
                    "start_ordinal": index,
                    "gap_ms": (timestamps[index + 1] - timestamps[index]) * 1000.0,
                }
            )
        for index in range(max(0, len(timestamps) - 11)):
            duration_ms = (timestamps[index + 11] - timestamps[index]) * 1000.0
            windows.append(
                {
                    "response_id": response_id,
                    "start_ordinal": index,
                    "end_ordinal": index + 11,
                    "duration_ms": duration_ms,
                    "passed": duration_ms <= 960.0 + 1e-9,
                }
            )
    worst_window = max(windows, key=lambda item: item["duration_ms"], default=None)
    worst_gap = max(gaps, key=lambda item: item["gap_ms"], default=None)
    failing_windows = sum(not window["passed"] for window in windows)
    max_gap = worst_gap["gap_ms"] if worst_gap is not None else None
    max_gap_passed = max_gap is None or max_gap <= max_gap_ms + 1e-9
    if not windows:
        outcome = "INSUFFICIENT"
    elif failing_windows == 0 and max_gap_passed:
        outcome = "ACCEPT"
    else:
        outcome = "REJECT"
    return {
        "normative": normative,
        "qualification_eligible": normative and bool(windows),
        "k": 12,
        "window_bound_ms": 960.0,
        "definition": "for every window, t[i+11] - t[i] <= 960 ms",
        "window_count": len(windows),
        "failing_window_count": failing_windows,
        "worst_window": worst_window,
        "max_gap_ms": max_gap,
        "max_gap_bound_ms": max_gap_ms,
        "worst_gap": worst_gap,
        "verdict": outcome if normative else None,
        "diagnostic_outcome": outcome if not normative else None,
    }


def _metrics_timestamp_groups(rows: Sequence[JsonObject]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for normalized in rows:
        if not normalized["delivered"]:
            continue
        event = normalized["event"]
        timestamp = _number(event.get("client_received_monotonic_s"))
        response_id = normalized.get("response_group")
        if timestamp is not None and isinstance(response_id, str):
            groups[response_id].append(timestamp)
    return dict(groups)


def _queue_timestamp_groups(
    queue_responses: dict[str, JsonObject],
) -> dict[str, list[float]]:
    return {
        response_id: [delta["timestamp_s"] for delta in response["deltas"]]
        for response_id, response in queue_responses.items()
    }


def _unavailable_structural_gate(reason: str, diagnostics: JsonObject) -> JsonObject:
    return {
        "normative": True,
        "qualification_eligible": False,
        "clock": "client_received_monotonic_s (response.output_audio.delta)",
        "verdict": "UNAVAILABLE",
        "unavailable_reason": reason,
        "k": 12,
        "window_bound_ms": 960.0,
        "window_count": 0,
        "failing_window_count": 0,
        "worst_window": None,
        "max_gap_ms": None,
        "diagnostics": diagnostics,
    }


def analyze_events(
    events: Sequence[JsonObject],
    *,
    reserves_ms: Sequence[float] = DEFAULT_RESERVES_MS,
    max_gap_ms: float = 160.0,
) -> JsonObject:
    rows = _normalize_model_rows(events)
    populations = classify_populations(rows)
    population_report = {name: _population_report(populations[name]) for name in POPULATION_NAMES}
    delivered = populations["delivered"]
    clustered = {clock: response_clustered_stats(delivered, clock) for clock in CLOCK_FIELDS}
    queue_clock, queue_responses, lifecycle_audit = extract_queue_responses(events, rows)
    queue = queue_diagnostics(queue_responses, reserves_ms)
    queue["arrival_clock"] = queue_clock
    queue["response_lifecycle_audit"] = lifecycle_audit
    queue["observed_playout_trace"] = validate_observed_playout_trace(events)
    has_audio_deltas = any(event.get("type") == "response.output_audio.delta" for event in events)
    reconciliation = (
        reconcile_audio_deltas(rows, queue_responses)
        if has_audio_deltas
        else {
            "provable": False,
            "reason": "artifact has no audio-delta receipt events",
            "delivered_count": len(delivered),
            "delta_count": 0,
        }
    )
    queue["delivered_frame_reconciliation"] = reconciliation
    diagnostics: JsonObject = {}
    metrics_groups = _metrics_timestamp_groups(rows)
    if metrics_groups:
        metrics_gate = structural_gate(metrics_groups, max_gap_ms=max_gap_ms, normative=False)
        metrics_gate["clock"] = "client_received_monotonic_s (delivered metrics row)"
        diagnostics["metrics_receipt"] = metrics_gate
    if not has_audio_deltas and queue_responses:
        server_gate = structural_gate(
            _queue_timestamp_groups(queue_responses),
            max_gap_ms=max_gap_ms,
            normative=False,
        )
        server_gate["clock"] = "monotonic_s (delivered model_step fallback)"
        diagnostics["server_emit"] = server_gate
    if reconciliation["provable"]:
        structural = structural_gate(
            _queue_timestamp_groups(queue_responses), max_gap_ms=max_gap_ms
        )
        structural["clock"] = "client_received_monotonic_s (response.output_audio.delta)"
        structural["reconciliation"] = reconciliation
        structural["diagnostics"] = diagnostics
    else:
        structural = _unavailable_structural_gate(reconciliation["reason"], diagnostics)
        structural["reconciliation"] = reconciliation
    return {
        "model_step_rows": len(rows),
        "population_relationships": {
            "delivered_partition": (
                "delivered-nonBOS and BOS-transition are disjoint and exhaustive"
            ),
            "abort_prefill_reset_relationship": (
                "abort-prefill-reset is a subset of BOS-transition"
            ),
            "response_null_predicate": "response_id == null",
            "idle_predicate": (
                "response_id == null and not audio_delivered and not function-cycle"
            ),
            "response_group_scope": "response clustering only",
            "high_mode_scope": "delivered-nonBOS only",
        },
        "delivery_classification": {
            "canonical_rows": sum(row["delivery_mode"] == "audio_delivered" for row in rows),
            "legacy_bytes_fallback_rows": sum(
                row["delivery_mode"] == "legacy_output_audio_bytes" for row in rows
            ),
            "legacy_fallback_normative": False,
        },
        "populations": population_report,
        "response_clustered_delivered": clustered,
        "queue_diagnostics": queue,
        "structural_gate": structural,
    }


def analyze_path(
    path: Path,
    *,
    reserves_ms: Sequence[float] = DEFAULT_RESERVES_MS,
    max_gap_ms: float = 160.0,
) -> JsonObject:
    identity = source_identity(path)
    report = analyze_events(
        read_jsonl(Path(identity["path"])),
        reserves_ms=reserves_ms,
        max_gap_ms=max_gap_ms,
    )
    report["source"] = identity
    return report


def analyze_paths(
    paths: Sequence[Path],
    *,
    reserves_ms: Sequence[float] = DEFAULT_RESERVES_MS,
    max_gap_ms: float = 160.0,
    step1_details: bool = False,
) -> JsonObject:
    if not paths:
        raise ValueError("at least one events.jsonl input is required")
    report = {
        "schema": "generation-frame-latency-step0-v2",
        "nearest_rank_percentiles": True,
        "sources": [
            analyze_path(path, reserves_ms=reserves_ms, max_gap_ms=max_gap_ms) for path in paths
        ],
    }
    if step1_details:
        report["step1_details"] = _build_step1_details(paths, report["sources"])
    return report


def _parse_reserves(values: Sequence[str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                reserve = float(item)
            except ValueError as exc:
                raise ValueError(f"invalid reserve value: {item!r}") from exc
            if not math.isfinite(reserve) or reserve < 0:
                raise ValueError(f"reserve must be a finite non-negative number: {item!r}")
            parsed.append(reserve)
    if not parsed:
        raise ValueError("reserve sweep is empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError("reserve sweep contains duplicate values")
    return tuple(parsed)


def _canonical_anchor_paths(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {
            "preflight": Path.home() / ".local/state/nemotron-voicechat/qualification/"
            "nano-sequential-preflight-20260810-r1/events.jsonl",
            "aged": Path.home() / ".local/state/nemotron-voicechat/qualification/"
            "sustained-aged-26h-20260812/events.jsonl",
            "pair": Path.home() / ".local/state/nemotron-voicechat/traces/model/"
            "f110f063-8c9a-46e1-87a9-97dc682b8022/events.jsonl",
            "high_mode": Path.home() / ".local/state/nemotron-voicechat/traces/model/"
            "209fe177-bbb8-403d-ba50-740d3b6f3bf4/events.jsonl",
            "function_cycle": Path.home() / ".local/state/nemotron-voicechat/traces/model/"
            "2a0fa2ae-766d-4455-bfb0-885f67cc958a/events.jsonl",
        }
    resolved = root.expanduser().resolve()
    aged_candidates = (
        resolved / "qualification/sustained-aged-26h-20260812/events.jsonl",
        resolved / "sustained-aged-26h-20260812/events.jsonl",
    )
    aged = next(
        (candidate for candidate in aged_candidates if candidate.is_file()), aged_candidates[0]
    )
    return {
        "preflight": resolved / "qualification/nano-sequential-preflight-20260810-r1/events.jsonl",
        "aged": aged,
        "pair": resolved / "traces/model/f110f063-8c9a-46e1-87a9-97dc682b8022/events.jsonl",
        "high_mode": resolved / "traces/model/209fe177-bbb8-403d-ba50-740d3b6f3bf4/events.jsonl",
        "function_cycle": resolved
        / "traces/model/2a0fa2ae-766d-4455-bfb0-885f67cc958a/events.jsonl",
    }


def _find_sweep(queue: JsonObject, reserve_ms: float) -> JsonObject:
    return next(
        item
        for item in queue["reserve_sweep"]
        if math.isclose(item["reserve_ms"], reserve_ms, rel_tol=0.0, abs_tol=1e-12)
    )


def _integrity_checks(
    reports: dict[str, JsonObject],
    expected_source_hashes: dict[str, str],
    expected_membership_hashes: dict[tuple[str, str], str],
) -> list[JsonObject]:
    """Return exact verifier checks for source and population membership."""

    checks: list[JsonObject] = []
    for source, expected_hash in expected_source_hashes.items():
        actual_hash = reports[source]["source"]["sha256"]
        checks.append(
            {
                "anchor": f"{source}.source.sha256",
                "expected": expected_hash,
                "actual": actual_hash,
                "difference": None,
                "passed": actual_hash == expected_hash,
            }
        )
    for (source, population), expected_hash in expected_membership_hashes.items():
        actual_hash = reports[source]["populations"][population]["membership"]["sha256"]
        checks.append(
            {
                "anchor": f"{source}.{population}.membership.sha256",
                "expected": expected_hash,
                "actual": actual_hash,
                "difference": None,
                "passed": actual_hash == expected_hash,
            }
        )
    return checks


def verify_anchors(root: Path | None = None) -> JsonObject:
    paths = _canonical_anchor_paths(root)
    reports = {
        name: analyze_path(path, reserves_ms=(160.0, 240.0, 320.0, 400.0))
        for name, path in paths.items()
    }
    checks: list[JsonObject] = []

    def check(label: str, actual: float | int | str | None, expected: Any) -> None:
        if isinstance(expected, float):
            passed = isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 0.001
            difference = float(actual) - expected if isinstance(actual, (int, float)) else None
        else:
            passed = actual == expected
            difference = None
        checks.append(
            {
                "anchor": label,
                "expected": expected,
                "actual": actual,
                "difference": difference,
                "passed": passed,
            }
        )

    expected_source_hashes = {
        "preflight": "166ec35a5e74045384c80042e01fe20e4795137719797634145dde6f6d9a74fd",
        "aged": "acb205f6b6e5c786a039a2e3e238f12a15dc0fc0dcc971341dd8838b33f7353f",
        "pair": "aba32f50ad8c7c03fce4c3c1663ea0d42da119d4a9a52c1d90379bb4596ea1f3",
        "high_mode": "baabf0a1203e8fa4bdcef9ebcbd9b2aeb7db952ec0eb83c58e18898fdcb2f880",
        "function_cycle": "2815a4e99d5ade234ade161cc83acd9e4b7fa93a4297bd55fb5e3f5ae596693d",
    }
    for source, population, expected_n, expected_mean in (
        ("preflight", "delivered", 34, 86.168),
        ("preflight", "response-null", 2692, 70.926),
        ("aged", "delivered", 193, 85.403),
        ("aged", "delivered-nonBOS", 188, 82.919),
        ("aged", "response-null", 2454, 70.627),
        ("aged", "idle", 2424, 71.469),
        ("pair", "delivered", 32, 75.116),
    ):
        stats = reports[source]["populations"][population]["clocks"]["server_step_ms"]
        check(f"{source}.{population}.server.n", stats["n"], expected_n)
        check(f"{source}.{population}.server.mean_ms", stats["mean"], expected_mean)

    expected_membership_hashes = {
        (
            "preflight",
            "delivered",
        ): "05295ce9e58ce7bd1685fb9a1a3f0f0e04f9f01d5ac923e0b6fdb13e6a64a155",
        (
            "preflight",
            "response-null",
        ): "3a5e63cffe022bf56d2d1444a3350a5f5e869e1d3682a4888f8ac2c987c49221",
        ("aged", "delivered"): "3f832f89b89e367c590e23cacda854b963a9f78a1960f615c85fc7db977ddb5e",
        (
            "aged",
            "delivered-nonBOS",
        ): "e4afedeb26f6bcd04194100475f4a53c733e50400d4d11ea9fdbcbd5529930fb",
        (
            "aged",
            "response-null",
        ): "fbe75b398c42f6252b08dba96784724329d3a9867869077f0ed1437e0fb4f366",
        ("aged", "idle"): "98f712d2c6b8e4da32454a687ed2713a637a214f12378d2a83b67e9af8cf78ef",
        (
            "aged",
            "function-cycle",
        ): "02b104caf316b636297f7a33ceff4d1253ff2c3ddada53495264aabd4f84dd9a",
        ("pair", "delivered"): "95ea829c87154c81b39e1154e6e67e02f22f39cb5ae1748c3746065eb4a67890",
        (
            "high_mode",
            "EarTTS-high",
        ): "d3bc56bd3fc35cd3e0f1638027836ba437de7de322823e01cb4c0532e6ae0a0e",
        (
            "function_cycle",
            "function-cycle",
        ): "8c9142aadcc4881c2543aad4e222e8b7691d5a40ff9a6b333629b429cd216c57",
        (
            "function_cycle",
            "BOS-transition",
        ): "c598fc6d6439a52109612cbe8712ad2bcb8ce09a7ae5ffc0f810f1cf39d183d4",
        (
            "function_cycle",
            "abort-prefill-reset",
        ): "209cba67016bc59f950585878e82f82a633d6a0ce7a6bb4c12155bfd2d1252b2",
        (
            "function_cycle",
            "residual-high",
        ): "b7908f96bb0eedfab203e54df198c6786fdd0352e75247fdbd66c6e72560cc45",
    }
    checks[0:0] = _integrity_checks(reports, expected_source_hashes, expected_membership_hashes)

    for source, response_null_n, idle_n in (
        ("high_mode", 245, 190),
        ("function_cycle", 490, 326),
    ):
        check(
            f"{source}.response-null.literal_count",
            reports[source]["populations"]["response-null"]["membership"]["n"],
            response_null_n,
        )
        check(
            f"{source}.idle.normative_count",
            reports[source]["populations"]["idle"]["membership"]["n"],
            idle_n,
        )

    check(
        "high_mode.EarTTS-high.semantic_count",
        reports["high_mode"]["populations"]["EarTTS-high"]["membership"]["n"],
        10,
    )
    check(
        "function_cycle.function-cycle.semantic_count",
        reports["function_cycle"]["populations"]["function-cycle"]["membership"]["n"],
        30,
    )

    check(
        "preflight.zero_reserve_prefix_debt_ms",
        reports["preflight"]["queue_diagnostics"]["zero_reserve_max_prefix_debt_ms"],
        293.889,
    )
    check(
        "pair.zero_reserve_prefix_debt_ms",
        reports["pair"]["queue_diagnostics"]["zero_reserve_max_prefix_debt_ms"],
        4.530,
    )
    gate = reports["preflight"]["structural_gate"]
    for label, actual, expected in (
        ("verdict", gate["verdict"], "REJECT"),
        ("worst_window_ms", gate["worst_window"]["duration_ms"], 1009.934),
        ("failing_window_count", gate["failing_window_count"], 13),
        ("max_gap_ms", gate["max_gap_ms"], 151.671),
    ):
        check(f"preflight.structural.audio_delta.{label}", actual, expected)
    metrics_gate = gate["diagnostics"]["metrics_receipt"]
    check(
        "preflight.structural.metrics_receipt.worst_window_ms",
        metrics_gate["worst_window"]["duration_ms"],
        1009.957,
    )
    check("preflight.structural.metrics_receipt.verdict", metrics_gate["verdict"], None)

    pair_gate = reports["pair"]["structural_gate"]
    check("pair.structural.verdict", pair_gate["verdict"], "UNAVAILABLE")
    diagnostic = pair_gate["diagnostics"]["server_emit"]
    for label, actual, expected in (
        ("diagnostic_outcome", diagnostic["diagnostic_outcome"], "ACCEPT"),
        ("worst_window_ms", diagnostic["worst_window"]["duration_ms"], 877.871),
        ("failing_window_count", diagnostic["failing_window_count"], 0),
        ("max_gap_ms", diagnostic["max_gap_ms"], 120.939),
    ):
        check(f"pair.structural.server_emit.{label}", actual, expected)

    combined_events = read_jsonl(paths["preflight"]) + read_jsonl(paths["aged"])
    _, combined_responses, _ = extract_queue_responses(combined_events)
    combined_queue = queue_diagnostics(combined_responses, (160.0, 240.0, 320.0, 400.0))
    for reserve, starved, starvation, delay in (
        (160.0, 4, 797.184, 116.202),
        (240.0, 4, 438.789, 202.734),
        (320.0, 2, 136.074, 289.819),
        (400.0, 0, 0.000, 375.563),
    ):
        sweep = _find_sweep(combined_queue, reserve)
        check(f"a6.reserve_{reserve:g}.responses", sweep["responses"], 6)
        check(f"a6.reserve_{reserve:g}.responses_starved", sweep["responses_starved"], starved)
        check(
            f"a6.reserve_{reserve:g}.total_starvation_ms",
            sweep["total_starvation_ms"],
            starvation,
        )
        check(
            f"a6.reserve_{reserve:g}.mean_added_release_delay_ms",
            sweep["mean_added_release_delay_ms"],
            delay,
        )
    return {
        "passed": all(item["passed"] for item in checks),
        "tolerance_ms": 0.001,
        "paths": {name: str(path.expanduser().resolve()) for name, path in paths.items()},
        "checks": checks,
    }


def _write_report(report: JsonObject, output: Path | None) -> str:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output = output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return encoded


def _format_report_value(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _artifact_id(source: JsonObject) -> str:
    return Path(source["source"]["path"]).parent.name


def render_step1_markdown(report: JsonObject) -> str:
    """Render the Step 1 evidence report strictly from analyzer JSON."""

    details = report.get("step1_details")
    if not isinstance(details, dict):
        raise ValueError("Step 1 Markdown requires --step1-details")
    lines = [
        "# Step 1 retained generation-frame evidence — 2026-08-12",
        "",
        "This report is an analyzer-rendered view of "
        "`step1-retained-evidence-20260812.json`. All latency aggregates use "
        "nearest-rank percentiles. Population membership hashes cover ordered row "
        "identifiers; the gate table pins every retained source byte-for-byte.",
        "",
        "The corpus contains "
        f"{details['artifact_count']} unique retained artifacts. The three schema anchors "
        "are the sequential preflight, aged sustained stream, and pair-era trace; the "
        "pair trace appears only once even though the task names it both as an anchor and "
        "as the pair comparator. No model, container, WebSocket, or browser run was used.",
        "",
        "## Corpus inventory",
        "",
        "For every population below, `n / mean / p95 / >80` uses the server clock "
        "`server_step_ms`; `mhash` is that population's full membership SHA-256.",
    ]
    for source in report["sources"]:
        artifact = _artifact_id(source)
        lines.extend(
            [
                "",
                f"### `{artifact}`",
                "",
                f"Source SHA-256: `{source['source']['sha256']}`",
                "",
                "| population | n | mean ms | p95 ms | >80 ms | mhash |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for population in POPULATION_NAMES:
            item = source["populations"][population]
            stats = item["clocks"]["server_step_ms"]
            lines.append(
                f"| {population} | {stats['n']} | {_format_report_value(stats['mean'])} | "
                f"{_format_report_value(stats['p95'])} | {stats['count_over_80_ms']} | "
                f"`{item['membership']['sha256']}` |"
            )
        gate = source["structural_gate"]
        diagnostic = gate.get("diagnostics", {}).get("server_emit")
        outcome = gate.get("verdict")
        if outcome == "UNAVAILABLE" and isinstance(diagnostic, dict):
            outcome = f"UNAVAILABLE (server diagnostic {diagnostic['diagnostic_outcome']})"
        delivered_hash = source["populations"]["delivered"]["membership"]["sha256"]
        worst_window = gate.get("worst_window")
        if worst_window is None and isinstance(diagnostic, dict):
            worst_window = diagnostic.get("worst_window")
        max_gap = gate.get("max_gap_ms")
        if max_gap is None and isinstance(diagnostic, dict):
            max_gap = diagnostic.get("max_gap_ms")
        lines.extend(
            [
                "",
                "Structural gate: "
                f"clock `{gate['clock']}`; eligible "
                f"`{_format_report_value(gate['qualification_eligible'])}`; outcome "
                f"`{outcome}`; worst K-window "
                + _format_report_value(worst_window.get("duration_ms") if worst_window else None)
                + " ms; "
                f"max gap {_format_report_value(max_gap)} ms. Delivered mhash "
                f"`{delivered_hash}`.",
            ]
        )
        queue = source["queue_diagnostics"]
        reserve = next(
            (item for item in queue["reserve_sweep"] if item["reserve_ms"] == 160.0), None
        )
        reconciliation_hash = queue["delivered_frame_reconciliation"].get(
            "ordered_membership_sha256"
        )
        queue_membership = reconciliation_hash or delivered_hash
        lines.append(
            "Queue diagnostics: clock "
            f"`{queue['arrival_clock']}`; normative `{_format_report_value(queue['normative'])}`; "
            f"responses {queue['response_count']}; zero-reserve max prefix debt "
            f"{_format_report_value(queue['zero_reserve_max_prefix_debt_ms'])} ms; "
            "at reserve 160 ms, responses starved / total starvation "
            f"{reserve['responses_starved'] if reserve else 0} / "
            f"{_format_report_value(reserve['total_starvation_ms'] if reserve else 0.0)} ms. "
            f"Queue/delivered membership hash `{queue_membership}`."
        )

    lines.extend(
        [
            "",
            "## Corpus-scale stage-timing strata",
            "",
            "Token phase is source-derived: PAD-tail means "
            f"`agent_token_id == {details['pad_token_id']}`; "
            "agent-control means `agent_control` is present or the token matches a non-PAD "
            "`control_ids` entry; text-emission means a remaining integer token. BOS remains "
            "separate. High modes retain the "
            "plan predicates over delivered-nonBOS only. Server columns use `server_step_ms`; "
            "stage columns are host-interval means in milliseconds.",
            "",
            "| stratum | n | server mean | server p95 | >80 | perception | Nano | EarTTS "
            "| codec | residual | wrapper | mhash |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    stage_order = (
        "delivered-nonBOS-stage-timed",
        "text-emission",
        "PAD-tail",
        "agent-control",
        "token-unavailable",
        "EarTTS-high",
        "residual-high",
        "BOS-transition",
        "abort-prefill-reset",
        "prepared-reuse-BOS",
    )
    for name in stage_order:
        item = details["stage_timing_strata"][name]
        server = item["clocks"]["server_step_ms"]
        stages = item["stage_timings"]
        lines.append(
            f"| {name} | {item['membership']['n']} | {_format_report_value(server['mean'])} | "
            f"{_format_report_value(server['p95'])} | {server['count_over_80_ms']} | "
            f"{_format_report_value(stages['perception']['mean'])} | "
            f"{_format_report_value(stages['nano_interface']['mean'])} | "
            f"{_format_report_value(stages['eartts_total']['mean'])} | "
            f"{_format_report_value(stages['codec_call']['mean'])} | "
            f"{_format_report_value(stages['wrapper_residual']['mean'])} | "
            f"{_format_report_value(stages['wrapper_total']['mean'])} | "
            f"`{item['membership']['sha256']}` |"
        )

    phase_names = ("text-emission", "PAD-tail", "agent-control", "token-unavailable")
    phase_total = sum(
        details["stage_timing_strata"][name]["membership"]["n"] for name in phase_names
    )
    delivered_nonbos_total = details["stage_timing_strata"]["delivered-nonBOS-stage-timed"][
        "membership"
    ]["n"]
    lines.extend(
        [
            "",
            f"The four delivered-nonBOS token-phase strata reconcile {phase_total}/"
            f"{delivered_nonbos_total} stage-timed rows; BOS is reported separately.",
        ]
    )

    comparison = details["pair_vs_sequential"]
    lines.extend(
        [
            "",
            "## Pair versus sequential retained rollup",
            "",
            f"Pair artifact: `{comparison['pair_artifact']}`. Sequential side: all "
            f"{len(comparison['sequential_artifacts'])} other retained artifacts. Both sides use "
            f"the identical analyzer populations and `{comparison['clock']}` clock.",
            "",
            "| population | pair n / mean / p95 / >80 | pair mhash "
            "| sequential n / mean / p95 / >80 | sequential mhash |",
            "|---|---|---|---|---|",
        ]
    )
    for population in POPULATION_NAMES:
        item = comparison["populations"][population]
        pair = item["pair"]
        sequential = item["sequential"]
        pair_stats = pair["clocks"][comparison["clock"]]
        sequential_stats = sequential["clocks"][comparison["clock"]]
        lines.append(
            f"| {population} | {pair_stats['n']} / {_format_report_value(pair_stats['mean'])} / "
            f"{_format_report_value(pair_stats['p95'])} / {pair_stats['count_over_80_ms']} | "
            f"`{pair['membership']['sha256']}` | {sequential_stats['n']} / "
            f"{_format_report_value(sequential_stats['mean'])} / "
            f"{_format_report_value(sequential_stats['p95'])} / "
            f"{sequential_stats['count_over_80_ms']} | "
            f"`{sequential['membership']['sha256']}` |"
        )
    schedule = comparison["pair_schedule"]
    lines.extend(
        [
            "",
            "Pair scheduling is classified only from the delta of "
            "`turn_state.vllm_request_positions.nano.session_positions`: zero is buffered, "
            "one is sequential, and two is packed-pair.",
            "",
            "| schedule class | n | server mean | server p95 | >80 | mhash |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name in ("buffered", "packed-pair", "sequential", "unavailable", "other-position-delta"):
        item = schedule["classes"][name]
        stats = item["clocks"]["server_step_ms"]
        lines.append(
            f"| {name} | {item['membership']['n']} | {_format_report_value(stats['mean'])} | "
            f"{_format_report_value(stats['p95'])} | {stats['count_over_80_ms']} | "
            f"`{item['membership']['sha256']}` |"
        )
    transitions = schedule["transitions"]
    lines.extend(
        [
            "",
            f"The pair-class subsequence contains {schedule['pair_class_row_count']} rows and "
            f"{schedule['adjacent_pair_class_transition_count']} adjacent transitions: "
            f"buffered→packed {transitions.get('buffered->packed-pair', 0)}, "
            f"packed→buffered {transitions.get('packed-pair->buffered', 0)}. "
            "The ordered pair-class membership SHA-256 is "
            f"`{schedule['pair_class_membership']['sha256']}`; transition counts and the "
            "complete per-row schedule are in `step1_details.pair_vs_sequential.pair_schedule`.",
        ]
    )

    lines.extend(["", "## High-mode row inventories for Step 4"])
    for mode in ("EarTTS-high", "residual-high"):
        inventory = details["high_mode_inventories"][mode]
        patterns = inventory["pattern_summary"]
        phase_text = (
            ", ".join(f"{name} {count}" for name, count in patterns["counts_by_phase"].items())
            or "none"
        )
        lines.extend(
            [
                "",
                f"### {mode}",
                "",
                f"Inventory n={inventory['membership']['n']}; membership SHA-256 "
                f"`{inventory['membership']['sha256']}`. Phase counts: {phase_text}. "
                "Position and periodicity diagnostics are analyzer outputs: response ordinal "
                f"p50/p95 {_format_report_value(patterns['response_delivered_ordinal']['p50'])}/"
                f"{_format_report_value(patterns['response_delivered_ordinal']['p95'])}; "
                "distance from response end p50/p95 "
                f"{_format_report_value(patterns['delivered_rows_from_response_end']['p50'])}/"
                f"{_format_report_value(patterns['delivered_rows_from_response_end']['p95'])}; "
                "within-response high-row ordinal gap p50/p95 "
                f"{_format_report_value(patterns['within_response_high_row_ordinal_gap']['p50'])}/"
                f"{_format_report_value(patterns['within_response_high_row_ordinal_gap']['p95'])}.",
                "",
                "| artifact | frame | phase/token | response pos | server | perception "
                "| Nano | EarTTS | codec | residual | wrapper | previous → next |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in inventory["rows"]:
            stages = row["stage_timings_ms"]

            def neighbor(value: JsonObject | None) -> str:
                if value is None:
                    return "none"
                return (
                    f"f{value['frame']} {value['phase']}/t{value['agent_token_id']} "
                    f"{_format_report_value(value['server_step_ms'])}ms"
                )

            lines.append(
                f"| `{row['artifact']}` | {row['frame']} "
                f"| {row['phase']}/t{row['agent_token_id']} | "
                f"{row['response_delivered_ordinal']}/{row['response_delivered_count']} | "
                f"{_format_report_value(row['server_step_ms'])} | "
                f"{_format_report_value(stages['perception'])} | "
                f"{_format_report_value(stages['nano_interface'])} | "
                f"{_format_report_value(stages['eartts_total'])} | "
                f"{_format_report_value(stages['codec_call'])} | "
                f"{_format_report_value(stages['wrapper_residual'])} | "
                f"{_format_report_value(stages['wrapper_total'])} | "
                f"{neighbor(row['previous_model_row'])} → {neighbor(row['next_model_row'])} |"
            )

    eartts_patterns = details["high_mode_inventories"]["EarTTS-high"]["pattern_summary"]
    residual_patterns = details["high_mode_inventories"]["residual-high"]["pattern_summary"]
    lines.extend(
        [
            "",
            "### Step 4 targeting summary",
            "",
            "EarTTS-high should be profiled first on PAD-tail states: the inventory is "
            f"{eartts_patterns['counts_by_phase'].get('PAD-tail', 0)} PAD-tail versus "
            f"{eartts_patterns['counts_by_phase'].get('text-emission', 0)} text-emission rows, "
            "with repeated within-response events rather than a single BOS/reset effect. "
            "Retain text-position captures as the control because the high predicate also "
            "occurs there. Residual-high should be investigated separately on text-emission "
            f"states ({residual_patterns['counts_by_phase'].get('text-emission', 0)} of "
            f"{details['high_mode_inventories']['residual-high']['membership']['n']} rows); "
            "its normal EarTTS column in the row inventory supports keeping tracing/logging "
            "overhead as the first residual hypothesis. The non-unit, variable high-row gaps "
            "reported above do not support selecting a fixed-period kernel hypothesis from "
            "this corpus alone.",
            "",
            "## Step 1 gate and source identities",
            "",
            "PASS. Every reported aggregate is emitted by the qualified analyzer and carries "
            "an ordered membership SHA-256; every member includes its artifact and source "
            "SHA-256. Structural and queue rows cite their delivered/reconciled membership. "
            "The JSON retains the raw analyzer output for every artifact.",
            "",
            "| artifact | source path | source SHA-256 |",
            "|---|---|---|",
        ]
    )
    for source in report["sources"]:
        lines.append(
            f"| `{_artifact_id(source)}` | `{source['source']['path']}` | "
            f"`{source['source']['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "The retained negative controls remain `reports/step8-overlap-ab-20260807.md` "
            "and `reports/step5-host-barriers-20260807.md`; they were not rerun or altered "
            "during this offline inventory.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="analyze retained JSONL files")
    analyze_parser.add_argument("events", type=Path, nargs="+")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument(
        "--reserve-sweep",
        nargs="+",
        default=[",".join(str(value) for value in DEFAULT_RESERVES_MS)],
        metavar="MS",
        help="comma- or space-separated reserve values (default: 0,80,...,560)",
    )
    analyze_parser.add_argument("--max-gap-ms", type=float, default=160.0)
    analyze_parser.add_argument(
        "--step1-details",
        action="store_true",
        help="include corpus stage strata, pair scheduling, and high-row context",
    )
    analyze_parser.add_argument(
        "--markdown-output",
        type=Path,
        help="render a Step 1 Markdown view from the same analyzer result",
    )

    verify_parser = subparsers.add_parser("verify-anchors", help="verify retained r3 anchors")
    verify_parser.add_argument(
        "--root",
        type=Path,
        help=("override fixture root containing qualification/, traces/, and sustained-aged-26h/"),
    )
    verify_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            reserves = _parse_reserves(args.reserve_sweep)
            if not math.isfinite(args.max_gap_ms) or args.max_gap_ms < 0:
                raise ValueError("max-gap-ms must be finite and non-negative")
            if args.markdown_output is not None and not args.step1_details:
                raise ValueError("--markdown-output requires --step1-details")
            report = analyze_paths(
                args.events,
                reserves_ms=reserves,
                max_gap_ms=args.max_gap_ms,
                step1_details=args.step1_details,
            )
            _write_report(report, args.output)
            if args.markdown_output is not None:
                markdown = render_step1_markdown(report)
                markdown_output = args.markdown_output.expanduser()
                markdown_output.parent.mkdir(parents=True, exist_ok=True)
                markdown_output.write_text(markdown, encoding="utf-8")
            print(
                f"analyzed {len(report['sources'])} source(s); report: "
                f"{args.output.expanduser().resolve()}"
            )
            return 0
        report = verify_anchors(args.root)
        encoded = _write_report(report, args.output)
        if args.output is None:
            print(encoded, end="")
        if report["passed"]:
            print(f"verify-anchors: PASS ({len(report['checks'])} anchors)")
            return 0
        print("verify-anchors: FAIL", file=sys.stderr)
        for item in report["checks"]:
            if not item["passed"]:
                print(
                    f"  {item['anchor']}: expected {item['expected']!r}, "
                    f"actual {item['actual']!r}, diff {item['difference']!r}",
                    file=sys.stderr,
                )
        return 1
    except (OSError, ValueError, StopIteration) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
