#!/usr/bin/env python3
"""Fail-closed contracts for the preregistered Step 2 live fixture campaign."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from stratified_latency_analyzer import validate_observed_playout_trace

PLAN_SCHEMA = "nemotron_voicechat.step2_fixture_plan.v2"
MARKER_SCHEMA = "nemotron_voicechat.step2_fixture_marker.v2"
SUMMARY_SCHEMA = "nemotron_voicechat.step2_fixture_summary.v2"
FIXTURE_RESULT_SCHEMA = "nemotron_voicechat.step2_fixture_result.v2"
BROWSER_TRACE_SCHEMA = "nemotron_voicechat.browser_playout.v1"
PLAYOUT_TRACE_SCHEMA = "nemotron_voicechat.playout.v1"
PLAYOUT_TRACE_CONNECTION_TOKEN = "{connection}"
PLAYOUT_TRACE_PUBLICATION_TIMEOUT_S = 90.0
PLAYOUT_TRACE_PUBLICATION_POLL_S = 1.0
RTVI_QUIESCENCE_WINDOW_MS = 2000.0
RTVI_QUIESCENCE_MIN_WINDOW_MS = 500.0
RTVI_QUIESCENCE_MAX_WINDOW_MS = 60_000.0
# Date.now() is millisecond-quantized. Permit at most one tick when comparing
# its retained duration with the configured monotonic-clock window.
RTVI_WALL_CLOCK_SKEW_TOLERANCE_S = 0.001
RTVI_TRANSCRIPT_IDENTITY = "ordered-channel-quiescence-v1"
TRANSCRIPT_MISMATCH_EMPTY = (
    "joined browser transcript is empty while response-owned traced text is nonempty"
)
TRANSCRIPT_MISMATCH_DIFFERENT = (
    "joined browser transcript differs from response-owned traced text"
)

CaptureMode = Literal["on", "off"]
FixtureStatus = Literal["not_run", "running", "completed", "timeout", "error"]

SCRIPTS = {
    "S1": "The red balloon rose over the quiet village square.",
    "L1": (
        "The quick brown fox jumps over the lazy dog while seventeen green dragons "
        "circle the ancient stone tower, counting every window, every door, and every "
        "flag that flies above the northern gate."
    ),
    "L2": (
        "On Tuesday morning the harbor filled with small boats carrying baskets of "
        "silver fish, and the old lighthouse keeper wrote down the number of gulls, "
        "the color of the clouds, and the exact minute the tide turned."
    ),
}
I1_REPLACEMENT_TEXT = "Stop. What is two plus two?"
I1_OFFSET_S = 4.0
I1_OFFSET_TOLERANCE_S = 0.25
CANONICAL_TIMEOUT_S = 180.0

_WAIT_KEYS = {"event", "timeout_s", "require_transcription"}
_FIXTURE_KEYS = {
    "fixture_id",
    "script_id",
    "typed_text",
    "capture",
    "wait",
    "interruption",
}
_INTERRUPTION_KEYS = {"after_first_audio_s", "typed_text"}
_BOUNDARIES = {"start", "interruption-send", "complete", "timeout", "error"}
_FIXTURE_STATUSES = {"not_run", "running", "completed", "timeout", "error"}
_SUMMARY_STATUSES = {"running", "completed", "failed"}
_SESSION_STATUSES = {"not_run", "running", "completed", "failed"}
_MARKER_STATUSES = {"not_attempted", "succeeded", "failed"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WaitCriteria:
    event: str
    timeout_s: float
    require_transcription: bool


@dataclass(frozen=True)
class InterruptionSpec:
    after_first_audio_s: float
    typed_text: str
    typed_text_sha256: str


@dataclass(frozen=True)
class FixtureSpec:
    plan_index: int
    fixture_id: str
    script_id: str
    typed_text: str
    typed_text_sha256: str
    capture: CaptureMode
    wait: WaitCriteria
    interruption: InterruptionSpec | None = None


@dataclass(frozen=True)
class FixturePlan:
    fixtures: tuple[FixtureSpec, ...]
    plan_sha256: str
    script_hashes_sha256: Mapping[str, str]
    schema: str = PLAN_SCHEMA

    def sessions(self) -> tuple[tuple[CaptureMode, tuple[FixtureSpec, ...]], ...]:
        """Preserve the validated declaration order and its sole mode transition."""

        return (("on", self.fixtures[:7]), ("off", self.fixtures[7:]))


def _canonical_fixture_document() -> list[dict[str, object]]:
    ids_and_scripts = (
        ("S1-1", "S1", "on"),
        ("S1-2", "S1", "on"),
        ("L1-1", "L1", "on"),
        ("L1-2", "L1", "on"),
        ("L2-1", "L2", "on"),
        ("L2-2", "L2", "on"),
        ("I1", "I1", "on"),
        ("capture-off-L1", "L1", "off"),
    )
    document: list[dict[str, object]] = []
    for fixture_id, script_id, capture in ids_and_scripts:
        text_id = "L1" if script_id == "I1" else script_id
        fixture: dict[str, object] = {
            "fixture_id": fixture_id,
            "script_id": script_id,
            "typed_text": SCRIPTS[text_id],
            "capture": capture,
            "wait": {
                "event": "bot-stopped-speaking",
                "timeout_s": CANONICAL_TIMEOUT_S,
                "require_transcription": True,
            },
        }
        if script_id == "I1":
            fixture["interruption"] = {
                "after_first_audio_s": I1_OFFSET_S,
                "typed_text": I1_REPLACEMENT_TEXT,
            }
        document.append(fixture)
    return document


CANONICAL_FIXTURE_DOCUMENT = _canonical_fixture_document()


def canonical_plan_json() -> str:
    """Return the canonical JSON bytes operators must preregister and supply."""

    return json.dumps(CANONICAL_FIXTURE_DOCUMENT, sort_keys=True, separators=(",", ":"))


CANONICAL_PLAN_SHA256 = sha256_text(canonical_plan_json())
CANONICAL_SCRIPT_HASHES_SHA256 = {
    **{script_id: sha256_text(text) for script_id, text in SCRIPTS.items()},
    "I1": sha256_text(SCRIPTS["L1"]),
    "I1-replacement": sha256_text(I1_REPLACEMENT_TEXT),
}


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _keys(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: object, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}")
    return number


def validate_rtvi_quiescence_window_ms(value: object) -> float:
    """Require the bounded positive interval that protects Step 2 gate 7."""

    return _finite_number(
        value,
        "RTVI quiescence window",
        minimum=RTVI_QUIESCENCE_MIN_WINDOW_MS,
        maximum=RTVI_QUIESCENCE_MAX_WINDOW_MS,
    )


def parse_rtvi_quiescence_window_ms(raw: str | None) -> float:
    """Parse the live environment override once during campaign preflight."""

    if raw is None:
        return RTVI_QUIESCENCE_WINDOW_MS
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("RTVI quiescence window must be numeric") from exc
    return validate_rtvi_quiescence_window_ms(value)


def _parse_wait(value: object, label: str) -> WaitCriteria:
    wait = _object(value, label)
    _keys(wait, _WAIT_KEYS, label)
    event = _nonempty_string(wait.get("event"), f"{label}.event")
    timeout_s = _finite_number(
        wait.get("timeout_s"), f"{label}.timeout_s", minimum=1.0, maximum=600.0
    )
    require_transcription = wait.get("require_transcription")
    if not isinstance(require_transcription, bool):
        raise ValueError(f"{label}.require_transcription must be a boolean")
    return WaitCriteria(event, timeout_s, require_transcription)


def _parse_interruption(value: object, label: str) -> InterruptionSpec:
    interruption = _object(value, label)
    _keys(interruption, _INTERRUPTION_KEYS, label)
    text = _nonempty_string(interruption.get("typed_text"), f"{label}.typed_text")
    return InterruptionSpec(
        after_first_audio_s=_finite_number(
            interruption.get("after_first_audio_s"),
            f"{label}.after_first_audio_s",
            minimum=0.0,
            maximum=300.0,
        ),
        typed_text=text,
        typed_text_sha256=sha256_text(text),
    )


def parse_fixture_plan(raw: str) -> FixturePlan:
    """Parse and require the exact preregistered eight-entry Step 2 plan."""

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fixture plan is not valid JSON: {exc}") from exc
    if not isinstance(document, list):
        raise ValueError("fixture plan must be the canonical ordered JSON list")
    if len(document) != len(CANONICAL_FIXTURE_DOCUMENT):
        raise ValueError("fixture plan must contain exactly eight canonical entries")

    fixtures: list[FixtureSpec] = []
    for index, (raw_fixture, expected) in enumerate(
        zip(document, CANONICAL_FIXTURE_DOCUMENT, strict=True)
    ):
        label = f"fixture[{index}]"
        fixture = _object(raw_fixture, label)
        _keys(fixture, _FIXTURE_KEYS, label)
        expected_keys = set(expected)
        if set(fixture) != expected_keys:
            raise ValueError(f"{label} fields do not match the canonical plan")
        fixture_id = _nonempty_string(fixture.get("fixture_id"), f"{label}.fixture_id")
        script_id = _nonempty_string(fixture.get("script_id"), f"{label}.script_id")
        typed_text = _nonempty_string(fixture.get("typed_text"), f"{label}.typed_text")
        capture = fixture.get("capture")
        if capture not in {"on", "off"}:
            raise ValueError(f"{label}.capture must be 'on' or 'off'")
        wait = _parse_wait(fixture.get("wait"), f"{label}.wait")
        interruption = (
            _parse_interruption(fixture["interruption"], f"{label}.interruption")
            if "interruption" in fixture
            else None
        )
        normalized = {
            "fixture_id": fixture_id,
            "script_id": script_id,
            "typed_text": typed_text,
            "capture": capture,
            "wait": asdict(wait),
            **(
                {
                    "interruption": {
                        "after_first_audio_s": interruption.after_first_audio_s,
                        "typed_text": interruption.typed_text,
                    }
                }
                if interruption
                else {}
            ),
        }
        if normalized != expected:
            raise ValueError(f"{label} does not match the canonical preregistered entry")
        fixtures.append(
            FixtureSpec(
                plan_index=index,
                fixture_id=fixture_id,
                script_id=script_id,
                typed_text=typed_text,
                typed_text_sha256=sha256_text(typed_text),
                capture=capture,
                wait=wait,
                interruption=interruption,
            )
        )

    modes = [fixture.capture for fixture in fixtures]
    if modes != ["on"] * 7 + ["off"]:
        raise ValueError("fixture plan requires seven capture-on entries then one capture-off")
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if sha256_text(canonical) != CANONICAL_PLAN_SHA256:
        raise ValueError("fixture plan hash does not match the preregistration")
    return FixturePlan(
        fixtures=tuple(fixtures),
        plan_sha256=CANONICAL_PLAN_SHA256,
        script_hashes_sha256=dict(CANONICAL_SCRIPT_HASHES_SHA256),
    )


def plan_document(plan: FixturePlan) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    for fixture in plan.fixtures:
        item = asdict(fixture)
        fixtures.append(item)
    return {
        "schema": plan.schema,
        "plan_sha256": plan.plan_sha256,
        "script_hashes_sha256": dict(plan.script_hashes_sha256),
        "fixtures": fixtures,
    }


def interruption_deadline(first_audio_monotonic_s: float, after_first_audio_s: float) -> float:
    first = _finite_number(
        first_audio_monotonic_s,
        "first_audio_monotonic_s",
        minimum=0.0,
        maximum=1e100,
    )
    delay = _finite_number(after_first_audio_s, "after_first_audio_s", minimum=0.0, maximum=300.0)
    return first + delay


async def wait_for_interruption_deadline(
    first_audio_monotonic_s: float,
    after_first_audio_s: float,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> float:
    deadline = interruption_deadline(first_audio_monotonic_s, after_first_audio_s)
    while (remaining := deadline - monotonic()) > 0:
        await sleep(remaining)
    return monotonic()


def fixture_marker(
    fixture: FixtureSpec,
    boundary: str,
    *,
    browser_performance_ms: float,
    browser_wall_time_s: float,
    detail: str | None = None,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "trace_schema": BROWSER_TRACE_SCHEMA,
        "type": "browser.fixture_boundary",
        "marker_schema": MARKER_SCHEMA,
        "fixture_index": fixture.plan_index,
        "fixture_id": fixture.fixture_id,
        "script_id": fixture.script_id,
        "capture_enabled": fixture.capture == "on",
        "boundary": boundary,
        "browser_performance_ms": browser_performance_ms,
        "browser_wall_time_s": browser_wall_time_s,
    }
    if detail is not None:
        marker["detail"] = detail
    validate_marker(marker)
    return marker


def validate_marker(marker: Mapping[str, object]) -> None:
    required = {
        "trace_schema",
        "type",
        "marker_schema",
        "fixture_index",
        "fixture_id",
        "script_id",
        "capture_enabled",
        "boundary",
        "browser_performance_ms",
        "browser_wall_time_s",
    }
    allowed = required | {
        "detail",
        "playwright_batch_received_monotonic_s",
        "playwright_batch_received_wall_time_s",
    }
    if set(marker) - allowed or not required <= marker.keys():
        raise ValueError("fixture marker field schema is invalid")
    if (
        marker["trace_schema"] != BROWSER_TRACE_SCHEMA
        or marker["type"] != "browser.fixture_boundary"
    ):
        raise ValueError("fixture marker has the wrong trace identity")
    if marker["marker_schema"] != MARKER_SCHEMA:
        raise ValueError("fixture marker has the wrong marker identity")
    if type(marker["fixture_index"]) is not int or marker["fixture_index"] < 0:
        raise ValueError("fixture marker index must be a non-negative integer")
    _nonempty_string(marker["fixture_id"], "fixture marker fixture_id")
    _nonempty_string(marker["script_id"], "fixture marker script_id")
    if not isinstance(marker["capture_enabled"], bool) or marker["boundary"] not in _BOUNDARIES:
        raise ValueError("fixture marker boundary or capture mode is invalid")
    for field in ("browser_performance_ms", "browser_wall_time_s"):
        _finite_number(marker[field], f"fixture marker {field}", minimum=0, maximum=1e100)
    if "detail" in marker and not isinstance(marker["detail"], str):
        raise ValueError("fixture marker detail must be a string")


_BROWSER_COMMON = {
    "trace_schema",
    "type",
    "playwright_batch_received_monotonic_s",
}
_AUDIO_FIELDS = {
    "worklet_id",
    "main_thread_received_performance_ms",
    "render_audio_context_time_s",
    "render_audio_context_frame",
    "ordinal",
    "sample_count",
    "channel_count",
    "rms",
    "underrun",
    "underrun_count",
}


def validate_browser_record(record: Mapping[str, object], *, persisted: bool = True) -> None:
    """Validate one member of the complete browser-artifact record union."""

    event_type = record.get("type")
    common = _BROWSER_COMMON if persisted else {"trace_schema", "type"}
    fields = set(record)
    if record.get("trace_schema") != BROWSER_TRACE_SCHEMA:
        raise ValueError("browser artifact has the wrong trace schema")
    if event_type == "browser.fixture_boundary":
        validate_marker(record)
        if (
            persisted
            and not {
                "playwright_batch_received_monotonic_s",
                "playwright_batch_received_wall_time_s",
            }
            <= fields
        ):
            raise ValueError("persisted fixture marker lacks receipt clocks")
        if persisted:
            _finite_number(
                record["playwright_batch_received_monotonic_s"],
                "fixture marker receipt monotonic clock",
                minimum=0,
                maximum=1e100,
            )
            _finite_number(
                record["playwright_batch_received_wall_time_s"],
                "fixture marker receipt wall clock",
                minimum=0,
                maximum=1e100,
            )
        return
    variants: list[set[str]] = []
    if event_type == "browser.playout.capability":
        variants = [
            common | {"audio_worklet"},
            common | {"audio_worklet", "media_stream_track_processor"},
            common
            | {
                "audio_worklet",
                "media_stream_track_processor",
                "zero_pcm_classification",
                "audio_context_sample_rate_hz",
                "render_path",
                "audible_element_render_path_observed",
                "render_clock",
                "main_thread_clock",
                "caveat",
                "export_chunk_records",
                "worklet_batch_records",
                "max_pending_binding_writes",
            },
        ]
    elif event_type == "browser.clock_mapping":
        variants = [
            common
            | {
                "phase",
                "main_thread_performance_ms",
                "audio_context_time_s",
                "output_context_time_s",
                "output_performance_time_ms",
            }
        ]
    elif event_type == "browser.audio_worklet.registered":
        variants = [common | {"worklet_id"}]
    elif event_type in {"browser.audio_worklet.dequeue", "browser.audio_worklet.onset"}:
        variants = [common | _AUDIO_FIELDS]
    elif event_type == "browser.audio_worklet.flush_ack":
        variants = [common | {"worklet_id", "final_ordinal", "delivered_last_ordinal"}]
    elif event_type == "browser.remote_audio.enqueue":
        variants = [
            common
            | {
                "main_thread_observed_performance_ms",
                "ordinal",
                "media_timestamp_us",
                "media_duration_us",
                "sample_count",
                "sample_rate_hz",
                "channel_count",
            }
        ]
    if not variants or fields not in variants:
        raise ValueError(f"browser artifact field schema mismatch for {event_type!r}")
    if persisted:
        _finite_number(
            record["playwright_batch_received_monotonic_s"],
            "browser receipt clock",
            minimum=0,
            maximum=1e100,
        )
    if event_type == "browser.playout.capability":
        if not isinstance(record.get("audio_worklet"), bool) or (
            "media_stream_track_processor" in record
            and not isinstance(record.get("media_stream_track_processor"), bool)
        ):
            raise ValueError("browser capability booleans are invalid")
    elif event_type == "browser.clock_mapping":
        if record.get("phase") not in {"startup", "teardown"}:
            raise ValueError("browser clock mapping phase is invalid")
        for field in ("main_thread_performance_ms", "audio_context_time_s"):
            _finite_number(record[field], field, minimum=0, maximum=1e100)
    elif event_type == "browser.audio_worklet.registered":
        _nonempty_string(record.get("worklet_id"), "browser worklet_id")
    elif event_type in {"browser.audio_worklet.dequeue", "browser.audio_worklet.onset"}:
        _nonempty_string(record.get("worklet_id"), "browser worklet_id")
        for field in ("ordinal", "sample_count", "channel_count", "underrun_count"):
            if type(record.get(field)) is not int or int(record[field]) < 0:
                raise ValueError(f"browser audio {field} is invalid")
        if int(record["ordinal"]) < 1 or int(record["channel_count"]) < 1:
            raise ValueError("browser audio ordinal/channel count is invalid")
        if not isinstance(record.get("underrun"), bool):
            raise ValueError("browser audio underrun flag is invalid")
        for field in (
            "main_thread_received_performance_ms",
            "render_audio_context_time_s",
            "render_audio_context_frame",
            "rms",
        ):
            _finite_number(record[field], field, minimum=0, maximum=1e100)
    elif event_type == "browser.audio_worklet.flush_ack":
        _nonempty_string(record.get("worklet_id"), "browser worklet_id")
        if any(
            type(record.get(field)) is not int or int(record[field]) < 0
            for field in ("final_ordinal", "delivered_last_ordinal")
        ):
            raise ValueError("browser flush acknowledgement ordinals are invalid")
    elif event_type == "browser.remote_audio.enqueue":
        for field in ("ordinal", "sample_count", "sample_rate_hz", "channel_count"):
            if type(record.get(field)) is not int or int(record[field]) < 1:
                raise ValueError(f"browser remote audio {field} is invalid")
        _finite_number(
            record["main_thread_observed_performance_ms"],
            "browser remote receipt clock",
            minimum=0,
            maximum=1e100,
        )


def validate_browser_artifact(
    records: Sequence[Mapping[str, object]], *, capture_enabled: bool
) -> dict[str, Any]:
    """Consume a complete canonical session trace and reconcile terminal ordinals."""

    if not records:
        raise ValueError("browser artifact is empty")
    registered: dict[str, int] = {}
    acknowledged: dict[str, int] = {}
    marker_count = 0
    marker_state: dict[int, list[str]] = {}
    marker_sequence: list[dict[str, object]] = []
    marker_brackets: list[dict[str, object]] = []
    last_fixture_index = -1
    last_receipt = -1.0
    last_marker_performance = -1.0
    last_marker_wall = -1.0
    for index, record in enumerate(records):
        validate_browser_record(record)
        receipt = float(record["playwright_batch_received_monotonic_s"])
        if receipt < last_receipt:
            raise ValueError(f"browser receipt clock regresses at record {index}")
        last_receipt = receipt
        event_type = record["type"]
        if event_type == "browser.fixture_boundary":
            marker_count += 1
            if record["capture_enabled"] is not capture_enabled:
                raise ValueError("fixture marker capture mode disagrees with its artifact")
            fixture_index = int(record["fixture_index"])
            boundary = str(record["boundary"])
            marker_performance = float(record["browser_performance_ms"])
            marker_wall = float(record["browser_wall_time_s"])
            if marker_performance < last_marker_performance or marker_wall < last_marker_wall:
                raise ValueError("fixture marker browser clocks regress")
            last_marker_performance = marker_performance
            last_marker_wall = marker_wall
            prior = marker_state.setdefault(fixture_index, [])
            if fixture_index < last_fixture_index:
                raise ValueError("fixture marker indices regress")
            if not prior and boundary != "start":
                raise ValueError("fixture terminal/interruption marker precedes its start")
            if prior and prior[-1] in {"complete", "timeout", "error"}:
                raise ValueError("fixture marker appears after a terminal boundary")
            if boundary == "start" and prior:
                raise ValueError("fixture start marker is duplicated")
            if boundary == "interruption-send" and (
                prior != ["start"] or record.get("script_id") != "I1"
            ):
                raise ValueError("interruption marker is out of order or not I1")
            if boundary in {"complete", "timeout", "error"} and prior not in (
                ["start"],
                ["start", "interruption-send"],
            ):
                raise ValueError("fixture terminal marker is out of order")
            prior.append(boundary)
            last_fixture_index = fixture_index
            marker_sequence.append(
                {
                    "fixture_index": fixture_index,
                    "fixture_id": record["fixture_id"],
                    "script_id": record["script_id"],
                    "capture_enabled": record["capture_enabled"],
                    "boundary": boundary,
                }
            )
            marker_brackets.append(
                {
                    "record_index": index,
                    "fixture_index": fixture_index,
                    "fixture_id": record["fixture_id"],
                    "script_id": record["script_id"],
                    "boundary": boundary,
                    "browser_performance_ms": marker_performance,
                    "browser_wall_time_s": marker_wall,
                    "playwright_batch_received_monotonic_s": receipt,
                    "playwright_batch_received_wall_time_s": record[
                        "playwright_batch_received_wall_time_s"
                    ],
                }
            )
        elif event_type == "browser.audio_worklet.registered":
            worklet_id = str(record["worklet_id"])
            if worklet_id in registered:
                raise ValueError("duplicate browser worklet registration")
            registered[worklet_id] = 0
        elif event_type == "browser.audio_worklet.dequeue":
            worklet_id = str(record["worklet_id"])
            ordinal = record["ordinal"]
            if (
                worklet_id not in registered
                or worklet_id in acknowledged
                or type(ordinal) is not int
                or ordinal != registered[worklet_id] + 1
            ):
                raise ValueError("browser worklet dequeue ordinal is not contiguous")
            registered[worklet_id] = ordinal
        elif event_type == "browser.audio_worklet.onset":
            worklet_id = str(record["worklet_id"])
            if (
                worklet_id not in registered
                or worklet_id in acknowledged
                or record["ordinal"] != registered[worklet_id]
            ):
                raise ValueError("browser onset is not a member of the current dequeue")
        elif event_type == "browser.audio_worklet.flush_ack":
            worklet_id = str(record["worklet_id"])
            final = record["final_ordinal"]
            if worklet_id not in registered or worklet_id in acknowledged or type(final) is not int:
                raise ValueError("invalid browser worklet terminal acknowledgement")
            if record["delivered_last_ordinal"] != final:
                raise ValueError("browser worklet delivered/final ordinal mismatch")
            acknowledged[worklet_id] = final
    if capture_enabled:
        if not any(
            record["type"] == "browser.playout.capability" and record.get("audio_worklet") is True
            for record in records
        ):
            raise ValueError("capture-on browser artifact lacks capability evidence")
        if not registered or registered != acknowledged:
            raise ValueError("browser artifact terminal ordinals do not reconcile")
    elif registered or acknowledged:
        raise ValueError("capture-off browser artifact contains worklet state")
    plan = parse_fixture_plan(canonical_plan_json())
    expected_fixtures = plan.sessions()[0 if capture_enabled else 1][1]
    expected_sequence = [
        {
            "fixture_index": fixture.plan_index,
            "fixture_id": fixture.fixture_id,
            "script_id": fixture.script_id,
            "capture_enabled": capture_enabled,
            "boundary": boundary,
        }
        for fixture in expected_fixtures
        for boundary in (
            ("start", "interruption-send", "complete")
            if fixture.interruption is not None
            else ("start", "complete")
        )
    ]
    if marker_sequence != expected_sequence:
        raise ValueError("browser artifact lacks the exact canonical fixture marker sequence")
    return {
        "valid": True,
        "record_count": len(records),
        "fixture_marker_records": marker_count,
        "marker_sequence": marker_sequence,
        "marker_brackets": marker_brackets,
        "persisted_last_ordinals": registered,
        "acknowledged_final_ordinals": acknowledged,
    }


def load_browser_artifact(
    path: Path,
    *,
    capture_enabled: bool,
    reservation: Step2PathReservation | None = None,
) -> dict[str, Any]:
    content = (
        reservation.read_child_text(path)
        if reservation is not None
        else path.read_text(encoding="utf-8")
    )
    records = [json.loads(line) for line in content.splitlines()]
    return validate_browser_artifact(records, capture_enabled=capture_enabled)


def connection_playout_trace_path(template: Path, ordinal: int) -> Path:
    if type(ordinal) is not int or not 1 <= ordinal <= 9999:
        raise ValueError("playout trace connection ordinal must be between 1 and 9999")
    text = str(template)
    if PLAYOUT_TRACE_CONNECTION_TOKEN not in text:
        raise ValueError("Pipecat playout trace path requires the literal {connection} token")
    return Path(text.replace(PLAYOUT_TRACE_CONNECTION_TOKEN, f"{ordinal:04d}"))


def _absolute_unresolved(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_directory_nofollow(path: Path) -> int:
    """Open an absolute directory one component at a time without following links."""

    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"Step 2 directory path contains a symlink or nondirectory: {path}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        raise ValueError("Step 2 directory descriptor is not a directory")
    return opened.st_dev, opened.st_ino


def _open_regular_file_nofollow(path: Path) -> int:
    parent_fd = _open_directory_nofollow(path.parent)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Step 2 input is not a regular file: {path}")
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            os.close(descriptor)
            raise ValueError(f"Step 2 input identity changed while opening: {path}")
        return descriptor
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"Step 2 input path contains a symlink: {path}") from exc
        raise
    finally:
        os.close(parent_fd)


def _reserve_session_directory(path: Path) -> int:
    parent_fd = _open_directory_nofollow(path.parent)
    try:
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(f"Step 2 session directory must not be a symlink: {path}")
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    f"Step 2 session directory is not a real directory: {path}"
                )
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            if (before.st_dev, before.st_ino) != _descriptor_identity(descriptor):
                os.close(descriptor)
                raise ValueError(
                    f"Step 2 session directory identity changed while opening: {path}"
                )
            return descriptor
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    f"Step 2 session directory is not a real directory: {path}"
                ) from exc
            raise
    finally:
        os.close(parent_fd)


class Step2PathReservation(dict[str, str]):
    """Resolved paths plus the retained descriptor for their reserved session."""

    def __init__(self, paths: Mapping[str, str], descriptor: int):
        super().__init__(paths)
        self._descriptor = descriptor
        self._identity = _descriptor_identity(descriptor)

    @property
    def identity(self) -> tuple[int, int]:
        return self._identity

    @property
    def closed(self) -> bool:
        return self._descriptor < 0

    def close(self) -> None:
        if not self.closed:
            descriptor = self._descriptor
            self._descriptor = -1
            os.close(descriptor)

    def _require_descriptor(self) -> int:
        if self.closed:
            raise ValueError("Step 2 session directory reservation is closed")
        if _descriptor_identity(self._descriptor) != self._identity:
            raise ValueError("Step 2 retained session directory identity changed")
        return self._descriptor

    def _child_name(self, path: Path) -> str:
        absolute = _absolute_unresolved(path)
        if absolute.parent != Path(self["output_dir"]):
            raise ValueError("Step 2 output must remain a direct child of its reservation")
        return absolute.name

    def revalidate_parent(self, path: Path) -> None:
        """Require the current pathname parent to still name the retained directory."""

        self._require_descriptor()
        absolute = _absolute_unresolved(path)
        if absolute.parent != Path(self["output_dir"]):
            raise ValueError("Step 2 Pipecat artifact escaped its reserved parent")
        try:
            parent_fd = _open_directory_nofollow(absolute.parent)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("Step 2 session pathname no longer names its reservation") from exc
        try:
            if _descriptor_identity(parent_fd) != self._identity:
                raise ValueError("Step 2 session pathname identity changed after reservation")
        finally:
            os.close(parent_fd)

    def open_child(
        self,
        path: Path,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        descriptor = self._require_descriptor()
        return os.open(
            self._child_name(path),
            flags | os.O_NOFOLLOW,
            mode,
            dir_fd=descriptor,
        )

    def read_child_text(self, path: Path) -> str:
        descriptor = self.open_child(path, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            return stream.read()

    def child_exists(self, path: Path) -> bool:
        descriptor = self._require_descriptor()
        try:
            child = os.stat(
                self._child_name(path),
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(child.st_mode):
            raise ValueError("Step 2 output child became a symlink")
        return True

    def replace_child(self, source: Path, destination: Path) -> None:
        descriptor = self._require_descriptor()
        os.replace(
            self._child_name(source),
            self._child_name(destination),
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    def unlink_child(self, path: Path) -> None:
        descriptor = self._require_descriptor()
        os.unlink(self._child_name(path), dir_fd=descriptor)


def resolve_step2_paths(
    *,
    microphone: Path,
    capture_on: Path,
    capture_off: Path,
    summary: Path,
    pipecat_trace: Path,
    output_dir: Path,
) -> Step2PathReservation:
    """Validate and reserve campaign paths through retained no-follow descriptors."""

    raw = {
        "microphone": _absolute_unresolved(microphone),
        "browser_capture_on": _absolute_unresolved(capture_on),
        "browser_capture_off": _absolute_unresolved(capture_off),
        "summary": _absolute_unresolved(summary),
        "pipecat_playout_trace": _absolute_unresolved(pipecat_trace),
        "output_dir": _absolute_unresolved(output_dir),
    }
    if PLAYOUT_TRACE_CONNECTION_TOKEN not in str(raw["pipecat_playout_trace"]):
        raise ValueError("Pipecat playout trace path requires the literal {connection} token")
    output = raw["output_dir"]
    output_names = {
        name: path.name
        for name, path in raw.items()
        if name in {"browser_capture_on", "browser_capture_off", "summary", "pipecat_playout_trace"}
        and path.parent == output
    }
    if set(output_names) != {
        "browser_capture_on",
        "browser_capture_off",
        "summary",
        "pipecat_playout_trace",
    }:
        raise ValueError("Step 2 output files must be direct children of output_dir")

    resolved = dict(raw)
    resolved["pipecat_playout_capture_on"] = connection_playout_trace_path(
        raw["pipecat_playout_trace"], 1
    )
    resolved["pipecat_playout_capture_off"] = connection_playout_trace_path(
        raw["pipecat_playout_trace"], 2
    )
    file_paths = {
        key: value
        for key, value in resolved.items()
        if key not in {"output_dir", "pipecat_playout_trace"}
    }
    reverse: dict[Path, str] = {}
    for name, path in file_paths.items():
        if path in reverse:
            raise ValueError(f"Step 2 paths collide after resolution: {reverse[path]} and {name}")
        reverse[path] = name
    try:
        microphone_fd = _open_regular_file_nofollow(resolved["microphone"])
    except FileNotFoundError as exc:
        raise ValueError(f"microphone input does not exist: {resolved['microphone']}") from exc
    os.close(microphone_fd)
    try:
        output_fd = _reserve_session_directory(output)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Step 2 session parent directory does not exist: {output.parent}"
        ) from exc
    try:
        for name in (
            "browser_capture_on",
            "browser_capture_off",
            "summary",
            "pipecat_playout_capture_on",
            "pipecat_playout_capture_off",
        ):
            try:
                existing = os.stat(
                    resolved[name].name, dir_fd=output_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(existing.st_mode):
                raise ValueError(f"Step 2 output must not be a symlink: {resolved[name]}")
            raise FileExistsError(f"refusing existing Step 2 output {name}: {resolved[name]}")
        try:
            descriptor = os.open(
                ".step2-live-fixture-session",
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=output_fd,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"refusing reused Step 2 session directory: {output}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps({key: str(value) for key, value in resolved.items()}, sort_keys=True)
                + "\n"
            )
    except BaseException:
        os.close(output_fd)
        raise
    return Step2PathReservation(
        {key: str(value) for key, value in resolved.items()},
        output_fd,
    )


def _revalidate_pipecat_parent(
    path: Path, reservation: Step2PathReservation | None
) -> Path:
    absolute = _absolute_unresolved(path)
    if reservation is not None:
        reservation.revalidate_parent(absolute)
    return absolute


def active_playout_trace_path(
    path: Path, *, reservation: Step2PathReservation | None = None
) -> Path:
    """Resolve the published trace or the writer's sole live immutable staging file."""

    path = _revalidate_pipecat_parent(path, reservation)
    pending = sorted(path.parent.glob(f".{path.name}.*.pending"))
    if len(pending) == 1:
        return pending[0]
    if not pending and path.is_file():
        return path
    raise ValueError(f"expected exactly one live Pipecat trace for {path}, found {len(pending)}")


def read_playout_trace(
    path: Path, *, reservation: Step2PathReservation | None = None
) -> list[dict[str, Any]]:
    active = active_playout_trace_path(path, reservation=reservation)
    _revalidate_pipecat_parent(active, reservation)
    content = active.read_text(encoding="utf-8")
    lines = content.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not content.endswith("\n"):
                break
            raise
        if not isinstance(record, dict) or record.get("trace_schema") != PLAYOUT_TRACE_SCHEMA:
            raise ValueError("Pipecat playout trace contains a foreign record")
        records.append(record)
    return records


def read_published_playout_trace(
    path: Path, *, reservation: Step2PathReservation | None = None
) -> list[dict[str, Any]]:
    path = _revalidate_pipecat_parent(path, reservation)
    if not path.is_file():
        raise ValueError(f"Pipecat playout trace was not published: {path}")
    pending = list(path.parent.glob(f".{path.name}.*.pending"))
    if pending:
        raise ValueError(f"published Pipecat trace still has live staging evidence: {path}")
    _revalidate_pipecat_parent(path, reservation)
    content = path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        raise ValueError("published Pipecat playout trace has a partial terminal line")
    records = [json.loads(line) for line in content.splitlines()]
    if not records:
        raise ValueError("published Pipecat playout trace is empty")
    if not all(
        isinstance(record, dict) and record.get("trace_schema") == PLAYOUT_TRACE_SCHEMA
        for record in records
    ):
        raise ValueError("published Pipecat playout trace contains a foreign record")
    return records


def validate_published_playout_artifact(
    path: Path, *, reservation: Step2PathReservation | None = None
) -> dict[str, Any]:
    records = read_published_playout_trace(path, reservation=reservation)
    checked = validate_observed_playout_trace(records)
    expected_terminal = {
        "trace_schema": PLAYOUT_TRACE_SCHEMA,
        "type": "voicechat.playout.trace_status",
        "trace_closed_monotonic_s": records[-1].get("trace_closed_monotonic_s"),
        "valid": True,
        "invalid_reason": None,
        "error": None,
        "dropped_records": 0,
    }
    if records[-1] != expected_terminal:
        raise ValueError("Pipecat playout trace terminal status is not exactly successful")
    if checked.get("observed") is not True or checked.get("valid") is not True:
        raise ValueError("canonical observed-playout consumer did not validate the artifact")
    return dict(checked)


async def wait_for_playout_trace_publication(
    path: Path,
    *,
    reservation: Step2PathReservation | None = None,
    timeout_s: float = PLAYOUT_TRACE_PUBLICATION_TIMEOUT_S,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    timeout = _finite_number(timeout_s, "playout publication timeout", minimum=0.001, maximum=300)
    if monotonic is None:
        monotonic = asyncio.get_running_loop().time
    deadline = monotonic() + timeout
    last_error: Exception | None = None
    while True:
        path = _revalidate_pipecat_parent(path, reservation)
        pending = list(path.parent.glob(f".{path.name}.*.pending"))
        if path.is_file() and not pending:
            return validate_published_playout_artifact(path, reservation=reservation)
        last_error = ValueError(
            f"published={path.is_file()} pending_staging_records={len(pending)}"
        )
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        await sleep(min(PLAYOUT_TRACE_PUBLICATION_POLL_S, remaining))
    raise TimeoutError(f"Pipecat playout trace did not publish cleanly: {path}: {last_error}")


def validate_connection_playout_trace_set(
    template: Path, *, reservation: Step2PathReservation | None = None
) -> tuple[Path, Path]:
    template = _revalidate_pipecat_parent(template, reservation)
    expected = tuple(connection_playout_trace_path(template, ordinal) for ordinal in (1, 2))
    prefix, suffix = template.name.split(PLAYOUT_TRACE_CONNECTION_TOKEN)

    def matching_trace(path: Path) -> bool:
        name = path.name
        candidates = (name, name[1:] if name.startswith(".") else name)
        for candidate in candidates:
            ordinal_end = len(prefix) + 4
            if (
                not candidate.startswith(prefix)
                or len(candidate) < ordinal_end + len(suffix)
                or not candidate[len(prefix) : ordinal_end].isdigit()
                or not candidate[ordinal_end:].startswith(suffix)
            ):
                continue
            tail = candidate[ordinal_end + len(suffix) :]
            if tail == "" or (tail.startswith(".") and not tail.endswith(".lock")):
                return True
        return False

    _revalidate_pipecat_parent(template, reservation)
    observed = {path for path in template.parent.iterdir() if matching_trace(path)}
    quarantine = template.parent / ".voicechat-playout-trace-quarantine"
    if quarantine.is_dir():
        observed.update(path for path in quarantine.iterdir() if matching_trace(path))
    _revalidate_pipecat_parent(template, reservation)
    if observed != set(expected) or not all(path.is_file() for path in expected):
        raise ValueError(
            "Step 2 requires exactly the closed 0001 capture-on and 0002 capture-off "
            "Pipecat artifacts with no staged/quarantined traces; "
            f"observed={sorted(map(str, observed))}"
        )
    return expected


def _clock(record: Mapping[str, object]) -> float:
    clocks = [value for key, value in record.items() if key.endswith("_monotonic_s")]
    if len(clocks) != 1:
        raise ValueError("playout record must carry exactly one receipt/release clock")
    return _finite_number(clocks[0], "playout record clock", minimum=0, maximum=1e100)


def correlate_response_lifecycles(
    records: Sequence[Mapping[str, object]], *, interruption: bool
) -> dict[str, Any] | None:
    """Require an exact one-response, or exact cancel-then-replace, trace segment."""

    relevant_types = {
        "response.created",
        "response.output_text.delta",
        "response.output_audio.delta",
        "response.done",
        "input_audio_buffer.speech_started",
        "voicechat.playout.interruption",
        "voicechat.playout.release",
        "voicechat.playout.downstream_push",
    }
    relevant = [record for record in records if record.get("type") in relevant_types]
    indexed_relevant = list(enumerate(relevant))
    clocks = [_clock(record) for record in relevant]
    if clocks != sorted(clocks):
        raise ValueError("fixture playout trace clocks regress")
    created = [
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "response.created"
    ]
    expected = 2 if interruption else 1
    if len(created) < expected:
        return None
    if len(created) != expected:
        raise ValueError("fixture trace contains extra response lifecycles")
    response_ids = [
        _nonempty_string(record.get("response_id"), "response.created response_id")
        for _, record in created
    ]
    if len(set(response_ids)) != expected:
        raise ValueError("fixture response identifiers are empty or repeated")
    foreign = {
        record.get("response_id")
        for record in relevant
        if record.get("response_id") is not None and record.get("response_id") not in response_ids
    }
    if foreign:
        raise ValueError("fixture trace contains stale or foreign response transitions")
    typed_edges = [
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "input_audio_buffer.speech_started" and record.get("source") == "typed"
    ]
    idle_typed_edges = [
        (index, record)
        for index, record in typed_edges
        if record.get("response_id") is None
    ]
    expected_typed_sends = 2 if interruption else 1
    if (
        len(idle_typed_edges) != expected_typed_sends
        or idle_typed_edges[0][0] >= created[0][0]
    ):
        if interruption:
            raise ValueError("fixture lacks two typed-send edges for its response lifecycle")
        raise ValueError("fixture lacks one typed-send edge before its response lifecycle")
    typed_send_edge = {
        "type": idle_typed_edges[0][1]["type"],
        "source": "typed",
        "monotonic_s": _clock(idle_typed_edges[0][1]),
    }
    interruption_typed_send_edge = (
        {
            "type": idle_typed_edges[1][1]["type"],
            "source": "typed",
            "monotonic_s": _clock(idle_typed_edges[1][1]),
        }
        if interruption
        else None
    )
    if interruption and not (
        created[0][0] < idle_typed_edges[1][0] < created[1][0]
    ):
        raise ValueError("I1 interruption typed-send edge is outside its response transition")
    lifecycle: dict[str, dict[str, Any]] = {}
    for response_id in response_ids:
        owned = [
            (index, record)
            for index, record in indexed_relevant
            if record.get("response_id") == response_id
        ]
        deltas = [
            (index, record)
            for index, record in owned
            if record["type"] == "response.output_audio.delta"
        ]
        text_deltas = [
            (index, record)
            for index, record in owned
            if record["type"] == "response.output_text.delta"
        ]
        terminals = [
            (index, record)
            for index, record in owned
            if record["type"] == "response.done"
        ]
        releases = [
            (index, record)
            for index, record in owned
            if record["type"] == "voicechat.playout.release"
        ]
        pushes = [
            (index, record)
            for index, record in owned
            if record["type"] == "voicechat.playout.downstream_push"
        ]
        if not terminals:
            return None
        if len(terminals) != 1 or not deltas or not text_deltas or not releases:
            raise ValueError("response lifecycle lacks unique terminal or audio membership")
        text_ordinals = [record.get("ordinal") for _, record in text_deltas]
        if text_ordinals != list(range(1, len(text_ordinals) + 1)) or not all(
            isinstance(record.get("delta"), str) and record["delta"]
            for _, record in text_deltas
        ):
            raise ValueError("response text delta membership is invalid")
        delta_ordinals = [record.get("ordinal") for _, record in deltas]
        if delta_ordinals != list(range(1, len(delta_ordinals) + 1)):
            raise ValueError("response audio delta ordinals are not contiguous")
        released = [
            ordinal
            for _, record in releases
            for ordinal in record.get("ordinals", [])
            if record.get("frames_released", 0)
        ]
        cleared = [
            ordinal
            for _, record in releases
            for ordinal in record.get("ordinals", [])
            if record.get("frames_cleared", 0)
        ]
        pushed = [record.get("ordinal") for _, record in pushes]
        if sorted(pushed + cleared) != delta_ordinals:
            raise ValueError("response audio deltas, releases, clears, and pushes disagree")
        created_position, _created_record = next(
            (index, record)
            for index, record in owned
            if record["type"] == "response.created"
        )
        if any(
            index <= created_position
            for index, record in owned
            if record["type"] != "response.created"
        ):
            raise ValueError("response transition precedes response.created")
        terminal_position, terminal_record = terminals[0]
        if any(index >= terminal_position for index, _record in deltas):
            raise ValueError("response audio delta appears after its terminal")
        delta_by_ordinal = {record["ordinal"]: (index, record) for index, record in deltas}
        release_by_ordinal: dict[int, tuple[int, Mapping[str, object]]] = {}
        for release_position, release in releases:
            ordinals = release.get("ordinals")
            if not isinstance(ordinals, list):
                raise ValueError("response release lacks ordinal membership")
            for ordinal in ordinals:
                if type(ordinal) is not int or ordinal in release_by_ordinal:
                    raise ValueError("response ordinal is released or cleared more than once")
                release_by_ordinal[ordinal] = (release_position, release)
        push_by_ordinal: dict[int, tuple[int, Mapping[str, object]]] = {}
        for push_position, push in pushes:
            ordinal = push.get("ordinal")
            if type(ordinal) is not int or ordinal in push_by_ordinal:
                raise ValueError("response ordinal is pushed more than once")
            push_by_ordinal[ordinal] = (push_position, push)
        playout_starts = [
            (index, release)
            for index, release in releases
            if bool(release.get("frames_released", 0))
        ]
        clears = [
            (index, release)
            for index, release in releases
            if release.get("reason") == "interruption-clear"
        ]
        for ordinal, (delta_position, _delta) in delta_by_ordinal.items():
            release_entry = release_by_ordinal.get(ordinal)
            push_entry = push_by_ordinal.get(ordinal)
            if release_entry is not None and delta_position >= release_entry[0]:
                raise ValueError("response ordinal release/clear does not follow its delta")
            if release_entry is not None and ordinal in released:
                if push_entry is None or release_entry[0] >= push_entry[0]:
                    raise ValueError("response ordinal downstream push does not follow its release")
            elif release_entry is not None:
                if push_entry is not None:
                    raise ValueError("cleared response ordinal has a downstream push")
            else:
                if push_entry is None or delta_position >= push_entry[0]:
                    raise ValueError("response ordinal downstream push does not follow its delta")
                if not any(
                    playout_start_position < delta_position
                    for playout_start_position, _playout_start in playout_starts
                ):
                    raise ValueError(
                        "response ordinal was pushed before playout started or without release"
                    )
                if any(
                    delta_position < clear_position < push_entry[0]
                    for clear_position, _clear in clears
                ):
                    raise ValueError("response ordinal downstream push follows its clear")
                if push_entry[0] != delta_position + 1 or push_entry[0] >= terminal_position:
                    raise ValueError(
                        "post-release response ordinal downstream push is not immediate"
                    )
            if ordinal in cleared and push_entry is not None:
                raise ValueError("cleared response ordinal has a downstream push")
        release_membership = [
            {
                "reason": record.get("reason"),
                "ordinals": list(record.get("ordinals", [])),
                "frames_released": record.get("frames_released"),
                "frames_cleared": record.get("frames_cleared"),
            }
            for _, record in releases
        ]
        lifecycle[response_id] = {
            "response_id": response_id,
            "created_monotonic_s": _clock(
                next(record for _, record in owned if record["type"] == "response.created")
            ),
            "audio_delta_ordinals": delta_ordinals,
            "output_text": "".join(str(record["delta"]) for _, record in text_deltas),
            "release_reasons": [record.get("reason") for _, record in releases],
            "release_membership": release_membership,
            "downstream_push_ordinals": pushed,
            "terminal": {
                "status": terminal_record.get("status"),
                "monotonic_s": _clock(terminal_record),
            },
        }
    if not interruption:
        if any(
            (
                record["type"] == "input_audio_buffer.speech_started"
                and record.get("response_id") is not None
            )
            or record["type"] == "voicechat.playout.interruption"
            or (
                record["type"] == "voicechat.playout.release"
                and record.get("reason") == "interruption-clear"
            )
            for record in relevant
        ):
            raise ValueError("ordinary fixture contains an interruption transition")
        only = lifecycle[response_ids[0]]
        if only["terminal"]["status"] != "completed":
            raise ValueError("ordinary fixture response terminal is not completed")
        return {
            "response_ids": response_ids,
            "typed_send_edge": typed_send_edge,
            "primary_response": only,
        }

    original_id, replacement_id = response_ids
    edges = [
        (index, record)
        for index, record in indexed_relevant
        if (
            record["type"] == "input_audio_buffer.speech_started"
            and record.get("response_id") is not None
        )
        or record["type"] == "voicechat.playout.interruption"
    ]
    clears = [
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "voicechat.playout.release"
        and record.get("reason") == "interruption-clear"
    ]
    if len(edges) != 1 or len(clears) != 1:
        raise ValueError("I1 requires exactly one interruption edge and clear release")
    if (
        edges[0][1].get("response_id") != original_id
        or clears[0][1].get("response_id") != original_id
    ):
        raise ValueError("I1 interruption edge is not attributed to the original response")
    original_terminal = next(
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "response.done" and record.get("response_id") == original_id
    )
    replacement_created = created[1]
    original_deltas = [
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "response.output_audio.delta"
        and record.get("response_id") == original_id
    ]
    original_pushes = [
        (index, record)
        for index, record in indexed_relevant
        if record["type"] == "voicechat.playout.downstream_push"
        and record.get("response_id") == original_id
    ]
    if not original_pushes:
        raise ValueError("I1 interruption lacks original audible downstream audio")
    if any(index >= edges[0][0] for index, _record in original_deltas):
        raise ValueError("I1 accepted original audio after its interruption edge")
    if any(index >= clears[0][0] for index, _record in original_pushes):
        raise ValueError("I1 accepted an original downstream push after its clear")
    ordered = [
        record[0]
        for record in (
            original_pushes[0],
            edges[0],
            clears[0],
            original_terminal,
            idle_typed_edges[1],
            replacement_created,
        )
    ]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("I1 interruption transitions are missing or out of order")
    if lifecycle[original_id]["terminal"]["status"] != "cancelled":
        raise ValueError("I1 original response terminal is not cancelled")
    if lifecycle[replacement_id]["terminal"]["status"] != "completed":
        raise ValueError("I1 replacement response terminal is not completed")
    return {
        "response_ids": response_ids,
        "typed_send_edge": typed_send_edge,
        "interruption_typed_send_edge": interruption_typed_send_edge,
        "original_response": lifecycle[original_id],
        "interruption_edge": {
            "type": edges[0][1]["type"],
            "monotonic_s": _clock(edges[0][1]),
        },
        "clear_edge": {
            "reason": "interruption-clear",
            "monotonic_s": _clock(clears[0][1]),
        },
        "replacement_response": lifecycle[replacement_id],
    }


def blank_fixture_result(fixture: FixtureSpec) -> dict[str, Any]:
    marker_emission = {
        boundary: {"status": "not_attempted", "error": None} for boundary in _BOUNDARIES
    }
    return {
        "schema": FIXTURE_RESULT_SCHEMA,
        "plan_index": fixture.plan_index,
        "fixture_id": fixture.fixture_id,
        "script_id": fixture.script_id,
        "capture": fixture.capture,
        "typed_text_sha256": fixture.typed_text_sha256,
        "response_ids": [],
        "response_attribution": None,
        "send_attempt_wall_time_s": None,
        "send_wall_time_s": None,
        "first_audio_wall_time_s": None,
        "last_audio_wall_time_s": None,
        "completion_wall_time_s": None,
        "completion_status": "not_run",
        "error": None,
        "marker_emission": marker_emission,
        "transcript_evidence": [],
        "transcript_gate_evidence": None,
        "quiescence_barrier": None,
        # Populated for timeout/error outcomes.  The ordered-channel index is
        # the causal authority; the two receipt clocks are retained only for
        # post-mortem evidence.
        "rtvi_window_start_index": None,
        "observed_rtvi_messages": [],
        "rtvi_observation_error": None,
        "interruption": (
            {
                "after_first_audio_s": fixture.interruption.after_first_audio_s,
                "typed_text": fixture.interruption.typed_text,
                "typed_text_sha256": fixture.interruption.typed_text_sha256,
                "send_attempt_wall_time_s": None,
                "send_wall_time_s": None,
                "rtvi_send_marker_event_index": None,
                "actual_offset_s": None,
                "second_answer_transcript_evidence": [],
                "second_answer_gate_evidence": None,
            }
            if fixture.interruption
            else None
        ),
    }


def _valid_marker_statuses(result: Mapping[str, object], *, completed: bool) -> None:
    emissions = _object(result.get("marker_emission"), "fixture marker_emission")
    if set(emissions) != _BOUNDARIES:
        raise ValueError("fixture marker_emission must enumerate every boundary")
    for boundary, raw_state in emissions.items():
        state = _object(raw_state, f"fixture marker {boundary}")
        if set(state) != {"status", "error"} or state["status"] not in _MARKER_STATUSES:
            raise ValueError("fixture marker emission status is invalid")
        if state["status"] == "failed" and not isinstance(state["error"], str):
            raise ValueError("failed marker emission requires an error")
        if state["status"] != "failed" and state["error"] is not None:
            raise ValueError("nonfailed marker emission cannot carry an error")
    if completed:
        required = {"start", "complete"}
        if result.get("interruption") is not None:
            required.add("interruption-send")
        if any(
            _object(emissions[name], "marker state")["status"] != "succeeded" for name in required
        ):
            raise ValueError("completed fixture lacks a required emitted marker")
        if any(
            _object(emissions[name], "marker state")["status"] != "not_attempted"
            for name in _BOUNDARIES - required
        ):
            raise ValueError("completed fixture contains an unexpected marker attempt")


def _validate_embedded_plan(value: object) -> tuple[Mapping[str, object], list[object]]:
    plan = _object(value, "summary plan")
    if set(plan) != {"schema", "plan_sha256", "script_hashes_sha256", "fixtures"}:
        raise ValueError("summary embeds an invalid plan field schema")
    fixtures = plan.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 8:
        raise ValueError("summary plan must contain eight fixtures")
    reconstructed: list[dict[str, object]] = []
    embedded_text: dict[str, str] = {}
    for index, raw_fixture in enumerate(fixtures):
        fixture = _object(raw_fixture, f"summary plan fixture {index}")
        text = _nonempty_string(fixture.get("typed_text"), "embedded fixture typed_text")
        if fixture.get("typed_text_sha256") != sha256_text(text):
            raise ValueError("summary fixture text hash does not match its embedded bytes")
        script_id = _nonempty_string(fixture.get("script_id"), "embedded fixture script_id")
        hash_id = "L1" if script_id == "I1" else script_id
        if hash_id in embedded_text and embedded_text[hash_id] != text:
            raise ValueError("summary embeds inconsistent bytes for a repeated script")
        embedded_text[hash_id] = text
        if script_id == "I1":
            embedded_text["I1"] = text
        interruption = fixture.get("interruption")
        raw: dict[str, object] = {
            "fixture_id": fixture.get("fixture_id"),
            "script_id": script_id,
            "typed_text": text,
            "capture": fixture.get("capture"),
            "wait": fixture.get("wait"),
        }
        if interruption is not None:
            embedded_interruption = _object(interruption, "embedded interruption")
            replacement = _nonempty_string(
                embedded_interruption.get("typed_text"), "embedded replacement text"
            )
            if embedded_interruption.get("typed_text_sha256") != sha256_text(replacement):
                raise ValueError("summary replacement hash does not match its embedded bytes")
            embedded_text["I1-replacement"] = replacement
            raw["interruption"] = {
                "after_first_audio_s": embedded_interruption.get("after_first_audio_s"),
                "typed_text": replacement,
            }
        reconstructed.append(raw)
    canonical_bytes = json.dumps(reconstructed, sort_keys=True, separators=(",", ":"))
    if plan.get("plan_sha256") != sha256_text(canonical_bytes):
        raise ValueError("summary plan hash does not match its embedded bytes")
    computed_hashes = {name: sha256_text(text) for name, text in embedded_text.items()}
    if plan.get("script_hashes_sha256") != computed_hashes:
        raise ValueError("summary script hashes do not match their embedded bytes")
    parsed = parse_fixture_plan(canonical_bytes)
    if plan != plan_document(parsed):
        raise ValueError("summary embeds the wrong canonical plan")
    return plan, fixtures


def _validate_rtvi_bracket(
    evidence: Mapping[str, object],
    *,
    response_id: str,
    output_text: str,
    transcripts: list[object],
    terminal_type: str | Sequence[str],
) -> tuple[float, float]:
    bracket = _object(evidence.get("rtvi_event_bracket"), "response RTVI event bracket")
    terminal_types = {terminal_type} if isinstance(terminal_type, str) else set(terminal_type)
    if (
        bracket.get("response_id") != response_id
        or bracket.get("transcript_binding") != RTVI_TRANSCRIPT_IDENTITY
        or bracket.get("protocol_response_identity") is not None
        or bracket.get("start_event_type") != "bot-started-speaking"
        or bracket.get("terminal_event_type") not in terminal_types
        or bracket.get("output_text_sha256") != sha256_text(output_text)
    ):
        raise ValueError("response RTVI bracket identity is invalid")
    start_index = bracket.get("start_event_index")
    terminal_index = bracket.get("terminal_event_index")
    transcript_indices = bracket.get("transcript_event_indices")
    transcript_clocks = bracket.get("transcript_performance_ms")
    if (
        type(start_index) is not int
        or type(terminal_index) is not int
        or start_index >= terminal_index
        or not isinstance(transcript_indices, list)
        or not isinstance(transcript_clocks, list)
        or len(transcript_indices) != len(transcript_clocks)
        or len(transcript_indices) != len(transcripts)
        or not all(
            type(index) is int and start_index < index < terminal_index
            for index in transcript_indices
        )
        or transcript_indices != sorted(set(transcript_indices))
    ):
        raise ValueError("response RTVI event-index bracket is invalid")
    start_performance = _finite_number(
        bracket.get("start_performance_ms"),
        "RTVI start performance clock",
        minimum=0,
        maximum=1e100,
    )
    terminal_performance = _finite_number(
        bracket.get("terminal_performance_ms"),
        "RTVI terminal performance clock",
        minimum=0,
        maximum=1e100,
    )
    checked_transcript_clocks = [
        _finite_number(clock, "RTVI transcript clock", minimum=0, maximum=1e100)
        for clock in transcript_clocks
    ]
    if not start_performance < terminal_performance or any(
        not start_performance < clock < terminal_performance
        for clock in checked_transcript_clocks
    ) or checked_transcript_clocks != sorted(checked_transcript_clocks):
        raise ValueError("response RTVI clock bracket is not strict")
    send_performance = _finite_number(
        bracket.get("fixture_send_performance_ms"),
        "fixture RTVI send-edge performance clock",
        minimum=0,
        maximum=1e100,
    )
    if not send_performance < start_performance or bracket.get(
        "messages_received_before_send_in_bracket"
    ) != 0:
        raise ValueError("response RTVI bracket contains messages from before its send edge")
    start_wall = _finite_number(
        bracket.get("start_wall_time_s"), "RTVI start wall clock", minimum=0, maximum=1e100
    )
    terminal_wall = _finite_number(
        bracket.get("terminal_wall_time_s"),
        "RTVI terminal wall clock",
        minimum=0,
        maximum=1e100,
    )
    send_wall = _finite_number(
        bracket.get("fixture_send_wall_time_s"),
        "fixture RTVI send-edge wall clock",
        minimum=0,
        maximum=1e100,
    )
    if not send_wall < start_wall < terminal_wall:
        raise ValueError("response RTVI wall-clock bracket is not strict")
    return start_performance, terminal_performance


def _validate_quiescence_barrier(value: object) -> None:
    barrier = _object(value, "fixture RTVI quiescence barrier")
    required = {
        "identity",
        "quiet_window_ms",
        "window_start_performance_ms",
        "window_end_performance_ms",
        "window_start_wall_time_s",
        "window_end_wall_time_s",
        "start_message_count",
        "end_message_count",
        "observed_message_count",
        "bot_message_count",
        "bot_message_types",
        "passed",
    }
    if set(barrier) != required or barrier.get("identity") != RTVI_TRANSCRIPT_IDENTITY:
        raise ValueError("fixture RTVI quiescence barrier schema is invalid")
    quiet_ms = validate_rtvi_quiescence_window_ms(barrier.get("quiet_window_ms"))
    start_performance = _finite_number(
        barrier.get("window_start_performance_ms"),
        "RTVI quiet-window start",
        minimum=0,
        maximum=1e100,
    )
    end_performance = _finite_number(
        barrier.get("window_end_performance_ms"),
        "RTVI quiet-window end",
        minimum=0,
        maximum=1e100,
    )
    start_wall = _finite_number(
        barrier.get("window_start_wall_time_s"),
        "RTVI quiet-window wall start",
        minimum=0,
        maximum=1e100,
    )
    end_wall = _finite_number(
        barrier.get("window_end_wall_time_s"),
        "RTVI quiet-window wall end",
        minimum=0,
        maximum=1e100,
    )
    counts = [
        barrier.get("start_message_count"),
        barrier.get("end_message_count"),
        barrier.get("observed_message_count"),
        barrier.get("bot_message_count"),
    ]
    if not all(type(count) is int and count >= 0 for count in counts):
        raise ValueError("RTVI quiet-window message counts are invalid")
    start_count, end_count, observed_count, bot_count = counts
    bot_types = barrier.get("bot_message_types")
    performance_window_complete = end_performance >= start_performance + quiet_ms
    wall_window_complete = (
        end_wall + RTVI_WALL_CLOCK_SKEW_TOLERANCE_S
        >= start_wall + quiet_ms / 1000.0
    )
    expected_passed = bot_count == 0 and performance_window_complete and wall_window_complete
    if (
        end_count - start_count != observed_count
        or bot_count > observed_count
        or not isinstance(bot_types, list)
        or not all(isinstance(item, str) and item.startswith("bot-") for item in bot_types)
        or len(bot_types) != bot_count
        or barrier.get("passed") is not expected_passed
    ):
        raise ValueError("RTVI quiet-window evidence is inconsistent")


def transcript_gate_evidence(
    browser_transcripts: Sequence[str], response_owned_traced_text: str
) -> dict[str, object]:
    """Record exact transcript equality as gate evidence, not fixture completion."""

    if not all(isinstance(text, str) and text for text in browser_transcripts):
        raise ValueError("browser transcript evidence must contain nonempty strings")
    response_text = _nonempty_string(response_owned_traced_text, "response-owned traced text")
    joined = "".join(browser_transcripts)
    equal = joined == response_text
    mismatch_reason = None
    if not equal:
        mismatch_reason = (
            TRANSCRIPT_MISMATCH_EMPTY if not joined else TRANSCRIPT_MISMATCH_DIFFERENT
        )
    return {
        "joined_browser_transcript": joined,
        "response_owned_traced_text": response_text,
        "transcript_equal": equal,
        "mismatch_reason": mismatch_reason,
    }


def _validate_transcript_gate_evidence(
    value: object,
    *,
    browser_transcripts: list[object],
    response_owned_traced_text: str,
) -> None:
    evidence = _object(value, "fixture transcript gate evidence")
    required = {
        "joined_browser_transcript",
        "response_owned_traced_text",
        "transcript_equal",
        "mismatch_reason",
    }
    if set(evidence) != required:
        raise ValueError("fixture transcript gate evidence schema is invalid")
    if type(evidence.get("transcript_equal")) is not bool:
        raise ValueError("fixture transcript gate evidence equality must be a boolean")
    expected = transcript_gate_evidence(
        [str(text) for text in browser_transcripts], response_owned_traced_text
    )
    if dict(evidence) != expected:
        raise ValueError("fixture transcript gate evidence is internally inconsistent")


def second_answer_gate_evidence(
    browser_transcripts: Sequence[str], replacement_response_owned_traced_text: str
) -> dict[str, object]:
    """Record the preregistered I1 answer check without gating completion."""

    if not all(isinstance(text, str) and text for text in browser_transcripts):
        raise ValueError("second-answer browser transcript evidence must contain strings")
    response_text = _nonempty_string(
        replacement_response_owned_traced_text,
        "replacement response-owned traced text",
    )
    return {
        "joined_browser_transcript": "".join(browser_transcripts),
        "response_owned_traced_text": response_text,
        "second_answer_adjudication": _is_second_answer(response_text),
    }


def _validate_second_answer_gate_evidence(
    value: object,
    *,
    browser_transcripts: list[object],
    replacement_response_owned_traced_text: str,
) -> None:
    evidence = _object(value, "I1 second-answer gate evidence")
    required = {
        "joined_browser_transcript",
        "response_owned_traced_text",
        "second_answer_adjudication",
    }
    if set(evidence) != required:
        raise ValueError("I1 second-answer gate evidence schema is invalid")
    if type(evidence.get("second_answer_adjudication")) is not bool:
        raise ValueError("I1 second-answer gate evidence adjudication must be a boolean")
    expected = second_answer_gate_evidence(
        [str(text) for text in browser_transcripts],
        replacement_response_owned_traced_text,
    )
    if dict(evidence) != expected:
        raise ValueError("I1 second-answer gate evidence is internally inconsistent")


def _validate_response_attribution(result: Mapping[str, object], response_ids: list[str]) -> None:
    attribution = _object(result.get("response_attribution"), "response attribution")
    if attribution.get("response_ids") != response_ids:
        raise ValueError("response attribution IDs disagree with the fixture IDs")
    typed_send = _object(attribution.get("typed_send_edge"), "typed send edge")
    if (
        typed_send.get("type") != "input_audio_buffer.speech_started"
        or typed_send.get("source") != "typed"
    ):
        raise ValueError("response attribution lacks its typed-send edge")
    typed_send_clock = _finite_number(
        typed_send.get("monotonic_s"),
        "typed send edge clock",
        minimum=0,
        maximum=1e100,
    )

    def response(
        value: object,
        label: str,
        response_id: str,
        status: str,
        *,
        terminal_type: str,
    ) -> tuple[float, float]:
        evidence = _object(value, label)
        if evidence.get("response_id") != response_id:
            raise ValueError(f"{label} has the wrong response ID")
        terminal = _object(evidence.get("terminal"), f"{label} terminal")
        if terminal.get("status") != status:
            raise ValueError(f"{label} has the wrong terminal status")
        created_clock = _finite_number(
            evidence.get("created_monotonic_s"),
            f"{label} creation clock",
            minimum=0,
            maximum=1e100,
        )
        terminal_clock = _finite_number(
            terminal.get("monotonic_s"),
            f"{label} terminal clock",
            minimum=0,
            maximum=1e100,
        )
        deltas = evidence.get("audio_delta_ordinals")
        releases = evidence.get("release_reasons")
        release_membership = evidence.get("release_membership")
        pushes = evidence.get("downstream_push_ordinals")
        output_text = evidence.get("output_text")
        if (
            not isinstance(deltas, list)
            or not deltas
            or deltas != list(range(1, len(deltas) + 1))
        ):
            raise ValueError(f"{label} lacks audio delta membership")
        if not isinstance(releases, list) or not releases:
            raise ValueError(f"{label} lacks release membership")
        if not isinstance(release_membership, list) or not release_membership:
            raise ValueError(f"{label} lacks exact release membership")
        if not isinstance(pushes, list) or not all(type(item) is int for item in pushes):
            raise ValueError(f"{label} lacks downstream-push membership")
        reconstructed: list[int] = []
        released_ordinals: list[int] = []
        cleared_ordinals: list[int] = []
        reconstructed_reasons: list[object] = []
        for raw_release in release_membership:
            release = _object(raw_release, f"{label} release membership")
            if set(release) != {
                "reason",
                "ordinals",
                "frames_released",
                "frames_cleared",
            }:
                raise ValueError(f"{label} release membership schema is invalid")
            ordinals = release.get("ordinals")
            frames_released = release.get("frames_released")
            frames_cleared = release.get("frames_cleared")
            if (
                not isinstance(ordinals, list)
                or not all(type(ordinal) is int and ordinal > 0 for ordinal in ordinals)
                or type(frames_released) is not int
                or type(frames_cleared) is not int
                or frames_released < 0
                or frames_cleared < 0
                or frames_released + frames_cleared != len(ordinals)
            ):
                raise ValueError(f"{label} release ordinal membership is invalid")
            reconstructed.extend(ordinals)
            if frames_released:
                released_ordinals.extend(ordinals)
            if frames_cleared:
                cleared_ordinals.extend(ordinals)
            reconstructed_reasons.append(release.get("reason"))
        if (
            len(set(reconstructed)) != len(reconstructed)
            or not set(reconstructed).issubset(deltas)
            or reconstructed_reasons != releases
            or sorted(pushes + cleared_ordinals) != deltas
            or not set(released_ordinals).issubset(pushes)
            or set(cleared_ordinals).intersection(pushes)
        ):
            raise ValueError(f"{label} audio membership fields disagree")
        output_text = _nonempty_string(output_text, f"{label} output text")
        transcripts = evidence.get("transcript_evidence", [])
        if not isinstance(transcripts, list) or not all(
            isinstance(text, str) and text for text in transcripts
        ):
            raise ValueError(f"{label} transcript membership is invalid")
        _validate_rtvi_bracket(
            evidence,
            response_id=response_id,
            output_text=output_text,
            transcripts=transcripts,
            terminal_type=terminal_type,
        )
        if created_clock > terminal_clock:
            raise ValueError(f"{label} terminal precedes response.created")
        return created_clock, terminal_clock

    if result.get("interruption") is None:
        created_clock, terminal_clock = response(
            attribution.get("primary_response"),
            "primary response",
            response_ids[0],
            "completed",
            terminal_type="bot-stopped-speaking",
        )
        if typed_send_clock > created_clock:
            raise ValueError("ordinary response creation precedes its typed-send edge")
        primary = _object(attribution.get("primary_response"), "primary response")
        speaking_times = [
            _finite_number(primary.get(field), field, minimum=0, maximum=1e100)
            for field in ("speaking_start_wall_time_s", "speaking_stop_wall_time_s")
        ]
        if speaking_times != sorted(speaking_times):
            raise ValueError("primary speaking lifecycle clocks are out of order")
        primary_bracket = _object(primary.get("rtvi_event_bracket"), "primary RTVI bracket")
        if speaking_times != [
            primary_bracket.get("start_wall_time_s"),
            primary_bracket.get("terminal_wall_time_s"),
        ]:
            raise ValueError("primary speaking clocks disagree with its RTVI bracket")
        return
    original_created, original_terminal = response(
        attribution.get("original_response"),
        "I1 original response",
        response_ids[0],
        "cancelled",
        terminal_type=("bot-interrupted", "bot-stopped-speaking"),
    )
    replacement_created, replacement_terminal = response(
        attribution.get("replacement_response"),
        "I1 replacement response",
        response_ids[1],
        "completed",
        terminal_type="bot-stopped-speaking",
    )
    edge = _object(attribution.get("interruption_edge"), "I1 interruption edge")
    interruption_typed_send = _object(
        attribution.get("interruption_typed_send_edge"),
        "I1 interruption typed-send edge",
    )
    clear = _object(attribution.get("clear_edge"), "I1 clear edge")
    if (
        edge.get("type")
        not in {
            "input_audio_buffer.speech_started",
            "voicechat.playout.interruption",
        }
        or interruption_typed_send.get("type")
        != "input_audio_buffer.speech_started"
        or interruption_typed_send.get("source") != "typed"
        or clear.get("reason") != "interruption-clear"
    ):
        raise ValueError("I1 attribution lacks its interruption send/clear edge")
    edge_clock = _finite_number(
        edge.get("monotonic_s"), "I1 interruption edge clock", minimum=0, maximum=1e100
    )
    clear_clock = _finite_number(
        clear.get("monotonic_s"), "I1 clear edge clock", minimum=0, maximum=1e100
    )
    interruption_typed_send_clock = _finite_number(
        interruption_typed_send.get("monotonic_s"),
        "I1 interruption typed-send edge clock",
        minimum=0,
        maximum=1e100,
    )
    lifecycle_clocks = [
        typed_send_clock,
        original_created,
        edge_clock,
        clear_clock,
        original_terminal,
        interruption_typed_send_clock,
        replacement_created,
        replacement_terminal,
    ]
    if lifecycle_clocks != sorted(lifecycle_clocks):
        raise ValueError("I1 response lifecycle clocks are out of causal order")
    original = _object(attribution.get("original_response"), "I1 original response")
    replacement = _object(attribution.get("replacement_response"), "I1 replacement response")
    speaking_times = [
        _finite_number(
            original.get("speaking_start_wall_time_s"),
            "original speaking start",
            minimum=0,
            maximum=1e100,
        ),
        _finite_number(
            original.get("interrupted_wall_time_s"),
            "original interruption",
            minimum=0,
            maximum=1e100,
        ),
        _finite_number(
            replacement.get("speaking_start_wall_time_s"),
            "replacement speaking start",
            minimum=0,
            maximum=1e100,
        ),
        _finite_number(
            replacement.get("speaking_stop_wall_time_s"),
            "replacement speaking stop",
            minimum=0,
            maximum=1e100,
        ),
    ]
    if speaking_times != sorted(speaking_times):
        raise ValueError("I1 browser response lifecycle clocks are out of order")
    original_bracket = _object(original.get("rtvi_event_bracket"), "original RTVI bracket")
    replacement_bracket = _object(
        replacement.get("rtvi_event_bracket"), "replacement RTVI bracket"
    )
    rtvi_window_start = result.get("rtvi_window_start_index")
    interruption_result = _object(result.get("interruption"), "I1 interruption result")
    marker_event_index = interruption_result.get("rtvi_send_marker_event_index")
    if (
        type(rtvi_window_start) is not int
        or rtvi_window_start < 0
        or type(marker_event_index) is not int
        or marker_event_index < rtvi_window_start
    ):
        raise ValueError("I1 lacks its ordered-channel interruption-send marker index")
    marker_window_index = marker_event_index - rtvi_window_start
    if not (
        int(original_bracket["start_event_index"])
        < marker_window_index
        <= int(original_bracket["terminal_event_index"])
        < int(replacement_bracket["start_event_index"])
    ):
        raise ValueError("I1 browser event indices violate its interruption marker ordering")
    if speaking_times != [
        original_bracket.get("start_wall_time_s"),
        original_bracket.get("terminal_wall_time_s"),
        replacement_bracket.get("start_wall_time_s"),
        replacement_bracket.get("terminal_wall_time_s"),
    ]:
        raise ValueError("I1 speaking clocks disagree with its RTVI brackets")


_RESPONSE_ARTIFACT_FIELDS = {
    "response_id",
    "created_monotonic_s",
    "audio_delta_ordinals",
    "output_text",
    "release_reasons",
    "release_membership",
    "downstream_push_ordinals",
    "terminal",
}


def _artifact_attribution_projection(
    attribution: Mapping[str, object], *, interruption: bool
) -> dict[str, object]:
    projected: dict[str, object] = {
        "response_ids": attribution.get("response_ids"),
        "typed_send_edge": attribution.get("typed_send_edge"),
    }
    response_names = (
        ("original_response", "replacement_response")
        if interruption
        else ("primary_response",)
    )
    for name in response_names:
        response = _object(attribution.get(name), f"artifact attribution {name}")
        projected[name] = {field: response.get(field) for field in _RESPONSE_ARTIFACT_FIELDS}
    if interruption:
        projected["interruption_typed_send_edge"] = attribution.get(
            "interruption_typed_send_edge"
        )
        projected["interruption_edge"] = attribution.get("interruption_edge")
        projected["clear_edge"] = attribution.get("clear_edge")
    return projected


def _partition_session_fixture_records(
    records: Sequence[Mapping[str, object]], fixtures: Sequence[FixtureSpec]
) -> list[Sequence[Mapping[str, object]]]:
    """Partition a serial session using the plan's typed-send cardinality."""

    typed_sends = [
        index
        for index, record in enumerate(records)
        if record.get("type") == "input_audio_buffer.speech_started"
        and record.get("source") == "typed"
        and record.get("response_id") is None
    ]
    expected_send_counts = [2 if fixture.interruption is not None else 1 for fixture in fixtures]
    if len(typed_sends) != sum(expected_send_counts):
        raise ValueError("retained Pipecat artifact has the wrong fixture-send membership")

    partitions: list[Sequence[Mapping[str, object]]] = []
    send_offset = 0
    for expected_send_count in expected_send_counts:
        start = typed_sends[send_offset]
        send_offset += expected_send_count
        stop = typed_sends[send_offset] if send_offset < len(typed_sends) else len(records)
        partitions.append(records[start:stop])
    return partitions


def reconstruct_session_response_attributions(
    records: Sequence[Mapping[str, object]], fixtures: Sequence[FixtureSpec]
) -> list[dict[str, object]]:
    """Rebuild each serial fixture attribution from one retained session artifact."""

    reconstructed: list[dict[str, object]] = []
    partitions = _partition_session_fixture_records(records, fixtures)
    for fixture_records, fixture in zip(partitions, fixtures, strict=True):
        attribution = correlate_response_lifecycles(
            fixture_records, interruption=fixture.interruption is not None
        )
        if attribution is None:
            raise ValueError("retained Pipecat artifact contains an incomplete response lifecycle")
        reconstructed.append(attribution)
    return reconstructed


def _validate_marker_lifecycle_brackets(
    marker_brackets: object, fixture_results: Sequence[Mapping[str, object]]
) -> None:
    if not isinstance(marker_brackets, list):
        raise ValueError("browser validation lacks retained marker brackets")
    by_identity: dict[tuple[int, str], Mapping[str, object]] = {}
    last_record_index = -1
    for raw_marker in marker_brackets:
        marker = _object(raw_marker, "persisted marker bracket")
        record_index = marker.get("record_index")
        if type(record_index) is not int or record_index <= last_record_index:
            raise ValueError("persisted marker record order is invalid")
        last_record_index = record_index
        key = (int(marker["fixture_index"]), str(marker["boundary"]))
        if key in by_identity:
            raise ValueError("persisted marker bracket is duplicated")
        by_identity[key] = marker

    for result in fixture_results:
        fixture_index = int(result["plan_index"])
        attribution = _object(result["response_attribution"], "marker response attribution")
        interruption = result.get("interruption")
        primary_name = "replacement_response" if interruption is not None else "primary_response"
        primary = _object(attribution[primary_name], "marker primary response")
        primary_bracket = _object(primary["rtvi_event_bracket"], "marker primary RTVI bracket")
        boundaries = (
            ("start", "interruption-send", "complete")
            if interruption is not None
            else ("start", "complete")
        )
        markers = [by_identity.get((fixture_index, boundary)) for boundary in boundaries]
        if any(marker is None for marker in markers):
            raise ValueError("fixture lifecycle lacks a persisted marker bracket")
        checked_markers = [_object(marker, "fixture marker bracket") for marker in markers]
        if [int(marker["record_index"]) for marker in checked_markers] != sorted(
            int(marker["record_index"]) for marker in checked_markers
        ):
            raise ValueError("fixture marker record brackets are out of order")

        def clocks(marker: Mapping[str, object]) -> tuple[float, float, float]:
            return (
                _finite_number(
                    marker.get("browser_performance_ms"),
                    "marker performance clock",
                    minimum=0,
                    maximum=1e100,
                ),
                _finite_number(
                    marker.get("browser_wall_time_s"),
                    "marker browser wall clock",
                    minimum=0,
                    maximum=1e100,
                ),
                _finite_number(
                    marker.get("playwright_batch_received_wall_time_s"),
                    "marker receipt wall clock",
                    minimum=0,
                    maximum=1e100,
                ),
            )

        start_performance, start_wall, start_receipt_wall = clocks(checked_markers[0])
        if not (
            float(result["send_attempt_wall_time_s"])
            < start_wall
            <= start_receipt_wall
            < float(result["send_wall_time_s"])
            and start_performance < float(primary_bracket["start_performance_ms"])
        ):
            raise ValueError("start marker does not strictly bracket the fixture send lifecycle")
        complete_performance, complete_wall, complete_receipt_wall = clocks(checked_markers[-1])
        if not (
            float(result["last_audio_wall_time_s"])
            < complete_wall
            <= complete_receipt_wall
            < float(result["completion_wall_time_s"])
            and float(primary_bracket["terminal_performance_ms"]) < complete_performance
        ):
            raise ValueError("complete marker does not strictly bracket the terminal lifecycle")
        if interruption is not None:
            interruption_result = _object(interruption, "marker interruption result")
            marker_performance, marker_wall, marker_receipt_wall = clocks(checked_markers[1])
            original = _object(attribution["original_response"], "marker original response")
            original_bracket = _object(
                original["rtvi_event_bracket"], "marker original RTVI bracket"
            )
            if not (
                float(interruption_result["send_attempt_wall_time_s"])
                < marker_wall
                <= marker_receipt_wall
                < float(interruption_result["send_wall_time_s"])
                and float(original_bracket["start_performance_ms"])
                < marker_performance
                < float(original_bracket["terminal_performance_ms"])
                < float(primary_bracket["start_performance_ms"])
            ):
                raise ValueError(
                    "interruption marker does not strictly bracket the replacement send lifecycle"
                )


def validate_fixture_summary(
    summary: Mapping[str, object],
    *,
    reservation: Step2PathReservation | None = None,
) -> None:
    required = {
        "schema",
        "status",
        "started_wall_time_s",
        "completed_wall_time_s",
        "plan",
        "resolved_paths",
        "sessions",
        "fixtures",
        "terminalization_errors",
    }
    if not required <= summary.keys():
        raise ValueError(f"fixture summary is missing fields: {sorted(required - summary.keys())}")
    status = summary["status"]
    if summary["schema"] != SUMMARY_SCHEMA or status not in _SUMMARY_STATUSES:
        raise ValueError("fixture summary identity or status is invalid")
    _finite_number(summary["started_wall_time_s"], "summary start", minimum=0, maximum=1e100)
    completed_time = summary["completed_wall_time_s"]
    if status == "running" and completed_time is not None:
        raise ValueError("running summary cannot have a terminal wall time")
    if status != "running":
        _finite_number(completed_time, "summary completion", minimum=0, maximum=1e100)
        if float(completed_time) < float(summary["started_wall_time_s"]):
            raise ValueError("summary completion precedes its start")
    _plan, plan_fixtures = _validate_embedded_plan(summary["plan"])
    paths = _object(summary["resolved_paths"], "summary resolved_paths")
    expected_path_keys = {
        "microphone",
        "browser_capture_on",
        "browser_capture_off",
        "summary",
        "pipecat_playout_trace",
        "pipecat_playout_capture_on",
        "pipecat_playout_capture_off",
        "output_dir",
    }
    if set(paths) != expected_path_keys or any(
        not isinstance(value, str) or not Path(value).is_absolute() for value in paths.values()
    ):
        raise ValueError("summary resolved path set is incomplete or unresolved")
    if reservation is not None and dict(paths) != dict(reservation):
        raise ValueError("summary resolved paths disagree with the retained reservation")
    if (
        len({paths[key] for key in expected_path_keys - {"output_dir", "pipecat_playout_trace"}})
        != 6
    ):
        raise ValueError("summary resolved file paths are not pairwise distinct")
    trace_template = Path(str(paths["pipecat_playout_trace"]))
    if PLAYOUT_TRACE_CONNECTION_TOKEN not in str(trace_template):
        raise ValueError("summary Pipecat trace path lacks the connection token")
    if paths["pipecat_playout_capture_on"] != str(
        connection_playout_trace_path(trace_template, 1)
    ) or paths["pipecat_playout_capture_off"] != str(
        connection_playout_trace_path(trace_template, 2)
    ):
        raise ValueError("summary Pipecat artifacts do not match connection ordinals")
    terminal_errors = summary["terminalization_errors"]
    if not isinstance(terminal_errors, list) or not all(
        isinstance(error, str) and error for error in terminal_errors
    ):
        raise ValueError("summary terminalization_errors must contain strings")
    sessions = summary["sessions"]
    fixtures = summary["fixtures"]
    if not isinstance(sessions, list) or len(sessions) != 2:
        raise ValueError("summary requires exactly two capture sessions")
    if not isinstance(fixtures, list) or len(fixtures) != 8:
        raise ValueError("summary requires exactly eight fixture results")
    for index, raw_session in enumerate(sessions):
        session = _object(raw_session, f"summary.sessions[{index}]")
        required_session = {
            "capture",
            "artifact",
            "status",
            "capture_shutdown",
            "error",
            "teardown_errors",
            "browser_artifact_validation",
            "pipecat_artifact",
            "pipecat_publication_wait_s",
            "observed_playout_trace",
        }
        if not required_session <= session.keys() or session["capture"] != (
            "on" if index == 0 else "off"
        ):
            raise ValueError("summary session does not match canonical capture order")
        _nonempty_string(session["artifact"], "session artifact")
        expected_artifact = paths[f"browser_capture_{session['capture']}"]
        if session["artifact"] != expected_artifact:
            raise ValueError("session artifact disagrees with the resolved path set")
        expected_pipecat = paths[f"pipecat_playout_capture_{session['capture']}"]
        if session["pipecat_artifact"] != expected_pipecat:
            raise ValueError("session Pipecat artifact disagrees with its connection mapping")
        if session["status"] not in _SESSION_STATUSES:
            raise ValueError("session status is invalid")
        publication_wait_s = session["pipecat_publication_wait_s"]
        if publication_wait_s is not None:
            _finite_number(
                publication_wait_s,
                "session Pipecat publication wait",
                minimum=0,
                maximum=1e100,
            )
        if session["error"] is not None and not isinstance(session["error"], str):
            raise ValueError("session error must be a string or null")
        if not isinstance(session["teardown_errors"], list) or not all(
            isinstance(error, str) and error for error in session["teardown_errors"]
        ):
            raise ValueError("session teardown_errors must contain strings")
    campaign_response_ids: set[str] = set()
    for index, raw_result in enumerate(fixtures):
        result = _object(raw_result, f"summary.fixtures[{index}]")
        planned = _object(plan_fixtures[index], f"summary.plan.fixtures[{index}]")
        if result.get("schema") != FIXTURE_RESULT_SCHEMA or result.get("plan_index") != index:
            raise ValueError("fixture result identity is invalid")
        for field in ("fixture_id", "script_id", "capture", "typed_text_sha256"):
            if result.get(field) != planned.get(field):
                raise ValueError("fixture result does not match its plan entry")
        result_status = result.get("completion_status")
        if result_status not in _FIXTURE_STATUSES:
            raise ValueError("fixture completion status is invalid")
        completed = result_status == "completed"
        rtvi_window_start = result.get("rtvi_window_start_index")
        if rtvi_window_start is not None and (
            type(rtvi_window_start) is not int or rtvi_window_start < 0
        ):
            raise ValueError("fixture RTVI window start index is invalid")
        observed_rtvi = result.get("observed_rtvi_messages")
        if not isinstance(observed_rtvi, list):
            raise ValueError("fixture observed RTVI diagnostics must be a list")
        for diagnostic_offset, raw_diagnostic in enumerate(observed_rtvi):
            diagnostic = _object(raw_diagnostic, "fixture observed RTVI diagnostic")
            if set(diagnostic) != {
                "event_index",
                "type",
                "received_performance_ms",
                "received_wall_time_s",
            }:
                raise ValueError("fixture observed RTVI diagnostic schema is invalid")
            event_index = diagnostic.get("event_index")
            if (
                type(event_index) is not int
                or rtvi_window_start is None
                or event_index != rtvi_window_start + diagnostic_offset
            ):
                raise ValueError("fixture observed RTVI diagnostic indices are not contiguous")
            if diagnostic.get("type") is not None and not isinstance(
                diagnostic.get("type"), str
            ):
                raise ValueError("fixture observed RTVI diagnostic type is invalid")
            for clock_field in ("received_performance_ms", "received_wall_time_s"):
                _finite_number(
                    diagnostic.get(clock_field),
                    f"fixture observed RTVI diagnostic {clock_field}",
                    minimum=0,
                    maximum=1e100,
                )
        observation_error = result.get("rtvi_observation_error")
        if observation_error is not None and not isinstance(observation_error, str):
            raise ValueError("fixture RTVI observation error must be a string or null")
        if (
            result_status in {"timeout", "error"}
            and rtvi_window_start is None
            and not observation_error
        ):
            raise ValueError("failed fixture lacks raw RTVI diagnostics or an observation error")
        if (result.get("interruption") is None) != (planned.get("interruption") is None):
            raise ValueError("fixture interruption result does not match its plan entry")
        _valid_marker_statuses(result, completed=completed)
        response_ids = result.get("response_ids")
        expected_responses = 2 if result.get("interruption") is not None else 1
        if not isinstance(response_ids, list) or not all(
            isinstance(item, str) and item for item in response_ids
        ):
            raise ValueError("fixture response_ids must contain nonempty strings")
        if completed and (
            len(response_ids) != expected_responses or len(set(response_ids)) != expected_responses
        ):
            raise ValueError("completed fixture has incorrect response-ID cardinality")
        if completed and campaign_response_ids.intersection(response_ids):
            raise ValueError("completed fixtures reuse a response ID across the campaign")
        if completed:
            campaign_response_ids.update(response_ids)
        if completed:
            _validate_response_attribution(result, response_ids)
        for field in (
            "send_attempt_wall_time_s",
            "send_wall_time_s",
            "first_audio_wall_time_s",
            "last_audio_wall_time_s",
            "completion_wall_time_s",
        ):
            value = result.get(field)
            if value is not None:
                _finite_number(value, f"fixture {field}", minimum=0, maximum=1e100)
        if completed and any(
            result.get(field) is None
            for field in (
                "send_attempt_wall_time_s",
                "send_wall_time_s",
                "first_audio_wall_time_s",
                "last_audio_wall_time_s",
                "completion_wall_time_s",
                "response_attribution",
            )
        ):
            raise ValueError("completed fixture lacks timing or attribution evidence")
        if completed:
            ordered_times = [
                float(result[field])
                for field in (
                    "send_attempt_wall_time_s",
                    "send_wall_time_s",
                    "first_audio_wall_time_s",
                    "last_audio_wall_time_s",
                    "completion_wall_time_s",
                )
            ]
            if ordered_times != sorted(ordered_times):
                raise ValueError("completed fixture wall times are out of order")
        if completed and result.get("error") is not None:
            raise ValueError("completed fixture cannot carry an error")
        transcripts = result.get("transcript_evidence")
        if not isinstance(transcripts, list) or not all(
            isinstance(text, str) and text for text in transcripts
        ):
            raise ValueError("fixture transcript evidence must contain strings")
        barrier = result.get("quiescence_barrier")
        if barrier is not None:
            _validate_quiescence_barrier(barrier)
        if completed and (
            barrier is None
            or _object(barrier, "completed quiescence barrier").get("passed") is not True
        ):
            raise ValueError("completed fixture lacks a successful RTVI quiescence barrier")
        if completed:
            attribution = _object(result.get("response_attribution"), "response attribution")
            attributed_response = _object(
                attribution.get(
                    "replacement_response"
                    if result.get("interruption") is not None
                    else "primary_response"
                ),
                "attributed transcript response",
            )
            if transcripts != attributed_response.get("transcript_evidence"):
                raise ValueError("redundant fixture transcript fields disagree")
            _validate_transcript_gate_evidence(
                result.get("transcript_gate_evidence"),
                browser_transcripts=transcripts,
                response_owned_traced_text=_nonempty_string(
                    attributed_response.get("output_text"),
                    "attributed response-owned traced text",
                ),
            )
            attributed_bracket = _object(
                attributed_response.get("rtvi_event_bracket"),
                "attributed response RTVI bracket",
            )
            checked_barrier = _object(barrier, "completed quiescence barrier")
            if (
                checked_barrier.get("window_start_performance_ms")
                != attributed_bracket.get("terminal_performance_ms")
                or checked_barrier.get("window_start_wall_time_s")
                != attributed_bracket.get("terminal_wall_time_s")
            ):
                raise ValueError("RTVI quiescence barrier does not begin at the terminal event")
        interruption_result = result.get("interruption")
        if interruption_result is not None:
            interruption_result = _object(interruption_result, "fixture interruption")
            second = interruption_result.get("second_answer_transcript_evidence")
            actual_offset = interruption_result.get("actual_offset_s")
            if completed:
                if (
                    interruption_result.get("after_first_audio_s")
                    != planned["interruption"]["after_first_audio_s"]
                    or interruption_result.get("typed_text_sha256")
                    != planned["interruption"]["typed_text_sha256"]
                ):
                    raise ValueError("completed I1 does not match its interruption policy")
                if (
                    interruption_result.get("send_attempt_wall_time_s") is None
                    or interruption_result.get("send_wall_time_s") is None
                ):
                    raise ValueError("completed I1 lacks interruption send evidence")
                interruption_times = [
                    float(result["first_audio_wall_time_s"]),
                    float(interruption_result["send_attempt_wall_time_s"]),
                    float(interruption_result["send_wall_time_s"]),
                    float(result["last_audio_wall_time_s"]),
                ]
                if interruption_times != sorted(interruption_times):
                    raise ValueError("completed I1 interruption wall times are out of order")
                offset = _finite_number(actual_offset, "I1 actual offset", minimum=0, maximum=300)
                if abs(offset - I1_OFFSET_S) > I1_OFFSET_TOLERANCE_S:
                    raise ValueError("completed I1 missed its offset policy")
                if not isinstance(second, list) or not all(
                    isinstance(text, str) and text for text in second
                ):
                    raise ValueError("completed I1 second-answer transcripts are invalid")
                if second != transcripts:
                    raise ValueError("redundant I1 transcript fields disagree")
                replacement_response = _object(
                    _object(result.get("response_attribution"), "I1 response attribution").get(
                        "replacement_response"
                    ),
                    "I1 replacement response",
                )
                _validate_second_answer_gate_evidence(
                    interruption_result.get("second_answer_gate_evidence"),
                    browser_transcripts=second,
                    replacement_response_owned_traced_text=_nonempty_string(
                        replacement_response.get("output_text"),
                        "I1 replacement response-owned traced text",
                    ),
                )
                replacement_text = _nonempty_string(
                    interruption_result.get("typed_text"), "I1 replacement text"
                )
                if (
                    interruption_result.get("typed_text_sha256") != sha256_text(replacement_text)
                    or replacement_text != planned["interruption"]["typed_text"]
                ):
                    raise ValueError("completed I1 replacement bytes do not match their hash")
    completed_results = [
        _object(result, "fixture result")
        for result in fixtures
        if _object(result, "fixture result").get("completion_status") == "completed"
    ]
    for previous, current in zip(completed_results, completed_results[1:]):
        if float(previous["completion_wall_time_s"]) > float(current["send_attempt_wall_time_s"]):
            raise ValueError("fixture results overlap despite canonical serial execution")
    if completed_results and (
        float(summary["started_wall_time_s"])
        > float(completed_results[0]["send_attempt_wall_time_s"])
        or (
            completed_time is not None
            and float(completed_results[-1]["completion_wall_time_s"]) > float(completed_time)
        )
    ):
        raise ValueError("fixture results fall outside the summary lifecycle")
    if status == "completed":
        if terminal_errors:
            raise ValueError("completed summary cannot carry terminalization errors")
        canonical_specs = parse_fixture_plan(canonical_plan_json()).fixtures
        for session in sessions:
            session = _object(session, "completed session")
            shutdown = session.get("capture_shutdown")
            validation = session.get("browser_artifact_validation")
            playout_validation = session.get("observed_playout_trace")
            if (
                session.get("status") != "completed"
                or session.get("error") is not None
                or session.get("teardown_errors")
            ):
                raise ValueError("completed summary requires every session to complete cleanly")
            if session.get("pipecat_publication_wait_s") is None:
                raise ValueError("completed summary requires a recorded Pipecat publication wait")
            if (
                not isinstance(shutdown, Mapping)
                or shutdown.get("shutdown_error") is not None
                or shutdown.get("sink_close_error") is not None
            ):
                raise ValueError("completed summary requires successful capture shutdown")
            if not isinstance(validation, Mapping) or validation.get("valid") is not True:
                raise ValueError("completed summary requires checked browser artifacts")
            if not isinstance(playout_validation, Mapping):
                raise ValueError("completed summary requires checked Pipecat artifacts")
            actual_browser_validation = load_browser_artifact(
                Path(str(session["artifact"])),
                capture_enabled=session["capture"] == "on",
                reservation=reservation,
            )
            if validation != actual_browser_validation:
                raise ValueError("summary does not contain the full checked browser result")
            actual_playout_validation = validate_published_playout_artifact(
                Path(str(session["pipecat_artifact"])), reservation=reservation
            )
            if playout_validation != actual_playout_validation:
                raise ValueError("summary does not contain the full checked Pipecat result")
            session_fixture_indices = list(
                range(7) if session["capture"] == "on" else range(7, 8)
            )
            retained_records = read_published_playout_trace(
                Path(str(session["pipecat_artifact"])), reservation=reservation
            )
            reconstructed = reconstruct_session_response_attributions(
                retained_records,
                [canonical_specs[index] for index in session_fixture_indices],
            )
            for index, actual_attribution in zip(
                session_fixture_indices, reconstructed, strict=True
            ):
                claimed = _object(
                    _object(fixtures[index], "fixture").get("response_attribution"),
                    "claimed response attribution",
                )
                interrupted = canonical_specs[index].interruption is not None
                if _artifact_attribution_projection(
                    claimed, interruption=interrupted
                ) != _artifact_attribution_projection(
                    actual_attribution, interruption=interrupted
                ):
                    raise ValueError(
                        "claimed response attribution disagrees with the retained Pipecat artifact"
                    )
            expected_markers = 15 if session["capture"] == "on" else 2
            if (
                shutdown.get("fixture_marker_records") != expected_markers
                or validation.get("fixture_marker_records") != expected_markers
            ):
                raise ValueError("completed summary has the wrong required marker count")
            session_fixture_results = [
                _object(fixtures[index], "session fixture") for index in session_fixture_indices
            ]
            _validate_marker_lifecycle_brackets(
                validation.get("marker_brackets"), session_fixture_results
            )
            claimed_sequence = [
                {
                    "fixture_index": index,
                    "fixture_id": fixtures[index]["fixture_id"],
                    "script_id": fixtures[index]["script_id"],
                    "capture_enabled": session["capture"] == "on",
                    "boundary": boundary,
                }
                for index in session_fixture_indices
                for boundary in (
                    ("start", "interruption-send", "complete")
                    if fixtures[index].get("interruption") is not None
                    else ("start", "complete")
                )
                if _object(
                    _object(fixtures[index], "fixture")["marker_emission"],
                    "marker emissions",
                )[boundary]["status"]
                == "succeeded"
            ]
            if validation.get("marker_sequence") != claimed_sequence:
                raise ValueError("summary marker claims disagree with the persisted artifact")
        validate_connection_playout_trace_set(trace_template, reservation=reservation)
        if any(
            _object(result, "completed fixture").get("completion_status") != "completed"
            for result in fixtures
        ):
            raise ValueError("completed summary requires every fixture to complete")
    elif status == "failed":
        failure_observed = (
            bool(terminal_errors)
            or any(
                _object(session, "failed summary session").get("status") == "failed"
                or _object(session, "failed summary session").get("error") is not None
                or bool(_object(session, "failed summary session").get("teardown_errors"))
                for session in sessions
            )
            or any(
                _object(result, "failed summary fixture").get("completion_status")
                in {"timeout", "error"}
                or _object(result, "failed summary fixture").get("error") is not None
                for result in fixtures
            )
        )
        if not failure_observed:
            raise ValueError("failed summary lacks any recorded failure evidence")


def _is_second_answer(text: object) -> bool:
    if not isinstance(text, str):
        return False
    return re.search(r"\bfour\b", text, re.IGNORECASE) is not None


def validate_rtvi_message_clocks(messages: Sequence[Mapping[str, object]]) -> None:
    last_performance = -1.0
    last_wall = -1.0
    for item in messages:
        performance = _finite_number(
            item.get("receivedPerformanceMs"),
            "RTVI message performance clock",
            minimum=0,
            maximum=1e100,
        )
        wall = _finite_number(
            item.get("receivedWallTimeS"),
            "RTVI message wall clock",
            minimum=0,
            maximum=1e100,
        )
        if performance < last_performance or wall < last_wall:
            raise ValueError("RTVI message clocks regress")
        last_performance = performance
        last_wall = wall


def rtvi_message_diagnostics(
    messages: Sequence[Mapping[str, object]], *, start_event_index: int
) -> list[dict[str, object]]:
    """Project one fixture's raw ordered RTVI window for failure diagnosis."""

    if type(start_event_index) is not int or start_event_index < 0:
        raise ValueError("RTVI diagnostic start event index is invalid")
    diagnostics: list[dict[str, object]] = []
    for offset, item in enumerate(messages):
        message = item.get("message")
        diagnostics.append(
            {
                "event_index": start_event_index + offset,
                "type": message.get("type") if isinstance(message, Mapping) else None,
                "received_performance_ms": _finite_number(
                    item.get("receivedPerformanceMs"),
                    "RTVI diagnostic performance clock",
                    minimum=0,
                    maximum=1e100,
                ),
                "received_wall_time_s": _finite_number(
                    item.get("receivedWallTimeS"),
                    "RTVI diagnostic wall clock",
                    minimum=0,
                    maximum=1e100,
                ),
            }
        )
    return diagnostics


def transcription_texts(
    messages: Sequence[Mapping[str, object]],
    *,
    after_performance_ms: float,
    before_performance_ms: float | None = None,
    after_event_index: int | None = None,
    before_event_index: int | None = None,
    expected_output_text: str | None = None,
) -> list[str]:
    validate_rtvi_message_clocks(messages)
    texts: list[str] = []
    for index, item in enumerate(messages):
        performance = float(item["receivedPerformanceMs"])
        if after_event_index is not None and index <= after_event_index:
            continue
        if before_event_index is not None and index >= before_event_index:
            continue
        if performance <= after_performance_ms:
            continue
        if before_performance_ms is not None and performance >= before_performance_ms:
            continue
        message = item.get("message")
        if not isinstance(message, Mapping) or message.get("type") != "bot-transcription":
            continue
        data = message.get("data")
        text = data.get("text") if isinstance(data, Mapping) else None
        if isinstance(text, str) and text:
            texts.append(text)
    if expected_output_text is not None and "".join(texts) != expected_output_text:
        return []
    return texts


def rtvi_transcript_evidence(
    messages: Sequence[Mapping[str, object]],
    *,
    start_event_index: int,
    terminal_event_index: int,
    start_event: Mapping[str, object],
    terminal_event: Mapping[str, object],
    response_id: str,
    expected_output_text: str,
    fixture_send_performance_ms: float,
    fixture_send_wall_time_s: float,
) -> tuple[list[str], dict[str, object]]:
    """Bind identity-less RTVI transcript records through strict ordered edges."""

    def revalidated_event(
        index: int, selected: Mapping[str, object], label: str
    ) -> Mapping[str, object]:
        if type(index) is not int or not 0 <= index < len(messages):
            raise ValueError(f"RTVI speaking {label} event index is out of bounds")
        current = messages[index]
        selected_message = selected.get("message")
        current_message = current.get("message")
        selected_type = (
            selected_message.get("type") if isinstance(selected_message, Mapping) else None
        )
        current_type = (
            current_message.get("type") if isinstance(current_message, Mapping) else None
        )
        selected_clocks = (
            _finite_number(
                selected.get("receivedPerformanceMs"),
                f"selected RTVI speaking {label} performance clock",
                minimum=0,
                maximum=1e100,
            ),
            _finite_number(
                selected.get("receivedWallTimeS"),
                f"selected RTVI speaking {label} wall clock",
                minimum=0,
                maximum=1e100,
            ),
        )
        current_clocks = (
            _finite_number(
                current.get("receivedPerformanceMs"),
                f"RTVI speaking {label} performance clock",
                minimum=0,
                maximum=1e100,
            ),
            _finite_number(
                current.get("receivedWallTimeS"),
                f"RTVI speaking {label} wall clock",
                minimum=0,
                maximum=1e100,
            ),
        )
        if current_type != selected_type or current_clocks != selected_clocks:
            raise ValueError(f"RTVI speaking {label} event revalidation mismatch")
        return current

    start_index = start_event_index
    terminal_index = terminal_event_index
    start_event = revalidated_event(start_index, start_event, "start")
    terminal_event = revalidated_event(terminal_index, terminal_event, "terminal")
    if start_index >= terminal_index:
        raise ValueError("RTVI speaking terminal does not follow its start event")
    start_performance = _finite_number(
        start_event.get("receivedPerformanceMs"),
        "RTVI speaking start performance clock",
        minimum=0,
        maximum=1e100,
    )
    terminal_performance = _finite_number(
        terminal_event.get("receivedPerformanceMs"),
        "RTVI speaking terminal performance clock",
        minimum=0,
        maximum=1e100,
    )
    send_performance = _finite_number(
        fixture_send_performance_ms,
        "fixture send performance clock",
        minimum=0,
        maximum=1e100,
    )
    send_wall = _finite_number(
        fixture_send_wall_time_s,
        "fixture send wall clock",
        minimum=0,
        maximum=1e100,
    )
    bracket_messages = messages[start_index : terminal_index + 1]
    before_send = sum(
        float(item.get("receivedPerformanceMs", -1)) <= send_performance
        or float(item.get("receivedWallTimeS", -1)) <= send_wall
        for item in bracket_messages
    )
    if before_send:
        raise ValueError("RTVI speaking bracket contains messages received before its send edge")
    texts = transcription_texts(
        messages,
        after_performance_ms=start_performance,
        before_performance_ms=terminal_performance,
        after_event_index=start_index,
        before_event_index=terminal_index,
    )
    transcript_indices = [
        index
        for index in range(start_index + 1, terminal_index)
        if messages[index].get("message", {}).get("type") == "bot-transcription"
    ]
    return texts, {
        "response_id": response_id,
        "transcript_binding": RTVI_TRANSCRIPT_IDENTITY,
        "protocol_response_identity": None,
        "fixture_send_performance_ms": send_performance,
        "fixture_send_wall_time_s": send_wall,
        "messages_received_before_send_in_bracket": before_send,
        "start_event_index": start_index,
        "terminal_event_index": terminal_index,
        "start_event_type": start_event.get("message", {}).get("type"),
        "terminal_event_type": terminal_event.get("message", {}).get("type"),
        "start_performance_ms": start_performance,
        "terminal_performance_ms": terminal_performance,
        "start_wall_time_s": start_event.get("receivedWallTimeS"),
        "terminal_wall_time_s": terminal_event.get("receivedWallTimeS"),
        "transcript_event_indices": transcript_indices,
        "transcript_performance_ms": [
            messages[index]["receivedPerformanceMs"] for index in transcript_indices
        ],
        "output_text_sha256": sha256_text(expected_output_text),
    }
