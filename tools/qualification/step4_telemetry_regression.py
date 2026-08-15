#!/usr/bin/env python3
"""Join Step 0 analyzer populations to Step 4 telemetry and regress frame cost."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from stratified_latency_analyzer import (
    STAGE_FIELDS,
    _membership_identifier,
    _normalize_model_rows,
    _stage_timings,
    classify_populations,
    distribution_stats,
    read_jsonl,
    source_identity,
)

JsonObject = dict[str, Any]
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260813
MAX_CAUSAL_JOIN_AGE_S = 1.1
TELEMETRY_SCHEMAS = {
    "nemotron_voicechat.step4_telemetry.v1",
    "nemotron_voicechat.step4_telemetry.v2",
}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _membership(rows: list[JsonObject]) -> JsonObject:
    members = [
        {
            "event_ordinal": row["event_ordinal"],
            "frame": row["frame"],
            "response_id": row.get("response_id"),
            "monotonic_timestamp_s": row["monotonic_timestamp_s"],
        }
        for row in rows
    ]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return {"n": len(members), "sha256": hashlib.sha256(encoded).hexdigest()}


def _require_number(value: object, label: str) -> float:
    number = _finite(value)
    if number is None:
        raise ValueError(f"telemetry {label} must be a finite number")
    return number


def validate_telemetry(samples: list[JsonObject]) -> None:
    if len(samples) < 2:
        raise ValueError("at least two telemetry samples are required")
    previous_start: float | None = None
    previous_end: float | None = None
    for index, sample in enumerate(samples):
        if sample.get("schema") not in TELEMETRY_SCHEMAS:
            raise ValueError(f"telemetry sample {index} has an unsupported schema")
        for field in ("gpu", "cpu_scaling_cur_freq_khz", "pressure"):
            if not isinstance(sample.get(field), dict):
                raise ValueError(f"telemetry sample {index} lacks {field}")
        _require_number(sample.get("wall_time_s"), f"sample {index} wall_time_s")
        for field in ("gpu", "cpu_scaling_cur_freq_khz"):
            for name, value in sample[field].items():
                if value is not None and _finite(value) is None:
                    raise ValueError(f"telemetry sample {index} has invalid {field}.{name}")
        if set(sample["pressure"]) != {"cpu", "memory", "io"}:
            raise ValueError(f"telemetry sample {index} has invalid pressure kinds")
        for kind, pressure in sample["pressure"].items():
            if not isinstance(pressure, dict) or set(pressure) != {
                "avg10",
                "avg60",
                "avg300",
                "total",
            }:
                raise ValueError(f"telemetry sample {index} has invalid {kind} pressure")
            for field, value in pressure.items():
                _require_number(value, f"sample {index} pressure.{kind}.{field}")
        start = _require_number(sample.get("monotonic_s"), f"sample {index} monotonic_s")
        end = start
        if sample["schema"].endswith(".v2"):
            acquisition = sample.get("acquisition")
            sources = sample.get("source_acquisition")
            if not isinstance(acquisition, dict) or not isinstance(sources, dict):
                raise ValueError(f"telemetry sample {index} lacks acquisition boundaries")
            if set(sources) != {"gpu", "cpu_scaling_cur_freq", "pressure"}:
                raise ValueError(f"telemetry sample {index} has invalid source boundaries")
            start = _require_number(
                acquisition.get("monotonic_start_s"), f"sample {index} acquisition start"
            )
            end = _require_number(
                acquisition.get("monotonic_end_s"), f"sample {index} acquisition end"
            )
            wall_start = _require_number(
                acquisition.get("wall_start_s"), f"sample {index} acquisition wall start"
            )
            wall_end = _require_number(
                acquisition.get("wall_end_s"), f"sample {index} acquisition wall end"
            )
            duration_ms = _require_number(
                acquisition.get("duration_ms"), f"sample {index} acquisition duration"
            )
            if end < start:
                raise ValueError(f"telemetry sample {index} acquisition ends before it starts")
            if wall_end < wall_start or duration_ms < 0:
                raise ValueError(f"telemetry sample {index} has invalid acquisition duration")
            for source_name, boundary in sources.items():
                if not isinstance(boundary, dict):
                    raise ValueError(f"telemetry sample {index} invalid {source_name} boundary")
                source_start = _require_number(
                    boundary.get("monotonic_start_s"),
                    f"sample {index} {source_name} start",
                )
                source_end = _require_number(
                    boundary.get("monotonic_end_s"), f"sample {index} {source_name} end"
                )
                source_wall_start = _require_number(
                    boundary.get("wall_start_s"), f"sample {index} {source_name} wall start"
                )
                source_wall_end = _require_number(
                    boundary.get("wall_end_s"), f"sample {index} {source_name} wall end"
                )
                source_duration = _require_number(
                    boundary.get("duration_ms"), f"sample {index} {source_name} duration"
                )
                if not start <= source_start <= source_end <= end:
                    raise ValueError(
                        f"telemetry sample {index} {source_name} boundary is out of acquisition"
                    )
                if source_wall_end < source_wall_start or source_duration < 0:
                    raise ValueError(f"telemetry sample {index} {source_name} has invalid duration")
        if previous_start is not None and start <= previous_start:
            raise ValueError("telemetry sample times must be strictly increasing")
        if previous_end is not None and end <= previous_end:
            raise ValueError("telemetry acquisition end times must be strictly increasing")
        previous_start, previous_end = start, end


def _source_end(sample: JsonObject, source: str) -> float:
    if sample["schema"].endswith(".v2"):
        return float(sample["source_acquisition"][source]["monotonic_end_s"])
    return float(sample["monotonic_s"])


def flatten_telemetry(samples: list[JsonObject]) -> tuple[list[JsonObject], list[str]]:
    validate_telemetry(samples)
    flattened: list[JsonObject] = []
    for index, sample in enumerate(samples):
        values: dict[str, float | None] = {}
        available: dict[str, float] = {}
        intervals: dict[str, list[float] | None] = {}
        for key, value in sample["gpu"].items():
            channel = f"gpu.{key}"
            values[channel] = _finite(value)
            available[channel] = _source_end(sample, "gpu")
            intervals[channel] = None
        for key, value in sample["cpu_scaling_cur_freq_khz"].items():
            channel = f"cpu{key}.scaling_cur_freq_khz"
            values[channel] = _finite(value)
            available[channel] = _source_end(sample, "cpu_scaling_cur_freq")
            intervals[channel] = None
        for kind, pressure in sample["pressure"].items():
            for window in ("avg10", "avg60", "avg300"):
                channel = f"pressure.{kind}.{window}"
                values[channel] = _finite(pressure[window])
                available[channel] = _source_end(sample, "pressure")
                intervals[channel] = None
            interval_fraction = None
            interval_bounds = None
            if index:
                previous = samples[index - 1]
                interval_start = _source_end(previous, "pressure")
                interval_end = _source_end(sample, "pressure")
                elapsed = interval_end - interval_start
                stalled_us = float(pressure["total"]) - float(previous["pressure"][kind]["total"])
                if elapsed > 0 and stalled_us >= 0:
                    interval_fraction = stalled_us / (elapsed * 1_000_000.0)
                    interval_bounds = [interval_start, interval_end]
            channel = f"pressure.{kind}.interval_stall_fraction"
            values[channel] = interval_fraction
            available[channel] = _source_end(sample, "pressure")
            intervals[channel] = interval_bounds
        acquisition = sample.get("acquisition", {})
        monotonic_start = float(acquisition.get("monotonic_start_s", sample["monotonic_s"]))
        monotonic_end = float(acquisition.get("monotonic_end_s", sample["monotonic_s"]))
        flattened.append(
            {
                "sample_index": index,
                "monotonic_s": monotonic_start,
                "acquisition_end_monotonic_s": monotonic_end,
                "wall_time_s": float(sample["wall_time_s"]),
                "values": values,
                "available_monotonic_s": available,
                "value_intervals_monotonic_s": intervals,
            }
        )
    channels = sorted({key for row in flattened for key in row["values"]})
    return flattened, channels


def join_population(
    members: list[JsonObject], events: list[JsonObject], telemetry: list[JsonObject]
) -> list[JsonObject]:
    channels = sorted({channel for row in telemetry for channel in row["values"]})
    channel_times = {
        channel: [row["available_monotonic_s"][channel] for row in telemetry]
        for channel in channels
    }
    joined: list[JsonObject] = []
    for member in members:
        ordinal = int(member["event_ordinal"])
        if ordinal < 1 or ordinal > len(events):
            raise ValueError(f"analyzer member ordinal is out of range: {ordinal}")
        event = events[ordinal - 1]
        timestamp = float(member["monotonic_timestamp_s"])
        if event.get("event") != "model_step" or event.get("frame") != member.get("frame"):
            raise ValueError(f"analyzer membership does not match source event {ordinal}")
        timestamp_field = member.get("timestamp_field")
        if timestamp_field != "monotonic_s" or _finite(event.get(timestamp_field)) != timestamp:
            raise ValueError(f"analyzer member timestamp does not match source event {ordinal}")
        server_step_ms = _finite(event.get("server_step_ms"))
        if server_step_ms is None:
            raise ValueError(f"event {ordinal} lacks server_step_ms")
        values: dict[str, float | None] = {}
        sample_indices: dict[str, int] = {}
        join_ages_ms: dict[str, float] = {}
        value_intervals: dict[str, list[float] | None] = {}
        for channel in channels:
            position = bisect.bisect_right(channel_times[channel], timestamp) - 1
            if position < 0:
                raise ValueError(f"event {ordinal} has no causally prior telemetry for {channel}")
            age = timestamp - channel_times[channel][position]
            if age > MAX_CAUSAL_JOIN_AGE_S:
                raise ValueError(
                    f"event {ordinal} causally prior telemetry for {channel} is {age:.6f}s old"
                )
            sample = telemetry[position]
            values[channel] = sample["values"][channel]
            sample_indices[channel] = position
            join_ages_ms[channel] = age * 1000.0
            value_intervals[channel] = sample["value_intervals_monotonic_s"][channel]
        joined.append(
            {
                **member,
                "server_step_ms": server_step_ms,
                "telemetry_sample_indices": sample_indices,
                "join_age_ms": max(join_ages_ms.values()),
                "channel_join_age_ms": join_ages_ms,
                "telemetry_value_intervals_monotonic_s": value_intervals,
                "telemetry": values,
                "event": event,
            }
        )
    return joined


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    xx = sum((value - mean_x) ** 2 for value in x)
    yy = sum((value - mean_y) ** 2 for value in y)
    if xx <= 0 or yy <= 0:
        return None
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / math.sqrt(xx * yy)


def slope(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 0:
        return None
    mean_y = sum(y) / len(y)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / denominator


def linear_residuals(time_values: list[float], values: list[float]) -> list[float] | None:
    fitted_slope = slope(time_values, values)
    if fitted_slope is None:
        return None
    mean_time = sum(time_values) / len(time_values)
    mean_value = sum(values) / len(values)
    return [
        value - (mean_value + fitted_slope * (timestamp - mean_time))
        for timestamp, value in zip(time_values, values, strict=True)
    ]


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _correlation_view(pairs: list[tuple[float, float, float, str]], view: str) -> float | None:
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    if view == "raw":
        return pearson(x, y)
    if view == "linear-time-adjusted":
        timestamps = [pair[2] for pair in pairs]
        x_residuals = linear_residuals(timestamps, x)
        y_residuals = linear_residuals(timestamps, y)
        if x_residuals is None or y_residuals is None:
            return None
        return pearson(x_residuals, y_residuals)
    if view == "within-response":
        by_response: dict[str, list[int]] = defaultdict(list)
        for index, pair in enumerate(pairs):
            by_response[pair[3]].append(index)
        demeaned_x: list[float] = []
        demeaned_y: list[float] = []
        for indices in by_response.values():
            mean_x = sum(x[index] for index in indices) / len(indices)
            mean_y = sum(y[index] for index in indices) / len(indices)
            demeaned_x.extend(x[index] - mean_x for index in indices)
            demeaned_y.extend(y[index] - mean_y for index in indices)
        return pearson(demeaned_x, demeaned_y)
    raise ValueError(f"unknown correlation view: {view}")


def clustered_correlation_ci(
    rows: list[JsonObject], channel: str, *, view: str = "raw"
) -> JsonObject:
    clusters: dict[str, list[tuple[float, float, float, str]]] = defaultdict(list)
    for row in rows:
        value = row["telemetry"].get(channel)
        response_id = row.get("response_id")
        if value is not None and isinstance(response_id, str):
            clusters[response_id].append(
                (
                    float(value),
                    float(row["server_step_ms"]),
                    float(row["monotonic_timestamp_s"]),
                    response_id,
                )
            )
    names = sorted(clusters)
    if len(names) < 2:
        return {"iterations": 0, "seed": BOOTSTRAP_SEED, "ci_95": None}
    generator = random.Random(BOOTSTRAP_SEED)
    correlations: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        pairs = [pair for _name in names for pair in clusters[generator.choice(names)]]
        correlation = _correlation_view(pairs, view)
        if correlation is not None:
            correlations.append(correlation)
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "valid_iterations": len(correlations),
        "seed": BOOTSTRAP_SEED,
        "ci_95": [
            _nearest_rank(correlations, 2.5),
            _nearest_rank(correlations, 97.5),
        ],
    }


def scatter_summary(rows: list[JsonObject], channel: str) -> list[JsonObject]:
    pairs = [
        (float(row["telemetry"][channel]), float(row["server_step_ms"]), row)
        for row in rows
        if row["telemetry"].get(channel) is not None
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda item: item[0])
    result: list[JsonObject] = []
    for band in range(4):
        start = math.floor(band * len(pairs) / 4)
        end = math.floor((band + 1) * len(pairs) / 4)
        selected = pairs[start:end]
        if not selected:
            continue
        telemetry_values = [item[0] for item in selected]
        timing_values = [item[1] for item in selected]
        result.append(
            {
                "quartile": band + 1,
                "n_rows": len(selected),
                "n_responses": len(
                    {item[2].get("response_id") for item in selected if item[2].get("response_id")}
                ),
                "telemetry_min": min(telemetry_values),
                "telemetry_max": max(telemetry_values),
                "telemetry_mean": sum(telemetry_values) / len(telemetry_values),
                "server_step_ms_mean": sum(timing_values) / len(timing_values),
                "server_step_ms_p95": _nearest_rank(timing_values, 95),
            }
        )
    return result


def regress(rows: list[JsonObject], channel: str, *, clustered: bool) -> JsonObject:
    usable = [row for row in rows if row["telemetry"].get(channel) is not None]
    x = [float(row["telemetry"][channel]) for row in usable]
    y = [float(row["server_step_ms"]) for row in usable]
    timestamps = [float(row["monotonic_timestamp_s"]) for row in usable]
    by_sample: dict[int, list[JsonObject]] = defaultdict(list)
    for row in usable:
        by_sample[int(row["telemetry_sample_indices"][channel])].append(row)
    sample_x = [float(group[0]["telemetry"][channel]) for group in by_sample.values()]
    sample_y = [
        sum(float(row["server_step_ms"]) for row in group) / len(group)
        for group in by_sample.values()
    ]
    sample_timestamps = [
        sum(float(row["monotonic_timestamp_s"]) for row in group) / len(group)
        for group in by_sample.values()
    ]
    x_residuals = linear_residuals(timestamps, x)
    y_residuals = linear_residuals(timestamps, y)
    sample_x_residuals = linear_residuals(sample_timestamps, sample_x)
    sample_y_residuals = linear_residuals(sample_timestamps, sample_y)
    unique = len(set(x))
    response_pairs = [
        (
            float(row["telemetry"][channel]),
            float(row["server_step_ms"]),
            timestamps[index],
            response,
        )
        for index, row in enumerate(usable)
        if isinstance((response := row.get("response_id")), str)
    ]
    return {
        "status": "estimated" if len(x) >= 2 and unique >= 2 else "undefined-no-variation",
        "n_rows": len(x),
        "n_telemetry_samples": len(by_sample),
        "n_unique_values": unique,
        "pearson_r_frame": pearson(x, y),
        "pearson_r_channel_vs_time": pearson(x, timestamps),
        "pearson_r_outcome_vs_time": pearson(y, timestamps),
        "ols_slope_ms_per_unit_frame": slope(x, y),
        "pearson_r_sample_mean": pearson(sample_x, sample_y),
        "pearson_r_frame_after_linear_time_detrend": (
            pearson(x_residuals, y_residuals)
            if x_residuals is not None and y_residuals is not None
            else None
        ),
        "pearson_r_sample_mean_after_linear_time_detrend": (
            pearson(sample_x_residuals, sample_y_residuals)
            if sample_x_residuals is not None and sample_y_residuals is not None
            else None
        ),
        "pearson_r_within_response_demeaned": _correlation_view(response_pairs, "within-response")
        if response_pairs
        else None,
        "response_clustered_bootstrap": (
            {
                "raw": clustered_correlation_ci(usable, channel),
                "linear_time_adjusted": clustered_correlation_ci(
                    usable, channel, view="linear-time-adjusted"
                ),
                "within_response": clustered_correlation_ci(
                    usable, channel, view="within-response"
                ),
            }
            if clustered
            else None
        ),
        "scatter_quartiles": scatter_summary(usable, channel),
    }


def _exact_membership(rows: list[JsonObject]) -> JsonObject:
    members = [_membership_identifier(row) for row in rows]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return {"n": len(members), "sha256": hashlib.sha256(encoded).hexdigest(), "members": members}


def _validate_analyzer_membership(
    source: JsonObject, populations: dict[str, list[JsonObject]]
) -> None:
    for name, analyzer_population in source["populations"].items():
        if name not in populations:
            raise ValueError(f"qualified analyzer has unknown population: {name}")
        expected = _exact_membership(populations[name])
        actual = analyzer_population.get("membership")
        if actual != expected:
            raise ValueError(f"qualified analyzer exact membership mismatch: {name}")


def _response_stage_means(rows: list[JsonObject]) -> list[JsonObject]:
    by_response: dict[str, list[JsonObject]] = defaultdict(list)
    for row in rows:
        response_id = row.get("response_id")
        if isinstance(response_id, str):
            by_response[response_id].append(row)
    result = []
    for response_id, response_rows in by_response.items():
        stages: dict[str, list[float]] = defaultdict(list)
        for row in response_rows:
            timings = _stage_timings(row["event"])
            for name in STAGE_FIELDS.values():
                value = timings.get(name)
                measured = _finite(value)
                if measured is not None:
                    stages[name].append(measured)
        result.append(
            {
                "response_id": response_id,
                "n_rows": len(response_rows),
                "server_step_ms_mean": sum(row["server_step_ms"] for row in response_rows)
                / len(response_rows),
                "stage_mean_ms": {
                    name: sum(values) / len(values) for name, values in sorted(stages.items())
                },
            }
        )
    return result


def analyze(analyzer_path: Path, server_trace_path: Path, telemetry_path: Path) -> JsonObject:
    analyzer = json.loads(analyzer_path.read_text(encoding="utf-8"))
    if analyzer.get("schema") != "generation-frame-latency-step0-v2":
        raise ValueError("input is not a qualified Step 0 analyzer result")
    if len(analyzer.get("sources", [])) != 1:
        raise ValueError("telemetry join requires exactly one analyzer source")
    source = analyzer["sources"][0]
    identity = source_identity(server_trace_path)
    if identity != source["source"]:
        raise ValueError("server trace identity does not match analyzer source")
    events = read_jsonl(server_trace_path)
    classified = classify_populations(_normalize_model_rows(events))
    _validate_analyzer_membership(source, classified)
    raw_telemetry = read_jsonl(telemetry_path)
    telemetry, channels = flatten_telemetry(raw_telemetry)
    intervals = [
        telemetry[index]["monotonic_s"] - telemetry[index - 1]["monotonic_s"]
        for index in range(1, len(telemetry))
    ]
    offsets = [row["wall_time_s"] - row["monotonic_s"] for row in telemetry]
    populations: dict[str, JsonObject] = {}
    for population in ("delivered", "delivered-nonBOS", "idle"):
        analyzer_population = source["populations"][population]
        rows = join_population(analyzer_population["membership"]["members"], events, telemetry)
        regressions = {
            channel: regress(rows, channel, clustered=population != "idle") for channel in channels
        }
        populations[population] = {
            "analyzer_membership": {
                "n": analyzer_population["membership"]["n"],
                "sha256": analyzer_population["membership"]["sha256"],
            },
            "analyzer_server_step_ms": analyzer_population["clocks"]["server_step_ms"],
            "join_distance_ms": distribution_stats([float(row["join_age_ms"]) for row in rows]),
            "effective_support": {
                "n_rows": len(rows),
                "n_responses": len(
                    {row.get("response_id") for row in rows if row.get("response_id")}
                ),
                "n_telemetry_samples": len(
                    {index for row in rows for index in row["telemetry_sample_indices"].values()}
                ),
            },
            "run_level_sensitivity": {
                "status": "unavailable-one-run",
                "n_runs": 1,
                "interpretation": (
                    "response resampling is conditional on this run and does not estimate "
                    "run-to-run uncertainty"
                ),
            },
            "regressions": regressions,
        }
        if population == "delivered-nonBOS":
            populations[population]["response_level_stage_means"] = _response_stage_means(rows)
            spike_rows = [row for row in rows if row["server_step_ms"] >= 88.0]
            populations["delivered-generation->=88ms"] = {
                "definition": "analyzer delivered-nonBOS membership and server_step_ms >= 88.0",
                "membership": _membership(spike_rows),
                "server_step_ms": distribution_stats(
                    [float(row["server_step_ms"]) for row in spike_rows]
                ),
                "regressions": (
                    {channel: regress(spike_rows, channel, clustered=True) for channel in channels}
                    if spike_rows
                    else None
                ),
            }
    unavailable = {
        channel: sum(row["values"].get(channel) is None for row in telemetry)
        for channel in channels
    }
    channel_sample_distributions = {}
    for channel in channels:
        values = [
            float(row["values"][channel])
            for row in telemetry
            if row["values"].get(channel) is not None
        ]
        channel_sample_distributions[channel] = {
            "n": len(values),
            "n_unique": len(set(values)),
            "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "schema": "nemotron_voicechat.step4_telemetry_regression.v2",
        "inference_scope": "single-run exploratory descriptive sensitivities; no causal null",
        "multiplicity_policy": {
            "status": "exploratory-no-familywise-adjustment",
            "interpretation": (
                "All channels and sensitivity views are reported; intervals are descriptive "
                "and no channel is promoted or rejected by nominal interval exclusion."
            ),
        },
        "clock_mapping": {
            "method": (
                "direct shared-host monotonic clock; each value joins to the latest "
                "causally prior source-acquisition end; PSI deltas describe their recorded "
                "preceding intervals"
            ),
            "telemetry_samples": len(telemetry),
            "poll_interval_s": {
                "min": min(intervals),
                "mean": sum(intervals) / len(intervals),
                "max": max(intervals),
            },
            "wall_minus_monotonic_s": {
                "min": min(offsets),
                "mean": sum(offsets) / len(offsets),
                "max": max(offsets),
                "range_ms": (max(offsets) - min(offsets)) * 1000.0,
            },
            "maximum_permitted_causal_join_age_s": MAX_CAUSAL_JOIN_AGE_S,
        },
        "sources": {
            "qualified_analyzer": source_identity(analyzer_path),
            "server_trace": identity,
            "telemetry": source_identity(telemetry_path),
        },
        "channels": channels,
        "unavailable_sample_counts": unavailable,
        "channel_sample_distributions": channel_sample_distributions,
        "populations": populations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--server-trace", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.analyzer, args.server_trace, args.telemetry)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
