#!/usr/bin/env python3
"""Reduce Step-7 same-engine PIECEWISE/FULL latency evidence."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def delta(
    baseline: dict[str, float | int], candidate: dict[str, float | int]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in ("mean_ms", "p50_ms", "p95_ms"):
        old = float(baseline[name])
        new = float(candidate[name])
        result[name.removesuffix("_ms")] = {
            "candidate_minus_baseline_ms": new - old,
            "candidate_minus_baseline_percent": (new - old) * 100.0 / old,
        }
    return result


def read_marked_json(log: Path, marker: str) -> list[dict[str, Any]]:
    records = []
    for line in log.read_text(errors="replace").splitlines():
        if marker in line:
            records.append(json.loads(line.split(marker, 1)[1]))
    return records


def load_model_steps(trace_root: Path) -> dict[int, list[dict[str, Any]]]:
    result = {}
    pattern = str(trace_root / "model" / "*" / "events.jsonl")
    for raw_path in glob.glob(pattern):
        events = [
            json.loads(line)
            for line in Path(raw_path).read_text().splitlines()
            if '"event":"model_step"' in line
        ]
        request_ids = {
            str(event["turn_state"]["vllm_request_positions"]["request_id"])
            for event in events
            if event["turn_state"]["vllm_request_positions"].get("request_id") is not None
        }
        if len(request_ids) != 1:
            raise RuntimeError(f"ambiguous request IDs in {raw_path}: {request_ids}")
        request_id = int(request_ids.pop())
        if request_id in result:
            raise RuntimeError(f"duplicate trace for request {request_id}")
        result[request_id] = events
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.evidence_root
    output = args.output or root / "analysis.json"

    workers = read_marked_json(root / "server.log", "VOICECHAT_PAIR_WORKER ")
    rpcs = read_marked_json(root / "server.log", "VOICECHAT_PAIR_RPC ")
    for worker in workers:
        matches = [
            rpc
            for rpc in rpcs
            if rpc["wait_started_ns"] <= worker["execute_started_ns"]
            and worker["execute_finished_ns"] <= rpc["wait_finished_ns"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"worker/RPC correlation count is {len(matches)}")
        worker["request_id"] = matches[0]["request_id"]

    steps = load_model_steps(root / "traces")
    measured_ids = {2, 3, 4, 5, 6, 7}
    if set(steps) != measured_ids:
        raise RuntimeError(f"unexpected measured request IDs: {sorted(steps)}")

    modes = {
        "piecewise": {2, 4, 6},
        "full": {3, 5, 7},
    }
    mode_results = {}
    session_results = []
    for mode, request_ids in modes.items():
        mode_workers = [worker for worker in workers if int(worker["request_id"]) in request_ids]
        expected_graph = mode == "full"
        if not mode_workers or any(
            bool(worker["pair_full_graph"]) != expected_graph for worker in mode_workers
        ):
            raise RuntimeError(f"invalid {mode} worker routing")
        mode_rpcs = [rpc for rpc in rpcs if int(rpc["request_id"]) in request_ids]
        pair_step_ms = []
        pair_cycle_ms = []
        correlation_lag_ms = []
        rpc_overhead_ms = []

        for rpc in mode_rpcs:
            request_id = int(rpc["request_id"])
            request_steps = steps[request_id]
            target_s = rpc["wait_finished_ns"] / 1e9
            index = min(
                range(len(request_steps)),
                key=lambda candidate: abs(request_steps[candidate]["monotonic_s"] - target_s),
            )
            lag_ms = (request_steps[index]["monotonic_s"] - target_s) * 1000.0
            if index == 0 or not 0.0 <= lag_ms < 50.0:
                raise RuntimeError(
                    f"invalid RPC/model-step correlation for request {request_id}: "
                    f"index={index}, lag_ms={lag_ms}"
                )
            pair_step_ms.append(float(request_steps[index]["server_step_ms"]))
            pair_cycle_ms.append(
                float(request_steps[index - 1]["server_step_ms"])
                + float(request_steps[index]["server_step_ms"])
            )
            correlation_lag_ms.append(lag_ms)

        for worker in mode_workers:
            rpc = next(
                record
                for record in mode_rpcs
                if record["request_id"] == worker["request_id"]
                and record["wait_started_ns"] <= worker["execute_started_ns"]
                and worker["execute_finished_ns"] <= record["wait_finished_ns"]
            )
            rpc_overhead_ms.append(float(rpc["response_wait_ms"]) - float(worker["execute_ms"]))

        all_server_steps = [
            float(event["server_step_ms"])
            for request_id in request_ids
            for event in steps[request_id]
        ]
        mode_results[mode] = {
            "request_ids": sorted(request_ids),
            "worker_execute": summary([float(worker["execute_ms"]) for worker in mode_workers]),
            "rpc_response_wait": summary([float(rpc["response_wait_ms"]) for rpc in mode_rpcs]),
            "rpc_non_execute_overhead": summary(rpc_overhead_ms),
            "pair_execution_server_step": summary(pair_step_ms),
            "buffer_plus_pair_cycle": summary(pair_cycle_ms),
            "pair_cycle_per_logical_frame": summary([value / 2.0 for value in pair_cycle_ms]),
            "all_server_steps": summary(all_server_steps),
            "rpc_to_model_trace_correlation_lag": summary(correlation_lag_ms),
        }

    for report_path in sorted((root / "sessions").glob("request-*/report.json")):
        report = json.loads(report_path.read_text())
        request_id = int(report_path.parent.name.split("-")[1])
        session_results.append(
            {
                "request_id": request_id,
                "mode": "piecewise" if request_id % 2 == 0 else "full",
                "metric_frames": report["metric_frames"],
                "metric_sequence_complete": report["metric_sequence_complete"],
                "typed_completed": report["typed_completed"],
                "typed_answered": report["typed_answered"],
                "unexpected_errors": report["unexpected_errors"],
                "short_run_queue_result": report["queue"],
            }
        )

    watchdog_events = [
        json.loads(line) for line in (root / "watchdog.jsonl").read_text().splitlines()
    ]
    watchdog_samples = [event for event in watchdog_events if event["event"] == "sample"]
    inspect = json.loads((root / "container-inspect.json").read_text())[0]
    comparisons = {}
    for metric in (
        "worker_execute",
        "rpc_response_wait",
        "rpc_non_execute_overhead",
        "pair_execution_server_step",
        "buffer_plus_pair_cycle",
        "pair_cycle_per_logical_frame",
        "all_server_steps",
    ):
        comparisons[metric] = delta(mode_results["piecewise"][metric], mode_results["full"][metric])

    result = {
        "schema": 1,
        "method": "single-engine alternating request IDs; even PIECEWISE, odd FULL",
        "modes": mode_results,
        "candidate_minus_baseline": comparisons,
        "sessions": session_results,
        "functional_fixture_passed": bool(session_results)
        and all(
            session["metric_sequence_complete"]
            and session["typed_completed"] == 3
            and session["typed_answered"] == 3
            and not session["unexpected_errors"]
            for session in session_results
        ),
        "memory_guardrail": {
            "samples": len(watchdog_samples),
            "minimum_mem_available_gib": min(
                event["mem_available_gib"] for event in watchdog_samples
            ),
            "maximum_mem_available_gib": max(
                event["mem_available_gib"] for event in watchdog_samples
            ),
            "terminal_event": watchdog_events[-1]["event"],
            "low_memory_kill": any(
                event["event"] == "low_memory_kill" for event in watchdog_events
            ),
            "container_memory_bytes": inspect["HostConfig"]["Memory"],
            "container_memory_swap_bytes": inspect["HostConfig"]["MemorySwap"],
            "container_oom_killed": inspect["State"]["OOMKilled"],
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
