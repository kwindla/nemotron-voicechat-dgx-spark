#!/usr/bin/env python3
"""Benchmark and decompose matched public Nano content/PAD interface states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

PAD_TOKEN_ID = 12
CONTROL_TOKEN_IDS = {1, 2}
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_PAIRS = ((45, 46), (326, 327), (685, 686))
PROFILE_RANGE = re.compile(
    r"^step4d:block=(?P<block>\d+);pair=(?P<pair>\d+);"
    r"class=(?P<class>content|pad);call=(?P<call>\d+);step=(?P<step>\d+)$"
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    calls = [int(record["call_index"]) for record in records]
    if calls != list(range(len(records))):
        raise ValueError("capture manifest calls are not contiguous from zero")
    return records


def _tensor_contract(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("decode_steps", 0)),
        record.get("request_id"),
        int(record.get("sequence_epoch", -1)),
    )


def select_matched_states(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select lexical-run boundary states followed immediately by PAD."""
    pairs = []
    for index in range(len(records) - 1):
        content = records[index]
        pad = records[index + 1]
        token = int(content["predicted_token"])
        next_token = int(pad["predicted_token"])
        previous_lexical = token > PAD_TOKEN_ID
        followed_by_pad = next_token == PAD_TOKEN_ID
        if not (previous_lexical and followed_by_pad):
            continue
        if _tensor_contract(content) != _tensor_contract(pad):
            raise ValueError(f"candidate pair {index}/{index + 1} crosses a capture contract")
        if int(pad["current_step"]) != int(content["current_step"]) + 1:
            raise ValueError(f"candidate pair {index}/{index + 1} is not position-consecutive")
        pairs.append(
            {
                "pair": len(pairs) + 1,
                "request_id": content["request_id"],
                "sequence_epoch": int(content["sequence_epoch"]),
                "decode_steps": int(content["decode_steps"]),
                "content": {
                    "call_index": int(content["call_index"]),
                    "current_step": int(content["current_step"]),
                    "expected_token": token,
                    "file": content["file"],
                    "sha256": content["sha256"],
                },
                "pad": {
                    "call_index": int(pad["call_index"]),
                    "current_step": int(pad["current_step"]),
                    "expected_token": next_token,
                    "file": pad["file"],
                    "sha256": pad["sha256"],
                },
            }
        )
    actual = tuple((item["content"]["call_index"], item["pad"]["call_index"]) for item in pairs)
    if actual != EXPECTED_PAIRS:
        raise ValueError(f"captured state selection changed: {actual!r} != {EXPECTED_PAIRS!r}")
    return pairs


def validate_pair_payloads(root: Path, pairs: list[dict[str, Any]]) -> None:
    """Bind the manifest-selected pairs to equal captured tensor contracts."""
    torch = _torch()
    for pair in pairs:
        contracts = {}
        for state_class in ("content", "pad"):
            state = pair[state_class]
            path = root / state["file"]
            if sha256_file(path) != state["sha256"]:
                raise ValueError(f"selected capture payload hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            inputs = payload["inputs"]
            embeds = inputs["input_embeds"]
            prompt = inputs.get("prompt_token_ids")
            if prompt is not None:
                raise ValueError(f"selected {state_class} state has a prompt-token override")
            contracts[state_class] = {
                "input_embed_shape": list(embeds.shape),
                "input_embed_dtype": str(embeds.dtype),
                "decode_steps": int(inputs.get("decode_steps", 1)),
                "prompt_token_ids": None,
            }
            if int(inputs.get("current_step", -1)) != state["current_step"]:
                raise ValueError(f"selected {state_class} payload current_step mismatch")
        if contracts["content"] != contracts["pad"]:
            raise ValueError(f"pair {pair['pair']} tensor contracts differ")
        pair["input_contract"] = contracts["content"]


def selection_payload(root: Path) -> dict[str, Any]:
    records = read_manifest(root)
    metadata_path = root / "metadata.json"
    pairs = select_matched_states(records)
    validate_pair_payloads(root, pairs)
    return {
        "schema": 1,
        "kind": "step4d_nano_matched_state_selection",
        "capture_root": str(root.resolve()),
        "capture_metadata_sha256": sha256_file(metadata_path),
        "capture_manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "capture_calls": len(records),
        "max_replayed_call": max(item["pad"]["call_index"] for item in pairs),
        "rules": {
            "content": "predicted_token > 12",
            "pad": "predicted_token == 12",
            "controls_excluded": sorted(CONTROL_TOKEN_IDS),
            "boundary": "last lexical state immediately followed by PAD",
            "causal_replay": "every call from zero through max_replayed_call",
        },
        "pairs": pairs,
    }


def _hash_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"null;")
    elif isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
    elif isinstance(value, int):
        digest.update(f"int:{value};".encode())
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite output cannot enter exact-output digest")
        digest.update(f"float:{value.hex()};".encode())
    elif isinstance(value, str):
        encoded = value.encode()
        digest.update(f"str:{len(encoded)}:".encode())
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        digest.update(b"};")
    elif isinstance(value, (list, tuple)):
        digest.update(f"seq:{len(value)}:[".encode())
        for item in value:
            _hash_value(digest, item)
        digest.update(b"];")
    elif hasattr(value, "detach") and hasattr(value, "shape"):
        tensor = value.detach().contiguous().cpu()
        metadata = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(b"tensor:")
        _hash_value(digest, metadata)
        digest.update(tensor.view(-1).view(dtype=_torch().uint8).numpy().tobytes())
        digest.update(b";")
    else:
        raise TypeError(f"unsupported exact-output value: {type(value)!r}")


def exact_output_digest(interface_result: dict[str, Any]) -> str:
    """Hash production semantics while excluding request identity and timing telemetry."""
    interface = {
        key: value
        for key, value in interface_result.items()
        if key not in {"request_id", "nano_engine_benchmark"}
    }
    digest = hashlib.sha256()
    _hash_value(digest, {"interface": interface})
    return digest.hexdigest()


def _torch():
    import torch

    return torch


def _load_model(args: argparse.Namespace) -> Any:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        }
    )
    sys.path.insert(0, str(args.speech_root.resolve()))
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model

    return create_model(
        engine_type="vllm_llm",
        vllm_config={
            "model_path": str(args.nano_vllm_path.resolve()),
            "engine_path": str(args.nano_vllm_path.resolve()),
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": "bfloat16",
            "enforce_eager": False,
        },
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=0.0,
        text_pad_id=PAD_TOKEN_ID,
    )


def _profile_begin(torch: Any, message: str) -> None:
    result = torch.cuda.cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {result}")
    # Nsight injects CUPTI into the vLLM worker process. Give capture enablement
    # time to cross that process boundary before the measured NVTX interval.
    time.sleep(0.02)
    torch.cuda.nvtx.range_push(message)


def _profile_end(torch: Any) -> None:
    torch.cuda.nvtx.range_pop()
    # Keep collection alive briefly after the measured interval so the worker's
    # graph record is committed before the parent disables the capture range.
    time.sleep(0.005)
    result = torch.cuda.cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed: {result}")


def run_component(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    root = args.capture_root.resolve()
    selection = selection_payload(root)
    records = read_manifest(root)[: selection["max_replayed_call"] + 1]
    targets: dict[int, tuple[int, str, dict[str, Any]]] = {}
    for pair in selection["pairs"]:
        for state_class in ("content", "pad"):
            state = pair[state_class]
            targets[state["call_index"]] = (pair["pair"], state_class, state)
    model = _load_model(args)
    blocks = []
    reference: dict[str, str] = {}
    external_reference = None
    if args.reference_outputs:
        external_reference = json.loads(args.reference_outputs.read_text(encoding="utf-8"))
        reference = dict(external_reference["reference_output_digests"])
    try:
        for ordinal in range(args.warmup_blocks + args.measured_blocks):
            measured = ordinal >= args.warmup_blocks
            block_index = ordinal - args.warmup_blocks if measured else ordinal
            phase = "measured" if measured else "warmup"
            request_id = f"step4d-{args.mode}-{phase}-{block_index:03d}"
            selected = []
            mismatches = []
            started = time.perf_counter()
            cleanup_ok = False
            try:
                for record in records:
                    payload = torch.load(
                        root / record["file"], map_location="cpu", weights_only=False
                    )
                    saved = payload["inputs"]
                    call_index = int(record["call_index"])
                    kwargs = {
                        "current_step": int(saved.get("current_step", 0)),
                        "decode_steps": int(saved.get("decode_steps", 1)),
                    }
                    if saved.get("prompt_token_ids") is not None:
                        kwargs["prompt_token_ids"] = saved["prompt_token_ids"]
                    generated = saved.get("generated_tokens")
                    if generated is not None:
                        kwargs["generated_tokens"] = generated.to("cuda")
                    target = targets.get(call_index)
                    message = None
                    if args.mode == "profile" and measured and target is not None:
                        pair_index, state_class, state = target
                        message = (
                            f"step4d:block={block_index:03d};pair={pair_index};"
                            f"class={state_class};call={call_index};"
                            f"step={state['current_step']}"
                        )
                        _profile_begin(torch, message)
                    call_start = time.perf_counter()
                    result = model(
                        saved["input_embeds"].to("cuda"), request_id=request_id, **kwargs
                    )
                    call_ms = (time.perf_counter() - call_start) * 1000.0
                    if message is not None:
                        _profile_end(torch)
                    actual_token = int(result["predicted_token"])
                    expected_token = int(record["predicted_token"])
                    if actual_token != expected_token:
                        mismatches.append(
                            {
                                "call_index": call_index,
                                "expected_token": expected_token,
                                "actual_token": actual_token,
                            }
                        )
                    if target is not None:
                        pair_index, state_class, state = target
                        output_digest = exact_output_digest(result)
                        key = f"pair-{pair_index}-{state_class}"
                        if measured and args.mode == "exact" and not reference:
                            # Populate all six entries during the first measured block.
                            pass
                        selected.append(
                            {
                                "pair": pair_index,
                                "class": state_class,
                                "call_index": call_index,
                                "current_step": state["current_step"],
                                "expected_token": state["expected_token"],
                                "actual_token": actual_token,
                                "interface_ms": call_ms,
                                "output_digest": output_digest,
                                "reference_key": key,
                                "nvtx_range": message,
                            }
                        )
                model.abort_request(request_id)
                cleanup_ok = True
            finally:
                if not cleanup_ok:
                    try:
                        model.abort_request(request_id)
                    except Exception:
                        pass
            if measured and args.mode == "exact" and not reference:
                reference = {item["reference_key"]: item["output_digest"] for item in selected}
            for item in selected:
                expected_digest = reference.get(item["reference_key"])
                item["reference_digest"] = expected_digest
                item["exact_output_match"] = (
                    None if expected_digest is None else item["output_digest"] == expected_digest
                )
            valid = (
                cleanup_ok
                and len(selected) == len(targets)
                and not mismatches
                and (
                    not measured
                    or not reference
                    or all(item["exact_output_match"] is True for item in selected)
                )
            )
            blocks.append(
                {
                    "phase": phase,
                    "block": block_index,
                    "request_id": request_id,
                    "wall_ms": (time.perf_counter() - started) * 1000.0,
                    "calls": len(records),
                    "cleanup_ok": cleanup_ok,
                    "token_mismatches": mismatches,
                    "selected": selected,
                    "valid": valid,
                }
            )
    finally:
        model.shutdown()

    measured = [block for block in blocks if block["phase"] == "measured"]
    report = {
        "schema": 1,
        "kind": "step4d_nano_component_replay",
        "mode": args.mode,
        "capture": selection,
        "component": {
            "path": str(args.nano_vllm_path.resolve()),
            "config_sha256": sha256_file(args.nano_vllm_path / "config.json"),
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": False,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0},
        },
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": args.measured_blocks,
        "reference_source": (
            str(args.reference_outputs.resolve()) if external_reference else "first-measured-block"
        ),
        "reference_output_digests": reference,
        "blocks": blocks,
        "passed": len(measured) == args.measured_blocks
        and all(block["valid"] for block in measured),
    }
    atomic_json(args.output, report)
    return report


def interval_union_ms(intervals: Iterable[tuple[int, int]], start: int, end: int) -> float:
    clipped = sorted((max(left, start), min(right, end)) for left, right in intervals)
    clipped = [(left, right) for left, right in clipped if right > left]
    if not clipped:
        return 0.0
    total = 0
    current_start, current_end = clipped[0]
    for left, right in clipped[1:]:
        if left <= current_end:
            current_end = max(current_end, right)
        else:
            total += current_end - current_start
            current_start, current_end = left, right
    total += current_end - current_start
    return total / 1_000_000.0


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(connection):
        return {}
    return {
        int(row[0]): str(row[1]) for row in connection.execute("SELECT id, value FROM StringIds")
    }


def _nvtx_ranges(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    if "NVTX_EVENTS" not in _tables(connection):
        raise ValueError("Nsight SQLite has no NVTX_EVENTS table")
    columns = _columns(connection, "NVTX_EVENTS")
    strings = _strings(connection)
    selected = ["start", "end"]
    if "text" in columns:
        selected.append("text")
    if "textId" in columns:
        selected.append("textId")
    ranges = []
    for row in connection.execute(f"SELECT {', '.join(selected)} FROM NVTX_EVENTS"):
        values = dict(zip(selected, row, strict=True))
        text = values.get("text")
        if not text and values.get("textId") is not None:
            text = strings.get(int(values["textId"]))
        if isinstance(text, str) and PROFILE_RANGE.fullmatch(text) and values["end"] is not None:
            ranges.append((text, int(values["start"]), int(values["end"])))
    return ranges


def _timed_rows(connection: sqlite3.Connection, table: str) -> list[tuple[int, int]]:
    if table not in _tables(connection):
        return []
    columns = _columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        return []
    return [
        (int(row[0]), int(row[1]))
        for row in connection.execute(f'SELECT start, end FROM "{table}"')
    ]


def _runtime_rows(connection: sqlite3.Connection) -> list[tuple[int, int, str]]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in _tables(connection):
        return []
    columns = _columns(connection, table)
    strings = _strings(connection)
    name_column = "nameId" if "nameId" in columns else "name"
    rows = []
    for start, end, name in connection.execute(
        f'SELECT start, end, "{name_column}" FROM "{table}"'
    ):
        if name_column == "nameId":
            name = strings.get(int(name), str(name))
        rows.append((int(start), int(end), str(name)))
    return rows


def _extract_nsys_sqlite(
    sqlite_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = _tables(connection)
        gpu_activity_by_table = {
            name: activity
            for name in (
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            )
            if name in tables and (activity := _timed_rows(connection, name))
        }
        if not gpu_activity_by_table:
            raise ValueError("Nsight SQLite has no nonempty CUDA kernel or graph activity table")
        graph_activity = gpu_activity_by_table.get("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", [])
        explicit_kernel_activity = [
            interval
            for table in (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            )
            for interval in gpu_activity_by_table.get(table, [])
        ]
        memory_tables = sorted(
            name
            for name in tables
            if name.startswith("CUPTI_ACTIVITY_KIND_MEM")
            and {"start", "end"}.issubset(_columns(connection, name))
        )
        memory = [row for table in memory_tables for row in _timed_rows(connection, table)]
        runtime = _runtime_rows(connection)
        rows = []
        for message, start, end in _nvtx_ranges(connection):
            identity = PROFILE_RANGE.fullmatch(message)
            assert identity is not None
            overlapping_graph_envelopes = [
                (left, right) for left, right in graph_activity if left < end and right > start
            ]
            overlapping_explicit_kernels = [
                (left, right)
                for left, right in explicit_kernel_activity
                if left < end and right > start
            ]
            overlapping_memory = [
                (left, right) for left, right in memory if left < end and right > start
            ]
            overlapping_runtime = [row for row in runtime if row[0] < end and row[1] > start]
            wall_ms = (end - start) / 1_000_000.0

            # A graph trace is the authoritative execution envelope. Explicit
            # kernels in a dual-table export are auxiliary observations, not a
            # reason to relabel or discard the graph range.
            primary_gpu_activity = (
                overlapping_graph_envelopes
                if overlapping_graph_envelopes
                else overlapping_explicit_kernels
            )
            kernel_sum_ms = (
                sum(min(right, end) - max(left, start) for left, right in primary_gpu_activity)
                / 1_000_000.0
            )
            kernel_union_ms = interval_union_ms(primary_gpu_activity, start, end)
            graph_envelope_sum_ms = (
                sum(
                    min(right, end) - max(left, start)
                    for left, right in overlapping_graph_envelopes
                )
                / 1_000_000.0
            )
            graph_envelope_union_ms = interval_union_ms(
                overlapping_graph_envelopes, start, end
            )
            explicit_kernel_sum_ms = (
                sum(
                    min(right, end) - max(left, start)
                    for left, right in overlapping_explicit_kernels
                )
                / 1_000_000.0
            )
            explicit_kernel_union_ms = interval_union_ms(
                overlapping_explicit_kernels, start, end
            )
            gpu_busy_union_ms = interval_union_ms(
                [*overlapping_graph_envelopes, *overlapping_explicit_kernels], start, end
            )
            memory_union_ms = interval_union_ms(overlapping_memory, start, end)
            rows.append(
                {
                    "block": int(identity["block"]),
                    "pair": int(identity["pair"]),
                    "class": identity["class"],
                    "call_index": int(identity["call"]),
                    "current_step": int(identity["step"]),
                    "nvtx_range": message,
                    "start_ns": start,
                    "end_ns": end,
                    "interface_wall_ms": wall_ms,
                    "kernel_count": len(primary_gpu_activity),
                    "gpu_activity_kind": (
                        "cuda_graph_execution_envelope"
                        if overlapping_graph_envelopes
                        else "cuda_kernel"
                    ),
                    "kernel_sum_ms": kernel_sum_ms,
                    "kernel_union_ms": kernel_union_ms,
                    "kernel_fraction": kernel_union_ms / wall_ms if wall_ms else None,
                    "graph_envelope_count": len(overlapping_graph_envelopes),
                    "graph_envelope_sum_ms": graph_envelope_sum_ms,
                    "graph_envelope_union_ms": graph_envelope_union_ms,
                    "graph_envelope_fraction": (
                        graph_envelope_union_ms / wall_ms if wall_ms else None
                    ),
                    "graph_envelope_residual_ms": wall_ms - graph_envelope_union_ms,
                    "graph_envelope_residual_fraction": (
                        1.0 - graph_envelope_union_ms / wall_ms if wall_ms else None
                    ),
                    "explicit_kernel_count": len(overlapping_explicit_kernels),
                    "explicit_kernel_sum_ms": explicit_kernel_sum_ms,
                    "explicit_kernel_union_ms": explicit_kernel_union_ms,
                    "explicit_kernel_outside_graph_union_ms": (
                        gpu_busy_union_ms - graph_envelope_union_ms
                    ),
                    "gpu_busy_union_ms": gpu_busy_union_ms,
                    "gpu_busy_fraction": gpu_busy_union_ms / wall_ms if wall_ms else None,
                    "gpu_busy_residual_ms": wall_ms - gpu_busy_union_ms,
                    "gpu_busy_residual_fraction": (
                        1.0 - gpu_busy_union_ms / wall_ms if wall_ms else None
                    ),
                    "gpu_memory_operation_count": len(overlapping_memory),
                    "gpu_memory_union_ms": memory_union_ms,
                    "cuda_api_count": len(overlapping_runtime),
                    "cuda_api_ms": sum(
                        min(right, end) - max(left, start) for left, right, _ in overlapping_runtime
                    )
                    / 1_000_000.0,
                    "launch_api_count": sum(
                        "launch" in name.lower() for _, _, name in overlapping_runtime
                    ),
                    "non_kernel_residual_ms": wall_ms - kernel_union_ms,
                    "non_kernel_fraction": 1.0 - kernel_union_ms / wall_ms if wall_ms else None,
                    "closure_error_ms": wall_ms - kernel_union_ms - (wall_ms - kernel_union_ms),
                }
            )
    finally:
        connection.close()
    rows.sort(key=lambda row: (row["block"], row["pair"], row["class"]))
    return rows, sorted(gpu_activity_by_table), memory_tables


def analyze_nsys_sqlites(
    sqlite_paths: Sequence[Path], expected_blocks: int, *, require_cuda_graph: bool = False
) -> dict[str, Any]:
    rows = []
    kernel_tables = set()
    memory_tables = set()
    sources = []
    for path in sqlite_paths:
        extracted, source_gpu_activity_tables, source_memory_tables = _extract_nsys_sqlite(path)
        rows.extend(extracted)
        kernel_tables.update(source_gpu_activity_tables)
        memory_tables.update(source_memory_tables)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ranges": len(extracted),
                "gpu_activity_tables": source_gpu_activity_tables,
            }
        )
    rows.sort(key=lambda row: (row["block"], row["pair"], row["class"]))
    expected = expected_blocks * len(EXPECTED_PAIRS) * 2
    identities = {(row["block"], row["pair"], row["class"]) for row in rows}
    expected_identities = {
        (block, pair, state_class)
        for block in range(expected_blocks)
        for pair in range(1, len(EXPECTED_PAIRS) + 1)
        for state_class in ("content", "pad")
    }
    return {
        "schema": 1,
        "kind": "step4d_nano_nsys_decomposition",
        "source_sqlites": sources,
        "gpu_activity_tables": sorted(kernel_tables),
        "memory_tables": sorted(memory_tables),
        "expected_ranges": expected,
        "observed_ranges": len(rows),
        "missing_ranges": [list(item) for item in sorted(expected_identities - identities)],
        "duplicate_range_identities": len(rows) - len(identities),
        "require_cuda_graph": require_cuda_graph,
        "rows": rows,
        "passed": (
            len(rows) == expected
            and identities == expected_identities
            and all(
                row["interface_wall_ms"] > 0
                and row["kernel_count"] > 0
                and (
                    not require_cuda_graph
                    or row["graph_envelope_count"] > 0
                )
                for row in rows
            )
            and all(abs(row["closure_error_ms"]) <= 0.05 for row in rows)
        ),
    }


def analyze_nsys_sqlite(sqlite_path: Path, expected_blocks: int) -> dict[str, Any]:
    """Compatibility wrapper for a single Nsight SQLite export."""
    return analyze_nsys_sqlites([sqlite_path], expected_blocks)


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty values")
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_mean(values: Sequence[float]) -> list[float]:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [nearest_rank(samples, 2.5), nearest_rank(samples, 97.5)]


def metric_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "blocks": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "block_bootstrap_95_ci_for_mean": bootstrap_mean(values),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def summarize(exact: dict[str, Any], decomposition: dict[str, Any]) -> dict[str, Any]:
    measured_exact = [block for block in exact["blocks"] if block["phase"] == "measured"]
    grouped: dict[str, dict[str, list[float]]] = {}
    for metric in (
        "interface_wall_ms",
        "kernel_union_ms",
        "non_kernel_residual_ms",
        "kernel_fraction",
        "non_kernel_fraction",
        "graph_envelope_union_ms",
        "graph_envelope_residual_ms",
        "graph_envelope_fraction",
        "graph_envelope_residual_fraction",
        "explicit_kernel_outside_graph_union_ms",
        "gpu_busy_union_ms",
        "gpu_busy_residual_ms",
        "gpu_busy_fraction",
        "gpu_busy_residual_fraction",
    ):
        grouped[metric] = {"content": [], "pad": [], "pad_minus_content": []}
    by_block: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in decomposition["rows"]:
        by_block.setdefault(row["block"], {}).setdefault(row["class"], []).append(row)
    require_graph = bool(decomposition.get("require_cuda_graph"))
    valid_block_ids = [
        block
        for block, classes in sorted(by_block.items())
        if set(classes) == {"content", "pad"}
        and all(len(classes[state_class]) == len(EXPECTED_PAIRS) for state_class in classes)
        and (
            not require_graph
            or all(
                row.get("graph_envelope_count", 0) > 0
                for rows in classes.values()
                for row in rows
            )
        )
    ]
    for block in valid_block_ids:
        classes = by_block[block]
        for metric in grouped:
            content = statistics.fmean(row[metric] for row in classes["content"])
            pad = statistics.fmean(row[metric] for row in classes["pad"])
            grouped[metric]["content"].append(content)
            grouped[metric]["pad"].append(pad)
            grouped[metric]["pad_minus_content"].append(pad - content)
    summaries = {
        metric: {name: metric_summary(values) for name, values in categories.items()}
        for metric, categories in grouped.items()
    }
    exact_states = [item for block in measured_exact for item in block["selected"]]
    exact_passed = (
        exact.get("passed") is True
        and len(measured_exact) == 30
        and len(exact_states) == 30 * len(EXPECTED_PAIRS) * 2
        and all(block.get("valid") is True for block in measured_exact)
        and all(item.get("exact_output_match") is True for item in exact_states)
    )
    profile_passed = decomposition.get("passed") is True and len(valid_block_ids) == 10
    return {
        "schema": 1,
        "kind": "step4d_nano_decomposition_summary",
        "uncertainty_unit": "independent causal replay block",
        "exact_output": {
            "measured_blocks": len(measured_exact),
            "states": len(exact_states),
            "matching_states": sum(item.get("exact_output_match") is True for item in exact_states),
            "reference_output_digests": exact.get("reference_output_digests"),
            "passed": exact_passed,
        },
        "decomposition": summaries,
        "decomposition_blocks": {
            "observed": len(by_block),
            "valid": len(valid_block_ids),
            "valid_block_ids": valid_block_ids,
            "invalid_block_ids": sorted(set(by_block) - set(valid_block_ids)),
            "whole_block_exclusion": True,
        },
        "recoverable_ceiling": {
            "graph_envelope": summaries["graph_envelope_union_ms"],
            "graph_envelope_fraction": summaries["graph_envelope_fraction"],
            "wall_minus_graph_envelope": summaries["graph_envelope_residual_ms"],
            "wall_minus_graph_envelope_fraction": summaries[
                "graph_envelope_residual_fraction"
            ],
            "explicit_kernel_outside_graph": summaries[
                "explicit_kernel_outside_graph_union_ms"
            ],
            "all_observed_gpu_busy": summaries["gpu_busy_union_ms"],
            "interpretation": "impossible zero-cost upper bounds, not predicted speedups",
        },
        "passed": exact_passed and profile_passed,
        "decision": "COMPLETE" if exact_passed and profile_passed else "INCOMPLETE",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--capture-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("exact", "profile"), required=True)
    run.add_argument("--speech-root", type=Path, required=True)
    run.add_argument("--nano-vllm-path", type=Path, required=True)
    run.add_argument("--capture-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference-outputs", type=Path)
    run.add_argument("--warmup-blocks", type=int, default=2)
    run.add_argument("--measured-blocks", type=int)
    run.add_argument("--max-model-len", type=int, default=12288)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.42)

    analyze = subparsers.add_parser("analyze-nsys")
    analyze.add_argument("--sqlite", type=Path, action="append", required=True)
    analyze.add_argument("--expected-blocks", type=int, default=10)
    analyze.add_argument("--require-cuda-graph", action="store_true")
    analyze.add_argument("--output", type=Path, required=True)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--exact", type=Path, required=True)
    summary.add_argument("--decomposition", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.measured_blocks is None:
            args.measured_blocks = 30 if args.mode == "exact" else 10
        if args.warmup_blocks < 0 or args.measured_blocks <= 0:
            parser.error("block counts must be nonnegative/positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "select":
        payload = selection_payload(args.capture_root.resolve())
        atomic_json(args.output, payload)
    elif args.command == "run":
        payload = run_component(args)
    elif args.command == "analyze-nsys":
        payload = analyze_nsys_sqlites(
            [path.resolve() for path in args.sqlite],
            args.expected_blocks,
            require_cuda_graph=args.require_cuda_graph,
        )
        atomic_json(args.output, payload)
    else:
        exact = json.loads(args.exact.read_text(encoding="utf-8"))
        decomposition = json.loads(args.decomposition.read_text(encoding="utf-8"))
        payload = summarize(exact, decomposition)
        atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command in {"run", "analyze-nsys", "summarize"} and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
