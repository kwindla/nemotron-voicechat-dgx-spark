#!/usr/bin/env python3
"""Run and summarize fresh-engine W8 graph reproducibility experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

CONTROL_TOKEN_IDS = {1, 2, 12, 20, 21, 22}
SELECTED_CALLS = {45, 46, 326, 327, 685, 686}
COMPLETE_PREFIX_CALLS = 687
BOUNDED_GATE_MEASURED_BLOCKS = 30
BOUNDED_GATE_FRESH_ENGINES = 5
BOUNDED_GATE_MAX_MARGIN = 0.05


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


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_manifest(root: Path, max_call_index: int) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    calls = [int(record["call_index"]) for record in records]
    if calls != list(range(len(records))):
        raise ValueError("capture manifest calls are not contiguous from zero")
    if max_call_index >= len(records):
        raise ValueError(f"max call {max_call_index} exceeds capture ending at {len(records) - 1}")
    return records[: max_call_index + 1]


def _torch():
    import torch

    return torch


def _hash_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"null;")
    elif isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
    elif isinstance(value, int):
        digest.update(f"int:{value};".encode())
    elif isinstance(value, float):
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
        digest.update(b"tensor:")
        _hash_value(digest, {"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        digest.update(tensor.view(-1).view(dtype=_torch().uint8).numpy().tobytes())
        digest.update(b";")
    else:
        raise TypeError(f"unsupported digest value: {type(value)!r}")


def production_output_digest(result: dict[str, Any]) -> str:
    interface = {
        key: value
        for key, value in result.items()
        if key not in {
            "agent_logit_trace",
            "function_logit_trace",
            "nano_engine_benchmark",
            "request_id",
        }
    }
    digest = hashlib.sha256()
    _hash_value(digest, {"interface": interface})
    return digest.hexdigest()


def divergence_runs(expected: Sequence[int], actual: Sequence[int]) -> list[dict[str, Any]]:
    if len(expected) != len(actual):
        raise ValueError("expected and actual vectors differ in length")
    runs = []
    start = None
    for index, (wanted, observed) in enumerate(zip(expected, actual, strict=True)):
        differs = wanted != observed
        if differs and start is None:
            start = index
        at_end = index == len(expected) - 1
        if start is not None and (not differs or at_end):
            end = index if differs and at_end else index - 1
            runs.append(
                {
                    "start_call": start,
                    "end_call": end,
                    "length": end - start + 1,
                    "reconverged_call": None if end + 1 == len(expected) else end + 1,
                }
            )
            start = None
    return runs


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    return True


def block_invalid_reasons(
    block: dict[str, Any],
    *,
    expected_calls: int,
    require_selected_outputs: bool = True,
) -> list[str]:
    reasons = []
    if block.get("valid") is not True:
        reasons.append("recorded_invalid")
    if block.get("cleanup_ok") is not True:
        reasons.append("cleanup_failed")
    if block.get("calls") != expected_calls:
        reasons.append("call_count_mismatch")
    if block.get("engine_requests_before") != 0:
        reasons.append("requests_present_before_block")
    if block.get("engine_requests_after_cleanup") != 0:
        reasons.append("requests_present_after_cleanup")

    if require_selected_outputs:
        expected_selected_calls = {call for call in SELECTED_CALLS if call < expected_calls}
        selected = block.get("selected_outputs", [])
        observed_selected_calls = {item.get("call_index") for item in selected}
        if observed_selected_calls != expected_selected_calls:
            reasons.append("selected_output_coverage_mismatch")
        if not all(item.get("reference_match") is True for item in selected):
            reasons.append("selected_output_digest_mismatch")

    actual_tokens = block.get("actual_tokens")
    if actual_tokens is not None and block.get("actual_token_vector_sha256") != stable_digest(
        actual_tokens
    ):
        reasons.append("actual_token_vector_digest_mismatch")
    if not _all_numeric_values_finite(block.get("diagnostics", [])):
        reasons.append("non_finite_diagnostic")
    return reasons


def evaluate_bounded_block(block: dict[str, Any]) -> dict[str, Any]:
    mismatches = block.get("token_mismatches", [])
    runs = block.get("divergence_runs", [])
    complete = block.get("calls") == COMPLETE_PREFIX_CALLS
    checks = {
        "valid": block.get("valid") is True,
        "cleanup_ok": block.get("cleanup_ok") is True,
        "mismatches_at_most_one": len(mismatches) <= 1,
        "mismatch_rate_at_most_one_per_687": (
            len(mismatches) <= 1 if complete else None
        ),
        "first_mismatch_no_earlier_than_39": not mismatches
        or int(mismatches[0]["call_index"]) >= 39,
        "lexical_divergences_only": all(
            int(item["expected_token"]) > 12 and int(item["actual_token"]) > 12
            for item in mismatches
        ),
        "control_token_mismatches_zero": all(
            int(item["expected_token"]) not in CONTROL_TOKEN_IDS
            and int(item["actual_token"]) not in CONTROL_TOKEN_IDS
            for item in mismatches
        ),
        "max_divergence_run_at_most_one": all(int(run["length"]) <= 1 for run in runs),
        "all_divergences_reconverge": all(run["reconverged_call"] is not None for run in runs),
        "selected_outputs_match": all(
            item.get("reference_match") is True for item in block.get("selected_outputs", [])
        ),
    }
    applicable_checks = [value for value in checks.values() if value is not None]
    return {
        "applicable": complete,
        "complete_687_call_block": complete,
        **checks,
        "passed": all(applicable_checks) if complete else None,
    }


def _engine_request_count(model: Any) -> int | None:
    requests = getattr(getattr(model, "engine", None), "requests", None)
    return len(requests) if isinstance(requests, dict) else None


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

    if args.diagnostic_call is not None:
        os.environ["S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK"] = "20"
    return create_model(
        engine_type="vllm_llm",
        vllm_config={
            "model_path": str(args.nano_vllm_path.resolve()),
            "engine_path": str(args.nano_vllm_path.resolve()),
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": "bfloat16",
            "enforce_eager": bool(args.enforce_eager),
            "seed": int(args.sampling_seed),
        },
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=0.0,
        text_pad_id=12,
        agent_logit_trace_ids={"pad": 12, "bos": 1},
    )


def _call_kwargs(saved: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "current_step": int(saved.get("current_step", 0)),
        "decode_steps": int(saved.get("decode_steps", 1)),
    }
    if saved.get("prompt_token_ids") is not None:
        kwargs["prompt_token_ids"] = saved["prompt_token_ids"]
    generated = saved.get("generated_tokens")
    if generated is not None:
        kwargs["generated_tokens"] = generated.to("cuda")
    return kwargs


def replay_request(
    model: Any,
    records: Sequence[dict[str, Any]],
    root: Path,
    request_id: str,
    *,
    sync_each_call: bool,
    references: dict[str, str],
    diagnostic_call: int | None = None,
    retain_details: bool = True,
    validate_selected_outputs: bool = True,
) -> dict[str, Any]:
    torch = _torch()
    actual_tokens = []
    mismatches = []
    selected_outputs = []
    diagnostics = []
    cleanup_ok = False
    started = time.perf_counter()
    before_requests = _engine_request_count(model)
    try:
        for record in records:
            payload = torch.load(root / record["file"], map_location="cpu", weights_only=False)
            saved = payload["inputs"]
            call_index = int(record["call_index"])
            call_kwargs = _call_kwargs(saved)
            if call_index == diagnostic_call:
                call_kwargs["capture_agent_logit_trace"] = True
            result = model(
                saved["input_embeds"].to("cuda"),
                request_id=request_id,
                **call_kwargs,
            )
            if sync_each_call:
                torch.cuda.synchronize()
            actual_token = int(result["predicted_token"])
            expected_token = int(record["predicted_token"])
            actual_tokens.append(actual_token)
            if actual_token != expected_token:
                mismatches.append(
                    {
                        "actual_token": actual_token,
                        "call_index": call_index,
                        "expected_token": expected_token,
                    }
                )
            if call_index in SELECTED_CALLS:
                digest = production_output_digest(result)
                reference = references.get(str(call_index))
                selected_outputs.append(
                    {
                        "call_index": call_index,
                        "digest": digest,
                        "reference_digest": reference,
                        "reference_match": None if reference is None else digest == reference,
                    }
                )
            if call_index == diagnostic_call:
                trace = result.get("agent_logit_trace")
                if not isinstance(trace, dict):
                    raise RuntimeError("diagnostic call did not return an agent logit trace")
                by_token = {int(item["token_id"]): item for item in trace["top_k"]}
                wanted = by_token.get(10592)
                alternate = by_token.get(1044)
                diagnostics.append(
                    {
                        "call_index": call_index,
                        "head_argmax_id": trace["head_argmax_id"],
                        "head_argmax_logit": trace["head_argmax_logit"],
                        "retained_token": wanted,
                        "alternate_token": alternate,
                        "retained_minus_alternate_logit": (
                            None
                            if wanted is None or alternate is None
                            else wanted["logit"] - alternate["logit"]
                        ),
                        "top_k": trace["top_k"],
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
    expected_tokens = [int(record["predicted_token"]) for record in records]
    runs = divergence_runs(expected_tokens, actual_tokens)
    block = {
        "actual_token_vector_sha256": stable_digest(actual_tokens),
        "calls": len(actual_tokens),
        "cleanup_ok": cleanup_ok,
        "divergence_runs": runs,
        "diagnostics": diagnostics,
        "engine_requests_after_cleanup": _engine_request_count(model),
        "engine_requests_before": before_requests,
        "expected_token_vector_sha256": stable_digest(expected_tokens),
        "request_id": request_id,
        "selected_outputs": selected_outputs,
        "token_mismatches": mismatches,
        "valid": cleanup_ok and len(actual_tokens) == len(records),
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }
    if retain_details:
        block["actual_tokens"] = actual_tokens
    invalid_reasons = block_invalid_reasons(
        block,
        expected_calls=len(records),
        require_selected_outputs=validate_selected_outputs,
    )
    block["invalid_reasons"] = invalid_reasons
    block["valid"] = not invalid_reasons
    block["bounded_gate"] = evaluate_bounded_block(block)
    return block


def run_engine(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    random.seed(args.parent_seed)
    torch.manual_seed(args.parent_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.parent_seed)
    root = args.capture_root.resolve()
    records = read_manifest(root, args.max_call_index)
    references = {}
    if args.reference_outputs:
        source = json.loads(args.reference_outputs.read_text(encoding="utf-8"))
        references = {str(key): str(value) for key, value in source.items()}
    model = _load_model(args)
    prelude = None
    preludes = []
    blocks = []
    try:
        if args.sequence == "history":
            history_plan = (("baseline", 0, 1), ("post-47", 47, 3), ("post-17", 17, 3))
            for history, prelude_calls, count in history_plan:
                if prelude_calls:
                    item = replay_request(
                        model,
                        records[:prelude_calls],
                        root,
                        f"{args.engine_label}-{history}-prelude",
                        sync_each_call=False,
                        references={},
                        diagnostic_call=args.diagnostic_call,
                        retain_details=False,
                        validate_selected_outputs=False,
                    )
                    item["history"] = history
                    preludes.append(item)
                for local_index in range(count):
                    block = replay_request(
                        model,
                        records,
                        root,
                        f"{args.engine_label}-{history}-{local_index:03d}",
                        sync_each_call=False,
                        references=references,
                        diagnostic_call=args.diagnostic_call,
                    )
                    block.update(
                        {
                            "block": len(blocks),
                            "history": history,
                            "request_policy": "unique",
                            "sync_each_call": False,
                        }
                    )
                    blocks.append(block)
        elif args.sequence == "scheduling":
            schedule_plan = (
                ("unique", False, False),
                ("reused", True, False),
                ("synchronized", False, True),
            )
            for policy, reuse, synchronize in schedule_plan:
                for local_index in range(3):
                    request_id = (
                        f"{args.engine_label}-reused"
                        if reuse
                        else f"{args.engine_label}-{policy}-{local_index:03d}"
                    )
                    block = replay_request(
                        model,
                        records,
                        root,
                        request_id,
                        sync_each_call=synchronize,
                        references=references,
                        diagnostic_call=args.diagnostic_call,
                    )
                    block.update(
                        {
                            "block": len(blocks),
                            "history": "baseline",
                            "request_policy": policy,
                            "sync_each_call": synchronize,
                        }
                    )
                    blocks.append(block)
        elif args.prelude_calls:
            prelude = replay_request(
                model,
                records[: args.prelude_calls],
                root,
                f"{args.engine_label}-prelude-{args.prelude_calls}",
                sync_each_call=args.sync_each_call,
                references={},
                diagnostic_call=args.diagnostic_call,
                retain_details=False,
                validate_selected_outputs=False,
            )
        if args.sequence == "standard":
            for block_index in range(args.blocks):
                request_id = (
                    f"{args.engine_label}-reused"
                    if args.reuse_request_id
                    else f"{args.engine_label}-block-{block_index:03d}"
                )
                block = replay_request(
                    model,
                    records,
                    root,
                    request_id,
                    sync_each_call=args.sync_each_call,
                    references=references,
                    diagnostic_call=args.diagnostic_call,
                )
                block.update(
                    {
                        "block": block_index,
                        "history": (
                            f"post-{args.prelude_calls}"
                            if args.prelude_calls
                            else "baseline"
                        ),
                        "request_policy": "reused" if args.reuse_request_id else "unique",
                        "sync_each_call": args.sync_each_call,
                    }
                )
                blocks.append(block)
    finally:
        model.shutdown()
    vector_digests = sorted({block["actual_token_vector_sha256"] for block in blocks})
    expected_blocks = {"standard": args.blocks, "history": 7, "scheduling": 9}[args.sequence]
    return {
        "blocks": blocks,
        "capture": {
            "manifest_sha256": sha256_file(root / "manifest.jsonl"),
            "max_call_index": args.max_call_index,
            "root": str(root),
        },
        "component": {
            "config_sha256": sha256_file(args.nano_vllm_path / "config.json"),
            "enforce_eager": args.enforce_eager,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "path": str(args.nano_vllm_path.resolve()),
        },
        "distinct_token_vectors": len(vector_digests),
        "engine_label": args.engine_label,
        "hypothesis_controls": {
            "parent_seed": args.parent_seed,
            "prelude_calls": args.prelude_calls,
            "reuse_request_id": args.reuse_request_id,
            "sampling_seed": args.sampling_seed,
            "sequence": args.sequence,
            "sync_each_call": args.sync_each_call,
        },
        "kind": "step5_graph_reproducibility_engine_run",
        "passed": len(blocks) == expected_blocks and all(block["valid"] for block in blocks),
        "prelude": prelude,
        "preludes": preludes,
        "schema": 1,
        "token_vector_sha256": vector_digests,
    }


def summarize_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for report in reports:
        controls = report["hypothesis_controls"]
        identity = (
            report["component"]["enforce_eager"],
            controls["sampling_seed"],
            controls["prelude_calls"],
            controls["reuse_request_id"],
            controls["sync_each_call"],
            report["capture"]["max_call_index"],
        )
        cells.setdefault(identity, []).append(report)
    summaries = []
    for identity, cell_reports in sorted(cells.items(), key=lambda item: repr(item[0])):
        blocks = [block for report in cell_reports for block in report["blocks"]]
        vector_digests = sorted({block["actual_token_vector_sha256"] for block in blocks})
        exact_blocks = [block for block in blocks if not block["token_mismatches"]]
        mismatch_distribution: dict[str, int] = {}
        for block in blocks:
            first = block["token_mismatches"][0] if block["token_mismatches"] else None
            key = "exact" if first is None else (
                f"call={first['call_index']};expected={first['expected_token']};"
                f"actual={first['actual_token']}"
            )
            mismatch_distribution[key] = mismatch_distribution.get(key, 0) + 1
        summaries.append(
            {
                "blocks": len(blocks),
                "distinct_token_vectors": len(vector_digests),
                "engines": len(cell_reports),
                "exact_blocks": len(exact_blocks),
                "exact_prefix_rate": len(exact_blocks) / len(blocks) if blocks else None,
                "identity": {
                    "enforce_eager": identity[0],
                    "sampling_seed": identity[1],
                    "prelude_calls": identity[2],
                    "reuse_request_id": identity[3],
                    "sync_each_call": identity[4],
                    "max_call_index": identity[5],
                },
                "mismatch_distribution": mismatch_distribution,
                "token_vector_sha256": vector_digests,
                "valid_blocks": sum(block["valid"] is True for block in blocks),
                "within_engine_stable": all(
                    report["distinct_token_vectors"] == 1 for report in cell_reports
                ),
            }
        )
    return {
        "cells": summaries,
        "engine_runs": len(reports),
        "kind": "step5_graph_reproducibility_summary",
        "passed": bool(reports) and all(report.get("passed") is True for report in reports),
        "schema": 1,
    }


def _report_identity(report: dict[str, Any]) -> tuple[Any, ...]:
    component = report.get("component", {})
    capture = report.get("capture", {})
    controls = report.get("hypothesis_controls", {})
    return (
        component.get("config_sha256"),
        component.get("gpu_memory_utilization"),
        component.get("max_model_len"),
        component.get("path"),
        capture.get("manifest_sha256"),
        capture.get("root"),
        controls.get("parent_seed"),
    )


def _expected_arm_values(group_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "prelude_calls": 0,
        "reuse_request_id": False,
        "sync_each_call": False,
    }
    if group_name in {"primary_graph_seed0", "graph_history", "graph_scheduling"}:
        values.update(enforce_eager=False, sampling_seed=0, max_call_index=46)
    elif group_name == "primary_eager_seed0":
        values.update(enforce_eager=True, sampling_seed=0, max_call_index=46)
    elif group_name == "graph_seed1":
        values.update(enforce_eager=False, sampling_seed=1, max_call_index=46)
    elif group_name == "diagnostic_graph":
        values.update(enforce_eager=False, sampling_seed=0, max_call_index=46)
    elif group_name == "diagnostic_eager":
        values.update(enforce_eager=True, sampling_seed=0, max_call_index=46)
    elif group_name == "confirmation_graph":
        values.update(enforce_eager=False, sampling_seed=0, max_call_index=686)
    elif group_name == "confirmation_eager":
        values.update(enforce_eager=True, sampling_seed=0, max_call_index=686)
    if group_name == "graph_history":
        values["sequence"] = "history"
    elif group_name == "graph_scheduling":
        values["sequence"] = "scheduling"
    return values


def _report_invalid_reasons(
    report: dict[str, Any],
    *,
    group_name: str,
    identity_matches: bool,
) -> list[str]:
    reasons = []
    expected = _expected_arm_values(group_name)
    controls = report.get("hypothesis_controls", {})
    component = report.get("component", {})
    capture = report.get("capture", {})
    if report.get("passed") is not True:
        reasons.append("recorded_engine_failure")
    if not identity_matches:
        reasons.append("artifact_capture_or_parent_control_identity_mismatch")
    for key, value in expected.items():
        if key == "enforce_eager":
            source = component
        elif key == "max_call_index":
            source = capture
        else:
            source = controls
        if source.get(key) != value:
            reasons.append(f"unexpected_{key}")

    expected_calls = int(capture.get("max_call_index", -1)) + 1
    if group_name == "graph_history":
        expected_blocks = 7
    elif group_name == "graph_scheduling":
        expected_blocks = 9
    else:
        expected_blocks = 3
    blocks = report.get("blocks", [])
    if len(blocks) != expected_blocks:
        reasons.append("block_count_mismatch")
    for index, block in enumerate(blocks):
        reasons.extend(
            f"block_{index}:{reason}"
            for reason in block_invalid_reasons(block, expected_calls=expected_calls)
        )
    preludes = [item for item in [report.get("prelude")] if item is not None]
    preludes.extend(report.get("preludes", []))
    for index, prelude in enumerate(preludes):
        prelude_calls = int(prelude.get("calls", -1))
        reasons.extend(
            f"prelude_{index}:{reason}"
            for reason in block_invalid_reasons(
                prelude,
                expected_calls=prelude_calls,
                require_selected_outputs=False,
            )
        )
    return reasons


def _group_summary(
    reports: Sequence[dict[str, Any]],
    invalid_reasons: Sequence[list[str]] | None = None,
) -> dict[str, Any]:
    invalid_reasons = invalid_reasons or [[] for _ in reports]
    valid_reports = [
        report
        for report, reasons in zip(reports, invalid_reasons, strict=True)
        if not reasons
    ]
    blocks = [block for report in valid_reports for block in report["blocks"]]
    selected = [item for block in blocks for item in block.get("selected_outputs", [])]
    mismatch_locations: dict[str, int] = {}
    for block in blocks:
        for item in block["token_mismatches"]:
            key = (
                f"call={item['call_index']};expected={item['expected_token']};"
                f"actual={item['actual_token']}"
            )
            mismatch_locations[key] = mismatch_locations.get(key, 0) + 1
    return {
        "blocks": sum(len(report.get("blocks", [])) for report in reports),
        "distinct_token_vectors": len(
            {block["actual_token_vector_sha256"] for block in blocks}
        ),
        "engines": len(reports),
        "exact_blocks": sum(not block["token_mismatches"] for block in blocks),
        "mismatch_locations": mismatch_locations,
        "selected_output_digests_compared": sum(
            item.get("reference_match") is not None for item in selected
        ),
        "selected_output_digests_matching": sum(
            item.get("reference_match") is True for item in selected
        ),
        "valid_blocks": sum(
            len(report.get("blocks", [])) for report in valid_reports
        ),
        "valid_engines": sum(not reasons for reasons in invalid_reasons),
        "invalid_engine_reasons": {
            report.get("engine_label", f"engine-{index}"): reasons
            for index, (report, reasons) in enumerate(zip(reports, invalid_reasons, strict=True))
            if reasons
        },
        "within_engine_stable": all(
            report.get("distinct_token_vectors") == 1 for report in valid_reports
        ),
    }


def _outcome_vectors(reports: Sequence[dict[str, Any]]) -> set[str]:
    return {
        block["actual_token_vector_sha256"]
        for report in reports
        for block in report.get("blocks", [])
    }


def _within_engine_stratum_association(
    reports: Sequence[dict[str, Any]],
    stratum,
) -> tuple[bool | None, list[str]]:
    tested_strata: set[str] = set()
    compared = False
    associated = False
    for report in reports:
        strata: dict[str, set[str]] = {}
        for block in report.get("blocks", []):
            key = stratum(block)
            if key is None:
                continue
            tested_strata.add(key)
            strata.setdefault(key, set()).add(block["actual_token_vector_sha256"])
        if len(strata) >= 2:
            compared = True
            associated = associated or len({frozenset(value) for value in strata.values()}) > 1
    return (associated if compared else None), sorted(tested_strata)


def _selector_conclusions(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seed0 = _outcome_vectors(groups["primary_graph_seed0"])
    seed1 = _outcome_vectors(groups["graph_seed1"])
    seed_association = seed0 != seed1 if seed0 and seed1 else None

    history_association, histories = _within_engine_stratum_association(
        groups["graph_history"], lambda block: block.get("history")
    )
    history_basins = sorted(
        {
            "exact" if not block.get("token_mismatches") else "divergent"
            for report in groups["graph_history"]
            for block in report.get("blocks", [])
            if block.get("history") == "baseline"
        }
    )
    request_id_association, request_policies = _within_engine_stratum_association(
        groups["graph_scheduling"],
        lambda block: (
            block.get("request_policy")
            if block.get("request_policy") in {"unique", "reused"}
            else None
        ),
    )
    synchronization_association, synchronization_policies = (
        _within_engine_stratum_association(
            groups["graph_scheduling"],
            lambda block: (
                "synchronized"
                if block.get("sync_each_call") is True
                else "unsynchronized_unique"
                if block.get("request_policy") == "unique"
                else None
            ),
        )
    )
    scheduling_basins = sorted(
        {
            "exact" if not report["blocks"][0].get("token_mismatches") else "divergent"
            for report in groups["graph_scheduling"]
            if report.get("blocks")
        }
    )
    return {
        "sampler_seed": {
            "conclusion": (
                "incomplete"
                if seed_association is None
                else "observed_association_not_isolated"
                if seed_association
                else "no_observed_association_not_isolated"
            ),
            "isolated": False,
            "observed_association": seed_association,
            "seed0_vectors": sorted(seed0),
            "seed1_vectors": sorted(seed1),
            "selects_result": None,
        },
        "request_history": {
            "conclusion": (
                "incomplete"
                if history_association is None
                else "observed_within_engine_association"
                if history_association
                else "not_observed_in_tested_basins"
            ),
            "histories": histories,
            "selects_result": history_association,
            "tested_basins": history_basins,
        },
        "request_id_reuse": {
            "conclusion": (
                "incomplete"
                if request_id_association is None
                else "observed_within_engine_association"
                if request_id_association
                else "not_observed_in_tested_basins"
            ),
            "policies": request_policies,
            "selects_result": request_id_association,
            "tested_basins": scheduling_basins,
        },
        "per_call_synchronization": {
            "conclusion": (
                "incomplete"
                if synchronization_association is None
                else "observed_within_engine_association"
                if synchronization_association
                else "not_observed_in_tested_basins"
            ),
            "policies": synchronization_policies,
            "selects_result": synchronization_association,
            "tested_basins": scheduling_basins,
        },
    }


def evaluate_global_bounded_gate(
    graph_reports: Sequence[dict[str, Any]],
    diagnostic_reports: Sequence[dict[str, Any]],
    *,
    production_config_sha256: str | None,
) -> dict[str, Any]:
    full_prefix_reports = [
        report
        for report in graph_reports
        if report.get("capture", {}).get("max_call_index") == COMPLETE_PREFIX_CALLS - 1
    ]
    measured_blocks = [
        block
        for report in full_prefix_reports
        for block in sorted(report.get("blocks", []), key=lambda item: item.get("block", -1))[1:]
    ]
    block_results = []
    for block in measured_blocks:
        result = evaluate_bounded_block(block)
        result["invalid_reasons"] = block_invalid_reasons(
            block, expected_calls=COMPLETE_PREFIX_CALLS
        )
        result["passed"] = result["passed"] is True and not result["invalid_reasons"]
        block_results.append(result)
    full_prefix_identity_valid = (
        bool(full_prefix_reports)
        and len({_report_identity(report) for report in full_prefix_reports}) == 1
        and all(
            report.get("component", {}).get("config_sha256")
            == production_config_sha256
            and report.get("component", {}).get("enforce_eager") is False
            for report in full_prefix_reports
        )
    )
    production_margins = [
        float(diagnostic["retained_minus_alternate_logit"])
        for report in [*graph_reports, *diagnostic_reports]
        if report.get("component", {}).get("config_sha256") == production_config_sha256
        for block in report.get("blocks", [])
        for diagnostic in block.get("diagnostics", [])
        if diagnostic.get("retained_minus_alternate_logit") is not None
        and _all_numeric_values_finite(diagnostic)
    ]
    overlay_margins = [
        float(diagnostic["retained_minus_alternate_logit"])
        for report in diagnostic_reports
        if report.get("component", {}).get("config_sha256") != production_config_sha256
        for block in report.get("blocks", [])
        for diagnostic in block.get("diagnostics", [])
        if diagnostic.get("retained_minus_alternate_logit") is not None
        and _all_numeric_values_finite(diagnostic)
    ]
    production_margin_status = (
        "unresolved"
        if not production_margins
        else "pass"
        if max(abs(margin) for margin in production_margins) <= BOUNDED_GATE_MAX_MARGIN
        else "fail"
    )
    requirements = {
        "all_engine_identities_match": full_prefix_identity_valid,
        "all_measured_blocks_pass": bool(block_results)
        and all(result["passed"] is True for result in block_results),
        "fresh_graph_engines_at_least_five": len(full_prefix_reports)
        >= BOUNDED_GATE_FRESH_ENGINES,
        "measured_blocks_at_least_30": len(measured_blocks)
        >= BOUNDED_GATE_MEASURED_BLOCKS,
        "production_margin_at_most_0_05": production_margin_status == "pass",
    }
    return {
        "fresh_graph_engines": len(full_prefix_reports),
        "measured_687_call_blocks": len(measured_blocks),
        "overlay_margin_values_supportive_only": sorted(set(overlay_margins)),
        "production_margin_status": production_margin_status,
        "production_margin_values": sorted(set(production_margins)),
        "requirements": requirements,
        "technical_requirements_passed": all(requirements.values()),
    }


def build_investigation_summary(reports_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reports_by_name = {
        name: report
        for name, report in reports_by_name.items()
        if "superseded" not in name
        and report.get("kind") == "step5_graph_reproducibility_engine_run"
    }

    def select(*prefixes: str) -> list[dict[str, Any]]:
        return [
            report
            for name, report in sorted(reports_by_name.items())
            if name.startswith(prefixes)
        ]

    groups = {
        "primary_graph_seed0": select("g0-"),
        "primary_eager_seed0": select("e0-"),
        "graph_seed1": select("g1-"),
        "graph_history": select("history-"),
        "graph_scheduling": select("schedule-"),
        "diagnostic_graph": select("diag-g-"),
        "diagnostic_eager": select("diag-e-"),
        "confirmation_graph": select("confirm-graph-"),
        "confirmation_eager": select("confirm-eager-"),
    }
    production_identity = {
        _report_identity(report)
        for name, reports in groups.items()
        if not name.startswith("diagnostic_")
        for report in reports
    }
    diagnostic_identity = {
        _report_identity(report)
        for name, reports in groups.items()
        if name.startswith("diagnostic_")
        for report in reports
    }
    invalid_reasons = {
        name: [
            _report_invalid_reasons(
                report,
                group_name=name,
                identity_matches=(
                    len(diagnostic_identity) == 1
                    if name.startswith("diagnostic_")
                    else len(production_identity) == 1
                ),
            )
            for report in reports
        ]
        for name, reports in groups.items()
    }
    valid_groups = {
        name: [
            report
            for report, reasons in zip(reports, invalid_reasons[name], strict=True)
            if not reasons
        ]
        for name, reports in groups.items()
    }
    group_summaries = {
        name: _group_summary(reports, invalid_reasons[name])
        for name, reports in groups.items()
    }
    production_graph = [
        *groups["primary_graph_seed0"],
        *groups["graph_seed1"],
        *groups["graph_history"],
        *groups["graph_scheduling"],
        *groups["confirmation_graph"],
    ]
    production_eager = [*groups["primary_eager_seed0"], *groups["confirmation_eager"]]
    production_graph_reasons = [
        *invalid_reasons["primary_graph_seed0"],
        *invalid_reasons["graph_seed1"],
        *invalid_reasons["graph_history"],
        *invalid_reasons["graph_scheduling"],
        *invalid_reasons["confirmation_graph"],
    ]
    production_eager_reasons = [
        *invalid_reasons["primary_eager_seed0"],
        *invalid_reasons["confirmation_eager"],
    ]
    production = {
        "graph": _group_summary(production_graph, production_graph_reasons),
        "eager": _group_summary(production_eager, production_eager_reasons),
    }
    diagnostics = [
        diagnostic
        for report in valid_groups["diagnostic_graph"] + valid_groups["diagnostic_eager"]
        for block in report["blocks"]
        for diagnostic in block.get("diagnostics", [])
    ]
    diagnostic_margins = sorted(
        {
            diagnostic["retained_minus_alternate_logit"]
            for diagnostic in diagnostics
            if diagnostic.get("retained_minus_alternate_logit") is not None
        }
    )
    primary_reports_valid = (
        group_summaries["primary_graph_seed0"]["valid_engines"]
        == group_summaries["primary_graph_seed0"]["engines"]
        and group_summaries["primary_eager_seed0"]["valid_engines"]
        == group_summaries["primary_eager_seed0"]["engines"]
    )
    graph_eager_discriminator = (
        primary_reports_valid
        and group_summaries["primary_graph_seed0"]["distinct_token_vectors"] > 1
        and group_summaries["primary_eager_seed0"]["distinct_token_vectors"] == 1
        and group_summaries["primary_eager_seed0"]["exact_blocks"]
        == group_summaries["primary_eager_seed0"]["blocks"]
    )
    selectors = _selector_conclusions(valid_groups)
    all_reports_valid = all(
        not reasons for group in invalid_reasons.values() for reasons in group
    )
    production_config_sha256 = next(
        (
            report.get("component", {}).get("config_sha256")
            for report in groups["primary_graph_seed0"]
        ),
        None,
    )
    bounded_gate = evaluate_global_bounded_gate(
        [
            *valid_groups["primary_graph_seed0"],
            *valid_groups["graph_seed1"],
            *valid_groups["graph_history"],
            *valid_groups["graph_scheduling"],
            *valid_groups["confirmation_graph"],
        ],
        [*valid_groups["diagnostic_graph"], *valid_groups["diagnostic_eager"]],
        production_config_sha256=production_config_sha256,
    )
    real_state_defect = graph_eager_discriminator
    classification = "REAL STATE DEFECT" if real_state_defect else "INCOMPLETE"
    bounded_gate.update(
        {
            "applicable": not real_state_defect,
            "exclusion_reason": (
                "REAL STATE DEFECT requires exact-gate repair"
                if real_state_defect
                else None
            ),
            "passed": bounded_gate["technical_requirements_passed"]
            and not real_state_defect,
            "status": (
                "excluded_real_state_defect"
                if real_state_defect
                else "pass"
                if bounded_gate["technical_requirements_passed"]
                else "incomplete"
            ),
        }
    )
    return {
        "bounded_gate": bounded_gate,
        "classification": classification,
        "classification_basis": {
            "all_reports_valid": all_reports_valid,
            "graph_eager_discriminator": graph_eager_discriminator,
            "graph_within_engine_stable": production["graph"]["within_engine_stable"],
            "eager_all_exact": production["eager"]["exact_blocks"]
            == production["eager"]["blocks"],
            "sampler_seed_selects_result": selectors["sampler_seed"]["selects_result"],
            "request_history_selects_result": selectors["request_history"]["selects_result"],
            "request_id_or_synchronization_selects_result": (
                selectors["request_id_reuse"]["selects_result"]
                or selectors["per_call_synchronization"]["selects_result"]
            ),
        },
        "diagnostic_retained_minus_alternate_logit_values": diagnostic_margins,
        "groups": group_summaries,
        "kind": "step5_graph_reproducibility_investigation_summary",
        "passed": classification != "INCOMPLETE" and all_reports_valid,
        "production": production,
        "schema": 1,
        "selectors": selectors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--speech-root", type=Path, required=True)
    run.add_argument("--nano-vllm-path", type=Path, required=True)
    run.add_argument("--capture-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--engine-label", required=True)
    run.add_argument("--reference-outputs", type=Path)
    run.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--sampling-seed", type=int, default=0)
    run.add_argument("--parent-seed", type=int, default=20260813)
    run.add_argument("--blocks", type=int, default=3)
    run.add_argument("--max-call-index", type=int, default=46)
    run.add_argument("--prelude-calls", type=int, default=0)
    run.add_argument("--reuse-request-id", action="store_true")
    run.add_argument("--sync-each-call", action="store_true")
    run.add_argument("--diagnostic-call", type=int)
    run.add_argument(
        "--sequence", choices=("standard", "history", "scheduling"), default="standard"
    )
    run.add_argument("--max-model-len", type=int, default=12288)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.42)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", type=Path, action="append", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.blocks <= 0:
            parser.error("--blocks must be positive")
        if args.max_call_index < 0:
            parser.error("--max-call-index must be nonnegative")
        if not 0 <= args.prelude_calls <= args.max_call_index + 1:
            parser.error("--prelude-calls must be within the replay prefix")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        payload = summarize_reports(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
        )
        atomic_json(args.output, payload)
    elif args.command == "aggregate":
        payload = build_investigation_summary(
            {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(args.run_dir.glob("*.json"))
            }
        )
        atomic_json(args.output, payload)
    else:
        try:
            payload = run_engine(args)
            atomic_json(args.output, payload)
        except BaseException as exc:
            failure = {
                "engine_label": args.engine_label,
                "error": f"{type(exc).__name__}: {exc}",
                "kind": "step5_graph_reproducibility_engine_failure",
                "passed": False,
                "schema": 1,
                "traceback": traceback.format_exc(),
            }
            atomic_json(args.output, failure)
            raise
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
