#!/usr/bin/env python3
"""Bandwidth attribution for the Nano graph envelope (preregistered side-study).

Subcommands:
  bytes-model     Host-only: per-decode-step streamed-bytes model from
                  safetensors headers and config.json, by declared family.
  ceiling         GPU: measured achievable device-memory bandwidth
                  (read-reduction and D2D copy at multiple sizes).
  analyze-kernels Attribute node-level Nsight kernel rows inside the Step 4d
                  NVTX interface ranges to families; report achieved GB/s
                  against the measured ceiling. Fail-closed: unmatched kernel
                  time is reported unattributed, never absorbed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import struct
import time
from pathlib import Path
from typing import Any

NVTX_RANGE = re.compile(
    r"^step4d:block=(?P<block>\d+);pair=(?P<pair>\d+);"
    r"class=(?P<class>content|pad);call=(?P<call>\d+);step=(?P<step>\d+)$"
)

SAFETENSORS_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

# Declared tensor-name -> family classification. Order matters; first match
# wins. Every tensor must match exactly one family or is reported
# unclassified.
TENSOR_FAMILIES: tuple[tuple[str, str, bool], ...] = (
    # (family, regex, streamed-per-step)
    ("embed_gather", r"embed_tokens|embed_asr_tokens", False),
    ("function_head_f32", r"function_head", True),
    ("lm_head_gptq", r"lm_head\.(qweight|scales|qzeros|g_idx)", True),
    ("attention_f32", r"\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.", True),
    (
        "mamba_proj_gptq",
        r"\.mixer\.(in_proj|up_proj|down_proj|out_proj)\.(qweight|scales|qzeros|g_idx)",
        True,
    ),
    ("mamba_small", r"\.mixer\.(A_log|D|dt_bias|conv1d|norm)", True),
    ("norms_misc", r"norm|\.bias$", True),
)

# Declared kernel-name -> family mapping used by analyze-kernels. First match
# wins; unmatched kernels stay unattributed. The mapping is intentionally
# conservative: only names whose family is unambiguous from the kernel type
# and problem shape are claimed.
KERNEL_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (
        "function_head_or_attention_f32_gemm",
        r"sgemm|gemv2N|gemvNSP|s16816gemm.*f32|cutlass.*f32|ampere_sgemm",
    ),
    ("gptq_dequant_gemv", r"gptq|marlin|dequant|awq|VecQuant"),
    ("unquantized_gemv_bf16", r"gemvx|cublas|gemv2N|gemvNSP"),
    ("mamba_scan", r"selective|mamba|ssm|scan|chunk_state|state_passing|bmm_chunk"),
    ("causal_conv", r"causal_conv|conv1d"),
    ("attention_core", r"triton.*attn|flash|paged_attn|attn_fwd|reshape_and_cache"),
    (
        "norm_elementwise",
        r"rms_norm|layer_norm|elementwise|vectorized_elementwise|act_and_mul|silu|softmax",
    ),
    ("gather_embed", r"gather|index_select|embedding"),
    ("sampling_argmax", r"argmax|topk|radix_sort|reduce_kernel|\bcub::"),
    ("memops", r"memcpy|memset|copy_device"),
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


def read_safetensors_headers(root: Path) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with shard.open("rb") as stream:
            (header_len,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor {name}")
            tensors[name] = {"shape": meta["shape"], "dtype": meta["dtype"], "shard": shard.name}
    if not tensors:
        raise ValueError(f"no safetensors tensors found under {root}")
    return tensors


def tensor_bytes(meta: dict[str, Any]) -> int:
    per = SAFETENSORS_BYTES.get(meta["dtype"])
    if per is None:
        raise ValueError(f"unknown safetensors dtype {meta['dtype']}")
    count = 1
    for dim in meta["shape"]:
        count *= dim
    return count * per


def classify_tensor(name: str) -> tuple[str, bool] | None:
    for family, pattern, streamed in TENSOR_FAMILIES:
        if re.search(pattern, name):
            return family, streamed
    return None


def bytes_model(args: argparse.Namespace) -> dict[str, Any]:
    root = args.model_root.resolve()
    tensors = read_safetensors_headers(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    families: dict[str, dict[str, Any]] = {}
    unclassified: list[str] = []
    for name, meta in tensors.items():
        size = tensor_bytes(meta)
        hit = classify_tensor(name)
        if hit is None:
            unclassified.append(name)
            continue
        family, streamed = hit
        entry = families.setdefault(
            family, {"bytes": 0, "tensors": 0, "streamed_per_step": streamed, "dtypes": set()}
        )
        entry["bytes"] += size
        entry["tensors"] += 1
        entry["dtypes"].add(meta["dtype"])
    for entry in families.values():
        entry["dtypes"] = sorted(entry["dtypes"])

    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    pattern = str(config.get("hybrid_override_pattern", ""))
    attention_layers = pattern.count("*")
    mamba_layers = pattern.count("M")
    heads = int(config.get("mamba_num_heads", 0))
    head_dim = int(config.get("mamba_head_dim", 0) or 0)
    if head_dim == 0 and heads:
        d_inner_cfg = config.get("mamba_d_inner")
        head_dim = int(d_inner_cfg // heads) if d_inner_cfg else 0
    state = int(config.get("ssm_state_size", 0))
    groups = int(config.get("n_groups", 0))
    conv_kernel = int(config.get("conv_kernel", config.get("d_conv", 4)) or 4)
    kv_heads = int(config.get("num_key_value_heads", 0))
    attn_heads = max(1, int(config.get("num_attention_heads", 1)))
    attn_head_dim = int(config.get("head_dim", 0) or (hidden // attn_heads))
    seq_len = int(args.sequence_length)

    d_inner = heads * head_dim if head_dim else None
    mamba_state_bytes = None
    conv_state_bytes = None
    if d_inner and state:
        # SSM state per layer: heads x head_dim x state, bf16; read + write per step.
        mamba_state_bytes = mamba_layers * heads * head_dim * state * 2 * 2
        conv_dim = d_inner + 2 * groups * state
        conv_state_bytes = mamba_layers * conv_dim * conv_kernel * 2 * 2
    kv_read_bytes = 2 * attention_layers * kv_heads * attn_head_dim * seq_len * 2

    streamed_weight_bytes = sum(e["bytes"] for f, e in families.items() if e["streamed_per_step"])
    dynamic = {
        "mamba_ssm_state_read_write": mamba_state_bytes,
        "mamba_conv_state_read_write": conv_state_bytes,
        "attention_kv_read_at_seq_len": kv_read_bytes,
        "assumed_sequence_length": seq_len,
    }
    dynamic_total = sum(v for v in (mamba_state_bytes, conv_state_bytes, kv_read_bytes) if v)
    payload = {
        "kind": "bandwidth_attribution_bytes_model",
        "model_root": str(root),
        "config_sha256": sha256_file(root / "config.json"),
        "families": families,
        "unclassified_tensors": sorted(unclassified),
        "derived_dims": {
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "attention_layers": attention_layers,
            "mamba_layers": mamba_layers,
            "mamba_num_heads": heads,
            "mamba_head_dim": head_dim,
            "ssm_state_size": state,
            "n_groups": groups,
            "kv_heads": kv_heads,
            "attn_head_dim": attn_head_dim,
        },
        "per_step_streamed_weight_bytes": streamed_weight_bytes,
        "per_step_dynamic_bytes": dynamic,
        "per_step_total_modeled_bytes": streamed_weight_bytes + dynamic_total,
        "assumptions": [
            "streamed families are read exactly once per decode step",
            "embed_tokens/embed_asr_tokens are a row gather, excluded from streaming",
            "activation traffic is unmodeled and disclosed",
            "dynamic state bytes counted read+write at bf16",
        ],
        "passed": not unclassified,
    }
    return payload


def _timed_gbps(fn, bytes_moved: int, repeats: int, warmup: int, sync) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        samples.append(bytes_moved / (time.perf_counter() - start) / 1e9)
    return {
        "median_gbps": statistics.median(samples),
        "mean_gbps": statistics.fmean(samples),
        "min_gbps": min(samples),
        "max_gbps": max(samples),
        "samples": samples,
    }


def ceiling(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ceiling requires CUDA")
    torch.cuda.init()
    device = torch.device("cuda:0")
    results: dict[str, Any] = {}
    sizes_gb = [float(s) for s in args.sizes_gb.split(",")]
    for size_gb in sizes_gb:
        elements = int(size_gb * 1e9 // 2)
        source = torch.ones(elements, dtype=torch.bfloat16, device=device)
        read_bytes = elements * 2
        results[f"read_reduction_{size_gb:g}GB"] = _timed_gbps(
            lambda s=source: s.sum(), read_bytes, args.repeats, args.warmup, torch.cuda.synchronize
        )
        half = elements // 2
        dst = torch.empty(half, dtype=torch.bfloat16, device=device)
        copy_bytes = half * 2 * 2  # read + write
        results[f"d2d_copy_{size_gb:g}GB"] = _timed_gbps(
            lambda d=dst, s=source: d.copy_(s[:half]),
            copy_bytes,
            args.repeats,
            args.warmup,
            torch.cuda.synchronize,
        )
        del source, dst
        torch.cuda.empty_cache()
    read_medians = [v["median_gbps"] for k, v in results.items() if k.startswith("read_reduction")]
    payload = {
        "kind": "bandwidth_attribution_ceiling",
        "device_name": torch.cuda.get_device_name(0),
        "results": results,
        "study_ceiling_gbps": max(read_medians),
        "spec_context_gbps": 273.0,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "passed": True,
    }
    return payload


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    return dict(connection.execute("SELECT id, value FROM StringIds"))


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _nvtx_interface_ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    tables = _tables(connection)
    if "NVTX_EVENTS" not in tables:
        raise ValueError("no NVTX_EVENTS table")
    strings = _strings(connection)
    ranges = []
    for text_id, text, start, end in connection.execute(
        "SELECT textId, text, start, end FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        label = text if text else strings.get(text_id, "")
        match = NVTX_RANGE.match(label or "")
        if match:
            ranges.append(
                {
                    "label": label,
                    "start": int(start),
                    "end": int(end),
                    **{k: match.group(k) for k in ("block", "pair", "class", "call", "step")},
                }
            )
    if not ranges:
        raise ValueError("no step4d NVTX interface ranges found")
    return ranges


def _kernel_rows(connection: sqlite3.Connection) -> list[tuple[int, int, str]]:
    tables = _tables(connection)
    strings = _strings(connection)
    rows: list[tuple[int, int, str]] = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        if table not in tables:
            continue
        columns = {r[1] for r in connection.execute(f"PRAGMA table_info({table})")}
        name_col = "demangledName" if "demangledName" in columns else "shortName"
        for start, end, name_id in connection.execute(
            f"SELECT start, end, {name_col} FROM {table}"
        ):
            rows.append((int(start), int(end), strings.get(name_id, str(name_id))))
    return rows


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    last_end = None
    for start, end in sorted(intervals):
        if last_end is None or start > last_end:
            total += end - start
            last_end = end
        elif end > last_end:
            total += end - last_end
            last_end = end
    return total


def classify_kernel(name: str) -> str | None:
    for family, pattern in KERNEL_FAMILY_RULES:
        if re.search(pattern, name, re.IGNORECASE):
            return family
    return None


def analyze_kernels(args: argparse.Namespace) -> dict[str, Any]:
    bytes_payload = json.loads(args.bytes_model.read_text(encoding="utf-8"))
    ceiling_payload = json.loads(args.ceiling.read_text(encoding="utf-8"))
    ceiling_gbps = float(ceiling_payload["study_ceiling_gbps"])

    per_range: list[dict[str, Any]] = []
    kernel_name_totals: dict[str, dict[str, Any]] = {}
    for sqlite_path in args.sqlite:
        connection = sqlite3.connect(sqlite_path)
        try:
            ranges = _nvtx_interface_ranges(connection)
            kernels = _kernel_rows(connection)
        finally:
            connection.close()
        if not kernels:
            raise ValueError(
                f"{sqlite_path} has no explicit kernel rows; was --cuda-graph-trace=node set?"
            )
        for rng in ranges:
            clipped = [
                (max(start, rng["start"]), min(end, rng["end"]), name)
                for start, end, name in kernels
                if end > rng["start"] and start < rng["end"]
            ]
            union_ns = interval_union_ns([(s, e) for s, e, _ in clipped])
            wall_ns = rng["end"] - rng["start"]
            span_ns = (
                max(e for _, e, _ in clipped) - min(s for s, _, _ in clipped) if clipped else 0
            )
            per_range.append(
                {
                    "label": rng["label"],
                    "class": rng["class"],
                    "wall_ms": wall_ns / 1e6,
                    "kernel_union_ms": union_ns / 1e6,
                    "kernel_span_ms": span_ns / 1e6,
                    "intra_span_idle_ms": (span_ns - union_ns) / 1e6 if clipped else 0.0,
                    "non_kernel_ms": (wall_ns - union_ns) / 1e6,
                    "kernel_count": len(clipped),
                }
            )
            for start, end, name in clipped:
                entry = kernel_name_totals.setdefault(
                    name,
                    {"total_ms": 0.0, "count": 0, "family": classify_kernel(name), "instances": []},
                )
                duration_ms = (end - start) / 1e6
                entry["total_ms"] += duration_ms
                entry["count"] += 1
                if entry["family"] == "unquantized_gemv_bf16":
                    entry["instances"].append(round(duration_ms, 4))

    range_count = len(per_range)
    if range_count == 0:
        raise ValueError("no ranges analyzed")

    families: dict[str, dict[str, Any]] = {}
    unattributed_ms = 0.0
    for name, entry in kernel_name_totals.items():
        per_range_ms = entry["total_ms"] / range_count
        family = entry["family"]
        if family is None:
            unattributed_ms += per_range_ms
            continue
        record = families.setdefault(
            family, {"ms_per_range": 0.0, "kernel_names": [], "instances_per_range": 0.0}
        )
        record["ms_per_range"] += per_range_ms
        record["instances_per_range"] += entry["count"] / range_count
        record["kernel_names"].append(name)

    weight_families = bytes_payload["families"]
    fh_bytes = weight_families.get("function_head_f32", {}).get("bytes", 0)
    attn_bytes = weight_families.get("attention_f32", {}).get("bytes", 0)
    gemv_family = families.get("unquantized_gemv_bf16")
    gemv_instances = sorted(
        (
            inst
            for entry in kernel_name_totals.values()
            if entry["family"] == "unquantized_gemv_bf16"
            for inst in entry["instances"]
        ),
        reverse=True,
    )
    quant_estimates = {}
    if gemv_family and fh_bytes:
        bf16_ms = gemv_family["ms_per_range"]
        # runtime casts checkpoint F32 modules to bf16: half the checkpoint bytes
        bf16_bytes = (fh_bytes + attn_bytes) // 2
        implied_gbps = bf16_bytes / (bf16_ms / 1e3) / 1e9 if bf16_ms else None
        quant_estimates = {
            "unquantized_gemv_ms_per_step": bf16_ms,
            "runtime_bf16_bytes": bf16_bytes,
            "implied_gbps": implied_gbps,
            "instance_durations_ms_desc": gemv_instances[: max(1, len(gemv_instances))][:40],
            "recoverable_ms_if_w8": bf16_ms * 0.5 if bf16_ms else None,
            "recoverable_ms_if_w4": bf16_ms * 0.75 if bf16_ms else None,
            "note": (
                "descriptive estimate for a NEW quantization-extension candidate; "
                "requires instance-count confirmation and its own qualification; "
                "no ranking authority"
            ),
        }

    top_kernels = sorted(
        (
            {
                "name": name[:160],
                "family": entry["family"],
                "ms_per_range": entry["total_ms"] / range_count,
                "instances_per_range": entry["count"] / range_count,
            }
            for name, entry in kernel_name_totals.items()
        ),
        key=lambda item: -item["ms_per_range"],
    )[:40]

    payload = {
        "kind": "bandwidth_attribution_kernels",
        "ranges_analyzed": range_count,
        "per_range": per_range,
        "mean_wall_ms": statistics.fmean(r["wall_ms"] for r in per_range),
        "mean_kernel_union_ms": statistics.fmean(r["kernel_union_ms"] for r in per_range),
        "mean_intra_span_idle_ms": statistics.fmean(r["intra_span_idle_ms"] for r in per_range),
        "mean_non_kernel_ms": statistics.fmean(r["non_kernel_ms"] for r in per_range),
        "families": families,
        "unattributed_ms_per_range": unattributed_ms,
        "top_kernels": top_kernels,
        "quantization_extension_estimates": quant_estimates,
        "study_ceiling_gbps": ceiling_gbps,
        "modeled_bytes_per_step": bytes_payload["per_step_total_modeled_bytes"],
        "implied_gbps_at_mean_union": (
            bytes_payload["per_step_total_modeled_bytes"]
            / (statistics.fmean(r["kernel_union_ms"] for r in per_range) / 1e3)
            / 1e9
        ),
        "passed": True,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bm = sub.add_parser("bytes-model")
    bm.add_argument("--model-root", type=Path, required=True)
    bm.add_argument("--sequence-length", type=int, default=4096)
    bm.add_argument("--output", type=Path, required=True)

    ce = sub.add_parser("ceiling")
    ce.add_argument("--sizes-gb", default="1,2,4")
    ce.add_argument("--repeats", type=int, default=20)
    ce.add_argument("--warmup", type=int, default=5)
    ce.add_argument("--output", type=Path, required=True)

    ak = sub.add_parser("analyze-kernels")
    ak.add_argument("--sqlite", type=Path, action="append", required=True)
    ak.add_argument("--bytes-model", type=Path, required=True)
    ak.add_argument("--ceiling", type=Path, required=True)
    ak.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bytes-model":
        payload = bytes_model(args)
    elif args.command == "ceiling":
        payload = ceiling(args)
    else:
        payload = analyze_kernels(args)
    atomic_json(args.output, payload)
    condensed = {k: v for k, v in payload.items() if k not in {"per_range", "top_kernels"}}
    print(json.dumps(condensed, indent=2, sort_keys=True, default=str)[:4000])
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
