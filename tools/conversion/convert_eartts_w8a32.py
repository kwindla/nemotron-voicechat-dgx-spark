#!/usr/bin/env python3
"""Create an immutable EarTTS FP32-activation/INT8-weight checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from artifact_io import ShardWriter, tensor_bytes

PROJECTION_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
SAMPLER_MLP_PREFIX = "model.sampler.mog_head.mlp_stack."
SAMPLER_MLP_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
GIB = 1024**3
DEFAULT_MIN_REMAINING_GIB = 100.0
# The final W8A32 tensor payload is smaller than the source FP32 payload.  Keep
# a conservative allowance for safetensors headers, scales, config/index files,
# and the conversion report rather than relying on that reduction in preflight.
OUTPUT_OVERHEAD_BYTES = 128 * 1024**2


def selected(name: str) -> bool:
    return (name.startswith("model.backbone.layers.") and name.endswith(PROJECTION_SUFFIXES)) or (
        name.startswith(SAMPLER_MLP_PREFIX) and name.endswith(SAMPLER_MLP_SUFFIXES)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument(
        "--min-remaining-gib",
        type=float,
        default=DEFAULT_MIN_REMAINING_GIB,
        help="Refuse conversion unless projected remaining disk meets this floor",
    )
    return parser.parse_args()


def _existing_disk_anchor(path: Path) -> Path:
    anchor = path
    while not anchor.exists():
        if anchor.parent == anchor:
            raise RuntimeError(f"No existing ancestor for output path {path}")
        anchor = anchor.parent
    return anchor


def projected_output_upper_bound(source: Path) -> int:
    """Conservative peak additional bytes required by an immutable conversion."""
    return sum(path.stat().st_size for path in source.iterdir() if path.is_file()) + (
        OUTPUT_OVERHEAD_BYTES
    )


def conversion_disk_preflight(
    source: Path,
    output: Path,
    *,
    min_remaining_gib: float = DEFAULT_MIN_REMAINING_GIB,
    free_bytes: int | None = None,
) -> dict[str, int | float | str]:
    if min_remaining_gib < DEFAULT_MIN_REMAINING_GIB:
        raise ValueError(
            "min_remaining_gib cannot be lower than the hard "
            f"{DEFAULT_MIN_REMAINING_GIB:.0f} GiB conversion floor"
        )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    anchor = _existing_disk_anchor(output.parent)
    available = shutil.disk_usage(anchor).free if free_bytes is None else free_bytes
    projected = projected_output_upper_bound(source)
    remaining = available - projected
    floor = int(min_remaining_gib * GIB)
    result: dict[str, int | float | str] = {
        "disk_anchor": str(anchor),
        "available_bytes": available,
        "projected_peak_additional_bytes": projected,
        "projected_remaining_bytes": remaining,
        "minimum_remaining_gib": min_remaining_gib,
    }
    if remaining < floor:
        raise RuntimeError(
            "EarTTS W8A32 conversion disk preflight failed: "
            f"{available / GIB:.2f} GiB free - "
            f"{projected / GIB:.2f} GiB projected peak = "
            f"{remaining / GIB:.2f} GiB remaining, below the "
            f"{min_remaining_gib:.2f} GiB floor"
        )
    return result


def quantize(weight, torch, device: str):
    source = weight.to(device=device, dtype=torch.float32)
    scales = source.abs().amax(dim=1).clamp_min_(torch.finfo(torch.float32).tiny) / 127
    quantized = torch.round(source / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    stride = max(1, source.numel() // 1_000_000)
    source_sample = source.reshape(-1)[::stride]
    restored_sample = (quantized.float() * scales[:, None]).reshape(-1)[::stride]
    error = restored_sample - source_sample
    metrics = {
        "shape": list(weight.shape),
        "sample_count": error.numel(),
        "sample_relative_rmse": (
            error.square().mean().sqrt() / source_sample.square().mean().sqrt()
        ).item(),
        "sample_max_abs_error": error.abs().max().item(),
    }
    return quantized.cpu(), scales[:, None].cpu(), metrics


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model_path = source / "model.safetensors"
    config_path = source / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise SystemExit(f"Missing EarTTS model/config in {source}")

    try:
        disk_preflight = conversion_disk_preflight(
            source,
            output,
            min_remaining_gib=args.min_remaining_gib,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"disk_preflight": disk_preflight}, indent=2), flush=True)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.time()
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        selected_names = [name for name in names if selected(name)]
    expected_selected = 28 * 7 + 3 * 3
    if len(selected_names) != expected_selected:
        raise SystemExit(
            f"Expected {expected_selected} EarTTS projections, found {len(selected_names)}"
        )

    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name in {
            "model.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "quantize_config.json",
        }:
            continue
        if path.is_file():
            shutil.copy2(path, output / path.name)

    quantization_config = {"quant_method": "ea_w8a32"}
    config = json.loads(config_path.read_text())
    config["quantization_config"] = quantization_config
    config["ea_w8a32_backbone"] = True
    config["ea_w8a32_sampler_mlp"] = True
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "quantize_config.json").write_text(json.dumps(quantization_config, indent=2) + "\n")

    writer = ShardWriter(output, int(args.max_shard_gib * 1024**3), save_file)
    layer_metrics = []
    source_bytes = 0
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for index, name in enumerate(names, 1):
            tensor = handle.get_tensor(name)
            if selected(name):
                qweight, scale, metrics = quantize(tensor, torch, args.device)
                writer.add(name, qweight)
                writer.add(name.removesuffix(".weight") + ".weight_scale", scale)
                source_bytes += tensor_bytes(tensor)
                metrics["source_name"] = name
                layer_metrics.append(metrics)
                print(
                    f"[{index}/{len(names)}] quantized {name}: "
                    f"relative RMSE {metrics['sample_relative_rmse']:.6f}",
                    flush=True,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                writer.add(name, tensor)

    manifest = writer.finish()
    # `safetensors.save_file` follows a restrictive process umask in the
    # conversion container. Derived checkpoints are immutable deployment
    # inputs and must remain readable by a non-root runtime/validator.
    for shard in manifest["shards"]:
        (output / shard["file"]).chmod(0o644)
    total_size = sum(item["bytes"] for item in manifest["shards"])
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": manifest["weight_map"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "schema": 1,
        "status": "complete",
        "source": str(source),
        "output": str(output),
        "method": "symmetric per-output-channel W8 weights, FP32 activations/accumulation",
        "selected_tensor_count": len(selected_names),
        "selected_source_bytes": source_bytes,
        "output_tensor_bytes": total_size,
        "elapsed_seconds": time.time() - started,
        "disk_preflight": disk_preflight,
        "shards": manifest["shards"],
        "layer_metrics": layer_metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "selected_tensor_count",
                    "selected_source_bytes",
                    "output_tensor_bytes",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
