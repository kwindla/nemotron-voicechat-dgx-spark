#!/usr/bin/env python3
"""Build the sealed attention-projection W8 Nano candidate.

The source is the unchanged fhw8 artifact.  Each q_proj/k_proj/v_proj/o_proj
weight in the four attention blocks is replaced in its original shard by the
same deterministic GPTQ-W8 packing used by ``build_fhw8_candidate.py``.
Unlike lm_head and function_head, these are normal mixer linears and require
only the standard GPTQ module routing in both quantization-config locations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import torch
from build_fhw8_candidate import (
    BITS,
    GROUP_SIZE,
    Q_NAMES,
    quantize_symmetric_group128,
    roundtrip_check,
    sha256_file,
)

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MODULE_ROUTES = tuple(f"mixer.{name}" for name in PROJECTIONS)
EXPECTED_ATTENTION_LAYERS = (14, 21, 30, 39)


def is_attention_weight(name: str) -> bool:
    return name.startswith("stt_model.llm.layers.") and name.endswith(
        tuple(f".mixer.{projection}.weight" for projection in PROJECTIONS)
    )


def attention_layer(name: str) -> int:
    return int(name.split(".")[3])


def packed_prefix(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name[: -len(".weight")]


def rewrite_shard(
    source: Path,
    target: Path,
    replacements: dict[str, dict[str, torch.Tensor]],
) -> list[str]:
    from safetensors.torch import load_file, save_file

    tensors = load_file(source)
    for weight_name, packed in replacements.items():
        if weight_name not in tensors:
            raise KeyError(f"{weight_name} is absent from {source}")
        del tensors[weight_name]
        prefix = packed_prefix(weight_name)
        for suffix, tensor in packed.items():
            tensors[f"{prefix}.{suffix}"] = tensor
    save_file(tensors, target)
    return sorted(tensors)


def remove_projection_exclusions(dynamic: dict[str, Any]) -> list[str]:
    removed = []
    for key in list(dynamic):
        if any(projection in key for projection in PROJECTIONS):
            removed.append(key)
            del dynamic[key]
    return removed


def add_projection_routes(config: dict[str, Any]) -> list[str]:
    routes = config.setdefault("modules_in_block_to_quantize", [])
    for route in MODULE_ROUTES:
        if route not in routes:
            routes.append(route)
    return list(routes)


def tensor_payload_bytes(path: Path) -> int:
    with path.open("rb") as stream:
        (header_len,) = struct.unpack("<Q", stream.read(8))
    return path.stat().st_size - 8 - header_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fhw8-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.fhw8_root.resolve()
    candidate = args.candidate_root.resolve()
    if candidate.exists() and any(candidate.iterdir()):
        raise SystemExit(f"candidate root {candidate} already exists and is non-empty")
    candidate.mkdir(parents=True, exist_ok=True)

    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    weights = sorted(name for name in weight_map if is_attention_weight(name))
    observed_layers = tuple(sorted({attention_layer(name) for name in weights}))
    if len(weights) != 16 or observed_layers != EXPECTED_ATTENTION_LAYERS:
        raise SystemExit(
            f"expected 16 attention weights in layers {EXPECTED_ATTENTION_LAYERS}; "
            f"found {len(weights)} in {observed_layers}"
        )

    from safetensors.torch import load_file

    replacements: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    checks: dict[str, dict[str, Any]] = {}
    layouts: dict[str, dict[str, Any]] = {}
    for weight_name in weights:
        shard_name = weight_map[weight_name]
        weight = load_file(source / shard_name)[weight_name]
        if weight.dtype != torch.float32 or weight.ndim != 2:
            raise SystemExit(
                f"unexpected attention layout {weight_name}: {weight.dtype} {weight.shape}"
            )
        if weight.shape[1] % GROUP_SIZE or weight.shape[0] % (32 // BITS):
            raise SystemExit(f"unpackable attention layout {weight_name}: {weight.shape}")
        packed = quantize_symmetric_group128(weight)
        check = roundtrip_check(weight, packed)
        if not check["rtn_exact"]:
            raise SystemExit(f"G0a round-trip failed for {weight_name}: {check}")
        replacements.setdefault(shard_name, {})[weight_name] = packed
        checks[weight_name] = check
        layouts[weight_name] = {
            "source_dtype": str(weight.dtype),
            "source_shape": list(weight.shape),
            "source_shard": shard_name,
            "packed": {
                suffix: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
                for suffix, tensor in packed.items()
            },
        }
        del weight

    source_shards: dict[str, str] = {}
    for shard in sorted(source.glob("*.safetensors")):
        source_shards[shard.name] = sha256_file(shard)
        target = candidate / shard.name
        if shard.name in replacements:
            rewrite_shard(shard, target, replacements[shard.name])
        else:
            shutil.copyfile(shard, target)

    for weight_name in weights:
        shard_name = weight_map.pop(weight_name)
        prefix = packed_prefix(weight_name)
        for suffix in Q_NAMES:
            weight_map[f"{prefix}.{suffix}"] = shard_name
    index["metadata"]["total_size"] = sum(
        tensor_payload_bytes(shard) for shard in sorted(candidate.glob("*.safetensors"))
    )
    (candidate / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    embedded = config.get("quantization_config")
    if not isinstance(embedded, dict):
        raise SystemExit("config.json has no embedded quantization_config")
    embedded_dynamic = embedded.setdefault("dynamic", {})
    embedded_removed = remove_projection_exclusions(embedded_dynamic)
    embedded_modules = add_projection_routes(embedded)
    (candidate / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    quant_config = json.loads((source / "quantize_config.json").read_text(encoding="utf-8"))
    dynamic = quant_config.setdefault("dynamic", {})
    removed = remove_projection_exclusions(dynamic)
    modules = add_projection_routes(quant_config)
    (candidate / "quantize_config.json").write_text(
        json.dumps(quant_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not removed or not embedded_removed:
        raise SystemExit(
            "projection exclusion was not removed from both quantization-config locations"
        )
    for extra in source.iterdir():
        if extra.name.endswith(".safetensors") or extra.name in {
            "model.safetensors.index.json",
            "config.json",
            "quantize_config.json",
            "artifact.sha256.json",
        }:
            continue
        target = candidate / extra.name
        if extra.is_file():
            shutil.copyfile(extra, target)
        elif extra.is_dir():
            shutil.copytree(extra, target)

    manifest = {
        path.name: sha256_file(path)
        for path in sorted(candidate.iterdir())
        if path.is_file()
    }
    (candidate / "artifact.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "kind": "attnw8_candidate_build",
        "fhw8_root": str(source),
        "candidate_root": str(candidate),
        "attention_weights": weights,
        "attention_layers": list(observed_layers),
        "layouts": layouts,
        "g0a_roundtrip": checks,
        "deltas": {
            "quantized": weights,
            "quantize_config_dynamic_removed": removed,
            "embedded_quantization_config_dynamic_removed": embedded_removed,
            "quantize_config_modules_in_block_to_quantize": modules,
            "embedded_modules_in_block_to_quantize": embedded_modules,
        },
        "packing": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "sym": True,
            "desc_act": False,
            "zero_point_convention": "gptqmodel-v2 qzeros=0, w = s*(q-128)",
            "construction": "deterministic RTN against stored bf16 scales",
        },
        "source_shards": source_shards,
        "candidate_manifest": manifest,
        "passed": all(check["rtn_exact"] for check in checks.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "attention_tensors": len(weights),
                "layers": observed_layers,
                "dynamic_removed": removed,
                "embedded_dynamic_removed": embedded_removed,
                "passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
