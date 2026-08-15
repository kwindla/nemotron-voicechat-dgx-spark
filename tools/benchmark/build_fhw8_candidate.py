#!/usr/bin/env python3
"""Build the preregistered function_head-W8 candidate Nano artifact.

Produces a new artifact directory from the unchanged released checkpoint with
exactly the three preregistered deltas (reports/fhw8-qualification/
preregistration.md, G0/candidate definition):

1. ``function_head.weight`` (F32) -> GPTQ-format W8 tensors (symmetric,
   group 128, desc_act=false, gptqmodel v2 zero-point convention: qzeros
   stored all-zero, dequant ``w = s * (q - 128)``), matching the shipped
   ``lm_head`` packing exactly (qweight I32 [in/4, out], qzeros I32
   [in/group, out/4], scales BF16 [in/group, out], trivial ascending g_idx).
2. ``config.json``: ``voicechat_quantized_output_heads`` gains
   ``"function_head"``.
3. ``quantize_config.json``: the ``-:.*function_head$`` dynamic exclusion is
   removed.

The released artifact is never modified. Construction is deterministic
round-to-nearest (no calibration). The in-process round-trip check (G0a)
fails the build on any mismatch between packed tensors and the intended RTN
values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import torch

GROUP_SIZE = 128
BITS = 8
PACK = 32 // BITS  # int8 lanes per int32
ZERO_POINT = 128  # gptqmodel v2 symmetric convention; qzeros stored all-zero
FUNCTION_HEAD = "stt_model.function_head.weight"
Q_NAMES = ("qweight", "qzeros", "scales", "g_idx")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shard_headers(root: Path) -> dict[str, dict[str, Any]]:
    headers = {}
    for shard in sorted(root.glob("*.safetensors")):
        with shard.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            headers[shard.name] = json.loads(stream.read(length))
    return headers


def quantize_symmetric_group128(weight_f32: torch.Tensor) -> dict[str, torch.Tensor]:
    """RTN symmetric W8 group-128 in exact gptqmodel/lm_head packing."""
    out_features, in_features = weight_f32.shape
    if in_features % GROUP_SIZE:
        raise ValueError("in_features must be divisible by the group size")
    groups = in_features // GROUP_SIZE
    # [groups, group, out]
    grouped = weight_f32.T.reshape(groups, GROUP_SIZE, out_features).to(torch.float32)
    scales = grouped.abs().amax(dim=1) / 127.0  # [groups, out]
    if (scales == 0).any():
        # a fully-zero group still needs a nonzero scale for exact zeros
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    # quantize against the bf16-rounded scales that will actually be stored,
    # so the packed tensors are exactly self-consistent (G0a)
    scales = scales.to(torch.bfloat16).to(torch.float32)
    q = torch.round(grouped / scales.unsqueeze(1)) + ZERO_POINT
    q = q.clamp(0, 255).to(torch.int32)  # [groups, group, out]
    q_in_out = q.reshape(in_features, out_features)  # in-feature major
    # pack 4 consecutive in-features per int32 word, lane j at bits [8j, 8j+8)
    q_words = q_in_out.reshape(in_features // PACK, PACK, out_features)
    qweight = (
        q_words[:, 0, :]
        | (q_words[:, 1, :] << 8)
        | (q_words[:, 2, :] << 16)
        | (q_words[:, 3, :] << 24)
    ).to(torch.int32)
    qzeros = torch.zeros((groups, out_features // PACK), dtype=torch.int32)
    g_idx = (torch.arange(in_features, dtype=torch.int32) // GROUP_SIZE).to(torch.int32)
    return {
        "qweight": qweight.contiguous(),
        "qzeros": qzeros.contiguous(),
        "scales": scales.to(torch.bfloat16).contiguous(),
        "g_idx": g_idx.contiguous(),
    }


def dequantize(packed: dict[str, torch.Tensor], out_features: int) -> torch.Tensor:
    qweight = packed["qweight"]
    lanes = [((qweight >> (8 * j)) & 0xFF) for j in range(PACK)]
    q = torch.stack(lanes, dim=1).reshape(qweight.shape[0] * PACK, out_features)
    groups = packed["scales"].shape[0]
    grouped = q.reshape(groups, GROUP_SIZE, out_features).to(torch.float32)
    scales = packed["scales"].to(torch.float32).unsqueeze(1)
    return ((grouped - ZERO_POINT) * scales).reshape(groups * GROUP_SIZE, out_features).T


def roundtrip_check(weight_f32: torch.Tensor, packed: dict[str, torch.Tensor]) -> dict[str, Any]:
    """G0(a): packed tensors must reproduce intended RTN values exactly."""
    deq = dequantize(packed, weight_f32.shape[0])
    scales = packed["scales"].to(torch.float32)
    groups = scales.shape[0]
    grouped = weight_f32.T.reshape(groups, GROUP_SIZE, -1).to(torch.float32)
    intended_q = (
        torch.round(grouped / scales.unsqueeze(1)).clamp(-ZERO_POINT, 255 - ZERO_POINT)
    )
    intended = (intended_q * scales.unsqueeze(1)).reshape(groups * GROUP_SIZE, -1).T
    exact = torch.equal(deq, intended)
    err = (deq - weight_f32.to(torch.float32)).abs()
    rel_scale = weight_f32.to(torch.float32).abs().amax()
    return {
        "rtn_exact": bool(exact),
        "max_abs_error": float(err.amax()),
        "max_error_over_global_max": float(err.amax() / rel_scale),
        "mean_abs_error": float(err.mean()),
    }


def rewrite_shard(
    source: Path, target: Path, drop: set[str], add: dict[str, torch.Tensor], prefix: str
) -> list[str]:
    from safetensors.torch import load_file, save_file

    tensors = load_file(source)
    kept = {name: tensor for name, tensor in tensors.items() if name not in drop}
    for suffix, tensor in add.items():
        kept[f"{prefix}.{suffix}"] = tensor
    save_file(kept, target)
    return sorted(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_root.resolve()
    candidate = args.candidate_root.resolve()
    if candidate.exists() and any(candidate.iterdir()):
        raise SystemExit(f"candidate root {candidate} already exists and is non-empty")
    candidate.mkdir(parents=True, exist_ok=True)

    index = json.loads((release / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    fh_shard = weight_map[FUNCTION_HEAD]
    lm_shard = weight_map["stt_model.lm_head.qweight"]

    from safetensors.torch import load_file

    fh_weight = load_file(release / fh_shard)[FUNCTION_HEAD]
    if fh_weight.dtype != torch.float32 or fh_weight.shape[1] % GROUP_SIZE:
        raise SystemExit(f"unexpected function_head layout {fh_weight.dtype} {fh_weight.shape}")

    packed = quantize_symmetric_group128(fh_weight)
    check = roundtrip_check(fh_weight, packed)
    if not check["rtn_exact"]:
        raise SystemExit(f"G0a round-trip failed: {check}")

    # shard layout: drop the F32 weight from its shard; add packed tensors to
    # the lm_head shard (keeping quantized heads co-located).
    provenance: dict[str, Any] = {"source_shards": {}, "written_files": {}}
    for shard in sorted(release.glob("*.safetensors")):
        provenance["source_shards"][shard.name] = sha256_file(shard)
        target = candidate / shard.name
        if shard.name == fh_shard:
            rewrite_shard(shard, target, {FUNCTION_HEAD}, {}, "")
        elif shard.name == lm_shard:
            rewrite_shard(
                shard, target, set(), dict(packed.items()), "stt_model.function_head"
            )
        else:
            shutil.copyfile(shard, target)

    del weight_map[FUNCTION_HEAD]
    for suffix in Q_NAMES:
        weight_map[f"stt_model.function_head.{suffix}"] = lm_shard
    total = 0
    for shard in sorted(candidate.glob("*.safetensors")):
        total += shard.stat().st_size - 8 - struct.unpack(
            "<Q", shard.open("rb").read(8)
        )[0]
    index["metadata"]["total_size"] = total
    (candidate / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    heads = list(config.get("voicechat_quantized_output_heads") or [])
    if "function_head" not in heads:
        heads.append("function_head")
    config["voicechat_quantized_output_heads"] = heads
    # vLLM consumes the quantization config embedded in config.json, so the
    # function_head exclusion must be removed there as well
    embedded = config.get("quantization_config")
    embedded_removed: list[str] = []
    if embedded and isinstance(embedded.get("dynamic"), dict):
        for key in [k for k in embedded["dynamic"] if "function_head" in k]:
            del embedded["dynamic"][key]
            embedded_removed.append(key)
    # the loader also requires membership in modules_in_block_to_quantize
    # (is_layer_gptq_quantized consults it for every layer incl. heads)
    if embedded and "function_head" not in embedded.get("modules_in_block_to_quantize", []):
        embedded.setdefault("modules_in_block_to_quantize", []).append("function_head")
    (candidate / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    quant_config = json.loads((release / "quantize_config.json").read_text(encoding="utf-8"))
    dynamic = dict(quant_config.get("dynamic") or {})
    removed = [key for key in dynamic if "function_head" in key]
    for key in removed:
        del dynamic[key]
    quant_config["dynamic"] = dynamic
    if "function_head" not in quant_config.get("modules_in_block_to_quantize", []):
        quant_config.setdefault("modules_in_block_to_quantize", []).append("function_head")
    (candidate / "quantize_config.json").write_text(
        json.dumps(quant_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for extra in release.iterdir():
        if extra.name.endswith(".safetensors") or extra.name in {
            "model.safetensors.index.json",
            "config.json",
            "quantize_config.json",
        }:
            continue
        if extra.is_file():
            shutil.copyfile(extra, candidate / extra.name)

    manifest = {}
    for path in sorted(candidate.iterdir()):
        if path.is_file():
            manifest[path.name] = sha256_file(path)
    (candidate / "artifact.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "kind": "fhw8_candidate_build",
        "release_root": str(release),
        "candidate_root": str(candidate),
        "deltas": {
            "quantized": FUNCTION_HEAD,
            "config_quantized_output_heads": heads,
            "quantize_config_dynamic_removed": removed,
            "embedded_quantization_config_dynamic_removed": embedded_removed,
        },
        "packing": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "sym": True,
            "zero_point_convention": "gptqmodel-v2 qzeros=0, w = s*(q-128)",
            "construction": "deterministic RTN (preregistered primary)",
        },
        "g0a_roundtrip": check,
        "source_shards": provenance["source_shards"],
        "candidate_manifest": manifest,
        "passed": check["rtn_exact"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("deltas", "g0a_roundtrip", "passed")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
