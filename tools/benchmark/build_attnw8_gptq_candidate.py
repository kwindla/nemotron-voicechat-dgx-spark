#!/usr/bin/env python3
"""Sequential Hessian-calibrated attention-W8 builder under the sealed contract."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
from pathlib import Path
from typing import Any

from attnw8_gptq_core import (
    ATTENTION_LAYERS,
    BLOCK_SIZE,
    DAMP_PERCENT,
    GROUP_SIZE,
    ArtifactReader,
    atomic_json,
    canonical_sha256,
    capture_inputs,
    hierarchy_hessian,
    load_fhw8_eager,
    load_sequences,
    packed_prefix,
    prove_shared_qkv,
    quantize_weight,
    run_layer,
    selected_request_ids,
    sha256_file,
    target_weight_name,
    tree_manifest,
)

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
Q_NAMES = ("qweight", "qzeros", "scales", "g_idx")
MODULE_ROUTES = tuple(f"mixer.{name}" for name in PROJECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal1", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--fhw8-root", type=Path, required=True)
    parser.add_argument("--model-code-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--a1-root", type=Path, action="append", required=True)
    parser.add_argument("--tool-capture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--intermediate-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell", default="full")
    return parser.parse_args()


def tensor_sha(tensor: Any) -> str:
    import hashlib
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def selected_stream_sha(activations: list[Any], indexes: list[list[int]]) -> str:
    return canonical_sha256(
        [
            {
                "shape": list(activation[:, rows].shape),
                "dtype": str(activation.dtype),
                "sha256": tensor_sha(activation[:, rows]),
            }
            for activation, rows in zip(activations, indexes, strict=True)
        ]
    )


def repaired_damped(hessian: Any, torch):
    h = hessian.to(torch.float32).clone()
    dead = torch.diag(h) == 0
    if bool(dead.any()):
        if bool((h[dead] != 0).any()) or bool((h[:, dead] != 0).any()):
            raise RuntimeError("dead Hessian column is not isolated")
        diagonal = torch.arange(h.shape[0], device=h.device)
        h[diagonal[dead], diagonal[dead]] = 1.0
    damping = DAMP_PERCENT * torch.diag(h).mean()
    diagonal = torch.arange(h.shape[0], device=h.device)
    h[diagonal, diagonal] += damping
    return h, float(damping)


def leverage_rows(hessian: Any, activations: list[Any], indexes: list[list[int]], torch):
    h, damping = repaired_damped(hessian, torch)
    chol = torch.linalg.cholesky(h)
    rows = torch.cat(
        [activation[0, selected].to(h.device, torch.float32) for activation, selected in zip(activations, indexes, strict=True)],
        dim=0,
    )
    solved = torch.cholesky_solve(rows.t(), chol)
    values = (rows * solved.t()).sum(dim=1)
    return {
        "damping": damping,
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "values_sha256": tensor_sha(values),
    }


def pack_roundtrip(packed: dict[str, Any], result: dict[str, Any], torch) -> dict[str, Any]:
    qweight = packed["qweight"].to(torch.int32)
    lanes = [((qweight >> (8 * lane)) & 0xFF) for lane in range(4)]
    signed = torch.stack(lanes, dim=1).reshape(qweight.shape[0] * 4, qweight.shape[1]).t()
    signed = signed.to(torch.int32) - 128
    codes_exact = torch.equal(signed.cpu(), result["codes"].to(torch.int32).cpu())
    scales_exact = torch.equal(
        packed["scales"].cpu(), result["scales"].to(torch.bfloat16).cpu()
    )
    metadata = (
        bool((packed["qzeros"] == 0).all())
        and torch.equal(
            packed["g_idx"].cpu().to(torch.int32),
            torch.arange(signed.shape[1], dtype=torch.int32) // GROUP_SIZE,
        )
    )
    if not (codes_exact and scales_exact and metadata):
        raise RuntimeError("independent packed-runtime round-trip failed")
    return {
        "signed_codes_exact": codes_exact,
        "stored_bf16_scales_exact": scales_exact,
        "zero_qzeros_and_natural_g_idx": metadata,
    }


def diagonal_selfcheck(torch, device: str) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.randn((8, 128), generator=generator, dtype=torch.float32).to(device)
    direct_scale = torch.maximum(
        weight.amax(dim=1).abs() / 127,
        weight.amin(dim=1).abs() / 128,
    )
    direct_scale = torch.where(direct_scale == 0, torch.ones_like(direct_scale), direct_scale)
    direct_codes = torch.round(weight / direct_scale[:, None]).clamp(-128, 127).to(torch.int32)
    checks = []
    for name, diagonal in (
        ("identity", torch.ones(128, device=device)),
        ("positive_diagonal", torch.linspace(0.5, 2.0, 128, device=device)),
    ):
        result, _packed, metrics = quantize_weight(weight, torch.diag(diagonal), torch)
        exact = torch.equal(result["codes"].to(device), direct_codes)
        scales_exact = torch.equal(result["scales"].reshape(-1).to(device), direct_scale)
        expected = (direct_codes.to(torch.float32) * direct_scale[:, None]).to(torch.bfloat16)
        dequant_exact = torch.equal(result["dequantized"].to(device), expected)
        if not (exact and scales_exact and dequant_exact):
            raise RuntimeError(f"diagonal Hessian self-check failed: {name}")
        checks.append(
            {
                "name": name,
                "codes_exact": exact,
                "scales_exact": scales_exact,
                "dequantized_exact": dequant_exact,
                "metrics": metrics,
            }
        )
    return {"passed": True, "checks": checks}


def remove_projection_exclusions(dynamic: dict[str, Any]) -> list[str]:
    removed = []
    for key in list(dynamic):
        if any(projection in key for projection in PROJECTIONS):
            removed.append(key)
            del dynamic[key]
    return removed


def add_routes(config: dict[str, Any]) -> list[str]:
    routes = config.setdefault("modules_in_block_to_quantize", [])
    for route in MODULE_ROUTES:
        if route not in routes:
            routes.append(route)
    return list(routes)


def cell_targets(cell: str) -> tuple[set[str], bool]:
    universe = {
        target_weight_name(layer, projection)
        for layer in ATTENTION_LAYERS
        for projection in PROJECTIONS
    }
    base_only = cell == "base-only"
    if cell in {"full", "base-only"}:
        return universe, base_only
    if cell.startswith("layer-only-"):
        layer = int(cell.removeprefix("layer-only-"))
        return {name for name in universe if f".layers.{layer}." in name}, False
    if cell.startswith("leave-layer-"):
        layer = int(cell.removeprefix("leave-layer-"))
        return {name for name in universe if f".layers.{layer}." not in name}, False
    if cell.startswith("family-only-"):
        projection = cell.removeprefix("family-only-") + "_proj"
        return {name for name in universe if f".{projection}.weight" in name}, False
    if cell.startswith("leave-family-"):
        projection = cell.removeprefix("leave-family-") + "_proj"
        return {name for name in universe if f".{projection}.weight" not in name}, False
    import re

    match = re.fullmatch(r"(14|21|30|39)-(q|k|v|o)-(only|leave)", cell)
    if match:
        layer = int(match.group(1))
        projection = match.group(2) + "_proj"
        target = target_weight_name(layer, projection)
        return ({target} if match.group(3) == "only" else universe - {target}), False
    raise ValueError(f"unknown sealed contingency cell: {cell}")


def payload_bytes(path: Path) -> int:
    with path.open("rb") as stream:
        (header,) = struct.unpack("<Q", stream.read(8))
    return path.stat().st_size - 8 - header


def assemble_candidate(
    source: Path,
    output: Path,
    packed_by_weight: dict[str, dict[str, Any]],
    selected_targets: set[str],
) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, dict[str, dict[str, Any]]] = {}
    for weight_name, packed in packed_by_weight.items():
        by_shard.setdefault(weight_map[weight_name], {})[weight_name] = packed
    for shard in sorted(source.glob("*.safetensors")):
        target = output / shard.name
        if shard.name not in by_shard:
            shutil.copyfile(shard, target)
            continue
        tensors = load_file(shard)
        for weight_name, packed in by_shard[shard.name].items():
            del tensors[weight_name]
            prefix = packed_prefix(weight_name)
            for suffix, tensor in packed.items():
                tensors[f"{prefix}.{suffix}"] = tensor.cpu().contiguous()
        save_file(tensors, target)
    for weight_name in sorted(packed_by_weight):
        shard = weight_map.pop(weight_name)
        prefix = packed_prefix(weight_name)
        for suffix in Q_NAMES:
            weight_map[f"{prefix}.{suffix}"] = shard
    index["metadata"]["total_size"] = sum(
        payload_bytes(path) for path in sorted(output.glob("*.safetensors"))
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    embedded = config["quantization_config"]
    embedded_removed = remove_projection_exclusions(embedded.setdefault("dynamic", {}))
    for weight_name in sorted(
        {
            target_weight_name(layer, projection)
            for layer in ATTENTION_LAYERS
            for projection in PROJECTIONS
        }
        - selected_targets
    ):
        module_name = weight_name.removesuffix(".weight")
        embedded["dynamic"][f"-:^{module_name}$"] = {}
    embedded_routes = add_routes(embedded)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quant = json.loads((source / "quantize_config.json").read_text(encoding="utf-8"))
    removed = remove_projection_exclusions(quant.setdefault("dynamic", {}))
    for weight_name in sorted(
        {
            target_weight_name(layer, projection)
            for layer in ATTENTION_LAYERS
            for projection in PROJECTIONS
        }
        - selected_targets
    ):
        module_name = weight_name.removesuffix(".weight")
        quant["dynamic"][f"-:^{module_name}$"] = {}
    routes = add_routes(quant)
    (output / "quantize_config.json").write_text(
        json.dumps(quant, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not removed or not embedded_removed:
        raise RuntimeError("projection exclusion was not removed from both config locations")
    for path in source.iterdir():
        if path.name.endswith(".safetensors") or path.name in {
            "model.safetensors.index.json",
            "config.json",
            "quantize_config.json",
            "artifact.sha256.json",
        }:
            continue
        target = output / path.name
        if path.is_file():
            shutil.copyfile(path, target)
        elif path.is_dir():
            shutil.copytree(path, target)
    manifest = {
        path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    (output / "artifact.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "quantized": sorted(packed_by_weight),
        "dynamic_removed": removed,
        "embedded_dynamic_removed": embedded_removed,
        "routes": routes,
        "embedded_routes": embedded_routes,
    }


def main() -> None:
    args = parse_args()
    seal1 = json.loads(args.seal1.read_text(encoding="utf-8"))
    a0 = json.loads(args.a0.read_text(encoding="utf-8"))
    if seal1.get("kind") != "attnw8_gptq_seal1" or not seal1.get("passed"):
        raise SystemExit("candidate construction requires a passing Seal-1")
    if seal1.get("a0_sha256") != sha256_file(args.a0):
        raise SystemExit("Seal-1/A0 hash mismatch")
    if args.intermediate_root.exists() or args.candidate_root.exists():
        raise FileExistsError("clean build outputs must not already exist")
    args.intermediate_root.mkdir(parents=True)
    spool_root = args.intermediate_root / "packed"
    spool_root.mkdir()

    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    selfcheck = diagonal_selfcheck(torch, args.device)
    selected_targets, base_only = cell_targets(args.cell)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    a1 = load_sequences(
        args.a1_root,
        torch=torch,
        include_request_ids=selected_request_ids(selection, "calibration"),
    )
    fixture_records = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))["records"]
    a2_fixture = [item for item in fixture_records if item["partition"] == "a2"]
    all_tool = load_sequences([args.tool_capture], torch=torch)
    all_tool = all_tool[-len([item for item in fixture_records if item["partition"] in {"a2", "c2"}]) :]
    a2 = all_tool[: len(a2_fixture)]
    for sequence, scenario in zip(a2, a2_fixture, strict=True):
        sequence.scenario_id = scenario["scenario_id"]
    if len(a1) != 3 or len(a2) != 24:
        raise RuntimeError("sealed A1/A2 source counts changed")
    a1_rows = [list(range(sequence.embeds.shape[1])) for sequence in a1]
    a2_rows = [a0["a2_windows"][scenario["scenario_id"]] for scenario in a2_fixture]
    sequences = a1 + a2
    hidden = [sequence.embeds.to(args.device, torch.bfloat16) for sequence in sequences]
    model, function_weight, _reader = load_fhw8_eager(
        args.fhw8_root, args.model_code_root, device=args.device, torch=torch
    )
    del function_weight
    packed_by_weight = {}
    metrics = []
    intermediate_streams = []
    try:
        for layer_index, layer in enumerate(model.backbone.layers):
            if layer_index not in ATTENTION_LAYERS:
                hidden = run_layer(layer, hidden, torch)
                continue
            _teacher_outputs, captures = capture_inputs(
                layer, hidden, ("q_proj", "k_proj", "v_proj"), torch
            )
            prove_shared_qkv(captures, torch)
            shared = captures["q_proj"]
            a1_h, a1_audit = hierarchy_hessian(
                shared[: len(a1)], a1_rows, torch=torch, device=args.device
            )
            a2_h, a2_audit = hierarchy_hessian(
                shared[len(a1) :], a2_rows, torch=torch, device=args.device, positive_count=12
            )
            hessian = a1_h if base_only else 0.75 * a1_h + 0.25 * a2_h
            qkv_stream_sha = selected_stream_sha(
                shared, a1_rows + a2_rows
            )
            for projection in ("q_proj", "k_proj", "v_proj"):
                name = target_weight_name(layer_index, projection)
                if name not in selected_targets:
                    continue
                module = getattr(layer.mixer, projection)
                result, packed, quant_metrics = quantize_weight(module.weight, hessian, torch)
                roundtrip = pack_roundtrip(packed, result, torch)
                module.weight.data.copy_(result["dequantized"].to(module.weight.device))
                packed_by_weight[name] = {key: value.cpu() for key, value in packed.items()}
                save_file(
                    {f"{packed_prefix(name)}.{key}": value.cpu().contiguous() for key, value in packed.items()},
                    spool_root / f"{name.replace('.', '__')}.safetensors",
                )
                metrics.append(
                    {
                        "weight": name,
                        "input_site": "shared_actual_qkv_argument",
                        "hessian_trace": float(torch.trace(hessian)),
                        "a1_sources": a1_audit,
                        "a2_sources": a2_audit,
                        "a2_final_trace_share": (
                            0.0
                            if base_only
                            else float(torch.trace(0.25 * a2_h) / torch.trace(hessian))
                        ),
                        "tool_row_leverage": leverage_rows(hessian, shared[len(a1) :], a2_rows, torch),
                        "quantizer": quant_metrics,
                        "packed_roundtrip": roundtrip,
                        "stream_sha256": qkv_stream_sha,
                    }
                )
                del result, packed
            _qkv_outputs, o_capture = capture_inputs(layer, hidden, ("o_proj",), torch)
            actual_o = o_capture["o_proj"]
            o_a1_h, o_a1_audit = hierarchy_hessian(
                actual_o[: len(a1)], a1_rows, torch=torch, device=args.device
            )
            o_a2_h, o_a2_audit = hierarchy_hessian(
                actual_o[len(a1) :], a2_rows, torch=torch, device=args.device, positive_count=12
            )
            o_hessian = o_a1_h if base_only else 0.75 * o_a1_h + 0.25 * o_a2_h
            o_module = layer.mixer.o_proj
            name = target_weight_name(layer_index, "o_proj")
            if name in selected_targets:
                result, packed, quant_metrics = quantize_weight(o_module.weight, o_hessian, torch)
                roundtrip = pack_roundtrip(packed, result, torch)
                o_module.weight.data.copy_(result["dequantized"].to(o_module.weight.device))
                packed_by_weight[name] = {key: value.cpu() for key, value in packed.items()}
                save_file(
                    {f"{packed_prefix(name)}.{key}": value.cpu().contiguous() for key, value in packed.items()},
                    spool_root / f"{name.replace('.', '__')}.safetensors",
                )
                metrics.append(
                    {
                        "weight": name,
                        "input_site": "actual_post_attention_o_argument_with_gptq_qkv",
                        "hessian_trace": float(torch.trace(o_hessian)),
                        "a1_sources": o_a1_audit,
                        "a2_sources": o_a2_audit,
                        "a2_final_trace_share": (
                            0.0
                            if base_only
                            else float(torch.trace(0.25 * o_a2_h) / torch.trace(o_hessian))
                        ),
                        "tool_row_leverage": leverage_rows(
                            o_hessian, actual_o[len(a1) :], a2_rows, torch
                        ),
                        "quantizer": quant_metrics,
                        "packed_roundtrip": roundtrip,
                        "stream_sha256": selected_stream_sha(actual_o, a1_rows + a2_rows),
                    }
                )
            intermediate_streams.append(
                {
                    "layer": layer_index,
                    "qkv_sha256": qkv_stream_sha,
                    "o_sha256": selected_stream_sha(actual_o, a1_rows + a2_rows),
                }
            )
            hidden = run_layer(layer, hidden, torch)
            del a1_h, a2_h, hessian, o_a1_h, o_a2_h, o_hessian
            gc.collect()
            torch.cuda.empty_cache()
            print(f"sequential GPTQ layer {layer_index} complete", flush=True)
    finally:
        del model, hidden
        gc.collect()
        torch.cuda.empty_cache()
    if set(packed_by_weight) != selected_targets:
        raise RuntimeError(
            f"sealed cell target mismatch: expected {len(selected_targets)}, built {len(packed_by_weight)}"
        )
    delta = assemble_candidate(
        args.fhw8_root.resolve(), args.candidate_root, packed_by_weight, selected_targets
    )
    spool_manifest = tree_manifest(spool_root)
    spool_manifest.pop("root", None)
    reproducible = {
        "selfcheck": selfcheck,
        "metrics": metrics,
        "intermediate_streams": intermediate_streams,
        "spool_manifest": spool_manifest,
    }
    atomic_json(args.intermediate_root / "reproducible.json", reproducible)
    report = {
        "schema": 1,
        "kind": "attnw8_gptq_sequential_build",
        "cell": args.cell,
        "seal1_sha256": sha256_file(args.seal1),
        "a0_sha256": sha256_file(args.a0),
        "numerics": {
            "dtype": "float32",
            "group_size": GROUP_SIZE,
            "block_size": BLOCK_SIZE,
            "damp_percent": DAMP_PERCENT,
            "desc_act": False,
            "projection_order": list(PROJECTIONS),
            "layer_order": list(ATTENTION_LAYERS),
            "seed": 0,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "delta": delta,
        "candidate_manifest": tree_manifest(args.candidate_root),
        "intermediate_manifest": tree_manifest(args.intermediate_root),
        "metrics": metrics,
        "passed": True,
    }
    atomic_json(args.report, report)
    print(json.dumps({"tensors": len(metrics), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
