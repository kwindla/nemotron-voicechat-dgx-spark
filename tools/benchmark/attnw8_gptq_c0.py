#!/usr/bin/env python3
"""C0 numeric, packing, delta, and dual-build qualification for attnW8-GPTQ."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from attnw8_gptq_core import (
    ATTENTION_LAYERS,
    ArtifactReader,
    atomic_json,
    capture_inputs,
    load_fhw8_eager,
    load_sequences,
    run_layer,
    selected_request_ids,
    sha256_file,
    target_weight_name,
    tree_manifest,
)

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal1", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--fhw8-root", type=Path, required=True)
    parser.add_argument("--gptq-root", type=Path, required=True)
    parser.add_argument("--gptq-repro-root", type=Path, required=True)
    parser.add_argument("--gptq-intermediate", type=Path, required=True)
    parser.add_argument("--gptq-repro-intermediate", type=Path, required=True)
    parser.add_argument("--rtn-root", type=Path, required=True)
    parser.add_argument("--model-code-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--q-root", type=Path, action="append", required=True)
    parser.add_argument("--tool-capture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fixed_sum(values: Any, numpy) -> float:
    flat = values.reshape(-1).astype(numpy.float64, copy=False)
    if flat.size == 0:
        return 0.0
    return float(numpy.add.accumulate(flat, dtype=numpy.float64)[-1])


def metric_cell(
    reference_rows: list[Any],
    variant_rows: list[Any],
    rtn_rows: list[Any],
    row_keys: list[str],
    *,
    numpy,
) -> dict[str, Any]:
    e_variant = 0.0
    e_rtn = 0.0
    denominator = 0.0
    count = 0
    worst_value = -1.0
    worst_key = None
    for reference, variant, rtn, key in zip(
        reference_rows, variant_rows, rtn_rows, row_keys, strict=True
    ):
        ref = reference.detach().to("cpu").float().numpy()
        var = variant.detach().to("cpu").float().numpy()
        rt = rtn.detach().to("cpu").float().numpy()
        var_sq = numpy.square(numpy.subtract(var, ref, dtype=numpy.float32), dtype=numpy.float32)
        rtn_sq = numpy.square(numpy.subtract(rt, ref, dtype=numpy.float32), dtype=numpy.float32)
        ref_sq = numpy.square(ref, dtype=numpy.float32)
        row_error = fixed_sum(var_sq, numpy)
        e_variant = float(numpy.float64(e_variant) + numpy.float64(row_error))
        e_rtn = float(numpy.float64(e_rtn) + numpy.float64(fixed_sum(rtn_sq, numpy)))
        denominator = float(
            numpy.float64(denominator) + numpy.float64(fixed_sum(ref_sq, numpy))
        )
        count += int(ref.size)
        if row_error > worst_value:
            worst_value = row_error
            worst_key = key
    if denominator == 0:
        normalized = 0.0 if e_variant == 0 else math.inf
        rtn_normalized = 0.0 if e_rtn == 0 else math.inf
    else:
        normalized = e_variant / denominator
        rtn_normalized = e_rtn / denominator
    if e_rtn == 0:
        ratio: float | str = (
            "ZERO_RTN_BOTH_ZERO" if e_variant == 0 else "ZERO_RTN_GPTQ_NONZERO"
        )
        passed = e_variant == 0
    else:
        ratio = e_variant / e_rtn
        passed = ratio <= 1.0
    return {
        "gptq_error_numerator": e_variant,
        "rtn_error_numerator": e_rtn,
        "reference_energy_denominator": denominator,
        "scalar_count": count,
        "row_count": len(reference_rows),
        "gptq_unnormalized_mse": e_variant / count,
        "rtn_unnormalized_mse": e_rtn / count,
        "gptq_normalized_mse": normalized,
        "rtn_normalized_mse": rtn_normalized,
        "gptq_over_rtn": ratio,
        "worst_row": worst_key,
        "worst_row_gptq_error": worst_value,
        "passed": passed,
    }


def apply_weights(layer: Any, reader: ArtifactReader, layer_index: int, torch, device: str):
    for projection in PROJECTIONS:
        name = target_weight_name(layer_index, projection)
        value = reader.dense_or_dequant(name, torch).to(device=device, dtype=torch.bfloat16)
        getattr(layer.mixer, projection).weight.data.copy_(value)


def selected_rows(
    tensors: list[Any], indexes: list[list[int]], keys: list[list[str]]
) -> tuple[list[Any], list[str]]:
    values = []
    flat_keys = []
    for tensor, positions, sequence_keys in zip(tensors, indexes, keys, strict=True):
        for position, key in zip(positions, sequence_keys, strict=True):
            values.append(tensor[0, position])
            flat_keys.append(key)
    return values, flat_keys


def linear_outputs(inputs: list[Any], weight: Any, torch) -> list[Any]:
    return [torch.nn.functional.linear(row, weight) for row in inputs]


def compare_trees(left: Path, right: Path) -> dict[str, Any]:
    a = tree_manifest(left)
    b = tree_manifest(right)
    a.pop("root", None)
    b.pop("root", None)
    return {"byte_identical": a == b, "left": a, "right": b}


def delta_inventory(source: Path, candidate: Path, torch) -> dict[str, Any]:
    from safetensors.torch import load_file

    source_reader = ArtifactReader(source)
    candidate_reader = ArtifactReader(candidate)
    expected = {
        target_weight_name(layer, projection)
        for layer in ATTENTION_LAYERS
        for projection in PROJECTIONS
    }
    changed_non_target = []
    replacement_errors = []
    for shard in sorted(source.glob("*.safetensors")):
        left = load_file(shard)
        right = load_file(candidate / shard.name)
        for name, tensor in left.items():
            if name in expected:
                prefix = name.removesuffix(".weight")
                required = {prefix + "." + suffix for suffix in ("qweight", "qzeros", "scales", "g_idx")}
                if name in right or not required <= set(right):
                    replacement_errors.append(name)
            elif name not in right or not torch.equal(tensor, right[name]):
                changed_non_target.append(name)
        allowed_added = {
            name.removesuffix(".weight") + "." + suffix
            for name in expected
            for suffix in ("qweight", "qzeros", "scales", "g_idx")
        }
        unexpected = set(right) - set(left) - allowed_added
        changed_non_target.extend(sorted(unexpected))
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    candidate_config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    source_quant = json.loads((source / "quantize_config.json").read_text(encoding="utf-8"))
    candidate_quant = json.loads((candidate / "quantize_config.json").read_text(encoding="utf-8"))
    return {
        "expected_replacements": sorted(expected),
        "replacement_count": len(expected),
        "replacement_errors": replacement_errors,
        "changed_non_target_tensors": changed_non_target,
        "config_changed": source_config != candidate_config,
        "quantize_config_changed": source_quant != candidate_quant,
        "passed": (
            not replacement_errors
            and not changed_non_target
            and source_config != candidate_config
            and source_quant != candidate_quant
        ),
    }


def packing_checks(root: Path, torch) -> dict[str, Any]:
    reader = ArtifactReader(root)
    checks = []
    for layer in ATTENTION_LAYERS:
        for projection in PROJECTIONS:
            name = target_weight_name(layer, projection)
            prefix = name.removesuffix(".weight")
            weight = reader.dequant_prefix(prefix, torch)
            scales = reader.tensor(prefix + ".scales")
            finite = bool(torch.isfinite(weight).all() and torch.isfinite(scales.float()).all())
            checks.append(
                {
                    "weight": name,
                    "shape": list(weight.shape),
                    "scales_shape": list(scales.shape),
                    "finite": finite,
                }
            )
    return {"checks": checks, "passed": all(item["finite"] for item in checks)}


def main() -> None:
    args = parse_args()
    seal1 = json.loads(args.seal1.read_text(encoding="utf-8"))
    a0 = json.loads(args.a0.read_text(encoding="utf-8"))
    if seal1.get("a0_sha256") != sha256_file(args.a0):
        raise SystemExit("C0 Seal-1/A0 hash mismatch")
    dual_artifact = compare_trees(args.gptq_root, args.gptq_repro_root)
    dual_intermediate = compare_trees(args.gptq_intermediate, args.gptq_repro_intermediate)

    import numpy
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    packing = packing_checks(args.gptq_root, torch)
    delta = delta_inventory(args.fhw8_root, args.gptq_root, torch)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    q = load_sequences(
        args.q_root,
        torch=torch,
        include_request_ids=selected_request_ids(selection, "evaluation"),
    )
    fixtures = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))["records"]
    tool_fixtures = [item for item in fixtures if item["partition"] == "c2"]
    all_tool_fixtures = [item for item in fixtures if item["partition"] in {"a2", "c2"}]
    tool = load_sequences([args.tool_capture], torch=torch)[-len(all_tool_fixtures) :]
    tool = tool[-len(tool_fixtures) :]
    for sequence, scenario in zip(tool, tool_fixtures, strict=True):
        sequence.scenario_id = scenario["scenario_id"]
    sequences = q + tool
    hidden = [sequence.embeds.to(args.device, torch.bfloat16) for sequence in sequences]
    base_indexes = [list(range(sequence.embeds.shape[1])) for sequence in q]
    base_keys = [[row["record_sha256"] for row in a0["indexes"]["q_base"]]]
    c2_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in a0["c2_rows"]:
        c2_by_scenario.setdefault(row["scenario_id"], []).append(row)
    tool_indexes = [
        [int(row["causal_coordinate"]) for row in c2_by_scenario[scenario["scenario_id"]]]
        for scenario in tool_fixtures
    ]
    tool_keys = [
        [str(row["canonical_row_key"]) for row in c2_by_scenario[scenario["scenario_id"]]]
        for scenario in tool_fixtures
    ]
    model, function_weight, fhw8_reader = load_fhw8_eager(
        args.fhw8_root, args.model_code_root, device=args.device, torch=torch
    )
    del function_weight
    gptq_reader = ArtifactReader(args.gptq_root)
    rtn_reader = ArtifactReader(args.rtn_root)
    cells = []
    try:
        for layer_index, layer in enumerate(model.backbone.layers):
            if layer_index not in ATTENTION_LAYERS:
                hidden = run_layer(layer, hidden, torch)
                continue
            reference_output, captured = capture_inputs(
                layer, hidden, ("q_proj", "k_proj", "v_proj", "o_proj"), torch
            )
            for projection in PROJECTIONS:
                source_inputs = (
                    captured[projection] if projection != "o_proj" else captured["o_proj"]
                )
                weights = {
                    "reference": fhw8_reader.dense_or_dequant(
                        target_weight_name(layer_index, projection), torch
                    ).to(args.device, torch.bfloat16),
                    "gptq": gptq_reader.dense_or_dequant(
                        target_weight_name(layer_index, projection), torch
                    ).to(args.device, torch.bfloat16),
                    "rtn": rtn_reader.dense_or_dequant(
                        target_weight_name(layer_index, projection), torch
                    ).to(args.device, torch.bfloat16),
                }
                for stratum, indexes, keys, offset in (
                    ("base", base_indexes, base_keys, 0),
                    ("tool", tool_indexes, tool_keys, len(q)),
                ):
                    inputs, row_keys = selected_rows(
                        source_inputs[offset : offset + len(indexes)], indexes, keys
                    )
                    ref = linear_outputs(inputs, weights["reference"], torch)
                    cand = linear_outputs(inputs, weights["gptq"], torch)
                    rtn = linear_outputs(inputs, weights["rtn"], torch)
                    cell = metric_cell(ref, cand, rtn, row_keys, numpy=numpy)
                    cells.append(
                        {
                            "kind": "linear",
                            "layer": layer_index,
                            "projection": projection,
                            "stratum": stratum,
                            **cell,
                        }
                    )
            saved = {
                projection: getattr(layer.mixer, projection).weight.detach().clone()
                for projection in PROJECTIONS
            }
            apply_weights(layer, gptq_reader, layer_index, torch, args.device)
            gptq_output = run_layer(layer, hidden, torch)
            apply_weights(layer, rtn_reader, layer_index, torch, args.device)
            rtn_output = run_layer(layer, hidden, torch)
            for projection, value in saved.items():
                getattr(layer.mixer, projection).weight.data.copy_(value)
            for stratum, indexes, keys, offset in (
                ("base", base_indexes, base_keys, 0),
                ("tool", tool_indexes, tool_keys, len(q)),
            ):
                ref, row_keys = selected_rows(
                    reference_output[offset : offset + len(indexes)], indexes, keys
                )
                cand, _ = selected_rows(
                    gptq_output[offset : offset + len(indexes)], indexes, keys
                )
                rtn, _ = selected_rows(
                    rtn_output[offset : offset + len(indexes)], indexes, keys
                )
                cells.append(
                    {
                        "kind": "attention_block",
                        "layer": layer_index,
                        "projection": None,
                        "stratum": stratum,
                        **metric_cell(ref, cand, rtn, row_keys, numpy=numpy),
                    }
                )
            hidden = reference_output
            print(f"C0 numeric layer {layer_index} complete", flush=True)
    finally:
        del model, hidden
        torch.cuda.empty_cache()
    numeric_passed = all(cell["passed"] for cell in cells)
    payload = {
        "schema": 1,
        "kind": "attnw8_gptq_c0_numeric",
        "seal1_sha256": sha256_file(args.seal1),
        "packing": packing,
        "dual_build_artifact": dual_artifact,
        "dual_build_intermediate": dual_intermediate,
        "delta_inventory": delta,
        "cells": cells,
        "numeric_passed": numeric_passed,
        "passed_without_routing": (
            packing["passed"]
            and dual_artifact["byte_identical"]
            and dual_intermediate["byte_identical"]
            and delta["passed"]
            and numeric_passed
        ),
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "cells": len(cells),
                "failed_cells": sum(not cell["passed"] for cell in cells),
                "passed_without_routing": payload["passed_without_routing"],
            },
            indent=2,
        )
    )
    if not payload["passed_without_routing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
