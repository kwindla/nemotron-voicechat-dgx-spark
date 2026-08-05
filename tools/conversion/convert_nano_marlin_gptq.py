#!/usr/bin/env python3
"""Reproduce the qualified exact-public calibrated Nano GPTQ artifact.

The calibration path consumes only captured ``combined_embeds`` tensors.  It
never tokenizes calibration text. The sole release policy uses W8, group-128
for all 105 selected tensors and zero-pads K=15680 down projections to K=15744.
The function head and embeddings are copied unchanged.

GPU use is intentionally explicit: run ``--preflight-only`` while another
VoiceChat process owns the GPU, and launch conversion only after that process
has stopped.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from artifact_io import ShardWriter, quantized_names
from nano_gptq_calibration import (
    FUNCTION_HEAD,
    GPTQ_POLICIES,
    POLICY_W8_ALL_G128,
    PUBLIC_REPOSITORY,
    PUBLIC_REVISION,
    TEXT_HEAD,
    assert_byte_identical,
    assert_disjoint_corpora,
    bits_for_target,
    conversion_disk_preflight,
    gptq_quantization_config,
    gptq_quantize_matrix,
    group_size_for_target,
    hessian_from_activations,
    inspect_replay_corpus,
    is_gptq_target,
    load_fused_sequences,
    pack_gptq_codes,
    projected_quantized_tensor_bytes,
    select_replay_corpus,
    sha256_file,
)


def method_for_policy(policy: str) -> str:
    if policy != POLICY_W8_ALL_G128:
        raise ValueError(f"unsupported GPTQ policy: {policy}")
    return (
        "calibrated symmetric GPTQ MLP+Mamba+text-head W8A16 group-128; "
        "K=15680 MLP-down padded to K=15744; function head and embeddings "
        "full precision"
    )


def runtime_identity() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "safetensors",
        "vllm",
        "compressed-tensors",
        "nvidia-modelopt",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver_version = None
    try:
        nvrm = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8")
        match = re.search(r"\b(\d{3,4}(?:\.\d+){1,3})\s+Release Build\b", nvrm)
        driver_version = match.group(1) if match else None
    except OSError:
        pass
    return {
        "image": os.environ.get("VOICECHAT_RUNTIME_IMAGE"),
        "image_id": os.environ.get("VOICECHAT_RUNTIME_IMAGE_ID"),
        "driver_version": driver_version,
        "kernel_version": platform.release(),
        "packages": packages,
        "converter_sha256": sha256_file(Path(__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="First deterministic output")
    parser.add_argument("--repro-output", type=Path)
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Produce one clean-process output; the orchestrator runs this twice.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-code-root", type=Path, required=True)
    parser.add_argument("--calibration-replay", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-replay", type=Path, action="append", required=True)
    parser.add_argument("--corpus-selection", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--policy", choices=GPTQ_POLICIES, default=POLICY_W8_ALL_G128)
    parser.add_argument("--minimum-remaining-gib", type=float, default=100.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-payload-hash-verification",
        action="store_true",
        help="Debug-only speed escape hatch; production manifests must not use it.",
    )
    return parser.parse_args()


def inventory(source: Path, policy: str = POLICY_W8_ALL_G128) -> tuple[list[dict[str, Any]], int]:
    from safetensors import safe_open

    model_path = source / "model.safetensors"
    items = []
    estimated = 0
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            info = handle.get_slice(name)
            shape = list(info.get_shape())
            dtype = info.get_dtype()
            element_bytes = {"F32": 4, "F16": 2, "BF16": 2}.get(dtype)
            if element_bytes is None:
                raise ValueError(f"unsupported source dtype {dtype}: {name}")
            source_bytes = element_bytes
            for dimension in shape:
                source_bytes *= int(dimension)
            selected = is_gptq_target(name)
            group_size = group_size_for_target(name, policy) if selected else None
            packed_shape = list(shape)
            if selected and name.endswith(".mixer.down_proj.weight"):
                if shape != [4480, 15680]:
                    raise ValueError(f"unexpected padded down-projection shape: {shape}")
                packed_shape[1] = 15744
            output_bytes = (
                projected_quantized_tensor_bytes(
                    packed_shape, bits_for_target(name, policy), group_size
                )
                if selected
                else source_bytes
            )
            estimated += output_bytes
            items.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "quantized": selected,
                    "bits": bits_for_target(name, policy) if selected else None,
                    "group_size": group_size,
                    "packed_shape": packed_shape if selected else None,
                    "source_bytes": source_bytes,
                    "estimated_output_bytes": output_bytes,
                }
            )
    selected_names = {item["name"] for item in items if item["quantized"]}
    if TEXT_HEAD not in selected_names:
        raise ValueError("public conversational text head is absent")
    if FUNCTION_HEAD in selected_names:
        raise AssertionError("function head selection leak")
    families = {
        suffix: sum(item["name"].endswith(suffix) for item in items if item["quantized"])
        for suffix in (
            ".mixer.up_proj.weight",
            ".mixer.down_proj.weight",
            ".mixer.in_proj.weight",
            ".mixer.out_proj.weight",
        )
    }
    if not all(families.values()):
        raise ValueError(f"incomplete MLP/Mamba selection: {families}")
    if len(selected_names) != 105:
        raise ValueError(f"W8 policy must select exactly 105 tensors, got {len(selected_names)}")
    if any(item["group_size"] != 128 for item in items if item["quantized"]):
        raise ValueError("W8 policy requires group-128 for every selected tensor")
    return items, estimated


def source_to_hf_name(name: str) -> str | None:
    if name.startswith("stt_model.llm."):
        return "backbone." + name.removeprefix("stt_model.llm.")
    if name == TEXT_HEAD:
        return "lm_head.weight"
    # Exact fused embeddings bypass both embedding tables.  They and the
    # function head are copied directly to output, never materialized here.
    return None


def load_calibration_model(source: Path, model_code_root: Path, torch, device: str):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    code_root = model_code_root.expanduser().resolve()
    config_payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    # AutoConfig's dynamic-module loader supplies the package context required
    # by modeling_nemotron_h.py's relative import. Direct module import does
    # not. Start from NVIDIA's base-model code/config, then overlay every exact
    # VoiceChat architecture field from the extracted public component.
    config = AutoConfig.from_pretrained(code_root, trust_remote_code=True)
    for key, value in config_payload.items():
        setattr(config, key, value)
    config.use_mamba_kernels = False

    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    finally:
        torch.set_default_dtype(old_default)
    modeling = sys.modules[model.__class__.__module__]
    # Hooks must observe in_proj/out_proj calls.  The fused Mamba fast path
    # accepts raw weights and would bypass nn.Linear hooks.
    modeling.is_fast_path_available = False
    # The exact fused capture means the large initialized table is never read.
    # Removing it saves ~1.1 GiB while calibration runs; source bytes are copied
    # unchanged when the final checkpoint is assembled.
    model.backbone.embeddings = torch.nn.Identity()
    parameters = dict(model.named_parameters())
    loaded = set()
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        for source_name in handle.keys():
            target_name = source_to_hf_name(source_name)
            if target_name is None:
                continue
            target = parameters.get(target_name)
            if target is None:
                raise KeyError(
                    f"source mapping has no HF parameter: {source_name} -> {target_name}"
                )
            value = handle.get_tensor(source_name).to(dtype=target.dtype)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"shape mismatch {source_name}: {tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.data.copy_(value)
            loaded.add(target_name)
            del value
    expected = {name for name in parameters if not name.startswith("backbone.embeddings.")}
    missing = expected - loaded
    if missing:
        raise ValueError(f"calibration model has unmapped parameters: {sorted(missing)[:12]}")
    model.eval().requires_grad_(False).to(device)
    return model, modeling


def _capture_module_inputs(layer, sequences, module_name: str, torch) -> list[Any]:
    module = getattr(layer.mixer, module_name)
    captured = []

    def hook(_module, args):
        captured.append(args[0].detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for hidden in sequences:
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                layer(hidden, cache_params=None, cache_position=positions, attention_mask=None)
    finally:
        handle.remove()
    if len(captured) != len(sequences):
        raise RuntimeError(
            f"{module_name} hook observed {len(captured)}/{len(sequences)} forwards; "
            "a fused kernel likely bypassed calibration"
        )
    return captured


def _run_layer(layer, sequences, torch) -> list[Any]:
    output = []
    with torch.inference_mode():
        for hidden in sequences:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            output.append(
                layer(
                    hidden, cache_params=None, cache_position=positions, attention_mask=None
                ).detach()
            )
    return output


def _spool_bundle(path: Path, bundle: dict[str, Any], save_file) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in bundle.items()},
        path,
        metadata={"format": "pt"},
    )


def _quantize_module(
    source_name: str,
    module,
    activations,
    *,
    torch,
    save_file,
    spool_root: Path,
    device: str,
    block_size: int,
    damp_percent: float,
    policy: str = POLICY_W8_ALL_G128,
) -> dict[str, Any]:
    bits = bits_for_target(source_name, policy)
    group_size = group_size_for_target(source_name, policy)
    hessian, rows = hessian_from_activations(activations, torch, device=device)
    weight = module.weight
    padded_input_columns = 0
    if source_name.endswith(".mixer.down_proj.weight"):
        if weight.shape[1] != 15680:
            raise ValueError(f"unexpected W8 down-projection K: {weight.shape[1]}")
        padded_input_columns = 64
        weight = torch.nn.functional.pad(weight, (0, padded_input_columns))
        hessian = torch.nn.functional.pad(
            hessian, (0, padded_input_columns, 0, padded_input_columns)
        )
    result, metrics = gptq_quantize_matrix(
        weight,
        hessian,
        torch,
        bits=bits,
        group_size=group_size,
        block_size=block_size,
        damp_percent=damp_percent,
    )
    dequantized = result["dequantized"][:, : module.weight.shape[1]]
    module.weight.data.copy_(dequantized.to(device=module.weight.device, dtype=module.weight.dtype))
    packed = pack_gptq_codes(
        result["codes"], result["scales"], torch, bits=bits, group_size=group_size
    )
    names = quantized_names(source_name)
    renamed = {names[kind]: value for kind, value in packed.items()}
    spool_path = spool_root / f"{source_name.replace('.', '__')}.safetensors"
    _spool_bundle(spool_path, renamed, save_file)
    metrics.update(
        {
            "source_name": source_name,
            "activation_rows": rows,
            "spool_file": str(spool_path),
            "spool_sha256": sha256_file(spool_path),
            "output_names": names,
            "padded_input_columns": padded_input_columns,
            "packed_input_size": int(weight.shape[1]),
        }
    )
    del hessian, result, packed, renamed
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def calibrate_and_spool(
    source: Path,
    model_code_root: Path,
    corpus,
    spool_root: Path,
    *,
    device: str,
    block_size: int,
    damp_percent: float,
    policy: str = POLICY_W8_ALL_G128,
) -> list[dict[str, Any]]:
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    model, _ = load_calibration_model(source, model_code_root, torch, device)
    sequences = [
        item.to(device=device, dtype=torch.bfloat16) for item in load_fused_sequences(corpus, torch)
    ]
    metrics = []
    metrics_log = spool_root.parent / "metrics.jsonl"

    def retain_metric(metric: dict[str, Any]) -> None:
        metrics.append(metric)
        metrics_log.parent.mkdir(parents=True, exist_ok=True)
        with metrics_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, sort_keys=True) + "\n")

    try:
        for layer_index, layer in enumerate(model.backbone.layers):
            if layer.block_type == "mlp":
                stages = ("up_proj", "down_proj")
            elif layer.block_type == "mamba":
                stages = ("in_proj", "out_proj")
            else:
                stages = ()
            for module_name in stages:
                source_name = f"stt_model.llm.layers.{layer_index}.mixer.{module_name}.weight"
                activations = _capture_module_inputs(layer, sequences, module_name, torch)
                retain_metric(
                    _quantize_module(
                        source_name,
                        getattr(layer.mixer, module_name),
                        activations,
                        torch=torch,
                        save_file=save_file,
                        spool_root=spool_root,
                        device=device,
                        block_size=block_size,
                        damp_percent=damp_percent,
                        policy=policy,
                    )
                )
                del activations
            sequences = _run_layer(layer, sequences, torch)
            print(f"calibrated layer {layer_index:02d} ({layer.block_type})", flush=True)

        with torch.inference_mode():
            final_hidden = [model.backbone.norm_f(hidden).detach() for hidden in sequences]
        retain_metric(
            _quantize_module(
                TEXT_HEAD,
                model.lm_head,
                final_hidden,
                torch=torch,
                save_file=save_file,
                spool_root=spool_root,
                device=device,
                block_size=block_size,
                damp_percent=damp_percent,
                policy=policy,
            )
        )
    finally:
        del model, sequences
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return metrics


def assemble_output(
    source: Path,
    output: Path,
    spool_root: Path,
    max_shard_gib: float,
    policy: str = POLICY_W8_ALL_G128,
) -> dict[str, Any]:
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name in {"model.safetensors", "config.json"}:
            continue
        if path.is_file():
            shutil.copy2(path, output / path.name)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = gptq_quantization_config(policy)
    config["voicechat_quantized_output_heads"] = ["lm_head"]
    config["ea_quantized_output_heads"] = False
    config["voicechat_gptq_calibrated_on_fused_embeddings"] = True
    config["voicechat_gptq_policy"] = policy
    config["voicechat_marlin_group128_input_padding"] = {
        "original_input_size": 15680,
        "padded_input_size": 15744,
        "selector": "backbone.layers.*.mixer.down_proj",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "quantize_config.json").write_text(
        json.dumps(gptq_quantization_config(policy), indent=2, sort_keys=True) + "\n"
    )

    writer = ShardWriter(output, int(max_shard_gib * 1024**3), save_file)
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_handle:
        for name in source_handle.keys():
            if is_gptq_target(name):
                spool_path = spool_root / f"{name.replace('.', '__')}.safetensors"
                if not spool_path.is_file():
                    raise RuntimeError(f"missing GPTQ spool bundle: {spool_path}")
                with safe_open(spool_path, framework="pt", device="cpu") as packed:
                    writer.add_bundle({key: packed.get_tensor(key) for key in packed.keys()})
            else:
                writer.add(name, source_handle.get_tensor(name))
    manifest = writer.finish()
    total_size = sum(item["bytes"] for item in manifest["shards"])
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": manifest["weight_map"]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"output_tensor_bytes": total_size, "shards": manifest["shards"]}


def run_conversion(args, corpus, run_index: int, destination: Path) -> dict[str, Any]:
    started = time.time()
    spool = args.work_dir / f"run-{run_index}" / "packed"
    metrics = calibrate_and_spool(
        args.source,
        args.model_code_root,
        corpus,
        spool,
        device=args.device,
        block_size=args.block_size,
        damp_percent=args.damp_percent,
        policy=args.policy,
    )
    assembled = assemble_output(args.source, destination, spool, args.max_shard_gib, args.policy)
    return {
        "run": run_index,
        "output": str(destination),
        "elapsed_seconds": time.time() - started,
        "layer_metrics": metrics,
        **assembled,
    }


def main() -> None:
    args = parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.repro_output = args.repro_output.expanduser().resolve() if args.repro_output else None
    args.work_dir = args.work_dir.expanduser().resolve()
    args.model_code_root = args.model_code_root.expanduser().resolve()
    if args.single_run == (args.repro_output is not None):
        raise SystemExit("choose exactly one of --single-run or --repro-output")
    if args.repro_output is not None:
        if args.output == args.repro_output:
            raise SystemExit("--output and --repro-output must differ")
        if args.repro_output.exists():
            raise SystemExit(f"reproducibility output already exists: {args.repro_output}")
    if args.skip_payload_hash_verification and not args.preflight_only:
        raise SystemExit("production conversion cannot skip replay payload hash verification")

    items, estimated = inventory(args.source, args.policy)
    calibration_all = inspect_replay_corpus(
        args.calibration_replay,
        verify_payload_hashes=not args.skip_payload_hash_verification,
    )
    evaluation_all = inspect_replay_corpus(
        args.evaluation_replay,
        verify_payload_hashes=not args.skip_payload_hash_verification,
    )
    selection_path = args.corpus_selection.expanduser().resolve()
    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    calibration = select_replay_corpus(calibration_all, selection_manifest, split="calibration")
    evaluation = select_replay_corpus(evaluation_all, selection_manifest, split="evaluation")
    assert_disjoint_corpora(calibration, evaluation)
    if not 3 <= calibration.conversations <= 5:
        raise SystemExit(
            "GPTQ calibration must contain 3-5 captured conversations; "
            f"found {calibration.conversations}"
        )
    if calibration.calls < 300:
        raise SystemExit(
            f"GPTQ calibration must contain at least 300 Nano calls; found {calibration.calls}"
        )
    if evaluation.conversations < 1:
        raise SystemExit("GPTQ evaluation corpus is empty")
    disk = conversion_disk_preflight(
        args.source,
        args.output,
        args.work_dir,
        estimated_output_bytes=estimated,
        reproducibility_runs=2,
        min_remaining_gib=args.minimum_remaining_gib,
    )
    source_config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    provenance = {
        "repository": source_config.get("voicechat_checkpoint_repository"),
        "revision": source_config.get("voicechat_checkpoint_revision"),
        "checkpoint_sha256": source_config.get("voicechat_checkpoint_sha256"),
        "derived_from_ea": source_config.get("voicechat_derived_from_ea"),
        "nano_sha256": sha256_file(args.source / "model.safetensors"),
    }
    if provenance["repository"] != PUBLIC_REPOSITORY or provenance["revision"] != PUBLIC_REVISION:
        raise SystemExit(f"source is not the pinned exact-public revision: {provenance}")
    if provenance["derived_from_ea"] is not False:
        raise SystemExit("EA-derived source is forbidden")
    for label, corpus in (("calibration", calibration), ("evaluation", evaluation)):
        if corpus.provenance.get("nano_sha256") != provenance["nano_sha256"]:
            raise SystemExit(
                f"{label} replay Nano hash does not match exact source component: "
                f"{corpus.provenance.get('nano_sha256')} != {provenance['nano_sha256']}"
            )

    base_report = {
        "schema": 1,
        "status": "preflight_only" if args.preflight_only else "converting",
        "method": method_for_policy(args.policy),
        "runtime": runtime_identity(),
        "calibration_engine": {
            "implementation": "in-tree classic GPTQ second-order error propagation",
            "external_gptq_package": None,
            "reason": (
                "release image has ModelOpt/compressed-tensors packing but no calibrated "
                "GPTQ driver; current GPTQModel requires Transformers >=5.4 while this "
                "public runtime is pinned to the release's Transformers 4.56 stack"
            ),
            "model_code": str(args.model_code_root),
            "mamba_fused_fast_path": False,
            "mamba_fast_path_reason": (
                "nn.Linear hooks must observe exact in_proj/out_proj activations"
            ),
            "true_sequential": True,
            "dead_column_policy": "isolated_quantize_preserve_weight",
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
        },
        "quantization_config": gptq_quantization_config(args.policy),
        "policy": args.policy,
        "corpus_selection": selection_manifest,
        "source": str(args.source),
        "source_provenance": provenance,
        "selected_tensor_count": sum(item["quantized"] for item in items),
        "selected_source_bytes": sum(item["source_bytes"] for item in items if item["quantized"]),
        "quantized_output_heads": ["lm_head"],
        "function_head_full_precision": True,
        "embeddings_full_precision": True,
        "attention_projections_full_precision": True,
        "shape_exception": {
            "selector": "stt_model.llm.layers.*.mixer.down_proj.weight",
            "shape": [4480, 15680],
            "requested_group_size": 128,
            "effective_group_size": 128,
            "reason": "zero-pad K by 64 so all 105 selected tensors remain group-128",
            "runtime_padding": True,
            "padded_shape": [4480, 15744],
            "other_group_size_fallbacks_allowed": False,
        },
        "inventory": items,
        "estimated_output_tensor_bytes": estimated,
        "disk_preflight": disk,
        "calibration": {
            "roots": [str(root) for root in calibration.roots],
            "calls": calibration.calls,
            "conversations": calibration.conversations,
            "effective_rows_lower_bound": calibration.effective_rows,
            "manifest_sha256": calibration.manifest_sha256,
            "input": "captured exact combined_embeds; no tokenization",
            "records": list(calibration.records),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
        },
        "evaluation": {
            "roots": [str(root) for root in evaluation.roots],
            "calls": evaluation.calls,
            "conversations": evaluation.conversations,
            "manifest_sha256": evaluation.manifest_sha256,
            "disjoint_from_calibration": True,
            "records": list(evaluation.records),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in base_report.items() if key != "inventory"}, indent=2
        )
    )
    if args.preflight_only:
        return

    args.work_dir.mkdir(parents=True, exist_ok=False)
    if args.single_run:
        try:
            run = run_conversion(args, calibration, 1, args.output)
        except Exception as error:
            base_report.update(
                {
                    "status": "conversion_failed",
                    "failure": {
                        "run": 1,
                        "type": type(error).__name__,
                        "message": str(error),
                        "partial_metrics": str(args.work_dir / "run-1" / "metrics.jsonl"),
                    },
                }
            )
            args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
            raise
        base_report.update(
            {
                "status": "complete",
                "runs": [run],
                "reproducibility": {
                    "mode": "external_clean_process_pair",
                    "verified": False,
                },
            }
        )
        args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "complete", "run": 1}, indent=2))
        return

    runs = []
    for run_index, destination in ((1, args.output), (2, args.repro_output)):
        try:
            runs.append(run_conversion(args, calibration, run_index, destination))
        except Exception as error:
            base_report.update(
                {
                    "status": "conversion_failed",
                    "runs": runs,
                    "failure": {
                        "run": run_index,
                        "type": type(error).__name__,
                        "message": str(error),
                        "partial_metrics": str(
                            args.work_dir / f"run-{run_index}" / "metrics.jsonl"
                        ),
                    },
                }
            )
            args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
            raise
        base_report.update(
            {
                "status": "reproducibility_pending"
                if run_index == 1
                else "verifying_reproducibility",
                "runs": runs,
            }
        )
        args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
    try:
        reproducibility = assert_byte_identical(args.output, args.repro_output)
    except Exception as error:
        base_report.update(
            {
                "status": "reproducibility_failed",
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
        raise
    base_report.update(
        {
            "status": "complete",
            "runs": runs,
            "reproducibility": reproducibility,
        }
    )
    args.report.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "reproducibility": reproducibility}, indent=2))


if __name__ == "__main__":
    main()
