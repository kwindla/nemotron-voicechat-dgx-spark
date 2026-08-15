#!/usr/bin/env python3
"""Device-side worker for :mod:`step5_phase1_real_executor`.

This module is deliberately import-light.  The qualified vLLM overlay is
assembled and selected before vLLM, NeMo, or the runtime server is imported.
Tensor-heavy replies are written with ``torch.save``; stdout is reserved for
one-line protocol acknowledgements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from step5_phase1_campaign import (  # noqa: E402
    CaseSpec,
    ExecutionResult,
    JsonValue,
    MaterializedFixture,
    MaterializedFrame22State,
    OraclePlan,
    StateSnapshot,
    TransactionKind,
    case_matrix,
    plan_for_case,
    stable_digest,
)

from nemotron_voicechat_runtime.runtime_optimizations import (  # noqa: E402
    _pad_pair_state,
    correct_add_fusion_pending,
)
from nemotron_voicechat_runtime.wedge_boundary_trace import (  # noqa: E402
    BOUNDARIES,
    RECORD_BYTES,
    record_boundary,
)

PAD_TOKEN_ID = 12
FRAME_SAMPLES = 1280
PCM_DISCARDED_PREFIX_SAMPLES = 304


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(path.encode() + b"\0")
            digest.update(str(item.dtype).encode() + b"\0")
            digest.update(str(tuple(item.shape)).encode() + b"\0")
            digest.update(_tensor_bytes(item))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            digest.update(path.encode() + b"\0" + repr(item).encode() + b"\n")

    visit(value, "root")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
        try:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return digest.hexdigest()


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"qualified tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"qualified tree contains a non-regular entry: {path}")
        inventory[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    if not inventory:
        raise RuntimeError(f"qualified tree is empty: {root}")
    return inventory


def _inventory_sha256(inventory: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative, seal in inventory.items():
        digest.update(relative.encode() + b"\0")
        digest.update(str(seal["bytes"]).encode() + b"\0")
        digest.update(str(seal["sha256"]).encode() + b"\n")
    return digest.hexdigest()


def _verify_used_runtime_identity(config: dict[str, Any]) -> None:
    """Reverify repository bytes used by this child before importing the runtime."""
    expected = {
        REPO_ROOT / "reports/step5-wedge/phase1-real-executor-config.json": config[
            "configuration_sha256"
        ],
        TOOLS_ROOT / "step5_phase1_real_executor.py": config["executor_source_sha256"],
        TOOLS_ROOT / "step5_phase1_gpu_worker.py": config["worker_source_sha256"],
        REPO_ROOT / "src/nemotron_voicechat_runtime/runtime_optimizations.py": config[
            "runtime_optimizations_source_sha256"
        ],
        REPO_ROOT / "src/nemotron_voicechat_runtime/pocket_worker.py": config[
            "pocket_worker_source_sha256"
        ],
        TOOLS_ROOT / "step5_phase1_typed_carrier.py": config[
            "typed_carrier_source_sha256"
        ],
        REPO_ROOT / "src/nemotron_voicechat_runtime/server.py": config["server_source_sha256"],
        Path(config["patch_manifest"]): config["patch_manifest_sha256"],
    }
    for path, expected_sha256 in expected.items():
        if _file_sha256(path.resolve()) != expected_sha256:
            raise RuntimeError(f"used runtime source differs from parent seal: {path}")
    observed_environment = {
        key: os.environ.get(key) for key in sorted(config["worker_environment"])
    }
    if observed_environment != config["worker_environment"]:
        raise RuntimeError("child semantic environment differs from the parent seal")
    if stable_digest(observed_environment) != config["worker_environment_sha256"]:
        raise RuntimeError("child semantic environment digest differs from the parent seal")


def _observed_model_artifact_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Rehash every model/config artifact that this child can load."""
    manifest_path = Path(config["vllm_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise RuntimeError("vLLM model manifest lacks components")
    observed: dict[str, Any] = {}
    for name, configured_root in (
        ("nano", config["nano_vllm_path"]),
        ("eartts", config["eartts_vllm_path"]),
    ):
        root = Path(configured_root).resolve()
        component = components.get(name)
        files = component.get("files") if isinstance(component, dict) else None
        model = component.get("model") if isinstance(component, dict) else None
        if not isinstance(files, dict) or not isinstance(model, dict):
            raise RuntimeError(f"model manifest component is incomplete: {name}")
        expected_inventory: dict[str, dict[str, Any]] = {}
        for relative, expected in sorted(files.items()):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(
                    f"used {name} artifact is absent or escapes its root: {relative}"
                )
            expected_inventory[relative] = expected
        inventory = _file_inventory(root)
        if inventory != expected_inventory:
            raise RuntimeError(f"used {name} artifact tree differs from the manifest file set")
        observed[name] = {
            "files": inventory,
            "manifest_model_sha256": model.get("sha256"),
            "used_tree_sha256": _inventory_sha256(inventory),
        }
    source = manifest.get("source")
    expected_parent = source.get("model", {}).get("sha256") if isinstance(source, dict) else None
    checkpoint_root = Path(config["checkpoint_root"]).resolve()
    checkpoint_inventory = _file_inventory(checkpoint_root)
    for relative in ("config.json", "model.safetensors", "rnnt_tokenizer/tokenizer.model"):
        if relative not in checkpoint_inventory:
            raise RuntimeError(f"used parent checkpoint artifact is absent: {relative}")
    parent_sha256 = checkpoint_inventory["model.safetensors"]["sha256"]
    if parent_sha256 != expected_parent:
        raise RuntimeError("used parent checkpoint differs from the model manifest")
    speech_root = Path(config["speech_root"]).resolve()
    speech_runtime_files: dict[str, dict[str, Any]] = {}
    for relative in (
        "examples/speechlm2/function_calling/template.jinja",
        "examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
    ):
        path = (speech_root / relative).resolve()
        if not path.is_relative_to(speech_root) or not path.is_file():
            raise RuntimeError(f"used Speech runtime artifact is absent: {relative}")
        speech_runtime_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    identity = {
        "checkpoint_files": checkpoint_inventory,
        "checkpoint_tree_sha256": _inventory_sha256(checkpoint_inventory),
        "eartts_model_sha256": observed["eartts"]["used_tree_sha256"],
        "hf_skeleton_tree_sha256": _inventory_sha256(
            _file_inventory(Path(config["hf_skeleton"]).resolve())
        ),
        "manifest_sha256": _file_sha256(manifest_path),
        "nano_model_sha256": observed["nano"]["used_tree_sha256"],
        "parent_model_sha256": parent_sha256,
        "speech_runtime_files": speech_runtime_files,
        "verified_files": observed,
    }
    identity["model_sha256"] = stable_digest(identity)
    return identity


def _verify_model_artifact_identity(config: dict[str, Any]) -> dict[str, Any]:
    observed = _observed_model_artifact_identity(config)
    if observed != config.get("model_artifact_identity"):
        raise RuntimeError("used model artifacts differ from the parent seal")
    if observed["model_sha256"] != config.get("model_sha256"):
        raise RuntimeError("used model artifact digest differs from the parent seal")
    return observed


def _remember_request_blocks(runner: Any) -> None:
    """Retain the small request-to-block mapping across idle input-batch cleanup."""
    retained = getattr(runner, "_phase1_request_blocks", None)
    if not isinstance(retained, dict):
        retained = {}
        runner._phase1_request_blocks = retained
    batch = runner.input_batch
    for request_id, request_index in batch.req_id_to_index.items():
        groups: dict[int, list[int]] = {}
        for group_index in range(len(runner.kv_cache_config.kv_cache_groups)):
            table = batch.block_table[group_index]
            count = int(table.num_blocks_per_row[request_index])
            groups[group_index] = [
                int(value) for value in table.get_numpy_array()[request_index, :count]
            ]
        retained[request_id] = groups


def _read_ring(path: Path) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if size == 0 or size % RECORD_BYTES:
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            for offset in range(0, size, RECORD_BYTES):
                raw = mapping[offset : offset + RECORD_BYTES].rstrip(b"\0")
                if not raw:
                    break
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    return records


def boundary_records(trace_dir: Path, request_id: str) -> list[dict[str, Any]]:
    records = [
        record
        for path in sorted(trace_dir.glob("*.ring"))
        for record in _read_ring(path)
        if record.get("request_id") == request_id
    ]
    return sorted(
        records,
        key=lambda item: (int(item.get("monotonic_ns", 0)), item.get("boundary")),
    )


def read_boundary_prefix(trace_dir: Path, request_id: str) -> tuple[str, ...]:
    """Return the latest transaction's canonical observed B0--B9 prefix."""
    records = boundary_records(trace_dir, request_id)
    starts = [index for index, item in enumerate(records) if item.get("boundary") == "B0"]
    if not starts:
        return ()
    observed = {item.get("boundary") for item in records[starts[-1] :]}
    return tuple(boundary for boundary in BOUNDARIES if boundary in observed)


def latest_transaction_records(trace_dir: Path, request_id: str) -> list[dict[str, Any]]:
    records = boundary_records(trace_dir, request_id)
    starts = [index for index, item in enumerate(records) if item.get("boundary") == "B0"]
    return [] if not starts else records[starts[-1] :]


def observed_dispatch(trace_dir: Path, request_id: str) -> tuple[str, dict[str, Any]]:
    """Derive dispatch solely from the latest transaction's B2/B3 records."""
    records = latest_transaction_records(trace_dir, request_id)
    b2 = [item for item in records if item.get("boundary") == "B2"]
    b3 = [item for item in records if item.get("boundary") == "B3"]
    if len(b2) != 1 or len(b3) != 1:
        raise RuntimeError("latest transaction lacks unique B2/B3 dispatch evidence")
    if b2[0].get("truncated") or b3[0].get("truncated"):
        raise RuntimeError("latest B2/B3 dispatch evidence was truncated")
    b2_details = b2[0].get("details")
    b3_details = b3[0].get("details")
    if not isinstance(b2_details, dict) or not isinstance(b3_details, dict):
        raise RuntimeError("latest B2/B3 records lack observed detail mappings")
    graph_mode = b3_details.get("graph_mode")
    selector = b3_details.get("phase1_pair_eager_selector")
    if not isinstance(selector, bool) or not isinstance(graph_mode, str):
        raise RuntimeError("latest B3 lacks graph-mode/selector evidence")
    graph_is_none = graph_mode in {"NONE", "CUDAGraphMode.NONE"}
    if selector and graph_is_none:
        mode = "pair-eager"
    elif not selector:
        mode = "normal"
    else:
        raise RuntimeError(
            "B3 selector fired without the required eager graph mode: "
            f"selector={selector!r} graph_mode={graph_mode!r}"
        )
    return mode, {
        "b2_descriptor_key": b2_details.get("descriptor_key"),
        "b3_descriptor_key": b3_details.get("descriptor_key"),
        "graph_mode": graph_mode,
        "selector_fired": selector,
    }


def _copy_qualified_overlay(config: dict[str, Any]) -> Path:
    """Copy only the sealed atomic patched tree and verify every output path."""
    import importlib.util

    patched = Path(config["patched_vllm_root"]).resolve()
    before = _file_inventory(patched)
    expected_count = int(config["patched_vllm_file_count"])
    expected_digest = str(config["patched_vllm_tree_sha256"])
    if len(before) != expected_count or _inventory_sha256(before) != expected_digest:
        raise RuntimeError("atomic patched tree differs from the parent seal")
    overlay_parent = Path(tempfile.mkdtemp(prefix="step5-phase1-vllm-overlay-"))
    overlay = overlay_parent / "vllm"
    shutil.copytree(patched, overlay, symlinks=False)
    after_source = _file_inventory(patched)
    copied = _file_inventory(overlay)
    if after_source != before:
        raise RuntimeError("atomic patched tree changed while the child copied it")
    if copied != before or _inventory_sha256(copied) != expected_digest:
        raise RuntimeError("qualified child tree has missing, extra, or changed files")
    sys.path.insert(0, str(overlay_parent))
    selected = importlib.util.find_spec("vllm")
    if selected is None or selected.origin is None:
        raise RuntimeError("qualified vLLM overlay is not importable")
    if Path(selected.origin).resolve().parent != overlay:
        raise RuntimeError("Python did not select the qualified vLLM overlay")
    runner = overlay / "v1/worker/gpu_model_runner.py"
    if _file_sha256(runner) != config["patched_runner_sha256"]:
        raise RuntimeError("qualified child selected the wrong patched runner")
    if _file_sha256(Path(config["patch_manifest"])) != config["patch_manifest_sha256"]:
        raise RuntimeError("qualified child patch manifest differs from the parent seal")
    return overlay


def install_phase1_forcing(worker: Any) -> dict[str, Any]:
    """Install request-scoped diagnostic token forcing outside CUDA graphs."""
    runner = worker.model_runner
    if getattr(runner, "_phase1_forcing_installed", False):
        return {"installed": True, "reused": True}
    original_forward = runner._model_forward
    original_logits = runner.model.compute_logits

    def forced_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        _remember_request_blocks(runner)
        targets = getattr(runner, "_phase1_function_targets", None)
        names = list(getattr(runner.model_config, "custom_outputs", None) or [])
        if targets is None or "function_tokens" not in names or not isinstance(output, tuple):
            return output
        # The qualified Nano is not a pipeline-parallel intermediate stage;
        # custom outputs follow hidden_states in configured order.
        index = 1 + names.index("function_tokens")
        if index >= len(output) or not isinstance(output[index], torch.Tensor):
            raise RuntimeError("Phase-1 cannot locate function_tokens in model output")
        values = output[index].clone()
        flat = values.reshape(-1)
        if len(targets) < flat.numel():
            raise RuntimeError("Phase-1 function target vector is too short")
        flat.copy_(torch.tensor(targets[: flat.numel()], device=flat.device, dtype=flat.dtype))
        changed = list(output)
        changed[index] = values
        return tuple(changed)

    def forced_logits(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        logits = original_logits(hidden_states, *args, **kwargs)
        targets = getattr(runner, "_phase1_text_targets", None)
        if targets is None:
            return logits
        forced = torch.full_like(logits, torch.finfo(logits.dtype).min)
        rows = forced.reshape(-1, forced.shape[-1])
        if len(targets) < rows.shape[0]:
            raise RuntimeError("Phase-1 text target vector is too short")
        for index in range(rows.shape[0]):
            rows[index, int(targets[index])] = 0
        return forced

    runner._model_forward = forced_forward
    runner.model.compute_logits = forced_logits
    runner._phase1_forcing_installed = True
    runner._phase1_text_targets = None
    runner._phase1_function_targets = None
    return {"installed": True, "reused": False}


def set_phase1_targets(worker: Any, text: list[int], function: list[int]) -> dict[str, Any]:
    runner = worker.model_runner
    if not getattr(runner, "_phase1_forcing_installed", False):
        raise RuntimeError("Phase-1 forcing hook is not installed")
    runner._phase1_text_targets = [int(item) for item in text]
    runner._phase1_function_targets = [int(item) for item in function]
    return {"text": list(text), "function": list(function)}


def clear_phase1_targets(worker: Any) -> dict[str, bool]:
    """Disable diagnostic logits forcing for exact system-prompt prefill."""
    runner = worker.model_runner
    if not getattr(runner, "_phase1_forcing_installed", False):
        raise RuntimeError("Phase-1 forcing hook is not installed")
    runner._phase1_text_targets = None
    runner._phase1_function_targets = None
    return {"cleared": True}


def capture_worker_state(worker: Any, request_id: str, output_path: str) -> dict[str, Any]:
    """Snapshot live request-owned KV/Mamba and rollback state in the GPU worker."""
    runner = worker.model_runner
    _remember_request_blocks(runner)
    layer_groups: dict[str, int] = {}
    for group_index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for layer_name in group.layer_names:
            layer_groups[layer_name] = group_index
    block_ids = getattr(runner, "_phase1_request_blocks", {}).get(request_id)
    if not isinstance(block_ids, dict) or not block_ids:
        raise RuntimeError(f"Phase-1 request block ownership is absent: {request_id}")
    request_state = getattr(runner, "requests", {}).get(request_id)
    committed_tokens = getattr(request_state, "num_computed_tokens", None)
    if isinstance(committed_tokens, bool) or not isinstance(committed_tokens, int):
        raise RuntimeError(f"Phase-1 committed token boundary is absent: {request_id}")
    if committed_tokens <= 0:
        raise RuntimeError(f"Phase-1 committed token boundary is invalid: {committed_tokens}")

    groups: dict[str, dict[str, torch.Tensor]] = {
        "kv": {},
        "mamba_conv": {},
        "mamba_ssm": {},
        "shadow": {},
    }
    shadow_ordinals: set[int] = set()
    shadow_locations: dict[str, dict[str, int]] = {}
    for module_name, module in runner.model.named_modules():
        group_index = layer_groups.get(module_name)
        for shadow_name in ("voicechat_pad_pair_shadow_conv", "voicechat_pad_pair_shadow_ssm"):
            shadow = getattr(module, shadow_name, None)
            if isinstance(shadow, torch.Tensor) and shadow.numel():
                groups["shadow"][f"{module_name}.{shadow_name}"] = shadow.detach().cpu().clone()
        index = getattr(module, "voicechat_pad_pair_shadow_index", None)
        if index is not None:
            if group_index is None or group_index not in block_ids:
                raise RuntimeError(
                    f"rollback shadow lacks a request-owned cache group: {module_name}"
                )
            physical_index = int(index.reshape(-1)[0])
            selected_ids = block_ids[group_index]
            if physical_index not in selected_ids:
                raise RuntimeError(
                    "rollback shadow index is not owned by the live request: "
                    f"{module_name} index={physical_index} blocks={selected_ids}"
                )
            request_block_ordinal = selected_ids.index(physical_index)
            shadow_ordinals.add(request_block_ordinal)
            shadow_locations[module_name] = {
                "cache_group": group_index,
                "physical_index": physical_index,
                "request_block_ordinal": request_block_ordinal,
            }
        cache = getattr(module, "kv_cache", None)
        if cache is None:
            continue
        if group_index is None:
            continue
        selected_ids = block_ids[group_index]
        module_is_mamba = "mamba" in (module_name + type(module).__name__).lower()
        stack = [(module_name, cache, None)]
        while stack:
            name, state, mamba_slot = stack.pop()
            if isinstance(state, torch.Tensor) and state.numel():
                maximum = max(selected_ids)
                if state.ndim > 1 and state.shape[0] in (1, 2) and maximum < state.shape[1]:
                    selected = state[:, selected_ids]
                elif maximum < state.shape[0]:
                    selected = state[selected_ids]
                else:
                    raise RuntimeError(
                        f"cannot locate request block axis: {name} {tuple(state.shape)}"
                    )
                lower = name.lower()
                category = (
                    "mamba_conv"
                    if "conv" in lower or (module_is_mamba and mamba_slot == 0)
                    else "mamba_ssm"
                    if "ssm" in lower or (module_is_mamba and mamba_slot == 1)
                    else "kv"
                )
                if category == "kv":
                    if selected.ndim < 3 or selected.shape[0] not in (1, 2):
                        raise RuntimeError(
                            f"unsupported KV block representation: {name} {tuple(selected.shape)}"
                        )
                    selected = selected.flatten(1, 2)
                    if committed_tokens > selected.shape[1]:
                        raise RuntimeError(
                            "committed KV boundary exceeds owned cache capacity: "
                            f"{name} tokens={committed_tokens} capacity={selected.shape[1]}"
                        )
                    # Compare logical committed tokens, never unused bytes in the
                    # request-owned cache block tail.
                    selected = selected[:, :committed_tokens]
                groups[category][name] = selected.detach().cpu().clone()
            elif isinstance(state, (list, tuple)):
                for index, child in enumerate(state):
                    stack.append((f"{name}[{index}]", child, index if module_is_mamba else None))
    if any(not groups[name] for name in ("kv", "mamba_conv", "mamba_ssm")):
        raise RuntimeError(
            "Phase-1 did not find complete KV/Mamba state: "
            + json.dumps({name: len(value) for name, value in groups.items()}, sort_keys=True)
        )
    if not groups["shadow"] or not shadow_locations:
        raise RuntimeError("Phase-1 true rollback shadow/index is absent")
    if len(shadow_ordinals) != 1:
        raise RuntimeError(
            "rollback modules disagree on the request-relative shadow index: "
            + json.dumps(shadow_locations, sort_keys=True)
        )
    shadow_index = next(iter(shadow_ordinals))
    payload = {
        "committed_token_count": committed_tokens,
        "committed_kv_state": groups["kv"],
        "committed_mamba_conv_state": groups["mamba_conv"],
        "committed_mamba_ssm_state": groups["mamba_ssm"],
        "rollback_shadow": groups["shadow"],
        "shadow_index": shadow_index,
        "shadow_physical_locations": shadow_locations,
        "virtual_engine_identity": stable_digest(
            {
                "engine_index": getattr(runner, "engine_index", 0),
                "model": str(getattr(runner, "model_path", "")),
                "kv_cache_groups": len(runner.kv_cache_config.kv_cache_groups),
            }
        ),
    }
    torch.save(payload, output_path)
    return {"path": output_path, "counts": {name: len(value) for name, value in groups.items()}}


def correct_fused_row(
    worker: Any,
    row_path: str,
    assumed_text: int,
    effective_text: int,
    assumed_function: int,
    effective_function: int,
    output_path: str,
) -> dict[str, Any]:
    """Apply AddFusion token deltas with the real Nano embedding table."""
    row = torch.load(row_path, map_location="cpu", weights_only=False)
    if not isinstance(row, torch.Tensor):
        raise RuntimeError("Phase-1 corrected-row input is not a tensor")
    model = worker.model_runner.model
    embed_tokens = None
    for candidate in (
        getattr(model, "embed_tokens", None),
        getattr(getattr(model, "model", None), "embed_tokens", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "embed_tokens", None),
    ):
        if callable(candidate):
            embed_tokens = candidate
            break
    if embed_tokens is None:
        raise RuntimeError("Phase-1 cannot locate the qualified Nano embedding table")
    corrected = correct_add_fusion_pending(
        row.to(device=worker.device),
        embed_tokens=embed_tokens,
        assumed_text=assumed_text,
        effective_text=effective_text,
        assumed_function=assumed_function,
        effective_function=effective_function,
        agent_text_weight=1.0,
        function_weight=2.0,
    )
    torch.save(corrected.detach().cpu(), output_path)
    return {"path": output_path, "sha256": _tensor_sha256(corrected)}


def export_embedding_row(worker: Any, token_id: int, output_path: str) -> dict[str, Any]:
    """Export one real model embedding for a non-scientific transport smoke."""
    model = worker.model_runner.model
    embed_tokens = None
    for candidate in (
        getattr(model, "embed_tokens", None),
        getattr(getattr(model, "model", None), "embed_tokens", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "embed_tokens", None),
    ):
        if callable(candidate):
            embed_tokens = candidate
            break
    if embed_tokens is None:
        raise RuntimeError("Phase-1 smoke cannot locate the qualified Nano embedding table")
    token = torch.tensor([int(token_id)], device=worker.device, dtype=torch.long)
    row = embed_tokens(token).detach().to(dtype=torch.bfloat16).cpu()
    if row.shape[-1] != 4480:
        raise RuntimeError(f"Phase-1 smoke embedding is not 4480-wide: {tuple(row.shape)}")
    torch.save(row, output_path)
    return {"path": output_path, "sha256": _tensor_sha256(row)}


def _pack_group(group: dict[str, torch.Tensor], label: str) -> torch.Tensor:
    """Make one byte-exact tensor while retaining leaf metadata separately."""
    if not group:
        raise RuntimeError(f"cannot pack empty {label} group")
    chunks = [
        value.contiguous().view(torch.uint8).reshape(-1) for _, value in sorted(group.items())
    ]
    return torch.cat(chunks)


def _leaf_metadata(group: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "byte_count": len(_tensor_bytes(value)),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
        for key, value in sorted(group.items())
    }


def _canonical_rollback_projection(raw_shadow: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Project physical pair-shadow buffers onto committed Mamba leaf names/layouts."""
    conv: dict[str, torch.Tensor] = {}
    ssm: dict[str, torch.Tensor] = {}
    for name, value in raw_shadow.items():
        if name.endswith(".voicechat_pad_pair_shadow_conv"):
            key = name.removesuffix(".voicechat_pad_pair_shadow_conv") + "[0][0]"
            conv[key] = value.transpose(-1, -2).unsqueeze(0).contiguous()
        elif name.endswith(".voicechat_pad_pair_shadow_ssm"):
            key = name.removesuffix(".voicechat_pad_pair_shadow_ssm") + "[0][1]"
            ssm[key] = value.unsqueeze(0).contiguous()
        else:
            raise RuntimeError(f"unknown rollback-shadow leaf: {name}")
    if not conv or not ssm:
        raise RuntimeError("rollback projection lacks Mamba conv/SSM leaves")
    return {
        "committed_mamba_conv_state": conv,
        "committed_mamba_ssm_state": ssm,
    }


def _committed_mamba_projection(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "committed_mamba_conv_state": raw["committed_mamba_conv_state"],
        "committed_mamba_ssm_state": raw["committed_mamba_ssm_state"],
    }


def _function_state(turn_state: dict[str, Any]) -> str:
    state = turn_state.get("function_calling")
    if not isinstance(state, dict):
        raise RuntimeError("frame-22 function state is absent")
    active = {
        key: state.get(key)
        for key in ("active", "awaiting_response", "injecting_response", "awaiting_eotr")
    }
    if any(active.values()) or int(state.get("forced_tokens", 0)):
        return "active"
    return "idle"


def _runtime_recorded_assumptions(
    prefix: dict[str, Any], frame: int, pad_state: dict[str, Any] | None = None
) -> tuple[int, int]:
    """Return the wrapper-recorded AddFusion predecessor identity for a row."""
    assumptions = prefix.get("assumptions")
    value = assumptions.get(frame) if isinstance(assumptions, dict) else None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"frame {frame} lacks valid runtime-recorded assumptions")
    observed = (int(value[0]), int(value[1]))
    if pad_state is not None and pad_state.get("assumed") != observed:
        raise RuntimeError("runtime-recorded correction assumptions differ from materialization")
    return observed


class GPUCampaignRuntime:
    def __init__(self, config: dict[str, Any], root: Path) -> None:
        self.config = config
        self.root = root
        _verify_used_runtime_identity(config)
        self.overlay = _copy_qualified_overlay(config)
        self.nano: Any | None = None
        self.pipeline: Any | None = None
        self.control_engine: Any | None = None
        self.rows: list[torch.Tensor] = []
        self.generated_counts: list[int] = []
        self.lanes: dict[str, dict[str, Any]] = {}
        self.trace_dir = Path(os.environ["NEMOTRON_WEDGE_TRACE_DIR"])

    def construct(self, mode: str) -> dict[str, Any]:
        if mode == "materializer":
            return {"mode": mode, "overlay": str(self.overlay), "lazy_pipeline": True}
        if mode != "epoch":
            raise ValueError(f"unknown construction mode: {mode}")
        self.pipeline = self._build_pipeline()
        self.nano = self.pipeline.s2s_model.model_llm_interface
        self._start_control_engine()
        self._enable_phase1_pairing()
        self._collective(install_phase1_forcing)
        return self._construction_metadata("full-pipeline-epoch")

    def _build_nano(self) -> Any:
        _verify_model_artifact_identity(self.config)
        self._enable_phase1_pairing()
        sys.path.insert(0, str(Path(self.config["speech_root"]).resolve()))
        from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model

        return create_model(
            engine_type="vllm_llm",
            vllm_config={
                "model_path": self.config["nano_vllm_path"],
                "engine_path": self.config["nano_vllm_path"],
                "max_model_len": 12288,
                "gpu_memory_utilization": float(self.config["gpu_memory_utilization"]),
                "dtype": "bfloat16",
                "enforce_eager": False,
            },
            top_p=1.0,
            repetition_penalty=1.0,
            temperature=0.0,
            text_pad_id=PAD_TOKEN_ID,
        )

    @staticmethod
    def _enable_phase1_pairing() -> None:
        """Enable the diagnostic intervention only after production validation."""
        os.environ["VOICECHAT_NANO_PAD_PAIR"] = "1"
        os.environ["VOICECHAT_NANO_PAD_PAIR_CONDITIONAL"] = "1"
        os.environ["VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER"] = "1"
        from nemotron_voicechat_runtime.runtime_optimizations import (
            _install_public_pad_pair_engine,
            install_public_wrapper_hooks,
        )

        install_public_wrapper_hooks()
        _install_public_pad_pair_engine()

    def _collective(self, function: Any, *args: Any) -> Any:
        if self.nano is None:
            raise RuntimeError("Nano engine is not constructed")
        return self.nano._run_async(self.nano.engine.engine.collective_rpc(function, args=args))[0]

    def _set_targets(self, text: list[int], function: list[int]) -> None:
        self._collective(set_phase1_targets, text, function)

    def _capture(self, request_id: str, label: str) -> dict[str, Any]:
        path = self.root / f"state-{label}.pt"
        self._collective(capture_worker_state, request_id, str(path))
        return torch.load(path, map_location="cpu", weights_only=False)

    def _correct_row(
        self,
        row: torch.Tensor,
        case: CaseSpec,
        label: str,
        assumptions: tuple[int, int],
    ) -> torch.Tensor:
        input_path = self.root / f"uncorrected-{label}.pt"
        path = self.root / f"corrected-{label}.pt"
        torch.save(row.detach().cpu(), input_path)
        self._collective(
            correct_fused_row,
            str(input_path),
            assumptions[0],
            case.text_token,
            assumptions[1],
            case.function_token,
            str(path),
        )
        return torch.load(path, map_location="cpu", weights_only=False)

    def _construction_metadata(self, kind: str) -> dict[str, Any]:
        return {
            "gpu_memory_utilization": float(self.config["gpu_memory_utilization"]),
            "kind": kind,
            "overlay": str(self.overlay),
            "patch_manifest_sha256": hashlib.sha256(
                Path(self.config["patch_manifest"]).read_bytes()
            ).hexdigest(),
            "patched_runner_sha256": hashlib.sha256(
                (self.overlay / "v1/worker/gpu_model_runner.py").read_bytes()
            ).hexdigest(),
            "schema": 1,
        }

    def _build_pipeline(self) -> Any:
        _verify_model_artifact_identity(self.config)
        from nemotron_voicechat_runtime import server

        os.environ["VOICECHAT_VLLM_NANO_MEMORY_UTILIZATION"] = str(
            self.config["gpu_memory_utilization"]
        )
        args = SimpleNamespace(
            checkpoint_root=self.config["checkpoint_root"],
            checkpoint_revision="fb0f94eaf4d03ddc430f39565229393fa1b50c26",
            checkpoint_sha256="d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b",
            eartts_vllm_path=self.config["eartts_vllm_path"],
            hf_skeleton=self.config["hf_skeleton"],
            nano_vllm_path=self.config["nano_vllm_path"],
            speaker_name="Aria",
            speech_root=self.config["speech_root"],
            vllm_manifest=self.config["vllm_manifest"],
        )
        return server.build_public_pipeline(args)

    def materialize(self, fixture: dict[str, Any]) -> dict[str, Any]:
        from nemotron_voicechat_runtime.server import VoiceChatEngine, render_tool_system_prompt

        self.pipeline = self._build_pipeline()
        wrapper = self.pipeline.s2s_model
        self.nano = wrapper.model_llm_interface
        stream = self.nano.engine
        original = stream.generate_next_token
        frame_rows: dict[int, torch.Tensor] = {}
        frame_assumptions: dict[int, tuple[int, int]] = {}
        prefill_calls: list[dict[str, Any]] = []
        capture_frame: int | None = None
        events_path = Path(self.config["source_pcm"]).with_name("events.jsonl")
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        observed_schedule = {
            int(item["frame"]): int(
                item["turn_state"]["vllm_request_positions"]["nano"]["generated_tokens"]
            )
            for item in events
            if item.get("event") == "model_step" and 0 <= int(item.get("frame", -1)) <= 22
        }
        if set(observed_schedule) != set(range(23)):
            raise RuntimeError("frozen runtime events lack the complete frame-0..22 schedule")
        recorded_generated_counts = [observed_schedule[frame] for frame in range(23)]

        async def capture(
            _engine: Any,
            input_tensors: list[torch.Tensor],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            row = input_tensors[0].detach().cpu().clone()
            if capture_frame is None:
                prompt_ids = args[0] if args else kwargs.get("prompt_token_ids")
                prefill_calls.append(
                    {
                        "prompt_token_ids": (
                            None if prompt_ids is None else [int(item) for item in prompt_ids]
                        ),
                        "row": row,
                    }
                )
            else:
                if capture_frame in frame_rows:
                    raise RuntimeError(f"phase1 frame {capture_frame} issued multiple Nano calls")
                frame_rows[capture_frame] = row
                call_context = getattr(stream, "_voicechat_pad_pair_call_context", {})
                assumed = call_context.get("assumed")
                if capture_frame > 0:
                    if (
                        not isinstance(assumed, tuple)
                        or len(assumed) != 2
                        or any(
                            isinstance(item, bool) or not isinstance(item, int)
                            for item in assumed
                        )
                    ):
                        raise RuntimeError(
                            f"phase1 frame {capture_frame} lacks runtime-recorded assumptions"
                        )
                    frame_assumptions[capture_frame] = (int(assumed[0]), int(assumed[1]))
            if capture_frame == 23:
                raise RuntimeError("phase1-frame23-captured")
            if capture_frame is None:
                return await original(input_tensors, *args, **kwargs)
            request_id = kwargs.get("request_id")
            if request_id is None and len(args) > 1:
                request_id = args[1]
            if request_id is None:
                request_id = "speech_stream"
            request = stream.requests.get(str(request_id))
            if request is None:
                raise RuntimeError("observed schedule replay lost the live Nano request")
            generated_before = len(request.generated_tokens)
            expected_after = recorded_generated_counts[capture_frame]
            expected_delta = expected_after - generated_before
            if expected_delta not in (0, 1, 2):
                raise RuntimeError(
                    f"frozen schedule has invalid frame delta at {capture_frame}: "
                    f"{generated_before}->{expected_after}"
                )
            pad_state = _pad_pair_state(stream, str(request_id))
            pad_state["previous_effective_pad"] = expected_delta != 1
            result = await original(input_tensors, *args, **kwargs)
            generated_after = len(request.generated_tokens)
            if generated_after != expected_after:
                raise RuntimeError(
                    f"live scheduler differs from frozen frame {capture_frame}: "
                    f"{generated_after} != {expected_after}"
                )
            return result

        stream.generate_next_token = MethodType(capture, stream)
        retained_pcm_bytes = Path(self.config["source_pcm"]).read_bytes()
        tool_events = [item for item in events if item.get("event") == "session_tools_configured"]
        typed_events = [item for item in events if item.get("event") == "typed_input_accepted"]
        discarded = [
            item
            for item in events
            if item.get("event") == "microphone_partial_discarded_for_typed_input"
        ]
        replay = fixture["reproduction_recipe"].get("typed_carrier_replay")
        if not isinstance(replay, dict):
            raise RuntimeError("frozen fixture lacks the typed-carrier replay contract")
        witness_path = Path(self.config["source_typed_carrier_witness"])
        witness_bytes = witness_path.read_bytes()
        if hashlib.sha256(witness_bytes).hexdigest() != replay.get(
            "historical_witness_sha256"
        ):
            raise RuntimeError("historical typed-carrier witness seal mismatch")
        witness_lines = witness_bytes.decode("utf-8", errors="strict").splitlines()
        carriers: list[bytes] = []
        for index, line in enumerate(witness_lines[:-1]):
            if line.strip() != "await text_box.fill(":
                continue
            quoted = witness_lines[index + 1].strip()
            try:
                candidate = json.loads(quoted)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, str):
                encoded = candidate.encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() == replay.get("utf8_sha256"):
                    carriers.append(encoded)
        if len(carriers) != 1:
            raise RuntimeError(
                f"historical witness yielded {len(carriers)} matching typed carriers"
            )
        typed_carrier_bytes = carriers[0]
        if len(typed_carrier_bytes) != replay.get("byte_count"):
            raise RuntimeError("historical typed-carrier byte count mismatch")
        if (
            len(tool_events) != 1
            or tool_events[0].get("tools") != ["get_current_utc_time"]
            or len(typed_events) != 1
            or typed_events[0].get("text_bytes") != len(typed_carrier_bytes)
            or typed_events[0].get("job_id") != replay.get("historical_event_job_id")
            or len(discarded) != 1
            or discarded[0].get("samples") != PCM_DISCARDED_PREFIX_SAMPLES
        ):
            raise RuntimeError("frozen typed-carrier/tool/discard context is not attested")
        carrier_path = self.root.parent / "historical-typed-carrier.utf8"
        replay_pcm_path = self.root.parent / "typed-carrier-replay.pcm"
        carrier_path.write_bytes(typed_carrier_bytes)
        helper_environment = dict(os.environ)
        helper_environment["HUGGINGFACE_HUB_CACHE"] = str(self.config["pocket_hf_hub"])
        helper_environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src"), helper_environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        synthesis = subprocess.run(
            [
                str(self.config["pocket_python"]),
                str(TOOLS_ROOT / "step5_phase1_typed_carrier.py"),
                "--carrier",
                str(carrier_path),
                "--pcm-output",
                str(replay_pcm_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=helper_environment,
        )
        try:
            synthesis_seal = json.loads(synthesis.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError("Pocket replay helper returned no valid seal") from error
        replay_pcm_bytes = replay_pcm_path.read_bytes()
        if (
            synthesis_seal.get("carrier_sha256") != replay.get("utf8_sha256")
            or synthesis_seal.get("carrier_bytes") != replay.get("byte_count")
            or synthesis_seal.get("pcm_sha256")
            != hashlib.sha256(replay_pcm_bytes).hexdigest()
        ):
            raise RuntimeError("Pocket replay helper seal differs from replay bytes")
        exact_prefix_bytes = int(replay.get("retained_pcm_exact_prefix_bytes", -1))
        retained_after_discard = retained_pcm_bytes[PCM_DISCARDED_PREFIX_SAMPLES * 2 :]
        if exact_prefix_bytes != 24 * FRAME_SAMPLES * 2:
            raise RuntimeError("typed-carrier replay prefix is not the 24-frame contract")
        if (
            len(replay_pcm_bytes) < exact_prefix_bytes
            or replay_pcm_bytes[:exact_prefix_bytes]
            != retained_after_discard[:exact_prefix_bytes]
        ):
            raise RuntimeError("Pocket replay does not reproduce retained 24-frame PCM bytes")
        pcm = np.frombuffer(replay_pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        tool = {
            "description": "Get the current clock time in UTC.",
            "name": "get_current_utc_time",
            "parameters": {"properties": {}, "required": [], "type": "object"},
        }
        template = (
            Path(self.config["speech_root"]) / "examples/speechlm2/function_calling/template.jinja"
        )
        rendered_prompt = render_tool_system_prompt(
            template, str(self.config["system_prompt"]), [tool]
        )
        if len(rendered_prompt.encode()) != tool_events[0].get("rendered_prompt_bytes"):
            raise RuntimeError("used prompt does not match the frozen tool-enabled prompt")
        engine = VoiceChatEngine(self.pipeline)
        self.control_engine = engine
        engine.configure_external_tools({"get_current_utc_time": lambda: None})
        engine.start(system_prompt=rendered_prompt)
        request_id = self.pipeline._request_id_for_stream(engine.stream_id)
        prefill_baseline = len(stream.requests[request_id].generated_tokens)
        if prefill_baseline != 1 or len(prefill_calls) != 1:
            raise RuntimeError(
                "Phase-1 materialization requires the frozen one-token prefill baseline"
            )
        self._enable_phase1_pairing()
        self._collective(install_phase1_forcing)
        self._set_targets([PAD_TOKEN_ID] * 4, [PAD_TOKEN_ID] * 4)
        frame22_state: dict[str, Any] | None = None
        try:
            for frame in range(24):
                capture_frame = frame
                try:
                    result = engine.process(
                        pcm[frame * FRAME_SAMPLES : (frame + 1) * FRAME_SAMPLES],
                        input_active=True,
                    )
                except RuntimeError as error:
                    if str(error) != "phase1-frame23-captured" or frame != 23:
                        raise
                    break
                nano_position = result.turn_state["vllm_request_positions"]["nano"]
                self.generated_counts.append(int(nano_position["generated_tokens"]))
                if self.generated_counts[-1] != recorded_generated_counts[frame]:
                    raise RuntimeError("live turn state differs from the frozen schedule")
                if frame == 22:
                    state = self._capture(request_id, "materialized-frame22")
                    pad_state = _pad_pair_state(stream, request_id)
                    pending = pad_state.get("pending")
                    if not isinstance(pending, torch.Tensor):
                        raise RuntimeError("frame-22 materialization has no pending fused row")
                    frame22_state = {
                        "pad": dict(pad_state),
                        "pending": pending,
                        "request_id": request_id,
                        "state": state,
                        "turn_state": result.turn_state,
                    }
        finally:
            stream.generate_next_token = original
        if set(frame_rows) != set(range(24)) or frame22_state is None:
            raise RuntimeError("materialization did not capture frame-22/frame-23 tensors")
        rows = [frame_rows[index] for index in range(24)]
        prefix_path = self.root.parent / "materialized-prefix-rows.pt"
        prefix_payload = {
            "assumptions": frame_assumptions,
            "generated_counts": self.generated_counts,
            "prefill_baseline": prefill_baseline,
            "prefill_calls": prefill_calls,
            "rows": rows,
            "sha256": [_tensor_sha256(row) for row in rows],
            "source_events_sha256": _file_sha256(events_path),
            "source_pcm_sha256": _file_sha256(Path(self.config["source_pcm"])),
            "tool_prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
            "typed_carrier_sha256": hashlib.sha256(typed_carrier_bytes).hexdigest(),
            "typed_carrier_replay_pcm_sha256": hashlib.sha256(replay_pcm_bytes).hexdigest(),
        }
        temporary_prefix = prefix_path.with_suffix(".pt.tmp")
        torch.save(prefix_payload, temporary_prefix)
        temporary_prefix.replace(prefix_path)
        prefix_payload_sha256 = _file_sha256(prefix_path)
        state = frame22_state["state"]
        leaf_metadata = {
            name: _leaf_metadata(state[name])
            for name in (
                "committed_kv_state",
                "committed_mamba_conv_state",
                "committed_mamba_ssm_state",
                "rollback_shadow",
            )
        }
        turn_state = frame22_state["turn_state"]
        nano_position = turn_state.get("vllm_request_positions", {}).get("nano", {})
        effective = frame22_state["pad"].get("last_effective_tokens")
        if not isinstance(effective, dict) or effective.get("frame_idx") != 22:
            raise RuntimeError("frame-22 effective PAD identity was not observed")
        pending = frame22_state["pending"]
        control_pending = bool(
            frame22_state["pad"].get("sequential_mode")
            or frame22_state["pad"].get("precontrol_reason")
        )
        materialized = MaterializedFixture(
            schema_version=1,
            fixture_id=str(fixture["fixture_id"]),
            fused_rows=(rows[22], rows[23]),
            frame_22_state=MaterializedFrame22State(
                frame=22,
                nano_generated_tokens=int(nano_position["generated_tokens"]),
                nano_session_positions=int(nano_position["session_positions"]),
                pending_row_count=int(isinstance(pending, torch.Tensor)),
                pending_text_token=int(effective["text"]),
                pending_function_token=int(effective["function"]),
                function_state=_function_state(turn_state),
                control_barrier_pending=control_pending,
                committed_kv_state=_pack_group(state["committed_kv_state"], "KV"),
                committed_mamba_conv_state=_pack_group(
                    state["committed_mamba_conv_state"], "Mamba conv"
                ),
                committed_mamba_ssm_state=_pack_group(
                    state["committed_mamba_ssm_state"], "Mamba SSM"
                ),
                rollback_shadow=_pack_group(state["rollback_shadow"], "rollback shadow"),
                shadow_index=int(state["shadow_index"]),
                shadow_pointer=f"{state['virtual_engine_identity']}/shadow/{state['shadow_index']}",
                virtual_engine_identity=str(state["virtual_engine_identity"]),
                construction_metadata=self._construction_metadata("full-prefix-materializer")
                | {
                    "leaf_tensors": leaf_metadata,
                    "shadow_physical_locations": state["shadow_physical_locations"],
                    "prefix_payload_sha256": prefix_payload_sha256,
                    "prefix_rows_sha256": stable_digest([_tensor_sha256(row) for row in rows]),
                    "source_attestation": {
                        "discarded_prefix_samples": int(discarded[0]["samples"]),
                        "events_sha256": _file_sha256(events_path),
                        "pcm_sha256": _file_sha256(Path(self.config["source_pcm"])),
                        "rendered_prompt_sha256": hashlib.sha256(
                            rendered_prompt.encode()
                        ).hexdigest(),
                        "tool_names": tool_events[0]["tools"],
                        "historical_witness_sha256": _file_sha256(witness_path),
                        "retained_pcm_exact_prefix_bytes": exact_prefix_bytes,
                        "retained_pcm_exact_prefix_sha256": hashlib.sha256(
                            retained_after_discard[:exact_prefix_bytes]
                        ).hexdigest(),
                        "typed_carrier_sha256": hashlib.sha256(
                            typed_carrier_bytes
                        ).hexdigest(),
                        "typed_carrier_synthesis": synthesis_seal,
                    },
                },
            ),
        )
        return {
            "materialized": materialized,
            "prefix_payload_sha256": prefix_payload_sha256,
            "prefix_rows_path": str(prefix_path),
        }

    def restore_lane(
        self,
        lane: str,
        request_id: str,
        fixture_sha256: str,
        materialized: MaterializedFixture,
        prefix_rows_path: str,
        prefix_payload_sha256: str,
    ) -> str:
        prefix_path = Path(prefix_rows_path)
        if _file_sha256(prefix_path) != prefix_payload_sha256:
            raise RuntimeError("retained prefix payload changed before lane restore")
        if self.nano is None:
            raise RuntimeError("epoch engine is not constructed")
        prefix = torch.load(prefix_rows_path, map_location="cpu", weights_only=False)
        rows = prefix["rows"]
        counts = prefix["generated_counts"]
        prefill_calls = prefix.get("prefill_calls")
        prefill_baseline = prefix.get("prefill_baseline")
        assumptions = prefix.get("assumptions")
        if (
            len(rows) != 24
            or len(counts) != 23
            or not isinstance(assumptions, dict)
            or set(assumptions) != set(range(1, 24))
            or not isinstance(prefill_calls, list)
            or len(prefill_calls) != 1
            or prefill_baseline != 1
        ):
            raise RuntimeError("retained prefix row/count contract is incomplete")
        if prefix.get("sha256") != [_tensor_sha256(row) for row in rows]:
            raise RuntimeError("retained prefix row content seal mismatch")
        if prefix.get("source_pcm_sha256") != _file_sha256(Path(self.config["source_pcm"])):
            raise RuntimeError("retained prefix PCM identity mismatch")
        stream = self.nano.engine
        self._collective(clear_phase1_targets)
        for prefill in prefill_calls:
            self.nano(
                prefill["row"].unsqueeze(0),
                request_id=request_id,
                decode_steps=0,
                prompt_token_ids=prefill["prompt_token_ids"],
            )
        actual_prefill = len(stream.requests[request_id].generated_tokens)
        if actual_prefill != prefill_baseline:
            raise RuntimeError(
                f"prefix prefill count mismatch: {actual_prefill} != {prefill_baseline}"
            )
        self._set_targets([PAD_TOKEN_ID] * 4, [PAD_TOKEN_ID] * 4)
        previous = prefill_baseline
        for index, target in enumerate(counts):
            state = _pad_pair_state(stream, request_id)
            state["previous_effective_pad"] = True
            delta = target - previous
            if delta == 0:
                if state.get("pending") is not None:
                    raise RuntimeError(f"prefix attempted a second buffer at frame {index}")
                state["pending"] = rows[index].detach().clone()
            elif delta == 1:
                pending = state.get("pending")
                if isinstance(pending, torch.Tensor):
                    self._original_step(pending, request_id)
                    state["pending"] = rows[index].detach().clone()
                else:
                    self._original_step(rows[index], request_id)
            elif delta == 2:
                if not isinstance(state.get("pending"), torch.Tensor):
                    raise RuntimeError(f"prefix pair lacks buffered row at frame {index}")
                self.nano._run_async(
                    stream.generate_next_token([rows[index]], request_id=request_id)
                )
            else:
                raise RuntimeError(
                    "prefix generated-token transition differs at frame "
                    f"{index}: {previous}->{target}"
                )
            actual = len(stream.requests[request_id].generated_tokens)
            if actual != target:
                raise RuntimeError(
                    f"prefix generated count mismatch at frame {index}: {actual} != {target}"
                )
            previous = target
        if len(stream.requests[request_id].generated_tokens) != 23:
            raise RuntimeError("restored Nano request does not end at generated-token 23")
        pad = _pad_pair_state(stream, request_id)
        if not isinstance(pad.get("pending"), torch.Tensor):
            raise RuntimeError("restored frame-22 row is not pending")
        if _tensor_sha256(pad["pending"]) != _tensor_sha256(materialized.fused_rows[0]):
            raise RuntimeError("restored pending row differs from the materialization seal")
        restored_state = self._capture(request_id, f"{lane}-restored-frame22")
        sealed_state = materialized.frame_22_state
        comparisons = {
            "committed_kv_state": sealed_state.committed_kv_state,
            "committed_mamba_conv_state": sealed_state.committed_mamba_conv_state,
            "committed_mamba_ssm_state": sealed_state.committed_mamba_ssm_state,
            "rollback_shadow": sealed_state.rollback_shadow,
        }
        for name, expected in comparisons.items():
            actual = _pack_group(restored_state[name], f"restored {name}")
            if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
                raise RuntimeError(f"restored {name} dtype/shape differs from materialization")
            if _tensor_sha256(actual) != _tensor_sha256(expected):
                raise RuntimeError(f"restored {name} bytes differ from materialization")
        if (
            int(restored_state["shadow_index"]) != sealed_state.shadow_index
            or str(restored_state["virtual_engine_identity"])
            != sealed_state.virtual_engine_identity
        ):
            raise RuntimeError(
                "restored shadow/virtual-engine identity differs: "
                f"shadow={restored_state['shadow_index']} != {sealed_state.shadow_index}, "
                f"virtual_engine={restored_state['virtual_engine_identity']} "
                f"!= {sealed_state.virtual_engine_identity}"
            )
        restored_shadow_locations = restored_state["shadow_physical_locations"]
        if not restored_shadow_locations or any(
            int(location["request_block_ordinal"]) != sealed_state.shadow_index
            for location in restored_shadow_locations.values()
        ):
            raise RuntimeError(
                "restored rollback shadow is not bound to the sealed request-relative index"
            )
        expected_leaf_metadata = sealed_state.construction_metadata.get("leaf_tensors")
        actual_leaf_metadata = {name: _leaf_metadata(restored_state[name]) for name in comparisons}
        if actual_leaf_metadata != expected_leaf_metadata:
            raise RuntimeError("restored per-leaf state seals differ from materialization")
        if _file_sha256(prefix_path) != prefix_payload_sha256:
            raise RuntimeError("retained prefix payload changed during lane restore")
        self.lanes[lane] = {
            "fixture_sha256": fixture_sha256,
            "materialized": materialized,
            "pending_origin": None,
            "prefix": prefix,
            "prefix_payload_sha256": prefix_payload_sha256,
            "prefix_rows_path": prefix_rows_path,
            "request_id": request_id,
            "restore_allocation_evidence": {
                "materialized_physical_locations": sealed_state.construction_metadata[
                    "shadow_physical_locations"
                ],
                "restored_physical_locations": restored_shadow_locations,
                "request_relative_shadow_index": sealed_state.shadow_index,
            },
            "transition": 0,
        }
        return fixture_sha256

    def _start_control_engine(self) -> None:
        """Start the production control owner before forcing or lane requests exist."""
        if self.pipeline is None:
            raise RuntimeError("control lane requires the qualified full pipeline")
        from nemotron_voicechat_runtime.server import VoiceChatEngine

        engine = VoiceChatEngine(self.pipeline)
        engine.configure_external_tools({"get_current_utc_time": lambda: None})
        engine.start(system_prompt=str(self.config["system_prompt"]))
        self.control_engine = engine

    def _discard_control_lane(self) -> None:
        """Release only the synthetic stream owners, never the live Nano lane."""
        if self.pipeline is None or self.control_engine is None:
            return
        stream_id = self.control_engine.stream_id
        abort = getattr(self.pipeline, "_abort_stream_request", None)
        if callable(abort):
            abort(stream_id)
        self.pipeline.delete_state(stream_id)
        manager = self.pipeline.context_manager
        slot = manager.streamidx2slotidx.get(stream_id)
        if slot is not None:
            manager.reset_slot(slot)
        self.control_engine = None

    def _control_owners(self) -> tuple[Any, Any, Any]:
        if self.pipeline is None or self.control_engine is None or not self.control_engine.started:
            raise RuntimeError("real control owners are not active")
        wrapper = self.pipeline.s2s_model
        manager = self.pipeline.context_manager
        stream_id = self.control_engine.stream_id
        slot = manager.streamidx2slotidx.get(stream_id)
        if slot is None:
            raise RuntimeError("real control context has no live stream slot")
        context = manager.slot_contexts[slot]
        state = self.pipeline._state_pool.get(stream_id)
        if context is None or state is None:
            raise RuntimeError("real control context/streaming state is absent")
        return wrapper, context, state

    def _control_snapshot(self) -> dict[str, Any]:
        wrapper, context, state = self._control_owners()
        stream_id = self.control_engine.stream_id
        fc = context.fc_state
        if not isinstance(fc, dict):
            raise RuntimeError("real control context lacks function-call state")
        return {
            "agent_idle": bool(wrapper._get_agent_idle(stream_id)),
            "agent_speaking": bool(
                torch.as_tensor(context.rnnt_partial_hypotheses["agent_speaking"])
                .reshape(-1)[0]
                .item()
            ),
            "context_frame_idx": int(context.frame_idx),
            "function_calling": {
                key: fc.get(key)
                for key in (
                    "active",
                    "awaiting_response",
                    "awaiting_eotr",
                    "injecting_response",
                    "trigger_async",
                )
            }
            | {
                "call_tokens": [int(item) for item in fc.get("call_tokens", [])],
                "forced_token_count": len(fc.get("forced_function_tokens") or []),
            },
            "post_fc_client_bos_requested": bool(
                getattr(wrapper, "_post_fc_client_bos_requested", False)
            ),
            "streaming_state_agent_idle": bool(state.is_agent_idle()),
        }

    def _original_step(self, row: torch.Tensor, request_id: str) -> Any:
        stream = self.nano.engine
        original = type(stream)._voicechat_original_generate_next_token
        record_boundary("B0", request_id, host_sequential_call=True, token_count=1)
        result = self.nano._run_async(original(stream, [row], request_id=request_id))
        record_boundary("B9", request_id, host_call_returned=True, iterator_awakened=True)
        return result

    @torch.inference_mode()
    def _execute_control_boundary(
        self, lane: str, control: str, row: torch.Tensor, request_id: str
    ) -> tuple[Any, dict[str, Any]]:
        """Run one Nano position through the live wrapper/turn-state owners."""
        del lane
        wrapper, context, _state = self._control_owners()
        stt = wrapper.model.stt_model
        if control == "agent_bos" and bool(
            torch.as_tensor(context.rnnt_partial_hypotheses["agent_speaking"])
            .reshape(-1)[0]
            .item()
        ):
            setup_frame = int(context.frame_idx)
            context.gen_text[:, setup_frame] = int(stt.text_eos_id)
            setup_blank = torch.ones(
                context.gen_text.shape[0], dtype=torch.bool, device=context.gen_text.device
            )
            wrapper._apply_rnnt_turn_taking(
                setup_frame,
                context.gen_text,
                setup_blank,
                context.rnnt_partial_hypotheses,
            )
            if int(context.gen_text[0, setup_frame].item()) != int(stt.text_eos_id):
                raise RuntimeError("real wrapper could not close the retained agent turn")
            wrapper._set_agent_idle(True, self.control_engine.stream_id)
            context.frame_idx = setup_frame + 1
        targets = {
            "agent_bos": (int(stt.text_bos_id), PAD_TOKEN_ID),
            "agent_eos": (int(stt.text_eos_id), PAD_TOKEN_ID),
            "function_sotc": (PAD_TOKEN_ID, int(wrapper._fc_sotc_id)),
            "function_eotr": (PAD_TOKEN_ID, int(wrapper._fc_eotr_id)),
            "post_fc_bos": (int(stt.text_bos_id), PAD_TOKEN_ID),
        }
        if control not in targets:
            raise RuntimeError(f"unknown named control operation: {control}")
        text, function = targets[control]
        self._set_targets([text] * 4, [function] * 4)
        result = self._original_step(row, request_id)
        if int(result.token_id) != text or self._observed_function_token(result) != function:
            raise RuntimeError(f"named control operation emitted the wrong tokens: {control}")
        before = self._control_snapshot()
        frame = int(context.frame_idx)
        wrapper_input_text = (
            PAD_TOKEN_ID if control in {"agent_bos", "post_fc_bos"} else int(result.token_id)
        )
        context.gen_text[:, frame] = wrapper_input_text
        context.gen_function_text[:, frame] = self._observed_function_token(result)
        if control in {"agent_bos", "agent_eos", "post_fc_bos"}:
            if control == "agent_bos":
                enable_external_eou = getattr(wrapper, "set_external_user_eou_mode", None)
                if not callable(enable_external_eou):
                    raise RuntimeError("real wrapper lacks external user EOU mode")
                enable_external_eou(True)
                wrapper.request_user_eou()
            elif control == "post_fc_bos":
                if int(wrapper._fc_eotr_id) not in context.fc_state.get("call_tokens", []):
                    raise RuntimeError(
                        "post-FC BOS attempted before real function EOTR observation"
                    )
                wrapper.request_user_eou()
            blank = torch.ones(
                context.gen_text.shape[0], dtype=torch.bool, device=context.gen_text.device
            )
            wrapper._apply_rnnt_turn_taking(
                frame,
                context.gen_text,
                blank,
                context.rnnt_partial_hypotheses,
            )
            effective = int(context.gen_text[0, frame].item())
            if effective != text:
                raise RuntimeError(f"real wrapper changed the named agent control: {control}")
            wrapper._set_agent_idle(
                effective == int(stt.text_eos_id), self.control_engine.stream_id
            )
        else:
            predicted = context.gen_text[:, frame : frame + 1].clone()
            predicted_function = context.gen_function_text[:, frame : frame + 1].clone()
            wrapper._apply_fc_state_machine(
                fc_state=context.fc_state,
                function_predicted_token=context.gen_function_text[:, frame],
                gen_text=context.gen_text,
                gen_function_text=context.gen_function_text,
                current_frame_idx=frame,
                frame_offset=0,
                predicted_tokens=predicted,
                function_predicted_tokens=predicted_function,
                tool_response_text=context.tool_response_text,
            )
        context.frame_idx = frame + 1
        after = self._control_snapshot()
        if before == after:
            raise RuntimeError(f"real wrapper/turn state did not transition: {control}")
        if control in {"agent_bos", "post_fc_bos"} and after["agent_idle"]:
            raise RuntimeError(f"real wrapper did not open the agent turn: {control}")
        if control == "agent_eos" and not after["agent_idle"]:
            raise RuntimeError("real wrapper did not close the agent turn")
        if control == "function_sotc" and after["function_calling"]["active"] is not True:
            raise RuntimeError("real wrapper did not enter the function cycle")
        if (
            control == "function_eotr"
            and int(wrapper._fc_eotr_id) not in after["function_calling"]["call_tokens"]
        ):
            raise RuntimeError("real wrapper did not observe function EOTR")
        return result, {
            "after": after,
            "before": before,
            "mechanism": (
                "NemotronVoicechatInferenceWrapper._apply_fc_state_machine"
                if control.startswith("function_")
                else (
                    "NemotronVoicechatInferenceWrapper._apply_rnnt_turn_taking/"
                    "_set_agent_idle"
                )
            ),
        }

    def _snapshot(
        self,
        lane: str,
        operation_name: str,
        observed_plan: OraclePlan,
        before: dict[str, Any],
        pending_before_sha: str | None,
        result: Any,
        selected_source_row: torch.Tensor | None,
        *,
        correction_evidence: dict[str, Any] | None = None,
        control_evidence: dict[str, Any] | None = None,
        position_one_state: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        lane_state = self.lanes[lane]
        request_id = lane_state["request_id"]
        lane_state["transition"] += 1
        transition_id = f"transition-{lane_state['transition']:02d}"
        raw = self._capture(request_id, f"{lane}-{lane_state['transition']:02d}-{operation_name}")
        lane_state["last_allocation_evidence"] = {
            "physical_locations": raw["shadow_physical_locations"],
            "request_relative_shadow_index": int(raw["shadow_index"]),
        }
        before_sha = _tree_sha256(before)
        after_sha = _tree_sha256(raw)
        raw_shadow_evidence = _leaf_metadata(raw["rollback_shadow"])
        if operation_name == "transaction" and position_one_state is not None:
            rollback_projection = _committed_mamba_projection(position_one_state)
            rollback_projection_source = "sequential-position-one-committed-boundary"
        else:
            rollback_projection = _canonical_rollback_projection(raw["rollback_shadow"])
            rollback_projection_source = "packed-pair-position-one-shadow"
        pad = _pad_pair_state(self.nano.engine, request_id)
        pending = pad.get("pending")
        pending_hash = _tensor_sha256(pending) if isinstance(pending, torch.Tensor) else None
        is_drain = observed_plan.kind == TransactionKind.PENDING_DRAIN
        origin = lane_state.get("pending_origin")
        corrected = correction_evidence is not None
        if corrected:
            lane_state["pending_origin"] = transition_id
            origin = transition_id
        elif is_drain:
            lane_state["pending_origin"] = None
        custom = (
            {}
            if observed_plan.kind == TransactionKind.ABORT
            else dict(getattr(result, "custom_outputs", None) or {})
        )
        emitted_text = None if observed_plan.kind == TransactionKind.ABORT else int(result.token_id)
        emitted_function = None
        if observed_plan.kind != TransactionKind.ABORT:
            function = custom.get("function_tokens")
            if not isinstance(function, torch.Tensor) or function.numel() != 1:
                raise RuntimeError("completed operation lacks one observed function token")
            emitted_function = int(function.reshape(-1)[0].item())
        selected = observed_plan.selected_packed_row
        if selected not in (0, 1) and observed_plan.kind != TransactionKind.ABORT:
            raise RuntimeError("observed selected row is outside the fused pair")
        selected_source_sha256 = (
            stable_digest("zero-position-abort")
            if observed_plan.kind == TransactionKind.ABORT
            else _tensor_sha256(selected_source_row)
            if isinstance(selected_source_row, torch.Tensor)
            else None
        )
        if selected_source_sha256 is None:
            raise RuntimeError("completed operation lacks its actually executed source row")
        if is_drain and selected_source_sha256 != pending_before_sha:
            raise RuntimeError(
                "pending drain source identity differs from the executed pending row"
            )
        return StateSnapshot(
            emitted_text_token=emitted_text,
            emitted_function_token=emitted_function,
            emitted_conjunction=(
                None
                if observed_plan.kind == TransactionKind.ABORT
                else emitted_text == PAD_TOKEN_ID and emitted_function == PAD_TOKEN_ID
            ),
            accepted_token_count=observed_plan.accepted_token_count,
            selected_packed_row=selected,
            selected_source_row_identity=selected_source_sha256,
            selected_custom_outputs=custom,
            committed_kv_state=raw["committed_kv_state"],
            committed_mamba_conv_state=raw["committed_mamba_conv_state"],
            committed_mamba_ssm_state=raw["committed_mamba_ssm_state"],
            rollback_shadow=rollback_projection,
            shadow_index=int(raw["shadow_index"]),
            shadow_pointer=f"{raw['virtual_engine_identity']}/shadow/{raw['shadow_index']}",
            virtual_engine_identity=str(raw["virtual_engine_identity"]),
            host_pending_rows=([] if pending is None else {"row": pending.detach().cpu().clone()}),
            correction_flags={
                "corrected": corrected,
                "needs_correction_after": bool(pad.get("needs_correction", False)),
            },
            correction_assumptions=(correction_evidence or {}).get("assumptions", {}),
            position_one_boundary_state={
                "committed_state_sha256": (
                    _tree_sha256(position_one_state)
                    if position_one_state is not None
                    else after_sha
                )
            },
            uncommitted_second_position_state=(
                {"committed_state_sha256": after_sha} if corrected else None
            ),
            operation_evidence={
                "accepted_token_count": observed_plan.accepted_token_count,
                "operation": operation_name,
                "request_present_after": request_id in self.nano.engine.requests,
                "selected_source_row_sha256": (
                    None
                    if observed_plan.kind == TransactionKind.ABORT
                    else selected_source_sha256
                ),
                "raw_rollback_shadow_leaves": raw_shadow_evidence,
                "rollback_projection_source": rollback_projection_source,
            }
            | ({"runtime_control_transition": control_evidence} if control_evidence else {}),
            transition_id=transition_id,
            originating_rejection_id=(origin if corrected or is_drain else None),
            pending_row_sha256_before=(pending_before_sha if is_drain else None),
            pending_row_sha256_after=(pending_hash if corrected else None),
            committed_state_sha256_before=before_sha,
            committed_state_sha256_after=after_sha,
        )

    @staticmethod
    def _observed_function_token(result: Any) -> int:
        outputs = dict(getattr(result, "custom_outputs", None) or {})
        value = outputs.get("function_tokens")
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeError("operation did not return exactly one function token")
        return int(value.reshape(-1)[0].item())

    def _observed_transaction_plan(
        self,
        *,
        accepted: int,
        result: Any,
        operation_kind: TransactionKind | None,
        corrected: bool,
    ) -> OraclePlan:
        if operation_kind == TransactionKind.ABORT:
            return OraclePlan(TransactionKind.ABORT, (), -1, 0, False, False)
        if accepted not in (1, 2):
            raise RuntimeError(f"observed accepted count is invalid: {accepted}")
        if operation_kind in {TransactionKind.CONTROL_BOUNDARY, TransactionKind.PENDING_DRAIN}:
            return OraclePlan(
                operation_kind,
                (0,),
                0,
                accepted,
                False,
                operation_kind == TransactionKind.PENDING_DRAIN,
            )
        text = int(result.token_id)
        function = self._observed_function_token(result)
        if accepted == 2 and text == PAD_TOKEN_ID and function == PAD_TOKEN_ID:
            kind = TransactionKind.PAD_ACCEPT
        elif text != PAD_TOKEN_ID and function == PAD_TOKEN_ID:
            kind = TransactionKind.TEXT_REJECTION
        elif text == PAD_TOKEN_ID and function != PAD_TOKEN_ID:
            kind = TransactionKind.FUNCTION_REJECTION
        else:
            kind = TransactionKind.CONJUNCTIVE_REJECTION
        return OraclePlan(
            kind,
            tuple(range(accepted)),
            accepted - 1,
            accepted,
            corrected,
            corrected,
        )

    def execute_case(
        self,
        lane: str,
        case: CaseSpec,
        plan: OraclePlan,
        execution: str,
        request_id: str,
    ) -> ExecutionResult:
        lane_state = self.lanes[lane]
        if request_id != lane_state["request_id"]:
            raise RuntimeError("lane request identity changed")
        prefix = lane_state["prefix"]
        row0, row1 = prefix["rows"][22], prefix["rows"][23]
        before = self._capture(request_id, f"{lane}-before-{lane_state['transition']:02d}")
        self._set_targets(
            [case.text_token, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
            [case.function_token, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
        )
        stream = self.nano.engine
        pad = _pad_pair_state(stream, request_id)
        pending_before = pad.get("pending")
        pending_before_sha = (
            _tensor_sha256(pending_before) if isinstance(pending_before, torch.Tensor) else None
        )
        generated_before = len(stream.requests[request_id].generated_tokens)
        result = None
        correction_evidence: dict[str, Any] | None = None
        control_evidence: dict[str, Any] | None = None
        executed_source_row: torch.Tensor | None = None
        position_one_state: dict[str, Any] | None = None
        operation_kind: TransactionKind | None = None
        abort_evidence: dict[str, JsonValue] | None = None
        if case.action == "pending_drain":
            operation_kind = TransactionKind.PENDING_DRAIN
            pending = pad.get("pending")
            if not isinstance(pending, torch.Tensor):
                raise RuntimeError("pending drain has no corrected row")
            executed_source_row = pending.detach().clone()
            result = self._original_step(pending, request_id)
            pad["pending"] = None
            pad["needs_correction"] = False
            pad["assumed"] = None
        elif (
            execution == "sequential-logical-position"
            and case.action == "transaction"
            and case.control_boundary is None
        ):
            pending = pad.get("pending")
            if not isinstance(pending, torch.Tensor):
                pending = row0
            executed_source_row = pending.detach().clone()
            result = self._original_step(pending, request_id)
            position_one_state = self._capture(
                request_id, f"{lane}-position-one-{lane_state['transition']:02d}"
            )
            observed_text = int(result.token_id)
            observed_function = self._observed_function_token(result)
            if observed_text == PAD_TOKEN_ID and observed_function == PAD_TOKEN_ID:
                executed_source_row = row1
                result = self._original_step(row1, request_id)
                pad["pending"] = None
            else:
                assumptions = _runtime_recorded_assumptions(prefix, 23)
                corrected = self._correct_row(
                    row1,
                    CaseSpec(case.name, observed_text, observed_function),
                    f"{lane}-{lane_state['transition']:02d}",
                    assumptions,
                )
                pad["pending"] = corrected
                pad["needs_correction"] = False
                pad["assumed"] = None
                correction_evidence = {
                    "assumption_source": "materialized-wrapper-call-context/frame-23",
                    "assumptions": {"function": assumptions[1], "text": assumptions[0]},
                    "corrected_row_sha256": _tensor_sha256(corrected),
                    "effective": {"function": observed_function, "text": observed_text},
                }
        elif case.action == "abort":
            operation_kind = TransactionKind.ABORT
            self.nano._run_async(stream.abort_generation(request_id))
            if request_id in stream.requests:
                raise RuntimeError("abort_generation returned with the request still present")
            transition = lane_state["transition"]
            self.restore_lane(
                lane,
                request_id,
                lane_state["fixture_sha256"],
                lane_state["materialized"],
                lane_state["prefix_rows_path"],
                lane_state["prefix_payload_sha256"],
            )
            self.lanes[lane]["transition"] = transition
            lane_state = self.lanes[lane]
            stream = self.nano.engine
            pad = _pad_pair_state(stream, request_id)
            abort_evidence = {
                "operation": "abort_generation",
                "request_absent_after": True,
                "restored_after_abort": True,
            }
        elif case.control_boundary is not None:
            operation_kind = TransactionKind.CONTROL_BOUNDARY
            result, control_evidence = self._execute_control_boundary(
                lane, case.control_boundary, row0, request_id
            )
            executed_source_row = row0
            pad["pending"] = None
        else:
            stream._voicechat_pad_pair_call_context = {
                "assumed": _runtime_recorded_assumptions(prefix, 23),
                "bypass": False,
            }
            if not isinstance(pad.get("pending"), torch.Tensor):
                pad["previous_effective_pad"] = True
                self.nano._run_async(stream.generate_next_token([row0], request_id=request_id))
            pad["previous_effective_pad"] = True
            result = self.nano._run_async(stream.generate_next_token([row1], request_id=request_id))
            accepted_now = len(stream.requests[request_id].generated_tokens) - generated_before
            executed_source_row = row1 if accepted_now == 2 else row0
            if accepted_now == 1 and isinstance(pad.get("pending"), torch.Tensor):
                # The direct engine path does not re-enter the wrapper finalizer.
                # Apply the same qualified correction helper explicitly.
                observed_text = int(result.token_id)
                observed_function = self._observed_function_token(result)
                recorded_assumptions = pad.get("assumed")
                recorded_assumptions = _runtime_recorded_assumptions(prefix, 23, pad)
                corrected = self._correct_row(
                    row1,
                    CaseSpec(case.name, observed_text, observed_function),
                    f"{lane}-{lane_state['transition']:02d}",
                    recorded_assumptions,
                )
                pad["pending"] = corrected
                pad["needs_correction"] = False
                pad["assumed"] = None
                correction_evidence = {
                    "assumption_source": "pad-pair-runtime-state",
                    "assumptions": {
                        "function": recorded_assumptions[1],
                        "text": recorded_assumptions[0],
                    },
                    "corrected_row_sha256": _tensor_sha256(corrected),
                    "effective": {"function": observed_function, "text": observed_text},
                }
        generated_after = len(stream.requests[request_id].generated_tokens)
        accepted = (
            0 if operation_kind == TransactionKind.ABORT else generated_after - generated_before
        )
        observed_plan = self._observed_transaction_plan(
            accepted=accepted,
            result=result,
            operation_kind=operation_kind,
            corrected=correction_evidence is not None,
        )
        snapshot = self._snapshot(
            lane,
            case.action if case.control_boundary is None else case.control_boundary,
            observed_plan,
            before,
            pending_before_sha,
            result,
            executed_source_row,
            correction_evidence=correction_evidence,
            control_evidence=control_evidence,
            position_one_state=position_one_state,
        )
        if operation_kind == TransactionKind.ABORT:
            boundaries: tuple[str, ...] = ()
            dispatch = "normal"
            dispatch_evidence = abort_evidence or {}
        else:
            boundaries = read_boundary_prefix(self.trace_dir, request_id)
            dispatch, dispatch_evidence = observed_dispatch(self.trace_dir, request_id)
        dispatch_evidence = dict(dispatch_evidence)
        dispatch_evidence["state_allocation_evidence"] = {
            "restore": lane_state["restore_allocation_evidence"],
            "after_operation": lane_state["last_allocation_evidence"],
        }
        return ExecutionResult(
            snapshot=snapshot,
            boundaries=boundaries,
            completed=True,
            terminal_unknown=False,
            dispatch_mode=dispatch,
            prefix_sha256=lane_state["fixture_sha256"],
            observed_plan=observed_plan,
            dispatch_evidence=dispatch_evidence,
        )

    def shutdown(self) -> None:
        self._discard_control_lane()
        if self.nano is not None:
            stream = self.nano.engine
            for request_id in list(stream.requests):
                try:
                    self.nano._run_async(stream.abort_generation(request_id))
                except Exception:
                    pass
            self.nano._run_async(stream.shutdown())
            self.nano = None


def _serve(config_path: Path) -> None:
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = GPUCampaignRuntime(config, config_path.parent)
    for raw in sys.stdin:
        command: dict[str, Any] = {}
        try:
            command = json.loads(raw)
            operation = command["operation"]
            payload = (
                torch.load(command["payload_path"], map_location="cpu", weights_only=False)
                if command.get("payload_path")
                else command.get("payload", {})
            )
            if operation == "construct":
                result = runtime.construct(str(payload["mode"]))
            elif operation == "materialize":
                result = runtime.materialize(payload["fixture"])
            elif operation == "restore_lane":
                result = runtime.restore_lane(
                    payload["lane"],
                    payload["request_id"],
                    payload["fixture_sha256"],
                    payload["materialized"],
                    payload["prefix_rows_path"],
                    payload["prefix_payload_sha256"],
                )
            elif operation == "execute_case":
                result = runtime.execute_case(
                    payload["lane"],
                    payload["case"],
                    payload["plan"],
                    payload["execution"],
                    payload["request_id"],
                )
            elif operation == "shutdown":
                runtime.shutdown()
                result = {"reason": payload.get("reason")}
            else:
                raise ValueError(f"unknown worker operation: {operation}")
            torch.save(result, command["output"])
            protocol.write(json.dumps({"ok": True, "sequence": command["sequence"]}) + "\n")
            if operation == "shutdown":
                return
        except BaseException as error:
            protocol.write(
                json.dumps(
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "ok": False,
                        "sequence": command.get("sequence"),
                        "traceback": traceback.format_exc(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _smoke_s(config_path: Path, output: Path) -> None:
    """Construct one real Nano engine and exercise one structural S lane.

    The row is a real PAD embedding from the qualified model, but it is not the
    frozen PCM-derived materialization and can never be used as campaign data.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = GPUCampaignRuntime(config, output.parent)
    construction = runtime.construct("epoch")
    row_path = output.parent / "smoke-pad-row.pt"
    runtime._collective(export_embedding_row, PAD_TOKEN_ID, str(row_path))
    row = torch.load(row_path, map_location="cpu", weights_only=False)
    fixture_path = REPO_ROOT / "reports/step5-wedge/phase1-frozen-fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    events_path = Path(config["source_pcm"]).with_name("events.jsonl")
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    counts = [
        int(item["turn_state"]["vllm_request_positions"]["nano"]["generated_tokens"])
        for item in events
        if item.get("event") == "model_step" and 0 <= int(item.get("frame", -1)) <= 22
    ]
    if len(counts) != 23 or counts[-1] != 23:
        raise RuntimeError("Phase-1 smoke could not recover the frozen scheduler shape")
    prefix_path = output.parent / "smoke-prefix.pt"
    torch.save(
        {
            "generated_counts": counts,
            "prefill_baseline": 1,
            "prefill_calls": [{"prompt_token_ids": [1], "row": row.clone()}],
            "rows": [row.clone() for _ in range(24)],
            "sha256": [_tensor_sha256(row)] * 24,
        },
        prefix_path,
    )
    request_id = "step5-phase1/S/smoke"
    restored = None
    cases: list[dict[str, Any]] = []
    try:
        restored = runtime.restore_lane(
            "S", request_id, fixture["fixture_sha256"], str(prefix_path)
        )
        for case in case_matrix(PAD_TOKEN_ID):
            result = runtime.execute_case(
                "S",
                case,
                plan_for_case(case, PAD_TOKEN_ID),
                "sequential-logical-position",
                request_id,
            )
            cases.append(
                {
                    "boundaries": list(result.boundaries),
                    "case": case.name,
                    "completed": result.completed,
                    "dispatch_mode": result.dispatch_mode,
                    "snapshot": result.snapshot is not None,
                }
            )
    except BaseException as error:
        output.write_text(
            json.dumps(
                {
                    "campaign_admissible": False,
                    "error": f"{type(error).__name__}: {error}",
                    "kind": "phase1-real-executor-structural-s-smoke-failure",
                    "schema": 1,
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        runtime.shutdown()
    payload = {
        "campaign_admissible": False,
        "construction": construction,
        "fixture_prefix_identity": restored,
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "kind": "phase1-real-executor-structural-s-smoke",
        "real_engine_constructions": 1,
        "s_lane_cases": cases,
        "schema": 1,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--smoke-s", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.serve == args.smoke_s:
        parser.error("select exactly one of --serve or --smoke-s")
    if args.smoke_s:
        if args.output is None:
            parser.error("--smoke-s requires --output")
        _smoke_s(args.config, args.output)
    else:
        _serve(args.config)


if __name__ == "__main__":
    main()
