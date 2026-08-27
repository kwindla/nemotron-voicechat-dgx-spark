#!/usr/bin/env python3
"""Shared sealed machinery for the attnW8-GPTQ campaign."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ATTENTION_LAYERS = (14, 21, 30, 39)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
GROUP_SIZE = 128
BLOCK_SIZE = 128
DAMP_PERCENT = 0.01
PAD_TOKEN = 12
AGENT_BOS_TOKEN = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def scalar(value: Any) -> int:
    if value is None:
        return -1
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError(f"expected scalar output token, got {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def tree_manifest(root: Path) -> dict[str, Any]:
    records = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(root.resolve()), "files": records, "sha256": canonical_sha256(records)}


def read_safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(length))


class ArtifactReader:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.weight_map: dict[str, str] = self.index["weight_map"]

    def tensor(self, name: str):
        from safetensors import safe_open

        shard = self.weight_map[name]
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def dense_or_dequant(self, weight_name: str, torch):
        if weight_name in self.weight_map:
            return self.tensor(weight_name).to(torch.float32)
        prefix = weight_name.removesuffix(".weight")
        return self.dequant_prefix(prefix, torch)

    def dequant_prefix(self, prefix: str, torch):
        qweight = self.tensor(prefix + ".qweight").to(torch.int32)
        scales = self.tensor(prefix + ".scales")
        qzeros = self.tensor(prefix + ".qzeros")
        g_idx = self.tensor(prefix + ".g_idx")
        if bool((qzeros != 0).any()):
            raise ValueError(f"{prefix}: qzeros is not all-zero")
        k = int(qweight.shape[0]) * 4
        n = int(qweight.shape[1])
        expected_g = torch.arange(k, dtype=torch.int32) // GROUP_SIZE
        if not torch.equal(g_idx.cpu().to(torch.int32), expected_g):
            raise ValueError(f"{prefix}: g_idx is not natural group membership")
        lanes = [((qweight >> (8 * lane)) & 0xFF) for lane in range(4)]
        unsigned = torch.stack(lanes, dim=1).reshape(k, n)
        signed = unsigned.to(torch.int32) - 128
        expanded = scales.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=0)
        if tuple(expanded.shape) != (k, n):
            raise ValueError(f"{prefix}: scale orientation mismatch")
        return (signed.to(torch.float32) * expanded).t().contiguous()


def source_to_hf_name(name: str) -> str | None:
    if name.startswith("stt_model.llm."):
        return "backbone." + name.removeprefix("stt_model.llm.")
    if name == "stt_model.lm_head.weight":
        return "lm_head.weight"
    return None


def load_fhw8_eager(
    root: Path,
    model_code_root: Path,
    *,
    device: str,
    torch,
):
    """Reconstruct the runtime fhw8 propagation weights in the eager HF model."""
    from transformers import AutoConfig, AutoModelForCausalLM

    reader = ArtifactReader(root)
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(model_code_root.resolve(), trust_remote_code=True)
    eager_chunk_size = int(config.chunk_size)
    for key, value in config_payload.items():
        setattr(config, key, value)
    # The packed runtime pins chunk_size=1 for cached one-row streaming.  A0
    # and sequential GPTQ propagate complete effective-position sequences
    # without a cache, so retain the base eager model's qualified chunk size.
    # Using the streaming value in the uncached torch path creates a
    # pathological sequence-by-sequence broadcast rather than the same causal
    # recurrence partitioned into eager chunks.
    config.chunk_size = eager_chunk_size
    config.use_mamba_kernels = False
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, attn_implementation="eager"
        )
    finally:
        torch.set_default_dtype(old_dtype)
    modeling = sys.modules[model.__class__.__module__]
    modeling.is_fast_path_available = False
    model.backbone.embeddings = torch.nn.Identity()
    parameters = dict(model.named_parameters())
    loaded: set[str] = set()
    for hf_name, target in parameters.items():
        if hf_name.startswith("backbone.embeddings."):
            continue
        if hf_name == "lm_head.weight":
            source_name = "stt_model.lm_head.weight"
        elif hf_name.startswith("backbone."):
            source_name = "stt_model.llm." + hf_name.removeprefix("backbone.")
        else:
            raise KeyError(f"unmapped eager parameter: {hf_name}")
        value = reader.dense_or_dequant(source_name, torch)
        if (
            source_name.endswith(".mixer.down_proj.weight")
            and value.shape[1] == target.shape[1] + 64
        ):
            # The qualified Marlin artifact shape-pads down-projection K from
            # 15,680 to 15,744. The 64 added columns are exactly zero and are
            # not part of the eager HF module's propagation weight.
            value = value[:, : target.shape[1]]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch {source_name}: {tuple(value.shape)} != {tuple(target.shape)}"
            )
        target.data.copy_(value.to(dtype=target.dtype))
        loaded.add(hf_name)
        del value
    expected = {name for name in parameters if not name.startswith("backbone.embeddings.")}
    if loaded != expected:
        raise ValueError(f"fhw8 eager reconstruction missing {sorted(expected - loaded)[:8]}")
    function_weight = reader.dequant_prefix("stt_model.function_head", torch).to(torch.bfloat16)
    model.eval().requires_grad_(False).to(device)
    return model, function_weight.to(device), reader


@dataclass
class Sequence:
    key: str
    source_root: str
    request_id: str
    scenario_id: str | None
    audio_sha256: str | None
    records: list[dict[str, Any]]
    embeds: Any


def read_capture_records(root: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(record["call_index"]) for record in records] != list(range(len(records))):
        raise ValueError(f"non-contiguous capture manifest: {root}")
    return records


def selected_request_ids(selection: dict[str, Any], split: str) -> dict[int, set[str]]:
    return {
        int(entry["root_index"]): {str(item) for item in entry["include_request_ids"]}
        for entry in selection[split]
    }


def load_sequences(
    roots: Iterable[Path],
    *,
    torch,
    include_request_ids: dict[int, set[str]] | None = None,
    scenario_by_ordinal: list[dict[str, Any]] | None = None,
) -> list[Sequence]:
    sequences: list[Sequence] = []
    for root_index, root in enumerate(roots):
        root = root.resolve()
        grouped: dict[tuple[int, str], list[tuple[dict[str, Any], Any]]] = {}
        for record in read_capture_records(root):
            request_id = str(record.get("request_id", "default"))
            if request_id == "-1":
                continue
            if include_request_ids is not None and request_id not in include_request_ids[root_index]:
                continue
            payload_path = root / record["file"]
            if payload_path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"capture payload size mismatch: {payload_path}")
            if sha256_file(payload_path) != record["sha256"]:
                raise ValueError(f"capture payload SHA-256 mismatch: {payload_path}")
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            saved = payload["inputs"]
            embeds = saved["input_embeds"]
            rows = int(record.get("decode_steps", 1))
            if int(record.get("current_step", 0)) == 0 and embeds.shape[1] > rows:
                rows = int(embeds.shape[1])
            else:
                rows = min(int(embeds.shape[1]), rows)
            if rows <= 0:
                continue
            enriched = dict(record)
            result = payload["outputs"]["interface_result"]
            # A public Nano request begins with a decode_steps=0 prefill call.
            # The interface returns the boolean readiness sentinel for that
            # call rather than the token mapping returned by decode calls.
            # It contributes the supplied prefix rows, but emits no token.
            result_mapping = result if isinstance(result, dict) else {}
            enriched["text_token"] = scalar(result_mapping.get("predicted_token"))
            enriched["function_token"] = scalar(
                result_mapping.get("function_predicted_token")
            )
            enriched["payload_sha256"] = record["sha256"]
            key = (int(record.get("sequence_epoch", 0)), request_id)
            grouped.setdefault(key, []).append((enriched, embeds[:, :rows].detach().cpu()))
        for key in grouped:
            parts = grouped[key]
            ordinal = len(sequences)
            scenario = scenario_by_ordinal[ordinal] if scenario_by_ordinal else None
            scenario_id = scenario["scenario_id"] if scenario else None
            audio_sha = scenario["pcm"]["sha256"] if scenario else None
            records = [item[0] for item in parts]
            if any(part[1].shape[1] != 1 for part in parts):
                # The canonical index still expands any true prefill exactly once.
                expanded: list[dict[str, Any]] = []
                for record, embed in parts:
                    for local in range(embed.shape[1]):
                        row = dict(record)
                        row["effective_local_row"] = local
                        expanded.append(row)
                records = expanded
            sequences.append(
                Sequence(
                    key=f"{root_index}:{key[0]}:{key[1]}",
                    source_root=str(root),
                    request_id=key[1],
                    scenario_id=scenario_id,
                    audio_sha256=audio_sha,
                    records=records,
                    embeds=torch.cat([item[1] for item in parts], dim=1),
                )
            )
    return sequences


def run_layer(layer: Any, sequences: list[Any], torch) -> list[Any]:
    outputs = []
    with torch.inference_mode():
        for hidden in sequences:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            outputs.append(
                layer(
                    hidden,
                    cache_params=None,
                    cache_position=positions,
                    attention_mask=None,
                ).detach()
            )
    return outputs


def capture_inputs(layer: Any, sequences: list[Any], names: Iterable[str], torch):
    captured = {name: [] for name in names}
    handles = []
    for name in names:
        module = getattr(layer.mixer, name)

        def hook(_module, args, site=name):
            captured[site].append(args[0].detach())

        handles.append(module.register_forward_pre_hook(hook))
    try:
        outputs = run_layer(layer, sequences, torch)
    finally:
        for handle in handles:
            handle.remove()
    for name in names:
        if len(captured[name]) != len(sequences):
            raise RuntimeError(f"hook {name} observed {len(captured[name])}/{len(sequences)} calls")
    return outputs, captured


def prove_shared_qkv(captured: dict[str, list[Any]], torch) -> None:
    for ordinal, (q, k, v) in enumerate(
        zip(captured["q_proj"], captured["k_proj"], captured["v_proj"], strict=True)
    ):
        if not torch.equal(q, k) or not torch.equal(q, v):
            raise RuntimeError(f"q/k/v actual module arguments differ in sequence {ordinal}")


def quantize_weight(weight: Any, hessian: Any, torch):
    from nano_gptq_calibration import gptq_quantize_matrix, pack_gptq_codes

    h = hessian.detach().to(dtype=torch.float32)
    dead = torch.diag(h) == 0
    if bool(dead.any()):
        dead_rows = h[dead]
        dead_cols = h[:, dead]
        if bool((dead_rows != 0).any()) or bool((dead_cols != 0).any()):
            raise RuntimeError("zero-diagonal Hessian column has nonzero row/column entries")
    result, metrics = gptq_quantize_matrix(
        weight.detach().to(dtype=torch.float32),
        h,
        torch,
        bits=8,
        group_size=GROUP_SIZE,
        block_size=BLOCK_SIZE,
        damp_percent=DAMP_PERCENT,
    )
    packed = pack_gptq_codes(
        result["codes"], result["scales"], torch, bits=8, group_size=GROUP_SIZE
    )
    return result, packed, metrics


def hierarchy_hessian(
    activations: list[Any],
    row_indexes: list[list[int]],
    *,
    torch,
    device: str,
    positive_count: int | None = None,
):
    if len(activations) != len(row_indexes):
        raise ValueError("activation/index source count mismatch")
    source_h: list[Any] = []
    audit = []
    for ordinal, (activation, indexes) in enumerate(zip(activations, row_indexes, strict=True)):
        if not indexes or len(set(indexes)) != len(indexes):
            raise ValueError(f"source {ordinal} has empty or duplicate Hessian rows")
        selected = activation[0, indexes].to(device=device, dtype=torch.float32)
        raw = selected.t().matmul(selected)
        weighted = raw * (2.0 / len(indexes))
        source_h.append(weighted)
        audit.append(
            {
                "source": ordinal,
                "rows": len(indexes),
                "raw_trace": float(torch.trace(raw)),
                "source_weighted_trace": float(torch.trace(weighted)),
            }
        )
    if positive_count is None:
        h = torch.stack(source_h).mean(dim=0)
    else:
        positive = torch.stack(source_h[:positive_count]).mean(dim=0)
        no_call = torch.stack(source_h[positive_count:]).mean(dim=0)
        h = 0.5 * positive + 0.5 * no_call
    return h, audit


def target_weight_name(layer: int, projection: str) -> str:
    return f"stt_model.llm.layers.{layer}.mixer.{projection}.weight"


def packed_prefix(weight_name: str) -> str:
    return weight_name.removesuffix(".weight")


def parse_layer(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        raise ValueError(name)
    return int(match.group(1))
