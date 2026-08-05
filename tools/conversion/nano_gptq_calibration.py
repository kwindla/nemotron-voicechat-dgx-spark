#!/usr/bin/env python3
"""Exact-public fused-embedding calibration primitives for Nano GPTQ.

This module deliberately has no dependency on NeMo, vLLM, or GPTQModel.  The
full converter imports it inside NVIDIA's public runtime image, where PyTorch
and safetensors are already pinned.  Small selection, disk, corpus, and GPTQ
math tests can therefore exercise the critical contracts independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GIB = 1024**3
HARD_MIN_REMAINING_GIB = 100.0
PUBLIC_REPOSITORY = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
PUBLIC_REVISION = "fb0f94eaf4d03ddc430f39565229393fa1b50c26"


MLP_SUFFIXES = (
    ".mixer.up_proj.weight",
    ".mixer.down_proj.weight",
)
MAMBA_SUFFIXES = (
    ".mixer.in_proj.weight",
    ".mixer.out_proj.weight",
)
TEXT_HEAD = "stt_model.lm_head.weight"
FUNCTION_HEAD = "stt_model.function_head.weight"
POLICY_W8_ALL_G128 = "w8-all-g128"
GPTQ_POLICIES = (POLICY_W8_ALL_G128,)
EMBEDDING_SUFFIXES = (
    ".embed_tokens.weight",
    ".embed_asr_tokens.weight",
)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_gptq_target(name: str) -> bool:
    """Select exactly MLP + Mamba projections and the conversational head."""
    if name == TEXT_HEAD:
        return True
    if not name.startswith("stt_model.llm.layers."):
        return False
    return name.endswith(MLP_SUFFIXES + MAMBA_SUFFIXES)


def _require_release_policy(policy: str) -> None:
    if policy != POLICY_W8_ALL_G128:
        raise ValueError(f"unsupported GPTQ policy: {policy}")


def bits_for_target(name: str, policy: str = POLICY_W8_ALL_G128) -> int:
    if not is_gptq_target(name):
        raise ValueError(f"not a GPTQ target: {name}")
    _require_release_policy(policy)
    return 8


def group_size_for_target(name: str, policy: str = POLICY_W8_ALL_G128) -> int:
    if not is_gptq_target(name):
        raise ValueError(f"not a GPTQ target: {name}")
    _require_release_policy(policy)
    return 128


def gptq_quantization_config(policy: str = POLICY_W8_ALL_G128) -> dict[str, Any]:
    """vLLM GPTQ/Marlin metadata for the one bounded public candidate."""
    _require_release_policy(policy)
    dynamic = {
        r"-:.*(?:q_proj|k_proj|v_proj|o_proj)$": {},
        r"-:.*function_head$": {},
        r"-:.*(?:embed_tokens|embed_asr_tokens)$": {},
    }
    return {
        "bits": 8,
        "checkpoint_format": "gptq",
        "desc_act": False,
        "group_size": 128,
        "lm_head": True,
        "modules_in_block_to_quantize": [
            "mixer.up_proj",
            "mixer.down_proj",
            "mixer.in_proj",
            "mixer.out_proj",
            "lm_head",
        ],
        "quant_method": "gptq",
        "sym": True,
        # Make exclusions machine-readable. Attention, embeddings, and the
        # always-on function head remain byte-for-byte source data.
        "dynamic": dynamic,
    }


def projected_quantized_tensor_bytes(shape: Iterable[int], bits: int, group_size: int = 128) -> int:
    n, k = (int(value) for value in shape)
    if group_size == -1:
        group_size = k
    if k % group_size:
        raise ValueError(f"group size {group_size} does not divide K={k}")
    pack_factor = 32 // bits
    if n % pack_factor or k % pack_factor:
        raise ValueError(f"W{bits} packed dimensions must be divisible by {pack_factor}: {(n, k)}")
    groups = k // group_size
    # qweight + BF16 scale + qzeros + g_idx in the pinned vLLM GPTQ layout.
    return k * n * bits // 8 + groups * n * 2 + groups * (n // pack_factor) * 4 + k * 4


def conversion_disk_preflight(
    source: Path,
    output: Path,
    work_dir: Path,
    *,
    estimated_output_bytes: int,
    reproducibility_runs: int = 2,
    min_remaining_gib: float = HARD_MIN_REMAINING_GIB,
    free_bytes: int | None = None,
) -> dict[str, Any]:
    """Refuse conversion unless projected peak leaves at least 100 GiB.

    Peak new bytes conservatively include an HF-name staging copy, one dense
    offload copy, packed spool data, and every reproducibility output.  Existing
    source and replay data are already reflected in filesystem free space.
    """
    if min_remaining_gib < HARD_MIN_REMAINING_GIB:
        raise ValueError(
            "minimum remaining space cannot be below the hard 100 GiB conversion floor"
        )
    if reproducibility_runs < 2:
        raise ValueError("GPTQ requires at least two reproducibility runs")
    for target, label in ((output, "output"), (work_dir, "work directory")):
        if target.exists():
            raise FileExistsError(f"{label} already exists: {target}")
    model_path = source / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    source_bytes = model_path.stat().st_size
    projected_new_bytes = (
        source_bytes * 2  # HF staging plus conservative dense/offload scratch
        + estimated_output_bytes  # packed per-module spool
        + estimated_output_bytes * reproducibility_runs
    )
    usage = shutil.disk_usage(output.parent.resolve())
    available = usage.free if free_bytes is None else int(free_bytes)
    remaining = available - projected_new_bytes
    floor_bytes = math.ceil(min_remaining_gib * GIB)
    result = {
        "filesystem": str(output.parent.resolve()),
        "free_bytes": available,
        "source_model_bytes": source_bytes,
        "estimated_output_bytes": int(estimated_output_bytes),
        "reproducibility_runs": reproducibility_runs,
        "projected_peak_new_bytes": projected_new_bytes,
        "projected_remaining_bytes": remaining,
        "minimum_remaining_bytes": floor_bytes,
        "minimum_remaining_gib": float(min_remaining_gib),
    }
    if remaining < floor_bytes:
        raise RuntimeError(
            "Nano GPTQ conversion disk preflight failed: projected peak would leave "
            f"{remaining / GIB:.2f} GiB, below the {min_remaining_gib:.2f} GiB floor"
        )
    return result


@dataclass(frozen=True)
class ReplayCorpus:
    roots: tuple[Path, ...]
    records: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    manifest_sha256: str
    calls: int
    effective_rows: int
    conversations: int


def _validated_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    manifest_path = root / "manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"replay root is incomplete: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "public_nano_deterministic_replay":
        raise ValueError(f"not a public Nano replay: {root}")
    provenance = metadata.get("checkpoint_provenance") or {}
    if provenance.get("repository") != PUBLIC_REPOSITORY:
        raise ValueError(f"wrong replay repository: {root}")
    if provenance.get("revision") != PUBLIC_REVISION:
        raise ValueError(f"wrong replay revision: {root}")
    derived_from_ea = provenance.get("derived_from_ea")
    if derived_from_ea is None:
        derived_from_ea = (provenance.get("vllm_artifacts") or {}).get("derived_from_ea")
    if derived_from_ea is not False:
        raise ValueError(f"EA-derived replay is forbidden: {root}")
    return metadata


def _provenance_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    artifacts = provenance.get("vllm_artifacts") or {}
    components = artifacts.get("components") or {}
    nano = components.get("nano") or {}
    return {
        "repository": provenance.get("repository"),
        "revision": provenance.get("revision"),
        "model_sha256": provenance.get("model_sha256"),
        "nano_sha256": provenance.get("nano_sha256") or nano.get("sha256"),
        "derived_from_ea": (
            provenance.get("derived_from_ea")
            if provenance.get("derived_from_ea") is not None
            else artifacts.get("derived_from_ea")
        ),
    }


def effective_rows(record: dict[str, Any], sequence_length: int) -> int:
    decode_steps = int(record.get("decode_steps", 1))
    if decode_steps < 0:
        raise ValueError(f"negative decode_steps in call {record.get('call_index')}")
    # Prefill consumes the entire supplied prefix; decode consumes at most the
    # requested rows (the release normally requests exactly one).
    return sequence_length if decode_steps == 0 else min(sequence_length, decode_steps)


def inspect_replay_corpus(
    roots: Iterable[Path], *, verify_payload_hashes: bool = True
) -> ReplayCorpus:
    roots = tuple(Path(root).expanduser().resolve() for root in roots)
    if not roots:
        raise ValueError("at least one replay root is required")
    all_records: list[dict[str, Any]] = []
    conversations: set[tuple[int, int]] = set()
    provenance: dict[str, Any] | None = None
    effective_total = 0
    for capture_index, root in enumerate(roots):
        metadata = _validated_metadata(root)
        current = metadata["checkpoint_provenance"]
        identity = _provenance_identity(current)
        if provenance is None:
            provenance = identity
        elif identity != provenance:
            raise ValueError(f"mixed checkpoint provenance at {root}")
        lines = [
            line for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()
        ]
        records = [json.loads(line) for line in lines]
        if [item.get("call_index") for item in records] != list(range(len(records))):
            raise ValueError(f"non-contiguous replay calls: {root}")
        request_conversations: dict[str, int] = {}
        active_conversation = 0
        for item in records:
            payload = root / item["file"]
            if not payload.is_file() or payload.stat().st_size != int(item["bytes"]):
                raise ValueError(f"missing or truncated replay payload: {payload}")
            if verify_payload_hashes and sha256_file(payload) != item["sha256"]:
                raise ValueError(f"replay payload hash mismatch: {payload}")
            enriched = dict(item)
            enriched["capture_index"] = capture_index
            enriched["capture_root"] = str(root)
            request_id = str(item.get("request_id", "default"))
            if request_id != "-1":
                if request_id not in request_conversations:
                    request_conversations[request_id] = len(request_conversations)
                active_conversation = request_conversations[request_id]
            enriched["normalized_conversation_index"] = active_conversation
            all_records.append(enriched)
            conversations.add((capture_index, active_conversation))
            # The tensor shape is intentionally read by load_fused_sequences;
            # use the manifest's causal contract here when no tensor library is
            # imported.  Every recorded call contributes at least one row.
            effective_total += max(1, int(item.get("decode_steps", 1)))
    manifest_payload = {
        "schema": 1,
        "kind": "exact_public_nano_gptq_calibration_corpus",
        "roots": [str(root) for root in roots],
        "provenance": provenance,
        "records": all_records,
    }
    return ReplayCorpus(
        roots=roots,
        records=tuple(all_records),
        provenance=provenance or {},
        manifest_sha256=canonical_json_sha256(manifest_payload),
        calls=len(all_records),
        effective_rows=effective_total,
        conversations=len(conversations),
    )


def assert_disjoint_corpora(calibration: ReplayCorpus, evaluation: ReplayCorpus) -> None:
    shared_roots = set(calibration.roots) & set(evaluation.roots)
    if shared_roots:
        raise ValueError(f"calibration/evaluation roots overlap: {sorted(map(str, shared_roots))}")
    calibration_hashes = {item["sha256"] for item in calibration.records}
    evaluation_hashes = {item["sha256"] for item in evaluation.records}
    shared_payloads = calibration_hashes & evaluation_hashes
    if shared_payloads:
        raise ValueError(f"calibration/evaluation payloads overlap: {len(shared_payloads)}")


def select_replay_corpus(
    corpus: ReplayCorpus,
    selection_manifest: dict[str, Any],
    *,
    split: str,
) -> ReplayCorpus:
    """Apply an exhaustive request-ID whitelist with rejection reasons.

    Entries are keyed by stable root index rather than host/container paths so
    the same manifest remains valid when captures are mounted read-only at a
    different location. Every non-prefill request must be explicitly accepted
    or rejected; silent/unreviewed evidence is a hard error.
    """
    if (
        selection_manifest.get("schema") != 1
        or selection_manifest.get("kind") != "nano_gptq_corpus_selection"
    ):
        raise ValueError("invalid Nano GPTQ corpus selection manifest")
    entries = selection_manifest.get(split)
    if not isinstance(entries, list):
        raise ValueError(f"selection manifest has no {split!r} entry list")
    by_root: dict[int, dict[str, Any]] = {}
    for entry in entries:
        root_index = int(entry.get("root_index", -1))
        if root_index in by_root or root_index < 0 or root_index >= len(corpus.roots):
            raise ValueError(f"invalid or duplicate {split} root_index: {root_index}")
        included = {str(value) for value in entry.get("include_request_ids", [])}
        excluded_raw = entry.get("exclude_request_ids", {})
        if not isinstance(excluded_raw, dict):
            raise ValueError(f"{split} excluded request IDs must map IDs to reasons")
        excluded = {str(key): str(value).strip() for key, value in excluded_raw.items()}
        if not included:
            raise ValueError(f"{split} root {root_index} has no accepted request IDs")
        if included & set(excluded):
            raise ValueError(f"{split} root {root_index} both accepts and rejects a request")
        if any(not reason for reason in excluded.values()):
            raise ValueError(f"{split} root {root_index} has an empty rejection reason")
        by_root[root_index] = {
            "include": included,
            "exclude": excluded,
            "label": entry.get("label"),
        }
    if set(by_root) != set(range(len(corpus.roots))):
        raise ValueError(
            f"{split} selection must classify every replay root: "
            f"actual={sorted(by_root)}, expected={list(range(len(corpus.roots)))}"
        )

    observed: dict[int, set[str]] = {index: set() for index in by_root}
    selected_records = []
    for record in corpus.records:
        root_index = int(record["capture_index"])
        request_id = str(record.get("request_id", "default"))
        if request_id != "-1":
            observed[root_index].add(request_id)
        rule = by_root[root_index]
        if request_id == "-1" or request_id in rule["include"]:
            selected_records.append(record)
    for root_index, request_ids in observed.items():
        rule = by_root[root_index]
        classified = rule["include"] | set(rule["exclude"])
        if request_ids != classified:
            raise ValueError(
                f"{split} root {root_index} request classification is not exhaustive: "
                f"observed={sorted(request_ids)}, classified={sorted(classified)}"
            )
        missing_included = rule["include"] - request_ids
        if missing_included:
            raise ValueError(
                f"{split} root {root_index} accepted IDs are absent: {sorted(missing_included)}"
            )

    selected_conversations = {
        (int(record["capture_index"]), int(record["normalized_conversation_index"]))
        for record in selected_records
        if str(record.get("request_id")) != "-1"
    }
    if not selected_conversations:
        raise ValueError(f"{split} selection produced no conversations")
    payload = {
        "schema": 1,
        "kind": "selected_exact_public_nano_gptq_corpus",
        "split": split,
        "source_manifest_sha256": corpus.manifest_sha256,
        "selection_manifest_sha256": canonical_json_sha256(selection_manifest),
        "records": selected_records,
    }
    return ReplayCorpus(
        roots=corpus.roots,
        records=tuple(selected_records),
        provenance=corpus.provenance,
        manifest_sha256=canonical_json_sha256(payload),
        calls=len(selected_records),
        effective_rows=sum(
            max(1, int(record.get("decode_steps", 1))) for record in selected_records
        ),
        conversations=len(selected_conversations),
    )


def load_fused_sequences(corpus: ReplayCorpus, torch, *, device: str = "cpu") -> list[Any]:
    """Rebuild exact causal fused-embedding sequences without tokenization."""
    sequences: dict[tuple[int, int, str], list[Any]] = {}
    for record in corpus.records:
        payload = torch.load(
            Path(record["capture_root"]) / record["file"],
            map_location="cpu",
            weights_only=False,
        )
        saved = payload["inputs"]
        embeds = saved.get("input_embeds")
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1:
            raise ValueError(f"invalid fused embeddings in {record['file']}")
        rows = effective_rows(record, int(embeds.shape[1]))
        if rows == 0:
            continue
        key = (
            int(record["capture_index"]),
            int(record.get("sequence_epoch", 0)),
            str(record.get("request_id", saved.get("request_id", "default"))),
        )
        sequences.setdefault(key, []).append(embeds[:, :rows].detach().cpu())
    result = [torch.cat(parts, dim=1).to(device=device) for _, parts in sorted(sequences.items())]
    if not result:
        raise ValueError("calibration corpus produced no fused sequences")
    return result


def hessian_from_activations(activations: Iterable[Any], torch, *, device: str) -> tuple[Any, int]:
    hessian = None
    rows = 0
    for activation in activations:
        flat = (
            activation.detach()
            .reshape(-1, activation.shape[-1])
            .to(device=device, dtype=torch.float32)
        )
        if flat.numel() == 0:
            continue
        contribution = flat.t().matmul(flat)
        hessian = contribution if hessian is None else hessian.add_(contribution)
        rows += int(flat.shape[0])
    if hessian is None or rows == 0:
        raise ValueError("no activation rows for Hessian")
    return hessian.mul_(2.0 / rows), rows


def _symmetric_group_scales(weight, torch, *, bits: int, group_size: int):
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"group size {group_size} does not divide K={columns}")
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    grouped = weight.reshape(rows, columns // group_size, group_size)
    maxima = grouped.amax(dim=2)
    minima = grouped.amin(dim=2)
    scales = torch.maximum(maxima.abs() / qmax, minima.abs() / -qmin)
    return torch.where(scales == 0, torch.ones_like(scales), scales)


def gptq_quantize_matrix(
    weight,
    hessian,
    torch,
    *,
    bits: int,
    group_size: int = 128,
    block_size: int = 128,
    damp_percent: float = 0.01,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Classic symmetric GPTQ with static groups and deterministic ordering.

    Returns signed integer codes in HF [N,K] orientation, group scales in
    [K/group,N] orientation, and the BF16 dequantized teacher-propagation
    weight.  Packing into vLLM's int32 layout is kept separate and testable.
    """
    if bits != 8:
        raise ValueError("the release converter supports only W8")
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("weight/Hessian shape mismatch")
    if group_size == -1:
        group_size = int(weight.shape[1])
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("group_size must divide the input dimension")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    work = weight.detach().to(device=hessian.device, dtype=torch.float32).clone()
    hessian = hessian.detach().to(device=work.device, dtype=torch.float32).clone()
    dead = torch.diag(hessian) == 0
    if bool(dead.any()):
        # A zero Hessian diagonal means this calibration corpus never activates
        # the input column. Isolate it from GPTQ error propagation, but retain
        # and quantize its weight. Zeroing the weight (the common reference
        # implementation shortcut) can create large held-out error even at W8.
        hessian[dead, dead] = 1.0
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    damp = float(damp_percent) * float(torch.diag(hessian).mean())
    hessian[diagonal, diagonal] += damp
    chol = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(chol)
    hessian_inverse = torch.linalg.cholesky(inverse, upper=True)
    scales_n_g = _symmetric_group_scales(work, torch, bits=bits, group_size=group_size)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    codes = torch.empty_like(work, dtype=torch.int32)
    losses = torch.zeros_like(work)
    for block_start in range(0, work.shape[1], block_size):
        block_end = min(block_start + block_size, work.shape[1])
        count = block_end - block_start
        block = work[:, block_start:block_end].clone()
        block_errors = torch.zeros_like(block)
        for offset in range(count):
            column = block[:, offset]
            absolute_column = block_start + offset
            group = absolute_column // group_size
            scale = scales_n_g[:, group]
            quant = torch.round(column / scale).clamp_(qmin, qmax).to(torch.int32)
            dequant = quant.to(torch.float32) * scale
            pivot = hessian_inverse[absolute_column, absolute_column]
            error = (column - dequant) / pivot
            block[:, offset:] -= error.unsqueeze(1).matmul(
                hessian_inverse[absolute_column, absolute_column:block_end].unsqueeze(0)
            )
            codes[:, absolute_column] = quant
            losses[:, absolute_column] = (column - dequant).square() / (pivot * pivot)
            block_errors[:, offset] = error
        work[:, block_end:] -= block_errors.matmul(
            hessian_inverse[block_start:block_end, block_end:]
        )
    expanded = scales_n_g.repeat_interleave(group_size, dim=1)
    dequantized = codes.to(torch.float32).mul_(expanded)
    metrics = {
        "bits": bits,
        "group_size": group_size,
        "block_size": block_size,
        "damp_percent": damp_percent,
        "hessian_damp": damp,
        "dead_columns": int(dead.sum()),
        "dead_column_policy": "isolated_quantize_preserve_weight",
        "gptq_loss_sum": float(losses.sum()),
        "relative_rmse": float(
            (dequantized - weight.to(dequantized)).square().mean().sqrt()
            / weight.to(dequantized).square().mean().sqrt()
        ),
    }
    return {
        "codes": codes.cpu(),
        "scales": scales_n_g.t().contiguous().cpu(),
        "dequantized": dequantized.to(dtype=torch.bfloat16),
    }, metrics


def pack_gptq_codes(codes, scales, torch, *, bits: int, group_size: int) -> dict[str, Any]:
    """Pack signed [N,K] GPTQ codes into the proven public vLLM layout."""
    n, k = codes.shape
    if group_size == -1:
        group_size = k
    pack_factor = 32 // bits
    if k % pack_factor or n % pack_factor or k % group_size:
        raise ValueError("codes are not aligned for GPTQ/Marlin packing")
    q_unsigned = codes.to(torch.int32).t().contiguous() + 2 ** (bits - 1)
    packed = torch.zeros((k // pack_factor, n), dtype=torch.int32)
    mask = (1 << bits) - 1
    for offset in range(pack_factor):
        packed.bitwise_or_((q_unsigned[offset::pack_factor] & mask) << (bits * offset))
    groups = k // group_size
    if tuple(scales.shape) != (groups, n):
        raise ValueError(f"scale shape mismatch: {tuple(scales.shape)} != {(groups, n)}")
    return {
        "qweight": packed,
        "scales": scales.to(torch.bfloat16).contiguous(),
        "qzeros": torch.zeros((groups, n // pack_factor), dtype=torch.int32),
        "g_idx": torch.arange(groups, dtype=torch.int32).repeat_interleave(group_size),
    }


def tree_digest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "name": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "sha256": canonical_json_sha256(files)}


def assert_byte_identical(left: Path, right: Path) -> dict[str, Any]:
    left_digest, right_digest = tree_digest(left), tree_digest(right)
    if left_digest != right_digest:
        raise ValueError("dual GPTQ conversion outputs are not byte-identical")
    return {"runs": 2, "byte_identical": True, "tree_sha256": left_digest["sha256"]}
