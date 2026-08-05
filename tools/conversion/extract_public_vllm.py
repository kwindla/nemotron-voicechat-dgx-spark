#!/usr/bin/env python3
"""Identity-locked, reproducible vLLM extraction for public VoiceChat weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPOSITORY = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
REVISION = "fb0f94eaf4d03ddc430f39565229393fa1b50c26"
MODEL_BYTES = 44_382_749_892
MODEL_SHA256 = "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
MINIMUM_POST_CONVERSION_FREE_GIB = 100.0
SPEECH_COMMIT = "911ec674ab40f04302ef33672be4179f45a7310f"
NANO_SKELETON_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
RUNTIME_NANO_SKELETON_PATH = "/models/NVIDIA-Nemotron-Nano-9B-v2"
NANO_PREFIXES = (
    "stt_model.llm",
    "stt_model.lm_head",
    "stt_model.embed_tokens",
    "stt_model.function_head",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFIED_BOM = REPO_ROOT / "config/qualified-candidate-1.json"


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated SafeTensors length: {path}")
        length = struct.unpack("<Q", raw_length)[0]
        return json.loads(handle.read(length))


def tensor_inventory(
    path: Path, predicate: Callable[[str], bool] = lambda _name: True
) -> dict[str, Any]:
    header = safetensors_header(path)
    tensors = {
        name: metadata
        for name, metadata in header.items()
        if name != "__metadata__" and predicate(name)
    }
    dtype_bytes: Counter[str] = Counter()
    for metadata in tensors.values():
        start, end = metadata["data_offsets"]
        dtype_bytes[metadata["dtype"]] += end - start
    return {
        "tensor_count": len(tensors),
        "payload_bytes": sum(dtype_bytes.values()),
        "payload_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "tensors": {
            name: {"dtype": value["dtype"], "shape": value["shape"]}
            for name, value in sorted(tensors.items())
        },
    }


def verify_hashed_files(
    root: Path,
    expected_hashes: dict[str, str],
    *,
    expected_sizes: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify immutable payloads without depending on HF cache sidecars."""

    result: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in sorted(expected_hashes.items()):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing exact-public input: {path}")
        actual_bytes = path.stat().st_size
        if expected_sizes is not None and actual_bytes != int(expected_sizes[relative]):
            raise ValueError(
                f"Exact-public input size differs for {path}: "
                f"{actual_bytes} != {expected_sizes[relative]}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Exact-public input SHA256 differs for {path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        result[relative] = {"bytes": actual_bytes, "sha256": actual_sha256}
    return result


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def require_empty_target(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing non-empty conversion target: {path}")
    path.mkdir(parents=True, exist_ok=True)


def available_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def disk_budget(
    source_model: Path,
    output_root: Path,
    component: str,
    reproducibility_runs: int,
    safety_margin_bytes: int,
) -> dict[str, int]:
    nano_payload = tensor_inventory(source_model, lambda name: name.startswith(NANO_PREFIXES))[
        "payload_bytes"
    ]
    eartts_payload = tensor_inventory(
        source_model, lambda name: name.startswith("tts_model.tts_model.")
    )["payload_bytes"]
    # The NVIDIA EarTTS export duplicates/pads small embedding/RVQ tensors.
    # Reserve 25% over the selected inner-model payload until a real output
    # replaces this estimate in the retained manifest.
    eartts_estimate = (eartts_payload * 5 + 3) // 4
    nano_final = output_root / "nano-vllm-fp32"
    eartts_final = output_root / "eartts-vllm-fp32"
    nano_exists = nano_final.is_dir() and any(nano_final.iterdir())
    eartts_exists = eartts_final.is_dir() and any(eartts_final.iterdir())
    nano_new_runs = reproducibility_runs - int(nano_exists)
    eartts_new_runs = reproducibility_runs - int(eartts_exists)
    if component == "nano":
        peak_outputs = nano_payload * nano_new_runs
    elif component == "eartts":
        peak_outputs = eartts_estimate * eartts_new_runs
    else:
        peak_outputs = max(
            nano_payload * nano_new_runs,
            eartts_estimate * eartts_new_runs,
        )
    free = available_bytes(output_root)
    return {
        "free_bytes": free,
        "nano_output_estimate_bytes": nano_payload,
        "eartts_output_estimate_bytes": eartts_estimate,
        "peak_output_bytes": peak_outputs,
        "safety_margin_bytes": safety_margin_bytes,
        "minimum_remaining_free_bytes": safety_margin_bytes,
        "projected_remaining_free_bytes": free - peak_outputs,
        "required_free_bytes": peak_outputs + safety_margin_bytes,
        "existing_nano_output": nano_exists,
        "existing_eartts_output": eartts_exists,
        "new_nano_runs": nano_new_runs,
        "new_eartts_runs": eartts_new_runs,
    }


def verify_source(speech_root: Path, checkpoint: Path, skeleton: Path) -> dict[str, Any]:
    model = checkpoint / "model.safetensors"
    bom = json.loads(QUALIFIED_BOM.read_text(encoding="utf-8"))
    checkpoint_contract = bom["public_sources"]["checkpoint"]
    skeleton_contract = bom["public_sources"]["nano_skeleton"]
    if (
        checkpoint_contract["repository"] != REPOSITORY
        or checkpoint_contract["revision"] != REVISION
        or checkpoint_contract["bytes"] != MODEL_BYTES
        or checkpoint_contract["sha256"] != MODEL_SHA256
        or skeleton_contract["revision"] != NANO_SKELETON_REVISION
    ):
        raise ValueError("checked-in public-source contract disagrees with the converter")
    checkpoint_files = verify_hashed_files(
        checkpoint,
        checkpoint_contract["files"],
        expected_sizes=checkpoint_contract["file_sizes"],
    )
    skeleton_files = verify_hashed_files(skeleton, skeleton_contract["files"])
    revision = checkpoint_contract["revision"]
    skeleton_revision = skeleton_contract["revision"]
    commit = git_commit(speech_root)
    if commit != SPEECH_COMMIT:
        raise ValueError(f"Speech commit {commit}; expected {SPEECH_COMMIT}")
    actual_sha = checkpoint_files["model.safetensors"]["sha256"]
    converter_paths = {
        "nano": speech_root
        / "nemo/collections/speechlm2/inference/vllm/scripts/convert_nemotronllm_checkpoint.py",
        "eartts": speech_root
        / "nemo/collections/speechlm2/inference/vllm/scripts/convert_eartts_checkpoint.py",
    }
    return {
        "repository": REPOSITORY,
        "revision": revision,
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": actual_sha,
            "inventory": tensor_inventory(model),
            "verified_files": checkpoint_files,
        },
        "nano_skeleton": {
            "path": str(skeleton),
            "revision": skeleton_revision,
            "weights_present": any(skeleton.glob("*.safetensors")),
            "verified_files": skeleton_files,
        },
        "speech": {
            "path": str(speech_root),
            "commit": commit,
            "converters": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in converter_paths.items()
            },
        },
        "runtime": {
            "image": os.environ.get("VOICECHAT_RUNTIME_IMAGE"),
            "image_id": os.environ.get("VOICECHAT_RUNTIME_IMAGE_ID"),
        },
    }


def annotate_nano_config(path: Path) -> None:
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "voicechat_checkpoint_repository": REPOSITORY,
            "voicechat_checkpoint_revision": REVISION,
            "voicechat_checkpoint_sha256": MODEL_SHA256,
            "voicechat_has_asr_head": False,
            "voicechat_has_function_head": True,
            "voicechat_derived_from_ea": False,
        }
    )
    custom_inputs = config.get("custom_input_specs") or []
    combined = [item for item in custom_inputs if item.get("name") == "combined_embeds"]
    if len(combined) != 1 or combined[0].get("dtype") != "bfloat16":
        raise ValueError(f"Unexpected Nano custom input contract: {custom_inputs}")
    if config.get("custom_outputs") != [
        "text_logits",
        "function_tokens",
        "function_logits",
    ]:
        raise ValueError(f"Unexpected Nano custom outputs: {config.get('custom_outputs')}")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def convert_nano(speech_root: Path, checkpoint: Path, skeleton: Path, target: Path) -> None:
    sys.path.insert(0, str(speech_root))
    from nemo.collections.speechlm2.inference.vllm.scripts.convert_nemotronllm_checkpoint import (
        convert_nemo_to_hf_format,
    )

    require_empty_target(target)
    convert_nemo_to_hf_format(
        checkpoint_path=str(checkpoint / "model.safetensors"),
        output_dir=str(target),
        config_path=str(checkpoint / "__force_local_nano_skeleton__.missing"),
        pretrained_llm=str(skeleton),
        tensors_to_keep=list(NANO_PREFIXES),
        dtype="bfloat16",
    )
    annotate_nano_config(target)


def convert_eartts(speech_root: Path, checkpoint: Path, skeleton: Path, target: Path) -> None:
    sys.path.insert(0, str(speech_root))
    import torch
    from nemo.collections.speechlm2.inference.vllm.scripts.convert_eartts_checkpoint import (
        convert,
    )

    require_empty_target(target)
    # NVIDIA's combined public config names the remote Nano tokenizer. The
    # project is intentionally offline after acquisition, so use a temporary
    # construction-only config that routes this lookup to the exact local
    # code/config/tokenizer skeleton. No source file or converted weights are
    # changed by this deviation.
    local_config_path = target / ".public-local-converter-config.json"
    local_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    local_config["model"]["speech_generation"]["model"]["pretrained_lm_name"] = str(skeleton)
    local_config_path.write_text(json.dumps(local_config, indent=2) + "\n", encoding="utf-8")
    torch.manual_seed(0)
    try:
        convert(
            str(target),
            str(local_config_path),
            str(checkpoint / "model.safetensors"),
        )
    finally:
        local_config_path.unlink(missing_ok=True)
    annotate_eartts_config(target, skeleton)


def annotate_eartts_config(path: Path, skeleton: Path) -> None:
    """Bind EarTTS's runtime tokenizer lookup to the verified local Nano skeleton.

    vLLM constructs two auxiliary EarTTS embeddings by calling
    ``AutoTokenizer.from_pretrained`` in its engine subprocess.  Leaving the
    upstream repository id in the converted config makes an otherwise local
    artifact attempt a Hugging Face lookup at startup.  The exact Nano
    skeleton is already a required, revision-checked runtime input, so record
    its container path explicitly and keep the full stack offline.
    """
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "pretrained_tokenizer_name": RUNTIME_NANO_SKELETON_PATH,
            "voicechat_checkpoint_repository": REPOSITORY,
            "voicechat_checkpoint_revision": REVISION,
            "voicechat_checkpoint_sha256": MODEL_SHA256,
            "voicechat_derived_from_ea": False,
            "voicechat_converter_seed": 0,
            "voicechat_tokenizer_source": {
                "repository": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                "revision": NANO_SKELETON_REVISION,
                "path": RUNTIME_NANO_SKELETON_PATH,
            },
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def verify_component(component: str, source_model: Path, target: Path) -> dict[str, Any]:
    model = target / "model.safetensors"
    if not model.is_file():
        raise ValueError(f"Missing converted {component} model: {model}")
    inventory = tensor_inventory(model)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config.get("voicechat_checkpoint_sha256") != MODEL_SHA256:
        raise ValueError(f"Converted {component} config lacks exact-public SHA")
    if config.get("voicechat_derived_from_ea") is not False:
        raise ValueError(f"Converted {component} is not explicitly public-only")
    if component == "nano":
        expected = tensor_inventory(source_model, lambda name: name.startswith(NANO_PREFIXES))
        if inventory["tensors"] != expected["tensors"]:
            raise ValueError("Nano output shape/dtype inventory differs from public source")
        if set(inventory["payload_bytes_by_dtype"]) != {"F32"}:
            raise ValueError("Nano stored tensors were unexpectedly cast")
        if "stt_model.function_head.weight" not in inventory["tensors"]:
            raise ValueError("Public Nano output dropped function head")
        if any("asr_head" in name for name in inventory["tensors"]):
            raise ValueError("Public Nano output contains retired EA ASR head")
    files = hash_tree(target)
    return {
        "path": str(target),
        "model": {
            "bytes": model.stat().st_size,
            "sha256": files["model.safetensors"]["sha256"],
            "inventory": inventory,
        },
        "files": files,
        "config": config,
    }


def convert_reproducibly(
    component: str,
    output_root: Path,
    converter: Callable[[Path], None],
    source_model: Path,
    runs: int,
) -> dict[str, Any]:
    final = output_root / f"{component}-vllm-fp32"
    final_exists = final.is_dir() and any(final.iterdir())
    targets = [] if final_exists else [final]
    targets.extend(output_root / f".{component}-repro-{uuid.uuid4().hex}" for _ in range(runs - 1))
    reports = [verify_component(component, source_model, final)] if final_exists else []
    for index, target in enumerate(targets, start=len(reports) + 1):
        print(
            f"Converting {component} reproducibility run {index}/{runs}: {target}",
            flush=True,
        )
        converter(target)
        reports.append(verify_component(component, source_model, target))
    reference_files = reports[0]["files"]
    for report in reports[1:]:
        if report["files"] != reference_files:
            raise ValueError(f"{component} conversion is not byte reproducible")
    duplicates = targets if final_exists else targets[1:]
    for duplicate in duplicates:
        shutil.rmtree(duplicate)
    reports[0]["reproducibility"] = {
        "runs": runs,
        "byte_identical": True,
        "verified_file_hashes": reference_files,
        "verified_duplicates_removed": [str(path) for path in duplicates],
    }
    return reports[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--nano-skeleton", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--component", choices=("nano", "eartts", "all"), default="all")
    parser.add_argument("--reproducibility-runs", type=int, default=2)
    parser.add_argument(
        "--safety-margin-gib",
        type=float,
        default=MINIMUM_POST_CONVERSION_FREE_GIB,
        help=(
            "Minimum free space that must remain after estimated peak output; "
            f"values below {MINIMUM_POST_CONVERSION_FREE_GIB:g} GiB are refused"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reproducibility_runs < 2:
        raise SystemExit("At least two reproducibility runs are required")
    if args.safety_margin_gib < MINIMUM_POST_CONVERSION_FREE_GIB:
        raise SystemExit(
            "Conversion safety margin cannot be lower than "
            f"{MINIMUM_POST_CONVERSION_FREE_GIB:g} GiB"
        )
    speech_root = args.speech_root.resolve()
    checkpoint = args.checkpoint_root.resolve()
    skeleton = args.nano_skeleton.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior_manifest_path = output_root / "manifest.json"
    prior_manifest = (
        json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        if prior_manifest_path.is_file()
        else {}
    )
    started = time.time()
    source = verify_source(speech_root, checkpoint, skeleton)
    source_model = checkpoint / "model.safetensors"
    budget = disk_budget(
        source_model,
        output_root,
        args.component,
        args.reproducibility_runs,
        int(args.safety_margin_gib * 1024**3),
    )
    print(
        json.dumps(
            {
                "source": {
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "model_bytes": source["model"]["bytes"],
                    "model_sha256": source["model"]["sha256"],
                    "model_tensors": source["model"]["inventory"]["tensor_count"],
                    "speech_commit": source["speech"]["commit"],
                    "runtime": source["runtime"],
                },
                "disk_budget": budget,
            },
            indent=2,
        ),
        flush=True,
    )
    if budget["free_bytes"] < budget["required_free_bytes"]:
        raise SystemExit(
            f"Insufficient free space: {budget['free_bytes']} available, "
            f"{budget['required_free_bytes']} required"
        )
    if args.preflight_only:
        return

    components: dict[str, Any] = {}
    if args.component in ("nano", "all"):
        components["nano"] = convert_reproducibly(
            "nano",
            output_root,
            lambda target: convert_nano(speech_root, checkpoint, skeleton, target),
            source_model,
            args.reproducibility_runs,
        )
    if args.component in ("eartts", "all"):
        # Upgrade a retained pre-fix conversion before creating the independent
        # comparison run.  The new run receives the same deterministic
        # annotation in convert_eartts(), so the full trees still have to match.
        retained_eartts = output_root / "eartts-vllm-fp32"
        if retained_eartts.is_dir() and any(retained_eartts.iterdir()):
            annotate_eartts_config(retained_eartts, skeleton)
        components["eartts"] = convert_reproducibly(
            "eartts",
            output_root,
            lambda target: convert_eartts(speech_root, checkpoint, skeleton, target),
            source_model,
            args.reproducibility_runs,
        )
    for name in ("nano", "eartts"):
        final = output_root / f"{name}-vllm-fp32"
        if name not in components and final.is_dir() and any(final.iterdir()):
            verified = verify_component(name, source_model, final)
            prior = (prior_manifest.get("components") or {}).get(name) or {}
            if prior.get("model", {}).get("sha256") == verified["model"]["sha256"]:
                if "reproducibility" in prior:
                    verified["reproducibility"] = prior["reproducibility"]
            components[name] = verified
    manifest = {
        "schema": 1,
        "kind": "exact_public_vllm_extraction",
        "derived_from_ea": False,
        "source": source,
        "disk_budget": budget,
        "components": components,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, report in components.items():
        (output_root / f"{name}-extraction.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote exact-public extraction manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
