"""Fail-closed validation for the immutable Step-9 vLLM compile cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

SCHEMA = "nemotron-voicechat-step9-vllm-cache-v1"
DEFAULT_CACHE_ROOT = Path("/root/.cache/vllm")
MANIFEST_NAME = "voicechat-cache-manifest.json"
COLD_CACHE_ROOT = Path("/tmp/voicechat-step9-cold-vllm-cache")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _runtime_contract() -> dict[str, Any]:
    import torch

    try:
        import vllm

        vllm_version = vllm.__version__
    except (ImportError, AttributeError):
        vllm_version = importlib.metadata.version("vllm")
    capability = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    return {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": vllm_version,
        "cuda_capability": capability,
        "capture_sizes": os.environ.get("VOICECHAT_STEP9_CAPTURE_SIZES", ""),
        "pair_full_graph": os.environ.get("VOICECHAT_NANO_PAIR_FULL_GRAPH", "0"),
    }


def _reject(reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    # Never expose a rejected tree to vLLM. A unique ephemeral root guarantees
    # the engine takes its ordinary cold compilation path without deleting the
    # baked evidence or trusting any stale/corrupt artifact.
    cold_root = COLD_CACHE_ROOT.with_name(f"{COLD_CACHE_ROOT.name}-{os.getpid()}")
    cold_root.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_CACHE_ROOT"] = str(cold_root)
    result = {
        "status": "cold_fallback",
        "reason": reason,
        "cache_root": str(cold_root),
        **(details or {}),
    }
    print(f"VOICECHAT_STEP9_CACHE {json.dumps(result, sort_keys=True)}", flush=True)
    return result

def validate_baked_vllm_cache(
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Validate the complete baked tree or redirect vLLM to a fresh cold root."""

    manifest_path = cache_root / MANIFEST_NAME
    expected_key = os.environ.get("VOICECHAT_STEP9_BAKED_CACHE_KEY", "")
    if not manifest_path.is_file():
        return _reject("manifest_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _reject("manifest_unreadable", {"error": type(exc).__name__})
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete":
        return _reject("manifest_incomplete")
    if not expected_key or manifest.get("cache_key") != expected_key:
        return _reject("cache_key_mismatch")
    runtime = _runtime_contract()
    if manifest.get("runtime_contract") != runtime:
        return _reject(
            "runtime_contract_mismatch",
            {"expected": manifest.get("runtime_contract"), "actual": runtime},
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return _reject("file_inventory_missing")
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            return _reject("unsafe_inventory_path")
        path = cache_root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(cache_root.resolve(strict=True))
        except (OSError, ValueError):
            return _reject("inventory_path_missing", {"path": relative})
        if not resolved.is_file() or resolved.stat().st_size != item.get("bytes"):
            return _reject("inventory_size_mismatch", {"path": relative})
        if _sha256(resolved) != item.get("sha256"):
            return _reject("inventory_digest_mismatch", {"path": relative})
    result = {
        "status": "validated",
        "cache_key": expected_key,
        "cache_root": str(cache_root),
        "files": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
    }
    print(f"VOICECHAT_STEP9_CACHE {json.dumps(result, sort_keys=True)}", flush=True)
    return result
