from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nemotron_voicechat_runtime import step9_cache_guard as guard


def write_cache(root: Path, *, key: str, contract: dict[str, str]) -> Path:
    payload = root / "torch_compile_cache" / "graph.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"qualified graph")
    manifest = {
        "schema": guard.SCHEMA,
        "state": "complete",
        "cache_key": key,
        "runtime_contract": contract,
        "files": [
            {
                "path": payload.relative_to(root).as_posix(),
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / guard.MANIFEST_NAME).write_text(json.dumps(manifest))
    return payload


def configure(monkeypatch, tmp_path: Path, *, key: str, contract: dict[str, str]) -> None:
    monkeypatch.setenv("VOICECHAT_STEP9_BAKED_CACHE_KEY", key)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.setattr(guard, "COLD_CACHE_ROOT", tmp_path / "cold-cache")
    monkeypatch.setattr(guard, "_runtime_contract", lambda: contract)


def test_valid_cache_is_exposed(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "baked"
    root.mkdir()
    contract = {"qualified": "yes"}
    write_cache(root, key="cache-v1", contract=contract)
    configure(monkeypatch, tmp_path, key="cache-v1", contract=contract)

    result = guard.validate_baked_vllm_cache(root)

    assert result["status"] == "validated"
    assert "VLLM_CACHE_ROOT" not in guard.os.environ


def test_stale_key_fails_closed_to_fresh_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "baked"
    root.mkdir()
    contract = {"qualified": "yes"}
    write_cache(root, key="cache-v1", contract=contract)
    configure(monkeypatch, tmp_path, key="cache-v2", contract=contract)

    result = guard.validate_baked_vllm_cache(root)

    assert result["status"] == "cold_fallback"
    assert result["reason"] == "cache_key_mismatch"
    assert Path(guard.os.environ["VLLM_CACHE_ROOT"]).is_dir()
    assert Path(guard.os.environ["VLLM_CACHE_ROOT"]) != root


def test_corrupt_payload_fails_closed_to_fresh_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "baked"
    root.mkdir()
    contract = {"qualified": "yes"}
    payload = write_cache(root, key="cache-v1", contract=contract)
    payload.write_bytes(b"corrupt")
    configure(monkeypatch, tmp_path, key="cache-v1", contract=contract)

    result = guard.validate_baked_vllm_cache(root)

    assert result["status"] == "cold_fallback"
    assert result["reason"] == "inventory_size_mismatch"
    assert Path(guard.os.environ["VLLM_CACHE_ROOT"]).is_dir()
