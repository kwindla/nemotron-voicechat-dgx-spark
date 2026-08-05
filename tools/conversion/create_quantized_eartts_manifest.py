#!/usr/bin/env python3
"""Compose an exact-public manifest with a reproducible W8A32 EarTTS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def composite_sha256(files: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["name"]):
        digest.update(item["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    base_manifest: dict,
    report: dict,
    eartts_root: Path,
    *,
    report_path: Path,
    reproducibility_runs: int,
    byte_identical: bool,
) -> dict:
    if report.get("status") != "complete":
        raise ValueError("W8A32 report is incomplete")
    config_path = eartts_root / "config.json"
    index_path = eartts_root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("W8A32 EarTTS artifact is incomplete")

    source = base_manifest.get("source") or {}
    source_model = source.get("model") or {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("voicechat_checkpoint_repository") != source.get("repository")
        or config.get("voicechat_checkpoint_revision") != source.get("revision")
        or config.get("voicechat_checkpoint_sha256") != source_model.get("sha256")
        or config.get("voicechat_derived_from_ea") is not False
    ):
        raise ValueError("W8A32 EarTTS provenance does not match base manifest")
    if config.get("quantization_config") != {"quant_method": "ea_w8a32"}:
        raise ValueError("Unexpected W8A32 quantization config")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set((index.get("weight_map") or {}).values()))
    if not names:
        raise ValueError("W8A32 EarTTS index has no shards")
    files = []
    for name in names:
        if Path(name).name != name:
            raise ValueError(f"Unsafe shard name: {name}")
        path = eartts_root / name
        if not path.is_file():
            raise ValueError(f"Missing W8A32 shard: {path}")
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest = copy.deepcopy(base_manifest)
    manifest["kind"] = "exact_public_vllm_quantized_eartts"
    manifest["derived_from_ea"] = False
    manifest["components"]["eartts"] = {
        "path": f"/derived/{eartts_root.name}",
        "model": {
            "bytes": sum(item["bytes"] for item in files),
            "sha256": composite_sha256(files),
            "files": files,
            "inventory": {
                "tensor_count": len(index.get("weight_map") or {}),
                "payload_bytes": (index.get("metadata") or {}).get("total_size"),
            },
        },
        "quantization": {
            "method": report.get("method"),
            "selected_tensor_count": report.get("selected_tensor_count"),
            "selected_source_bytes": report.get("selected_source_bytes"),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
        "reproducibility": {
            "runs": reproducibility_runs,
            "byte_identical": byte_identical,
        },
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--quant-report", type=Path, required=True)
    parser.add_argument("--eartts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproducibility-runs", type=int, default=2)
    parser.add_argument("--byte-identical", action="store_true")
    args = parser.parse_args()

    base_path = args.base_manifest.expanduser().resolve()
    report_path = args.quant_report.expanduser().resolve()
    eartts_root = args.eartts_root.expanduser().resolve()
    manifest = build_manifest(
        json.loads(base_path.read_text(encoding="utf-8")),
        json.loads(report_path.read_text(encoding="utf-8")),
        eartts_root,
        report_path=report_path,
        reproducibility_runs=args.reproducibility_runs,
        byte_identical=args.byte_identical,
    )
    output = args.output.expanduser().resolve()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    component = manifest["components"]["eartts"]
    print(
        json.dumps(
            {
                "output": str(output),
                "eartts_bytes": component["model"]["bytes"],
                "eartts_sha256": component["model"]["sha256"],
                "shards": len(component["model"]["files"]),
                "byte_identical": args.byte_identical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
