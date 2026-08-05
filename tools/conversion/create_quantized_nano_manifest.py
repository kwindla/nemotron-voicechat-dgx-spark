#!/usr/bin/env python3
"""Compose an exact-public vLLM manifest with a sharded quantized Nano."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--quant-report", type=Path, required=True)
    parser.add_argument("--nano-root", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproducibility-runs", type=int, default=2)
    parser.add_argument("--byte-identical", action="store_true")
    args = parser.parse_args()

    base_path = args.base_manifest.expanduser().resolve()
    report_path = args.quant_report.expanduser().resolve()
    nano_root = args.nano_root.expanduser().resolve()
    manifest = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index_path = nano_root / "model.safetensors.index.json"
    config_path = nano_root / "config.json"
    if report.get("status") != "complete" or not index_path.is_file() or not config_path.is_file():
        raise SystemExit("Quantized Nano artifact is incomplete")

    source = manifest.get("source") or {}
    provenance = report.get("source_provenance") or {}
    if (
        provenance.get("repository") != source.get("repository")
        or provenance.get("revision") != source.get("revision")
        or provenance.get("checkpoint_sha256") != (source.get("model") or {}).get("sha256")
        or provenance.get("derived_from_ea") is not False
    ):
        raise SystemExit("Quantized Nano provenance does not match base manifest")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("voicechat_derived_from_ea") is not False:
        raise SystemExit("Quantized Nano config is not exact-public")
    if "function_head" in (config.get("voicechat_quantized_output_heads") or []):
        raise SystemExit("Public function head must remain unquantized")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set((index.get("weight_map") or {}).values()))
    if not shard_names:
        raise SystemExit("Quantized Nano index has no model shards")
    files = []
    for name in shard_names:
        if Path(name).name != name:
            raise SystemExit(f"Unsafe shard name: {name}")
        path = nano_root / name
        if not path.is_file():
            raise SystemExit(f"Missing quantized Nano shard: {path}")
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    component = {
        "path": f"/derived/{nano_root.name}",
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
            "config": report.get("quantization_config"),
            "selected_tensor_count": report.get("selected_tensor_count"),
            "selected_source_bytes": report.get("selected_source_bytes"),
            "quantized_output_heads": report.get("quantized_output_heads"),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
        "reproducibility": {
            "runs": args.reproducibility_runs,
            "byte_identical": args.byte_identical,
        },
    }
    manifest["kind"] = "exact_public_vllm_quantized_nano"
    manifest["components"]["nano"] = component
    report_runtime = report.get("runtime") or {}
    manifest["source"]["runtime"] = {
        "image": args.runtime_image,
        "image_id": args.runtime_image_id,
        "driver_version": report_runtime.get("driver_version"),
        "kernel_version": report_runtime.get("kernel_version"),
    }
    manifest["derived_from_ea"] = False
    output = args.output.expanduser().resolve()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "nano_bytes": component["model"]["bytes"],
                "nano_sha256": component["model"]["sha256"],
                "shards": len(files),
                "byte_identical": args.byte_identical,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
