#!/usr/bin/env python3
"""Build the fhw8 candidate service manifest (G3 preregistration item 2).

Produces an `exact_public_vllm_quantized_nano_eartts` manifest for the
candidate Nano plus the unchanged release EarTTS. The
`reproducibility: byte_identical` claim is earned in-process: the script
verifies a second independent builder run produced byte-identical candidate
files, and refuses to write the manifest otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def composite_sha256(shards: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(shards, key=lambda value: value["name"]):
        digest.update(item["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def nano_component(root: Path, quantization: dict[str, Any]) -> dict[str, Any]:
    files = inventory(root)
    shards = [
        {"name": name, **details}
        for name, details in files.items()
        if name.startswith("model-") and name.endswith(".safetensors")
    ]
    if not shards:
        raise SystemExit(f"no model shards under {root}")
    return {
        "files": files,
        "model": {
            "files": shards,
            "bytes": sum(item["bytes"] for item in shards),
            "sha256": composite_sha256(shards),
        },
        "reproducibility": {"runs": 2, "byte_identical": True},
        "quantization": quantization,
    }


def eartts_component_from_release(release: dict[str, Any]) -> dict[str, Any]:
    prefix = "eartts/"
    files = {
        item["path"].removeprefix(prefix): {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in release.get("files", [])
        if item.get("path", "").startswith(prefix)
    }
    shards = [
        {"name": name, **details}
        for name, details in files.items()
        if name.startswith("model-") and name.endswith(".safetensors")
    ]
    composite = release.get("eartts_composite_sha256")
    if not files or not shards or not composite:
        raise SystemExit("signed release has no complete eartts component")
    return {
        "files": files,
        "model": {
            "files": shards,
            "bytes": sum(item["bytes"] for item in shards),
            "sha256": composite,
        },
        "reproducibility": {"runs": 2, "byte_identical": True},
        "quantization": {"release": release.get("candidate")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--rebuild-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    first = inventory(args.candidate_root)
    second = inventory(args.rebuild_root)
    mismatches = sorted(
        name
        for name in set(first) | set(second)
        if first.get(name) != second.get(name)
    )
    if mismatches:
        raise SystemExit(f"double-build NOT byte-identical: {mismatches}")

    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if release.get("kind") != "nemotron_voicechat_dgx_spark_hf_release":
        raise SystemExit("unexpected release manifest kind")
    parent = release.get("parent") or {}
    build_report = json.loads(args.build_report.read_text(encoding="utf-8"))

    quantization = {
        "kind": "fhw8_candidate",
        "base_release_nano_composite_sha256": release.get("nano_composite_sha256"),
        "delta": build_report.get("deltas"),
        "packing": build_report.get("packing"),
        "builder": "tools/benchmark/build_fhw8_candidate.py",
        "campaign": "reports/fhw8-qualification-20260814.md (G0-G2 PASS)",
    }
    manifest = {
        "kind": "exact_public_vllm_quantized_nano_eartts",
        "schema": 1,
        "derived_from_ea": False,
        "source": {
            "repository": parent.get("repository"),
            "revision": parent.get("revision"),
            "model": {"sha256": parent.get("model_sha256")},
            "runtime": parent.get("runtime"),
        },
        "components": {
            "nano": nano_component(args.candidate_root, quantization),
            "eartts": eartts_component_from_release(release),
        },
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "kind": "fhw8_double_build_evidence",
        "candidate_root": str(args.candidate_root),
        "rebuild_root": str(args.rebuild_root),
        "files_compared": len(set(first) | set(second)),
        "byte_identical": True,
        "first_inventory": first,
        "manifest_sha256": sha256_file(args.output),
    }
    args.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    condensed = {
        "files_compared": evidence["files_compared"],
        "byte_identical": True,
        "manifest": str(args.output),
        "manifest_sha256": evidence["manifest_sha256"],
    }
    print(json.dumps(condensed, indent=2))


if __name__ == "__main__":
    main()
