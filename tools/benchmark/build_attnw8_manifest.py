#!/usr/bin/env python3
"""Build the attnW8 live-service manifest after proving double-build identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_fhw8_manifest import (
    eartts_component_from_release,
    inventory,
    nano_component,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--rebuild-root", type=Path, required=True)
    parser.add_argument("--fhw8-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    first = inventory(args.candidate_root)
    second = inventory(args.rebuild_root)
    mismatches = sorted(
        name for name in set(first) | set(second) if first.get(name) != second.get(name)
    )
    if mismatches:
        raise SystemExit(f"double-build NOT byte-identical: {mismatches}")

    fhw8 = json.loads(args.fhw8_manifest.read_text(encoding="utf-8"))
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    report = json.loads(args.build_report.read_text(encoding="utf-8"))
    if fhw8.get("kind") != "exact_public_vllm_quantized_nano_eartts":
        raise SystemExit("unexpected fhw8 manifest kind")
    if release.get("kind") != "nemotron_voicechat_dgx_spark_hf_release":
        raise SystemExit("unexpected release manifest kind")

    parent = release.get("parent") or {}
    fhw8_nano = fhw8["components"]["nano"]
    quantization = {
        "kind": "attnw8_candidate",
        "base_fhw8_nano_composite_sha256": fhw8_nano["model"]["sha256"],
        "delta": report.get("deltas"),
        "packing": report.get("packing"),
        "builder": "tools/benchmark/build_attnw8_candidate.py",
        "campaign": "reports/attnw8-qualification/preregistration.md",
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
        "kind": "attnw8_double_build_evidence",
        "candidate_root": str(args.candidate_root),
        "rebuild_root": str(args.rebuild_root),
        "files_compared": len(set(first) | set(second)),
        "byte_identical": True,
        "first_inventory": first,
        "fhw8_base_composite_sha256": fhw8_nano["model"]["sha256"],
        "manifest_sha256": sha256_file(args.output),
    }
    args.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files_compared": evidence["files_compared"],
                "byte_identical": True,
                "manifest": str(args.output),
                "manifest_sha256": evidence["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
