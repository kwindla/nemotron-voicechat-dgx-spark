#!/usr/bin/env python3
"""Seal an extracted vLLM cache tree with an explicit immutable cache key."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nemotron_voicechat_runtime.step9_cache_guard import MANIFEST_NAME, SCHEMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    root = args.cache_root.resolve()
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"cache tree contains a symlink: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not files:
        raise SystemExit("refusing to seal an empty cache")
    manifest = {
        "schema": SCHEMA,
        "state": "complete",
        "cache_key": args.cache_key,
        "runtime_contract": json.loads(args.runtime_contract.read_text()),
        "provenance": json.loads(args.provenance.read_text()),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "files": len(files),
                "bytes": sum(item["bytes"] for item in files),
                "sha256": sha256(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
