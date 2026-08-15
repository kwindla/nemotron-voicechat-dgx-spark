#!/usr/bin/env python3
"""Seal the complete retained Step 4c evidence package with SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(data_root: Path, preregistration: Path, output: Path) -> dict[str, object]:
    root = Path.cwd().resolve()
    data_root = data_root.resolve(strict=True)
    preregistration = preregistration.resolve(strict=True)
    output = output.resolve()
    paths = [preregistration]
    paths.extend(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.resolve() != output
    )
    files = []
    for path in sorted(paths):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact is outside the repository: {resolved}") from exc
        files.append(
            {
                "path": str(relative),
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )
    return {
        "schema": "nemotron_voicechat.step4c_artifact_sha256.v1",
        "hash": "sha256",
        "self_excluded": str(output.relative_to(root)),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root, args.preregistration, args.output)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sealed {manifest['file_count']} Step 4c artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
