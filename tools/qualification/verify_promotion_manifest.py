#!/usr/bin/env python3
"""Fail closed when the promotion launch surface drifts from its checksum seal."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

LINE = re.compile(r"([0-9a-f]{64})  ([^\n]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_promotion_manifest(repo_root: Path, manifest: Path) -> list[str]:
    repo_root = repo_root.resolve()
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise RuntimeError(f"promotion manifest is missing: {manifest}")
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = LINE.fullmatch(raw)
        if match is None:
            raise RuntimeError(f"malformed promotion manifest line {line_number}")
        digest, relative = match.groups()
        if relative in entries:
            raise RuntimeError(f"duplicate promotion manifest entry: {relative}")
        entries[relative] = digest

    launch_root = repo_root / "deploy/promotion-candidate"
    production_scripts = {
        path.relative_to(repo_root).as_posix()
        for path in launch_root.glob("start-production-*.sh")
        if path.is_file()
    }
    missing = sorted(production_scripts - entries.keys())
    if missing:
        raise RuntimeError(
            "promotion manifest omits production launch scripts: " + ", ".join(missing)
        )

    verified: list[str] = []
    for relative, expected in entries.items():
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(f"promotion manifest path escapes repository: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"promotion manifest path is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"promotion checksum drift: {relative}: expected {expected}, got {actual}"
            )
        verified.append(relative)
    return verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("deploy/promotion-candidate/manifest.sha256"),
    )
    args = parser.parse_args()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = args.repo_root / manifest
    verified = verify_promotion_manifest(args.repo_root, manifest)
    print(f"promotion manifest verified: {len(verified)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
