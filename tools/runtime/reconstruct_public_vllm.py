#!/usr/bin/env python3
"""Reconstruct the qualified vLLM Python package from pinned public trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOM_PATH = REPO_ROOT / "config/qualified-candidate-1.json"
INDEX_PATH = REPO_ROOT / "tools/provenance/qualified-deltas/index.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_file(root: Path, relative: str) -> Path:
    path = root / "vllm" / relative
    if not path.is_file():
        raise FileNotFoundError(f"public source file is missing: {path}")
    return path


HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def apply_unified_patch(source: bytes, patch: bytes) -> bytes:
    """Apply a one-file text diff while validating every context/deletion line.

    The retained escrow was captured from complete file pairs. One historical
    diff has an inaccurate final hunk count, which GNU patch rejects even
    though all context and result hashes are sound. Parsing to the next hunk
    header and validating the final SHA is stricter about content and tolerant
    only of that non-semantic header defect.
    """
    original = source.decode("utf-8").splitlines(keepends=True)
    diff = patch.decode("utf-8").splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    position = 0
    while position < len(diff) and not diff[position].startswith("@@ "):
        position += 1
    while position < len(diff):
        match = HUNK_HEADER.match(diff[position])
        if not match:
            raise ValueError(f"invalid unified-diff hunk header: {diff[position].rstrip()}")
        old_start = int(match.group(1))
        hunk_cursor = max(old_start - 1, 0)
        if hunk_cursor < cursor:
            raise ValueError("overlapping unified-diff hunks")
        output.extend(original[cursor:hunk_cursor])
        cursor = hunk_cursor
        position += 1
        while position < len(diff) and not diff[position].startswith("@@ "):
            line = diff[position]
            position += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            marker, value = line[:1], line[1:]
            if marker == "+":
                output.append(value)
            elif marker in {" ", "-"}:
                if cursor >= len(original) or original[cursor] != value:
                    actual = original[cursor] if cursor < len(original) else "<EOF>"
                    # difflib concatenated the first added line to a deleted
                    # source line that lacked its final newline in one escrow
                    # patch. Split only that mechanically recognizable form;
                    # the final per-file SHA remains the authority.
                    if (
                        marker == "-"
                        and actual != "<EOF>"
                        and not actual.endswith("\n")
                        and value.startswith(actual + "+")
                    ):
                        cursor += 1
                        output.append(value[len(actual) + 1 :])
                        continue
                    raise ValueError(
                        "unified-diff context does not match public source at "
                        f"line {cursor + 1}: expected {value!r}, found {actual!r}"
                    )
                if marker == " ":
                    output.append(value)
                cursor += 1
            else:
                raise ValueError(f"invalid unified-diff line: {line.rstrip()}")
    output.extend(original[cursor:])
    return "".join(output).encode("utf-8")


def reconstruct(
    *,
    native_root: Path,
    voicechat_root: Path,
    output: Path,
    bom_path: Path = BOM_PATH,
    index_path: Path = INDEX_PATH,
) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    bom = json.loads(bom_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_index_sha = bom["qualified_runtime_reference"]["delta_escrow"][
        "index_sha256"
    ]
    if sha256_file(index_path) != expected_index_sha:
        raise ValueError("qualified delta index hash differs from the BOM")
    sources = bom["public_sources"]
    expected_revisions = {
        native_root: sources["native_vllm"]["revision"],
        voicechat_root: sources["voicechat_vllm"]["revision"],
    }
    for root, expected in expected_revisions.items():
        actual = git_revision(root)
        if actual != expected:
            raise ValueError(f"source revision mismatch for {root}: {actual} != {expected}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix="voicechat-vllm-reconstruct-", dir=output.parent)
    )
    temporary = temporary_parent / "vllm"
    try:
        temporary.mkdir()
        for entry in index["files"]:
            relative = entry["path"]
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            reference = entry["reference"]
            if reference == "none":
                destination.touch()
            else:
                source_root = (
                    voicechat_root if reference in {"voicechat", "both"} else native_root
                )
                source = _source_file(source_root, relative)
                expected_source_sha = entry.get("reference_sha256", entry["sha256"])
                if sha256_file(source) != expected_source_sha:
                    raise ValueError(f"public source hash mismatch: {reference}:{relative}")
                if reference == "both":
                    native_source = _source_file(native_root, relative)
                    if sha256_file(native_source) != expected_source_sha:
                        raise ValueError(f"public source trees disagree: {relative}")
                shutil.copyfile(source, destination)

            patch_name = entry.get("patch")
            if patch_name:
                patch = index_path.parent / patch_name
                if sha256_file(patch) != entry["patch_sha256"]:
                    raise ValueError(f"qualified patch hash mismatch: {patch_name}")
                if patch.stat().st_size:
                    try:
                        patched = apply_unified_patch(
                            destination.read_bytes(), patch.read_bytes()
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"qualified patch did not apply: {patch_name}: {error}"
                        ) from error
                    destination.write_bytes(patched)
            if sha256_file(destination) != entry["sha256"]:
                raise ValueError(f"reconstructed file hash mismatch: {relative}")

        reconstructed = sorted(path for path in temporary.rglob("*") if path.is_file())
        if len(reconstructed) != index["qualified_python_files"]:
            raise ValueError("reconstructed file count differs from the qualified index")
        tree_digest = hashlib.sha256()
        for path in reconstructed:
            relative = path.relative_to(temporary).as_posix()
            tree_digest.update(relative.encode() + b"\0")
            tree_digest.update(str(path.stat().st_size).encode() + b"\0")
            tree_digest.update(sha256_file(path).encode() + b"\n")
        if tree_digest.hexdigest() != index["qualified_tree_sha256"]:
            raise ValueError("reconstructed tree digest differs from the qualified index")
        temporary.replace(output)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    return {
        "output": str(output),
        "files": index["qualified_python_files"],
        "tree_sha256": index["qualified_tree_sha256"],
        "native_revision": expected_revisions[native_root],
        "voicechat_revision": expected_revisions[voicechat_root],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--voicechat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconstruct(
        native_root=args.native_root.resolve(),
        voicechat_root=args.voicechat_root.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
