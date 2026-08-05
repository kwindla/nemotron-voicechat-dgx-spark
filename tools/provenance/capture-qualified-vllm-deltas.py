#!/usr/bin/env python3
"""Escrow every qualified vLLM Python delta against pinned public trees."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tree_sha256(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_bytes(payload).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def unified_patch(relative: str, source: bytes | None, qualified: bytes) -> bytes:
    before = [] if source is None else source.decode("utf-8").splitlines(keepends=True)
    after = qualified.decode("utf-8").splitlines(keepends=True)
    from_name = "/dev/null" if source is None else f"a/vllm/{relative}"
    lines = difflib.unified_diff(
        before,
        after,
        fromfile=from_name,
        tofile=f"b/vllm/{relative}",
        lineterm="\n",
    )
    return "".join(lines).encode("utf-8")


def capture(
    qualified_root: Path,
    voicechat_root: Path,
    native_root: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    entries: list[dict[str, Any]] = []
    patches: dict[str, bytes] = {}
    tree_files: list[tuple[str, bytes]] = []
    counts = {
        "equal_both": 0,
        "equal_voicechat": 0,
        "equal_native": 0,
        "delta": 0,
        "added": 0,
    }
    for qualified_path in sorted(qualified_root.rglob("*.py")):
        relative = qualified_path.relative_to(qualified_root).as_posix()
        qualified = qualified_path.read_bytes()
        tree_files.append((relative, qualified))
        references: list[tuple[str, bytes]] = []
        for label, root in (("voicechat", voicechat_root), ("native", native_root)):
            path = root / relative
            if path.is_file():
                references.append((label, path.read_bytes()))

        equal = [label for label, payload in references if payload == qualified]
        entry: dict[str, Any] = {
            "path": relative,
            "bytes": len(qualified),
            "sha256": sha256_bytes(qualified),
        }
        if len(equal) == 2:
            entry.update(classification="equal_both", reference="both")
            counts["equal_both"] += 1
        elif equal:
            entry.update(classification=f"equal_{equal[0]}", reference=equal[0])
            counts[f"equal_{equal[0]}"] += 1
        else:
            candidates = [
                (len(unified_patch(relative, payload, qualified)), label, payload)
                for label, payload in references
            ]
            if candidates:
                _, label, source = min(candidates, key=lambda item: (item[0], item[1]))
                patch = unified_patch(relative, source, qualified)
                classification = "delta"
                entry["reference_sha256"] = sha256_bytes(source)
            else:
                label = "none"
                patch = unified_patch(relative, None, qualified)
                classification = "added"
            patch_name = f"patches/{relative}.patch"
            patches[patch_name] = patch
            entry.update(
                classification=classification,
                reference=label,
                patch=patch_name,
                patch_sha256=sha256_bytes(patch),
            )
            counts[classification] += 1
        entries.append(entry)
    return (
        {
            "schema_version": 1,
            "kind": "qualified_vllm_python_delta_escrow",
            "qualified_python_files": len(entries),
            "qualified_tree_sha256": tree_sha256(tree_files),
            "counts": counts,
            "files": entries,
        },
        patches,
    )


def write_capture(output: Path, index: dict[str, Any], patches: dict[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for name, payload in sorted(patches.items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-root", type=Path, required=True)
    parser.add_argument("--voicechat-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index, patches = capture(
        args.qualified_root.resolve(),
        args.voicechat_root.resolve(),
        args.native_root.resolve(),
    )
    write_capture(args.output.resolve(), index, patches)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "qualified_python_files": index["qualified_python_files"],
                "counts": index["counts"],
                "patches": len(patches),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
