"""Content identities for the public runtime payload and its build recipe."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

PUBLIC_RUNTIME_SOURCE_LABEL = "ai.pipecat.voicechat.project-source-sha256"
PUBLIC_RUNTIME_PAYLOAD_LABEL = "ai.pipecat.voicechat.project-payload-sha256"
PUBLIC_RUNTIME_RECIPE_LABEL = "ai.pipecat.voicechat.build-recipe-sha256"

PUBLIC_RUNTIME_PAYLOAD_FILES = (
    "README.md",
    "config/qualified-candidate-1.json",
    "container/constraints-pocket.txt",
    "pyproject.toml",
    "tools/runtime/reconstruct_public_vllm.py",
)
PUBLIC_RUNTIME_PAYLOAD_TREES = (
    "src",
    "tools/provenance/qualified-deltas",
)
PUBLIC_RUNTIME_RECIPE_FILES = (
    ".dockerignore",
    "container/Dockerfile.public-runtime",
    "container/Dockerfile.public-vllm",
    "container/build-public-runtime.sh",
)


def _included_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
        and path.suffix != ".pyc"
    )


def _resolve_paths(
    repo_root: Path,
    *,
    files: tuple[str, ...],
    trees: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    paths = [repo_root / relative for relative in files]
    for relative in trees:
        tree = repo_root / relative
        if not tree.is_dir() or tree.is_symlink():
            raise RuntimeError(f"public runtime source tree is invalid: {relative}")
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(
                    f"public runtime source tree contains a symlink: {path.relative_to(repo_root)}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(
                    "public runtime source tree contains a special entry: "
                    f"{path.relative_to(repo_root)}"
                )
            if _included_file(path):
                paths.append(path)
    missing = [str(path.relative_to(repo_root)) for path in paths if not _included_file(path)]
    if missing:
        raise RuntimeError(f"public runtime source inputs are invalid: {missing}")
    return tuple(sorted(set(paths), key=lambda path: path.relative_to(repo_root).as_posix()))


def public_runtime_payload_paths(repo_root: Path) -> tuple[Path, ...]:
    """Resolve the exact files copied into `/opt/project` by the runtime build."""

    return _resolve_paths(
        repo_root,
        files=PUBLIC_RUNTIME_PAYLOAD_FILES,
        trees=PUBLIC_RUNTIME_PAYLOAD_TREES,
    )


def public_runtime_recipe_paths(repo_root: Path) -> tuple[Path, ...]:
    """Resolve the Docker/build recipe files retained for in-image audit."""

    return _resolve_paths(repo_root, files=PUBLIC_RUNTIME_RECIPE_FILES)


def public_runtime_source_paths(repo_root: Path) -> tuple[Path, ...]:
    """Resolve the full payload-plus-recipe inventory."""

    return tuple(
        sorted(
            {*public_runtime_payload_paths(repo_root), *public_runtime_recipe_paths(repo_root)},
            key=lambda path: path.relative_to(repo_root).as_posix(),
        )
    )


def _paths_sha256(repo_root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix().encode()
        contents = path.read_bytes()
        mode = stat.S_IMODE(path.stat().st_mode)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def public_runtime_payload_sha256(repo_root: Path) -> str:
    return _paths_sha256(repo_root, public_runtime_payload_paths(repo_root))


def public_runtime_recipe_sha256(repo_root: Path) -> str:
    return _paths_sha256(repo_root, public_runtime_recipe_paths(repo_root))


def public_runtime_recipe_snapshot(repo_root: Path) -> bytes:
    """Return the canonical recipe manifest retained inside the image."""

    records = []
    for path in public_runtime_recipe_paths(repo_root):
        records.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "mode": stat.S_IMODE(path.stat().st_mode),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()


def public_runtime_source_sha256(repo_root: Path) -> str:
    """Bind payload and recipe identities into one checkout/image identity."""

    digest = hashlib.sha256()
    for name, value in (
        ("payload", public_runtime_payload_sha256(repo_root)),
        ("recipe", public_runtime_recipe_sha256(repo_root)),
    ):
        digest.update(name.encode())
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()
