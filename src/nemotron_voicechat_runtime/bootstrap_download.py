"""Download and verify every immutable model input used by Voicechat."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import hf_hub_download, snapshot_download

from .artifacts import (
    DEFAULT_CONFIG,
    Layout,
    atomic_json,
    default_layout,
    is_lfs_pointer,
    load_config,
    sha256_file,
)


@contextmanager
def _artifact_stage(number: int, total: int, description: str) -> Iterator[None]:
    """Make long download and checksum stages visible to non-interactive users."""

    started = time.monotonic()
    print(f"[artifacts {number}/{total}] {description}...", flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        print(f"[artifacts {number}/{total}] failed after {elapsed:.1f}s", flush=True)
        raise
    elapsed = time.monotonic() - started
    print(f"[artifacts {number}/{total}] complete ({elapsed:.1f}s)", flush=True)


def _download_error(repository: str, exc: Exception) -> RuntimeError:
    terms = f"https://huggingface.co/{repository}"
    return RuntimeError(
        f"Unable to download {repository}. If access is gated, accept its terms at {terms}, "
        "authenticate with `hf auth login` (or set HF_TOKEN for this invocation), and rerun "
        f"`./voicechat bootstrap`. Original error: {exc}"
    )


def _download_snapshot(
    *,
    repository: str,
    revision: str,
    destination: Path,
    cache_dir: Path,
    allow_patterns: list[str],
    offline: bool,
) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        snapshot_download(
            repository,
            revision=revision,
            local_dir=temporary,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            local_files_only=offline,
            token=os.environ.get("HF_TOKEN"),
        )
        cache_metadata = temporary / ".cache"
        if cache_metadata.exists():
            shutil.rmtree(cache_metadata)
        os.replace(temporary, destination)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise _download_error(repository, exc) from exc


def _verify_files(root: Path, expected: dict[str, dict[str, Any] | str]) -> None:
    expected_names = set(expected)
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.relative_to(root).as_posix().startswith(".cache/")
    }
    if actual_names != expected_names:
        raise RuntimeError(f"artifact inventory mismatch in {root}")
    for name, details in expected.items():
        path = root / name
        if is_lfs_pointer(path):
            raise RuntimeError(f"partial Hugging Face LFS pointer found: {path}")
        expected_sha = details if isinstance(details, str) else details["sha256"]
        expected_bytes = None if isinstance(details, str) else details.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise RuntimeError(f"artifact byte count mismatch: {path}")
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"artifact SHA-256 mismatch: {path}")


def verify_release(root: Path, expected_release_sha: str) -> dict[str, Any]:
    manifest_path = root / "manifests/release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_release_sha = manifest.pop("release_sha256", None)
    import hashlib

    canonical = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["release_sha256"] = actual_release_sha
    if actual_release_sha != expected_release_sha or canonical != expected_release_sha:
        raise RuntimeError("signed Voicechat release identity mismatch")
    expected = {
        item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for item in manifest["files"]
    }
    expected["manifests/release.json"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    _verify_files(root, expected)
    return manifest


def _release_patterns(
    repository: str, revision: str, cache_dir: Path, offline: bool
) -> tuple[list[str], dict[str, Any]]:
    try:
        path = hf_hub_download(
            repository,
            "manifests/release.json",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=offline,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise _download_error(repository, exc) from exc
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item["path"] for item in manifest["files"]] + ["manifests/release.json"], manifest


def _download_pocket(config: dict[str, Any], layout: Layout, offline: bool) -> None:
    pocket = config["artifacts"]["pocket"]
    for revision, patterns in (
        (
            pocket["model_revision"],
            [
                "languages/english_2026-04/model.safetensors",
                "languages/english_2026-04/tokenizer.model",
            ],
        ),
        (
            pocket["voice_revision"],
            ["languages/english_2026-04/embeddings/alba.safetensors"],
        ),
    ):
        try:
            snapshot_download(
                pocket["repository"],
                revision=revision,
                cache_dir=layout.hf_home / "hub",
                allow_patterns=patterns,
                local_files_only=offline,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            raise _download_error(pocket["repository"], exc) from exc


def download_all(config: dict[str, Any], layout: Layout, *, offline: bool) -> dict[str, Any]:
    layout.cache.mkdir(parents=True, exist_ok=True)
    layout.hf_home.mkdir(parents=True, exist_ok=True)
    release_cfg = config["artifacts"]["release"]
    with _artifact_stage(1, 4, "download and verify converted Voicechat release"):
        patterns, remote_manifest = _release_patterns(
            release_cfg["repository"],
            release_cfg["revision"],
            layout.hf_home / "hub",
            offline,
        )
        if remote_manifest.get("release_sha256") != release_cfg["release_sha256"]:
            raise RuntimeError("remote release manifest does not match the checked-in identity")
        _download_snapshot(
            repository=release_cfg["repository"],
            revision=release_cfg["revision"],
            destination=layout.release,
            cache_dir=layout.hf_home / "hub",
            allow_patterns=patterns,
            offline=offline,
        )
        release = verify_release(layout.release, release_cfg["release_sha256"])

    bom = json.loads(
        (Path(__file__).resolve().parents[2] / "config/qualified-candidate-1.json").read_text()
    )
    parent_cfg = config["artifacts"]["parent"]
    parent_files = bom["public_sources"]["checkpoint"]["files"]
    parent_expected = {
        name: {
            "bytes": bom["public_sources"]["checkpoint"]["file_sizes"][name],
            "sha256": digest,
        }
        for name, digest in parent_files.items()
    }
    with _artifact_stage(2, 4, "download and verify NVIDIA parent checkpoint"):
        _download_snapshot(
            repository=parent_cfg["repository"],
            revision=parent_cfg["revision"],
            destination=layout.parent,
            cache_dir=layout.hf_home / "hub",
            allow_patterns=list(parent_expected),
            offline=offline,
        )
        _verify_files(layout.parent, parent_expected)

    skeleton_cfg = config["artifacts"]["nano_skeleton"]
    skeleton_expected = bom["public_sources"]["nano_skeleton"]["files"]
    with _artifact_stage(3, 4, "download and verify Nano tokenizer skeleton"):
        _download_snapshot(
            repository=skeleton_cfg["repository"],
            revision=skeleton_cfg["revision"],
            destination=layout.nano_skeleton,
            cache_dir=layout.hf_home / "hub",
            allow_patterns=list(skeleton_expected),
            offline=offline,
        )
        _verify_files(layout.nano_skeleton, skeleton_expected)
    with _artifact_stage(4, 4, "download and verify Pocket TTS assets"):
        _download_pocket(config, layout, offline)
    return {
        "release_sha256": release["release_sha256"],
        "release_revision": release_cfg["revision"],
        "parent_revision": parent_cfg["revision"],
        "nano_skeleton_revision": skeleton_cfg["revision"],
    }


def extract_replay_corpus(release_root: Path, destination: Path) -> None:
    if destination.exists():
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        with tarfile.open(release_root / "calibration/replay.tar") as archive:
            archive.extractall(temporary, filter="data")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root")
    parser.add_argument("--trace-root")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    layout = default_layout(args.cache_root, args.trace_root)
    result = download_all(config, layout, offline=args.offline)
    atomic_json(layout.cache / "artifact-state.json", result)
    print("[artifacts] all immutable inputs are ready", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
