#!/usr/bin/env python3
"""Create an exact-weight EarTTS sliding-window discriminator artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

MINIMUM_FREE_BYTES = 100 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_variant(
    source_dir: Path,
    source_manifest: Path,
    output_dir: Path,
    output_manifest: Path,
    sliding_window: int,
) -> dict:
    source_dir = source_dir.resolve()
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    output_manifest = output_manifest.resolve()
    if sliding_window < 128:
        raise ValueError("EarTTS sliding-window discriminator must be at least 128 positions")
    if not source_dir.is_dir() or not source_manifest.is_file():
        raise FileNotFoundError("Source EarTTS directory and manifest are required")
    if output_dir.exists() or output_manifest.exists():
        raise FileExistsError("Output EarTTS directory and manifest must not exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("Refusing to run below the 100 GiB free-space floor")

    source_config_path = source_dir / "config.json"
    source_config = json.loads(source_config_path.read_text())
    original_window = int(source_config.get("sliding_window", 0))
    layer_types = list(source_config.get("layer_types") or [])
    sliding_layers = sum(value == "sliding_attention" for value in layer_types)
    full_layers = sum(value == "full_attention" for value in layer_types)
    if original_window <= 0 or not sliding_layers or not full_layers:
        raise ValueError("Source EarTTS config lacks the expected hybrid attention contract")
    if sliding_window >= original_window:
        raise ValueError("Discriminator window must be smaller than the shipped window")

    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    copy_modes: dict[str, str] = {}
    try:
        temporary_dir.mkdir()
        for source_file in sorted(source_dir.iterdir()):
            if not source_file.is_file():
                raise ValueError(f"Unexpected non-file in EarTTS artifact: {source_file}")
            if source_file.name != "config.json":
                destination = temporary_dir / source_file.name
                try:
                    os.link(source_file, destination)
                    copy_modes[source_file.name] = "hardlink"
                except PermissionError:
                    # The derived public artifacts may be root-owned; Linux's
                    # protected_hardlinks then rejects a user-created link even
                    # on the same filesystem. ext4 on this host has no reflink,
                    # so preserve exact bytes with a normal copy as the bounded
                    # fallback and verify the digest below.
                    shutil.copy2(source_file, destination)
                    copy_modes[source_file.name] = "copy"
        config = dict(source_config)
        config["sliding_window"] = int(sliding_window)
        config["voicechat_eartts_window_experiment"] = {
            "status": "experimental",
            "weights_changed": False,
            "source_config_sha256": sha256_file(source_config_path),
            "shipped_sliding_window": original_window,
            "candidate_sliding_window": int(sliding_window),
            "sliding_attention_layers": sliding_layers,
            "full_attention_layers": full_layers,
            "quality_gate_required": True,
        }
        config_path = temporary_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    manifest = json.loads(source_manifest.read_text())
    if manifest.get("derived_from_ea") is not False:
        raise ValueError("Source manifest is not exact-public")
    manifest.pop("production_candidate", None)
    manifest["eartts_window_experiment"] = config["voicechat_eartts_window_experiment"]
    runtime_optimization = dict(manifest.get("runtime_optimization") or {})
    runtime_optimization["production_candidate"] = None
    runtime_optimization["eartts_sliding_window_experiment"] = int(sliding_window)
    manifest["runtime_optimization"] = runtime_optimization
    component = manifest["components"]["eartts"]
    component["path"] = f"/derived/{output_dir.name}"
    component["files"] = {
        "config.json": {
            "bytes": (output_dir / "config.json").stat().st_size,
            "sha256": sha256_file(output_dir / "config.json"),
        }
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    hardlinked = [
        path.name for path in source_dir.iterdir() if path.is_file() and path.name != "config.json"
    ]
    return {
        "output_dir": str(output_dir),
        "output_manifest": str(output_manifest),
        "weights_changed": False,
        "shipped_sliding_window": original_window,
        "candidate_sliding_window": int(sliding_window),
        "sliding_attention_layers": sliding_layers,
        "full_attention_layers": full_layers,
        "config_sha256": sha256_file(output_dir / "config.json"),
        "all_non_config_files_hardlinked": all(
            (source_dir / name).stat().st_ino == (output_dir / name).stat().st_ino
            for name in hardlinked
        ),
        "all_non_config_files_byte_identical": all(
            sha256_file(source_dir / name) == sha256_file(output_dir / name) for name in hardlinked
        ),
        "copy_modes": copy_modes,
        "production_candidate_removed": "production_candidate" not in manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--sliding-window", type=int, required=True)
    args = parser.parse_args()
    report = create_variant(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
