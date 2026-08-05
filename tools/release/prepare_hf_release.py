#!/usr/bin/env python3
"""Build a content-addressed Hugging Face release directory from qualified inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark"
MINIMUM_REMAINING_BYTES = 100 * 1024**3


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix() not in excluded
        and not path.relative_to(root).as_posix().startswith(".cache/huggingface/")
    ]


def verify_artifact(root: Path, expected: dict[str, Any], label: str) -> None:
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected["files"]:
        raise ValueError(f"{label} artifact differs from the qualified BOM")


def _copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    destination.chmod(0o644)
    return mode


def compact_nano_report(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "portable_nano_conversion_report",
        "status": source.get("status"),
        "method": source.get("method"),
        "source_provenance": source.get("source_provenance"),
        "policy": source.get("policy"),
        "quantization_config": source.get("quantization_config"),
        "selected_tensor_count": source.get("selected_tensor_count"),
        "selected_source_bytes": source.get("selected_source_bytes"),
        "quantized_output_heads": source.get("quantized_output_heads"),
        "function_head_full_precision": source.get("function_head_full_precision"),
        "embeddings_full_precision": source.get("embeddings_full_precision"),
        "attention_projections_full_precision": True,
        "shape_exception": source.get("shape_exception"),
        "calibration": {
            key: (source.get("calibration") or {}).get(key)
            for key in (
                "calls",
                "conversations",
                "effective_rows_lower_bound",
                "manifest_sha256",
                "selection_manifest_sha256",
                "input",
            )
        },
        "evaluation": {
            key: (source.get("evaluation") or {}).get(key)
            for key in (
                "calls",
                "conversations",
                "manifest_sha256",
                "selection_manifest_sha256",
                "disjoint_from_calibration",
            )
        },
        "runtime": {
            key: (source.get("runtime") or {}).get(key)
            for key in ("driver_version", "kernel_version", "packages")
        },
        "reproducibility": source.get("reproducibility"),
        "layer_metrics": [
            metric for run in source.get("runs") or [] for metric in run.get("layer_metrics") or []
        ][:105],
    }


def compact_eartts_report(
    source: dict[str, Any], reproducibility: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "portable_eartts_conversion_report",
        "status": source.get("status"),
        "method": source.get("method"),
        "selected_tensor_count": source.get("selected_tensor_count"),
        "selected_source_bytes": source.get("selected_source_bytes"),
        "output_tensor_bytes": source.get("output_tensor_bytes"),
        "reproducibility": reproducibility,
        "layer_metrics": source.get("layer_metrics"),
    }


def _copy_artifact(source: Path, destination: Path) -> dict[str, str]:
    modes = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            modes[relative.as_posix()] = _copy(path, destination / relative)
    return modes


def _verified_corpus_files(
    roots_by_release_path: dict[str, Path],
    corpus_manifest: dict[str, Any],
) -> list[tuple[Path, str]]:
    files = []
    for root in corpus_manifest["roots"]:
        release_path = root["release_path"]
        source = roots_by_release_path.get(release_path)
        if source is None:
            raise ValueError(f"missing corpus root: {release_path}")
        actual = {
            path.relative_to(source).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source.rglob("*")
            if path.is_file()
        }
        expected = {
            item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
            for item in root["files"]
        }
        if actual != expected:
            raise ValueError(f"corpus root differs from manifest: {release_path}")
        for relative in sorted(actual):
            files.append((source / relative, f"{release_path}/{relative}"))
    return files


def build_deterministic_tar(files: list[tuple[Path, str]], destination: Path) -> None:
    """Archive verified replay files without host-dependent metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for source, archive_name in sorted(files, key=lambda item: item[1]):
            info = archive.gettarinfo(str(source), arcname=archive_name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with source.open("rb") as stream:
                archive.addfile(info, stream)


def stage_release(
    *,
    nano_root: Path,
    eartts_root: Path,
    nano_report: Path,
    eartts_report: Path,
    corpus_roots: dict[str, Path],
    output: Path,
    bom_path: Path = REPO_ROOT / "config/qualified-candidate-1.json",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    projected_bytes = sum(
        path.stat().st_size
        for root in (nano_root, eartts_root, *corpus_roots.values())
        for path in root.rglob("*")
        if path.is_file()
    )
    if shutil.disk_usage(output.parent).free - projected_bytes < MINIMUM_REMAINING_BYTES:
        raise RuntimeError(
            "refusing release staging when projected output would leave less than 100 GiB free"
        )
    bom = json.loads(bom_path.read_text(encoding="utf-8"))
    verify_artifact(nano_root, bom["qualified_artifacts"]["nano"], "Nano")
    verify_artifact(eartts_root, bom["qualified_artifacts"]["eartts"], "EarTTS")
    corpus_manifest_path = REPO_ROOT / "config/calibration-corpus-production-candidate-1.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest["release_sha256"]
        != bom["qualified_artifacts"]["calibration"]["portable_release_sha256"]
    ):
        raise ValueError("corpus release identity differs from the qualified BOM")
    license_path = REPO_ROOT / "licenses/OpenMDW-1.1.txt"
    if sha256_file(license_path) != bom["public_sources"]["checkpoint"]["license_sha256"]:
        raise ValueError("OpenMDW license copy differs from the public checkpoint")

    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    copy_modes: dict[str, str] = {}
    try:
        temporary.mkdir()
        copy_modes.update(
            {
                f"nano/{name}": mode
                for name, mode in _copy_artifact(nano_root, temporary / "nano").items()
            }
        )
        copy_modes.update(
            {
                f"eartts/{name}": mode
                for name, mode in _copy_artifact(eartts_root, temporary / "eartts").items()
            }
        )
        build_deterministic_tar(
            _verified_corpus_files(corpus_roots, corpus_manifest),
            temporary / "calibration/replay.tar",
        )
        for source, relative in (
            (corpus_manifest_path, "calibration/manifest.json"),
            (
                REPO_ROOT / "config/nano-gptq-corpus-selection.json",
                "calibration/selection.json",
            ),
            (REPO_ROOT / "release/MODEL_CARD.md", "README.md"),
            (license_path, "LICENSE"),
            (REPO_ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        ):
            copy_modes[relative] = _copy(source, temporary / relative)
        for fixture in sorted((REPO_ROOT / "tools/conversion/fixtures").glob("*.json")):
            relative = f"calibration/fixtures/{fixture.name}"
            copy_modes[relative] = _copy(fixture, temporary / relative)

        reports = temporary / "reports"
        reports.mkdir()
        nano_portable = compact_nano_report(json.loads(nano_report.read_text(encoding="utf-8")))
        eartts_portable = compact_eartts_report(
            json.loads(eartts_report.read_text(encoding="utf-8")),
            bom["qualified_artifacts"]["eartts"]["reproducibility"],
        )
        (reports / "nano-conversion.json").write_text(
            json.dumps(nano_portable, indent=2, sort_keys=True) + "\n"
        )
        (reports / "eartts-conversion.json").write_text(
            json.dumps(eartts_portable, indent=2, sort_keys=True) + "\n"
        )

        files = inventory(temporary)
        release = {
            "schema": 1,
            "kind": "nemotron_voicechat_dgx_spark_hf_release",
            "repository": REPO_ID,
            "community_release": True,
            "nvidia_release": False,
            "parent": {
                "repository": bom["public_sources"]["checkpoint"]["repository"],
                "revision": bom["public_sources"]["checkpoint"]["revision"],
                "model_sha256": bom["public_sources"]["checkpoint"]["sha256"],
                "license": "OpenMDW-1.1",
            },
            "candidate": bom["candidate"],
            "runtime_contract": bom["runtime_contract"],
            "derived_from_ea": False,
            "corpus_release_sha256": corpus_manifest["release_sha256"],
            "nano_composite_sha256": bom["qualified_artifacts"]["nano"]["composite_sha256"],
            "eartts_composite_sha256": bom["qualified_artifacts"]["eartts"]["composite_sha256"],
            "files": files,
            "files_sha256": canonical_sha256(files),
        }
        release["release_sha256"] = canonical_sha256(release)
        manifest_path = temporary / "manifests/release.json"
        manifest_path.parent.mkdir()
        manifest_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "release_sha256": release["release_sha256"],
        "file_count": len(files),
        "payload_bytes": sum(item["bytes"] for item in files),
        "copy_modes": {
            mode: sum(value == mode for value in copy_modes.values())
            for mode in sorted(set(copy_modes.values()))
        },
    }


def verify_staged_release(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifests/release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != REPO_ID or manifest.get("derived_from_ea") is not False:
        raise ValueError("release repository or lineage is invalid")
    expected_release_sha = manifest.pop("release_sha256", None)
    if canonical_sha256(manifest) != expected_release_sha:
        raise ValueError("release manifest self-hash mismatch")
    files = inventory(root, exclude={"manifests/release.json"})
    if files != manifest["files"] or canonical_sha256(files) != manifest["files_sha256"]:
        raise ValueError("staged release payload differs from release manifest")
    manifest["release_sha256"] = expected_release_sha
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-root", type=Path, required=True)
    parser.add_argument("--eartts-root", type=Path, required=True)
    parser.add_argument("--nano-report", type=Path, required=True)
    parser.add_argument("--eartts-report", type=Path, required=True)
    parser.add_argument(
        "--corpus-root",
        action="append",
        required=True,
        metavar="RELEASE_PATH=LOCAL_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {}
    for value in args.corpus_root:
        release_path, separator, local_path = value.partition("=")
        if not separator or release_path in roots:
            raise SystemExit(f"invalid or duplicate --corpus-root: {value}")
        roots[release_path] = Path(local_path).expanduser().resolve()
    result = stage_release(
        nano_root=args.nano_root.expanduser().resolve(),
        eartts_root=args.eartts_root.expanduser().resolve(),
        nano_report=args.nano_report.expanduser().resolve(),
        eartts_report=args.eartts_report.expanduser().resolve(),
        corpus_roots=roots,
        output=args.output.expanduser().resolve(),
    )
    verify_staged_release(args.output.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
