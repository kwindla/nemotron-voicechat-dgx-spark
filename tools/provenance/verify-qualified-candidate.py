#!/usr/bin/env python3
"""Validate the immutable Production Candidate 1 comparison manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROHIBITED_BUILD_INPUTS = (
    "nemotron-local/voicechat-vllm:ea-",
    "nvcr.io/nvidia/nemotron-voicechat",
)
HOST_LOCAL_PREFIXES = ("/home/", "/tmp/", "/mnt/", "/data/", "/var/", "/srv/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))
    else:
        yield path, value


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("derived_from_ea") is not False:
        errors.append("derived_from_ea must be false")
    if payload.get("candidate") != "production-candidate-1":
        errors.append("unexpected candidate identity")

    runtime = payload.get("qualified_runtime_reference") or {}
    if "comparison-only" not in str(runtime.get("use", "")):
        errors.append("qualified opaque image must be comparison-only")

    for field_path, value in _walk(payload):
        field = field_path[-1] if field_path else ""
        if field == "sha256" or field.endswith("_sha256"):
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                errors.append(f"invalid SHA-256 at {'.'.join(field_path)}")
        if isinstance(value, str) and value.startswith(HOST_LOCAL_PREFIXES):
            errors.append(f"host-local path at {'.'.join(field_path)}")

    for section_path, section in _walk_named_dicts(payload):
        if section_path[-1:] == ("files",):
            for name, digest in section.items():
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    errors.append(f"invalid file SHA-256 at {'.'.join(section_path)}.{name}")

    serialized = json.dumps(payload, sort_keys=True)
    for prohibited in PROHIBITED_BUILD_INPUTS:
        if prohibited in serialized:
            errors.append(f"prohibited public build input: {prohibited}")

    sources = payload.get("public_sources") or {}
    if not (sources.get("checkpoint") or {}).get("files"):
        errors.append("checkpoint file inventory is empty")
    build = (sources.get("native_vllm") or {}).get("build") or {}
    for key in ("cuda_version", "flashinfer_version", "python_version", "torch_cuda_arch_list"):
        if not build.get(key):
            errors.append(f"native vLLM build parameter is missing: {key}")
    if not payload.get("runtime_contract"):
        errors.append("runtime contract is missing")
    calibration = (payload.get("qualified_artifacts") or {}).get("calibration") or {}
    if calibration.get("redistributable") not in {"unknown", "yes", "no"}:
        errors.append("calibration redistributable status is invalid")

    for component in ("nano", "eartts"):
        artifact = (payload.get("qualified_artifacts") or {}).get(component) or {}
        reproduction = artifact.get("reproducibility") or {}
        if reproduction != {"runs": 2, "byte_identical": True}:
            errors.append(f"{component} lacks dual byte-identical conversion")
        if not artifact.get("files"):
            errors.append(f"{component} file inventory is empty")
    return errors


def _walk_named_dicts(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk_named_dicts(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_named_dicts(child, (*path, str(index)))


def verify_tree(
    root: Path,
    files: dict[str, str],
    label: str,
    *,
    sizes: dict[str, int] | None = None,
    strict: bool = False,
) -> list[str]:
    errors: list[str] = []
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            errors.append(f"{label}: missing {name}")
            continue
        if sizes is not None and name in sizes and path.stat().st_size != int(sizes[name]):
            errors.append(
                f"{label}: {name} bytes {path.stat().st_size} != {int(sizes[name])}"
            )
            continue
        try:
            actual = sha256_file(path)
        except OSError as error:
            errors.append(f"{label}: cannot read {name}: {error}")
            continue
        if actual != expected:
            errors.append(f"{label}: {name} SHA-256 {actual} != {expected}")
    if strict:
        actual_names = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") for part in path.relative_to(root).parts
            )
        }
        unexpected = sorted(actual_names - set(files))
        errors.extend(f"{label}: unexpected {name}" for name in unexpected)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/qualified-candidate-1.json"),
    )
    parser.add_argument("--nano-root", type=Path)
    parser.add_argument("--eartts-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--nano-skeleton-root", type=Path)
    parser.add_argument("--application-root", type=Path)
    parser.add_argument("--strict-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(payload)
    artifacts = payload["qualified_artifacts"]
    sources = payload["public_sources"]
    baseline = payload["baseline_application_source"]
    optional_checks = (
        (args.nano_root, artifacts["nano"]["files"], "nano", None, args.strict_artifacts),
        (args.eartts_root, artifacts["eartts"]["files"], "eartts", None, args.strict_artifacts),
        (
            args.checkpoint_root,
            sources["checkpoint"]["files"],
            "checkpoint",
            sources["checkpoint"]["file_sizes"],
            False,
        ),
        (
            args.nano_skeleton_root,
            sources["nano_skeleton"]["files"],
            "nano skeleton",
            None,
            args.strict_artifacts,
        ),
        (args.application_root, baseline["files"], "baseline application", None, False),
    )
    for root, files, label, sizes, strict in optional_checks:
        if root is not None:
            errors.extend(verify_tree(root, files, label, sizes=sizes, strict=strict))
    if args.application_root is not None:
        escrow = payload["qualified_runtime_reference"]["delta_escrow"]
        index_path = args.application_root / escrow["path"] / "index.json"
        if not index_path.is_file():
            errors.append(f"delta escrow: missing {index_path}")
        elif sha256_file(index_path) != escrow["index_sha256"]:
            errors.append("delta escrow index SHA-256 mismatch")
        else:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("qualified_python_files") != escrow["qualified_python_files"]:
                errors.append("delta escrow file count mismatch")
            if index.get("qualified_tree_sha256") != escrow["qualified_tree_sha256"]:
                errors.append("delta escrow tree SHA-256 mismatch")
            if index.get("counts") != escrow["classification"]:
                errors.append("delta escrow classification mismatch")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print(f"qualified candidate manifest valid: {args.manifest}")


if __name__ == "__main__":
    main()
