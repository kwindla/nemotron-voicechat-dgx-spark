#!/usr/bin/env python3
"""Validate and inventory the exact replay corpus used by Production Candidate 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from nano_gptq_calibration import (
    assert_disjoint_corpora,
    canonical_json_sha256,
    inspect_replay_corpus,
    select_replay_corpus,
    sha256_file,
)

FIXTURE_LICENSE = (
    "Project-authored text fixtures; replay tensors are outputs of the "
    "OpenMDW-1.1 NVIDIA parent model and contain no raw user recordings."
)


def _portable_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    portable = []
    for record in records:
        item = dict(record)
        item.pop("capture_root", None)
        portable.append(item)
    return portable


def _root_inventory(
    root: Path,
    index: int,
    split: str,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    metadata = root / "metadata.json"
    manifest = root / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    files = [
        {
            "path": "metadata.json",
            "bytes": metadata.stat().st_size,
            "sha256": sha256_file(metadata),
        },
        {
            "path": "manifest.jsonl",
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
    ]
    files.extend(
        {
            "path": str(record["file"]),
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    )
    return {
        "split": split,
        "root_index": index,
        "release_path": f"{split}/{index}",
        "source_name": source_name if source_name is not None else root.name,
        "files": files,
        "files_sha256": canonical_json_sha256(files),
    }


def _fixture_inventory(fixture_root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(fixture_root.glob("*.json")):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"no fixture definitions found in {fixture_root}")
    return files


def build_release_manifest(
    calibration_roots: list[Path],
    evaluation_roots: list[Path],
    selection_path: Path,
    fixture_root: Path,
    *,
    verify_payload_hashes: bool = True,
    source_names: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    """Build a path-independent inventory without modifying the replay payloads."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    calibration_all = inspect_replay_corpus(
        calibration_roots, verify_payload_hashes=verify_payload_hashes
    )
    evaluation_all = inspect_replay_corpus(
        evaluation_roots, verify_payload_hashes=verify_payload_hashes
    )
    calibration = select_replay_corpus(calibration_all, selection, split="calibration")
    evaluation = select_replay_corpus(evaluation_all, selection, split="evaluation")
    assert_disjoint_corpora(calibration, evaluation)

    root_keys = {
        *(("calibration", index) for index in range(len(calibration_roots))),
        *(("evaluation", index) for index in range(len(evaluation_roots))),
    }
    if source_names is not None and set(source_names) != root_keys:
        raise ValueError(
            "explicit replay source names must cover every split/root index exactly"
        )
    roots = [
        *(
            _root_inventory(
                root,
                index,
                "calibration",
                source_name=(source_names or {}).get(("calibration", index)),
            )
            for index, root in enumerate(calibration_roots)
        ),
        *(
            _root_inventory(
                root,
                index,
                "evaluation",
                source_name=(source_names or {}).get(("evaluation", index)),
            )
            for index, root in enumerate(evaluation_roots)
        ),
    ]
    fixtures = _fixture_inventory(fixture_root)
    selected = {}
    for name, corpus in (("calibration", calibration), ("evaluation", evaluation)):
        records = _portable_records(corpus.records)
        selected[name] = {
            "calls": corpus.calls,
            "conversations": corpus.conversations,
            "effective_rows_lower_bound": corpus.effective_rows,
            "records_sha256": canonical_json_sha256(records),
        }

    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "voicechat_gptq_replay_release",
        "redistributable": True,
        "contains_raw_user_recordings": False,
        "license_basis": FIXTURE_LICENSE,
        "checkpoint_provenance": calibration.provenance,
        "selection": {
            "path": "nano-gptq-corpus-selection.json",
            "bytes": selection_path.stat().st_size,
            "sha256": sha256_file(selection_path),
            "canonical_sha256": canonical_json_sha256(selection),
        },
        "fixtures": {
            "release_path": "fixtures",
            "license": "Apache-2.0 (this repository)",
            "files": fixtures,
            "files_sha256": canonical_json_sha256(fixtures),
        },
        "roots": roots,
        "selected": selected,
    }
    payload["release_sha256"] = canonical_json_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-root", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-payload-hash-verification",
        action="store_true",
        help="Development-only; release generation must not use this option.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_release_manifest(
        [path.expanduser().resolve() for path in args.calibration_root],
        [path.expanduser().resolve() for path in args.evaluation_root],
        args.selection.expanduser().resolve(),
        args.fixture_root.expanduser().resolve(),
        verify_payload_hashes=not args.skip_payload_hash_verification,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"refusing to overwrite a different corpus manifest: {output}")
    output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "release_sha256": manifest["release_sha256"],
                "calibration_calls": manifest["selected"]["calibration"]["calls"],
                "evaluation_calls": manifest["selected"]["evaluation"]["calls"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
