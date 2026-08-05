#!/usr/bin/env python3
"""Reproduce the two qualified Voicechat checkpoints from exact public inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nano_gptq_calibration import assert_byte_identical, canonical_json_sha256, sha256_file
from package_replay_corpus import build_release_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--nano-skeleton", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPO_ROOT / "config/nano-gptq-corpus-selection.json",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=TOOLS_ROOT / "fixtures",
    )
    parser.add_argument(
        "--expected-bom",
        type=Path,
        default=REPO_ROOT / "config/qualified-candidate-1.json",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--asr-model",
        type=Path,
        help="Local Parakeet model used by the required EarTTS component gate.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def _tree_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _write_stage_marker(work_root: Path, name: str, outputs: list[Path]) -> None:
    files = {
        str(output.relative_to(work_root)): _tree_files(output)
        if output.is_dir()
        else {
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        for output in outputs
    }
    payload = {"schema": 1, "stage": name, "outputs": files}
    payload["sha256"] = canonical_json_sha256(payload)
    marker = work_root / "stages" / f"{name}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)


def _validate_stage_marker(work_root: Path, name: str) -> bool:
    marker = work_root / "stages" / f"{name}.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected_sha = payload.pop("sha256", None)
    if canonical_json_sha256(payload) != expected_sha:
        raise RuntimeError(f"corrupt stage marker: {marker}")
    actual = {}
    for relative, recorded in payload["outputs"].items():
        output = work_root / relative
        if isinstance(recorded, list):
            if not output.is_dir():
                raise RuntimeError(f"completed stage {name} lost directory: {output}")
            actual[relative] = _tree_files(output)
        else:
            if not output.is_file():
                raise RuntimeError(f"completed stage {name} lost file: {output}")
            actual[relative] = {
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
    if actual != payload["outputs"]:
        raise RuntimeError(f"completed stage {name} no longer matches its marker")
    return True


def _run_stage(
    work_root: Path,
    name: str,
    outputs: list[Path],
    action,
) -> None:
    if _validate_stage_marker(work_root, name):
        print(f"[resume] validated completed stage: {name}", flush=True)
        return
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(
            f"stage {name} has unvalidated partial outputs: {existing}; move them aside and rerun"
        )
    action()
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"stage {name} did not create declared outputs: {missing}")
    _write_stage_marker(work_root, name, outputs)


def _verify_expected_artifact(root: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    expected_files = expected["files"]
    actual_names = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if actual_names != sorted(expected_files):
        raise RuntimeError(
            f"{label} file inventory differs from Production Candidate 1: "
            f"{actual_names} != {sorted(expected_files)}"
        )
    mismatches = {}
    for name, digest in expected_files.items():
        actual = sha256_file(root / name)
        if actual != digest:
            mismatches[name] = {"expected": digest, "actual": actual}
    if mismatches:
        raise RuntimeError(f"{label} differs from Production Candidate 1: {mismatches}")
    return {
        "label": label,
        "root": str(root),
        "composite_sha256": expected["composite_sha256"],
        "files": expected_files,
    }


def _frozen_source_names(expected_corpus: dict[str, Any]) -> dict[tuple[str, int], str]:
    """Restore provenance labels lost when replay roots are archived by index."""

    result: dict[tuple[str, int], str] = {}
    for root in expected_corpus["roots"]:
        key = (str(root["split"]), int(root["root_index"]))
        source_name = root.get("source_name")
        if key in result or not isinstance(source_name, str) or not source_name:
            raise ValueError("expected corpus has invalid or duplicate replay source names")
        result[key] = source_name
    return result


def _make_release_readable(root: Path) -> None:
    """Make root-container outputs consumable by the invoking host user."""

    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)


def main() -> None:
    args = parse_args()
    speech = args.speech_root.expanduser().resolve()
    checkpoint = args.checkpoint_root.expanduser().resolve()
    skeleton = args.nano_skeleton.expanduser().resolve()
    calibration = [path.expanduser().resolve() for path in args.calibration_root]
    evaluation = [path.expanduser().resolve() for path in args.evaluation_root]
    selection = args.selection.expanduser().resolve()
    fixtures = args.fixture_root.expanduser().resolve()
    asr_model = args.asr_model.expanduser().resolve() if args.asr_model else None
    work = args.work_root.expanduser().resolve()
    if work.exists() and not work.is_dir():
        raise SystemExit(f"work root is not a directory: {work}")
    work.mkdir(parents=True, exist_ok=True)

    expected_corpus = json.loads(
        (REPO_ROOT / "config/calibration-corpus-production-candidate-1.json").read_text()
    )
    corpus_manifest = build_release_manifest(
        calibration,
        evaluation,
        selection,
        fixtures,
        source_names=_frozen_source_names(expected_corpus),
    )
    if corpus_manifest != expected_corpus:
        raise SystemExit("supplied replay corpus differs from Production Candidate 1")

    extraction_command = (
        str(TOOLS_ROOT / "extract_public_vllm.py"),
        "--speech-root",
        str(speech),
        "--checkpoint-root",
        str(checkpoint),
        "--nano-skeleton",
        str(skeleton),
        "--output-root",
        str(work / "fp32"),
        "--reproducibility-runs",
        "2",
    )
    if args.preflight_only:
        _run(*extraction_command, "--preflight-only")
        print(
            json.dumps(
                {
                    "status": "preflight_complete",
                    "corpus_release_sha256": corpus_manifest["release_sha256"],
                },
                indent=2,
            )
        )
        return
    if asr_model is None:
        raise SystemExit("--asr-model is required for the EarTTS component gate")

    fp32 = work / "fp32"
    _run_stage(
        work,
        "extract-public-components",
        [fp32],
        lambda: _run(*extraction_command),
    )

    nano_run1 = work / "nano-run1"
    nano_run2 = work / "nano-run2"
    nano_report1 = work / "reports/nano-conversion-run1.json"
    nano_report2 = work / "reports/nano-conversion-run2.json"

    def convert_nano() -> None:
        for run_index, output, report in (
            (1, nano_run1, nano_report1),
            (2, nano_run2, nano_report2),
        ):
            command = [
                str(TOOLS_ROOT / "convert_nano_marlin_gptq.py"),
                str(fp32 / "nano-vllm-fp32"),
                str(output),
                "--single-run",
                "--work-dir",
                str(work / f"scratch/nano-run{run_index}"),
                "--report",
                str(report),
                "--model-code-root",
                str(skeleton),
                "--corpus-selection",
                str(selection),
                "--device",
                args.device,
            ]
            for root in calibration:
                command.extend(("--calibration-replay", str(root)))
            for root in evaluation:
                command.extend(("--evaluation-replay", str(root)))
            _run(*command)

    _run_stage(
        work,
        "quantize-nano-w8-gptq",
        [nano_run1, nano_run2, nano_report1, nano_report2],
        convert_nano,
    )
    assert_byte_identical(nano_run1, nano_run2)

    eartts_run1 = work / "eartts-run1"
    eartts_run2 = work / "eartts-run2"
    eartts_report1 = work / "reports/eartts-conversion-run1.json"
    eartts_report2 = work / "reports/eartts-conversion-run2.json"

    def convert_eartts() -> None:
        for output, report in (
            (eartts_run1, eartts_report1),
            (eartts_run2, eartts_report2),
        ):
            _run(
                str(TOOLS_ROOT / "convert_eartts_w8a32.py"),
                str(fp32 / "eartts-vllm-fp32"),
                str(output),
                "--report",
                str(report),
                "--device",
                args.device,
            )

    _run_stage(
        work,
        "quantize-eartts-w8a32",
        [eartts_run1, eartts_run2, eartts_report1, eartts_report2],
        convert_eartts,
    )
    assert_byte_identical(eartts_run1, eartts_run2)

    manifest_nano = work / "manifests/nano.json"
    manifest_combined = work / "manifests/nano-eartts.json"
    nano_final = work / "release/nano"
    eartts_final = work / "release/eartts"
    manifest_final = work / "release/manifest.json"

    def finalize() -> None:
        manifest_nano.parent.mkdir(parents=True, exist_ok=True)
        _run(
            str(TOOLS_ROOT / "create_quantized_nano_manifest.py"),
            "--base-manifest",
            str(fp32 / "manifest.json"),
            "--quant-report",
            str(nano_report1),
            "--nano-root",
            str(nano_run1),
            "--runtime-image",
            "pending-public-runtime",
            "--runtime-image-id",
            "pending-phase-4",
            "--output",
            str(manifest_nano),
            "--byte-identical",
        )
        _run(
            str(TOOLS_ROOT / "create_quantized_eartts_manifest.py"),
            "--base-manifest",
            str(manifest_nano),
            "--quant-report",
            str(eartts_report1),
            "--eartts-root",
            str(eartts_run1),
            "--output",
            str(manifest_combined),
            "--byte-identical",
        )
        _run(
            str(TOOLS_ROOT / "finalize_nano_release.py"),
            "--source-dir",
            str(nano_run1),
            "--output-dir",
            str(nano_final),
        )
        _run(
            str(TOOLS_ROOT / "create_eartts_window_variant.py"),
            "--source-dir",
            str(eartts_run1),
            "--source-manifest",
            str(manifest_combined),
            "--output-dir",
            str(eartts_final),
            "--output-manifest",
            str(manifest_final),
            "--sliding-window",
            "1500",
        )

    _run_stage(
        work,
        "finalize-qualified-artifacts",
        [nano_final, eartts_final, manifest_final],
        finalize,
    )

    bom = json.loads(args.expected_bom.expanduser().resolve().read_text())
    nano_verification = _verify_expected_artifact(
        nano_final, bom["qualified_artifacts"]["nano"], "nano"
    )
    eartts_verification = _verify_expected_artifact(
        eartts_final, bom["qualified_artifacts"]["eartts"], "eartts"
    )

    nano_gate = work / "gates/nano"
    eartts_gate_eager = work / "gates/eartts-eager"
    eartts_gate_graph = work / "gates/eartts-graph"

    def run_component_gates() -> None:
        _run(
            str(TOOLS_ROOT / "nano_component_gate.py"),
            "--speech-root",
            str(speech),
            "--nano-vllm-path",
            str(nano_run1),
            "--teacher-replay",
            str(evaluation[0]),
            "--output-dir",
            str(nano_gate),
            "--max-calls",
            str(corpus_manifest["selected"]["evaluation"]["calls"]),
        )
        for output, graph_flag in (
            (eartts_gate_eager, ()),
            (eartts_gate_graph, ("--no-enforce-eager",)),
        ):
            _run(
                str(TOOLS_ROOT / "eartts_component_gate.py"),
                "--speech-root",
                str(speech),
                "--checkpoint-root",
                str(checkpoint),
                "--nano-skeleton",
                str(skeleton),
                "--eartts-vllm-path",
                str(eartts_final),
                "--asr-model",
                str(asr_model),
                "--output-dir",
                str(output),
                *graph_flag,
            )

    _run_stage(
        work,
        "component-gates",
        [nano_gate, eartts_gate_eager, eartts_gate_graph],
        run_component_gates,
    )

    # Docker writes bind-mounted conversion outputs as root. Normalize only
    # permissions (never bytes) so the developer who invoked bootstrap can
    # inspect, compare, and publish the verified release from the host.
    _make_release_readable(work / "release")

    verification = {
        "schema": 1,
        "kind": "voicechat_release_artifact_reproduction",
        "corpus_release_sha256": corpus_manifest["release_sha256"],
        "nano": nano_verification,
        "eartts": eartts_verification,
        "dual_conversion": {
            "nano": assert_byte_identical(nano_run1, nano_run2),
            "eartts": assert_byte_identical(eartts_run1, eartts_run2),
        },
        "component_gates": {
            "nano": str(nano_gate),
            "eartts_eager": str(eartts_gate_eager),
            "eartts_graph": str(eartts_gate_graph),
        },
    }
    verification["sha256"] = canonical_json_sha256(verification)
    verification_path = work / "release/reproduction.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
