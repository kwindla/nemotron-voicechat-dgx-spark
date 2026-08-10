#!/usr/bin/env python3
"""Re-evaluate retained response audio without mutating or rerunning model runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from transcribe_sustained import report_kind, resolve_audio_path, runtime_gate_passed

from nemotron_voicechat_runtime.artifacts import atomic_json

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATIONS = (
    ("multiturn", "multiturn_strict_v3", None),
    ("memory-matrix", "memory_matrix_strict_v3", 0.0),
    ("sustained", "sustained_strict_v3", None),
)
INHERITED_EVIDENCE = (
    "report.json",
    "browser-attempts.json",
    "restart-cycles/report.json",
    "conversion-tensor-tests.log",
    "nano-component.log",
    "eartts-eager-component.log",
    "eartts-graph-component.log",
    "pocket-component.log",
    "stack.log",
    "multiturn.log",
    "memory-matrix.log",
    "sustained.log",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, source_run: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required retained artifact is missing: {path}")
    try:
        name = str(path.resolve().relative_to(source_run.resolve()))
    except ValueError:
        name = str(path.resolve())
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve_manifest(source_run: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    candidate = source_run / path.name
    if candidate.is_file():
        return candidate
    raise RuntimeError(f"retained manifest is missing: {recorded}")


def collect_consumed_inputs(source_run: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    records: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, expected_kind, _max_silent_rate in QUALIFICATIONS:
        run_dir = source_run / name
        report_path = run_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_kind = report_kind(report)
        if actual_kind != expected_kind:
            raise RuntimeError(
                f"{name}: expected {expected_kind}, found {actual_kind or 'unknown/ambiguous'}"
            )
        if not runtime_gate_passed(report):
            raise RuntimeError(f"{name}: retained pre-ASR runtime gate is red")
        response_records = []
        for index, response in enumerate(report.get("responses") or [], 1):
            recorded = response.get("audio_path") if isinstance(response, dict) else None
            if not isinstance(recorded, str) or not recorded:
                raise RuntimeError(f"{name}: response {index} has no audio_path")
            audio_path = resolve_audio_path(run_dir, recorded)
            if not audio_path.resolve().is_relative_to(run_dir.resolve()):
                raise RuntimeError(f"{name}: response {index} audio_path escapes its run dir")
            record = file_record(audio_path, source_run)
            if record["bytes"] <= 44:
                raise RuntimeError(f"{name}: response {index} WAV is empty: {audio_path}")
            response_records.append(record)
            paths[f"{name}:response:{index}"] = audio_path
        if not response_records:
            raise RuntimeError(f"{name}: retained report has no responses")
        manifest_record = None
        if isinstance(report.get("manifest"), str):
            manifest_path = resolve_manifest(source_run, report["manifest"])
            if not manifest_path.resolve().is_relative_to(source_run.resolve()):
                raise RuntimeError(f"{name}: manifest escapes the retained source run")
            manifest_record = file_record(manifest_path, source_run)
            paths[f"{name}:manifest"] = manifest_path
        records[name] = {
            "report_kind": actual_kind,
            "runtime_gate_recomputed": True,
            "report": file_record(report_path, source_run),
            "manifest": manifest_record,
            "response_audio": response_records,
        }
        paths[f"{name}:report"] = report_path
    return records, paths


def collect_inherited_evidence(source_run: Path) -> dict[str, Any]:
    result = {name: file_record(source_run / name, source_run) for name in INHERITED_EVIDENCE}
    browser = json.loads((source_run / "browser-attempts.json").read_text(encoding="utf-8"))
    restart = json.loads(
        (source_run / "restart-cycles/report.json").read_text(encoding="utf-8")
    )
    if browser.get("passed") is not True or restart.get("passed") is not True:
        raise RuntimeError("retained inherited browser or restart verdict is red")
    return result


def verify_snapshot(paths: dict[str, Path], records: dict[str, Any]) -> None:
    for name, expected in records.items():
        current = file_record(paths[name], paths[name].parent)
        if (current["bytes"], current["sha256"]) != (expected["bytes"], expected["sha256"]):
            raise RuntimeError(f"retained input changed during recovery: {name}")


def resolve_runtime_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        capture_output=True,
        check=False,
    )
    image_id = result.stdout.strip()
    if result.returncode or not image_id.startswith("sha256:") or len(image_id) != 71:
        raise RuntimeError(f"could not resolve immutable runtime image ID for {image}")
    return image_id


def evaluator_command(
    args: argparse.Namespace,
    name: str,
    output_dir: Path,
    max_silent_rate: float | None,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "-v",
        f"{REPO_ROOT}:/workspace/project:ro",
        "-v",
        f"{(args.source_run / name).resolve()}:/qualification-input:ro",
        "-v",
        f"{output_dir.resolve()}:/qualification-output",
        "-e",
        f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID={args.asr_evaluator_image_id}",
        args.runtime_image_id,
        "python3",
        "-P",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/qualification-input",
        "--output-dir",
        "/qualification-output",
    ]
    if max_silent_rate is not None:
        command.extend(("--max-silent-rate", str(max_silent_rate)))
    return command


def run_checked(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
    if result.returncode:
        raise RuntimeError(f"offline ASR failed with status {result.returncode}: {log_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.source_run = args.source_run.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.source_run.is_dir():
        raise RuntimeError(f"source run does not exist: {args.source_run}")
    if args.output.exists():
        raise RuntimeError(f"recovery output already exists: {args.output}")
    if args.source_run == args.output or args.source_run in args.output.parents:
        raise RuntimeError("recovery output must not be inside the retained source run")
    args.runtime_image_id = resolve_runtime_image_id(args.asr_evaluator_image)
    args.asr_evaluator_image_id = args.runtime_image_id
    subprocess.run(
        [
            str(REPO_ROOT / "container/audit-asr-evaluator.sh"),
            args.asr_evaluator_image_id,
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    consumed, consumed_paths = collect_consumed_inputs(args.source_run)
    inherited = collect_inherited_evidence(args.source_run)
    evaluator_path = REPO_ROOT / "tools/qualification/transcribe_sustained.py"
    consumed_paths["evaluator"] = evaluator_path
    flat_snapshot = {
        name: file_record(path, args.source_run) for name, path in consumed_paths.items()
    }
    source_report = json.loads((args.source_run / "report.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    atomic_json(
        args.output / "input-snapshot.json",
        {
            "report_kind": "offline_asr_recovery_input_snapshot_v1",
            "consumed_inputs": consumed,
            "evaluator": flat_snapshot["evaluator"],
            "asr_evaluator_image": args.asr_evaluator_image,
            "asr_evaluator_image_id": args.runtime_image_id,
            "inherited_evidence": inherited,
        },
    )

    results = {}
    for name, expected_kind, max_silent_rate in QUALIFICATIONS:
        output_dir = args.output / name
        output_dir.mkdir()
        log_path = args.output / f"{name}-asr.log"
        run_checked(evaluator_command(args, name, output_dir, max_silent_rate), log_path)
        evaluated = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        if report_kind(evaluated) != expected_kind:
            raise RuntimeError(f"{name}: recovered report kind changed")
        if not runtime_gate_passed(evaluated):
            raise RuntimeError(f"{name}: recovered runtime gate is red")
        if evaluated.get("external_asr", {}).get("passed") is not True:
            raise RuntimeError(f"{name}: recovered external ASR gate is red")
        if evaluated.get("passed") is not True:
            raise RuntimeError(f"{name}: recovered aggregate gate is red")
        results[name] = {
            "report_kind": expected_kind,
            "runtime_gate_passed": True,
            "external_asr": evaluated["external_asr"],
            "report": file_record(output_dir / "report.json", args.output),
            "log": file_record(log_path, args.output),
        }

    verify_snapshot(consumed_paths, flat_snapshot)
    if collect_inherited_evidence(args.source_run) != inherited:
        raise RuntimeError("inherited evidence changed during recovery")
    report = {
        "report_kind": "offline_asr_recovery_v1",
        "passed": bool(
            len(results) == len(QUALIFICATIONS)
            and all(
                result["runtime_gate_passed"] and result["external_asr"]["passed"]
                for result in results.values()
            )
        ),
        "source_run": str(args.source_run),
        "source_suite_report": {
            "passed": source_report.get("passed"),
            "error": source_report.get("error"),
            "artifact": inherited["report.json"],
            "preserved_unmodified": True,
        },
        "asr_evaluator_image": args.asr_evaluator_image,
        "asr_evaluator_image_id": args.runtime_image_id,
        "evaluator": flat_snapshot["evaluator"],
        "reverified": {
            "retained_input_integrity_before_and_after": True,
            "pre_asr_runtime_gates": [kind for _name, kind, _rate in QUALIFICATIONS],
            "external_asr": results,
        },
        "inherited_not_reexecuted": {
            "cold_start_timing": "not recoverable from the failed top-level report",
            "restart_cycles": inherited["restart-cycles/report.json"],
            "browser": inherited["browser-attempts.json"],
            "component_and_live_stack_logs": {
                name: record
                for name, record in inherited.items()
                if name.endswith(".log")
            },
        },
        "limitations": [
            "This is not a fresh full live-suite run.",
            (
                "Cold-start, restart, browser, component, and live-stack gates are inherited "
                "artifacts and were not re-executed."
            ),
            (
                "Input hashes establish integrity across this recovery operation, not "
                "provenance before the snapshot was created."
            ),
        ],
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-evaluator-image", required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "report_kind": "offline_asr_recovery_v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
