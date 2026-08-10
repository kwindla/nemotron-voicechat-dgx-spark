#!/usr/bin/env python3
"""Rerun the legacy NeMo ASR judge with an evidence-bearing Docker invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

LEGACY_DECODER_GIT_PATH = "HEAD:tools/qualification/transcribe_sustained.py"
LEGACY_DECODER_GIT_BLOB = "8b8af889db779aeeaf660b4f57d676ebdcd9ed6f"
LEGACY_DECODER_SHA256 = "674a9ba6a97cd09186e16a69b355c59d67708d2773405b97d82b2a0f2d38cee3"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def checked_output(*argv: str) -> str:
    return subprocess.check_output(argv, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite legacy ASR rerun: {output_dir}")

    source_report = run_dir / "report.json"
    retained_source_report = output_dir / "source-report.json"
    shutil.copy2(source_report, retained_source_report)
    shutil.copy2(source_report, output_dir / "report.json")
    source_document = json.loads(source_report.read_text(encoding="utf-8"))
    for response in source_document.get("responses", []):
        recorded = Path(str(response.get("audio_path", "")))
        candidates = (recorded, run_dir / recorded, run_dir / recorded.name)
        source_audio = next((path for path in candidates if path.is_file()), None)
        if source_audio is None:
            raise FileNotFoundError(f"retained source audio is missing: {recorded}")
        shutil.copy2(source_audio, output_dir / source_audio.name)

    image_document = json.loads(
        checked_output("docker", "image", "inspect", args.runtime_image_id)
    )[0]
    inspected_image_id = image_document.get("Id")
    if inspected_image_id != args.runtime_image_id:
        raise ValueError("runtime image did not resolve to the requested immutable ID")

    decoder_bytes = subprocess.check_output(
        ("git", "show", LEGACY_DECODER_GIT_PATH), cwd=Path(__file__).resolve().parents[2]
    )
    decoder_blob = checked_output(
        "git", "rev-parse", LEGACY_DECODER_GIT_PATH
    )
    decoder_sha256 = sha256_bytes(decoder_bytes)
    if decoder_blob != LEGACY_DECODER_GIT_BLOB or decoder_sha256 != LEGACY_DECODER_SHA256:
        raise ValueError("legacy decoder no longer matches its locked Git object")
    retained_decoder = output_dir / "legacy-transcribe-sustained.py"
    retained_decoder.write_bytes(decoder_bytes)

    container_hash_argv = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        f"{model}:/models/asr.nemo:ro",
        "--volume",
        f"{retained_decoder}:/legacy/transcribe_sustained.py:ro",
        args.runtime_image_id,
        "python",
        "-c",
        (
            "import hashlib,json,pathlib;"
            "p=['/models/asr.nemo','/legacy/transcribe_sustained.py'];"
            "print(json.dumps({x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() "
            "for x in p},sort_keys=True))"
        ),
    ]
    container_hashes = json.loads(checked_output(*container_hash_argv))

    docker_argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{output_dir}:/work",
        "--volume",
        f"{model}:/models/asr.nemo:ro",
        "--volume",
        (
            f"{retained_decoder}:"
            "/workspace/project/tools/qualification/transcribe_sustained.py:ro"
        ),
        args.runtime_image_id,
        "python",
        "/workspace/project/tools/qualification/transcribe_sustained.py",
        "--run-dir",
        "/work",
        "--model",
        "/models/asr.nemo",
        "--max-wer",
        "0.5",
        "--silence-dbfs",
        "-55",
        "--max-silent-rate",
        "0",
    ]
    header = {
        "kind": "legacy_asr_execution",
        "schema": 2,
        "runtime_image_id": args.runtime_image_id,
        "image_inspect_id": inspected_image_id,
        "image_repo_digests": image_document.get("RepoDigests") or [],
        "model_mount": "/models/asr.nemo:ro",
        "source_run_dir": str(run_dir),
        "source_report_sha256": sha256_file(source_report),
        "retained_source_report": str(retained_source_report),
        "retained_source_report_sha256": sha256_file(retained_source_report),
        "model_sha256": sha256_file(model),
        "container_model_sha256": container_hashes["/models/asr.nemo"],
        "decoder_script_git_path": LEGACY_DECODER_GIT_PATH,
        "decoder_script_git_blob": decoder_blob,
        "decoder_script_sha256": decoder_sha256,
        "container_decoder_script_sha256": container_hashes[
            "/legacy/transcribe_sustained.py"
        ],
        "retained_decoder_script": str(retained_decoder),
        "network": "none",
        "gpu": "all",
        "ipc": "host",
        "docker_argv": docker_argv,
        "docker_argv_sha256": canonical_sha256(docker_argv),
    }
    log_path = output_dir / "execution.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        log.flush()
        completed = subprocess.run(
            docker_argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
