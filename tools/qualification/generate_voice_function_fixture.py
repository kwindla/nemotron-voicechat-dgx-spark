#!/usr/bin/env python3
"""Generate the frozen Pocket c014 carrier used by the voice logit control."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import wave
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager
from nemotron_voicechat_runtime.voice_function_fixture import (
    FIXTURE_BASE_SEED,
    FIXTURE_CASE_ID,
    FIXTURE_KIND,
    FIXTURE_LANGUAGE,
    FIXTURE_MANIFEST_NAME,
    FIXTURE_MAX_SAMPLES,
    FIXTURE_PCM_NAME,
    FIXTURE_SAMPLE_RATE,
    FIXTURE_SCHEMA,
    FIXTURE_VOICE,
    FIXTURE_WAV_NAME,
    corpus_sha256,
    fixture_case,
    sha256_bytes,
    sha256_path,
    validate_fixture_source_provenance,
    validate_voice_function_fixture,
)


def _source_hashes() -> dict[str, str]:
    project = Path("/workspace/project")
    package = project / "src/nemotron_voicechat_runtime"
    paths = {
        "src/nemotron_voicechat_runtime/pocket_controller.py": package
        / "pocket_controller.py",
        "src/nemotron_voicechat_runtime/pocket_worker.py": package / "pocket_worker.py",
        "src/nemotron_voicechat_runtime/voice_function_fixture.py": package
        / "voice_function_fixture.py",
        "tools/qualification/generate_voice_function_fixture.py": project
        / "tools/qualification/generate_voice_function_fixture.py",
    }
    installed_worker = Path(
        "/opt/pocket-tts/lib/python3.12/site-packages/"
        "nemotron_voicechat_runtime/pocket_worker.py"
    )
    if installed_worker.is_file():
        paths["pocket_environment/pocket_worker.py"] = installed_worker
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"voice fixture generator sources are missing: {missing}")
    return {name: sha256_path(path) for name, path in sorted(paths.items())}


async def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("voice function fixture output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    _, case = fixture_case(args.corpus)
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        language=FIXTURE_LANGUAGE,
        voice=FIXTURE_VOICE,
        threads=args.threads,
        seed=FIXTURE_BASE_SEED,
        cpus=tuple(args.cpus),
        startup_timeout=args.startup_timeout,
    )
    try:
        await manager.start()
        health = dict(manager.health)
        result = await manager.synthesize(
            case["prompt"],
            job_id=f"voice-function-{FIXTURE_CASE_ID}",
            cancel_event=asyncio.Event(),
        )
    finally:
        await manager.stop()
    if result.sample_rate != FIXTURE_SAMPLE_RATE or len(result.pcm16) % 2:
        raise RuntimeError("Pocket returned an invalid voice function carrier format")
    sample_count = len(result.pcm16) // 2
    if not 0 < sample_count <= FIXTURE_MAX_SAMPLES:
        raise RuntimeError("Pocket voice function carrier length is outside the bound")

    pcm_path = args.output / FIXTURE_PCM_NAME
    wav_path = args.output / FIXTURE_WAV_NAME
    pcm_path.write_bytes(result.pcm16)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(FIXTURE_SAMPLE_RATE)
        output.writeframes(result.pcm16)
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "case_id": FIXTURE_CASE_ID,
        "corpus_sha256": corpus_sha256(args.corpus),
        "prompt": case["prompt"],
        "pcm": {
            "path": FIXTURE_PCM_NAME,
            "sample_rate": FIXTURE_SAMPLE_RATE,
            "sample_count": sample_count,
            "bytes": len(result.pcm16),
            "sha256": sha256_bytes(result.pcm16),
        },
        "wav": {
            "path": FIXTURE_WAV_NAME,
            "bytes": wav_path.stat().st_size,
            "sha256": sha256_path(wav_path),
        },
        "synthesis": {
            "language": health["language"],
            "voice": health["voice"],
            "quantize": True,
            "threads": health["threads"],
            "cpus": health["cpus"],
            "base_seed": health["seed"],
            "seed_scheme": health["seed_scheme"],
            "package_version": health["package_version"],
            "assets": health["assets"],
        },
        "provenance": {
            "runtime_image": args.runtime_image,
            "runtime_image_id": args.runtime_image_id,
            "source_sha256": _source_hashes(),
        },
    }
    (args.output / FIXTURE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_voice_function_fixture(
        args.output,
        corpus_path=args.corpus,
        expected_runtime_image_id=args.runtime_image_id,
    )
    validate_fixture_source_provenance(
        manifest,
        project_root=Path("/workspace/project"),
        installed_worker=Path(
            "/opt/pocket-tts/lib/python3.12/site-packages/"
            "nemotron_voicechat_runtime/pocket_worker.py"
        ),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--cpus", type=int, nargs="+", default=[7, 8, 9, 15])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    manifest = asyncio.run(generate(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
