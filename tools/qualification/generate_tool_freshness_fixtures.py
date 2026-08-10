#!/usr/bin/env python3
"""Generate the immutable Pocket speech twin for tool freshness parity."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import wave
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager
from nemotron_voicechat_runtime.tool_freshness_contract import scenario_document
from nemotron_voicechat_runtime.tool_freshness_fixture import (
    FIXTURE_BASE_SEED,
    FIXTURE_INSTALLED_WORKER_SOURCE,
    FIXTURE_KIND,
    FIXTURE_LANGUAGE,
    FIXTURE_MANIFEST_NAME,
    FIXTURE_PROJECT_SOURCE_PATHS,
    FIXTURE_SAMPLE_RATE,
    FIXTURE_SCHEMA,
    FIXTURE_VOICE,
    TOOL_ONLY_PROMPTS,
    canonical_sha256,
    self_hash,
    sha256_bytes,
    sha256_path,
    validate_fixture_source_provenance,
    validate_tool_freshness_fixtures,
)


def source_hashes() -> dict[str, str]:
    project = Path("/workspace/project")
    paths = {name: project / name for name in FIXTURE_PROJECT_SOURCE_PATHS}
    paths[FIXTURE_INSTALLED_WORKER_SOURCE] = Path(
        "/opt/pocket-tts/lib/python3.12/site-packages/"
        "nemotron_voicechat_runtime/pocket_worker.py"
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"tool freshness fixture sources are missing: {missing}")
    return {name: sha256_path(path) for name, path in sorted(paths.items())}


async def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("tool freshness fixture output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    scenario_sha256 = canonical_sha256(scenario_document())
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        language=FIXTURE_LANGUAGE,
        voice=FIXTURE_VOICE,
        threads=args.threads,
        seed=FIXTURE_BASE_SEED,
        cpus=tuple(args.cpus),
        startup_timeout=args.startup_timeout,
    )
    fixtures = []
    try:
        await manager.start()
        health = dict(manager.health)
        for ordinal, prompt in enumerate(TOOL_ONLY_PROMPTS, 1):
            case_id = f"tool-freshness-{ordinal}"
            result = await manager.synthesize(
                prompt,
                job_id=case_id,
                cancel_event=asyncio.Event(),
            )
            if result.sample_rate != FIXTURE_SAMPLE_RATE or len(result.pcm16) % 2:
                raise RuntimeError("Pocket returned an invalid tool freshness carrier")
            pcm_path = args.output / f"{case_id}.pcm"
            wav_path = args.output / f"{case_id}.wav"
            pcm_path.write_bytes(result.pcm16)
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(FIXTURE_SAMPLE_RATE)
                output.writeframes(result.pcm16)
            samples = len(result.pcm16) // 2
            fixtures.append(
                {
                    "case_id": case_id,
                    "ordinal": ordinal,
                    "text": prompt,
                    "pcm": {
                        "path": pcm_path.name,
                        "sample_rate": FIXTURE_SAMPLE_RATE,
                        "sample_count": samples,
                        "bytes": len(result.pcm16),
                        "sha256": sha256_bytes(result.pcm16),
                    },
                    "wav": {
                        "path": wav_path.name,
                        "sample_rate": FIXTURE_SAMPLE_RATE,
                        "channels": 1,
                        "sample_width": 2,
                        "samples": samples,
                        "bytes": wav_path.stat().st_size,
                        "sha256": sha256_path(wav_path),
                    },
                }
            )
    finally:
        await manager.stop()
    manifest: dict[str, object] = {
        "schema": FIXTURE_SCHEMA,
        "kind": FIXTURE_KIND,
        "scenario_sha256": scenario_sha256,
        "fixtures": fixtures,
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
            "source_sha256": source_hashes(),
        },
    }
    manifest["sha256"] = self_hash(manifest)
    (args.output / FIXTURE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_tool_freshness_fixtures(
        args.output,
        expected_runtime_image_id=args.runtime_image_id,
        expected_scenario_sha256=scenario_sha256,
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
