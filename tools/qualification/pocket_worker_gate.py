#!/usr/bin/env python3
"""Verify the isolated in-container Pocket worker and its real-time factor."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemotron_voicechat_runtime.pocket_controller import PocketWorkerManager


async def run(args: argparse.Namespace) -> dict[str, object]:
    manager = PocketWorkerManager(
        socket_path=args.output / "pocket.sock",
        cpus=tuple(args.cpus),
        threads=args.threads,
        startup_timeout=args.startup_timeout,
    )
    samples = []
    try:
        await manager.start()
        health = manager.health
        for index, text in enumerate(args.text):
            import asyncio as _asyncio

            result = await manager.synthesize(
                text,
                job_id=f"pocket-gate-{index}",
                cancel_event=_asyncio.Event(),
            )
            samples.append(
                {
                    "text": text,
                    "audio_bytes": len(result.pcm16),
                    "generated_seconds": result.generated_seconds,
                    "wall_seconds": result.wall_seconds,
                    "realtime_speed": result.generated_seconds / result.wall_seconds,
                    "time_to_first_audio_seconds": result.time_to_first_audio_seconds,
                }
            )
    finally:
        await manager.stop()
    minimum_speed = min(float(sample["realtime_speed"]) for sample in samples)
    report = {
        "passed": (
            health.get("ready") is True
            and health.get("cpus") == sorted(args.cpus)
            and minimum_speed >= args.minimum_realtime_speed
            and all(int(sample["audio_bytes"]) > 0 for sample in samples)
        ),
        "health": health,
        "minimum_realtime_speed": minimum_speed,
        "required_minimum_realtime_speed": args.minimum_realtime_speed,
        "samples": samples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", default=[7, 8, 9, 15])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--minimum-realtime-speed", type=float, default=1.0)
    parser.add_argument(
        "--text",
        action="append",
        default=[
            "The answer is five.",
            "The codeword is sapphire.",
            "This longer sentence verifies sustained synthesis faster than real time.",
        ],
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
