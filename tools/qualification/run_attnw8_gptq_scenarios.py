#!/usr/bin/env python3
"""Run the Seal-0 attnW8-GPTQ A2/C2 baseline scenarios in fixed order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:8786/v1/realtime")
    parser.add_argument("--health-url", default="http://127.0.0.1:8786/health")
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()
    fixtures = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    records = [r for r in fixtures["records"] if r["partition"] in {"a2", "c2"}]
    if len(records) != 48:
        raise SystemExit(f"expected 48 A2/C2 records, found {len(records)}")
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(args.health_url, timeout=10) as response:
        health = json.loads(response.read())
    results = []
    for ordinal, record in enumerate(records):
        pcm = args.fixture_root / record["pcm"]["path"]
        if sha256(pcm) != record["pcm"]["sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {pcm}")
        out = args.output / record["scenario_id"]
        command = [
            sys.executable,
            str(args.harness),
            "--url", args.url,
            "--scenario", record["scenario_id"],
            "--speech-pcm", str(pcm),
            "--tool", "benchmark_word",
            "--max-injections", "1",
            "--run-text-chars", "20",
            "--trailing-silence-s", "8.0",
            "--turn-quiesce-s", "1.0",
            "--max-silence-frames", "15",
            "--turn-quiesce-cap-s", "25.0",
            "--observe-s", "15.0",
            "--output", str(out),
        ]
        started = time.time()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        item = {
            "ordinal": ordinal,
            "scenario_id": record["scenario_id"],
            "expected": record["expected"],
            "command": command,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        results.append(item)
        (args.output / "progress.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"scenario process failed: {record['scenario_id']}")
        time.sleep(args.settle_seconds)
    receipt = {
        "schema": 1,
        "kind": "attnw8_gptq_baseline_tool_capture",
        "health_before": health,
        "fixture_manifest_sha256": sha256(args.fixture_manifest),
        "harness_sha256": sha256(args.harness),
        "scenario_count": len(results),
        "results": results,
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
