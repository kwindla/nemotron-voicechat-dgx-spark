#!/usr/bin/env python3
"""Replay exact historical UTF-8 carrier bytes through the production Pocket backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path

from nemotron_voicechat_runtime.pocket_worker import PocketBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--pcm-output", type=Path, required=True)
    args = parser.parse_args()
    carrier = args.carrier.read_bytes()
    text = carrier.decode("utf-8", errors="strict")
    if text.encode("utf-8") != carrier or not text:
        raise ValueError("carrier is not a non-empty canonical UTF-8 byte sequence")
    backend = PocketBackend(
        language="english_2026-04",
        voice="alba",
        quantize=True,
        threads=4,
        seed=0,
    )
    backend.load_and_prewarm()
    result = backend.synthesize(text, threading.Event())
    args.pcm_output.write_bytes(result.pcm16)
    print(
        json.dumps(
            {
                "assets": backend.assets,
                "carrier_bytes": len(carrier),
                "carrier_sha256": hashlib.sha256(carrier).hexdigest(),
                "pcm_bytes": len(result.pcm16),
                "pcm_sha256": hashlib.sha256(result.pcm16).hexdigest(),
                "pocket_package_version": backend.package_version,
                "sample_rate": 16000,
                "seed_scheme": "sha256-text-plus-base-v1",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
