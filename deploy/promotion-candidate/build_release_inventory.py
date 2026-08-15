#!/usr/bin/env python3
"""Build the fhw8 release inventory (manifests/release.json) for upload.

Replaces the nano/* entries of the current signed release inventory with the
qualified fhw8 candidate files, recomputes the nano composite and release
hashes with the runtime's own algorithms, and verifies the result against a
hardlink-staged tree using the actual bootstrap verifier before printing the
new pins.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
from nemotron_voicechat_runtime.bootstrap_download import verify_release  # noqa: E402

cache = Path(
    os.environ.get("NEMOTRON_VOICECHAT_CACHE", Path.home() / ".cache/nemotron-voicechat")
)
release_root = cache / "artifacts/release"
candidate = cache / "artifacts/candidates/nano-fhw8"
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="fhw8-release-"))
out_dir.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = json.loads((release_root / "manifests/release.json").read_text(encoding="utf-8"))
old_release_sha = manifest["release_sha256"]

# replace nano/* entries with the candidate's actual files
kept = [item for item in manifest["files"] if not item["path"].startswith("nano/")]
nano_entries = []
for path in sorted(candidate.iterdir()):
    if path.is_file():
        nano_entries.append(
            {
                "path": f"nano/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
manifest["files"] = sorted(kept + nano_entries, key=lambda item: item["path"])

# nano composite: the server validator's shard digest algorithm
digest = hashlib.sha256()
shards = sorted(
    (
        e
        for e in nano_entries
        if e["path"].split("/", 1)[1].startswith("model-") and e["path"].endswith(".safetensors")
    ),
    key=lambda e: e["path"].split("/", 1)[1],
)
for item in shards:
    name = item["path"].split("/", 1)[1]
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(item["bytes"]).encode("ascii"))
    digest.update(b"\0")
    digest.update(item["sha256"].encode("ascii"))
    digest.update(b"\n")
manifest["nano_composite_sha256"] = digest.hexdigest()
manifest["candidate"] = "production-candidate-3-fhw8"

# canonical release hash (verify_release's exact form)
manifest.pop("release_sha256", None)
release_sha = hashlib.sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
manifest["release_sha256"] = release_sha

(out_dir / "release.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

# stage a hardlink tree and verify with the real bootstrap verifier
stage = Path(tempfile.mkdtemp(prefix="fhw8-release-stage-"))
(stage / "manifests").mkdir()
staged_manifest = stage / "manifests/release.json"
staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
for item in manifest["files"]:
    rel = item["path"]
    src = candidate / rel.split("/", 1)[1] if rel.startswith("nano/") else release_root / rel
    dst = stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
verified = verify_release(stage, release_sha)
payload_bytes = sum(item["bytes"] for item in manifest["files"])
print(json.dumps({
    "old_release_sha256": old_release_sha,
    "new_release_sha256": release_sha,
    "nano_composite_sha256": manifest["nano_composite_sha256"],
    "payload_bytes": payload_bytes,
    "payload_file_count": len(manifest["files"]),
    "verified_with_bootstrap_verifier": bool(verified),
    "changed_nano_files": [e["path"] for e in nano_entries],
    "output": str(out_dir / "release.json"),
    "stage": str(stage),
}, indent=2))
