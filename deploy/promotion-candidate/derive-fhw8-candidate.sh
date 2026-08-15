#!/usr/bin/env bash
# Derive the qualified fhw8 candidate Nano from the bootstrapped release
# artifact, deterministically, and verify it byte-for-byte against the
# committed expected hashes. Safe to re-run; fails closed on any drift.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cache=${NEMOTRON_VOICECHAT_CACHE:-$HOME/.cache/nemotron-voicechat}
release="$cache/artifacts/release/nano"
release_manifest="$cache/artifacts/release/manifests/release.json"
candidates="$cache/artifacts/candidates"
candidate="$candidates/nano-fhw8"
proof=$(mktemp -d "${TMPDIR:-/tmp}/fhw8-rebuild-proof.XXXXXX")
trap 'rm -rf "$proof"' EXIT
python=${VOICECHAT_PYTHON:-"$repo_root/.venv/bin/python"}
[ -x "$python" ] || python=python3

[ -d "$release" ] || { echo "release nano missing: run bootstrap first" >&2; exit 1; }
mkdir -p "$candidates"

if [ -d "$candidate" ]; then
  echo "candidate exists; verifying against committed pins..."
else
  "$python" "$repo_root/tools/benchmark/build_fhw8_candidate.py" \
    --release-root "$release" --candidate-root "$candidate" \
    --report "$proof/build-report-1.json"
fi
# reproducibility proof: independent second build must be byte-identical
"$python" "$repo_root/tools/benchmark/build_fhw8_candidate.py" \
  --release-root "$release" --candidate-root "$proof/rebuild" \
  --report "$proof/build-report-2.json"
"$python" "$repo_root/tools/benchmark/build_fhw8_manifest.py" \
  --candidate-root "$candidate" --rebuild-root "$proof/rebuild" \
  --release-manifest "$release_manifest" \
  --build-report "$proof/build-report-2.json" \
  --output "$candidates/fhw8-candidate-manifest.json" \
  --evidence "$proof/double-build-evidence.json"
# fail closed against the committed qualified pins
"$python" - "$candidate" "$repo_root/deploy/promotion-candidate" <<'PY'
import json, sys
from pathlib import Path
candidate, pins = Path(sys.argv[1]), Path(sys.argv[2])
expected = json.loads((pins / "fhw8-artifact-expected.sha256.json").read_text())
actual = json.loads((candidate / "artifact.sha256.json").read_text())
if expected != actual:
    diff = {k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}
    raise SystemExit(f"DERIVED CANDIDATE DOES NOT MATCH QUALIFIED PINS: {sorted(diff)}")
committed = json.loads((pins / "fhw8-candidate-manifest.json").read_text())
built = json.loads((candidate.parent / "fhw8-candidate-manifest.json").read_text())
if committed["components"]["nano"]["model"]["sha256"] != built["components"]["nano"]["model"]["sha256"]:
    raise SystemExit("DERIVED MANIFEST COMPOSITE DOES NOT MATCH QUALIFIED PIN")
print("fhw8 candidate verified against qualified pins:",
      built["components"]["nano"]["model"]["sha256"][:16], "...")
PY
echo "fhw8 candidate ready at $candidate"
