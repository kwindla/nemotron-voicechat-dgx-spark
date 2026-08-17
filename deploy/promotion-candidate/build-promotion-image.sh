#!/usr/bin/env bash
# Build the promotion-candidate runtime image from the CURRENT repository
# source, so the qualified runtime code (tool-call watchdog, pre-EOU
# settlement recovery, fence-on-real-audio + early arming) is baked in.
#
# Produces the canonical tag the start scripts default to. Override with
# VOICECHAT_PROMOTION_IMAGE. Requires the native vLLM base per
# docs/provenance.md; container/build-public-runtime.sh builds/uses it.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
image=${VOICECHAT_PROMOTION_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:promotion-candidate}

VOICECHAT_RUNTIME_IMAGE="$image" bash "${repo_root}/container/build-public-runtime.sh"

# attest that the built image carries this working tree's runtime source
expected=$(sha256sum "${repo_root}/src/nemotron_voicechat_runtime/server.py" | awk '{print $1}')
actual=$(docker run --rm --entrypoint sha256sum "$image" \
  /usr/local/lib/python3.12/dist-packages/nemotron_voicechat_runtime/server.py | awk '{print $1}')
if [ "$expected" != "$actual" ]; then
  echo "BUILT IMAGE DOES NOT CARRY THIS SOURCE: $expected != $actual" >&2
  exit 1
fi
echo "built and attested: $image (server.py $expected)"
