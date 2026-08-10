#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage=${EARTTS_EXACT_CONSTANTS_STAGE:-b}
base_image=${VOICECHAT_BASE_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:production-candidate-1}
output_image=${VOICECHAT_OUTPUT_IMAGE:-nemotron-local/voicechat-vllm:step3-eartts-exact-${stage}}

if [[ "${stage}" != "a" && "${stage}" != "b" ]]; then
  echo "EARTTS_EXACT_CONSTANTS_STAGE must be a or b" >&2
  exit 2
fi

exec docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "EARTTS_EXACT_CONSTANTS_STAGE=${stage}" \
  --file "${repo_root}/container/Dockerfile.eartts-exact-constants" \
  --tag "${output_image}" \
  "${repo_root}"
