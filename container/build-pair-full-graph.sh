#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_image=${VOICECHAT_BASE_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:production-candidate-1}
output_image=${VOICECHAT_OUTPUT_IMAGE:-nemotron-local/voicechat-vllm:step7-pair-full-graph}

exec docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --file "${repo_root}/container/Dockerfile.pair-full-graph" \
  --tag "${output_image}" \
  "${repo_root}"
