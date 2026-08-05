#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_image=${VOICECHAT_BASE_IMAGE:-nemotron-local/voicechat-vllm:public-runtime-flat-f3e41ccc}
output_image=${VOICECHAT_OUTPUT_IMAGE:-nemotron-local/voicechat-vllm:standalone-production-candidate1}

exec docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --file "${repo_root}/container/Dockerfile.runtime-overlay" \
  --tag "${output_image}" \
  "${repo_root}"
