#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${VOICECHAT_ASR_EVALUATOR_IMAGE:-pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b}
build_args=()
if [[ ${VOICECHAT_BUILD_NO_CACHE:-0} == 1 ]]; then
  build_args+=(--no-cache)
fi

DOCKER_BUILDKIT=1 docker build \
  "${build_args[@]}" \
  --network host \
  --platform linux/arm64 \
  --file "${repo_root}/container/Dockerfile.asr-evaluator" \
  --tag "${image}" \
  "${repo_root}"

image_id=$(docker image inspect --format '{{.Id}}' "${image}")
"${repo_root}/container/audit-asr-evaluator.sh" "${image_id}"
