#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cache_root=${VOICECHAT_BUILD_CACHE:-${repo_root}/.cache/public-build}
native_root=${cache_root}/vllm-native-b31e9326a
native_image=${VOICECHAT_NATIVE_IMAGE:-nemotron-voicechat-public-vllm-native:0.17.1-b31e9326a-fi065}
runtime_image=${VOICECHAT_RUNTIME_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:local}
native_revision=b31e9326a7d9394aab8c767f8ebe225c65594b60

mkdir -p "${cache_root}"
build_flags=()
if [[ "${VOICECHAT_BUILD_NO_CACHE:-0}" == "1" ]]; then
  build_flags+=(--no-cache)
fi
if [[ ! -d "${native_root}/.git" ]]; then
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "${native_root}"
fi
git -C "${native_root}" fetch origin "${native_revision}"
git -C "${native_root}" checkout --detach "${native_revision}"
test "$(git -C "${native_root}" rev-parse HEAD)" = "${native_revision}"

DOCKER_BUILDKIT=1 docker build \
  "${build_flags[@]}" \
  --platform linux/arm64 \
  --file "${repo_root}/container/Dockerfile.public-vllm" \
  --tag "${native_image}" \
  "${native_root}"

DOCKER_BUILDKIT=1 docker build \
  "${build_flags[@]}" \
  --platform linux/arm64 \
  --build-arg "NATIVE_IMAGE=${native_image}" \
  --file "${repo_root}/container/Dockerfile.public-runtime" \
  --tag "${runtime_image}" \
  "${repo_root}"

exec "${repo_root}/container/audit-public-runtime.sh" \
  "${runtime_image}" "${cache_root}/audit"
