#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cache_root=${VOICECHAT_BUILD_CACHE:-${repo_root}/.cache/public-build}
native_root=${cache_root}/vllm-native-b31e9326a
native_image=${VOICECHAT_NATIVE_IMAGE:-nemotron-voicechat-public-vllm-native:0.17.1-b31e9326a-fi065}
runtime_image=${VOICECHAT_RUNTIME_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:local}
read -r project_source_sha256 project_payload_sha256 build_recipe_sha256 < <(
  PYTHONPATH="${repo_root}/src" python3 -c \
    'import sys; from pathlib import Path; from nemotron_voicechat_runtime.public_runtime_identity import public_runtime_payload_sha256, public_runtime_recipe_sha256, public_runtime_source_sha256; root=Path(sys.argv[1]); print(public_runtime_source_sha256(root), public_runtime_payload_sha256(root), public_runtime_recipe_sha256(root))' \
    "${repo_root}"
)
recipe_snapshot_root=$(mktemp -d)
cleanup() {
  rm -rf -- "${recipe_snapshot_root}"
}
trap cleanup EXIT
PYTHONPATH="${repo_root}/src" python3 -c \
  'import sys; from pathlib import Path; from nemotron_voicechat_runtime.public_runtime_identity import public_runtime_recipe_snapshot; Path(sys.argv[2]).write_bytes(public_runtime_recipe_snapshot(Path(sys.argv[1])))' \
  "${repo_root}" "${recipe_snapshot_root}/recipe-snapshot.json"
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
  --network host \
  --platform linux/arm64 \
  --file "${repo_root}/container/Dockerfile.public-vllm" \
  --tag "${native_image}" \
  "${native_root}"

DOCKER_BUILDKIT=1 docker build \
  "${build_flags[@]}" \
  --network host \
  --platform linux/arm64 \
  --build-context "voicechat_recipe_snapshot=${recipe_snapshot_root}" \
  --build-arg "NATIVE_IMAGE=${native_image}" \
  --build-arg "VOICECHAT_SOURCE_SHA256=${project_source_sha256}" \
  --build-arg "VOICECHAT_PAYLOAD_SHA256=${project_payload_sha256}" \
  --build-arg "VOICECHAT_RECIPE_SHA256=${build_recipe_sha256}" \
  --file "${repo_root}/container/Dockerfile.public-runtime" \
  --tag "${runtime_image}" \
  "${repo_root}"

exec "${repo_root}/container/audit-public-runtime.sh" \
  "${runtime_image}" "${cache_root}/audit"
