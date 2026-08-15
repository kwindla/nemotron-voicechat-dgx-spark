#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_image=${VOICECHAT_RUNTIME_IMAGE:-pipecat-ai/nemotron-voicechat-dgx-spark:local}
diagnostic_image=${VOICECHAT_TOOL_LATENCY_IMAGE:-nemotron-voicechat-tool-latency-benchmark:local}
installer=${repo_root}/tools/benchmark/install_vllm_tool_latency_ledger.py
installer_sha256=$(sha256sum "${installer}" | awk '{print $1}')
runtime_id=$(docker image inspect --format '{{.Id}}' "${runtime_image}")
runtime_tag="nemotron-voicechat-tool-latency-base:${runtime_id#sha256:}"
docker tag "${runtime_id}" "${runtime_tag}"

DOCKER_BUILDKIT=1 docker build \
  --network none \
  --build-arg "RUNTIME_IMAGE=${runtime_tag}" \
  --build-arg "RUNTIME_IMAGE_ID=${runtime_id}" \
  --build-arg "BENCHMARK_PATCH_SHA256=${installer_sha256}" \
  --file "${repo_root}/container/Dockerfile.tool-latency-benchmark" \
  --tag "${diagnostic_image}" \
  "${repo_root}"

diagnostic_id=$(docker image inspect --format '{{.Id}}' "${diagnostic_image}")
test "$(docker image inspect --format '{{index .Config.Labels "ai.pipecat.voicechat.benchmark-base-image-id"}}' "${diagnostic_id}")" = "${runtime_id}"
test "$(docker image inspect --format '{{index .Config.Labels "ai.pipecat.voicechat.benchmark-patch-sha256"}}' "${diagnostic_id}")" = "${installer_sha256}"
printf '%s\n' "${diagnostic_id}"
