# Public runtime build graph

This document is the Phase 1 boundary for reconstructing Production Candidate
1 without its opaque local image lineage. The machine-readable identity is
[`config/qualified-candidate-1.json`](../config/qualified-candidate-1.json).

## What is qualified today

The derived Nano and EarTTS artifacts are exact-public: they originate from
`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@fb0f94ea...`, their source and output
payloads are hashed, and each conversion was byte-identical across two runs.
The application and retained Speech patch are checked into this repository.

The qualified image is **not** a permissible release input. Docker reports one
35.1 GB `Imported from -` layer beneath four reviewed patch layers. Historical
records show that layer passed through EA/NGC-derived intermediates. The image
and its final file hashes are retained solely as a behavioral and binary
comparison oracle.

## Public reconstruction graph

| Stage | Immutable public input | Project-owned transformation | Result |
|---|---|---|---|
| CUDA/PyTorch base | `nvcr.io/nvidia/pytorch:25.12-py3@sha256:1dc787...` | None | ARM64 CUDA/PyTorch toolchain |
| Native vLLM | `vllm-project/vllm@b31e9326a7d9...` | Build for `sm_121`; FlashInfer 0.6.5; CUDA 13.0.1 | Qualified native ABI baseline |
| Voicechat vLLM | `vklimkov-nvidia/vllm@237ba3cafc14...` | Overlay/reconcile the public Voicechat Python implementation onto the matching native build | EarTTS, CFG, custom inputs and Nano heads |
| NVIDIA Speech | `NVIDIA-NeMo/Speech@911ec674ab40...` | Apply and byte-check the retained RNNT/streaming patch | Streaming inference wrapper |
| Quantized runtime | Sources above | Apply checked-in W8A32, BF16-head, Marlin K-tail, unused-logit, PAD-pair/control-barrier, codec and server patches | Candidate runtime |
| Typed input | Pocket TTS snapshots pinned in the BOM | Isolated CPU-only worker environment inside the model container | `strict-v3` typed-input PCM injection |
| Application | This repository | Install runtime server in the image; install Pipecat on the host from the locked environment | Foreground clone-to-Playground stack |

Phase 4 must construct every row directly. It must not use the qualified image,
any `nemotron-local/voicechat-vllm:ea-*` tag, the retired EA checkpoint, or an
NGC Voicechat image as a base or copy source.

### Python dependency identity

The public base image is pinned by registry digest, and source-built packages
are pinned by full Git revision. Runtime additions are pinned to exact package
versions, but the upstream vLLM and NeMo dependency graphs do not publish a
complete ARM64 hash lock. The build therefore records the resolved environment
in the deterministic SPDX package inventory and runtime identity report and
treats any resolution change as a new runtime candidate requiring the full
Phase 7 qualification. This is a documented limitation of package-level
reproducibility, not a claim of a byte-reproducible container. The release gate
additionally requires a clean `--no-cache` build and records its image and SBOM
digests. The SBOM generator runs inside the completed image and inventories
both Python distributions and Debian packages; it does not depend on an
optional Docker Desktop plugin.

`UV_INDEX_STRATEGY=unsafe-best-match` is limited to the native vLLM build. It is
required because that pinned build combines the NVIDIA base environment with
the official PyTorch CUDA wheel index; the post-build SBOM is the authority for
the selected distributions.

## Runtime source ownership

The following final vLLM files define the behavioral comparison surface. Their
qualified hashes are in the BOM; their release versions must be produced from
the public forks plus the named project patches.

- `vllm/model_executor/models/eartts.py`
- `vllm/model_executor/models/nemotron_h.py`
- `vllm/model_executor/layers/cfg_ops.py`
- `vllm/v1/cfg_metadata.py`
- `vllm/model_executor/layers/quantization/ea_w8a32.py`
- `vllm/model_executor/layers/ea_bf16_head.py`
- `vllm/model_executor/layers/quantization/gptq_marlin.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`

The historical `ea_w8a32` name is a serialized artifact ABI only. Its source is
project-owned and checkpoint-independent; it does not authorize an EA weight or
image dependency.

## Qualified-package delta escrow

Phase 1 enumerated the complete installed Python package instead of assuming a
handwritten target list was complete. The reproducible capture sequence is:

```bash
container_id=$(docker create --entrypoint /bin/true \
  nemotron-local/voicechat-vllm:public-candidate1-hotfix5-20260803)
docker cp "$container_id:/usr/local/lib/python3.12/dist-packages/vllm" \
  /tmp/qualified-vllm
docker rm "$container_id"

git -C /path/to/vllm-voicechat.git archive \
  237ba3cafc14d514551539ad980f17d40d273dfa vllm | tar -x -C /tmp/voicechat
git -C /path/to/vllm.git archive \
  b31e9326a7d9394aab8c767f8ebe225c65594b60 vllm | tar -x -C /tmp/native

python3 tools/provenance/capture-qualified-vllm-deltas.py \
  --qualified-root /tmp/qualified-vllm \
  --voicechat-root /tmp/voicechat/vllm \
  --native-root /tmp/native/vllm \
  --output tools/provenance/qualified-deltas
```

The capture classified all 966 installed Python files: 922 are byte-identical
to at least one pinned public source, 36 differ, and 8 are added. All 44
differing/added files have per-file unified patches under
`tools/provenance/qualified-deltas/`, and the index records every qualified file,
reference choice, source/final hash, and patch hash. The full installed Python
tree digest is `d66baad763ef...`; the index digest is `8854e6eb87c7...`.

These patches are an escrow and Phase 4 porting worklist, not a blind patch
stack. Phase 4 must classify each delta as public-source selection, generated
build metadata, required project behavior, or unrelated dependency skew, then
port and test only the necessary behavior against the new public image.

## Build-time refusal rules

The public Docker build added in Phase 4 must fail if any of these are true:

- the base image is not the pinned public digest;
- a source checkout is not at its pinned full commit;
- a retained patch does not apply exactly once or its post-image source hashes
  are not recorded;
- an input path or Docker history includes a retired EA tag, an NGC Voicechat
  image, an unqualified local image, or the retired checkpoint;
- model payloads do not match the published release manifest; or
- a runtime dependency is fetched from an unpinned branch, tag, or package
  version.

## Remaining parity work

The public Voicechat fork and the qualified local oracle do not share identical
ancestry. Phase 4 therefore treats the public image as a new runtime candidate:
patches are ported against the public sources, final file deltas are reviewed,
and all component and live qualification gates run before promotion. Matching a
historical file hash is useful evidence but is not a substitute for those gates.
