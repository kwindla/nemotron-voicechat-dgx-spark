---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
pipeline_tag: audio-to-audio
tags:
  - speech-to-speech
  - nemotron
  - dgx-spark
  - gptq
  - vllm
---

# NVIDIA NemotronLabs VoiceChat 11B — DGX Spark community quantization

This is a community conversion of
nvidia/NVIDIA-NemotronLabs-VoiceChat-11B at immutable parent revision
fb0f94eaf4d03ddc430f39565229393fa1b50c26. It is not an NVIDIA release.
It targets one NVIDIA DGX Spark (GB10, Linux ARM64) and the runtime in
github.com/pipecat-ai/nemotron-voicechat-dgx-spark.

## Contents

- nano/: calibrated symmetric GPTQ W8A16, group 128, for 105 MLP, Mamba, and
  text-head tensors. The function head, embeddings, and attention projections
  remain full precision. K=15,680 down projections are zero-padded to 15,744
  for group-128 Marlin execution.
- eartts/: symmetric per-output-channel W8 weights with FP32 activations and
  accumulation. The qualified runtime configuration uses a 1,500-position
  sliding window.
- calibration/replay.tar: exact fused-embedding calibration/evaluation replays
  in a deterministic archive; calibration/manifest.json inventories and hashes
  every archived file. The corpus contains no raw user recordings.
- manifests/release.json: SHA-256 and byte count for every release file.
- reports/: portable conversion evidence.

The ordinary setup path downloads this release. The initial source release
contains signed-artifact verification and the Nano/EarTTS component gates. The
deterministic production conversion pipeline, corpus selection, dual
clean-process conversion check, and exact reproduction command will be
published in a follow-up commit after the long-running reproduction gate
finishes.

## Qualification

Production Candidate 1 passed the held-out Nano control-token gate, EarTTS
decode with independent Parakeet transcription, real browser/Pipecat
SmallWebRTC multi-turn tests, and function calling. Its two-frame mean passed
the 80 ms cadence target and its two-frame p95 was a documented near miss near
82 ms. Browser playback uses a small adaptive jitter buffer because delivery
jitter can exceed one 80 ms audio slice.

A sustained paced campaign exposed linear EarTTS step-time drift and growing
playback queue lag in long sessions. The weights are therefore published as a
qualified component/browser candidate, not as proof that sustained realtime
operation has passed its release gate. Sustained no-queue-growth qualification
remains a runtime release blocker.

The qualified platform identity is:

- NVIDIA DGX Spark / GB10, compute capability 12.1
- NVIDIA driver 580.142
- Linux kernel 6.17.0-1014-nvidia
- NVIDIA PyTorch 25.12 lineage

Other hardware and software combinations are not qualified by this release.

## Known limitations

The upstream model can self-talk, repeat canned replies, choose or skip the
wrong tool, mis-speak tool results, drop transcript words, or occasionally
produce a silent response. The deployment adds bounded watchdog and recovery
behavior but cannot eliminate model-level errors. Typed input is synthesized
to audio and is half-duplex with microphone input. Session length and unified
memory headroom are explicit runtime limits. Long sessions are not yet
production-qualified because EarTTS latency and the playback queue grow over
time in the current runtime.

The replay metadata field `runtime.image` records the local image label used
when the exact calibration/evaluation tensors were captured. The label is
provenance only, is not a downloadable dependency, and is not used by the
conversion or runtime paths. Reproduction uses the pinned public source graph
and the tensor/file hashes in the release manifest.

See the source repository's known-limitations and provenance documents for the
complete operational record.

## License and attribution

The parent model and these redistributed derivatives are governed by
OpenMDW-1.1; see LICENSE. NVIDIA origin and copyright notices are retained in
THIRD_PARTY_NOTICES.md. This repository identifies the exact parent above and
does not claim NVIDIA sponsorship or endorsement.
