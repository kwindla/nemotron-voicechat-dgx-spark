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

The ordinary setup path downloads this release. The source repository also
contains the deterministic production conversion code, corpus selection,
100 GiB disk guard, dual clean-process conversion check, component gates, and
exact reproduction command.

## Qualification

Production Candidate 1 passed the held-out Nano control-token gate, EarTTS
decode with independent Parakeet transcription, real browser/Pipecat
SmallWebRTC multi-turn tests, and function calling. Its two-frame mean passed
the 80 ms cadence target and its two-frame p95 was a documented near miss near
82 ms. Browser playback uses a small adaptive jitter buffer because delivery
jitter can exceed one 80 ms audio slice.

The current public runtime passed paced 12,000-frame qualification on both the
downloaded artifact and a byte-identical public-source reproduction. Late-block
queue slopes were 0.289 and 0.375 ms/minute, with peak implied playback debt of
382 and 400 ms respectively; the earlier linear EarTTS drift did not recur.
All 18 typed turns in each successful sustained run produced non-silent audio
that passed independent ASR. Sessions close gracefully at the declared model
frame limit.

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
behavior but cannot eliminate model-level errors. One retained source-runtime
campaign also encountered a malformed late function-call recovery that disabled
two-frame drafting and failed the queue-debt gate; a fresh-stack repeat passed.
Typed input is synthesized to audio and is half-duplex with microphone input.
Session length and unified-memory headroom are explicit runtime limits.

The replay metadata field `runtime.image` records the local image label used
when the exact calibration/evaluation tensors were captured. The label is
provenance only, is not a downloadable dependency, and is not used by the
conversion or runtime paths. Reproduction uses the pinned public source graph
and the tensor/file hashes in the release manifest.

See the source repository's known-limitations and provenance documents for the
complete operational record.

This source copy includes qualification completed after immutable weight
revision `a20c68547d27421c1e66bd501bf295569a0cc309` was published. It supersedes
the model-card prose frozen at that revision; the qualified weight bytes and
their signed manifest are unchanged.

## License and attribution

The parent model and these redistributed derivatives are governed by
OpenMDW-1.1; see LICENSE. NVIDIA origin and copyright notices are retained in
THIRD_PARTY_NOTICES.md. This repository identifies the exact parent above and
does not claim NVIDIA sponsorship or endorsement.
