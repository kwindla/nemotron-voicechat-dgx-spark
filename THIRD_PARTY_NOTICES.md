# Third-party notices

This repository is licensed under Apache License 2.0. It interoperates with,
and contains modifications targeting, the following third-party projects.

## NVIDIA NeMo Speech

The inference server imports NVIDIA NeMo Speech at runtime from a separately
cloned checkout pinned to commit
`911ec674ab40f04302ef33672be4179f45a7310f`. NeMo Speech is distributed under
Apache License 2.0. The retained
`nemotron-voicechat-rnnt-turn-taking.patch` is a derivative modification of
those Apache-licensed sources. The upstream copyright and license remain in
effect.

## vLLM and NVIDIA's Voicechat fork

The container overlay scripts modify files in vLLM and NVIDIA's public
Voicechat vLLM fork. vLLM and the referenced fork are distributed under Apache
License 2.0. The authoritative public fork revision used during development was
`237ba3cafc14d514551539ad980f17d40d273dfa` from
`https://github.com/vklimkov-nvidia/vllm.git`.

## Pipecat

The adapter and demo depend on Pipecat pinned to commit
`5415f28af9407937610d68abb2c195365650a862`. Pipecat is not vendored here; its
own license and notices apply when installed.

## NVIDIA model materials

The companion Hugging Face release redistributes community-quantized
derivatives of `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` at revision
`fb0f94eaf4d03ddc430f39565229393fa1b50c26`. The parent and derived model
materials are governed by OpenMDW-1.1. A verbatim copy is retained at
`licenses/OpenMDW-1.1.txt` and published as the model release's `LICENSE`.
NVIDIA is the origin of the parent model. This project and its derived release
are community work and are not NVIDIA releases.

The published fused-embedding calibration replay contains outputs of the parent
model and project-authored Apache-2.0 prompt fixtures. It contains no raw user
recordings.

## Qualified container image

The exact locally qualified image is derived from an EA/NGC image lineage and
is not distributed by this repository. Its presence on one host does not grant
redistribution rights. The consolidated Dockerfile is source for an overlay;
its base image remains an explicit external prerequisite.
