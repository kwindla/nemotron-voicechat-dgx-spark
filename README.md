# Nemotron Voicechat on DGX Spark

Run NVIDIA NemotronLabs VoiceChat 11B on one DGX Spark and use it through a
Pipecat SmallWebRTC bot and Pipecat's built-in Playground.

```bash
git clone https://github.com/pipecat-ai/nemotron-voicechat-dgx-spark.git
cd nemotron-voicechat-dgx-spark
./voicechat bootstrap
./voicechat up
```

Open **http://127.0.0.1:7860/client/**. Stop the foreground stack with Ctrl-C.

The first bootstrap downloads about 65 GiB and builds the CUDA runtime image;
allow 45–120 minutes and at least 90 GiB free. It may require accepting the
[NVIDIA VoiceChat model terms](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
and running `hf auth login`, or supplying `HF_TOKEN` only to the bootstrap
invocation. Runtime operation is local and offline: there is no `.env`, API
key, cloud model, system service, or persisted application credential.

## What ships

- public NVIDIA VoiceChat parent revision `fb0f94e...`;
- published, signed Production Candidate 1 weights at immutable revision
  `a20c685...` in
  [`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`](https://huggingface.co/pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark);
- calibrated GPTQ W8 Nano and W8A32 EarTTS;
- strict-v3 realtime WebSocket inference with RNNT turn taking, native audio,
  function calling, watchdogs, and a server-owned Pocket TTS typed-input bridge;
- a foreground Pipecat SmallWebRTC bot using the installed Playground client.

The GPU/CUDA/model stack runs in Docker. Pipecat runs from the pinned host `uv`
environment. Pocket TTS is internal to the model container in an isolated
CPU-only Python worker, so its PyTorch does not conflict with CUDA PyTorch.

## Useful commands

```bash
./voicechat doctor
./voicechat status
./voicechat test
./voicechat down                 # orphan cleanup after an unclean exit
./voicechat bootstrap --offline # proven cache-only reinstall/revalidation
```

The initial public release includes the inference runtime, downloaded-artifact
verification, component gates, and end-to-end qualification suite. The
deterministic production conversion pipeline and dual-source verification will
be published in a follow-up commit after their long-running reproduction gate
finishes. The converted weights and their signed provenance are already public
at the immutable Hugging Face revision above.

See [deployment](docs/deployment.md), [architecture](docs/architecture.md),
[strict-v3 protocol](docs/protocol-v3.md),
[weight reproduction](docs/weight-reproduction.md),
[qualification](docs/qualification.md), and
[known limitations](docs/known-limitations.md). Maintainers close the
[release checklist](docs/release-checklist.md) before publishing a new image or
weight revision.
