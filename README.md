# Nemotron Voicechat on DGX Spark

[NVIDIA NemotronLabs VoiceChat](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
is an end-to-end, speech-to-speech, full-duplex model for conversational AI.

This is the first open source, full-duplex model to support tool
calling. NVIDIA released the model as a research-oriented "Labs" checkpoint.

This repository runs VoiceChat 11B on one DGX Spark. To sustain real-time
inference on Spark, we quantized the Nano and EarTTS weights, patched vLLM, and
implemented conditional two-frame PAD drafting.

## Getting started

```bash
git clone https://github.com/pipecat-ai/nemotron-voicechat-dgx-spark.git
cd nemotron-voicechat-dgx-spark
./voicechat bootstrap
./voicechat up
```

The first bootstrap downloads about 65 GiB and builds the CUDA runtime image.
Allow 1–2 hours and at least 90 GiB of free space. You may need to accept the
[NVIDIA model terms](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
and run `hf auth login`, or provide `HF_TOKEN` only to the bootstrap command.
No credential is copied into the image or required at runtime.
Runtime is local and offline, once all the model weights and dependencies are downloaded.
The separate English Nemotron ASR evaluator is qualification-only and is built
on demand by `./voicechat test --live` or source conversion, not by normal serving bootstrap.

`./voicechat up` takes about seven minutes to start. You will see the
`PIPECAT DEVELOPMENT RUNNER` banner when the full stack is ready.

Open **http://127.0.0.1:7860/client/**.

The client uses Pipecat's
[SmallWebRTCTransport](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc)
for a low-latency peer-to-peer connection between the browser and the host
Pipecat bot. The bot talks to the Docker inference server over a separate,
loopback-only WebSocket.

Remote browser testing needs an HTTPS origin because browsers do not grant
microphone access to insecure origins other than localhost or loopback. You can use ngrok for this.

```bash
# Run in another terminal.
ngrok http http://127.0.0.1:7860
```

Open the HTTPS URL printed by ngrok. The tunnel exposes the Pipecat Playground UI and a
WebRTC signaling endpoint. If you are behind a restrictive firewall, you may need to use a different Pipecat transport (WebSocket, or commercial WebRTC cloud).

Stop the foreground stack with Ctrl-C.

## What's in this repo

This repository includes the inference runtime, downloaded-artifact
verification, component gates, end-to-end qualification suite, Pipecat service,
and sample bot in
[`demo.py`](src/nemotron_voicechat_pipecat/demo.py).

The deterministic production conversion pipeline is included for developers
who want to reproduce or extend the quantization work. The converted weights
and their content-addressed provenance are
[public on Hugging Face](https://huggingface.co/pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark),
so ordinary installations do not run the conversion.

The current release contains:

- the public NVIDIA VoiceChat parent revision `fb0f94e...`;
- published, hash-verified Production Candidate 1 weights at immutable revision
  `a20c685...` in
  [`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`](https://huggingface.co/pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark);
- calibrated GPTQ W8 Nano and W8A32 EarTTS;
- strict-v3 realtime WebSocket inference with Pipecat Smart Turn endpointing,
  explicit client commits, RNNT transcription, native audio, function calling,
  watchdogs, and a server-owned Pocket TTS typed-input bridge;
- a foreground Pipecat SmallWebRTC bot using the installed Playground client.

The GPU/CUDA/model stack runs in Docker. Pipecat runs from the pinned host `uv`
environment. Pocket TTS is internal to the model container in an isolated
CPU-only Python worker, so its PyTorch does not conflict with CUDA PyTorch.

## Todo

- Qualify Smart Turn latency, false endpoints, and host CPU contention against
  the captured RNNT-owned-turn baseline.
- Check input text transcription chunking. We may be inserting spaces in RTVI
  messages where we should not.
- Improve inference stack startup time.
- Experiment with application-layer tool routing instead of relying exclusively
  on the model's function head.
- Add an example for using VoiceChat as the front-end to NemoClaw.
- Benchmark context degradation in long conversations and investigate safe
  session rollover or compaction strategies.

## Known limitations

- NVIDIA trained the model with audio context windows no longer than two
  minutes; conversational context beyond that window may not be retained
  reliably.
- NVIDIA classifies the checkpoint as research-only. Knowledge, reasoning,
  transcript quality, and tool selection can be unreliable. See
  [known limitations](docs/known-limitations.md) for measured runtime behavior
  and upstream model limitations.

## Editing the bot (system instruction, tools list, etc)

The system prompt, tool schemas, and handlers live in
[`demo.py`](src/nemotron_voicechat_pipecat/demo.py). Keep the loaded model
resident while restarting only Pipecat from another terminal:

```bash
./voicechat restart-bot
```

For automatic restarts whenever a Pipecat Python file changes:

```bash
./voicechat up --reload-bot
```

Bot restarts close the current WebRTC session, so reconnect the Playground.
Changes apply to new sessions and require neither bootstrap nor an image build.

## Useful commands

```bash
./voicechat doctor
./voicechat status
./voicechat test
./voicechat down                 # orphan cleanup after an unclean exit
./voicechat bootstrap --offline # proven cache-only reinstall/revalidation
```

Developers reproducing the conversion can run:

```bash
./voicechat bootstrap --convert-from-source
```

That path uses the exact published calibration/evaluation corpus, performs two
conversions in separate processes, compares every output byte with the other
run and Production Candidate 1, and runs the Nano and EarTTS component gates.
After every check passes, bootstrap verifies the complete signed release and
transactionally activates the locally reproduced Nano and EarTTS files. Subsequent
`up` and `test --live` commands therefore exercise those local files. The
conversion refuses to start if projected output would leave less than 100 GiB
free.

## Docs and links

See [deployment](docs/deployment.md), [architecture](docs/architecture.md),
[strict-v3 protocol](docs/protocol-v3.md),
[weight reproduction](docs/weight-reproduction.md),
[qualification](docs/qualification.md), and
[known limitations](docs/known-limitations.md). Maintainers close the
[release checklist](docs/release-checklist.md) before publishing a new image or
weight revision.
