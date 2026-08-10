# Reproducible clone-to-Playground release plan

Status: implementation-ready; final Fable verdict APPROVE (2026-08-04)

## Outcome

A developer on a qualified DGX Spark can clone this repository and run:

```bash
./voicechat bootstrap
./voicechat up
```

The first command validates the host, downloads pinned public inputs and the
qualified derived weights, verifies every payload, builds the Voicechat model
container (including its internal CPU Pocket TTS bridge), and installs Pipecat
on the host with `uv`. The second command starts the model container and host
Pipecat process from the command line, waits for readiness, and prints the
Pipecat Playground URL.
`Ctrl-C` stops the entire stack. No `.env`, user systemd unit, pre-existing
local image, hand-created checkout, or undocumented artifact is required.

An expert can instead reproduce the published weights with:

```bash
./voicechat bootstrap --convert-from-source
```

That path runs the checked-in calibration and conversion recipes twice,
requires byte-identical outputs, runs the component gates, and verifies the
result against the published release manifest.

The initial support target is one DGX Spark with the qualified driver/kernel
contract. Other machines are rejected with a useful diagnostic unless they
have an explicit, separately qualified platform profile.

## Release principles

- All source repositories, Python packages, base images, model snapshots, and
  conversion inputs are pinned by immutable revision and checksum.
- The public NVIDIA Voicechat checkpoint is the only source checkpoint. No EA
  artifact or EA-derived image layer may enter the build graph.
- The normal installation path downloads qualified derived weights. Source
  conversion is a reproducibility and release path, not a tax on every user.
- Large weights live in a content-addressed host cache mounted read-only at
  runtime; they are not copied into Docker image layers.
- Runtime defaults are checked-in configuration. CLI flags are reserved for
  deployment values such as bind address, port, cache directory, and trace
  directory.
- Hugging Face credentials, if the upstream or derived repository is gated,
  are accepted only for the invoking download/upload command and are never
  written to a project `.env` or Docker layer.
- Voicechat runs in one GPU container with CUDA PyTorch and serves its WebSocket
  on loopback. Its typed-input bridge runs Pocket TTS in an isolated CPU worker
  environment inside that container. Pipecat runs in the host `uv` environment
  without Pocket TTS or PyTorch. Both processes are started and supervised by
  foreground command-line tooling; no systemd integration is installed or
  invoked.
- A release is not reproducible merely because it builds: artifact identity,
  component correctness, browser operation, tool calling, and sustained
  realtime behavior are all release gates.

## Phase 1: freeze and inventory the qualified candidate

1. Export a machine-readable bill of materials from Production Candidate 1:
   public checkpoint revision and file hashes, Nano and EarTTS tensor
   inventories, quantization reports, NVIDIA Speech and vLLM revisions,
   retained patches, Python/CUDA packages, driver/kernel, Pocket TTS assets,
   and runtime settings.
2. Trace every layer of the qualified local image. Classify each file as
   upstream public source, generated artifact, retained project patch, or
   prohibited private/EA lineage.
3. Rebuild a clean allowlist of the files the runtime actually imports. Make
   the build fail if a necessary file has no public source or checked-in patch.
4. Preserve the current qualified hashes as comparison evidence, not as inputs
   to the new public build.

Exit gate: the complete public build graph is documented and contains no
unresolved local-image dependency.

## Phase 2: promote deterministic artifact tooling

1. Move the successful public-checkpoint tooling out of the research tree and
   into `tools/conversion/`, retaining history where practical:
   - fused-embedding Nano capture and corpus selection;
   - calibrated GPTQ W8, group-128 conversion for all 105 selected Nano
     tensors, with function head and embeddings full precision;
   - EarTTS per-output-channel W8/FP32-activation conversion;
   - safetensors sharding, manifest composition, hashing, and disk preflight;
   - component replay and first-divergence reports.
2. Remove rejected W4/AWQ/RTN experiments and EA terminology except the
   serialized `ea_w8a32` ABI name that the qualified artifact requires. Explain
   that exception in code and provenance.
3. Package a redistributable, disjoint calibration/evaluation corpus. Prefer
   deterministic fused-embedding replay tensors and metadata over raw user
   recordings. Record prompt/fixture provenance and license.
   Decision fork: if the exact Production Candidate 1 calibration corpus is
   redistributable, ship it and the release weights remain Production Candidate
   1, so current qualification evidence carries. If any calibration input must
   be substituted, the output is a new candidate: its hashes need not match
   Production Candidate 1, the old hashes become comparison-only evidence, and
   the full Phase 7 ladder, including the 15-minute sustained qualification,
   must pass on the new weights before publication. Generate the release
   manifest from the candidate actually published; never copy it from the old
   candidate.
4. Put every conversion setting in a versioned recipe: source hashes, seed,
   dtype, tensor allowlist, group size, padding rule, damping, shard size,
   library versions, minimum 100 GiB post-conversion disk headroom, and target
   hardware profile.
5. Make conversion immutable and resumable only at validated stage boundaries.
   Never overwrite an output directory.
6. Run two clean conversions and require byte-identical payloads and manifests.
   If a library introduces nondeterministic metadata, canonicalize that metadata
   or fail explicitly; do not weaken the weight comparison.
7. Add unit tests for tensor selection, group packing, K-tail padding,
   full-precision exclusions, corpus disjointness, disk refusal, manifests, and
   corrupt/incomplete artifact rejection.

Exit gate: one checked-in command recreates both derived checkpoints from the
pinned public snapshot, and two runs match each other and the release manifest.

## Phase 3: publish a versioned Hugging Face artifact release

1. Satisfy the public checkpoint's OpenMDW-1.1 distribution obligations: ship
   the license text and NVIDIA copyright/origin notices, identify
   `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@fb0f94eaf4d03ddc430f39565229393fa1b50c26`
   as the parent and Hugging Face `base_model`, and label this a community
   quantization rather than an NVIDIA release. Record the conclusion in
   `docs/provenance.md` and `THIRD_PARTY_NOTICES.md`. Resolve the calibration
   corpus license before upload. Do not republish the Nano skeleton or Pocket
   TTS assets; bootstrap obtains them from their own repositories.
2. Publish one versioned model repository at
   `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` containing:

   ```text
   nano/
   eartts/
   manifests/release.json
   reports/nano-conversion.json
   reports/eartts-conversion.json
   calibration/manifest.json
   README.md
   LICENSE / notices required by upstream
   ```

3. Write a model card that names the exact public parent revision, conversion
   method, full-precision exclusions, DGX Spark target, runtime commit, quality
   gates, known limitations, and reproduction command.
4. Add `tools/release/upload-hf` with dry-run inventory, repository/revision
   allowlists, LFS checks, payload hashes, and post-upload read-back
   verification. Authentication comes from the process environment or the
   user's Hugging Face CLI session and is never persisted by this repository.
5. Pin downloads to the immutable Hugging Face commit produced by the release,
   never to `main`. Check every downloaded file against `release.json`.

Exit gate: a fresh cache can download the release at an immutable revision and
produce the exact qualified artifact inventory.

## Phase 4: build a public, reproducible runtime image

1. Replace the private local base with a public NVIDIA CUDA/PyTorch base pinned
   by registry digest and compatible with DGX Spark ARM64, CUDA, and driver
   constraints.
2. Build the NVIDIA Voicechat vLLM fork from its pinned public commit. Apply
   only reviewed, checked-in patches for the qualified public runtime:
   EarTTS/CFG reconciliation, W8A32 registration, GPTQ/Marlin group-128 tail
   support, conditional PAD-pair scheduling and control barriers, delta output,
   CPU codec path, RNNT turn taking, strict-v3 server behavior, and production
   watchdogs.
3. Clone NVIDIA Speech at the pinned commit during the image build, apply the
   retained patch, and byte-verify its expected diff. Eliminate the external
   `prepare-speech.sh` prerequisite.
4. Install the Voicechat GPU inference server and CUDA dependencies in the main
   container environment. Build a second, isolated Python environment in the
   same image for Pocket TTS and its CPU-only PyTorch wheel. Run Pocket TTS as a
   prewarmed CPU worker over a narrow local IPC contract so it never imports its
   PyTorch build into the CUDA server process. Pin and hash both environments,
   validate the Pocket model snapshot and voice state, and test that synthesis
   remains CPU-bound. Use a length-prefixed JSON or flat-binary Unix-socket
   protocol, never pickle. Scrub CUDA and Python library-path variables from the
   worker environment, and assign it CPU cores that exclude the latency-critical
   codec-worker cores. Detect worker death, return a typed-input error without
   damaging the voice path, and either restart/re-prewarm the worker or mark the
   typed-input health capability degraded explicitly.
5. Download the small Pocket TTS and tokenizer assets during `bootstrap` into
   the same verified host cache as the Voicechat artifacts and mount those exact
   files read-only into the model container. Run all services in Hugging
   Face/Transformers offline mode after bootstrap.
6. Keep `VLLM_ALLOW_INSECURE_SERIALIZATION` unset. The qualified runtime handles
   Voicechat custom inputs without enabling insecure serialization, and the
   public rebuild must prove the same behavior with an EarTTS custom-input
   prefill smoke test. Treat any apparent need for the variable as a runtime
   parity defect to diagnose, not as a supported workaround.
7. Generate an SBOM and provenance attestation. Audit `docker history`, image
   contents, package sources, retired EA image names, and NGC registry paths to
   prove that no private/EA layer or credential is present.
8. Build with BuildKit cache mounts. If a gated input needs a token, expose it
   only as a BuildKit secret or, preferably, download it before the build into
   the verified host cache so it never enters image history.

Exit gate: `docker build --no-cache` succeeds from public inputs on a clean DGX
Spark, and the resulting runtime passes artifact and source-diff validation.

## Phase 5: replace `.env` and systemd with checked-in configuration and a CLI

1. Replace `.env.example` and `check-environment.py` with:
   - `config/production-candidate-1.toml`, containing the frozen model,
     turn-taking, watchdog, session, and performance contract;
   - a small CLI schema for cache/trace roots, ports, bind address, and optional
     tunnel settings;
   - `./voicechat doctor`, which reports host, GPU, driver, kernel, Docker,
     disk, ports, artifacts, and expected memory headroom.
   Pin the host Python interpreter with `uv python pin`; `doctor` verifies that
   version and the completeness of the host wheel cache. The launcher translates
   the checked-in TOML contract into explicit child-process environment values
   and Docker `-e` flags. Removing `.env` means eliminating dotenv files, not
   changing the already-qualified environment-variable interfaces gratuitously.
2. Move the typed-input bridge behind the Voicechat WebSocket boundary and name
   the resulting single supported contract `strict-v3`, advertised by `/health`;
   do not retain a v2 mode. The typed-input event family comprises: request with
   bounded text and client job ID, accept/reject with reason,
   `injection_started`, and `injection_finished` with disposition `completed`,
   `cancelled`, `replaced`, or `error`. The model server asks its Pocket worker
   for PCM and feeds it through the same RNNT-owned input path as microphone
   audio.

   The server is the sole half-duplex authority: it gates incoming microphone
   PCM from typed-job acceptance until synthetic injection and trailing-silence
   closure complete; client ducking is advisory UI state only. Port the
   qualified closure rules unchanged: derive trailing silence from the RNNT EOU
   profile and margin, emit it whenever any synthetic audio was injected,
   including cancellation/replacement, and finish closure before replacement
   audio starts. Serialize typed jobs; a new request during an active job uses
   cancel-then-replace.

   Tag typed-turn transcription events with source and client job ID. Pipecat
   commits the byte-exact typed text to context exactly once and does not also
   commit the RNNT transcript for that turn. Preserve interruption and explicit
   error events. The server must fail health readiness if the isolated Pocket
   worker cannot load and complete its initial prewarm synthesis.
3. Remove the loopback API key under an explicit auth-disabled local mode. This
   is a runtime change, so it passes all runtime-parity and live gates. The model
   container publishes its WebSocket only on host loopback; only Pipecat is
   exposed to the browser or LAN.
4. Replace `deploy/stack` and all `systemd-run`/`systemctl` paths with a
   foreground launcher. It starts the Docker model container, tails useful
   startup status, waits for `/health`, starts Pipecat from the host `uv`
   environment, prints the Playground URL, forwards signals, removes the model
   container on exit, and returns nonzero if either child exits. Use `docker run
   --rm` and a deterministic container name; detect stale containers clearly,
   and make `down` remove a stale container after an unclean launcher exit.
5. Provide these stable commands:

   ```text
   ./voicechat bootstrap [--convert-from-source]
   ./voicechat doctor
   ./voicechat up [--host 127.0.0.1] [--port 7860]
   ./voicechat test [--live]
   ./voicechat down        # only for orphan cleanup; normal stop is Ctrl-C
   ./voicechat release ... # maintainer-only conversion/upload workflow
   ```

6. Store mutable data only beneath one documented cache root and one trace
   root. Do not write into the Git checkout during normal use.
7. Keep ngrok optional and manually invoked. It must target the Pipecat origin,
   never expose the raw model WebSocket, and must not be required for LAN/local
   Playground use.

Exit gate: there is no `.env` parsing, persisted runtime credential, or systemd
reference in executable deployment code. Hugging Face credentials exist only
for an invoking download/upload process; optional tunnel authentication remains
the user's external tool configuration.

## Phase 6: make bootstrap safe, resumable, and understandable

1. Implement `./voicechat bootstrap` as an idempotent state machine:
   preflight, artifact download, checksum verification, `uv sync --frozen`,
   model-image build, image inspection, and smoke test. A rerun skips only
   stages whose content hashes still match.
2. Print disk/download/build estimates before work begins. Preserve the 100 GiB
   conversion floor and add separate minimums for normal download/build.
3. Use temporary paths plus atomic rename for downloads. Reject partial LFS
   pointers, wrong snapshot revisions, unexpected files, and incompatible
   manifests.
4. Support `--offline` after a successful bootstrap and prove that it makes no
   network request, including a successful `uv sync --frozen --offline` from
   the bootstrap-populated aarch64 wheel cache. The runtime itself always
   operates offline.
5. Provide actionable failures, including the exact missing prerequisite and
   the retry command. Detect a gated Hugging Face 401/403, print the exact terms
   acceptance URL and retry command, and never turn a missing model into a late
   engine hang.

Exit gate: interrupting bootstrap at every stage and rerunning it yields the
same verified final state.

## Phase 7: test the complete developer journey

1. Run source-level unit and contract tests on ordinary CI: strict-v3 protocol,
   Pipecat lifecycle, the CPU-only Pocket worker and IPC contract, Pipecat's
   typed-input wire adapter, function calls, malformed results/timeouts, turn
   IDs, barge-in, artifact validation, CLI parsing, and process teardown. Move
   the qualified host bridge's eight invariants into worker/IPC, Pipecat contract,
   and DGX integration tiers; remove host `pocket-tts`, CPU-PyTorch, and SciPy
   dependencies and synthesis code from Pipecat. Mark the former host-bridge
   live report as evidence for the retired architecture. GPU/model integration
   remains a DGX-only gate.
2. On a clean DGX Spark cache, run the documented commands exactly as written.
   No undeclared file from the development machine may be mounted or copied.
3. Exercise component gates before live E2E:
   - Nano held-out replay, control-token margins, and function head;
   - EarTTS decode plus independent Parakeet transcription;
   - Pocket TTS prewarm and real-time factor measured inside the isolated
     container worker;
   - model health and cold-start-to-ready measurement.
4. Run a real Chromium test through Pipecat's built-in Playground and
   SmallWebRTC transport. Cover microphone speech, typed input, mixed modality,
   exact spoken tool results, interruption, silence/noise, multi-turn memory,
   adaptive playback buffering, and clean shutdown.
   Assert separately that the raw model port is bound only to `127.0.0.1` and
   is unreachable through a non-loopback host interface.
5. Run the retained multi-turn network client with independent input synthesis
   and output ASR, followed by the 15-minute paced qualification. Require no
   unbounded queue growth, graceful session limit, expected silent-turn base
   rate, and retained full traces. Interleave typed-input turns throughout so
   Pocket synthesis, CPU affinity, codec work, and live 80 ms pacing are tested
   together in the shipped container configuration.
6. Re-run the live release suite against both (a) downloaded published weights
   and (b) locally converted weights. Their component outputs and qualification
   verdicts must agree.
7. Add a restart-cycle gate to `./voicechat test --live`: reach health, send
   SIGINT to `./voicechat up`, wait for full teardown, restart immediately, and
   require health within the cold-start budget for three consecutive cycles.
   Also kill the launcher once with SIGKILL and require `./voicechat down` to
   cleanly recover the stale container before the next successful start.
8. Add a release checklist and, when available, a self-hosted DGX Spark CI job.
   Standard hosted CI must still test everything that does not require the GPU
   or gated multi-gigabyte artifacts.

Exit gate: an uninvolved developer follows the README from a fresh clone to a
working Playground, and a second developer reproduces the weights from source.

## Phase 8: rewrite the public documentation around the supported path

1. Make the README's first screen the two-command quick start, expected disk
   and startup time, supported hardware, and final Playground URL.
2. Document cache layout, CLI reference, manual foreground lifecycle, LAN/tunnel
   exposure, artifact provenance, weight reproduction, release publication,
   offline operation, logs, and troubleshooting.
3. Replace current statements that conversion tooling and the runtime image are
   external prerequisites. Remove `.env`, systemd, and manual Speech checkout
   instructions.
4. Retain known model limitations honestly, distinguishing upstream model
   behavior from deployment defects.

## Suggested implementation order

1. Record the resolved release topology and Hugging Face destination in the
   release manifest.
2. Freeze the current candidate inventory and import conversion tooling.
3. Publish a private/staging weight release and validate download-by-revision.
4. Reconstruct and qualify the public runtime image.
5. Introduce the foreground CLI and checked-in configuration.
6. Close component, browser, and sustained gates.
7. Publish the final weights and rewrite the README.
8. Have a clean-machine developer repeat the entire procedure without access to
   either source worktree or the old local Docker image.

## Resolved product decisions

1. **Hugging Face destination.** Publish the derived Nano and EarTTS weights to
   `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`, subject to the
   redistribution gate. Use immutable release revisions in bootstrap.
2. **Runtime topology.** Run Voicechat GPU inference, CUDA PyTorch, and the
   internal Pocket TTS typed-input bridge in one Docker container. Pocket TTS
   runs in an isolated CPU-only worker environment inside the image. Install and
   run only Pipecat on the host with `uv`. One foreground CLI owns both
   lifecycles; systemd and Docker Compose are unnecessary.
3. **Artifact paths.** Ordinary developers download the published, qualified
   converted weights. The repository also ships every calibration, conversion,
   verification, and manifest script needed to reproduce those weights from
   the pinned public NVIDIA checkpoint. `--convert-from-source` is an explicit
   maintainer/research path, not the default installation path.
