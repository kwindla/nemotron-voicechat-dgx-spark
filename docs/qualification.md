# Qualification

The ordinary developer path ends when the Playground is working. The release
qualification path is intentionally longer and runs only on the qualified DGX
Spark.

## Source and contract tests

```bash
./voicechat test
```

This runs the locked host suite without a GPU model. It covers strict-v3 event
validation, Pipecat lifecycle and function calls, typed-input invariants, the
Pocket worker IPC contract, artifact validation, and foreground teardown.
The four PyTorch-dependent conversion tensor modules/tests are visible skips in this
torch-free host environment; the full DGX command executes those same modules
offline inside the pinned public runtime image before any GPU component gate.

## Full DGX gate

Obtain the pinned Parakeet/Nemotron streaming ASR checkpoint, install Chromium
for Playwright once, and run:

```bash
uv run playwright install chromium
./voicechat test --live --asr-model /absolute/path/to/asr.nemo
```

No speech fixture is required. The command uses the in-container CPU-only
Pocket worker to create a three-turn 16 kHz conversation plus a browser
microphone fixture. It then runs, in order:

1. torch-dependent component-gate contracts inside the public image;
2. Nano's complete 1,172-call held-out component replay;
3. EarTTS eager and CUDA-graph decode with independent ASR;
4. Pocket prewarm, CPU affinity, and faster-than-realtime synthesis;
5. three clean SIGINT cycles and one SIGKILL/down/restart recovery cycle;
6. Chromium through Pipecat Playground and SmallWebRTC, including typed input,
   real microphone input, audio playback, and cross-modality memory;
7. the retained strict-v3 multi-turn network client with input/output ASR;
8. a paced run longer than 15 minutes, with typed turns every minute, queue
   growth checks, silent-turn accounting, and graceful 12,000-frame closure;
9. offline ASR of every retained assistant response.

The raw model port is checked from every discovered non-loopback IPv4 address;
an HTTP error still counts as exposure and fails the gate. Every subprocess has
a deadline, every failure writes a report, and the stack is torn down before
the GPU ASR pass.

The Chromium gate permits at most two attempts because the upstream checkpoint
has a measured wrong-tool base rate when tools are advertised. Every attempted
session is retained in `browser-attempts.json` with its log hash. One success
closes the browser gate; two failures fail the suite. This is a bounded evidence
policy, not an application-layer tool router or an unbounded retry-until-pass.

Outputs go beneath the configured trace root, normally
`~/.local/state/nemotron-voicechat/traces/live-suite-<UTC>/`. The top-level
`report.json` is the verdict. Component reports, browser logs, response WAVs,
ASR transcripts, restart timings, event streams, and server traces are retained
for audit.

For a shorter diagnostic run that is not a release qualification, explicitly
disable the session-limit expectation:

```bash
./voicechat test --live --asr-model /path/to/asr.nemo \
  --duration-seconds 180 --no-expect-session-limit
```

## Downloaded versus converted artifacts

The initial public release qualifies the downloaded, signed artifact. The
production conversion pipeline will be published in a follow-up commit. After
that stage-2 pipeline recreates `artifacts/release/`, run this same full suite
against the converted cache and compare component outputs and live verdicts:

```bash
uv run python tools/qualification/compare_candidate_reports.py \
  --downloaded /path/to/downloaded-suite \
  --converted /path/to/converted-suite \
  --output /path/to/source-agreement.json
```

The shipped comparator ignores timing and machine-specific paths. It requires exact
Nano component outputs, exact EarTTS component structure/transcription, and
matching browser, multi-turn, sustained, and overall verdicts.

## Release evidence

The public image audit writes image inspection, complete history, main and
Pocket runtime identities, deterministic SPDX 2.3 package inventory, and a
provenance attestation. A release build uses:

```bash
./voicechat bootstrap --no-cache
```

The no-cache build is a release gate, not the normal installation path. After
one successful online bootstrap, `./voicechat bootstrap --offline` must also
pass with network-independent artifact and host-lock validation.
