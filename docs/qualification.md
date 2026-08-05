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

1. torch-dependent conversion tensor contracts inside the public image;
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

Run the full suite once with the published release cache and once with a cache
whose `artifacts/release/` is populated by `bootstrap --convert-from-source`.
Then compare the component outputs and live verdicts:

```bash
uv run python tools/qualification/compare_candidate_reports.py \
  --downloaded /path/to/downloaded-suite \
  --converted /path/to/converted-suite \
  --output /path/to/source-agreement.json
```

The gates may be resumed on fresh stacks after a retained stochastic model
failure. When the converted sustained gate is retained separately, make that
staging explicit rather than copying or rewriting evidence:

```bash
uv run python tools/qualification/compare_candidate_reports.py \
  --downloaded /path/to/downloaded-suite \
  --converted /path/to/converted-component-browser-voice-suite \
  --converted-sustained-report /path/to/sustained/report.json \
  --output /path/to/source-agreement.json
```

The comparator ignores timing and machine-specific paths. It requires exact
Nano component outputs, exact EarTTS component structure/transcription, and
matching browser, multi-turn, sustained, and overall verdicts.

On 2026-08-05, this comparison passed between the published HF artifact and a
public-source reproduction. The source-agreement report SHA-256 is
`1bf4ea21391865f293b5dca755088e4f9fe904ba1811744ade6aa9517df578b8`.
The locally reproduced sustained report SHA-256 is
`ee62c4547da3efa7f53a1e545d5acc7ad5c1ca440da1d91a437163236cca4608`.
The latter answered 18/18 typed turns, passed independent ASR with zero
near-silent responses, closed at exactly 12,000 model frames, measured a
0.375 ms/minute late-block queue slope, and peaked at 399.69 ms implied debt.
A separate failed source-runtime run that accumulated 2.98 seconds of debt
after malformed late function-call recovery is retained as negative evidence.

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
