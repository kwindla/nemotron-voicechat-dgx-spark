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

Install Chromium for Playwright once, and run:

```bash
uv run playwright install chromium
./voicechat test --live
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
8. six fresh-session seeded memory cases covering voice, typed-via-Pocket, and
   system-prompt establishment crossed with voice and typed recall, plus one
   fresh-session unseeded negative control;
9. a paced run longer than 15 minutes, with typed turns every minute, queue
   growth checks, silent-turn accounting, and graceful 12,000-frame closure;
10. offline Nemotron English ASR of every retained assistant response, including all
    eleven seed-acknowledgement, recall, and negative-control WAVs from the
    memory matrix.

Memory recall prompts never contain a secret. Every seeded cell has a unique
secret; its runtime gate requires the model text channel to return its own
secret and none of the other five, and its independent Parakeet pass applies
the same inclusion/exclusion predicate to rendered EarTTS audio. The unseeded
negative control must return none of the six secrets in either channel. Seed
acknowledgements must be non-empty, audible, and structurally complete but need
not repeat the secret; their audio is still independently transcribed and must
match the model text within the ASR threshold. Thus all eleven WAVs are checked
for audio/text fidelity, while secret recall is evaluated on the six seeded
recalls plus the unseeded negative recall. Each matrix cell and the negative
control use a fresh
model session. Reports distinguish `typed_pocket` from microphone PCM and
system prefill so a green matrix cannot be mistaken for direct-text injection
evidence. The qualified Pocket carrier uses the frozen base seed from the
production environment and a SHA-256-derived per-text seed; repeated exact text
therefore produces byte-identical internal PCM instead of changing the model
stimulus between qualification sessions.

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
./voicechat test --live \
  --duration-seconds 180 --no-expect-session-limit
```

### Independent ASR evaluator

Qualification uses `nvidia/nemotron-speech-streaming-en-0.6b`, NVIDIA's
recommended checkpoint for English-only transcription, at immutable Hugging Face
revision `ebe59e5a817142986528bbbee5dba8db7b38ed50`. Its float32 safetensors,
processor, tokenizer, and configuration files are checked against the complete
inventory in `config/nemotron-asr-en-0.6b.json`.

The evaluator is a separate image based on the pinned NVIDIA PyTorch 25.12 image.
It pins Transformers 5.14.1 and uses `AutoModelForRNNT`/`AutoProcessor` directly:
16 kHz mono float audio, batch size one, the checkpoint-fixed English language,
13 lookahead tokens, greedy whole-WAV generation with the official encoder-derived
stop condition, the checkpoint's supported SDPA attention, no TF32, no
`trust_remote_code`, and no network
at qualification time. Reports include the immutable evaluator image ID,
model revision and snapshot hashes, package/CUDA/device identities, decoder
configuration, input WAV hashes, token IDs, and transcript. Missing provenance,
unexpected files, hash drift, unsupported deterministic CUDA operations, malformed
audio, or an unavailable GPU fails closed.

The qualification-only image is built and audited lazily by `test --live` and
`bootstrap --convert-from-source`; ordinary bootstrap and serving do not require
it. The audit compares the image-baked backend, manifest, and requirements with
the checked-in bytes, runs `pip check`, and performs deterministic short and long
silence decodes on the GPU before any verdict is accepted.
The non-silent canary is one unmodified, hash-pinned LibriSpeech clip used under
CC BY 4.0; attribution and exact source details are in
`licenses/LibriSpeech-canary-NOTICE.md`. No reference text or word boost is passed
to the ASR model.

This evaluator is independent of the VoiceChat serving environment and shares no
weights or decoder with EarTTS, though both the old and new evaluators are from the
NVIDIA speech-model family. EarTTS component generation remains in the serving
image. Its two canonical WAVs are handed to the evaluator through a byte-count and
SHA-256 manifest, then the finalized report is rejected if either file changed.

Changing the ASR judge is itself calibrated under
`docs/asr-judge-swap-policy.md`. The original component-identity policy and its red
result remain preserved. The prospective policy requires zero classification,
semantic inclusion/exclusion, per-response, or aggregate verdict changes; exact
separate-rerun determinism with unchanged contracts/WAV identities; and positive
and negative EarTTS controls. The artifact also records the shared pre-ASR silence
classifier control and exact-image reruns that bind the archived old-judge rows to
their `.nemo` model. Every WER-threshold
change remains disclosed and may be accepted only when another unchanged false
conjunct proves it cannot affect the archived verdict. A load-bearing WER change
blocks promotion, and the first green-path run receives an additional WER-margin
review.

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

### Capturing a post-function-call pairing recurrence

The retained failure predates per-frame effective-token telemetry. When a long
qualification is being run for diagnostic purposes, start the model server with:

```bash
./voicechat up --trace-pad-pair
```

The opt-in trace records scalar scheduler decisions, effective text/function
token IDs, request identity, and a log-only no-pair watchdog. Default serving
does not enable it. Treat trace-enabled timing as diagnostic composition data,
not as release latency evidence, because collection adds small per-frame host
work. A behavioral scheduler change requires a telemetry-complete recurrence;
the historic lossy trace alone is not a sufficient oracle.

### Direct-text semantic ladder

The direct-text diagnostic runs the same checked-in 20-case corpus through
Pocket+RNNT, FP32 direct positions, and W8 direct positions. Pocket must finish
every case with valid runtime provenance and structural evidence. Its RNNT
transcript is then evaluated by a per-case mechanical carrier predicate.
An explicitly failed response terminal can be structurally complete: balanced
terminal events and session closure count as evidence, while the failed status
remains a red semantic result. This prevents a model-selected loop-limit or
other valid failure terminal from masquerading as missing test execution.
Carrier-inconclusive cases are reported as inconclusive rather than success;
carrier-preserved model errors remain visible reference failures. Pocket is not
a conjunctive direct-text oracle because synthesis/RNNT can lose punctuation,
colors, digits, or other predicate-bearing content before Nano receives it.

Qualification was designed to be decided independently on both clean-token
direct lanes. Each token form had to meet the 95% semantic threshold, pass every
tool name and argument predicate, pass every safety-critical predicate, and
satisfy every hidden-position transaction invariant. The experiment failed
that production gate: direct text met only 10/20 semantic cases. A subsequent
neutral off/on/off trace compared c014 with a frozen acoustic carrier. The
acoustic path selected SOTC at settlement/EOU and produced the exact tool call;
direct injection kept PAD ahead of SOTC and emitted no call. Production typed
input therefore remains Pocket-to-acoustic conditioning. The direct machinery
is diagnostic-only evidence for evaluating a future trained adapter or native
multimodal checkpoint interface.

A direct response that does not reach its terminal within the fixed 240-frame
case bound remains a red, structurally incomplete case; the runner does not
increase that bound or reinterpret it as success. It aborts the stream and
starts the next case from a clean session. Transport reset explicitly restores
the wrapper-level agent-idle fallback before direct preflight, preventing an
open-turn bit from the bounded abort from poisoning the next fresh stream.

The FP32 qualification artifacts retain their immutable extraction manifest.
Before the isolated FP32 container starts, the runner writes an audited derived
manifest beside the qualification results. It verifies the expected extraction
kind, exact Nano/EarTTS component set, and exact build-time component paths;
then it changes only those two paths to their read-only `/derived` mount paths
and adds the source-manifest SHA-256 plus the path mapping. The source manifest
is re-read to prove it was not modified. The server receives this adapter as a
separate read-only mount and applies its ordinary strict path, per-file,
aggregate-model, checkpoint, and reproducibility validation. Production
manifest validation has no qualification bypass.

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
