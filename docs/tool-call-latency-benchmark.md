# Tool-call latency benchmark protocol

Status: preregistration draft. Do not use results for an implementation decision
until the benchmark artifact and the subsequent implementation plan pass separate
adversarial reviews.

## Question

Why does a successful production tool turn spend roughly ten seconds inside the
NVIDIA inference stack after end of user input, and what is the smallest safe
change that materially reduces that time without weakening function-call,
turn-taking, text, or audio correctness?

This benchmark measures the production external-tool path. It is not an ASR or
general speech-quality benchmark, and it will use ordinary English without
acronyms, digits, dates, times, or alphanumeric verification strings for semantic
gates.

## Evidence already established

The retained production trace
`6bf2fe21-4249-4a01-8f54-b12ae97e0682/events.jsonl` and the corresponding model
log establish the following for one complete external-tool cycle:

- The running wrapper reported `fc_async_enabled=True`,
  `fc_async_two_phase=True`, and `fc_async_use_real_audio=False`.
- SOTC was committed at transport/model-step frame 45.
- Async phase one advanced 28 Nano positions in 2.351 seconds. Its existing
  selected-stage timing log reports 1.723 seconds in the Nano interface
  (61.5 ms/position) and 0.619 seconds in EarTTS warmup
  (22.1 ms/position).
- The host tool completed in approximately 11 ms.
- Async phase two advanced 73 Nano positions in 6.237 seconds. Its existing
  selected-stage timing log reports 4.836 seconds in the Nano interface
  (66.2 ms/position) and 1.385 seconds in EarTTS warmup
  (19.0 ms/position).
- The result contained 72 forced function tokens followed by exactly one natural
  EOTR position. The post-FC terminal BOS was then generated on the normal path.

The approximately 80 ms change cadence visible in main-thread `model_step` rows is
therefore not evidence that async function calling is disabled or paced by input
audio. While the background worker runs, those rows are cheap polling/no-op
transport ticks. The background worker's own aggregate shows that sequential Nano
and EarTTS work consumes approximately 84--85 ms per async position. The benchmark
must bind the resolved async configuration and must never attribute main-thread
`inference_ms` to background-worker compute.

The pinned NVIDIA Speech commit `911ec674ab40f04302ef33672be4179f45a7310f`
already implements the two-phase `_run_fc_async_steps` path. Our patch extends and
hardens that path; it does not introduce the basic async architecture. The source
comparison will identify which behaviors are pristine NVIDIA behavior and which
are local reliability/correctness changes.

The async path has prior reliability history. A retained tool-freshness attempt
intermittently stalled inside the background worker; a subsequent unchanged retry
completed. That incident produced the opt-in FC-async stage heartbeat and verified
SIGUSR1 all-thread dump path now present in the runtime. This benchmark enables
that diagnostic and treats any recurrence as retained failure evidence. It does
not infer a fixed stall cause from the earlier nondeterministic attempt.

## Configuration and provenance contract

Every run records and exact-compares:

- immutable public-runtime image ID;
- NVIDIA Speech repository commit and applied patch SHA-256;
- server, benchmark driver, and analyzer SHA-256;
- model/config/artifact identities already exposed in runtime provenance;
- resolved values of `fc_async_enabled`, `fc_async_two_phase`,
  `fc_async_use_real_audio`, EarTTS backend mode, Nano sampling parameters, and
  all active fixed-stage optimizations;
- tool schema, system instructions, prompt, exact tool-result UTF-8 bytes,
  tokenizer token IDs, and the canonical scenario hash;
- whether input was real-time paced or deliberately sent faster than real time.

Sampling remains the qualified greedy Nano configuration. A run is called
deterministic only if repeated calls produce identical function-channel token IDs
through EOTC and identical forced result token IDs. Otherwise the report uses the
observed token counts and does not claim determinism.

## Clock and frame domains

All new benchmark durations use `time.monotonic()`/`time.perf_counter()`
boundaries on the host. The existing NVIDIA aggregate uses `time.time()` and is
retained only as corroborating evidence, not mixed into the new ledger.
Each event declares its clock and its index domain. The domains are not
interchangeable:

- `transport_frame`: server PCM/control processing invocation;
- `model_step_frame`: ordinary server-visible model step;
- `nano_position`: Nano sequence position, including hidden async positions;
- `eartts_position`: EarTTS sequence position;
- `browser_performance`: browser `performance.now()`; comparable only within the
  same browser page.

Mappings are retained as explicit before/after position counters. In particular,
the async EOTR frame is a Nano position and must not be presented as a transport
frame.

The direct-model benchmark stops its primary latency clock at first server-audible
PCM. Browser audible onset is a separate validation measurement using the existing
same-page WebAudio method: start is the final speech-like input frame in the
synthetic sequence and stop is the first speech-like received frame. It is not
mixed into inference-stage sums.

## Per-cycle timeline

Each correlated call retains these monotonic boundaries:

1. client commit sent and received;
2. SOTC committed;
3. async phase-one entry and exit/EOTC;
4. function call published;
5. host tool result produced, sent, received, and applied;
6. result tokenization start/end and exact forced-token count;
7. async phase-two entry, forced-token drain, and natural EOTR observation;
8. background result committed to the main context;
9. post-FC terminal BOS;
10. first model text, first server-audible PCM, response text EOS, acoustic end,
    and response done.

For every async phase, retain the following per-phase fields and counts:

- positions;
- embed, perception-poll, fusion, Nano-interface, EarTTS, codec, RNNT, and total
  elapsed milliseconds;
- named-stage sum and residual;
- starting/ending Nano and EarTTS positions;
- real-audio/perception frames consumed;
- EOTR wait positions;
- EarTTS reset/reuse/transition evidence.

The existing NVIDIA timing text is not sufficient as the benchmark ledger. Its
reported `total` is the sum of selected intervals, not a separately measured loop
wall time; RNNT and perception-thread join time are not independently accumulated;
and `LLM` combines input preparation/copy, vLLM append/submission, output wait, and
result parsing. Before the GPU sweep, add one opt-in structured record that:

- measures true phase wall time with `perf_counter`;
- separately accumulates RNNT and perception-thread shutdown/join time;
- splits each Nano and EarTTS engine call into existing input conversion/copy,
  request append or initial submission, async output wait, and result handling;
- retains a bounded per-position vector plus totals so p95/p99 can be rederived;
- records named-stage sum and wall residual and rejects non-finite, negative, or
  overlapping accounting outside a small absolute floating-point tolerance;
- is published once per phase with request/cycle/phase identity; and
- is strictly opt-in, with no clock calls, tensor reads, CUDA synchronization, or
  retained diagnostic state on the default path.

The independently measured phase wall duration and the sum of nonoverlapping
stage durations must agree within 150 ms. That tolerance is preregistered to
cover thread handoff, logging, and Python scheduling without hiding a material
unmeasured interval. A larger residual invalidates the latency attribution and is
investigated rather than assigned to Nano or EarTTS. Instrumentation is run once
enabled and once disabled on the same short cell; configuration, heartbeat
setting, outputs, and token IDs must match, and any material latency shift is
reported as measurement perturbation.

The per-position record contains `(async_step, absolute_nano_position,
nano_interface_ms, eartts_ms, monotonic_timestamp)` plus the split Nano-interface
subdurations. It is a bounded append-only phase ledger, not the existing
last-writer-wins heartbeat. The heartbeat remains enabled and unchanged for stall
localization.

For main-thread steps, retain `dt_since_previous_model_step`, `inference_ms`,
`server_step_ms`, and `server_step_ms - inference_ms`. These fields describe the
transport loop only while an async background worker is active; the analyzer must
not add them to background-worker stage totals.

Typed input also records Pocket synthesis time and generated audio duration.
Input-carrier time is reported separately from commit-to-SOTC and never included in
the function-cycle duration.

## Workload

The benchmark driver uses one external no-argument tool whose returned `spoken`
surface is an ordinary-English sentence. Production limits the complete wrapped
function output to 128 tokens, so it constructs four production-valid result
sizes near 20, 48, 72, and 120 forced tokens. It tokenizes the complete serialized
result after the same number-conversion and wrapper logic used by
`validate_function_output`, then selects the smallest sentence at or above each
target without exceeding 128. The exact wrapped UTF-8 bytes, IDs, and realized
counts are retained before the first measured run. The first word is a natural
verification word and subsequent turns exclude all earlier words. Candidate
sentences contain no digits, and the preflight proves that conversion leaves each
selected serialization and token vector unchanged.

Call-emission cost is measured with two preregistered tool schemas that differ in
name/argument serialization length while preserving task meaning. Because model
selection can vary, the independent variable is the realized SOTC-through-EOTC
token count, not the nominal schema length.

The workload is deliberately not a full factorial. The preregistered run ledger
is:

- primary result slope: four result-token sizes by ten fresh typed-input sessions
  each, using the short schema (40 calls);
- call-emission slope: short and long schemas by ten fresh typed-input sessions
  each, using the approximately 72-token result (20 calls; the short-schema rows
  may reuse the matching primary rows);
- same-session drift: one fresh session containing ten serialized calls with the
  approximately 72-token result;
- modality check: ten fresh qualified-Pocket speech sessions at the approximately
  72-token result, compared with the already measured ten typed rows;
- mechanism probe: three real-time and three faster-than-real-time speech carriers
  at the approximately 72-token result, explicitly non-promotable;
- safety cells: three fresh runs each for no-tool response, chained two-call,
  barge-in during call emission, and barge-in during result injection.

This is 88 measured tool calls after reusing the ten matching short-schema primary
rows, rather than multiplying every factor. If any primary cell has a failed or
invalid call, the failure remains in its ordinal slot; the cell is not topped up
after results are visible.

Before GPU activation, the driver performs a dry preflight against the production
16-KiB/128-token output limits, the one-position EOTR contract, the async position
budget, model context bound, and the agent text/audio watchdog bounds. A cell that
cannot complete within those unchanged contracts is removed before any results
exist and the preregistration is amended; limits are never raised to preserve a
nominal payload target.

The primary quantitative sweep is direct model WebSocket with real-time input
pacing. One deliberately faster-than-real-time carrier arm is a mechanism probe;
it is never a promotion result. Because the deployed async worker is already
backgrounded, the preregistered expectation is that carrier speed will not
materially change phase-one or phase-two duration after conditioning on position
count. A large change instead triggers a contention investigation. The fast
carrier also changes main-thread/GIL activity, so a modest shift is not by itself
attributed to perception contention.

The exact primary payload order by replicate block is:

| block | ordered wrapped-token targets |
|---|---|
| one | 20, 48, 72, 120 |
| two | 48, 72, 120, 20 |
| three | 72, 120, 20, 48 |
| four | 120, 20, 48, 72 |
| five | 20, 48, 72, 120 |
| six | 120, 72, 48, 20 |
| seven | 20, 120, 72, 48 |
| eight | 48, 20, 120, 72 |
| nine | 72, 48, 20, 120 |
| ten | 120, 72, 48, 20 |

Blocks six through ten are the row-wise reverse of blocks one through five.
Alternate short and long schema order by replicate. Run the same-session and
safety cells only after the fresh-session primary sweep. The measured-call
arithmetic is `40 primary + 10 additional long-schema + 10 same-session + 10
speech + 6 carrier probes + 12 safety = 88`. The GPU wall-time budget is two
hours, including startup and retained failure evidence; reaching it stops the run
without replacements. Failed, timed-out, or semantically invalid calls remain in
the artifact and are not silently replaced. Browser validation is
smaller: at least three fresh sessions covering typed, speech, one real tool, and
one barge-in after the direct-model mechanism is established. Barge-in is
triggered from an observed correlated protocol event: SOTC/function-call
activation for call-emission interruption and forced-result injection start for
the result-recovery serialization probe. In the latter case, the client sends
during the observed recovery phase, but the server's inline recovery loop does
not receive that input until recovery completes; the ordered transport, rather
than the application deferred-input queue, holds it first. It is never scheduled
by a fixed delay from user commit, and it is not described as a server-observed
model interruption.

## Analysis model

Analyze phases separately. For each phase:

```text
phase wall time
  = embed + perception/fusion + Nano interface + EarTTS/codec + RNNT + residual
```

Fit elapsed time against realized positions only after checking the named-stage
ledger. Report median, mean, p50, p95, p99, bootstrap confidence intervals, and
per-position distributions. The useful decision quantities are:

- Nano milliseconds per call-emission position;
- Nano milliseconds per forced-result/EOTR position;
- EarTTS/codec milliseconds per hidden position;
- fixed phase-entry/exit residual;
- post-EOTR BOS-to-first-text and BOS-to-first-audible durations;
- drift by same-session call ordinal.

The Nano-interface total is not treated as Nano kernel time. A separate
standalone Nano decode microbenchmark runs in the same immutable base image and
model configuration, batch one, greedy, with controlled initial context lengths
and back-to-back single-token positions. The qualified runtime's reconstructed
vLLM 0.17.1 core has no built-in iteration-detail clock, despite the newer pinned
native source containing one. The benchmark therefore uses a derived diagnostic
image with one opt-in, no-CUDA-synchronization `perf_counter` ledger around the
actual reconstructed `EngineCore.step` boundaries: scheduler, blocking model
executor, and scheduler-update. It emits bounded structured rows keyed by the
scheduled request ID and iteration. The patch SHA and derived image ID are
retained, its disabled branch is byte-for-byte the original step path, and it is
subject to the same enabled/disabled perturbation control as the phase ledger.
The enclosing `model_llm_interface` call is independently split into input/copy,
append-or-submit, output wait, and parse time. The analysis regresses per-step
latency against absolute Nano position and compares scheduler, model-executor,
update, and wrapper-interface time. These are CPU-observed execution boundaries,
not CUDA-kernel duration; any remaining executor-internal scheduling is reported
as unresolved rather than labeled model compute. Phase-one vs phase-two averages
alone are not used to infer context-length scaling because the phases differ in
other work.

The plan produced from the benchmark must show arithmetic for both expected saving
and remaining floor. For example, an optimization affecting only EarTTS can claim
at most `positions * measured EarTTS milliseconds`; it cannot claim the Nano time.
The retained 101-position example puts that ceiling near two seconds for its
approximately 20-ms EarTTS line, while its approximately 65-ms Nano-interface line
is about 6.6 seconds. Only the independently measured wrapper/scheduler fraction
of that Nano line can be claimed as host-overhead savings.

## Source comparison

Before interpreting results, produce a focused diff table for pristine NVIDIA vs
the applied runtime covering:

- async phase entry and two-phase handoff;
- Nano custom-input submission and output wait;
- EarTTS warmup, codec work, and silence delivery;
- live perception/RNNT catch-up and barge-in;
- forced result construction and EOTR validation;
- post-FC context merge and terminal BOS;
- local heartbeat, timeout, physical-request-generation, and prepared-epoch
  hardening.

The comparison explicitly verifies that `[INJECTED_RESPONSE]` and
`[CONVERTED_RESPONSE]` are diagnostic strings only and do not duplicate forced
tokens.

Codec domains remain separate. The worker's per-hidden-position EarTTS/codec-cache
synchronization is measured inside the async ledger. The 80-ms silence chunks
made available to the normal output path and their main-thread decode/delivery
work are reported separately; neither is silently charged to Nano. NVIDIA's
source comment states that omitting per-step TTS KV/cache synchronization leaves
the post-function-call audio path behind and causes choppy subsequent audio. Any
omission candidate must therefore test that adverse mechanism, including the
abort path where no later reset repairs the state.

## Safety and interpretation gates

A candidate is not promotable unless the eventual qualification retains:

- exact call cardinality, call ID, response ID, and turn correlation;
- exactly one SOTC/EOTC/EOTR cycle per call and the required post-FC BOS;
- zero ordinary function-channel leakage into agent output;
- unchanged model-text semantics and independent response-audio ASR;
- typed, voice, repeated, chained, and long-session behavior;
- barge-in during both call emission and result injection;
- exact Nano/EarTTS position accounting and post-burst perception/RNNT alignment;
- retained audio, traces, reports, provenance, and hashes;
- bounded failure/timeout behavior with no stale background worker or request.

Benchmark diagnostics may intentionally alter carrier pace, but they cannot be
cited as qualification evidence. No threshold, tokenizer, result corpus, or
semantic gate is relaxed after observing results.

## Decision sequence after results

Candidates will be ranked by semantic risk and measured expected value, not by
implementation convenience. The plan must evaluate at least:

1. whether EarTTS/codec advancement on hidden function positions can be omitted or
   reduced given the mandatory later reset, with explicit barge-in and recurrent
   state analysis;
2. which part of the 60--66 ms Nano-interface cost is model compute versus vLLM
   per-position scheduling/submission overhead;
3. whether a generation-internal, sequential multi-position API can preserve the
   exact Nano recurrence and forced-token sequence while amortizing host/vLLM
   overhead;
4. whether any proven result serialization overhead is removable (the duplicated
   diagnostic display strings are not presumed to be duplicate model input).

A conventional parallel fused kernel is not assumed safe: consecutive Nano
positions are causally dependent. Any fused/device-side proposal must preserve
that dependency and distinguish byte-exact forced token IDs from the separate
question of which audio/perception embeddings the hidden positions consume.

The quantitative implementation/qualification plan receives a second Fable
adversarial review before inference behavior is changed.
