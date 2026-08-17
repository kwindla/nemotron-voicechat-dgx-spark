# Generation-frame latency plan

Status: **draft r3, under adversarial review.** r1 review reversed the
draft's PAD-pair story and replaced its queue model; r2 review
(`docs/reviews/generation-latency-plan-review-r2.md`) returned NOT-READY
with four blockers: inconsistent measurement populations, a structural
target incompatible with the leading candidate, an underspecified watchdog
fix, and a pairing "recovery" precondition that is not recovery. r3
addresses all six of its closure conditions. Nothing here is promoted.

## Measurement schema (normative for every number and gate in this plan)

Every table and gate names its **population** and **clock** from this
schema. Percentiles are nearest-rank over the stated population; frames
within a response are autocorrelated, so cross-response claims use
response-clustered statistics (bootstrap by response).

Populations (per model-step row; r3.1 refinements from the Step 0
implementation review):
- `delivered` — canonical predicate `audio_delivered == true`; a
  bytes-based inference from legacy server traces lacking the field is an
  explicitly labeled degraded mode.
- `BOS-transition` — any EarTTS BOS transition row;
  `abort-prefill-reset` — the subset that performed an actual reset
  (prepared-reuse transitions carry no reset cost and must not be
  conflated with it). Both reported separately.
- `delivered-nonBOS` — delivered minus all BOS-transition rows.
- `response-null` — **literal** `response_id == null` on the source row
  (legacy population, kept for anchor reproduction; on legacy server
  traces this includes delivered rows that lack the field).
- `idle` (normative, r3.2) — `response-null` AND non-delivered, minus
  function-cycle pseudo-steps. (Inferred response grouping is used for
  response clustering only, never to define a literal source population.)
- High-mode strata are selected by **stage predicates over
  delivered-nonBOS only** (EarTTS-high: EarTTS ≥ 18 ms; residual-high:
  residual ≥ 10 ms), never by the total latency they are supposed to
  explain, and never overlapping the BOS strata.

Clocks:
- `server` — `server_step_ms` (model-step wall time; excludes event
  construction and send).
- `arrival` — client-observed `response.output_audio.delta` receipt times
  with exact per-delta sample counts.
- `browser` — AudioWorklet enqueue/dequeue/underrun timestamps plus
  Pipecat receipt/release timestamps with response ID, ordinal, and sample
  count (Step 2b instrumentation; relative intervals only unless a clock
  offset is established).

Anchor values (retained artifacts, `server` clock, delivered population):

| Artifact | n | mean |
|---|---:|---:|
| Sequential preflight (fresh, 2026-08-10) | 34 | 86.168 ms |
| Aged 26 h sustained probe (2026-08-12) | 193 | 85.403 ms |
| Aged, delivered-nonBOS | 188 | 82.919 ms |
| Pair-era baseline `f110f063…` | 32 | 75.116 ms |
| `response-null`, fresh (legacy idle) | 2,692 | 70.926 ms |
| `response-null`, aged (legacy idle) | 2,454 | 70.627 ms |
| `idle` (normative, function-cycle excluded), aged | 2,424 | 71.469 ms |

Clock rules (r3.1): the **normative** arrival clock is
`response.output_audio.delta` receipt time, reconciled to delivered
logical frames (fail closed or mark unavailable if the mapping is not
provable). Metrics-row receipt and server-emit times are diagnostic-only
(`normative: false`) and never issue a qualification verdict. The
sequential structural anchor on the normative delta clock is worst window
1009.934 ms (the metrics-clock 1009.957 value is retained as a
diagnostic anchor). Server-only traces cannot observe `response.done`;
below-threshold done-flush outcomes there are `done_flush_unknown`, never
fabricated. A structural evaluation with no complete K=12 window returns
`INSUFFICIENT`, not ACCEPT. Replay validates the real wire contract
(PCM16 / 22,050 Hz / mono, per `events.decode_audio_delta()`) and fails
closed on nonconforming payloads; simulated starvation is labeled
simulation-only until browser enqueue/dequeue evidence exists.

## Problem

The realtime contract is one 80 ms audio frame per model step. On the
current sequential production image, delivered generation frames average
~83–86 ms (anchors above; 34/34 over 80 ms in the preflight, 52/55 in the
browser math turn) and client arrival intervals average 86.6 ms pooled
(preflight-only mean 88.9 ms). The playout reserve (repo default 160 ms —
the deployed value was not inspected; Pipecat's TTFA metric is computed
upstream of it) absorbs a bounded deficit; retained ~3 s responses starve
it (A6), and the browser underruns audibly.

Established during review:

1. **PAD-pair history, corrected.** The retired conditional PAD-pair
   scheduler drafted whenever the previous effective token was PAD — which
   includes the generated audio tail, where it was heavily exercised
   (buffered PAD rows ~45 ms alternating with packed ~96 ms). The closest
   retained configuration comparison shows delivered generation at
   75.1 ms (pair era) vs 86.2 ms (sequential) on the server clock — an
   ~11 ms/frame association, though not a counterbalanced A/B. Retirement
   remains justified by the unrecoverable packed-call wedge. Note the pair
   schedule's per-call p95 was 118.3 ms while its arrival-clock peak queue
   debt was ~4.5 ms — which is why this plan's targets are queue/throughput
   gates, not per-call percentiles.
2. **Stage attribution** (A3, aligned populations): during
   delivered-nonBOS generation, Nano interface ≈ 52.3–52.8 ms dominates;
   perception ≈ 12.5–13.1; EarTTS ≈ 12.2–12.4 steady; codec call ≈ 0.6–0.7;
   residual ≈ 3.7–4.6. Distinct high-modes exist: EarTTS-high (≥ 18 ms,
   ~10 rows in the analyzed session), residual-high (~15 ms residual with
   normal EarTTS, 2 rows), and the BOS reset (own stratum). Host intervals
   are attribution hints, not additive causal GPU costs.
3. **Playout is a release-aware queue.** The service releases buffered
   audio downstream in a burst when the reserve threshold is crossed (or on
   `response.done` for short responses); audible behavior then follows the
   recovering queue over arrival intervals. The normative model is the
   exact state machine (A6's simulator), not any summed-excess scalar.
4. **No uptime drift demonstrated** (A2); the live stack was never
   restarted during this work. Not proven impossible; arms were not
   workload-identical.

### Blocking interaction: the no-text watchdog

Confirmed truncation: in the aged probe's tool response, frame 1791 reaches
no-text count 30 while decoded output is still −47 dBFS; frame 1792 is
forced to `agent_eos`. All six retained sustained responses end via this
watchdog (some during already-silent tails, which is benign). Consequence:
no ≥ 8 s response fixture can exist until this is handled.

**r3 decision (per r2 review):** this plan takes the
**qualification-scoped override** path — an explicitly qualification-only
mechanism (env-gated, refused in production images) that suspends the
no-text limit for latency fixtures. The **production** watchdog repair is
moved to its own correctness plan, which must specify a two-dimensional
liveness contract (the naive "pause while audible" rule would let
continuous audible non-text output evade all three watchdogs — a bounded
audio-only/max-response rule is required) and a full boundary test matrix
(BOS+silence, BOS+continuous-audible-no-text, legitimate long tails,
prosodic gaps, function-only output, fused/post-FC BOS, interruption,
session caps). The companion transport-gate defect (quiet first-turn
speech discarded at −40 dBFS) is likewise tracked separately.

## Targets

Preregistered, on schema populations. The former per-call p95 target is
**removed** (it would reject safe bursty schedules; see Problem 1). In its
place:

1. **Audible (normative):** running the exact release-aware playout state
   machine (exact chunk sample counts, threshold release, `response.done`
   flush, interruption at every arrival) over browser-clock measurements:
   zero underruns for responses up to the declared product envelope
   (duration bins, transport conditions, and independent-response counts
   declared before the run), at a reserve whose added release delay is
   explicitly accepted.
2. **Structural (normative):** logical media throughput ≥ 1.0 — for every
   consecutive 12-frame window of delivered logical frames, literally
   `t[i+11] − t[i] ≤ 960 ms` on the arrival clock — plus a bounded maximum
   single arrival gap (preregistered per fixture; default ≤ 160 ms) so a
   burst-and-stall scheduler cannot pass on averages. This exact convention
   is the one Step 0 encodes (r3 review verified it: pair-era worst window
   877.9 ms / 0 failures → accept; sequential 1010.0 ms / 13 failing
   windows → reject; discrimination is robust to the interval-counting
   variant). Verified against retained schedules in A7.
3. No regression to idle percentiles, voice-to-voice latency, barge-in
   measured at the output device (tested just before release, at release,
   and after each downstream frame — the release burst changes the worst
   case), or any semantic/WER gate. Semantic gates compare against known
   fixtures for **audio completeness**, not transcript presence (truncated
   responses can still show complete text).

## Plan

### Step 0 — Land the measurement schema and analyzers (offline) — READY

Implement the schema above in the sustained/qualification analyzers: exact
membership lists + source hash per aggregate, both queue diagnostics (the
release-aware state machine as normative; zero-reserve max debt as
informational), clocks/thermal/power telemetry columns, response-clustered
statistics.

Gate: reproduces every anchor value in the schema table from retained
artifacts, with published membership.

### Step 1 — Finish mining retained evidence (offline) — READY

Complete and record with aligned populations: stage-timing strata (A3),
pair-vs-sequential comparison on one clock (A4), EarTTS-high /
residual-high row inventories with token/control context, and the
negative-control reports (already copied into `reports/`, byte-verified;
A5).

### Step 2 — Unblock long-response fixtures — qualification parts READY

2a. Qualification-only no-text override (refused in production), enabling
≥ 8–10 s continuous-speech fixtures; fixture validity gate = audio
completeness against a known script, not transcript presence.

2b. Browser-side instrumentation: Pipecat receipt/release timestamps
(response ID, ordinal, sample count) and AudioWorklet
enqueue/dequeue/underrun capture; reconciliation with the Step 0 replay on
the same session using relative intervals.

### Step 3 — Stopgap reserve, sized by replay on real fixtures

Re-run the A6 sweep on genuine 8–10 s fixtures (blocked on 2a) with the
exact state machine, including interruption sweeps at every arrival
position. Pick reserve for the declared envelope; deploy via
`NEMOTRON_VOICECHAT_PREBUFFER_MS`; verify with 2b browser measurements.
Preliminary retained-data result (A6): 160 ms starves 4/6 of even ~3 s
responses; 400 ms was the first clean grid point for those samples (longer
responses *may* require more — path-dependent, not certain). Report added
release delay, browser first-audible delay, and reserve audio duration as
separate quantities (the reserve fills with all PCM including leading
silence). Evaluate adaptive reserve as fast-follow. Interim; revisited
after Step 5.

### Step 4 — Complete the profile and produce the candidate ranking table

Additions per r2 review — Step 4 must be able to rank Step 5:

- **Nano kernel/CPU-gap profile** (the dominant 52 ms term): Nsight
  Systems/Compute on captured content-position and PAD-position states;
  exact-output component benchmark; measure kernel time vs launch/gap time.
- EarTTS-high forensics on the stage-predicate row set (not
  total-latency-selected); residual-high separately (trace/logging overhead
  A/B on a deterministic fixture is the first residual hypothesis).
- Thermal/DVFS/memory-pressure regression against frame timestamps.

Output (the Step 5 decision input): a candidate table with, per candidate:
measured recoverable ceiling, correctness scope, reliability risk, and a
preregistered promotion floor.

### Step 5 — Structural candidate (selected BY the Step 4 table, not before)

Candidates to be ranked (no winner is declared in this plan):

- **Repaired tail-restricted PAD pairing.** Strongest retained latency
  precedent (A4) — but **inadmissible until its reliability precondition
  is closed**, which per the retirement doc's authority means one of:
  (a) eliminate the two-position packed vLLM call while keeping an
  amortized schedule; (b) find and fix the specific worker/graph/
  output-wakeup defect, then prove deterministic replay plus long
  worker-level stress; or (c) isolate the engine behind a supervisor that
  can kill and recreate the worker with proven clean reinitialization (no
  stale state or client lock survives), single-session failure only, and
  an explicit service-recovery SLO. A bounded timeout with session-fatal
  close is **not** closure — it cannot restore a wedged EngineCore.
  Additionally requires an executable tail state machine (entry/exit
  predicates, pending-row handling, rejection/rollback across BOS/EOS,
  SOTC/EOTR, post-FC BOS, interruption, abort, session caps) and a
  same-image, queue-clock, counterbalanced A/B with independent responses.
- **Nano batch-1 kernel work** (Marlin tile/split-K tuning, fused dequant,
  launch-gap reduction) — ranked by the Step 4 Nano profile against exact
  W8 outputs.
- **Wrapper small-launch-path graphing/fusion** — only if measured
  recoverable gaps ≥ 1.5 ms/frame.
- **EarTTS-high elimination** — worth up to ~8–10 ms of the p95 tail if
  the forensics find a fixable kernel/sampler cause.
- Selective EarTTS precision, two-frame perception batching — behind the
  above.

Negative controls (retained reports, `reports/`; byte-verified against the
source tree — do not rerun absent a changed-stack hypothesis): Step-8
concurrent perception overlap (+1.076 ms, defended stop, both stages
inflate under overlap), Step-5 barrier coalescing (seven barriers, none
removable; safe subset regressed).

Deferred, unchanged: Nano W4; CFG guidance changes (blinded non-inferiority
protocol required); low-buffer catch-up playback (offline evaluation
first).

## Constraints

- Single-session, batch-1, stateful runtime: check the single-client
  health/lock immediately before any live probe; fail closed if occupied.
- No container restarts are required by this plan; never use
  `./voicechat up` for probes against the current checkout.
- House rules: hash-qualified anchor patches, fail-closed, preregistered
  gates, response-clustered statistics, immutable-image evidence, negative
  results retained.

## Implementation log

- **2026-08-13 — Step 2 LIVE CAMPAIGN COMPLETED (attempt 10).** After
  driver qualification (7 offline review rounds) the campaign required ten
  live attempts; each failure was a fail-closed catch of a real
  fake-vs-real gap, fixed with the retained artifact as regression:
  post-release immediate-push semantics, object-identity brackets across
  the marshaling boundary, completion-vs-transcript-equality decoupling
  (twice: ordinary and I1), I1 dual typed-send windows, exact-shape RTVI
  predicates replaced by minimal five-anchor index matching (diagnosed via
  the retained observation the fix-12 diagnosability change added),
  teardown-latency publication racing (fixed with Playground Disconnect +
  90 s wait), and a frame-0 schema off-by-one. Two launch-environment
  deviations recorded (config mount for the validator's installed-layout
  path bug; launcher-injected image-identity envs dropped). **Results**
  (`reports/step2-live/session-20260813T083303Z/`): all 8 fixtures
  completed with terminal validation; short controls 3/3 full renditions
  (incl. capture-off); long bins: L1 1/2, L2 1/2 full — the model
  truncates long scripts stochastically (~50%), confirmed model-level
  (decoded-silence watchdog closes after EarTTS goes silent; no-text
  override present in every per-frame snapshot); I1 cancel-then-replace
  chain complete with second answer correct. **Step 3 decision inputs**:
  capture-on delivered 756 frames mean 83.2 ms p95 86.5 (75% over
  budget); real-response reserve sweep shows even 560 ms leaves 3
  responses starving — fixed reserves cannot clean long renditions,
  elevating the Step 5 structural fix; the capture-off full L1 rendition
  passed the K=12 arrival gate (worst window 936.8 ms) despite 84%
  over-80 server frames — arrival cadence smooths relative to server
  steps. **Adjudication (final):** Gate 2 PASS (zero no-text watchdog
  interference in retained traces); I1 replacement PASS (evaluator: "Two
  plus two is four."); Gate 1 FAIL campaign-wide per the preregistered
  rule — audio-complete renditions: L1 1/2, L2 2/2, capture-off-L1 0/1.
  The text-vs-audio divergence was demonstrated in both directions:
  L2-1's browser transcript mismatched while its audio was complete
  (text-channel flush artifact), and capture-off-L1's text was exact
  while its audio truncated — text evidence alone misjudges both. 35-file
  manifest sealed (`manifest.sha256`, self-hash `b62222f4…`); evaluator
  image digest recorded. Step 2 is COMPLETE; Step 3's reserve decision is
  a product trade (fixed reserves cannot clean long renditions) and
  Step 4 profiling opens next.

- **2026-08-13 — Step 2 CODE-COMPLETE (review round 4: PASS).** All deltas
  closed within the recorded threat model and verified with fresh probes
  (`docs/reviews/step2-implementation-review-r4.md`): consume-once
  validator authority surviving visible-field rewrites and two-thread
  races; real concurrent-process publication races with exactly-one-winner
  and orphan quarantine; the four round-3 producer-impossible traces plus
  two novel ones all rejected; full copy/deepcopy/pickle rejection matrix.
  Final state: 902 passed / 67 anchors / task-scoped lint green. Four
  review rounds total. The **live phase is gated** on the review's seven
  preconditions (immutable reviewed image, single-client check, fresh
  artifact paths, explicit qualification identity, pre-fixture /health
  capture, preregistered fixtures/gates, paired capture-off control) and
  its evidence-retention manifest; it requires an image build and a stack
  restart, and is scheduled explicitly, not autonomously.

- **2026-08-13 — Step 2 review round 3: FAIL (4 deltas remain); browser
  flush-ack delta closed with live-Chromium fault injection.** Remaining
  (`docs/reviews/step2-implementation-review-r3.md`): the proof seal is
  reachable-mutable (fix: validator-owned identity-keyed registry;
  raw-label grammar without normalization); artifact publication is not
  no-clobber and staging files persist on failure (fix: link-based
  no-replace publish, cleanup, quarantine of crash orphans); the consumer
  accepts four producer-impossible traces (post-clear audio, non-typed
  speech-start records, zero-sample chains, self-contradictory
  terminals); deepcopy reconstructs a half-initialized traced service.
  **Threat-model boundary (recorded):** the qualification fence defends
  against misconfiguration and careless in-process code; arbitrary
  hostile in-process code (e.g. `object.__setattr__` rewriting, module
  patching) can defeat any pure-Python design and is out of scope beyond
  the registry hardening. Fix round 3 delegated.

- **2026-08-13 — Step 2 review round 2: FAIL (5 boundary deltas); live
  phase still blocked.** Round-1 conditions largely closed (environment
  matcher rejects all fresh near-miss attacks; off-paths verified
  functionally byte-identical by independent code-object comparison;
  bounded writer and batched browser capture confirmed). Remaining
  (`docs/reviews/step2-implementation-review-r2.md`): proof-object
  forgeability (`dataclasses.replace`/subclass attribute spoofing at the
  engine boundary), false-valid persisted trace terminal on close failure
  plus non-finite timeout handling, consumer acceptance of
  clockless/unknown/misordered records and no prebuffer identity to check
  threshold releases against, unacknowledged final browser worklet batch,
  and shallow-copy producing a half-initialized traced service. Spec note:
  the qualification mode label must match `^[a-z0-9][a-z0-9-]{2,63}$`,
  is recorded verbatim in health/traces, and the label `production` is
  explicitly refused. Fix round 2 delegated.

- **2026-08-12 — Step 2 review round 1: FAIL (6 closure conditions); live
  qualification blocked.** The adversarial review
  (`docs/reviews/step2-implementation-review-r1.md`) found production-path
  defects: fail-open production refusal (any one-key drift admits the
  override — including the live recipe's own prompt change), trace-on
  behavior changes (stale malformed deltas become fatal instead of
  ignored; receipt clock sampled post-parse/post-decode), trace-off branch
  additions on both hot paths, unbounded writer queue + blocking cleanup,
  analyzer ignoring the new release/downstream-push records (verified by
  impossible-record injection), and per-quantum browser-binding load that
  can perturb measured underruns. **Design decision (r3.3):** the override
  becomes a positive, fail-closed qualification identity — it requires an
  explicit `VOICECHAT_QUALIFICATION_MODE=<label>` opt-in AND an
  environment equal to a checked-in candidate TOML environment except for
  an explicit allowlist of qualification deltas (the system prompt, the
  override itself, the mode flag); any other drift refuses. Off-path code
  must be byte-identical (construction-time variant selection). Fix round
  delegated.

- **2026-08-12 — Step 1 COMPLETE (review round 2: PASS).** All closure
  conditions verified with fresh raw-artifact probes
  (`docs/reviews/step1-evidence-review-r2.md`): corrected phase split
  209/615/7 member-for-member identical to independent recomputation, the
  seven moved rows exactly the round-1 `agent_eos` set, neighbor labels
  fixed, CLI validation pinned, reports byte-reproducible, 67/67 anchors,
  30/30 tests, 14/14 source hashes. Step 1 gate ("every number traces to
  a retained artifact + membership hash") is met. Deliverables:
  `reports/step1-retained-evidence-20260812.{md,json}`.

- **2026-08-12 — Step 1 review round 1: FAIL (one blocking finding).**
  The retained-evidence report's arithmetic engine fully verified by raw
  recomputation (10/10 sampled claims exact, all 14 source hashes match,
  pair-class walk reproduced exactly, PAD/BOS predicates sound, flake
  attribution confirmed pre-existing). Blocking: `_token_phase()` folds
  seven delivered `agent_eos` control rows into text-emission (true split:
  209 text / 615 PAD / 7 agent-EOS control of 831); closure requires the
  control stratum, EOS/CLI tests, and report regeneration
  (`docs/reviews/step1-evidence-review-r1.md`). Plan A4 prose corrected to
  the request-delta classification (12 packed, p95 118.710). Closure round
  delegated.

- **2026-08-12 — Step 0 COMPLETE (review round 3: PASS).** All five round-2
  deltas closed and verified with fresh counterexamples
  (`docs/reviews/step0-analyzer-review-r3.md`): ordered interruption/clear
  replay matching `llm.py` (cleared media never resurrected by a later
  `response.done`), at-release whole-burst exposure and elapsed drain
  before pre-arrival, typed-only cancellation trigger, literal
  `response-null` + r3.2 `idle` reproduced on both legacy traces
  (245/190, 490/326), and full verifier integrity (real corruption of a
  copied anchor tree exits nonzero on source and membership hashes).
  Final state: `verify-anchors PASS (67 anchors)`, analyzer tests 27/27,
  full suite 824 passed / 16 skipped. Three review rounds total; spec
  evolved r3 → r3.1 → r3.2 along the way. The Step 0 gate ("reproduces
  every anchor with published membership") is met.

- **2026-08-12 — Step 0 review round 2: FAIL (narrowing); spec refined to
  r3.2.** The fix round closed most of r1 (verified with fresh
  counterexamples: high-mode partition, canonical delivery, BOS strata,
  normative delta clock + fail-closed reconciliation, done_flush_unknown,
  INSUFFICIENT, hardened 60-anchor verifier; full suite 817 passed; a
  previously unchecked artifact — `nano-sequential-production-r1` —
  reproduced independently across all nine populations). Five fixes
  remain (`docs/reviews/step0-analyzer-review-r2.md`): observed
  interruption/clear must be an ordered replay event (a cancelled
  `response.done` currently "releases" media `llm.py` already cleared);
  add the at-release/pre-first-push interruption point and drain elapsed
  playout before pre-arrival; cancellation trigger must match
  `llm.py` (`speech_started` only with `source == "typed"`); spec
  decision recorded: `response-null` is literal on the source row and
  normative `idle` = response-null ∧ non-delivered − function-cycle
  (r3.2); complete verifier membership coverage + negative integrity
  tests. Fix round 2 delegated.

- **2026-08-12 — Step 0 review round 1: FAIL; spec refined to r3.1.** The
  implementation passed all 38 anchors and my independent non-anchor
  cross-checks (arithmetic, K=12 enumeration, nearest-rank, and the A6
  replay all verified correct), but the adversarial review
  (`docs/reviews/step0-analyzer-review-r1.md`) correctly failed it on
  population semantics: high-modes not scoped to delivered-nonBOS,
  idle/function-cycle overlap silently resolving a genuine spec/anchor
  inconsistency (spec now fixed: `response-null` legacy vs normative
  `idle`, new anchor n=2424 mean 71.469), BOS transition vs
  abort-prefill-reset conflation, wrong normative structural clock
  (metrics receipt instead of audio-delta receipt), fabricated server
  `response.done`, single-point interruption model, permissive wire
  metadata, vacuous ACCEPT on n<12, and an anchor verifier too narrow to
  catch any of it. Aged artifact moved to a durable location
  (`~/.local/state/nemotron-voicechat/qualification/sustained-aged-26h-20260812/`,
  events SHA-256 `acb205f6…`). Ambiguity-resolution records now live in
  this log rather than the transient scratchpad. Fix round delegated.

- **2026-08-12 — Step 0 started.** Implementation delegated to Codex
  (brief: `stratified_latency_analyzer` CLI under `tools/qualification/`,
  schema populations/clocks, membership manifests + source hashes,
  release-aware replay + informational zero-reserve debt, literal K=12
  throughput gate, anchor-verification subcommand). Review protocol per
  step: hands-on verification by the orchestrating session in parallel
  with a fresh-session Codex adversarial review; iterate until the step
  gate passes; notes recorded here.

## Data appendix

### A1 — Corpus sweep (2026-08-12; superseded in part)

Post-retirement sessions show delivered-generation means ~82–88 ms;
pair-era sessions run lower (75.1 ms in `f110f063…`). The sweep's deficit
figures used a non-recovering sum and are superseded by A6.

### A2 — Aged vs fresh sustained comparison (2026-08-12)

Schema-tabled anchors above. Conclusion: no uptime drift demonstrated
(non-identical workloads, one fresh response; no statistical test claimed);
the earlier "59 → 72 ms drift" reading was a workload-composition artifact.
The probe's aggregate p95 (82.2 ms at ~9% generation share vs 73.99 at
1.4%) is a live demonstration of why aggregate gates hid the problem.

### A3 — Stage timings, aligned populations (server clock, delivered-nonBOS)

| Artifact | n | perception | Nano | EarTTS | codec | residual | wrapper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential preflight | 33 | 13.06 | 52.29 | 12.15 | 0.70 | 4.57 | 82.77 |
| Aged sustained | 188 | 12.52 | 52.79 | 12.40 | 0.61 | 3.75 | 82.06 |

BOS stratum reported separately (preflight BOS row: EarTTS 84.5 ms
abort/prefill reset). High-mode inventory for `209fe177…` (55 delivered
rows): 10 EarTTS-high rows (EarTTS ≥ 18 ms; mean 20.4 vs 10.86 on the
other non-BOS rows), 2 residual-high rows (~15 ms residual, EarTTS
normal), 1 BOS row. Nano is flat (52.5–52.8) across all strata — the
dominant constant term.

### A4 — Pair era vs sequential (server clock, delivered)

Pair `f110f063…`: n=32, mean 75.116 (non-BOS 72.699), 13/32 over 80 ms.
Under the normative request-position-delta classification (Step 1 report):
12 buffered rows at 45.0 ms alternating exactly with 12 packed-pair rows
at 97.49 ms (p95 118.710, nearest-rank), plus 8 sequential/control rows
(one BOS, six text, one PAD) before pairing begins; arrival-clock peak
queue debt ≈ 4.5 ms.
Sequential preflight: n=34, mean 86.168 (non-BOS 84.024), 34/34 over
80 ms, client-arrival debt 293.9 ms. Association ≈ 11.1–11.3 ms/frame;
this is the closest retained configuration comparison, not a
counterbalanced A/B.

### A5 — Negative controls

`reports/step8-overlap-ab-20260807.md` and
`reports/step5-host-barriers-20260807.md`, byte-identical to the
source-artifact tree (SHA-256 verified in r2 review). Summaries in Step 5.

### A6 — Reserve replay sweep (2026-08-12, arrival clock, release-aware)

Six retained responses (2.7–3.4 s, all watchdog-truncated), exact
state-machine replay on `client_received_monotonic_s`; numbers reproduced
independently by the r2 review:

| Reserve | Responses starved | Total starvation | Mean added release delay |
|---:|---:|---:|---:|
| 160 ms (repo default) | 4/6 | 797 ms | 116 ms |
| 240 ms | 4/6 | 439 ms | 203 ms |
| 320 ms | 2/6 | 136 ms | 290 ms |
| 400 ms | 0/6 | 0 | 376 ms |

Pooled arrival-interval mean 86.589 ms over 221 intervals (preflight-only
88.906 ms; response means 82.5–89.0). "Added release delay" is the
model-client first-delta-to-release interval, not browser audible onset
(Step 2b measures that). 400 ms is the first clean 80 ms-grid point for
these truncated samples; longer responses may, not necessarily will,
require more.

### A7 — Structural-gate acceptance/rejection verification (2026-08-12)

The Target-2 gate (worst 12-frame window mean ≤ 80 ms, max single gap
≤ 160 ms) evaluated on retained schedules:

| Schedule (clock) | Worst 12-frame window | Max gap | Verdict |
|---|---:|---:|---|
| Pair era `f110f063…` (server emit — arrival unavailable in model trace) | 77.1 ms | 120.9 ms | accepts |
| Sequential preflight (client arrival) | 91.4 ms | 151.7 ms | rejects |
| Aged sequential probe (client arrival) | 92.4 ms | 147.2 ms | rejects |

The gate accepts the safe bursty scheduler and rejects the starving
sequential one, as required. Caveat: the pair-era row uses the server-emit
clock; a future pairing candidate must be gated on the arrival clock.

## Follow-ups outside this plan's scope

- **Deferred-tool-timeout test race (needs its own correctness ticket).**
  `test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails`
  (`tests/runtime/test_realtime_websocket_endpoint.py`, committed in
  `79cd152d`) patches a 30 ms function timeout and then polls for
  asynchronous abort completion; the failing signature is
  `engine.abort_count == 0` in the `invalidation_failure=False` subcase.
  Throughout Steps 0–4 it failed intermittently on full-suite runs in a
  collection-order-dependent way (observed independently by at least five
  review/implementation sessions, always passing on isolated rerun — the
  attribution analysis in `docs/reviews/step1-evidence-review-r1.md`
  confirmed it predates and is untouched by this plan's changes). On
  2026-08-13, during the Step 4a/4b round, it failed **in isolation** for
  the first time, so the race is drifting worse, not staying benign. The
  follow-up must determine whether the race is test-only (polling window
  too tight for a legitimately asynchronous abort) or a real
  abort-delivery liveness gap in the deferred-tool timeout path, and fix
  whichever it is; until then it remains a known flake that must never be
  "fixed" by loosening the assertion without that determination.
- Already recorded in the implementation log, listed here for
  discoverability: the qualification validator's installed-layout config
  path resolution bug (worked around by a hash-verified read-only mount;
  proper fix is searching `/opt/project/config`), and documenting the
  launcher-injected `VOICECHAT_RUNTIME_IMAGE`/`_ID` env keys as
  non-managed so qualification runs need not strip them ad hoc.

- **2026-08-13 — STEP 4 CLOSED (4e ranking review round 2: PASS).**
  Evidence base: 4a/4b descriptive (thermal/DVFS/pressure exonerated;
  server trace cost UNMEASURED pending the designed experiment), 4c
  INSUFFICIENT (EarTTS-high mode absent in 0/764 valid rows; retained-state
  pivot standing), 4d descriptive/INCOMPLETE (ten-block decomposition:
  content 53.56/45.85/7.71 ms, PAD 53.60/45.58/8.02 — ~86% of the Nano
  interface inside the CUDA-graph envelope; exact-output gate exposed
  fresh-engine graph-path replay divergence, a standing prerequisite).
  Byproduct: the deferred-tool "flake" was root-caused as a real
  abort-delivery liveness gap and fixed via a cancellation-safe
  transferred-ownership terminal path (3 review rounds; commit bc367bb).
  The ranking table (`reports/step4-candidate-ranking-20260813.md`)
  grants no candidate ranking authority yet; its reviewed recommendation:
  (1) close the W8 graph-path reproducibility gate first (unlocks the
  kernel and wrapper classes), (2) run pairing's wedge-closure decision as
  a design-only reliability track in parallel, (3) Step 5 selection only
  after at least one row turns `measured`.

- **2026-08-13 — Step 5 prerequisite phase opened.** Per the closed
  Step 4 recommendation: track 1 (W8 graph-path reproducibility
  root-cause, preregistered hypothesis discrimination with a
  benign-nondeterminism vs real-defect classification rule) delegated;
  track 2 (pairing wedge-closure design-only study) queued behind it.
  Step 5 candidate selection remains gated until a ranking row turns
  `measured`.

- **2026-08-14 — Step 5 prerequisite tracks CONCLUDED (both reviewed to
  PASS).** Track 1: the W8 graph-path divergence is a characterized
  bistable graph-construction defect (REAL STATE DEFECT); kernel-class
  candidates remain blocked pending an exact-gate repair at the
  vLLM graph-construction level; long-lived production instances carry a
  construction-time state qualification. Track 2: pairing's reliability
  precondition is NOT CLOSED — packed-call elimination and supervisor
  recovery are NO-GO on reviewed static grounds; wedge root-cause repair
  is the only bounded route, NEEDS-EXPERIMENT with repaired preregistered
  designs (matched within-epoch lanes, transactional sequential oracle,
  exposure-census-calibrated stress gate). Step 5 selection remains
  gated: no ranking row is `measured`. The three funding axes are (1) the
  graph exact-gate repair, (2) the pairing wedge experiments, (3) Nano W4
  requalification — an explicit product/engineering decision.

- **2026-08-14 — Bandwidth attribution side-study (user-funded)
  CONCLUDED.** Preregistered (SHA `0f638644…`) three-phase descriptive
  study; report `reports/step5-bandwidth-attribution-20260814.md`. The
  graph envelope is bytes-bound at the measured 239.6 GB/s ceiling, not
  idle: intra-envelope idle is 1.048 ms/frame, below the preregistered
  1.5 ms floor, so the wrapper graphing/fusion row **screens out**. The
  dominant recoverable cost is the unquantized bf16 `function_head`
  (7.292 ms/frame median, one vocab-sized GEMV per step) plus bf16
  attention projections (≈1.9 ms): a **quantization-extension candidate**
  worth ≈4.6 ms/frame (`function_head` W8 at measured Marlin throughput)
  to ≈5.5+ ms/frame (with attention W8) — descriptive, new-candidate
  qualification required; in-recipe precedent is the GPTQ-W8 `lm_head`.
  Attempt 1 of the profile phase is retained-invalid (call-39 divergent
  basin — one more divergent fresh graph construction for the Step 5
  defect census); attempt 2 (exact basin, 12/12 valid) is designated.

- **2026-08-14 — function_head W8 qualification G0–G2 CONCLUDED: all
  gates PASS; first `measured` ranking row.** Preregistered (SHA
  `40a36a2c…`); report `reports/fhw8-qualification-20260814.md`. G0:
  RTN-exact packing (max err 0.391% of weight max); 100% argmax agreement
  on all 1,172 retained-capture hidden states; loader fails closed when
  routing is incomplete (two retained attempts; candidate definition
  amended to include `modules_in_block_to_quantize`). G1: zero decision
  flips — baseline exact-basin lane 15/15 vs retained capture; matched-
  divergent comparison 439,500 paired decisions, 0 flips across five
  candidate engines (recorded deviation; both lanes share the
  characterized call-39 basin). New defect evidence: candidate
  constructions selected the divergent basin 5/5 — checkpoint layout
  influences basin selection. G2: **−4.437 ms/frame traced** (54.986 →
  50.549), mechanism exact (function_head bf16 GEMV −7.21 ms → Marlin
  +2.51 ms), no other family regressed. Projected untraced content wall
  ≈ 49.1 ms — generation frames near the 80 ms budget on this change
  alone. **G3 (live/behavioral promotion) remains required before any
  production use.** Follow-ons recorded: attention-projection W8; W4
  head variant; basin-bias → exact-gate repair case.

- **2026-08-14 — Wedge campaign increment 1 (Phase 0) CONCLUDED after
  three review rounds: v3 PASS confirmed.** v1 (synchronous dry run) and
  v2 (six round-2 defects) are retained as retracted/superseded; the v3
  campaign runs the exact qualified async task topology with exact
  FCFS-admission scheduling, 1,000,000 schedules realizing **829,062
  distinct event-vector interleavings**, genuine operation-loss injection
  10/10 (incl. a true stranded-host B9) plus a separate record-suppression
  self-test 10/10, all-966-file escrow verification, sealed identity
  enforcement with representative-restoring resume, and truthful
  production-call-site noninterference (78.94 ns disabled cost). Zero
  lost B0–B9 completions; the pass weakens only the transport/scheduler/
  output hypothesis. The revision guard's post-run rejection of a
  later commit is recorded as correct sealed behavior. **Phase 1 (12
  fresh-engine GPU discrimination of the B1 graph-dispatch hypothesis)
  may now be designed.** Large batch evidence (1.1/2.3 GB) retained on
  disk uncommitted; small artifacts and manifests committed.

- **2026-08-14 — fhw8 G3 live behavioral phase CONCLUDED: PASS.** Report
  `reports/fhw8-qualification-g3-20260814.md` (prereg sealed
  `671784e2…`). Candidate-backed service (earned-reproducibility manifest
  `fbb12132…`, accepted by the server's own fail-closed validation) ran
  the canonical 8-fixture plan: 8/8, zero terminalization errors,
  categorical parity with the retained Step 2 baseline on every gate
  field — no new failure class. Function-calling: baseline 20/20 vs
  candidate 20/20, identical decisions. Descriptive latency: candidate
  faster on 5/8 fixtures, long fixtures ~1–1.5 s faster to first audio.
  Startup attempt 1 hit the known release-artifact EarTTS warmup stall
  (retained; retry clean). **The fhw8 candidate has now passed its entire
  preregistered G0–G3 contract.** Remaining before production: the
  promotion contract's counterbalanced A/B latency qualification and the
  explicit user promotion decision.

- **2026-08-14 — fhw8 counterbalanced A/B CONCLUDED: AB PASS. Realtime
  budget met end-to-end.** Report
  `reports/fhw8-qualification-ab-20260814.md` (prereg `1eb8f6dc…`).
  Four sessions R1,C1,C2,R2, all 8/8 first-try. Δ = **5.452 ms/frame**
  (release 84.12 → candidate **78.67 ms/frame**, under the 80 ms
  budget), response-clustered 95% CI [3.55, 7.20]. Structural K=12 gate:
  candidate lanes **ACCEPT**, release lanes REJECT (the chronic overrun,
  recorded as a release-arm observation). Recorded deviation: normative
  analysis on pipecat playout traces (baseline-matching), server-clock
  analyses retained in parallel with identical contrast. **The fhw8
  candidate has completed the entire promotion contract (G0–G3 + A/B);
  production promotion awaits the explicit user decision.** Pairing
  (~11 ms further) remains in root-cause; Phase 1 harness in design.

- **2026-08-14 — fhw8 PROMOTED (local production mode) on user
  instruction.** `reports/fhw8-qualification/promotion/promotion-record.md`.
  Production stack running for manual testing: frozen candidate-1 env +
  generation-step2 image + fhw8 manifest/nano; pipecat on 0.0.0.0:7860;
  ngrok tunnel. Recorded follow-up: signed release-chain assembly (HF
  publication, user-owned). Wedge Phase 1 harness in fix round (four
  review blockers); its GPU campaign awaits a service-downtime window
  after manual testing.

- **2026-08-14 — Tool-call weirdness root-caused: PRE-EXISTING model/
  runtime behavior, NOT an fhw8 regression.** New diagnostic suite
  `tools/qualification/tool_call_behavior_suite.py` (direct-WS replay of
  retained live-session audio with production session config, energy
  turn segmentation, protocol-aware pacing, variation-tolerant metrics:
  loop incidence, max call-run, 4-gram repetition). Deterministic A/B on
  the loop-triggering session audio: candidate and release are
  **byte-identical** (same text, same 2 tool calls, same fatal) — 3+3
  replicates. Findings recorded for the follow-up ledger: (1) the live
  391-call tool loop and text repetition loops are model behavior under
  greedy decoding with overlapped user speech during function cycles
  (release-reachable; user-observed on candidate); (2)
  `UserEouSettlementFailure` (`_reject_initial_activity`) deterministically
  fatals the session when the model self-starts before client EOU commit
  — a real, client-reachable server defect in the settlement family;
  (3) suite-vs-server protocol contracts documented (1-based turn ids,
  model_output negotiation). fhw8 remains promoted.

- **2026-08-14 — Both live defects FIXED and validated on the
  deterministic reproduction.** (1) Function-repetition watchdog: the
  model-level TOOLCALL emission counter joins `AgentSilenceEosWatchdog`
  (limit 4/response, graceful EOS, reason `function_repetition_watchdog`;
  4 new unit tests) — guards the 391-call class the published-call budget
  could not see (pipeline self-feeds injections). (2) Pre-EOU settlement
  recovery: three-tier ladder before the unweakened strict fatal —
  administrative closure of a dangling self-interrupted response
  (`fail_active_response`, no fabricated output), bounded drain of an
  actively-producing response to natural terminal, bounded cancel
  escalation. Validation: the replay that fataled at turn 10 every run
  now completes 3/3 with zero fatals (12 recoveries observed). Image
  `defect-fixes-1` (overlay on generation-step2) now serves the promoted
  fhw8 production stack. Two intermediate attempts (cancel-first, then
  drain-first) are retained in `toolloop/fixed-*`/`fixed2-*` evidence —
  each exposed one more layer of the self-start taxonomy. Codex
  adversarial review of both fixes queued behind the wedge Phase 1
  confirmation.

- **2026-08-14 — Browser-path validation of the defect fixes: fixes
  exonerated; I1 fixture sensitivity isolated as a pre-existing open
  item.** The Chrome canonical plan on the fixed image failed only at I1;
  the unmodified generation-step2 control reproduced the identical I1
  timeout (6/8 + I1), and a fresh-service run refuted service-age
  degradation. The typed-site recovery call was removed anyway on review
  of I1 semantics (the typed path owns deliberate cancel-then-replace;
  recovery remains microphone-commit-only where the original fatal
  occurred). Failing I1 streams show BOTH answers completing behaviorally
  but 4 bot-interrupted events (vs the historical 2) and an anchor-order
  mismatch: the completion predicate requires the original answer's stop
  AFTER the interruption marker, which the known stochastic
  early-truncation can defeat. Open item: I1 predicate robustness (order-
  tolerant anchors) and the doubled interruption-event signature; all
  three failing runs + the exoneration control retained under
  reports/fhw8-qualification/browser-fix-validation/. S1/L1/L2 browser
  fixtures pass on the fixed image (6/6), and the direct-WS fatal
  reproduction remains 3/3 clean.

- **2026-08-15 — I1 root cause fully characterized: fixture premise vs
  stochastic truncation.** With the order-tolerant anchors (e17e57e) the
  RTVI predicate completes, and the failure moved to the driver's strict
  lifecycle contract (`step2_live_fixture_driver.py:1675`: exactly one
  interruption edge + clear release). When the original answer truncates
  early (the known pre-existing stochastic truncation), the scripted
  interruption fires against an already-stopped response, so no clean
  mid-speech edge/release exists. All non-interruption browser fixtures
  pass 6/6 on the fixed image; the WS fatal reproduction is 3/3 clean;
  the step2-image control exonerates the defect fixes. Open follow-up
  (qualified-driver scope, needs its own reviewed change): either an I1
  alternate lifecycle for interruption-after-early-stop, or a fixture
  retry-on-truncation rule — coupled to the standing stochastic-truncation
  known-limitation. Retained: four failing runs with distinct signatures
  + control evidence under browser-fix-validation/.

- **2026-08-15 — MAJOR FINDING (wedge round 4): genuine
  packed-vs-sequential divergence under the corrected committed-boundary
  comparator.** After a complete audit classified all 1,416 prior
  cross-lane differences (lane-absolute evidence excluded; KV projected
  to committed tokens; shadows canonicalized), the S-oracle-gated smoke
  found the first B case genuinely divergent: function_logits max delta
  **0.125** (the Step 5 basin-margin magnitude), committed Mamba conv
  state diverging from conv-history channel 3 (layer 0 exact), all 27
  SSM leaves differing — while tokens, decisions, pending/correction
  state, and committed KV are EXACT. Typed-carrier replay is now
  byte-faithful (synthesized PCM exact vs the retained 24-frame prefix).
  Phase 1 remains campaign-inadmissible under its exactness rule. The
  B-lane conflates packing with dispatch mode; the A-lane discriminator
  (packed+graph vs sequential+graph) is running to isolate eager-mode
  numerics vs true pair-path divergence, deciding between a mode-matched-
  oracle design amendment and a Step-7-contradicting pair-path finding.

- **2026-08-15 — Tool-call × interruption matrix built and PASSING
  12/12 on the fixed image.** New qualified harness
  `tools/qualification/tool_call_interrupt_matrix.py`: elicits a real
  spoken tool call, then lands a genuine second utterance at phase-locked
  offsets (call outstanding pre-response / during result injection /
  mid-acknowledgment) plus a no-interruption control, 3 replicates each,
  with a continuous silence clock for realtime pacing. All cells: exactly
  one tool call (no loops), zero fatals (the settlement recovery held at
  every FC phase, its first live exercise), and a second answer lifecycle
  after every interruption. Bonus negative-control: with a mismatched
  advertised tool the model declined 12/12 per the production instruction.
  This closes the coverage gap at the tool-call × barge-in intersection;
  optional follow-up: a release-arm baseline of the same matrix.

- **2026-08-15 — Wedge Phase 1 track CONSOLIDATED at an evidentiary
  stopping point: pairing-as-retained is disqualified from
  exact-replacement revival by two sealed findings.** (1) bf16
  recurrent-state drift from packed scan reduction order (decisions
  exact; production served the whole pair era this way). (2) **NEW
  BUG-CLASS: post-rejection corrected-drain committed-attention-KV
  non-equivalence** — at verified-aligned logical position 377 the packed
  path commits different K/V content than sequential in all four
  attention leaves, deterministically across five fresh constructions,
  with divergence accumulating thereafter
  (`phase1-postdrain-kv-adjudication-20260815.md`). The five-construction
  calibration campaign fail-closed REJECTED its own envelope amendment on
  this finding; no waiver, projection shift, or tolerance was adopted;
  the 12 held-out replicates remain unrun and Phase 1 unscheduled. The
  full real-executor stack (atomic 1,439-file tree, observed-only
  snapshots, evidence-verified dispatch, schema-3 used-runtime identity)
  is review-hardened through five adversarial rounds and committed.
  **Pairing's ~11 ms now requires a drain-path KV repair in the pair
  implementation, then recalibration and requalification — a defined
  follow-on project.** The B1 wedge-liveness question remains open but
  moot until the semantic repair lands.

- **2026-08-16 — NEW open item: user-EOU blank-fence settlement cost
  (perceived-latency dominant in rapid conversation).** Characterized
  from live session 10c4b983: each turn's settlement steps ~8+ synthetic
  silence frames (target `nonblank_reset_after_silence` = 10, a model-
  config constraint — forcing BOS earlier opens empty responses per the
  retained qualification note at `user_eou_settlement_blank_frames`),
  costing ~0.6 s wall per turn during which live mic audio queues; with
  only ms-level per-frame headroom the backlog accumulates over
  back-to-back turns (TTFB 0.75→2.35 s over six turns, draining ~1.2 s in
  a 10 s pause). Reduction directions, none yet attempted: (1) consume
  the queued REAL post-EOU audio (gated below the speech threshold) for
  fence advancement instead of adding synthetic frames on top — removes
  the double-count entirely; (2) arm the fence at client speech-stopped
  rather than at commit — the client's own smart-turn silence window and
  the fence currently serialize while observing the SAME silence; (3)
  measure empty-response incidence at fence 6–8 vs 10; (4) investigate
  extending fused-BOS to tool-advertising sessions (`fuse_final_bos =
  not session_tools` forces the slower separate-BOS path exactly when
  tools are registered). Also NEW open item: model tool-call propensity —
  casual asks answered by text hallucination incl. confabulated "I did
  make the tool call" (A/B-proven pre-existing); candidate mitigation:
  strengthened tool-obligation instruction, testable with the behavior
  suite's propensity metrics.

- **2026-08-16 — fhw8 weights published to the signed release chain and
  made clone+bootstrap reproducible.** HF release commit
  `fdf70edf9d8b…` on `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`
  (only ~537 MB of genuinely new LFS data after chunk dedupe); new
  inventory verified pre-upload with the bootstrap's own `verify_release`
  and re-verified by download round-trip (`release_sha256 426eecc0…`,
  nano composite `dd4ce6a9…` = the qualified candidate manifest value).
  Repo side: `config/production-candidate-3.toml` +
  `config/artifact-release-fhw8.json` pin the new revision;
  `deploy/promotion-candidate/` carries the portable launch scripts, the
  deterministic derivation script with fail-closed qualified pins (the
  provenance proof that the uploaded bytes are exactly the committed
  transform), the inventory builder, and the runbook. Recorded follow-up:
  fresh-machine end-to-end test of the candidate-3 `./voicechat up`
  bootstrap-state swap.

- **2026-08-16 — User-EOU settlement latency increment CONCLUDED (2
  review rounds, r2 PASS): the conversational-lag accumulation is
  eliminated.** Two settlement-path changes on the unchanged fence
  target/terminal contracts: fence-on-real-audio (queued sub-gate
  microphone frames advance the fence; barge-in stops and requeues
  fail-closed) and revocable early arming at the speech-stopped edge.
  Round-1 review caught a commit-edge padded-partial-frame arming defect,
  fixed by explicit model-input provenance (only pre-commit complete
  `microphone_audio` frames can arm) with the reviewer's reproduction as
  a committed regression test. Regenerated r2 gates: V1 zero empty
  responses (80/80, tool and no-tool); V2 six-turn median
  commit-to-first-audio baseline `861→2,919 ms` vs candidate flat
  `894→753 ms` (slope −2,058 ms → −141 ms); 78/80 settlements consumed
  real queued audio; V3 full battery green; V4 canonical browser plan
  8/8. Evidence repo-relative under `reports/fence-latency/` with
  manifest `4956d89c…`.

- **2026-08-16 — Attention-W8 campaign CONCLUDED: FAIL at G3, candidate
  RETIRED per preregistration — with a major finding.** G0 exact (16
  tensors, double-build byte-identical); G1 zero flips across 10,548
  matched decisions; G2 **−1.546 ms/frame** (49.00 vs 50.55, triple the
  gate). But G3's interruption matrix: **0/12 tool calls** — the
  candidate transcribes the directive carrier perfectly and answers in
  text without ever calling, deterministic over all replicates, where
  fhw8 elicits 12/12 on the identical carrier. FINDING: attention-
  projection quantization (RTN W8 g128) collapses tool-call propensity
  while leaving frozen-corpus decisions untouched — the tool-affordance
  decision lives in the attention pathway and is precision-sensitive.
  Consequences recorded: (1) the G1 frozen corpus contains no
  tool-elicitation contexts and cannot gate this behavior — future
  interior-quantization campaigns must include tool-decision replay in
  G1 or run G3 first; (2) unsealed future options: calibrated GPTQ
  (Hessian-weighted, likely far better than RTN for attention),
  per-projection ablation, tighter groups; (3) strengthens the
  tool-propensity ledger item: the behavior is fragile at bf16 already.
  ~1.5 ms/frame remains on the table behind a better quantizer. All
  evidence sealed under reports/attnw8-qualification/ (root manifest
  ff3f5797…).

- **2026-08-17 — attnW8-GPTQ campaign STOPPED fail-closed at A0, with two
  findings; ~1.5 ms/frame remains blocked.** Plan sealed after three
  adversarial review rounds (`17647bd7…`). (1) **Tool propensity is NOT
  as fragile as the sequential-curation churn implied.** After the
  pooled-selection deviation, ONE baseline over a sealed 48-scenario pool
  validated **48/48** (24/24 positive-intent, 24/24 no-call-intent), min
  transcript similarity 0.889. The 39 prior Seal-0 revisions were almost
  entirely ASR transcript-exactness artifacts, not model behavior — and
  sequential replacement was additionally an adaptive-selection bias the
  pool removes. (2) **NEW FINDING — eager/graph categorical disagreement
  on tool decisions:** sealed no-call scenario `a2-nocall-r25-01`
  (audio `5de8732c…`) is an exact zero-call/zero-injection under the
  packed production runtime but emits function tokens under eager
  reconstruction, raising the fail-closed A0 stop. This is the same
  eager-vs-graph divergence family as the Step 5 basin defect, and it
  **invalidates eager-mode activation capture as a faithful basis for
  calibrating tool-relevant behavior** — the premise of this campaign
  (and a caveat on fhw8's eager-derived G0(b) evidence, whose G1/G3
  decision gates were nevertheless run on the production path). No
  Hessian, candidate, or gate result was produced; Seal-1 never created.
  Prerequisites for any resumption: adjudicate the three recorded
  criterion deviations, and resolve the eager/graph tool disagreement
  (either explain it or capture activations on the production path).
