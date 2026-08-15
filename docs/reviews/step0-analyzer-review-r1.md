# Adversarial review: Step 0 stratified latency analyzer, r1

Review date: 2026-08-12. Scope was offline/read-only inspection of
`docs/generation-frame-latency-plan.md`, the r2/r3 plan reviews,
`tools/qualification/stratified_latency_analyzer.py`,
`tests/runtime/test_stratified_latency_analyzer.py`, retained artifacts, and
the actual playout path in `src/nemotron_voicechat_pipecat/llm.py`. I did not
connect to, probe, restart, or otherwise touch the running stack.

The requested `scratchpad/step0-impl-done.md` is absent from this checkout and
was not found anywhere under `/home/khkramer`. Consequently, the three claimed
ambiguity resolutions could only be assessed from the implementation's module
comments, report labels, and behavior; their original rationale is not
auditable.

## Verdict

**FAIL.** The basic arithmetic is mostly competent: nearest-rank percentiles
are correct, the literal `t[i+11] - t[i]` indexing is correct, and the simple
threshold/done-flush replay reproduces the retained A6 anchors. Those successes
mask normative errors. High-mode populations are wrong, function-cycle rows
are knowingly included in `idle`, the structural gate can select the wrong
event clock, server-only traces receive invented completion events and
unqualified normative-looking verdicts, and interruption does not model the
actual buffer clear/release burst. `verify-anchors` passes all 38 checks while
at least one retained non-anchor artifact is misreported.

## 1. Population predicates do not implement the schema

### High-mode strata are overbroad

`classify_populations()` applies the EarTTS >= 18 ms and residual >= 10 ms
predicates to every model row. The plan's A3 population is
**delivered-nonBOS**, with BOS always separate. The test at lines 54-122
explicitly locks in the wrong behavior: its expected `EarTTS-high` count of 3
includes a BOS row, a delivered non-BOS row, and an idle row.

Concrete counterexamples:

| Row | Schema membership | Analyzer membership |
|---|---|---|
| `audio_delivered=false`, `response_id=null`, EarTTS = 18 ms | `idle` | `idle`, `EarTTS-high` |
| delivered row with `eartts_reset={...}`, EarTTS = 80 ms | `delivered`, `BOS` | `delivered`, `BOS`, `EarTTS-high` |
| delivered non-BOS row, residual = 10 ms | `delivered`, `delivered-nonBOS`, `residual-high` | same (this boundary is correct) |

This is observable on retained data. On `209fe177…`, the analyzer reports 11
EarTTS-high rows; the independent delivered-nonBOS predicate yields 10. On
`2a0fa2ae…`, it reports 5 EarTTS-high rows, whereas only 1 is delivered non-BOS;
the four extras are three reset/BOS rows and a non-delivering function-cycle
row.

### `idle` contradicts the explicit function-cycle exclusion

The plan says both `response_id == null` and “function-cycle pseudo-steps are
excluded and counted separately.” The implementation chooses the literal
anchor count instead: a function-cycle row is appended to both `idle` and
`function-cycle`. This is disclosed in the module docstring and
`population_relationships`, but disclosure does not make it conformant.

The aged anchor demonstrates the consequence:

| Predicate | n | server-step mean |
|---|---:|---:|
| literal `response_id == null` | 2,454 | 70.626775 ms |
| excluding the 30 function pseudo-steps | 2,424 | 71.469253 ms |

Thus the published aged idle anchor and the normative prose are inconsistent.
Step 0 must not silently resolve that conflict in favor of the anchor. Either
the spec/anchor must be corrected or the report must carry two distinctly
named populations (for example, `response-null` for legacy reproduction and
normative `idle` with function-cycle excluded). The unqualified name `idle`
must follow the spec.

There is a second legacy/new-schema hazard: server `model_step` delivery is
classified from positive `output_audio.bytes`, ignoring canonical
`audio_delivered`. A new row with retained PCM bytes but
`audio_delivered=false` is classified delivered even though the spec's exact
predicate says it is not. Current retained server traces already carry
`audio_delivered`; use it when present, and allow any bytes-based legacy
fallback only as an explicitly degraded, schema-versioned mode.

### BOS is not defined tightly enough for prepared reuse

`is_bos()` treats any non-null `eartts_bos_epoch_step` as BOS and the module
docstring says this means EarTTS “actually reset.” Retained prepared-reuse rows
counterexample that claim: `eartts_reset` is null, backend generation is
unchanged, `reset_count_after == reset_count_before`, and only the reuse and
transition counts advance. If the desired stratum is every EarTTS BOS
transition, the code is plausible but the plan's “performed the
BOS/abort-prefill reset” wording and the docstring are false. If the desired
stratum is reset-cost rows, the classifier is wrong. Resolve this explicitly
and, if both analyses matter, report `BOS-transition` and `abort-prefill-reset`
separately.

## 2. The structural gate can silently use the wrong clock

The schema defines `arrival` as receipt time of
`response.output_audio.delta`. `_structural_timestamps()` instead prefers
`client_received_monotonic_s` from delivered `voicechat.metrics` rows whenever
those rows exist, even when exact audio-delta receipt events are present.
Those are different events and need not have the same delay or ordering.

A 12-frame counterexample with metrics events 80 ms apart and audio deltas
100 ms apart produces:

| Input selected | `t[11]-t[0]` | Verdict |
|---|---:|---|
| analyzer-selected metrics-event clock | 880 ms | ACCEPT |
| normative audio-delta receipt clock | 1,100 ms | REJECT |

The retained preflight happens to hide this error because its two streams are
only tens of microseconds apart. Even there, the anchor verifier checks the
metrics-derived 1009.957 ms rather than the delta-derived 1009.934 ms. The
verdict happens to be unchanged; the boundary selection is still wrong.

For server-only traces the tool falls back to `model_step.monotonic_s` and
still emits an unqualified `ACCEPT`/`REJECT`. The r3 review permits that clock
only for historical gate-shape evidence and explicitly forbids substituting it
for arrival-clock promotion evidence. A clock string in the report is useful
but insufficient: normative status and verdict eligibility must be explicit.

Required behavior:

1. Use audio-delta receipt timestamps for the normative structural gate.
2. Reconcile delta count/order/response IDs with delivered logical frames, and
   fail closed or mark the result unavailable when mapping is not provable.
3. Report metrics-event and server-emit calculations only as named
   diagnostics with `normative: false`; never issue a qualification `ACCEPT`
   from them.

## 3. Release-aware replay is only partially faithful

The core recovering-queue arithmetic is correct for the simple model used by
A6: audio does not drain before threshold release; after release each gap
drains the queue; variable chunk duration is added; deficit becomes starvation;
and a short response can flush at `response.done`. The reproduced A6 sweep is
therefore not evidence for the missing boundary semantics below.

### Server traces fabricate `response.done`

For every inferred server response, `extract_queue_responses()` sets
`done_time_s` to the final delivered row. There is no completion event in that
trace and the last delivered PCM is not a completion boundary. On a one-delta
40 ms response with a 160 ms reserve, the analyzer will report an immediate
`release_reason: "response.done"` and zero added delay. The real done event
could occur much later; it is unknowable from that artifact.

The fallback is mentioned in a source comment and the overall clock label, but
the fabricated per-response field and release reason are not honestly labeled.
Do not invent a done time. A below-threshold server-only response must be
`done_flush_unknown`/non-normative (or unavailable), while threshold-crossing
replay may remain an explicitly server-emit diagnostic.

### Interruption is modeled at only one ordering point

The sweep's ordinal means “immediately after this arrival has been fully
processed.” It does not cover just-before-arrival, just-before threshold
release, or partial progress through the release burst. Before release, the
real service calls `_clear_playout_buffer()`, leaving zero service-held media
and resetting `_playout_started`; the analyzer breaks with the just-arrived
media still in `queue_ms_at_stop`. At/after release it may already have marked
the entire reserve threshold-released without distinguishing which frames have
actually crossed the awaited downstream pushes.

This matters because `_release_playout_buffer()` starts TTS and pushes every
buffered frame one by one through awaited downstream calls. Interruption can
therefore race with the threshold arrival and with individual frames in that
burst. The plan and r2 review expressly call for just-before-release,
at-release, and after-each-downstream-frame coverage. The current arrival-only,
post-processing sweep cannot answer barge-in tail or released-frame exposure.

Add an explicit event-order state machine with at least pre-arrival,
post-buffer/pre-release, and post-release-frame interruption points. Clearing
must zero held media; released downstream media must be accounted separately
from service-held media. Do not label simulated starvation episodes as browser
`underrun_events` until browser enqueue/dequeue evidence or an explicit
simulation-only label exists.

### “Exact samples” accepts wire data the real service rejects

Payload bytes are counted exactly, which is good. However, the analyzer
silently defaults a missing sample rate to 22,050 Hz and does not validate
`encoding` or `channels`. `llm.py` calls `events.decode_audio_delta()`, which
requires exactly PCM16, 22,050 Hz, mono. A stereo, wrong-encoding, or missing
metadata artifact can therefore produce a plausible but wrong queue duration
instead of failing closed. Validate the same wire contract before replay. Also
reject/reconcile stale deltas that the real service would ignore for a
non-current or cancelling response.

## 4. The three ambiguity resolutions are not release-quality

| Resolution inferred from code | Sound? | Honestly labeled? |
|---|---|---|
| Idle/function-cycle overlap to preserve n=2,454 | **No** — contradicts the normative exclusion and changes the mean materially | **Mostly yes** in the module/report, but the public population remains misleadingly named `idle` |
| Structural clock prefers delivered metrics, else server emit | **No** — metrics receipt is not audio-delta arrival, and server emit is historical-only | **Partly** — the clock string is present, but an unqualified gate verdict implies normative eligibility |
| Server final delivered row stands in for done flush | **No** — completion timing is unobservable | **No** — downstream results claim `response.done` even though none exists |

The missing scratchpad is itself a provenance defect. Restore the decision
record or move the complete rationale and normative/degraded status into a
tracked spec/review before accepting Step 0.

## 5. K=12 and nearest-rank probes

The implemented window enumeration is correct:

- n=11 -> 0 windows;
- n=12 -> 1 window, `t[11]-t[0]`;
- n=13 -> 2 windows;
- n=24 -> 13 windows.

The 960 ms inclusive boundary and the adjacent-gap check are also correctly
separated. The tiny `1e-9` ms numerical tolerance is immaterial at the retained
timestamp precision.

One fail-closed issue remains: n<12 returns `verdict: "ACCEPT"` despite testing
zero throughput windows. The universal predicate is mathematically vacuous,
but a qualification analyzer must not present absence of evidence as a passing
gate. Return `INSUFFICIENT` (or an explicit non-eligible status) unless the
declared fixture supplies at least one K=12 window; apply analogous fixture
coverage requirements outside this helper as appropriate.

`nearest_rank()` correctly implements `ceil(p*n/100)-1`; an independent exact
rational-index sweep through n=5,000 found no off-by-one errors for the reported
percentiles, including 2.5, 50, 95, 97.5, and 99. The current test is weak,
though: p95 over n=4 always selects the maximum and does not distinguish many
wrong implementations. Add n=20 (rank 19) and n=21 (rank 20) boundary probes,
plus the corresponding p99 boundary.

## 6. Non-anchor retained-artifact check proves anchor overfitting

`verify-anchors` reports `PASS (38 anchors)`. Independently computing
`209fe177-bbb8-403d-ba50-740d3b6f3bf4` with `jq`/`awk`, without calling analyzer
helpers, gives:

| Quantity | Independent | Analyzer |
|---|---:|---:|
| delivered n / mean | 55 / 84.763491 ms | 55 / 84.763491 ms |
| delivered p50 / p95 / p99 | 83.132 / 94.383 / 97.988 ms | same |
| delivered-nonBOS n / mean | 54 / 84.518593 ms | same |
| BOS | 1 | 1 |
| delivered-nonBOS EarTTS-high | 10 | **11 reported under `EarTTS-high`** |
| delivered-nonBOS residual-high | 2 | 2 |
| server-emit K=12 windows / failures | 44 / 34 | 44 / 34 |
| server-emit worst window / max gap | 995.004 / 97.390 ms | same |

So the arithmetic generalizes, but the population semantics do not. The tool
also calls the server-emit result `REJECT`, although only an arrival-clock
result is normative.

The verifier is too narrow to catch this:

- it checks no high-mode counts or membership hashes;
- it checks no BOS/function-cycle partition invariants;
- it checks no expected source SHA-256 values, so aggregate-compatible source
  substitution can pass;
- its default aged path is a transient absolute `/tmp/claude-1000/...`
  scratchpad path, not a durable retained fixture;
- it enshrines the metrics-event structural timestamp and the overlapping idle
  anchor.

Move the aged input to a durable retained location or require a manifest,
verify expected source and membership hashes, and add semantic anchors that
exercise every population and both normative clocks. Anchor reproduction must
not override schema conformance.

## 7. Test quality

All 10 tests pass, but they primarily pin the implementation's chosen happy
path. Several are valuable: the queue recovery example, short done flush,
literal K=12 boundary, no cross-response windows, cumulative prefix debt, and
non-interpolated p50 do constrain real semantics. The suite nevertheless
mirrors the bugs in the critical areas:

- the population test explicitly expects idle/function overlap and BOS/idle in
  `EarTTS-high`;
- no server row tests canonical `audio_delivered` against retained bytes;
- no prepared-reuse BOS row distinguishes transition from actual reset;
- no test makes metrics-event and audio-delta clocks disagree;
- no test reconciles delivered logical-frame count/order with delta count/order;
- interruption is only post-arrival and asserts neither buffer clearing nor
  downstream release exposure;
- the exact-sample test omits the `encoding` and `channels` fields required by
  the real decoder, thereby rewarding analyzer permissiveness;
- no short server-only response proves done-flush is unknowable;
- no n<12 gate-eligibility or strong nearest-rank boundary test exists;
- no retained non-anchor artifact or `verify-anchors` integrity test is run.

## Required fixes before PASS

1. Restrict EarTTS-high and residual-high to delivered-nonBOS and keep BOS
   disjoint; add counterexample tests.
2. Make normative `idle` exclude function-cycle pseudo-steps. Resolve the aged
   anchor inconsistency in the spec, or retain a separately named legacy
   response-null population.
3. Use canonical `audio_delivered` when present; make legacy bytes inference
   explicit and degraded. Clarify/split BOS transition versus actual reset.
4. Select `response.output_audio.delta` receipt timestamps for normative
   structural results, reconcile them to logical delivered frames, and make
   metrics/server clocks diagnostic-only with no qualification verdict.
5. Remove fabricated server `response.done`; mark below-threshold done-flush
   outcomes unknowable. Preserve provenance on every replay result.
6. Model interruption ordering and real buffer clearing/release exposure, and
   align audio metadata validation with `events.decode_audio_delta()`.
7. Return an insufficient/non-eligible structural status when no K=12 window
   exists.
8. Harden `verify-anchors` with durable paths, expected source and membership
   hashes, and semantic/non-anchor checks; replace the tests that currently
   bless wrong populations and permissive wire data.
9. Restore or supersede the missing ambiguity-resolution record so all three
   choices and their normative status are reviewable.
