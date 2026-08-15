# Adversarial review: generation-frame latency plan, r2

Review date: 2026-08-12. Scope was offline/read-only inspection of r2, the r1
review, retained model and qualification traces under
`~/.local/state/nemotron-voicechat/`, the pair-era `f110f063…` trace, the new
`sustained-aged-26h` artifact, repository code, and the retained Step-5/Step-8
reports in the adjacent source-artifact tree. I did not inspect or change the
running stack, connect to the model WebSocket, or run a live probe.

## Verdict

**NOT-READY.** R2 is a substantial correction: it reverses the false PAD-pair
story, uses output-arrival intervals and a recovering queue, makes the watchdog
an explicit dependency, mines the already-retained stage timings, and correctly
retires the Step-5 and Step-8 revivals. The A6 replay table also recomputes
exactly.

Four blockers remain:

1. **The Step-0 oracle still mixes populations and clocks.** “Delivered
   generation” denotes 34 delivered rows in one place, an aged 184-row
   post-audible/window population in another, and 188 non-BOS rows in A3. The
   comparison table also mixes `server_step_ms` with `inference_ms`. The idle
   comparison uses different predicates in its two arms. An analyzer cannot
   reproduce the gate from the definitions r2 gives it.
2. **The structural p95 target rejects the leading candidate by design.** The
   pair baseline has delivered-generation `server_step_ms` p95 **118.274 ms**, not
   at most 80 ms, even though its fast/slow bursts recover and its retained
   output-arrival peak debt is only about 4.53 ms. Keeping both the per-physical-
   call p95 gate and PAD pairing as the leading fix is internally contradictory.
3. **The Step-2 watchdog fix has no complete liveness contract.** Pausing the
   no-text counter on audible decoded audio fixes the observed truncation, but
   continuous audible non-text output defeats both the no-audio and decoded-
   silence watchdogs. The claim that those watchdogs preserve all dead-turn
   coverage is false without another bounded audio-only/max-response rule.
4. **The example PAD-pair recovery precondition is not recovery.** The retained
   retirement analysis says a wedged packed vLLM/CUDA request cannot be
   transactionally cancelled and resumed and explicitly forbids an inference
   timeout/retry. A bounded per-call timeout followed by “session-fatal,
   never-retry” does not make the engine, output iterator, recurrent state, or
   single-client lock healthy. Either remove the packed request mechanism or
   prove process/worker teardown and clean reinitialization; tail restriction
   only lowers exposure and does not close the failure mode.

The concrete closure conditions are in the final section.

## 1. R1 finding-by-finding closure

Verdicts here mean **CLOSED** (r2 corrects or withdraws the claim), **PARTIAL**
(the direction is corrected but the replacement still has an evidence or
contract defect), and **OPEN** (the r1 issue remains materially present).

| R1 # | R2 status | Verification against primary artifacts |
|---:|---|---|
| 1 | **CLOSED** | R2 limits the claim to the current sequential image and explicitly shows the pair-era exception. The retained counts remain 34/34 over 80 ms in the sequential preflight, 52/55 in `209fe177…`, and 13/32 in `f110f063…`. |
| 2 | **CLOSED** | The invalid 59 ms idle endpoint is removed and the function-cycle/workload-composition confound is acknowledged. |
| 3 | **CLOSED** | The preflight aggregate remains correctly reported as informational (2,729 rows, mean 71.131 ms, p95 73.985 ms) and aggregate p95 is demoted. |
| 4 | **OPEN** | Step 0 names distinct populations, but the plan and its reproduction gate still conflate them. Aged delivered rows are n=193, delivered non-BOS rows are n=188, while the n=184/82.79 figure is a different response-window classifier. A2 also compares aged `response_id == null` with fresh `audio_delivered != true`. Details are below. |
| 5 | **CLOSED** | R2 correctly states that eligibility is based on the previous effective PAD and that pairing operated in the generated tail. Frames 63–87 of `f110f063…` show the expected entry into PAD followed by the fast/packed alternation. |
| 6 | **PARTIAL** | The whole-workload p99 argument is withdrawn and the approximately 11 ms association is shown. However, A4 mixes timing fields in its rounded values and calls a historical, separately run comparison a causal “regression from retirement.” Use one field and describe the estimate as the closest retained configuration comparison unless a counterbalanced A/B is supplied. |
| 7 | **PARTIAL** | R2 admits that the live environment was not inspected, but “the deployed prebuffer is the repo default 160 ms” is still self-contradictory. Only “the repository default is 160 ms; deployed value unknown” is supported by `llm.py:130,168-178`. |
| 8 | **CLOSED** | The missing fallback/load exclusion is no longer used. R2 instead schedules clocks/thermal/memory telemetry and retains zero Step-9 fallback only as trace-local evidence. |
| 9 | **PARTIAL** | The categorical no-uptime-effect claim is withdrawn, the arms are called non-identical, and no statistical test is claimed. “Match within response-to-response variance” is still descriptive evidence from five aged responses versus one fresh response, not a matched experiment, but the conservative “not demonstrated” conclusion is acceptable. |
| 10 | **CLOSED** | R2 no longer claims a gradual 10–15 ms ramp and carries forward the workload-composition explanation. |
| 11 | **CLOSED by withdrawal** | GC is no longer offered as either a demonstrated cause or a proved non-cause. The cheaper abnormal-cleanup audit from r1 is optional hygiene, not a blocker for this plan. |
| 12 | **CLOSED** | The non-recovering sum is explicitly superseded. A6 uses a release-aware recovering queue over `response.output_audio.delta` receipt times; its numbers reproduce. |
| 13 | **CLOSED** | The faulty 450 ms/8 s arithmetic is gone and sizing is trace-replay based. |
| 14 | **PARTIAL** | Upstream Pipecat TTFA and browser audible onset are correctly separated. The replay contract still needs exact short-response `response.done` flush, variable chunk duration, interruption, and downstream-release semantics. A6's “onset delay” is model-client first-delta-to-release delay, not a measured browser audible onset. |
| 15 | **PARTIAL** | Long-response qualification is correctly ordered behind Step 2. The proposed production watchdog behavior is not safe enough to implement yet; see the dedicated finding below. |
| 16 | **CLOSED** | R2 requires browser/output-device barge-in tail measurement rather than relying on the service's unreleased list. |
| 17 | **PARTIAL** | Existing stage timing is now mined before new profiling, and its raw A3 means reproduce. The table is not population-aligned, and the claimed EarTTS “mode” merges BOS, EarTTS-high, and residual-high cases. |
| 18 | **CLOSED** | Step 8 is a negative control, not a revival. The source report confirms +1.076 ms/logical frame and a defended stop. |
| 19 | **CLOSED** | Live perception overlap is deleted as a candidate absent a changed-stack hypothesis. |
| 20 | **CLOSED** | Step 5 is a negative control. The source report confirms seven synchronizations and the +0.747 ms run-mean / +3.962 ms pooled-p95 regression. |
| 21 | **CLOSED** | CFG changes are deferred behind a blinded non-inferiority audio-quality protocol rather than WER alone. |
| 22 | **CLOSED** | R2 acknowledges both the 75–79% PAD share and that the old scheduler already paired the PAD tail. Recomputed token-PAD shares are 27/34 (79.4%) in preflight and 145/193 (75.1%) in the aged probe. |
| 23 | **PARTIAL** | Response/session clustering, BOS splitting, and hardware telemetry are added. The eventual structural A/B still needs multiple independent responses, counterbalanced arm order where possible, and an explicitly identical image/workload apart from the candidate. |

The retained report copies currently present under this repository's `reports/`
are byte-identical to the adjacent source reports: Step 8 SHA-256
`75fa9190…f7db565` and Step 5 `4b7b7ad2…abd105`. A5's statement that they
“currently live only” in the source tree and its copy action are therefore
already stale, though this is not a technical blocker.

## 2. Recomputing the new material

### A6 reserve replay: arithmetic confirmed, interpretation tightened

The six inputs are the 34-delta sequential preflight response and the five aged
responses (34, 40, 43, 42, and 34 deltas). Every delta in this set contains one
80 ms PCM frame. I replayed the actual `client_received_monotonic_s` sequence
with the implementation's threshold-release behavior: at release, the queue
contains `k * 80 ms`; later arrivals consume elapsed playout time, append 80 ms,
and accumulate only time for which the queue was empty.

| Reserve | Recomputed affected | Recomputed starvation | Recomputed mean first-delta-to-release |
|---:|---:|---:|---:|
| 160 ms | 4/6 | 797.184 ms | 116.202 ms |
| 240 ms | 4/6 | 438.789 ms | 202.734 ms |
| 320 ms | 2/6 | 136.074 ms | 289.819 ms |
| 400 ms | 0/6 | 0 ms | 375.563 ms |

These round exactly to A6. The table is a valid preliminary replay, not a
browser result. Three wording/contract corrections are needed:

- 88.9 ms is the preflight's 33-interval mean (88.906 ms), not the six-response
  cadence. The pooled 221-interval mean is **86.589 ms**; response means range
  from 82.508 to 89.043 ms.
- 400 ms is the first clean point on this 80 ms grid for these six truncated
  samples. Longer responses **may**, not necessarily will, require more reserve;
  the required maximum prefix is path-dependent and can recover later.
- The plan's target `max debt <= reserve` is not the same computation as A6.
  Debt accrued before threshold release does not consume a queue that has not
  started playing. The normative gate should run the exact release-aware state
  machine (including early `response.done` flush), not compare a zero-reserve
  global debt scalar with the threshold.

The exact replay should also accept chunk sample counts rather than assume 80 ms,
release short responses on `response.done`, simulate interruption at every
arrival, and report both service release and browser playout. That matters
because `_release_playout_buffer()` pushes the whole reserve downstream in a
burst (`llm.py:1251-1258`); whether it becomes browser reserve is an empirical
transport property.

### A3 stage tables: raw means reproduce, comparison and “mode” do not

The A3 raw means are accurate. The comparable non-BOS populations are:

| Artifact, delivered non-BOS | n | perception | Nano | EarTTS | codec | residual | wrapper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential preflight | 33 | 13.055 | 52.288 | 12.150 | 0.702 | 4.574 | 82.769 |
| Aged sustained | 188 | 12.516 | 52.790 | 12.404 | 0.605 | 3.747 | 82.061 |

R2 instead places the 34-row preflight result, which includes the 84.53 ms
EarTTS BOS/reset, beside aged non-BOS. That is why preflight EarTTS becomes
14.279 ms. It is not a 14.3 ms “steady” value. A3 should publish aligned rows,
with BOS/reset in its own stratum.

The `209fe177…` high set also needs a non-circular classifier. Selecting 13
delivered rows by total/server time at least 88 ms yields wrapper 92.827 and
EarTTS 20.360 ms, as stated, but that set contains:

- one BOS/reset row;
- ten non-BOS rows with EarTTS above 18 ms; and
- two non-BOS residual outliers with EarTTS only 12.028 and 10.889 ms but
  residual about 15 ms.

The other 42 rows have EarTTS mean **10.855 ms**, not the copied 11.7 ms figure.
Step 4 should preregister separate BOS, EarTTS-high, and residual-high predicates
using the stage being diagnosed rather than selecting on the total latency it is
supposed to explain.

### A4 pair versus sequential: direction supported, table clock mixed

For `f110f063…`, the 32 delivered rows have:

- `server_step_ms` mean 75.116 ms, 13/32 over 80 ms;
- `inference_ms` mean 75.039 ms;
- non-BOS `server_step_ms` mean 72.699 ms;
- 12 fast PAD rows at 44.999 ms and 13 other PAD rows at 95.952 ms; and
- delivered `server_step_ms` p95 118.274 ms.

For the sequential preflight, the 34 delivered rows have `server_step_ms` mean
86.168 ms and non-BOS mean 84.024 ms. Thus the r1 server-clock differences are
11.052 ms including BOS and 11.325 ms excluding BOS.

A4's rounded 75.0 uses the pair `inference_ms` value, while its 86.2 uses the
sequential `server_step_ms` value. Use 75.1 versus 86.2 on the server clock, or
publish both fields explicitly. The configuration association is strong—the
whole pair-era run is the 2,718-row, 68.239 ms baseline identified by the
retirement artifact—but the response-only comparison was not a same-process,
counterbalanced A/B. “Strongest measured precedent” is fair; “retirement raised
cost by exactly 11 ms” is too causal.

More importantly, A4 demonstrates why r2's target is inconsistent. Pairing
works by alternating cheap and expensive physical calls. A per-call p95 at most
80 ms rejects that schedule despite its recovering queue behavior. Replace that
gate with logical-media throughput and output-arrival queue metrics, plus a
separate bounded maximum arrival gap/burst rule if individual pauses matter.

### Step 2 watchdog precondition: defect confirmed, fix underspecified

All six cited responses do end with `no_text_since_bos_watchdog`. The aged tool
response proves actual audible truncation: frame 1791 has no-text count 30 while
decoded output is still −47.118 dBFS, and frame 1792 is forced to `agent_eos`.
Other short responses often hit the same watchdog during an already-silent tail,
so “every response truncates audible speech” would be too broad, but the blocker
and the correctness defect are real.

The proposed rule—do not advance the no-text counter while decoded output is
audible—does not preserve the stated dead-turn envelope. Persistent audible
non-speech with no text resets the no-audio watchdog and never satisfies decoded
silence, so all three guards can be bypassed. Before changing production code,
r2 must choose one of two paths:

1. use an explicitly qualification-only override for the long latency fixture,
   while moving the production watchdog repair to its own correctness plan; or
2. specify a two-dimensional liveness contract with a bounded audio-only/max-
   response timer and test BOS+silence, BOS+continuous audible/no-text, long
   legitimate audio after text, prosodic gaps, function-only output, fused and
   post-function BOS, interruption, and model/session caps.

The Step-2 gate must require semantic/audio completeness against a known fixture,
not merely “full transcript”: the present responses already have apparently
complete short text even when the acoustic tail is force-closed.

### Step 5 ranking and preconditions: leading candidate is not yet admissible

The historical latency evidence makes repaired tail pairing worth investigating,
but it does not make it the leading implementable candidate under the current
preconditions.

First, the suggested watchdog is contradicted by the authoritative retirement
document. `docs/nano-step-stall-fix-plan.md:92-97,151-153` states that the packed
request cannot be cancelled/resumed transactionally and forbids timeout/retry.
Timing out the coroutine and closing the session leaves the underlying GPU/
EngineCore call potentially wedged. A valid precondition is one of:

- eliminate the two-position packed vLLM call while retaining a safe amortized
  schedule;
- identify and fix the specific worker/graph/output-wakeup defect, then prove
  deterministic replay and long worker-level stress; or
- isolate the engine behind a supervisor that can kill and recreate the worker,
  prove no stale state or lock survives, fail the affected session once, and
  meet an explicit service-recovery SLO.

Statistical stress alone cannot prove “cannot wedge,” and tail restriction does
not reduce severity. A timeout that cannot restore service must not be listed as
an example of closure.

Second, “tail-restricted” needs an executable state machine: exact entry token
and prior-token predicate, response/boundary phase, exit conditions, pending-row
handling, and rejection/rollback across agent BOS/EOS, function SOTC/EOTR,
post-FC BOS, interruption, abort, and max caps. “Control-boundary rollback proof
on response boundaries” is narrower than the historical scheduler's actual
function/control surface.

Third, Step 4 cannot yet rank the alternatives it names. It profiles EarTTS,
residual/logging, and telemetry, but not the 52 ms Nano interface. Before Marlin
tile/split-K, fusion, or selective precision is ranked against a repaired
scheduler, add a captured content and PAD Nano kernel/CPU-gap profile and an
exact-output component benchmark. Publish an estimated recoverable ceiling for
each candidate, its reliability risk, and the gate it must beat. Otherwise the
plan chooses pairing before the stated decision point has produced its ranking.

## 3. Second creative pass: newly opened or foreclosed risks

1. **The two queue contracts have diverged.** Step 0 asks for zero-reserve maximum
   debt; A6 correctly simulates delayed release with an initial queue; the Target
   compares the former with the latter's reserve. Publish both as diagnostics,
   but make the exact state machine normative.
2. **“Client clock” is not one boundary.** The retained qualification timestamp
   is direct model-WebSocket receipt. The product path adds Pipecat receipt,
   threshold release, transport enqueue, WebRTC/browser receipt, and device
   playout. To reconcile Step 0 with Step 2 on the same session, record response
   ID, ordinal, sample count, and monotonic timestamps at Pipecat receipt and
   release, then AudioWorklet enqueue/dequeue/underrun timestamps in the browser.
   Compare relative intervals or establish clock offset; do not compare unrelated
   host/browser monotonic clocks directly.
3. **The reserve includes leading silence.** The service counts all PCM bytes,
   while Step 0 also proposes an “audible” population. That is correct for queue
   capacity but means audible-frame counts must not drive reserve fill. Report
   playout release delay, browser first-audible delay, and reserve audio duration
   as separate quantities.
4. **Zero underruns needs a declared envelope.** “Zero” in a fixed corpus is a
   qualification result, not evidence for a product percentile. Specify response
   duration bins, network/transport conditions, number of independent responses,
   and the allowed statistical statement. Keep deterministic maximum-duration
   fixtures as correctness tests and clustered samples as performance evidence.
5. **A fixed 400 ms stopgap changes cancellation shape.** R2 correctly adds an
   output-device barge-in gate, but the release burst means the worst tail can
   occur just after threshold crossing. Simulate and test interruptions just
   before release, at release, and after each downstream 80 ms frame—not only a
   single conversational barge-in point.

## 4. Concrete resolution required for READY-TO-IMPLEMENT

1. **Publish one machine-executable measurement schema.** For every table/gate,
   state the predicate, timing field, BOS policy, response window, and percentile
   method. At minimum correct the current anchors to:
   - preflight delivered/server: n=34, mean 86.168 ms;
   - aged delivered/server: n=193, mean 85.403 ms; aged delivered non-BOS:
     n=188, mean 82.919 ms;
   - pair delivered/server: n=32, mean 75.116 ms;
   - fresh `response_id == null` idle: n=2692, mean 70.926 ms, comparable with
     aged n=2454, mean 70.627 ms.
   If the 184-row rendition window remains useful, give it a separate name and
   exact membership predicate rather than calling it delivered generation.
2. **Make the release-aware replay the normative playout gate.** Include exact
   chunk sizes, threshold crossing, `response.done` flush, interruption, and
   downstream/browser timestamps. Rename A6 onset to “added release delay” until
   browser audible onset is measured, and correct the pooled cadence statement.
3. **Replace the incompatible physical-call p95 target.** Gate on 80 ms logical
   media throughput and release-aware browser underflow, with a separately
   justified maximum arrival-gap/seam bound. Demonstrate that the target can
   accept a safe bursty scheduler and reject a genuinely starving one.
4. **Close the watchdog liveness matrix before a production fix.** Either scope
   an override solely to qualification or add an independent bounded audio-only/
   max-response rule and the full boundary tests described above.
5. **Remove the non-recovering PAD-pair timeout example.** Do not rank pairing as
   implementable until the packed-call wedge is eliminated or whole-worker
   recovery is demonstrated. Define the tail state machine and requalify all
   function/control/abort boundaries, then run a same-image queue-clock A/B with
   independent responses and counterbalanced order.
6. **Make Step 4 actually rank Step 5.** Add Nano content/PAD profiling and a
   candidate table containing measured recoverable ceiling, correctness scope,
   reliability risk, and preregistered promotion floor. Pairing may lead that
   table after its reliability precondition is satisfiable; it should not be
   declared the winner beforehand.

With those changes, Steps 0–1 and the qualification-only parts of Step 2 are
ready to execute. The stopgap and structural promotion path are not.
