# Adversarial review: Step 0 stratified latency analyzer, r2

Review date: 2026-08-12. Spec of record: `docs/generation-frame-latency-plan.md`
at r3.1, plus the nine required fixes in
`docs/reviews/step0-analyzer-review-r1.md`. Scope was read-only inspection and
execution of `tools/qualification/stratified_latency_analyzer.py`,
`tests/runtime/test_stratified_latency_analyzer.py`, retained artifacts, and
the actual decoder/playout path in `src/nemotron_voicechat_pipecat/events.py`
and `src/nemotron_voicechat_pipecat/llm.py`. I did not connect to, probe,
restart, or otherwise touch the running stack.

## Verdict

**FAIL.** The fix round is substantial and corrects most of r1: high modes,
idle/function-cycle separation, canonical delivery, the two BOS strata, the
normative delta clock, reconciliation failure closure, server-only
`done_flush_unknown`, `INSUFFICIENT`, diagnostic clock labeling, durable
anchors, wire metadata, and simulated-starvation labeling all improved.
The named non-anchor checks and every retained anchor now pass, and the full
repository suite has no failures.

The remaining blocker is not cosmetic. The new lifecycle extractor identifies
cancellation but never applies the corresponding real buffer clear to the
accepted media being replayed. It can consequently claim that
`response.done` released media which `llm.py` had already discarded. The
explicit interruption sweep also lacks the required at-release/pre-first-push
point and places a post-release `pre-arrival` interruption before elapsed
playout is drained. One rewritten test directly blesses lifecycle behavior
opposite to `llm.py`. In addition, `response-null` is not the literal predicate
the r3.1 report claims on legacy server traces, and anchor membership coverage
still omits three semantic populations.

## Disposition of the nine r1 required fixes

| # | Required fix | Result | Evidence |
|---:|---|---|---|
| 1 | Restrict high modes to delivered-nonBOS and keep BOS disjoint | **PASS** | `classify_populations()` gates both high modes on `delivered and not bos_transition` (lines 295–300). Fresh BOS, idle, and threshold-boundary counterexamples partition correctly. `209fe177…` now has EarTTS-high = 10, not 11. |
| 2 | Normative idle excludes function cycles; retain a distinct legacy response-null population | **PARTIAL** | The intended retained checks pass: aged idle is 2,424, and `2a0fa2ae…` is function-cycle = 30 / idle = 326. But classification uses inferred `response_group` (line 287), not literal source `response_id`, while the report claims `response_id == null` (line 1026). On legacy server traces, `response-null` silently excludes delivered rows whose source field is absent. |
| 3 | Canonical `audio_delivered`; explicit degraded bytes fallback; split BOS transition/reset | **PASS** | `_delivery_classification()` honors an existing `audio_delivered` even against positive retained bytes and labels the fallback non-normative. Prepared reuse is BOS-transition but not abort-prefill-reset. Fresh false-delivery/positive-bytes and prepared-reuse probes pass. |
| 4 | Normative delta receipt clock, provable reconciliation, diagnostics cannot qualify | **PASS** | A fresh 12-frame disagreement (metrics every 70 ms, deltas every 105 ms) produces normative delta-clock REJECT at 1,155 ms while metrics reports only diagnostic ACCEPT at 770 ms with `verdict: null`. Fresh count, response-order, sample-count, and missing-sample failures all make reconciliation unprovable. |
| 5 | Do not fabricate server done; preserve replay provenance | **PARTIAL** | A server-only 20 ms response correctly remains unreleased with `done_flush_unknown`; the pair-era normative gate is UNAVAILABLE. But an observed cancellation followed by done is replayed as a done flush of media that the runtime cleared, so observed completion provenance is not enough to make the replay faithful. |
| 6 | Correct interruption ordering/buffer clear/release exposure; match wire contract | **FAIL** | PCM16/22,050 Hz/mono validation is correct, stale/cancelling events are filtered before decode, held media clears to zero, and the detached remainder of a release burst is exposed. Actual lifecycle clears are not fed into replay; at-release is absent; post-release pre-arrival does not drain elapsed media; and any `speech_started` is treated as cancellation although `llm.py` only does so for `source == "typed"`. Details below. |
| 7 | No-window structural result is not ACCEPT | **PASS** | Fresh normative n=11 gives `INSUFFICIENT`, zero windows, and `qualification_eligible: false`; diagnostic n=11 gives only `diagnostic_outcome: INSUFFICIENT` and no verdict. |
| 8 | Harden anchors, add semantic/non-anchor coverage, replace bad tests | **PARTIAL** | `verify-anchors` now has durable paths, five source hashes, ten membership hashes, semantic non-anchor counts, both structural clocks, and 60 passing checks. It does not pin BOS-transition, abort-prefill-reset, or residual-high membership, and no test proves source/membership tampering fails. The interruption/cancellation test still pins the wrong runtime semantics. |
| 9 | Restore/supersede ambiguity-resolution record | **PASS** | The r3.1 clock, population, done-flush, wire, and eligibility decisions are now recorded in the plan's normative schema and implementation log (plan lines 18–71 and 276–292); the missing scratchpad is no longer authoritative. |

## 1. Cancellation is recognized but its buffer clear is not replayed

The runtime ordering is unambiguous:

1. `llm.py:_speech_started()` calls `_reset_response_after_interruption()`.
2. `_reset_response_after_interruption()` calls `_clear_playout_buffer()`
   before setting `_response_cancelling` (lines 1165–1178).
3. `_clear_playout_buffer()` removes all held frames and resets playout state
   (lines 1260–1263).
4. Cancelling `response.done` calls `_release_playout_buffer()`, but that
   buffer is empty (lines 1265–1288).

Fresh counterexample: one accepted 882-sample/40 ms delta, a matching delivered
metrics row, then typed `input_audio_buffer.speech_started`, then cancelled
`response.done`, at a 160 ms reserve. The analyzer reports:

| Quantity | Analyzer | `llm.py` reality |
|---|---:|---:|
| done observed | true | true |
| release reason | `response.done` | no media release |
| media released downstream | **40 ms** | **0 ms** |

`extract_queue_responses()` sets a boolean `cancelling` and ignores later
deltas (lines 431–464), but it leaves earlier accepted deltas in the response
and records the later done time (lines 486–495). `queue_diagnostics()` then
replays that response as if no actual clear occurred. The separate synthetic
interruption sweep does not repair the base replay or the reserve aggregate.

This can corrupt exactly the barge-in tail/release evidence the r3.1 target is
supposed to provide. Lifecycle observations must become ordered replay events,
not merely a delta filter. On interruption, clear service-held media; retain
already pushed downstream media; account for any detached release coroutine;
and prevent the later done event from flushing the cleared prefix.

## 2. Two explicit interruption points still disagree with real time/order

### No at-release/pre-first-push point

At threshold, `llm.py:_release_playout_buffer()` detaches the whole list before
awaiting `_start_playout()` and each downstream push (lines 1251–1258). An
interruption can occur after detach but before the first audio frame push. The
detached coroutine can then expose the entire burst after interruption.

The analyzer offers `pre-arrival`, `post-buffer/pre-release`, and
`post-release-frame` only (line 602). Its earliest release-burst interruption
is after one frame has already been counted downstream (lines 630–645). For a
two-frame 160 ms threshold burst it can expose at most 80 ms as
`release_burst_after_interrupt_ms`; the required at-release point can expose
all 160 ms. This omits the explicit r3.1 test point “at release” (plan lines
156–159).

The detached-list behavior after a downstream push is otherwise modeled
correctly. A fresh three-frame/240 ms burst interrupted after frame 2 reports
160 ms released at interruption, 80 ms released afterward, and 240 ms total.

### `pre-arrival` occurs before elapsed playback is drained

`replay_response()` checks a pre-arrival interruption at lines 661–663, before
the elapsed-gap drain at lines 666–674. Fresh counterexample: two 80 ms frames
release 160 ms at t=0.08; interruption just before the next arrival at t=0.18.
One hundred milliseconds has elapsed, so downstream queue media at interruption
is 60 ms. The analyzer reports **160 ms**. Its point is effectively
“immediately after the preceding arrival,” not “just before this arrival.”

Either give every interruption point an explicit timestamp and advance the
queue to it, or rename the point to its actual ordering. For the target as
written, the former is required.

## 3. Stale/cancelling filtering does not exactly match `llm.py`

Filtering stale or genuinely cancelling deltas before wire decode is correct.
Fresh malformed Opus deltas for a stale response and for a typed-cancelling
current response are ignored and audited 1/1, while a valid current delta is
accepted. Missing encoding, stereo, 24 kHz, and invalid base64 current deltas
all fail closed, matching the runtime contract.

However, analyzer line 431 treats every
`input_audio_buffer.speech_started` as interruption. `llm.py` lines 1006–1008
call `_speech_started()` only when `source == "typed"`; other server events of
that type are ignored. Fresh counterexample: response created, a
`speech_started` event with `source: "microphone"`, then a valid current delta.
The analyzer accepts zero deltas and audits one cancelling delta; the runtime
accepts the delta.

The rewritten test makes this worse: its cancelling event has no `source`
(`test_stratified_latency_analyzer.py` line 296), yet it expects the following
malformed delta to be ignored. In the actual handler the speech event does
nothing and the malformed delta reaches `events.decode_audio_delta()`, which
raises. This test pins the analyzer implementation, not runtime semantics.

## 4. `response-null` is not literal on legacy server traces

The r3.1 schema says `response-null` is literal `response_id == null`. The
analyzer instead classifies on normalized `response_group`; delivered server
rows with no response field receive an inferred response group at lines
213–217 and are excluded at line 287.

Independent raw predicates show:

| Artifact | Literal null model rows | Analyzer `response-null` | Difference |
|---|---:|---:|---:|
| `209fe177…` | 245 | 190 | 55 delivered rows |
| `2a0fa2ae…` | 490 | 356 | 134 delivered rows |

The requested `2a0fa2ae…` idle value of 326 does pass, because the inferred
response groups also remove those 134 delivered rows before the 30 function
cycles are removed. But that exposes an inconsistency in the record: literal
`response-null` minus function-cycle would be 460, not 326. Preserve the useful
inferred response group for response clustering, but do not use it to claim a
literal source population. A concrete resolution is to make `response-null`
raw/literal and define normative `idle` explicitly as response-null **and
non-delivered** minus function-cycle; otherwise rename the reported legacy
population and update r3.1.

## 5. Delta reconciliation and clock separation withstand fresh probes

The new reconciliation is fail-closed in the tested failure modes:

| Fresh mismatch | Result |
|---|---|
| 2 delivered rows / 1 accepted delta | unprovable: count mismatch |
| response IDs appear in the opposite order | unprovable: response order/sample mismatch |
| equal count/order but one sample count differs | unprovable: response order/sample mismatch |
| delivered row lacks integral `output_audio.samples` | unprovable: explicit ID/sample unavailable |

When reconciliation is unprovable, the normative gate is UNAVAILABLE and
ineligible. When it is provable, only audio-delta receipt timestamps drive the
verdict. Metrics and server-emit calculations have `normative: false`,
`verdict: null`, and a separately named `diagnostic_outcome`. This closes the
r1 clock-substitution defect.

## 6. Retained artifacts and anchors

### Requested r1 non-anchor checks

Independent predicate implementations, without calling analyzer classifiers,
agree with the fixed analyzer on the requested quantities:

| Artifact | Check | Independent | Analyzer |
|---|---|---:|---:|
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | delivered-nonBOS EarTTS-high | 10 | 10 |
| `2a0fa2ae-766d-4455-bfb0-885f67cc958a` | function-cycle | 30 | 30 |
| same | idle under intended non-delivered/inferred grouping | 326 | 326 |

The other corrected `2a0fa2ae…` semantic strata are also sensible: delivered
134, delivered-nonBOS 130, BOS-transition 4, abort-prefill-reset 3,
EarTTS-high 1, and residual-high 1.

### Previously unchecked artifact

I independently checked
`qualification/nano-sequential-production-r1/events.jsonl` (SHA-256
`3d3157480b009d107109594953b07a32231150654e7bb2f1d85d072b8a4bef3b`).
All nine population counts/means agree: delivered 34, delivered-nonBOS 33,
BOS-transition 1, abort-prefill-reset 1, response-null/idle 1,190,
function-cycle 0, and both high modes 0. Its 34 deltas reconcile one-to-one
with 34 delivered logical frames and exact sample counts. Independent arrival
math agrees on 23 K=12 windows, 1,009.394 ms worst window, 164.430 ms maximum
gap, and REJECT.

### Hardened anchor command

`python3 tools/qualification/stratified_latency_analyzer.py verify-anchors`
returns **PASS (60 anchors)**. Independent `sha256sum` checks match all five
pinned sources. The command now pins:

- five source hashes;
- ten exact population membership hashes;
- EarTTS-high = 10 and function-cycle = 30 semantic non-anchor counts;
- the 1,009.934 ms normative delta-clock anchor and the 1,009.957 ms
  metrics diagnostic anchor, including a null diagnostic verdict;
- pair server-emit as diagnostic ACCEPT while normative is UNAVAILABLE;
- both prefix debts and all sixteen A6 reserve cells.

No retained anchor regressed. The remaining hardening gap is coverage:
`BOS-transition`, `abort-prefill-reset`, and `residual-high` have no expected
membership hash in `verify_anchors`, despite r1 asking semantic anchors across
every population. Add them (the `2a0fa2ae…` strata are convenient), plus a
test that substitutes a wrong source hash and a wrong membership hash and
asserts verifier failure.

## 7. Test quality and full repository result

The rewritten tests are materially better. They now pin the intended high-mode
partition, prepared reuse versus reset, canonical delivery, strong nearest-rank
boundaries, observed versus unknown done flush, wire metadata, n<12
eligibility, delta/metrics clock disagreement, and one reconciliation count
failure.

They are not yet release-quality because:

- the stale/cancelling test omits `source: "typed"` and asserts the opposite of
  `llm.py` behavior;
- no test feeds an observed interruption through extraction and base replay,
  so clear-then-done fabrication survives;
- no interruption test covers at-release/pre-first-push;
- the pre-arrival test occurs before release and cannot expose missing elapsed
  drain;
- reconciliation tests cover count only, not response order, sample mismatch,
  or missing explicit sample evidence;
- no server-trace test distinguishes literal `response_id` from inferred
  response grouping;
- no test invokes `verify_anchors` or proves source/membership hash mismatch
  fails.

Commands run:

- analyzer-specific: **20 passed** in 0.09 s;
- full repository: **833 collected; 817 passed, 16 skipped, 4 warnings** in
  13.99 s.

There is no test failure or collateral damage. The warnings were three
dependency/deprecation warnings and one existing runtime coroutine warning in
`test_server_cancelled_terminal_discards_incomplete_tail_without_local_interrupt`;
none came from the analyzer tests.

## Required fixes before PASS

1. Represent observed interruption/clear as an ordered replay event. Clear
   service-held frames in the base replay, keep downstream/detached exposure
   separate, and do not let later `response.done` resurrect cleared media.
2. Add an at-release/post-detach/pre-first-audio-push interruption point and
   advance queue time before a true pre-arrival interruption. Add fresh tests
   for the 160 ms whole-burst exposure and the 60 ms elapsed-drain case above.
3. Match the actual lifecycle trigger: only server `speech_started` with
   `source == "typed"` invokes this cancellation path. Correct the existing
   stale/cancelling test and add non-typed and clear-then-done cases.
4. Make `response-null` truly literal, then explicitly resolve the legacy
   missing-response-ID effect on normative idle (recommended: also require
   non-delivered), or revise the r3.1 predicate/name. Pin the chosen behavior
   with legacy server counterexamples.
5. Complete verifier membership coverage for BOS-transition,
   abort-prefill-reset, and residual-high, and add negative integrity tests.

