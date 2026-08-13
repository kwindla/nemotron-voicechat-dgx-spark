# Step 2 live fixture driver review, round 8

Review date: 2026-08-13 UTC. Scope: live-failure fix 7 in the Step 2
fixture-driver response correlator, checked against the current `llm.py`
producer; the retained live regression and SHA pin; the analyzer consumer;
fresh adversarial traces; and the requested offline checks. No live-stack or
browser-E2E command was run.

## Verdict

**FAIL - the correlator fix is not complete at the driver-consumer boundary.**

The new correlator branch accepts the producer's post-threshold immediate
pushes and correctly rejects the requested pre-release, post-clear, duplicate,
and unaccounted cases. However, `_validate_response_attribution()` still
requires release membership to cover every accepted delta and every downstream
push. It therefore rejects the exact valid attribution that fix 7 newly
produces. A completed live fixture using the observed producer semantics can
still fail terminal summary validation.

## Findings

### 1. Blocking: completed-response validation retains the original release-coverage defect

`correlate_response_lifecycles()` now returns the right representation for a
threshold release followed by immediate pushes: all accepted ordinals appear
in `audio_delta_ordinals` and `downstream_push_ordinals`, while only buffered
ordinals appear in `release_membership`. This matches `llm.py`.

The downstream validator does not accept that representation:

- `tools/qualification/step2_live_fixture_driver.py:1826-1854` reconstructs
  accepted and pushed ordinals only from `release_membership`.
- Lines 1855-1860 require that reconstruction to equal all deltas and all
  pushes.
- A post-release ordinal is intentionally absent from `release_membership`, so
  this check necessarily raises `ValueError` with `primary response audio
  membership fields disagree`.

This validator is invoked for every completed result at line 2293, before the
completed-summary artifact reconciliation, so the mismatch is on the live
terminal path rather than an unused helper.

I reproduced this directly with the SHA-pinned retained artifact: correlate
the artifact, pass the resulting attribution to
`_validate_response_attribution()`, and the latter rejects it with that exact
error. The checked-in positives at
`tests/runtime/test_step2_live_fixture_driver.py:701-779` stop at the
correlator and do not exercise completed-summary validation with immediate
pushes.

This is a release blocker for fix 7: update the response-attribution consumer
to account for both release-owned ordinals and post-release immediate pushes,
then add a completed-summary regression using that representation.

### 2. Correlator-only ordering remains weaker than the producer

Two fresh negative traces are accepted by `correlate_response_lifecycles()`:

- a post-release delta whose downstream push occurs after `response.done`;
- two post-release deltas whose pushes occur in reverse ordinal order.

Both are impossible in the traced producer. The receive loop awaits each
event handler (`llm.py:1781-1789`); the accepted delta is recorded at lines
1889-1903; and, once playout has started, that same handler awaits its push at
lines 1912-1917 before returning. The push record itself is emitted at lines
1938-1949. `response.done` is a later server event and is traced before its
done-flush at lines 1832-1840 and 1853-1867, so an already-started ordinal
cannot first be pushed after that terminal.

The correlator enforces only each immediate push's individual
delta-before-push relation (`step2_live_fixture_driver.py:1356-1370`). It does
not enforce push-before-terminal or the producer's per-delta FIFO/immediate
ordering. The canonical analyzer does enforce both with its `pending_push`
queue and atomic transition check.

This is secondary to finding 1 because the terminal artifact is also checked
by the analyzer, but it means the correlator still does not model `llm.py`
line-for-line as requested.

## Producer and exactly-once audit

The core fix-7 rules themselves are sound:

- Before playout starts, `llm.py:1919-1926` buffers a frame and the release at
  lines 1928-1936 names every detached buffered ordinal before pushing it.
  The correlator requires release membership for such pushes; my fresh
  never-buffered pre-release push was rejected.
- After the first threshold release, `llm.py:1912-1917` records no new release
  membership: it starts/retains playout and immediately pushes the accepted
  frame. The positive synthetic trace and the retained live trace both
  correlate.
- Interruption clearing records only the currently buffered frames at
  `llm.py:1972-1990`, after the interruption edge at lines 1992-2002. The
  correlator's intervening-clear test rejected a post-clear immediate push.
- `step2_live_fixture_driver.py:1310-1311` requires the multiset of pushed plus
  cleared ordinals to equal the contiguous accepted deltas; lines 1322-1336
  separately reject repeated release/clear membership and repeated pushes.
  Fresh duplicate-push, duplicate-release, and missing-push cases were all
  rejected. Thus correlator-level exactly-once accounting holds for the
  canonical validated record schema.

## Retained artifact and SHA pin

The retained regression is present at
`reports/step2-live/session-20260813T060844Z/driver/pipecat-playout-0001.jsonl`.
Its SHA-256 is exactly
`acea2107db42d664fb0c9997d62fc8a42998f64bd6be9e2e8c29c6af5654b23b`.

It correlates cleanly as one completed response with:

- 67 contiguous accepted audio-delta ordinals;
- one threshold release containing ordinals 1-2;
- 67 downstream pushes.

The regression test reads the artifact bytes and asserts that literal digest
before invoking the correlator (`test_step2_live_fixture_driver.py:755-779`).
A missing or changed artifact therefore fails the focused suite; the pin is
enforced rather than documentary.

## Analyzer-consumer audit

The claim that `validate_observed_playout_trace()` never shared the defect is
confirmed. I independently constructed a schema-complete 160 ms prebuffer
trace with two 80 ms buffered deltas, their threshold release and pushes, then
a third delta followed directly by its downstream push. The analyzer accepted
it.

Fresh in-memory mutations were rejected as follows:

- removing the threshold release: `orphaned or reordered downstream push`;
- duplicating the post-release push: `orphaned or reordered downstream push`;
- inserting interruption-clear between the third delta and its push:
  `playout release/push ordering is not atomic`.

This follows its existing state machine: post-threshold deltas enter
`pending_push` at `stratified_latency_analyzer.py:1519-1536`; any intervening
non-push is rejected at lines 1468-1473; and the FIFO ordinal is consumed
exactly once at lines 1637-1649. No analyzer change is needed for this defect.

## Offline validation

- Focused driver, analyzer, and producer-trace suites: **162 passed** (the
  handoff's 134 driver/analyzer tests plus 28 producer trace tests), with one
  dependency deprecation warning.
- Canonical retained verifier: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff over `llm.py`, both qualification consumers, their runtime
  tests, and the producer trace tests: **passed**.

No browser-E2E test, live URL, service endpoint, container, or running-stack
operation was used. No commit was created.
