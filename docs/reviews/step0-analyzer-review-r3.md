# Adversarial review: Step 0 stratified latency analyzer, r3

Review date: 2026-08-12. Spec of record:
`docs/generation-frame-latency-plan.md` at r3.2, plus the five required fixes
in `docs/reviews/step0-analyzer-review-r2.md`. Scope was read-only inspection
and execution of the analyzer, its tests, retained artifacts, and the actual
playout path in `src/nemotron_voicechat_pipecat/llm.py`. I wrote only this
review and temporary corrupted fixture copies under `/tmp`; I did not connect
to, probe, restart, or otherwise touch the running stack.

## Verdict

**PASS. Step 0 is complete.** All five r2 deltas are closed. Fresh
counterexamples with different response IDs, timings, sample counts, reserve
sizes, and event sources agree with `llm.py`; the literal r3.2 populations
reproduce on both legacy server traces; real source and semantic membership
corruption makes the anchor verifier exit nonzero; all 67 canonical anchors
and the full repository test suite pass. No new blocker emerged.

## Disposition of the five r2 required fixes

| # | Required fix | Result | Fresh evidence |
|---:|---|---|---|
| 1 | Replay observed interruption/clear in order; do not resurrect cleared media at done | **PASS** | A 1,102-sample (49.977 ms) prefix at a 170 ms reserve, followed by typed clear and then observed done, clears 49.977 ms, releases 0 ms, remains `released: false`, and reports `interruption-clear`. A separate already-released 125.034 ms burst drains to 56.034 ms at the observed clear and is not flushed again by done. |
| 2 | Add at-release whole-burst exposure and drain elapsed playout before pre-arrival | **PASS** | An unequal three-frame burst totaling 155.011 ms, interrupted post-detach/pre-first-push at a 150 ms reserve, reports 0 ms released at interruption and all 155.011 ms exposed afterward. A separate 120 ms release interrupted 75 ms after its last arrival reports 45.000 ms downstream, with only the first two deltas processed. |
| 3 | Only typed `speech_started` invokes cancellation | **PASS** | `source: "server-vad"` followed by a valid 517-sample current delta records one accepted delta, no clear, and zero cancelling/stale ignores. Typed events still clear and suppress later cancelling deltas before decode. This matches `llm.py`'s `source == "typed"` guard. |
| 4 | Make `response-null` literal and apply the r3.2 idle predicate on legacy traces | **PASS** | A fresh four-row legacy matrix keeps a delivered row with an inferred response group in literal `response-null`, but excludes it from `idle`; it also excludes a function-cycle row. Independent raw-field evaluation of retained traces gives `209fe177…`: response-null 245, null-delivered 55, idle 190; `2a0fa2ae…`: response-null 490, null-delivered 134, function-cycle 30, idle 326. These agree with the analyzer and r3.2. |
| 5 | Complete semantic membership anchors and negative integrity coverage | **PASS** | The verifier pins BOS-transition, abort-prefill-reset, and residual-high membership. On a copied anchor tree, the pristine copy passes; a whitespace-only source mutation exits 1 on `preflight.source.sha256`; changing the first delivered BOS row exits 1 on source, BOS-transition, and abort-prefill-reset membership (and the affected idle count); lowering the retained residual-high row below 10 ms exits 1 on source and residual-high membership. |

## Ordered replay check against `llm.py`

The analyzer now preserves the runtime's relevant order:

1. `llm.py` accepts `input_audio_buffer.speech_started` for this path only
   when `source == "typed"` (lines 1006–1008).
2. `_reset_response_after_interruption()` clears the service playout buffer
   before marking the response cancelling (lines 1170–1176). The extractor
   records that clear with event order and receipt time; replay drains elapsed
   downstream media, clears only service-held frames, and stops before a later
   done flush.
3. `_release_playout_buffer()` detaches the whole buffered list before its
   first await/push (lines 1251–1258). The explicit
   `at-release/post-detach/pre-first-push` point therefore correctly exposes
   the whole detached burst after interruption, while already-pushed and
   still-service-held media remain separately accounted.
4. `_response_done()` may call release, but after an ordinary typed clear the
   service buffer is empty (lines 1265–1288); the analyzer no longer
   resurrects the cleared prefix.

The implementation's pre-arrival point advances queue time before recording
the interruption, so it now denotes a true interruption immediately before
the next arrival rather than immediately after the preceding one.

## Anchor-integrity verification

The corruption exercises used a complete copied fixture tree at
`/tmp/step0-r3-integrity.9ZUJOv`:

| Copy state | Exit | Result |
|---|---:|---|
| Untouched copy | 0 | PASS |
| Source bytes changed without changing parsed events | 1 | Source SHA mismatch |
| Delivered BOS membership changed | 1 | Source, BOS-transition, abort-prefill-reset, and affected semantic-count failures |
| Residual-high membership changed | 1 | Source and residual-high membership failures |

This confirms the CLI failure path, not only the unit-level comparison helper.
Canonical verification also passes:

```text
python3 tools/qualification/stratified_latency_analyzer.py verify-anchors \
  --output /tmp/step0-r3-canonical-anchors.json
verify-anchors: PASS (67 anchors)
```

## Tests

```text
.venv/bin/pytest -q tests/runtime/test_stratified_latency_analyzer.py
27 passed in 0.09s

.venv/bin/pytest -q
824 passed, 16 skipped, 4 warnings, 29 subtests passed in 14.24s
```

The warnings are dependency/deprecation warnings plus an existing coroutine
warning in a Pipecat service test; none is from the analyzer tests and none is
a Step 0 regression.
