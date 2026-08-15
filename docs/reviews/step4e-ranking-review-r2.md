# Step 4e candidate-ranking final review — round 2

Review date: 2026-08-13 UTC

**Verdict: PASS - STEP 4 CLOSED.** All three round-1 document deltas are
closed, the ranking table retains its evidence and authority boundaries, the
spot-checked numbers match their admitted sources, and every requested offline
gate passes.

This review was offline. I made no health query, model or browser connection,
container operation, live probe, or commit. The repository was read-only except
for this review file; pre-existing worktree changes were preserved.

## Round-1 delta closure

1. **Two-frame perception batching is now represented.** The candidate table
   has an explicit `NR` row with `retained-negative-adjacent, then blocked`
   standing. It distinguishes serial two-frame batching from the retained
   concurrent-overlap negative control, states correctness scope and reliability
   risk, preserves the common promotion contract, and gives an exact unlock
   without inventing ranking authority.
2. **Pairing recovery option (c) now matches the normative closure.** It requires
   supervised worker recreation with proven clean reinitialization, explicitly
   including that no stale state or client lock survives, single-session failure
   containment, and a service-recovery SLO. The remaining tail-state-machine and
   counterbalanced A/B requirements are retained.
3. **Evidence and judgment are now separated.** The evidence section limits the
   reserve conclusion to the tested grid and expressly does not generalize to
   every larger or adaptive reserve. The primary-solution preference appears
   only under judgment. The Nano envelopes now permit investigation without the
   evidence itself selecting the highest-information prerequisite.

## Number-drift spot-check

No drift was found. I checked the values in the new or moved passages and a
cross-section of every quantitative evidence family against the admitted
sources:

- the batching negative control remains **+1.076 ms/logical frame**;
- the pairing association remains **11.1–11.3 ms/frame**;
- the A3 Nano and EarTTS ranges remain **52.29–52.79** and
  **12.15–12.40 ms/frame**;
- the EarTTS-high inventory remains **10** rows, **20.4 versus 10.86 ms**, a
  **9.54 ms** difference, with current exposure **0 / 764**;
- the logging associations and conditional intervals remain **+1.479 ms
  [−0.758, +3.489]** and **+0.988 ms [+0.058, +1.851]**;
- the corrected Nano decomposition remains content **53.560786 / 45.847206 /
  7.713580 ms (85.608% / 14.392%)** and PAD **53.595332 / 45.579065 /
  8.016267 ms (85.053% / 14.947%)**;
- the graph prerequisite remains **687 calls**, the wrapper screening floor
  remains **1.5 ms/frame**, and the common cadence floors remain **12 frames /
  960 ms** and **160 ms** maximum gap; and
- the moved reserve evidence remains **560 ms**, **3 of 8** responses starving,
  and **499.0 ms** mean added release delay; the retained JSON value is
  499.0167216164991 ms and therefore rounds as reported.

The table still has no row with `measured` standing and makes no unauthorized
promotion, winner, or implementation claim.

## Offline validation

- `.venv/bin/pytest -q -p no:cacheprovider -m 'not browser_e2e'`:
  **1171 passed, 15 skipped, 3 deselected, 29 subtests passed** in 33.57 s.
- `.venv/bin/python tools/qualification/stratified_latency_analyzer.py
  verify-anchors`: **PASS (67 anchors)**.
- `.venv/bin/ruff check --no-cache .`: **All checks passed!**

Pytest emitted four non-failing warnings, including the previously documented
unawaited `FrameProcessor.__process_frame_task_handler` warning in
`test_local_interruption_discards_incomplete_sentence_tail`. No requested gate
failed and no live test was run.
