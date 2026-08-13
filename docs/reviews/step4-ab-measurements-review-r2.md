# Step 4a+4b measurement-validity review — round 2

**Verdict: FAIL.** Two closure deltas remain.

This review was entirely offline. I did not query health, open a model or
browser connection, run a live probe, or modify the running stack. Apart from
this review file, I made no file change.

## Remaining deltas

1. **The original completion note was not repaired or explicitly superseded.**
   `/tmp/claude-1000/-home-khkramer-src-nemotron-voicechat-dgx-spark/9dab1982-03af-4f77-9d25-ddf7139af66f/scratchpad/step4a-done.md`
   still calls the fixed-order client A/B interval a “bounded null” (lines
   31–36), bases the server follow-up on at least 30 response clusters rather
   than independently assigned run blocks (lines 54–61), and says the
   observational telemetry run deprioritizes thermal/DVFS/memory-pressure work
   while proposing repeated sessions without randomized interventions or
   causal acquisition ordering (lines 62–65). The new
   `step4a-closure-done.md` describes the corrections, but it neither edits nor
   declares that stale completion note withdrawn. Closure conditions 1, 2,
   and 4 therefore remain open at the completion-note surface. Replace the old
   statements with the reports' descriptive/`UNMEASURED` conclusions and the
   run-block/intervention designs, or mark that note prominently as superseded
   by the closure note.

2. **Several named cases are not end-to-end tests of the repaired analysis
   path.** All synthetic telemetry analysis calls construct only v1 samples
   (`tests/runtime/test_step4_telemetry_regression.py:21-32`), so no test sends
   v2 per-source acquisition boundaries through
   `step4_telemetry_regression.analyze()`. Nonmonotonic telemetry and causal
   PSI interval-boundary assignment stop at `flatten_telemetry()` or
   `join_population()` (lines 128–160), rather than exercising analyzer
   identity/membership validation, the complete join, regressions, and emitted
   result together. Likewise, the unequal-weight estimator regression is only
   against the private `_paired_cluster_difference()` helper
   (`tests/runtime/test_step4_ab_compare.py:93-106`); the complete A/B path uses
   equal-size single-row response clusters and asserts only the equal-response
   result (lines 140–147), so it would not detect miswiring among the declared
   frame-weighted, equal-response, and equal-run outputs. Add complete synthetic
   `analyze()` tests with v2 acquisition boundaries/nonmonotonicity/interval
   placement and with unequal response lengths that distinguish all declared
   A/B estimands. Closure condition 5's requested end-to-end coverage is not
   yet met, even though the focused tests pass.

## Required independent recomputation and gates

Direct parsing of the retained raw server JSONL and v1 telemetry JSONL, without
copying result-JSON statistics, reproduced the revised figures: direct typed
`delivered-nonBOS` was n=293 at 80.552856655 ms; its nearest-sample support was
31 telemetry samples and seven responses, with 7 unique joined CPU-PSI-avg300
values and 10 GPU-temperature values; accepted capture-on browser non-BOS was
n=748 at 82.163147059 ms, making the matched browser-minus-direct contrast
1.610290404 ms; trace-on minus trace-off was +1.479124850 ms frame-weighted and
+1.294669472 ms equal-response; wrapper residual was +0.987960222 ms
frame-weighted and +0.927109601 ms equal-response. Thus no additional numerical
delta was found.

The four Step 4 modules passed **25/25** tests. The full offline suite passed
**1124 tests, 15 skipped, 2 browser-E2E tests deselected, and 29 subtests
passed**. The canonical verifier passed **67/67 anchors**; task-scoped Ruff
reported **All checks passed**; and both Step 4 checksum manifests verified
every local and external artifact as `OK`.
