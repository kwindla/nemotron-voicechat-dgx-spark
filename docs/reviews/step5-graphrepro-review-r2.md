# Step 5 graph-reproducibility measurement-validity review — round 2

## Verdict

**PASS — track 1 concluded: characterized graph-construction defect; exact-gate
repair required; the kernel-class row remains blocked with that prerequisite;
production-instance qualification recorded. Track 2, the pairing wedge-closure
design study, may start.**

This conclusion closes the five deltas from
[the round-1 review](step5-graphrepro-review-r1.md). It concludes the
graph-reproducibility characterization track; it does not qualify the released
graph path or remove the exact-gate repair prerequisite.

## Review scope

This was an offline, read-only review of
[the revised Step 5 report](../../reports/step5-graph-reproducibility-20260813.md),
the retained Step 5 records and aggregate, the revised tool and regression
tests, [the Step 4e ranking table](../../reports/step4-candidate-ranking-20260813.md),
and the closure note at
`/tmp/claude-1000/-home-khkramer-src-nemotron-voicechat-dgx-spark/9dab1982-03af-4f77-9d25-ddf7139af66f/scratchpad/step5-graphrepro-closure-done.md`.
No engine, service, container, GPU, model, network, or other live probe was run.

I regenerated the aggregate from the retained run JSON into `/tmp` and compared
it byte-for-byte with the retained `summary.json`; they match. I also rechecked
the 83-entry SHA-256 manifest and the preregistration prefix seal. All 83
manifest entries pass, and the seal remains
`6c5b6c05dc3ca1677015574dce0818f2ca84d2786c4ef1859e0e651277fde071`.

## Round-1 delta closure

1. **Seed and history calibration — closed.** Seed is now reported as “not
   supported; not isolated,” with fresh-engine construction explicitly retained
   as a confound and no rejection or distribution claim. History is limited to
   no observed movement in the tested exact basin, with ordinary cleanup
   reassuring but divergent-engine history and fresh capture initialization
   unresolved.
2. **All-exact confirmation reconciliation — closed.** The report records the
   9/9 exact graph confirmation as evidence that an exact basin can persist
   through call 686, as negative evidence against a high or stationary
   prevalence claim, and as leaving a temporal/construction-epoch factor
   plausible. It preserves the replicated graph divergence and correctly uses
   4/12 divergent production graph engines overall rather than presenting the
   primary 2/3 cell as prevalence.
3. **Overlay-margin calibration — closed.** The overlay values `+0.125` and
   `0.0` are used only for greedy-boundary localization. They do not pass or fail
   the production `0.05` clause; the production margin is explicitly unresolved.
4. **Aggregate validity and bounded-gate tooling — closed.** Selector decisions
   are derived from retained strata. The aggregate fails closed on report/block
   count, call count, cleanup, nonempty engine request state, selected-output
   digest mismatch or coverage, artifact/capture/control identity mismatch,
   token-vector digest mismatch, and non-finite diagnostics; invalid engines are
   excluded from inferential summaries while their reasons are retained.
   Localization bounded-rate and gate results are N/A. The global evaluator
   enforces 30 measured 687-call blocks, five fresh graph engines, and a
   production-workload margin. On the retained evidence it reports three
   engines, six measured blocks, an unresolved production margin, and
   `excluded_real_state_defect`. Regression tests cover the corrected selector,
   validity, localization, and global-gate behavior.
5. **Ranking consequence and production qualification — closed.** The report
   and ranking table now state “characterized graph-construction defect;
   exact-gate repair required,” explicitly exclude the benign bounded gate, and
   leave the Nano W8 kernel/fusion/launch-gap row `NR`, `BLOCKED`, and without
   ranking authority. The long-lived-instance qualification is recorded:
   tested engines were basin-stable, but the restored production engine's basin
   was not measured, so production is potentially affected rather than proven
   universally wrong or unaffected. The wrapper graphing/fusion row remains
   blocked by the same exact-gate prerequisite.

## Number and classification audit

No reported measurement or validation number drifted.

| cell | engines | blocks | exact | call-39 divergent | selected digests |
|---|---:|---:|---:|---:|---:|
| primary graph, seed 0 | 3 | 9 | 3 | 6 | 18/18 |
| primary eager, seed 0 | 3 | 9 | 9 | 0 | 18/18 |
| graph, seed 1 | 2 | 6 | 3 | 3 | 12/12 |
| graph history | 2 | 14 | 14 | 0 | 28/28 |
| graph request-ID/synchronization | 2 | 18 | 9 | 9 | 36/36 |
| full-prefix graph confirmation | 3 | 9 | 9 | 0 | 54/54 |
| full-prefix eager confirmation | 3 | 9 | 9 | 0 | 54/54 |
| **all production graph** | **12** | **56** | **38** | **18** | **148/148** |
| **all production eager** | **6** | **18** | **18** | **0** | **72/72** |

All 18 authoritative production graph divergences remain the single mismatch
`call 39: expected 10592, actual 1044`; all 220 production-interface digest
comparisons match. The regenerated aggregate reports all authoritative reports
valid, within-engine graph stability, the graph/eager discriminator true, and
classification **`REAL STATE DEFECT`**. The later confirmation cell changes the
prevalence qualification, not the preregistered graph/eager classification.

## Fresh offline validation

- Focused Step 5 plus Step 4d regressions:
  **27 passed**.
- Full suite excluding `browser_e2e`:
  **1190 passed, 15 skipped, 3 deselected, 4 warnings, 29 subtests passed**.
- Canonical qualification verifier: **67/67 anchors passed**.
- Task-scoped Ruff: **PASS**.

The four warnings match the retained validation record and are non-failing:
the existing `audioop`, turn-strategy reset, and Starlette/httpx deprecations,
plus the existing unawaited `FrameProcessor` coroutine warning in the Pipecat
interruption test.

## Track disposition

Track 1 is concluded at characterization, with the repair gate intentionally
still closed: the released W8 graph-enabled path requires exact-gate repair
before kernel/fusion or wrapper graph ranking can resume. This prerequisite does
not block beginning the separate design-only reliability work. **Track 2 — the
pairing wedge-closure design study — may now start.**
