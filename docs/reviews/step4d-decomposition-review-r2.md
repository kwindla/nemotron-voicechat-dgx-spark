# Step 4d decomposition measurement-validity review — round 2

**Verdict: FAIL for the round-2 confirmation gate, with one validation delta.**
The round-1 analyzer table-precedence defect itself is closed: the retained
attempt-2 SQLite files support the corrected ten-block decomposition, the seven
dual-table ranges are handled correctly, the report withdraws the false loss and
preregistration claims, and Step 4d remains honestly `INCOMPLETE`. However, I
cannot issue the requested clean PASS because the required full suite excluding
`browser_e2e` failed in four independent full runs during this review. Every run
failed the same previously documented asynchronous abort-count poll. The test
passes in isolation, and no Step 4d-specific check failed, but no current full
run was green.

No live health query, container operation, profiler run, service mutation,
commit, or push was performed. The repository was read-only except for this
review file.

## 1. Independent retained-SQLite recomputation

I independently read all 60 files under
`nsys-sqlite-attempt2/`, resolved each file's sole Step 4d NVTX identity, clipped
the graph and explicit-kernel intervals to that wall range, unioned overlaps,
and formed the preregistered mean of the three positions in each block and
class. This did not call the production analyzer.

The retained inventory is exactly:

| condition | ranges | blocks |
|---|---:|---|
| positive graph envelope | **60/60** | 0–9 |
| graph envelope only | 53/60 | 0–9 |
| graph plus explicit kernel rows | **7/60** | 2, 3, 6, 8 |
| graph envelope absent | **0/60** | none |

The seven dual-table identities are block/pair/class `(2,3,PAD)`, `(3,1,PAD)`,
`(3,2,PAD)`, `(6,1,PAD)`, `(6,2,content)`, `(8,1,PAD)`, and `(8,3,PAD)`, from
SQLite exports 18, 20, 22, 38, 39, 50, and 54 respectively. All seven canonical
rows are labeled `cuda_graph_execution_envelope`. Their graph envelopes range
from 44.794208 to 47.347520 ms; their explicit outside-graph unions range from
0.001280 to 2.649376 ms.

The corrected analyzer now inspects every nonempty GPU activity table. Within a
range it uses a nonempty graph trace as the graph-envelope estimand, reports
explicit kernels separately, and unions graph and explicit intervals without
double counting. Consequently all 60 identities and all ten blocks pass the
graph requirement. This directly closes the round-1 defect.

The independent ten-block result is:

| class | interface wall, ms | graph envelope, ms | wall − envelope, ms | graph / residual split |
|---|---:|---:|---:|---:|
| content | **53.560786** | **45.847206** | **7.713580** | **85.608% / 14.392%** |
| PAD | **53.595332** | **45.579065** | **8.016267** | **85.053% / 14.947%** |

The all-observed-GPU sensitivity union is 45.933728 ms for content and
45.841732 ms for PAD, leaving residuals of 7.627058 and 7.753599 ms. These are
descriptive impossible-zero-cost ceilings, not predicted gains or ranking
authority.

## 2. Number stability and exact gate

The retained recomputation produced zero differences from the canonical JSON
for all 60 wall, graph-envelope, explicit-kernel, combined-GPU, residual, and
fraction fields checked. Independently rebuilding the ten block series and the
seed-20260813, 10,000-resample summaries likewise produced zero differences in
their means, medians, and percentile intervals. The report's ten block rows and
rounded aggregate table agree with those values.

Comparing the superseded attempt-2 analysis with the canonical correction:

- all common fields in the 53 graph-only ranges are unchanged;
- only the expected primary-activity label/count/timing/fraction/residual fields
  change in the seven dual-table ranges, plus the newly added graph/explicit/all-
  GPU fields;
- the exact-output object, `passed:false`, and `decision:"INCOMPLETE"` are
  unchanged; and
- the canonical JSON and its stdout copy are byte-identical, as are the summary
  JSON and its stdout copy.

The authoritative exact file still contains two warm-ups and 30 measured
blocks. All 32/32 blocks mismatch only at call 39, actual token 1044 versus
expected 10592; all 30 measured blocks are invalid. All 180/180 selected output
digests still match. `summary.json` therefore correctly records ten valid timing
blocks while retaining `exact_output.passed:false`, aggregate `passed:false`,
and `decision:"INCOMPLETE"`. No correctness or standing number drifted.

The artifact manifest also verifies **334/334** entries.

## 3. Narrative correction and supersession

The current report now states that all 60 ranges have graph envelopes, that the
former four-block exclusion was a table-precedence artifact rather than CUPTI
loss, and that it was not preregistered. It correctly identifies
`preregistration.json` as the retained machine-readable authority, acknowledges
that the detailed report seal followed health/inspect and container stop, and
states that the exact sealed prose was not retained and cannot be audited.

Historical attempt-specific JSON remains available as history, while the report
names `nano-interface-decomposition.json` and `summary.json` as the corrected
canonical files. The closure note explicitly supersedes `step4d-done.md` and
withdraws that note's six-block/CUPTI-loss narrative. The old note remains
retained but is no longer presented as current authority. I found no stale
six-block result presented as the current decomposition.

## 4. Validation

The following independently passed:

- task-scoped Ruff: all checks passed;
- the mixed graph-plus-kernel precedence/union regression and the prefix-
  divergence exact-gate regression: 2 passed;
- the complete focused Step 4d file: 8 passed;
- qualification anchors: 67/67 passed;
- artifact hashes: 334/334 passed; and
- whitespace validation: passed.

The required full suite excluding `browser_e2e` did not produce a green run.
Four full invocations each ended with **1 failed, 1168 passed, 15 skipped, 3
deselected, and 28 subtests passed**. The sole failure each time was
`RealtimeWebSocketEndpointTest.test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails`,
at the final `engine.abort_count >= 1` poll; failures occurred in both parameter
subcases across the runs. Its isolated rerun passed with 1 test and 2 subtests.
This is the same asynchronous poll already documented as transient in the
round-1 and closure validation records, and it is outside the Step 4d analyzer,
but the closure note's clean full-suite claim was not independently reproduced.

## Remaining delta

Obtain a clean full non-browser suite result, or establish and record an explicit
project-level disposition for the known asynchronous poll. Once that requested
gate is green or formally waived, the prescribed substantive verdict is
otherwise supported: **Step 4d is admissible as descriptive input with
`INCOMPLETE` standing**, using the corrected content split
**53.560786 / 45.847206 / 7.713580 ms (85.608% / 14.392%)** and PAD split
**53.595332 / 45.579065 / 8.016267 ms (85.053% / 14.947%)** for the Step 4e
table. It remains inadmissible for candidate ranking.
