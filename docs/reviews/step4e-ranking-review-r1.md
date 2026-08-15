# Step 4e candidate-ranking final review — round 1

Review date: 2026-08-13 UTC

**Verdict: FAIL — Step 4 is not closed.** The admitted numbers and their
standing are accurately represented, no measured candidate authority is
invented, all requested offline gates pass, and the EarTTS patch cleanup is
provably formatting-only. Three document deltas remain against the literal Step
4/5 contract.

This review was offline. I made no health query, model or browser connection,
container operation, live probe, or commit. The repository was read-only except
for this review file; pre-existing worktree changes were preserved.

## Blocking deltas

1. **The table omits a plan-listed Step 5 candidate.** The plan's Step 5 list
   places “selective EarTTS precision, two-frame perception batching” behind the
   higher candidates. The table has a selective-precision row but no two-frame
   perception-batching row. This is not harmless omission: W4 is likewise
   deferred and unmeasured yet receives an explicit `NR` row, while the Step 4
   output contract requires the ceiling, correctness scope, reliability risk,
   and promotion floor for each candidate. Add a two-frame perception-batching
   row with its honest unmeasured/blocked standing and no ranking authority, or
   amend the normative candidate list before claiming the synthesis complete.

2. **Pairing recovery option (c) does not reproduce the reviewed closure
   exactly.** The table requires supervised kill/recreate with “clean
   reinitialization, single-session failure containment, and an explicit
   recovery SLO,” but omits the plan's explicit proof that **no stale state or
   client lock survives**. “Clean reinitialization” does not independently pin
   client-lock release. Restore that clause verbatim. Options (a) and (b), the
   rejection of timeout/session-close as recovery, the executable tail state
   machine, and the subsequent same-image counterbalanced arrival/queue-clock
   A/B otherwise match the reviewed wedge contract.

3. **The evidence/judgment boundary leaks one product-priority conclusion.**
   Under “What the evidence licenses,” the report says the evidence licenses
   “not a larger fixed reserve as the primary solution.” The cited capture-on
   replay supports only the narrower statement that the tested grid through
   560 ms left 3/8 retained responses starving and imposed 499.0 ms mean
   first-delta-to-release delay; it expressly does not test every larger or
   adaptive reserve and does not itself decide what should be primary. Move the
   primary-solution choice to “Recommended next actions (judgment)” and leave
   the bounded replay result in the evidence section. For the same clean
   boundary, phrase the Nano-envelope statement as permission to investigate,
   not as evidence itself choosing the highest-information prerequisite.

## Evidence and authority audit

Every reported measurement in the table and recommendation traces to an
admitted source with the retained standing intact:

- the 12-frame/960 ms, 160 ms gap, browser-underrun, and no-regression floors
  are normative plan constraints, not achieved results;
- the 11.1–11.3 ms/frame pairing value is correctly labeled a historical,
  non-counterbalanced association rather than a repaired-candidate ceiling;
- the retained W8 stage opportunities (52.29–52.79 ms/frame and 12.15–12.40
  ms/frame) are not transferred into W4 or selective-precision benefit claims;
- the corrected ten-block Nano values and fractions match the admitted Step 4d
  disposition exactly and remain descriptive `INCOMPLETE` ceilings;
- the EarTTS values (10 rows, 20.4 versus 10.86 ms, arithmetic difference 9.54
  ms, and current 0/764 exposure) retain precedent-plus-`INSUFFICIENT`
  standing;
- the Step 4a associations and conditional intervals match the admitted fixed-
  order periods, while server trace cost remains `UNMEASURED`;
- the telemetry row states only the narrow recorded-channel observation and
  does not promote or deprioritize a thermal/pressure candidate; and
- the Step 2 capture-on reserve row reproduces 560 ms, 3/8, and 499.0 ms and is
  correctly labeled simulation-only rather than browser-underrun evidence.

The ordinal “1, precedent only” is explicitly scoped to historical evidence for
further work, not promotion authority. Every implementation row remains
`NR`, blocked, deferred, inadmissible, or not selectable; no row is represented
as `measured`. Apart from the missing candidate row, I found no unauthorized
candidate ordering or promotion claim.

The graph-path prerequisite correctly requires a reproducible released W8
baseline across fresh graph-enabled engines and profiling contexts, the full
687-call causal prefix, and all six dtype/shape/byte-sensitive outputs before
candidate timing. The W4 registry correctly treats W4 as a fresh
weights/runtime candidate and requires provenance-complete conversion and
calibration, candidate-appropriate component and quality/non-inferiority gates,
and full runtime requalification on that exact artifact; no W8 result transfers.
The pairing prerequisite is correct except for delta 2.

## EarTTS formatting-only audit

`git diff` for
`src/nemotron_voicechat_runtime/patch_eartts_exact_constants.py` contains only
two explanatory/Ruff comments and removal of one extra blank line (3 insertions,
1 deletion). An independent `HEAD`-versus-worktree parse produced identical
module ASTs, so no executable statement, expression, constant, or control flow
changed.

I also extracted every `replace_once` payload from `apply_change_a` and
`apply_change_b` and compared the rendered UTF-8 bytes position by position:

| payload class | sites | bytes | aggregate SHA-256 (`HEAD` and worktree) |
|---|---:|---:|---|
| source anchors | 11 | 1,135 | `badf5acde8061a7184ca154cc25c8f7dba2f9b47c8740820ee5f0c5b085d8dbb` |
| replacement payloads | 11 | 5,819 | `48e9f5a0aa62eb9a787259ce54c94ef74ae4e01bce936ee9f48d34019b5d292a` |

All 11 individual anchor comparisons and all 11 individual replacement
comparisons were byte-identical. The cleanup changes neither runtime logic nor
the embedded patch bytes.

## Offline validation

- `.venv/bin/pytest -q -p no:cacheprovider -m 'not browser_e2e'`:
  **1171 passed, 15 skipped, 3 deselected, 29 subtests passed** in 33.52 s.
- `tools/qualification/stratified_latency_analyzer.py verify-anchors`:
  **PASS (67 anchors)**.
- `.venv/bin/ruff check --no-cache .`: **All checks passed!**

Pytest emitted four non-failing warnings, including the already documented
unawaited `FrameProcessor.__process_frame_task_handler` warning in
`test_local_interruption_discards_incomplete_sentence_tail`. No requested gate
failed, and no live test was run.

Step 4 may close after the three document deltas above are corrected and the
same offline gates remain green.
