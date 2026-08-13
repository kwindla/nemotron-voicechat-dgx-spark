# Step 1 retained-evidence adversarial review — r1

Date: 2026-08-12 UTC

## Verdict

**FAIL.** The retained-source identities, sampled arithmetic, pair-schedule
derivation, high-mode inventories, BOS reset/reuse split, and Markdown/JSON
rendering all check out. One required semantic result does not: the analyzer
calls every non-PAD integer token `text-emission`, which puts seven delivered
`agent_eos` control rows into the text stratum. The report therefore does not
yet provide a valid text-emission-versus-PAD-tail phase rollup.

The new tests also leave that case and both new CLI flags uncovered. These are
closure items, not reasons to distrust the unchanged Step 0 populations.

## Blocking finding: `agent_eos` is not text emission

`_token_phase()` in
`tools/qualification/stratified_latency_analyzer.py:335-346` returns
`text-emission` for every integer token other than PAD after checking BOS. In
the retained corpus, seven delivered, stage-timed, non-BOS rows have all of:

- `agent_token_id == 2`;
- `agent_control == "agent_eos"` (also identified as `agent_eos` by the row's
  `control_ids`);
- `response_boundary.phase == "tail_draining"`;
- one delivered 1,764-sample audio frame.

They are frames 1792 in `sustained-aged-26h-20260812`, 74 in `209fe177…`,
288 in `de64cc34…`, 285 in `feca4ec0…`, 285 in `97f9bacc…`, 290 in
`543bfc97…`, and 285 in `70d3f7b5…`. They are control/tail rows, not text
emissions.

Consequences:

- The report's `text-emission n=216` at
  `reports/step1-retained-evidence-20260812.md:284` is semantically wrong.
  A source-only recomputation gives 209 actual text rows, 615 PAD rows, seven
  agent-EOS control rows, and zero token-unavailable rows, totaling the same
  831 delivered-nonBOS stage-timed rows.
- Corrected actual-text server-clock statistics are n=209, mean 82.364 ms,
  nearest-rank p95 90.569 ms, and 156 rows over 80 ms. The seven EOS-control
  rows are mean 82.322 ms, p95 84.829 ms, and 7/7 over 80 ms.
- The EarTTS-high and residual-high *member rows* are unaffected: none of the
  seven EOS rows satisfies either high predicate. Two high-row neighbor labels
  are nevertheless wrong: the next rows shown as `text-emission/t2` at report
  lines 354 and 360 are `agent_eos` controls.

Required fix:

1. Make `_token_phase()` recognize source control evidence before calling a
   token text. At minimum, identify `agent_eos` from `agent_control` and/or the
   row's `control_ids`; do not silently fold it into either text or PAD.
2. Add an explicit `agent-control`/`agent-EOS` stratum so the phase strata
   reconcile to 831. Keep BOS separate as they are now.
3. Regenerate both reports so the text membership, aggregate, definitions, and
   the two neighbor labels are corrected.
4. Add retained-shape tests containing delivered EOS, delivered BOS,
   delivered PAD, non-delivered idle/settlement PAD, and ordinary text.

## Independent raw-artifact recomputation

I parsed the 14 pinned JSONL files directly with a standalone standard-library
script. It did not import or invoke the analyzer. Delivery, BOS, reset, high
mode, stage, and request-position predicates were reimplemented directly from
the raw fields. Nearest-rank p95 used `ceil(0.95*n)` on sorted raw values.

| Sampled claim | Markdown/JSON | Raw recomputation | Result |
|---|---:|---:|---|
| Preflight delivered n / mean / p95 / >80 | 34 / 86.168 / 86.636 / 34 | 34 / 86.168441 / 86.636 / 34 | match |
| Aged delivered-nonBOS n / mean / p95 / >80 | 188 / 82.919 / 93.109 / 139 | 188 / 82.919356 / 93.109 / 139 | match |
| Aged EarTTS-high n / server mean / p95 | 10 / 91.763 / 96.453 | 10 / 91.762800 / 96.453 | match |
| `2a0fa2ae…` abort-reset n / mean / p95 | 3 / 177.141 / 205.267 | 3 / 177.141333 / 205.267 | match |
| Corpus delivered-nonBOS stage n / server mean / p95 / >80 | 831 / 82.420 / 90.990 / 655 | 831 / 82.420213 / 90.990 / 655 | match |
| Same 831-row rollup: perception / Nano / EarTTS / codec / residual / wrapper means | 12.642 / 51.989 / 11.979 / 0.846 / 3.982 / 81.437 | 12.641648 / 51.989324 / 11.978694 / 0.846224 / 3.981520 / 81.437409 | match |
| EarTTS-high corpus n / server mean / EarTTS mean | 38 / 92.156 / 21.700 | 38 / 92.156342 / 21.699928 | match |
| EarTTS-high source-token split | 34 PAD / 4 non-PAD text | 34 PAD / 4 non-control text | match |
| BOS reset n / server mean / EarTTS mean | 19 / 172.456 / 102.261 | 19 / 172.455579 / 102.260536 | match |
| BOS prepared-reuse n / server mean / EarTTS mean | 4 / 87.904 / 18.017 | 4 / 87.904250 / 18.016502 | match |

All 14 source SHA-256 values in the JSON and Markdown also match fresh hashes
of the retained files.

This audit confirms the numerical engine for those samples. It does not cure
the semantic name assigned to the seven EOS rows.

## PAD and BOS predicate audit

The stage-detail population is first restricted to delivered, non-BOS,
stage-timed rows at analyzer lines 592-598. On the current retained corpus:

- All 615 selected token-12 rows are delivered, non-BOS, `agent_control=pad`,
  and contain 1,764 output samples.
- Their source boundary phases are 597 `responding` and 18
  `tail_draining`. Thus `PAD-tail` is best read as “delivered generation PAD
  position,” not as proof that every member is in the boundary state's tail
  phase.
- Of the other 12,338 token-12 model rows, none is delivered. Their boundary
  phases are 12,297 `idle`, 34 `internal_drain`, and seven `complete`.
  Consequently, no current settlement/internal-drain/non-response PAD row
  leaks into the 615-row PAD stratum.
- Legacy server model traces have literal `response_id == null` even for
  delivered audio. That null cannot be used to call their 416 selected PAD
  rows “non-response”; the analyzer's inferred response grouping and the raw
  `responding`/`tail_draining` states are consistent with delivered responses.
- All 23 delivered BOS rows are excluded from text/PAD by the early BOS check
  and are reported separately. The 19 actual reset and four prepared-reuse
  rows are correctly disjoint.

The current delivered/non-BOS guard is sound for the retained corpus. The
problem is the remaining non-PAD branch, not PAD or BOS membership.

## Pair-class derivation

I independently walked every `model_step` in `f110f063…`, maintained the prior
`nano.session_positions` by raw `request_id`, and classified only delivered
rows by delta. This reproduces the analyzer exactly:

| Delta class | n | server mean | nearest-rank p95 | >80 |
|---|---:|---:|---:|---:|
| buffered, delta 0 | 12 | 44.999 ms | 45.959 ms | 0 |
| packed-pair, delta 2 | 12 | 97.485 ms | 118.710 ms | 12 |
| sequential/control, delta 1 | 8 | 86.740 ms | 150.050 ms | 1 |

The eight delta-1 rows are frames 56-63: one BOS, six ordinary text tokens,
and one PAD control row immediately before pairing begins. Frames 64-87 are
exactly 12 buffered/packed pairs. Removing the eight sequential/control rows
leaves 24 pair-class rows and 23 adjacent transitions: 12
buffered-to-packed and 11 packed-to-buffered. No latency threshold participates
in the classification.

The Step 1 report is correct here. The older plan appendix at
`docs/generation-frame-latency-plan.md:370-372` is stale: it says 13 packed PAD
rows and p95 118.27 ms. Under the normative request-delta and nearest-rank
rules, the correction is 12 packed rows plus one sequential PAD control and
p95 118.710 ms. This plan-text correction is not itself a defect in the Step 1
report, but it should be made when edits beyond this review file are allowed.

## Analyzer isolation and test coverage

The Step 1 computation is gated by `step1_details=False` in `analyze_paths()`;
the old per-artifact analyzer runs before the optional detail builder. The
render path only runs when Markdown is requested. Static inspection found no
change to a Step 0 population, clock, structural gate, queue simulation, or
anchor definition, and the current verifier passes all 67 anchors. The
analyzer-specific suite passes 28 tests. These results support “no qualified
semantic perturbation.”

They do not fully cover the extension:

- The sole new test (`tests/runtime/test_stratified_latency_analyzer.py:638-725`)
  calls `analyze_paths(step1_details=True)` and `render_step1_markdown()`
  directly. It exercises one positive buffered/packed schedule, one PAD high
  row, one text residual-high row, and BOS resets.
- It constructs no `agent_eos` control and no non-delivered PAD/settlement
  row, which is why the blocking phase error passed.
- It does not invoke `main()` with either `--step1-details` or
  `--markdown-output`, so neither new CLI route nor their combination is
  regression-tested.
- `--markdown-output` without `--step1-details` currently writes the JSON
  output first and only then raises because rendering lacks details
  (`stratified_latency_analyzer.py:2257-2268`). Validate the flag combination
  before any output write, or make Markdown output imply Step 1 details, and
  test that behavior.

The 67 anchors are valuable regression protection for the Step 0 path; they
are not coverage of the new Step 1 details.

## Findings wording and periodicity

The periodicity conclusion is appropriately cautious. Independently derived
within-response EarTTS-high ordinal gaps are:

`2x5, 3x6, 4x2, 5x3, 6x3, 7x3, 8x1, 10x2, 13x1` (26 gaps total).

Nearest-rank p50/p95 are therefore 4/10, but only 2/26 gaps equal four. The
full distribution supports the report's statement that this corpus does not
justify a fixed-period kernel hypothesis; it does not claim periodicity from
the two percentiles. The recommendation to prioritize PAD-state EarTTS
captures is also bounded by the observed 34/4 source-token split.

The one material wording overclaim is `text-emission`, addressed above. The
residual tracing/logging sentence is framed as a hypothesis to test, not as a
causal conclusion, and is acceptable in that form.

## Markdown, JSON, and negative-control consistency

- Rendering the committed JSON in memory with `render_step1_markdown()` is
  byte-for-byte equal to the Markdown report (45,475 characters).
- JSON has 14 source objects and 14 Step 1 artifact IDs. Every source hash
  matches the current retained artifact.
- The 38 EarTTS-high and six residual-high inventory memberships reconcile to
  their row arrays. The pair schedule reconciles to 32 delivered rows.
- The Markdown and JSON are therefore mutually consistent, including their
  shared EOS misclassification.
- `reports/step5-host-barriers-20260807.md` is byte-identical to its source-tree
  copy at SHA-256
  `4b7b7ad2e1bae26d1ba37173cc8226167e9fd4fca8ca68bdccfa3e4927abd105`.
  `reports/step8-overlap-ab-20260807.md` is likewise byte-identical at
  `75fa919039308a6a7cbb7ebf3de8622a39e8240992d62d2519e96d9a0f7db565`.
  Their short summaries in the plan/report do not overstate their negative
  results.

## Full-suite flake attribution

The flagged
`test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails` failure
is genuinely unrelated and pre-existing:

- The test and its runtime path are tracked code; `git blame` dates the test to
  commit `79cd152d`. Step 1 changed only the untracked analyzer, its analyzer
  test file, and the two reports. No runtime or WebSocket file changed.
- The test uses a 30 ms patched function timeout and then polls asynchronous
  abort completion (`tests/runtime/test_realtime_websocket_endpoint.py:1045-1089`).
- In this review, one isolated invocation passed, the immediately following
  isolated invocation failed in the `invalidation_failure=False` subcase
  because `engine.abort_count` remained zero, and a fresh isolated invocation
  then passed both subcases. That reproduces the timing-sensitive behavior
  without importing or running the Step 1 analyzer and without touching the
  deployed stack.

It should be tracked separately, but it is not collateral from this
deliverable and is not an additional Step 1 blocker.

## Closure conditions

1. Separate delivered agent-control rows from true text emission, add the
   reconciled control stratum, and correct the two neighbor labels.
2. Add tests for EOS/control, non-delivered PAD/settlement exclusion, BOS
   exclusion, and both new CLI flags, including their invalid combination or
   documented implication.
3. Regenerate the JSON and Markdown from the corrected analyzer; rerun the 28+
   analyzer tests and all 67 anchors; confirm byte-exact Markdown rendering and
   raw-source hashes again.
4. Separately correct the stale A4 prose from 13 packed PAD rows / p95 118.27
   to the request-delta, nearest-rank result.

Until conditions 1-3 are complete, the report's internal `PASS` gate statement
at line 387 is premature.
