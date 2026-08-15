# Step 4c incidence measurement-validity review — round 2

**Verdict: FAIL.** The retained campaign correction is substantively right: a
fresh public analysis exactly reproduces five valid and four consumed-invalid
blocks, **0 / 764** valid `delivered-nonBOS` rows (direct 0/458, native Pipecat
0/306, browser unavailable), `INSUFFICIENT`, and `matched-states: not-triggered`.
The known block-02 manifest-flip exploit is closed, the browser limitation is
registered prospectively without relaxing the retained v1 fixture, and the
narrowed conclusion matches the defensible round-1 wording. However, the
analyzer still does not fail closed on the preregistered 120-second paced-source
window. A new mutation of a declared-valid retained block is accepted, and the
new end-to-end tests omit—and in their fixtures normalize—that missing gate.
Step 4c therefore is not yet admissible under the requested analyzer-enforcement
criterion.

This review was entirely offline. I made no health query, model connection,
browser connection, live probe, container change, or commit. Apart from this
review file, I made no intentional repository change.

## Blocking remaining delta

### Paced-source duration is declared, not evidence-enforced

The preregistration fixes a 120.0-second connected/source window measured from
the start of paced input (`reports/step4-eartts-incidence-20260813.md:49-52`).
Every retained direct/native report exposes the corresponding `source_frames`;
at the configured 80 ms pacing, the expected value is 1,500. Public `analyze()`
does not require that value for ordinary direct or Pipecat blocks:

- `_client_validity_failures()` checks only the literal `warmup_seconds`,
  `duration_seconds`, and `drain_seconds` values
  (`tools/qualification/step4c_incidence.py:221-232`).
- `source_frames == 1500` is checked only inside block 01's exceptional legacy
  timing-sidecar path (`:167-202`), not for blocks whose three timing fields are
  present.
- The browser branch likewise never checks its retained
  `session.connected_window_seconds` (`:312-391`).

I copied the retained manifest and block-06 report to a temporary directory,
changed only block 06's `source_frames` from 1,500 to **1**, pointed the copied
manifest at that report, and invoked public `analyze()`. It accepted block 06 as
valid with `validity_failures: []` and still returned five valid blocks and
0/764. This is a direct fail-open mutation of fixed-exposure evidence, not an
incidence or membership change.

The synthetic public-analysis tests do not protect this boundary. Their valid
direct and Pipecat report builders omit `source_frames` entirely
(`tests/runtime/test_step4c_incidence.py:171-220`), and the browser builder omits
`connected_window_seconds` (`:223-265`), yet all are accepted. The negative
mutation matrix has no source-frame or connected-window case (`:431-509`). Thus
the report's statement that negative mutations cover every listed gate
(`reports/step4-eartts-incidence-20260813.md:287-296`) is too broad.

Required closure is local and does not require another live exposure:

1. For valid direct/native reports, require the exact expected paced-source
   frame count, with the same explicit evidence rule applied to the block-01
   sidecar path.
2. For a potentially valid browser block, require the retained connected window
   to cover the registered duration plus drain under a narrow, stated clock
   tolerance; do not accept a plan's configured numbers as proof of execution.
3. Add public `analyze()` negative tests for truncated/missing `source_frames`
   and truncated/missing browser `connected_window_seconds`, then rerun the
   offline gates and regenerate the analysis record if its bytes change.

## Closed round-1 findings

- **Corrected ledger and population:** Fresh analysis is byte-for-byte identical
  to retained `incidence-analysis.json` and `matched-states.json`. Blocks 02, 03,
  05, and 07 are retained and excluded. The valid population is exactly direct
  0/458 plus native Pipecat 0/306, totaling 0/764; no browser block is valid and
  the browser and stratified-overall summaries are unavailable.
- **Known analyzer exploit:** Flipping retained block 02 to `valid:true` and
  clearing its reason now raises `ValueError` because the evidence independently
  yields `PLAYOUT_TRACE_NOT_PUBLISHED`. Invalid reason codes are nonempty,
  enumerated, and must correspond to an observed failure.
- **Browser fixture-contract registration:** The future browser path is a
  separately versioned `nemotron_voicechat.step4c_fixture_plan.v2` contract. It
  records exact-zero mono PCM16 at source, a one-PCM16-LSB server-ingress bound,
  and the Chrome fake-microphone/WebRTC limitation
  (`tools/qualification/step4c_contract.py:22-31` and
  `tools/qualification/step2_live_fixture_driver.py:213-238`). The exact-plan
  parser and contract test verify those fields, while the retained campaign
  remains v1 and zero-tolerance.
- **Conclusion and pivot:** The result text at
  `reports/step4-eartts-incidence-20260813.md:248-264` preserves the round-1
  qualification: the older state is a reconstruction/profiler starting point,
  not a same-condition comparator; the five direct/native blocks weaken only
  persistence/commonness claims; browser traces are sensitivity observations;
  and removal, future incidence, transport equivalence, and cause are not
  established. `INSUFFICIENT` and the older-state pivot recommendation remain
  correct.
- **Evidence inventory:** Fresh verification exactly matches the retained
  92-file SHA-256/byte manifest.

## Fresh offline gates

- public Step 4c analyzer end-to-end tests: **22 passed**;
- full suite excluding `browser_e2e`: **1,153 passed, 15 skipped, 3
  deselected** (29 subtests passed; four warnings);
- canonical retained verifier: **PASS (67 anchors)**;
- task-scoped Ruff over the closure file set: **all checks passed**.

These green gates confirm that the closure is stable, but they do not cure the
reproduced source-duration bypass. **Step 4d should not proceed directly yet.**
Close this offline analyzer/test delta first; after that, the corrected
`INSUFFICIENT` result can stand without new live probes and its older-state pivot
should proceed.
