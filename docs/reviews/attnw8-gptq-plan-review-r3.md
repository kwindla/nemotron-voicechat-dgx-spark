# Confirmation review: attnW8-gptq campaign plan, R3

Date: 2026-08-16 UTC  
Reviewed artifacts: `reports/attnw8-gptq/plan.md`,
`reports/attnw8-gptq/plan-revision-notes.md`  
Prior reviews: `docs/reviews/attnw8-gptq-plan-review-r1.md`,
`docs/reviews/attnw8-gptq-plan-review-r2.md`  
Review mode: static, CPU-only; no campaign, model, capture, construction, or
qualification run performed

## Verdict

**PASS — record Seal-0 and begin Phase A immediately.**

All R2 numbered deltas and both precision edits are resolved in the operative
plan. No new blocking inconsistency was introduced. Execution must follow the
staged order now specified: record Seal-0 before any run or capture; perform
only the baseline A0/effective-position-index stage; record Seal-1 after that
stage passes; and begin A3 Hessian construction and Phase B candidate work only
after Seal-1.

## R2 delta audit

### 1. Circular seal/capture order: resolved

Seal-0 now fixes and hashes the proposed partition inputs, complete scenario
payloads, fixture/code/runtime identities, build parameters, A2 count/window
and weighting rules, algorithm/numerics, gates, replacement rules, and sealed
contingency before A0 or any baseline run/capture (`plan.md:28-87`). It
explicitly excludes baseline-emitted tokens, causal rows, accepted windows,
and completed indexes that cannot exist until the fhw8 baseline is run
(`plan.md:52-55`). This removes the former impossible pre-capture index seal.

The interval between the seals is narrowly authorized. Seal-0 permits only A0
and baseline passes needed to populate and validate the A1/A2/Q indexes, and
limits their use to exact eager-fhw8 behavior, hook transparency/q-k-v input
equality, causal alignment, and window availability. Hessians, quantized
weights, candidate artifacts/outputs, and rule or threshold tuning are
prohibited (`plan.md:89-96`).

The replacement path is fail-closed rather than post hoc: a wrong baseline
outcome or unavailable boundary window permits only whole-scenario replacement
under the preregistered rule; it requires a recorded Seal-0 revision with the
full replacement payload and renewed partition audit, invalidates the affected
baseline evidence, restarts the baseline-only stage, and prohibits invalidated
evidence from entering Seal-1 (`plan.md:98-104,179-190`).

Seal-1 then hash-chains to the exact Seal-0 parent and freezes the final
partitions and payloads, complete ordered indexes and emitted-head records,
selected A2 windows, A0 evidence, final C0 measurement manifest, and exact C2
row list (`plan.md:106-116`). A3/Phase B cannot start earlier. Neither scenario
nor window may change after Seal-1, neither seal may mutate after Hessian or
candidate work begins, and any later required change abandons the candidate
and evidence and starts a new complete Seal-0 -> baseline-only -> Seal-1 cycle
(`plan.md:118-121,185-187`). The same ordering is repeated consistently in the
sequential procedure, contingency/rescue rules, and evidence contract
(`plan.md:208-225,492-505,507-514`). Thus the staged protocol removes the
circularity without opening a candidate-outcome-driven mutation path.

### 2. Q qualification firewall wording: resolved

Q is now excluded specifically from Hessians, scales, candidate construction,
calibration/scenario choice, threshold tuning, and post-result gate revision,
while its preregistered use for qualification accept/reject and sealed
contingency screening is expressly allowed (`plan.md:65-72`). This is
consistent with C0/C2 qualification and with the fixed offline contingency
screen (`plan.md:313-394,417-441,462-470`); the contradictory blanket
"candidate-selection" prohibition is gone.

### 3. R2 precision edits: resolved

Seal-1's C0 manifest now fixes the exact dense-fhw8 eager reference, the
isolated-linear and full-block activation/cache/mask/position/call-state
sources, variant isolation, row and feature aggregation order, float32
subtraction/squaring with sequential float64 accumulation, the normalized-MSE
denominator and zero-energy rule, non-pooled cells, and deterministic worst-row
selection (`plan.md:319-354`). It also freezes and hashes the complete ordered
C2 boundary/continuation record list and forbids inferred, substituted, or
later-selected rows (`plan.md:355-360`).

The RTN-zero case is exact: zero RTN error requires zero GPTQ error and uses
`ZERO_RTN_BOTH_ZERO`; positive GPTQ error is an unconditional failure using
`ZERO_RTN_GPTQ_NONZERO` (`plan.md:387-394`). This removes the former ambiguous
"nonfinite zero denominator" wording.

## Consistency conclusion

The revised seal boundaries agree with the A0, A1/A2, A3, C0/C2, contingency,
and rescue sections. The original R1 design and gate thresholds remain intact,
and this review found no new internal contradiction or unsealed decision point.
Seal-0 is authorized now; Phase A begins with the A0/baseline-only index stage,
not Hessian construction. Seal-1 remains mandatory before A3 or any candidate
work.
