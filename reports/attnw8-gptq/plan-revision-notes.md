# attnW8-gptq plan R1 revision notes

Date: 2026-08-16 UTC  
Authority: `docs/reviews/attnw8-gptq-plan-review-r1.md`  
Revised artifact: `reports/attnw8-gptq/plan.md`

## Disposition

The plan is revised for R2 review and remains unsealed and non-executable. R1
does not authorize Phase A. This edit preserves the endorsed direction:
calibrated GPTQ, exact q/k/v and o input sites, construction integrity before
the C1-first live behavioral gate, RTN construction/latency controls, C3/C4,
fail-closed evidence retention, and explicit promotion authority.

## 1. Calibration and qualification firewall

A1 now uses the original three-conversation, 3,934-call calibration selection
(or a separately reviewed calibration-only replacement). The physically
separate 1,172-call evaluation replay is untouched and reserved for C0/C2.
Canonical hashed manifests must prove disjoint source roots,
conversation/request IDs, audio hashes, and whole scenario IDs. Qualification
carriers, prompts, schemas, system prompts, injected results, and harness
revisions are excluded from A1/A2. If a C1 carrier must overlap, C1 becomes
regression-only and a new disjoint frozen/live qualification set is mandatory.

The plan replaces position-modulo splitting and raw count coverage with a
hashed unique effective-position index. Prefill rows count once and subsequent
Nano calls contribute only new decode rows. It records source, conversation,
turn, call, causal row, both emitted heads, phase, scenario, and audio hash;
row `r` predicts emitted coordinate `r+1`. A2 is fixed at 12 positive and 12
hard no-call scenarios, nine boundary rows each. Hierarchical equal-scenario
weighting fixes `H = 0.75 H_A1 + 0.25 H_A2`, preventing A1 volume from drowning
tool rows; trace, count, and leverage audits are required.

## 2. In-tree GPTQ and sequential propagation

B-path-1 now names and reuses
`tools/conversion/nano_gptq_calibration.py` and the sequential stage pattern in
`tools/conversion/convert_nano_marlin_gptq.py`, combined with the fhw8-derived
shard rewrite in `tools/benchmark/build_attnw8_candidate.py`. Layers propagate
in order: capture shared q/k/v input, quantize/install q/k/v, recapture the
real o input, quantize/install o, then propagate the quantized layer. This
removes the draft's accidental fixed-teacher mismatch and preserves every
non-target fhw8 payload byte-for-byte.

## 3. Executable numerics and corrected self-check

The fallback prose and false RTN identity oracle are replaced by the actual
core contract: float32 Hessian/work, natural columns, group/block 128, static
127/128 range scales, exact zero-group and dead-column policies, damping after
dead-column repair, lower Cholesky -> inverse -> upper Cholesky, ties-to-even
rounding and `[-128,127]` clamp, BF16 scale serialization after code selection,
TF32 off, deterministic algorithms on, and exact software/hardware/image
identity. The correct checks are diagonal-H equality to the same GPTQ
no-feedback quantizer and independent packed-runtime reconstruction of codes,
BF16 scales, `+128`, zero `qzeros`, natural `g_idx`, shapes, and orientation.
RTN remains a numeric comparator. Two clean builds must be byte-identical.

## 4. Hard C0 and C2 surfaces

C0 is now wholly objective: exact/finite packing, two identical builds, an
exact 16-weight/config delta inventory, proof every other fhw8 file/tensor is
unchanged, fresh-engine routing with BF16-GEMV 8 -> 0 and Marlin 156 -> 168,
and non-waivable `GPTQ/RTN <= 1.0` held-out MSE gates for every linear tensor
and every layer's complete attention block on both base and tool strata.

C1 is unconditional 12/12 with exact calls and no stochastic adjudication. An
adjacent hash-pinned fhw8 12/12 control is required for validity. C2 retains
matched-basin replay of the 1,172 calls and adds a whole-scenario-disjoint
frozen tool replay with positive and hard no-call cases. Zero categorical,
argument, extra-call, continuation, or loop-bound changes is the gate;
function-vs-text margins are reported. Retained live sessions are supplemental.
Passing C0-C4 yields measured/eligible status only; promotion still requires a
counterbalanced A/B and explicit user decision. Pre-stop identity and verified
stable-field restoration are restored to the operations contract.

## 5. Sealed contingency

Tool-sensitive failure in C0, C1, or C2 now triggers a fixed localization
order: eight reciprocal layer cells, eight reciprocal projection-family cells,
then eight reciprocal q/k/v/o cells only within layers implicated by a sealed
paired rule. All coarse cells run; arbitrary subset search and optional
stopping are prohibited. Offline C0/tool replay screens cells before any live
matrix. Cells are localization evidence only; a rescued mixed-precision subset
needs a fresh seal and full C0-C4 ladder. Group size 64 is a separate full
candidate with deterministic build, routing, semantic, and latency gates. A
mandatory A1-only versus A1+A2 comparison measures whether tool calibration
actually helps.

---

# attnW8-gptq plan R2 consistency-fix notes

Date: 2026-08-16 UTC  
Authority: `docs/reviews/attnw8-gptq-plan-review-r2.md`  
Revised artifact: `reports/attnw8-gptq/plan.md`

## Disposition

The R2 consistency deltas are incorporated without changing the confirmed R1
quantization design or qualification thresholds. The plan remains unsealed and
non-executable; no campaign, model, capture, or qualification run was
performed.

## 1. Circular seal/capture order replaced by two stages

The former requirement to seal emitted-token effective-position indexes before
the baseline capture needed to create them is removed. Seal-0 now hashes the
candidate input partitions, complete scenario payloads, fixture/code/runtime
identities, parameters, A2 count/window and weighting rules, algorithm,
acceptance/replacement rules, gates, and contingency before any run or capture.

Seal-0 authorizes a baseline-only A0/index stage whose evidence is limited to
exact eager fhw8 behavior, hook transparency, q/k/v equality, causal alignment,
and required-window availability. A preflight-driven whole-scenario replacement
requires a recorded Seal-0 revision, invalidates the affected evidence, and
restarts that stage.

Seal-1 then hashes the final accepted partitions and payloads, complete ordered
effective-position indexes, emitted-head and window records, A0 evidence, final
C0 measurement manifest, and exact C2 boundary/continuation row list. No
Hessian, quantized weight, candidate construction/output, threshold change, or
outcome-driven tuning may precede Seal-1. Neither seal may be mutated after
candidate work begins; a change requires a new two-stage campaign cycle.

## 2. Q qualification firewall corrected

Q remains excluded from Hessians, scales, candidate construction,
calibration/scenario choice, threshold tuning, and post-result gate revision.
The plan now explicitly permits Q's preregistered use for qualification
accept/reject decisions and the already sealed contingency screening, removing
the contradictory blanket prohibition on candidate-selection decisions.

## 3. C0 and C2 precision repairs

The final Seal-1 C0 manifest now fixes the exact eager dense-fhw8 reference;
the hidden-state, linear-input, cache, mask, position, and call-state sources
for isolated-linear and complete-attention-block comparisons; the normalized
MSE denominator and zero-reference-energy behavior; fixed row/feature
aggregation order and arithmetic; per-cell non-pooling; and the exact hashed C2
boundary/continuation row inventory.

The RTN-zero case is now exact: when RTN's error numerator and unnormalized MSE
are zero, GPTQ's must also be zero. The ratio field uses
`ZERO_RTN_BOTH_ZERO` when both are zero and
`ZERO_RTN_GPTQ_NONZERO` for the unconditional failure when GPTQ is positive.
