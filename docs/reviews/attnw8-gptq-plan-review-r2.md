# Confirmation review: revised attnW8-gptq campaign plan, R2

Date: 2026-08-16 UTC  
Reviewed artifacts: `reports/attnw8-gptq/plan.md`,
`reports/attnw8-gptq/plan-revision-notes.md`  
Round-1 authority: `docs/reviews/attnw8-gptq-plan-review-r1.md`  
Review mode: static, CPU-only; no campaign, model, or qualification runs
performed

## Verdict

**FAIL — do not seal the current text or begin Phase A.**

The revised plan faithfully incorporates the substantive R1 design
dispositions. However, it introduces a circular sealing order that makes its
own A0/A1/A2 acceptance procedure impossible to execute. One small related
wording contradiction also needs correction. These are narrow deltas; no
change to the quantization design or qualification thresholds is required.

## Round-1 disposition audit

### Three-partition firewall and disjointness audit: incorporated

The plan now defines A1 calibration, A2 tool calibration, and Q qualification
as separate partitions (`plan.md:27-45`). A1 uses the original three selected
calibration conversations and 3,934 calls; the physically separate 1,172-call
evaluation conversation is reserved for C0/C2. The audit is pairwise over
resolved source roots, conversation/request IDs, audio hashes, and scenario
IDs, and tool material additionally inventories the carrier, transcript,
prompt, schema, system prompt, injected result, and harness revision
(`plan.md:47-57`). Whole-scenario and near-duplicate containment, fail-closed
collision handling, the C1-overlap demotion/replacement rule, and the mandatory
disjoint C2 tool set are all present.

### A0 preflight proofs: incorporated

A0 requires an exact eager numerical reconstruction of fhw8 rather than the
original dense release, pins the artifact/config/runtime/code identities,
requires the expected tool name and exact arguments, and requires hooks to
leave both output heads' decisions unchanged (`plan.md:65-72`). It hooks the
actual q/k/v module arguments in all four attention layers, requires
element-for-element equality before sharing a stream, and captures the real
post-attention argument to `o_proj` (`plan.md:74-79`).

### Effective-position index and causal convention: incorporated

The qualified measure is restored: prefill rows count once and subsequent
streaming calls contribute only new decode rows (`plan.md:81-86`), consistent
with the in-tree loader's `effective_rows` and sequence construction. The
canonical record includes the requested provenance and both emitted heads,
defines row `r` as predicting emitted coordinate `r+1`, identifies `r*` and
`r*-1`, hashes the ordered index, and rejects duplicate prefix rows and count
or alignment mismatches (`plan.md:88-106`).

### Twenty-four-scenario A2 design, initiation quotas, and Hessian weighting: incorporated

A2 is fixed at 12 independently worded/audio-hashed positive scenarios and 12
hard no-call controls (`plan.md:108-116`). Each scenario contributes a fixed
nine-row window; every positive includes `r*-1` and `r*`, and only those 24
rows satisfy the initiation quota (`plan.md:118-128`). The hierarchical
measure is fixed as equal conversations for A1, equal scenarios within
positive/no-call halves for A2, and
`H = 0.75 H_A1 + 0.25 H_A2`, with fixed ordering and row-count, trace, trace-
share, and leverage audits (`plan.md:130-143`). This prevents raw A1 volume or
one tool scenario from dominating the Hessian.

### In-tree GPTQ reuse, sequential propagation, executable numerics, and self-check: incorporated

The preferred path names the in-tree Hessian/quantizer/packing core and
sequential converter, combines it with the fhw8-derived shard-rewrite pattern,
and limits replacement to the 16 attention weights (`plan.md:145-177`). The
layer-order procedure quantizes and installs q/k/v, recaptures the actual
candidate-dependent `o_proj` input, quantizes o, and propagates the quantized
layer output, eliminating the former fixed-teacher mismatch.

The numerical contract pins float32 work/Hessian/error arithmetic, natural
column order, group and block size 128, static asymmetric signed-range scale
divisors, zero-group and dead-column behavior, damping placement, the exact
Cholesky sequence, ties-to-even rounding and clamp, BF16 serialization timing,
packing ABI, propagation-weight construction, determinism controls, and the
qualified software/hardware identity (`plan.md:184-228`). These agree with the
reused core's static scale calculation, GPTQ update order, and packing layout.
The impossible RTN identity oracle is correctly replaced by same-quantizer
diagonal-H/no-feedback equality and an independent packed-runtime unpacking
check; RTN is only a numeric comparator and two clean builds must be byte
identical (`plan.md:230-243`).

### Hard C0/C2 surfaces: incorporated

C0 is now non-waivable: exact packing and finite checks, two byte-identical
builds, an exact 16-weight/config delta inventory, unchanged carried fhw8
content, fresh-engine routing with the retained 8-to-0 BF16-GEMV and
156-to-168 Marlin transition, per-tensor held-out GPTQ/RTN limits, and complete
attention-block limits on both base and disjoint tool strata
(`plan.md:255-285`).

C2 keeps the untouched 1,172-call matched-basin replay with replay blocks as
the uncertainty unit and adds the mandatory A1/A2/C1-disjoint frozen tool
replay with positive and hard no-call scenarios. It gates categorical,
argument, extra-call, continuation, and loop behavior and reports decision-
margin erosion; retained live-session replay is explicitly supplemental
(`plan.md:308-331`).

### C1 qualification surface rules: incorporated

C1 pins the fixture and harness identities, separates harness validity from
model behavior, retains but excludes invalid attempts, and requires an
adjacent fhw8 12/12 control (`plan.md:287-304`). Candidate passage is
unconditionally 12/12 with zero fatals, exact tool name/arguments, and the
sealed call/loop bound. There is no adjudication or stochastic allowance. The
partition-firewall replacement rule is explicitly carried into C1
(`plan.md:305-306`).

### Tightened contingency and retained operating/promotion controls: incorporated

Tool-sensitive failure in C0, C1, or C2 triggers a sealed, fixed-order family:
all eight reciprocal layer cells, all eight reciprocal projection-family
cells, and eight reciprocal q/k/v/o cells for each layer implicated by the
fixed paired rule (`plan.md:352-380`). Offline C0/tool surfaces screen cells;
arbitrary subsets and optional stopping are prohibited. The A1-only versus
A1+A2 control is mandatory, mixed-precision rescue requires a fresh seal and
full ladder, and group size 64 is a separate full candidate with routing and
latency qualification (`plan.md:382-394`). The plan also restores
measured/eligible-not-promoted status, counterbalanced A/B plus explicit user
authority, pre-stop identity capture, and verified stable-field restoration
(`plan.md:348-350,396-409`).

## Required deltas

### 1. Resolve the circular seal/capture order

The prerun section requires canonical manifests **and their effective-position
indexes** before "any capture" (`plan.md:29-31`), says the seal precedes every
Phase-A action (`plan.md:59-61`), and repeats that all exact manifests/hashes
must be sealed before Phase A (`plan.md:398-399`). But the index contains
baseline-emitted text/function tokens and causal rows (`plan.md:88-105`), A0
must first prove baseline tool behavior and hook noninterference
(`plan.md:65-79`), and A2 permits scenario replacement when the baseline
outcome or required boundary window is unavailable (`plan.md:118-125`). Those
facts cannot be known and the replacement rule cannot operate before the
prohibited baseline capture.

Replace this with an explicit two-stage, fail-closed seal:

1. Before A0 or any baseline capture, seal the input partition candidates,
   complete scenario payloads, fixture/code/runtime identities, A2 counts and
   windows rule, weighting, algorithm, and acceptance/replacement rules.
2. Authorize baseline-only A0/index capture under that input seal. It may be
   used only to prove eager fhw8 outcomes, hook transparency/q-k-v equality,
   causal alignment, and window availability. Any scenario replacement
   invalidates the affected baseline evidence and restarts this stage under a
   recorded input-seal revision.
3. After A0 and effective-position validation, seal the final accepted
   partitions and complete ordered indexes. No candidate construction,
   quantized weight, candidate output, threshold change, or outcome-driven
   tuning may occur before this final capture/index seal. Phase A3/B may begin
   only after it.

This preserves R1's required pre-candidate preregistration while making A0-A2
possible.

### 2. Correct Q's contradictory "candidate-selection" prohibition

The Q description says the 1,172 calls never contribute to a
"candidate-selection decision" (`plan.md:41-45`), while C0/C2 use them to pass
or fail the main candidate and contingency uses the frozen offline surfaces to
screen localization cells (`plan.md:251-253,312-317,354-359`). The intended
firewall is clear, but the literal rule prohibits the qualification decisions
the plan later requires.

State instead that Q never contributes to a Hessian, scale, candidate
construction, calibration/scenario choice, threshold tuning, or post-result
gate revision. Explicitly permit its preregistered use for qualification
accept/reject decisions and the sealed contingency screening already defined.

## Non-blocking precision edits

- At `plan.md:283-285`, replace "a nonfinite RTN zero denominator" with the
  exact case intended: if RTN MSE is zero, GPTQ MSE must also be exactly zero
  and the ratio is reported by a fixed sentinel convention.
- When the final C0 manifest is sealed, spell out the normalized-MSE
  denominator, dense fhw8 reference, activation/cache/mask source for each
  linear and full-block comparison, aggregation order, and the exact frozen
  C2 boundary/continuation row list. The current ratio gates are objective,
  but these details should not be left to implementation-time choice.

After the two required deltas are incorporated, the plan can be sealed under
the corrected two-stage order and Phase A can begin. No further R1 design
disposition remains open.
