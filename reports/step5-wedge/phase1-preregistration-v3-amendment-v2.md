# Step 5 wedge Phase 1 validity-contract amendment — v2

**Reseal date:** 2026-08-15 UTC  
**Status:** **BLOCKED — calibration rejected; Phase 1 may not be scheduled**

## Authority, opening scope, and complete weakening disclosure

This document supersedes the rejected v1 amendment at
`reports/step5-wedge/phase1-preregistration-v3-amendment.md`, SHA-256
`0492c6f8a43951716c2989a8098370ac64acccb57b1f65d23aabc7c839a00057`,
and responds to the authoritative round-1 review at
`docs/reviews/step5-wedge-phase1-amendment-review-r1.md`, SHA-256
`24282768951ff72983b8d639ede0d73741155c8727b001e0b0400a0e1070a859`.

The proposed weakening is confined to selected `function_logits`, committed
Mamba conv/SSM state, and their canonical transaction rollback mirrors in
packed-lane comparisons against sequential S. It does not normalize tensor
metadata, non-finite values, decisions, source-row identity, pending/correction
state, committed logical KV, lifecycle evidence, or any other selected custom
output. Canonical rollback is in the opening scope because it is a repeated
assertion of the committed recurrent boundary, not a second independent drift
allowance.

The empirical envelope in amendment v1 came only from the first rejection in
exactly one fresh B construction and one fresh A construction. It was not a
distribution and is withdrawn. The adjudication supports only that packing
shape, descriptor/capture, or another pair-only numerical detail remains
unresolved; it does not support a specific BF16 reduction-order mechanism.

At the terminal causal case, released A state is unavailable precisely when A
loses completion before B4. Pair-vs-pair exactness therefore does not protect
that causal boundary, and this amendment makes no such claim.

## Complete immutable authority chain

The pre-amendment authority is pinned as follows:

| Authority | SHA-256 |
|---|---|
| v2 governing preregistration, `phase1-preregistration.md` | `f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5` |
| v3 compatibility reseal | `8c3494e8f755f9b6f252ba2306b34af7fcd8cecd38e18285a83ee65f202c1d64` |
| v4 executor reseal | `a97b197b1f80c61b438fa5f0c6c15627ca407f56664ef809a1b79a2203e6160d` |
| v5 executor round-3 reseal | `c685a247623233522993cd0511193fc959e97b3debd42f65a4427cde740c370a` |
| v6 executor round-4 reseal | `abb801a19c38172ead8df050d5f6a713e5fc50e078fd286b5eba56ffe1e6088f` |

If and only if a future accepted calibration activates an amendment, it would
supersede these exact v6 clauses:

1. `Cross-lane exactness boundary`, paragraph 4, the sentence requiring every
   remaining selected-output, committed-Mamba, and canonical-rollback byte
   difference to stop the campaign; only the three named numeric channel
   classes would move to a sealed directional envelope.
2. `Gated structural smoke`, paragraph 1, the classification of every
   corrected comparator difference as semantic divergence; the three named
   classes would instead be classified by the independently sealed envelope.
3. `Gated structural smoke`, paragraph 2, the all-B-case bitwise-exact
   viability sentence; it would become exact for non-waived fields plus
   fail-closed envelope conformance for the named numeric classes.

Those clauses are **not currently superseded**, because calibration did not
pass. Every v2–v6 non-oracle clause remains byte-for-byte authoritative,
including lane/case order, 12 fresh replicates, terminal destruction/no retry,
B4 localization, the two-discordant threshold, fallback, resume, identity, and
evidence-retention disciplines.

## Independently sealed calibration contract

Before calibration execution, the machine-readable contract was sealed at
`reports/step5-wedge/phase1-calibration-contract-v1.json`, SHA-256
`c22f3cb477c898fe3f4caa655c751345b363df47c509aa5960b166fb36b4ee38`.
It preregistered:

- exactly five fresh single-engine S+A constructions;
- all twelve applicable non-causal case/boundary classes;
- delta orientation `candidate_packed_minus_sequential_s`;
- per full tensor-leaf signed lower/upper deltas, positive/negative/zero and
  differing-element counts, and L1/L2/Linf norms;
- exact tensor presence/key set/dtype/shape/element count and finite values;
- a conv-layer-0 zero envelope;
- multiplier 1.0 extrema derivation and inclusive ceilings;
- failure on a missing statistic, orientation mismatch, new/missing leaf,
  metadata/non-finite error, sign/support range violation, signed-bound
  violation, norm-ceiling exceedance, or zero-envelope violation; and
- all 12 Phase 1 replicates held out, with no Phase 1 recalibration ever.

The comparator implementation expresses those rules per lane, case, boundary,
and leaf. B's pre-causal rules would be separately keyed but copied from the
independently observed released-A ordinary-packed baseline. That copy is not a
causal-case artifact-matched oracle and cannot support a semantic claim.

## Calibration result and failed activation

The calibration campaign report is
`reports/step5-wedge/phase1-calibration-v1.md`, SHA-256
`8f8916b8a29b77cd998a58b3778477bae1d11e2d86ddcf5095f9f5f7bf5ef197`.
The five-construction receipt is SHA-256
`a5a45dc07ad5981c77600b7bd68b9f3bb40899835937168fcf4a4602897f0ec8`.

The population and statistics completed, but the acceptance rule failed. In
all five fresh constructions, all four committed logical KV leaves diverged at
the same ten post-first boundaries, for 40 exact-field failures per
construction. Only the initial `text-only-rejection` retained the first-smoke
exact-KV result. The prior two-point evidence therefore did not generalize over
the case sequence.

No directional envelope is authorized or emitted. The observed statistics may
not be used as Phase 1 ceilings, and widening the weakening to KV after seeing
this result would be post-hoc recalibration. The 12 Phase 1 replicates remain
unrun and held out.

## Liveness-only terminal conclusion

Even after a future artifact-matched oracle closes the validity gap, the
maximum conclusion authorized by this amendment family is only **B1 liveness
localization**: reproduction of the released-path wedge and a completion flip
under the pair-only eager intervention, localized by the retained B0–B9 trace.

The following are explicitly prohibited:

- claiming semantic equivalence of B to the released or sequential path at the
  terminal causal case;
- claiming the numeric intervention effect is ruled out there;
- calling a B1 liveness flip an exact repair, promoting it as an exact repair,
  or advancing an exact-repair disposition; and
- using pre-causal pair-vs-pair exactness as protection for the unavailable
  terminal A comparison.

Semantic-equivalence and exact-repair advancement require a future,
independently sealed artifact-matched oracle that can distinguish ordinary
packed drift from a B-specific terminal numeric effect.

## Scheduling and reseal discipline

The calibration contract SHA above was recorded before the GPU campaign. This
amendment-v2 document is sealed after the rejected calibration and before any
Phase 1 scheduling. Because there is no accepted envelope, there is no second
pre-scheduling activation seal and Phase 1 scheduling is prohibited.

Any future resubmission must be adversarially reviewed, must name a new
independently justified oracle/contract, and must record that document's SHA
before any new calibration observation and again in the final immutable
pre-Phase-1 authority chain. Neither this rejected calibration nor the held-out
Phase 1 replicates may be used to widen it.

