# Step 5 wedge Phase 1 validity-contract amendment review — round 1

**Review date:** 2026-08-15 UTC  
**Verdict:** **FAIL — the Phase 1 campaign may not be scheduled under this amendment**

The amendment seal is authentic:
`0492c6f8a43951716c2989a8098370ac64acccb57b1f65d23aabc7c839a00057`.
The two decision receipts, the A-lane adjudication, the v2 preregistration, the
v3 compatibility reseal, and the historical Step-7 source/report hashes also
match the values cited by the amendment.

The amendment gets the qualitative correction right. The retained evidence no
longer supports treating packed pair execution as bitwise equal to the Phase 1
sequential logical-position oracle. B-vs-S and A-vs-S have the same 107-path
corrected divergence class at the adjudicated first rejection: decisions and
committed logical KV are exact; selected function logits, 26 of 27 committed
Mamba conv leaves, all 27 committed Mamba SSM leaves, and their 53 canonical
rollback mirrors differ. The historical Step-7 gate is correctly relabeled as
PIECEWISE-packed-pair versus FULL-packed-pair evidence.

That evidence justifies reconsidering pair-vs-S bitwise exactness. It does not
justify the particular campaign-wide numeric envelopes now proposed. Those
envelopes are calibrated from one B first-case observation and one A first-case
observation in different fresh constructions, have no signed/directional
constraint, and are then applied to every relevant boundary in 12 new
constructions. In the terminal causal case, the new B-vs-A exactness guard is
unavailable precisely when A loses completion before B4. A B-only numeric
intervention effect at that boundary can therefore be admitted as ordinary
pair-vs-S drift if it stays below the very coarse count and absolute-maximum
ceilings. The amendment consequently does not preserve the original claim that
semantic exactness has ruled out a numeric intervention effect on the B1
contrast.

This was a read-only CPU/static review. No campaign, GPU operation, engine,
container, service, or endpoint was run or changed. The only repository write
is this review file.

## Requirement disposition

| Requirement | Disposition | Evidence and assessment |
|---|---|---|
| (a) Grounding in the A-lane and Step-7 evidence | **FAIL quantitatively; supported qualitatively** | The two current receipts support the shared *path-class* finding and defeat an eager-only interpretation of the first-case `0.125` logit maximum (`phase1-a-lane-adjudication-20260815.md:5-25,108-149`). They do not establish a drift distribution: each lane contributes one observation of one boundary in one construction, and the A adjudication expressly says it cannot identify packing shape, PIECEWISE capture, or another pair-only numerical detail as the narrowest cause (`:20-25`). Step-7 supplies strong historical packed-pair reproducibility evidence—1,389 output/decision and 675 persistent-cache/rollback comparisons—but only in one same-process construction, on a different replay/runtime, between PIECEWISE pair and dedicated FULL pair (`step7_pair_bitwise_gate.py:2-7,181-224,291-309`; `step9-cold-start-20260807.md:169-177`). It cannot calibrate current pair-vs-S, cross-construction, or per-case drift. Amendment lines 42-44 overstate the evidence by declaring BF16 packed-scan reduction order the supported mechanism. |
| (b) Leak-proof envelopes | **FAIL** | The table itself is transcribed correctly. An independent parse of both 107-entry observations recovered the same 54 independent numeric leaves (function logits, 26 conv, 27 SSM), and all 55 table rows including zero-drift conv 0 equal the componentwise observed maxima. Lines 106-118 also fail closed on metadata, non-finite values, a new leaf, and either listed ceiling; lines 178-185 correctly prevent rollback from becoming a second independent allowance. But the gate defines only differing-element count and maximum *absolute* delta. It never defines comparator orientation (`packed - S` or `S - packed`), signed lower/upper bounds, positive/negative counts, or stable element support. Retaining first differing values is evidence, not a pass criterion. Arbitrary sign reversal, support migration, or compensating drift across almost an entire SSM leaf can pass. The table is also leaf/layer keyed but not case/boundary keyed even though it is calibrated solely on `text-only-rejection`; “every comparison” does not specify whether the same first-case ceiling is intentionally universal. Exceeding the two stated ceilings does unambiguously fail the replicate, but those two ceilings do not constitute the directional per-leaf envelope claimed by the review request. |
| (c) Wedge/liveness discrimination logic | **PASS as a textual diff** | Against the sealed v2 plus v3 reseal, no lane, case, liveness, or disposition rule changes. S/B/terminal-A order, the 13-case matrix, 12 fresh replicates, first-missing-B4 localization, terminal destruction/no retry, the two-discordant-replicate threshold, all-12 reporting, and the construction-stratified fallback are expressly preserved (`phase1-preregistration-v3-amendment.md:192-210,232-243`). The substantive changes are confined to oracle validity: relaxing three pair-vs-S numeric channels, adding per-leaf evidence/ceilings, and adding a stricter within-replicate B-vs-A exactness gate. The final review/authorization stop is procedural, not a change to B1. This scope pass does not cure the causal-boundary oracle blind spot described above. |
| (d) Explicit and complete weakening statement | **FAIL** | Lines 214-220 clearly enumerate selected logits, committed Mamba conv/SSM, and canonical rollback mirrors as no longer pair-vs-S byte-exact, while preserving decisions, selected-row/pending state, KV, evidence completeness, lifecycle, and pair-vs-pair exactness. That is a useful disclosure. It is not complete enough for the operative contract: the opening narrow-scope statement at lines 12-14 omits canonical rollback; the amendment incorporates later executor reseals only generically rather than pinning the v6 authority whose lines 37-40 are being reversed; it does not disclose that the ceilings come only from the first rejection in one B and one A construction; and “No ... pair-vs-pair state ... is weakened” omits the decisive limitation that no causal-case A state exists when the qualifying first-missing-B4 outcome occurs. The asserted reduction-order mechanism is also stronger than the adjudication permits. |
| Amendment changes only the oracle gate | **PASS, with the authority-chain delta below** | A semantic diff against v2/v3 found no hidden change to the intervention, execution order, liveness boundary, replicate threshold, fallback, resume, or identity disciplines. The amendment changes the cross-lane oracle and its evidence requirements only. Its unpinned reference to later reseals should nevertheless be made immutable so that “only” is auditable against the full effective pre-amendment contract. |

## Why the current envelope can mask a distinct effect

The exact B-vs-A gate is the amendment's strongest protection. At every
boundary returned by both packed lanes, it detects even a one-byte intervention
effect regardless of the broader pair-vs-S envelope. That is a sound
within-replicate control and is consistent with the historical Step-7 pattern.

It does not close the boundary that matters most. A qualifying B1 replicate has
B complete the final causal case while A reaches B0-B3 and fails to return at
B4. Amendment lines 91-94 correctly refuse to invent A state, but the result is
that only B-vs-S is observable for the causal post-forward outputs and recurrent
state. Under the proposed envelope, as many as 96,813 function-logit elements
may differ by up to `0.125`; individual SSM leaves may differ at roughly
1.28-1.31 million elements, with maxima up to about `1.04887`; and their signs
and locations are unconstrained. A numeric effect specific to eager B at the
causal transaction can fit those bounds and be classified as the common packed
arithmetic artifact even though there is no returned A tensor with which to
establish commonality.

This does not rewrite the raw liveness observation—B returned and A did not—but
it weakens its interpretation. The campaign could support “pair-only eager
changes completion behavior” while no longer supporting “the intervention is
semantically exact and the waived numeric drift is known to be unrelated to the
causal contrast.” The latter was part of the original exact Phase 1 admission
logic and is required before treating the result as an exact repair rather than
a liveness-only localization.

## Required deltas before resubmission

1. **Replace the two-point maxima with a justified, independently sealed
   calibration contract.** Cover multiple fresh constructions and every
   applicable case/boundary class, or supply a conservative numerical error
   bound derived independently of these observed outcomes. Predeclare the
   calibration population, construction count, statistics, and acceptance rule;
   keep the 12 Phase 1 replicates as held-out evidence. Do not widen or
   recalibrate from Phase 1 observations.

2. **Make every numeric envelope directional and boundary-specific.** Define
   the delta orientation explicitly (for example, `packed - S`) and seal, per
   lane, case/boundary, and full tensor-leaf path, signed lower/upper delta
   bounds plus positive/negative/zero counts or an equivalently discriminating
   signed statistic. Constrain support or a prespecified norm so element drift
   cannot migrate arbitrarily while count and maximum remain unchanged. State
   that a missing statistic, orientation mismatch, new leaf, support violation,
   or any inclusive-ceiling exceedance fails that replicate. Retain the existing
   exact metadata/finite-value checks and zero envelope for conv 0.

3. **Close or explicitly limit the terminal causal-case claim.** Either retain
   exact S equivalence for the waived B channels on the causal case, add an
   independently sealed artifact-matched oracle capable of distinguishing
   ordinary packed drift from B-specific drift there, or narrow the authorized
   conclusion to liveness-only localization and prohibit semantic-equivalence or
   exact-repair advancement until that ambiguity is resolved. Do not describe
   the unavailable causal-case B-vs-A comparison as protected by the
   pair-vs-pair gate.

4. **Correct the weakening and mechanism disclosure.** Include canonical
   rollback in the opening scope statement; state that the empirical envelope
   comes from the first rejection in exactly one B and one A construction;
   replace the reduction-order attribution with the adjudication's supported
   statement (“packing shape, descriptor/capture, or another pair-only numerical
   detail remains unresolved”); and disclose the causal-case comparison gap.

5. **Pin the complete authority chain.** Name and hash the incorporated v4, v5,
   and especially v6 reseals. Identify the exact v6 clauses superseded by this
   amendment. Then restate that all non-oracle clauses remain byte-for-byte
   authoritative.

The current text correctly preserves the wedge and liveness design, and its
per-leaf table faithfully reports the two available observations. Those are
necessary but not sufficient conditions for scheduling. Until the directional,
cross-construction, cross-boundary, and causal-case gaps above are closed, the
amended contract can admit an uncharacterized numeric intervention effect and
**FAIL** is the schedule-safe verdict.
