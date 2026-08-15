# Step 5 wedge Phase 1 preregistration — v3 validity-contract amendment

**Amendment date:** 2026-08-15 UTC  
**Status:** preregistered design amendment; sealed before execution; pending
adversarial review; campaign not run

## Authority and narrow scope

This amendment follows disposition (ii) of
`reports/step5-wedge/phase1-a-lane-adjudication-20260815.md`, SHA-256
`13be6c132cef0f12a7b485a960839cc0023fa8c12be67d04a88ee345ce0650a8`.
It replaces only the Phase 1 requirement that a packed lane reproduce the S
sequential logical-position oracle bitwise in selected logits and committed
Mamba recurrent tensors.

The governing v2 preregistration is
`reports/step5-wedge/phase1-preregistration.md`, SHA-256
`f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5`.
The v3 compatibility reseal is
`reports/step5-wedge/phase1-preregistration-v3.md`, SHA-256
`8c3494e8f755f9b6f252ba2306b34af7fcd8cecd38e18285a83ee65f202c1d64`.
The later executor reseals remain incorporated, including the v6 logical-state
projections and identity discipline. Nothing below authorizes normalization,
sampling, omission, or post-run selection of evidence.

The retained receipts defining the amendment are:

- B-vs-S round 4:
  `reports/step5-wedge/phase1-executor-smoke/smoke-result-r4.json`, SHA-256
  `47a5ba8a1e8d6969c426007463fabee93df1b0270f07f05db9ea04904fdd19b1`;
- A-vs-S adjudication:
  `reports/step5-wedge/phase1-executor-smoke/smoke-result-alane-20260815.json`,
  SHA-256
  `55b750bb1d8d01b132fc83693c43b05983e1bc8523a35a238b25a1b8d4006df2`.

Both same-construction comparisons found the same corrected path-class
signature: discrete decisions and committed logical KV were exact, while the
selected BF16 function logits and committed Mamba conv/SSM state differed.
The released A pair path (`PIECEWISE`) therefore is not bitwise equal to the S
sequential transaction (`FULL`) on this pinned runtime. The same classes also
differed for eager B (`NONE`), so the drift is not an eager-intervention
discriminator. The supported mechanism is BF16 recurrent-state drift from the
packed scan's reduction order relative to sequential logical-position
execution.

There is no historical Step-7 contradiction. The retained Step-7 gate compared
an unmodified `PIECEWISE` packed pair with a dedicated `query_len=2` `FULL`
packed pair in the same process; it did not construct this Phase 1 S
one-logical-position rejection oracle. Its 1,389/1,389 output/decision and
675/675 cache/rollback exact comparisons establish a pair-vs-pair
reproducibility basis, not pair-vs-sequential bitwise equality. The retained
gate source is `tools/qualification/step7_pair_bitwise_gate.py`, SHA-256
`1d9f4c4d9596af3e25ff4a3e98646b6fd1d7431f8c512d886cdeddb4e5a8c0be`.

## Amended Phase 1 validity contract

### 1. Decision-level equivalence to S remains exact

At every applicable completed boundary, B and A must each equal the S oracle
exactly for:

- emitted text token, emitted function token, and their conjunction;
- accepted-token count;
- selected packed-row number and executed source-row identity;
- retained `function_tokens` and every other retained custom output except the
  selected-logit tensor expressly characterized below;
- host pending-row presence and contents, corrected-row identity, correction
  flags, correction assumptions, and rejection-to-drain continuity;
- committed logical KV, projected to the exact committed-token count; and
- request-relative rollback/shadow index and pointer, virtual-engine identity,
  and all other non-numeric transaction metadata.

These are `EXACT` gates: equality means identical values and, for tensors,
identical dtype, shape, and bytes. A missing field, missing tensor leaf, empty
source identity, changed schema, or any mismatch fails the replicate. The
retained B and A smokes already prove these fields exact at the adjudicated
boundary.

### 2. Packed pair paths must be bitwise reproducible

Within each fresh replicate, every boundary completed by both packed lanes is
compared B-vs-A after restoring both lanes from that replicate's same sealed
prefix. The full committed state must be bitwise identical: selected outputs,
committed KV, committed Mamba conv/SSM tensors, canonical rollback projection,
pending/correction state, and the exact decision fields above. This
pair-vs-pair gate must pass in every qualifying replicate.

“Across replicates” means that this same within-replicate B-vs-A exact gate is
required throughout the preregistered replicate series. It does not license a
transitive byte comparison between unrelated fresh constructions or different
fixture-materialization seals. If terminal A does not return from the causal
case, no post-return state is invented; all preceding common completed
boundaries must have passed the pair-vs-pair gate, and B must have completed
the causal case under the remaining qualification rules.

### 3. Pair-vs-S numeric drift is bounded recorded evidence

Only these S-oracle comparison channels change from exactness gates to
characterized numeric evidence:

1. the selected `function_logits` tensor (and any future selected-logit tensor
   only after a separately sealed amendment supplies its envelope);
2. committed Mamba conv state; and
3. committed Mamba SSM state.

Tensor presence, key set, dtype, shape, byte count, and finite values remain
exact/fail-closed. Every comparison must retain, per leaf, both seals,
different-byte count, first differing byte and offset, different-element
count, first differing element and values, and maximum absolute delta. No
aggregate-only result is admissible.

For each leaf, the preregistered envelope is the componentwise maximum of the
retained round-4 B-vs-S and adjudication A-vs-S observations. A replicate fails
if a listed leaf exceeds either its differing-element ceiling or its
maximum-absolute-delta ceiling. A new differing leaf also fails. Conv layer 0
was exact in both receipts and therefore retains a zero-drift envelope. The
selected-logit maximum-absolute-delta ceiling is exactly `0.125`, not an open
or adjustable tolerance.

| Channel / layer | B differing elements | A differing elements | Ceiling: differing elements | Ceiling: max absolute delta |
|---|---:|---:|---:|---:|
| function logits | 66,528/131,072 | 96,813/131,072 | 96,813/131,072 | `0.125` |
| conv 0 | 0/36,864 | 0/36,864 | 0/36,864 | `0` |
| conv 2 | 9,559/36,864 | 9,559/36,864 | 9,559/36,864 | `0.0703125` |
| conv 4 | 9,226/36,864 | 9,229/36,864 | 9,229/36,864 | `0.0625` |
| conv 6 | 9,069/36,864 | 9,164/36,864 | 9,164/36,864 | `0.0625` |
| conv 7 | 9,319/36,864 | 9,343/36,864 | 9,343/36,864 | `0.0625` |
| conv 9 | 9,207/36,864 | 9,504/36,864 | 9,504/36,864 | `0.0625` |
| conv 11 | 9,317/36,864 | 9,702/36,864 | 9,702/36,864 | `0.125` |
| conv 13 | 10,742/36,864 | 10,787/36,864 | 10,787/36,864 | `0.3125` |
| conv 16 | 11,157/36,864 | 11,296/36,864 | 11,296/36,864 | `0.078125` |
| conv 18 | 11,246/36,864 | 11,311/36,864 | 11,311/36,864 | `0.0625` |
| conv 20 | 11,277/36,864 | 11,553/36,864 | 11,553/36,864 | `0.09375` |
| conv 23 | 11,138/36,864 | 11,646/36,864 | 11,646/36,864 | `0.078125` |
| conv 25 | 10,891/36,864 | 11,342/36,864 | 11,342/36,864 | `0.08203125` |
| conv 27 | 11,021/36,864 | 11,317/36,864 | 11,317/36,864 | `0.0625` |
| conv 29 | 10,813/36,864 | 11,179/36,864 | 11,179/36,864 | `0.092529296875` |
| conv 32 | 10,707/36,864 | 11,093/36,864 | 11,093/36,864 | `0.0625` |
| conv 34 | 10,627/36,864 | 11,052/36,864 | 11,052/36,864 | `0.046875` |
| conv 36 | 10,466/36,864 | 11,026/36,864 | 11,026/36,864 | `0.05126953125` |
| conv 38 | 10,412/36,864 | 11,050/36,864 | 11,050/36,864 | `0.078125` |
| conv 41 | 10,505/36,864 | 11,086/36,864 | 11,086/36,864 | `0.10546875` |
| conv 43 | 10,499/36,864 | 11,044/36,864 | 11,044/36,864 | `0.109375` |
| conv 44 | 10,448/36,864 | 10,966/36,864 | 10,966/36,864 | `0.0625` |
| conv 46 | 10,334/36,864 | 10,951/36,864 | 10,951/36,864 | `0.09375` |
| conv 48 | 10,343/36,864 | 10,850/36,864 | 10,850/36,864 | `0.078125` |
| conv 50 | 10,177/36,864 | 10,666/36,864 | 10,666/36,864 | `0.0665283203125` |
| conv 52 | 10,032/36,864 | 10,579/36,864 | 10,579/36,864 | `0.0703125` |
| conv 54 | 10,183/36,864 | 10,621/36,864 | 10,621/36,864 | `0.0625` |
| SSM 0 | 1,300,844/1,310,720 | 1,300,844/1,310,720 | 1,300,844/1,310,720 | `0.1800537109375` |
| SSM 2 | 1,292,865/1,310,720 | 1,292,907/1,310,720 | 1,292,907/1,310,720 | `1.0488662719726562` |
| SSM 4 | 1,297,204/1,310,720 | 1,298,057/1,310,720 | 1,298,057/1,310,720 | `0.12976837158203125` |
| SSM 6 | 1,300,040/1,310,720 | 1,297,957/1,310,720 | 1,300,040/1,310,720 | `0.236175537109375` |
| SSM 7 | 1,304,495/1,310,720 | 1,303,906/1,310,720 | 1,304,495/1,310,720 | `0.03092193603515625` |
| SSM 9 | 1,281,447/1,310,720 | 1,284,473/1,310,720 | 1,284,473/1,310,720 | `0.0077686309814453125` |
| SSM 11 | 1,305,488/1,310,720 | 1,306,057/1,310,720 | 1,306,057/1,310,720 | `0.010873198509216309` |
| SSM 13 | 1,309,525/1,310,720 | 1,308,923/1,310,720 | 1,309,525/1,310,720 | `0.0101662278175354` |
| SSM 16 | 1,309,248/1,310,720 | 1,309,773/1,310,720 | 1,309,773/1,310,720 | `0.0027780625969171524` |
| SSM 18 | 1,309,205/1,310,720 | 1,309,348/1,310,720 | 1,309,348/1,310,720 | `0.0022537708282470703` |
| SSM 20 | 1,309,234/1,310,720 | 1,309,490/1,310,720 | 1,309,490/1,310,720 | `0.0009598731994628906` |
| SSM 23 | 1,310,339/1,310,720 | 1,310,433/1,310,720 | 1,310,433/1,310,720 | `0.0009040674194693565` |
| SSM 25 | 1,309,740/1,310,720 | 1,310,046/1,310,720 | 1,310,046/1,310,720 | `0.0036001205444335938` |
| SSM 27 | 1,309,791/1,310,720 | 1,309,861/1,310,720 | 1,309,861/1,310,720 | `0.0009972825646400452` |
| SSM 29 | 1,310,253/1,310,720 | 1,310,323/1,310,720 | 1,310,323/1,310,720 | `0.004374504089355469` |
| SSM 32 | 1,309,338/1,310,720 | 1,309,882/1,310,720 | 1,309,882/1,310,720 | `0.0006764233112335205` |
| SSM 34 | 1,309,978/1,310,720 | 1,310,313/1,310,720 | 1,310,313/1,310,720 | `0.0013350173830986023` |
| SSM 36 | 1,309,440/1,310,720 | 1,309,749/1,310,720 | 1,309,749/1,310,720 | `0.0036103129386901855` |
| SSM 38 | 1,307,964/1,310,720 | 1,309,254/1,310,720 | 1,309,254/1,310,720 | `0.17057418823242188` |
| SSM 41 | 1,309,928/1,310,720 | 1,310,296/1,310,720 | 1,310,296/1,310,720 | `0.0022635385394096375` |
| SSM 43 | 1,309,741/1,310,720 | 1,310,235/1,310,720 | 1,310,235/1,310,720 | `0.0055048465728759766` |
| SSM 44 | 1,307,521/1,310,720 | 1,307,988/1,310,720 | 1,307,988/1,310,720 | `0.008563995361328125` |
| SSM 46 | 1,308,886/1,310,720 | 1,309,942/1,310,720 | 1,309,942/1,310,720 | `0.008513808250427246` |
| SSM 48 | 1,309,857/1,310,720 | 1,310,275/1,310,720 | 1,310,275/1,310,720 | `0.006132166832685471` |
| SSM 50 | 1,309,415/1,310,720 | 1,309,765/1,310,720 | 1,309,765/1,310,720 | `0.008420825004577637` |
| SSM 52 | 1,309,573/1,310,720 | 1,309,502/1,310,720 | 1,309,573/1,310,720 | `0.012223243713378906` |
| SSM 54 | 1,309,578/1,310,720 | 1,309,812/1,310,720 | 1,309,812/1,310,720 | `0.013503193855285645` |

The canonical rollback conv/SSM tensors are duplicate logical projections of
the committed recurrent state at the transaction boundary, not independent
numeric channels. Their S-oracle differences must be recorded leaf by leaf and
must match the corresponding committed-state statistics and envelope above.
Rollback leaf-set, dtype, shape, logical layout, pointer/index, and
virtual-engine identity remain exact. Any extra rollback difference or any
rollback statistic that does not mirror its committed leaf fails the
replicate.

This envelope is fixed before execution. It may not be widened, rounded upward,
or recalculated from campaign observations. Falling within it is validity
evidence only; it is not proof that pair and sequential arithmetic are bitwise
equal.

### 4. Every other sealed rule is unchanged

All other v2/v3 rules and subsequent executor reseals remain in force,
including:

- fixed lane order S, B, then terminal A; no post-A flip back;
- the frozen 13-case order and rejection/drain transaction definition;
- exactly 12 fresh-engine replicates and the same frozen-prefix restoration;
- B0--B9 completeness for S/B and first-missing-B4 localization for causal A;
- terminal destruction after A timeout, crash, unknown completion, or invalid
  evidence, with no resume, retry, cancellation recovery, or replay on that
  epoch;
- the two-discordant-replicate rule for B1 and reporting of all 12 outcomes;
- all zero-record and per-replicate `INCOMPLETE` dispositions;
- construction-stratified fallback, noninterference, append-only evidence, and
  contiguous resume discipline; and
- complete image, model, checkpoint-tree, qualified-source, patched-tree,
  implementation, configuration, fixture-materialization, request, row, and
  source-artifact identity checks.

## What is weakened, and why it is justified

Relative to v2, this amendment weakens exactly one assertion: B-vs-S and A-vs-S
are no longer required to be bitwise equal for selected logits, committed
Mamba conv/SSM contents, and the canonical rollback tensors that mirror those
recurrent contents. Those channels instead must remain inside the sealed
per-leaf envelope above. No decision, selected-row identity, pending/correction
state, committed KV, pair-vs-pair state, evidence-completeness, or lifecycle
gate is weakened.

The A-lane evidence justifies the change because the released packed graph path
and eager B path independently have the same 107-path corrected divergence
signature against their within-construction S oracle, including an exact
`0.125` selected-logit maximum, exact committed KV, the same 26 nonzero conv
leaf paths, and all 27 SSM leaf paths. A and B differ in dispatch mode but share
packed scan arithmetic. The parsimonious validity boundary is therefore exact
decisions/KV and exact packed-pair reproducibility, with BF16 reduction-order
drift against sequential execution characterized rather than mistaken for an
intervention effect.

## B1 discrimination is unchanged

B1 is a liveness question: under the same frozen prefix and fixed order, does
pair-only eager B complete while the released terminal A path loses completion
with first missing B4? The amended numeric contract removes a common
pair-vs-sequential arithmetic artifact from the admissibility gate; it does not
alter either lane's dispatch, execution, boundary trace, completion predicate,
or required B-then-A discordance. B1 still requires at least two qualifying
discordant fresh replicates, every B completing, all exact decision/KV and
pair-vs-pair gates passing, every numeric drift observation remaining inside
the sealed envelope, and all 12 outcomes being reported. Thus wedge
discrimination is unaffected.

## Seal and stop condition

The SHA-256 of this amendment must be recorded externally before any preflight,
smoke, replicate, or campaign execution. This increment authorizes no run.
After the SHA is recorded, work stops and the amendment goes to adversarial
review. Any execution before that review and a subsequent explicit
authorization is campaign-inadmissible.
