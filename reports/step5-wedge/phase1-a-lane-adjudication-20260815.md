# Step 5 wedge Phase-1 A-lane adjudication — 2026-08-15

## Disposition

**Outcome (ii): A-vs-S diverges in the same corrected semantic classes as
B-vs-S.** The released packed graph path does not reproduce the sequential
logical-position ground truth bitwise on this pinned runtime. The round-4
B-vs-S failure therefore cannot be isolated to the eager intervention. Phase 1
remains campaign-inadmissible; no campaign was started and no exact leaf was
normalized away.

This is a same-construction, S-oracle-gated result. The completed S lane ran all
13 structural cases. A was restored from the same materialized prefix, completed
the first case (`text-only-rejection`) through B0–B9, and was compared inside
`_run_lane` before the harness stopped. S recorded `graph_mode=FULL`; A recorded
the released packed path with `graph_mode=PIECEWISE`, `selector_fired=false`,
and `dispatch_mode=normal`. The receipt was written atomically and the terminal
worker/process group was destroyed.

The safe claim is about the released packed pair path versus sequential ground
truth. This run does not by itself identify whether packing shape, its
PIECEWISE descriptor/capture, or another pair-only numerical detail is the
narrowest cause. In particular, the repeated `0.125` logit maximum is not an
eager-mode discriminator: it occurs in both B (`NONE`, eager) and A
(`PIECEWISE`, graph) comparisons against S (`FULL`, sequential submissions).

## Why retained round-3 evidence was not sufficient

Offline adjudication was attempted first, as requested. Round 3 contains all
S/B/A cases and 151 snapshot-seal leaves for the first case. Its A and B first
case agree on 147/151 leaf hashes; the only A/B differences are four
lane/history whole-state digest leaves. That is useful historical evidence that
the old packed eager and packed graph observations agreed with one another.

It is not a corrected A-vs-S adjudication:

- round 3 is receipt schema 2, predates the corrected logical committed-KV and
  canonical rollback projections, and retains only hashes of the old full
  tensors; a committed-token slice cannot be recovered from a full-tensor hash;
- among the 97 leaf paths common to the round-3 and round-4 first-B snapshots,
  58 hashes differ, including `function_logits`, committed KV, and committed
  recurrent-state leaves; and
- the retained tensor seals contain digests, not tensor bytes, so the corrected
  projection and element-level comparison cannot be reconstructed after the
  fact.

The round-3 receipt is therefore retained context, not the decision receipt.
Its SHA-256 is
`cb8acf48a1840f99b3c20c1895fe83bf87c11a6c49d77a62789342a7602ca3e4`.

## Authorized fallback smoke

One non-campaign single-engine structural smoke was run with the existing
corrected runner after overriding only its in-process lane order to `("S",
"A")`. No source file was changed for the override. The existing `_run_lane`
call received the completed S result map as A's oracle, so A was exactness-gated
before it could advance to a second case. B was not run.

An initial container launch lacked the frozen fixture's host-absolute read-only
source-artifact mount and failed in `load_frozen_fixture()` before runtime
construction, materialization, or lane execution; it emitted no receipt. The
mount-only correction then produced exactly one real runtime construction and
the retained receipt below. This bounded pre-construction failure is not an
extra engine observation.

Decision receipt:
`phase1-executor-smoke/smoke-result-alane-20260815.json`

- receipt SHA-256:
  `55b750bb1d8d01b132fc83693c43b05983e1bc8523a35a238b25a1b8d4006df2`;
- kind: `phase1-real-pcm-single-engine-structural-smoke`;
- `campaign_admissible=false`;
- `real_runtime_constructions=1`;
- `destroy_reason=single-engine-smoke-exactness-failure`;
- `terminal_worker_destroyed=true`;
- sealed image:
  `ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`;
- model identity:
  `bd12062f1afc92079c92b739c56b542c50778f8edab44b2a3896d5a79fbe6dc2`;
- model-artifact identity:
  `9d5ccdb6fa38bd23c90447f529225e4b60083196312faced11347fce76dd8f9e`;
- patched runner:
  `32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447`;
- patched 1,439-file tree:
  `7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260`;
- comparator/oracle source:
  `be23e35df47155d4790941f89605346a60e5377b7de7954c61b66fddce2d0ebd`;
- worker source:
  `a1dd31b6a7841f798865f23cd2a44a7bd3952d5095f826fad5d6694e54e054f5`;
- fixture materialization:
  `6a8aae9e884e28ed27c6f3bbfe2730a457a6e3e66479bdb262fd818c7e7f4c9b`;
- first-case S snapshot seal:
  `1bf1d40e7b3f35e6c2bc790f19b5c3b37c46a8cb9264b85418207a57cb4ee0a8`;
- first-case A snapshot seal:
  `4efc9a73138d5efe10c786ee6a4153d9fd39d1c09c876f2113951f7aaa0f3e8e`.

The two materialized fused-row seals are the same as round 4
(`8c814df8…` and `87fc1e8f…`). Fresh construction nevertheless produced a
different full frame-22 state/materialization seal. This is another reason not
to compare B and A numerical values transitively across receipts; each decisive
comparison is the within-construction lane against its own S oracle.

## Three-way corrected class table

Both gates stopped at the same first case. “Rollback” below is the canonical
transaction projection and duplicates the committed recurrent divergence as a
rollback-boundary assertion; it is not 53 additional independent state tensors.

| Corrected class | Round-4 B-vs-S (`packed`, eager `NONE`) | A-vs-S (`packed`, released graph `PIECEWISE`) | Adjudication |
|---|---:|---:|---|
| emitted tokens/conjunction, accepted count, selected row/source identity | exact | exact | discrete decision exact |
| `function_tokens` | exact | exact | selected function decision exact |
| selected `function_logits` | 1 differing leaf; 66,528/131,072 BF16 elements; max delta `0.125` | 1 differing leaf; 96,813/131,072 BF16 elements; max delta `0.125` | same class and maximum; detailed values differ across fresh constructions |
| committed logical KV | 0 differences; all 4 tensor leaves exact | 0 differences; all 4 tensor leaves exact | exact after committed-token projection |
| committed Mamba conv | 26/27 tensor leaves differ; 268,715 elements total; per-leaf max-delta range `0.03125`–`0.3125` | 26/27 tensor leaves differ; 277,916 elements total; range `0.0390625`–`0.3125` | same 26 layer paths; layer 0 exact in both |
| committed Mamba SSM | 27/27 tensor leaves differ; 35,270,937 elements total; per-leaf max-delta range about `0.000540093`–`1.04887` | 27/27 tensor leaves differ; 35,279,411 elements total; range about `0.000446640`–`1.04887` | same 27 layer paths |
| canonical rollback conv/SSM | 53 differing leaves | 53 differing leaves | same recurrent paths repeated at the transaction boundary |
| **total comparator differences** | **107** | **107** | **identical corrected path-class signature** |

The A receipt has 108 textual `exactness_errors`: the 107 comparator
differences plus the expected fail-closed bookkeeping error that the rejected
row was left undrained when the gate stopped immediately. The scientific count
is the 107 entries in `cross_lane_observations.A[text-only-rejection]`.

## Selected leaf evidence

The receipt retains complete per-leaf dtype, shape, byte count, and SHA-256
evidence. Representative leaves are:

| Leaf | S seal | A seal | Result/evidence |
|---|---|---|---|
| `function_tokens` | `220dde27afee4d537cac96d85d6546f825153b90e828931b74e103807541bc42` | same | exact |
| `function_logits` | `6b6c2dfe25c6291105a46cb72a9d4ec61076d44c784f8265617df7b9007a2922` | `d2cd7f2789f69c09af6a8631878664ec05d71b11139d8a508c60d353544a178e` | 97,175/262,144 bytes differ; first values `-4.21875` vs `-4.25`; max delta `0.125` |
| committed KV layer 14 | `6c81e4effb4156dd733ff941c6b70e0c21bb639d77e0cb67af49976d7fea5857` | same | exact |
| committed KV layer 21 | `74a2366a7f8038ba5292e455463e975768bfafa262ecfda0e221019e01f1a197` | same | exact |
| committed KV layer 30 | `1d7afb06b084b6f1b2a5b9e3be14236470878d8c8a1f83797ef5ee4b0fc665a2` | same | exact |
| committed KV layer 39 | `b14db3d0a178e1db81b2444ffe707a6da6726135d305c6914bdba13b4e529c66` | same | exact |
| Mamba conv layer 0 | `9702442ca5b14cf2f9fe7c6ac3a506c87573055a8e4a9df89666d2ff05779928` | same | exact |
| Mamba conv layer 11 | `aa983d46a8dbd972741af41450fb6d23eba230e9c7957d0d6ba778daf2c0fd68` | `74d24c61182147d96e24d8ae431785723698daf63292b2a67889b3c4a22a7480` | 9,702/36,864 BF16 elements differ; divergence starts at offset 24,576; max delta `0.125` |
| Mamba SSM layer 0 | `fdb2a2abe75351dff811d3c719d620b2e0a33bc580912afcd1360afc49de36a3` | `cd83c69d2c21582a4c28bbf1d756d7b7229ce5587708078a1e90b5da4a384095` | 1,300,844/1,310,720 FP32 elements differ; max delta `0.1800537109375` |

The last SSM pair is particularly strong three-way evidence: round 4 retained
the identical S layer-0 seal `fdb2…` and B layer-0 seal `cd83…`. Thus both B
and A independently reach the same differing layer-0 contents against S even
though they used eager and graph dispatch respectively.

Round-4 B receipt:
`phase1-executor-smoke/smoke-result-r4.json`, SHA-256
`47a5ba8a1e8d6969c426007463fabee93df1b0270f07f05db9ea04904fdd19b1`.

## Step-7 reconciliation question

The current result conflicts with any reading of the historical Step-7 gate as
proof that packed pair execution was bitwise equal to sequential
logical-position ground truth. The retained Step-7 summaries report
1,389/1,389 output/decision comparisons and 675/675 cache/rollback comparisons
exact. However, the retained gate source describes and implements two packed
requests: an “old” unmodified PIECEWISE pair lane and a “new” dedicated
`query_len=2` FULL pair-graph lane. It does not construct the Phase-1 S oracle's
one-logical-position rejection transaction. It also uses a different silent
turn teacher replay, enables the dedicated pair FULL graph, and predates the
current FHW8/runtime/fixture seals.

The reconciliation question that must be answered before any further exactness
claim is:

> Did Step 7 retain any mode- and artifact-matched comparison of the packed
> pair boundary against a true sequential logical-position boundary, including
> selected `function_logits`, committed-token KV, committed Mamba conv/SSM, and
> canonical rollback? If yes, identify that receipt and explain why its
> descriptor, runtime, prefix materialization, and boundary differ from the
> current same-construction failure. If no, relabel Step 7 as PIECEWISE-pair
> versus FULL-pair equivalence rather than pair-versus-sequential exactness.

Relevant historical seals:

- `tools/qualification/step7_pair_bitwise_gate.py`:
  `1d9f4c4d9596af3e25ff4a3e98646b6fd1d7431f8c512d886cdeddb4e5a8c0be`;
- `reports/step9-cold-start-20260807.md`:
  `8fd41c233a772dc1297a23dbff3b654073273d84361b88ab5ab1aab90df31152`;
- `reports/pc2-hotfix-regate-20260807.md`:
  `211d75b3ab746b9f9635a81b5a03c291b0706be080f3ce56a7ca2852f48ecec8`;
- wedge-closure design:
  `a7c8c44419d9d7eb53d559e82e2cb0402985601487e9702751a29253dff322af`.

The separate Step-5 graph-basin report remains relevant localization context,
not a waiver. Its `+0.125`/`0.0` diagnostic margins involved graph-engine token
basins and a workload-changing text-logit overlay. The present independent
`0.125` maxima in both eager B and graph A show that magnitude alone cannot
attribute the Phase-1 recurrent divergence to eager mode.

## Operational disposition

- No 12-replicate or other campaign was run.
- No comparator tolerance or design amendment was adopted.
- The exactness gate remains failed for both B and A on the pinned runtime.
- No commit was created.
- The original production service was restarted on
  `sha256:f90a0776c9189721c67244f1e3faf47ba1fbbfd872b58b9ceb2555933d506c49`;
  final readiness is recorded in the completion note.

