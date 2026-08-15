# Step 5 Phase-1 round-4 gated-smoke audit

Receipt: `smoke-result-r4.json`  
Receipt SHA-256: `47a5ba8a1e8d6969c426007463fabee93df1b0270f07f05db9ea04904fdd19b1`  
Outcome: **campaign blocked by genuine B-vs-S semantic divergence**

The smoke passed the completed S result map as the oracle argument to B. It
stopped at the first B case, `text-only-rejection`, retained the receipt, and
destroyed the worker/process group. A was not run after this fail-closed stop.

## Audit of all round-3 difference classes

The round-3 receipt reported the same 1,416 differences for B-vs-S and A-vs-S:

| Round-3 class | Count per lane | Classification and round-4 treatment |
|---|---:|---|
| `position_one_boundary_state` whole-state digest | 13 | Representation/provenance artifact. The digest includes physical/lane-history state and is retained but not cross-compared. |
| pre/post committed whole-state digests | 26 | Representation artifact. They hash lane-absolute physical state; retained for within-lane restore/drain validation only. |
| `operation_evidence` | 30 | Retained evidence containing shared-control-owner frame/function state and physical allocation. Lane-absolute by construction; not cross-compared. |
| uncommitted-position whole-state digest | 3 | Within-lane rejection/restore evidence, not a committed boundary equality field. |
| selected `function_logits` | 12 | Committed semantic output. It remains in the equality set; any difference is genuine. |
| committed KV | 48 | The full owned block included unused tail bytes. Round 4 projects ordered KV to the exact committed-token count. No KV difference remains in the first gated B comparison. |
| committed Mamba conv | 312 | Already a lane-invariant committed state. It remains in the equality set; any difference is genuine. |
| committed Mamba SSM | 324 | Already a lane-invariant committed state. It remains in the equality set; any difference is genuine. |
| raw rollback shadow | 648 | Raw B/A pair buffers and stale S buffers had different names/layout/history. Round 4 compares a canonical transaction position-one Mamba projection and retains raw leaves as evidence. Any canonical difference is genuine. |

Counts sum to 1,416 per compared lane. No round-3 class is unclassified.

## Corrected first-B result

The first gated B case has 107 committed-boundary differences:

| Corrected class | Leaves | Evidence |
|---|---:|---|
| selected `function_logits` | 1 | 66,528/131,072 BF16 elements differ; first differing element is `-4.125` (S) vs `-4.09375` (B); max absolute delta `0.125`. |
| committed Mamba conv | 26 | Differences begin in the third 12,288-wide conv-history channel (first offsets 24,576 or 24,577); per-leaf max absolute delta is retained. Layer 0 is exact. |
| committed Mamba SSM | 27 | Each retained SSM leaf differs, generally across nearly all 1,310,720 FP32 elements; leaf-level first values and max absolute deltas are retained. |
| canonical rollback projection | 53 | Exactly the same 26 conv plus 27 SSM position-one divergence, now expressed in committed Mamba names/layouts. This confirms the raw-shadow representation artifact was fixed without hiding the semantic state difference. |
| committed KV | 0 | Exact after excluding only unused physical block-tail bytes. |

Tokens, selected packed/source-row identity, accepted count, `function_tokens`,
pending/correction state, request-relative shadow identity, and committed KV are
exact. The selected logits and committed recurrent state are not. The evidence
therefore demonstrates a genuine numerical/semantic difference between the
sequential logical-position oracle and packed pair execution under the pinned
runtime. Per the preregistered exactness rule, the campaign must not start and
the comparator must not normalize these leaves.

## Typed-carrier materialization

The smoke reached the lane executions only after materialization successfully:

1. verified the historical browser-log witness SHA-256;
2. extracted its unique 58 historical UTF-8 carrier bytes and matched their
   SHA-256/event job identity;
3. synthesized those bytes through the production deterministic Pocket
   backend; and
4. required the synthesized first 61,440 PCM bytes to equal the retained
   post-discard 24-frame prefix byte-for-byte.

Any failure in those checks raises before S is restored, so the retained S and
B lane evidence proves the replay gate completed. The final materialization
seal is `e3ab063bb9e74c2ca6463a62b890a3e578b2cf8b2f697502b958e6e0789082c9`.
