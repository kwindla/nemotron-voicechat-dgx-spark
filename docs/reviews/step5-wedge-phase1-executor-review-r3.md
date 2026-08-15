# Step 5 wedge Phase 1 real-executor review — round 3

**Review date:** 2026-08-15 UTC  
**Verdict:** **FAIL — the 12-replicate campaign may not be scheduled**

The three narrowed round-2 implementation blockers are closed: drains seal the
corrected tensor actually executed, named controls call the production wrapper
transitions and retain real state, and parent/materializer/epoch paths completely
rehash the model trees before construction. Item (a) and the earlier closed
transport, dispatch, abort, terminal-process, and campaign fail-closed contracts
did not regress.

Scheduling is nevertheless not authorized. The retained round-3 receipt itself
records **0/13 exact B-versus-S cases** (1,416 differences), while the smoke still
passes no S oracle into `_run_lane` and therefore leaves `exactness_errors` empty.
The real campaign does pass S into B and stops on the first such difference. The
round-2 typed-carrier replay delta is also unchanged: materialization checks only
the historical event's byte count, hashes the fixture string, and then sends only
PCM frames through the pipeline. These are pre-campaign validity failures, not
scientific outcomes.

This was a read-only CPU/static confirmation review. No campaign, GPU operation,
container, service, or endpoint was run or changed. The only repository write is
this review file.

## Seal and check results

- Every completion-note hash for the patcher, trace source, oracle, campaign,
  executor, worker, tests, config, v5 preregistration, fixture, and final receipt
  matches the reviewed bytes. The receipt is SHA-256
  `cb8acf48a1840f99b3c20c1895fe83bf87c11a6c49d77a62789342a7602ca3e4`.
- Current `HEAD` is the v5-required
  `1d8163831ef0ba29a1a41e6cab9f0cb4a411bd87`; there is no revision drift.
- The retained receipt binds the 1,439-file atomic tree
  `7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260`,
  manifest `f1e472e...`, and runner `32e0c36c...`. The patcher remains the same
  sealed `eb9324f...` source, and child copy/reinventory/equality checks remain
  intact. The container-only atomic tree was not mounted in this CPU review, so
  no third rebuild was attempted.
- The focused boundary/oracle/campaign suite passed CPU-only: **85 passed**.

## Narrow round-3 blocker disposition

| Item | Disposition | Round-3 evidence |
|---|---|---|
| (c) Executed drain row and live correction assumptions | **CLOSED** | A drain clones `pad["pending"]`, passes that same tensor to `_original_step`, and passes the clone to `_snapshot`; `_snapshot` fails if its content digest differs from `pending_row_sha256_before` (`step5_phase1_gpu_worker.py:1598-1610,1458-1471`). The oracle independently enforces both rejection-to-drain digest continuity and `selected_source_row_identity == pending_row_sha256_before` (`step5_phase1_oracle.py:422-432`). All nine receipt drains satisfy the equality. The exact digests are `2a45d382465463f4ea177e4f106ba2a8b7583d7641b5b874dd12ddbbe15371e2` (text-only), `16a32f7c5a9bd356e7135ce82bf606182096b2cda26e5ef570f533bedaf69348` (function-only), and `00f7d391d7b5e02953e76f97d4f2cfb131ecbdd0003a509d72c2625edc85ac55` (conjunctive), each repeated once in S/B/A. Materialization captures frame-23 `assumed` from the wrapper call context; sequential correction reads that retained value, while packed correction requires live pad state to equal it before use (`step5_phase1_gpu_worker.py:650-664,840-857,973-983,1620-1647,1684-1703`). The receipt retains `{text: 12, function: 12}` for all nine rejections, but source provenance—not the coincidental numeric value—is now enforced. |
| (d) Production named controls and real state | **CLOSED for the requested substitution defect** | The case matrix routes exactly five controls through `_execute_control_boundary`. That function resolves the live production wrapper, turn context, and streaming state, executes `_apply_rnnt_turn_taking` plus `_set_agent_idle` for the three agent controls, and `_apply_fc_state_machine` for the two function controls; it captures real state immediately before and after and rejects a no-op (`step5_phase1_gpu_worker.py:1231-1287,1289-1406`; `step5_phase1_campaign.py:117-121`). There is no `control_state` mapping or alternate control branch elsewhere in the matrix. The receipt retains 15/15 transitions with the correct mechanism and a one-frame context advance: S 23→28, B 29→34, and A 35→40 across the five ordered controls. |
| (f) Complete pre-build model identity in parent and every child | **CLOSED for model artifacts** | The parent completely inventories and hashes Nano, EarTTS, the parent-checkpoint tree, and the HF skeleton, plus the manifest and the template/YAML read during construction (`step5_phase1_real_executor.py:90-190,534-553`). Each child rebuilds the same identity and compares the full mapping and composite digest (`step5_phase1_gpu_worker.py:156-236`). `_build_pipeline` performs that verification immediately before importing/building the pipeline; the otherwise-unused `_build_nano` path does likewise (`:693-705,777-800`). The materializer reaches `_build_pipeline` when `materialize` is requested, and each ordinary epoch reaches it during `construct`, so the lazy materializer return does not bypass the check. A read sweep found model/config consumers only inside the completely inventoried checkpoint, Nano, EarTTS, and skeleton trees, plus the separately hashed manifest and Speech template/YAML. The optional temporary Nano diagnostic route is disabled because the child environment removes `S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK`. The receipt binds composite model SHA-256 `bd12062f...` and full-identity SHA-256 `9d5ccdb6...`; successful materialization necessarily passed the child comparison. |

## No-regression confirmation

- **(a):** the patcher still rejects symlinks, non-regular entries, bytecode, and
  unqualified Python; copies only the 966 indexed sources plus enumerated data
  grafts; validates source and staging; and atomically publishes. Every child
  inventories the complete source before copy, verifies it did not change during
  copy, requires exact copy equality, and selects only that tree.
- Exact tensor transport remains torch-file based; true rollback capture,
  per-leaf receipt seals, observed B2/B3 dispatch, and real abort remain present.
- S/B/A order, terminal A placement, no retry after unknown completion,
  process-group destruction, the two-discordance rule, zero-record preflight
  failure, schema-5 identity/resume discipline, and pair-only B dispatch remain
  fail closed. The focused suite exercises these contracts.

## Remaining campaign blockers

### 1. The retained smoke proves campaign exactness failure

`smoke-result-r3.json` records all 13 B comparisons as `exact: false`, totaling
1,416 differences. The first B case, `text-only-rejection`, already has 116
differences, including `function_logits`, committed KV leaves, Mamba conv/SSM
leaves, rollback-shadow leaves, and committed-state hashes. All 13 A comparisons
are likewise inexact.

The empty top-level `exactness_errors` is not contrary evidence. The smoke calls
`_run_lane(..., oracle_results=None)` for S, B, and A, then computes cross-lane
comparisons only as non-gating observations (`step5_phase1_real_executor.py:816-856`).
The real campaign instead supplies the S results to every non-S lane; `_run_lane`
adds every comparator difference to errors and stops immediately, and
`run_replicate` destroys the epoch on that failure
(`step5_phase1_campaign.py:746-760,780-793`). Therefore the proposed campaign
would reject the first B case, not produce a usable replicate.

The live controls also expose one concrete source of guaranteed inequality: the
single production control owner is deliberately not restored between lanes, so
the retained before/after frame indices are S 23→28, B 29→34, and A 35→40. Those
absolute real-state values are part of `operation_evidence` and thus cannot be
cross-lane exact.

### 2. The frozen typed-carrier recipe is still not replayed or content-verified

The frozen fixture requires replay of retained PCM **and** the frozen typed
carrier. The worker verifies only that the historical `typed_input_accepted`
event's `text_bytes` equals the fixture string length, then hashes the fixture
string into metadata (`step5_phase1_gpu_worker.py:894-910,973-983`). After
`engine.start`, the only submitted inputs are 24 PCM slices through
`engine.process` (`:924-945`). No API consumes `typed_carrier`, and a byte count
cannot establish its content. This is the same unresolved round-2 delta.

## Required deltas before resubmission

1. Resolve the real S/B state differences. Run the non-campaign validation with
   the actual campaign call contract—pass retained S results as `oracle_results`
   for B/A—and fail the smoke if any `cross_lane_observations` entry is inexact.
2. Restore or recreate each control lane at an identical production wrapper,
   context, and streaming-state boundary before executing its five controls, so
   the real before/after evidence is both genuine and cross-lane comparable.
   Do not remove control evidence from the exactness contract to make it pass.
3. Replay and content-bind the frozen typed carrier through the pinned typed-input
   materialization path, or formally reseal the fixture/preregistration with an
   evidence-backed explanation that retained PCM is the complete model input.
   A length check alone is insufficient.
4. Retain the corrected drain/control/model-identity repairs unchanged, then
   produce a fresh one-construction non-campaign receipt that passes the campaign
   comparator before scheduling any of the 12 fresh constructions.

Until those deltas close, **FAIL** is the only schedule-safe verdict.
