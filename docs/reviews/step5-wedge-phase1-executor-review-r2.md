# Step 5 wedge Phase 1 real-executor review — round 2

**Review date:** 2026-08-15 UTC  
**Verdict:** **FAIL — the 12-replicate campaign may not be scheduled**

The atomic-tree and recorded-dispatch repairs are genuine, and the focused CPU
suite passes. The round-1 blockers are not all closed, however. Child workers do
not reverify the model artifacts included in the parent's schema-3 identity;
pending-drain snapshots identify the original frame-22 row rather than the
corrected row actually executed; the typed carrier is attested but never replayed;
and named controls remain ordinary token steps plus harness-local state. The final
smoke also retained 13/13 inexact B-versus-S comparisons while excluding those
differences from `exactness_errors`. That receipt is honestly non-campaign, but it
cannot authorize spending 12 fresh constructions on a path whose campaign
comparator would reject the first B case.

This was a CPU-only/static confirmation review. No campaign, GPU operation,
container, service, or endpoint was run or changed. The only repository write is
this review file.

## Seal and check results

- The completion-note hashes for the patcher, trace source, oracle, campaign,
  executor, worker, focused tests, frozen config, frozen fixture, and final smoke
  receipt match the reviewed bytes.
- Two retained independent patch builds each contain exactly 1,439 regular files
  and reproduce tree SHA-256
  `7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260`,
  manifest SHA-256
  `f1e472e597dc846a3d366fe1514fbc6bcb50599ef5cd3d56a9d88a9c514559d2`,
  and runner SHA-256
  `32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447`.
  The tree consists of 966 qualified Python files, 471 explicitly inventoried
  binary/data grafts, the added trace module, and the patch manifest; it has no
  bytecode or symlinks.
- The focused boundary/oracle/campaign suite passed: **78 passed**. Task-scoped
  Ruff and `py_compile` passed.
- The current checkout is at `1d8163831ef0ba29a1a41e6cab9f0cb4a411bd87`,
  while the completion note recorded `e17e57e...` and the campaign currently
  requires `cf0e9ca...`. This drift fails closed before execution; it is not the
  basis of the verdict, but the current checkout cannot launch the campaign as-is.

## Requirement disposition

| Item | Disposition | Round-2 evidence |
|---|---|---|
| (a) 1,439-file atomic tree, per-child reverification, no image-package leakage | **CLOSED** | The patcher rejects symlinks, non-regular entries, caches, and unqualified Python, copies only the 966 qualified sources plus the explicit 471-file graft inventory, adds the trace and manifest, and atomically publishes. `_copy_qualified_overlay` inventories and hashes the parent tree before copy, inventories it again after copy, requires the child copy to be exactly equal, selects only that copy on `sys.path`, and rechecks runner/manifest hashes before any vLLM import (`patch_step5_wedge_boundary_trace.py:58-120,458-538`; `step5_phase1_gpu_worker.py:99-149,254-284,562-582`). There is no installed-image package copy or overlay merge. |
| (c) Runtime-observed snapshots, true shadow, row identities, retained leaf seals | **OPEN — BLOCKER** | Emitted outputs, accepted count, KV/Mamba state, true shadow/index, pending state, and most correction state are now read from the live runtime; missing shadow fails closed (`step5_phase1_gpu_worker.py:356-463,1125-1274`). `_result_summary` now retains canonical per-leaf seals for snapshots and dispatch evidence (`step5_phase1_oracle.py:105-176`; `step5_phase1_campaign.py:594-610`). But every pending drain actually executes the corrected pending tensor (`step5_phase1_gpu_worker.py:1306-1314`) while `_snapshot` always hashes `source_rows[selected]`, where `source_rows` is the original frame-22/frame-23 pair (`:1172-1189,1409-1418`). In the retained receipt, all nine lane/rejection drains report frame-22 SHA-256 `8c814d...` as `selected_source_row_identity`, while their executed `pending_row_sha256_before` values are `2a45d3...`, `16a32f...`, or `00f7d3...`. The source-row evidence therefore identifies a tensor that was not executed. Correction assumptions are also manufactured as constant PAD/PAD mappings (`:1342-1346,1394-1398`) rather than read from the runtime's recorded assumption state. |
| (d) Recorded B2/B3 dispatch and real named operations | **OPEN — BLOCKER (dispatch and abort closed; controls open)** | Dispatch is now derived from the latest unique, non-truncated B2/B3 detail records. Selected B requires the observed selector plus pinned eager spellings `NONE`/`CUDAGraphMode.NONE`; non-B requires selector false, and the campaign requires full SHA-256 descriptor keys (`step5_phase1_gpu_worker.py:203-251`; `step5_phase1_campaign.py:613-638`). Abort invokes and verifies real `abort_generation` before restore (`step5_phase1_gpu_worker.py:1347-1369`). Named controls, however, still mutate only a harness-local `control_state` mapping and invoke the same `_original_step(row)` used for ordinary one-token work (`:1065-1069,1074-1123,1370-1374`). Verifying forced output token IDs is an improvement, but it does not execute or observe the named agent/function transition in the actual wrapper/turn state; it is still the round-1 ordinary-step substitution with local bookkeeping. |
| (f) One schema-3 used-runtime identity and child reverification | **OPEN — BLOCKER** | The selected config bytes must equal the frozen repository config, their direct file hash is used, and the launcher-supplied image ID must equal the configured immutable ID (`step5_phase1_real_executor.py:234-260,513-535,695-715`). The parent derives the model identity from manifested Nano/EarTTS trees, the parent model, the skeleton, and manifest (`:117-164`). But `_verify_used_runtime_identity` in every child checks only the repository config, executor, worker, runtime-optimization source, server source, patch manifest, and semantic environment (`step5_phase1_gpu_worker.py:126-149`); it never rehashes or compares `model_artifact_identity`, the model trees, parent checkpoint, skeleton, or manifest before `_build_nano`/`_build_pipeline` uses them. This leaves a direct parent-seal-to-child-use TOCTOU gap and contradicts the completion note's per-child artifact-reverification claim. The parent model identity also omits checkpoint files the pipeline demonstrably reads—at least `checkpoint_root/config.json` and `rnnt_tokenizer/tokenizer.model` (`server.py:4491-4503`)—so it is not yet the identity of all used model artifacts. |

## Smoke-loop repair assessment

The process-lifecycle changes do not weaken the campaign contract. Workers use a
new session, timeout/failed shutdown kills the complete process group, process
waits are bounded, and inherited stderr collection is bounded
(`step5_phase1_real_executor.py:304-392,427-452`). Nonterminal timeouts raise;
only the preregistered causal A case can return `terminal_unknown`, and the lane
then stops before epoch destruction (`:625-669`; `step5_phase1_campaign.py:678-754`).

The structural/scientific distinction is safe inside the actual campaign:
`run_replicate` still passes S snapshots into B/A and fails on any exactness
difference. The smoke is also explicitly `campaign_admissible: false` and retains
all comparisons. The scheduling interpretation is not safe, though:
`run_single_engine_smoke` passes `oracle_results=None` for S, B, and A, so its
`exactness_errors` cannot include cross-lane failures
(`step5_phase1_real_executor.py:787-825`). The final receipt records every one of
the 13 B-versus-S cases as inexact (1,386 retained differences total), including
116 differences on the first text-only rejection. The campaign would classify
that first B comparison as an exactness failure and stop B. Exactness is a
validity prerequisite, not a scientific causal verdict, so reclassifying it as
merely a non-campaign observation makes this smoke unsuitable as the schedule
gate even though it does not alter campaign adjudication.

The smoke also did not exercise terminal-unknown process-group teardown: all 13
A cases completed, `destroy_reason` is `single-engine-smoke-complete`, and the
worker exited through ordinary close. Finally, materialization hashes the typed
carrier and checks that the historical event reports the same byte length, but
after `engine.start` it processes only PCM frames; the carrier string is never
submitted to the pipeline (`step5_phase1_gpu_worker.py:761-790,791-838`). This
does not satisfy the frozen fixture's requirement to replay retained PCM **and**
the typed carrier.

## Required deltas before resubmission

1. Reverify the complete model-artifact identity inside every materializer and
   epoch child immediately before model construction/use. Include every used
   checkpoint artifact, including config and tokenizer, and fail closed on any
   parent/child inventory difference.
2. Pass the tensor actually executed by each operation into snapshot creation.
   For a pending drain, `selected_source_row_identity` must equal the corrected
   pending-row content SHA-256. Derive correction assumptions from observed pad
   state and add a negative test for substituted row identities/assumptions.
3. Execute named agent/function transitions through the real runtime mechanism
   and snapshot the resulting real control/turn state; do not use a harness-only
   state machine as evidence. Retain the existing real abort path.
4. Replay the exact frozen typed-carrier content through the pinned pipeline, not
   merely its recorded byte length, before sealing frame 22/23.
5. Resolve the retained S/B exactness failures and run a separately authorized,
   non-campaign one-construction validation that applies the campaign comparator
   to S/B. If terminal destruction is claimed as smoke evidence, make that smoke
   actually enter the causal terminal-unknown path and verify group teardown.

Until those deltas close, the correct disposition remains **FAIL**.
