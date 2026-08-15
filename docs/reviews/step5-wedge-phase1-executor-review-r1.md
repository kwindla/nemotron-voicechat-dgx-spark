# Step 5 wedge Phase 1 real-executor review — round 1

**Review date:** 2026-08-14 UTC  
**Verdict:** **FAIL — the 12-replicate campaign may not be scheduled**

The implementation is a substantial structural advance, and the final smoke is
honestly labelled inadmissible.  It does not yet implement an auditable real
campaign under the sealed v2/v3 contract.  In particular, the worker reports
several oracle values and the B dispatch mode from the requested plan rather
than from device/runtime observations, the retained campaign record discards
the exactness snapshots, and the only GPU smoke bypassed the PCM materializer
and all packed/terminal lanes.

This was a read-only CPU/static review except for this file.  No GPU campaign,
container, service, or endpoint was run or changed.

## Seal and check results

- The completion-note hashes for the executor, worker, executor-config file,
  v3 amendment, patcher, rebuilt patch manifest, patched runner, and final
  smoke receipt all match the reviewed bytes.
- An independent CPU rebuild from the unchanged qualified source reproduced
  patch-manifest SHA-256 `2e5ab58a...` and runner SHA-256 `b73d2876...`
  byte-for-byte.  The B5 change to
  `int(sum(spec_decode_metadata.num_draft_tokens))` is trace-only and is
  correct for the pinned list representation.
- The focused boundary/Phase-1 suite passed: **67 passed**.  Task-scoped Ruff
  and `py_compile` passed.
- A fresh CPU fake run completed 12/12 with the expected synthetic
  `B1_SUPPORTED` result, two discordances, `SEALED` fake preflight, and a
  verified manifest.
- Direct torch-file round trips of float32 (including a noncontiguous view),
  bfloat16, int64, and bool tensors preserved dtype, shape, stride, and
  contiguous content bytes.

These checks validate the harness control flow and the chosen RPC mechanism;
they are not evidence that the real materializer, B lane, or terminal A path
works.

## Requirement disposition

| Item | Disposition | Evidence |
|---|---|---|
| (a) Atomic patched-tree overlay only | **OPEN — BLOCKER** | `_copy_qualified_overlay` first copies the image's entire installed `vllm` package and then overlays the patched tree with `dirs_exist_ok=True`.  It neither rejects image files absent from the atomic tree nor verifies the resulting file set/hashes; checking only the selected top-level `__init__.py` origin does not prove submodule provenance (`step5_phase1_gpu_worker.py:151-170`).  The executor hashes the mutable source tree only after the materializer has already copied and used it, and later epoch workers copy it again without rechecking the sealed digest (`step5_phase1_real_executor.py:319-340,380-398`).  Thus image-package leakage and post-seal/tree-use TOCTOU remain possible. |
| (b) Exact tensor transport | **CLOSED for RPC transport** | Requests and replies use `torch.save`/`torch.load(map_location="cpu")`; there is no RPC serialization through JSON and no dtype cast (`step5_phase1_real_executor.py:206-242`; `step5_phase1_gpu_worker.py:908-950`).  The CPU byte probe above confirms representative dtypes/shapes.  `_pack_group` deliberately reinterprets materializer leaves as flat `uint8`, but that is a separate evidence-schema choice, not an RPC cast (`step5_phase1_gpu_worker.py:385-392`). |
| (c) Fused rows and complete per-leaf exactness capture | **OPEN — BLOCKER** | Frame-22/23 fused rows are intercepted before Nano dispatch and preflight KV/Mamba/shadow leaf metadata contains dtype/shape/hash (`step5_phase1_gpu_worker.py:507-641`).  The per-case contract is not auditable: `_snapshot` fills emitted tokens, conjunction, accepted count, selected row, source-row identity, correction flags, and assumptions from `CaseSpec`/`OraclePlan`, not observed results (`:766-808`).  The source-row identity is the constant `packed-row-N`, not a row hash.  Missing real rollback shadow is silently replaced by a copy of committed Mamba state (`:297-305`).  Most decisively, `_result_summary` retains no snapshot or per-leaf seals at all, so the ledger/artifact manifest cannot substantiate any KV/Mamba/shadow/custom-output/pending-row comparison after the process exits (`step5_phase1_campaign.py:566-575,728-739`). |
| (d) True S transactions and sealed pair-only B selector | **OPEN — BLOCKER** | The core S accept/reject transaction uses one-position `_original_step` calls and a separate drain, and the patch predicate itself is the sealed request/PAD/one-request/two-position selector.  However, the worker reports `pair-eager` solely because the requested lane/case is B; it never reads B3's actual `graph_mode` or proves the selector fired (`step5_phase1_gpu_worker.py:873-885`).  Control cases and abort are not their named operations: all execute the same one-token `_original_step(row0)`, while the snapshot claims the plan's control/zero-position semantics (`:858-860`, `:766-808`).  Therefore S/B semantic equivalence can be manufactured even when the real transaction or dispatch differs. |
| (e) Terminal-A timeout destruction | **CLOSED** | A causal timeout returns one terminal-unknown result, the lane loop stops, and `finally` destroys the epoch.  There is no execute retry, resume, replay, or post-A lane (`step5_phase1_real_executor.py:435-475`; `step5_phase1_campaign.py:673-741`).  Graceful shutdown may wait before killing the stuck process, but it cannot resubmit the transaction. |
| (f) Complete identity and honest smoke/preflight | **OPEN — BLOCKER** | Executor, worker, config-file, and full patched-tree hashes are present across the compatibility identity, but they are not closed as one used-runtime identity.  The executor's `configuration_sha256` hashes a canonical mapping plus source/tree hashes, not the bytes of the config file selected by `NEMOTRON_PHASE1_EXECUTOR_CONFIG`; the separately frozen repository config is never required to equal that selected file.  The patched tree is not reverified in each child, and image/model hashes are accepted as configured strings rather than derived from the used artifacts (`step5_phase1_real_executor.py:105-178,380-398,478-486`; `step5_phase1_campaign.py:437-525`).  The v9 smoke is correctly `campaign_admissible: false` and used repeated real-model PAD embeddings, not PCM-derived fused rows.  It proves only one structural S lane. |
| (g) Four smoke-discovered fixes | **PARTIALLY CLOSED; no independent contract weakening from the fixes themselves** | Capturing/replaying the one-token prefill baseline is correct; built-in `sum()` fixes the pinned B5 list type without changing execution; retaining request block ownership fixes idle input-batch cleanup for the still-live single request; and torch-file RPC removes lossy conversion.  The tests cover fake-backend parity and idle-map retention, but not an actual subprocess tensor RPC.  The retained map has no stale/reallocation guard, and the surrounding snapshot fallback/self-reporting defects above still weaken the contract even though the four localized repairs do not. |
| (h) B5 reseal | **CLOSED** | The unchanged 966-file qualified digest is recovered, patcher/manifest/runner hashes match v4, and an independent rebuild produced byte-identical manifest and runner. |

## Critical PCM-materialization assessment

The v9 smoke provides no evidence for campaign preflight.  Before
`create_epoch(0)`, the real preflight must:

1. verify the frozen event/PCM sources and the selected image, model, config,
   worker, executor, manifest, and patched-tree identity;
2. build the pinned full perception/fusion/Nano pipeline and replay the retained
   typed-carrier PCM with the frozen 304-sample discard and prompt/tool context;
3. observe the real one-token prefill, the frame schedule through completed
   frame 22, and exactly one pre-dispatch fused row for frames 22 and 23;
4. derive—not assert—generated/session positions, pending PAD/PAD identity,
   idle function state, absent control barrier, request-owned KV, Mamba
   conv/SSM, true rollback shadow/index/pointer, virtual-engine identity, and
   construction identity;
5. seal every tensor's dtype, shape, byte count, and content hash, bind the
   retained prefix file to those seals, restore a fresh lane to the same
   initial exact state, and atomically write the matching materialization seal
   before epoch 0.

The materializer as written cannot establish those facts.  It does capture the
PCM-derived input rows and device tensors, but it hard-codes
`nano_session_positions=22`, pending token IDs, `function_state="idle"`, and
`control_barrier_pending=False`; it validates only generated-token count and
the presence of some pending tensor (`step5_phase1_gpu_worker.py:567-575,
610-639`).  It does not consume/attest the frozen typed-carrier text or the
recorded tool-enabled prompt context.  Restore trusts an unhashed prefix `.pt`
file, ignores the transported `MaterializedFixture`, checks counts/pending
presence only, and returns the claimed fixture hash without comparing the
restored initial state to the sealed frame-22 state (`:643-728`; payload ignored
at `:927-933`).

Consequently it is not correct to predict that this implementation will merely
stop `INCOMPLETE` again.  If full-pipeline construction/capture exceeds the
120-second materialize RPC or any observed shape/count check fails, the harness
will correctly emit zero-record `INCOMPLETE`.  But if those limited checks
pass, the current hard-coded metadata can produce `SEALED` preflight while the
required logical state and restored-state identity remain unproven.  That
false-admission possibility is the more serious blocker.

## Required deltas before resubmission

1. Build the child import tree from the atomic patched tree as the authority;
   explicitly enumerate any required image-only binary/data grafts, reject
   extra Python/module collisions, verify every resulting path, and recheck the
   sealed tree/manifest/runner hashes inside every materializer/epoch child
   before importing vLLM.
2. Bind the exact selected config file bytes and the actually used image/model
   artifacts.  Remove the tree/config TOCTOU between materialization, identity
   construction, and every fresh epoch.
3. Make materialization derive all frame-22 logical leaves from live pipeline
   state, require a real rollback source, attest the typed-carrier/prompt/tool
   context, hash the retained prefix payload, and verify every lane's restored
   initial snapshot against the materialization seal.  Apply the construction
   timeout to lazy full-pipeline construction rather than the 120-second
   ordinary RPC limit.
4. Populate `StateSnapshot` from observed outputs/state only.  Execute the real
   agent/function control transitions and a real abort; do not translate them
   into ordinary token steps or copy expected plan values into observed fields.
5. Parse and validate the latest transaction's B2/B3 details.  Report
   `pair-eager` only when the actual sealed selector descriptor and B3
   `CUDAGraphMode.NONE` are observed; require normal B3 mode for S/A/control/
   drain/abort work.
6. Persist canonical per-leaf dtype/shape/byte/hash seals for every exactness
   contract leaf (including custom outputs, pending rows, correction state,
   KV/Mamba/shadow, and row identity) in each replicate's retained evidence.
   Add negative tests that demonstrate a wrong GPU result, wrong B3 mode,
   tampered prefix file, missing true shadow, or hard-coded logical field cannot
   pass.
7. After those CPU-reviewable deltas, run a separately authorized one-engine
   **real PCM preflight plus S/B/terminal-destruction structural check** marked
   non-campaign.  Only a valid materialization seal and observed dispatch/state
   evidence can authorize scheduling the 12 fresh scientific replicates.

Until these deltas are closed, the correct disposition is **FAIL**, not
permission to schedule and rely on a possible zero-record `INCOMPLETE` stop.
