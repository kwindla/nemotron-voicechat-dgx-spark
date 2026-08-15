# Step 5 wedge Phase 1 design and harness review — round 2

**Review date:** 2026-08-14 UTC  
**Verdict:** **PASS — Phase 1 GPU execution may be scheduled immediately**

The four round-1 blockers are closed in the reviewed v2 sources. The four
previously closed deltas remain intact. This authorizes the preregistered Phase
1 GPU execution; it is not a Phase 1 result and does not weaken any runtime
identity, preflight, exactness, terminality, or fallback gate.

The v2 preregistration seal is authentic:
`f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5`.
The reviewed harness, oracle, patcher, tests, and fixture match every hash in
the round-2 completion note. The retained Phase 1 patch manifest hashes to
`818906e4c83ecbd8a62a82193a3df86b563f8fc402d07fbf3453deec1f946786`;
its source runner is `2ce42b6f...`, deterministic patched runner is
`1cc698d1...`, qualified tree is `d66baad7...`, and indexed file count is 966.
An independent read-only rehash of all 966 indexed source files recovered the
same qualified-tree digest, and the patched runner compiled from bytes.

This was a CPU-only/static confirmation review. No Phase 1 campaign, GPU,
engine, container, service, endpoint, or model was run or changed. The only
repository write is this review file. The current worktree `HEAD` is
`832da7f...`, not the sealed application revision; this is not a regression:
the harness still refuses execution unless `HEAD` is exactly
`15bcb8e5fb16dfa1fa987def60e7c130c00c82bc`.

## Requirement disposition

| Item | Disposition | Round-2 evidence |
|---|---|---|
| (a) Fixed S/B/(C)/A order and terminal A | **CLOSED — no regression** | `LANE_ORDER = ("S", "B", "A")`; each epoch executes that fixed order, breaks on any earlier error or unknown completion, and destroys in `finally`. The causal case is last in the case matrix and A remains the terminal lane; C remains explicitly omitted because it changes construction (`step5_phase1_campaign.py:48,72-105,652-709`). |
| (b) Logical-position transaction oracle | **CLOSED** | The oracle fixes acceptance to positions `(0,1)`/row 1/count 2 and rejection to position `(0,)`/row 0/count 1. `StateSnapshot` now carries both selected row and canonical source-row identity, and semantic validation checks them against the plan. Each of the three rejection cases is immediately followed by a separate drain. Campaign state links each pair by origin rejection ID, pending-row digest, rejection-after/drain-before committed-state digest, and required committed-state movement; rejection and drain counts are independently validated as one (`step5_phase1_oracle.py:35-95,236-301,318-352`; `step5_phase1_campaign.py:72-115,546-649`). |
| (c) Exactness and fail-closed evidence | **CLOSED** | Snapshot fields use recursive exact-value/tensor-tree types. Every completed nonterminal result requires a snapshot. Presence validation requires nonempty tensor leaves for KV, Mamba conv, Mamba SSM, and rollback shadow; non-abort cases require `function_tokens`; identities and state hashes are required. Semantic and restore checks fail closed before cross-lane byte-exact comparison. The retained negative tests cover missing snapshots, missing tensors in both lanes, missing custom output, and missing row identity (`step5_phase1_oracle.py:16-22,67-95,110-233,236-315`; `step5_phase1_campaign.py:575-648`; tests at `test_step5_phase1_campaign.py:196-227` and `test_step5_phase1_oracle.py:157-182`). |
| (d) Pair-only B eager patch and manifest binding | **CLOSED** | The patch predicate now requires the sole speculative proposal value to equal sealed PAD ID 12 in addition to B namespace, one request, two scheduled positions, and one proposal, then records B3 after the override. `requires_pair_eager` covers every B transaction—three rejection pairs plus final acceptance—while drains, controls, abort, S, and A require normal dispatch. Both manifest CLI arguments are mandatory, and verification binds the file bytes and JSON contents to the qualified tree/count, boundary marker, Phase 1 marker/namespace/PAD/target, source/output runner hashes, patcher hash, and trace hash (`patch_step5_wedge_boundary_trace.py:325-376,409-482`; `step5_phase1_campaign.py:118-120,260-307,570-574,1015-1032`). |
| (e) Record-computed two-discordance rule | **CLOSED — no regression** | `run_replicate` computes exact B completion and first-missing-B4 A loss from that replicate's results. `campaign_verdict` requires all 12 records, zero B/exactness failures, no post-forward loss, and at least two qualifying discordances; fewer than two remains `INCOMPLETE` (`step5_phase1_campaign.py:682-740`). |
| (f) Materialize/validate/seal preflight | **CLOSED** | The executor must return exactly two fused tensors plus completed-frame-22 state before any epoch. The harness copies tensor bytes to CPU, records dtype/shape/byte count/content hash, validates 4480 width and the exact frame/token/session/pending PAD/function/barrier state, requires nonempty KV/Mamba/shadow tensors and construction/shadow/virtual-engine identities, computes the aggregate materialization hash, matches the sealed executor identity, and atomically writes the seal before `create_epoch(0)`. Initial preflight failure writes an `INCOMPLETE` zero-record summary and never reaches epoch creation (`step5_phase1_campaign.py:123-177,200-214,310-391,442-494,765-829`). |
| (g) Identity and resume discipline | **CLOSED — no regression** | The exact `15bcb8e...` application revision, case/lane/replicate shapes, patch manifest, executor hashes, and frozen file hashes remain in compatibility identity. Executor hashes must be non-null lowercase 64-hex. Resume still requires a contiguous zero-based, hash-valid ledger no longer than 12 and resumes only at the next fresh epoch (`step5_phase1_campaign.py:426-494,504-531,765-829`). |
| (h) Construction-stratified fallback | **CLOSED — no regression** | The v2 preregistration preserves non-destructive preclassification, a separate seal, 12 matched pairs per observed stratum, randomized assignment, B-before-terminal-A, within-stratum two-discordance, and fail-closed stratum mismatch. The primary runner still refuses an executor that reports non-lane-local dispatch (`phase1-preregistration.md:131-151`; `step5_phase1_campaign.py:652-660`). |

## Targeted confirmation

- All seven published v2 source/fixture hashes and the authoritative round-1
  review hash matched the completion note.
- Task-scoped Ruff passed with cache disabled.
- The 30 oracle tests passed CPU-only; the combined Phase 1 suite collected the
  claimed 46 tests. Campaign tests were not executed in this review.
- The retained dry-run artifacts were inspected, not rerun: 12 contiguous
  records, qualifying synthetic replicates 0 and 1, no B dispatch mismatch, no
  non-B eager dispatch, ledger SHA-256 `dc9f4eb...`, SEALED preflight, and the
  published identity/materialization/summary/manifest hashes all matched.
- `git diff --check` was clean before this review write.

No round-2 delta remains. The GPU-window operator may schedule Phase 1 now,
using the sealed application revision, exact patch manifest, real executor
identity, and mandatory materialization preflight. Any failure of those gates
must remain `INCOMPLETE`/refused as preregistered; it is not permission to
substitute synthetic inputs or an unstratified construction fallback.
