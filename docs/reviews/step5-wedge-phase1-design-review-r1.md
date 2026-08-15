# Step 5 wedge Phase 1 design and harness review — round 1

**Review date:** 2026-08-14 UTC  
**Verdict:** **FAIL — Phase 1 may not be scheduled for GPU execution**

The preregistration seal is authentic: SHA-256 is
`d076ccbdd46b2c174c6d6656b2ce79b257d33104872c643e70a0e2b28d887789`.
The retained patch manifest is byte-identical in its two published copies and
hashes to `1c6781819becd9073690a34edd9ddf23a5f5f0b4e27eb67e8f1054ed41a67fda`.
An independent read-only rehash verified all 966 indexed escrow files and
recovered tree SHA-256
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.
The frozen harness-file hashes in the completion note also match the files
reviewed.

Those seals do not make the GPU acceptance path executable as reviewed. The
harness can accept a complete S/B/A replicate with every state snapshot absent,
does not bind a pending drain to each rejected row, and can enter replicate 0
with a null materialization hash. The B patch also selects any one speculative
token rather than the preregistered PAD token, while the campaign verifies eager
dispatch only for the final B acceptance case. These are acceptance-path
defects, not documentation omissions.

This was a static and CPU-only adversarial review. No campaign, GPU, container,
service, endpoint, or model was run. The only repository write is this review.
The shared worktree was already dirty and its current `HEAD` is `d129eda…`; the
harness correctly seals the earlier `15bcb8e5fb16dfa1fa987def60e7c130c00c82bc`
revision and therefore refuses the current revision.

## Requirement disposition

| Item | Disposition | Evidence |
|---|---|---|
| (a) S, B, (C), A order and terminal A | **CLOSED** | `LANE_ORDER = ("S", "B", "A")`; the loop stops and destroys the epoch on any earlier error or unknown completion; A is last, and there is no retry, cancel/replay, or post-A arm (`step5_phase1_campaign.py:38-41,365-424`). C is explicitly sealed as omitted because it is construction-changing. |
| (b) Logical-position transaction oracle | **OPEN — BLOCKER** | Acceptance and rejection plans have the reviewed position/count/row values (`step5_phase1_oracle.py:35-54`), but selected-row identity is not present in `StateSnapshot`, and `pending_drain_is_separate_transition` is never consumed by campaign logic. The lane is restored once and runs three rejections followed by controls/abort and only one generic drain (`step5_phase1_campaign.py:58-81,294-362`). No drain is linked to an originating corrected row. |
| (c) Exactness contract and fail-closed tensors | **OPEN — BLOCKER** | The dataclass names the contract fields and byte-compares tensors, arrays, and nested values (`step5_phase1_oracle.py:57-153`). However, fields are `Any`, required tensor/output presence is never validated, and all semantic/comparison work is guarded by `snapshot is not None` (`step5_phase1_campaign.py:328-355`). Missing snapshots therefore pass. Missing KV/Mamba/shadow tensors or required custom-output keys also pass when both lanes omit them. |
| (d) Pair-only B eager patch and atomic 966 verification | **OPEN — BLOCKER** | Atomic staging, all-966 verification, compile, manifest, and `os.replace` are retained (`patch_step5_wedge_boundary_trace.py:403-479`) and independently verified. But the selector checks only that the speculative-token list has length one; it never checks that its value is the sealed PAD ID (`:344-361`). The harness requires `pair-eager` only on B's final causal case (`step5_phase1_campaign.py:395-404`), so B rejection pairs can report normal dispatch and still pass. The supplied patch-manifest hash is also accepted without format or content verification (`:202-241`). |
| (e) Two-discordant rule and record-based dispositions | **CLOSED** | `campaign_verdict` computes the 12-record completeness gate, exactness/B failures, post-forward rejection, the at-least-two rule, and fewer-than-two `INCOMPLETE` result (`step5_phase1_campaign.py:427-456`). The fixed-order qualifying predicate is also computed in `run_replicate`. |
| (f) GPU preflight before replicate 0 | **OPEN — BLOCKER** | There is no materialize/validate/seal preflight transition in `run_campaign`; it loads the textual fixture and immediately starts `create_epoch`/replicate execution (`step5_phase1_campaign.py:480-504`). `compatibility_identity` checks only the presence of executor-identity keys, not non-null 64-hex values or frame-22/two-row contents (`:202-221`). It has no zero-replicate `INCOMPLETE` disposition. |
| (g) Identity and resume discipline | **CLOSED for the stated primary path** | The revision constant is exactly `15bcb8e…`; lane/case/replicate shapes, patch namespace/manifest hash, executor identity, and frozen files enter the canonical identity (`step5_phase1_campaign.py:32-45,187-242`). Resume verifies a contiguous zero-based ledger, compatibility digest, record digest, and maximum count, then starts only the next fresh epoch (`:261-279,480-503`). |
| (h) Construction-stratified fallback preregistration | **CLOSED at preregistration level** | The document requires a non-destructive construction/capture/layout preclassifier, a separate seal, 12 matched pairs per observed stratum, randomized assignment, B-before-terminal-A, within-stratum two-discordance, and fail-closed mismatches (`phase1-preregistration.md:129-148`). It cites the FHW8 G1 5/5 candidate divergent-basin/checkpoint-layout evidence in `docs/generation-frame-latency-plan.md`. The primary runner also refuses an asserted non-lane-local executor rather than silently using unstratified epochs (`step5_phase1_campaign.py:365-372`). |

## Blocking findings

### 1. Missing exactness evidence can produce a qualifying causal replicate

`StateSnapshot` has all requested field names, but that is not a required
schema. `ExecutionResult.snapshot` is optional, and `_run_lane` simply skips
restore validation, semantic validation, and cross-lane comparison when it is
`None`. An adversarial CPU control-flow probe returned `snapshot=None` for every
S, B, and A case. The resulting replicate had:

```text
exactness_errors=[]
b_exact_complete=True
qualifying_discordance=True
```

Likewise, `compare_snapshots` reports exact when both snapshots use `None` for
KV, both Mamba states, rollback shadow, shadow pointer, virtual-engine identity,
correction evidence, and position-one/uncommitted state. An empty
`selected_custom_outputs` mapping matches another empty mapping; the code does
not know which retained output keys are mandatory. This directly violates the
reviewed fail-closed exactness contract and can falsely contribute to the
two-discordant threshold.

The row contract is also only advisory. `OraclePlan.selected_packed_row` is
echoed through `observed_plan`, but the snapshot contains neither the selected
row number nor source-row identity. A real adapter can return row 1 for a
rejection, echo the supplied plan, and pass if S returns the same values.

### 2. Rejection drain is not an executable transaction

The plan correctly says a rejection accepts position one, retains corrected row
two, and marks its drain separate. Nothing schedules or verifies that separate
transition for that rejection. There is one lane restoration, followed by the
entire case list. After `text-only-rejection` leaves a row pending, the next
case is `function-only-rejection`, not the required drain of the corrected text
row. The same happens for function-only and conjunctive rejection. The later
`next-action-pending-drain` case has no originating rejection ID, pending-row
digest, correction assumptions, or before/after linkage. The fake executor is
stateless across cases and therefore masks the defect.

The harness must either restore an independent frozen lane for every matrix
case and execute that rejection's drain immediately as its next action, or
model a coherent lane sequence in which every rejected row is drained before
another transaction. In either form, the rejection and drain need distinct
records joined by origin and pending-row digest, with rejected accepted count
remaining one and drain count reported separately.

### 3. The materialization seal is narrative, not a gate

The preregistration correctly says the historical fused rows and device state
are unavailable and requires materialization before replicate 0. The executable
path does not implement that condition. A real executor identity containing all
six required keys is accepted even when
`fixture_materialization_sha256=None`; an arbitrary value such as
`patch_manifest_sha256="not-a-sha"` is also accepted. There is no verification
of two 4480-wide fused rows, dtype/shape/bytes, completed-frame-22 KV/Mamba and
rollback state, pending row/flags, construction metadata, or their sealed hash.
There is also no code path that writes `INCOMPLETE` with zero epochs constructed
when materialization fails.

This must be a harness-owned state transition before `create_epoch(0)`, not an
unchecked promise made by a future adapter's `sealed_identity()` method.

### 4. B dispatch is not exactly the preregistered intervention

The patch is narrow by request namespace, request count, and scheduled-position
count, and forcing `CUDAGraphMode.NONE` with a null graph descriptor matches the
dispatcher representation of eager execution. But the predicate calls any
single proposed speculative token the “exact PAD-pair descriptor.” It does not
compare the proposal value to the sealed PAD token. This is broader than the
preregistered intervention.

Separately, the campaign's fake makes rejection B cases report `normal` because
their logical accepted-position plan has length one, even though the packed
worker dispatch necessarily occurs before rejection and schedules two
positions. No harness assertion rejects that mismatch. B3 must prove eager mode
for every B packed two-position dispatch, including all three rejection cases,
and normal mode for S, A, one-position work, and every non-B request. The
manifest supplied to the campaign must be opened and verified to contain the
qualified tree digest, count 966, target hash, Phase 1 marker/namespace, and its
claimed SHA rather than being accepted as an opaque string.

## Required round-2 deltas

1. Make every nonterminal execution require a complete, typed snapshot. Define
   mandatory tensor leaves and retained custom-output keys, require selected-row
   identity, and fail closed on any absent snapshot, field, tensor, output, or
   rejection-boundary state. Add negative tests where all lanes omit the same
   evidence.
2. Bind each rejection to its own immediately following pending-row drain (or
   restore an independent fixture per rejection), and verify before/after row
   digest, correction flags/assumptions, accepted counts, and committed-state
   movement as two distinct transitions.
3. Add a pre-replicate GPU preflight API and state machine that materializes,
   validates, and seals both fused rows and the complete frame-22 state before
   any epoch construction. Failure must produce a retained `INCOMPLETE` summary
   with zero replicate records and no synthetic substitution.
4. Compare the B proposal token value to the sealed PAD ID, verify actual B3
   dispatch mode for every applicable/non-applicable case, and bind the supplied
   patch manifest to its on-disk content and qualified 966-file identity.
5. Retain the fixed S/B/A terminal ordering, code-computed two-discordant rule,
   `15bcb8e…` revision seal, contiguous resume ledger, and construction-stratified
   fallback wording unchanged while adding adversarial regressions for the four
   blockers above.

Until those deltas are implemented and CPU-verified, the sealed document may be
retained as a preregistration draft, but Phase 1 is **not schedulable on GPU**.
