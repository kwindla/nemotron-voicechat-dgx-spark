# Step 5 wedge Phase 1 preregistration — v2

**Status:** round-2 sealed design and executable harness; the GPU campaign is
not run in this increment. This v2 document supersedes the reviewed v1 seal
`d076ccbdd46b2c174c6d6656b2ce79b257d33104872c643e70a0e2b28d887789`.
The authoritative correction source is
`docs/reviews/step5-wedge-phase1-design-review-r1.md`. This document governs the
later coordinated 12-engine run and concretizes
`docs/step5-pairing-wedge-closure-design.md` §6.3.

## Immutable source and frozen causal prefix

The increment-1 Phase 0 conclusion is commit `b98fe9f`.  The Phase 1 harness is
based on application revision `15bcb8e5fb16dfa1fa987def60e7c130c00c82bc`,
which is the next repository revision and contains the independently completed
FHW8 G3 evidence.  A run is refused if `HEAD`, the batch shape, lane shape, case
matrix, patch manifest, executor identity, or any frozen file hash differs from
`compatibility-identity-preregistered.json` in its output directory.

The qualified vLLM escrow tree remains the only source.  Its 966 regular files
must match `tools/provenance/qualified-deltas/index.json` (index SHA-256
`8854e6eb87c7adec5ed0cdba9a9b9d5f16cecf62182c8e928be91058aabc1117`)
and tree SHA-256
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`
before the first staging write.  The source root is never modified.

The retained failing source is session
`2d08e743-39bb-4119-866d-8ab5c215581c`: events SHA-256
`75716a7d18859341c3bed237a5df5f95816f4e63e50453ce19fdba357d9c67d6`
and input PCM SHA-256
`d0ff4a30523aadfef6324d6291641f369e5ae747b43e6e8aeacf3dd88eb4470e`.
Those artifacts retain frames and audio but not either 4480-wide fused row,
committed KV, Mamba conv/SSM state, rollback shadows, or graph-construction
identity.  Exact historical tensor replay is therefore impossible.  This is
the inability-to-replay disclosure required by §6.3.

The frozen replacement is
`reports/step5-wedge/phase1-frozen-fixture.json` (pre-edit SHA-256 recorded by
the sealed compatibility identity). During the coordinated GPU increment the
harness calls the executor's `materialize_fixture` preflight before
`create_epoch(0)`. The pinned perception/fusion stack replays the retained PCM
and typed carrier, captures the frame-22 pending row and frame-23 row before
Nano dispatch, and returns exactly two 4480-wide tensor seals plus the complete
frame-22 state. The harness validates nonempty dtype/shape/byte count and
lowercase 64-hex content hashes, then validates and atomically seals
completed-frame-22 KV, Mamba conv/SSM, rollback shadow, pointer/index,
virtual-engine identity, construction identity, pending PAD/PAD row, idle
function state, and absent control barrier before any epoch construction.
Every lane is restored independently to completed frame 22: Nano generated
tokens 23, session positions 22, one PAD/PAD row pending, function state idle,
and no control barrier pending. Failure to materialize, validate, hash-match the
executor identity, or seal that state retains an `INCOMPLETE` summary with zero
replicate records and no `create_epoch` call; it is not replaced by synthetic
GPU inputs after the run starts.

## Patch and lanes

The increment-1 atomic patch-on-copy implementation is reused with the Phase 1
flag.  It verifies all 966 source hashes and every anchor before writing,
copies to a temporary sibling, applies B0--B9 tracing plus the intervention,
reverifies staging, compiles every Python file, writes a manifest, and publishes
with `os.replace`.  The intervention target is hash-qualified
`v1/worker/gpu_model_runner.py`.  Only a request in namespace
`step5-phase1/B/` with one request, two scheduled positions, and exactly one
PAD speculative token has its final dispatch changed to
`CUDAGraphMode.NONE`; one-position work and all other requests retain normal
dispatch. The selector compares the sole speculative token value to sealed PAD
ID 12. B3 must record this final eager mode for every B packed two-position
transaction, including all three rejection cases and the final acceptance; it
must record normal mode for S, A, controls, drains, abort, one-position work,
and non-B requests. The campaign opens the supplied patch manifest and verifies
its claimed file hash, qualified tree digest, 966-file count, runner source and
output hashes, Phase 1 marker, B namespace, PAD ID, and target before identity
construction. The ordinary graph-enabled one-position path, speculative PAD
check, and commit/rollback paths are not changed.

There are exactly 12 independent fresh engine replicates.  Within every primary
epoch, independent requests are restored from the same prefix and run in this
fixed order:

1. **S** — pair-disabled sequential logical-position transaction.
2. **B** — packed candidate-1 semantics with the pair-only eager intervention.
3. **C omitted** — the global-eager control changes engine construction and is
   not a lane-local selector in the pinned runtime.
4. **A** — released candidate-1 packed path, normal graph-enabled mode, with
   the dedicated pair FULL graph disabled; A is terminal and always last.

Candidate-2 dedicated FULL graph lane D is excluded from this primary study.

## Logical-position transaction oracle and matrix

For PAD/PAD conjunctive acceptance, S executes positions one and two; packed
row 1 (zero based) is compared at the committed position-two boundary.  For a
text-only, function-only, or conjunctive rejection, S executes only position
one; packed row 0 is compared immediately at that committed boundary.  The
actual position-one text and function results correct the second fused row,
which remains pending in both lanes. Its next scheduler drain/roll is an
immediately following, separately counted transition and is never included in
the rejected transaction's accepted count. The rejection and drain are
distinct records joined by originating rejection ID, corrected-row SHA-256,
the rejection's after-state/drain's before-state SHA-256, and committed-state
movement across the drain.

Every lane executes, in order, text-only rejection and its drain, function-only
rejection and its drain, conjunctive rejection and its drain, agent BOS, agent
EOS, function SOTC, function EOTR, post-function BOS, abort, and finally the
frozen PAD-accept causal case. A's causal case is last within terminal A.

At the applicable boundary equality is bit-exact—not tolerance based—for:

- emitted text token, emitted function token, and their conjunction;
- accepted-token count;
- every retained custom output in packed row 1 for acceptance or row 0 for
  rejection;
- committed KV state and committed Mamba conv and SSM state;
- rollback-shadow contents, shadow index/pointer, and virtual-engine identity;
- host pending-row contents, correction flags, and correction assumptions.

Every completed nonterminal execution must return a typed `StateSnapshot`.
The snapshot records the selected packed-row number and nonempty source-row
identity. Committed KV, both Mamba states, and rollback shadow must each contain
nonempty tensor leaves, and every non-abort case must contain the retained
`function_tokens` custom output. Missing snapshots, tensors, identities,
digests, or required output keys fail closed even when S and the candidate lane
omit the same evidence.

On rejection, uncommitted second-position KV/recurrent state must be absent or
bit-exactly restored to the position-one boundary.  Any difference invalidates
the replicate before causal, timing, or reliability interpretation.

## Boundary, lifecycle, and decision rules

The increment-1 bounded B0--B9 sidecar is enabled for every future lane.
Sequence records are joined by request ID and engine epoch.  Completed S and B
require all B0--B9 boundaries; A causal loss is preregistered as first missing
**B4** (B0--B3 present, forward return absent).  Trace overflow, duplicate or
out-of-order boundaries, request/epoch routing mismatch, or trace-on/off
noninterference failure invalidates the run.  A wall timeout only triggers
diagnosis and is never recovery.

An unknown completion, timeout, crash, or exactness failure makes the epoch
terminal.  In particular A is never resumed, cancelled-and-retried, or replayed
on that epoch.  The harness records A's last boundary, destroys the whole
epoch, and proceeds only to the next preregistered fresh engine.  A replicate
qualifies only when exact S and exact/completing B precede A at the same frozen
prefix and A loses completion with first missing B4.  There is no post-A flip
back.

B1 is supported only if at least two of 12 fresh epochs show that exact
boundary-localized B-then-A discordance, every B completes, and no exactness
check fails.  All 12 outcomes are reported.  Fewer than two reproductions is
**INCOMPLETE**, never evidence that stress fixed the wedge.  If A records B4 or
later before loss, the forward-non-returning form of B1 is rejected and the
first missing post-forward/output transition becomes the only next
intervention; the complete fixed-order design must then be preregistered and
rerun.

## Construction-stratified fail-closed fallback

The primary patch is lane-local by inspection and patch tests.  If the real
adapter demonstrates that per-request dispatch changes anything else, the
primary run stops before replicate 0.  It must first run a non-destructive
preclassifier that records graph capture keys, selected compiled artifact
hashes, allocation/layout signatures, and dispatcher construction identity
without restoring or exercising the packed failing prefix.  This requirement
is evidence-driven: FHW8 G1 candidate constructions selected the divergent
call-39 basin 5/5 while baseline constructions did not; checkpoint layout
therefore biases basin selection (`docs/generation-frame-latency-plan.md`).

After a separate seal, run 12 fresh matched A/B epoch pairs **for each observed
construction stratum**.  Preclassify both untouched constructions, randomize
their arm assignment from the sealed replicate index, require identical
stratum identities, then execute the assigned B epoch before terminal A.
Apply the same S oracle, exactness contract, first-missing-B4 criterion, and
two-discordant-replicate rule inside each stratum.  Mismatched strata are not
silently paired.  Unstratified/counterbalanced independent epochs and C/D
controls are non-causal and cannot advance Phase 2.

## Evidence and resume

Each output directory contains the v2 sealed identity, materialized-fixture
seal, append-only replicate ledger, per-replicate lane/case boundary summaries
and exactness differences, terminal destruction reason, final summary, and a
recursive hash manifest.
Resume accepts only a contiguous 0-based prefix whose identity and canonical
record hashes verify.  It rejects gaps, overlaps, revision/shape/hash drift,
or more than 12 records.  A completed terminal-A record is never resumed; the
next action is construction of the next fresh epoch.

This preregistration authorizes no GPU, container, service, or endpoint action
in increment 2.
