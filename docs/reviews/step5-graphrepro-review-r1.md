# Step 5 graph-reproducibility measurement-validity review — round 1

## Verdict

**FAIL as a measurement-validity-complete report, with the central result
preserved.** The retained measurements support the preregistered
`REAL STATE DEFECT` disposition and keep the released W8 graph path blocked.
The failure is not a count or artifact-integrity failure. It is caused by four
material calibration/tooling defects:

1. sampling seed was only varied across different fresh engines, not isolated
   as a selector, yet the report calls it rejected;
2. request-history evidence comes only from exact-basin engines, so it rejects
   a history-triggered transition only in that observed basin, not history/cache
   initialization generally;
3. the diagnostic overlay changed the workload, so its `+0.125` versus `0.0`
   margins cannot be used to fail the preregistered production-margin bound;
4. the aggregate tool hard-codes the three selector conclusions to `false` and
   does not implement the complete preregistered validity or bounded-gate rule.

These defects require report/tooling deltas, but they do **not** change the
classification, reopen the bounded-divergence gate, or unlock the ranking-table
kernel-class row.

## Scope and review method

This was an offline review of
[the Step 5 report](../../reports/step5-graph-reproducibility-20260813.md), its
retained data directory, the completion note at
`/tmp/claude-1000/-home-khkramer-src-nemotron-voicechat-dgx-spark/9dab1982-03af-4f77-9d25-ddf7139af66f/scratchpad/step5-graphrepro-done.md`, the Step 4e ranking table,
and the Step 5 tool and tests. No service, container, GPU, or model probe was
run.

I independently:

- verified every entry in the 83-entry SHA-256 manifest;
- recomputed the preregistration prefix seal and all five frozen Step 4 input
  hashes;
- recomputed every stated arm count directly from the authoritative run JSON,
  excluding the two explicitly superseded attempts;
- recomputed every stored token-vector SHA-256 from the retained token arrays;
- checked validity, cleanup, request counts, digest matches, and diagnostic
  finiteness in the retained records;
- inspected the aggregation and bounded-gate implementation and its tests; and
- reran only the offline task-scoped Ruff and focused test commands, with the
  pytest cache disabled.

## 1. Preregistration seal and execution

The seal is valid.

- Hashing the report bytes through and including
  `STEP5_RESULTS_BEGIN` gives
  `6c5b6c05dc3ca1677015574dce0818f2ca84d2786c4ef1859e0e651277fde071`,
  exactly the value in `preregistration.sha256`.
- `preregistration.json` hashes to
  `dac72808133b905c124e16673ab76f909e78e9d7d60627bcc3e4272f69421aa4`,
  also exactly sealed.
- The five preregistered Step 4 evidence hashes all match the current retained
  inputs.
- The preregistration files were sealed before the first retained replay log:
  the JSON/seal precede the first `g0-1` attempt, whose log begins at
  `2026-08-13 23:06:06Z`.

The fixed hypothesis order is supported by the ledger and retained log
chronology. The two deviations are disclosed and retained: the first `g0-1`
digest encoder was superseded and rerun, and the first history aggregate-count
check was superseded and rerun. Their inference records were not silently
substituted into the reported totals. Diagnostic overlays were run only after
the preregistered trigger occurred. No evidence indicates an image rebuild.

There are 84 files in the data directory including `artifacts.sha256`; that
manifest contains 83 entries and all 83 verify. Thus “83-file inventory” means
83 inventoried payload files, not 83 files including the manifest, and is
accurate with that convention.

## 2. Independent recomputation

### Production arms

| arm | engines | blocks | exact | call-39 `10592 -> 1044` | selected digests |
|---|---:|---:|---:|---:|---:|
| primary graph, seed 0 | 3 | 9 | 3 | 6 | 18/18 |
| primary eager, seed 0 | 3 | 9 | 9 | 0 | 18/18 |
| graph, seed 1 | 2 | 6 | 3 | 3 | 12/12 |
| graph history sequence | 2 | 14 | 14 | 0 | 28/28 |
| graph request-ID/sync sequence | 2 | 18 | 9 | 9 | 36/36 |
| full-prefix graph confirmation | 3 | 9 | 9 | 0 | 54/54 |
| full-prefix eager confirmation | 3 | 9 | 9 | 0 | 54/54 |
| **all production graph** | **12** | **56** | **38** | **18** | **148/148** |
| **all production eager** | **6** | **18** | **18** | **0** | **72/72** |

The requested totals therefore reproduce exactly: graph `38/56`, with all 18
divergences having the same single recorded mismatch at call 39; eager `18/18`;
primary seed-0 graph `3/9` versus `6/9`; and confirmation `9/9` in each mode.
All 220 authoritative production-interface digest comparisons match.

All authoritative production blocks are marked valid, have successful cleanup,
and record zero requests both before the block and after cleanup. Rehashing the
retained token arrays reproduced every stored vector digest. The diagnostic
records are finite and internally consistent as well.

### What “bistable construction” is allowed to mean

The evidence supports **two observed graph-engine-stable output basins at the
fresh graph-enabled/compiled construction boundary**:

- every tested engine was stable within itself;
- the primary graph engines split one exact to two divergent;
- the seed-1 graph engines split one exact to one divergent;
- the scheduling engines supplied one exact and one divergent engine, both
  stable across their within-engine interventions; and
- no eager production engine diverged.

Across authoritative production graph runs, the independent construction unit
is 12 engines, of which 8 were exact and 4 divergent. The `38/56` block rate is
not an incidence estimate because engines contributed unequal numbers of
correlated blocks. “Bistable construction” must not be expanded to mean a known
random construction probability, a proven vLLM child-kernel cause, or a proven
lifetime guarantee beyond the tested requests.

### Honest reconciliation of the confirmation cell

The three graph confirmation engines and all nine of their 687-call blocks are
genuinely exact. That cell does two things:

1. it shows that an exact graph basin can remain exact through call 686; and
2. it is negative evidence against any high or stationary divergence-incidence
   claim.

It does **not** erase the four independently observed divergent production graph
engines. It does, however, weaken the primary cell's apparent two-of-three
divergent prevalence: after confirmation and all other production arms, the
engine-level observation is 4/12 divergent, not 2/3. Because execution was
fixed-order rather than interleaved, the all-exact later confirmation cell also
leaves a temporal or other uncontrolled construction-epoch factor plausible.
The report's statement that the cell “does not erase” the defect is correct but
incomplete unless this negative-evidence/prevalence qualification is added.

The confirmation cell did not obtain a divergent full-prefix engine. Therefore
Step 5 does not establish the post-call-46 behavior of its divergent engines or
run the preregistered 30-block/five-engine bounded gate. The older Step 4d
full-prefix divergence remains separate retained evidence.

## 3. Selector discrimination

| selector | what was actually tested | licensed conclusion |
|---|---|---|
| sampling seed | two new seed-1 engines, one exact and one divergent, compared with three different seed-0 engines | No observed seed association; **not isolated and not rejected**. Fresh-engine construction is a confound and the cell is too small for a distribution claim. |
| prior request history | baseline followed by 47-call and 17-call sacrificial requests in two engines; both engines were exact throughout | History did not move either observed exact engine. This is a real within-engine intervention, but it does not show whether history can move or perturb a divergent engine. Ordinary cleanup evidence is reassuring; capture/fresh initialization remains unresolved. |
| request-ID reuse | three unique-ID, three reused-ID, then three synchronized unique-ID blocks in one exact and one divergent engine | Reuse did not change either basin. This is a meaningful within-engine, both-basin discriminator for the tested single-client path. |
| per-call synchronization | same two scheduling engines, with three synchronized blocks after the unsynchronized blocks | Synchronization did not change either basin. This is a meaningful within-engine, both-basin discriminator, subject to the fixed sequence. |

The report should replace “sampler/seed state rejected” with “not supported; not
isolated,” and narrow the history claim to the exact basin plus the general
cleanup observations. Reuse and synchronization are fairly described as tested
non-selectors. These calibration deltas do not affect the graph/eager
classification trigger.

## 4. Classification and logit localization

`REAL STATE DEFECT` is the correct **preregistered rule label**. The primary
matrix meets the H1 discriminator exactly as preregistered: multiple
engine-stable vectors occurred only in graph mode while all three fresh eager
engines were exact. Under the sealed classification rule, that graph/eager or
capture/construction discriminator selects the real-defect branch. The later
all-exact confirmation cell changes prevalence uncertainty, not the existence
of the replicated graph-only defect.

The narrowest supported faulty boundary is the released hybrid W8 Nano's fresh
vLLM graph-enabled/compiled decode construction. “State defect” here is a
classification label; it does **not** prove stale KV/SSM state or corrupted
buffers. The evidence does not distinguish capture-time state/buffer binding,
input-copy behavior, compilation/autotuning, or a graph-contained numerical
kernel/reduction. Naming one of those as root cause would overclaim.

The weight-identical diagnostic overlay licenses only this localization:

- its exact graph/eager observations recorded `logit(10592)-logit(1044) =
  +0.125` and selected 10592;
- its divergent graph observation recorded a BF16 tie, `0.0`, and selected
  1044; and
- therefore the overlay's corresponding token flip occurs at the greedy
  decision boundary and is compatible with construction-associated graph
  numerical/state variation.

It does not license the claim that production has those same margins, that a
specific cache is corrupt, or that ordinary BF16 numerical sensitivity alone is
benign or defective. Most importantly, the overlay changes graph workload. The
sealed rule says an overlay-only margin cannot satisfy the production margin
clause; symmetrically it cannot be used as the decisive proof that the
production `0.05` bound failed. The report's use of the exact overlay basin's
`0.125` to declare that bound exceeded must be removed or labeled supportive
only.

`BENIGN-NONDETERMINISM` is unavailable regardless: no within-engine intrinsic
variation or equivalent kernel demonstration was obtained, the production
margin remains unresolved, and the required 30 measured 687-call blocks over
five fresh graph engines were not run. Because `REAL STATE DEFECT` was selected,
the preregistered bounded-divergence gate is **off the table**, not a fallback
acceptance route. The exact-output gate remains in force.

## 5. Tooling and tests

The runner successfully retains full token arrays, vector hashes, mismatch
runs, selected-output digests, cleanup observations, request counts, and
diagnostic top logits. The retained validation outputs are manifest-covered and
report Ruff passing, 14 focused tests passing, 67/67 anchors, and the full
non-browser result of 1177 passed, 15 skipped, 3 deselected, and 29 subtests.
The offline rerun in this review reproduced the Ruff pass and `14 passed`.

However, the aggregate/tool claim is materially broader than the code:

- `build_investigation_summary()` assigns
  `sampler_seed_selects_result`, `request_history_selects_result`, and
  `request_id_or_synchronization_selects_result` literal `False` values. It
  never computes those decisions from the blocks.
- Aggregate classification does not require all contributing reports/blocks to
  pass the preregistered validity conditions or verify artifact/capture/control
  identity before using their vectors.
- A block's `valid` flag checks cleanup and call count, but not nonzero request
  counts, selected-digest mismatch, identity mismatch, or non-finite diagnostic
  values, even though the preregistration makes those engine-invalidating.
- `evaluate_bounded_block()` defaults its denominator to 687 even on 47-call
  localization blocks. It consequently labels one mismatch in 47 calls as
  satisfying the `1/687` rate. That field should be not-applicable outside a
  complete 687-call block.
- The implementation does not enforce the global 30-measured-block/five-engine
  requirement or the production-margin clause. Thus it implements component
  checks, not the complete bounded gate.
- The tests cover hashing compatibility, divergence runs, a few block-bound
  failures, cell separation, and the graph/eager classification. They do not
  catch the hard-coded selector decisions or the omitted validity/global-gate
  requirements.

The retained dataset itself passes those omitted validity checks under this
manual audit, so this tooling defect does not reverse the observed counts or
classification. It does prevent accepting the completion note's unqualified
claim that the tool covers aggregate hypothesis classification and the complete
bounded gate.

## 6. Consequences

### Step 4e ranking-table kernel-class row

The **Nano W8 kernel/fusion/launch-gap** row remains `NR`, `BLOCKED`, and without
ranking authority. Step 5 changes the prerequisite's description, not its
status:

- before Step 5, graph-path reproducibility was an unresolved prerequisite;
- after Step 5, it is a characterized graph-enabled fresh-construction defect
  with two observed engine-stable outputs.

The next correctness work is a vLLM graph/capture-boundary repair investigation
on the unchanged W8 artifact: bisect full versus piecewise graph/compiled modes,
input copying, capture warm-up and dummy requests, and hybrid Mamba/KV fixed
buffer identity/zeroing; instrument the existing production sampler without
adding a duplicate text head. The evidence supports this boundary, not yet a
specific upstream vLLM fix.

A repair must then pass the original exact full-prefix and six-output gate over
fresh graph engines and profiling contexts before decomposition, child-kernel
identification, timing, or ranking resumes. Eager mode is a correctness control
and possible workaround only; adopting it creates a different latency baseline
that needs decomposition and full runtime qualification. Neither eager nor the
bounded-divergence rule unlocks the existing graph-path row.

The wrapper graphing/fusion row shares the same defect prerequisite and remains
blocked as well; its separate attributable `>=1.5 ms/frame` screening floor is
still unmet.

### Production exposure

Production uses a long-lived graph-enabled engine, so the construction result is
operationally relevant rather than automatically averaged away per session. In
the tested replay, a constructed engine stayed in its basin across new request
IDs, abort/cleanup, block allocation, reuse, and synchronization. The evidence
therefore predicts instance-stable behavior for the tested path: an exact-basin
instance remains exact under those controls, while a divergent-basin instance
repeats the sensitive decision. Restarts, worker replacement, scale-out, or new
replicas can construct a different basin.

That does **not** establish an 18/56 per-session defect rate, a 4/12 production
incidence, or that arbitrary conversations diverge at their 39th call. The
retained causal prefix is one controlled sequence, and its mismatch reconverges
at the next recorded call through the measured 47-call divergent runs. User-
visible semantic/audio impact on general traffic is unmeasured.

The restored production container's artifact identity does not reveal its
basin. Step 5 did not replay the retained prefix against that restored long-lived
service, and this review intentionally performed no live probe. Consequently,
the qualified identity was restored, but whether that particular running Nano
engine is in the exact or divergent basin is unknown. Production is therefore
**potentially affected at the released graph-path correctness level**, not
proven universally wrong and not proven unaffected by long-lived operation.

## 7. Restoration stall/retry

The retained restoration evidence is internally consistent and passes.

- Pre-stop health was `ready` with `active_client:false`.
- The original container was stopped, then its stopped instance was removed
  only because the retained v4 launch required the fixed name; no image or
  volume removal is recorded.
- Attempt 1 used image
  `sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`,
  completed model construction, reached the first EarTTS pipeline-warm-up
  request, and remained there for more than seven minutes without binding the
  port. Its logs, inspect, and stop/removal record are retained.
- Attempt 2 used the same image and retained v4 launch path. It reached
  `application_ready`; final `/health` recorded `ready` and
  `active_client:false`.
- Pre/post revision, parent-model SHA-256, Nano SHA-256, EarTTS SHA-256, and
  image ID match. The same six mounts are present, including the repository
  config mounted read-only at `/usr/local/lib/python3.12/config`.

Thus the stall/retry left the stack at the recorded qualified identity. This is
a verification of the retained final evidence, not a claim about state after
that evidence was captured.

## Required deltas for a passing revision

1. Downgrade seed from “rejected” to “not supported/not isolated”; narrow the
   history conclusion to the observed exact basin and cleanup evidence.
2. Reconcile the confirmation cell as negative evidence on prevalence and a
   possible temporal/construction-epoch factor, while preserving that it does
   not erase replicated divergence.
3. Remove the overlay `+0.125` as a dispositive production-margin failure;
   retain it only as greedy-boundary localization. State that the production
   margin is unresolved.
4. Make aggregate selector conclusions data-derived; enforce every sealed
   engine-invalidating condition; make localization bounded-rate fields N/A;
   and implement or explicitly decline the global 30-block/five-engine and
   production-margin gate. Add regression tests for each case.
5. Update the ranking consequence to “characterized graph-construction defect,
   exact-gate repair required,” explicitly excluding the bounded gate, and add
   the long-lived production-instance qualification above.
