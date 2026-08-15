# Step 5 track 2: tail-pairing wedge-closure design decision

**Date:** 2026-08-14

**Scope:** design and static source study only

**Decision:** the reliability precondition is **not closed**. Tail-restricted PAD
pairing remains **inadmissible** for Step 5 selection.

The only technically credible route is a bounded offline investigation of the
retained packed request, with a pair-only graph-disabled execution hypothesis.
That route is **NEEDS-EXPERIMENT**, not a finding that the wedge is fixed.
Eliminating the packed request while preserving its amortization is **NO-GO**
on the pinned public fork, and supervisor recovery is **NO-GO** for the current
Step 5 candidate because it is a multi-minute cold reconstruction rather than
an in-place worker recovery.

This document makes no runtime or product change. No live service, GPU probe,
or container was used.

## 1. Decision contract and evidence labels

The governing [Step 4 ranking](../reports/step4-candidate-ranking-20260813.md)
places repaired tail pairing first as historical precedent only: its retained
association is approximately 11.1--11.3 ms per delivered frame, but it was not
a same-image, counterbalanced arrival-clock A/B. The
[Step 5 plan](generation-frame-latency-plan.md#step-5--structural-candidate-selected-by-the-step-4-table-not-before)
keeps pairing inadmissible until exactly one of these closures exists:

1. remove the two-position packed call while retaining amortization;
2. repair the specific worker/graph/output-wakeup defect and pass deterministic
   replay plus long worker stress; or
3. prove that a supervisor can kill and recreate the worker without stale state
   or a surviving client lock, with single-session failure and an explicit
   service-recovery SLO.

A timeout or session close alone is not closure. Even after wedge closure, the
tail-state matrix and same-image counterbalanced A/B required by the plan still
apply.

The findings below use four labels deliberately:

- **[CODE]** follows from static reading of the pinned or retained source.
- **[EVIDENCE]** is an admitted retained measurement or failure record.
- **[EXPERIMENT]** requires a future offline execution; it was not performed
  here.
- **[JUDGMENT]** is a design decision drawn from code and admitted evidence.

## 2. Static source boundary

The study read these exact source inputs:

- the retained host implementation in
  [`runtime_optimizations.py`](../src/nemotron_voicechat_runtime/runtime_optimizations.py)
  and scheduler/Mamba changes in
  [`patch_public_pad_pair_scheduler.py`](../src/nemotron_voicechat_runtime/patch_public_pad_pair_scheduler.py);
- the separately gated FULL-graph experiment in
  [`patch_pair_full_graph.py`](../src/nemotron_voicechat_runtime/patch_pair_full_graph.py);
- NVIDIA Speech revision
  `911ec674ab40f04302ef33672be4179f45a7310f`, the public VoiceChat vLLM fork
  revision `237ba3cafc14d514551539ad980f17d40d273dfa`, and native vLLM revision
  `b31e9326a7d9394aab8c767f8ebe225c65594b60`, all pinned by
  [`qualified-candidate-1.json`](../config/qualified-candidate-1.json);
- the 44-file qualified vLLM delta escrow in
  [`tools/provenance/qualified-deltas`](../tools/provenance/qualified-deltas),
  reconstructed statically to the manifest's 966-file tree SHA-256
  `d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.

No inference was made from a newer upstream API. The lifecycle conclusions are
for the pinned single-GPU `AsyncLLM`/`AsyncMPClient`/`UniProcExecutor` stack.

## 3. What the retained pair actually does

**[CODE]** The retained host wrapper has one persistent Speech request and one
generation iterator. On the first eligible tail row it copies the fused
embedding into `state["pending"]` and returns a synthetic PAD without asking
vLLM to run. On the next eligible row it concatenates pending and current
embeddings into a two-row tensor, calls `_generate_packed_pad_pair()`, sends one
`append_request()` for the existing backend request ID, and waits on the same
iterator's `__anext__()`.

The scheduler interprets the two rows as one real position plus one speculative
PAD position. Acceptance is conjunctive across the text and function-token
streams. The hybrid Mamba patch records a shadow of the position-one recurrent
state and restores it if position two is rejected. This is not ordinary request
batching: it is a custom, dependent two-position speculative step for one
request.

**[EVIDENCE]** The retained wedge trace in
[`nano-step-stall-fix-plan.md`](nano-step-stall-fix-plan.md) ends after perception
for frame 23, at the first entry into `_generate_packed_pad_pair()` for the row
buffered at frame 22 plus the new row. The packed Nano call never returned and
no later model-step event appeared. It is one exact missing-completion
signature; it does not identify which internal boundary failed.

Candidate 1 used the ordinary graph-enabled path and did **not** enable the
separate pair FULL graph. Candidate 2 did enable that experiment. Therefore the
retained trace does not support blaming the dedicated FULL graph.

## 4. Verdict summary

| Option | Feasibility established by reading | Verdict now | Exact reopening condition |
|---|---|---|---|
| (a) Remove packed call, retain amortization | No exact supported form exists in the pinned fork | **NO-GO** | A new primitive must prove two dependent positions in one amortized model execution with exact rejection/rollback; that would be a new speculative implementation, not one of the proposed substitutions |
| (b) Root-cause and fix retained wedge | A narrow, testable pair-only graph-dispatch hypothesis exists; the trace cannot select it statically | **NEEDS-EXPERIMENT** | Matched within-epoch causal localization and repair, exact logical-position replay, then the powered engine/session-block stress gate with zero missing completions |
| (c) Supervisor kill/recreate | Clean state requires replacing the whole Nano engine frontend/core epoch; current APIs do not restart a dead worker in place | **NO-GO** | Product acceptance of a multi-minute recovery SLO plus a separately qualified process-isolation/reconstruction project |

**[JUDGMENT]** The answer to the Step 5 precondition today is **no**. Option (b)
is a research path that could change that answer; it is not present evidence of
closure.

## 5. Option (a): eliminate the packed call and retain amortization

### 5.1 Two sequential submissions with fused host-side scheduling

**Feasibility — NO-GO.**

**[CODE]** Position `t+1` is causally dependent on the sampled text/function
result at position `t`: an unexpected non-PAD result changes the fused embedding
and the control state used for the next position. An exact implementation must
therefore await the first result, correct the next embedding if necessary, and
only then submit the second position. Host-side fusion can reduce Python or IPC
bookkeeping, but it cannot fuse those two model executions.

The current append channel also supplies one replaceable custom-input slot per
request. Two appends issued before the first is consumed are not a request-local
FIFO. Adding such a FIFO would serialize two worker steps after delivery; it
would not recreate the retained one-forward-pass-for-two-positions
amortization.

**Implementation scope.** A safe sequential variant would touch the Speech
wrapper and perhaps coalesce host preparation. It would not need the speculative
scheduler or Mamba shadow. It is essentially the already retained sequential
repair, whose closest preflight had mean 86.168 ms overall, 84.024 ms excluding
BOS, and 293.9 ms of client-arrival debt. It cannot claim the pair-era benefit.

**Residual risk.** Host fusion could obscure ordering or control-boundary bugs
without removing a model pass. Any performance claim would need a new A/B and
cannot inherit the 11.1--11.3 ms/frame historical association.

**Qualification scope.** Exact causal-prefix outputs and ordinary sequential
reliability would be necessary, followed by the common cadence gates. Even if
green, this qualifies a sequential scheduler, not wedge-closed pairing.

### 5.2 Public-fork batched-token, single-request forms

**Feasibility — NO-GO.**

**[CODE]** The public VoiceChat fork originally had a multi-row custom-input
“auto-extend” form. If custom inputs exceeded the uncomputed sequence gap it
inserted placeholder output IDs and ran the extra rows as prefill. The qualified
delta deliberately removes that path and maps the two-row PAD case onto
`spec_token_ids` instead.

The original form is not an exact substitute:

- ordinary non-speculative multi-row execution samples the last scheduled
  position, not both positions;
- it therefore cannot observe and reject a non-PAD text/function result at the
  first position;
- blindly committing both hybrid recurrent positions corrupts the second
  position whenever the assumed first PAD is false.

Scheduler batching of two request IDs is also inapplicable. Those requests own
independent KV/recurrent state. The fork's CFG shadow request is a special
lockstep clone, not a continuation-state fork/merge API. Making a second request
share and conditionally merge the first request's Mamba/KV state would recreate
a custom speculative primitive with the same state-commit problem.

**Implementation scope.** Restoring auto-extend would cross the qualified fork,
GPU runner, sampler, hybrid cache, output protocol, and exact-output contract.
Making it correct requires first-position sampling plus conditional recurrent
commit/rollback—in substance, a new version of the retained packed path.

**Residual risk.** Silent first-position misacceptance is worse than a visible
wedge because it can corrupt text/function control and future recurrent state.
Request-alias approaches add cache ownership, cancellation, preemption, and
cleanup hazards.

**Qualification scope.** Every accept/reject combination for text and function
tokens, recurrent-state equivalence at both positions, preemption/cancellation,
and all tail/control transitions would need exact replay. Passing them would
qualify a new speculative engine feature, not a supported public-fork form.

### 5.3 Persistent-request continuation tricks

**Feasibility — NO-GO.**

**[CODE]** The retained implementation already uses the strongest continuation
available: one persistent request ID, one `RequestState`, one generation
iterator, and repeated `append_request()` calls. The API has no snapshot,
continuation alias, transactional append, or request-state merge primitive.
Starting a new request loses the active Nano recurrent/KV history unless the
entire causal prefix is replayed. Appending twice cannot pipeline the dependency
and, as above, overwrites the single pending custom-input slot if consumption
has not intervened.

**Implementation scope.** Any true continuation trick would require an engine
protocol plus scheduler/cache ownership and rollback work. It converges on the
same custom speculative mechanism rather than eliminating it.

**Residual risk.** Stale iterator outputs, request-ID/epoch confusion, cache
alias lifetime, and abort/preemption races.

**Qualification scope.** Transactional state equivalence and lifecycle stress
would be required in addition to the normal tail matrix. There is no smaller
supported API surface to qualify.

### 5.4 Option (a) decision

**[JUDGMENT] Verdict: NO-GO.** Two exact sequential executions discard the
amortization. Every one-execution alternative needs speculative observation and
conditional recurrent commit, which is the packed mechanism under another API
shape. Option (a) cannot close the precondition with the pinned public fork's
supported forms.

## 6. Option (b): localize and repair the retained wedge

### 6.1 What reading establishes—and does not

**[CODE]** `append_request()` sends one `EngineCoreAppendRequest` through the
async multiprocessing client. The core installs custom inputs, the scheduler
moves a waiting-input request back to running, the worker executes the
two-position speculative descriptor, the sampler and Mamba bookkeeping produce
an output, the core sends it through ZMQ, the frontend output processor routes
it to the request queue, and `__anext__()` wakes.

**[EVIDENCE]** The retained application trace proves only that completion did
not reach the caller. It has no boundary markers inside that chain. Thus source
reading cannot distinguish:

- a scheduler transition or descriptor-construction loss;
- a graph/compiled worker execution that never returned;
- sampler or Mamba shadow/rollback bookkeeping that stalled;
- core-output send, async receive, request routing, or iterator wakeup loss.

The separate
[`step5-graph-reproducibility-20260813.md`](../reports/step5-graph-reproducibility-20260813.md)
finding increases concern about graph-enabled construction: across 12 fresh
engines, 18 of 56 graph blocks diverged at call 39 while the eager control was
18/18 exact. That is adjacent evidence, not a root cause for this wedge.

### 6.2 Concrete primary fix hypothesis

**Hypothesis B1:** the ordinary graph-enabled/compiled hybrid-Mamba
two-position speculative descriptor has a shape- or construction-specific
non-returning path. Route only the one-request/two-position PAD speculative
step through graph-disabled eager execution while retaining the normal
one-position graph path, speculative PAD checking, and Mamba rejection restore.

This is narrow enough to test and preserves the essential amortization: both
rows still traverse the Nano backbone in one worker call. Its latency is
unknown; “eager” here is pair-only execution routing, not global Nano eager
mode.

**Alternative B2:** if boundary evidence shows worker execution returned, the
defect lies in scheduler/output wakeup. The fix must then target the first
missing transition—request wake/run bookkeeping, output enqueue/send, frontend
routing, or iterator notification—rather than add a timeout around
`__anext__()`.

### 6.3 Preregistered offline experiment

No part of this experiment was run during the design study.

#### Artifact and independence

- Use one immutable image reconstructed from the pinned qualified tree and the
  reviewed diagnostic patch only. Record source, patch, image, model, fixture,
  and configuration hashes before execution.
- Use the retained failing causal prefix and fused-input rows if the source
  artifact contains them; otherwise reproduce the same state transitions from
  a frozen fixture and mark the inability to replay the exact tensors.
- Every engine epoch is fresh. Do not reuse an engine after any timeout, crash,
  unknown completion, or failed exactness check. The primary arms are matched
  request lanes inside the same worker/engine epoch, reconstructed independently
  from the same frozen prefix; independent fresh engines are replicates, not
  substitutes for the within-epoch match.

#### Boundary localization

Add bounded, nonblocking sequence records at:

| ID | Required boundary |
|---|---|
| B0 | host packed-call entry and append intent |
| B1 | `EngineCoreAppendRequest` decoded by the core |
| B2 | scheduler custom-input install, waiting/running transition, and finalized descriptor |
| B3 | worker execute entry with request/epoch, token count, graph mode, and descriptor key |
| B4 | model forward returned |
| B5 | sampling plus speculative accept/reject and Mamba commit/restore returned |
| B6 | core output enqueued |
| B7 | core output sent |
| B8 | async client received and output processor routed the request |
| B9 | per-request iterator awakened and host call returned |

Records must use a preallocated ring or sidecar path and must themselves pass a
trace-off/trace-on noninterference check. A wall-clock timeout is only the
diagnostic trigger; it is never counted as recovery.

#### Phase 0: CPU-only protocol race test

Use a fake executor/model-output source against the exact request, scheduler,
output-processor, and iterator path. Randomize append arrival, waiting-input
transition, output batches, cancellation, and task-error timing. Require at
least 1,000,000 schedules with no lost B0-to-B9 completion and exact request
routing. Inject failures at each boundary to prove the harness detects loss.

A failure localizes the transport/scheduler/output hypothesis to the first
missing boundary. A pass only weakens that hypothesis; it cannot clear the
GPU/graph hypothesis or the reliability precondition.

#### Retained-exposure census and calibration consequence

The historical census is a prerequisite to, not an output of, stress-gate
calibration. The artifact-level counts recoverable from the admitted documents
are:

| Unique retained exposure | Exact-signature events | Eligible packed calls | Requests/sessions | Engine constructions |
|---|---:|---:|---|---|
| Missing-completion trace `2d08e743…` | 1 | 11 attempts: 10 completed and the terminal frame-23 attempt | one request in one live session | one running engine; construction identifier absent |
| Closest pair-era trace `f110f063…` | 0 observed | 12 completed | one response in one live session | construction identifier absent |
| Step-7 same-process teacher replay, reported identically in Step 8, Step 9, PC2, and promotion assembly and therefore counted once | 0 observed | 675 completed: 663 accepted and 12 rejected | component request lanes, not product sessions | one same-process engine execution; no population sampling |
| Strict-v3 host-barrier session `e9915410…` | 0 observed | 519 completed accepted pairs | one live measurement session | one running engine; construction identifier absent |
| Other retained post-FC campaigns and pair-era qualification reports | none reported | not recoverable as an audited eligible-call denominator from the admitted summaries | some session counts are reported, but their union and request membership are not audited | construction/reuse mapping absent |

The 1,217 countable attempts above are a selected mixture of one failure trace,
one latency comparator, component exactness replay, and one profiling session.
They are not an enumerated sampling frame and must not be pooled into `1/1217`:
the trace scan that found the exact signature does not retain an audited union
of all eligible pair-era calls, requests/sessions, or engine epochs, and several
reports repeat the same Step-7 exposure. **[EVIDENCE] The observed historical
base rate is therefore unidentified.** A future census must publish the
artifact manifest, de-duplicate request and engine identities, and count exact
signatures, eligible packed calls, requests/sessions, and fresh constructions
before historical-rate calibration is reconsidered.

Because the denominator was not recovered, Phase 2 below is calibrated instead
to an explicitly selected unacceptable clustered-failure rate. Its call total
is not described as calibrated to the retained event.

#### Phase 1: offline worker discrimination

Use these fixed logical lanes:

| Lane | Configuration | Purpose |
|---|---|---|
| S | pair-disabled sequential execution of the logical-position transaction | exact state oracle from the same frozen prefix |
| B | identical to A except pair-only graph-disabled/eager worker dispatch | primary B1 intervention |
| C | the same packed semantics forced through the global-eager route by a lane-local selector | optional graph-independent localization control; omit if this requires a construction change |
| A | released candidate-1 pair path; normal graph-enabled mode; dedicated pair FULL graph off | terminal reproduction/localization attempt |
| D | candidate-2 dedicated pair FULL graph | optional localization in a separate matched study only; never substitute it for A |

Preregister **12 independent fresh engine replicates**. In every replicate,
create independent S, B, optional C, and A request lanes in the same worker,
restore each lane from the identical frozen causal prefix, and execute them in
the fixed order **S, B, optional C, A**. S and B must finish before A starts.
A is always last because a missing or unknown completion is terminal for that
engine. Such a request is never resumed, cancelled-and-retried, or replayed on
that engine; record its last boundary, destroy the epoch, and continue only on
the next preregistered fresh engine.

The S lane is a **logical-position transaction**, not an unconditional
two-position execution:

- For conjunctive PAD acceptance, execute sequential logical positions one and
  two and compare the packed lane at the committed boundary after position two.
  The packed result exposes its last accepted position, so compare custom-output
  row 1 (the second row, zero-based) with S position two.
- For text-only, function-only, or conjunctive rejection, execute only logical
  position one. Compare at the committed boundary immediately after position
  one; do not execute position two. The packed result exposes its last accepted
  position, so compare custom-output row 0 with S position one. Correct the
  second fused input using the actual position-one text/function result, retain
  that corrected row as pending in both lanes, and execute it only on the next
  logical scheduler action. That next action is a separately counted pending
  drain/roll transition, not part of the rejected transaction.

At the applicable boundary require exact equality of emitted text token,
function token and their conjunction, accepted-token count, every retained
custom output in the selected row, committed KV state, committed Mamba conv/SSM
state, rollback-shadow contents, shadow index/pointer and virtual-engine
identity, and host pending-row contents plus correction flags/assumptions.
Also require the uncommitted second-position KV/recurrent state to be absent or
restored to the position-one boundary on rejection. Cover PAD accept,
text-only rejection, function-only rejection, conjunctive rejection, control
boundaries, abort, and the next-action pending drain. Any difference invalidates
the lane before causal, timing, or reliability interpretation. Same-worker
independent lanes follow the established same-process Step-7 pattern and make
cache comparisons without crossing a construction epoch.

The executable causal decision is preregistered:

1. A **qualifying replicate** is one engine epoch in which S is exact, B is
   exact and completes, and terminal A reaches the same frozen prefix but loses
   B0-to-B9 completion at the same preregistered boundary. B then A is the
   intervention-on/intervention-removed flip; no post-A “flip back” is attempted.
2. B1 is supported only with the same boundary-localized discordance in at
   least two of the 12 independent fresh engines, no B missing completion, and
   no exactness failure. Report all 12 outcomes; the threshold identifies a
   repeatable causal case and is not an incidence estimate.
3. If A reaches B4 or later before loss, the forward-non-returning form of B1 is
   rejected; repair the first missing post-forward/output transition and rerun
   the complete fixed-order design with that single intervention.
4. If fewer than two A lanes reproduce, the result is **INCOMPLETE**, not “fixed
   by stress.” Absence of another rare wedge cannot identify its cause.

If per-request within-engine dispatch cannot be implemented without changing
anything beyond the dispatch intervention, the primary design does not silently
fall back to counterbalanced epochs. First run a non-destructive discriminator
that preclassifies the relevant graph construction without exercising the
packed failing prefix. Then preregister 12 fresh matched A/B epoch pairs **per
observed construction stratum**, randomize A/B assignment within each pair,
run the B epoch before the terminal A epoch, and apply the same exactness and
two-discordant-replicate rule inside a stratum. An A timeout remains terminal.
Construction-stratified agreement may identify the intervention;
counterbalanced independent epochs without that
preclassification remain non-causal and cannot advance to Phase 2. C and D are
localization controls only and cannot repair a failed primary comparison.

#### Phase 2: post-fix reliability gate

Only a causal and exact Phase 1 repair enters stress. The separate estimands
are (i) missing B0-to-B9 completion per eligible packed call, reported as
descriptive exposure; (ii) the probability that a fresh engine construction
has any missing completion, death, timeout trigger, scheduler/output task
error, or exactness failure; and (iii) the probability that a fresh product
session has any such failure or fails clean teardown. Engine and session blocks,
not calls inside them, are the uncertainty units for (ii) and (iii).

The retained base rate is unidentified, so the conservative design declares
**1% per engine and 1% per session unacceptable**. To have at least 95% chance
to observe one or more failures when the true block rate is 1%, the zero-event
sample size is
`ceil(log(0.05) / log(0.99)) = 299`. Round up to **300 independent fresh engine
constructions** and require **two fresh sessions per engine (600 sessions)**.
One session per engine, selected by a frozen balanced first/second-session
schedule, forms the 300 independent session blocks for the powered session
estimand; the other 300 sessions exercise repeated lifecycle and contribute to
the zero-event gate but are not miscounted as independent of their engine mate.
With zero events the exact one-sided 95% binomial upper bound is about 0.994%
for both the 300 engine blocks and the 300 powered session blocks. These
calculations assume independent fresh engines; if construction provenance shows
dependence, the affected engine and sentinel-session blocks do not count and
the campaign expands.

Across those blocks, retain floors of **24 aggregate hours** and **250,000
eligible accepted-or-rejected packed calls**. Distribute the calls rather than
allowing long PAD-accept stretches to dominate. The preregistered transition
minima are:

| Exposure | Campaign minimum and distribution rule |
|---|---|
| accepted pairs | 100,000, with at least one in every engine and session |
| text-only rejection | 25,000, with at least one in every engine |
| function-only rejection | 25,000, with at least one in every engine |
| conjunctive rejection | 25,000, with at least one in every engine |
| corrected pending-row next-action drains | all 75,000 rejection cases, classified by the originating rejection; at least one of each class in every engine |
| abort and cancel | 300 of each, at least one of each per engine and placed across pre-dispatch, post-dispatch-known-complete, and pending-row states; never cancel/retry an unknown completion |
| clean session teardown | all 600 sessions, including pending and just-drained states in balanced assignments |
| repeated request/session creation | at least two fresh sessions and four fresh Nano requests per engine, with no request ID, queue, cache, or pair-state reuse |

The remaining calls needed to reach 250,000 must be allocated across these
classes by a frozen schedule; raw PAD acceptance may not replace any minimum.
Run the deterministic failing-prefix replay before and after every engine stress
segment. Require:

- zero missing B0-to-B9 completions, engine deaths, timeout triggers, or
  scheduler/output task errors;
- zero token, function-token, selected-row custom-output, accept/reject,
  committed KV/Mamba, rollback-shadow, shadow-pointer, pending-row, or
  correction-state differences from the logical-position oracle;
- all 300 engine constructions to pass the same deterministic failing-prefix
  replay before and after their stress segment; and
- no monotonic queue, request-state, GPU-memory, or host-memory growth after
  request/session teardown.

At 250,000 clean calls, `3/N_calls = 1.2e-5` is retained only as descriptive
per-call exposure accounting. It is not a reliability confidence claim, does
not use calls as independent Bernoulli trials, and is not calibrated to the
unidentified historical event rate. The powered claims use the 300 independent
engine blocks and one preregistered session block per engine; all 600 sessions
must nevertheless remain event-free. Deterministic causal replay remains
mandatory.

### 6.4 Implementation and qualification scope

**Implementation scope if B1 wins.** Add the narrow pair descriptor dispatch,
keep normal one-token graph behavior unchanged, and retain the current
text/function conjunction and Mamba shadow/restore. Add permanent bounded
boundary/error counters but not production high-volume tracing. Review the
qualified fork delta, its provenance, and immutable image as a coupled change.

**Implementation scope if B2 wins.** Change only the first proven broken
scheduler/output transition, with epoch-aware request routing and explicit task
failure propagation. Do not retry a request whose worker completion is unknown.

**Residual risk.** A one-off trace may remain irreproducible; instrumentation
may perturb a timing race; pair-only eager may expose an untested descriptor or
lose the latency advantage; a construction classifier may not fully capture
the neighboring graph basins; and clustered stress bounds only the declared
engine/session rates and does not prove impossibility or recovery.

**Qualification scope after wedge closure.** The Step 5 executable tail-state
matrix remains required, including entry/exit, pending rows, rejection/rollback,
BOS/EOS, SOTC/EOTR, post-function-call BOS, interruption, abort, and session
caps. Then run the same-image, arrival/queue-clock, counterbalanced A/B over
independent responses and the common structural/audio gates. Historical
11.1--11.3 ms/frame association is context, not a promotion floor.

### 6.5 Option (b) decision

**[JUDGMENT] Verdict: NEEDS-EXPERIMENT.** B1 is the narrowest credible fix
hypothesis and the recommended next action. Static reading cannot turn it into
a GO. Option (b) becomes **GO** only after causal localization, exact replay,
and the Phase 2 gate all pass; no-reproduction, timeout-only handling, or
stress-only survival leaves it **NEEDS-EXPERIMENT** or retires the candidate.

## 7. Option (c): supervisor kill and recreate

### 7.1 Pinned lifecycle and surviving state

**[CODE]** Speech constructs an `AsyncLLM`, which constructs an
`AsyncMPClient` and a background EngineCore process. On the single-GPU path the
core selects `UniProcExecutor`; its driver worker, CUDA context, model weights,
KV/recurrent caches, and graphs live in the same EngineCore process. There is no
separately restartable GPU-worker child beneath that core.

When the core dies, the client marks `resources.engine_dead`, shuts down its
communication resources, and subsequent operations raise `EngineDeadError`.
The process manager can close its engines, but neither the client nor Speech has
an API that replaces a dead core and resumes the existing request.

| State after killing EngineCore | Location | Required disposition |
|---|---|---|
| CUDA context, weights, caches, graphs, scheduler requests | killed EngineCore process | disappears only after process exit and GPU resource reclamation |
| ZMQ sockets, output-handler task, `engine_dead`, output queue | parent `AsyncMPClient`/`AsyncLLM` | destroy; never reconnect or reuse |
| frontend `OutputProcessor.request_states`, collectors, per-request queues | parent `AsyncLLM` | destroy with the old frontend epoch |
| Speech request map, generated-token history, generation iterator | parent Speech engine wrapper | destroy; cannot resume |
| retained `_voicechat_pad_pair_states` and pending fused row | parent process | clear by destroying the whole Nano wrapper epoch |
| outer voice session, RNNT/turn/function/audio state and client lock | server/session layer | fail the active session, unwind fully, and epoch-fence before releasing the lock |
| EarTTS engine and unrelated server models | separate objects/processes | may remain only if isolation and GPU-memory reclamation are proven |

Thus a “worker restart” that preserves the parent client or Speech wrapper is
not clean. The minimum safe unit is a new Nano engine frontend **and** core
epoch. Active Nano recurrent state is unrecoverable, so the current session
must fail.

### 7.2 Minimum credible supervisor design

A future architecture would need to:

1. detect a missing completion and atomically mark the Nano service
   unavailable;
2. increment an epoch and reject all old request/output messages;
3. force-close the single active session and prove all session state and its
   client lock unwind;
4. terminate, then if necessary kill, the old EngineCore; verify PID exit and
   GPU allocation reclamation;
5. destroy the old `AsyncLLM`, client tasks/sockets, Speech Nano wrapper,
   request maps, pair state, and pending row;
6. construct a fresh Speech Nano wrapper/`AsyncLLM`/EngineCore, load weights,
   allocate caches, and recapture or compile graphs;
7. run an exact fresh-request self-test; and
8. publish the new epoch as ready only after every check passes.

For stronger fault containment, the entire Nano frontend plus EngineCore should
be a supervised service subprocess with an epoch-tagged IPC contract. That
makes parent-state destruction structurally provable, but it is a substantial
architecture project and still does not save the failed session.

### 7.3 Recovery-time plausibility

**[EVIDENCE]** The admitted
[cold-start accounts](../reports/step9-cold-start-20260807.md) record fresh Nano
vLLM engine initialization from **115.571 to 154.015 seconds**: 115.571 and
115.725 seconds for the two reduced-cache runs, 122.127 seconds for the keyed
exact run, 126.666 seconds for cache-default, and 154.015 seconds for the
uncached baseline. Admitted optimized full-service readiness spans **247.954 to
256.016 seconds** across the Step 9 and PC2 records. These are startup
measurements, not hot-recovery measurements; full-service readiness is scale
and context only, not a Nano-only lower bound.

A Nano-only reconstruction can in principle leave EarTTS and perception alive.
Conversely, even the 115.571-second fresh Nano datum excludes forced
termination, uncertain CUDA cleanup, wrapper reconstruction, readiness
self-test, publication, and lock/session unwind. The repository contains no
measured hot-recreate distribution from which to derive a product SLO or a
recovery lower bound.

**[JUDGMENT]** A defensible recovery-planning assumption is on the order of
minutes, not seconds. For qualification planning, preregister **ready within
240 seconds after detection in every trial** solely as a conservative
experiment ceiling.
It is a chosen judgment with explicit margin beyond the admitted fresh-Nano
range for termination, GPU reclamation, wrapper creation, readiness self-test,
and publication; it is not evidence-derived, a measured lower bound, or a
product-approved SLO. Report p50/p95/max for detection, termination, GPU
reclamation, engine initialization, self-test, publication, and total outage
separately. Failure to reclaim GPU state or become ready by 240 seconds is a
failed experiment, never a reason to extend the timer post hoc. Product
acceptance and a measured recovery distribution remain required before any
service-recovery SLO can close option (c).

### 7.4 What proving recovery would require

**Implementation scope.** External Nano service boundary or equivalent full
frontend/core epoch factory; supervisor and health state machine; epoch-fenced
IPC; forced session teardown; lock ownership audit; staged termination; GPU
reclamation check; clean model construction; readiness self-test; and failure
injection. This scope is materially larger than the tail scheduler.

**Residual risk.** A stuck CUDA operation may delay process exit or memory
reclamation; the outer server may retain a task, iterator, lock, or session
reference; surviving EarTTS allocations may prevent Nano reconstruction;
repeated deaths can cause an outage loop; and a minutes-long outage may be
operationally equivalent to restarting the product.

**Qualification scope.** In a future offline campaign, inject a hang at every
B0--B9 boundary and SIGKILL the EngineCore during load, graph capture, active
pairing, rejection restore, and teardown. Across at least 30 independent
recovery trials, require zero old-epoch outputs, zero surviving requests/tasks
or locks, zero stale pair/session state, exact self-test output, bounded GPU and
host memory, successful new-session service, and the fixed 240-second ceiling.
Repeat-kill trials must prove that recovery itself is restartable. Product
owners must explicitly accept the single-session loss and recovery SLO before
this work can count toward the Step 5 precondition.

### 7.5 Option (c) decision

**[JUDGMENT] Verdict: NO-GO for Step 5.** Clean recreation is technically
conceivable, but it is not supported by the current lifecycle and is not a
bounded worker retry. It requires whole-Nano-epoch replacement and has a
multi-minute cold path. Reopen as a separate availability project only if that
outage and implementation scope are explicitly accepted; do not keep pairing
eligible on the promise of future supervision.

## 8. Recommendation and stop/go rules

Run only the preregistered option-(b) diagnostic as the next pairing action.
Start with the CPU protocol race test, then the four-arm offline worker
discrimination. Do not implement the tail state machine or spend a live
qualification window until the wedge has a causal, exact repair.

The decision rules are:

- **GO for (a): impossible on current evidence; verdict NO-GO.** Reconsider only
  for a genuinely new transactional dependent-token primitive, reviewed as a
  new engine feature.
- **GO for (b):** only after repeatable causal localization, intervention
  removal in the terminal matched ordering, exact logical-position-oracle
  replay, and the powered 300-engine/300-independent-session gate across 600
  exposed sessions, with the 24-hour/250,000-call exposure floors. Until then
  the verdict is NEEDS-EXPERIMENT.
- **GO for (c): impossible within current Step 5 scope; verdict NO-GO.** Reopen
  only after product acceptance of single-session loss and a minutes-scale SLO,
  followed by a separate supervisor qualification.
- **GO for the pairing candidate:** one closure above must be GO **and** the
  plan's tail-state and same-image counterbalanced arrival-clock qualification
  must pass. None has done so.

If option (b) cannot reproduce and causally localize the wedge, or its repaired
path loses exactness or the amortized latency benefit, retire pairing from Step
5 despite its historical performance lead.

## 9. Validation disposition

This was a documentation-only study. The existing suite, anchor checks, and
ruff result were intentionally left untouched and were not rerun. No code,
configuration, tests, runtime image, container, or live service was changed.
