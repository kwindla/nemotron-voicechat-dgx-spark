# Step 5 pairing wedge-closure design review — round 1

**Review date:** 2026-08-14 UTC
**Verdict:** **FAIL**

The design reaches the right present-tense disposition: the pairing reliability
precondition is **NOT CLOSED**, the pairing row remains **INADMISSIBLE / retained
precedent only**, and no ranking-table promotion or Step 5 implementation is
authorized. Option (a)'s supported-form NO-GO and option (c)'s current-lifecycle
NO-GO are substantially correct.

The failure is in the preregistration that would be used to turn option (b) into
a GO. Independent engine arms do not discriminate a pair-dispatch repair from
the already admitted fresh-construction graph-basin effect; the sequential
oracle is underspecified at rejection; and the 24-hour/250,000-call/five-epoch
gate is not calibrated from the sole observed wedge or from an exposure
denominator. Those defects could let a non-causal or under-sampled repair pass.
There is also a smaller evidence correction: minutes-scale reconstruction is
supported, but the selected 240-second ceiling is not derived from all admitted
startup measurements.

This was a static/read-only review. I inspected the pinned Speech revision
`911ec674ab40f04302ef33672be4179f45a7310f`, VoiceChat vLLM revision
`237ba3cafc14d514551539ad980f17d40d273dfa`, and the reconstructed 966-file
qualified vLLM tree. The reconstruction recomputed the manifest tree digest
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.
No live service, GPU, container, or model probe was used.

## Findings

### 1. BLOCKER — Phase 1 is confounded across engine constructions

The design assigns A, B, and C to independent fresh engine epochs and says B1
is supported when A wedges while B and C do not
([design, lines 266–340](../step5-pairing-wedge-closure-design.md#phase-1-offline-worker-discrimination)).
That comparison is not causal on this evidence base. The adjacent admitted
graph study found engine-stable but construction-varying graph basins: 4 of 12
production graph engines contained all 18 divergent blocks, no engine changed
basin, and eager was 18/18 exact. The report explicitly says those correlated
epochs do not estimate an incidence probability
([graph study, lines 260–279](../../reports/step5-graph-reproducibility-20260813.md#primary-discrimination),
[lines 316–327](../../reports/step5-graph-reproducibility-20260813.md#full-prefix-confirmation)).

Therefore, “A fails on epoch X; B passes on epoch Y” cannot distinguish the
pair-only eager intervention from construction-epoch selection. The design
acknowledges the confound as residual risk at lines 373–376, but does not remove
it from the acceptance rule. The reversal clause is also operationally
ambiguous: after a genuine A wedge, that engine cannot safely execute the
reverse arm, and a reversal on a different construction retains the confound.

Required delta:

1. Make the primary comparison matched within an engine epoch, using independent
   request lanes reconstructed from the same frozen prefix. Run the exact
   pair-disabled/sequential lane and pair-only graph-disabled B lane before the
   terminal A attempt; run A last. A qualifying replicate is one engine in which
   B is exact and completes, then A reaches the same prefix and loses completion
   at the preregistered boundary. Repeat that matched ordering across a
   preregistered number of independent fresh engines.
2. If within-engine routing cannot be implemented without changing more than the
   dispatch intervention, preclassify the relevant graph construction on a
   non-destructive discriminator and compare within construction strata. Do not
   treat counterbalanced independent epochs alone as causal identification.
3. Rewrite the “flip / flip back” rule in executable order. An unknown-completion
   request is terminal for that engine; it is never resumed or retried.

This preserves the retirement document's no-timeout-retry discipline. The
existing requirements to discard an engine after timeout, use timeout only as a
diagnostic trigger, and never retry unknown worker completion are correct.

### 2. BLOCKER — the sequential oracle is not defined for rejection

The design asks every packed call to match a “two-sequential-position
reference” while also covering text, function-token, and conjunctive rejection
([design, lines 322–327](../step5-pairing-wedge-closure-design.md#phase-1-offline-worker-discrimination)).
Taken literally, that oracle is wrong for a rejected draft. The qualified
scheduler accepts either two tokens or only the first replacement, decrements
`num_computed_tokens` for the rejected speculative position, and the retained
wrapper keeps the second fused row pending for later correction/execution. A
sequential lane that executes both positions is one committed position ahead
and cannot be used for recurrent/KV equivalence.

Required delta:

1. Define the oracle as a logical-position transaction. For PAD acceptance, the
   sequential lane advances two positions and compares state after position two.
   For any rejection, it advances only position one, compares committed Mamba/KV
   state at that boundary, corrects and retains position two as pending, and does
   not execute it until the next logical scheduler action.
2. State exactly which custom-output row is compared in each case and require
   text/function conjunction, accepted-token count, committed KV/Mamba state,
   rollback shadows, shadow index/pointer, and pending-row contents to match.
3. Use independent request lanes in the same worker where possible, following
   the established same-process Step-7 gate pattern, so cache state can be
   compared without crossing a construction epoch.

Without this definition, “zero recurrent-state differences from the sequential
oracle” in Phase 2 is not an executable acceptance criterion.

### 3. BLOCKER — the stress gate is not base-rate calibrated and uses the wrong independence unit

The retained retirement record supplies one exact missing-completion signature,
but no audited denominator of packed calls, sessions, or engine epochs. The
design nevertheless fixes 24 hours, 250,000 packed calls, and five engine
constructions, then reports `3/N = 1.2e-5` as a zero-event 95% upper bound
([design, lines 342–359](../step5-pairing-wedge-closure-design.md#phase-2-post-fix-reliability-gate)).

The arithmetic is the usual zero-event approximation, but it is not calibrated
to the observed event and its Bernoulli interpretation is not warranted here.
Calls within a persistent request and engine share scheduler, recurrent cache,
graph construction, and output tasks. The adjacent graph evidence demonstrates
exactly this kind of engine-level clustering. Five long engine segments can
produce 250,000 highly correlated calls while sampling only five construction
epochs and few request/session teardown transitions. Neither 250,000 nor five
was derived from the unknown historical exposure denominator.

Required delta:

1. First produce a retained-exposure census: exact-signature events, eligible
   packed calls, requests/sessions, and engine constructions for every admitted
   pair-era artifact whose instrumentation permits the count. If the denominator
   cannot be recovered, say that the observed base rate is unidentified and do
   not describe the chosen N as calibrated to it.
2. Preregister separate estimands and gates for per-call missing completion and
   per-engine/per-session failure. Use engine/session blocks as the uncertainty
   unit for clustered failures; retain `3/N_calls` only as descriptive exposure
   accounting, not the reliability confidence claim.
3. Derive the minimum engine/session count from a stated detection-power target
   against the retained exposure or a conservatively selected unacceptable
   rate. Keep 24 hours and 250,000 calls as floors if desired, but increase and
   distribute engine epochs as the power calculation requires.
4. Add minimum exposure counts for accepted pairs, each rejection class,
   pending drains, abort/cancel/teardown, and repeated request/session creation.
   Raw PAD-accept calls must not swamp the state transitions most capable of
   exposing rollback and lifecycle defects.

The deterministic failing-prefix replay before and after every stress segment
is valuable and should remain mandatory, but it does not replace adequate
sampling of construction and lifecycle epochs.

### 4. IMPORTANT — the 240-second recovery ceiling is a judgment, not a derived result

The lifecycle conclusion is correct. In the pinned single-GPU path,
`AsyncLLM` owns an `AsyncMPClient` and background EngineCore; `UniProcExecutor`
constructs its driver worker in that EngineCore process. Core death marks the
client dead and shuts its IPC resources; neither vLLM nor Speech exposes a
supported dead-core replacement that can resume the request. The frontend
output/request queues, Speech request iterator/history, retained pair state,
and the active session therefore cannot be reused. A clean recovery requires a
new logical Nano frontend/core epoch and failure of the active session. An
explicit factory that destroys and reconstructs all those objects would be
equivalent to that epoch replacement; preserving the old client or request is
not a smaller safe recovery.

The qualitative **minutes, not seconds** conclusion is also supported, but the
evidence paragraph is incomplete. It quotes the uncached baseline Nano init at
154.015 seconds while the admitted optimized startup accounts record Nano init
at 115.571 and 115.725 seconds for the two reduced-cache runs, 122.127 seconds
for the keyed exact run, and 126.666 seconds for cache-default. Those are still
roughly two-minute fresh initializations, and none measures hot recovery. They
support a minutes-scale planning assumption; they do not derive exactly 240
seconds.

Required delta: report the admitted 115.571–154.015-second Nano initialization
range, keep full-service 247.954–256.016-second readiness only as scale/context,
and label 240 seconds solely as a conservative preregistered experiment ceiling
chosen with explicit margin for termination, GPU reclamation, wrapper creation,
self-test, and publication. Do not call it an evidence-derived SLO or lower
bound. Product acceptance and a measured recovery distribution remain required.

### 5. VERIFIED — option (a) missed no supported exact form in the pinned fork

The actual fork supports the study's NO-GO:

- `Request.set_custom_inputs()` replaces one dictionary and resets one consumed
  offset; it is not a FIFO. `AsyncLLM.append_request()` awaits ZMQ submission,
  not scheduler consumption, so two unacknowledged appends can replace the same
  slot.
- The public fork's removed auto-extend path added placeholder output IDs and
  treated surplus custom-input rows as prefill. Ordinary multi-row prefill
  samples at the request's final scheduled position; it cannot observe the
  first text/function result and conditionally reject/rollback position two.
- The qualified delta correctly maps the second row to `spec_token_ids`, after
  which the scheduler's existing speculative path returns one or two accepted
  tokens and rewinds the rejected computed position.
- CFG creates a special unconditional clone with separate request/cache
  ownership. It is not a general continuation snapshot or merge primitive.
  Separate request IDs therefore cannot preserve and conditionally commit the
  active request's hybrid recurrent history.

A worker-resident, two-forward micro-loop with device/worker-side fusion could
be invented, but it does not exist in the pinned fork, would be a new engine
protocol/model feature, and would execute two backbones rather than preserve
the retained one-forward pair saving. It is not a missed supported option-(a)
form and could not inherit the historical 11.1–11.3 ms/frame association.

### 6. VERIFIED — NOT-CLOSED and the ranking-table consequence are correct

Option (a) has no supported exact substitute, option (b) has not been run and
is not yet adequately preregistered, and option (c) is a separate availability
architecture with active-session loss and minutes-scale reconstruction. Thus
none of the Step 5 closure alternatives is presently GO.

The ranking consequence is exactly the current table state: repaired tail
pairing remains ordinal **1, precedent only**, **retained-precedent, then
blocked**, **INADMISSIBLE**, with **NO current ranking authority**. The study
must not remove the row yet because a corrected option-(b) experiment remains a
bounded research route; it must not promote the row because no closure has
passed. If option (b) cannot causally reproduce/localize the wedge, loses exact
transactional state, or loses the amortized latency benefit, retirement from
Step 5 is the correct next consequence.

## Closure checklist for round 2

- Replace cross-epoch A/B/C causal inference with a matched construction design
  and executable terminal-failure ordering.
- Define the rejection-aware sequential oracle and state comparison boundary.
- Add the historical exposure census, clustered reliability estimand, powered
  engine/session count, and state-stratified stress minima.
- Correct the startup range and distinguish the proposed 240-second test ceiling
  from measured evidence and product SLO acceptance.
- Retain the supported-form option-(a) NO-GO, whole-Nano-epoch option-(c)
  lifecycle conclusion, no-timeout-retry rules, NOT-CLOSED verdict, and current
  ranking-table standing.
