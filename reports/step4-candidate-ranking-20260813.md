# Step 4e candidate ranking and Step 5 decision input — 2026-08-13

## Disposition

**Step 4 evidence synthesis: COMPLETE. Step 5 candidate-selection gate: NOT
OPEN.** The admitted record does not authorize an implementation winner. The
repaired tail-pair scheduler has the strongest retained latency precedent, but
its packed-call reliability precondition is open. The Nano measurements expose
large descriptive opportunity envelopes, but the fresh graph-enabled W8
baseline failed its exact-output reproducibility gate. EarTTS-high did not recur
in the capped campaign, and the logging and telemetry studies are descriptive
only. Nano W4, selective EarTTS precision, and two-frame perception batching
have no candidate measurement.

Consequently, “rank” below means **strength of evidence for further work**, not
permission to promote or ship. Only the pairing precedent occupies an ordinal
position. All other candidates are `NR` (not rankable), including candidates
whose descriptive ceiling is numerically larger.

## Standing and common promotion contract

The standings used here are:

- **measured** — a candidate-specific, gate-valid recoverable estimate that can
  order Step 5 candidates;
- **descriptive** — an observed association or impossible-zero-cost envelope
  admitted for orientation, not candidate ordering;
- **retained-precedent** — older, non-counterbalanced evidence that motivates a
  candidate but does not measure the proposed candidate;
- **retained-negative-adjacent** — a retained negative result on a distinct
  mechanism that constrains the hypothesis without measuring the proposed
  candidate;
- **blocked** — the needed state, correctness baseline, causal comparison, or
  candidate measurement is absent or failed.

No row currently has `measured` standing. Every eventual promotion must satisfy
the plan's common absolute floor on the arrival clock: every complete 12-frame
window must have `t[i+11] - t[i] <= 960 ms`, maximum single arrival gap must be
`<= 160 ms` unless a different bound was preregistered for the fixture, and the
release-aware browser replay must show zero underruns over the declared product
envelope at an explicitly accepted reserve. It must also show no regression to
idle latency, voice-to-voice latency, device-level barge-in, semantics/WER, or
audio completeness. These are absolute promotion floors, not evidence that any
candidate meets them. ([P-T], normative/preregistered)

Where the admitted record has no class-specific numeric benefit floor, the
table says so. Such a floor must be preregistered before candidate timing; it
must not be selected after seeing results. The common absolute floor above
still applies.

## Candidate table

| Evidence position | Candidate | Recoverable ceiling and evidentiary standing | Correctness scope | Reliability risk | Preregistered promotion floor | Current Step 5 admissibility and exact unlock |
|---|---|---|---|---|---|---|
| **1, precedent only** | **Repaired tail-restricted PAD pairing** | No measured ceiling for the proposed repaired scheduler. The closest retained configurations show an approximately **11.1–11.3 ms/frame association** in favor of the pair era; this was not a same-image counterbalanced A/B. **Standing: retained-precedent, then blocked.** ([P-A4], retained-precedent) | Executable tail state machine: entry/exit, pending rows, rejection/rollback, BOS/EOS, SOTC/EOTR, post-FC BOS, interruption, abort, and session caps; then same-image, arrival/queue-clock, counterbalanced A/B over independent responses. ([P-5], normative prerequisite) | A two-position packed vLLM call can wedge EngineCore. A bounded timeout and session-fatal close do not recover the worker. Tail restriction reduces exposure, not failure severity. ([P-5], normative prerequisite) | Common promotion contract, plus the plan-required same-image counterbalanced queue-clock A/B. There is no separate preregistered minimum mean saving; the historical association is not a floor. | **INADMISSIBLE. NO current ranking authority.** Unlock only by one of: (a) eliminate the packed two-position call while preserving amortization; (b) fix the worker/graph/output-wakeup defect and pass deterministic replay plus long worker stress; or (c) isolate the engine behind a supervisor that can kill and recreate the worker with proven clean reinitialization (no stale state or client lock survives), single-session failure only, and an explicit service-recovery SLO. The executable tail state machine and A/B remain required after wedge closure. ([P-5], normative prerequisite) |
| **NR** | **Nano W4** | None. The retained W8 Nano interface is **52.29–52.79 ms/frame**, but that is only a W8 opportunity envelope and supplies no W4 speedup, correctness, or quality estimate. **Standing: blocked; no ranking authority.** ([P-A3], retained-precedent; [4D-R], descriptive/blocking constraint) | W4 is a new weights/runtime candidate: fresh conversion and calibration provenance; candidate-appropriate component correctness and quality/non-inferiority rather than borrowed W8 exactness; semantic/function/control and acoustic-quality gates; then the complete common runtime qualification on the new artifact. No W8 qualification result carries by inference. | Quantization and recurrent-state drift, control/function margin changes, semantic or voice-quality loss, new kernel/graph behavior, and artifact-specific failures. | No W4-specific benefit floor is preregistered. Before any timing, preregister the non-inferiority margins and a candidate-specific latency floor; after requalification the common absolute promotion contract applies. | **DEFERRED / INADMISSIBLE. NO ranking authority.** Unlock requires the full new-candidate requalification scope just stated, followed by candidate-specific timing. The W8 graph-path result cannot promote, demote, or ceiling W4. ([P-5], deferred status; [4D-R], blocking disposition) |
| **NR** | **Nano W8 kernel/fusion/launch-gap work** | Corrected ten-block descriptive split: content wall **53.560786 ms**, graph envelope **45.847206 ms**, wall-minus-envelope **7.713580 ms**; PAD wall **53.595332 ms**, graph envelope **45.579065 ms**, wall-minus-envelope **8.016267 ms**. The graph and residual fractions are **85.608% / 14.392%** for content and **85.053% / 14.947%** for PAD. These are overlapping impossible-zero-cost envelopes, not predicted or additive gains, and no tunable kernel family was identified. **Standing: descriptive / `INCOMPLETE`, then blocked.** ([4D], descriptive-admissible/`INCOMPLETE`) | First establish the released baseline; then preserve every retained causal-prefix token and the six dtype/shape/byte-sensitive production outputs in every block and profiling context. A future W8 candidate must pass exact output before timing can rank it; the captured-state gate remains necessary, not proof over arbitrary prompts. ([4D-R], descriptive/blocking constraint) | Fresh graph-enabled engines did not reproduce the prefix: every authoritative exact-output block diverged at call 39 even though the six selected outputs later reconverged. Graph-envelope time also does not reveal which child kernel is tunable. ([4D-R], descriptive/blocking constraint) | Exact-output validity is a hard floor, followed by the common promotion contract. No kernel-class minimum net saving is yet preregistered; it must be fixed before candidate timing. | **BLOCKED. NO ranking authority.** Exact unlock: make the released graph-enabled baseline reproducible across fresh engines and profiling contexts for the full causal prefix and six output digests; rerun the exact-output/decomposition gate; then use per-kernel profiling to identify a candidate family. Only a candidate that passes exact W8 output may be timed and ranked. ([4D-R], blocking constraint) |
| **NR** | **EarTTS-high elimination** | Older retained state: **10** high rows with mean EarTTS **20.4 ms** versus **10.86 ms** on other non-BOS rows, an affected-row difference of **9.54 ms** and only a symptom-class ceiling. In the capped current campaign, five valid blocks had **0 / 764** qualifying rows; the decision was `INSUFFICIENT`, not a null. **Standing: retained-precedent plus blocked.** ([P-A3], retained-precedent; [4C], `INSUFFICIENT` pivot standing) | Preserve acoustic-code distribution/sampling contract, complete audio, voice identity and quality, ASR/semantic gates, and all control boundaries. A high/low component result would constrain only the reconstructed state until broader qualification passes. | The mode is state/image/workload/period dependent and presently unavailable; post-selecting a “high” state or attributing its absence to transport would be invalid. Sampler/kernel changes can alter RNG order and audible output. | The common promotion contract applies. No candidate-specific reduction floor exists yet; preregister it only after the mode recurs and a fixable cause is identified. | **BLOCKED. NO current ranking authority.** Unlock by the admitted pivot: reconstruct the older `sustained-aged-26h-20260812` state difference, establish recurrence under a qualified capped design, retain a matched high/low PAD-tail pair, and identify a fixable kernel/sampler cause before implementing or timing elimination. ([4C-R], `INSUFFICIENT` pivot standing) |
| **NR** | **Wrapper small-launch-path graphing/fusion** | The only ceiling is the same wall-minus-graph envelope: **7.713580 ms content / 8.016267 ms PAD**. It includes wrapper/CPU delay, synchronization, graph-interpretation limits, and outside-graph GPU work; it is not pure launch gap or predicted gain. **Standing: descriptive / `INCOMPLETE`, then blocked.** ([4D], descriptive-admissible/`INCOMPLETE`) | Exact state/control behavior, stable descriptors, buffer identity, no stale replay, full causal-prefix/output gate, and end-to-end external cadence. Graph capture must cover all named branch and boundary shapes it can encounter or fail safely. | Stateful control branches and descriptor growth can replay stale buffers or wedge a graph path. The unreproducible fresh-engine baseline prevents distinguishing a candidate regression from baseline variance. | **Attributable recoverable gaps must be >= 1.5 ms/frame** before this class proceeds, followed by exact output and the common promotion contract. ([P-5], preregistered screening floor) | **BLOCKED; not screened out, but not rankable.** Unlock shares the graph-path reproducibility prerequisite above, then requires a new profile that attributes at least the **1.5 ms/frame** floor specifically to graphable/fusible wrapper launch gaps. The present 7.7–8.0 ms residual does not satisfy attribution. ([4D], descriptive/blocking disposition; [P-5], floor) |
| **NR** | **Selective EarTTS precision** | None. Steady EarTTS occupies **12.15–12.40 ms/frame** in retained A3, but zero-costing the whole stage is not a precision-specific recoverable ceiling. **Standing: retained opportunity only, then blocked.** ([P-A3], retained-precedent) | Module-by-module component non-inferiority for acoustic codes/logits as applicable, recurrent-state stability, waveform/voice quality and ASR, then full semantic/audio/runtime gates on the changed artifact and graph inventory. | Accumulated recurrent drift, sampling changes, audible artifacts, and precision-specific graph caches; a normal steady-state total does not identify bandwidth-bound modules. | Common promotion contract. No selective-precision benefit or non-inferiority floor is preregistered; both must be fixed before measurement. | **BEHIND THE PROFILED CANDIDATES / INADMISSIBLE. NO ranking authority.** Unlock requires per-module profiling that identifies a precision-sensitive bottleneck, an explicit non-inferiority protocol, and candidate-specific timing. ([P-5], ordering status) |
| **NR** | **Two-frame perception batching** | None. The retained Step 8 candidate was concurrent perception/EarTTS overlap, which regressed by **1.076 ms/logical frame**; serial two-frame perception batching is a distinct mechanism, so that negative result supplies no batching ceiling or ranking authority. **Standing: retained-negative-adjacent, then blocked.** ([P-5], retained negative control and ordering status) | From identical cloned cache, compare two sequential one-frame perception calls with one two-frame call and require exact cache and encoded outputs before timing; then preserve input order, recurrent-state semantics, control boundaries, end-to-end audio completeness, and device-level barge-in behavior. | Waiting for a second input frame can add input latency and worsen barge-in responsiveness; the new batch shape can alter cache semantics, output exactness, memory pressure, and graph inventory. | Common promotion contract. No batching-specific benefit floor is preregistered; fix it before candidate timing, after component exactness establishes that the mechanism is viable. | **BEHIND THE PROFILED CANDIDATES / INADMISSIBLE. NO ranking authority.** Unlock requires the cloned-cache component comparison, an attributable batching-amortization profile, a preregistered candidate-specific latency floor, and full candidate timing and runtime qualification. The concurrent-overlap stop neither promotes nor screens out batching. ([P-5], ordering status and retained negative-control boundary) |
| **NR** | **Residual/logging** | Server trace cost is **UNMEASURED**. The client Pipecat trace-on minus trace-off periods showed **+1.479 ms** server-step association with conditional 95% interval **[-0.758, +3.489] ms** and **+0.988 ms** residual association with conditional interval **[+0.058, +1.851] ms**. Fixed order, one period per arm, and a client-side toggle prevent a causal or ranking interpretation. **Standing: descriptive only.** ([4A], descriptive) | Identical protocol event sequence, response text, delivered PCM sample counts/hashes, no trace/queue drops or mode leak, and clean teardown; whole-block invalidation on any mismatch. The tested clock must include `send_step`/arrival work. | Removing observability can hide correctness failures; instrumentation can move the cost; order/warm-state and run-period effects dominate with one period per arm. | Existing preregistration: pursue only if the lower 95% bound for recoverable server `send_step`/arrival cost exceeds **0.5 ms per delivered-generation frame**, or residual-high occurrence/tail is materially reduced without correctness or cadence regression. ([4A-F], preregistered follow-up floor) | **NOT SELECTABLE. NO ranking authority.** Unlock requires a startup-frozen server `full`/`sink-null`/`off` experiment in one immutable qualification image with randomized/counterbalanced independently restarted blocks and run-block uncertainty. An empty residual-high stratum remains `INSUFFICIENT`. ([4A-F], prerequisite) |
| **NR** | **Thermal/DVFS (including host pressure)** | No recoverable ceiling. In one soak, raw frame associations changed materially under time adjustment and within-response views; the run neither establishes nor refutes a thermal, DVFS, CPU-pressure, or memory-pressure cause. The narrow observation is only that no recorded GPU clock collapse, pinned-core frequency change, or host memory/I/O pressure occurred. **Standing: descriptive only / blocked.** ([4B], descriptive) | Same artifact, workload, response membership, and clocks; supported and safe intervention; named-stage and external-cadence outcomes; no power/thermal safety, correctness, or scheduling regression. | Observational time/order confounding; telemetry acquisition and locking can perturb scheduling; locked clocks may increase power/temperature; unmeasured scheduler contention remains possible. | Common promotion contract. No causal-effect or benefit floor is preregistered; intervention arms, effect estimand, safe operating bounds, and minimum worthwhile effect must be fixed before a run. | **NOT SELECTABLE. NO ranking authority.** Unlock requires randomized run blocks for safe hot/cool, locked/normal clock, and controlled pressure/idle interventions, with causally preceding telemetry intervals and run block as the uncertainty unit. Observational repeats or detrending cannot unlock promotion. ([4B-F], required follow-up) |

The Nano kernel and wrapper ceilings overlap; they must never be added. The
pairing association is likewise not comparable to either single-position
envelope because it comes from an amortized schedule on a different retained
configuration and clock context. ([P-A4], retained-precedent; [4D],
descriptive/blocking disposition)

## Prerequisite registry

This registry is the fail-closed boundary for opening Step 5 selection:

1. **Kernel-class and wrapper graph work — graph-path reproducibility.** Across
   fresh graph-enabled released W8 engines and profiling contexts, reproduce
   the entire 687-call retained causal prefix and all six selected
   dtype/shape/byte-sensitive output digests in every block. Only then repeat
   the exact-output decomposition. Kernel work additionally needs an identified
   tunable kernel family; wrapper work needs specifically attributable gaps at
   or above 1.5 ms/frame. ([4D-R], descriptive/blocking constraint; [P-5],
   preregistered wrapper floor)
2. **Tail pairing — wedge closure.** Satisfy one of the three recovery closures
   in the pairing row. A timeout, session close, or stress-only claim is not
   closure. Then qualify the executable state machine and same-image
   counterbalanced arrival-clock A/B. ([P-5], normative prerequisite)
3. **Nano W4 — new-candidate requalification.** Produce a fresh, provenance-
   complete W4 artifact; preregister and pass candidate-appropriate component
   correctness plus quality/non-inferiority; then rerun semantic/function/
   control, acoustic/audio-completeness, idle/latency, browser/arrival,
   interruption/barge-in, sustained reliability, and recovery qualification on
   that exact artifact. W8 hashes, exact-output observations, and runtime
   qualification do not transfer. ([P-5], deferred status; [4D-R],
   descriptive/blocking disposition)
4. **EarTTS-high — state reconstruction.** Follow the older-state pivot, prove
   recurrence without extending or post-selecting the capped current campaign,
   and profile a qualified matched high/low state before naming an eliminable
   cause. ([4C-R], `INSUFFICIENT` pivot standing)
5. **Logging and thermal/DVFS — causal assignment.** Run the already specified
   server-side logging blocks and safe telemetry intervention blocks,
   respectively. The existing fixed-order A/B and observational soak cannot be
   upgraded by reinterpretation. ([4A-F], descriptive follow-up; [4B-F],
   descriptive follow-up)

## Step 5 recommendation — evidence versus judgment

### What the evidence licenses

In the Step 2 campaign's release-aware simulation, the maximum tested **560
ms** reserve still left **3 of 8** responses starving and added a mean **499.0
ms** first-delta-to-release delay; the artifact explicitly labels starvation
as simulation-only rather than browser underrun evidence. ([S2-R], measured
campaign evidence) This establishes that the tested fixed reserve grid did not
clean the retained long-response set; it does not prove that every larger or
adaptive reserve fails.

The evidence also licenses three narrower conclusions:

- pairing is the strongest historical performance lead, not an admissible
  implementation;
- Nano kernel and wrapper work have descriptive envelopes that permit further
  investigation; their shared reproducibility prerequisite remains open, and
  the evidence does not authorize ordering or implementation;
- the current EarTTS-high, residual/logging, and thermal records do not support
  promotion or deprioritization. Nano W4, selective precision, and two-frame
  perception batching remain outside the selection set.

### Recommended next actions (judgment)

Prefer continued structural work over a larger fixed reserve as the primary
solution; the reserve evidence above bounds the tested tradeoff but does not
rule out every larger or adaptive reserve.

1. **First, close the released W8 graph-path reproducibility gate.** This is the
   highest-information prerequisite because one correctness closure unlocks
   both Nano kernel work and wrapper launch-path attribution. Do not tune a
   kernel or add a graph while the baseline itself can diverge.
2. **In parallel as a design-only reliability track, choose whether pairing can
   eliminate the two-position packed call or obtain real worker recovery.** Do
   not implement the tail scheduler until one closure is concrete. If none is
   feasible, retire pairing from Step 5 despite its latency precedent.
3. **After W8 reproducibility passes, repeat the exact-output decomposition.**
   If attributable wrapper gaps clear 1.5 ms/frame, benchmark that lower-scope
   path. If per-kernel evidence identifies a tunable W8 family, benchmark it
   against exact outputs. Compare qualified candidates only on the common
   arrival/browser promotion contract.
4. **Reconstruct the older EarTTS-high state next, as forensics rather than a
   selected optimization.** It becomes a candidate only after recurrence and a
   fixable cause. Run the server logging and telemetry intervention studies only
   if those hypotheses remain operationally important; neither is on the
   critical path to a structural candidate today.
5. **Keep Nano W4, selective EarTTS precision, and two-frame perception
   batching deferred.** Reopen W4 only as a separately resourced full
   requalification project, selective precision only after profiling supplies
   a module-specific ceiling and a preregistered non-inferiority contract, and
   batching only after the cloned-cache component comparison establishes
   exactness and attributable amortization.

This ordering is judgment about information gain and engineering scope. The
admitted measurements do **not** prove that W8 kernel work will beat repaired
pairing, that wrapper graphing will clear its floor, or that any one candidate
will satisfy the structural and audible gates. Step 5 selection should be made
only after at least one row changes from `blocked`/`descriptive` to `measured`.

## Admitted source register

- **[P-T]** [Generation-frame latency plan — Targets](../docs/generation-frame-latency-plan.md#targets):
  normative/preregistered common promotion contract.
- **[P-5]** [Generation-frame latency plan — Step 5](../docs/generation-frame-latency-plan.md#step-5--structural-candidate-selected-by-the-step-4-table-not-before):
  normative candidate list, pairing prerequisites, wrapper floor, ordering,
  retained negative-control boundary, and deferred status.
- **[P-A3]** [Generation-frame latency plan — A3](../docs/generation-frame-latency-plan.md#a3--stage-timings-aligned-populations-server-clock-delivered-nonbos):
  retained-precedent stage measurements.
- **[P-A4]** [Generation-frame latency plan — A4](../docs/generation-frame-latency-plan.md#a4--pair-era-vs-sequential-server-clock-delivered):
  retained-precedent pair/sequential association.
- **[4A]** [Step 4a trace/logging report](step4-trace-overhead-ab-20260813.md#result):
  descriptive-only admitted by [the 4a/4b round-3 PASS](../docs/reviews/step4-ab-measurements-review-r3.md).
- **[4A-F]** [Step 4a required server-side follow-up](step4-trace-overhead-ab-20260813.md#required-server-side-follow-up-experiment):
  preregistered logging prerequisite and benefit floor; the retained A/B itself
  remains descriptive.
- **[4B]** [Step 4b telemetry report](step4-telemetry-regression-20260813.md#result):
  descriptive-only admitted by [the 4a/4b round-3 PASS](../docs/reviews/step4-ab-measurements-review-r3.md).
- **[4B-F]** [Step 4b required telemetry follow-up](step4-telemetry-regression-20260813.md#required-telemetry-follow-up):
  required causal design; current evidence remains descriptive.
- **[4C]** [Step 4c result](step4-eartts-incidence-20260813.md#results):
  `INSUFFICIENT` with pivot standing, admitted by [the round-3 PASS](../docs/reviews/step4c-incidence-review-r3.md).
- **[4C-R]** [Step 4c Nsight decision and pivot](step4-eartts-incidence-20260813.md#nsight-decision-and-pivot):
  blocked-state reconstruction prerequisite.
- **[4D]** [Step 4d corrected ten-block disposition](../docs/reviews/tool-race-fix-and-step4d-gate-r3.md#b-three-run-non-browser-confirmation-and-step-4d-disposition):
  descriptive-admissible/`INCOMPLETE`; explicitly inadmissible for ranking.
- **[4D-R]** [Step 4d exact-output constraint](step4-nano-decomposition-20260813.md#exact-output-component-benchmark-and-constraint):
  descriptive/blocking graph-path prerequisite.
- **[S2-R]** [Step 2 campaign reserve sweep](step2-live/session-20260813T083303Z/arrival-analysis.json):
  measured campaign replay, explicitly simulation-only for starvation rather
  than browser-underrun evidence; admitted in the plan's Step 2 implementation
  log.
