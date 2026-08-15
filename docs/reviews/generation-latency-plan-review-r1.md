# Adversarial review: generation-frame latency plan, r1

Review date: 2026-08-12. Scope was offline/read-only inspection of the retained
traces, qualification artifacts, repository code, experiment branches, and
retained experiment reports. I did not inspect the running container because
the review brief's explicit rule forbids touching containers or the model
WebSocket. Consequently, container-only assertions are not treated as verified.

## 1. Findings

Verdict meanings: **CONFIRMED-WRONG** means the retained primary evidence
contradicts the claim; **UNSUPPORTED** means the evidence cannot establish it;
**MISLEADING** means a quoted number can be reproduced only with a materially
different population or interpretation; **RISK** identifies an unclosed
implementation or qualification hazard; **OK** means I reproduced the claim as
stated.

| # | Verdict | Claim under review | Recomputed evidence and consequence |
|---:|---|---|---|
| 1 | **CONFIRMED-WRONG** | Generation misses 80 ms on every generation frame and has done so for as long as retained traces exist (`docs/generation-frame-latency-plan.md:9-21`). | The sequential preflight does have 34/34 delivered frames over 80 ms, but retained sessions do not. `2a0fa2ae…/events.jsonl` has 101/134 delivered frames over 80 ms (75.4%); `209fe177…` has 52/55 (94.5%); the closest PAD-pair baseline `f110f063…` has 13/32 (40.6%); and Aug-10 session `b6ac34bb…` has 1/38 (2.6%). The sweep artifact itself records generation means of 71.1 and 75.0 ms at `scratchpad/drift-and-deficit.txt:260,276`, contradicting Appendix A1's “82–88 ms in every session” (`plan:202-217`). The structural regression is real on the current sequential image, but the historical/100% formulation is false. |
| 2 | **MISLEADING** | Fresh idle cost is “~59–71 ms,” while real microphone idle is consistently ~71 ms (`plan:13-17,34-45`). | In `2a0fa2ae…`, phase=`idle` is n=352, mean 58.663 ms only because it includes 68 background function-cycle pseudo-steps averaging 3.694 ms. Excluding those gives n=284, mean 71.824 ms. In `209fe177…`, idle is n=189, mean 71.781 ms. The later workload-composition explanation is supported, but the problem table retains the invalid 59 ms endpoint and “settlement” still has no executable definition. |
| 3 | **OK** | Aggregate sequential-preflight p95 is 73.99 ms (`plan:27-30`). | Recomputed over all 2,729 `voicechat.metrics` rows: mean 71.131, p50 71.544, p95 73.9848, p99 84.6381, max 156.923 ms. This correctly demonstrates that a silence-dominated aggregate p95 hides the response phase. |
| 4 | **MISLEADING** | The preflight split is 37 generation / 2,692 other at means 86.0 / 70.9, and Step 0 will classify generation as “inside response.created → response.done with nonzero delivered audio” (`plan:27-30,90-102`). | The 37/2,692 split is reproducible only as `response_id != null` versus everything else: 37 rows at 86.0386 ms and 2,692 at 70.9262 ms. Three of those 37 are internal-drain rows with `audio_delivered=false` (`nano-sequential-preflight…/events.jsonl:145-147`). The delivered population is 34 rows at 86.1684 ms; all non-delivered rows are n=2,695 at 70.9414 ms. A “nonzero delivered audio” classifier therefore cannot reproduce the stated 37/2,692 gate. A consistent post-first-audible classifier produces yet another result: n=34 at 84.1701 ms. |
| 5 | **CONFIRMED-WRONG** | PAD-pair was ineligible during generation and accelerated the wrong phase (`plan:54-58`). | `pad_pair_should_draft()` returns true when the *previous effective token is PAD* under the conditional policy (`runtime_optimizations.py:444-446`), and the serving path uses that result (`runtime_optimizations.py:1014-1045`). In the closest retained baseline, generation frame 63 enters PAD; frame 64 is then buffered at 45.912 ms and frame 65 executes the packed pair at 89.111 ms, with the same alternation continuing through the audio tail (`f110f063…/events.jsonl:319-439`). Of 32 delivered response frames, 25 are PAD: 12 buffered PAD frames average 44.999 ms and 13 accepted/transition PAD frames average 95.952 ms. PAD-pair was explicitly eligible in, and heavily exercised by, the generated post-text tail. |
| 6 | **MISLEADING** | Both retirement arms had the same ~85 ms generation cost; baseline p99 94.8 versus 84.6 proves PAD-pair was irrelevant or worse (`plan:54-58`). | The quoted 94.787/84.638 p99 values are whole-workload distributions with different burst shapes, not generation-phase costs. In the closest retained pair baseline, delivered generation is n=32, mean 75.116 ms; in the sequential preflight it is n=34, mean 86.168 ms—a measured 11.052 ms/frame regression after retirement. Excluding each BOS gives 72.699 versus 84.024 ms. Pairing lowered average tail cost and allowed fast frames to repay slow ones; its unresolved wedge risk can still justify retirement, but “same cost/wrong phase” cannot. |
| 7 | **UNSUPPORTED** | The deployed prebuffer is definitely 160 ms with no environment override (`plan:47-50`). | The repository default is 160 ms and an environment override is accepted (`llm.py:130,168-178`); the repo contains no production override. Neither the model traces nor the new sustained artifact records the Pipecat process environment. Confirming the live value requires container/process inspection, which this review was forbidden to perform. State this as “repo default,” not deployed fact. |
| 8 | **MISLEADING** | Step-9 fallback and host contention are excluded (`plan:51-53`). | Every inspected metric row in the retained preflight and aged trace reports fallback count zero, so graph fallback is reasonably excluded for those runs. The cited `/tmp/voicechat-step9-fallbacks.jsonl` is not currently retained on the host, however, and “load 0.32” has no timestamped source in the trace/qualification artifacts. One host-load snapshot would not exclude GPU clocks, thermal throttling, LPDDR contention, I/O, or scheduler stalls anyway. |
| 9 | **UNSUPPORTED** | A matched A/B establishes that no uptime drift exists; a fresh arm was unnecessary (`plan:34-45,104-110,219-245`). | This was not a matched A/B: the aged arm is a 220 s, five-response typed/mixed sustained run, while the “fresh reference” is a different one-response preflight, and no fresh arm of the same workload was run (`plan:221-234`). The table also mixes classifiers. Reapplying the aged “after first audible until response.done” rule yields aged n=184, mean 82.7863, but fresh n=34, mean 84.1701—not the table's n=37, 86.04. Aged response means range 79.57–85.70 ms, versus only one fresh response, so “statistically indistinguishable” has neither an independence unit nor a reported test/interval. The data strongly refute the original 59→72 idle-drift reading, but do not prove zero uptime effect. |
| 10 | **OK** | The two highlighted single-session traces do not show gradual 10–15 ms aging. | In `209fe177…`, pre-response idle is n=23, mean 70.514 and post-response idle n=166, mean 71.956; early versus late generation means are 85.130 versus 85.522 ms. Its high generation mode is episodic: 13 frames at ≥88 ms average 92.827 wrapper ms, driven mainly by EarTTS (20.36 versus 11.71 ms on low frames) and occasionally residual (for example frame 34), not a linear ramp. In `2a0fa2ae…`, the apparent 58.7 ms idle value is the function-cycle mixture described in finding 2. This evidence should replace the categorical A2 conclusion with “no drift demonstrated; previous large estimate was confounded.” |
| 11 | **OK** | Cyclic GC is not a persuasive normal-session drift explanation (`plan:104-110`, superseding the earlier draft hypothesis). | `RealtimeCyclicGCGuard.finish()` re-enables GC when the final active stream ends (`runtime_optimizations.py:354-383`), and the normal `is_last` path calls fixed-stage finalization (`server.py:3861-3868`). There is no retained heap/RSS/generation-counter evidence for GC pressure. However, direct `reset_session()` paths in abort handling (`server.py:4054-4084`) do not visibly call this finalizer, so an abnormal-cleanup audit of guard membership is cheaper and more decisive than asserting there is no possible GC issue. |
| 12 | **CONFIRMED-WRONG** | `Σ max(0, step_ms - 80)` is the cumulative buffer deficit and maps to audible glitches (`plan:74-83,213-217`). | A playback queue recovers on fast frames. The correct recurrence is `debt_i = max(0, debt_(i-1) + interval_i - 80)` and the required reserve is its maximum, using audio-output arrival intervals. In the PAD-pair response, summing positive `server_step_ms` excess gives 279.867 ms, but fast paired frames make total step excess −156.277 ms; using consecutive model-output intervals, peak queue debt is only 4.53 ms. The plan's metric would reject exactly the bursty schedule that maintains realtime. Conversely, the sequential preflight has 209.727 ms server-step excess but 293.916 ms of client-observed audio-delta interval debt. `server_step_ms` stops before turn-state/event construction (`server.py:3936`), while audio conversion and send begin later (`server.py:5604-5634`), so it is not the playback clock. |
| 13 | **CONFIRMED-WRONG** | A 450 ms prebuffer covers 8 s at 5.5 ms/frame (`plan:112-121`). | Eight seconds contains 100 80-ms frames, so the plan's own formula gives **550 ms**, not 450 ms. Ten seconds is 125 frames and does give 687.5 ms (~700). The empirical values also do not justify one constant rate: 312/38 = 8.21 ms/frame and 307/22 = 13.95 ms/frame, while 279/55 = 5.07. BOS and output/send overhead matter. Size from the maximum-prefix distribution of actual `response.output_audio.delta` arrival intervals, clustered by response, not from the average of positive server-step excesses. |
| 14 | **MISLEADING** | The current 160 ms reserve absorbs ~2–3 s, and every added prebuffer millisecond adds the same amount to TTFA (`plan:19-23,114-125`). | `160/(5–6)` is indeed 26.7–32 frames (2.13–2.56 s) under the simplifying constant-rate model. The retained sequential preflight is worse at the client: 33 audio-delta intervals average 88.9065 ms and accumulate 293.916 ms debt in 2.934 s. Pipecat also calls `process_ttfa_metrics()` **before** buffering (`llm.py:1217-1242`), so its existing TTFA metric will not move at all when the buffer changes. Browser audible onset should rise by approximately the threshold delta only for responses long enough to fill it; short responses flush immediately on `response.done` (`llm.py:1265-1288`) and pay less. The ~1.4 s baseline is a valid point sample, not a distribution (`voice-to-voice-latency.md:315-326`). |
| 15 | **RISK** | A long browser response can decide the larger-buffer gate without interacting with response watchdogs (`plan:60-70,123-125`). | Every one of the five new sustained responses ended with `no_text_since_bos_watchdog` (`sustained-aged-26h/report.json:39-89`), as did the sequential preflight (`events.jsonl:150`). A ≥8 s fixture can therefore be truncated/early-flushed by the 30-frame (2.4 s) no-text watchdog before it reaches the intended buffer stress. Use a fixture that continuously emits text, or fix/explicitly gate the watchdog first. Otherwise “glitch-free” and buffer coverage are not decidable. |
| 16 | **RISK** | “Barge-in discard unchanged” is sufficiently covered by the existing buffer behavior (`plan:123-125`). | Speech start clears only the service's *unreleased* list (`llm.py:1165-1178`). Once the threshold is reached, the entire reserve is pushed downstream and the list is empty (`llm.py:1251-1258`). A larger threshold makes more audio cancellable before release but also releases a larger burst into transport. The gate must measure post-barge audio at the browser/output device, not only the internal list/unit test. |
| 17 | **CONFIRMED-WRONG** | Retained stage attribution covers only settlement, so Step 3 needs a new diagnostic run before optimization decisions (`plan:127-137,236-246`). | Both the sequential preflight and the new aged artifact already carry `model_stage_timings_ms` on relevant rows. Recomputed over 34 delivered preflight frames: perception 13.048, Nano interface 52.282, EarTTS 14.279, codec call 0.703, residual 4.596, wrapper total 84.908 ms; excluding BOS (n=33), EarTTS is 12.150 and wrapper 82.769. In the aged artifact's 188 delivered non-BOS rows: 12.516 / 52.790 / 12.404 / 0.605 / 3.747 / 82.061 ms. These are enough to falsify the appendix's “unattributed 15–20 ms” inference. A larger current-image run may improve confidence, but existing data must be mined first, and host intervals are not additive causal GPU costs because queued work can be charged to the next interval (`voice-to-voice-latency.md:32-42`). |
| 18 | **MISLEADING** | Step 8 should be revived because measured results were not retained in-repo (`plan:139-149`). | The result is retained in the adjacent source artifact tree: `.../reports/step8-overlap-ab-20260807.md:1-18,131-135` records a defended stop. Candidate logical-frame mean was 64.034 versus 62.957 ms, a **+1.076 ms regression**, with p95 88.646 versus 88.336. On overlap-only frames, perception inflated 11.854→20.165 ms and EarTTS 12.521→23.261 ms (`report:39-64`). The exactness gate passed 93/93 cache/output comparisons (`report:93-107`). Rerunning may be justified only by a concrete changed-stack hypothesis, not by claiming the result is unknown. |
| 19 | **RISK** | Perception(N+1) can simply leave the serial live path and should improve generation p95 by ≥8 ms (`plan:139-149`). | The historical implementation cloned `cache_last_channel`, `cache_last_time`, and length, ran `PerceptionCacheManager.step()` on a dedicated stream, joined after EarTTS, and exact-compared the recurrent outputs (`765a024:patch_step8_perception_overlap.py:43-60,109-215`); this establishes old-stack dependency legality. It does not establish current reentrancy with the BF16/captured-graph override (`runtime_optimizations.py:386-428`). More fundamentally, the branch supplied `next_frame_hint` only for pre-synthesized typed audio and typed silence; the server diff routes those hints at typed synthesis, not the microphone path (`765a024:server.py`, `next_frame_hint` hunk). The serial WebSocket loop does not possess microphone frame N+1 while computing N. Full voice qualification could pass while overlap is inactive. Implementing live overlap first requires a bounded input-ingestion queue, cache ownership proof, and a live launch/consume coverage gate; the prior GB10 contention result still predicts no speedup. |
| 20 | **CONFIRMED-WRONG** | Step 5 can coalesce four CUDA synchronizations for an expected single-digit-ms win (`plan:151-159`). | The retained report found **seven**, not four, synchronizations and concluded none was independently removable; the 10.4–10.7 ms perception wait was prerequisite GPU work (`.../reports/step5-host-barriers-20260807.md:5-15,46-76`). The implemented safe changes (ten RNNT reads packed, constants reused, PAD detokenization shortcut) regressed run-mean wall time by 0.747 ms and pooled p95 by 3.962 ms (`report:126-146`). Step 5 is already a negative experiment. It should not follow Step 4 as an expected win; only isolated subchange experiments with a new mechanism would merit work. |
| 21 | **RISK** | CFG scale 0 can be accepted with WER 0.0 and “no audible-quality regression” (`plan:161-171`). | WER on a small retained fixture is not a prosody/naturalness gate, and “audible” has no blinded evaluator, threshold, or sample count. The measured steady non-BOS EarTTS interval is about 12.2–12.4 ms, so it is a plausible lever, but scale 0 needs deterministic acoustic/code comparisons plus a blinded preference/MUSHRA-style gate across speakers, phonetic stress cases, long tails, and seams. Preregister a minimum latency win and a non-inferiority bound before changing checkpoint-intended guidance. |
| 22 | **MISLEADING** | Only ~60% of analyzed generation is PAD, and tail-restricted pairing is a future successor aimed at the phase the old pair missed (`plan:173-183`). | In the sequential preflight, 27/34 delivered frames (79.4%) have PAD agent tokens; in the aged sustained artifact, 144/193 delivered rows (74.6%) are PAD. More importantly, the old conditional scheduler already paired the PAD tail (findings 5–6). A safer successor may still be valuable, but it is a repair/requalification of a historically effective tail optimizer—not a new targeting insight—and its claimed ~15 ms saving needs a queue-clock A/B, not a per-frame subtraction. |
| 23 | **RISK** | Existing sampling and gates are sufficient once phase-stratified (`plan:87-98,129-149`). | Frames within one response are autocorrelated and BOS is a distinct, very expensive event. The aged “n=184” represents only five responses; the fresh “n=37” only one. Report response-clustered intervals or bootstrap by response/session, split BOS/leading silence/steady speech/PAD tail, counterbalance arms in one resident process where exact RNG permits, and log temperature/clocks/power. A raw frame count of ≥200 is not an independence guarantee on GB10. |

### Bottom line

The core diagnosis—current sequential generated audio does not sustain the
80 ms media clock—is supported. The proposed explanation and remedy ordering
are not. The draft uses inconsistent “generation” populations, an invalid
queue-deficit equation, a non-matched “A/B,” and two optimization branches
whose retained results are already negative. The most damaging factual error
is the PAD-pair claim: historical tail pairing demonstrably reduced average
generation cost by about 11 ms/frame in the closest retained comparison,
although its correctness/wedge risk remains a legitimate blocker.

## 2. Ordering and gate critique

The plan should be reordered as follows.

1. **Freeze the measurement contract offline.** Define separate, non-conflated
   populations: response-compute (`response_id`), delivered PCM
   (`audio_delivered`), audible PCM (an explicit dBFS threshold), boundary phase,
   BOS/leading silence, steady speech, PAD tail, settlement, and function-cycle.
   Use `response.output_audio.delta` receipt intervals for playout, and compute
   maximum queue debt with the recovering recurrence. Emit row membership and a
   source hash so aggregate reproduction is auditable. Do not use the impossible
   37/nonzero-audio Step-0 oracle.

2. **Mine evidence that already exists before any new run or build.** The
   preflight plus aged artifact already provide 227 delivered stage-timed rows;
   the PAD-pair trace answers eligibility; and the Step-8/Step-5 reports answer
   the old branches' performance questions. This should delete, not merely
   postpone, the current Step 4 and Step 5 unless a specific changed condition
   predicts reversal.

3. **Make the long-response fixture valid.** The current watchdog closes every
   sustained response. Either use a prompt/model fixture with text activity
   throughout the ≥8–10 s response or resolve the no-text watchdog first. Add an
   objective browser-side underrun counter/audio timestamp capture and a real
   output-device barge-in tail measurement.

4. **If uptime remains a question, run an actual matched experiment.** Use the
   same image, prompt/audio sequence, pacing, trace mode, and classifier before
   and after the age condition; include multiple responses per arm, thermal
   soak, counterbalanced order where possible, and GPU clock/temperature/power
   telemetry. A fresh restart is not warranted by current evidence alone, but
   neither is the categorical “no uptime component” conclusion.

5. **Attribute the observed current tail, not hypothetical barriers.** Start
   with the recurrent EarTTS/residual high mode and actual output-arrival gap.
   The retained data already show perception and Nano are nearly flat while
   selected aged frames add roughly 8–11 ms in EarTTS or residual. Use CUDA
   events/kernel traces on a small fixed-state fixture before a whole-image A/B.

6. **Size any stopgap buffer through trace replay.** Sweep reserve values against
   per-response output-arrival sequences, report underrun probability and
   browser-onset cost, and select an explicit product percentile/maximum response
   duration. Keep it separate from the structural p95 objective. A fixed 450 ms
   value is neither derived correctly nor guaranteed for eight seconds.

7. **Only then choose a structural candidate.** The old Step-8 and Step-5 paths
   are negative controls. A corrected, fail-closed tail-pair design may have the
   strongest measured precedent, but it must first close the packed-call wedge
   and prove response-boundary rollback. Otherwise prioritize isolated EarTTS,
   Nano-kernel, launch-overhead, or scheduling mechanisms based on the new
   profiling evidence.

The current gates also need these changes to become decidable:

- Replace “byte-exactly reproduce rounded aggregates” with exact membership,
  source hash, and a documented percentile convention.
- Measure browser audible onset, because Pipecat's TTFA instrumentation is
  upstream of the prebuffer and cannot validate the configured delta.
- Replace subjective “glitch-free” and “no audible regression” with timestamped
  underrun/seam counts and a blinded audio-quality non-inferiority protocol.
- Treat stage timings as attribution hints, not summable savings; CUDA work can
  be charged to a following host interval.
- Require Step-8 live microphone launch/consume coverage before correctness or
  performance results count. A green general voice suite is not proof that the
  overlap path ran.
- Check the single-client health/lock immediately before any live probe and
  fail closed if occupied. Budget cold-start/build time and the multiple
  response/session repetitions, not merely timed fixture duration.
- Record GB10 clocks, temperature, power state, memory pressure, and arm order.
  The prior Step-8 result directly demonstrated shared-resource contention even
  when physical read volume was unchanged.

## 3. Creative suggestions, ranked by expected information per effort

These are additions, not endorsements for immediate production. Each begins
with a cheap falsification measurement.

| Rank | Approach | Mechanism and expected win | Main risk | Cheapest useful measurement |
|---:|---|---|---|---|
| 1 | **Offline playout replay + adaptive reserve** | Replay actual audio-delta timestamps through the recovering queue equation; select reserve from response-length prediction or early observed production rate rather than one worst-case constant. Expected win is the same underrun protection with materially less median onset delay than a 550–700 ms fixed buffer. | A bad length/rate prediction can underrun late; short-response completion changes the release rule. | Feed every retained response's delta timestamps into a 0–1,000 ms reserve sweep, then simulate barge-in at every frame. Report underrun, onset, and discarded-audio distributions without touching the model. |
| 2 | **Trace/logging overhead A/B** | High-frequency JSON trace construction, PCM statistics, and per-frame logging may inflate wrapper residual or its tail. Buffering trace writes or reducing hot-path log formatting could recover ~1–5 ms or remove residual outliers. | Disabling observability can hide correctness failures and changes timing enough to confound comparison. | On a deterministic component fixture, compare trace fully on, trace sink buffered, and trace off; retain identical model outputs and measure both wrapper total and external output cadence. Do not change production logging first. |
| 3 | **EarTTS/MoG-sampler spike forensics and fusion** | `209fe177…` shows high frames with EarTTS near 20–23 ms versus ~12 ms normally while perception/Nano stay flat. Fuse/vectorize mixture sampling, codebook selection, or small post-forward kernels; graph a stable sampler shape. Expected p95 win is up to 8–11 ms on the high mode, with little mean win if spikes are rare. | RNG order, acoustic-code distribution, and voice quality can change even when text WER is unchanged. | Capture CUDA events and one Nsight Systems trace around EarTTS on saved low/high input states; correlate time with token/control type and kernel sequence before changing code. |
| 4 | **Thermal/DVFS and unified-memory telemetry** | Establish whether response-to-response modes track GPU clock, temperature, power, CPU frequency, or memory pressure; if so, use a steady thermal soak or supported clock/power policy. Expected direct win is uncertain, but it can prevent false optimization conclusions. | Sampling can perturb scheduling; locked clocks can increase power/temperature and may be unsupported. | Poll low-rate clocks/temp/power and `/proc` pressure during an existing qualification, align to frame timestamps, and regress stage time against telemetry. No model mutation is needed. |
| 5 | **Executor affinity and priority experiment** | Pin the model-call executor, WebSocket event loop, trace writer, and CPU codec to noncompeting cores; optionally test scheduler priority within safe limits. Expected benefit is primarily reduced wrapper/output-arrival p95, perhaps 1–3 ms. | Starving network/codec work can worsen end-to-end cadence despite a faster wrapper number. | Record per-thread CPU migrations and run-queue delay with `perf sched`/eBPF on a short fixed fixture, then test one reversible affinity map and compare external audio-delta intervals. |
| 6 | **Selective EarTTS precision below global W4** | Keep recurrent/norm and sensitive output heads at qualified precision while trying BF16/FP16 on bandwidth-heavy projections or the sampler. Expected win is a few milliseconds if those kernels are bandwidth-bound. | Accumulated recurrent drift, changed sampling, or audible artifacts; graph caches may need separate captures. | Change one module at a time in an offline component harness; compare logits/acoustic-code divergence, waveform metrics, ASR, and module CUDA time before any live image. |
| 7 | **W8 batch-1 Marlin/kernel tuning** | The plan jumps from current W8 to W4, but batch-1 decode may admit better Marlin tile/split-K choices, fused dequant/activation, or persistent small-shape kernels without recalibration. Expected win is roughly 1–5 ms if kernel occupancy/launch overhead is the limiter. | Hardware-specific tuning, compile/cache complexity, and regressions on non-steady positions. | Profile one captured PAD and one content position with Nsight Compute; benchmark only alternative kernel configs against exact W8 outputs before integration. |
| 8 | **Two-frame perception batching instead of concurrency** | When two input frames are already queued, process a two-frame Conformer chunk and consume embeddings sequentially, amortizing frontend and launch overhead without contending concurrently with EarTTS. Expected win is modest (1–4 ms/frame) if the cache manager supports the shape. | Additional input latency, altered cache semantics, a new CUDA-graph shape, and worse barge-in responsiveness. | In the component harness, compare two sequential one-frame calls with one two-frame call from identical cloned cache; require exact cache/output or quantify drift before timing. |
| 9 | **Low-buffer catch-up playback** | When reserve falls, apply bounded pitch-preserving time compression (for example, render 80 ms of generated PCM in 76–78 ms) until the reserve recovers. This converts a 2.5–5% production deficit into a small temporary speaking-rate change without model changes. | Prosody artifacts, altered conversational timing, and poor behavior on music/non-speech or at boundaries. | Apply WSOLA/phase-vocoder variants offline to retained WAVs, run ASR and objective pitch/formant checks, conduct a small blinded preference test, and replay queue depth. |
| 10 | **Graph/fuse the wrapper's small launch path** | Capture or fuse stable RNNT/control/sampler operations around the already-graphed large models, reducing Python and launch gaps rather than deleting necessary synchronizations. Expected win is sub-ms to a few ms. | Stateful control branches and graph descriptor explosion; incorrect capture can wedge or replay stale buffers. | Use one Nsight Systems trace to total CPU gaps and kernel-launch overhead between named stages. Proceed only if recoverable gaps exceed a preregistered floor (for example 1.5 ms/frame). |
| 11 | **Broader Nano(N+1)/EarTTS(N) pipeline feasibility probe** | Decouple model ingestion so the next audio/Nano position can become available while the current acoustic tail runs, potentially hiding more than perception alone. | The old Step-8 experiment predicts severe GB10 shared-resource contention; live N+1 availability, state ordering, and function/control boundaries make this high risk. | Before scheduler work, run fixed-state Nano and EarTTS kernels concurrently on separate streams and measure combined critical path, clocks, and exact outputs. Stop immediately if either stage inflates enough to erase overlap. |

## 4. What I would do first Monday morning

I would not restart the stack or revive either old branch. I would first fix the
offline analyzer so it emits exact frame membership and maximum-prefix debt from
audio-delta arrival times, then run it over the preflight, the aged artifact,
the PAD-pair baseline, and the Aug-10/11 corpus. In the same morning I would
attach the retained Step-8 and Step-5 negative reports to the plan and mine the
already-present stage timings, splitting BOS, steady content, PAD tail, and the
EarTTS/residual spike mode. Those zero-model-cost steps should determine whether
the first live experiment is a trace-overhead check, an EarTTS spike profile, or
a carefully repaired tail-pair candidate—and they prevent spending a day
rerunning experiments whose answers are already retained.
