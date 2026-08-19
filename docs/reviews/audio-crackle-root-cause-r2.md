# Audio crackle: independent root-cause assessment (R2)

Date: 2026-08-19 UTC  
Scope: direct `RawAudioTrack` telemetry from the human session
`803c1d4a-1a4d-4276-86a6-d934c7da7ed5`, the chunk=1/chunk=4 replay arms,
and the earlier full playout trace  
Constraint: analysis and measurement only; no source, test, dependency, or
environment changes

## Executive conclusion

The new counter confirms that `RawAudioTrack` really takes its auto-silence
branch many times. It does **not** establish that all 722 ticks happened during
audible bot speech. Its `in_speech` classifier has a deterministic boundary
error: after the last real chunk of every response, it continues labeling
empty ticks as in-speech for 300 ms. That contributes approximately 29–30
ticks and one run per completed or interrupted response.

For the two A/B arms, four response boundaries occurred before the fifth
response was cut off. Thus approximately 116–120 of the 233/215 measured ticks
are known response-tail false positives. Exact replay of each arm's actual
model arrivals, through the warm stateful SOXR output-size pattern, configured
Pipecat chunking, and ten track phases, predicts another 28–29 ticks for the
chunk=1 arm and only 7–8 for chunk=4. The remaining unexplained counts are:

```text
chunk=1: 233 - (116..120) - (28..29) = 84..89 ticks
chunk=4: 215 - (116..120) - ( 7.. 8) = 87..92 ticks
```

That residual is strikingly similar between arms. It is evidence of a delay
downstream of the model-only replay (or another classifier limitation), not
evidence that chunk=4 is better. A depth-zero observation immediately before
the real `recv()` queue test is a real 10 ms zero frame in the sender media
timeline; a later audio arrival cannot erase it. Therefore “the counter counts
events rather than displacement” does not explain the residual. Each counted
tick is exactly 10 ms of inserted sender-side silence. What remains unproven is
whether the inserted interval replaces audible source content and whether it
survives WebRTC/browser playout as a heard crack.

The most likely unmeasured region is Pipecat's post-service output path:
BaseOutput resampling/queueing, its consumption-gated batch handoff to
`RawAudioTrack`, and aiortc track scheduling/catch-up. The earlier full trace
shows that model-to-WebSocket receipt and receipt-to-downstream-push delays are
small, but its last timestamp is before BaseOutput processes the frame. The
current evidence cannot identify the exact operation responsible for the
approximately 0.85–0.9 seconds of residual zeros in each A/B arm.

The chunk-size A/B conclusion is **not sound**. Identical microphone input did
not produce identical output: the arms generated 273 versus 336 frames, and
their timing debt was materially different. The 21-tick difference predicted
from those unequal traces already exceeds the 18-tick measured difference in
the direction observed. Keep the default chunk=4 setting, but do not claim a
quality win from this n=1 comparison.

## Verdicts on the R2 claims and candidates

| Claim | Verdict | Evidence / correction |
|---|---|---|
| Direct telemetry confirms many small auto-silence ticks. | **CONFIRMED.** | The wrapper samples the queue after the track's pacing wait and immediately before the original `recv()` tests it. A zero observation takes the real auto-silence path unless the duplicate pacing check unexpectedly suspends again. |
| `in_speech_ticks` is the number that degrades speech. | **REFUTED.** | “In speech” means only “within 300 ms of the last nonempty dequeue.” Every response tail is therefore counted even after its audio is finished. Digital source silence is also not distinguished from audible PCM. |
| The 4–6x excess is merely event count rather than net displacement. | **REFUTED.** | Every zero dequeue emits 10 ms of PCM zero and advances PTS by 10 ms. If real audio arrives 1 ms later, it plays behind that zero: that is precisely a permanent 10 ms insertion/displacement in the sender media timeline. |
| Empty windows between 80 ms bursts can be counted even if no starvation occurs. | **REFUTED for the current counter.** | The queue may be empty between due times without consequence. The counter samples only when a 10 ms frame is due. Track phase changes the exact A/B replay by only one tick per arm, far below the discrepancy. |
| Model timing alone predicts 56/35 ticks from `sum(max(0, gap-80))`. | **REFUTED as a queue model.** | That sum discards every early arrival and ignores initial stock, recovery, discrete SOXR output, chunk exposure, tick phase, and batch handoff. Signed A/B receipt drift was only +120.0 ms for chunk=1 and −203.3 ms for chunk=4. Exact queue replay predicts 28–29 and 7–8 stream-period ticks. |
| Event-loop delay from service receipt to transport may explain the excess. | **PARTLY REFUTED upstream; UNPROVEN downstream.** | In the earlier 802-delta trace, model-to-client receipt was p50 1.294 ms, p95 1.841 ms, max 2.224 ms; post-buffer downstream-push delay for ordinals 3+ was p50 0.089 ms, p95 0.139 ms, max 1.363 ms. BaseOutput, track handoff, and aiortc scheduling occur after that marker and remain unmeasured. |
| Response boundaries and interruptions inflate the total. | **CONFIRMED, materially.** | The 300 ms grace contributes about 29–30 ticks and one run after every last real chunk, including an interrupted response. It explains about half the A/B ticks and 377–390 of the human session's 722 ticks. |
| `chunks=4` beats `chunks=1`. | **UNPROVEN; the reported comparison is confounded.** | Exact input-timing replay predicts a 21-tick advantage for the chunk=4 session before chunk size itself is credited, versus an observed 18-tick advantage. Normalized residuals overlap: 84–89 versus 87–92 ticks. |
| 722 ticks / 94 runs is compatible with intermittent crackle. | **COMPATIBLE, not causal proof.** | After removing 13 terminal grace windows, at most 332–345 ticks remain within streaming windows, about 3.32–3.45 s. That is intermittent rather than continuous, but some remaining ticks can still overlap intentional digital silence. No listener-aligned PCM exists. |

## Audit of what the counter measures

`PlayoutUnderflowCounter.observe()` has two independent behaviors:

1. `depth_chunks == 0` records the same queue-empty condition that the wrapped
   `RawAudioTrack.recv()` will use to emit a 10 ms zero frame.
2. It calls that zero “in speech” if no more than 0.3 seconds have elapsed since
   any positive-depth observation.

The first behavior is a useful direct measurement. The second is not a speech
state machine. It does not know about response IDs, TTS start/stop, interruption,
the PCM's audibility, BaseOutput's upstream queue, or its residual byte buffer.

For an isolated response, let the final real track chunk be consumed at time
`t`. Empty dequeues at approximately `t+10, t+20, ..., t+290/t+300 ms` all
satisfy the grace test. The next response begins seconds later in all three
observed sessions, so this creates one artificial in-speech run of about 29–30
ticks per response. This is not a statistical theory: it follows directly from
the code and the 10 ms track clock.

The logs corroborate it. Both A/B arms contain four bot stops before their
fifth response is interrupted by disconnect. Their last reported in-speech
timestamps occur at the end of response 4's terminal window, before response 5
starts. The human log has 13 bot starts and 13 bot stops, including two hard
interruptions, so it has 13 such windows.

The counter also cannot tell whether a real zero replaces speech. Every model
response begins with three digital-silence frames, and completed responses have
roughly 7–11 trailing silence frames in these sessions. A zero inserted inside
those intervals changes timing but is acoustically indistinguishable from the
source PCM at that moment.

## Correct event/displacement model

At the track boundary, use a discrete tick queue, not a sum of positive model
step overruns. Let:

- `Q_k` be real 10 ms chunks present immediately before due tick `k`;
- `A_k` be chunks made available between ticks `k-1` and `k`;
- `Z_k = 1` when `Q_k == 0`, otherwise zero.

Ignoring notation for the producer's consumption-gated batch handoff, the
track recursion is:

```text
emit real chunk if Q_k > 0, otherwise emit a 10 ms zero
Q_(k+1) = max(0, Q_k - 1) + A_(k+1)
inserted sender-side silence = 10 * sum(Z_k) ms
```

This makes three points explicit:

- early arrivals replenish stock and must not be discarded;
- an empty interval matters only if it contains a due 10 ms tick;
- once a zero tick is emitted, later audio resumes one media slot later. The
  zero is not undone.

The quoted deficit metric,
`sum(max(0, interarrival_ms - 80))`, is neither signed drift nor a simulation of
this recursion. For the A/B traces:

| Arm | Frames | Gross positive gap excess | Signed gap drift | Exact stream-period replay |
|---|---:|---:|---:|---:|
| chunk=1 | 273 | 560.4 ms | +120.0 ms | 28–29 ticks |
| chunk=4 | 336 | 354.4 ms | −203.3 ms | 7–8 ticks |

The exact replay used the session's recorded PCM and model `monotonic_s`
arrivals, a single warm `SOXRStreamAudioResampler` across responses, the
configured one- or four-chunk BaseOutput exposure unit, release on model delta
2, and ten phases across the 10 ms track period. It stops stream-period
classification at the final model arrival and treats the later empty interval
as a terminal-classification problem. Track phase changed each total by only
one tick.

The replay deliberately assumes instantaneous transfer of already available
BaseOutput batches to the track. That is the important unmeasured assumption.
In reality `write_audio_frame()` awaits the future attached to the final 10 ms
chunk of each submitted batch. Only after the track pops that final chunk can
the audio task be scheduled to submit the next batch. When the RTP sender or
event loop is late relative to the track's creation-anchored wall clock,
`recv()` can run catch-up calls without a pacing sleep. This can expose an
empty RawTrack queue even when complete audio remains upstream. It is a
plausible source of the residual, but the current summary-only telemetry cannot
prove it.

## Quantitative reconciliation

### Controlled A/B arms

| Component | chunk=1 | chunk=4 | Status |
|---|---:|---:|---|
| Measured `in_speech_ticks` | 233 | 215 | Direct RawTrack observation, contaminated classification |
| Four 0.3 s terminal windows | 116–120 | 116–120 | Proven classifier artifact |
| Model-arrival/SOXR/tick replay | 28–29 | 7–8 | Deterministic prediction from each arm's unequal output trace |
| Residual | **84–89** | **87–92** | Real zero observations not attributed by available timestamps |

The two residual ranges overlap almost exactly. There is no remaining measured
advantage for chunk=4 after accounting for the different generated timelines.
The residual could include post-push event-loop/track scheduling, unobserved
A/B-specific WebSocket delay, and zeros that occur over intentional source
silence. Per-tick records are required to divide it further.

The run counts support “longer existing episodes” more than “hundreds of new
independent failures.” The chunk=1 arrival replay predicts roughly 11–14
stream-period runs; chunk=4 predicts 6–7. Adding four terminal runs gives
15–18 and 10–11, versus 19 and 14 observed. Most of the run-count difference is
already explained, while observed tick duration is substantially longer.

### Human session

The human trace contains 881 delivered frames, 70.48 seconds of model audio.
Of those, 217 frames (17.36 seconds) are below −90 dBFS; 39 are the three-frame
leading silence on 13 responses and 120 are terminal silence frames. The
remaining source content is at most 53.12 seconds above the threshold.

| Component | Ticks / runs | Interpretation |
|---|---:|---|
| Raw counter | 722 ticks / 94 runs | Includes terminal grace and intentional source silence |
| 13 terminal grace windows | 377–390 ticks / 13 runs | Known false “in-speech” classification |
| Exact model-arrival replay | 125–131 ticks / roughly 64–82 runs across phase choices | Expected queue-empty due ticks before final arrivals |
| Residual after both | 201–220 ticks | Unattributed duration; not necessarily new runs |

Removing only the known terminal windows leaves 332–345 ticks, or 3.32–3.45
seconds, inside the counter's streaming windows. That is 4.7–4.9% of all
generated-audio duration and an upper bound of about 6.3–6.5% relative to the
above-threshold source duration. The 81 nonterminal runs average approximately
41–43 ms if all remaining ticks are assigned to them.

During roughly 53 seconds of above-threshold source audio, 81 runs would be one
episode about every 0.66 seconds. That can reasonably be described as frequent
intermittent crackle or breakup rather than continuous distortion. It is also
large enough that, if every run replaced speech, a listener might call it
choppy. Because the counter has no PCM-content labels and no browser capture,
the wording of the report cannot adjudicate the remaining fraction.

## What the earlier full trace rules out

`/tmp/playout-traces-baseline/playout-0003.jsonl` contains 802 deltas across nine
responses. It can be paired ordinal-for-ordinal with model session
`79ee9a96-b531-4cd3-8745-56ce8bff7da2` because the Linux monotonic clock is
shared and the response lengths match exactly.

Measured model-step publication to Pipecat receipt:

```text
p50 1.294 ms, p95 1.841 ms, p99 2.010 ms, max 2.224 ms
within-response first-to-last hop change: -1.145 to +0.621 ms
```

Measured Pipecat receipt to `client_downstream_push` for ordinals 3 and later:

```text
p50 0.089 ms, p95 0.139 ms, p99 0.726 ms, max 1.363 ms
```

This strongly disfavors the model-to-service WebSocket and service buffer as a
large systematic silence source in that earlier run. It does **not** measure
the A/B runs, and `client_downstream_push` is recorded before BaseOutput
resamples, chunks, queues, writes, and waits for RawTrack consumption. The
highest-value server instrumentation belongs after that marker.

## Why the queue-depth histograms do not prove margin

The histograms are mechanically determined by the batch handshake:

- with chunk=1, BaseOutput submits one 10 ms chunk and waits for its future, so
  RawTrack can only be observed at depth 1 before a real pop;
- with chunk=4, it submits four chunks, so successful drains naturally produce
  the repeating `4,3,2,1` sequence.

BaseOutput may have additional complete frames in its own audio queue while
RawTrack shows only 1–4. Therefore `{1: 7038}` does not mean the whole output
path had only 10 ms of audio, and the clean even 1/2/3/4 histogram does not show
that chunk=4 maintained 40 ms of end-to-end safety depth. Neither histogram
records the upstream queue or byte residual.

Chunk=1 also requires a producer/consumer future handoff every 10 ms instead of
every 40 ms, so it is plausible that it is more scheduling-sensitive. The
current A/B does not isolate that effect.

## Highest-value next measurement

The proposed browser PCM plus WebRTC stats is directionally right, with one
essential addition: capture the exact server track PCM and per-tick state on
the same clock. The decisive single experiment is a synchronized, controlled
replay with all three views:

1. At every `RawAudioTrack.recv()` due time, record timestamp, scheduled-clock
   lateness, real/zero choice, raw queue depth, response ID/state, source audio
   ordinal, and whether the logical source interval is above an audibility
   threshold. Also record BaseOutput full-frame queue depth, residual samples,
   and last-batch-future-to-next-submit delay.
2. Record the exact 10 ms PCM frames emitted by RawTrack, including its inserted
   zeros.
3. In Chromium, record post-WebRTC PCM with an AudioWorklet and collect inbound
   RTP statistics, especially packets lost, jitter, jitter-buffer delay,
   concealed samples/events, silent-concealed samples, and inserted/removed
   samples for rate adjustment.

Browser PCM is the final arbiter of audible damage. `getStats()` alone will not
flag source zeros because they are valid RTP audio. Server PCM alone cannot
separate a source splice from network loss, jitter-buffer concealment, browser
rate adjustment, or device output. Aligning both ends lets each observed crack
be classified rather than inferred from aggregate counts.

Before interpreting a new session summary, replace the 300 ms heuristic in the
measurement design with explicit lifecycle/content labels. This is a telemetry
requirement, not a production fix recommendation. A useful breakdown is:

- zero tick after logical response audio is exhausted: expected idle;
- zero tick while more source audio is outstanding: source/output starvation;
- zero tick replacing source PCM below the audibility threshold: timing damage,
  probably inaudible;
- zero tick replacing above-threshold PCM: candidate audible dropout.

## A sound chunk-size experiment

The microphone WAV is not a sufficient control because model output is
nondeterministic. A sound comparison should replay the **same prerecorded
output PCM with the same arrival timestamps and response boundaries** into the
post-service output path for both settings. Use randomized paired AB/BA order,
enough repeated pairs to estimate a confidence interval, and the lifecycle-
gated per-tick metric above. Primary endpoints should be above-threshold zero
duration and browser-observed concealment/dropout duration per emitted speech
second, not raw session totals.

If a full live-model test is also desired, retain every output trace and use a
paired model-based adjustment, but do not substitute that for the frozen-output
test. The current one-run arms differ in response count timing, length, and
generation debt enough to reverse the apparent treatment result.

## Ranked conclusions and actions

1. **Fix the measurement contract before tuning production.** The current
   counter is valid for total RawTrack zero emissions but not for “in-speech”
   damage. Explicit response/content state and per-event timestamps are the
   highest-confidence gain.
2. **Run the synchronized server/browser PCM experiment.** This determines
   whether residual source zeros coincide with heard crackle and whether
   WebRTC/browser concealment adds another mechanism.
3. **Keep chunk=4 as the default pending a controlled repeat.** Chunk=1 has more
   handoffs and showed no normalized advantage. The current data do not prove
   that chunk=4 improves quality, only that changing to 1 is unjustified.
4. **Investigate the post-push handoff if residual zeros align with audible
   content.** Batch-future scheduling delay and track clock lateness are the
   leading hypotheses. They are not yet proven root causes.
5. **Treat extra prebuffer as a mitigation experiment, not resolution of this
   discrepancy.** It can absorb arrival debt identified in R1, but it will not
   remove terminal misclassification and may not cover track scheduling
   catch-up. Measure its browser-visible effect against latency before adopting
   it.

## Reproducibility and limitations

Throwaway scripts under `/tmp` parsed the three model traces, paired the earlier
model/playout trace, streamed recorded PCM through the installed warm SOXR
implementation, reproduced BaseOutput complete-chunk exposure, and swept ten
track phases. No file under `src/` or `tests/` was modified, no dependency was
changed, and no commit was created.

The A/B runs lack per-push and per-zero timestamps, so the residual cannot be
localized precisely. The human run has only the terminal summary. No browser
PCM or WebRTC stats were captured. Accordingly, this R2 establishes a broken
classification, refutes the current A/B inference, and localizes the remaining
unknown to the post-model/output path; it does not claim a final audible-crackle
root cause.
