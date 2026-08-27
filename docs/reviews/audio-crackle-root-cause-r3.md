# Audio crackle: adversarial review of corrected underflow telemetry (R3)

Date: 2026-08-19 UTC  
Code reviewed: `10ab675` and parent `0adc1a8`  
Evidence reviewed: `reports/audio-crackle/` and the installed Pipecat source in
`~/.cache/nemotron-voicechat/venv/`  
Constraint: analysis only; no source, test, dependency, or commit changes

## Executive conclusion

The corrected result is useful but narrower than “the server-side playout path
is exonerated.” Across two local 95-second runs, the counter saw no queue-empty
run that both:

1. was later closed by another RawAudioTrack chunk;
2. remained under the same **shared control epoch**; and
3. lasted no more than the five-second backstop.

That is strong evidence against ordinary mid-response RawAudioTrack starvation
in those two local replays. It is not evidence about unresolved terminal gaps,
boundary-straddling gaps, post-RawTrack encoding/network/browser behavior, or
the actual remote session in which the crackle was heard.

The main remaining measurement flaw is structural. `PlayoutEpochTagger` does
not tag queued audio. It increments one mutable value when a TTS control frame
passes, while previously queued audio continues draining asynchronously. A
stop, start, or interruption can therefore relabel audio already buffered in
BaseOutput or RawAudioTrack. This creates a conservative false-negative window
around every boundary. The window is unlikely to explain crackle distributed
through the middle of long responses, but it can hide an audible tail dropout,
hard splice, or underflow while old audio remains queued across a control
boundary.

The Pipecat stop-flush mechanism is confirmed: if BaseOutput has a residual,
it appends zero padding to one full output chunk, enqueues that chunk, and then
enqueues the `TTSStoppedFrame`. The upstream tagger advances the shared epoch
before BaseOutput performs either enqueue. The padded chunk therefore reaches
the track before BaseOutput processes the stop frame but after the epoch has
already changed. The retained evidence does **not** independently substantiate
the quoted 550–670 ms or 9–10 ms timings; those values appear only in comments,
tests, and the README, not in retained per-event logs.

The 200 ms negative control proves the live hook detects long same-epoch
starvation. It does not close the most relevant sensitivity gap for crackle:
one or two 10 ms zero ticks. The cheapest closure is the same write-delay probe
at approximately 15–25 ms, with injection timestamps and epoch-before/after
logged. The existing probe's comment assumes 40 ms of RawTrack slack, but its
hook runs only after the preceding batch has drained; the real deadline is
approximately the next 10 ms tick. That is why a 200 ms sleep produced 19–20,
not about 16, ticks.

I would describe the result as: **local, resumed, same-epoch RawTrack underflow
is not observed and is now a lower-probability explanation.** The real remote
path still needs synchronized PCM and WebRTC evidence before network, jitter
buffer, Opus, generated content, or device playback can be ranked decisively.

## Evidence and test result

The two clean summaries report:

| Run | Positive-depth observations | In-speech ticks/runs | Epoch advances | Stale-epoch runs | Backstop runs |
|---|---:|---:|---:|---:|---:|
| `underflow-bot2` | 4,180 | 0 / 0 | 15 | 8 | 0 |
| `underflow-bot3` | 2,743 | 0 / 0 | 15 | 9 | 0 |

Together they cover 6,923 real 10 ms track observations, or 69.23 seconds of
track audio. The equal 1/2/3/4 histograms are expected from four-chunk batch
submission; they verify orderly batch draining but do not expose BaseOutput's
upstream queue depth. The zero underflow count, not the sawtooth itself, is the
relevant evidence.

The negative control reports 508 ticks in 26 runs for 26 nominal injections,
with every run 19–20 ticks. That tight signature makes accidental substitution
by natural events implausible. It validates the installed wrapper, counter,
epoch wiring, and summary path for long same-epoch stalls.

All 29 current `tests/pipecat/test_playout_transport.py` tests pass. Those tests
pin direct counter behavior, including one-tick runs, but do not exercise a
one-tick fault through live RawAudioTrack pacing, BaseOutput handoff, the epoch
tagger, aiortc, and the browser together.

## A. Remaining false-negative classes

### 1. Shared epoch relabels queued audio at boundaries

This is the important unacknowledged class.

The epoch is a mutable integer (`playout_transport.py:81-98`). The tagger
increments it when a boundary frame passes (`127-131`). The recv wrapper later
samples its current value (`328-339`). No epoch is stored on an
`OutputAudioRawFrame`, BaseOutput batch, RawAudioTrack queue entry, or returned
10 ms frame.

Meanwhile, Pipecat has two asynchronous stores downstream of the tagger:

- BaseOutput's `_audio_queue`, which can hold complete output frames;
- RawAudioTrack's `_chunk_queue`, which holds the currently submitted batch.

Consequently, “same epoch” means “the shared control value happened not to
change between the empty dequeue and the resuming dequeue.” It does not prove
that the chunks on either side belong to the same or different audio stream.

A real false-negative sequence is:

```text
old-stream chunk drains under epoch N
RawAudioTrack emits one or more zeros; pending epoch = N
TTSStopped/Interruption/next TTSStarted advances shared epoch to N+1
an old-stream chunk already queued downstream resumes, now observed as N+1
counter discards the real gap as stale epoch
```

The inverse timing also matters. If the stop advances the shared epoch before
the preceding real audio drains, that old audio, the later zero run, and the
padded flush can all appear under the new epoch. The current fence is therefore
not guaranteed by content ordering alone; its behavior still depends on when
the control frame overtakes physical playout.

The robust version of this measurement would capture an epoch/content ID when
each output batch is enqueued and carry it in a sidecar entry for every 10 ms
RawTrack chunk. A pending gap should retain the ID of the last consumed real
chunk and compare it with the ID of the actual resuming chunk. That avoids
retroactively relabeling queued audio when a control frame passes.

Likelihood for the reported crackle: **low to moderate, depending on where the
artifact occurs.** It is unlikely to explain repeated crackle throughout the
middle of long responses because no epoch changes there. It remains credible
for end-of-response crackle, back-to-back responses, function/tool transitions
that create rapid boundaries, and interruption splices.

### 2. Any gap that never resumes

Pending zeros are reported as tail and never counted. This includes the
accepted normal response drain, but also a real starvation event that removes
the final audible phoneme and is followed by stop, disconnect, or failure. A
listener can hear that as a cutoff or click even though no later chunk closes
the gap.

Likelihood for frequent mid-speech crackle: **low.** Likelihood for a reported
end click or clipped last syllable: **material.** A PCM/content-aware capture is
needed to distinguish the two.

### 3. Any gap spanning a stream boundary

This is deliberately accepted, but broader than interruption alone. It covers
TTS stop, the next TTS start, and interruption; an interruption with an open
TTS stream can advance twice because the service first emits
`TTSStoppedFrame` and then forwards `InterruptionFrame`.

The policy correctly excludes idle/prebuffer silence. It also excludes real
physical discontinuities when queued audio crosses one of those control
boundaries. RawAudioTrack's currently submitted batch is not cleared by
BaseOutput interruption handling, so up to the active batch can survive the
logical cancellation.

Likelihood: **localized but plausible.** Hard interruption splices have no
fade and are a known independent click mechanism. They do not explain crackle
in uninterrupted speech.

### 4. Same-epoch gaps longer than five seconds

The backstop discards them. A five-second audio loss would sound like an outage,
not crackle, and both clean runs report zero backstop discards. This is a real
false-negative class but not a credible explanation of the symptom.

### 5. Track replacement after the held counter is attached

The disconnect handler logs the counter captured from the output track at
initial connection. If a future reconnect/renegotiation path replaces that
RawAudioTrack without reattaching and updating the held reference, the wrapper
will auto-create a counter on the new track while the session summary still
logs the old one. The installed transport creates/replaces its track in
`_handle_client_connected`; the observed single-connection replay does not
exercise replacement.

Likelihood in the reported one-session crackle: **low.** It is a coverage risk
for future renegotiation/reconnect tests.

### 6. Failures after RawAudioTrack

These are not errors in the gate, but they are outside its measurement point:
Opus encoding, RTP scheduling/transmission, packet loss/reordering, receiver
jitter buffering and concealment, browser rate adjustment, audio rendering,
and the physical device. A zero result says nothing about them.

## B. Independent verification of the stop-frame flush

The mechanism and ordering are confirmed directly from the installed source.

1. `handle_audio_frame` resamples incoming audio, appends it to
   `_audio_buffer`, and moves only complete `audio_chunk_size` frames to
   `_audio_queue` (`base_output.py:580-606`). A partial residual can remain.
2. On TTS stop, `handle_tts_stopped` calls
   `_enqueue_flushed_audio_buffer()` before it queues the stop frame
   (`644-657`).
3. The flush returns if there is no residual. Otherwise it constructs
   `bytes(_audio_buffer) + bytes(missing_length)`, clears the residual, and
   enqueues the full frame (`707-729`). The appended portion is unequivocally
   digital zero.
4. `_audio_task_handler` consumes that queue in FIFO order, handles and writes
   each audio frame, and only later handles the queued TTS stop (`887-914`).
5. In this application, `PlayoutEpochTagger` sits immediately before
   `transport.output()` (`demo.py:317-329`) and increments before forwarding
   the stop (`playout_transport.py:127-131`). Thus the shared epoch changes
   before BaseOutput enqueues either the padded frame or its stop frame.

Two wording corrections matter:

- The frame is **padded with silence**, not necessarily pure silence. It can
  contain up to almost one full chunk of residual source audio followed by
  zeros. Whether the residual was inaudible in the observed cases is not
  retained in the summaries.
- At the output transport, the padded frame is queued **ahead of** the stop
  frame and physically reaches RawAudioTrack before BaseOutput processes the
  stop. It is only the upstream tagger's shared epoch that has already moved.

The source imposes no 550–670 ms delay. That observed delay is plausible: real
audio can drain while response finalization is still pending, and only the
later TTS stop causes the residual flush. But `reports/audio-crackle/` retains
only final counter summaries. Discarded stale-epoch runs have no timestamps or
tick counts, and no raw log contains the asserted response-done deltas.
Therefore:

- silence padding and FIFO ordering: **CONFIRMED**;
- epoch advances before the flush is enqueued: **CONFIRMED**;
- four observed instances at 550–670 ms and within 9–10 ms of response.done:
  **NOT INDEPENDENTLY VERIFIABLE FROM THE RETAINED EVIDENCE**.

## C. Adequacy of the negative control

The 200 ms control licenses three claims:

- the monkeypatched write path was active;
- a long, resumed, same-epoch starvation becomes a recorded run;
- tick duration and injection duration agree to about one 10 ms quantum.

It does not license “zero at all click-relevant durations.” A one-tick
auto-silence splice can be audible when it cuts a nonzero waveform, so the
missing 10–20 ms live control matters materially.

The probe's stated slack calculation is also incorrect. BaseOutput's audio
task awaits `SmallWebRTCOutputTransport.write_audio_frame`; that method awaits
the future attached to the submitted batch's final RawTrack chunk. The patched
function is next entered only after the previous batch has been fully consumed.
At that point the next RawTrack deadline is about 10 ms away; there are not 40
ms of old chunks still queued. The observed 19–20 ticks for a 200 ms sleep are
consistent with:

```text
expected ticks ~= floor((stall_ms - time_to_next_tick_ms) / 10) + phase term
               ~= 19 or 20 for a 200 ms stall
```

### Cheapest closing control

Reuse `stall_injector.py`; do not add browser instrumentation.

1. Delay selected `write_audio_frame` calls by 15 ms and 25 ms in separate
   replay runs. Because the previous batch has drained, these should produce
   approximately one and two zero ticks depending on scheduler phase.
2. Log every injection's monotonic start/end, write ordinal, and epoch before
   and after the sleep. Exclude injections that cross an epoch boundary from
   the same-epoch sensitivity denominator.
3. Require one counter run aligned to each stable-epoch injection and a 1–2
   tick distribution consistent with measured sleep-to-next-deadline.
4. Retain the per-run and per-injection records, not only the terminal summary.

A 10 ms sleep exactly on a 10 ms deadline is phase-fragile; 15/25 ms gives a
cleaner live test while still exercising the click-relevant regime. A direct
unit test of `observe(1), observe(0), observe(1)` remains useful but is not a
substitute for this integration control.

Until that control passes, the correct interpretation of the clean replay is
“no detected same-epoch stall at the already demonstrated 200 ms scale,” plus
code-level confidence—not an empirical bound of zero at 10 ms.

## D. What the local replay does and does not exonerate

I do not agree with the unqualified statement that the server-side playout path
is exonerated.

What is strongly disfavored:

- ordinary, resumed, same-epoch RawAudioTrack queue starvation in the two local
  replay sessions;
- the R1 theory that those particular generated responses necessarily create
  repeated mid-stream RawTrack zeros.

What remains open on the server:

- boundary-adjacent and unresolved gaps hidden by the conservative gate;
- resampler or generated-PCM artifacts that do not empty RawTrack;
- aiortc/Opus/RTP behavior after RawAudioTrack;
- differences between the two replay outputs and the actual reported session;
- scheduling/backpressure behavior with the real remote peer.

Headless Chromium on loopback still runs Opus and a receiver jitter buffer. It
does not remove them; it merely exercises them under a low-jitter local path.
The server counter cannot tell whether either produced an artifact. The replay
also captured no decoded browser PCM and had no listener, so an audible problem
could have occurred without appearing in the server summary.

Loading the page through an ngrok URL does not by itself prove that RTP media
traverses ngrok. WebRTC commonly uses the selected ICE candidate pair after
HTTP signaling. The next experiment must record local/remote candidates,
protocol, relay status, and the nominated selected pair before describing the
route as “the ngrok media path.”

The leading candidates should now shift toward:

1. remote network loss/jitter and receiver concealment/rate adjustment;
2. browser or operating-system audio playout;
3. Opus encode/decode behavior;
4. generated or resampled PCM content;
5. boundary-local server splices missed by the epoch gate.

This ranking is provisional. R1's waveform checks reduce the prior for a
periodic generated/resampler boundary click, but they did not examine the
actual browser-decoded waveform from the reported remote session.

## E. Concrete synchronized remote-path experiment

### Required artifacts

Every run should produce one immutable directory containing:

```text
manifest.json                 versions, flags, clocks, selected ICE route
model-output-22050.pcm         existing model trace reference
server-track-24000.pcm         exact 10 ms RawAudioTrack output, including zeros
server-track-events.jsonl      PTS, monotonic time, queue/epoch/content metadata
browser-decoded-f32.pcm        AudioWorklet output and sample rate
browser-blocks.jsonl           currentFrame/currentTime/performance clock mapping
webrtc-internals.json          inbound RTP, codec, jitter buffer, ICE candidates
listener-markers.jsonl         user key/button timestamps and optional notes
capture-health.json            drops, overruns, CPU, event-loop/callback lateness
sha256.txt                     hashes of every artifact
```

The model trace distinguishes generated content from transport changes. The
server-track PCM distinguishes pre-network audio from downstream damage. The
browser PCM is the decoded waveform the browser actually rendered into Web
Audio; an optional operating-system loopback capture can further distinguish
browser PCM from device/backend faults.

### Server capture without hot-path I/O

At the RawAudioTrack boundary, record the returned 10 ms PCM frame, its PTS,
scheduled due time, actual monotonic call/return times, pre-pop queue depth,
real-versus-auto-silence choice, and queue-side content epoch.

Do not synchronously write or JSON-encode in `recv()`. Copy the 480-byte mono
24 kHz frame and a fixed-size binary metadata record into a bounded in-memory
ring; a background task writes batches. Report ring drops explicitly. At this
rate PCM is only about 48 kB/s.

For attribution rather than merely reproducing the current gate, capture the
epoch/content ID at `write_audio_frame`/`add_audio_bytes` enqueue time and keep a
parallel ID for each 10 ms queued chunk. Do not sample only the mutable shared
epoch at dequeue time. This both closes the R3 false-negative class and labels
the TTS-stop padded chunk correctly.

### Browser capture without wrapping constructors

Do not wrap `RTCPeerConnection`, subclass `AudioContext`, or replace prebuilt
UI objects.

After the UI has connected and attached its remote media, obtain the existing
remote audio track from the media element, clone that track, and feed only the
clone into a separate, ordinary `AudioContext` and `AudioWorkletNode`. Leave the
UI's original playback element and graph untouched. The worklet should batch
at least 250 ms of Float32 samples into transferable buffers before posting to
the page; per-quantum Playwright callbacks or console logging are prohibited.
Record `sampleRate`, `currentFrame`, `currentTime`, `performance.timeOrigin`,
and `AudioContext.getOutputTimestamp()` where available. Track worklet callback
gaps and dropped batches.

Collect WebRTC statistics through Chromium's native `chrome://webrtc-internals`
dump rather than monkeypatching the peer connection. Sample/export cumulative
inbound RTP fields including packets/bytes received, packets lost, jitter,
jitter-buffer delay and emitted count, concealed samples/events,
silent-concealed samples, and inserted/removed samples for acceleration or
deceleration. Retain codec and selected candidate-pair records.

If the exact Chromium build cannot export the needed native dump, the fallback
is a small environment-gated UI diagnostic that exposes the already-created PC
by reference; it must not wrap its methods or constructor. Poll `getStats()` no
faster than once per second and batch results locally.

### Clock and sample alignment

Use media samples, not wall-clock subtraction, as the primary alignment:

1. RawTrack PTS and browser `currentFrame` each define exact sample timelines.
2. Resample server PCM from 24 to 48 kHz with a persistent reference resampler.
3. Cross-correlate non-silent windows, then fit a piecewise-linear mapping from
   server sample index to browser sample index. Piecewise fitting reveals
   inserted/removed/concealed intervals instead of smearing them into one
   offset.
4. Use lightweight data-channel ping/pong timestamps only for coarse event and
   listener-marker alignment; network asymmetry makes them less precise than
   waveform correlation.

The listener should press one dedicated key when a crack is heard. The marker
is a search hint, not ground truth, because human response latency is variable.
The retained PCM must support blind offline confirmation around every marker.

### Attribution rules

For every detected discontinuity or listener marker:

| Server track PCM | Browser PCM | Stats delta | Attribution |
|---|---|---|---|
| Artifact present | Same artifact present | any | Generated/resampler/server-track origin; compare model PCM to split it further |
| Clean | Missing/repeated/concealed region | loss/concealment increase | Network loss/jitter plus receiver concealment |
| Clean | Time insertion/removal | jitter-buffer or acceleration fields change | Browser jitter-buffer/rate adjustment |
| Clean | Artifact present | no network/jitter delta | Opus decode/browser Web Audio candidate; reproduce locally and inspect codec path |
| Clean | Clean | user/OS capture has artifact | Browser output backend, OS mixer, or device |

An all-zero server interval is not automatically a fault; compare it with the
content ID and model PCM to distinguish intended source silence, stop padding,
and RawTrack auto-silence.

### Prove the capture is non-perturbing

Before trusting a remote human run, perform randomized capture-off/capture-on
A/A pairs on a frozen output source, first over loopback and then over the real
selected remote path. Include a server-only mode to isolate browser overhead.
Predeclare acceptance criteria:

- no change in connect/bot-ready/session completion success;
- zero server ring drops and zero browser worklet batch drops;
- no new RawTrack zeros in the clean local replay;
- capture-on/off equivalence for RawTrack scheduled lateness and host event-loop
  lag within a small predeclared margin;
- no material shift in inbound jitter/concealment or decoded PCM duration;
- bounded CPU and memory overhead with no AudioWorklet deadline misses.

Use enough randomized pairs to estimate a confidence interval; one successful
instrumented run does not demonstrate non-interference. Frozen generated output
is preferable because the existing two nominally identical microphone replays
produced 41.80 versus 27.43 seconds of track audio.

Finally, validate attribution with two positive controls:

1. inject one server-side 10 ms zero inside a known non-silent segment and
   require it at the aligned position in both server and browser PCM;
2. apply a small, logged `tc netem` loss/jitter burst to the actual selected
   media route and require clean server PCM plus browser concealment/stats
   evidence.

Do not proceed to the human ngrok-path assessment until both positive controls
and the capture-on/off non-interference gate pass.

## Bottom line

The current zero is credible evidence against mid-response RawTrack starvation
in two clean local runs, but it is not a universal zero and does not yet license
“server exonerated.” The next cheapest action is the 15/25 ms live stall
control. The decisive action remains a non-perturbing, synchronized server and
browser PCM capture on the actual selected remote media route, with native
WebRTC statistics and predeclared A/A interference checks.
