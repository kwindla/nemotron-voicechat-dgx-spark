# Voice-to-voice latency investigation

Status: measured and root-caused; onset-preroll, tool-free turn-boundary, and
idle-PAD EarTTS remediations are qualified, but the current runtime does not yet
meet a 400 ms last-speech-to-first-speech target.

## Measurement boundary

The browser probe measures PCM RMS on one `performance.now()` clock. Its start
is the last callback above -40 dBFS on the local microphone track, before
WebRTC. Its end is the first callback above -40 dBFS on the remote audio track,
before the physical DAC. The callback window is 256 samples at 48 kHz (about
5.33 ms).

Pipecat's TTFB is narrower: commit enqueue in
`NemotronVoicechatLLMService` to the first non-empty
`response.output_audio.delta`. TTFA extends that interval to the first model
audio frame whose PCM exceeds the silence threshold. Neither metric is
model-only inference latency.

Smart Turn v3.2 itself takes about 28--35 ms on the retained microphone
fixture. The browser's last-speech-to-`user-stopped-speaking` interval also
contains Silero VAD (`stop_secs=0.2`), media chunking, pipeline scheduling, and
the RTVI return event.

`session.updated` is now a model-ready barrier. Pipecat admits microphone media
only after that server acknowledgement, its initial context, and its audio
sender are ready. Capture racing startup is discarded before resampling; it is
never buffered and replayed into the ordered input stream. A turn beginning
before readiness is discarded in full through its matching commit.

The next diagnostic image reports per-position stage timing without adding a
CUDA synchronization point. Each value is therefore a host-observed interval
between synchronization boundaries that already existed in the qualified
wrapper; GPU work queued by one stage can be charged to the following interval.
`nano_interface_ms` names the interface deliberately because the PAD-pair path
can return a buffered token without advancing Nano. The persistent CPU codec is
one frame pipelined: its call on frame F waits for F-1 while submitting F. The
wrapper emits the named-stage sum, the one computed residual, and the wrapper
total and fails if their full-precision accounting is inconsistent. EarTTS BOS
reset timing is a nested abort/prefill wall-time breakdown and is not added a
second time to the top-level sum.

Before collecting this diagnostic, the analysis classes are fixed as settlement
zero-input positions, fused or legacy BOS, post-BOS leading-silence positions,
and steady speech. Results will be reported separately for tool-free direct,
tool-advertised direct, and browser-through-Pipecat turns. The initial
optimization ranking is also fixed: first remove a proven discardable idle
EarTTS/codec or BOS-reset cost; next address a dominant Nano/blank-fence cost
only with a newly qualified boundary design; then address generated EarTTS
onset; finally isolate downstream Pipecat/WebRTC delivery. Transmitted-silence
suppression is not considered an audible-latency improvement because it cannot
make EarTTS generate the first speech frame earlier.

## Results

All figures below are milliseconds. The traces are retained under
`~/.local/state/nemotron-voicechat/traces/model/`.

### Stage-attribution run

The non-perturbing stage-timing image is
`sha256:2e71bf2894d8f02b96c4511fb12255b6cc9d70a9ae73c2c67927e76cc672e494`.
Its browser-through-Pipecat, tool-advertised run passed the microphone, typed
math, typed recall, audio-carrier, and clean-close checks. The measured
browser voice-to-voice latency was 1470.7 ms; Pipecat TTFB was 732.341 ms and
TTFA was 982.772 ms, including 250.431 ms of generated leading silence.

For the correlated microphone commit, the server received the commit with
four RNNT blanks already observed. Six settlement positions reached the
ten-blank fence in 517.914 ms, followed by a separate 213.137 ms BOS position.
The six model calls themselves totalled 417.902 ms. Their wrapper accounting
was:

| Settlement stage | Six-position total |
|---|---:|
| Perception | 70.708 |
| Nano interface | 186.495 |
| EarTTS | 96.836 |
| One-frame-pipelined codec calls | 45.012 |
| Wrapper residual | 12.008 |
| Wrapper total | 411.059 |

The roughly 100 ms difference between commit receipt and those six model
calls is the wait for the already-running input position plus small controller
overhead; it is not Pipecat queueing or network delay. The separate BOS call
took 211.838 ms at the server. Its wrapper total was 211.202 ms: 11.680 ms
perception, 51.536 ms Nano interface, 145.023 ms EarTTS, 0.382 ms codec, and
2.580 ms residual. EarTTS's nested total includes a 0.200 ms abort and a
132.744 ms speaker-prefill reset. The first speech-like server PCM was three
positions after BOS.

A repeated direct no-tools diagnostic reproduced the timing shape. With five
RNNT blanks already present, four ordinary settlement positions plus the
fused fifth BOS reached the boundary in 495.582 ms. The fused BOS alone took
167.889 ms, including a 75.645 ms EarTTS reset; first speech-like PCM again
arrived three positions after BOS. Those two direct diagnostics were retained
as red rather than promoted because the model omitted the third recall answer,
despite nonempty output audio. The browser/tool run is the semantic and
end-to-end gate for this diagnostic image.

These data rule out the CPU codec as the primary remaining cost. The next
optimization target is discarded idle EarTTS work and the EarTTS reset on the
response-critical BOS. Nano/RNNT settlement and generated speech onset remain
separate later targets; no threshold or semantic gate is relaxed by this
attribution.

The eager-start and onset-preroll remediations are qualified in immutable runtime
`sha256:523b03877062099481d200d01f3a72c08237b8a289d0d8d6416cbdc5dc712792`.
Eager per-session start moved a 4190.9 ms prompt prefill in the latest run
entirely ahead of `session.updated`. Onset-suffix preroll replay then retained
the two buffered active frames plus two context frames and skipped eight older
frames (`12 -> 4`, 290.485 ms). The browser's first-turn voice-to-voice latency
fell from 2260.8 to 1461.3 ms. The full expected voice transcript was retained
on both microphone turns, and the browser audio, typed-math, and typed-memory
checks all passed.

The no-tools blank-fence/BOS fusion is separately qualified in immutable
runtime
`sha256:b289c9bfd332866823166b4a36ded8755c772d6c6e1192cce278e18888cc8539`.
For each of three live turns, the exact RNNT position that reached blank count
10 also emitted the one agent BOS and opened the response; the prior separate
180--198 ms BOS position was absent. The tool/function channel was inactive and
effective PAD at each boundary. Complete, non-duplicated input transcripts
passed the 0.35 WER gate, all model-text answers were correct, and the promoted
immutable Nemotron English evaluator classified all three retained outputs as
speech with response WER 0.0 and the required semantics. This fast path is
strictly limited to sessions that advertise no tools. Tool-advertised sessions,
including the current Playground configuration, retain the qualified legacy
settlement followed by a separate BOS position.

The ordinary-PCM idle-PAD EarTTS bypass is qualified in immutable runtime
`sha256:c73087684a544984b86509dd028c47c623d4a28b7fb3d0f8d3476b5fc82b7577`
at canonical patch
`3805cbacc06dbb444ec54ea4247635a813fe28b91fc6a3ea7810922802f18026`.
The correlated model trace contains 439 actual bypassed positions and 203
enabled-but-ineligible positions. Every bypass reports zero EarTTS time, emits
the qualified cloned decoder-silence codes, and preserves the recurrent code
and cache identity. All four BOS positions retain the legacy abort-and-prefill
reset, measured at 72.9--78.3 ms.

The tool-advertised browser run passed two microphone turns, typed math, and
voice-to-typed memory recall. Voice-to-voice latency measured 1360.0 ms on the
first turn and 1322.5 ms on the warm turn; Pipecat TTFB was 632.625 and 550.840
ms, and TTFA was 876.072 and 795.284 ms. The first six settlement positions
took 365.624 ms at the server versus 417.902 ms in the stage-attribution run.
The bypass removed the entire measured 96.836 ms EarTTS contribution, while
one-frame-pipelined codec variation reduced the wall-time saving to 52.278 ms
inside that bracket. The roughly 111 ms end-to-end difference is therefore a
useful observed result, not a claim that every run will save exactly that much.

An actual external-tool cycle independently passed its structural boundary
gate and the immutable Nemotron English evaluator transcribed the response
exactly at WER 0.0. A separate four-turn no-tools run had input WER 0.0 and
four independently correct model-text and audio-ASR rows; its top ASR report
remains red because that legacy report kind is intentionally defined for
exactly three turns. A fixed-duration four-tool stress artifact also remains
red and is not promotion evidence. C2a promotion rests on the green browser
and single-tool reports plus the exact mechanism trace.

The current post-remediation budget is:

| Component | Voice turn 1 | Voice turn 2 |
|---|---:|---:|
| Browser last speech to first speech | 1461.3 | 1407.5 |
| Browser endpoint/VAD/Smart-Turn/RTVI | about 365.0 | about 377.7 |
| Pipecat commit queue | 0.110 | 0.322 |
| Commit send to server handler read | 0.545 | 47.942 |
| Server read to ten-blank fence | 510.045 | 430.089 |
| Blank fence to BOS/first audio bytes | 197.544 | 180.533 |
| BOS to first speech-like model PCM | about 249.945 | 248.445 |
| Model-speech to browser residual | about 138 | about 123 |
| Pipecat TTFB | 709.613 | 659.712 |
| Pipecat TTFA | 958.048 | 904.157 |

For turn two, the additive browser budget is therefore approximately
`377.7 + 47.942 + 430.089 + 180.533 + 248.445 + 123 = 1407.5 ms`.
Pipecat TTFB independently reconciles commit delivery, settlement, and BOS;
TTFA adds the generated EarTTS onset. The 128.7 ms difference from the prior
warm measurement is treated as run variance because warm turns do not replay
preroll.

| Component | Cold session `8f46562a...` | Later session, turn 1 `bd98038f...` | Same session, turn 2 `bd98038f...` |
|---|---:|---:|---:|
| Browser last speech to first speech | 8971.2 | 4021.3 | 3329.0 |
| Browser endpoint/Smart-Turn/RTVI interval | 266.5 | 366.3 | included in 499.3 residual |
| Pipecat commit queue | 0.114 | 0.104 | 0.293 |
| Commit send to server handler read | 7555.354 | 2511.209 | 1936.369 |
| Server handler read to RNNT blank fence | 510.894 | 540.181 | 421.782 |
| Blank fence to BOS/first audio bytes | 172.504 | 211.642 | 222.050 |
| First bytes to first speech-like model PCM | 243.447 | 248.435 | 249.544 |
| Server-speech/browser endpoint residual | about 222 | about 143 | combined endpoint/output residual 499.3 |
| Pipecat TTFB | 8239.599 | 3263.807 | not retained by the first two-turn probe |
| Pipecat TTFA | 8483.046 | 3512.242 | not retained by the first two-turn probe |

The first cold trace spent 6939.919 ms in per-session `engine.start()`. The
first-only 12-frame speech-gate pre-roll bracket was 7877.373 ms total, so it
contains that start and 937.454 ms of replay. In the later session the same
figures were 1802.656 ms and 2832.474 ms, respectively. These intervals overlap
the commit send-to-read interval and must not be added to it.

For the same-session second turn, the exact internal commit-send-to-first-model-
speech path was:

```
1936.369 input backlog
 421.782 missing RNNT blank-fence positions
 222.050 EOU/BOS and first output frame
 249.544 three silent output frames
--------
2829.745 ms
```

The remaining 499.255 ms in the browser measurement combines endpoint
detection before commit and output delivery after the first speech-like server
PCM.

## Root cause

The 2.725-second TTFB originally observed was real, but it was not a fixed model
TTFB and it was not Smart Turn inference.

Pipecat's local audio queue is effectively empty at commit (0.1--0.3 ms). Its
send task writes audio and the commit promptly. The server, however, receives
one ordered WebSocket message and awaits `process_pcm()`--including synchronous
model work--before reading the next message. On the same-session second turn the
commit was behind 108 already-sent 320-sample packets, or 2160 ms of audio. The
server consumed those packets in about 1930 ms before it could read the commit,
draining rather than creating roughly 230 ms of backlog during that window.
Per-session startup and first-only pre-roll create a large initial deficit;
serial input and response stepping preserve enough of it to affect later turns.
This sample proves persistent residual session-onset backlog, not an intrinsic
1.936-second cost on every turn. A queue-lag-zero marker or at least three
same-session turns is needed before claiming a steady-state distribution.

Even with zero input backlog, the currently qualified boundary contract has a
substantial floor. The retained run needed 422 ms to reach ten consecutive RNNT
blank positions, 222 ms for the forced EOU/BOS response-start step, about 250 ms
of generated leading silence, plus endpoint and output delivery. Removing the
entire 1.936-second backlog from this sample still leaves about 1393 ms in the
browser. Previous live
isolation found that zero and five settlement blanks could produce empty or bad
responses; ten blanks passed. The fence must not be shortened merely to improve
the metric.

## Remediation order

1. Instrument `engine.start()` subphases, then eagerly start and prefill the
   real session after `session.update`, before user speech, once instructions
   and tools are immutable. This reliably removes the variable 1.8--6.9 second
   operation from speech-turn latency. The existing global realtime warmup had
   already completed before the 6.9-second sample, so stronger boot warmup is
   only a hypothesis until the subphase trace identifies reusable work. Make
   session readiness explicit and fail closed if startup does not finish.
2. Separate WebSocket ingestion from model consumption with one bounded,
   ordered server queue. Retain exact audio/control barriers and capacity
   failure. This makes commit arrival immediate and exposes, rather than hides,
   model backlog. It does not by itself make unprocessed audio disappear or
   permit a commit to jump over it. Backlog elimination additionally needs a
   qualified catch-up mechanism and input consumption that keeps pace during
   response generation. Perception batching may help its front-end only; Nano
   and EarTTS recurrent positions remain sequential unless proven otherwise.
3. Remove the first-turn 12-frame replay stall without losing onset context.
   Candidate designs must preserve the qualified client `turn_start` authority
   and recurrent ordering; they require a fresh voice-input qualification.
   **Complete:** the onset-suffix selector retained frames 8 through 11 in the
   live 12-frame buffer, skipped eight older frames, preserved the complete
   input transcript, and passed the browser audio/text qualification.
4. Prove queue-lag-zero with at least three same-session turns before assigning
   a recurring steady-state cost.
5. Investigate whether blank settlement can safely overlap the 200 ms VAD plus
   roughly 30 ms Smart-Turn window, without allowing silence alone to commit a
   turn. Speculation needs rollback/checkpoint semantics if speech resumes. Any
   shorter fence or alternative explicit-EOU path requires prospective
   qualification against empty responses.
   **Partially complete:** for sessions with no advertised tools, the final
   exact blank-10 position is now fused with the agent BOS. Tool sessions remain
   byte-for-byte on the legacy path because the pre-RNNT function-call guard
   must observe every settlement position. This removes one serial model
   position without shortening the ten-blank contract.
6. Profile and reduce the 172--222 ms BOS step, the three 80 ms leading-silence
   frames, and roughly 143--222 ms of Pipecat/WebRTC/browser output delay.
   **Prepared-epoch substep complete:** C2b prepares a request-bound EarTTS
   speaker epoch at
   explicit microphone `turn_start`, while the user is speaking, and arms it
   only after the matching commit has flushed all PCM. A BOS may reuse it only
   if the active request, iterator, generated-token count, speaker inputs,
   prompt-token IDs, initial code, session epoch, prepare epoch, and committed
   client turn still match exactly. Every real intervening EarTTS call consumes
   or invalidates it; typed/direct input never arms it. Mismatch and preparation
   failure retain the byte-equivalent legacy abort-and-prefill reset.

   The boundary evidence is prospectively versioned around one successful BOS
   epoch transition: `delta transition = 1 = delta reset + delta prepared
   reuse`. Exactly one of legacy `(1, 1, 0)` or prepared reuse `(1, 0, 1)` is
   accepted, with the latter bound to the committed microphone turn. The arm is
   consumed before BOS inference so a failed call cannot reuse it. Fused,
   separate, and post-function-recovery BOS positions are validated before
   publication. Preparation timing remains a separate turn-start interval and
   is never charged to C1's BOS-step EarTTS timing. The preregistered maximum
   expected recovery is the measured 73--78 ms reset, not a claim of reaching
   400 ms.

   The promoted runtime is
   `sha256:97f6f9d21b49fdc6c1fbe8c75aae37eca14bc71dd2a3eb52e48442b960c1c6a7`.
   Its canonical three-turn no-tools voice run passed all input-transcription,
   response-text, and independent Nemotron English ASR gates with response WER
   0.0. Its actual-tool run completed one call, one EOTR, and one post-function
   BOS; two internal FC resets were retained separately and the terminal step
   was exactly legacy `(transition, reset, reuse) = (1, 1, 0)`. Independent ASR
   transcribed its spoken fresh-word answer at WER 0.0.

   The correlated Chromium/SmallWebRTC run also passed microphone input, typed
   math, typed recall of voice-only memory, audio-carrier, and clean-close
   checks. The local route measured 1425.6 ms browser voice-to-voice, 648.2 ms
   Pipecat TTFB, and 896.7 ms TTFA, including 248.4 ms of generated leading
   silence. A final browser run through the deployed ngrok HTTPS/WebSocket route
   also passed and measured 1343.5 ms voice-to-voice, 552.2 ms TTFB, and 795.7
   ms TTFA, including 243.4 ms of generated leading silence. The
   prepared reset is removed from an eligible microphone BOS, but one point
   sample cannot distinguish its roughly 75 ms saving from run-to-run endpoint
   and model variation. The remaining headline latency is still dominated by
   endpoint/blank-fence processing and generated onset silence.
7. After changes, promote a latency gate with several cold and same-session
   turns and percentile limits. The current values are repeatable point
   measurements, not an SLA distribution.

The proposed 400 ms budget is therefore an architectural target, not a tuning
change. With the present qualified fence and output behavior, the post-commit
path alone exceeds it.
