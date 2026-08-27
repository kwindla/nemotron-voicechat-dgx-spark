# Live session 431ac6df: paired adversarial analysis R1

Date: 2026-08-19 UTC

Session: `431ac6df-573c-45af-8746-7cd9f66bd8c3`

Constraint: analysis only; no source, test, deployment, or commit changes

## Executive conclusion

Three separate mechanisms occurred and should not be collapsed into one story.

1. **The server kept real time in this session, with a thin compute margin and a
   materially larger queued-playout margin.** All 1,128 model steps completed
   below their 80 ms media budget. Speaking-step `server_step_ms` was 72.82 ms
   on average, 75.08 ms at p95, and 78.25 ms worst case. The previously cited
   78--83 ms speaking numbers describe completion/arrival cadence, not model
   compute. Cadence includes waiting for the next 80 ms input packet and event
   loop scheduling, so it can exceed 80 ms while each compute step remains
   below budget. The 160 ms response-local prebuffer then released two 80 ms
   frames on every response and absorbed that cadence jitter. The retained
   RawAudioTrack counter recorded 4,744 positive-depth 10 ms observations and
   zero resumed, same-epoch underflow ticks.

2. **Responses 2 and 3 really lost their audio tails, but this is not the
   no-text watchdog defect fixed yesterday.** Pinned offline ASR hears only
   22/41 and 68/86 generated words, respectively. In both cases Nano finished
   emitting all text many seconds before EarTTS fell into sustained digital
   silence. The `decoded_silence_watchdog` observed 20 silent decoder frames
   and forced EOS on the next step. It was downstream of the loss, not its
   initiating event. None of the named ratio, idle-bypass, PAD-tail-silence, or
   response-delivery gates initiated the silence. The strongest localization
   supported by the retained evidence is **EarTTS acoustic generation (or the
   immediately following acoustic-code/codec path) entering terminal silence
   with semantic text still unrendered**. Raw generated acoustic codes were not
   retained, so the evidence cannot make the final split between a model
   silence decision and a lower acoustic decoding/codec fault.

3. **The tool failure is partly a concrete boundary-timing loss, not merely a
   generic “several turns deep” reluctance.** Token 20 is `<SPECIAL_20>`, the
   function-channel SOTC token. It was produced at frame 832, with the user
   blank fence at 9/10, one model position before the authoritative EOU
   boundary, and the pre-EOU guard replaced it with PAD. No later SOTC was
   produced. That mechanically explains the lost tool opportunity preceding
   response 5. The guard behaved according to its current exact-boundary
   contract; this trace alone does not justify executing or staging every
   one-frame-early SOTC because retained prior diagnostics include early false
   starts. For the later explicit requests (responses 6 and 7), the function
   channel simply remained PAD. Without function logits, the trace cannot say
   why. Response 5's 20 silent frames were not a tool-wait state: every FC state
   flag was false.

Confidence in those three headline findings is respectively **high**, **high
for localization / medium for the final EarTTS-internal split**, and **high for
the mechanics / medium for semantic intent**.

## Evidence and clock alignment

The server trace and client playout trace both use `CLOCK_MONOTONIC` on this
host. Server response boundaries precede the corresponding client lifecycle
record by approximately 0.7--1.5 ms, so they align without a fitted offset.
The playout trace is valid and closed with zero dropped records.

The response topology is:

| Response | Server BOS--end frames | Client status | Emitted audio | Text/audio result | EOS reason |
|---|---:|---|---:|---|---|
| 1 | 54--105 | completed | 51 frames / 4.08 s | 13/13 words, complete | model/turn-taking |
| 2 | 185--304 | completed | 109 / 8.72 s | 22/41, truncated | decoded-silence watchdog |
| 3 | 398--668 | completed | 260 / 20.80 s | 68/86, truncated | decoded-silence watchdog |
| 4 | 775--813 | cancelled | 33 / 2.64 s | interrupted at 10/14 words | user barge-in, then model/turn-taking |
| 5 | 834--892 | completed | 48 / 3.84 s | 11/11, complete | decoded-silence watchdog |
| 6 | 1003--1059 | cancelled | 49 / 3.92 s | 11/11, complete before cancellation | user barge-in/model-turn-taking |
| 7 | 1078--1124 | completed | 46 / 3.68 s | 11/11, complete | model/turn-taking |

The ASR transcripts were generated from the frame-exact emitted PCM using the
pinned `pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b` image with
GPU access, host IPC, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and no network. This
supersedes the earlier characters-per-second inference.

One unrelated terminal event occurred after response 7 had completed: the
server claimed `internal_error` at 44362.133 because publication attempted a
WebSocket send after close. It is 307 ms after response 7's `response.done`
and has no temporal or state connection to the two truncated responses.

## 1. What the real-time metrics actually mean

### Compute duration is not completion cadence

The full-session distributions are:

| Measure | n | p50 | mean | p95 | p99 | max | >80 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `inference_ms`, all model steps | 1,128 | 71.219 | 68.404 | 74.741 | 75.755 | 78.135 | 0 |
| `server_step_ms`, all model steps | 1,128 | 71.305 | 68.500 | 74.865 | 75.843 | 78.250 | 0 |
| model completion-to-completion | 1,127 | 78.668 | 78.104 | 82.712 | 87.519 | 178.919 | 354 |
| client metric receive-to-receive | 1,127 | 78.624 | 78.105 | 82.804 | 87.226 | 178.943 | 357 |

The near equality of the last two rows shows that the client trace did not add
meaningful timing distortion to server event cadence. It does **not** make
completion cadence a measurement of inference duration.

For steady agent-speaking steps:

| Measure | n | p50 | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `inference_ms` | 633 | 72.730 | 72.718 | 74.948 | 75.945 | 78.135 |
| `server_step_ms` | 633 | 72.852 | 72.821 | 75.080 | 76.085 | 78.250 |
| completion interval, both endpoints speaking | 626 | 78.169 | 77.779 | 82.385 | 84.435 | 99.781 |

For idle steps, `server_step_ms` averaged 62.97 ms and had a 60.23 ms median.
The stage timing explains the difference. EarTTS cost averaged 11.31 ms while
speaking but only 0.16 ms while idle because the qualified idle-PAD bypass was
active. Perception and Nano remained approximately 11.9 and 44.6--44.8 ms in
both strata.

Thus the earlier “78--83 ms speaking versus 62 ms idle” comparison mixed two
different clocks:

- approximately 62 ms is an idle **compute** duration;
- approximately 78--83 ms is usually a speaking **completion/arrival interval**
  paced by 80 ms source frames plus scheduler phase.

The server can finish a queued speaking step in about 73 ms, then wait for the
next source packet. Conversely, a completion interval can be 82 ms even though
the work inside that step took 73 ms. The 80 ms budget applies to work needed
per 80 ms media position; it is not a promise that two event timestamps are
exactly 80 ms apart.

### The compute margin is real but not large

Measured as `80 - server_step_ms`, speaking compute headroom was approximately:

- 7.15 ms at the median;
- 4.92 ms at p95;
- 3.92 ms at p99;
- 1.75 ms in the worst observed step.

That is a clean pass for this one session, not a generous worst-case capacity
claim. A future GPU/CPU contention event larger than a few milliseconds could
miss an isolated compute deadline. The evidence does not license extrapolation
to a different load, thermal state, or simultaneous session count.

### The prebuffer converted thin per-step margin into larger playout margin

The playout config was 22.05 kHz mono with a 160 ms prebuffer. Every response
did exactly the same thing:

- the prebuffer re-armed at response creation;
- the threshold release contained ordinals 1 and 2, 3,528 samples total;
- those samples equal exactly 160 ms of media;
- all seven threshold releases report `frames_released=2` and
  `frames_cleared=0`.

The threshold was reached only about 74--76 ms after `response.created`
because ordinal 1 arrived with response creation and ordinal 2 arrived one
model step later. “160 ms prebuffer” is buffered **media duration**, not a
160 ms startup timer.

An analytical queue reconstruction from downstream-push timestamps, adding 80
ms for each pushed frame and consuming wall time between pushes, found no
queue-empty arrival in any response. Immediately before ordinal 3, the smallest
remaining upstream media lead was 85.16--86.87 ms by response. Thereafter the
lead generally grew because average production cadence was slightly faster
than real time. Even response 3's largest single push interval, 100.84 ms, was
absorbed by already accumulated lead.

This reconstruction is not a substitute for the RawAudioTrack observation: it
sits upstream of BaseOutput and does not include Opus, RTP, browser jitter
buffering, or device playback. It does explain why occasional 81--84 ms push
intervals did not imply starvation.

At interruptions, the response-local prebuffer reported zero held frames to
clear. That is expected because it had released at startup; it does **not** say
that all already-downstream BaseOutput/RawTrack audio was synchronously erased.

### What the zero-underflow result says

The same-session Pipecat disconnect summary in `scratchpad/bot3.log` reports:

```text
recv_calls=9048
in_speech_ticks=0
in_speech_runs=0
positive depth observations=1186*4=4744
depth histogram={1:1186, 2:1186, 3:1186, 4:1186}
stale_epoch_runs_discarded=9
backstop_runs_discarded=0
epoch_advances=21
```

The equal 1/2/3/4 histogram is the normal four-chunk RawAudioTrack batch
sawtooth. `min_depth_ms_while_playing=10` means the sampler saw the last 10 ms
chunk of a batch; it does **not** mean the whole pipeline had only 10 ms of
headroom. BaseOutput and the response prebuffer sit upstream.

The result is strong evidence that no ordinary queue-empty period resumed with
real audio under the same mutable control epoch in this session. It matches the
user hearing no crackle and the upstream queue reconstruction.

It retains the limitations established in the prior R3 review: nine runs were
discarded after an epoch change, terminal zero runs never resume and are
reported as tails, queued audio is not itself epoch-tagged, and everything
after RawAudioTrack (encoding, RTP/network, browser, device) is outside the
counter. Therefore the accurate statement is **“no detected mid-response,
resumed, same-epoch RawTrack starvation in this session,”** not “the complete
audio path can never crackle.”

### Frame-count time is not wall time

The decoded-silence guard's 20 frames represent 1.6 seconds of nominal decoded
media. The observed wall time from the first all-silent frame completion to
the request edge was about 1.42--1.52 seconds because these model positions
were sometimes produced faster than media time. This guard is frame-counted.
The separate 30-second response deadline is the true monotonic wall timer and
did not expire in this session.

## 2. The two incomplete audio responses

### Direct confirmation

The ASR/content result is unambiguous:

- Response 2 emitted the generated text only through “I am Nemotron, created
  by NVIDIA.” The never-spoken tail is “I can help with anything from answering
  questions to generating creative content. What would you like to talk
  about?”
- Response 3 emitted through “...and never stopped visiting Luna.” The
  never-spoken tail is “From then on, the forest was filled with laughter and
  magic, and everyone lived in peace. The end.”
- Response 5 emitted every generated word despite closing through the same
  watchdog.

These are clean final-tail omissions, not mid-response transport gaps.

### Exact decoder/watchdog timeline

| Response | Last text-delta frame | Last non-silent decoder frame | First sustained digital-silence frame | Watchdog request | Forced EOS |
|---|---:|---:|---:|---:|---:|
| 2 | 236 | 283 | 284 | 303 | 304 |
| 3 | 504 | 647 | 648 | 667 | 668 |
| 5 | 847 | 871 | 872 | 891 | 892 |

At request frames 303, 667, and 891:

- control was PAD;
- `model_output_audio.rms_dbfs` was -120.0;
- decoded `silent_frames`, `no_text_frames`, and `no_audio_frames` were all 20;
- `required_frames` was 20, while both no-text/no-audio requirements were 30;
- all function-cycle flags were false and completed-call count was zero.

The request is one-shot. `request_agent_eos()` is observed on the next model
position, where `agent_control` becomes `agent_eos` and
`agent_eos_edge_pending=true`. Client `response.done` follows about 80 ms after
the request edge and about 1 ms after the forced-EOS model result.

I agree with the first-pass conclusion that the watchdog closed responses the
model did not close itself. I also agree that watchdog occurrence and audio
truncation are not equivalent: response 5 is the direct counterexample.

### This is not yesterday's no-text timeout defect

The runtime has multiple completion/rescue predicates under the historical
contract name `production-hotfix-notext-watchdog-v1`:

- no semantic or acoustic progress for 30 frames:
  `no_text_since_bos_watchdog`;
- no function or acoustic progress for 30 frames:
  `no_audio_since_bos_watchdog`;
- 20 consecutive silent decoded frames after audible audio:
  `decoded_silence_watchdog`;
- a separate 30-second monotonic response deadline.

Yesterday's repair made acoustic progress reset the no-text counter and
qualified dead-response rescue. The current session never reached either
30-frame predicate. The 20-frame decoded-silence path is intentionally retained
as the normal cleanup path for an acoustic model that becomes silent without
self-EOS. The project test-matrix amendment explicitly calls tail loss followed
by this closure “model-silence truncation,” a known limitation, and distinguishes
it from a watchdog that fires while audio is live.

So the answer to “do we still have a watchdog timeout issue?” is: **not the
fixed no-text issue. A different decoded-silence cleanup guard still exists and
fired as designed. The underlying EarTTS tail-loss limitation is still live.**

### Named policy gates did not initiate the loss

| Candidate mechanism | Evidence | Finding |
|---|---|---|
| 16x text-token ratio cap | At request, response 2 reported 954 frames remaining, response 3 reported 1,427, response 5 reported 151. | Not reached. It cannot be the truncation trigger. |
| Per-turn text counter | Response 2 inherited 16 tokens from response 1, so its reported 67 is not a clean per-turn count. Even recomputed from its own 51 tokens, 118 talking frames is far below `16*51=816`. | Telemetry/reset anomaly, but it enlarges rather than shortens the cap and cannot explain loss. |
| EarTTS idle-PAD bypass | Every open-response position reports `enabled_no_skip`; the bypass predicate requires agent idle. | Not active during the lost tails. |
| PAD decoder-silence tail policy | Its strict predicate is prior PAD count `> 3 * content count`. At first sustained silence, responses 2 and 3 had roughly 47/51 and 143/106 PAD/content positions, far below 153 and 318. | Not reached when silence began or before their watchdogs. |
| PAD policy in response 5 | The 3x threshold is crossed only late in its 20-frame silent tail, after EarTTS had already emitted about 16 all-silent frames. | May preserve already-existing silence late; did not initiate it. |
| Delivery-silence gate | It marks acoustic completion after 12 frames at or below -60 dBFS. In responses 2/3/5, the withheld frames were already -120 dBFS. It is reversible before EOS if audible output returns. | Withheld redundant silence; did not make EarTTS silent or remove audible words. |
| Post-EOS tail deadline | All three watchdog EOS and boundary-end events coincide with `post_eos_frames=0`; no 25-frame tail deadline occurred. | Not implicated. |
| BOS reset/prepared epoch | Each affected response has one attested prepared-reuse BOS transition and no reset or epoch change during the response. | No proximate failure evidence; raw state capture would be needed to exclude a deeper prepared-state effect absolutely. |

The waveform-tail argument in the first pass is not a valid discriminator. The
watchdog responses decayed from -78.0, -68.7, and -69.4 dBFS to -120 dBFS in
one frame. Self-EOS responses did the same from -59.4 and -73.2 dBFS. Earlier
multi-frame quiet pauses occurred and recovered in the complete responses.
There is no unique “hard watchdog cut” shape: the watchdog arrived 20 silent
decoder frames later.

### Root-cause boundary and missing measurement

What is proved, in causal order, is:

```text
all semantic text reaches the streaming turn
        -> EarTTS/decoder continues for seconds on PAD positions
        -> decoded PCM naturally falls to sustained silence with tail text unspoken
        -> delivery gate stops forwarding redundant silence
        -> decoded-silence guard requests EOS after 20 silent frames
        -> next model step force-closes the response
```

Calling the watchdog “the cause” reverses this observed order. However, the
retained evidence cannot prove the counterfactual that EarTTS would *never*
resume after more than 20 silent frames. The complete responses show recovered
pauses of four frames, not 20, which makes a very late recovery unlikely but
does not make it impossible.

To close both remaining uncertainties, replay the captured input with:

1. raw EarTTS generated code, post-policy decoder code, codec PCM, current text
   token, `_tts_in_turn_content`, `_tts_in_turn_pads`, and `tail_done` captured
   per position; and
2. the decoded-silence threshold temporarily extended or disabled in a
   diagnostic image.

If raw generated codes become codec-silence codes before any policy predicate,
the model selected silence. If raw codes remain non-silent but post-policy or
codec PCM becomes silent, the lower runtime path is responsible. If audio
resumes only after frame 20 with the guard extended, then the current guard is
also a truncating timeout. None of those distinctions can be recovered from
PCM and aggregate stage timings alone.

## 3. Why no tools were called

### Tools were genuinely configured

At 44273.694, before any response, `session_tools_configured` records:

- `model_ready=true`;
- tools `get_current_time` and `set_timezone`;
- `rendered_prompt_bytes=2161`;
- client-authored v1 function-output mode.

Across all 1,128 model steps, every `function_delta` is empty,
`completed_calls=0`, and every effective function token is PAD. The only raw
non-PAD evidence is the one suppression counter transition at frame 832.

### Token 20 is SOTC, and it arrived one position early

The installed and canonical wrapper initialize:

```text
_fc_sotc_id = tokenizer("<SPECIAL_20>")[0]
_fc_eotc_id = tokenizer("<SPECIAL_21>")[0]
_fc_eotr_id = tokenizer("<SPECIAL_22>")[0]
```

The trace independently reports EOTR id 22, so raw id 20 is the Start Of Tool
Call token.

The relevant aligned timeline for user turn 5 is:

| Frame/time | User/turn state | Function result |
|---|---|---|
| 823 / 44338.497 | transcript reaches “I'm in the Pacific time” | PAD |
| 824 / 44338.568 | post-commit audio frame and EarTTS commit arm | PAD |
| 831 / 44339.067 | RNNT blank fence 8/10 | PAD |
| 832 / 44339.140 | blank fence 9/10; agent idle | raw SOTC 20, suppressed to PAD |
| 833 / 44339.211 | fence reaches 10/10 and EOU settles | PAD |
| 834 / 44339.285 | authoritative BOS opens response 5 | PAD |

The guard accepts an idle-agent SOTC only when
`_external_user_eou_requested` is latched on that exact position. At frame 832
it was not; executing a function call before authoritative EOU would race the
user's still-open turn. Replacement with PAD is therefore faithful to the
current safety/ownership contract.

This is nevertheless a real recall loss: a semantically plausible tool token
one frame before EOU is destroyed rather than deferred, and no later SOTC
reappears. The trace supports “the guard suppressed the only tool-call opening
the model produced” with **high confidence**. It supports “that early token was
definitely final user intent and should always be staged” only with **medium
confidence**. Prior repository diagnostics document a one-frame-early SOTC
false start (case c009) that was deliberately discarded, while legitimate
calls persisted to the authoritative boundary. A single current session cannot
separate those populations safely.

The closing measurement is an early-SOTC corpus with the function logits and
later boundary decision retained: count how often a 9/10 SOTC persists at
10/10, and compare true tool prompts with non-tool/continued-speech controls.
A change to buffer early SOTC requires that false-positive evidence; blindly
executing frame 832 is not justified.

### The other advertised tool intentions have a different explanation

Response 4 says it can get the time but first asks for the user's timezone; it
was then interrupted. No call is necessarily due before that clarification.

For response 5, the one pre-boundary SOTC was suppressed. During response 5
and its silent tail, all of the following remained false/zero:

- `active`, `awaiting_response`, `awaiting_eotr`, `injecting_response`, and
  `background_active`;
- forced tokens and completed calls;
- agent-open suppression count.

Therefore response 5 was **not in a runtime tool-wait state** after saying
“Just a second.” It was an ordinary still-open verbal turn whose acoustic
decoder had finished speaking and whose function head emitted PAD. It is
reasonable to infer semantic tool intent from the sentence plus the earlier
SOTC, but not to describe the 20 PAD frames as the runtime waiting for a tool.

Responses 6 and 7 explicitly ask/confirm `set_timezone`, yet produce no raw
non-PAD token and no suppression transition. That is a function-head/model
decision, not the pre-EOU guard. It is compatible with known tool reluctance,
context contamination after the lost SOTC, prompt/model behavior, or a PAD
margin in the quantized function head. The retained trace has no function
logits, so it cannot choose among them. Calling it specifically the “several
turns deep” effect is unsupported without a fresh-versus-deep counterbalanced
reproduction of the same utterance.

The needed measurement is the existing function-logit diagnostic over voice
input, settlement, EOU boundary, BOS, and early response positions for turns 5,
6, and 7. It would show whether SOTC was runner-up to PAD, far below PAD, or
semantically absent.

### The duplicated-function-token truncation comment is not implicated

The adjacent `agent_open_guard` comment warns that a duplicated agent token on
the function channel can feed back on the next position and truncate a spoken
**post-tool** answer. Responses 2 and 3 had no FC cycle, no tool response, no
non-PAD function token, and `agent_open_suppressed_tokens=0` throughout. The
only suppression was pre-EOU at frame 832, hundreds of frames later.

That code path is ruled out as the cause of response 2/3 tail loss with **high
confidence**.

## Agreement/disagreement with the first pass

| First-pass claim | Assessment |
|---|---|
| Three decoded-silence requests at frames 303/667/891, next-frame EOS | **Agree; directly confirmed.** |
| The request is cleanup after 20 silent frames, not the initiating truncation event | **Agree on observed order; reserve the unmeasured >20-frame recovery counterfactual.** |
| Yesterday's fix was a different no-text path | **Agree.** The no-text/no-audio 30-frame predicates never fired. |
| Responses 2/3 were truncated based on rate | **Conclusion confirmed directly by pinned ASR; rate argument is superseded.** |
| Watchdog and truncation are not equivalent because response 5 is complete | **Agree.** |
| Watchdog tails cut within one frame while self-EOS tails decay over about five | **Disagree.** Both terminal classes show a one-frame fall to digital silence; this shape does not identify the closer. |
| Response 5 was waiting for a function call | **Disagree as a runtime-state claim.** The earlier SOTC shows plausible semantic intent, but all FC wait/activity flags were false throughout response 5 and its tail. |
| Pre-EOU suppression cannot explain all four verbal tool intentions | **Agree.** It explains one concrete lost opportunity only; later turns stayed raw PAD. |
| The guard's duplicated-token warning may explain responses 2/3 | **Refute.** No tool/post-tool cycle or agent-open function suppression existed then. |

## Bottom line and confidence

- **Real-time compute margin in this session:** high confidence. Every model
  step was under 80 ms; speaking worst case left 1.75 ms.
- **Effective server playout margin in this session:** high confidence for the
  measured segment. Two-frame prebuffering, upstream queue reconstruction, and
  the RawTrack counter all agree that ordinary mid-response starvation did not
  occur. Generalization beyond this session and post-RawTrack stages remains
  unmeasured.
- **Responses 2/3 lost final spoken content:** very high confidence from pinned
  ASR over exact emitted PCM.
- **Watchdog was downstream cleanup, not the initiating event:** high confidence
  for observed causality; medium confidence that a diagnostic >20-frame run
  would never resume.
- **Named runtime caps/policies initiated the two losses:** high confidence they
  did not, based on live counters and exact source predicates.
- **Final internal root cause:** medium confidence at EarTTS/acoustic-generation
  boundary; raw-code capture is required to split model selection from decoder
  or codec behavior.
- **Frame-832 token identity and suppression mechanics:** very high confidence.
- **Frame-832 guard is categorically right or wrong:** not decidable from one
  session; current exact-boundary behavior is by design, while tool recall loss
  is real.
- **Later PAD-only tool failures are specifically conversation-depth
  reluctance:** low confidence; function logits and a depth-controlled replay
  are required.

## Evidence inventory

- Server model trace:
  `~/.local/state/nemotron-voicechat/traces/model/431ac6df-573c-45af-8746-7cd9f66bd8c3/`
- Client playout trace:
  `~/.local/state/nemotron-voicechat/playout-traces/pipecat-playout-r3-0002.jsonl`
- Frame-exact response WAVs and pinned ASR output:
  `scratchpad/clips/response_*.wav`, `scratchpad/clips/transcripts.json`
- Same-session RawAudioTrack summary: `scratchpad/bot3.log`
- Watchdog/boundary implementation: `src/nemotron_voicechat_runtime/server.py`
- Function guard and EarTTS streaming implementation:
  `src/nemotron_voicechat_runtime/patches/nemotron-voicechat-rnnt-turn-taking.patch`
- PAD-silence and idle-bypass predicates:
  `src/nemotron_voicechat_runtime/runtime_optimizations.py`
- Prior scope interpretations:
  `docs/reviews/audio-crackle-root-cause-r3.md` and
  `reports/notext-watchdog/test-matrix-amendment.md`
