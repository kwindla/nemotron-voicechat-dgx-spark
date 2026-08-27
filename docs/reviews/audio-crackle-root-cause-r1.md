# Audio crackle: independent root-cause assessment (R1)

Date: 2026-08-18 UTC  
Scope cutoff: session `3f6bc203-7b57-4a34-bcc8-df3ccceff8ea` (later live sessions were excluded)  
Constraint: analysis only; no source, test, dependency, or environment changes

## Executive conclusion

The strongest explanation in the available evidence is repeated source-side underrun, but it is not yet a directly observed root cause. The traces do not record `RawAudioTrack` queue depth or its auto-silence branch, and there is no browser PCM/WebRTC-stats capture for the reported human session. The conclusion is therefore **high-confidence mechanism plus trace replay**, not proof that every heard crack was an underrun.

Two corrections materially strengthen the underrun case:

1. The raw-frame sawtooth model is right that a 160 ms one-shot buffer has only about 80 ms at its first pre-arrival trough, not 160 ms. The earlier `160 / drift` estimate was wrong.
2. The production path is worse than that ideal model. Stateful SOXR initially exposes only 135.5 ms of the first 160 ms, and Pipecat exposes only complete 40 ms chunks. Thus the second model delta releases **120 ms of immediately playable transport audio**, not 160 ms. The first trough is approximately `120 - interarrival`, about 37 ms for an 82.8 ms interval.

Replaying the seven human responses from sessions `b2545a28-...` and `3f6bc203-...` through the installed SOXR output-size pattern, the 40 ms Pipecat chunker, and five possible 10 ms track phases predicts:

- only 2/7 responses clean at 160 ms for every track phase (3/7 clean for at least one phase);
- 62–63 auto-silence ticks in the 227-frame response;
- its first tick 0.84–1.01 s after release, or about **0.59–0.76 s after audible speech begins**;
- 800 ms raw prebuffer clean for all 211 responses in the frozen last-40-session cohort and all seven human responses in this deterministic replay.

This repeated-underflow behavior explains “more crackling than one predicted underrun”: after the queue drains, many subsequent arrivals that are slightly over 80 ms can each cause another 10 ms zero splice.

The proposed 12.5-click/s sample-rate-conversion mechanism is **refuted**. Pipecat uses one persistent `SOXRStreamAudioResampler` per media sender. It retains filter history across model chunks. No relevant inter-delta gap exceeded its 200 ms reset threshold, and actual PCM boundary measurements do not show anomalous per-frame discontinuities.

## Claim-by-claim verdict

| Claim | Verdict | Evidence / correction |
|---|---|---|
| Delivered/speaking work is near the 80 ms budget; idle is materially faster. | **CONFIRMED (qualitative); exact quoted means are cohort-dependent.** | In the seven human responses, wrapper time was 78.396 ms for audio+text and 78.324 ms for delivered audio without text. Comparable model-step idle wrapper time was 66.167 ms. In the latest session, server-step means were 80.751, 79.731, and 67.646 ms respectively. |
| Audio decode, rather than text emission, consumes the margin. | **CONFIRMED as an attribution, not as exclusive causation.** | Human audio+text and no-text wrapper means differ by only 0.072 ms. `eartts_total_ms` is 12.239/13.168 ms in those groups and 0 in bypassed idle rows; that approximately accounts for the 12 ms wrapper gap. Other shared stages still consume most absolute time. |
| The latest long/short responses average 82.79/81.53 ms. | **CONFIRMED only for `monotonic_s` interarrival.** | Long: 227 deltas, 82.789 ms mean interarrival. Short: 34 deltas, 81.534 ms. Their `server_step_ms` means are instead 80.156 and 79.111 ms. Queue drain is governed by arrival spacing, so interarrival is the correct value for playout modeling. |
| Performance is not a regression from fhw8 certification. | **CONFIRMED in direction; exact 77.69/12.9 and 79.52/31.8 figures are UNPROVEN without a session manifest.** | Frozen last-40 window ending at `3f6bc203`: 77.425 ms server-step mean, 12.8% over 80. The five late certification-day sessions at 22:26–23:11 UTC: 79.823 ms and 30.9%. Whole-day aggregates differ because they include stress/qualification populations. |
| Last-40 aggregate is automation-biased; 5/7 human responses have positive drift. | **CONFIRMED.** | At least 24/40 sessions match retained `reports/session-repair` artifacts; they contribute 172/211 responses and 6,773/8,855 delivered frames. In the two identified human sessions, exactly 5/7 response interarrival means exceed 80 ms. |
| “97.6% clean at 160 ms.” | **REFUTED for the relevant population and production queue.** | Transport-correct replay gives 158/211 (74.9%) clean for every 10 ms phase, 173/211 (82.0%) clean for at least one phase. Human seven: 2/7 and 3/7. This remains a prediction because actual underflow telemetry is absent. |
| 160 ms prebuffer is active and releases on delta 2. | **CONFIRMED.** | Each delta is exactly 80 ms. Both service implementations append until `>= 160`; retained traced runs record `reason="threshold"`, `ordinals:[1,2]`, `frames_released:2`. The first-delta-plus-85-ms-propagation alternative is refuted. |
| Buffer is one-shot, not maintained. | **CONFIRMED.** | After `_playout_started=True`, all later frames bypass `_playout_buffer`. No downstream component maintains a target depth. |
| Buffer resets correctly between ordinary responses and interruptions. | **CONFIRMED.** | It clears on a new response, response completion, and interruption. The traced override records bookkeeping and delegates to the base clear. Targeted existing tests passed (9 tests). |
| Function calls/back-to-back responses can inherit stale prebuffer state. | **REFUTED for the observed protocol paths; one slow same-response tool gap remains an unobserved edge.** | Function events do not clear because they remain part of the same open response. A `response.done` followed by a new `response.created` clears twice. If a future same-response tool pause exceeds 200 ms, SOXR can reset while service prebuffer remains started; no such gap occurred (human max 95.782 ms; frozen last-40 max 168.402 ms). |
| First three response frames are silence. | **CONFIRMED for all seven human responses.** | Frames 1–3 are `-120 dBFS`; frame 4 is the first above -90 dBFS. The raw silence-to-frame-4 boundary itself has zero sample jump. |
| Leading silence simply “burns” all 240 ms of buffer. | **PARTLY CONFIRMED, but that wording is incomplete.** | Silence consumes playout time while new model frames arrive. After SOXR/chunking, the first nonzero output occurs 242.5–250.1 ms into content. At that point the actual full-chunk lead is only about 30–110 ms at 160 ms prebuffer, depending on response cadence and track phase. |
| The revised `160 - 82.8 ≈ 77 ms`, then lose drift each frame model is correct. | **CONFIRMED for an ideal raw-frame queue, but incomplete for production.** | It fixes the earlier factor-of-two error. Production starts with 120 ms playable, making the analogous trough about 37 ms and underrun earlier. |
| SOXR is stateless per 80 ms delta and clicks at 12.5 Hz. | **REFUTED.** | The installed resampler owns one persistent `soxr.ResampleStream`; it explicitly retains history. Actual raw and resampled boundaries are not anomalous. |
| Setting output to 22,050 Hz avoids all resampling and is a clean fix. | **REFUTED.** | It bypasses Pipecat SOXR but aiortc still resamples to Opus's 48 kHz. Also 22,050 Hz has no integral 10 ms sample count (`22050*10//1000 = 220`, not 220.5), weakening the transport's cadence contract. It is technically encodable, but not a no-resample path or a justified crackle fix. |

## Exact path and reset audit

The service default is 160 ms and the environment override is read at construction (`llm.py:464-477`). For a 1,764-sample mono PCM16 delta at 22,050 Hz, the duration expression at `llm.py:1531-1538` is exactly 80 ms. Delta 1 leaves 80 ms buffered. Delta 2 reaches 160 ms and `_release_playout_buffer()` pushes both frames in order (`llm.py:1547-1554`).

`stop_ttfb_metrics()` runs before buffering (`llm.py:1519`), while `Bot started speaking` is emitted by the output task on its first complete resampled output frame, before the physical WebRTC write. Consequently the observed 80–94 ms TTFB-stop-to-BotStarted gap is the wait for delta 2 plus small propagation, not 85 ms propagation of delta 1. Retained playout traces independently record two-frame threshold releases.

Reset paths:

- interruption: `llm.py:1466-1474` clears unpushed service audio and closes TTS;
- new non-overlapping response: `llm.py:1476-1494` clears before opening;
- response terminal: `llm.py:1561-1584` releases a sub-threshold short response, emits stop/end, then clears;
- traced implementation: `llm.py:1986-2004` only adds trace bookkeeping, then calls `super()._clear_playout_buffer()`;
- traced interruption: `llm.py:2006-2015` invokes that same clear with a trace reason.

There are not two independent playout states. Python dynamic dispatch means base response-start/end call sites invoke the traced override when tracing is enabled. The two `_audio_delta` implementations have the same buffer/release semantics; the traced version adds ordinals and timestamps.

Function calls do not create an implicit reset at `response.function_call_arguments.done`; this is correct while the protocol regards pre-tool and post-tool output as one continuous response. A subsequent protocol response necessarily passes through `response.done`/`response.created`. The retained traced fixture also demonstrates interruption clear followed by a new threshold release of ordinals 1 and 2.

Existing tests exercised the ordinary fill/drain, short-response drain, disabled buffer, environment override, traced release, and interruption clear paths. Command run:

```text
.venv/bin/pytest -q tests/pipecat/test_nemotron_voicechat_service.py \
  -k 'playout_prebuffer or interruption_discards' \
  tests/pipecat/test_playout_trace.py -k 'prebuffer or interruption'
```

Result: 9 passed, 88 deselected. One pre-existing coroutine warning appeared in the trace test.

## Correct underrun model

### Ideal raw-frame model

Let:

- source frame duration `F = 80 ms`;
- prebuffer threshold `B = mF` source frames;
- source arrival times `a_i`;
- constant interarrival `S = F + d`, with positive drift `d`;
- release occur at arrival `a_m`.

Immediately after release, ideal queue depth is `mF`. Just before the `k`th later arrival (`a_(m+k)`), only `k-1` later frames have arrived:

```text
Q_k^- = mF + (k - 1)F - (a_(m+k) - a_m)
```

For constant `S`:

```text
Q_k^- = (m - 1)F - k d = (B - F) - k d
k_underrun ≈ (B - F) / d
```

The earlier `B / d` calculation incorrectly credited the next 80 ms frame before it arrived. With `B=160` and `d=2.789`, the ideal constant-rate model gives `k≈28.7`, about 2.37 s after release or about 2.13 s after a 240 ms speech onset. The user's revised model is therefore the correct **ideal** model.

The actual long trace is burstier than its mean. Applying its exact arrival sequence to an ideal raw queue crosses zero at 1.60 s after release, about 1.36 s into audible speech.

### Production transport correction

The production chain is:

```text
22,050 Hz / 80 ms model delta
  -> persistent SOXR stream (22,050 -> 24,000)
  -> persistent BaseOutput byte buffer
  -> complete 960-sample / 40 ms frames only
  -> RawAudioTrack 240-sample / 10 ms queue
  -> auto-silence when empty
  -> aiortc persistent resampler to Opus 48 kHz
```

On a fresh SOXR stream, the first per-delta output lengths (24 kHz samples) are:

```text
1623, 1629, 1628, 2442, 1629, 1628, 2442, ...
```

Although a source delta represents exactly 1,920 output samples in steady state, SOXR retains initial filter latency and emits variable bursts. The first two calls expose 3,252 samples (135.5 ms). Pipecat's 960-sample chunker exposes only three complete chunks, so immediately playable depth is **120 ms**, with 372 samples (15.5 ms) retained in the byte buffer.

For the common steady case where each later delta exposes two 40 ms chunks, substitute playable release depth `P=120`:

```text
Q_k^- ≈ (P - F) - k d = 40 - k d
```

At `d=2.789`, that approximation reaches zero after about 14.3 later arrivals. Exact replay is earlier because the beginning of this response has accumulated bursts of lateness: a 10 ms tick-phase simulation first empties at 0.84–1.01 s after release.

Once empty, queue depth cannot remain negative: `RawAudioTrack` emits a 10 ms zero chunk, advances its timestamp, and later audio resumes behind it. This makes the splice permanent in the content timeline. More importantly, every later deficit can trigger another zero tick. The exact 227-frame replay predicts 62–63 such ticks, not one large terminal underrun.

### Human-response replay

“First tick” is relative to buffer release. Subtract approximately 243–250 ms for time into audible speech. Counts span five possible track phases (0, 2.5, 5, 7.5, and nearly 10 ms).

| Session / response | Frames | Mean interarrival | Predicted 10 ms zero ticks at B=160 | First tick after release |
|---|---:|---:|---:|---:|
| `b2545a28` r1 | 34 | 77.882 ms | 0 | none |
| `b2545a28` r2 | 32 | 77.085 ms | 0 | none |
| `b2545a28` r3 | 40 | 83.155 ms | 9–10 | 1.72–1.89 s |
| `b2545a28` r4 | 34 | 81.638 ms | 1–2 | 2.44–2.53 s |
| `b2545a28` r5 | 54 | 81.355 ms | 4–5 | as early as 1.96 s |
| `3f6bc203` r1 | 34 | 81.534 ms | 0–1 | 2.28 s if phase triggers it |
| `3f6bc203` r2 | 227 | 82.789 ms | 62–63 | 0.84–1.01 s |

This replay assumes model `monotonic_s` spacing closely approximates Pipecat receipt spacing and negligible processing between receipt and output buffering. The model and Pipecat run locally across a WebSocket, but only client-side playout trace timestamps could close that remaining measurement gap.

## Leading silence and effective speech-onset lead

All seven human responses have the same onset pattern: three `-120 dBFS` model frames, then audible frame 4. The PCM does not jump at the frame boundary: the first sample of frame 4 and last sample of frame 3 are both zero.

The streaming-resampled first nonzero sample appears 242.5–250.1 ms into output content. By then, playback has consumed the initial 120 ms plus later chunks as they arrive. At 160 ms configured prebuffer, the complete, playable lead at first speech is only approximately:

- 30 ms in the slower cases where the next 80 ms contribution has not arrived before onset;
- 110 ms where it has just arrived;
- never a maintained 160 or 240 ms target.

Thus leading silence does consume the available transport lead, but it does not simply subtract 240 ms from a fixed buffer: generation continues during the silence, and discrete arrivals/chunk exposure determine whether another 80 ms is available at onset.

## Sample-rate conversion findings

Installed versions are Pipecat `1.7.1.dev21` at pinned commit `5415f28...`, `soxr 1.0.0`, `aiortc 1.15.0`, and `av 17.1.0`.

Pipecat creates one stream resampler when the media sender is constructed (`base_output.py:434-435`) and reuses it on every audio frame (`base_output.py:589-606`). `create_stream_resampler()` selects `SOXRStreamAudioResampler`. That object owns one `soxr.ResampleStream` and calls `resample_chunk()` on it (`soxr_stream_resampler.py:68-115`). Its own documentation states that history is retained to avoid chunk-boundary clicks.

It clears history after more than 0.2 s without a resample call. In the frozen last-40 cohort, none of 8,644 within-response gaps exceeds 200 ms; the maximum is 168.402 ms. In the human seven, none of 448 gaps exceeds 200 ms; maximum 95.782 ms. Therefore it does not clear at the 80 ms boundaries in question.

Chunk alignment is unusually favorable:

- `24000 / 22050 = 160 / 147`;
- `1764 = 12 * 147` source samples;
- steady-state output is exactly `12 * 160 = 1920` samples, exactly 80 ms at 24 kHz;
- 24 kHz has exactly 240 samples per 10 ms.

SOXR's streaming latency makes individual call outputs variable, but Pipecat's persistent byte buffer concatenates them before making exact 40 ms frames. There is no drop or pad at each 80 ms boundary. Only the terminal partial chunk is padded on `TTSStoppedFrame`; SOXR itself is not explicitly flushed, but each normal response ends in many silence frames, so the retained tail is silence in the examined responses.

aiortc's Opus encoder owns a second persistent `AudioResampler`, converting transport audio to Opus's 48 kHz. At the current 24 kHz this is an exact 2x conversion.

### Waveform checks

Across the seven human response PCMs:

- no sample clipped; maximum absolute raw sample was 8,168/32,768 (about -12.1 dBFS);
- per-response DC means were within 0.13 integer PCM least-significant bits of zero;
- raw 80 ms boundary jumps were not elevated (latest long response: boundary p95 324.5, max 1,258; all adjacent differences p95 412, max 3,970);
- after stateful SOXR, latest-long call-boundary jumps had p95 381 and max 1,449, versus all-adjacent p99.9 1,557 and max 3,583;
- silence-to-speech frame-boundary jump was zero in every response.

These measurements refute a discontinuity at every model frame boundary. They do not exclude codec/network loss, device playback faults, or a hard interruption splice.

### Why 22,050 Hz is not the fix

At 22,050 Hz, Pipecat and `RawAudioTrack` truncate the nominal 10 ms size to 220 samples. That is 9.9773 ms, not 10 ms. BaseOutput would make 880-sample chunks and preserve model samples through its residual byte buffer, so it is technically capable of running. A direct encoder probe also accepted 22,050 Hz input.

However, aiortc Opus always resamples to 48 kHz. The change would merely move the non-integer conversion from the explicit stateful SOXR stage to PyAV's stateful encoder resampler, while abandoning the exact 10 ms transport cadence. It is neither resampling-free nor supported by crackle evidence. Keep 24 kHz unless a controlled PCM/browser A/B proves otherwise. A 48 kHz transport would remove aiortc's second resample and retain exact 10 ms chunks, but it is likewise an optimization hypothesis, not a root-cause fix.

## Other click mechanisms

### Supported as possible but not measured here

- **Hard interruption splice.** Service and BaseOutput correctly discard pending interruptible audio, but there is no fade. Up to the already-submitted 40 ms WebRTC write may remain in `RawAudioTrack`. Cutting a nonzero waveform to silence can click. The reported uninterrupted long response does not require this mechanism.
- **WebRTC packet loss/jitter or browser/device output.** No `RTCPeerConnection.getStats()` loss/jitter data or browser PCM exists for the human report, so these remain open alternatives.
- **Slow same-response tool pause.** A pause over 200 ms could clear SOXR state without rearming the one-shot service buffer. It was not present in the relevant or last-40 traces.

### Not supported by the examined PCM

- clipping;
- DC-offset steps;
- periodic raw model-frame discontinuities;
- discontinuity at the first silence-to-speech frame boundary;
- SOXR state reset every 80 ms.

## Ranked recommendations

### 1. A/B the existing environment override at 800 ms — highest expected immediate impact

Set only `NEMOTRON_VOICECHAT_PREBUFFER_MS=800` in a controlled production-equivalent run, capture browser PCM and WebRTC stats, and compare against 160 ms. This needs no dependency upgrade or code change. In deterministic replay, 800 ms is the first 80 ms-aligned threshold that eliminates all predicted ticks in the 227-frame response; it also eliminates ticks in all 211 frozen last-40 responses.

Expected cost: release moves from delta 2 to delta 10, adding about eight interarrival intervals, roughly 0.65–0.67 s, before the UI's BotStarted event and audible response. Configured 720 ms still predicts 3–4 ticks late in the long response; 800 ms retains about 50 ms minimum margin in that exact replay. This is a strong mitigation and diagnostic, not a universal bound for arbitrary 30-second responses or future load.

### 2. Add direct underflow and queue-depth evidence before declaring final root cause — highest confidence gain

Instrument `RawAudioTrack.recv()` to count/ timestamp auto-silence while a TTS response is open, and record BaseOutput residual/full-chunk depth. Align those events with a browser PCM capture and `getStats()` packet-loss/jitter fields. The decisive test is whether audible crack timestamps coincide with source zero ticks. This recommendation is measurement work, not part of this analysis-only change.

### 3. Restore sustained generation below real time — best durable quality/latency fix

The model must average under 80 ms including scheduling/publication, not merely have `server_step_ms` near 80. The roughly 12–13 ms EARTTS decode stage accounts for the idle-to-speaking margin loss. Reducing that cost or other delivery overhead avoids accumulating debt without adding startup latency.

### 4. Replace one-shot buffering with an explicitly measured playout policy — architectural fix

A one-shot buffer is consumed permanently by positive drift. A production design should expose queue depth and choose a policy: a sufficiently sized startup jitter buffer based on response-length SLO, adaptive delay before speech, or controlled time-scale correction. Simply turning `audio_out_auto_silence` off is risky: the track timestamp remains behind its creation-anchored wall clock and may burst frames on recovery.

### 5. Treat sample-rate and cosmetic splice changes as experiments, not primary fixes

Do not switch to 22,050 Hz on the current evidence. Testing 48 kHz output, 10 ms BaseOutput chunks, or short fades around true interruption/underflow boundaries may reduce latency or click audibility, but none addresses the demonstrated cumulative arrival debt. A 10 ms BaseOutput chunk can recover at most about 30 ms of chunk-quantization headroom, far below the long response's roughly 0.63 s accumulated deficit.

## Reproducibility notes

Throwaway analysis scripts were written under `/tmp` only. They:

- streamed each recorded 1,764-sample frame through the installed Pipecat SOXR class;
- retained the exact 960-sample BaseOutput residual state;
- replayed model `monotonic_s` arrivals;
- consumed one 240-sample chunk at each 10 ms tick;
- swept five track phases;
- froze the dataset at `3f6bc203-...` to avoid later live-session contamination.

No files under `src/` or `tests/` were modified, dependencies were not changed, and no commit was created.
