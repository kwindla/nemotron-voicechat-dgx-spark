# PC-2 hotfix and regate — 2026-08-07

## 1. Disposition

**PROMOTE `nemotron-local/voicechat-vllm:promotion-candidate-2a`.**

Both PC2-1 fixes are implemented. The final image and its new immutable cache
were built from the expanded component-specific capture contract. All required
qualification criteria passed: the 1,389-call bitwise gate, retained 3/3
post-FC fixture, independent cache-validating cold start, recorded live-audio
arm, six-turn near-ceiling adversarial arm, exhaustive capture sweep, and fresh
sequential 900-second baseline/candidate soaks.

The production capture policy is now **graphless fallback plus persistent
observability**, never a default hard raise. Qualification uses
`VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE=1`; production defaults it to `0`.
Every candidate live and soak metric carried a zero fallback count.

Evidence root:

`/home/khkramer/src/nvidia-voice-chat/artifacts/container/dgx_spark/latest_voicechat/profiles/strict-v3-pair-20260806/pc2-hotfix-regate/`

## 2. Fix 1 — service-level response-start playout prebuffer

`NemotronVoicechatLLMService` now owns response-start playout buffering at the
`TTSAudioRawFrame` push path in `src/nemotron_voicechat_pipecat/llm.py`.

- Constructor parameter: `prebuffer_ms`, default `160`.
- Environment override: `NEMOTRON_VOICECHAT_PREBUFFER_MS`.
- The service withholds only the first audio frames of a response. At 160 ms,
  the normal two 80-ms chunks are released in order, followed by normal
  streaming with no further delay.
- A response that ends below threshold drains immediately before
  `TTSStoppedFrame`/response close.
- Existing speech-start and interruption paths discard an unplayed buffer
  immediately, preserving barge-in responsiveness.
- `0` disables buffering and retains immediate playout.
- Invalid or negative values fail configuration rather than silently changing
  behavior.
- `demo.py` is unchanged; application developers do not wire this feature.

Unit coverage verifies fill, ordered drain, short-response drain,
flush/discard on interruption, disabled-at-zero behavior, and environment
override. The full repository suite passed: **258 passed, 8 skipped**.

This directly addresses the measured response-start failure mode: playback no
longer begins with zero reserve against the 45/90-ms pair cadence and its
100–122-ms tail.

## 3. Fix 2 — live graph coverage without a production wedge

The Step-9 dispatcher patch is now version 2.

For an uncovered descriptor after application readiness:

1. Production (`VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE=0`) records the
   descriptor, emits `logger.warning_once(...)`, and returns vLLM's
   `CUDAGraphMode.NONE, None` graphless path.
2. Qualification (`VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE=1`) retains the hard
   `RuntimeError` so gaps fail a gate.
3. Neither branch captures or compiles a new graph, so
   `VOICECHAT_STEP9_ASSERT_NO_POST_READY_COMPILE=1` remains intact.

The dispatcher appends one JSON record per fallback to
`/tmp/voicechat-step9-fallbacks.jsonl`. The server exposes the container-lifetime
count and last descriptor as `step9_capture_fallback` in `/health`, every
`voicechat.metrics` event, and the persisted model trace. Thus coverage is
provably complete below the ceiling and production traffic above it is
rate-limited in logs, counted persistently, and expected to remain zero.

The final production configuration explicitly sets the hard coverage assertion
to `0`; every qualification engine arm overrode it to `1`.

An explicit final-image production-mode probe dispatched uncovered descriptor
`BatchDescriptor(num_tokens=769, uniform_decode=False, query_len=0)`. It returned
`NONE, None` without raising, emitted the single `warning_once` message, and
appended exactly one counter record containing token count 769. This directly
checks the monitored-above-ceiling path; all ordinary live traffic is still
expected to keep the counter at zero.

## 4. Live shape inventory and capture ceiling

### Manual PC-2 inventory

The retained five-turn session
`0d098361-b70e-4050-9b05-d4fb51ae6199` in
`~/.local/state/nemotron-voicechat/traces/pc2-manual-20260807/` produced this
agent-BOS growth curve:

| Turn | Model frame | Nano generated-token position | Nano stateful position | EarTTS position |
|---:|---:|---:|---:|---:|
| 1 | 30 | 31 | 30 | 1 |
| 2 | 150 | 150 | 149 | 1 |
| 3 | 241 | 240 | 239 | 1 |
| 4 | 317 | 315 | 314 | 1 |
| 5 | 391 | 388 | 387 | 1 |

Post-first-turn Nano growth was **+119, +90, +75, +73 tokens**, mean
**89.25 tokens/turn**. The other two traces reached two BOS events but reported
zero Nano position after PC2-1 had wedged the engine. The defect log and
investigation report record the exact uncovered turn prefill:
`BatchDescriptor(num_tokens=320, uniform_decode=False, query_len=0)`.

### Selected contract

The old contract was `257,80,16,2,1`. The new component-specific contracts are:

- Nano: **`768,512,384,257,80,16,2,1`**
- EarTTS: **`257,80,16,2,1`** (live prefills reset to one token)

The added Nano ladder points have explicit roles:

- `384` covers the observed 320-token defect and the remainder above 257.
- `512` covers the measured fifth-turn region around 388 with immediate
  headroom.
- `768` is the ceiling. Relative to the observed 388-token maximum, it adds
  380 tokens, or **4.26 further turns at the measured mean growth**. Projecting
  four such turns gives 745, just below 768.

The hard-assert adversarial replay independently reached its sixth agent BOS at
Nano position **731**, within 37 tokens of the ceiling, with no fallback. This
validates the selected ceiling against actual conversational growth rather than
only the original 320-token observation.

## 5. Capture cost, cache, and image

The closest old/new cache-source comparison isolates the Nano mixed-prefill
capture stage:

| Measurement | Old 5-size contract | New 8-size contract | Added cost |
|---|---:|---:|---:|
| Nano PIECEWISE mixed-prefill capture | 0.68 s | 2.30 s | **+1.62 s** |
| Nano total graph-capture log | 67 s | 71 s | +4 s |
| Nano reported graph memory | 0.89 GiB | 0.97 GiB | **+0.08 GiB** |

The +1.62 s is the directly attributable three-size capture addition. The
coarser total includes variable decode/full-graph work and is reported without
assigning all four seconds to the new sizes. The fresh uncached cache-source arm
reached ready in 362.703 s versus 364.735 s for the prior cache source, showing
that whole-start variance is larger than this isolated addition. This is well
below the removed sweep's measured 28.2-second cost.

The old cache tree was not reused. The final cache was generated from a fresh
source arm and sealed under the new component-specific contract:

- Cache key:
  `pc2a-be0e0af6-serverc19e83bb-vllm0171-gb10sm121-nano768-512-384-257-80-16-2-1-eartts257-80-16-2-1-v2`
- Files: 2,865
- Payload: 121,965,805 bytes
- Manifest SHA256:
  `7d981de484fffa75afed6d3fc16c2863d504af5568407b4138c799421cf8cac1`
- Final image ID:
  `sha256:ba12b10103f669a70d6294335f2afd3a02f0419b043e89a9a421d179dacc574b`
- Final image size: 35,494,008,306 bytes

The cache guard now includes both Nano and EarTTS capture inventories in its
runtime contract, preserving fail-closed key/manifest validation.

## 6. Mandatory gates

| Gate | Result | Evidence/result |
|---|---|---|
| A. Step-7-method same-process pair gate | **PASS** | 1,389/1,389 calls; all outputs and decisions bitwise equal; 675 cache/rollback-shadow checks; first mismatch `null`. Final decisions: 663 accepted, 12 rejected, 664 buffered, 49 conditional singles. |
| B. Retained post-FC fixture | **PASS (functional 3/3)** | 3/3 typed turns completed and answered; both FC turns made exactly two calls; middle turn returned `five`; all three produced audio; zero unexpected errors. |
| C. Cold start/cache validation | **PASS** | Independent final-image ready in **247.936 s**; exact cache validation succeeded; fallback count 0; preflight 117.995 GiB; minimum 46.678 GiB; no watchdog action. |
| D. Recorded live-audio arm | **PASS by specified live criterion** | Four fake-microphone/replay WAV turns; 4/4 transcribed, answered, and audible; zero protocol/EngineCore execution errors, zero uncovered warnings/fallbacks; normal client stop. |
| E. Fresh sequential release soak | **PASS** | Baseline then 2a, 900 s each, 11,250 source frames each, 8,882/8,927 settled positive frames; both queue gates passed. |
| F. Exhaustive capture sweep | **PASS** | Under hard assert, every raw prefill token count 1…768 dispatched through a graph; decode shape 1 and pair-decode shape 2 both dispatched FULL; zero failures. |
| Adversarial long live arm | **PASS** | Six replay turns, 6/6 audible, BOS growth to position 731, hard assert enabled, zero fallback, normal client stop. |
| Production fallback probe | **PASS** | Uncovered descriptor 769 returned graphless, logged once, persisted one counter record, and did not raise or compile. |

The post-FC sustained harness's aggregate `passed` field is false because its
third response exceeded the auxiliary serialized-response timeout and its
settled queue slope was +2.245 ms/min. That is not the retained functional
fixture criterion: the third response subsequently completed, all 3 jobs were
answered, both FC turns issued exactly two calls, and there were zero unexpected
errors. The result is disclosed rather than relabeled.

Likewise, the generic live replay report's aggregate is false only because two
fixture-specific semantic word lists were narrower than valid responses (“I am
not sure…” and “I can generate…”). The user-defined live gate requires audible
answers and engine/capture health, not those semantic tokens; all four turns had
nonzero audio, valid text, acceptable ASR WER, normal completion, and zero
fallback.

Startup logs retain the pre-existing non-fatal optional
`lora_hf_hub_resolver` import diagnostic. There were zero post-ready EngineCore
execution failures, descriptor assertions, uncovered warnings, or post-ready
compile assertions in the live/adversarial/candidate-soak arms.

### Exhaustive rounding coverage

The dense sweep's raw-to-capture histogram was:

| Capture size | Raw token counts mapped |
|---:|---:|
| 1 | 1 |
| 2 | 1 |
| 16 | 14 |
| 80 | 64 |
| 257 | 177 |
| 384 | 127 |
| 512 | 128 |
| 768 | 256 |

All 768 raw prefills and both decode descriptors had a non-`NONE` graph key.
This proves there are no rounding-ladder holes below the ceiling.

## 7. Live-audio frame evidence

The four-turn recorded replay used the retained
`/tmp/voicechat-manual-replay-fixture/manifest.json`, including the
`turn-math.wav` class. Each 80-ms input frame passed through the strict-v3
`input_audio_buffer.append` fake-microphone path.

| Turn | Turn-start transport frame | Agent-BOS transport frame | Agent-BOS model frame | Start→BOS |
|---:|---:|---:|---:|---:|
| 1 | 13 | 34 | 31 | 21 frames / 1,680 ms |
| 2 | 115 | 158 | 155 | 43 frames / 3,440 ms |
| 3 | 236 | 281 | 278 | 45 frames / 3,600 ms |
| 4 | 416 | 440 | 437 | 24 frames / 1,920 ms |

Audible response payloads were 197,568; 190,512; 391,608; and 229,320 bytes.
Input WERs were 0.25, 0, 0, and 0. The persistent fallback counter remained 0
in every metric and final health.

The six-turn adversarial arm's agent-BOS model frames were
31, 153, 276, 436, 580, and 737; corresponding Nano positions were
31, 151, 273, 433, 576, and 731. All six responses were audible.

## 8. Fresh sequential soak results

Method: select `voicechat.metrics` with `server_step_ms > 0`, discard the first
1,500 positive frames independently per arm, use linear percentiles, and form
each sliding two-frame sample as `(frame[i] + frame[i+1]) / 2`. Arms ran in the
mandated order with no overlap.

### Raw per-frame time

| Statistic | production-candidate-1 | promotion-candidate-2a | 2a − baseline |
|---|---:|---:|---:|
| samples | 8,882 | 8,927 | +45 |
| mean | 69.473 ms | 65.595 ms | **−3.878 ms (−5.582%)** |
| p50 | 80.247 ms | 48.527 ms | −31.720 ms (bimodal boundary shift) |
| p95 | 98.078 ms | 95.134 ms | **−2.944 ms (−3.002%)** |
| p99 | 106.564 ms | 103.005 ms | **−3.559 ms (−3.340%)** |

The large raw-p50 change reflects the known bimodal distribution: candidate
had 4,325/8,927 settled frames above 80 ms versus baseline 4,452/8,882, moving
the median from the high mode to the low mode. It is not interpreted as a
31.7-ms uniform speedup.

### Sliding two-frame time

| Statistic | production-candidate-1 | promotion-candidate-2a | 2a − baseline |
|---|---:|---:|---:|
| windows | 8,881 | 8,926 | +45 |
| mean | 69.471 ms | 65.592 ms | **−3.879 ms (−5.584%)** |
| p50 | 69.897 ms | 67.466 ms | **−2.431 ms (−3.478%)** |
| p95 | 75.348 ms | 73.525 ms | **−1.823 ms (−2.419%)** |
| p99 | 82.236 ms | 82.579 ms | +0.343 ms (+0.417%) |

Candidate had 118 two-frame windows above 80 ms versus baseline 174. The
governing two-frame p95 passes 80 ms on both and improves on 2a. The small p99
regression is reported and does not outweigh the mean/p95/raw-p99 improvements.

### Functional, queue, and memory soak results

| Item | baseline | candidate 2a |
|---|---:|---:|
| duration / source frames | 900 s / 11,250 | 900 s / 11,250 |
| metric / positive frames | 10,385 / 10,382 | 10,429 / 10,427 |
| typed completed / answered | 15 / 15 | 15 / 14 |
| unexpected errors | 0 | 0 |
| first / last block mean | 60.347 / 70.616 ms | 57.149 / 68.249 ms |
| queue slope | +0.386 ms/min | +0.357 ms/min |
| peak implied debt | 308.430 ms | 381.821 ms |
| queue gate | pass | pass |
| startup ready | 382.889 s | 256.016 s |
| minimum MemAvailable | 42.638 GiB | 42.571 GiB |

The candidate's one unanswered typed turn is within the harness's permitted
one-turn allowance at the fp32 base injection rate; all 15 injections completed,
metrics stayed continuous, and the session closed normally. Candidate final
health and all soak metrics reported fallback count 0.

## 9. Guardrails and recommendation

Before the first promotion launch, the manual bot/ngrok processes were stopped,
`voicechat-pc2-manual` was stopped, and `docker ps -q` was verified empty.
Every engine launch used Nano `0.35`, EarTTS `0.10`, Docker
`--memory 90g --memory-swap 90g`, an empty-container and >100-GiB preflight,
and a 10-second watchdog with a 15-GiB kill threshold. No two engines overlapped
and no watchdog fired.

The new ladder closes the reproduced PC2-1 hole, the dense sweep proves every
shape through 768, real and adversarial live traffic remains at zero under hard
assert, and production can no longer permanently wedge on an unforeseen shape.
The residual above-ceiling risk is bounded by graphless fallback and continuously
visible through the persistent counter and rate-limited warning.

**Recommendation: PROMOTE promotion-candidate-2a.**
