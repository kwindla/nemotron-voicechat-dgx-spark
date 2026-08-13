# STEP 5: measure host barriers, do not delete by count

Date: 2026-08-07 UTC

## Result

STEP 5 is complete. Phase 1 found real host barriers, but it did **not** justify removing any of the seven `torch.cuda.synchronize()` calls. The dominant 10.4–10.7 ms perception wait is queued prerequisite GPU work; removing that synchronization would move the wait to the next dependent consumer. The other six synchronization sites are small and likewise have no independent work available before their values are consumed.

Phase 2 therefore leaves all seven synchronizations intact and changes only measured recoverable host work:

1. the ten early RNNT scalar reads are coalesced into one raw-byte packed tensor and one D2H transfer;
2. the one-token EarTTS BOS mask and zero speaker latent are reused, with the original dynamic path retained for non-serving shapes;
3. one-token PAD detokenization returns the same empty string without tokenizer work.

The correctness and integration gates pass. The bounded end-to-end result is nevertheless negative: candidate run-mean wall time is **70.113 ± 0.383 ms versus 69.367 ± 0.101 ms**, a candidate-minus-baseline regression of **+0.747 ms**. Pooled p50/p95 regress by **+0.226/+3.962 ms**. This branch is useful as a measured, exact implementation, but the combined change is **not recommended for latency promotion** on this evidence. A follow-up would have to isolate the three changes and explain the regression; it must not claim the small local savings as an end-to-end win.

The 12.727–16.683 ms EngineCore/RPC-wait category was observed as prior context only. No EngineCore, RPC, process-boundary, scheduling, manifest, or production-configuration code was changed.

## Provenance and isolation

- source repository: `/home/khkramer/src/nemotron-voicechat-dgx-spark`
- isolated worktree: `/home/khkramer/src/nemotron-voicechat-dgx-spark-step5`
- starting HEAD: `4fd86fa203961cce85c19c874d23049cc371adef`
- branch: `codex/step5-host-barriers`
- commit: `415d59648e8ce62a5b0b7094a0f3a3f6da2ebf90`
- production baseline image: `sha256:6294053537329500a17a1684421cd3909224babd7423d677c6c5d6bceead0482`
- measurement image: `sha256:078b7efa311f3eeea5b4841d188434c5edee0522501f11d4e1af140fc3126f18`
- optimized image: `sha256:2c417392a28afd69e03699426f73edfc3450c152b326a183cd9935059a2dcc0e`

The overlay fails closed unless the three baked Speech inputs have the qualified hashes `bb194225…`, `a74dbeb…`, and `b05c7efa…`. It also regenerates the image's retained Speech diff rather than weakening the runtime provenance check.

Frozen inputs were unchanged at the end:

| input | SHA-256 |
|---|---|
| `config/qualified-candidate-1.json` | `a7361ef00df0b34de900ef1d8b08658915608501b6a548bbcf5bb143729e5c84` |
| `config/artifact-release.json` | `628dcf0eb3de8ffc635e0e647151f4d0ed8231120628e4d482508020ce981899` |
| `config/production-candidate-1.toml` | `434692a7c1563c488ac249b0be0d3447c570d48875b97577df59981d5a06b5f6` |

## Phase 1 method

The measurement image adds opt-in `VOICECHAT_HOST_BARRIER_PROFILE=1` spans around the seven serving synchronizations, ten early RNNT scalar reads, the two fresh EarTTS allocations, and detokenization. Each synchronization/scalar span records monotonic host time and CUDA-event time; allocation and detokenization spans record monotonic host time. The flag defaults off and creates no events, additional synchronization, captured values, or log records when off. More importantly, the optimized A/B image does not contain the Phase-1 helpers at all, so measurement overhead is exactly absent from both timed arms.

The valid strict-v3 measurement session was `e9915410-339e-48dd-beb2-0fc7990faff1`: 1,039 model frames and 20,780 in-session span records. Frame classes are derived from the fixture trace: 519 accepted PAD-pair frames, 511 buffered PAD-pair frames, six serial-content frames, plus one BOS pending drain, one EOS bypass, and one unclassified terminal frame. The table below reports host wait as `mean / p50 / p95 / max` in milliseconds for accepted PAD, buffered PAD, and serial-content frames. Single-observation control rows remain in `barrier-summary.json` and are not generalized here.

| barrier site | accepted PAD ms | buffered PAD ms | serial content ms | recovery classification and Phase-2 action |
|---|---:|---:|---:|---|
| `perception.substep_sync` | 10.3765 / 10.3567 / 10.5268 / 11.2389 | 10.6957 / 10.5176 / 11.3064 / 11.8032 | 10.6893 / 10.5816 / 11.1316 / 11.1849 | **not recoverable**: next substep/consumer depends on the queued work; retained |
| `perception.encoder_sync` | 0.0076 / 0.0060 / 0.0202 / 0.0623 | 0.0074 / 0.0060 / 0.0184 / 0.0378 | 0.0144 / 0.0145 / 0.0161 / 0.0163 | **not recoverable**; retained |
| `wrapper.perception_sync` | 0.0053 / 0.0041 / 0.0146 / 0.0890 | 0.0054 / 0.0041 / 0.0160 / 0.0482 | 0.0128 / 0.0126 / 0.0143 / 0.0147 | **not recoverable**; retained |
| `wrapper.stt_model_sync` | 0.0396 / 0.0389 / 0.0486 / 0.1382 | 0.0080 / 0.0061 / 0.0244 / 0.0509 | 0.0277 / 0.0190 / 0.0472 / 0.0480 | **not recoverable**; retained |
| `wrapper.tts_model_sync` | 0.0240 / 0.0230 / 0.0334 / 0.0486 | 0.0322 / 0.0319 / 0.0415 / 0.0672 | 0.0332 / 0.0362 / 0.0381 / 0.0381 | **not recoverable**; retained |
| `wrapper.audio_codec_sync` | 0.0088 / 0.0078 / 0.0177 / 0.0372 | 0.0340 / 0.0339 / 0.0457 / 0.1001 | 0.0173 / 0.0162 / 0.0218 / 0.0235 | **not recoverable**; retained |
| `wrapper.one_step_sync` | 0.0081 / 0.0054 / 0.0229 / 0.2131 | 0.0139 / 0.0110 / 0.0333 / 0.3026 | 0.0139 / 0.0137 / 0.0150 / 0.0151 | **not recoverable**; retained |
| `rnnt.blank_count.item` | 0.0097 / 0.0082 / 0.0214 / 0.0372 | 0.0093 / 0.0076 / 0.0222 / 0.0336 | 0.0222 / 0.0201 / 0.0302 / 0.0334 | value wait is required; redundant transfer overhead recoverable; packed |
| `rnnt.nonblank_consec.item` | 0.0088 / 0.0067 / 0.0192 / 0.1734 | 0.0082 / 0.0067 / 0.0196 / 0.0560 | 0.0185 / 0.0184 / 0.0191 / 0.0191 | same; packed |
| `rnnt.nonblank_total.item` | 0.0073 / 0.0053 / 0.0183 / 0.0677 | 0.0074 / 0.0064 / 0.0188 / 0.0287 | 0.0186 / 0.0185 / 0.0200 / 0.0204 | same; packed |
| `rnnt.speech_confirmed.item` | 0.0087 / 0.0057 / 0.0189 / 0.1691 | 0.0071 / 0.0056 / 0.0188 / 0.0374 | 0.0181 / 0.0182 / 0.0188 / 0.0189 | same; packed |
| `rnnt.first_turn.item` | 0.0073 / 0.0053 / 0.0186 / 0.0560 | 0.0067 / 0.0053 / 0.0186 / 0.0365 | 0.0177 / 0.0178 / 0.0183 / 0.0184 | same; packed |
| `rnnt.rolling_density.item` | 0.0099 / 0.0080 / 0.0197 / 0.0855 | 0.0089 / 0.0079 / 0.0188 / 0.0232 | 0.0186 / 0.0184 / 0.0197 / 0.0200 | same; packed |
| `rnnt.user_first_turn.item` | 0.0077 / 0.0054 / 0.0181 / 0.1425 | 0.0068 / 0.0054 / 0.0183 / 0.0295 | 0.0192 / 0.0176 / 0.0257 / 0.0281 | same; packed |
| `rnnt.agent_speaking.item` | 0.0076 / 0.0051 / 0.0180 / 0.1716 | 0.0063 / 0.0050 / 0.0179 / 0.0280 | 0.0168 / 0.0168 / 0.0177 / 0.0178 | same; packed |
| `rnnt.current_token.item` | 0.0071 / 0.0052 / 0.0181 / 0.0689 | 0.0069 / 0.0051 / 0.0182 / 0.2010 | 0.0180 / 0.0177 / 0.0193 / 0.0196 | same; packed |
| `rnnt.forced_bos.item` | 0.0072 / 0.0053 / 0.0182 / 0.0698 | 0.0068 / 0.0052 / 0.0183 / 0.0310 | 0.0170 / 0.0170 / 0.0176 / 0.0176 | same; packed when present |
| ten RNNT reads, summed site means | **0.0811** | **0.0745** | **0.1848** | nine redundant scalar materializations are recoverable; one packed D2H remains causally required |
| `eartts.alloc_bos_mask` | 0.0170 / 0.0133 / 0.0353 / 0.1843 | 0.0159 / 0.0134 / 0.0395 / 0.0522 | 0.0313 / 0.0301 / 0.0342 / 0.0344 | **recoverable pure host work**; preallocated for one-token decode |
| `eartts.alloc_speaker_latent` | 0.0078 / 0.0061 / 0.0141 / 0.1434 | 0.0075 / 0.0064 / 0.0175 / 0.0294 | 0.0122 / 0.0121 / 0.0132 / 0.0135 | **recoverable pure host work**; preallocated after prompt shape is known |
| `wrapper.detokenize` | 0.1433 / 0.1169 / 0.3772 / 0.5096 | 0.2235 / 0.1899 / 0.5854 / 0.8051 | 0.3368 / 0.3391 / 0.3433 / 0.3437 | **recoverable on one-token PAD**; narrow empty-string fast path |

For the large perception wait, CUDA event means are only 0.017–0.027 ms because the event pair brackets the synchronization call itself while monotonic time includes the already-queued prerequisite work. That distinction is the central I2 result: deleting the synchronization would not make the prerequisite disappear.

## Phase 2 exact changes

### Seven synchronization sites

No change. Although their comments/logging make them look like timing synchronizations, Phase 1 found no fillable interval between these sites and the next dependent consumer. They were not demoted behind a debug guard merely because the count was seven.

### RNNT scalar transfer

The candidate byte-views each source scalar in its native dtype, concatenates those raw bytes on device, performs one `.cpu().numpy().tobytes()` D2H, and unpacks `=qqq??f??q` plus the optional forced-BOS boolean. No scalar is numerically cast. `VOICECHAT_RNNT_PACK_VERIFY=1` is an opt-in gate only: it re-reads the legacy ten values, packs them with the identical format, and fails closed on any byte mismatch. This verifier was off in all performance runs.

The later post-EOS fallback `.item()` outside the measured early block was not changed. The task was not treated as a textual `.item()` deletion count.

### EarTTS constants

The one-value fp32 BOS epsilon is allocated once at `VllmEARTTSModel` construction. The one-token fp32 zero speaker latent is allocated after the prompt reveals the production latent width (1,024). Non-one-token shapes retain the original `full_like`/`zeros` path, so prefill and unusual shapes are not broadened.

### PAD detokenization

Only batch size one, chunk size one, current-token PAD takes the fast path. Baseline maps ID 12 to `<SPECIAL_12>`, removes that token, and calls `tokens_to_text([])`, producing `""`; the candidate directly returns `[""]`. All non-PAD and broader shapes retain the original tokenizer path.

## Correctness and integration gates

Different gate levels are stated explicitly rather than conflated.

| change / output | gate level | result |
|---|---|---|
| packed RNNT scalar bytes | real wrapper fixture, opt-in oracle | pass: 110 live `model_step` frames, zero `voicechat_step5_rnnt_packed_scalar_byte_mismatch` |
| output token and turn-taking decision trace | deterministic same-process baseline/candidate method execution | pass: 4,096 mixed cases, exact generated-token, mutable-state, and wrapper-decision equality; trace SHA-256 `31f06cd4af11c4d9fc2509c88be875ac0d6dd88359e210eb76bb8f793386482a` |
| BOS mask bytes | first-affected-input, same process | pass: shape `[1]`, fp32, SHA-256 `6dff7c7c50f1543854765270504719763dad4409b22e16a281c71077ee265878` |
| speaker latent bytes | first-affected-input, same process | pass: shape `[1,1024]`, fp32, SHA-256 `ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7` |
| live tokens/control/acoustic codes before process-RNG divergence | separate-process fixture negative control | 56 substantive frames exact, then baseline emitted BOS while candidate remained PAD; not used as an acceptance gate |
| bounded acoustic fixture | fresh candidate production component | pass: 135 frames, non-eager vLLM, ASR `The answer is five.`, WER 0 |
| acoustic codes | same-process first-affected-input/decision gate | pass with exact bytes and no tolerance; separate-process component files differ in 7 entries across 2 frames and are explicitly invalid as a fixed-RNG gate per STEP 3 |
| streaming waveform | production cached-codec component versus STEP-3 baseline | pass, byte equal, SHA-256 `89168aa27da6ac603486213162526859c595a0727ffc50e44c9c49b81603e9d8` |
| offline waveform | production whole-history component versus STEP-3 baseline | pass, byte equal, SHA-256 `4faea495879404e5ea272a0e0369710da0355900ff94f296d0dda47910dd8c9a` |
| vLLM integration | fresh optimized-image startup | pass: all 35 mixed PIECEWISE and 19 FULL CUDA-graph captures, then 135-frame component completion |

The separate live baseline and candidate outputs were “The current UTC time is two plus three hours, which is five AM UTC.” and “The sum of two and three is five.” That mismatch is preserved, not hidden. STEP 3 established that an unchanged vLLM baseline differs from its own repeat across separate launches even with the same seed, and the Phase-1 measurement baseline also differed from the uninstrumented baseline here. Consequently, cross-process live output equality is an invalid exact-RNG gate. The same-process token/decision and first-affected-input gates are authoritative; the fresh component run closes waveform and integration coverage.

No correctness gate failed after the final candidate was built. Therefore no Phase-2 change was reverted for correctness. The cross-process mismatch is a negative control, not a failed exact gate.

## Bounded uninstrumented A/B

Method matches STEP 3/4:

- loopback websocket at `127.0.0.1:8786` only;
- GPU idle before each arm;
- one resident server per image, three independent 90 s client runs;
- absolute 80 ms pacing, one dense typed prompt at frame zero, 20 s drain;
- existing trace environment enabled;
- no profiler, Phase-1 helper, acoustic capture, or RNNT verifier in either timed arm.

All six runs had complete metric sequences, one answered prompt, `client_stop`, and no unexpected errors. `passed: false` means only that a bounded 90 s run cannot reach the harness's unchanged 1,500-settled-frame endurance threshold.

| arm | run | server frames | mean ms | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1,050 | 69.363 | 45.128 | 46.483 | 86.777 | 91.735 | 94.824 | 99.144 | 160.141 |
| baseline | 2 | 1,052 | 69.245 | 45.076 | 46.445 | 79.781 | 91.821 | 95.314 | 99.923 | 157.803 |
| baseline | 3 | 1,061 | 69.493 | 45.596 | 46.713 | 80.513 | 91.983 | 93.664 | 100.562 | 157.916 |
| candidate | 1 | 1,032 | 70.650 | 45.340 | 46.469 | 80.202 | 92.537 | 99.859 | 101.019 | 386.832 |
| candidate | 2 | 1,057 | 69.782 | 45.631 | 46.854 | 80.968 | 92.631 | 94.852 | 99.584 | 160.797 |
| candidate | 3 | 1,058 | 69.909 | 46.225 | 47.603 | 84.036 | 91.930 | 94.440 | 96.773 | 156.218 |

| distribution | baseline | candidate | candidate − baseline |
|---|---:|---:|---:|
| run-mean wall, mean ± population std | 69.367 ± 0.101 | 70.113 ± 0.383 | **+0.747 ms** |
| pooled wall mean | 69.367 | 70.109 | **+0.742 ms** |
| pooled wall p50 | 80.742 | 80.968 | **+0.226 ms** |
| pooled wall p95 | 94.686 | 98.648 | **+3.962 ms** |
| pooled wall p99 | 99.956 | 100.870 | **+0.915 ms** |
| run-mean EarTTS host stage | 13.118 | 13.480 | **+0.362 ms** |

Positive deltas are regressions. Candidate run 1 contains a 386.832 ms accepted-PAD outlier; it remains in pooled and run-mean results. Removing that run or outlier is not justified. Even candidate runs 2–3 do not establish a win over the tight baseline distribution.

This negative end-to-end result is compatible with Phase 1: the locally recoverable work is sub-millisecond, while the critical path is dominated by alternating PAD-pair execution, queued perception work, and the unchanged EngineCore/RPC boundary. “Measured recoverable” does not imply “measurable application win.”

## Failed/setup attempts retained

1. The first measurement image launch correctly failed the server's retained-Speech-diff provenance check because the initial overlay changed qualified Speech source without regenerating the retained diff. The build was fixed to hash-check and regenerate that diff; the valid measurement was rerun from a clean server.
2. The first fixture invocation was given an already-created output directory and stopped before a session. It is preserved under `phase1/fixture-failed-precreated`; it is not included in any distribution.
3. Separate-process live/component token and code mismatches are preserved as negative controls. Per the established STEP-3 baseline-repeat evidence, they are not valid exact-RNG gates and were not “fixed” with a tolerance.

## Cleanup

All STEP-5 servers and component containers were stopped. Final checks found:

- no NVIDIA compute process;
- GPU 0% utilization, 208 MHz, P8;
- no listener on 8786, 7860, or 4040;
- no profiling process, tunnel, or STEP-5 container left running.

Pre-existing stopped/created containers were not deleted because they are outside this task's ownership.

## Evidence index

Root: `artifacts/container/dgx_spark/latest_voicechat/profiles/strict-v3-pair-20260806/step5/`

- `phase1/server.log`, `phase1/fixture/`, `phase1/barrier-summary.json`: valid measured session and all 20 site distributions
- `phase1/fixture-failed-precreated/` and Phase-1 build/failure logs: retained setup evidence
- `gate-baseline/`, `gate-candidate/`, and their server logs: bounded live wrapper fixtures
- `gates/rnnt-live-verifier.txt`: real-fixture packed-byte oracle
- `gates/same-process-gate.json`: 4,096-case exact token/turn-state and constant-byte gate
- `gates/live-cross-process-comparison.json`: 56-frame exact prefix and explicit RNG negative control
- `gates/candidate-component/`, `gates/component-comparison.json`: fresh production component, codes, WAVs, ASR, and baseline comparison
- `perf-baseline/`, `perf-candidate/`, `perf-ab-summary.json`: all six uninstrumented timed runs and honest distributions
- `provenance/`: image IDs, frozen hashes, clean commit, source diff, and static validation
- `cleanup/` plus each arm's `post-stop.txt`: final machine state

