# Step 4b telemetry regression — 2026-08-13

> Supersession note: the prior completion note `step4a-done.md` is withdrawn; this repaired report and `step4-trace-overhead-ab-20260813.md` are the sole measurement record.

## Result

The 300 s sustained soak completed cleanly, but it does **not** establish or
refute a thermal, DVFS, CPU-pressure, or memory-pressure cause for
delivered-generation cost. On the qualified `delivered-nonBOS` population and
`server_step_ms` clock, the strongest raw associations were CPU PSI `avg300`
(Pearson r=0.772; response-cluster bootstrap 95% interval 0.105–0.905) and GPU
temperature (r=0.466; interval 0.004–0.822). Both co-moved with run time and
response order. Linear-time-adjusted correlations were 0.243 and −0.623, while
within-response demeaned sensitivities were −0.215 and −0.026. These are
separate exploratory views: the sign change neither establishes nor refutes a
thermal/pressure effect, and no ranking conclusion follows from it.

The nominal population has 293 frame rows, but effective support is only **31
joined telemetry samples and seven responses**. CPU PSI `avg300` and integer
GPU temperature have **7 and 10 unique joined values**, respectively; all 293
rows have observed values for both channels. Seven response clusters inside one
run do not estimate run-to-run uncertainty. The retained bootstrap intervals
apply only to the raw frame correlation; adjusted-statistic uncertainty was not
computed in this retained v1 result and is unavailable rather than implied by
the adjacent raw interval.

The narrower operational observations are: all five pinned big cores stayed at
3,900,000 kHz in all 331 samples; GPU SM clock did not downclock in the measured
generation range (2,457–2,535 MHz, r=−0.215; time-detrended r=0.099); memory and
I/O PSI averages were identically zero. The v1 nearest-sample instantaneous
CPU-stall sensitivity was r=−0.012 (95% interval −0.221–0.274), but its interval
counter join is not causally ordered and is only a coarse descriptive check.
Integrated-GPU memory clock,
memory-used/total, and power-limit fields were reported by `nvidia-smi` as
`[N/A]` in all 331 samples and remain null.

There were **zero** delivered-generation rows at `server_step_ms >= 88.0` (n=0;
membership SHA-256
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`).
Spike-specific regression is therefore unavailable rather than green.  All
numbers in this section come from
`session-20260813T150500Z/telemetry-regression.json`, population
`delivered-nonBOS`, clock `server_step_ms`; the stratum itself comes from the
qualified Step 0 analyzer output `stratified-analysis.json`.

Thus this run supports only “no observed GPU clock collapse, pinned-core
frequency change, or host memory/I/O pressure in the recorded channels.”
Constant frequency does not exclude scheduler contention, and the late
codec/residual shift coincides with the highest global CPU `avg300`; CPU
pressure is not deprioritized by this run.

## Completion conclusion

Step 4b is a single-run descriptive observation, not a causal or robust-null
candidate-ranking input. It does not deprioritize thermal, DVFS, or CPU-pressure
work beyond the narrow recorded-channel observations above. Step 4c uses the
counterbalanced incidence-first design below; the browser/WebRTC EarTTS-high
interaction claim is withdrawn.

## Qualified soak and provenance

| item | retained value |
|---|---|
| accepted wall interval | 2026-08-13 15:10:27.695797799–15:15:57.744095864 UTC |
| direct model session | `3c888978-02de-4b81-982b-10d696c86328` |
| source duration / cadence | 300.0 s / 80 ms PCM frames; typed prompt every 45 s; 15 s drain |
| runner result | PASS; 3,750 sent frames; 3,650 contiguous metric frames; 7/7 typed jobs answered; clean `client_stop` |
| telemetry | 331 samples spanning 330.000492 s; requested interval 1.0 s; observed min/mean/max 0.999166/1.000001/1.000764 s |
| model image | `pipecat-ai/nemotron-voicechat-dgx-spark:generation-step2`; image ID `sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a` |
| container | `nemotron-voicechat-model`, ID `6efe0b3ed9e24df9830a3b5666652ec57d3991d01a3bbf4a876b2dfd1b7ff650`; not restarted or replaced |
| checkout commit at preflight | `e5924c7b57503ea03e2a72d104f8decfbfd0f46d` |
| host runner | project package 0.1.0 in `.venv`; Python 3.12.13 |
| runtime server source | health/runtime provenance SHA-256 `8f5855d5d62529367d574836caae99b36dda31c06a2aea60953bcf117a975a87`; Speech commit `911ec674ab40f04302ef33672be4179f45a7310f` |
| checkpoint | `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` revision `fb0f94eaf4d03ddc430f39565229393fa1b50c26`; model SHA-256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b` |
| qualification deltas | `VOICECHAT_QUALIFICATION_MODE=step2-longform-fixture`; `VOICECHAT_QUAL_NO_TEXT_WATCHDOG_OVERRIDE_FRAMES=160`; qualification system prompt recorded in health/session-created evidence |
| scheduler/runtime | sequential: `VOICECHAT_NANO_PAD_PAIR=0`, conditional=0, control barrier=0; BF16 perception; CPU codec process on cores 5,6; typed input on cores 7,8,9,15 |
| immediate lock evidence | `preflight/retry-health-immediately-before.json`: `active_client:false` |
| exact full configuration | `preflight/container-inspect.json`, `preflight/image-inspect.json`, and the runner's retained `session.created` record in `soak/events.jsonl` |

The exact server command retained in `container-inspect.json` was:

```text
python3 -m nemotron_voicechat_runtime.server --speech-root /opt/Speech --checkpoint-root /models/voicechat --hf-skeleton /models/NVIDIA-Nemotron-Nano-9B-v2 --vllm-manifest /models/derived/manifests/release.json --nano-vllm-path /models/derived/nano --eartts-vllm-path /models/derived/eartts --speaker-name Aria --host 0.0.0.0 --port 8786 --trace-dir /trace
```

Retained paths are relative to
`reports/step4-telemetry-regression-20260813-data/session-20260813T150500Z/`
unless absolute:

- raw telemetry: `telemetry.jsonl`, SHA-256
  `a1b745b1399927be490ea0b7f9d88c59f88ba1c9ff4b740a32d87300ebf6169f`;
- client events/report: `soak/events.jsonl` / `soak/report.json`, SHA-256
  `3607f180c19f982dd5cf48b72aecbc092ba67c78107f5f9a10b61ee4b7288aa8` /
  `119d7f1af75120158ab077045d506a557e4a9d2e445430f71f25f53585bac889`;
- server model trace:
  `~/.local/state/nemotron-voicechat/traces/model/3c888978-02de-4b81-982b-10d696c86328/events.jsonl`,
  SHA-256 `72d4dff418111e4c1034ec57c82f8f35cde597023cb8ef640dc2efebeb422834`;
- qualified strata: `stratified-analysis.json`; joined regression:
  `telemetry-regression.json`; hashes are in the refreshed `artifacts.sha256`.

## Clock mapping and join

Each poll sampled `time.monotonic()` and `time.time()` immediately before the
telemetry reads.  Linux containers share the host monotonic clock namespace, so
server `model_step.monotonic_s` joins directly to the nearest telemetry
`monotonic_s`; wall time is reconstructed as `monotonic +
wall_minus_monotonic`.  Across 331 samples, that offset's range was 0.003099 ms.
Every analyzer-issued member was checked back against its server-trace event
ordinal and frame before joining. The retained v1 tool did not independently
verify member timestamps or exact full membership equality; the repaired tool
now does both. The maximum permitted nearest-sample
distance was 550 ms; the delivered-generation join had n=293, mean 249.881 ms,
p95 478.242 ms, and max 499.313 ms.  No row was dropped.  Source:
`telemetry-regression.json`, `clock_mapping` and
`populations.delivered-nonBOS.join_distance_ms`.

The retained poll timestamp was taken before `nvidia-smi`, CPU, and PSI reads
and had no acquisition-end/source-boundary fields. Moreover, a PSI delta at
sample *i* describes `(i-1, i]`, while nearest-sample joining could attach that
future-ended interval to an earlier frame. Consequently the v1 interval-stall
association is not causal. The repaired poller records wall/monotonic overall
and per-source start/end/duration fields; the repaired analyzer validates schema
and strictly increasing times, assigns counter deltas to their actual preceding
interval, and joins only values whose acquisition has ended by the event time.

The qualified Step 0 analyzer issued these soak strata on the `server_step_ms`
clock:

| population | n | mean ms | p50 | p95 | p99 | max | membership SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| delivered | 300 | 82.355 | 79.583 | 85.717 | 157.894 | 162.310 | `7138c2eb10406d0b1348e231b619ba55259a73d72842705cc0fb77d61e24aac4` |
| delivered-nonBOS | 293 | 80.553 | 79.522 | 85.384 | 86.882 | 87.167 | `bccf6f80f74b8a2a59073bd21cb845785425c4420d44507e32610ef15672b672` |
| BOS-transition | 7 | 157.799 | 157.894 | 162.310 | 162.310 | 162.310 | `6a2010c186b0205ac891c43fb5b5c81ae876fcb48e25eee5df11cd91b0d3dbfd` |
| EarTTS-high | 0 | null | null | null | null | null | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| residual-high | 0 | null | null | null | null | null | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| idle | 3,315 | 70.033 | 69.708 | 75.006 | 80.371 | 129.257 | `ed72af26a6f80d8f04248178888554c11be06e6bc2acd3e816bf8232c526f726` |

These values and membership hashes are copied from
`stratified-analysis.json`; no table value was hand-computed.

## Channel regressions

Effective support and the principal sensitivity views are:

| channel | rows / samples / responses | unique | r(frame) | r(sample) | r(channel,time) | r(outcome,time) | r(time adjusted) | r(within response) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU PSI avg300 | 293 / 31 / 7 | 7 | 0.772 | 0.820 | 0.982 | 0.756 | 0.243 | −0.215 |
| GPU temperature C | 293 / 31 / 7 | 10 | 0.466 | 0.502 | 0.877 | 0.756 | −0.623 | −0.026 |

CPU `avg300` is almost a proxy for time, GPU temperature also strongly follows
time, and frame cost follows time. Conditioning on time can remove a genuine
time-mediated effect or fail to remove an unrelated state transition. The
within-response view shows that the positive raw associations are primarily
between-response/order associations, but does not prove that pathway spurious.

The table covers every available, varying channel for qualified
`delivered-nonBOS` rows.  `r(frame)` is the per-frame Pearson correlation;
`r(sample)` first averages all joined frames sharing a 1 Hz telemetry sample;
`r(time-adjusted)` is the frame correlation after independently removing each
channel's and `server_step_ms`'s linear monotonic-time trend.  The interval is a
10,000-iteration response-cluster bootstrap of `r(frame)` (seed 20260813).

| telemetry channel | raw sample range | r(frame) | r(sample) | r(time-adjusted) | response-cluster 95% interval |
|---|---:|---:|---:|---:|---:|
| GPU SM clock MHz | 2,398–2,554 | −0.215 | −0.269 | 0.099 | [−0.791, 0.115] |
| GPU power W | 10.44–36.77 | 0.014 | 0.012 | −0.442 | [−0.259, 0.310] |
| GPU temperature C | 48–70 | 0.466 | 0.502 | −0.623 | [0.004, 0.822] |
| GPU utilization % | 0–96 | −0.371 | −0.295 | −0.061 | [−0.585, −0.027] |
| CPU PSI avg10 | 0–2.35 | −0.006 | −0.018 | 0.383 | [−0.655, 0.584] |
| CPU PSI avg60 | 0–0.51 | 0.326 | 0.370 | 0.379 | [−0.405, 0.736] |
| CPU PSI avg300 | 0–0.22 | 0.772 | 0.820 | 0.243 | [0.105, 0.905] |
| CPU PSI interval stall fraction | 0.000136–0.222793 | −0.012 | 0.123 | 0.178 | [−0.221, 0.274] |
| I/O PSI interval stall fraction | 0–0.002955 | −0.057 | −0.015 | −0.163 | [−0.162, 0.101] |

Undefined/non-varying channels are also findings: CPU5–CPU9 were each exactly
3,900,000 kHz; GPU memory utilization was 0%; memory PSI avg10/60/300 and
interval fraction were zero; I/O PSI avg10/60/300 were zero.  GPU memory clock,
memory used/total, and power limit were null in all samples.  Complete channel
cardinalities, ranges, null counts, slopes, correlations, intervals, and
quartiles are retained in `telemetry-regression.json`.

This was an exploratory all-channel screen with **no multiplicity adjustment**.
All channels and sensitivity views are disclosed; no nominal interval exclusion
is used to promote, reject, or rank a candidate. The repaired analyzer states
this policy in its report-facing JSON and emits raw, sample-mean,
linear-time-adjusted, within-response, and adjusted-statistic bootstrap views
separately.

## Response-order sensitivity

The response sequence is piecewise rather than well modeled by one linear time
term:

| response | non-BOS n | server mean ms | perception | Nano | EarTTS | codec | wrapper residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 79.078 | 11.896 | 52.238 | 11.699 | 0.394 | 2.225 |
| 2 | 42 | 78.856 | 11.923 | 52.056 | 11.549 | 0.396 | 2.310 |
| 3 | 50 | 78.833 | 12.047 | 51.978 | 11.443 | 0.401 | 2.340 |
| 4 | 48 | 78.731 | 11.781 | 52.180 | 11.677 | 0.386 | 2.193 |
| 5 | 34 | 79.843 | 12.052 | 52.876 | 11.598 | 0.416 | 2.294 |
| 6 | 41 | 83.980 | 13.104 | 52.449 | 11.850 | 0.700 | 4.615 |
| 7 | 44 | 84.609 | 13.269 | 52.734 | 11.998 | 0.704 | 4.641 |

The first five responses contain 208 rows at 79.019 ms; the final two contain
85 rows at 84.306 ms. Aggregate perception changes 11.936→13.190 ms, codec
0.398→0.702 ms, and wrapper residual 2.274→4.629 ms, while Nano changes only
52.230→52.597 ms and EarTTS 11.586→11.926 ms. This late regime shift is why a
single linear detrend is not a causal disambiguator.

## Scatter-table summaries

Each row is a telemetry-value quartile after joining the qualified
`delivered-nonBOS` population.  Timing columns are analyzer
`server_step_ms`; quartiles are descriptive and do not make independent rows
out of autocorrelated frames.

| channel | quartile telemetry range | rows / responses | mean server ms | p95 server ms |
|---|---:|---:|---:|---:|
| CPU PSI avg300 | 0.09–0.12 | 73 / 2 | 78.937 | 81.901 |
| CPU PSI avg300 | 0.12–0.15 | 73 / 3 | 78.854 | 81.147 |
| CPU PSI avg300 | 0.15–0.21 | 73 / 3 | 80.140 | 85.440 |
| CPU PSI avg300 | 0.21–0.22 | 74 / 2 | 84.230 | 86.333 |
| GPU temperature C | 54–61 | 73 / 2 | 78.937 | 81.901 |
| GPU temperature C | 61–67 | 73 / 3 | 78.854 | 81.147 |
| GPU temperature C | 67–68 | 73 / 4 | 80.770 | 85.280 |
| GPU temperature C | 68–69 | 74 / 3 | 83.609 | 86.439 |
| GPU utilization % | 52–75 | 73 / 5 | 82.296 | 86.291 |
| GPU utilization % | 75–77 | 73 / 6 | 80.193 | 85.384 |
| GPU utilization % | 77–81 | 73 / 6 | 80.374 | 85.440 |
| GPU utilization % | 81–83 | 74 / 5 | 79.364 | 83.827 |
| GPU SM clock MHz | 2,457–2,502 | 73 / 4 | 82.827 | 85.782 |
| GPU SM clock MHz | 2,502–2,509 | 73 / 4 | 79.803 | 85.384 |
| GPU SM clock MHz | 2,509–2,515 | 73 / 4 | 80.564 | 85.121 |
| GPU SM clock MHz | 2,515–2,535 | 74 / 3 | 79.038 | 81.901 |
| GPU power W | 29.19–31.81 | 73 / 4 | 80.018 | 84.728 |
| GPU power W | 31.81–32.11 | 73 / 4 | 80.833 | 85.572 |
| GPU power W | 32.11–33.35 | 73 / 4 | 80.891 | 85.507 |
| GPU power W | 33.35–36.77 | 74 / 6 | 80.470 | 85.269 |

The identical first two quartile timing rows for CPU `avg300` and temperature
show their shared slow time ordering; the last two responses are also the two
slowest response means in the qualified analyzer (85.828 and 86.336 ms).  This
is why the raw positive correlations are labeled confounded. The observed
clock/frequency/memory/I/O facts are operational descriptions only; this single
run supplies neither a causal finding nor a robust null for Step 5 ranking.

## Matched client populations and corrected Step 4c design

The typed-soak observation is 293 `delivered-nonBOS` rows at 80.553 ms with no
row at or above 88 ms. Comparing that number to an all-`delivered` browser mean
would mix populations. The matched retained populations are:

| retained trace | client | all delivered n / mean | delivered-nonBOS n / mean | non-BOS >=88 | EarTTS-high |
|---|---|---:|---:|---:|---:|
| `3c888978...` | direct sustained typed | 300 / 82.355 ms | 293 / 80.553 ms | 0 | 0 |
| `42a2717a...` | browser campaign, capture on | 756 / 83.226 ms | 748 / 82.163 ms | 2 | 0 |
| `a543b72d...` | browser campaign, capture off | 93 / 81.879 ms | 92 / 81.137 ms | 0 | 0 |
| `310af616...` | native Pipecat, no browser, trace on | 276 / 84.887 ms | 269 / 82.935 ms | 0 | 0 |
| `bb89df0d...` | native Pipecat, no browser, trace off | 279 / 83.494 ms | 272 / 81.456 ms | 0 | 0 |

The matched accepted-browser contrast against the direct typed run is 1.610 ms,
not roughly 2.3–2.7 ms, and remains confounded by prompt/content, period, audio
path, and client stack. The two no-browser Pipecat arms span the browser mean,
so the retained data show no monotonic browser penalty. Across the eight
substantial same-image Step 2 browser attempts, `delivered-nonBOS` was n=5,539,
pooled mean 82.050878 ms, 31 rows were >=88 ms, and zero were EarTTS-high. The
two >=88 ms rows in the accepted capture-on trace were Nano-high and
perception-high, not EarTTS-high.

Earlier retained evidence already has EarTTS-high in a direct sustained/typed
artifact (`sustained-aged-26h-20260812`: 10/188 non-BOS rows). Browser/WebRTC is
therefore neither necessary nor, on the current image, observed sufficient for
the high mode. The prior interaction claim is removed. The safer descriptive
hypothesis is state/image/workload/period dependence.

Step 4c must measure **incidence first** in counterbalanced run blocks over (a)
direct WebSocket, (b) native Pipecat without a browser, and (c) browser/WebRTC.
Every condition uses the same deterministic long PAD-tail fixture and matched
`delivered-nonBOS` predicate, with prompts, acoustic input, response target,
image, warm-up, and run duration fixed; stage rows and telemetry are retained.
Run block is the randomization/uncertainty unit, and EarTTS-high counts/rates are
analyzed with cluster/run uncertainty. Only if the high mode recurs does Nsight
capture matched high/low PAD-position states within each client condition. If
it does not recur by a preregistered exposure cap, Step 4c is `INSUFFICIENT` and
pivots to the older reproducible state difference; it does not declare a client
path causal.

## Required telemetry follow-up

Repeated observational sessions and detrending cannot identify a cause. Use
randomized intervention blocks where safe: hot/cool starts for thermal state,
locked-clock versus normal-clock blocks where supported, and controlled
CPU-pressure versus idle-host blocks. Run block—not frame or response—is the
intervention and uncertainty unit. Record acquisition start/end for every
source, GPU throttle/event reasons, process/core utilization, and per-core
scheduler/pressure evidence. Use causally ordered preceding-window joins for
interval counters, analyze lags, and model named stages as outcomes. If an
intervention is unsafe or unavailable, label the result observational and do
not use detrending to issue a causal null.

## Exact commands and excluded launch

Accepted polling and runner commands:

```bash
.venv/bin/python tools/qualification/step4_telemetry_poll.py \
  --output "$STEP4_SESSION_DIR/telemetry.jsonl" \
  --duration-seconds 330 --interval-seconds 1

curl -fsS http://127.0.0.1:8786/health \
  > "$STEP4_SESSION_DIR/preflight/retry-health-immediately-before.json"
.venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("active_client") is False, "single-client lock occupied"' \
  "$STEP4_SESSION_DIR/preflight/retry-health-immediately-before.json"

.venv/bin/python tools/qualification/sustained_strict_v3.py \
  --url ws://127.0.0.1:8786/v1/realtime \
  --output "$STEP4_SESSION_DIR/soak" \
  --duration-seconds 300 --typed-interval-seconds 45 \
  --drain-seconds 15 --no-require-queue --no-expect-session-limit
```

Analysis commands:

```bash
.venv/bin/python tools/qualification/stratified_latency_analyzer.py analyze \
  ~/.local/state/nemotron-voicechat/traces/model/3c888978-02de-4b81-982b-10d696c86328/events.jsonl \
  --output "$STEP4_SESSION_DIR/stratified-analysis.json"

.venv/bin/python tools/qualification/step4_telemetry_regression.py \
  --analyzer "$STEP4_SESSION_DIR/stratified-analysis.json" \
  --server-trace ~/.local/state/nemotron-voicechat/traces/model/3c888978-02de-4b81-982b-10d696c86328/events.jsonl \
  --telemetry "$STEP4_SESSION_DIR/telemetry.jsonl" \
  --output "$STEP4_SESSION_DIR/telemetry-regression.json"
```

An initial operator invocation at 15:04:33 UTC pre-created the runner output
directory.  `sustained_strict_v3.py` failed before `websockets.connect` with
`FileExistsError`; it acquired no model session and generated no client event
file.  Its 331 telemetry-only samples and failure report are retained as
`telemetry-launch-failure.jsonl`, `runner-launch-failure.stdout.log`, and
`soak-launch-failure/report.json`, and are excluded from every result above.
The accepted retry changed only that invocation error: the runner was allowed
to create its own `soak/` directory.
