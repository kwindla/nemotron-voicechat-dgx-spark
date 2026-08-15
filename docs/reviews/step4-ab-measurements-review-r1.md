# Step 4a+4b measurement-validity review — round 1

**Verdict: FAIL for use as a Step 4 candidate-ranking input.** The retained
artifacts are intact and the headline arithmetic reproduces, but the reports
overstate what one telemetry period and one fixed-order A/B period per arm can
bound. The failure is inferential, not a reason to discard or rerun the raw
evidence. The reports should be repaired in place, the current results relabeled
as descriptive/single-sequence evidence, and the follow-up designs corrected
before they drive Step 4c or Step 5 prioritization.

This review was entirely offline. I did not query `/health`, open a model
connection, touch the running stack, or perform any live probe. I read the
three retained server traces cited by the Step 4 reports, the retained Step 2
campaign traces, the raw telemetry and analyzer artifacts, and the new tools
and tests. Both Step 4 checksum manifests verify in full, including the three
external server traces.

## 1. Independent recomputation

I recomputed the following directly from the retained JSONL events and
telemetry samples, using the qualified analyzer's published membership only to
identify the normative `delivered-nonBOS` rows. Percentiles below are
nearest-rank. Correlations, time residuals, and bootstrap draws were separately
reimplemented for this review rather than copied from the result JSON.

### Telemetry soak

Session `3c888978-02de-4b81-982b-10d696c86328` reproduces exactly:

| quantity | independent result | report/result JSON |
|---|---:|---:|
| `delivered-nonBOS` n | 293 | 293 |
| mean / p95 / max `server_step_ms` | 80.552857 / 85.384 / 87.167 ms | 80.552857 / 85.384 / 87.167 ms |
| rows at `server_step_ms >= 88` | 0 | 0 |
| nearest-telemetry join mean / p95 / max | 249.881 / 478.242 / 499.313 ms | same |
| CPU PSI avg300 raw r / time-residual r | 0.772115 / 0.243306 | same |
| CPU PSI avg300 cluster interval | [0.104785, 0.905045] | same |
| GPU temperature raw r / time-residual r | 0.466421 / -0.623441 | same |
| GPU temperature cluster interval | [0.003511, 0.821858] | same |
| GPU SM clock raw r / time-residual r | -0.215477 / 0.098851 | same |
| CPU interval-stall raw r / cluster interval | -0.011640 / [-0.220797, 0.274284] | same |

All other correlations in the report's channel table also reproduce to the
displayed precision. The 331-sample duration/cadence, constant 3.9 GHz values
for CPU5-CPU9, null integrated-GPU fields, zero memory/I/O PSI averages, and
the empty spike membership reproduce.

The effective support is much smaller than the 293 frame rows suggest. The
delivered rows join to only **31 telemetry samples and seven responses**. CPU
PSI avg300 has only **seven unique joined values**, and integer GPU temperature
has ten. The report's phrase “only 7/10 observed delivered-row values” should
say “7 and 10 unique joined values”; all 293 delivered rows have observed
values for both channels.

### Pipecat trace on/off comparison

Sessions `310af616-906b-4b14-905a-81176f97e387` and
`bb89df0d-33ef-45c5-9fbe-0238fa789e40` also reproduce exactly:

| outcome | A trace-on | B trace-off | A-B | reported-style paired response bootstrap |
|---|---:|---:|---:|---:|
| `server_step_ms` | n=269, 82.935320 ms | n=272, 81.456195 ms | +1.479125 ms | [-0.757924, +3.489044] ms |
| wrapper residual | n=269, 4.385639 ms | n=272, 3.397679 ms | +0.987960 ms | [+0.058149, +1.850613] ms |

The seven per-response `server_step_ms` effects reproduce as +4.787, +4.684,
+2.340, -2.148, -2.061, -1.475, and +2.935 ms. The residual effects reproduce
as +2.334, +2.352, +1.853, -0.269, -0.418, -0.454, and +1.092 ms. Thus the
report accurately discloses the sign changes.

The analyzer's estimand is a frame-weighted pooled arm difference. A sensitivity
calculation that gives every response pair equal weight yields +1.294669 ms
with a same-seed response bootstrap interval [-0.858701, +3.402792] for
`server_step_ms`, and +0.927110 ms [0.028703, 1.805896] for residual. This does
not reverse either descriptive result, but the intended weighting must be
preregistered in the follow-up because response lengths are treatment-period
outcomes, not fixed weights.

## 2. Telemetry methodology does not identify a cause or a robust null

Linear time detrending is mathematically implemented correctly: it is the
partial correlation obtained after removing a linear monotonic-time fit from
both the channel and `server_step_ms`. It is not, however, a sound causal
disambiguator in this run.

- CPU avg300 is almost a proxy for time (`r(channel,time)=0.981676`), GPU
  temperature also strongly follows time (`r=0.876995`), and frame cost follows
  time (`r=0.755590`). If temperature or accumulated pressure causes a real
  slowdown through time, conditioning on time removes the pathway of interest.
  Conversely, if an unrelated late-run state change causes the slowdown,
  linear detrending need not remove it. The adjusted correlations therefore
  neither establish nor refute a thermal/pressure effect.
- The response sequence is visibly piecewise rather than linear. The first
  five responses contain 208 non-BOS rows at mean 79.019 ms; the final two
  contain 85 rows at 84.306 ms. The shift is concentrated in perception
  (11.936 to 13.190 ms), codec (0.398 to 0.702 ms), and wrapper residual
  (2.274 to 4.629 ms), while Nano changes only 52.230 to 52.597 ms and EarTTS
  11.586 to 11.926 ms. A single linear time term is a poor model for this
  regime change.
- A within-response demeaning sensitivity gives r=-0.215 for CPU avg300 and
  r=-0.026 for temperature. That shows the positive raw results are almost
  entirely between-response/run-order associations. It does not prove the
  between-response pathway is spurious.
- The response-cluster bootstrap correctly avoids claiming 293 independent
  frames, but seven clusters inside one run do not estimate run-to-run
  variability. The raw intervals are also shown next to time-adjusted point
  estimates even though no interval was computed for the adjusted statistic.
  Multiple exploratory channels were examined without a multiplicity plan.
- `step4_telemetry_poll.sample()` timestamps before invoking `nvidia-smi` and
  before reading CPU/PSI, but records no acquisition-end time or per-source
  acquisition interval. More importantly, PSI interval fractions describe
  `(previous sample, current sample]`, while the nearest-sample join can attach
  the current value to a frame before the current sample. That can leak future
  stalls into a frame's predictor. The instantaneous PSI null is useful only as
  a coarse descriptive check until this is fixed.

The operational observations are still valuable: there was no GPU SM-clock
collapse, the selected CPU frequency files stayed at 3.9 GHz, and host memory
and I/O PSI averages were zero. Those facts support “no observed clock collapse
or host memory/I/O pressure in this run.” They do **not** support the completion
note's broader decision to deprioritize thermal/DVFS/**CPU pressure** work. In
particular, constant frequency does not exclude scheduler contention, and the
late codec/residual step occurs exactly where global CPU avg300 is highest.

Required report fix: make raw, sample-mean, linear-time-adjusted, and
within-response results explicitly separate sensitivity views; print
`n_telemetry_samples=31`, unique-value counts, response-level stage means, and
channel-time/outcome-time correlations in the main report; remove any causal
or ranking conclusion from the sign reversal; and narrow the operational null
to the channels actually observed.

## 3. Seven clusters do not rescue the fixed-order A/B

The A/B report is commendably explicit that the switch is client-side, that
`server_step_ms` ends before Pipecat observes the frame, that order was fixed
on-to-off, and that residual-high was empty. Its arithmetic treatment of frame
autocorrelation is better than a row-wise interval. The remaining label is
still too strong.

Treatment was assigned once to an entire run: one A period followed by one B
period. The experimental unit for arm/order variance is therefore the **run
period**, of which there is one per arm. Resampling seven responses inside
each period quantifies response-sequence variability conditional on those two
periods; it cannot estimate period, warm-state, or order variance. Calling the
result a “bounded null” and saying it rules out a large tracing effect makes the
bootstrap interval sound like a bound on a general treatment effect. It is not.
The interval still permits +3.489 ms even under its conditional model, and
unmeasured period bias is unbounded by it.

The correct conclusion is narrower: “In this one on-then-off sequence, the
observed pooled difference was +1.479 ms; seven-pair resampling conditional on
these periods crossed zero. No causal Pipecat tracing effect and no server
tracing cost were identified.” The residual interval excluding zero remains a
period association only. It should not promote or demote a candidate.

This was also not a direct server tracing experiment. The report accurately
states that the server's JSON/PCM/stdout work is after `server_step_ms` and
unchanged in both arms. The fallback satisfies the brief as an observational
negative control, but Step 4a's server-side hypothesis remains **UNMEASURED**,
not a bounded server-trace null.

## 4. The typed-only/browser side observation is population-confounded

The raw typed-soak observation itself is true: session `3c888978...` has 293
`delivered-nonBOS` rows, mean 80.552857 ms, and zero rows at or above 88 ms.
The proposed comparison to “82.8-83.2 ms in browser campaigns” is not valid as
stated because that browser range is based primarily on **all `delivered`
rows**, including BOS transitions, while 80.553 is `delivered-nonBOS`.

Matched populations from the accepted Step 2 campaign are:

| retained trace | client | all delivered n / mean | delivered-nonBOS n / mean | non-BOS >=88 | EarTTS-high |
|---|---|---:|---:|---:|---:|
| `3c888978...` | direct sustained typed | 300 / 82.355 ms | 293 / 80.553 ms | 0 | 0 |
| `42a2717a...` | browser campaign, capture on | 756 / 83.226 ms | 748 / 82.163 ms | 2 | 0 |
| `a543b72d...` | browser campaign, capture off | 93 / 81.879 ms | 92 / 81.137 ms | 0 | 0 |
| `310af616...` | native Pipecat, no browser, trace on | 276 / 84.887 ms | 269 / 82.935 ms | 0 | 0 |
| `bb89df0d...` | native Pipecat, no browser, trace off | 279 / 83.494 ms | 272 / 81.456 ms | 0 | 0 |

The matched accepted-browser contrast is therefore 1.610 ms on non-BOS rows,
not roughly 2.3-2.7 ms, and it remains confounded by prompts, response content,
run period, audio path, and client stack. The no-browser Pipecat arms span both
sides of the accepted browser mean, so the retained data do not show a monotonic
browser penalty.

I also checked the eight substantial Step 2 browser-attempt traces retained on
the same `generation-step2` image. Across those traces, `delivered-nonBOS` is
n=5,539, pooled mean 82.050878 ms, with 31 rows >=88 ms and **zero EarTTS-high
rows**. Individual non-BOS means range 81.610-82.522 ms. The two >=88 ms rows
in the final accepted capture-on trace are not EarTTS-high: one has elevated
Nano (56.298 ms), and the other elevated perception (15.511 ms).

Most importantly, retained earlier evidence already contains EarTTS-high in a
direct sustained/typed artifact (`sustained-aged-26h-20260812`: 10/188
non-BOS rows). Browser/WebRTC concurrency is therefore neither necessary nor,
on the current Step 2 image, observed to be sufficient for EarTTS-high. The
interaction hypothesis is currently unsupported. A safer inference is that
the high mode is state/image/workload/period dependent and failed to recur in
all three Step 4 sessions and all current-image browser campaign traces.

Required 4c design change: first measure **incidence**, not just latency means,
in counterbalanced run blocks using the same deterministic long PAD-tail
fixture and matched `delivered-nonBOS` predicate across (a) direct WebSocket,
(b) native Pipecat without browser, and (c) browser/WebRTC. Hold prompts,
acoustic input, response target, image, warm-up, and run duration fixed; retain
stage rows and telemetry. Treat run block as the randomization unit and analyze
EarTTS-high counts/rates with cluster/run uncertainty. Only if the high mode
recurs should Nsight capture matched high/low PAD-position states within each
client condition. If it does not recur by a preregistered exposure cap, 4c is
`INSUFFICIENT` and should pivot to the older reproducible state difference,
not declare the current client path causal.

## 5. Follow-up experiment designs need treatment-unit and intervention fixes

### Server trace experiment

The proposed `full` / `sink-null` / `off` decomposition, separate
`send_step` clock, external arrival clock, correctness gates, and startup-frozen
qualification-only control are good. The statistical design is not complete.

`full/off/off/full` and `full/sink-null/sink-null/full` provide only two periods
per compared mode even if each period contains 30 responses. Thirty response
clusters are not 30 independent treatment assignments. Replace this with
multiple randomized/counterbalanced run blocks over all three modes (for
example balanced Latin-square orders repeated across independently restarted
periods), and choose the number of blocks from a pilot variance/power target for
the preregistered 0.5 ms floor. Report both equal-run and declared frame-weighted
estimands. Keep identical low-overhead timing instrumentation in every arm and
separate record construction, serialization, file/stdout write, WebSocket send,
and total `send_step` intervals so the measurement does not merely move the
cost. A missing response, output mismatch, trace/queue drop, or mode leak must
invalidate a whole run block before timing analysis.

### Telemetry experiment

Repeated observational sessions alone still cannot separate time from a
thermal/pressure cause. The follow-up needs randomized interventions or strong
negative controls: randomized hot/cool start blocks for thermal state; locked
clock versus normal-clock blocks where supported; and controlled CPU-pressure
versus idle-host blocks for the CPU hypothesis. Record acquisition start/end
for each source, GPU throttle/event reasons, process/core utilization, and
per-core scheduler/pressure evidence where available. Use causally ordered
preceding-window joins for interval counters, analyze lags, model named stages
as outcomes, and make run block—not frame or response—the unit for intervention
uncertainty. If intervention is unsafe or unavailable, label the result
observational and do not use detrending to issue a causal null.

## 6. Tool and test quality

The tools are readable, deterministic, preserve unavailable values as null,
use source hashes, and fail on a grossly distant join. The tests cover parsing,
basic correlation math, join distance, bootstrap sign, the health lock, and
the fixed prompt list. They are not yet campaign-grade for the conclusions
being drawn.

Concrete defects/fixes:

1. `step4_telemetry_poll.py` records only a pre-read timestamp. Add monotonic
   and wall start/end timestamps plus per-source read boundaries/durations.
2. `step4_telemetry_regression.py` trusts the analyzer member timestamp and
   checks only source ordinal/frame. Require the member timestamp to match the
   raw event clock, validate telemetry schema and strictly increasing sample
   times, and verify exact analyzer membership, not just source identity.
3. Assign counter deltas to their actual preceding interval and join only
   causally prior/containing intervals. Do not attach a future-ended PSI delta
   by nearest timestamp.
4. Emit effective sample/unique-value counts, channel-time correlations,
   response/run sensitivities, and adjusted-statistic uncertainty in the
   report-facing summary. Add an explicit exploratory multiplicity policy.
5. `step4_ab_compare.py` validates freshly classified population cardinality
   but not exact equality with the analyzer's published members/hashes. A
   tampered same-cardinality analyzer can therefore be cited without detection.
   Compare exact ordered members and hashes.
6. A/B pairing is based only on sorted inferred response IDs and equal cluster
   counts. Pair by retained prompt/job ordinal and prompt identity; fail on a
   missing or duplicated ordinal. Declare equal-run/equal-response/frame-weighted
   estimands before analysis.
7. `step4_pipecat_soak.py` considers `responses_done >= prompts_sent - 1` a
   pass and does not retain prompt injection/mapping records. For an A/B
   qualification driver, require exactly one terminal response per submitted
   job (unless a specifically preregistered disposition says otherwise) and
   retain injection time, job ID, prompt ID, response ID, and terminal status.
8. Add end-to-end synthetic tests for corrupted member timestamps, wrong
   same-cardinality membership, non-monotonic telemetry, interval-boundary
   assignment, missing middle responses, unequal prompt identities, constant
   and null channels, empty/nonempty spike strata, deterministic bootstrap
   regression values, and both declared weighting estimands. The current four
   test modules are useful smoke tests but do not exercise any complete
   analysis path.

## Closure conditions

This review can pass without new live measurements if the authors:

1. revise both reports and the completion note to use the narrower descriptive
   conclusions above, including effective sample counts and the matched
   browser population table;
2. relabel Step 4a server trace cost as `UNMEASURED` and the present client A/B
   interval as conditional on one fixed run sequence;
3. remove the browser/WebRTC EarTTS-high interaction claim from 4c motivation
   and replace it with the counterbalanced incidence-first design;
4. repair the two follow-up designs around run-block treatment assignment and
   causal telemetry ordering/intervention; and
5. fix the membership/join/pairing/pass-gate defects and add the end-to-end
   tests listed above.

No retained raw artifact needs to be deleted. The exact numerical reproduction,
hash-complete provenance, explicit empty high strata, and honest disclosure of
the unsupported server switch are strong foundations; they are simply not yet
a valid causal or ranking result.
