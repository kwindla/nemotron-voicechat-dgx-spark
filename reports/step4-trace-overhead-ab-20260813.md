# Step 4a trace/logging overhead A/B — 2026-08-13

> Supersession note: the prior completion note `step4a-done.md` is withdrawn; this repaired report and `step4-telemetry-regression-20260813.md` are the sole measurement record.

## Result

There is no supported server-side trace switch on the running image, so this
round used the brief's permitted fallback: a real 300 s Pipecat-layer playout
trace on/off A/B, while observing both model sessions with the unchanged server
trace and qualified analyzer.  No production code, container, browser harness,
or server configuration was changed.

**Server trace cost: UNMEASURED.**  The image had no supported server-side trace
switch, and the client-side fallback does not identify that cost.

For qualified delivered generation (`delivered-nonBOS`, `server_step_ms`), arm A
Pipecat trace-on minus arm B trace-off was **+1.479 ms** in this one fixed
on-then-off sequence.  A paired response-cluster bootstrap 95% interval was
**−0.758 to +3.489 ms** (seven paired prompt/response clusters, 10,000
iterations, seed 20260813).  This interval is **conditional on the two observed
run periods**: it describes response-sequence variability within them and does
not bound run-period, warm-state, or order effects.  It is not a treatment-effect
confidence interval or a bounded null.
Trace-on p95 was 85.282 ms versus trace-off 85.446 ms even though the pooled
means were 82.935 and 81.456 ms, another reason not to interpret the mean
difference as tail improvement from disabling tracing.

Delivered-generation wrapper residual had a +0.988 ms period association
(conditional response bootstrap interval +0.058 to +1.851 ms), but it is **not
a qualified causal result**:
the toggle runs in the client Pipecat process, the server clock ends before
Pipecat sees the frame, the arms ran once in fixed on→off order, and response
lengths differed (269 versus 272 delivered-generation rows).  Per-response
effects changed sign: the first three favored trace-off, responses 4–6 favored
trace-on, and response 7 favored trace-off.  Both arms had zero qualified
`residual-high` rows (`wrapper_residual_ms >= 10`, n=0/n=0), so the retained
residual-outlier hypothesis was not exercised and its effect is unavailable.

The effective treatment support is one run period per arm, not 269 or 272
independent assignments.  The seven response pairs are conditional clusters;
the 269/272 rows are autocorrelated frames.  No causal Pipecat tracing effect
and no server tracing cost were identified, and neither observed association
promotes or demotes a Step 4/5 candidate.

All effect sizes above come from `ab-comparison.json`.  Populations and clocks
are issued by `arm-a-stratified-analysis.json` and
`arm-b-stratified-analysis.json`; no campaign number was hand-computed.

## Completion conclusion

Step 4a does not supply a candidate-ranking input. Server trace cost is
`UNMEASURED`; the client Pipecat A/B is descriptive single-sequence evidence,
and its interval is conditional on one fixed on-then-off pair of run periods.
No tracing candidate is promoted or demoted. The next valid server experiment
uses randomized/counterbalanced independently restarted run blocks and the
correctness-invalidates-whole-block gate specified below.

## Why the server switch was unavailable

The deployed command contains fixed `--trace-dir /trace`; the only server CLI
control is `--trace-dir`, not an environment- or per-session trace-mode switch
(`src/nemotron_voicechat_runtime/server.py:7873-7876,8223-8230`).  Omitting the
directory would still construct and `json.dumps` every record and synchronously
`print(..., flush=True)`; it would only omit the JSONL/PCM files
(`server.py:2277-2319`).  Restarting the container to change even that partial
control was forbidden.

The hot-path audit also establishes the measurement boundary:

- `server_step_ms` stops at `server.py:3990`, before result-side audio statistics
  are built at 4046-4047;
- `send_step` then converts and synchronously flushes delivered PCM, constructs
  protocol/metrics/trace objects, sends WebSocket events, JSON-encodes the trace,
  prints/flushed stdout, and writes/flushed JSONL (`server.py:5660-5754` and
  `SessionTrace` above);
- the Pipecat switch is supported and branch-free at construction:
  `NEMOTRON_VOICECHAT_PLAYOUT_TRACE` selects the traced subclass once
  (`src/nemotron_voicechat_pipecat/llm.py:395-398`).  That subclass records
  receipt/release/push events through a bounded writer thread; the off arm uses
  the base hot methods (`llm.py:1721-1898`).

Consequently this fallback could observe a period association consistent with
gross host contention or backpressure in following server steps, but it cannot
directly time the server's own post-`server_step_ms` trace construction/writes.
The server-side hypothesis remains **UNMEASURED** and has the preregistered
follow-up below.

## A/B design and full provenance

Both arms used the task-scoped native Pipecat driver
`tools/qualification/step4_pipecat_soak.py`, not a browser.  It instantiated the
same `NemotronVoicechatLLMService`, submitted exact 80 ms zero PCM frames for
300 s, injected one fixed tool-free typed prompt every 45 s, drained for 15 s,
and captured downstream Pipecat response/audio counts.  Each invocation called
`/health`, required `active_client:false`, retained the complete response, and
only then opened the model WebSocket.

Fixed prompt cycle (seven sends at source seconds 0, 45, 90, 135, 180, 225,
270):

1. `What is two plus three? Answer in one short sentence.`
2. `Remember that the codeword is sapphire and acknowledge it briefly.`
3. `What codeword did I ask you to remember?`
4. `Say the word latency once.`

The service instruction was exactly `Answer briefly. Do not use tools.` and its
tool list was empty.  Both arms used Pipecat prebuffer 160 ms (constructor
default, no environment override), input 16 kHz mono PCM16, native model output
22.05 kHz mono PCM16, and pipeline output declaration 24 kHz.  The server image,
container ID/command, checkpoint, sequential Nano policy, qualification no-text
override, fixed-stage optimizations, and runtime source hashes are identical to
the configuration table in `reports/step4-telemetry-regression-20260813.md`.
The host driver used checkout commit
`e5924c7b57503ea03e2a72d104f8decfbfd0f46d`, project package 0.1.0,
Pipecat 1.7.1.dev21, and Python 3.12.13; the Pipecat/Python identity is also the
first retained line of each arm's stdout log.
The full immutable configuration is retained in
`reports/step4-trace-overhead-ab-20260813-data/preflight/container-inspect.json`.

| item | A: Pipecat trace on | B: Pipecat trace off |
|---|---|---|
| wall interval UTC | 15:22:35.006161495–15:27:51.533346291 | 15:28:05.268804108–15:33:21.808271923 |
| immediate lock | `arm-a-trace-on/health-immediately-before.json`: false | `arm-b-trace-off/health-immediately-before.json`: false |
| model session | `310af616-906b-4b14-905a-81176f97e387` | `bb89df0d-33ef-45c5-9fbe-0238fa789e40` |
| Pipecat trace env | absolute `arm-a-trace-on.playout.jsonl` | explicitly absent via `env -u` |
| source frames / prompts / completed responses | 3,750 / 7 / 7 | 3,750 / 7 / 7 |
| downstream audio | 276 frames / 973,728 bytes | 279 frames / 984,312 bytes |
| Pipecat trace status | valid, published, closed; 4,304 written, 0 dropped, writer stopped | null/disabled |
| server trace SHA-256 | `1f3a61aeb2b75d06ba2e1646f404977fa579ccb5d52ab63dd276fb0ddcb64333` | `e613be6d646a3171a86d34551d856599e867fa16493c4deea80edacb8a5591d1` |
| server-trace path | `~/.local/state/nemotron-voicechat/traces/model/310af616-906b-4b14-905a-81176f97e387/events.jsonl` | `~/.local/state/nemotron-voicechat/traces/model/bb89df0d-33ef-45c5-9fbe-0238fa789e40/events.jsonl` |

Arm A's Pipecat artifact SHA-256 is
`93eb66eac5c1f0b12f4eacf3ec2bbe897681863280ea54ccb91c05a62179b317`.
The qualified analyzer reconciled its 276 audio deltas exactly and validated
the normative client-receipt clock; that one-sided artifact is retained as
`arm-a-playout-analysis.json` for integrity, not used as a cross-arm latency
comparison because tracing-off intentionally emits no equivalent artifact.

A short excluded trace-on pilot preceded the arms solely to qualify the new
driver: session `c61e37f5-51a3-4673-86c7-620d8d72dcd8`, 8 s source, one prompt,
one response, 40 audio frames, valid 197-record trace, zero drops.  It is under
`reports/step4-trace-overhead-ab-20260813-data/pilot/` and contributes no A/B
number.

## Analyzer populations

The qualified analyzer's `server_step_ms` results are:

| population | A n | A mean / p50 / p95 / max ms | B n | B mean / p50 / p95 / max ms |
|---|---:|---:|---:|---:|
| delivered | 276 | 84.887 / 83.053 / 86.393 / 165.191 | 279 | 83.494 / 81.152 / 86.015 / 195.579 |
| delivered-nonBOS | 269 | 82.935 / 83.015 / 85.282 / 87.361 | 272 | 81.456 / 80.993 / 85.446 / 86.733 |
| residual-high | 0 | null / null / null / null | 0 | null / null / null / null |
| idle | 3,382 | 71.079 / 71.517 / 74.430 / 121.443 | 3,374 | 71.485 / 71.996 / 74.888 / 200.685 |

Membership provenance:

| population | A membership SHA-256 | B membership SHA-256 |
|---|---|---|
| delivered | `7df97b4c90de61639021791ca8571dbb2041af7bee72a92c36eaab468da1381e` | `23d8735f21b533f4da371c76cd986e64aec854502fb15a2bd2ce4af888b94878` |
| delivered-nonBOS | `e840942dc7782605a5d574b32721d44a66cd3e07558d0b80191b0b5a8d4499a5` | `7339a1b6b99d08c53e6c1e937e5e84ee9380ef772724149ad11e45486ffa0e1d` |
| residual-high | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | same empty membership hash |
| idle | `5aefd62206970ff3ad8eaa52796fa2ede0d28bb61617ccb16b669505bb797d69` | `5f657bf4083a5b44495211285d5c6ff9095908c3a01e58584497d3643c023df3` |

The delivered-generation stage means further show that the between-arm change
was not confined to one named server component:

| `delivered-nonBOS`, host-interval stage clock | A trace-on mean ms | B trace-off mean ms |
|---|---:|---:|
| perception | 12.807 | 12.447 |
| Nano interface | 52.081 | 52.308 |
| EarTTS total | 11.921 | 11.914 |
| codec call | 0.702 | 0.536 |
| wrapper residual | 4.386 | 3.398 |
| wrapper total | 81.897 | 80.603 |

Stage values are qualified analyzer means from each arm's
`populations.delivered-nonBOS.stage_timings`.  Only the server-step and wrapper
residual cross-arm effects were response-cluster bootstrapped.

## Response-clustered effects and interpretation

| outcome | A n / mean | B n / mean | trace-on − trace-off | paired cluster 95% interval | conclusion |
|---|---:|---:|---:|---:|---|
| delivered-generation `server_step_ms` | 269 / 82.935 ms | 272 / 81.456 ms | +1.479 ms | [−0.758, +3.489] ms | descriptive; interval conditional on these periods |
| delivered-generation wrapper residual | 269 / 4.386 ms | 272 / 3.398 ms | +0.988 ms | [+0.058, +1.851] ms | period association only; fixed order/client-side toggle |
| residual-high membership | 0 | 0 | unavailable | unavailable | no outlier rows in either arm |

The first row's conclusion is: **descriptive difference; conditional interval
crosses zero in this fixed sequence**.  The retained v1 analyzer paired sorted
inferred response ordinals because the schedule was identical.  The repaired
tool now requires retained contiguous prompt/job ordinals and identical prompt
IDs/text, exact analyzer membership/hashes, and one terminal response per job;
it fails on missing/duplicate ordinals or unequal prompt identities.  It then
resamples the seven declared pairs.  It does not pretend the 269/272
autocorrelated frame rows are independent.

The preregistered primary estimand in this retained result was frame-weighted:
+1.479125 ms, conditional interval [−0.757924, +3.489044].  An equal-response
sensitivity was +1.294669 ms [−0.858701, +3.402792].  The residual counterparts
were +0.987960 ms [+0.058149, +1.850613] and +0.927110 ms
[+0.028703, +1.805896].  Equal-run inference is unavailable with one run period
per arm.  The repaired analyzer emits equal-run, equal-response, and
frame-weighted definitions before its results.

These two periods do not rule out a tracing effect: the conditional interval
still permits +3.489 ms and unmeasured period/order bias is not bounded.  The
experiment does not establish any server trace cost and does not explain the
retained residual-high corpus because those rows did not recur.

## Required server-side follow-up experiment

The unresolved server hypothesis requires a qualification-only code-change
experiment in a later round:

1. Add a startup-frozen trace mode, refused outside a qualification image, with
   three explicit arms: `full` (current JSONL/stdout/PCM), `sink-null` (same
   record/PCM construction and serialization but discard writes), and `off`
   (no record construction, serialization, stdout, JSONL, PCM writes, or
   per-frame trace-field construction).  Do not overload `--trace-dir=None`,
   because it still serializes/prints today.
2. Keep protocol `voicechat.metrics` and audio deltas available to an external
   client in every arm.  Capture client receipt cadence and an explicit
   `send_step` entry/exit interval so the tested work, which lies after
   `server_step_ms`, is actually in a clock.  Preserve the existing
   `server_step_ms` clock separately for next-frame contention.
3. Use one immutable image/config, fixed Pocket seed, and exact typed/acoustic
   schedule.  Randomize/counterbalance **multiple independently restarted run
   blocks** across all three modes, for example repeated balanced Latin-square
   orders.  Choose the block count from pilot run-to-run variance and a power
   target for the preregistered 0.5 ms floor; response count inside a period is
   not treatment-assignment count.  Report equal-run and the declared
   frame-weighted estimand, with run block as the uncertainty unit.
4. Keep identical low-overhead timing instrumentation in every arm.  Separate
   record construction, serialization, file/stdout write, WebSocket send, and
   total `send_step` intervals so instrumentation does not merely move the
   measured work.  Repeat telemetry polling for order/state diagnostics.
5. Gate correctness before timing: identical protocol event sequence,
   response text, delivered PCM sample counts and hashes, no trace/queue drops,
   no mode leak, and clean session teardown.  A missing response, output
   mismatch, trace/queue drop, or mode leak invalidates the **whole run block**
   before timing analysis.  Analyze qualified `delivered-nonBOS`,
   `residual-high`, and external arrival clocks with run-block uncertainty.
6. Preregister promotion: pursue logging work only if the lower 95% bound for
   recoverable `send_step`/arrival cost exceeds 0.5 ms per delivered-generation
   frame or the residual-high occurrence/tail is materially reduced without a
   correctness or cadence regression.  An empty residual-high stratum remains
   `INSUFFICIENT`, never PASS.

## Exact commands

Pilot (excluded):

```bash
NEMOTRON_VOICECHAT_PLAYOUT_TRACE="$STEP4_AB_ROOT/pilot/playout.jsonl" \
PYTHONSAFEPATH=1 .venv/bin/python tools/qualification/step4_pipecat_soak.py \
  --output "$STEP4_AB_ROOT/pilot/run" --duration-seconds 8 \
  --typed-interval-seconds 45 --drain-seconds 8
```

Arm A and B:

```bash
NEMOTRON_VOICECHAT_PLAYOUT_TRACE="$STEP4_AB_ROOT/arm-a-trace-on.playout.jsonl" \
PYTHONSAFEPATH=1 .venv/bin/python tools/qualification/step4_pipecat_soak.py \
  --output "$STEP4_AB_ROOT/arm-a-trace-on" --duration-seconds 300 \
  --typed-interval-seconds 45 --drain-seconds 15

env -u NEMOTRON_VOICECHAT_PLAYOUT_TRACE PYTHONSAFEPATH=1 \
.venv/bin/python tools/qualification/step4_pipecat_soak.py \
  --output "$STEP4_AB_ROOT/arm-b-trace-off" --duration-seconds 300 \
  --typed-interval-seconds 45 --drain-seconds 15
```

Qualified and task-scoped analyzers:

The retained `ab-comparison.json` was produced by the v1 interface before job
mapping records were retained. The repaired command below is the required
interface for future v2 run reports; the historical reports are not silently
backfilled or represented as newly measured evidence.

```bash
.venv/bin/python tools/qualification/stratified_latency_analyzer.py analyze \
  ~/.local/state/nemotron-voicechat/traces/model/310af616-906b-4b14-905a-81176f97e387/events.jsonl \
  --output "$STEP4_AB_ROOT/arm-a-stratified-analysis.json"
.venv/bin/python tools/qualification/stratified_latency_analyzer.py analyze \
  ~/.local/state/nemotron-voicechat/traces/model/bb89df0d-33ef-45c5-9fbe-0238fa789e40/events.jsonl \
  --output "$STEP4_AB_ROOT/arm-b-stratified-analysis.json"
.venv/bin/python tools/qualification/stratified_latency_analyzer.py analyze \
  "$STEP4_AB_ROOT/arm-a-trace-on.playout.jsonl" \
  --output "$STEP4_AB_ROOT/arm-a-playout-analysis.json"

.venv/bin/python tools/qualification/step4_ab_compare.py \
  --analyzer-a "$STEP4_AB_ROOT/arm-a-stratified-analysis.json" \
  --trace-a ~/.local/state/nemotron-voicechat/traces/model/310af616-906b-4b14-905a-81176f97e387/events.jsonl \
  --run-report-a "$STEP4_AB_ROOT/arm-a-trace-on/report.json" \
  --analyzer-b "$STEP4_AB_ROOT/arm-b-stratified-analysis.json" \
  --trace-b ~/.local/state/nemotron-voicechat/traces/model/bb89df0d-33ef-45c5-9fbe-0238fa789e40/events.jsonl \
  --run-report-b "$STEP4_AB_ROOT/arm-b-trace-off/report.json" \
  --output "$STEP4_AB_ROOT/ab-comparison.json"
```

The raw reports, stdout logs, health checks, analyzer JSON, checksums, exact
container inspect, pilot, and trace artifact are retained under
`reports/step4-trace-overhead-ab-20260813-data/`.
