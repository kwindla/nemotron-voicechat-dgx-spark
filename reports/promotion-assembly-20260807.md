# Promotion assembly: production-candidate-2

Date: 2026-08-07 UTC

Branch: `codex/promotion-assembly`

Worktree: `/home/khkramer/src/nemotron-voicechat-dgx-spark-promo`

Base: `codex/step7-pair-graph` at `9ee8d21eb1c8dd197cf6e2a17f9d62deb827de4f`

Candidate image: `nemotron-local/voicechat-vllm:promotion-candidate-2`

Image ID: `sha256:c497c7d27c2ba47168ffadf23edc1bf3348b1226259c503e58eb6ae87ca8d16d`

## 1. Recommendation

**PROMOTE the assembled image as production-candidate-2.** All five mandatory
gates passed. The two 900-second release-soak arms ran strictly sequentially
and both exceeded the 1,500-settled-frame requirement. Against the measured
production-candidate-1 arm, the fully assembled candidate improves settled raw
mean by **1.717 ms (-2.465%)**, raw p95 by **6.774 ms (-6.649%)**, and the newly
captured raw p99 by **9.542 ms (-8.751%)**. Its sliding two-frame p95 improves
by **4.834 ms (-6.304%)** and is 71.847 ms, below the unchanged 80 ms gate.

This recommendation uses only the measured stacked soak. No savings from the
individual step reports are added together.

The Step-4 W8A32 v2 launch table is **permanently RETIRED** and is not present
in this candidate. Its stacked mean delta was smaller than observed run-to-run
noise, while pooled p99 regressed.

## 2. Assembled-delta manifest

Exactly the promoted deltas were assembled:

| Delta | Source branch and requested commit | Assembly commit | Candidate disposition |
|---|---|---|---|
| Step 7 pair FULL graph | `codex/step7-pair-graph` at `9ee8d21` | base `9ee8d21` | included in full |
| Step 3 Change A, pre-padded `rvq_embs` | `codex/step3-eartts-exact-constants` at `b002cb2` | cherry-pick `fd3a079` | included as stage `a` only |
| Step 9 cache/capture/assertions | `codex/step9-cold-start` at `ca368e6` | cherry-pick `b5da5d1` | included; cache regenerated for this stack |
| Step 3 Change B, cached norms | same source commit, stage `b` | none in image | excluded because it regressed |
| Step 4 W8A32 v2 | `codex/step4-w8a32-gemv-v2` at `314762f` | test-only overlay | excluded and permanently retired |

The final image label reports `eartts_exact_constants_stage=a`. Installed
source hashes are:

- EarTTS stage-A source: `82c5e5a370060baa7f85ace1a286467616214e1161d8bbb91508ff88d9d7a88c`
- Step-7 pair runner: `909614c832ebe2b24702d0698429379f301bb70560594fbcd670735f97753189`
- runtime server: `015fb6e0468abac44696d179e4d80d6fe7cbeb2b2c07d6d3b123b7d5b07138c9`

## 3. Fresh assembled-image baked cache

The Step-9 cache tree was not reused. A new uncached instrument image was
built after applying Change A and the Step-7 graph, then cold-booted to compile
the assembled sources. The extracted `/root/.cache/vllm` became the only cache
tree copied into the final image.

| Item | Assembled value |
|---|---|
| Cache-source image | `sha256:bff4d33067f386937899bf1a3fb5900a5f165219b85838ff67ddd98a8fd413b7` |
| Fresh cache-source cold boot | 364.735 s |
| Cache key | `promotion-bff4d330-eartts82c5e5a-vllm0171-gb10sm121-shapes257-80-16-2-1-v1` |
| Capture contract | `257,80,16,2,1` |
| Manifest | 2,865 files; 121,999,842 bytes |
| Manifest SHA256 | `e2a57703cfc504a54f2083717f30a098798f17025f54890b4244a5ca60556d8e` |

The provenance records are `reports/promotion-cache-provenance.json` and
`reports/promotion-cache-runtime-contract.json`.

## 4. Mandatory gates

| Gate | Result | Evidence |
|---|---|---|
| A. Step-3-method single-process bitwise gate | **PASS** | Stage A; padded RVQ, depth-summed embedding, and bounded acoustic codes are byte-equal. Padded RVQ SHA256 `983b522d64d868419ef8a29589067d890430b33e8584f282c2a8ece8ba130fe9`; acoustic-code SHA256 `a33a9958d41fd6e9577af218a7bfc58c2785904c12488045bbb378b10382961e`. |
| B. Step-7-method same-process pair gate | **PASS** | 1,389/1,389 calls; outputs and decisions equal; 675/675 cache/rollback-shadow comparisons equal; first mismatch `null`. Final decisions: 663 accepted, 12 rejected, 664 buffered, 49 conditional singles. |
| C. Retained post-FC functional fixture | **PASS** | 3/3 typed jobs completed and answered, including both two-call tool turns; zero unexpected errors; normal `client_stop`. The short harness reports only `insufficient settled frames`, which is not this fixture's functional criterion. |
| D. Cold-start/cache/assertion gate | **PASS** | Independent final-image cold boot 253.983 s; cache validated at 2,865 files/121,999,842 bytes; contract `257,80,16,2,1`; no uncovered post-ready descriptor and no post-ready compile assertion. |
| E. Release soak | **PASS** | Both 900 s arms passed the pacing/queue harness, completed 11,250 source frames and 15/15 typed turns, had zero unexpected errors, and retained 8,923/8,938 settled positive server frames respectively. |

Cold-start assertion detail: all logged `torch.compile` activity preceded
`VOICECHAT_STARTUP application_ready`. The full candidate soak produced zero
`Step-9 detected torch.compile after application readiness` and zero
`Step-9 uncovered post-ready cudagraph descriptor` failures. Because both
assertions fail closed, the uninterrupted 10,438-positive-frame session is
also an end-to-end assertion exercise.

## 5. Headline 80 ms frame-time distribution

Method: for each arm, select `voicechat.metrics` records with
`server_step_ms > 0`, discard the first 1,500 positive server frames, then
compute the empirical distribution over the remaining frames. Percentiles use
linear interpolation. The arms used the same 900-second fixture and config and
ran baseline first, candidate second, with no overlap.

### Raw per-frame `server_step_ms`

| Statistic | production-candidate-1 | assembled candidate | candidate - baseline | Delta |
|---|---:|---:|---:|---:|
| sample count | 8,923 | 8,938 | +15 | — |
| mean | 69.666 ms | 67.948 ms | **-1.717 ms** | **-2.465%** |
| p50 | 79.433 ms | 79.647 ms | +0.214 ms | +0.269% |
| p95 | 101.884 ms | 95.109 ms | **-6.774 ms** | **-6.649%** |
| **p99** | **109.040 ms** | **99.498 ms** | **-9.542 ms** | **-8.751%** |

The baseline had 4,443/8,923 raw frames above 80 ms; the candidate had
4,458/8,938. The candidate's benefit is a shorter high tail and lower mean,
not a change in the alternating distribution's approximately half-over-budget
count. The small +0.214 ms p50 change is not presented as an improvement.

### Sliding two-frame metric used at candidate-1 freeze

Each window is `(frame[i] + frame[i+1]) / 2` over the same settled raw series.

| Statistic | production-candidate-1 | assembled candidate | candidate - baseline | Delta |
|---|---:|---:|---:|---:|
| window count | 8,922 | 8,937 | +15 | — |
| mean | 69.666 ms | 67.947 ms | **-1.719 ms** | **-2.467%** |
| p50 | 69.566 ms | 68.501 ms | **-1.066 ms** | **-1.532%** |
| p95 | 76.681 ms | 71.847 ms | **-4.834 ms** | **-6.304%** |
| **p99** | **84.392 ms** | **84.776 ms** | +0.384 ms | +0.455% |

The candidate has 161/8,937 windows above 80 ms versus 153/8,922 for the
baseline. The governing p95 improves and passes; the +0.384 ms two-frame p99
movement is small and is reported rather than hidden.

For frozen-reference comparability, the prior three-turn candidate-1 numbers
were 69.635 ms mean and 82.799 ms sliding two-frame p95 over 445 steady frames.
The long-soak measurements here are 69.666/76.681 ms for the current baseline
and 67.947/71.847 ms for the candidate. The current sequential A/B is the
promotion evidence; the shorter historical fixture is context only.

P99 caveat: 8,923 and 8,938 settled raw observations put roughly 89 samples
in each empirical upper 1%, so this is not a thin short-run p99. It is still
one sequential soak per image, not a repeated-arm confidence interval; normal
run-to-run tail movement remains possible. That qualification is especially
relevant to the small two-frame-p99 difference, but not large enough to erase
the measured 9.542 ms raw-p99 improvement.

## 6. Release-soak functional and queue results

| Item | production-candidate-1 | assembled candidate |
|---|---:|---:|
| duration / source frames | 900 s / 11,250 | 900 s / 11,250 |
| metric / positive server frames | 10,427 / 10,423 | 10,441 / 10,438 |
| settled positive frames | 8,923 | 8,938 |
| typed completed / answered | 15/15 | 15/15 |
| unexpected errors | 0 | 0 |
| queue first / last block mean | 61.728 / 70.837 ms | 58.395 / 69.543 ms |
| fitted queue slope | +0.347 ms/min | +0.355 ms/min |
| peak implied serial debt | 307.872 ms | 430.347 ms |
| queue gate | pass | pass |

Both slopes are below the +1 ms/min limit, both late means are below 80 ms,
and both peak debts are below 1 second.

## 7. Step-4 stacked retest and final disposition

The byte-exact winning table from `314762f` was overlaid on the Change-A +
Step-7 + Step-9 instrument stack. Control and v2 each used one resident engine
and three independent 90-second strict-v3 runs. Both arms compiled fresh; no
Step-9 cache tree was shared across the source-changing comparison. All six
runs completed their metric sequence and typed fixture without unexpected
errors.

| Statistic | stacked control | stacked Step-4 v2 | v2 - control |
|---|---:|---:|---:|
| pooled frames | 3,141 | 3,148 | +7 |
| pooled mean | 67.980 ms | 67.404 ms | -0.576 ms |
| pooled p50 | 84.374 ms | 81.380 ms | -2.995 ms |
| pooled p95 | 93.443 ms | 90.434 ms | -3.009 ms |
| pooled p99 | 94.841 ms | 96.770 ms | **+1.929 ms** |
| mean of three run means | 67.982 ms | 67.404 ms | -0.578 ms |
| sample SD of run means | 0.666 ms | 0.149 ms | — |

Control run means were 68.703, 67.853, and 67.390 ms; v2 run means were
67.250, 67.415, and 67.547 ms. The -0.578 ms mean-of-runs delta is smaller
than the control's 0.666 ms run-mean sample deviation, the arm order was not
counterbalanced, and p99 regressed. This does not meet the required
clear-outside-noise standard. **Step-4 v2 is permanently RETIRED.**

## 8. Cold start and memory guardrails

The final image independently reached readiness in 253.983 s; its release-soak
launch reached readiness in 255.999 s. Both are in the promoted ~250-second
class. The baseline soak launch took 380.920 s. Cache validation succeeded on
both candidate launches. The startup comparison is measured directly and is
not combined with any steady-state step-report savings.

Every durable engine launch used Nano memory utilization 0.35, EarTTS 0.10,
Docker `--memory 90g --memory-swap 90g`, an empty-`docker ps -q` preflight,
MemAvailable above 100 GiB, and a 10-second watchdog that kills below 15 GiB.
No two engines ran concurrently.

| Launch | Preflight/first watchdog GiB | Minimum GiB | End state |
|---|---:|---:|---|
| fresh assembled cache source | 118.051 | 46.476 | stopped; no watchdog trip |
| Step-7 1,389-call gate | 117.975 | 70.595 | exit 0; not OOM-killed |
| independent final cold start | 118.081 | 46.624 | stopped; not OOM-killed |
| retained post-FC fixture | 118.077 | 42.918 | planned stop; not OOM-killed |
| Step-4 stacked control | 118.018 | 43.230 | planned stop; not OOM-killed |
| Step-4 stacked v2 | 118.045 | 43.024 | planned stop; not OOM-killed |
| release soak baseline | 118.009 | 42.732 | planned stop; not OOM-killed |
| release soak candidate | 117.929 | 42.619 | planned stop; not OOM-killed |

The Step-3 bitwise gate was a bounded single-process EarTTS check rather than
a durable two-engine service launch; it used the same 90 GiB cap after an
empty-container, 118.070 GiB preflight. The candidate soak's final watchdog
record is a normal sample four seconds before the planned stop because the
wrapper removed the stopped container before the next 10-second poll; its
final Docker inspect is `ExitCode=143`, `OOMKilled=false`, and the watchdog has
no low-memory or concurrency event. Final host state was `docker ps -q` empty
with 117.927 GiB MemAvailable.

## 9. Evidence and cleanup

Authoritative evidence root:

`/home/khkramer/src/nvidia-voice-chat/artifacts/container/dgx_spark/latest_voicechat/profiles/strict-v3-pair-20260806/promotion-assembly/`

It contains the fresh cache-source boot, independent cold start, Step-3 and
Step-7 bitwise reports, post-FC fixture, six Step-4 retest runs, both release
soaks, server logs, Docker inspections, and memory watchdog streams. No global
Docker prune was run. Only promotion-created intermediate/test image tags were
removed after qualification; the final candidate image was retained.
