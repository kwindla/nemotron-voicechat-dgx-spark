# Step 8: bounded perception/EarTTS overlap A/B

Date: 2026-08-07  
Branch: `codex/step8-overlap-ab`  
Base commit: `9ee8d21eb1c8dd197cf6e2a17f9d62deb827de4f` (`codex/step7-pair-graph`)  
Disposition: **DEFENDED STOP — do not promote the overlap scheduler**

## Result

The dependency hypothesis is valid, but the performance hypothesis is not. Perception for frame `t+1` can be computed on a second CUDA stream during EarTTS for frame `t` without changing its encoded outputs or recurrent state. In the fixed end-to-end A/B, however, the candidate was **1.076 ms slower per logical frame**:

| arm | mean logical-frame cycle | p50 | p95 | n |
|---|---:|---:|---:|---:|
| Step-7 FULL baseline | 62.957 ms | 61.001 ms | 88.336 ms | 543 |
| perception/EarTTS overlap | 64.034 ms | 62.056 ms | 88.646 ms | 544 |
| candidate minus baseline | **+1.076 ms (+1.71%)** | +1.055 ms | +0.310 ms | — |

Net gain, defined as baseline minus candidate, is **-1.076 ms**. This is below the pre-committed 2 ms promotion floor and is a regression. Perception and EarTTS both inflated under actual overlap, erasing the schedulable interval. The stop rule therefore fired; no scheduler variants were attempted.

The retained Step-7 FULL result was 63.595 ms/logical frame. The Step-8 resident baseline was 0.638 ms faster than that historical run, while the candidate was 0.439 ms slower. The decision uses the same-process Step-8 baseline rather than that cross-run difference.

## A/B design and implementation

The base image was `nemotron-local/voicechat-vllm:step7-pair-full-graph-hf1` (`sha256:e935fbf2e2bda49e152a12a5373e02b56a44823572fd7d3bb450aa3da37d4be1`). The uninstrumented A/B image was `sha256:3d2aac7ee8af4ef8c6ac0523815641a61c4d1c2e43e03d88460a91fde9fe6886`.

- Baseline: Step-7 FULL pair graph unchanged.
- Candidate: stage the already-known next 80 ms audio frame; immediately before the synchronous EarTTS call, dispatch `perception(t+1)` on one dedicated `torch.cuda.Stream`; join it after EarTTS and before codec decode.
- No EarTTS bridge was added. Nano, RNNT, function handling, EarTTS invocation, and codec order were otherwise unchanged.
- The CPU codec remained process-backed and capped to two cores/threads.
- One resident process alternated by request ID: baseline requests 2/4/6 and candidate requests 3/5/7. This avoids cross-process vLLM RNG comparisons.
- Each request used the retained 75-second, three-prompt post-FC traffic. The short-run queue model reports `insufficient settled frames` by construction; the binding fixture result is three completed and answered typed jobs with no unexpected protocol errors.

The candidate launched 275 overlaps. Mean actual overlap was 20.112 ms. It consumed 272 prefetched perception results; three launches occurred at terminal/reset boundaries and were joined but not consumed.

Evidence: `step8/e2e-overlap-ab/analysis.json`, `server.log`, `sessions/`, `traces/`, `container-inspect*.json`, and `watchdog.jsonl` under the strict-v3 pair profile root.

## Contention control

The E06-style stage control reproduced the contested behavior on the Step-7 FULL stack:

| measurement | baseline | candidate/all frames | candidate/overlap-only | overlap-only inflation vs baseline |
|---|---:|---:|---:|---:|
| perception | 11.854 ms | 14.155 ms | 20.165 ms | **+8.311 ms (+70.1%)** |
| EarTTS round trip | 12.521 ms | 15.509 ms | 23.261 ms | **+10.740 ms (+85.8%)** |

Across all candidate frames, perception inflated by 2.301 ms (+19.4%) and EarTTS by 2.988 ms (+23.9%). These are diluted because only 275 of 1,092 candidate frames carried an overlap. On the overlapped subset, both stages approximately doubled toward a common 20–23 ms critical interval instead of hiding one another.

E06 reported perception increasing from 21.9 to 31.9 ms under overlap plus codec contention. Step 8 excludes E06's EarTTS bridge and retains the two-core process codec, yet the direction repeats. This separates dependency legality from shared-resource viability: the bridge conflation was incorrect, but the contention warning was real.

## LPDDR/unified-memory counter attribution

Nsight Compute 2026.1 captured the GB10 counter validated in Step 2:

`physical read bytes = lts__d_sectors_fill_sysmem.sum * 32`

The matched counter ranges were request-local frame 1. Kernel replay was injected into the application process only, which owns perception; injecting replay into the vLLM children consumed EarTTS's fixed 0.10 allocation and was rejected rather than relaxing the guardrail. EarTTS remained live as the concurrent load. Instrumented durations are not used for performance because kernel replay perturbs scheduling.

| arm | profiled kernels | sysmem-fill sectors | physical read bytes |
|---|---:|---:|---:|
| baseline | 1,785 | 39,249,313 | 1,255,978,016 |
| candidate | 1,784 | 39,159,315 | 1,253,098,080 |
| candidate minus baseline | -1 | -89,998 | **-2,879,936 (-0.229%)** |

The physical-read totals agree within 0.23%, inside Step 2's 0.8% validation envelope. Thus the candidate did not introduce a larger perception working set or extra unified-memory fill volume. The same approximately 1.25 GB application-process traffic became slower only when it was co-scheduled with EarTTS. Together with the uninstrumented overlap-only inflation, this supports LPDDR/shared-GPU contention rather than extra work or paging as the cause. The counter does not prove LPDDR is the sole contended resource; SM and cache competition can coexist.

Evidence: `step8/bandwidth/bandwidth-summary.json`, arm CSVs, raw `.ncu-rep` files, server logs with `STEP8_NCU_START/STOP`, container inspections, and watchdog logs.

Excluded profiler attempts are retained in the same evidence tree:

1. an invalid range-replay option failed before engine initialization;
2. all-process profiler injection left no EarTTS KV-cache headroom at the mandatory 0.10 cap;
3. range replay rejected `cuGraphLaunch` and emitted no valid range;
4. the first candidate routing preflight used silence only, did not instantiate model request 2, and emitted no counter report.

None contributed a counter or timing sample. The successful pair used identical application-only kernel replay.

## Exactness and retained functional gate

All binding checks passed.

### Pair accept/reject and rollback

The same-process Step-7 pair gate completed 1,389/1,389 calls with no first mismatch:

- 675 cache checks, all equal;
- 663 accepted pairs;
- 12 rejected/rollback cases;
- 664 buffered steps and 49 conditional singles;
- outputs, decisions, and compared cache tensors were bit-exact.

This gate ran on image `sha256:44572ec67145351cb1302b2862d07ff5f05bbd0e6143f356b410b2e4266b0734`. The subsequent `3d2aac...` rebuild changed only the retained Speech diff used by the runtime audit; the installed wrapper and Nano worker runtime were unchanged.

### Scheduler/recurrent-state exactness

The full stack ran baseline request 2 and candidate request 3 in one process with `VOICECHAT_STEP8_EXACT_GATE=1`. For every candidate prefetch, the wrapper cloned the input recurrent cache, recomputed perception serially, and byte-compared:

- source encoding;
- raw ASR encoding;
- `cache_last_channel`;
- `cache_last_time`;
- `cache_last_channel_len`.

All **93/93** overlap checks passed, and 92 prefetched results were consumed. Diagnostic recomputation invalidates this run's timings; it is exactness evidence only.

### Retained post-FC fixture

The exactness run completed/answered 3/3 prompts in both arms with no unexpected errors. The uninstrumented A/B completed/answered 3/3 prompts in each of six sessions, **18/18 total**, again with no unexpected errors. The analyzer's retained functional fixture result is `true`.

The final local image (`sha256:096d51df7b267c3911bb55d750feb361ffaca327452c9632a4b18d010b7e54fa`) adds only dormant, environment-gated NCU start/stop hooks to the measured `3d2aac...` runtime. They are inactive unless a counter target request is explicitly configured.

## Memory guardrails

Every launch used the required controls:

- Nano `--gpu-memory-utilization 0.35`;
- EarTTS `0.10`;
- Docker `--memory 90g --memory-swap 90g`;
- empty running-container preflight and host MemAvailable greater than 100 GiB;
- one engine container at a time;
- 10-second watchdog, kill below 15 GiB.

| run | minimum MemAvailable | terminal event | watchdog/OOM kill |
|---|---:|---|---|
| pair exactness | 70.435 GiB | container stopped | no |
| full-stack scheduler exactness | 42.818 GiB | container stopped | no |
| uninstrumented six-session A/B | 41.959 GiB | container stopped | no |
| successful NCU arms | 41.963 GiB | container stopped | no |

All inspected containers had 96,636,764,160-byte memory and swap limits and `OOMKilled=false`. At completion, Docker had no running containers and host MemAvailable was approximately 118.18 GiB.

## Disposition

**STOP. Do not promote perception/EarTTS overlap. Keep the Step-7 FULL pair graph as the campaign endpoint.**

The experiment establishes that the scheduler move is dependency-legal and bit-exact, resolving the E06 bridge conflation. It also establishes that, on GB10 with the retained stack and codec cap, the overlap creates enough shared-resource contention to regress end-to-end logical-frame latency. The pre-committed rule forbids further overlap variants after this result.
