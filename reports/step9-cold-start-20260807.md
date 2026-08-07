# Step 9 cold-start campaign — 2026-08-07

## Decision

Promote exactly these two coupled startup levers:

1. the keyed, image-baked, fully inventoried vLLM cache; and
2. the production-shape capture contract `257,80,16,2,1`, with strict
   post-ready coverage and compile assertions.

The promotion candidate is
`nemotron-local/voicechat-vllm:step9-cache-reduced`
(`sha256:6d90d390496d...`). It reached Playground-ready in 247.954 s and
254.087 s on two valid launches, saving 128.796–134.929 s (33.6–35.2%)
from the 382.883 s measured strict-v3 baseline. This beats the plan's
270–300 s intermediate target but does not reach the 200–230 s stretch
target.

Do **not** promote exact-prompt warmup. After generating a separate cache
from the exact production prompt and keying that prompt into the image, it
still took 259.997 s. The fully controlled exact-prompt results show no
startup saving. Do not promote prefetch or meta-init: neither received a
valid A/B in this campaign (see Dispositions).

All reported engine launches were sequential. No launch crossed the 15 GiB
watchdog floor and no watchdog killed a container.

## Scope and provenance

- Branch base: `9ee8d21` (`codex/step7-pair-graph`).
- Baseline image:
  `nemotron-local/voicechat-vllm:step7-pair-full-graph-hf1`,
  `sha256:e935fbf2e2bd...`.
- Instrument image: `nemotron-local/voicechat-vllm:step9-instrument`,
  `sha256:15bdcc8677c9...`.
- Promotion image: `nemotron-local/voicechat-vllm:step9-cache-reduced`,
  `sha256:6d90d390496d...`.
- Exact-prompt diagnostic image:
  `nemotron-local/voicechat-vllm:step9-cache-reduced-exact-keyed`,
  `sha256:038830d3ee21...` (not promoted).
- External evidence root:
  `/home/khkramer/src/nvidia-voice-chat/artifacts/container/dgx_spark/latest_voicechat/profiles/strict-v3-pair-20260806/step9`.
- Nano/EarTTS memory-utilization contracts were 0.35/0.10; every full
  launch used Docker `--memory 90g --memory-swap 90g`.

## 1. Additive strict-v3 startup account

The authoritative baseline arm is `baseline-a2`. Machine-readable markers
were added around the server's phases and reconciled with Docker log
timestamps and the host monotonic launch/readiness clock. Adjacent phase
boundaries are mutually exclusive, so the table is additive rather than a
collection of overlapping profiler totals.

| Phase | Seconds |
|---|---:|
| Container launch and Python import | 0.383459 |
| Module import to main entry | 0.001729 |
| Main dispatch to pipeline build | 0.000471 |
| Pre-model pipeline setup | 33.968048 |
| NeMo CPU model-structure initialization | 40.656654 |
| Checkpoint hydration and EarTTS handoff | 2.194285 |
| EarTTS vLLM engine initialization | 119.375010 |
| Inter-engine fixed setup | 1.782465 |
| Nano vLLM engine initialization | 154.015132 |
| Post-engine pipeline setup | 13.059035 |
| Pipeline fallback-prompt warmup | 10.315650 |
| Realtime engine construction | 0.000074 |
| Exact production-prompt continuation warmup | 2.004017 |
| Application construction and server entry | 0.068079 |
| Uvicorn startup | 3.306856 |
| Ready sentinel to successful health poll | 1.752032 |
| **Additive sum** | **382.882994** |
| **Host launch-to-ready wall** | **382.882994** |

The wall/phase reconciliation error is 0.000000272 s, caused by using the
host monotonic clock for wall duration and wall-clock nanoseconds for log
boundaries. The checked-in machine-readable account is
`reports/step9-baseline-startup-account.json`.

This confirms that the current cold-start centers are the two serialized
vLLM engines (273.390 s combined), then NeMo CPU structure initialization
(40.657 s) and pre-model setup (33.968 s). It also corrects older logs for
this exact image: the current fallback warmup is 10.316 s, not 35.9 s.

## 2. Per-lever A/B

| Arm | Cache | Capture sizes | Warmup prompt | Ready (s) | Saved vs baseline (s) | Gate status |
|---|---|---|---|---:|---:|---|
| `baseline-a2` | ephemeral cold | vLLM default 54 | fallback | 382.883 | — | post-FC 3/3 |
| `cache-default` | default baked | default 54 | fallback | 268.050 | 114.833 | cache validation/recovery pass |
| `capture-reduced` | ephemeral cold | 257/80/16/2/1 | fallback | 354.671 | 28.212 | post-FC + strict coverage pass |
| `reduced-cache-a` | reduced baked | 257/80/16/2/1 | fallback | 247.954 | 134.929 | bitwise gate pass |
| `reduced-cache-fixture2` | reduced baked | 257/80/16/2/1 | fallback | 254.087 | 128.796 | promotion-image post-FC pass |
| `reduced-cache-exact` | fallback-built reduced cache | 257/80/16/2/1 | exact | 259.973 | 122.910 | diagnostic only; cache lacked exact-prefill artifact |
| `capture-reduced-exact-source` | ephemeral cold | 257/80/16/2/1 | exact | 356.647 | 26.236 | exact cache source |
| `reduced-cache-exact-keyed` | exact-built reduced cache | 257/80/16/2/1 | exact | 259.997 | 122.886 | post-FC pass; performance STOP |

The fallback-prompt promotion runs span 6.133 s. Their mean is 251.021 s.
The two exact-prompt cached arms are 259.973/259.997 s; the prompt-keyed
result is therefore not a cache-miss artifact. Its pipeline warmup fell
from 10.021 s to 2.598 s when baked from the correct prompt, but normal
engine-init variance consumed that local saving and total readiness did not
improve. Against the promotion mean, exact-prompt readiness regressed by
8.976 s.

## 3. Immutable cache and recovery gates

The cache manifest includes an explicit key, `state=complete`, runtime
contract, provenance, and SHA-256 plus byte size for every file. Validation
runs before pipeline construction. The runtime contract covers architecture,
Python, torch, CUDA, vLLM, CUDA capability, capture sizes, and the Step-7
pair-FULL-graph contract. A rejected tree is never exposed to vLLM:
`VLLM_CACHE_ROOT` is redirected to a unique empty `/tmp` directory and the
ordinary cold compilation path runs.

Reduced promotion cache:

- key:
  `step9-reduced-e935fbf2-vllm0171-gb10sm121-shapes257-80-16-2-1-v1`;
- 2,865 inventoried files, 121,855,721 bytes;
- manifest SHA-256:
  `d7285c7ffae8c6f247226eb2f7e204fa2fe7c6e37e19fdd8dea113d2d8f475f2`
  (the manifest is embedded in the image and provenance is in
  `reports/step9-cache-reduced-provenance.json`).

Recovery probes on the reduced image produced:

- intact/key-matched: `validated`, all 2,865 files and 121,855,721 bytes;
- stale key: `cold_fallback`, reason `cache_key_mismatch`;
- mutated first payload: `cold_fallback`, reason
  `inventory_size_mismatch`;
- missing `VOICECHAT_NANO_PAIR_FULL_GRAPH=1`: `cold_fallback`, reason
  `runtime_contract_mismatch`, followed by a real cold compile rather than
  reuse.

The last probe arose during a confirmation launch and demonstrates the
runtime invalidation gate under an actual engine boot. It was intentionally
excluded from timing results. Unit tests cover valid, stale-key, and corrupt
payload paths.

The exact-prompt diagnostic cache used a distinct prompt-qualified key:
`step9-reduced-exact-c56b27c2-e935fbf2-vllm0171-gb10sm121-shapes257-80-16-2-1-v1`.
It contains 2,849 files/121,001,767 bytes; its manifest SHA-256 is
`e3acee3883b807a537dc37bdd81af5e84661c611291c6fad106575585105cdf1`.

## 4. Shape inventory, no-late-compile gate, and regate

The baseline inventory separates vLLM's startup capture sweep from real
post-ready descriptors. The retained post-FC and Step-7 workloads exercised
these post-ready token counts:

- `1`: uniform decode;
- `2`: uniform decode, including `query_len=2` paired Nano execution;
- `257`: nonuniform tool-prompt prefill;
- `80`: nonuniform tool-response prefill.

The current production system-prompt prefill adds the pre-ready size `16`.
This yields the descending capture contract `257,80,16,2,1`. Startup-only
default sweep descriptors (4, 8, 24 ... 256) are not production demand and
were not retained.

The vLLM patch is hash-qualified to the exact source files in vLLM
`0.17.1.dev0+gb31e9326a.fi065`. It inventories descriptors before capture
rounding, raises on an uncovered post-ready descriptor, and raises if the
compiler backend is invoked after the application-ready sentinel. Both the
cold reduced arm and the final promotion-image post-FC fixture completed
without either assertion firing.

The binding Step-7 same-process bitwise pair gate used the full 1,389-call
teacher replay, not the earlier 157-call smoke run:

- output/decision comparisons: **1,389/1,389 exact**;
- cache comparisons: **675/675 exact**;
- final pair decisions: 663 accepted, 12 rejected, 664 buffered,
  49 conditional singles;
- first mismatch: `null`;
- result: **PASS**.

The promotion image's retained post-FC fixture completed and answered 3/3
serialized prompts, produced 3 responses, closed with `client_stop`, and
reported no unexpected errors. Its top-level sustained report says
`passed=false` only because the intentionally short fixture has fewer than
the 1,500 settled frames required by the unrelated flat-queue soak metric;
the functional fixture criterion passed. The exact-prompt keyed diagnostic
also completed 3/3 with no error. An earlier non-keyed exact diagnostic hit
one codec-worker EOF on its third turn; the correctly keyed rerun did not
reproduce it, but the lever is rejected on performance regardless.

## 5. Memory-guardrail telemetry

| Arm | Preflight MemAvailable (GiB) | Minimum sampled (GiB) | Watchdog kill |
|---|---:|---:|---|
| baseline | 118.129 | 45.709 | no |
| default cache | 118.041 | 47.759 | no |
| cold reduced capture | 118.150 | 46.625 | no |
| reduced cache A | 118.126 | 47.929 | no |
| reduced cache promotion fixture | 117.307 | 45.816 | no |
| first exact diagnostic | 118.153 | 46.617 | no |
| exact cache source | 117.980 | 46.194 | no |
| exact keyed cache | 118.054 | 47.890 | no |
| binding bitwise gate | >100 preflight | 71.112 | no |

Every arm checked that `docker ps -q` was empty and MemAvailable exceeded
100 GiB immediately before launch. Sampling was every 10 s. No two engines
or startup arms ran concurrently.

## 6. Dispositions and promotion recommendation

1. **Promote — immutable baked cache.** This is the largest measured lever
   (114.833 s alone with default captures) and it fails closed on every
   invalidation/recovery probe.
2. **Promote only with the cache — reduced capture sizes
   `257,80,16,2,1`.** It saves 28.212 s cold and contributes to the
   247.954–254.087 s combined result. The required bitwise, post-FC,
   coverage, and no-post-ready-compile gates pass.
3. **Do not promote — exact-prompt warmup.** Both the naively reused cache
   and a separately generated prompt-keyed cache took about 260 s. The
   correctly keyed functional fixture passes, but there is no startup win.
4. **Defer — Nano weight prefetch.** Repeated launches have already warmed
   the host page cache. Producing a valid cold-prefetch A/B would require a
   host-wide page-cache eviction, which was neither safe nor authorized for
   this shared campaign host. No saving is claimed.
5. **Defer — NeMo meta-init.** This requires audited Speech fork surgery and
   a new numerical regate. With caching/capture already at 248–254 s and no
   clean prefetch control remaining, it was not started speculatively. No
   saving is claimed.

Thus the exact safe adoption set is **image-baked keyed cache + reduced
capture contract + the validation/assertion machinery**. Keep the existing
fallback warmup and all steady-state Step-7 runtime behavior unchanged.

## Verification

- `52 passed` for the cache-guard tests plus the existing realtime server
  test module.
- Ruff: all modified Python files pass.
- `git diff --check`: clean.
- No global Docker prune was run; all pre-existing campaign images remain.
