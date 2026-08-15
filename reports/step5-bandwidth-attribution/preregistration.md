# Preregistration: Nano graph-envelope bandwidth attribution (Step 5 side-study)

Status: sealed before any measurement run. Descriptive study. No ranking
authority is created or implied by any result below; the Step 4 ranking's
blocked/`descriptive` dispositions and the Step 5 exact-gate prerequisite are
unchanged by this study.

## Motivation and prior numbers (fixed before runs)

Step 4d measured, for the released W8 Nano at the three matched
content/PAD interface states (ten-block estimates): content interface wall
53.560786 ms with CUDA-graph envelope 45.847206 ms; PAD 53.595332 /
45.579065 ms. The graph envelope was captured at whole-graph granularity, so
no per-kernel attribution inside the envelope exists yet.

Checkpoint facts established from safetensors headers and
`quantize_config.json` before profiling (read-only host inspection,
2026-08-14): total checkpoint 13.866 GB across 4 shards. GPTQ W8 (group 128,
sym, desc_act=false) covers `mixer.{in,out,up,down}_proj` and `lm_head`
(qweight+scales+qzeros ≈ 8.22 GB). Deliberately excluded from quantization
and stored F32: `embed_tokens` (2.349 GB), `function_head` (2.349 GB),
attention `q/o` (0.367 GB × 4 each family) and `k/v` (0.073 GB × 4 each) —
≈ 0.881 GB attention total.

Pre-stated hypotheses (to be confirmed or refuted by per-kernel evidence):

- H-BYTES: the per-decode-step streamed read set is ≈ 11.4–11.6 GB
  (quantized projections + lm_head + F32 attention projections + F32
  function_head + states; embeddings are a gather). At the measured envelope
  this implies ≈ 250 GB/s achieved — near bus saturation — leaving little
  idle-bus headroom.
- H-FH: a single F32 vocab-sized (4480×131072) GEMV/GEMM kernel instance per
  step, attributable to `function_head`, accounts for ≈ 8–10 ms of the
  envelope.
- H-ATTN: unquantized F32 attention projection kernels account for
  ≈ 3–4 ms of the envelope.
- H-GAP: intra-envelope non-kernel time (bus idle between graph nodes) is
  small relative to the above (< 3 ms).

## Phases and methods

**Phase A — bytes model (host, read-only).** Parse safetensors headers;
classify every tensor into declared families; compute per-decode-step read
bytes under explicit assumptions: all quantized projection payloads, F32
attention projections, F32 function_head, and lm_head payload read once per
step; `embed_tokens`/`embed_asr_tokens` counted as gather (excluded from
streaming, disclosed); Mamba conv/SSM state read+write and 4-layer KV read
computed from `config.json` dims at the capture's observed sequence lengths;
activations disclosed as unmodeled. Every tensor must land in exactly one
family or be listed unclassified — no silent drops.

**Phase B — bandwidth ceiling (GPU, component container).** Measured
achievable device-memory bandwidth: read-reduction over contiguous tensors of
1, 2, and 4 GB and device-to-device copy at the same sizes; ≥20 timed
repeats each after ≥5 warmups; bytes counted explicitly (copy = read+write);
report per-size medians and the study ceiling = maximum sustained read-side
median. Spec 273 GB/s is context only; utilization is reported against the
measured ceiling.

**Phase C — node-level kernel profile (GPU, component container).** Rerun the
unchanged Step 4d harness (`tools/benchmark/nano_interface_decomposition.py`,
`run --mode profile`, 2 warmup + 10 measured blocks, same fixed state
selection; capture root manifest SHA-256 must equal
`55d9e815ec235a588b0aa6eacb4a97a2d5a77f19aa00954d0c2e205d4108d1aa`) under
Nsight Systems with `--cuda-graph-trace=node` added so child kernels of the
CUDA graph are recorded. Same capture-range gating (cudaProfilerApi), same
NVTX range naming. Export to SQLite and retain raw artifacts.

**Analysis (fail-closed).** For each matched Step 4d NVTX range: clip kernel
intervals to the range; aggregate by demangled kernel name (sum, count,
mean); compute kernel interval-union and range-wall-minus-union. Kernel-name
→ family mapping is by a declared regex table; unmatched kernel time is
reported as unattributed, never absorbed. Family achieved GB/s = Phase A
family bytes / measured family kernel time, reported against the Phase B
ceiling. Instance-count consistency checks: function_head and lm_head must
appear a fixed number of instances per step for their attribution to stand;
otherwise the corresponding hypothesis is reported unresolved, not assumed.

**Perturbation rule.** Node-level tracing adds per-node overhead. The traced
NVTX interface wall distribution is reported against the Step 4d untraced
wall (53.56/53.60 ms). If traced wall inflates by more than 15%, absolute
per-kernel milliseconds are reported as traced-condition values and the
family *shares* of the envelope become the primary estimands; this rule is
fixed now.

## Decision rules (fixed before runs)

1. **P-5 screening for the wrapper/fusion row:** the row's ≥1.5 ms/frame
   floor is satisfied only if measured intra-envelope non-kernel time plus
   kernel time in families whose achieved bandwidth is below 60% of the
   measured ceiling (and whose bytes are already minimal) sums to
   ≥1.5 ms/frame. Satisfying the floor does not unblock ranking (exact-gate
   prerequisite unchanged).
2. **Bytes-reduction candidates:** if H-FH or H-ATTN is confirmed by
   matched kernel instances, report estimated recoverable milliseconds for
   W8/W4 quantization of the confirmed family as
   `measured_family_ms × (1 − target_bytes/current_bytes)`, labeled a
   descriptive estimate for a NEW candidate family (quantization extension)
   requiring its own qualification; it acquires no ranking authority here.
3. Negative/ambiguous results are retained and reported as such.

## Operational plan

Follow the retained Step 4d component pattern: record the running
qualification service container's `/health` identity, stop it with retained
stop evidence, run Phases B and C in a fresh component container from the
unchanged image `sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`
(engine args: `/models/derived/nano`, max_model_len 12288,
gpu_memory_utilization 0.42, dtype bfloat16, enforce_eager False, speech root
`/opt/Speech`), then restart the retained v4 script and verify identity
(revision, model/Nano/EarTTS SHA-256) matches pre-stop. Deviations are
recorded; failed runs are retained, never silently retried.

## Deliverables

`reports/step5-bandwidth-attribution-20260814.md`; data under
`reports/step5-bandwidth-attribution/data/` (raw `.nsys-rep`, SQLite,
bytes-model JSON, ceiling JSON, analyzer JSON, preflight/identity evidence,
SHA-256 manifest). Analyzer: `tools/benchmark/bandwidth_attribution.py` with
tests. This file is sealed by SHA-256 recorded in the session log and report
before any Phase B/C execution.
