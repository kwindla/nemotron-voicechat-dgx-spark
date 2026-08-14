# Nano graph-envelope bandwidth attribution (Step 5 side-study)

**Date:** 2026-08-14 UTC
**Preregistration:** `reports/step5-bandwidth-attribution/preregistration.md`,
SHA-256 `0f63864456889ad12b8758624036609f1911b2dba5d189afc08b25a034eb99a6`,
sealed before any Phase B/C execution.
**Disposition:** descriptive. No ranking authority is created; the Step 4
ranking's blocked dispositions and the Step 5 exact-gate prerequisite are
unchanged.

## Summary of findings

1. **The graph envelope is bytes-bound at the achievable bus ceiling, not
   idle.** Measured achievable device-memory read bandwidth on this GB10 is
   **239.6 GB/s** (87.8% of the 273 GB/s spec). The per-step streamed byte
   set, corrected for the runtime's bf16 cast of unquantized modules
   (≈ 10.1 GB), divided by the measured kernel-union time reproduces the
   envelope at near-ceiling throughput. The prior "≈ 25% unexplained
   envelope" was unnecessary bytes plus unreachable spec headroom, not
   recoverable idle bandwidth.
2. **`function_head` costs ≈ 7.3 ms of every frame.** The GPTQ recipe
   quantizes `lm_head` but deliberately excludes `function_head`,
   `embed_tokens`, and the four attention layers' `q/k/v/o` projections.
   At runtime these run as bf16 cuBLAS GEMVs: exactly 9 instances per
   decode step — one vocab-sized (131072×4480) `function_head` GEMV at
   **7.292 ms median**, plus 8 attention-projection GEMVs totaling
   ≈ 1.9 ms — together 9.176 ms/step at an implied 176 GB/s (73% of
   ceiling).
3. **The already-quantized path is near-optimal.** The 105 Marlin W8 GEMVs
   (54 Mamba projections, 50 MLP projections, `lm_head`) plus their fused
   Triton neighbors run 37.7 ms/step at ≈ 220 GB/s — 92% of the measured
   ceiling. No kernel family with materially sub-ceiling throughput and
   already-minimal bytes was found.
4. **The fusion/wrapper screening floor is NOT met.** Mean intra-envelope
   idle between graph-node kernels is **1.048 ms/step**, and no family
   qualifies under the preregistered sub-60%-of-ceiling-with-minimal-bytes
   clause, so the preregistered ≥ 1.5 ms/frame floor for the wrapper
   graphing/fusion row is not satisfied. That row screens out on this
   evidence.
5. **New candidate (descriptive): quantization extension.** Quantizing
   `function_head` to the same GPTQ-W8/Marlin path as `lm_head` would, at
   the measured Marlin throughput, reduce its cost from ≈ 7.3 ms to
   ≈ 2.7 ms — **≈ 4.6 ms/frame recovered**. Adding the attention
   projections at W8 yields ≈ 5.5 ms/frame total; a W4 `function_head`
   raises the family estimate toward ≈ 6.9 ms/frame. These are descriptive
   estimates for a NEW candidate family requiring its own qualification
   (numeric change by design); precedent exists in-recipe: the equally
   vocab-sized `lm_head` already ships GPTQ-W8.

## Hypothesis dispositions (preregistered)

- **H-BYTES: CONFIRMED (with correction).** The checkpoint-level F32 model
  (11.73 GB/step) overcounts: kernel evidence shows the unquantized modules
  execute in bf16, halving their bytes (streamed set ≈ 10.1 GB/step). At
  the measured ceiling this reproduces the envelope; the F32-based implied
  throughput (255.9 GB/s) exceeded the measured ceiling, which is what
  exposed the cast.
- **H-FH: CONFIRMED in bf16 form.** One vocab-sized GEMV per step,
  7.292 ms median (preregistered F32 magnitude 8–10 ms was halved by the
  bf16 cast; instance count and per-step presence confirmed exactly).
- **H-ATTN: CONFIRMED in bf16 form,** ≈ 1.9 ms rather than the F32-based
  3–4 ms.
- **H-GAP: CONFIRMED.** 1.048 ms intra-envelope idle (< 3 ms bound).

## Methods and evidence

- **Phase A (bytes model):** safetensors headers + `config.json`
  (`tools/benchmark/bandwidth_attribution.py bytes-model`); fail-closed
  family classification (a first run failed on unclassified GPTQ `g_idx`
  tensors and was corrected before any GPU phase); output
  `data/bytes-model.json`.
- **Phase B (ceiling):** read-reduction and D2D copy at 1/2/4 GB, 20
  repeats after 5 warmups; read-side medians 236.9/238.6/239.6 GB/s; copy
  219–223 GB/s; study ceiling 239.59 GB/s; output `data/ceiling.json`.
- **Phase C (node-level profile):** unchanged Step 4d harness, 2 warmup +
  10 measured blocks, fixed state selection (capture manifest SHA-256
  `55d9e815ec235a588b0aa6eacb4a97a2d5a77f19aa00954d0c2e205d4108d1aa`),
  Nsight Systems `--cuda-graph-trace=node`, capture-range gating unchanged.
  **Attempt 1 is retained-invalid:** the fresh graph engine landed in the
  Step 5-characterized call-39 divergent basin (token 1044 vs 10592 in all
  12 blocks) and failed the harness token contract
  (`data/profile-component-attempt1-call39-divergent-basin.json`,
  `data/nsys-attempt1-call39-divergent-basin/`). **Attempt 2 (designated)**
  landed in the exact basin: 12/12 valid blocks, no token mismatches, exit
  0 (`data/profile-component.json`). 60 capture ranges exported to SQLite
  (`data/sqlite/`) and analyzed (`data/kernel-attribution.json`).
- **Perturbation:** traced mean interface wall 54.99 ms vs untraced Step 4d
  53.56/53.60 ms — 2.6% inflation, within the preregistered 15% rule;
  absolute traced milliseconds are reported directly.
- **Accounting closure:** per range, wall 54.99 = kernel union 48.37 +
  non-kernel 6.62 (the known Step 4d wrapper/CPU residual, measured there
  as 7.7–8.0 ms untraced). Unattributed kernel time 0.18 ms/range.
  Family table (ms/step at trace-on): Marlin+fused 37.70 (×155), bf16
  GEMV 9.18 (×9), Mamba scan 0.73 (×27), norm/elementwise 0.41 (×257),
  causal-conv 0.11 (×27), sampling 0.08, attention-core 0.01.

## Decision-rule outcomes (preregistered)

1. **P-5 wrapper/fusion screening: floor NOT met** (1.048 ms qualifying).
   The row screens out on this profile; the Step 4 ranking's demand for an
   attributed ≥ 1.5 ms/frame stands unmet on the best available evidence.
2. **Bytes-reduction candidate: CONFIRMED at instance level.** Reported
   estimates: `function_head` W8 ≈ 4.6 ms/frame; + attention W8
   ≈ 5.5 ms/frame; W4 variants larger. Descriptive only; requires new
   candidate qualification (exactness gates do not apply to a deliberate
   numeric change; behavioral qualification per the plan's promotion
   contract does).
3. Negative/ambiguous results retained: none ambiguous; the divergent-basin
   attempt is retained as further prevalence evidence for the Step 5 graph
   construction defect (one additional divergent fresh construction, one
   additional exact fresh construction).

## Operations

Pre-stop service identity recorded; qualification container stopped with
evidence; component container `bwattr-component` (unchanged image
`sha256:ea15693c...`) ran Phases B/C; service restored with the retained v4
script and identity re-verified (`data/preflight/`). Component container
logs and inspect retained; container removed after retention.

## Relation to the funded wedge campaign

Independent and complementary: pairing amortizes the whole streamed set
across two frames; the quantization extension shrinks the set. Both together
target ≈ 84 ms → well under budget; either alone covers the ≈ 4–6 ms
overrun if it qualifies.
