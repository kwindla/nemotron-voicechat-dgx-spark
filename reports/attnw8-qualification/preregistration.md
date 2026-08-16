# Preregistration: attention-projection W8 quantization extension

Status: sealed before candidate construction and before any gate run.
Funded by user "Proceed!" on the measured basis in
`reports/step5-bandwidth-attribution-20260814.md` and the fhw8 precedent
(`reports/fhw8-qualification-20260814.md`): 8 unquantized bf16 attention
GEMVs cost ≈1.9 ms/frame at ~176 GB/s; GPTQ-W8 on the Marlin path
(~220 GB/s measured) projects ≈0.9–1.0 ms/frame recovered.

## Candidate definition (fixed)

A derived Nano artifact built from the **fhw8 release artifact** (the
promoted production nano, HF revision `fdf70edf…`, composite `dd4ce6a9…`)
with exactly these deltas:

1. The 16 attention projection tensors (`q_proj/k_proj/v_proj/o_proj`
   weights across the 4 attention layers) replaced by GPTQ-format W8
   tensors — symmetric, group 128, desc_act=false, gptqmodel-v2 zero-point
   (qzeros all-zero, `w = s·(q−128)`), deterministic RTN against stored
   bf16 scales — the identical scheme/packing as the shipped lm_head and
   fhw8 function_head.
2. Config routing: the attention-projection dynamic exclusion
   (`-:.*(?:q_proj|k_proj|v_proj|o_proj)$`) removed from BOTH
   `quantize_config.json` and the embedded `quantization_config`, and the
   four projection module names added to `modules_in_block_to_quantize` in
   both locations (the loader's `is_layer_gptq_quantized` gate, per the
   fhw8 loader findings).

Everything else byte-identical to the fhw8 artifact; new artifact
directory; released artifacts never modified. Construction must be
deterministic (double-build byte-identical proof before any manifest
claims reproducibility).

## Gates (fail-closed, order fixed)

**G0 — construction integrity.** (a) RTN round-trip exact per tensor,
recorded max/mean error. (b) Load/routing proof: a fresh engine constructs;
the kernel profile (or an equivalent runtime check) shows the bf16 GEMV
family instance count drop 8 → 0 per step and the Marlin family gain
accordingly. There is NO leaf-argmax analog gate: attention projections are
interior, so numeric effects are adjudicated at G1's decision level — this
is a preregistered difference from fhw8, justified because interior bf16
drift is already characterized on this runtime (wedge findings) and the
validity question is decision equivalence, not tensor equality.

**G1 — decision-level replay (primary).** Paired replay of the frozen
1,172-call capture (manifest `55d9e815…`): baseline lane = fhw8 artifact,
candidate lane = attnW8 artifact, matched basins per the fhw8 matched-basin
rule (exact-basin preferred; the characterized call-39-divergent signature
handled identically in both lanes if it arises). Gate: zero per-call
text-token, function-token, or conjunction flips across ≥3 valid blocks per
lane. Any decision flip = FAIL (no tolerance; a failed G1 retires the
candidate or motivates a per-tensor ablation, each requiring a new seal).

**G2 — latency confirmation.** Step-4d harness node-level profile on the
candidate: mean content-interface wall must improve ≥0.5 ms/frame vs the
fhw8 baseline (traced 50.549 ms) with no other kernel family regressing
>0.3 ms. The projected ≈0.9 ms is the estimate, not the gate.

**G3 — behavioral batteries (same session).** The rapid-turn settlement
suite (V1 semantics: zero empty responses, 5+5 sessions) and the tool-call
interruption matrix (12 cells, zero fatals, bounded calls) on the live
candidate stack; canonical browser plan with no new failure class (I1
adjudicated per its ledger entry).

**Promotion** additionally requires a counterbalanced A/B per the fhw8
precedent and an explicit user decision; G0–G3 alone qualify the candidate
as `measured`.

## Discipline

Replay blocks are the uncertainty unit; failed engines retained with basin
evidence; deviations recorded, never absorbed; evidence under
`reports/attnw8-qualification/` with SHA-256 manifests; this file sealed by
SHA-256 before G0 execution.
