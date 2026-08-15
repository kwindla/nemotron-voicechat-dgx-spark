# Preregistration: function_head W8 quantization-extension qualification

Status: sealed before candidate construction and before any gate run.
Funded by user greenlight 2026-08-14 following
`reports/step5-bandwidth-attribution-20260814.md` (function_head bf16 GEMV
= 7.292 ms/frame median; W8 estimate ≈ 4.6 ms/frame recovered).

## Candidate definition (fixed)

A derived Nano checkpoint variant, built from the unchanged released
artifact `/models/derived/nano` (model SHA-256 recorded at build), with
exactly three deltas:

1. `function_head.weight` (F32, 131072×4480) replaced by GPTQ-format W8
   tensors (`qweight`/`qzeros`/`scales`/`g_idx`), symmetric, group 128,
   desc_act=false — the same scheme and packing conventions as the shipped
   `lm_head`. Primary construction is deterministic round-to-nearest in
   GPTQ packing (no calibration-set choice); if gate G1 fails, one fallback
   construction with Hessian-based GPTQ over retained-capture hidden states
   is permitted and must be reported as the fallback.
2. `config.json`: `voicechat_quantized_output_heads` = `["lm_head",
   "function_head"]`.
3. `quantize_config.json`: the `-:.*function_head$` dynamic exclusion
   removed.

Everything else byte-identical to the released artifact. The candidate is
a NEW artifact directory; the released artifact is never modified. The
attention-projection extension is explicitly out of scope (recorded
follow-on).

## Gates (all fail-closed; order fixed)

**G0 — construction integrity (CPU then GPU).**
(a) Packing round-trip: dequantized candidate tensors reproduce the
intended RTN values exactly; recorded max |w_deq − w_f32| and relative
error consistent with W8 group-128.
(b) Numeric validation on real hidden states: on ≥ 10,000 final-hidden-state
vectors drawn from retained capture calls, candidate-dequant logits vs F32
reference logits must agree on argmax ≥ 99.9% and on the top-2 margin sign
wherever |margin| > 0.25 (the Step 5 defect margin scale); any packing-
convention mismatch (detected as gross error) is a build bug, not a gate
trade-off.
(c) Load check: a fresh engine constructs from the candidate; the
function_head routes through the GPTQ/Marlin path (verified in G2's profile
by bf16 GEMV instances dropping 9 → 8 per step and a new Marlin instance).

**G1 — decision-level replay gate (GPU, primary).** Paired replay of the
frozen capture (root manifest SHA-256
`55d9e815ec235a588b0aa6eacb4a97a2d5a77f19aa00954d0c2e205d4108d1aa`, 687
calls): baseline lane on the released artifact, candidate lane on the
candidate artifact, matched fresh engines, ≥ 3 valid blocks per lane.
Baseline-lane token contract vs the retained capture must hold (exact-basin
check; a call-39 divergent-basin engine is retained-invalid and
reconstructed, per Step 5/attribution precedent — never counted as a
candidate failure or success). Gate: across all valid paired blocks, per
call, the emitted text token, function token, and their conjunction
decision must be identical between lanes, zero flips allowed. Logit-level
byte equality is expected to fail and is not the gate (deliberate numeric
change); the decision layer is.

**G2 — latency confirmation (GPU).** Step 4d harness profile mode on the
candidate (2 warmup + 10 measured blocks, same fixed states, node-level
tracing): mean content-interface wall must improve by **≥ 3.5 ms/frame**
vs the Step 4d baseline 53.56 ms with no other kernel family regressing by
> 0.5 ms/frame. Below 3.5 ms the candidate is not promoted on latency
grounds regardless of G1.

**G3 — behavioral/live qualification (promotion phase, separately
scheduled).** The plan's promotion contract: Step 2-style live fixture
campaign plus function-calling fixtures (tool-call correctness), structural
and audio gates. G3 is preregistered as required for production promotion;
its detailed prereg is written when scheduled. G0–G2 alone qualify the
candidate only for ranking entry as `measured`.

## Populations and discipline

The uncertainty unit in G1/G2 is the replay block. Failed engines are
retained with their basin evidence. No criterion may be weakened after
runs begin; deviations are recorded. Negative results are retained. The
operational pattern for GPU phases is the retained stop/component/restore
sequence with stable-field identity verification
(`identity-verification.json` rule from the attribution study).

## Deliverables

Builder: `tools/benchmark/build_fhw8_candidate.py` (+ tests). Candidate at
`~/.cache/nemotron-voicechat/artifacts/candidates/nano-fhw8/` with SHA-256
manifest. Report `reports/fhw8-qualification-20260814.md`; evidence under
`reports/fhw8-qualification/data/`. This file is sealed by SHA-256 recorded
before G0 execution.
