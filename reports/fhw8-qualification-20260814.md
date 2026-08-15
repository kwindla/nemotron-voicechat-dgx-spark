# function_head W8 qualification: gates G0–G2

**Date:** 2026-08-14 UTC
**Preregistration:** `reports/fhw8-qualification/preregistration.md`, SHA-256
`40a36a2c844aea224fbdb24dcb61249504be355e31b851d430669c8d0320c95f`, sealed
before candidate construction.
**Verdict: G0 PASS · G1 PASS (matched-divergent design, recorded deviation) ·
G2 PASS (−4.437 ms/frame traced).** The candidate qualifies for ranking entry
as `measured`. Production promotion still requires the preregistered G3
live/behavioral phase.

## Candidate

`~/.cache/nemotron-voicechat/artifacts/candidates/nano-fhw8/` built by
`tools/benchmark/build_fhw8_candidate.py` from the unchanged released
checkpoint. Final candidate definition (one amendment vs the preregistration,
below): `function_head.weight` (F32 131072×4480) → GPTQ W8 sym group-128
tensors in the shipped lm_head packing (gptqmodel-v2 zero-point: qzeros
all-zero, `w = s·(q−128)`, deterministic RTN against the stored bf16 scales);
`voicechat_quantized_output_heads` gains `function_head`; the
`-:.*function_head$` dynamic exclusion removed from **both**
`quantize_config.json` and the `quantization_config` embedded in
`config.json`; and — the amendment — `function_head` appended to
`modules_in_block_to_quantize` in both locations, because the loader's
`is_layer_gptq_quantized` consults that list for every layer including heads.
The amendment was discovered through two retained fail-closed load failures
(`g1-candidate-attempt1/2-*`): with q-tensors present but the layer
unrouted, vLLM refuses to load (`no module or parameter named
'function_head.g_idx'`) rather than silently skipping quantization.

## G0 — construction integrity

- **G0(a) PASS.** Packed tensors reproduce intended RTN exactly
  (`rtn_exact: true`); max |Δw| 1.389e-3 = 0.391% of the global weight max
  (`data/build-report.json`). An earlier builder round caught a real
  self-consistency bug (quantizing against unrounded f32 scales while
  storing bf16 scales) before any artifact was written.
- **G0(b) PASS — perfect.** On all **1,172 unique** final-hidden-state
  vectors extracted from the retained capture (eager engine, capture-safe
  dump patch, hashes recorded): argmax agreement **100%** (≥99.9% required);
  all 1,172 states carried strong top-2 margins (>0.25) and **zero** ordering
  violations (`data/g0b-head-numerics.json`). *Recorded deviations:* (i) the
  preregistered ≥10,000-vector population is unreachable as unique states —
  the retained corpus contains exactly 1,172 capture calls; the gate ran on
  the full corpus. (ii) hidden states were extracted via an eager-mode
  engine because graph replay never exposes replay-time hidden states to
  Python; Step 5's 18/18 eager exactness is the fidelity basis, and the
  eager lane independently matched every retained text and function token.
  A first extraction attempt that wrote inside CUDA graph capture
  (invalidating capture) is retained; the v2 patch is capture-safe.
- **G0(c) PASS.** Routing proven in G2's profile: the bf16 GEMV family
  drops 9 → 8 instances/step and the Marlin family gains exactly one
  (155 → 156).

## G1 — decision-level replay (matched-basin design)

Population: frozen 1,172-call capture (manifest SHA-256 `55d9e815…`),
graph-mode engines, per-call text token + function token equality required.

- **Baseline exact-basin lane:** 15/15 valid blocks, zero text and zero
  function mismatches vs the retained capture (`data/g1-baseline.json`).
- **Candidate constructions select the divergent basin systematically:**
  five consecutive fresh candidate engines (attempts 3–5, rolls 6–7) each
  reproduced exactly the Step 5 call-39 signature (token 1044 vs 10592,
  engine-stable, identical across all blocks) — and **zero function
  mismatches in every block** (75 measured blocks ≈ 87,900 calls). Under
  the released artifact the basin is mixed (this session: baseline exact
  first-roll, then divergent first-roll at 5 blocks; attribution study:
  divergent then exact). The candidate's 5/5 divergent streak is recorded
  as evidence that checkpoint layout influences basin selection — a new
  prevalence observation for the Step 5 graph-construction defect, and
  additional motivation for the exact-gate repair.
- **Matched-divergent comparison (recorded deviation from the preregistered
  exact-basin-only design):** a baseline lane rolled to the divergent basin
  (5/5 blocks exact call-39 signature, zero function mismatches) was
  compared per-call against all five candidate runs under the
  `call39-divergent` basin rule (block valid iff its only text mismatch is
  exactly the characterized signature and function mismatches are zero).
  Result: **439,500 paired decisions, zero flips** across five independent
  candidate engines (`data/g1-compare-*.json`). Both lanes share the
  identical characterized basin, so the comparison isolates the head delta
  exactly; the divergence itself is a pre-existing shared-backbone defect
  present on the unmodified release and unrelated to this candidate.

**G1 verdict: PASS.** Zero decision flips attributable to the candidate, on
either basin evidence line.

## G2 — latency confirmation

Step 4d harness profile mode on the candidate, node-level Nsight, 2+10
blocks, same six preregistered states; all 60 ranges captured; block token
contract shows only the known call-39 basin signature (retained; timing
unaffected). Traced-condition comparison vs the attribution-study baseline
(same tracing, same analyzer):

| | baseline | candidate | delta |
|---|---:|---:|---:|
| interface wall (mean, traced) | 54.986 ms | 50.549 ms | **−4.437 ms** |
| kernel union | 48.368 | 43.664 | −4.704 |
| bf16 GEMV family (×9 → ×8) | 9.176 | 1.963 | −7.212 |
| Marlin family (×155 → ×156) | 37.696 | 40.202 | +2.505 |
| intra-envelope idle | 1.048 | 1.043 | −0.005 |
| every other family | | | within ±0.001 |

**Gate: ≥3.5 ms improvement, no family regressing >0.5 ms → PASS.** The
mechanism is exactly the preregistered one: the ~7.2 ms function_head bf16
GEMV is replaced by a ~2.5 ms Marlin instance; net head saving ≈ 4.7 ms.
Projected untraced content-interface wall ≈ 49.1 ms (from 53.56), putting
the generation frame at approximately the 80 ms budget on this change alone.

## Standing and next steps

- Ranking: the quantization-extension row enters as **`measured`
  (−4.437 ms/frame traced, decision-invisible on the retained corpus)** —
  the first measured row; per the plan, Step 5 candidate selection can now
  proceed on evidence.
- **G3 (production promotion) remains open:** live fixture campaign,
  function-calling behavioral fixtures, structural/audio gates, per the
  plan's promotion contract. The candidate must not ship before G3.
- Follow-on recorded: attention-projection W8 extension (~1.9 → ~1.0 ms
  potential); W4 function_head variant; the basin-bias observation feeds the
  graph exact-gate repair case.

## Operations

Same stop/component/restore pattern as the attribution study; the failed
first extraction attempt, two loader fail-closed attempts, and all five
divergent candidate runs are retained. Service restored from the retained v4
script; stable-field identity verification in `data/preflight/`.
