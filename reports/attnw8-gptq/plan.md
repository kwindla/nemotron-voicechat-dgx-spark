# Plan: Hessian-calibrated GPTQ for attention projections (attnW8-gptq)

Status: **REVISED AFTER R2 REVIEW; NOT SEALED; DO NOT EXECUTE ANY CAMPAIGN
RUN.** This revision incorporates all required dispositions from
`docs/reviews/attnw8-gptq-plan-review-r1.md` and all deltas from
`docs/reviews/attnw8-gptq-plan-review-r2.md`. Seal-0 must be recorded before
any baseline run or capture, and Seal-1 must be recorded before any Hessian or
candidate work. Neither review authorized a run. Branch:
`agent/reduce-eou-settlement-latency`.

## Motivation and prior result

The RTN attention-W8 candidate was retired at G3 per
`reports/attnw8-qualification/`: G0 was exact, G1 had zero flips over 10,548
frozen-corpus decisions, and G2 measured -1.546 ms/frame, but the candidate
made 0/12 tool calls on the interruption matrix where fhw8 made 12/12 on the
same audio, deterministically. Tool-call affordance is therefore
attention-precision-sensitive and was invisible to the old frozen corpus.

GPTQ proper (per-column quantization with Hessian-weighted error compensation
over real activations) minimizes layer-output error rather than weight error
and is how the shipped quantized layers were built
(`config/nano-gptq-corpus-selection.json` and
`config/conversion-recipe.json`). The objective is to recover at least 0.5
ms/frame while preserving tool behavior, or to localize the sensitivity
without weakening any semantic gate.

## Two-stage prerun seals and partition firewall

This campaign uses two immutable, hash-chained seals. There is no unsealed
capture stage and no candidate work between them.

### Seal-0: inputs and rules, before any run

Before A0 or any other baseline run or capture, create Seal-0 and SHA-256 hash
the following:

- this plan and the complete candidate input manifests for all three
  partitions;
- the complete payload and inventory for every proposed tool scenario,
  including carrier/audio, expected transcript and outcome, user prompt,
  registered-tool schema, system prompt, injected result, scenario ID, audio
  hash, and harness revision;
- the fhw8 artifact and both config locations, model/runtime image, package
  freeze, hardware/driver, capture code, harness, quantizer, sequential driver,
  shard rewriter, and every build parameter; and
- the A2 scenario counts, boundary-window and initiation-quota rules,
  hierarchical weighting, effective-position convention, algorithm and
  numerics, acceptance thresholds, scenario acceptance/replacement rules,
  qualification rules, and contingency definitions and order.

Seal-0 fixes candidate inputs and derivation rules; it does not claim
baseline-emitted tokens, causal rows, accepted windows, or completed
effective-position indexes that can only be learned by running the fhw8
baseline. Its input partitions are:

1. **A1 calibration:** use the original three-conversation, 3,934-call
   calibration selection in `config/nano-gptq-corpus-selection.json`, from the
   corpus release `ad8ce2fe3a66fae72ce45703216265180066b2745dc32d829ee550b45c487bd5`
   recorded in `config/conversion-recipe.json`. A newly sealed
   calibration-only replacement is permitted only as a reviewed deviation;
   it may not draw from either qualification partition.
2. **A2 tool calibration:** whole tool scenarios selected under the contract
   below. No scenario may be split between calibration and qualification.
3. **Q qualification:** the untouched, physically separate 1,172-call
   evaluation conversation (retained manifest `55d9e815...`) plus the frozen
   tool-decision scenarios and live C1 scenarios defined below. The 1,172
   calls are used only by C0/C2. Q never contributes to a Hessian, scale,
   candidate construction, calibration/scenario choice, threshold tuning, or
   post-result gate revision. Its preregistered use for qualification
   accept/reject decisions and for the sealed contingency screening defined
   below is explicitly permitted.

The partition audit must show pairwise disjointness by resolved source root,
conversation/request ID, audio SHA-256, and scenario ID. For tool material it
must additionally inventory the carrier/audio, expected transcript, user
prompt, registered-tool schema, system prompt, injected result, and harness
revision. Every qualification carrier and all of those scenario components
must be absent from A1/A2. Near-duplicate or derived scenarios remain in one
partition. Any collision is fail-closed. If an unavoidable collision places a
C1 interruption carrier in A2, C1 is demoted to regression-only evidence and
an independently worded/audio, frozen, disjoint live tool-scenario matrix must
replace it as the qualification surface under the same 12/12 rule. The
scenario-level-disjoint offline C2 tool set is mandatory in all cases.

The audit of the proposed inputs is part of Seal-0. A collision or missing
payload prevents A0; it is not deferred to Seal-1.

### Baseline-only A0/index stage

Seal-0 authorizes only A0 and the baseline passes needed to populate and
validate the effective-position indexes for A1, A2, and Q. This stage may be
used only to prove exact eager fhw8 outcomes, hook transparency and q/k/v
input equality, causal alignment, and required-window availability. It may
not construct or inspect a Hessian, quantized weight, candidate artifact, or
candidate output, and it may not tune any threshold or rule.

If a proposed scenario has the wrong baseline categorical outcome or lacks
its required boundary window, replace the whole scenario only under the
Seal-0 replacement rule. Record the reason and a revised Seal-0 containing the
replacement's complete payload and updated partition audit. The replacement
invalidates all affected baseline evidence, and the baseline-only A0/index
stage restarts under that Seal-0 revision. No evidence from the invalidated
stage may enter Seal-1.

### Seal-1: accepted partitions and indexes, before Hessian work

After A0 and all effective-position checks pass, create Seal-1. It identifies
its exact Seal-0 parent and SHA-256 hashes the final accepted partition
manifests and scenario payloads; the complete ordered effective-position
indexes, emitted-head records, selected A2 windows, and A0 evidence; the final
C0 measurement manifest specified below; and the exact frozen C2
boundary/continuation row list. Phase A3 and Phase B may begin only after
Seal-1. In particular, no Hessian construction, quantized weight, candidate
construction or output, threshold change, or outcome-driven tuning is allowed
before it.

Seal-0 and Seal-1 are both immutable once any Hessian or candidate work
starts. No post-candidate mutation of either seal is permitted. A required
change abandons the candidate and its evidence and starts a new campaign from
a recorded Seal-0 revision followed by a new baseline-only stage and Seal-1.

## Phase A - corpus measure and sequential activation capture

### A0: baseline and hook preflight

Under Seal-0, reconstruct the exact fhw8 baseline numerically in the eager
engine; do not fall back to the original dense release. Verify the Seal-0
hashes of the fhw8 artifact, config, runtime image/packages, hardware identity,
capture code, and harness before loading it. On the Seal-0 positive tool
preflight, the eager fhw8 model must emit the expected tool name and exact
arguments. Repeating the same preflight with hooks enabled must leave every
text/function decision unchanged.

At each attention layer (14, 21, 30, and 39), hook the actual module arguments
to all of `q_proj`, `k_proj`, and `v_proj` and prove equality element-for-element
on the preflight corpus. Only after that proof may their input be represented
by one shared stream. Hook the actual argument passed to `o_proj` after head
reshape and attention mixing; a layer input or inferred intermediate is not a
substitute.

### A1: effective-position index

Preserve the qualified replay measure from
`tools/conversion/nano_gptq_calibration.py`: a prefill contributes its supplied
prefix exactly once, while each subsequent streaming Nano call contributes
only its newly decoded row. Never dump all causal prefixes on every call.

Define one canonical record for each unique effective position containing:

`(source root/hash, conversation/request, turn, Nano call, causal row, emitted
text token, emitted function token, phase, scenario ID, audio SHA-256)`.

The causal convention is fixed: activation row `r` is the row whose output
logits select the emitted token at sequence coordinate `r+1`. For a first
function token, `r*` is the activation row predicting that token; `r*-1` is
the immediately preceding decision row. This convention applies to both the
shared q/k/v input and the actual o input generated while computing row `r`.
Record both heads' emitted values (including no-function) at each indexed row.

Canonicalize and SHA-256 hash every provenance record and the ordered full
index. Reject duplicated keys, duplicated source-prefix rows, missing causal
alignment, or disagreement between indexed and captured row counts. Report
counts by source, conversation, scenario, phase, layer, and input site. Final
candidate-dependent activation streams are deterministic build intermediates;
Seal-1 fixes the corpus index, while Seal-0 fixes the weighting rule,
algorithm, and runtime identity.

### A2: stratified tool-boundary calibration

A2 contains exactly 24 whole, calibration-only scenarios accepted in Seal-1
before a candidate is built:

- 12 independently worded and independently recorded/audio-hashed positive
  scenarios in which eager fhw8 emits the expected first function token; and
- 12 independently worded/audio hard no-call controls, including plausible
  tool-adjacent requests for which eager fhw8 emits text and no function call.

For each scenario select a fixed nine-row boundary window. A positive window
is `r*-4 ... r*+4` around the first-function-token decision row and must include
both `r*-1` and `r*`; these 24 rows across the 12 positives are the initiation
quota. A no-call window is nine consecutive rows centered on a scenario row
declared in Seal-0 before capture. Thus A2 has 216 unique boundary-window
positions. If a required window is unavailable or the baseline categorical
outcome is wrong, apply the recorded Seal-0 revision and baseline-stage
restart protocol above before Seal-1; do not shrink or move the window. No
scenario replacement or window mutation is permitted after Seal-1. Post-call
rows may be present inside a positive window but cannot satisfy the initiation
quota. No other scenario may contribute, so one scenario is bounded to 1/24
of the final A2 weight.

Use fixed hierarchical weights rather than raw `sum(xxT)`. For each source
conversation/scenario with `n` selected effective positions, compute
`H_source = (2/n) * sum(xxT)` in sealed row order. Define A1 as the
equal-weight mean of its three whole conversations. Define A2 as 50% the
equal-weight mean of the 12 positive scenarios and 50% the equal-weight mean
of the 12 hard no-call scenarios. The Hessian supplied to GPTQ is fixed at:

`H = 0.75 * H_A1 + 0.25 * H_A2`.

The weights, quotas, window derivation, and ordering rules are fixed in
Seal-0; the accepted scenarios and exact ordered windows are fixed in Seal-1.
For every layer/input site report the row count, raw and weighted Hessian trace
for every stratum/scenario, A2's final trace share, and tool-row leverage
`x^T(H + lambda I)^-1 x`. A2's coefficient may not be tuned after seeing
results.

### A3: sequential propagation contract

After Seal-1, freeze the corpus, reconstruct fhw8 eagerly, and process model
layers in natural order. Propagate non-target layers unchanged. At each target
attention layer:

1. capture and re-prove the shared actual q/k/v input;
2. build its weighted Hessian and quantize q, k, and v in fixed q/k/v order;
3. install each BF16 dequantized GPTQ propagation weight in the eager model;
4. with quantized q/k/v active, recapture the real `o_proj` input, build its
   weighted Hessian, quantize o, and install its BF16 dequantized weight; then
5. run the complete quantized layer and propagate its output to the next
   layer.

This is the in-tree converter's sequential stage/teacher-propagation contract,
not a fixed-teacher eight-stream variant. The corpus and algorithm are sealed;
the q/k/v-dependent o inputs and later-layer inputs are reproducible build
intermediates and are hashed in each clean build.

## Phase B - quantizer and candidate construction

### B-path-1 (required preferred path)

Reuse the qualified in-tree GPTQ core in
`tools/conversion/nano_gptq_calibration.py` (effective-row loading, float32
Hessian construction, `gptq_quantize_matrix`, and `pack_gptq_codes`) and the
sequential propagation pattern in
`tools/conversion/convert_nano_marlin_gptq.py`. Do not extend the full
105-tensor dense-source converter blindly. Combine those math/packing helpers
with the fhw8-derived shard-rewrite pattern in
`tools/benchmark/build_attnw8_candidate.py`, so exactly the 16 dense attention
weights are replaced and all existing fhw8 packed payloads are carried
byte-identically.

B-path-2 is allowed only if B-path-1 is technically unusable and the reason is
recorded before implementation. It must independently implement the identical
contract below and pass all self-checks; it is not permission to choose a
different GPTQ variant.

### Executable numerical contract

The following is byte-deterministic and applies to both paths:

- Work weights, activations, Hessians, damping, inverses, losses, and error
  feedback are float32. Accumulate the fixed weighted Hessian in the sealed
  index order. Process input-feature columns in natural order with
  `desc_act=false`, group size 128, and block size 128. Group scales are static
  and computed once from the original float32 work weight, not recomputed
  after feedback.
- For each output row/group use
  `scale = max(abs(max)/127, abs(min)/128)`; replace an all-zero group's scale
  with float32 1.0. Codes are `torch.round(value/scale)` under the pinned
  runtime's round-to-nearest-ties-to-even behavior, then clamped to signed
  `[-128, 127]` and cast to int32. Do not BF16-round the scale before deriving
  codes.
- Before damping, identify columns whose Hessian diagonal is exactly zero;
  their rows/columns must also be exactly zero or construction stops. Isolate
  each by setting only its diagonal to 1.0, preserve and quantize its weight,
  and allow no dead-column error propagation. Then compute
  `lambda = 0.01 * mean(diag(H))` on that repaired Hessian and add lambda to
  every diagonal element.
- Compute lower `L = cholesky(H_damped)`, then
  `H_inverse = cholesky_inverse(L)`, then the upper Cholesky factor
  `U = cholesky(H_inverse, upper=True)`. Within each 128-column block, use
  `U[j,j]` as the pivot, apply the in-block row slice update, collect block
  errors, and apply the block-to-remainder matrix update exactly as
  `gptq_quantize_matrix` does. Natural column and q/k/v/o order are fixed.
- Pack signed `[N,K]` codes after transposition by adding 128 and packing four
  consecutive input-feature lanes per int32. Serialize scales once to
  contiguous BF16 in `[K/128,N]` orientation; `qzeros` is all-zero int32;
  `g_idx` is natural group membership; shapes and orientation must match the
  gptqmodel-v2/fhw8 loader ABI. Form the eager propagation weight as signed
  int32 codes converted to float32 times the repeated original float32 group
  scales, then cast that product once to BF16, exactly as the in-tree core
  specifies.
- Set seed 0, `torch.use_deterministic_algorithms(True)`, float32 matmul
  precision `highest`, and CUDA TF32 off. No stochastic operation is allowed.
  Pin and verify the conversion recipe's qualified identity: linux/arm64,
  NVIDIA GB10 compute capability 12.1, driver 580.142, kernel
  6.17.0-1014-nvidia, torch `2.10.0a0+b4e4ee81d3.nv25.12`, transformers
  4.56.0, safetensors 0.8.0, vLLM `0.17.1.dev0+gb31e9326a.fi065`,
  compressed-tensors 0.13.0, and nvidia-modelopt 0.37.0. Seal-0 contains the
  exact image digest, GPU identity, package freeze, source hashes, and build
  parameters. A mismatch stops construction and requires a reviewed new
  two-stage seal cycle.

The self-check contract is:

1. identity and arbitrary positive diagonal Hessians produce exactly the
   no-feedback codes, float32 scales, dequantized propagation weights, and
   metrics of this same GPTQ base quantizer; and
2. an independently implemented packed-runtime dequantizer reconstructs the
   intended signed codes and stored BF16 scales exactly, including signed-code
   `+128`, zero `qzeros`, natural `g_idx`, shapes, and orientation.

The retired RTN artifact is only a numeric comparator; it is not a byte oracle.
Its absmax/127 and pre-code BF16-scale rules intentionally differ. Two full
clean-process builds must match every artifact byte. Hash the quantizer source,
sequential driver, shard rewriter, runtime image/packages, corpus and position
indexes, parameters, intermediate-stream index, and output tree.

Build `nano-attnw8g` from the fhw8 artifact in a new directory. Apply only the
retired RTN campaign's sealed routing deltas in both `quantize_config.json` and
the embedded `quantization_config`. Never modify a released artifact.

## Phase C - fixed fail-closed qualification ladder

Construction integrity C0 precedes all live behavior. C1 remains the first
live behavioral gate. A gate failure stops the main candidate ladder and
retains all evidence; it cannot be explained away.

### C0 - hard construction and numeric qualification

Seal-1 includes the final C0 measurement manifest; it is not completed at
implementation time. The manifest freezes all of the following:

- The reference is the hash-pinned exact eager fhw8 artifact with its original
  dense BF16 q/k/v/o weights at layers 14, 21, 30, and 39, not the original
  dense release. C0 uses the exact ordered Seal-1 rows for the untouched base
  Q stratum and disjoint frozen tool-decision stratum.
- For each isolated linear comparison, q/k/v use the actual module argument
  captured from that fhw8 reference replay. The o comparison uses the actual
  `o_proj` argument produced by the reference's dense q/k/v attention path.
  The upstream hidden states, inbound KV cache, attention mask, position IDs,
  cache positions, and all other call state are the exact fhw8 reference state
  named by the row record. Reference, packed GPTQ, and packed RTN receive the
  same frozen linear input; only the one measured weight differs.
- For each full-block comparison, reference, GPTQ, and RTN begin with the same
  actual fhw8 layer-entry hidden state, inbound KV cache, attention mask,
  position IDs, cache positions, and call state named by the row record. The
  reference runs that layer's unmodified dense fhw8 attention block. GPTQ and
  RTN replace all four projections in that layer, derive their own current
  q/k/v values and attention-mixed `o_proj` argument, and produce their own
  block output. Inbound caches are copied from the reference state for each
  variant and row so that one variant cannot mutate another's input.
- Metrics are computed independently for each tensor-or-block/stratum cell;
  strata and cells are never pooled. In ordered-index row order and then
  natural output-feature order, let `E_m = sum((y_m - y_fhw8)^2)`,
  `D = sum(y_fhw8^2)`, and `N` be the compared scalar count. Subtraction and
  squaring use float32 outputs; the terms are accumulated sequentially in
  IEEE-754 float64 in that fixed order. Unnormalized MSE is `E_m/N`, and
  normalized MSE is `E_m/D`. If `D` is exactly zero, normalized MSE is exactly
  zero when `E_m` is zero and positive infinity otherwise. Because GPTQ and
  RTN share `D`, their gate ratio is computed as `E_GPTQ/E_RTN` when
  `E_RTN > 0`, including when both normalized MSE values are infinite; it is
  never computed as an infinity/infinity division. A worst-row tie is resolved
  to the earliest Seal-1 row.
- The C2 portion lists, in evaluation order, every exact boundary and
  continuation row: canonical row key, scenario ID and positive/no-call type,
  boundary/continuation role, causal coordinate, both fhw8 emitted heads,
  expected tool identity and arguments, extra-call count, post-result
  continuation, and loop bound. The ordered list and each record are hashed in
  Seal-1; C2 may evaluate no inferred, substituted, or later-selected row.

C0 passes only if every item below passes:

1. Exact independent packed round-trip for all 16 tensors; exact agreement
   with stored BF16 scales and metadata; all source, work, scale, dequantized,
   and block-output values finite.
2. Two clean-process builds are byte-identical over their complete artifact
   trees and reproducible intermediates.
3. A delta inventory proves exactly 16 dense q/k/v/o weights were replaced,
   both config locations changed exactly as sealed, and every other fhw8
   file, tensor, packed payload, and config field was carried unchanged.
4. A fresh engine loads the candidate and proves routing. Under the retained
   RTN profiling fixture, unquantized BF16-GEMV instances must change 8 to 0
   and Marlin instances 156 to 168 per measured step, with the selected replay
   tokens correct. Any changed runtime requires an equally exact check sealed
   in an R2-reviewed amendment before execution.
5. On the untouched 1,172-call base evaluation and the disjoint frozen
   tool-decision stratum, packed-runtime GPTQ normalized linear-output MSE must
   be no greater than packed-runtime RTN MSE for every one of the 16 tensors:
   `GPTQ/RTN <= 1.0` with no waiver.
6. At each of layers 14, 21, 30, and 39, run the real all-four-projection
   attention block with q/k/v perturbations passing through attention and the
   resulting input reaching o. On both the base and tool-decision strata its
   normalized block-output MSE must be no greater than RTN's:
   `GPTQ/RTN <= 1.0` for all eight layer/stratum cells, with no waiver.

Report absolute normalized MSE, unnormalized MSE, GPTQ/RTN ratio, row count,
and worst row for every linear tensor and attention-block cell. If RTN's error
numerator `E_RTN` and therefore its unnormalized MSE are exactly zero, GPTQ's
`E_GPTQ` and MSE must also be exactly zero and the ratio field is the literal
sentinel `ZERO_RTN_BOTH_ZERO`. A positive GPTQ value is an unconditional
failure and its ratio field is the literal sentinel
`ZERO_RTN_GPTQ_NONZERO`. The limits are fixed here before measurement; prose
cannot waive a regression.

### C1 - first live gate

Run the hash-pinned fhw8 positive control adjacent to the candidate matrix,
using the correct audio SHA-256 (the retained precedent carrier is
`1c7856088512ab3b865665682af1eaa43947ef3b1b403f6f8a13654be0a545ee`), exact
transcript, registered-tool schema, system prompt, injected result,
environment, scenario inventory, and harness revision.

Before each matrix, prove process lifetime, readiness, client attachment, and
quiescence; harness liveness is not model behavior. Retain wrong-fixture,
startup, bridge-lifecycle, stalled, and invalid attempts but exclude them from
the verdict with reasons.

The fhw8 control and candidate must each pass 12/12 cells with zero fatals,
the exact tool name and arguments, and the sealed call/loop bound. There are no
replicates, adjudication, or stochastic miss allowance. Candidate evidence is
interpretable only when its adjacent fhw8 control is 12/12; otherwise the
whole session is retained-invalid and rerun cleanly. A valid candidate result
below 12/12 fails C1. The partition-firewall replacement rule applies if this
carrier ever overlaps A2.

### C2 - hard offline decision qualification

C2 has two mandatory surfaces:

1. Run `fhw8_decision_replay` on the untouched 1,172-call evaluation corpus,
   candidate versus hash-pinned fhw8, with matched basins and at least three
   valid replay blocks per lane. Gate zero text-token, function-token, or
   conjunction flips. The replay block, not an individual position, is the
   uncertainty unit. Retain invalid/divergent-basin attempts and apply the
   characterized basin rule symmetrically.
2. Run a sealed, whole-scenario, A1/A2/C1-disjoint frozen tool-decision replay
   containing at least 12 independently worded/audio positive-call scenarios
   and 12 hard no-call controls. At every row in the exact ordered Seal-1 C2
   boundary/continuation list, compare fhw8 and candidate for call/no-call,
   first function token, tool identity and exact arguments, extra-call count,
   post-result continuation, and loop bound. The gate is zero categorical
   flips, zero argument changes, zero extra calls, and zero loop-bound changes
   in every scenario. Record the fhw8 and candidate function-vs-text decision
   margin at each frozen boundary row and report absolute/relative margin
   erosion; categorical parity is the gate and a barely preserved sign is
   explicitly reported as risk.

The deterministic replay of the two retained live sessions remains a
supplemental trajectory regression only and cannot substitute for the frozen
tool-decision set.

### C3 - latency

Run the retained node-level profile against the fhw8 traced baseline of
50.549 ms. Gate at least 0.5 ms/frame improvement and no non-target kernel
family regression above 0.3 ms. Report the same kernel-family and bandwidth
attribution used in the RTN campaign; the expected approximately 1.55
ms/frame is an estimate, not the gate.

### C4 - behavioral battery

Run the rapid-turn settlement suite (5+5 sessions, zero empty responses), the
canonical browser plan (no new failure class; I1 per its ledger entry), and the
full non-browser suite, anchors, lint, compile, and bridge-accounted checks from
the retained campaign. Failed and stalled attempts remain evidence.

Passing C0-C4 makes the candidate **measured and eligible**, not promoted.
Promotion still requires the fhw8 precedent's separately sealed
counterbalanced A/B protocol and an explicit user decision.

## Contingency - preregistered localization and rescue

Contingency triggers on any tool-sensitive failure in C0's tool stratum, C1,
or C2. Seal-0 fixes all cell definitions, build-identity rules, screening
rules, order, and stopping rules below; Seal-1 fixes the exact indexed
screening inputs. Every cell is built deterministically and screened first
with C0 numerics and the frozen offline tool replay; live matrices are reserved
for cells that pass both offline surfaces. Cells are localization evidence and
are never automatically promotable.

The fixed coarse-to-fine order is:

1. **Layer localization:** in layer order 14, 21, 30, 39, build four
   layer-only-W8 cells; then in the same order build four reciprocal
   leave-one-layer-full-precision cells. Run all eight; no result-dependent
   early stop.
2. **Projection-family localization:** in q, k, v, o order, build four
   family-only-W8 cells across all four layers; then in the same order build
   four reciprocal leave-one-family-full-precision cells. Run all eight; no
   result-dependent early stop.
3. **Within-layer localization:** a layer is implicated only if its layer-only
   cell reproduces a triggering offline failure and its reciprocal
   leave-that-layer-full-precision cell clears that same failure. For every
   implicated layer, in layer order, build q-only, k-only, v-only, and o-only
   cells followed by the four reciprocal leave-q/k/v/o-full-precision cells.
   Run all eight for every implicated layer. If no layer meets the paired rule,
   stop individual testing and report localization inconclusive; do not search
   arbitrary subsets. Projection-family results are corroborating evidence and
   do not alter this stopping rule.

A mandatory algorithmic control builds base-only GPTQ (`H = H_A1`) and
base-plus-tool GPTQ (the fixed 0.75/0.25 design) with otherwise identical
inputs and code, then compares both on C0 and the frozen tool-decision replay.
This measures whether A2 helped tool preservation rather than attributing a
lucky outcome to it. The base-only cell is not promotion-eligible without a
fresh two-stage seal cycle and full ladder.

Any mixed-precision subset rescued by localization receives a fresh candidate
definition and complete Seal-0/baseline-only/Seal-1 cycle, then must pass the
complete C0-C4 ladder and the existing C3 >=0.5 ms/frame benefit. Group size 64
is a separate rescue variant, not a localization cell: it requires its own
two-stage seal cycle, deterministic double build, metadata and kernel-routing
proof, complete C0-C4 ladder, and latency gate. No contingency outcome may
weaken a gate or authorize optional subset search.

## Evidence and operational discipline

Seal-0 and its exact input/rule hashes must exist before A0 or any baseline
capture. Seal-1 and its final accepted partition, index, A0-evidence, C0, and
C2 hashes must exist before A3 Hessian work or Phase B. Deviations are recorded
and reviewed, never absorbed; failed, invalid, and stalled attempts are
retained. Evidence lives under `reports/attnw8-gptq/` with SHA-256 manifests.
No released artifact is modified.

Before stopping any service, retain its full identity and stable fields,
including revision, model/Nano/EarTTS hashes, image ID, mounts, environment,
GPU state, health/readiness, and active-client/quiescence state. Use retained
stop/start procedures. After every capture or qualification window, restore
the release service and prove every stable field matches the pre-stop record;
a claimed restore without that verification is invalid.
