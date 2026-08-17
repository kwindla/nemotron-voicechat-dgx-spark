# Adversarial review: attnW8-gptq campaign plan, R1

Date: 2026-08-16 UTC  
Reviewed artifact: `reports/attnw8-gptq/plan.md`  
Review mode: static, CPU-only; no campaign or model runs performed

## Verdict

**FAIL — do not seal or execute this plan yet.**

The direction is right: use calibrated GPTQ, put the tool gate first, retain
the RTN latency and construction controls, and fail closed if genuine tool
contexts cannot be captured. But the current draft would calibrate on the
held-out 1,172-call evaluation corpus, gives rare tool rows no guaranteed
influence on the Hessian, describes a fixed-teacher capture while claiming the
original converter's sequential GPTQ contract, and contains an identity-Hessian
self-check that is not byte-compatible with the repository's RTN builder. Its
qualification side also reuses likely training scenarios, leaves C0 partly
subjective, and does not add the frozen tool-decision surface that the RTN miss
proved is necessary.

These are design defects, not documentation polish. The required deltas below
must be incorporated before the preregistration is sealed.

## Findings

### 1. Phase A has the right linear input sites, but the wrong split and an underspecified sampling measure

The two conceptual sites per attention layer are correct for linear GPTQ:

- `q_proj`, `k_proj`, and `v_proj` consume the same normalized hidden-state
  input, so one shared input stream is sufficient **after a hook-level proof
  that all three actual module arguments are equal**.
- `o_proj` must use the actual argument passed to `o_proj`, after head reshape
  and attention mixing. It must not use the layer input or a guessed attention
  intermediate.

That does not make the proposed eight-stream capture sound as written.

First, A1 selects the 1,172-call replay (`plan.md:27-28`) that the original
conversion receipt calls the physically disjoint **evaluation** corpus:
`config/conversion-recipe.json:23-27` records 3,934 calibration calls and 1,172
evaluation calls, and `config/nano-gptq-corpus-selection.json:25-31` identifies
that evaluation conversation. Feeding those 1,172 calls into the Hessian makes
C2's frozen-corpus comparison in-sample. An every-eighth-position split does
not repair this: adjacent autoregressive positions from the same turn share
nearly all context and are highly correlated.

Second, the capture must preserve the original replay's effective-row rule.
The qualified loader takes a prefill once and only the newly decoded row from
subsequent calls (`nano_gptq_calibration.py:424-444`). Dumping every causal
prefix on every streaming call would count old positions repeatedly and turn
the Hessian into an accidental length/quadratic-prefix weighting.

Third, `>=200` tool-turn positions and `>=24` positions “at or after TOOLCALL”
do not establish useful tool coverage. The initiating decision is made by the
hidden row predicting the first function token; most post-TOOLCALL rows govern
arguments, acknowledgement, or post-tool continuation, not call initiation.
The 12 interruption cells also are four post-call interrupt phases times three
replicates of one carrier, so they do not provide 12 independent initiation
contexts. Finally, raw `sum(xxT)` lets a few hundred A2 rows disappear under a
much larger A1 population unless source/scenario weights are fixed.

Required Phase-A changes:

1. Use the original 3,934-call calibration partition (or a newly sealed
   calibration-only partition) for A1. Preserve the 1,172-call evaluation
   replay untouched for C0/C2. Enforce disjointness by source root,
   conversation/request ID, audio hash, and scenario ID—not by position
   modulo.
2. Split tool material by whole scenario before capture. The carrier/audio,
   prompt, tool schema, system prompt, injected result, and harness revision
   used for qualification must be absent from GPTQ calibration. If the
   interruption carrier is used in A2, C1 is only a regression check and a
   separate frozen tool-scenario set becomes mandatory.
3. Define a unique effective position and record `(source, conversation,
   turn, Nano call, causal row, emitted text token, emitted function token,
   phase)`. Hash this index. Prove there are no duplicated prefix rows and
   specify the causal alignment of the activation row to the emitted function
   token, including the off-by-one convention.
4. Replace the two count-only A2 thresholds with a stratified boundary-window
   contract: multiple independently worded/audio positive scenarios, hard
   no-call controls, bounded contribution from any one scenario, and rows
   immediately before and at first-function-token emission. Post-call rows may
   be included but cannot satisfy the initiation quota. Seal a source mixture
   or explicit per-scenario weighting (for example, normalized A1/A2 Hessians
   with a fixed non-negligible A2 coefficient); report each stratum's Hessian
   trace and row count. The exact quotas and weight must be chosen before any
   candidate output is observed.
5. Capture from an exact fhw8-baseline numerical model, not the original dense
   release by accident. Record artifact/config/runtime/code hashes. Before
   accepting captures, prove the eager baseline emits the expected tool calls
   and that enabling hooks does not change decisions. Hook all q/k/v inputs in
   a preflight and seal their equality; thereafter one shared stream is fine.

The held-out set should consist of whole base conversations and whole tool
scenarios. “Every eighth position” is not a usable generalization split.

### 2. B-path-1 should reuse the in-tree GPTQ core, not blindly extend the full converter

The original implementation is already present:
`tools/conversion/convert_nano_marlin_gptq.py` and
`tools/conversion/nano_gptq_calibration.py`. It is not necessary to search an
unspecified repo/container. Reusing this qualified math and packing is the
sound preference.

Blindly extending its target selector is not sound. The converter selects and
rebuilds the 105 shipped tensors from a dense source, while this campaign's
candidate must be derived from fhw8 and leave all existing packed tensors
byte-identical. The attention builder should therefore reuse the GPTQ Hessian,
quantizer, and packing helpers while using the RTN campaign's shard-rewrite
pattern. It must prove that exactly the 16 dense attention weights are replaced
and all carried fhw8 payloads are unchanged.

There is also a propagation mismatch. The original converter quantizes a stage,
writes its dequantized teacher-propagation weight back into the eager model,
then captures the next stage and propagates the resulting layer output
(`convert_nano_marlin_gptq.py:405-432`). Pre-capturing all eight streams from
the untouched fhw8 teacher means `o_proj` is calibrated on an input produced by
full-precision q/k/v, and later attention layers are calibrated before earlier
attention layers are quantized. That is a valid fixed-teacher variant only if
explicitly preregistered; it is not the original pipeline's contract.

The preferred path should instead freeze the input corpus, load an eager
numerical reconstruction of fhw8, and process layers 0..N in order. At each
attention layer, capture/prove the shared q/k/v input, quantize q/k/v, recapture
the real `o_proj` input with quantized q/k/v, quantize o, then propagate the
quantized layer output. The candidate-dependent streams are reproducible build
intermediates; the corpus and algorithm—not all final streams—are what must be
sealed before construction.

### 3. The fallback GPTQ description is incomplete, and its RTN self-check is contradictory

The high-level math is otherwise appropriate: input-feature columns in natural
order, `desc_act=false`, 1% mean-diagonal damping, group-128 W8, and
gptqmodel-v2 packing. However, the executable contract must pin the existing
implementation details: float32 Hessian and work matrix, block size 128,
static group scales, exact dead-column policy, damping placement, Cholesky
sequence, rounding/clamping rule, BF16 scale serialization, TF32 disabled,
deterministic algorithms enabled, and the pinned software/hardware identity.
“Standard dampening” and “Cholesky inverse rows” are not sufficient to
reimplement this byte-deterministically.

The proposed “identity Hessian reproduces RTN exactly” check is false across
the two existing implementations. The qualified GPTQ core computes separate
positive/negative extrema with divisors 127/128 and derives codes before BF16
scale serialization (`nano_gptq_calibration.py:466-476, 520-548`). The retired
RTN builder uses absolute max/127 and rounds the scale to BF16 before deriving
codes (`build_fhw8_candidate.py:61-77`). An identity Hessian removes error
feedback, but it cannot make those different base quantizers byte-identical.

Replace that check with both of the following:

- identity/diagonal-H produces exactly the no-feedback result of the **same
  GPTQ base quantizer**; and
- an independent packed-runtime dequantizer reproduces the intended codes and
  stored BF16 scales exactly, including signed-code `+128`, all-zero `qzeros`,
  natural `g_idx`, shapes, and orientation.

Keep the retired RTN artifact as the numeric comparator, not as an impossible
byte oracle. Two clean-process full builds must match every artifact byte, and
the report must hash the quantizer source, runtime image/packages, corpus
index, build parameters, and output tree.

### 4. C1-first is right, but C0 and C2 need hard qualification surfaces

Putting the tool matrix first among **live behavioral** gates is exactly the
RTN lesson. Construction integrity must still precede it.

C0 currently says output MSE is diagnostic, then permits a worse tensor if it
is “explained” (`plan.md:65-69`). That is a subjective post-result escape
hatch. C0 must be a hard gate containing at least:

- exact packed round-trip and finite-value checks;
- two clean byte-identical builds;
- a delta inventory proving exactly 16 dense weights were replaced, both
  config locations changed as sealed, and every other fhw8 file/tensor was
  carried unchanged;
- fresh-engine load/routing proof, retaining the RTN campaign's observed
  bf16-GEMV 8 -> 0 and Marlin 156 -> 168 per-step transition (or an equally
  exact preregistered runtime check); and
- objective held-out numeric limits chosen before the run. At minimum,
  packed-runtime GPTQ must not exceed RTN's held-out linear-output MSE on any
  tensor, and the all-four attention-block output must not exceed RTN at any
  layer on either the base or tool-decision stratum. Report both absolute
  normalized error and GPTQ/RTN ratios; do not make a per-tensor regression
  waivable by prose.

The end-to-end attention-block check is essential: fixed-input q/k/v and o
linear MSE does not measure q/k perturbation through the attention softmax or
q/k/v perturbation of the input subsequently seen by o.

C1 should be an unconditional 12/12 candidate pass with zero fatals, exact
tool name/arguments, and bounded calls. The current text simultaneously says
11/12 is FAIL and allows a single-cell adjudication. Remove that contradiction
and the stochastic loophole: the retained behavior is deterministic. Run a
hash-pinned fhw8 positive control adjacent to the candidate matrix, using the
correct carrier, exact transcript, registered-tool schema, environment, and
harness revision. A failed candidate matrix is interpretable only if that
control is 12/12; otherwise the session is retained-invalid, not a candidate
failure or success.

C2 should retain the 1,172-call matched-basin replay, but only after removing
it from A1. The two retained live sessions are not an adequate new decision
surface and “loop metrics must match” is not a sufficiently defined oracle.
Add a sealed, scenario-level-disjoint tool-decision replay containing positive
calls and hard no-call controls. Compare fhw8 and candidate at every relevant
row for call/no-call, first function token, tool identity/arguments, extra-call
count, post-result continuation, and loop bound. Zero categorical flips should
be the gate. Retained sessions may remain a supplemental trajectory replay.

C3 and C4 are otherwise correctly carried forward. Restore the omitted
promotion contract: passing C0-C4 makes the candidate measured/eligible, not
promoted; promotion still requires the counterbalanced A/B protocol and an
explicit user decision. Also retain pre-stop service identity and prove final
stable-field restoration, rather than only saying the stack “must be restored.”

### 5. The contingency plan is too open-ended and triggers too late

Contingency must trigger on a tool-sensitive failure in C0, C1, **or C2**, not
only C1. “Per-layer and per-projection ablation grid” is not a sealed grid; it
permits outcome-driven subset search, and sealing each cell only after seeing
the preceding result invites optional stopping.

Before the main run, seal an ordered localization family and its stopping
rules. A useful coarse-to-fine design is:

1. four layer-only and four leave-one-layer-full-precision candidates;
2. four projection-family-only and four leave-one-projection-family-full-
   precision candidates; then
3. individual q/k/v/o tests only inside an implicated layer.

Run the frozen offline tool replay and C0 numerics as the screening surface;
reserve live matrices for preregistered survivors. Treat these cells as
localization evidence, not automatically promotable candidates. Any rescued
mixed-precision subset must receive a fresh seal and the full C0-C4 ladder.

Group size 64 is a separate rescue variant, not a clean localization ablation:
it changes scales/metadata, potentially kernel behavior, and latency. It needs
its own deterministic build, routing, C0-C4, and latency gate. Also preregister
a base-only-GPTQ versus base-plus-tool-GPTQ comparison; otherwise the campaign
cannot tell whether A2 improved tool preservation or the candidate merely got
lucky.

## RTN lessons the draft still misses

- **The old frozen corpus was demonstrably blind.** It remains useful for
  broad regression, but cannot double as calibration data or the only offline
  semantic gate.
- **Correct fixture identity is part of the result.** The RTN campaign retained
  an invalid wrong-carrier attempt. Pin audio SHA-256, transcript expectation,
  tool schema, injected result, system prompt, harness code, and scenario
  inventory before capture and before C1.
- **Harness liveness is not model behavior.** The invalid browser bridge
  lifecycle attempt requires explicit process-lifetime/ready/quiescence checks
  and baseline controls.
- **Construction routing mattered.** The new plan dropped the exact load/kernel
  routing proof that RTN G0 established; it belongs in C0.
- **“Everything else unchanged” must be measured.** Build from fhw8 and compare
  every carried tensor/file, not just config intent.
- **Basins and failed attempts remain evidence.** Keep the matched-basin rule,
  retain invalid/stalled attempts, and use the replay block—not individual
  positions—as the uncertainty unit.
- **Qualification is not promotion.** Restore counterbalanced A/B and explicit
  user authorization.

## Creative additions worth sealing

1. At frozen tool-decision rows, record the fhw8 and candidate function-vs-text
   decision margin, not only the argmax. Gate the categorical decision and
   report margin erosion. A candidate that barely preserves the sign is a
   predictable live-regression risk.
2. Report tool-row leverage under each layer Hessian (for example,
   `x^T(H+lambda I)^-1x`) and the A2 share of Hessian trace. This is a direct
   audit that tool contexts occupy a represented subspace rather than merely
   satisfying a row counter.
3. Add an all-four-projections-at-one-layer held-out block-output test alongside
   per-linear MSE. This can identify q/k softmax amplification that linear MSE
   hides.
4. If full attnW8 still fails, preregister a latency-aware mixed-precision
   rescue: retain the implicated layer or projection family in FP32/BF16 and
   quantize the rest, but require the existing >=0.5 ms/frame C3 benefit. This
   turns localization into a bounded engineering option without weakening
   semantic gates.

## Required disposition

Revise the plan with all required changes above, hash the exact calibration and
qualification scenario partitions, name the in-tree GPTQ implementation and
its sequential propagation contract, replace the impossible RTN identity
self-check, make C0 objective, add the disjoint frozen tool-decision replay,
remove C1's adjudication loophole, and seal the complete contingency order.
Then submit the revised plan for R2 review. **R1 does not authorize Phase A.**
