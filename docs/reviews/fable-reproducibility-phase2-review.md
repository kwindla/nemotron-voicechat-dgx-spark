# Fable adversarial review — Reproducibility Phase 2 (deterministic conversion tooling)

Date: 2026-08-04 · Scope: uncommitted Phase 2 boundary on
`codex/pocket-tts-typed-input` — `tools/conversion/` (18 tools),
`tests/conversion/` (12 files), `config/conversion-recipe.json`,
`config/calibration-corpus-production-candidate-1.json`,
`config/nano-gptq-corpus-selection.json`, `docs/weight-reproduction.md`, BOM
update — against plan Phase 2 and the Phase 1 BOM. Accepted evidence: ruff
clean; 53 conversion tests pass; full suite 204 pass / 1 skip; live
`--preflight-only` run against the real 44,382,749,892-byte checkpoint
(SHA `d553750c…` verified, 1,632 tensors, corpus release SHA `ad8ce2fe…`,
~1.54 TB projected remaining). No changes implemented.

## Verdict: APPROVE WITH CHANGES (no blockers; three IMPORTANT).
## Exit gate: code-complete but EXECUTION-PENDING — see final section.

## Challenged items that held

- **W4/AWQ/RTN removal:** zero matches for `awq|rtn|w4|int4` across
  `tools/conversion/`. The `ea_w8a32` ABI exception is documented in the
  recipe (`eartts.abi_note`). Phase 2.2 met.
- **W8 selection and padding:** recipe pins the exact contract — 105 tensors
  (MLP up/down, Mamba in/out, text head), group 128, seed 0, damp 0.01,
  block 128, the K-tail padding selector
  (`…mixer.down_proj.weight`, source_k 15680), 4 GiB shards, and both
  expected composite hashes matching the BOM. Phase 2.4 met.
- **Corpus identity / path independence / disjointness:**
  `build_release_manifest` inventories by content hash with portable records
  and canonical JSON digests (no absolute paths in identity), asserts
  calibration/evaluation disjointness, and the orchestrator refuses to run
  if the supplied corpus manifest differs from the checked-in PC1 manifest —
  so a PC1-parity run is impossible on a substituted corpus. Path
  independence and corruption rejection are tested
  (`test_release_manifest_is_path_independent_and_disjoint`,
  `…rejects_corrupt_payload`).
- **Licensing / PC1 carryover:** the corpus is declared
  `redistributable: true` with explicit reasoning — project-authored text
  fixtures (Apache-2.0) plus replay tensors that are *outputs* of the
  OpenMDW-1.1 parent model, which the license explicitly leaves unrestricted;
  `contains_raw_user_recordings: false`. The BOM's Phase 1
  `redistributable: "unknown"` is resolved to `"yes"` consistently. Because
  the exact PC1 corpus ships and is identity-enforced, the plan's Phase 2.3
  decision fork resolves to the carryover branch: the release weights remain
  PC1. The reasoning is sound under the license text I verified in the plan
  review.
- **Immutability / 100 GiB / resume integrity:** `extract_public_vllm`
  enforces `MINIMUM_POST_CONVERSION_FREE_GIB = 100` with a projected-peak
  disk budget (live preflight showed the computation working);
  `--repro-output` must differ from `--output` and must not pre-exist; stage
  markers are self-authenticating (canonical-hash over their own payload),
  resume re-hashes every recorded output and hard-fails on drift, and
  unvalidated partial outputs abort with a move-aside instruction rather than
  overwrite. All of these behaviors are unit-tested, including marker
  tampering. Phases 2.5/2.6 mechanics met.
- **End-to-end executability:** the orchestrator's wiring is coherent
  (sibling imports resolve via script-dir sys.path; subprocess stages use
  `sys.executable`; stage outputs are declared and checked), and the live
  preflight proves the input-validation path executes against real inputs.
  The final `_verify_expected_artifact` hard-requires byte-equality with the
  BOM's PC1 file inventory and hashes — the run cannot "succeed" without
  literally recreating PC1.
- **Stale research paths:** no `/home/` or `/tmp/` literals in tools or
  configs; the only container-absolute path is the Nano skeleton runtime
  mount, which correctly preserves the load-bearing "Nemotron" name
  substring.

## IMPORTANT

**P2-I1 — Six promoted tools are unwired and unmentioned.**
`nano_component_gate.py`, `eartts_component_gate.py`, `nano_replay.py`,
`nano_attribution.py`, `create_nano_pad_pair_variant.py`, and
`create_nano_skip_custom_text_logits.py` are invoked by neither
`reproduce_release_artifacts.py` nor `docs/weight-reproduction.md` (grep:
zero references). Plan Phase 2.1 explicitly includes "component replay and
first-divergence reports" in the promoted tooling, and the plan's Outcome
says `--convert-from-source` "runs the component gates." Either wire the
gates/replay into the reproduction flow (or the future `voicechat test`),
and document each variant script's role — or delete what is genuinely
residual research. Undocumented executables in a provenance tree invite
exactly the staleness this phase exists to eliminate.

**P2-I2 — The Nano dual run is in-process; EarTTS's is not.** EarTTS runs two
separate subprocess invocations (strong independence), but
`convert_nano_marlin_gptq.py` produces run 1 and run 2 inside one process
(`for run_index, destination in ((1, output), (2, repro_output))`). A single
process shares allocator state, cuBLAS/cuDNN workspace heuristics, and
library initialization, so in-process byte-identity is weaker evidence of
determinism than clean-process identity. Make run 2 a fresh subprocess (the
orchestrator already has the machinery), or document explicitly why
in-process repetition is considered sufficient for the Nano path.

**P2-I3 — full-precision lists disagree between recipe and BOM.** The recipe
says `["function head", "embeddings", "attention projections"]`; the BOM
says `["function_head", "embeddings"]`. Attention projections are indeed
unquantized (105 selected tensors exclude them), so the recipe is the more
complete statement — align the BOM (and normalize the naming style) so the
two machine-readable sources cannot be cited against each other.

## SUGGESTIONS (non-gating)

- Conversion reports embed `elapsed_seconds` wall-clock — fine, since only
  artifact trees are byte-compared, but say so in
  `docs/weight-reproduction.md` ("reports are diagnostics; determinism is
  claimed only for artifact trees and manifests").
- `_verify_expected_artifact` inventories only top-level files
  (`root.iterdir()`); an unexpected subdirectory in a release root would
  escape detection. Use a recursive listing.
- The BOM has no pointer to `config/conversion-recipe.json` (path + hash);
  add one so the recipe is discoverable from the single machine-readable
  identity document.
- The manifest placeholders `pending-public-runtime` / `pending-phase-4` are
  honest; ensure Phase 4 fails closed if they survive into a published
  release manifest.

## Exit-gate assessment (execution vs. code, as requested)

Phase 2's exit gate — "one checked-in command recreates both derived
checkpoints … and two runs match each other and the release manifest" — is
**structurally guaranteed but not yet demonstrated**. The code cannot
falsely report success: corpus identity is pre-checked, dual-run
byte-identity is asserted twice, and final hashes must equal PC1's. The live
preflight validated the real inputs, tensor count, corpus SHA, and disk
budget. What has NOT happened is the multi-hour GPU conversion itself.
Declare Phase 2 "code-complete, reproduction-pending," and let the first
real `--convert-from-source` run — which Phase 7.6 requires anyway for the
downloaded-vs-converted agreement gate — retire the remaining risk. The
three IMPORTANT items should land before that run so the executed evidence
covers the final shape of the tooling.

---

# Closure re-review — 2026-08-04, Phase 2 amendments

## Verdict: APPROVE (code boundary closed; exit gate remains
reproduction-pending as previously ruled).

All three IMPORTANT items verified closed by inspection:

- **P2-I1 — CLOSED.** `--asr-model` is now required; the orchestrator has a
  marker-protected `component-gates` stage running `nano_component_gate`
  over all held-out evaluation calls (`--max-calls` driven by the corpus
  manifest, 1,172) plus `eartts_component_gate` in both eager and
  `--no-enforce-eager` (graph) modes; gate outputs are recorded in
  `reproduction.json`. The two obsolete config-variant executables and their
  stale tests are deleted (test count 53→46, explained), and
  `docs/weight-reproduction.md` names the direct diagnostic tools.
- **P2-I2 — CLOSED.** Nano runs 1 and 2 are now separate subprocess
  invocations with distinct scratch dirs and reports, byte-compared
  afterward. The converter's `--single-run` guard is XOR-correct: it rejects
  both flags together and neither flag (`args.single_run ==
  (args.repro_output is not None)`).
- **P2-I3 — CLOSED.** BOM and recipe both read
  `["function_head", "embeddings", "attention_projections"]`, naming
  normalized.
- **Suggestions — landed:** `_verify_expected_artifact` now inventories
  recursively (`rglob`); the BOM carries a `conversion_recipe` pointer with
  path + SHA-256; diagnostic-report determinism is clarified in docs; the
  Phase 4 fail-closed obligation for `pending-*` placeholders is recorded.

One documented-choice observation (non-gating): the Nano component gate
targets `nano-run1` (converter config) while the EarTTS gate correctly
targets the final window-1500 release artifact. This is acceptable because
`finalize_nano_release` is weight-preserving (hardlinks) and the final
config is hash-pinned byte-identical to the live-qualified PC1 config
(enforced by the final inventory verification, plus the
`test_qualified_nano_config_chain` hash chain) — so the gate exercises the
identical weights and the shipped config is separately guaranteed. Add one
sentence to `docs/weight-reproduction.md` stating this is deliberate (or
point the gate at `release/nano` if the harness tolerates the qualified
config), so the asymmetry is not mistaken for an oversight.

Phase 2 status: code boundary APPROVED with no open findings. The exit gate
remains "reproduction-pending" until the first real `--convert-from-source`
run demonstrates PC1 hash equality end-to-end, per the original ruling.
