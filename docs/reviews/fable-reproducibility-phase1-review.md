# Fable adversarial review — Reproducibility Phase 1 (freeze & inventory)

Date: 2026-08-04 · Scope: Phase 1 boundary only —
`config/qualified-candidate-1.json`, `docs/public-build-graph.md`,
`tools/provenance/verify-qualified-candidate.py`,
`tests/runtime/test_qualified_candidate_manifest.py` — against Phase 1 of
`docs/reproducible-clone-to-playground-plan.md` and the qualified project
evidence. Accepted evidence: 6 focused tests pass; manifest validates; full
artifact verification ran inside the qualified container against the Nano,
EarTTS, Nano-skeleton, and application roots. No changes implemented.

## Verdict: APPROVE WITH CHANGES — one blocker, four IMPORTANT

## Spot-checks that PASSED (verified against independent records)

- Checkpoint revision `fb0f94ea…` and combined-model SHA `d553750c…` match
  `docs/provenance.md` and the frozen-candidate memory record; the OpenMDW-1.1
  license is now hash-pinned.
- Speech commit `911ec674…`, Voicechat-fork commit `237ba3ca…`, qualified
  image ID `fa11b6…`, driver 580.142, kernel 6.17.0-1014 — all match.
- Pocket TTS snapshot revisions `d29db797…`/`e041936c…` match the deployed
  `check-environment.py` pins (upgraded from byte-size to SHA-256 — good).
- `native_vllm` version string `0.17.1.dev0+gb31e9326a.fi065` is internally
  consistent with its pinned commit `b31e9326…`.
- Nano 105 quantized tensors / dual byte-identical runs matches provenance.
- The opaque-image quarantine is honest and enforced in three places: the
  manifest (`use: comparison-only; prohibited as a public-build input`), the
  validator (hard error if the string is absent), and the build-graph refusal
  rules. The EA/NGC lineage of the 35.1 GB imported layer is admitted plainly
  rather than laundered. Phase 1.4 is genuinely met.
- The build-graph doc's "Remaining parity work" section honestly states the
  public fork and the oracle do not share ancestry and that Phase 4 output is
  a NEW candidate requiring full requalification — matching hash is evidence,
  not a gate substitute. This is exactly the right epistemic posture.

## BLOCKER

**B1 — The qualified vLLM deltas exist only inside the opaque image, so "no
unresolved local-image dependency" is not yet true in substance.** The BOM
pins final hashes for 11 vLLM files and the build graph says their "release
versions must be produced from the public forks plus the named project
patches" — but no checked-in patch or diff produces those deltas today, and
the 11-file list itself is hand-asserted rather than derived. Two
consequences: (a) if the oracle image were lost, the qualified behavioral
deltas would be unreconstructible — a hash you cannot reach is a dependency,
not a comparison; (b) per plan Phase 1.3, a necessary file with "no public
source or checked-in patch" must fail classification — these 11 currently
fail it. Exact fix, purely extractive (no design work, oracle still exists):

1. Inside the qualified container, run a full-tree diff of the installed
   `vllm/` package against a checkout of `vklimkov-nvidia/vllm@237ba3ca…`
   (and, where files come from the native baseline, against
   `vllm-project/vllm@b31e9326…`). Enumerate EVERY differing/added file —
   this proves the 11-file comparison surface complete or corrects it.
2. Check the resulting per-file reference diffs into
   `tools/provenance/qualified-deltas/` (text patches, hash-recorded in the
   BOM). These are Phase 4's porting worklist and the reconstruction escrow.
3. Record the diff-enumeration command and its file count in
   `docs/public-build-graph.md` so the completeness claim is reproducible.

With B1 done, the Phase 1 exit gate is honestly met; without it, the graph
documents intent but the deltas remain image-resident.

## IMPORTANT

**I1 — Checkpoint file coverage and verifiability.** The BOM hashes only
`model.safetensors` and `LICENSE` from the checkpoint repository, but the
runtime also consumes `config.json` and the `rnnt_tokenizer/` files (the RNNT
turn-taking sidecar is load-bearing). Add their hashes to
`public_sources.checkpoint.files`, and add a `--checkpoint-root` option to the
verifier — as shipped, the 44 GB checkpoint is the one artifact the tool
cannot verify at all, so the reported "full artifact verification" run
structurally excluded it.

**I2 — Build parameters live only in prose.** CUDA 13.0.1, FlashInfer 0.6.5,
and the `sm_121` target appear only in the build-graph markdown table. Phase 4
will consume the machine-readable BOM; move these into a
`public_sources.native_vllm.build` object (cuda, flashinfer, arch) so the
reproducible build cannot silently drift from the documented one.

**I3 — "Runtime settings" are absent from the BOM.** Phase 1.1 explicitly
lists runtime settings in the bill of materials, but the frozen contract
(EOU/BOU frames, watchdogs, session limits, VLLM/streaming max lengths, the
typed-input profile) appears nowhere in `qualified-candidate-1.json`. Either
add a `runtime_contract` section now or add one line to the manifest and the
build-graph doc stating the contract is deliberately deferred to Phase 5's
`config/production-candidate-1.toml` — silence invites the assumption it was
forgotten.

**I4 — Calibration corpus identity is opaque.** The artifacts' inputs were the
checkpoint AND the calibration corpus, but the corpus is represented only
indirectly through `source_manifest_sha256` (a hash of an external,
unpublished manifest). The Phase 2.3 decision fork — the plan's B1 — turns
entirely on corpus identity. Add an explicit
`qualified_artifacts.calibration` entry (corpus identity/hash, fixture count,
redistributable: unknown|yes|no) so the fork is decidable from the BOM alone.

## SUGGESTIONS (non-gating)

- `composite_sha256` is unverifiable as shipped: the composition function is
  undocumented and the verifier never computes it. Document the exact
  construction (e.g., SHA-256 over sorted `name:hash` lines) or drop the
  field.
- `verify_tree` accepts unlisted extra files in artifact roots; add a
  `--strict` mode that fails on unexpected files, matching Phase 6.3's
  "reject unexpected files."
- Record and check per-file byte sizes alongside hashes — a fast preflight
  that catches truncation before a 14 GB hash pass.
- The host-local path guard only catches `/home/` and `/tmp/`; add `/mnt/`,
  `/data/`, `/var/`, and `/srv/`.
- The prohibited-input scan covers only `public_sources`; scan the whole
  document.
- `captured_at_utc` is a midnight placeholder; use the real capture time.
- Test gaps worth one more pass: malformed-SHA rejection, a
  `reproducibility != {runs:2, byte_identical:true}` mutation, and a
  comparison-only-string mutation are untested validator branches.

## Exit-gate assessment

Phase 1.1 (BOM): substantially met, with the I1–I4 completeness gaps.
Phase 1.2 (layer classification): met at the honest granularity — the opaque
layer is quarantined whole rather than per-file laundered, which is the
defensible reading. Phase 1.3 (import allowlist / fail-on-unsourced): NOT yet
met — B1 is precisely this requirement. Phase 1.4 (hashes as comparison
evidence only): fully met and triple-enforced. Overall: the boundary is well
built and honestly framed, but the exit gate cannot be declared until B1's
delta escrow and completeness enumeration land; I1's checkpoint verifiability
should land with it since it touches the same manifest schema.

---

# Re-review — 2026-08-04, Phase 1 amendments

## Verdict: APPROVE. The Phase 1 exit gate is honestly met.

All findings verified closed by direct inspection, not by report:

- **B1 — CLOSED, and the fix proved its own necessity.** The capture tool
  deterministically enumerated all 966 installed Python files against both
  pinned public trees (922 exact: 838 matching the Voicechat fork, 84 matching
  both; 36 delta; 8 added). I confirmed all 44 non-exact files have escrowed
  patches on disk whose SHA-256 each matches its index entry (44/44), the
  index digest `8854e6eb…` matches the build-graph record, and the qualified
  tree digest is recorded. Decisively: the enumeration *corrected* the
  hand-asserted 11-file surface — `cfg_ops.py` and `cfg_metadata.py` turned
  out byte-identical to the public fork, and 33 differing files the original
  list missed were surfaced (including `vllm_flash_attn/*`,
  `transformers_utils/*`, scheduler/KV-cache internals). The deltas are now
  reconstructible without the oracle image, and the build graph records the
  capture command and counts as the Phase 4 classification worklist.
- **I1 — CLOSED.** `public_sources.checkpoint.files` now hashes `config.json`,
  `model.safetensors`, `LICENSE`, and all three `rnnt_tokenizer/` files;
  `--checkpoint-root` exists, and the reported strict in-container
  verification included the 44 GB checkpoint.
- **I2 — CLOSED.** `native_vllm.build` records CUDA 13.0.1, FlashInfer 0.6.5,
  Python 3.12, `torch_cuda_arch_list: 12.1`, and build parallelism
  machine-readably.
- **I3 — CLOSED.** A 38-key `runtime_contract` is in the BOM; spot-checked
  against the frozen deployed contract: `rnnt_eou_frames: 20`,
  `rnnt_bou_frames: 3`, `rnnt_function_call_interrupt_ms: 240`, speech gate
  −40 dBFS / 3 frames, `transport_vad_min_rnnt_tokens: 2`, max lengths
  12288/12000 — all match.
- **I4 — CLOSED.** `qualified_artifacts.calibration` now carries source and
  selected manifest hashes (file + canonical) for both calibration and
  evaluation sets with explicit counts (3 conversations / 3,934 calls
  calibration; 1 / 1,172 evaluation) and `redistributable: "unknown"` — the
  Phase 2.3 decision fork is now decidable from the BOM and honestly marked
  undecided.
- **Suggestions — landed:** `--strict-artifacts` mode, per-file byte sizes on
  the checkpoint inventory, expanded path guard, whole-document prohibited
  scan, real capture timestamp (`2026-08-04T18:04:05Z`), composite-algorithm
  documentation, and mutation tests. I re-ran the manifest suite locally:
  10/10 pass in this file (the reported 12 presumably includes the two
  updated runtime-optimization tests; not material).

Non-gating observations for Phase 4's classification pass: several escrowed
deltas look incidental rather than intentional (`_version.py`, wheel-adjacent
`vllm_flash_attn` shims, `lfm2_moe`/`quark`/mx-format utilities) — the
worklist should classify each as project-patch, upstream-build artifact, or
incidental before porting; and the calibration corpus is strikingly small
(3 conversations), which raises the value of resolving `redistributable`
early since re-calibration on a substitute corpus would trigger the full
requalification fork.

Phase 1 is complete against its exit gate: the public build graph is
documented, machine-readable, verifier-enforced, and — with the delta
escrow — no longer depends on the opaque image for anything but its role as
a comparison oracle.
