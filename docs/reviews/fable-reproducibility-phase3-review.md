# Fable adversarial review — Reproducibility Phase 3 (HF release)

## Pre-publication check — 2026-08-04

Scope: uncommitted `release/MODEL_CARD.md`, `licenses/OpenMDW-1.1.txt`,
`tools/release/prepare_hf_release.py`, `upload_hf.py`, `upload-hf`,
`tests/release/`, provenance/notices updates, and the staged tree at
`/home/khkramer/models/NVIDIA-NemotronLabs-VoiceChat-11B-Spark-hf-release-v1`
(8,152 files, 19,690,226,144 bytes on disk). Target repo does not exist yet.
Nothing uploaded or modified.

## Verdict: SAFE TO UPLOAD — as a PRIVATE staging release first, per the
plan's implementation order. Two notes below should be addressed before the
public promotion, not before staging.

### Independently verified (not taken from reports)

- **Exact PC1 weights.** I re-hashed every Nano file (4 shards + index +
  quantize_config) and both EarTTS files in the staged tree: all match the
  Phase 1 BOM's qualified hashes exactly. The staged weights ARE Production
  Candidate 1.
- **Release manifest integrity.** I recomputed the canonical-JSON release
  SHA from the staged `manifests/release.json` independently:
  `2c829a3f7cf350dd136dc17168ef9d49cf41318370552f4ae5707980d3b6fc7e` —
  matches both the embedded value and the dry-run report. Manifest inventories
  8,151 files + itself = 8,152, consistent with the dry run.
- **No secrets, private paths, or EA lineage.** Targeted scans of every
  text/JSON file: zero hits for `/home/`, the username, ngrok, tokens
  (`hf_` matches only the field name `hf_release`), `:ea-` tags, or the
  retired checkpoint. One benign finding — see note 2.
- **License/origin compliance (OpenMDW-1.1 obligations).** LICENSE (NVIDIA
  copyright + OpenMDW-1.1) is at the release root; THIRD_PARTY_NOTICES.md
  present; README front-matter sets
  `base_model: nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`, names parent
  revision `fb0f94ea…`, and states "community conversion … not an NVIDIA
  release." Both retained-notice obligations are met; the non-affiliation
  line addresses the branding concern from the plan review.
- **Corpus identity + redistribution.** The staged calibration tree carries
  the manifest/selection whose identity the Phase 2 tooling enforces; the
  redistribution rationale (replay tensors are OpenMDW model outputs;
  fixtures Apache-2.0; no raw recordings) is carried into the release.
- **Uploader safety.** `REPO_ID` is a hardcoded constant equal to the
  intended target (allowlist by construction); staging is re-verified via
  `verify_staged_release` before any network call; readback pins the
  returned immutable commit (`info.sha`) and byte-compares the downloaded
  `release.json` against the staged one; per-file verification uses
  server-side LFS sha256 metadata; auth comes from the process
  environment/CLI session and is never persisted.

### Notes to close before PUBLIC promotion (non-gating for private staging)

1. **Readback hash coverage boundary.** LFS-tracked payloads are verified
   via server-side sha256 metadata and the manifest is byte-verified, but
   small non-LFS files expose no sha256 through the HF API, so they are not
   hash-verified at readback. Consumer-side bootstrap verification against
   `release.json` closes the loop at download time, but the release gate
   should not depend on the consumer: extend readback to download and hash
   all non-LFS files (they are small) or document the boundary explicitly.
2. **`nemotron-local/voicechat-vllm:public-runtime` appears in the three
   replay `metadata.json` files.** Not a secret and not the EA/qualified
   private tag — it names the local capture-environment image, which public
   consumers cannot resolve. Either document the field as capture-environment
   provenance in the model card, or sanitize it to the runtime commit/digest
   it corresponds to.

Also verified: 8,152 files is comfortably inside HF repo limits; Xet/LFS
dedup does not alter content hashes on download, and the consumer-side
`release.json` verification is transformation-proof regardless of storage
backend.

---

## Delta re-review — 2026-08-04, pre-upload amendments

### Verdict: SAFE to stage a new v2 release and upload PUBLICLY.

Both pre-publication notes verified closed by inspection, plus one honesty
improvement I had not demanded but which materially strengthens the release:

- **Note 1 (readback coverage) — CLOSED.** `upload_hf.py` now tolerates HF
  LFS metadata as dict or object, and for every file whose server-side
  sha256 is absent (non-LFS) it downloads the file at the immutable readback
  revision with `force_download` and hashes it locally, comparing against
  the manifest entry — full-inventory hash verification at readback, no
  consumer-side dependence. Regression-tested (4 release tests pass, ruff
  clean).
- **Note 2 (`runtime.image` label) — CLOSED.** The model card documents the
  replay metadata field as provenance-only capture-environment labeling
  (MODEL_CARD.md:77-78).
- **Honesty upgrade:** the card now removes any sustained-pass implication
  and explicitly discloses the linear EarTTS step-time drift and growing
  playback queue lag from the sustained paced campaign, publishing the
  weights as "a qualified component/browser candidate" with sustained
  realtime as "a runtime release blocker" (MODEL_CARD.md:51-55). This
  matches the recorded sustained-qualification findings exactly — the public
  claims now align with the evidence, which was my remaining unstated
  concern about public promotion order: publishing weights publicly ahead of
  the Phase 4 public runtime is defensible precisely because the card no
  longer implies more than the weights' actual qualification status.

Conditions carried into the upload (already satisfied by the tooling):
upload as an immutable revision, record the returned commit for bootstrap
pinning, and keep v1 staging untouched as the superseded immutable
comparison point.

---

## Delta re-review 2 — 2026-08-04, replay transport packaging (tar)

### Verdict: APPROVE — safe to stage and publicly upload the tar-packaged
release. Stopping the 8,120-file upload was the right call.

- **Determinism — verified in source.** `build_deterministic_tar` sorts
  members by archive name, and normalizes every host-dependent field:
  `uid=0`, `gid=0`, `uname=""`, `gname=""` (explicitly cleared — no
  username leakage), `mtime=0`, `mode=0o644`, PAX format, uncompressed (no
  gzip timestamp trap). Covered by
  `test_replay_archive_is_deterministic_and_normalized`; 5 release tests
  pass.
- **Corpus identity chain — intact.** Source files are byte-verified
  against the calibration manifest BEFORE archiving; the outer release
  manifest hashes the archive; the per-file manifest ships for
  post-extraction verification. Transport packaging cannot alter identity
  without failing one of the three checks. The release SHA necessarily
  changes — correctly handled as a new v2 release.
- **HF practicality — sound.** One ~4–5 GB LFS object replaces 8,120
  transactions; total repo file count drops to ~32, making fresh bootstrap
  and readback verification fast and robust. Well inside HF file-size
  limits.
- **Replacing the partial repo — sound**, provided the incomplete public
  upload is deleted (or force-superseded) so no consumer can pin a
  half-populated revision, and v1 local staging remains immutable as the
  comparison point.
- **Carried requirement for bootstrap (extraction contract):** the docs
  promise verify-archive-then-extract-then-verify-files; when Phase 5/6
  implements extraction, use Python's tarfile extraction `filter="data"`
  (or equivalent member validation) as defense-in-depth against
  traversal/symlink members — the pre-extraction hash check already makes a
  malicious archive unreachable, but the extractor should not rely on that
  alone.

---

## Post-publication boundary review — 2026-08-04, published revision `a20c6854…`

### Verdict: APPROVE — Phase 3 publication boundary verified against the
LIVE public repository. No blockers.

Independently verified against the published repo (not local staging):

- **Signed manifest self-hash.** I downloaded `manifests/release.json` from
  `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` at immutable revision
  `a20c68547d27421c1e66bd501bf295569a0cc309` and recomputed the
  canonical-JSON self-hash:
  `1d8c3db8efbb87e0da3be86f1045957018082e1e8a339edaaf7e9bf4fb70671b` —
  exact match to the embedded value and the expected hash.
- **`runtime_contract` embedded** in the published manifest (38 keys — the
  frozen deployment contract travels with the weights).
- **Inventory** includes `calibration/replay.tar` (the deterministic
  transport archive from delta review 2).
- **License/notices:** `LICENSE` resolves at the immutable revision
  (HTTP 200).
- **No EA lineage:** zero matches for EA tags or local-image names in the
  published manifest.
- **Uploader tolerance is minimal:** only HF's autogenerated
  `.gitattributes` is allowlisted (`HF_AUTOGENERATED_FILES`); anything else
  unexpected fails verification.
- **Revision pinning:** both `config/artifact-release.json` and
  `config/production-candidate-1.toml` pin exactly `a20c6854…` — bootstrap
  cannot drift to `main`.
- **Tests:** 6 release tests pass.

### NIT

- Downloader fail-closed behavior was verified through the release test
  suite and the pinned pointer contract rather than line-by-line code
  inspection in this pass; the Phase 5/6 bootstrap review should walk the
  actual download/verify/extract code path once it consumes this layout
  (including the carried `filter="data"` extraction requirement above).

Phase 3 is complete: qualified PC1 weights, honest model card, deterministic
transport, immutable pinned revision, and a self-hashing manifest verified
end-to-end against the live public artifact.
