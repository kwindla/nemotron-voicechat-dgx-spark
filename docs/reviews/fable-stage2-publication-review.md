# Fable Stage-2 Publication Review (adversarial)

- Date: 2026-08-06
- Scope: public worktree `/home/khkramer/src/nemotron-voicechat-public`, branch `main`,
  candidate `cfdcb5e` vs public baseline `origin/main` = `f24f31b`; private source
  commit `043f3da`. Read-only review: history/boundary, conversion/release tooling,
  bootstrap activation, claims-vs-evidence, static tests/lint, README preservation.
- Mid-review correction (orchestrator): `cfdcb5e` is SUPERSEDED. User README-only
  commit `0941475` ("README.md changes", 2026-08-05) had not landed on public main
  and its delta is being restored; the candidate will be amended. This review's
  verdict therefore applies to the **Stage-2 content at `cfdcb5e`**; a final
  README-only delta check on the amended SHA is explicitly deferred to a follow-up
  pass. `cfdcb5e` itself is NOT declared push-ready.

## Verdict

**SAFE WITH CHANGES** — the Stage-2 conversion/release/qualification content at
`cfdcb5e` is publication-safe (0 blockers in that content); the candidate must be
amended to restore user commit `0941475`'s README edits (in progress), and the two
IMPORTANT documentation findings below should be fixed or consciously accepted
before push. The amended SHA needs a final README-only delta re-check.

## Findings

### BLOCKER

None in the Stage-2 content. (The missing README commit `0941475` would have been a
blocker for pushing `cfdcb5e` as-is; it was independently discovered and is being
remedied by amendment — tracked under README preservation below.)

### IMPORTANT

1. **Nine-session wrong-tool sample membership is undocumented and not uniquely
   derivable from evidence.** `docs/known-limitations.md:20-24` claims "four
   sessions selected the intended response path and five selected the unrelated
   UTC-time tool ... a measured nine-session sample." I could reconstruct exactly
   one 9-session set matching 4/5: {r11, r15, r17, r18-attempt-1, r18-attempt-2,
   converted-r2-attempt-1, converted-r2-attempt-2, converted-r3-attempt-1,
   converted-r3-attempt-2} (details in the session table below). However, four
   additional intended-path browser passes from the same 2026-08-05 downloaded
   campaign (r12, r13, r14, r16, each "1 passed" on the same shipped pytest gate)
   and two non-tool browser failures (r9 locator timeout, r10 `assert 0 > 0`
   audio assertion) are silently excluded, with no retention criterion stated
   anywhere. The stated 5 wrong-tool events are fully evidenced; the "nine-session"
   denominator is a curatorial choice. Exclusion is conservative (it overstates the
   wrong-tool rate), so the claim does not flatter the model, but "measured sample"
   implies a defined population. **Fix:** add a private evidence index enumerating
   the nine retained sessions (log path + outcome each) and one sentence in
   `docs/known-limitations.md` stating the retention criterion (e.g., "sessions
   retained from full qualification campaign runs"), or recount over all countable
   model sessions (8 intended / 5 wrong-tool / 2 non-tool failures ≈ 15 incl. the
   two non-tool fails).

2. **`release/MODEL_CARD.md` now diverges from the published HF card at the pinned
   immutable revision, and the card is a signed-release input.**
   `tools/release/prepare_hf_release.py:253` copies `release/MODEL_CARD.md` into
   the staged release as `README.md`; its SHA-256 enters `files_sha256` and thus
   `release_sha256`. Stage-2 rewrote the card's qualification section (sustained
   pass, 0.289/0.375 slopes) — correct against evidence — but the HF release at
   `a20c68547d27` (pinned in `config/artifact-release.json:5-6`, release
   `1d8c3db8...`) still carries the stage-1 text ("sustained ... remains a runtime
   release blocker"). Consequences: (a) HF viewers read stale, now-falsified
   limitations; (b) re-running `stage_release` from this tree yields a
   `release_sha256` ≠ the pinned `1d8c3db8...`, so `verify_staged_release` only
   matches the historical staging. Nothing breaks at runtime (bootstrap verifies
   the downloaded release against its own signed manifest), but the divergence is
   undocumented. **Fix:** either upload the updated card to HF `main` (weights
   untouched; record the new revision alongside the immutable weight revision) or
   add a note in `docs/weight-reproduction.md`/`release/MODEL_CARD.md` that the
   in-repo card supersedes the frozen copy inside the signed release.

### NIT

1. **Millisecond crash window and stranded backup in atomic activation.**
   `src/nemotron_voicechat_runtime/cli.py:383-391`: between
   `os.replace(layout.release, backup)` and `os.replace(candidate, layout.release)`
   a SIGKILL/power loss leaves no `artifacts/release`; the still-valid downloaded
   release is stranded in a hidden `.source-converted-release-*/downloaded-release`
   dir that nothing sweeps (the `finally` only runs in-process). Recovery is
   implicit (next bootstrap re-downloads and re-verifies), so this is not a
   corruption risk — the design is otherwise sound: candidate is fully staged and
   `verify_release`-checked before the swap (cli.py:381), re-verified after
   (cli.py:386), rollback on exception (cli.py:387-391). EXDEV is handled by
   design: the staging dir is created inside `layout.artifacts` (same filesystem as
   `release`, artifacts.py:65-70) so both `os.replace` calls are same-device, and
   `_link_or_copy` (cli.py:328-335) falls back to `copy2` on cross-device
   hardlinks. **Fix:** sweep stale `.source-converted-release-*` dirs at bootstrap
   start, or document the recovery path.

2. **Hardlink aliasing between the active release and the conversion work tree.**
   `_link_or_copy` hardlinks files from `~/.cache/nemotron-voicechat/conversion/release/*`
   into the activated `artifacts/release`. A later in-place edit under the work
   tree would alias into the active release. Mitigated: conversion outputs are
   0644, `bootstrap --offline` re-verifies, and stage markers detect drift on
   rerun. **Fix (optional):** always `copy2`, or note the aliasing in
   `docs/weight-reproduction.md`.

3. **Private refs share the public worktree's object store.** The public worktree
   uses the private repo's `.git`; local refs `private/main-archive`, `codex/*`
   exist. `git push origin main` is safe (only curated ancestry is reachable), but
   `git push --all` / `--mirror` from either worktree would publish private
   history. **Fix:** push exactly `git push origin main` from the public worktree.

4. `tools/release/upload_hf.py:56-63` passes `revision=`/`private=` to
   `upload_large_folder`; some `huggingface_hub` versions reject those kwargs. The
   pinned lock evidently works (upload succeeded) — just keep the lock pinned.

## History / publication boundary (verified clean)

- `git log --oneline` on public `main`: exactly `cfdcb5e -> f24f31b -> 2a1d37e`.
- `cfdcb5e` tree vs private `043f3da` tree: identical except README (the intended
  divergence, now superseded) and removal of `docs/reviews/**` (22 files) and
  `docs/reproducible-clone-to-playground-plan.md`. No review/plan docs in the
  candidate tree (`git ls-tree -r cfdcb5e` grep: zero hits).
- Secrets/paths: `git grep -E 'hf_[A-Za-z0-9]{20,}|ngrok\.app|khkramer|api[_-]?key\s*=|HF_TOKEN\s*=' cfdcb5e` → zero hits (rc=1). Working-tree hit only in the
  untracked `.git` worktree pointer file. "modal" hits are `multimodal/*` path
  substrings; no `nemotron-voicechat-ea`/Modal/old-env references.
- Large blobs: largest is `config/calibration-corpus-production-candidate-1.json`
  (1.4 MB, pre-existing in f24f31b); nothing new above ~30 KB except tooling
  source. No unexpected >5 MB blobs.

## Conversion/release tooling (verified)

- Exact-public input pinning: `tools/conversion/extract_public_vllm.py:21-26`
  (checkpoint revision `fb0f94ea...`, model SHA, Nano skeleton revision) with
  per-file size+SHA-256 verification (lines 81-104, 190-207).
- Replay-corpus identity: `tools/conversion/reproduce_release_artifacts.py:197-208`
  rebuilds the manifest with frozen provenance labels
  (`_frozen_source_names`, lines 161-171) and refuses on any mismatch with
  `config/calibration-corpus-production-candidate-1.json`.
- Dual fresh-process conversion: reproduce spawns each Nano run as a separate
  subprocess with `--single-run` (lines 251-283) and each EarTTS run as a separate
  subprocess (291-311); `assert_byte_identical` after each pair (284, 312) and
  again in the verification payload (443-446). `convert_nano_marlin_gptq.py:546-547`
  enforces exactly one of `--single-run`/`--repro-output`.
- Disk floor: `extract_public_vllm.py:24,485-489` (100 GiB post-conversion floor,
  refuses lower via arg validation at 429-444); `convert_nano_marlin_gptq.py:116`
  (default 100 GiB); `prepare_hf_release.py` staging floor; bootstrap-side
  `require_free_space` with `minimum_conversion_remaining_gib = 100`
  (`config/production-candidate-1.toml:42`).
- Resumable self-authenticating stage markers:
  `reproduce_release_artifacts.py:69-133` — canonical-JSON self-hash, full
  per-file re-hash on resume, hard refusal of unvalidated partial outputs.
- Component gates wired: stage `component-gates` (lines 386-430) runs Nano replay
  gate plus EarTTS eager and CUDA-graph gates before verification is written.
- Permission normalization: `_make_release_readable` (174-179, invoked at 435)
  chmods only, after all gates; documented in `docs/weight-reproduction.md`.
- BOM byte-match: `_verify_expected_artifact` (136-158) requires exact file
  inventory + SHA-256 equality with `config/qualified-candidate-1.json`.
- HF uploader fail-closed: `tools/release/upload_hf.py` — `REPO_ID` hardcoded and
  the `--repo-id` flag only accepts it (`choices=(REPO_ID,)`, line 115); readback
  at the immutable `info.sha` (76, 102); non-LFS files (no `lfs.sha256`) are
  downloaded and hashed (97-106); inventory must equal manifest ∪
  {manifests/release.json} with only `.gitattributes` tolerated (14, 89-91);
  manifest read-back byte-compared (80-81). `verify_staged_release`
  (`prepare_hf_release.py:321-328`) checks repository identity, `derived_from_ea`
  is False, manifest self-hash, and payload/files-hash equality.

## Bootstrap `--convert-from-source` + activation (verified, see NITs 1-2)

- `cli.py:319-325` requires `--asr-model`; `406-475` extracts the corpus atomically
  (`tarfile ... filter="data"`, tmpdir + `os.replace`, 416-424), runs the
  containerized reproducer with read-only input mounts, then
  `_activate_converted_release` (343-403): reproduction-report self-hash check
  (354-362), staged candidate built from the verified downloaded release with only
  nano/eartts swapped (373-380), **full signed-release verification before the
  swap** (381), same-filesystem two-step `os.replace` swap with rollback (383-391),
  post-swap re-verification (386). Bootstrap state records
  `artifact_materialization: source-converted` + `reproduction_sha256` only after
  success (493-511). Stale-cache: a work root without valid stage markers is
  refused for reuse by the reproducer.

## Static verification

- `./voicechat test` (public tree, cfdcb5e): **239 passed, 7 skipped, 3 warnings
  in 1.96s** — matches the checklist claim "239 passed, 7 explicit dependency
  skips".
- `uv run --frozen ruff check .`: **All checks passed!**

## Evidence verification (what I checked in each file)

- `/home/khkramer/.cache/nemotron-voicechat/conversion/release/reproduction.json`:
  kind `voicechat_release_artifact_reproduction`; self `sha256` =
  `ab80bbcb19c5...` (matches `docs/release-checklist.md`); nano
  `composite_sha256` = `3acecd057f73...` and eartts = `6e103b3cbcbb...` — both
  equal the BOM values in `config/qualified-candidate-1.json:118,139` and
  `config/conversion-recipe.json:55,63`; `dual_conversion.{nano,eartts}.byte_identical
  = true`, runs = 2 each; component-gate paths recorded.
- `/tmp/voicechat-public-live-20260805-source-agreement.json`: file SHA-256 =
  `1bf4ea21391865f2...` — matches `docs/qualification.md` claim exactly. `passed:
  true`; converted root = converted-r2, downloaded root = downloaded-r18; converted
  sustained explicitly staged from converted-sustained-r4 (matching the documented
  `--converted-sustained-report` staging); all six per-side live verdicts true.
- `/tmp/voicechat-public-live-20260805-converted-sustained-r4/sustained/report.json`:
  SHA-256 = `ee62c4547da3...` (matches docs); passed; 18/18 typed answered;
  ASR 18/18 speech, 0 near-silent; closed at 12,000 frames
  (`session_position_limit`); slope 0.375 ms/min; peak debt 399.69 ms; 1200 s.
- `/tmp/voicechat-public-live-20260805-downloaded-r18/report.json`: overall passed;
  browser passed on attempt 2 of max 2 with both attempt logs SHA-256-recorded in
  `browser-attempts.json`; sustained slope 0.289 ms/min, peak debt 381.673 ms,
  18/18 typed.
- `/tmp/voicechat-public-live-20260805-converted-r2/sustained/report.json`
  (negative evidence): failed, peak debt 2979.732 ms ≈ the documented "2.98
  seconds", 16/18 typed — supports the known-limitations malformed
  function-call-recovery paragraph verbatim.
- converted-r1 `browser-attempts.json`: both attempts failed in <1 s with a
  Playwright executable-missing error — infrastructure, correctly not counted as
  model sessions.

## Claims-vs-evidence table

| Claim (public tree) | Evidence | Verdict |
|---|---|---|
| MODEL_CARD: sustained passed on downloaded + reproduction; slopes 0.289 / 0.375 ms/min; peak debt 382 / 400 ms; 18/18 typed; graceful frame-limit close | r18 report (0.289, 381.673, 18/18) and converted-sustained-r4 (0.375, 399.69, 18/18), both `session_position_limit` close | **Supported** |
| qualification.md: agreement SHA `1bf4ea21...`, sustained SHA `ee62c454...`, 18/18 typed, 0.375 slope, 399.69 ms debt, 12,000 frames | Recomputed both SHA-256; values match byte-for-byte | **Supported** |
| known-limitations: failed source run with 2.98 s debt retained as negative evidence; fresh-stack repeat passed at 400 ms | converted-r2 sustained (2979.732 ms, failed) + converted-sustained-r4 (passed) | **Supported** |
| known-limitations: "four intended / five UTC-tool ... measured nine-session sample" | 5 wrong-tool events fully evidenced (r17, r18a1, cr2a1, cr3a1, cr3a2 — all UTC-time responses to the sapphire-memory prompt). Unique matching 9-set exists {r11, r15, r17, r18×2, cr2×2, cr3×2 → 4 intended/5 wrong}, but r12/r13/r14/r16 (4 more intended passes) and r9/r10 (2 non-tool failures) are excluded without a stated criterion | **Numerically supportable; membership undocumented (IMPORTANT #1)** |
| release-checklist: 239 passed / 7 skips, Ruff clean | Reproduced this review | **Supported** |
| release-checklist: reproduction self-hash `ab80bbcb...`; revision `a20c685...`; signed release `1d8c3db8...` | reproduction.json self-hash matches; pins consistent across `config/artifact-release.json` and `production-candidate-1.toml` | **Supported** |
| release-checklist: dual conversions byte-identical and match candidate manifest | reproduction.json `dual_conversion` + `_verify_expected_artifact` against BOM | **Supported** |

Browser session inventory (2026-08-05 campaigns): r8 infra (Playwright strict-mode
locator bug, 12 s), r9 fail non-tool (response timeout), r10 fail non-tool (audio
assertion, memory answered), r11-r16 passed (intended), r17 wrong-tool fail,
r18 a1 wrong-tool / a2 pass, converted-r1 infra ×2, converted-r2 a1 wrong-tool /
a2 pass, converted-r3 a1+a2 wrong-tool.

## README preservation

`git diff f24f31b..cfdcb5e -- README.md` contains **only** the intended Stage-2
additions: the stage-2 pending paragraph replaced by the published-pipeline
paragraph, plus the new `--convert-from-source` section (dual conversion, PC1
byte comparison, gates, atomic activation, 100 GiB floor). It does not touch the
stage-1 sections. **Known gap (being remedied):** user commit `0941475`
("README.md changes" — "first open source, full-duplex" phrasing, simplified
offline wording, relocated/reworded ngrok section, shortened tunnel caveats) was
never merged to public main, so those edits are absent from `cfdcb5e`; the
orchestrator is restoring that exact delta and amending the candidate. The
amended SHA requires a follow-up README-only delta check (verify: amended tree =
cfdcb5e tree + exactly the 0941475 README delta rebased onto the stage-2 README).

---

## FINAL VERDICT against amended candidate `c4a02c2` — 2026-08-06

Structural verification (performed directly): `c4a02c2` is HEAD of
public `main` with clean three-commit curated ancestry
(`2a1d37e` → `f24f31b` → `c4a02c2`), clean worktree, README-only delta
from the content-reviewed `cfdcb5e` (26 lines, zero secret hits), and the
README is BYTE-IDENTICAL to the private Stage-2 source `043f3da` — the
user's 0941475 wording and Stage-2 additions are both present exactly.

**Ruling: SAFE TO PUSH after one paragraph — close IMPORTANT-1 first.**

- **IMPORTANT-1 (pre-push, one edit):** enumerate the nine wrong-tool
  sample sessions (or state the retention criterion) in
  `docs/known-limitations.md`. The claim is numerically reconstructible
  and conservative, but an undocumented curated subset in a
  qualification claim is exactly what this project refuses elsewhere.
  One paragraph; amend; I will delta-check the final SHA in minutes.
- **IMPORTANT-2 (same-day, does not gate the git push):** the HF model
  card at the pinned revision still carries stage-1 "sustained is a
  release blocker" text now contradicted by evidence. Update the card on
  HF `main` from the private tooling (maintainer path per the carve-out
  ruling) and record that re-staging the release would change
  `release_sha256` versus the pinned `1d8c3db8…` — pinned-revision
  consumers see the historical card; that divergence must be documented.
- **Push discipline (NIT, mandatory practice):** the public worktree
  shares the private object store — push EXACTLY `git push origin main`;
  never `--all`, `--mirror`, or tag pushes from this checkout.
- Tests on the public tree: 239 passed / 7 skipped; ruff clean.

---

## FINAL DELTA VERDICT — 2026-08-06, authoritative candidate `be66414`

### SAFE TO PUSH. All findings closed. This is the release commit.

Verified on the actual `c4a02c2..be66414` diff (7 files, +103/−16; zero
secret/path hits; ancestry unchanged at three curated commits; clean
worktree):

- **IMPORTANT-1 CLOSED correctly:** `known-limitations.md` now reports
  the COMPLETE retained population — 15 usable sessions: 8 intended, 5
  wrong-tool, 2 non-tool failures — with the sole exclusion criterion
  stated explicitly (missing Chromium / locator defect ⇒ no usable model
  verdict) and the "complete count, not an estimate" framing plus the
  bounded-retry contract. No curated subset remains.
- **IMPORTANT-2 CLOSED as documented divergence:** frozen signed HF
  revision `a20c6854` vs the superseding source card is distinguished in
  weight-reproduction, the model card, and an unchecked release-checklist
  follow-up; the HF metadata refresh is correctly sequenced after the
  source push with weights unchanged.
- **NIT hardening verified in code:**
  `_remove_stale_conversion_activations` runs only AFTER the active
  release is downloaded and verified (ordering and rationale in-code);
  source-converted components use `copy_function=shutil.copy2` with the
  explicit no-hardlink-aliasing comment; preflight reserves the copied
  component bytes atop the 100-GiB floor; docs state transactional
  behavior and crash recovery.
- Public-tree tests verified locally (19/19 in test_cli); reported clean
  detached run 241 passed / 7 skips, ruff clean.

Push exactly `git push origin main` from the public worktree. With this
push, Stage 2 publication completes the reproducibility project's public
release; the only remaining action is the sequenced HF model-card
refresh.

---

## Trivial-delta confirmation — 2026-08-06, PUSH SHA `227f91c`

Verified: the `be66414..227f91c` delta is exactly one file
(`docs/release-checklist.md`, +6/−2) checking two boxes, both true
against retained evidence — the restart/SIGKILL recovery gate (verified
live in my Delta 10 review of r8) and the completed adversarial review
boundary (accurately described). No code, no claims, no other files.

**`227f91c` is CONFIRMED as the push SHA. The verdict chain is complete.**
