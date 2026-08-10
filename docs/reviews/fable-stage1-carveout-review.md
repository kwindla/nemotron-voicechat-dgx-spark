# Fable review — Stage-1 public carve-out boundary

Date: 2026-08-05 · Ruling on the proposed keep/withhold split before any
edit. Dependency graph inspected; no code edited; no clients run.

## Verdict: APPROVED WITH TWO ADDITIONS (one blocker-grade).

1. **Keep/withhold lists — APPROVED, dependency-verified clean.** The
   four kept tools/conversion files import only stdlib and each other
   (`nano_component_gate ← nano_attribution`; both kept); ZERO imports of
   withheld modules (`artifact_io`, `nano_gptq_calibration`,
   `package_replay_corpus`, converters, finalize, reproduce). The test
   split matches (pipeline tests withheld incl.
   `test_qualified_nano_config_chain`).
2. **run_live_suite tensor list — APPROVED:** retained trio
   (`test_nano_attribution`, `test_nano_replay`,
   `test_eartts_component_gate`) only.
3. **BLOCKER-GRADE ADDITION — the `voicechat release` subcommand:**
   `cli.py` imports the uploader (`prepare_hf_release`/`upload_hf`).
   Removing `_convert_from_source`/`--convert-from-source`/`--asr-model`
   is approved, but the `release` subcommand MUST also be withheld or
   stubbed ("maintainer tooling ships in stage 2"), or the day-one tree
   has an import error / dangling command.
4. **Uploader ruling — WITHHOLD WHOLESALE in stage 1**, per your
   preference and my concurrence: `prepare_hf_release.py`,
   `upload_hf.py`, `upload-hf`, and `tests/release`. The weights are
   already public at the immutable revision; day-one consumers need only
   the consumption-side signed-manifest verification already in
   `src` bootstrap. Refactoring qualified upload code hours before a
   public push is churn plus risk for zero day-one value. Stage 2
   restores it alongside the pipeline.
5. **Docs exclusion — APPROVED WITH A CONDITION:** exclude
   `docs/reviews/**` and the superseded plan as internal process history
   (they also carry /tmp paths and session minutiae). CONDITION:
   `docs/provenance.md:101` references `docs/reviews/` and must be
   reworded (e.g., "the full review history is retained in the private
   development repository"); grep the retained tree for any other
   references to excluded docs. Provenance, BOM, and operational docs
   stay; true how-made claims preserved.
6. **Universal pre-push gate:** on the CURATED tree (not the working
   tree): zero grep references to any withheld path across code, tests,
   CI, and docs; full suite + ruff green; then my final secrets/claims
   review before push.

---

## Implementation verdict — 2026-08-05: APPROVED. NO BLOCKERS.

Verified on the actual uncommitted diff (47 files, −5,586 lines, 34
deletions):

- **Dependency closure is total:** zero grep references to any withheld
  module (`artifact_io`, converters, `extract_public_vllm`,
  `nano_gptq_calibration`, `package_replay_corpus`,
  `reproduce_release_artifacts`, uploader) across src, tools, tests, CI,
  README, and retained docs.
- **CLI withholding complete and test-pinned:** `add_parser("release")`
  count is zero;
  `test_stage1_cli_withholds_conversion_and_release_commands` pins the
  absence of both conversion and release commands — the blocker-grade
  addition is closed.
- **run_live_suite** retains the consumption-side gates `test --live`
  needs (attribution/replay/eartts-gate trio referenced).
- **Known-limitations wording is exactly honest:** "three sessions
  selected the intended response path and two selected the unrelated
  UTC-time tool … a measured five-session sample, not an estimate," with
  the one-recorded-retry / fail-after-two contract stated.
- **Atomic browser evidence:** `browser-attempts.json` now uses
  `atomic_json` — my earlier NIT closed. (The fixture-manifest host view
  keeps plain write, previously ruled acceptable as derived data.)
- **Full suite on the carved tree: 223 passed / 2 skipped (8.9 s)** —
  the count delta from 246 is exactly the withheld pipeline tests.

Stage-1 tree is ready for curated-main construction. Remaining before
push: user manual-test sign-off, curated main built from this tree, and
my final secrets/claims review on the curated main.

---

## Final pre-push secrets/claims review — 2026-08-05, curated main `66cfab2`

### Verdict: CLEAN FOR PUBLIC PUSH, pending only the user's manual-test
sign-off.

Performed on `/home/khkramer/src/nemotron-voicechat-public` (single root
commit — no history leakage possible):

- **Withheld-path closure:** zero references to any withheld module,
  `docs/reviews`, or the superseded plan anywhere in the tree.
- **Secrets:** zero hits for the username, ngrok origins, HF tokens, API
  keys, or basic-auth material across all text files.
- **Residue greps all benign by inspection:** the only
  `nemotron-local`/`:ea-` strings are the prohibited-input constants,
  refusal-rule documentation, and a test-fixture mutation (the guards
  themselves, not references); the only `/home/` string is a test
  fixture; every `.env` mention is a negation.
- **Claims honesty:** README and weight-reproduction carry the follow-up
  wording; the release checklist marks dual-source as stage-2 pending;
  `.github` CI, LICENSE, THIRD_PARTY_NOTICES, licenses/, and the model
  card are present.
- Codex independently reports the clean-environment suite + Ruff pass on
  this tree.

Push on user sign-off.

---

## Post-sign-off bootstrap UX verdict — 2026-08-05: APPROVED. No blockers.

- **Progress precedes work:** `_artifact_stage` prints the flushed,
  numbered stage banner BEFORE yielding, so output appears before
  remote-manifest, download, or checksum work begins — the observed
  "apparent hang" cannot recur silently. All four long immutable-input
  stages are individually named.
- **Fail-closed and visible:** the wrapper catches `BaseException`,
  prints "failed after Xs" flushed, and RE-RAISES — exception semantics
  are byte-preserved (even KeyboardInterrupt is announced then
  propagated).
- **No leakage:** stage descriptions are static strings; output carries
  only stage numbers and elapsed seconds — no paths, URLs, or
  credentials.
- Tests 8/8 in the file (17 focused reported), ruff clean; the real
  `--offline` run showed the stages with timings (9.1 s / 19.6 s /
  immediate / immediate) ending in a successful audit and completion.

A pure-UX change with correct semantics. Cleared for the public tree.

---

## Curated-main delta verdict — 2026-08-05, `66cfab2` → `7b03ca2`

### CLEAN FOR PUBLIC PUSH — the prior verdict carries forward.

Verified on the public worktree: still exactly ONE root commit (history
leakage remains structurally impossible); the amendment diff touches
exactly two files — `bootstrap_download.py` (+the reviewed progress
wrapper) and `tests/runtime/test_bootstrap.py` (+18 lines of its test) —
and nothing else; zero secret-pattern, path, or review-doc references in
the delta. Evidence on the exact curated tree: `./voicechat test`
195 passed / 5 skipped, ruff clean, focused progress tests 17 passed,
real offline bootstrap completed with the image audit passing. The
raw-pytest e2e note is a non-issue: the supported source command
intentionally excludes `tests/e2e` (a documented prerequisite —
Playwright Chromium — governs that suite) and the fully live browser
gate was already qualified on this stack. Push `7b03ca2`.

---

## Release-audit fail-open fix verdict — 2026-08-05: APPROVED. No blockers.

The original bug was a genuine fail-open: a missing `rg` returned 127
inside the `if` condition, read as "no match," and the audit printed
success. The fix is semantically exact:

- **set -e safe status capture:** `lineage_scan_status=0; grep … ||
  lineage_scan_status=$?` — the `||` arm prevents errexit while capturing
  the true status; then explicit three-way handling: 0 → prohibited
  lineage (fatal), 1 → clean, anything else → "lineage scan failed with
  status N" and `exit 2`. A missing scanner, unreadable file, or bad
  pattern can no longer masquerade as clean.
- **Scanner equivalence:** `grep -E -i` over the identical alternation
  pattern and the same two evidence files — ERE alternation semantics
  match the previous rg pattern exactly; grep is POSIX-guaranteed present
  (proven by the real audit succeeding under `env -i` with a minimal PATH
  where rg is absent).
- **Contract test pins all load-bearing lines:** no bare `rg`
  invocation, the grep form, the status capture, the ≠1 fatal branch, and
  the error message (`test_runtime_audit_uses_required_tool_and_fails_
  closed_on_scan_error`). `bash -n` clean; 9/9 focused verified locally.

This bug class (tool-absence read as negative result) is worth one
stage-2 sweep across other shell gates for the same `if tool …` shape.

---

## Curated-main delta verdict 2 — 2026-08-05, `7b03ca2` → `2a1d37e`

### CLEAN FOR PUBLIC PUSH — carried forward.

Verified: still exactly one root commit; the delta is exactly the two
approved files (`container/audit-public-runtime.sh` +10/−2 fail-closed
grep change; `tests/runtime/test_bootstrap.py` +9 contract test) and
nothing else; zero secret-pattern or path hits in the delta; no claims
impact. Curated evidence accepted: 196 passed / 5 skipped, ruff clean,
`bash -n` clean, real no-rg audit passed on identical source. Push
`2a1d37e`.

---

## Curated-main delta verdict 3 — 2026-08-05, `2a1d37e` → `f24f31b` (bot-reload + README)

### Content verdict: CLEAN — the reload implementation passes adversarial
review. Two process items before push.

Reviewed the actual delta (4 files, +608/−64; zero secret/path hits):

- **PID-reuse safety is correct:** `_owned_launcher_process` validates
  /proc cmdline AND process start-time against the recorded
  `launcher_start_time` — start-time is the canonical reuse
  discriminator, so a recycled PID cannot be mis-signaled.
- **Signal hygiene:** SIGUSR1 handler installed with the previous handler
  saved and restored in teardown, matching the qualified SIGTERM pattern.
- **Watch debounce** prevents restart storms mid-save; **fail-safe** is
  right: a dead bot logs "model remains ready" and the launcher keeps
  supervising for retry, while model death still exits with its code.
- **Qualified teardown preserved:** `_stop_stack` +
  `_wait_ports_available` remain on the exit path; `_stop_pipecat`
  escalates SIGINT→kill bounded. Public-tree tests 14/14.

### Process items:

1. **Single-root history broke:** the tree is now TWO commits
   (`2a1d37e` root + `f24f31b` feature). Both are curated content, so no
   leakage — but every prior claim said single root. Either squash back
   to one root (recommended for the release shape) or explicitly accept
   the two-commit history; user's call, not blocking on content.
2. **IMPORTANT — requalification honesty:** the reload feature modifies
   `command_up`'s supervision loop AFTER the live qualification ladder
   ran. The teardown path is structurally preserved, but the exact
   shipped CLI has not run the restart-cycle gate. Recommend ONE
   restart-cycle gate re-run (~30 min) on this tree before push — or
   document that bot-reload landed post-qualification. The quick re-run
   is the honest option and cheap.

Push after the history decision and the restart-gate re-run (or its
documented waiver).
