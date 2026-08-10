# Fable adversarial review — standalone repository extraction plan

Date: 2026-08-04 · Plan: `docs/standalone-repository-extraction-plan.md` (draft) ·
Cross-checked against the LIVE deployment (container `voicechat-pipecat-v2` mounts/env,
user units), the strict server/bot source, the pipecat service plan, and branch
`codex/pipecat-voicechat-service`. Nothing changed or restarted; no secrets recorded.

## Verdict

**APPROVE WITH CHANGES.** Scope, exclusions, phase/gate structure, rollback posture,
and the external-prerequisite stance on weights/image are right. Four blockers are
concrete file-level gaps between the plan's allowlist and what the live server
actually requires or forbids; all are small plan edits.

## Blockers

**B1 — The retained Speech wrapper patch is a hard runtime dependency and is absent
from the repository shape.** `realtime_web_server.py:1560-1572` resolves
`patches/nemotron-voicechat-rnnt-turn-taking.patch` relative to the server file,
exits if missing, and byte-compares the live Speech working-tree diff against it at
startup. Edit: add `src/nemotron_voicechat_runtime/patches/` (with the patch) to the
repository shape and allowlist, and keep the startup validation path working after
the package move.

**B2 — The NVIDIA Speech tree is an undeclared external prerequisite with two live
couplings.** The container mounts `$HOME/src/Speech-nemotron-voicechat →
/workspace/Speech` (ro) at pinned commit `911ec674` with the retained patch applied
(3 files, +377/−29), and the server *dynamically loads*
`examples/speechlm2/function_calling/template.jinja` from it per tools-enabled
session (`realtime_web_server.py:2786-2791`, hard exit if missing). Edit: add the
Speech checkout (pinned commit + deterministic patch-apply step/script) to the
Non-negotiable boundaries, `docs/provenance.md`, and `docs/deployment.md`; the
allowlist gate ("every retained import and subprocess target") must also cover
runtime-opened FILES, not just imports.

**B3 — The planned warm-up WAV is EA-container-derived.** `en-US_sample.wav` exists
only under `artifacts/s2s/` and `artifacts/container/listings/s2s/` — both extracted
from the NVIDIA EA container. Shipping it in a shareable repo violates the plan's own
EA exclusion and has unknown redistribution rights. Edit: generate a fresh warm-up
WAV (Pocket TTS or recorded-by-project) for `assets/`, or document explicit rights.

**B4 — `patch_public_pad_pair_scheduler.py` is missing from the repository shape.**
It is one of the five manifest-pinned runtime sources and is enumerated in the
server's own manifest validation (`realtime_web_server.py:387`). Edit: add it to
`src/nemotron_voicechat_runtime/` in the shape and allowlist.

## Important

- **I1 — Env contract is the deployment interface.** The live container carries ~40
  `VOICECHAT_*/EA_*/S2S_*` variables (e.g. `VOICECHAT_RNNT_EOU_FRAMES=20`,
  `VOICECHAT_WEB_SPEECH_GATE_DBFS=-40`, DELTA/BOS-reset/pair-barrier flags). Phase 1
  provenance must capture them verbatim and `deploy/stack` must reproduce them; any
  drift silently changes qualified behavior. Add an explicit env-diff check to the
  Phase 6 gate.
- **I2 — Scrub reports before committing them into the new repo.** §2 saves reviews
  into the repository; three existing reports contain public tunnel origins. Strip
  tunnel URLs/hostnames (and any operator paths) from anything committed if the repo
  may be shared.
- **I3 — Sustained qualification is silently orphaned.** Neither §7 nor the service
  plan's outstanding true Phase 4 (Pocket TTS/Parakeet ten cases, sustained
  queue-lag) runs through the Pipecat path before migration. Add an explicit
  statement that true Phase 4 remains an outstanding post-migration gate, so the
  extraction cannot be read as retiring it.
- **I4 — Scope the shareability claim.** The qualified image is built on an
  EA/NGC-derived base (proprietary NVIDIA Torch build) and is not redistributable;
  `docs/provenance.md` should say so explicitly. `THIRD_PARTY_NOTICES.md` must cover
  NVIDIA Speech (Apache-2.0), the wrapper patch as a derivative work, the vLLM-fork
  monkey-patch targets, and Pipecat.
- **I5 — Manifest enumeration test.** The current manifest pins
  `realtime_web_client.html`, which the new repo excludes. Make Phase 4 assert that
  the generated manifest's runtime-source set equals the retained file set exactly
  (generate twice, byte-compare, and diff against the allowlist).

## Nice-to-have

- Provenance notes for every binary asset committed (`assets/`).
- A CI stub running the Phase 4 offline gates (lint, tests, scans, manifest
  reproducibility) so the new repo is verifiable without this host.
- Verify `docs/protocol-v2.md` is copied from the reviewed strict-v2 doc, not
  re-derived.

## Positives verified

- Bot tests are path-clean (no `/tmp` or worktree references) and carry their own
  fixtures/fakes — unit + browser E2E gates can run inside the new repo as claimed.
- Exclusion lists correctly cover the EA adapters, v1 protocol, experiment/benchmark
  campaign scripts (~100 files), calibration corpora, and vendored Pipecat.
- Rollback is operational-only with old worktrees preserved — consistent with the
  freeze discipline that saved the demo twice today.
- Migration gate ("mounts point only at the new repo or documented external
  model/Speech paths") is the right final check — after B2 it becomes accurate.
