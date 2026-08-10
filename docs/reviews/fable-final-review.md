# Fable final code/deployment review — standalone stack

Date: 2026-08-04 · HEAD: `b737414` · Live E2E: PASS (see `fable-live-e2e.md`).
Verdict: **APPROVE for live operation**, with **one repository-scope cleanup
blocker** to land in the next commit (not deployment-affecting).

## Blocker (repo scope, not runtime)

**B-final-1 — legacy standalone-codec layout violates strict scope.**
`src/nemotron_voicechat_runtime/cpu_codec_worker.py:24-31`
(`load_codec_state`) accepts a standalone `codec.safetensors` checkpoint layout
("legacy standalone codec" per its own docstring) before falling back to the
combined `model.safetensors` slice. The production public checkpoint ships the
combined file only, and the repository's stated contract is strict-v2 with no
backward compatibility. Dead on the production path but contradicts the
minimality guarantee. Fix: delete the standalone branch (lines 26-31), reword
the docstring, and remove/adjust the tests that pin the legacy path —
`tests/runtime/test_cpu_codec_worker.py:26-40`
(`test_load_codec_state_prefers_legacy_standalone`) and the loader-guard lambda
at `:50`. Keep the missing-file error message accurate (combined-only).

## Verified

- **Minimality:** commit history since extraction is exactly the reviewed fixes
  (`aea7a0d` closure, `b737414` skeleton-dispatch contract); no stray or
  experiment files; working tree clean apart from review docs.
- **Security:** no tunnel origins, credentials, or private paths in tracked
  files; deployment `.env` tightened to mode **0600** (good); served client
  pages carry no tokens; model and Pipecat ports loopback-only; public origin
  is the Pipecat UI only, posture documented.
- **Provenance:** running container image `fa11b6ef4ea2…` matches the
  regenerated `manifest-production-candidate1-standalone.json` pinned image ID;
  15 runtime sources hash-pinned (repo-relative paths, including the Dockerfile
  overlay and both patch modules). The `VLLM_ALLOW_INSECURE_SERIALIZATION`
  question is **investigated and cleared**: the flag is absent by design; the
  one-time prefill hang was residual GPU/driver state from a failed boot,
  falsified as an env dependency by the clean-memory determinism retry.
- **Runtime mounts/ownership:** all model-container mounts read-only except the
  dedicated trace dir; repo mounted read-only; Speech tree read-only at the
  pinned commit; `nemotron-voicechat-pipecat` and `nemotron-voicechat-ngrok`
  user units both active and own their processes.
- **Strict-v2-only:** protocol v2 advertised in health; no legacy wire branches
  (remaining "legacy" strings are the B-final-1 codec loader).
- **Tests:** `uv run --frozen pytest -q tests` → **107 passed, 3 skipped**.
- **Known limitations documented:** `docs/known-limitations.md` covers the
  tool-skip rate, single-client backend, session frame limit, and EOU tuning.

## Post-fix closure (commit 8e0255135424fd007b6c69778ca1f3c9d90e12c6)

**B-final-1 CLOSED without regression.** Adversarial review of `b737414..8e02551`:

- `load_codec_state` no longer accepts a `load_file` callable or probes
  `codec.safetensors`; the combined public `model.safetensors` is the sole
  loader path with a precise `FileNotFoundError` otherwise
  (`cpu_codec_worker.py:26-28`). The legacy branch, its import surface, and its
  test (`test_cpu_codec_worker.py`, −24 lines) are removed together.
- The only remaining `codec.safetensors` string in the tree is inside the
  byte-frozen upstream RNNT wrapper patch, which must not be edited — correct
  to retain.
- Old-stack prose removed from protocol/deployment docs (zero
  hotfix/legacy-stack references remain in those files).
- Gates re-verified independently at `8e02551`: **106 passed, 3 skipped**
  (9.5 s), `ruff` clean. External manifest regenerated at this commit with 15
  pinned hashes (per deployment note); to be confirmed live at next
  health-ready.

No further blockers. The repository now satisfies the strict
no-backward-compatibility scope in full.

## Final smoke after B-final-1 cleanup (2026-08-04 04:34 UTC)

Post-redeploy health confirms `manifest-production-candidate1-standalone.json`
and codec layout `public_combined`. One real-Chromium SmallWebRTC session via
the unchanged public origin, trace `25e55794-5430-4a99-becc-586bd774c17d`:
normal turn answered correctly ("The code word is sapphire. Two plus three is
five."), explicit tool prompt produced a real `get_current_utc_time` call with
a **0.005 s** round trip and spoken "The current UTC time is four thirty four
UTC" matching actual 04:34 UTC; 88,600 inbound RTP bytes with a live playing
audio track; clean disconnect; `active_client: false` after. No anomalies.
Evidence: `/tmp/fable-final-e2e-20260804/smoke/`.

**Final verdict: PASS.**
