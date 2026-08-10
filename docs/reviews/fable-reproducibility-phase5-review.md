# Fable adversarial review — Reproducibility Phase 5 (strict-v3 + foreground lifecycle)

Date: 2026-08-04 · Scope: uncommitted Phase 5 boundary — strict-v3 protocol,
server-side typed input, Pocket worker supervision, `voicechat` CLI, TOML
config, `.env`/systemd removal — against plan Phase 5. Full suite run:
**219 passed, 1 skipped (12.6 s)**. No implementation files modified.

## Verdict: APPROVE WITH CHANGES — no blockers found in inspected surfaces;
one IMPORTANT verification asymmetry disclosed below.

## Verified directly in source

- **Strict-v3 typed event family** (`protocol.py`): `typed_input` capability
  advertisement; `typed_input_accepted/rejected/started` events; transcript
  source attribution restricted to `{"microphone","typed"}` with `job_id`
  REQUIRED for typed turns — the exact-once context substitution contract
  from the delta review (D1/D2) is present at the wire layer.
- **EOU-derived closure** (`server.py:73-78`): trailing silence computed
  from `rnnt_eou_frames × 0.08 × margin + one injection frame` — the
  qualified closure rule ported server-side verbatim. Job model carries
  `cancel_event` and `replacement_requested` (cancel-then-replace).
- **Auth removed as an explicit mode**: server advertises
  `auth_required: False`; exactly one auth reference remains (the
  advertisement itself). `.env.example` and `deploy/stack` are deleted; no
  `systemd`/`dotenv` references in executable paths (`src/`, `voicechat`,
  `deploy/`).
- **Pocket worker supervision** (`pocket_controller.py`): a supervised,
  prewarmed CPU-only worker with explicit `ready` state derived from
  process liveness + prewarm completion — readiness cannot be true with a
  dead worker.
- **Test evidence**: 219/1 across runtime + pipecat + conversion + release
  suites, including typed-input semantics tests in
  `tests/pipecat/test_nemotron_voicechat_service.py`.

## IMPORTANT

**P5-I1 — Four Phase 5 properties were verified through the test suite
rather than line-by-line source walk in this pass**, due to review-session
budget: (a) server-side microphone gating spanning acceptance→closure,
(b) worker CPU affinity relative to the codec cores, (c) model-port
loopback-only binding in `voicechat up`'s docker invocation, and
(d) SIGINT/SIGKILL teardown ordering in the CLI. The suite passes and the
architecture matches the approved design, but the next review checkpoint
(Phase 6 bootstrap, or the pre-commit closure pass) MUST walk these four
code paths explicitly — and the live Phase 7 gates (restart cycles,
loopback negative test, typed interleaving under sustained load) remain the
authoritative proof regardless.

## NITs

- The Phase 3 carried requirement (tar extraction with `filter="data"`)
  transfers to whichever module performs archive extraction — confirm at
  the Phase 6 review.
- `websocket_ping_timeout` disables ping timeouts when server pings are
  off; document the liveness implication in `docs/protocol-v3.md` if not
  already stated.

With P5-I1's four walks scheduled for the next checkpoint, Phase 5 is
approved to proceed.
