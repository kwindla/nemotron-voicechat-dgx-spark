# Fable implementation review — Pocket TTS typed-input bridge

Date: 2026-08-04 · Scope: uncommitted diff (text_input.py, aggregators.py,
demo.py, env/deploy docs, tests) vs `docs/pocket-tts-text-input-plan.md` and
`docs/reviews/fable-pocket-tts-plan-review.md`. Gates accepted from Codex's
independent run (136 passed + 1 runtime-only skip; Chromium voice gate green;
offline Pocket gate 1.95× realtime, ~990 MB RSS). No code or services modified.

## Verdict: APPROVE WITH CHANGES (no blockers; two IMPORTANT items)

Both plan-review blockers are correctly closed:

- **B1 (cancel-then-replace + silence closure) — CLOSED.** The downstream
  `InterruptionFrame` bumps the epoch (`text_input.py:332-334, 423-424`); the
  serialized single-worker queue (maxsize 1) guarantees job B never starts
  before job A's teardown; and the decisive detail is right: the trailing-
  silence loop checks only `self._closing`, deliberately **not** job staleness
  (`:510-531`), while `_run_job` emits it whenever `job.audio_started`
  (`:482-483`) — so a job cancelled mid-injection still closes the server-side
  RNNT turn before the replacement's audio. The queued-but-unstarted case
  correctly skips closure (no audio was ever emitted). Covered by
  `test_interruption_closes_partial_audio_before_replacement`.
- **B2 (EOU-derived trailing silence) — CLOSED.** Derived from
  `VOICECHAT_RNNT_EOU_FRAMES` × 0.08 s × margin (validated > 1.0) plus one
  injection frame, ceiled to the 80 ms grid (`:52-56`); ≈2.08 s at the deployed
  profile. Covered by `test_deployed_eou_profile_has_margin_...`.

Also verified sound: exactly-once context (bridge forwards the append, waits on
the aggregator's commit ack via the per-connection coordinator before any
synthesis; `aggregators.py` VoicechatUserAggregator calls `committed()` after
`super()`); duck engages synchronously at accept, well before the first
synthetic sample, releases only when the queue drains, and is restored on
error/Cancel/End/cleanup; StartFrame creates the worker via `create_task` with
deterministic `cleanup()` cancellation; `skip_tts`/`run_llm` rejection matches
the plan (`:404-421`); 24→16 kHz via gcd-reduced `resample_poly` with correct
duration math; singleton backend with settings-equality guard + per-connection
bridge, prewarm-by-default exercising the full synth path before bot-ready;
`torch.set_num_threads(4)` caps CPU; new env vars documented and checked in
`check-environment.py`.

## IMPORTANT

1. **Replaced-job synthesis is not cooperatively cancelled.** A stale job's
   `to_thread` synthesis runs to completion under the generation lock, so a
   replacement typed send can wait up to the full synthesis wall time
   (~0.5× utterance length at 1.95× realtime) before starting. Smallest fix:
   a per-job `threading.Event`; check it in `_synthesize_sync`'s chunk loop
   and break early.
2. **Barge-in during typed playback drops the user's speech onset.** On a real
   interruption, injection stops but the mandatory ~2.08 s silence closure runs
   with the mic still ducked, so roughly the first two seconds of the user's
   barge-in utterance never reach the model, and the model may respond to the
   partial typed text. This is the least-bad resolution of the RNNT merge race
   (the closure must run), but it must be **documented as a known limitation**
   of typed input; a future refinement could shorten the closure when the
   cancellation source is a genuine user interruption.

## SUGGESTIONS

- The `"use run_immediately to replace"` error text is confusing since
  `run_immediately=true` is already mandatory; reword.
- The worker queue's `None` sentinel branch (`:432-433`) is dead code
  (shutdown goes through `cancel_task`); remove or use it.
- Coordinator keys on `id(frame)`; harmless today (same object flows through)
  but a weakref or explicit job id would be sturdier.
- Consider `os.nice`/affinity for the synthesis thread to further insulate the
  model server's codec cores.
- Test gaps worth closing later: context-commit timeout path, disconnect
  mid-synthesis, prewarm-disabled cold path, and the queue-full error branch.

## Test quality

The five bridge tests are behavior-level and cover the load-bearing invariants
(exactly-once commit-then-inject, duck window, rejection side-effect-freedom,
EOU margin, and the B1 replacement closure). With the two IMPORTANT items
addressed or documented, this is ready to land.

---

# Post-commit adversarial re-review — commit 29d11bc

Date: 2026-08-04 · Target: `codex/pocket-tts-typed-input` @ `29d11bc9…`
(fixed) · Emphases per request: concurrency, interruption, context
consistency, dependency/offline behavior, live-test evidentiary strength.

## Verdict: APPROVE — no blockers, no code changes required, so no live
re-qualification is needed.

### Both IMPORTANT items from the pre-commit review are closed

1. **Cooperative synthesis cancel — CLOSED.** Each job carries a
   `threading.Event` (`text_input.py:318`); `_synthesize_sync` checks it
   before generation and at every chunk (`:214-219`) and raises
   `TypedInputSynthesisCancelled` — no partial-audio return path exists. The
   event is set by `_invalidate_active_generation` (`:463`, reached from
   downstream `InterruptionFrame`, `CancelFrame`/`EndFrame`, and
   `cleanup()`) and by `_drop_stale_queued_job` (`:438`). I verified that
   cancelled ⇒ stale in every reachable path, so the `raise` fallback in
   `_run_job:498-501` is defensive-only. A disconnect mid-synthesis now
   orphans the thread for at most one chunk boundary.
2. **Half-duplex limitation — CLOSED.** `docs/known-limitations.md`
   documents the ducked-mic ~2.08 s closure window, its rationale (RNNT
   merge prevention), and user guidance.

Two pre-commit suggestions were also addressed: the misleading
"use run_immediately to replace" error was reworded (`text_input.py:394`),
and the suite grew from 5 to 8 tests covering replacement-cancel,
synthesis-failure recovery, and disconnect-mid-synthesis (10 cases pass
locally in 3.6 s; Codex's full run: 139 passed + 1 container-only skip).

### New verifications on the committed state

- **Concurrency / event loops.** The dev-runner prewarm uses
  `asyncio.run(...)` (temporary loop) while per-connection `start()` runs on
  the runner loop. Safe: `start()` short-circuits on `ready` before touching
  `_start_lock` (`:151-153`), so the lock never crosses loops after a
  successful prewarm; a failed prewarm crashes the dev runner at startup
  (fail-fast). `_drop_stale_queued_job` correctly re-queues a non-stale job.
- **Dependency / offline behavior.** Pipecat pinned to a git SHA with the
  `pocket-tts` extra; torch/torchaudio forced to the CPU wheel index via an
  explicit uv source (no CUDA torch can leak in); scipy/soundfile bounded.
  `deploy/check-environment.py` validates the three Pocket artifacts by
  exact HF-cache snapshot path and byte size, honoring `HF_HUB_CACHE` —
  first-use network fetches cannot silently satisfy a "qualified" deploy.
- **Live-test evidentiary strength.** `fable-pocket-tts-live-e2e.md` claims
  were cross-checked against `/tmp/fable-pocket-tts-live-e2e/` (driver
  scripts, RTVI captures, per-turn RTP stats, audio, screenshots for all
  three sessions) and are consistent; the report itself correctly discloses
  that failure-injection behaviors carry component-level, not live-session,
  evidence. Note: that report originated from a parallel Fable session, not
  this one; its evidence stands on its own.

### Residual (SUGGESTION only, non-gating)

- `_load_and_prewarm_sync` assigns `_model`/`_voice_state` before the
  prewarm-exercise check (`:180-186`), so a prewarm-synthesis failure leaves
  `ready=True` with an unexercised path; move the assignments after the
  check.
- The worker queue's `None`-sentinel branch remains dead code.
- Coordinator still keys on `id(frame)` (harmless with the current
  same-object flow).
