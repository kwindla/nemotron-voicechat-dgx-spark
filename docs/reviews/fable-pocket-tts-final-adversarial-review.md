# Fable final adversarial review — Pocket TTS typed-input bridge

Date: 2026-08-04 · Scope: commit `29d11bc` ("feat: bridge Playground text
input through Pocket TTS") vs parent `4a8f71c`, against
`docs/pocket-tts-text-input-plan.md` and both prior reviews. Method: full-diff
read, pinned-Pipecat source inspection (frame priorities, RTVI processor,
observer, interruption machinery), one targeted runtime experiment, full test
gates re-run at this commit (`tests/pipecat` + `tests/runtime`: 138 passed,
1 runtime-only skip; focused bridge suite 10/10; ruff clean). No code or live
services were modified.

## Verdict: APPROVE WITH CHANGES

No blocker found. Two IMPORTANT items (one documentation, one one-line
defensive hardening) and several suggestions. **No finding requires a
runtime-behavior change, so the three live browser sessions remain valid and
no new live E2E is required.** If the optional hardening in I2 lands, a
component-suite re-run is sufficient because it is provably behavior-preserving
under the current pin.

## Attack results — verified sound

**The two live interruption events are explained and harmless.** Event 1 is
RTVI's unconditional `interrupt_bot()` on `send-text`
(`pipecat/processors/frameworks/rtvi/processor.py:471-480`): it broadcasts a
DOWNSTREAM `InterruptionFrame`, then `flush_pipeline()` guarantees it fully
propagates (bumping the bridge epoch, `text_input.py:356-359, 448-451`)
**before** the append is pushed — deterministic cancel-then-replace ordering.
Event 2 fires when the model's RNNT detects the *synthetic* speech onset:
`llm.py:505-513` calls `broadcast_interruption()`, which reaches the bridge in
the UPSTREAM direction. The bridge invalidates only on
`direction == DOWNSTREAM` (`text_input.py:356`), so the active injection
survives — exactly what the complete live transcripts show. This direction
filter is load-bearing and correct: the upstream copy still resets the
bridge's (idle) process task via the base class, which cannot touch the
worker task created by `create_task`.

**Exact-once context.** The coordinator barrier (register at accept,
`committed()` fired by `VoicechatUserAggregator` after
`super()._handle_llm_messages_append` — which exists in the pinned Pipecat,
`llm_response_universal.py:1114`) guarantees the context commit precedes the
first synthetic sample. The RNNT transcript is pushed downstream of the
service only, so it can never re-enter the upstream user aggregator. Live
evidence: 9 typed turns across three sessions produced exactly 9 byte-exact
`user-llm-text` context events, once each.

**`skip_tts` bracket ordering cannot be subverted.** `LLMConfigureOutputFrame`
and `LLMMessagesAppendFrame` are both DataFrames (`frames.py`), strictly
ordered through processor queues; the RTVI processor emits open-bracket,
append, close-bracket in one task (`processor.py:481-494`). The bridge's
`_skip_tts` tracking is therefore race-free, and rejected appends leave the
bracket state consistent.

**Cancel-then-replace with mandatory RNNT closure.** The trailing-silence loop
deliberately checks only `_closing`, not staleness (`text_input.py:545-566`),
and runs iff `job.audio_started` — a mid-injection cancel still closes the
server-side turn before replacement audio or un-duck. The maxsize-1 queue plus
`_drop_stale_queued_job` is `task_done`-balanced in all paths, and prior-review
IMPORTANT #1 (cooperative cancellation of a stale generation) is genuinely
fixed: `_synthesize_sync` checks the per-job `threading.Event` before the lock
and at every chunk (`text_input.py:213-227`), so a replacement waits only to
the next chunk boundary, and stale-thread output is discarded via the epoch.

**Cleanup/disconnect races.** Client disconnect drives `worker.cancel()` →
`CancelFrame` (SystemFrame, immediate): the bridge sets `_closing`, cancels the
active generation, un-ducks, and `coordinator.clear()` releases any commit
wait (`text_input.py:365-370`); `cleanup()` cancels the worker with a 2 s
bound. The orphaned synthesis thread exits at its next chunk check, so a
subsequent session's first `synthesize` blocks on the generation lock only
briefly. Covered by
`test_disconnect_during_synthesis_cancels_job_and_releases_microphone`.

**Singleton / event-loop reuse.** The `__main__` prewarm runs in a temporary
`asyncio.run` loop, binding `_start_lock` there; every later `start()` hits the
`if self.ready: return` guard *before* lock acquisition
(`text_input.py:153-156`), so the loop-bound lock is never re-acquired from
the server loop. Prewarm failure exits the process before HTTP binds
(`demo.py:207-211`). Sequential sessions get fresh bridges over the shared
backend; the cached voice state is used with `copy_state=True`; a
settings-mismatched second initialization raises loudly
(`text_input.py:262-263`).

**Prewarm-before-ready, offline cache, provenance.** Prewarm exercises the
complete generate/codec/resample path (`text_input.py:184-187`); the voice-only
mock browser E2E explicitly disables it. `pocket-tts==2.1.0` and
`torch==2.9.1+cpu` resolve solely from the official CPU wheel index (0 CUDA
wheels in `uv.lock`); `HF_HUB_OFFLINE=1` is in the deployed environment;
`deploy/check-environment.py:13-30,87-100` rejects a missing cache; SHA-256
digests are recorded in `docs/provenance.md`.

**Unsupported RTVI options** are rejected pre-forward with zero context/audio
side effects and a single non-fatal error (component-tested for `skip_tts`,
`run_llm=false`, and multi-message appends).

**Live-session substantiation.** The three sessions' primary evidence (full
bidirectional RTVI captures) supports every claim in
`docs/reviews/fable-pocket-tts-live-e2e.md`: exact-once context items,
semantically equivalent RNNT transcripts, closed turn lifecycles, one real
`get_current_utc_time` round trip with a wall-clock-correct spoken answer,
zero spurious microphone turns against a live-but-silent capture device, and
cross-modal memory in the mixed session. The report's scope statements
(component-only evidence for failure paths; live coverage for the rest) match
what actually ran.

## IMPORTANT

**I1 — Typed messages that never became audio silently remain in
context/history the model has never heard.** When synthesis fails
(`text_input.py:466-471`), when a replacement cancels a job before its first
sample, or when the 5 s aggregation barrier times out, the append has already
been committed: Pipecat context and the client's history retain the message,
but the audio-only model never receives it, and the service never re-syncs
context mid-session (`llm.py:290-321` processes only function-call results
after the first context). The component tests *codify* this (both
`test_synthesis_failure_...` and `test_replacement_cancels_...` assert the
unheard message is present in context). The plan tolerates it for replacement
(RTVI append semantics) but `docs/known-limitations.md` documents only the
barge-in audio drop. Smallest fix: one known-limitations paragraph stating
that on typed-input synthesis failure or pre-audio replacement, the visible
transcript can contain a message the model never heard, and the error frame is
the authoritative signal. Docs-only; no runtime change; no new live E2E.

**I2 — Microphone-duck release depends on an atomicity invariant of the
pinned Pipecat, with no re-assertion at job start.** Duck is set only in
`_handle_append` (`text_input.py:399`) and released in the worker's `finally`
when the queue is empty (`text_input.py:476-477`). If the accept window
(`duck=True` → `push_frame` → `put_nowait`) could interleave with a finishing
job's `finally`, the duck would release with a job pending and never re-assert,
mixing live microphone audio into a synthetic utterance. I verified empirically
that this cannot happen today: an instrumented run (background loop-iteration
counter around `_handle_append`) measured **zero event-loop yields** in the
window — Pipecat's queues are unbounded and the RTVI observer's send path never
suspends, so the handler is atomic and `CancelledError` from
`_start_interruption`'s process-task cancellation (`frame_processor.py:869-889`)
can only land between frames (which also makes the `except Exception` at
`text_input.py:410-413` not catching `CancelledError` unreachable today).
This proof is pin-exact and will silently rot on the next Pipecat bump.
Smallest fix: re-assert `self._duck_microphone = True` when the worker
dequeues a job (one line at `text_input.py:461`), or release only when
`self._jobs.empty() and self._active_job is None`. Behavior-preserving under
the current pin → component-suite re-run suffices; no new live E2E.

## SUGGESTIONS

- **Preflight verifies artifact sizes, not hashes.** Provenance records
  SHA-256, but `deploy/check-environment.py` accepts any equal-size file.
  Hashing 219 MB adds ~1 s to preflight and closes a size-preserving swap.
- **Silent loss window for a queued append during genuine VAD barge-in.**
  Base-class interruption resets a processor's data queue
  (`frame_processor.py:880-889`); an append still queued at the bridge in that
  microsecond window vanishes with no error (RTVI's own interruption cannot
  hit this — it flushes before pushing its append). Consistent state, silent
  UX; at most worth a note.
- Dead `None`-sentinel branches remain in the worker loop and
  `_drop_stale_queued_job` (`text_input.py:419-421, 458-459`) — carried over
  from the implementation review.
- The 5 s aggregation-timeout path (`text_input.py:480-486`) is the one
  recovery branch still without a component test.
- The mixed live session's voice fixture is itself Pocket TTS `alba` audio —
  the same voice family the bridge injects. It exercised the real
  capture→WebRTC→RNNT path, but a human-recorded fixture would make the
  voice-turn evidence marginally more independent.
- Coordinator keys on `id(frame)`; safe with today's object flow, sturdier as
  an explicit job id (carried over).

## Changes required and E2E impact

Required for closure: the I1 known-limitations paragraph (docs-only) and a
decision on the I2 one-liner (recommended now; provably no behavior change
under the pinned Pipecat, so the existing live results stand). **Neither
requires code that alters runtime behavior, therefore no additional live E2E
is required.** If I2's hardening is applied, re-run the focused bridge suite;
if the Pipecat pin is ever bumped, re-validate the I2 atomicity assumption or
land the hardening first.

## Verdict: APPROVE WITH CHANGES
