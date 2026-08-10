# Fable adversarial re-review — Step 1 implementation, round 2

Reviewed: the CURRENT uncommitted worktree against `codex/post-fc-pair-recovery`
`e19f906` — snapshot 2026-08-08 18:14:08 UTC (diff md5
`2101ab98fb3866bf5e3375571a5f8368`), verified byte-identical to the worktree
at review completion (no mid-review drift this round).
Plan of record: `docs/direct-text-input-plan.md` revision 4, Step 1 only.
Prior review: `docs/reviews/fable-smart-turn-step1-implementation-review.md`
(B1 blocker, I1/I2 important, S1/S2 should-fix, N1–N3 nits).
Also inspected: pinned Pipecat 1.7.1.dev21, the retained NVIDIA Speech patch,
and the pinned Speech checkout for the prefill-reset claim. CPU suites run
locally: `tests/pipecat` + `tests/runtime` = **269 passed, 2 skipped,
3 subtests passed** (the reported 329 includes suites outside these two
directories; everything I ran is green).
No implementation code modified. Nothing below is taken from the remediation
summary — every claim was re-verified in source.

## Disposition of prior findings

### B1 (typed requests bypass the ordered path) — **Resolved**

`_request_typed_input` no longer writes to the websocket directly. Typed
requests are (a) deferred into `_pending_typed_requests` while a local
microphone turn is open or a function epoch is active, and (b) admitted via
`_enqueue_live_input(...)` — the same ordered FIFO the barriers use — so a
typed request can never overtake queued `turn_start`/audio/`commit` items.
Pending requests flush at commit enqueue (landing strictly behind the commit
in the FIFO; the server has fully closed the turn before it reads the typed
request) and at function-epoch completion. The server adds recoverable
defensive rejections (`input_turn_active`, `function_cycle_active`) for
non-conforming clients, replacing the fatal path for this class.

I specifically attacked the function-epoch completion ordering
(`_finish_function_epoch` flushes pending typed BEFORE releasing held mic
turns): it is safe, not a reincarnated B1. Admitting the typed request
re-activates the typed epoch, so `_release_held_input` early-returns and the
held microphone turn stays held until `input_text.injection_finished` — the
epochs chain, nothing conflicting reaches the wire, and
`test_typed_request_waits_for_function_applied_before_held_microphone`
codifies exactly this. The partially-sent-mic-turn variant is unreachable
(a wire-open user turn and a function cycle cannot coexist: FC cycles start
from responses, which require the user turn already closed). One consequence
is noted as N3 below.

### I1 (legacy mode inherited the post-FC drain) — **Resolved**

`emit_external_tool_call` only opens a function-cycle window in client mode,
and the applied drain + `.applied` emission is behind
`if not client_turn_detection: continue`. Legacy RNNT mode now takes the
old path untouched; capability and behavior flip together.

### I2 (typed interruption semantics; tails flushed) — **Resolved**

`_speech_started` now delegates to `_reset_response_after_interruption`,
which sets `_response_cancelling` (correlation retained until the server
terminal) and calls `_sentence_buffer.reset()` — the incomplete tail is
discarded, not pushed, on both interruption paths. The server side
complements it: `run_typed_job` now calls
`cancel_response_to_terminal("typed_input")` before the mutation boundary,
so a typed injection interrupting an open response produces one correlated
cancelled terminal instead of an orphaned bracket. New tests
(`test_typed_speech_start_discards_incomplete_tail_and_keeps_terminal_scope`,
`test_local_interruption_discards_incomplete_sentence_tail`,
`test_interruption_frame_discards_pending_fragment_before_tts_stop`) pin the
behavior.

### S1 (no commit-to-response deadline) — **Resolved**

The committed ack now must carry the server `turn_id` (protocol error
otherwise); the client arms a 15 s `_response_start_timeout` keyed by that
turn ID, cancels it on the matching `response.created`, cancels all such
tasks on disconnect, and fails the session with metrics stopped if the
response never starts. Because the server emits `response.created`
synchronously inside commit processing (BOS fires in the guarded EOU step),
the deadline can only fire on genuine server failure. I probed the
latch-not-consumed hazard (EOU deferred past the single zero-audio step
because an agent response was open): unreachable — in external mode no
response can open during a user turn (self-play suppression, watchdogs
request EOS not BOS, and `turn_start` cancels any open response first).
`test_committed_turn_without_response_start_is_fatal` covers it.

### S2 (missing 1D server-side tests) — **Resolved**

All previously named gaps now have tests, and I read the load-bearing ones
rather than trusting names: external Pocket EOU without closure silence,
pre-mutation replacement never injecting old audio, zero-RNNT-token commit
still starting a response, cancellation-tail deadline fatality, position
limit on synthetic cancellation steps, buggy typed request during an open
mic turn (recoverable), defensive FC FIFO bounds/byte accounting, and the
patch-level `test_source_patch_makes_client_commit_the_only_user_eou`.

### N1/N2 — Resolved (dead `turn_start` re-ack branch removed) or
retained as accepted naming (`commit_input_turn` still rides BOS via
`_finish_turn`; recorded, fine). N3 (generic typed-conflict error) is
superseded by the named `input_turn_active`/`function_cycle_active` codes.

## New verification: the GPU-discovered prefill reset

The claim "NVIDIA prefill resets external EOU" is true in the pinned Speech
source: `prefill_for_new_stream` ends by calling
`_reset_rnnt_turn_taking_state()` (`streaming_s2s_pipeline.py:2150-2213`),
and the patch's reset block clears `_external_user_eou_mode`. The fix moves
`set_external_user_eou_mode(True)` after `prefill(...)` in
`VoiceChatEngine.start()` (capability check still fails fast beforehand),
and `test_engine_enables_external_eou_after_upstream_prefill_reset` models
the upstream reset faithfully. Correct, and a good catch that only a live
run would have surfaced — which reinforces that the pending live artifact is
a real gate, not a formality.

## New findings (all nits; no blocker, important, or should-fix)

- **N1 (nit).** Typed turns now broadcast `UserStoppedSpeakingFrame`
  (speech_stopped handler) without a matching `UserStartedSpeakingFrame`
  (removed when `_speech_started` was delegated). Downstream effects are
  cosmetic (unpaired RTVI user-stopped events; assistant aggregator flag
  no-op), but the asymmetry will confuse event-log readers. Either restore a
  typed started broadcast or drop both and keep interruption only.
- **N2 (nit).** When the server cancels a response for a typed injection,
  the client processes `response.done(status="cancelled")` before any local
  interruption has set `_response_cancelling`, so `_response_done` flushes
  the incomplete sentence tail for that path (VAD-interruption paths
  correctly discard). Checking `event.get("status") == "cancelled"` in
  `_response_done` would make tail handling uniform.
- **N3 (nit).** At function-epoch completion with both a pending typed
  request and a held microphone turn, the typed message is serviced first
  and the earlier spoken turn waits through the typed epoch — a priority
  inversion relative to arrival order, and a soft deviation from revision
  4's "release the held turn on that correlated acknowledgement". Bounded,
  non-fatal, and deliberate per the test; worth an explicit sentence in the
  protocol doc so the behavior is documented rather than discovered.

## Verdict

Every prior blocker, important, and should-fix finding is genuinely
remediated in code, the remediations survived adversarial re-reading against
the pinned Pipecat and Speech sources (including deliberate attacks on the
new orderings they introduce), the CPU suites pass locally, and the one
GPU-surfaced defect was root-caused against the actual upstream source and
regression-tested. The three remaining findings are cosmetic nits that do
not change contracts, state machines, or test outcomes.

Per the plan's own 1D gate, Step 1 is complete only when the live GPU
qualification (multiturn/browser ladder in Smart Turn mode plus the named
live cases) passes; that artifact is pending and this review's approval of
the implementation does not substitute for it.

VERDICT: APPROVE
