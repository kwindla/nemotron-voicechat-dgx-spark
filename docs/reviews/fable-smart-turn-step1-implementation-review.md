# Fable adversarial review — Step 1 implementation (Smart Turn microphone turns)

Reviewed: the uncommitted worktree against `codex/post-fc-pair-recovery`
`e19f906` — snapshot taken 2026-08-08 17:45:18 UTC (diff md5
`d4a1af1af34c49f10214d82674e74638`, 2,900 patch lines across 17 files),
verified identical to the worktree at review completion. Note: implementation
files were being edited while this review started (mtimes 17:35–17:42); every
finding below was re-verified against the frozen snapshot, not the earlier
partial reads.
Plan of record: `docs/direct-text-input-plan.md` revision 4, Step 1 only
(1A–1D, cross-cutting invariants 1–9, 11).
Also inspected: pinned Pipecat 1.7.1.dev21 in `.venv`, the retained NVIDIA
Speech patch, and the full test suites (`tests/pipecat` + `tests/runtime`:
258 passed, 2 skipped at snapshot).
No implementation code was modified.

## Verified correct (non-exhaustive, checked against code, not prose)

- **1A wiring** (`aggregators.py`): Silero `VADParams(confidence=0.7,
  min_volume=0.6, start_secs=0.2, stop_secs=0.2)`, keyword-only
  `TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=1),
  wait_for_transcript=False)`, `user_turn_stop_timeout=5.0` — exactly the
  pinned-API shapes. `LLMServiceMetadataFrame` keeps
  `is_realtime_service=True` with `user_turn_strategies=None`. External
  `UserStartedSpeakingFrame` broadcasts cannot phantom-trigger the VAD
  strategies (verified against the pinned turn controller), and mic turn
  frames still flow downstream so `VoicechatAssistantAggregator`'s
  function-result carve-out keeps working.
- **Ordering and deadlines** (`llm.py`): start/audio/commit share the single
  outbound FIFO; commit is enqueued only after all observed audio; ack
  deadlines (15 s, matching the docs) are armed in the sender when the
  barrier is actually transmitted — held barriers carry no ticking clock.
  Byte budgets: 256 KiB live PCM with capacity waiting and fatal sustained
  overflow; 1 MiB held-input budget.
- **Held-turn arbiter** (`llm.py::_hold_audio`): unassigned rolling pre-roll
  trims oldest-first under a 256 KiB sub-cap; an overflowing *held* turn is
  dropped atomically (start, audio, and commit all withheld; metrics
  stopped; correlation records purged; non-fatal
  `voicechat.input_turn_dropped reason=arbiter_capacity` diagnostic), earlier
  completed held turns survive, and the FIFO releases in order on the epoch
  terminal. Matches revision 4's R3-1 resolution.
- **Wrapper EOU** (Speech patch): `request_user_eou()` is mode-guarded,
  consumed only on an idle-agent position, forces BOS, clears blank/nonblank
  counters AND the full hypothesis state (`y_sequence`, punctuation
  accumulator, predictor state), sets `forced_bos`, resets transport
  counters; `reset_transport_session_state` clears every new latch; the
  blank-count, first-turn-fallback, and transport-energy EOU branches are
  gated by `_external_user_eou_mode` while speech confirmation, native EOS,
  barge-in EOS, and watchdogs are preserved. Combined with
  `engine.reset_user_transcript()` after commit, the pre-start
  pending-evidence capture in `protocol.step_events` (skipped for
  `synthetic_control` steps), and evidence freezing in the `turn_start`
  handler (pre-roll + partial drained before the cancellation drain), the
  stale-transcript fence (r3 F4 / r2 I1) is genuinely implemented.
- **Cancellation drain** (`server.py::cancel_response_to_terminal`):
  request-EOS latch, position-guarded zero-input steps to the acoustic
  boundary, output suppression at both the send and protocol layers, one
  `response.done(status="cancelled", reason="client_vad")` terminal, bounded
  by `max_tail_frames + 2` with a fatal `response_cancel_timeout`. The
  endpoint test asserts terminal-before-new-turn ordering.
- **Defensive function-cycle deferral** (`server.py`): a
  `DeferredClientInputQueue` (1 MiB, fatal overflow with
  `deferred_input_overflow`) defers `turn_start`/`append`/`commit` for the
  whole cycle (call emission → applied), replays afterward, and thereby
  guarantees `cancel_response_to_terminal` can never fire through live FC
  state. Correlated `conversation.item.function_call_output.applied` is
  emitted only after the fc-state flags are idle; the client holds mic input
  until that ack with a send-time 25 s (20 + 5) deadline.
- **Pocket migration**: `input_text.accepted` is admission-only client-side
  (context commits on `injection_started`); server `begin_mutation` runs
  under `model_input_lock`, re-checks cancellation, emits
  started/`begin_input_turn` immediately before the first complete-or-padded
  80 ms model frame, forwards PCM with no pacing sleeps, ends with the
  common explicit EOU (no closure silence), makes post-mutation
  cancellation/replacement queue behind (`input_text.request` no longer
  cancels a mutated job), and aborts the session on post-mutation failure.
- **Idempotency**: 16-deep `started_client_turns`/`completed_client_turns`
  re-ack retries without re-mutation (endpoint-tested for commit);
  capability negotiation flips `input_turn_detection`,
  `server_turn_detection`, and `settings_mutable_until` together, and
  un-negotiated barrier events get recoverable `unsupported_event` errors.

## Findings

### B1 (blocker). Typed requests bypass the ordered outbound path; ordinary speaking-while-typing can kill the session

`_request_typed_input` still sends `input_text.request` directly over the
websocket (`llm.py:361-371`), while microphone `turn_start`/audio/`commit`
travel through `_audio_queue`. Nothing on the client defers typed admission,
and the server's FC deferral set covers only the three
`input_audio_buffer.*` types. Consequences, all reachable by ordinary users:

1. User is speaking (their `turn_start` is queued or already open
   server-side) and submits typed text. The typed request overtakes queued
   barriers. If the mic `turn_start` then arrives while `active_typed` is
   set, the server's `input_turn_conflict` is **fatal** (`server.py:3310`) —
   session death from normal concurrent input. The same race exists at
   held-turn release: released barriers sit in `_audio_queue` while a fresh
   typed request jumps ahead of them.
2. If instead the mic turn is already open server-side when the typed job
   starts, `begin_input_turn("typed")` raises (`an input turn is already
   open`) and the typed job dies with `typed_input_failed` — the user's
   message is silently lost instead of being queued, which revision 4
   explicitly rejects ("admitted to the bounded local arbiter but not sent
   until that acknowledgement").
3. Typed-during-function-cycle is unguarded on both sides: the client sends
   immediately, the server admits it mid-cycle, and Pocket PCM is injected
   into a live function cycle — the exact violation the FC atomic epoch was
   designed to prevent (invariant 8; 1B arbiter paragraph).

Required: route typed requests through the same ordered arbiter — defer
admission while a local mic turn is open/queued, while a typed epoch is
active, and until the FC `applied` ack — or equivalently defer server-side
and make the conflict paths non-fatal for conforming-ordering overlaps.
Invariant 2 ("audio and turn barriers share one ordered path") must extend
to every input-initiating event, or it isn't an ordering invariant.

### I1 (important). Legacy RNNT mode inherits the post-FC applied drain — the capability flip is not atomic

The 64 × 80 ms zero-audio drain and `.applied` emission after
`conversation.item.function_call_output` run unconditionally
(`server.py:3202-3233`), not gated on `client_turn_detection`. In legacy
RNNT mode this splices up to 5.12 s of synthetic silence into the model
timeline after every tool result — silence the old qualified path never saw,
feeding RNNT blank-count machinery that is *not* disabled in that mode — and
emits a new event legacy clients never negotiated. Invariant 9 and 1B
require capability and behavior to flip atomically; 1D requires the
rewritten negative tests to prove it. Gate the drain and the `applied` ack
on client-turn mode (the legacy path kept its old semantics everywhere else
— this is the one leak).

### I2 (important). Typed-path interruption still uses pre-plan semantics, and cancelled sentence tails are flushed instead of discarded

`_speech_started` (now reachable only for `source=="typed"`) still
hard-clears `_response_id` (`llm.py:850-864`), so when a typed injection
interrupts an open response, the client drops the server's terminal
`response.done` and never emits `LLMFullResponseEndFrame` — precisely the
lost-terminal-correlation defect invariant 7 prohibits, and inconsistent
with the `_response_cancelling` machinery correctly used for
`InterruptionFrame`. Additionally, *both* interruption paths push the
sentence-buffer remainder as a `TTSTextFrame`; revision 4's 1B says "flush
already-delivered sentence text … exactly once, **discard its incomplete
local sentence tail**". The incomplete tail is the buffer remainder — it
must be dropped, in `_speech_started` and `_reset_response_after_interruption`
both. (The done-path double-flush guard is correct; it is the wrong half
that survived.)

### S1 (should-fix). No missing-response-boundary deadline after a committed ack

1B makes a "missing response boundary" fatal, but the client arms deadlines
only for the two acks. After `committed`, nothing bounds the wait for the
BOS-bearing response: the 45 s liveness loop never fires because
`voicechat.metrics` events keep arriving on subsequent audio, so a server
that acknowledged the commit but never produced BOS leaves the conversation
hung with TTFB metrics running forever. Arm a bounded
committed-to-response-created deadline (the commit-to-BOS path is fully
synchronous server-side, so even a conservative bound is decidable).

### S2 (should-fix). 1D-promised test coverage is substantially missing on the server side

Present and good: client-side held-FIFO/overflow/ack-timeout/telemetry/
byte-exact-typed tests; protocol contract tests for capability lifecycle,
pre-start evidence across synthetic drains, and cancelled-terminal
correlation; endpoint tests for the barrier lifecycle, commit idempotency,
and cancel-to-terminal ordering. Missing but explicitly promised by 1D:

- function-cycle deferral: buggy-client `turn_start` during a cycle is
  deferred and replayed; `deferred_input_overflow` is fatal;
- Pocket client-mode migration: admission-only accept, mutation-anchored
  `injection_started`, replacement-queues-behind, completion without
  closure silence, cancellation before first mutation;
- zero-RNNT-token commit (the invariant-5 acceptance path);
- position-limit failure inside each synthetic path (cancellation drain,
  commit EOU step, applied drain);
- cancellation tail-deadline fatality (`response_cancel_timeout`);
- live-queue burst larger than the byte budget (bounded memory,
  audio-before-commit ordering under pressure).

### N1 (nit). `commit_input_turn` from the plan does not exist as a protocol primitive

Turn closure rides the forced BOS through `_finish_turn` inside
`step_events`. Functionally equivalent and event-ordering-correct; record
the naming deviation in the implementation notes so the plan and code
vocabularies stay reconcilable.

### N2 (nit). Dead branch and coarse metric stop

`open_client_turn_id == client_turn_id` re-ack in the `turn_start` handler
is unreachable (the `started_client_turns` check precedes it), and the
dropped-held-turn path calls `stop_all_metrics()`, which also cancels
metrics unrelated to the dropped turn. Both harmless; both worth a tidy.

### N3 (nit). Un-arbitrated typed failure surfaces as a generic error

Until B1 is fixed, `begin_input_turn("typed")` colliding with an open mic
turn surfaces as a non-fatal generic `typed_input_failed`. After B1 this
path should be unreachable; if it is kept as a defensive guard, give it a
distinct error code.

## Verdict

The implementation is far more complete than a first pass suggests — the
wrapper EOU fence, evidence freezing, cancellation drain, FC deferral queue,
held-turn arbiter, and Pocket mutation anchoring all match revision 4 and
survived adversarial reading against the pinned Pipecat and Speech sources.
But one ordering hole (typed requests outside the FIFO) makes ordinary
concurrent input session-fatal, the legacy-mode drain leak breaks the
atomic-flip invariant, the typed interruption path contradicts the
correlation/discard semantics the plan settled in two review rounds, and the
server-side test ledger 1D promised is still substantially unwritten.

VERDICT: CHANGES REQUIRED

Required changes, in order: B1 (typed-request arbitration through the
ordered path, with the conflict-fatality boundary re-drawn), I1 (gate the
applied drain on client-turn mode), I2 (typed-path cancelling semantics +
discard both incomplete tails), S1 (committed-to-response deadline), S2
(server-side 1D test coverage). N1–N3 at the implementer's discretion.
