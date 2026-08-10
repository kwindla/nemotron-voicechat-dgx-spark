# Fable final incremental review — Smart Turn and direct-text input plan, revision 4

Reviewed document: `docs/direct-text-input-plan.md` (revision 4, 2026-08-08)
Reviewed against: repo `codex/post-fc-pair-recovery` at `e19f906`; pinned
Pipecat in `.venv` (1.7.1.dev21, commit 5415f28); patched NVIDIA Speech
checkout at `/home/khkramer/src/Speech-nemotron-voicechat`.
Prior review: `docs/reviews/fable-smart-turn-direct-text-plan-r3-review.md`
(one important finding R3-1; nits R3-2, R3-3, R3-4).
Scope: incremental — the revision-4 deltas plus a final read of the whole
file. No implementation code modified.

## R3-1 — held-turn arbitration sizing, degradation, plurality, deadlines

**Resolved on all four required points, verified in the final text.**

(a) *Sizing.* 1B now allocates "a separate 1 MiB held-input budget (about
32 s of actual PCM, comfortably beyond the 20 s tool timeout plus Smart Turn
fallback)". Arithmetic checks: 16 kHz mono PCM16 is 32,000 B/s; 1 MiB /
32,000 B/s = 32.77 s. Worst-case epoch = 20 s function-call timeout + 5 s
post-FC recovery allowance + 3 s Smart Turn fallback = 28 s < 32.77 s. The
budget is explicitly *separate* from the 256 KiB live-sender FIFO
(256 KiB = 8.19 s, matching its "about 8 s" claim), which removes the
revision-3 ambiguity about a shared bound.

(b) *Graceful degradation.* The new 1B bullet makes held-input exhaustion "a
non-fatal overload outcome, not a wire protocol failure": the newly
overflowing held turn is dropped atomically — never sending its start,
partial audio, or commit — its remaining audio ignored through the local
commit, its metrics/ack timers cancelled, its per-turn state moved to a
terminal `dropped`, with a surfaced
`voicechat.input_turn_dropped(reason="arbiter_capacity")` diagnostic.
Earlier complete held turns are preserved and later turns are accepted when
capacity allows. Invariant 5 was consistently reworded to "a well-formed
commit **that reaches the server** … is always accepted", so the acceptance
guarantee no longer contradicts a local drop. Fatality is retained exactly
where it belongs and nowhere else: the live sender queue on sustained
backpressure ("partial transmission cannot be rolled back") and the
defensive server queue (see R3-3 below).

(c) *Plurality.* "Multiple locally completed turns may queue … and are
released in order"; the client state bullet keeps "one such record per
locally completed held turn, in FIFO order"; Step 3's plural "bounded FIFO"
now agrees. 1D adds the two-turns-held-in-one-epoch ordered-release test and
a third-turn overflow test proving one atomic drop with one diagnostic while
the session and earlier turns remain usable.

(d) *Deadline clocks.* "Acknowledgement deadlines begin only when the sender
actually transmits the held barrier, not while it is legitimately deferred",
with the fatal list narrowed to "post-send acknowledgement timeout" and a 1D
test for deferred acknowledgement deadlines.

## R3-2 — `function_call_output.applied` deadline

**Resolved.** 1B bounds the acknowledgement wait to "the configured 20 s
tool timeout plus a 5 s post-FC recovery allowance" — a decidable 25 s tied
to the existing configured values (`function_call_timeout_seconds` 20 s /
the 5 s function-event allowance already in the capability set) — after
which a missing acknowledgement is fatal.

## R3-3 — defensive server queue overflow policy

**Resolved.** 1C now states capacity exhaustion in the defensive
deferred-input queue is protocol-fatal, with the correct rationale: a
negotiated client that bypassed its required arbiter may already have sent a
partial turn. 1D separately proves live-send and defensive-server overflow
remain fatal, completing the fatal/non-fatal boundary enumeration.

## R3-4 — keyword-only constructor

**Resolved.** 1A now reads
`TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=1),
wait_for_transcript=False)`, matching the pinned keyword-only signature, and
additionally pins `user_turn_stop_timeout=5.0` explicitly with the
crash-recovery framing (matches the pinned default, so no behavioral
surprise).

## Late atomic Pocket first-model-mutation wording

**Verified sound.** 1C now anchors Pocket's `injection_started` /
`begin_input_turn` / non-cancellability to "the first actual model
mutation—not merely when a sub-frame PCM chunk enters `PcmFrameBuffer`",
performed under `model_input_lock` with a cancellation recheck, emitting
started/begin and processing "the first complete or final-padded 80 ms
frame as one non-cancellable transition". This closes a real race: Pocket
sends 20 ms chunks, so `PcmFrameBuffer` accumulates four sub-frame chunks
before the first model call (`server.py`, `pcm_buffer.push` yields only
complete 1,280-sample frames) — under the previous wording a job could
become non-cancellable, and context could be committed, before any model
state changed. The new wording makes the context-commit point coincide with
the first true mutation, consistent with invariant 8, and the
"final-padded" clause covers sub-80 ms syntheses. Cancellation/replacement
before that transition remains recoverable; after it, the job runs to
commit — matching the direct-mode atomicity model.

## Residual observations (non-blocking, editorial)

- Invariant 2 still says "holds the complete microphone turn" in the
  singular; the operative sections (1B, 1D, Step 3) are consistently plural
  FIFO. The invariant's emphasis — turns are held *complete*, in *bounded*
  memory — is correct as written; a plural noun would read cleaner.
- `voicechat.input_turn_dropped` borrows server-side diagnostic naming for a
  client-side (never-sent) event; "surfaced diagnostic" states the
  requirement clearly enough, and the implementation will choose the
  concrete channel. No behavior, test, or contract hinges on the name.

Neither observation changes behavior, tests, or contracts; neither reaches
the should-fix bar.

## Verdict

Every remaining finding from revision 3 is resolved in the final file, the
new numbers are arithmetically sound and traceable to configured limits, the
drop semantics are atomic and correctly scoped, the fatal/non-fatal boundary
is fully enumerated and tested, and the Pocket mutation-anchored transition
closes the last known race. No blocker, important, or should-fix finding
remains.

VERDICT: APPROVE
