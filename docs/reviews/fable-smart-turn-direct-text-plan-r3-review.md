# Fable adversarial re-review — Smart Turn and direct-text input plan, revision 3

Reviewed document: `docs/direct-text-input-plan.md` (revision 3, 2026-08-08)
Reviewed against: repo `codex/post-fc-pair-recovery` at `e19f906`; pinned
Pipecat in `.venv` (1.7.1.dev21, commit 5415f28); patched NVIDIA Speech
checkout at `/home/khkramer/src/Speech-nemotron-voicechat`.
Prior reviews:
`docs/reviews/fable-smart-turn-direct-text-plan-r2-review.md` (F1–F6) and
`docs/reviews/codex-smart-turn-direct-text-plan-r2-review.md` (B1, I1–I4).
Reviewer: Fable (adversarial pass; no implementation code modified).

## Disposition of revision-2 findings

### Fable F1 — typed/microphone overlap was protocol-fatal

**Resolved.** 1B introduces a client-side half-duplex outbound-input arbiter:
from typed-request enqueue through `input_text.injection_finished` (and,
for tool cycles, until the correlated
`conversation.item.function_call_output.applied` acknowledgement), a detected
microphone turn is held intact (pre-roll, start, audio, commit) and released
after the epoch's terminal event. 1C narrows protocol-fatality to "an overlap
that bypassed both documented arbiters, or another impossible state", adds a
bounded server-side deferred-input queue as defense against buggy clients,
and explicitly forbids forcing agent EOS through live FC state. 1D adds
speak-during-typed-injection unit, runtime, and live cases. The shape is
right; one consequence of the chosen bound is escalated below as R3-1.

### Fable F2 — `turn_start` during an open function-call cycle

**Resolved.** The function cycle is now an explicit atomic epoch: the client
withholds the mic turn until result injection *and post-FC recovery* complete,
gated on the new correlated `…function_call_output.applied` ack; the server
never forces agent EOS through live FC state and buffers barriers from
non-conforming clients. 1D adds "microphone barge-in while a function result
is pending and after post-FC recovery" plus a barge-in-during-tool-call live
case. The `fc_state` surfaces needed to define "function state idle" exist
(`server.py:1328-1337`).

### Fable F3 — false-positive VAD policy was implicit

**Resolved as an explicit policy.** Invariant 5 now states the design bet in
the open: a pinned-VAD-passing turn is authoritative "because Nano may
understand perception features even when RNNT emits no text", zero-token
commits produce one BOS, and the 1D gate spells out the accepted consequence
("may yield an empty/odd response rather than silently dropping possible
quiet speech"). The Silero operating point is now pinned
(`confidence=0.7, min_volume=0.6, start_secs=0.2, stop_secs=0.2` — all real
`VADParams` fields), and 1D adds both ambient-noise rejection and a
deliberately induced false positive with defined expected outcomes. This is
the "explicit, parameter-pinned acceptance" branch of my required change,
taken cleanly. The missing-response-boundary fatal rule is now consistent,
since every commit yields a BOS.

### Fable F4 — stale transcript inheritance on rapid re-speak

**Resolved.** Each commit snapshots the final transcript, resets/fences RNNT
display/hypothesis state, and clears the pending ledger; 1D's rapid
speak→commit→speak case now asserts transcript correctness, not just event
order; cancellation-drain tests prove synthetic blanks cannot mutate the
frozen ledger.

### Fable F5 — stale `settings_mutable_until` value

**Resolved.** Renamed to `"first_model_input_frame"` with the rationale
stated (1B).

### Fable F6 — sentence-buffer flush on cancellation

**Resolved.** 1B defines flush-once semantics: already-delivered sentence
text flushes into context exactly once at local interruption, the incomplete
local tail is discarded, and no second flush happens at the deferred
`response.done(status="cancelled")` terminal. 1D adds a single-flush unit
test. (Discard-the-tail is a policy change from today's flush-the-tail
`4972c86` behavior; it is explicit and defensible for cancelled speech.)

### Codex B1 — agent-EOS injection does not itself yield `response.done`

**Resolved.** 1C replaces the single "required guarded model step" with
`cancel_response_to_terminal(reason="client_vad")`: under the model lock,
latch `request_agent_eos`, run position-guarded zero-input steps until
`AcousticResponseBoundary` emits `end`, suppress newly produced output, emit
exactly one cancelled-status terminal, bounded by the acoustic tail deadline
(25 frames = 2 s worst case), session-fatal on deadline or position-limit
failure. 1D tests both immediate-silence and tail-deadline completion. This
matches the actual boundary machinery (`AcousticResponseBoundary`,
`server.py:983` ff.).

### Codex I1 — pre-start RNNT evidence

**Resolved.** The bounded pending-turn ledger holds evidence from
pre-`turn_start` audio without public lifecycle events, attaches it on the
matching start, never replays consumed model audio, is frozen and tagged
against synthetic-control blanks during cancellation drains, and is cleared
at commit and teardown.

### Codex I2 — Pocket context timing in Step 1

**Resolved.** 1C makes `input_text.accepted` admission-only in Step 1 and
moves exact context/transcription commitment to the existing
`input_text.injection_started` event (which the current client receives and
ignores — `llm.py` handles it as a no-op today, so the change is localized);
1D adds cancellation/replacement-before-start tests.

### Codex I3 — Pocket wall-clock pacing

**Resolved.** The 20 ms pacing loop (`server.py:2652-2670`, verified present)
is removed: PCM forwards immediately through serialized model calls. The
accompanying change making Pocket jobs non-cancellable after first PCM is a
deliberate tightening that removes the partial-utterance-replacement hazard
flagged in round 1 — a defensible trade, explicitly tested
("non-cancellable Pocket completion after first PCM").

### Codex I4 — serialized memory accounting

**Resolved.** 1B bounds raw PCM and serialized JSON/base64 event bytes and
control items separately, forbids an unbounded intermediate WebSocket queue,
and 1D adds a serialized-memory-bounds test with process high-water capture.

## New findings

### R3-1 (important). Held-turn overflow escalates ordinary user behavior to session death, and the hold capacity is never sized

Invariant 2 says a typed/tool epoch "holds the complete microphone turn in
bounded memory," and 1B routes `turn_start`, audio, and commit through "one
FIFO" whose raw-PCM bound is 256 KiB (~8 s) — with "Overflow or a missing
acknowledgement is fatal."

Put together: a user who talks for more than ~8 s while a typed job is
injecting kills the session. That is ordinary behavior, not a client bug —
and it is strictly worse than today's contract, which *discards* mic audio
during typed jobs non-fatally (`microphone_packet_gated`,
`docs/known-limitations.md`). The epochs are not necessarily short, either:
maximum typed text (1,000 chars) synthesizes tens of seconds of Pocket audio
whose model-speed consumption still takes a two-digit number of seconds, and
a function cycle can hold the arbiter for up to the 20 s function-call
timeout. The plan also leaves the held-turn capacity unspecified if it is
*not* the sender FIFO's 256 KiB (1B is ambiguous on whether the hold shares
that bound), and 1B speaks of holding "the complete microphone turn"
(singular) while Step 3 says "microphone turns … in a bounded FIFO" (plural)
— second-turn-during-one-epoch semantics are undefined.

Required change: (a) size the hold explicitly from the worst-case epoch
(max typed-audio duration at model-speed consumption plus the function-call
timeout plus margin), stating whether it shares the sender FIFO budget;
(b) on capacity overflow of a *held* turn, degrade gracefully — drop the
held microphone turn with a diagnostic event and cleanly reset the client
turn state machine to `idle` (this matches today's documented limitation and
loses one turn, not the session) — reserving session-fatality for the send
path's sustained overflow, where it belongs; (c) define
multiple-held-turn semantics (FIFO of turns, matching Step 3's wording); and
(d) state that acknowledgement timeouts start when a barrier is actually
sent, not while it is held.

### R3-2 (nit). The `function_call_output.applied` acknowledgement has no time bound

"A missing acknowledgement is fatal" — but no deadline is named. The
capability set already carries `function_call_timeout_seconds` (20 s) and
`function_event_emit_timeout_seconds` (5 s); the applied-ack wait should be
bounded by a stated function of these plus the post-FC recovery/drain
deadline, so "missing" is decidable.

### R3-3 (nit). Server deferred-input queue overflow policy unstated

1C's bounded deferred-input queue for non-conforming clients has no overflow
behavior. Fatality is fine there (it is a buggy-client path); say so, to
keep the fatal/non-fatal boundary fully enumerated.

### R3-4 (nit). 1A constructor pseudo-code would not run as written

`TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3(cpu_count=1),
wait_for_transcript=False)` — the pinned constructor is keyword-only
(`*, turn_analyzer: BaseTurnAnalyzer`,
`turn_analyzer_user_turn_stop_strategy.py:50-53`), so `turn_analyzer=` must
be spelled. Editorial; the implementation would discover it in seconds.

## Verdict

Revision 3 resolves every outstanding revision-2 finding from both
reviewers, and the resolutions are grounded in real seams: the cancellation
drain matches the actual `AcousticResponseBoundary` machinery, the Pocket
pacing loop it removes is where the plan says it is, the pinned VAD
operating point uses real `VADParams` fields, and the false-positive policy
is now an explicit, testable design decision rather than an implication. One
important gap remains — the held-turn arbitration that revision 3 itself
introduced is unsized and turns ordinary overlapping speech into a fatal
session error — plus three editorial nits.

**VERDICT: APPROVE WITH CHANGES**

Required changes:
1. R3-1 — size the held-turn budget from the worst-case typed/tool epoch,
   specify graceful drop-with-diagnostic degradation (not session-fatality)
   for held-turn capacity overflow, define multiple-held-turn FIFO
   semantics consistently across 1B and Step 3, and start acknowledgement
   timeout clocks at send time.
2. R3-2 through R3-4 are editorial and may land in the same revision
   without further substantive re-review.
