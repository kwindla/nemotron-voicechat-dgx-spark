# Codex adversarial Step 1 implementation review

Date: 2026-08-08

Scope: the uncommitted Step 1 implementation against revision 4 of
`docs/direct-text-input-plan.md`. This pass was performed in parallel with the
Fable implementation review.

## Findings

### I1 — Fixed: the original five-second barrier deadline could expire behind a legal burst

The live FIFO admits roughly eight seconds of PCM and sends it immediately. A
commit written behind that burst can therefore wait more than five seconds for
the single-threaded model consumer even though the connection is healthy. The
deadline is now configurable and pinned to 15 seconds, and begins only after
the barrier is written.

### I2 — Fixed: cancellation control steps could consume or erase pre-start evidence

Audio can precede the VAD start barrier. The first implementation allowed the
agent-EOS drain to consume a partial PCM frame and update the pending RNNT
hypothesis. The server now detaches both gate pre-roll and partial PCM before
the drain, tags drain positions as synthetic, suppresses their response deltas,
and the protocol preserves pending transcript evidence until the explicit turn
opens. A pure protocol regression test covers this race.

### I3 — Fixed: typed rejection and malformed function calls could strand held audio

Typed rejection now removes the admission epoch and releases held input.
Function epochs begin only after arguments validate, carry one correlated
`call_id`, release on either the matching applied acknowledgement or failure,
and have a 25-second result-application deadline.

### I4 — Important: the defensive server function-cycle queue is not implemented

The approved plan requires a bounded server-side queue when a nonconforming or
mismatched client transmits start/audio/commit during a live function cycle.
The current client arbiter prevents this normally, but the server can still
force agent EOS through function state or consume microphone messages at the
wrong point. Add the bounded deferred queue, replay it after function recovery,
and make only its overflow protocol-fatal.

### I5 — Gate pending: Step 1 live GPU qualification has not run

Static and CPU suites are being run, but the named complete/incomplete,
barge-in, Pocket, tool, and rapid re-speech live cases remain required before
Step 1 can be approved.

VERDICT: CHANGES REQUIRED
