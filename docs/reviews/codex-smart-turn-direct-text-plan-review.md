# Codex adversarial review — Smart Turn and direct text plan

Date: 2026-08-08  
Plan reviewed: `docs/direct-text-input-plan.md` (initial draft)

## Blockers

### B1 — An end edge alone cannot make barge-in deterministic

The plan sends only a Smart Turn commit. If the user starts while the agent is
speaking and RNNT misses or delays its BOU, the commit arrives while
`agent_speaking=True`. Forcing agent BOS in that state is invalid, while waiting
for RNNT cannot complete once the utterance has ended. The plan also lacks a
server-side fact with which to distinguish a real open local turn from an empty
or duplicate commit.

Add an ordered local-turn-start event driven by Pipecat's VAD
`UserStartedSpeakingFrame`, correlated with the later commit by a client turn
sequence. It must establish commit eligibility and request/drive agent EOS when
a response is open. The later commit can then force user EOU/BOS from an idle
agent state. Test the missed-RNNT-barge-in case explicitly.

### B2 — Disabling autonomous RNNT EOU deadlocks the retained Pocket mode

Step 1 globally disables RNNT blank/transport EOU, while Step 3 initially keeps
Pocket as the default typed mode. The current Pocket job closes solely by
injecting an RNNT-sized silence tail. It will never finish a turn under the new
policy.

Step 1 must move the Pocket path to the same internal explicit EOU primitive
after its final PCM (and on cancellation after partial PCM), removing closure
silence. This must land before autonomous EOU is disabled. It also needs a test
showing that microphone Smart Turn and Pocket typed turns coexist.

### B3 — Typed turns do not have a protocol turn-opening operation

`RealtimeProtocolSession` currently opens a turn only when RNNT reports
`speech_confirmed`. Direct injection deliberately bypasses RNNT. Merely emitting
an exact typed transcript does not allocate a fresh turn ID; response events can
therefore correlate to no turn or to the preceding microphone turn.

Add a protocol-owned `begin_typed_turn(job_id, text)` operation before the first
state-mutating injection step. It must allocate a turn, emit the exact
speech/transcript start as appropriate, and let the common forced BOS finish
that same turn. Acceptance alone cannot open it, because a pre-injection
cancellation is allowed.

### B4 — The direct primitive is underspecified at the hardest state boundary

“Add `inject_user_text_step()` at the seam” is not implementable enough. The
live state is owned jointly by `StreamingS2SPipeline`, its slot context, the
wrapper, two vLLM requests, EarTTS state, perception cache, RNNT hypotheses,
codec state, pad-pair state, function state, and output buffers. A wrapper-only
method can advance some of these while leaving the pipeline context inconsistent.

The plan must first inventory the exact mutable fields and define one
pipeline-level transaction for a position. State that the perception cache and
RNNT hypotheses intentionally do not advance, while timeline/gen tensors,
Nano, function outputs, EarTTS/codec, subword masks, request positions, and
frame index do. Add before/after assertions for every field and fail before
mutation if the required public seam is unavailable. A diagnostic prototype
must prove this transaction before production code is planned around it.

## Important findings

### I1 — The first injected position must preserve both feedback channels

The draft preserves the actual previous agent token but sets the function input
to PAD unconditionally. Normal inference feeds the previous function token as
well. On the first injected position, use the exact previous agent and function
tokens from context; only later positions see the PAD outputs deliberately
forced by injection.

### I2 — Empty partial-buffer flushing is ambiguous

`PcmFrameBuffer.flush()` always returns a zero frame even when there is no
partial data. The plan says both “drain the final partial frame” and then “run
one zero-audio model step,” which can accidentally consume two positions.
Expose pending sample count and process a padded tail only when non-empty; then
consume exactly one explicit EOU step.

### I3 — Smart Turn is not the only configured timeout owner

Pipecat's user-turn controller retains its independent
`user_turn_stop_timeout`. The Smart Turn fallback is 3 seconds and the default
controller watchdog is 5 seconds, so normal behavior is safe, but the plan's
“exactly one owner” invariant should explicitly configure and test that the
watchdog cannot preempt Smart Turn. Treat it as crash recovery, not a competing
endpointing policy.

### I4 — Server events still need separate typed and microphone handling

Suppressing all server speech broadcasts is correct for locally controlled
microphone turns but typed turns currently depend on server-owned lifecycle
events. The plan must name the source-aware behavior during the Pocket
transition and ensure RTVI/assistant aggregation sees exactly one coherent
turn lifecycle for each source.

### I5 — Protocol evolution policy is unstated

Strict v3 explicitly documents commits as unsupported. Adding start, commit,
acknowledgement, and changed capabilities can be an additive v3 extension only
if capability negotiation is the compatibility rule. State that rule and make
clients refuse local-turn mode when the capability is absent; otherwise bump
the protocol version.

### I6 — The semantic gate needs objective failure criteria

“Preserve intent and tool selection” is too subjective to decide whether a
distribution-shifted embedding path can ship. Use a checked-in prompt corpus
with expected intent slots/tool name and argument values, blinded response
ratings or deterministic predicates, minimum pass rate, zero critical tool
misroutes, and explicit FP32/W8 parity bounds. Keep free-form wording out of the
hard equality gate.

### I7 — Direct-input failure timing must be observable before context commit

Pipecat commits the typed message when the server accepts it today. The plan
allows pre-first-position cancellation, but acceptance/context commit can then
leave visible history the model never consumed. Either redefine acceptance as
“admitted and about to mutate,” or add a distinct `input_text.started` barrier
before Pipecat commits context. Document the one unavoidable failure window
after mutation as session-fatal.

### I8 — No-pacing still needs a measurable memory bound

Removing the redundant pacing sleep is right. Specify the maximum Pipecat FIFO
bytes/time, WebSocket `max_queue` assumptions, and the failure/backpressure
behavior for prerecorded bursts. Test a burst larger than one queue and prove
memory remains bounded and the commit stays ordered.

## Suggestions

- Include Smart Turn's model probability and reason (classifier vs silence
  fallback) in the commit diagnostics if Pipecat exposes them through metrics.
- Make start/commit source-specific names rather than overloading server-emitted
  `speech_started`/`speech_stopped` event names.
- Test local VAD false starts that never reach a Smart Turn completion, mute
  while speaking, and disconnect while the analyzer thread is running.
- Treat the pad-pair optimization as disabled for the first direct prototype;
  enable it only after sequential behavior is proven.

## Verdict

**REJECT pending the four blocker corrections.** The overall architecture is
promising and removing client-side pacing is correct, but the initial plan can
deadlock Pocket turns, cannot deterministically handle missed barge-in, lacks a
typed protocol turn, and does not yet define a complete state transaction for
direct positions.

