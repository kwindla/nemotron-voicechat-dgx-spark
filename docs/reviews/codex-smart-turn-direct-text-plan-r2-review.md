# Codex adversarial review — revision 2

Date: 2026-08-08  
Plan: `docs/direct-text-input-plan.md`, revision 2

## Prior findings

All four prior blockers are materially addressed: the plan now has an ordered
VAD-start barrier, migrates Pocket before disabling RNNT EOU, introduces a
protocol-owned typed turn opening, and requires a pipeline-level direct-position
transaction with an explicit state inventory. The prior important findings are
also covered by first-position function feedback, optional partial flushing,
watchdog ordering, source-aware events, capability negotiation, objective
semantic gates, injection-started context commit, and bounded queues.

## Blocker

### B1 — Agent EOS injection does not itself produce `response.done`

Step 1C says to request agent EOS and run the required model step so the normal
response terminal is emitted before opening the new input turn. In the current
runtime, that step forces the text EOS, but `AcousticResponseBoundary` then
enters `tail_draining`; it emits `event="end"` only after additional silent
EarTTS output or `response_tail_max_frames`. Therefore a single step cannot
guarantee the terminal event, and opening the user turn immediately would leave
the old response protocol bracket open.

Required change: define a bounded cancellation-drain operation. Under the model
input lock it requests agent EOS, runs guarded zero-input positions until the
response boundary emits `end`, emits/suppresses output according to the
interruption policy, and fails the session on its explicit maximum or any
position-limit error. Only then may `begin_input_turn` run. Test both immediate
silence completion and tail-deadline completion, and assert one terminal event.

## Important findings

### I1 — Pre-start audio can already have mutated RNNT/protocol state

The pinned universal aggregator pushes an `InputAudioRawFrame` downstream before
its asynchronously queued VAD and user-start frames. If the server transport
gate is already continuous/open, that triggering audio is processed rather than
held in gate pre-roll. RNNT can produce speech-confirmation/transcript evidence
before `turn_start` arrives. The plan says RNNT no longer allocates the turn but
does not specify what happens to that early evidence.

Required change: in external-turn mode, buffer turn-scoped RNNT evidence and
withhold public speech/transcript lifecycle events while no explicit input turn
is open. On matching `turn_start`, open the turn and attach/emit the buffered
evidence. Bound this buffer and clear it on fatal teardown. Do not replay model
audio that has already been consumed.

### I2 — Pocket context timing must change in Step 1, not only for direct mode

The current Pipecat service commits typed context on `input_text.accepted` and
ignores `input_text.injection_started`. Revision 2's invariant correctly says
context commits at injection start, but Step 1's Pocket migration does not name
this client change. A Pocket synthesis cancelled before audio would still leave
a user message that Nano never heard until Step 3 happens to change the handler.

Required change: as part of Step 1, make accepted admission-only for Pocket and
move exact typed context/transcription emission to the existing
`input_text.injection_started` event. Add cancellation-before-start and
replacement-before-start tests.

### I3 — Pocket's wall-clock pacing contradicts the new single pacing policy

The Pocket worker loop currently sleeps between 20 ms chunks. Step 1 removes
the Pipecat sender's clock and the cross-cutting invariant says neither client
nor server adds wall-clock sleeps, but it does not explicitly remove Pocket's
server-side sleep. Move Pocket PCM through the same immediate serialized input
path; model-call serialization/backpressure remains the consumption clock.

### I4 — Queue accounting must include serialized memory

The proposed 256 KiB PCM bound is useful, but each packet becomes base64 JSON
before WebSocket send. The implementation should bound both raw queued PCM and
the maximum serialized event/control count, avoid materializing an unbounded
second queue, and test the actual process-memory high-water mark. This is a
clarification, not a change to the architecture.

## Verdict

**APPROVE WITH CHANGES.** Fix B1 and I1–I4, then re-review. No implementation
should start while the acoustic cancellation boundary remains underspecified.
