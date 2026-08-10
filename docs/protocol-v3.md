# Voicechat strict-v3 realtime WebSocket protocol

This is the only wire contract supported by the model server. There is no
legacy mode. The server first emits `session.created` with
`{"name":"voicechat.realtime","version":3}` and fixed capabilities. The
client confirms version 3 before sending input:

```json
{
  "type": "session.update",
  "session": {"protocol_version": 3, "instructions": "...", "tools": []}
}
```

`session.updated` echoes the stable session ID and effective settings. Tools
omitted from an update are retained, `tools: []` clears them, and
`instructions: null` restores the checked-in default. The acknowledgement is
also the model-ready barrier: the final prompt and tool registry have been
prefilled before it is sent. Settings lock when the first typed or microphone
input turn begins, even if that turn has not yet produced a model frame.
Pipecat does not admit microphone media until this acknowledgement, its initial
LLM context, and its audio sender are all ready. Capture received during model
startup is discarded rather than replayed as stale audio; if a Smart Turn opens
during startup, that entire local turn is discarded through its matching commit.
The UI should therefore expose microphone readiness only after the bot-ready
barrier. The discard is a bounded, logged race fallback, not a startup buffer.
Production advertises
`input_turn_detection:"client_smart_turn_v1"` and
`server_turn_detection:null`; diagnostic construction can still expose the
legacy RNNT-owned mode.

## Audio and client-owned turns

Microphone input is JSON/base64 PCM16 mono at 16 kHz. Each
`input_audio_buffer.append` declares its encoding, sample rate, channel count,
and non-empty `audio`. Binary frames and client-created responses are
unsupported. Pipecat's VAD opens a turn and Smart Turn chooses its end; RNNT
remains the transcript/evidence source inside the audio model.

The client assigns a positive `client_turn_id` and sends this ordered logical
sequence:

```json
{"type":"input_audio_buffer.turn_start","event_id":"...","client_turn_id":1}
{"type":"input_audio_buffer.append","event_id":"...","encoding":"pcm16","sample_rate":16000,"channels":1,"audio":"..."}
{"type":"input_audio_buffer.commit","event_id":"...","client_turn_id":1}
```

Audio that triggered VAD may arrive before `turn_start`; the server retains it
in bounded onset pre-roll and the start barrier releases it. Start and commit
produce correlated `input_audio_buffer.turn_started` and
`input_audio_buffer.committed` acknowledgements. Matching retries are
idempotent. The 15-second acknowledgement deadline starts when a barrier is
actually written, leaving headroom to consume the bounded eight-second audio
burst ahead of it. Missing acknowledgements, mismatched IDs, and overlapping turns are
fatal because a partially transmitted audio turn cannot be rolled back. A
commit is authoritative even when RNNT decoded no tokens: it requests exactly
one model user-EOU/BOS transition. A committed turn must produce its correlated
`response.created` within 15 seconds; metrics traffic cannot extend that
deadline. A new start first drives an open assistant response to one correlated
`response.done(status:"cancelled")` terminal, and failure to reach the acoustic
terminal within its bounded tail is fatal.

The server emits turn-local transcript deltas and completions, speech start and
stop, response creation, text/audio deltas, function calls, output completion,
and `response.done`. Every event has a unique `event_id`; turn, response, and
tool events carry stable `turn_id`, `response_id`, and `call_id` values as
applicable. A textless or audioless response still receives a complete outer
lifecycle.

## Typed input

Typed input is synthesized inside the model container and injected through the
same RNNT path as microphone PCM:

```json
{"type":"input_text.request","job_id":"client-generated-id","text":"Hello"}
```

The server responds with exactly one accept or reject, then brackets accepted
work with:

- `input_text.accepted`
- `input_text.injection_started`
- `input_text.injection_finished`, whose disposition is `completed`,
  `cancelled`, `replaced`, or `error`

Text is bounded to 1,000 characters. A replacement is serialized as
cancel-then-close-then-start. Pipecat's outbound arbiter holds microphone turns
in a bounded FIFO from typed request through its terminal event. The server
emits `injection_started` under its serialized model lock immediately before
the first complete 80 ms mutation; cancellation is recoverable before that
boundary and a later failure is session-fatal. Pocket audio is forwarded
without wall-clock pacing and ends with the same explicit model EOU used by
microphone commits. Before that EOU, the server advances only the missing
bounded zero-input model positions needed to reach the pinned RNNT phrase-
settlement fence; this is model-state settlement, not a second turn detector or
client pacing. Typed-turn transcript events
are tagged `source:"typed"` and the original `job_id`; microphone transcripts
use `source:"microphone"`. Pipecat commits the byte-exact request text once and
suppresses the derived typed RNNT transcript from context.

During client-owned turn settlement, a function-head prediction cannot become
effective until the explicit EOU has opened the assistant turn. The model
wrapper records any raw pre-EOU function prediction for telemetry and forces
its effective token to PAD before function-call state mutation. An SOTC on the
exact EOU-consuming position is retained in a one-token latch and replayed once
after Nano prediction, before the normal function state machine, on the first
post-BOS position. Settlement-position SOTCs are discarded. If active,
awaiting, injecting, forced-token, or background function state nevertheless
appears before EOU, the server fails the session with fatal
`pre_eou_function_activity`; it never advances a frozen settlement loop or
reinterprets a settlement prediction as a user commit. Staged/replayed counts,
absolute model-frame edges, and pending state are exposed in turn telemetry;
there are deliberately no `this_frame` flags because FC async holds the ordinary
context head at the replay position while its separate model clock advances.
Direct-position
preflight fails closed while the latch is pending.

If both a typed request and a complete microphone turn accumulate during one
function cycle, Pipecat admits the typed request first after the correlated
function-result acknowledgement. The held microphone turn remains intact and
is released after typed injection finishes. This fixed priority keeps the
transition deterministic and preserves both turns.

## Function calls, errors, and closure

Function arguments arrive in `response.function_call_arguments.done`. Clients
return the matching `call_id` in `conversation.item.create`; unknown, duplicate,
late, or malformed results yield typed recoverable errors. A bounded tool wait
emits `response.function_call.failed` before restoring turn state. The server
emits `conversation.item.function_call_output.applied` only after result
injection and post-function recovery are idle; Pipecat holds microphone input
until that correlated acknowledgement. Before unblocking model-side injection,
the server rejects output larger than 16 KiB or 128 tokens. Recovery is bounded
to 256 model ticks, scheduled at a minimum 80 ms cadence (a 20.48-second minimum
scheduled span), inside Pipecat's shared 30-second applied-ack deadline. Slow
model work can extend the wall-clock duration; either bound failing closes the
session rather than releasing partially ordered input. A cycle still open after
the 256th tick fails fatally with `function_output_apply_timeout`. When recovery
completes, that model tick ends at the function-cycle boundary so the server can
acknowledge the applied result. Normal inference resumes on the next tick; a
genuine chained tool call therefore begins as a distinct protocol transaction. A
model-independent circuit breaker permits at most the advertised
`max_function_calls_per_response` (currently eight) calls within one user-turn
response scope. The ninth attempted call is not sent to the client: the server
emits empty text/audio terminals followed by
`response.done{status:"failed",reason:"function_call_loop_limit"}`, a fatal
typed error, and failed `session.closed`. It never fabricates an assistant
answer, so qualification continues to grade the underlying semantic case as
failed. A new user turn starts a fresh call budget. A defensive one-MiB server
FIFO retains microphone events sent by a mismatched
client during the function cycle and replays them afterward; overflowing that
queue is fatal because partial client transmission may already have occurred.

Recoverable errors have `code`, `message`, `fatal:false`, `event_id`, and an
optional echoed client event ID. Internal model failure and the session-position
guard are fatal. A commit whose bounded model settlement cannot reach the RNNT
blank fence fails with fatal `input_turn_settle_timeout`; a partially
model-mutated turn is never retried. `session.stop` closes permanently with
`session.closed` and
WebSocket code 1000. Busy connections use code 1013. The position guard emits a
fatal typed error and a graceful `session.closed` rather than crashing or
hanging.

The qualified runtime disables protocol-level WebSocket pings and ping
timeouts. Long model steps or a deliberately lagging qualification consumer
must not be disconnected by transport keepalive. Pipecat instead monitors the
flow of application events with a bounded liveness timer, while TCP/WebSocket
closure still terminates the session immediately.
