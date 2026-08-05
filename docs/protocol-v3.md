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
`instructions: null` restores the checked-in default. Settings lock when the
first above-gate model frame is processed.

## Audio and model-owned turns

Microphone input is JSON/base64 PCM16 mono at 16 kHz. Each
`input_audio_buffer.append` declares its encoding, sample rate, channel count,
and non-empty `audio`. Binary frames, client-created responses, and explicit
audio commits are unsupported: RNNT owns speech boundaries and response
creation.

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
cancel-then-close-then-start. The server—not the browser or Pipecat—is the sole
half-duplex authority: it discards microphone PCM from acceptance until Pocket
TTS audio and EOU-derived closure silence finish. If any synthetic audio began,
closure silence is sent even after cancellation. Typed-turn transcript events
are tagged `source:"typed"` and the original `job_id`; microphone transcripts
use `source:"microphone"`. Pipecat commits the byte-exact request text once and
suppresses the derived typed RNNT transcript from context.

## Function calls, errors, and closure

Function arguments arrive in `response.function_call_arguments.done`. Clients
return the matching `call_id` in `conversation.item.create`; unknown, duplicate,
late, or malformed results yield typed recoverable errors. A bounded tool wait
emits `response.function_call.failed` before restoring turn state.

Recoverable errors have `code`, `message`, `fatal:false`, `event_id`, and an
optional echoed client event ID. Internal model failure and the session-position
guard are fatal. `session.stop` closes permanently with `session.closed` and
WebSocket code 1000. Busy connections use code 1013. The position guard emits a
fatal typed error and a graceful `session.closed` rather than crashing or
hanging.

The qualified runtime disables protocol-level WebSocket pings and ping
timeouts. Long model steps or a deliberately lagging qualification consumer
must not be disconnected by transport keepalive. Pipecat instead monitors the
flow of application events with a bounded liveness timer, while TCP/WebSocket
closure still terminates the session immediately.
