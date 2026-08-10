# Fable adversarial re-review — Smart Turn and direct-text input plan, revision 2

Reviewed document: `docs/direct-text-input-plan.md` (revision 2, 2026-08-08)
Reviewed against: repo `codex/post-fc-pair-recovery` at `e19f906`; pinned
Pipecat in `.venv` (1.7.1.dev21, commit 5415f28); patched NVIDIA Speech
checkout at `/home/khkramer/src/Speech-nemotron-voicechat`.
Prior review: `docs/reviews/fable-smart-turn-direct-text-plan-review.md`
(verdict: APPROVE WITH CHANGES; blockers B1–B5, should-fixes S1–S7, nits
N1–N5).
Reviewer: Fable (adversarial pass; no implementation code modified).

## Disposition of prior findings

Every prior finding was re-checked against the revision-2 text and, where the
resolution makes new source claims, against the actual code.

| Prior | Status | Where resolved / evidence |
|---|---|---|
| B1 Pocket breaks in Step 1 | **Resolved** | 1C moves Pocket to internal start/commit barriers *before* external EOU is enabled; cadence rule 3 makes the ordering explicit; invariant 4 generalizes it; 1D adds Pocket coexistence, partial-PCM cancellation, and no-closure-silence tests. Committing "after its last PCM" only affects RNNT transcript telemetry lag, not model state — the model's user-audio channel has already consumed every frame, and typed context text is byte-exact from protocol state, so this is sound. |
| B2 VAD interruption desync | **Resolved (new consequences — see F2, F3)** | 1C latches `request_agent_eos` on a new microphone start so the server closes the response through normal machinery; the wrapper already has this exact seam (`_external_agent_eos_requested`, patch lines 383–406, 557–564), so the mechanism is proven. Client side, 1B/invariant 7 keep the response ID until the server terminal event. The desync and dropped-terminal bugs are gone. |
| B3 Gate/pre-roll ignored | **Resolved** | 1C start barrier force-opens `TransportModelGate`, replays `GatePrerollBuffer` FIFO, starts the engine, asserts pre-roll covers the VAD detection window (12 × 80 ms = 0.96 s comfortably covers Silero start latency), bypasses the onset threshold until commit, and has the commit defensively repeat the release. Invariant 6 now covers onset audio. |
| B4 Rejected-commit limbo | **Resolved** | Invariant 5 + the 1B client state machine replace "recoverable rejection" with fail-fast: well-formed commits are always accepted (including zero-token turns), duplicates are idempotent, and conflicts/timeouts fail the session rather than stranding a closed aggregator. Internally consistent and testable. But see F1: one *unavoidable* conflict class remains. |
| B5 Metadata frame loss | **Resolved** | 1A keeps `LLMServiceMetadataFrame` with `is_realtime_service=True` and `user_turn_strategies=None`, exactly as required. |
| S1 Step-1 gate too light | **Resolved** | 1D runs the existing multiturn and browser qualification ladder in Smart Turn mode; re-baselining is restricted to intentional latency changes. |
| S2 Env var unreachable in container | **Resolved** | Step 3 uses a CLI option with env default for local use and a compiled default for the frozen container; production flips are explicit source/rebuild changes. |
| S3 Contract tests / atomic flip | **Resolved** | 1B advertises `client_smart_turn_v1` only when active, mandates negotiation, rewrites the negative contract tests; cadence rule 3 makes capability advertising the final atomic change. |
| S4 Position limits on synthetic steps | **Resolved** | 1C enforces the guard per replay/tail/EOU step with the existing limit error; 1D tests the limit at every synthetic step; 2C rejects over-limit text before mutation. |
| S5 speak→commit→speak ordering | **Resolved** | 1C defines the two-turn FIFO semantics explicitly; 1D tests it. |
| S6 Unbounded fatal window | **Resolved** | 2C caps direct input at 64 source tokens, fails before mutation, sets a p95 started-to-commit target; invariant 10 keeps the fatal/rollback rule honest. |
| S7 CPU budget | **Resolved** | 1A measures host latency/contention before any affinity decision; 1D captures Smart Turn p50/p95 CPU latency and host CPU load. |
| N1 int8 claim | **Resolved** — precision claim dropped; `cpu_count=1` pinned. |
| N2 onset invariant | **Resolved** — invariant 6. |
| N3 FIFO overflow policy | **Resolved** — 256 KiB PCM cap (arithmetic checks out: ~8 s at 32 KB/s), separate control budget, barriers never dropped, fail on sustained overflow, burst test. |
| N4 Pocket reference transcripts | **Resolved** — recorded as qualification fixtures in Step 3. |
| N5 cadence | Retained. |

Additional revision-2 claims verified: `user_turn_stop_timeout` defaults to
5.0 s in the pinned aggregator, so 1A's "controller timeout longer than the
3 s fallback" is real and near-default; the `request_agent_eos` seam has the
`_external_agent_eos_requested` precedent noted above; 2A/2B's
transaction-inventory approach correctly supersedes the earlier
wrapper-only sketch and explicitly disables pad-pair drafting until an
equivalence gate.

## New findings

### F1 (blocker). Microphone `turn_start` during an active typed job is unavoidable, and the plan makes it protocol-fatal

1C: "A mismatched ID, source conflict, **typed/microphone overlap**, or
invalid state is protocol-fatal." But the client cannot prevent this event
class: Silero VAD runs on the live microphone, and users demonstrably speak
while a typed job is injecting — today that audio is silently gated
(`server.py:3077-3080`, `microphone_packet_gated`) and the documented
limitation says a live barge-in may be dropped. Under revision 2, the same
user behavior emits a `turn_start` barrier from the aggregator, and the
server kills the session.

Suppressing the barrier client-side reintroduces exactly the deadlock
invariant 5 was written to prevent (the aggregator has opened a local turn
that will never see a response), and the plan assigns the service no role in
gating aggregator-originated turn frames during typed jobs.

Required change: define non-fatal arbitration for microphone turn starts
that collide with an active typed job, client and server sides together.
Acceptable shapes: (a) the service defers the queued `turn_start` (and its
audio) until the typed job's terminal disposition, preserving today's
half-duplex semantics with a bounded queue; (b) mic-speech-cancels-typed:
the service cancels a not-yet-mutating typed job (Step 2/3 admission phase)
or, once mutation has begun, defers as in (a); or (c) the service drops the
turn locally and force-closes the aggregator turn via the documented
external-stop path, with metrics cancellation. Whichever is chosen, 1D needs
a speak-during-typed-job live case, and the "protocol-fatal overlap" rule
must be narrowed to overlaps that indicate a client bug rather than normal
user behavior.

### F2 (important). `turn_start` during an open function-call cycle strands the tool round-trip

1C's new-start path latches `request_agent_eos` whenever "an agent response
is open." An agent response is also open while a function call is awaiting
its result (`fc_state.active` / `awaiting_response` — `server.py:1328-1337`;
the provider keeps the speech turn open pending `function_call_output`, which
is why `VoicechatAssistantAggregator` exists). Forcing agent EOS in that
window closes the response the function result needs to re-enter, colliding
with the post-function-call recovery machinery this branch spent weeks
stabilizing. The plan rejects *direct injection* during an open function
cycle (2C) but says nothing about microphone starts.

Required change: specify `turn_start` semantics while a function cycle is
open — defer the forced EOS until result injection completes (the natural
choice, mirroring how barge-in behaves today), or explicitly force-fail the
call through the existing `response.function_call.failed` path. Add
barge-in-during-tool-call to the 1D runtime and live test lists (1D currently
tests "a tool call" only on the happy path).

### F3 (important). Every VAD false positive now yields a spurious agent response, by construction

Revision 2 resolves the old desync by making VAD starts authoritative — a
noise-triggered start kills the live response (F2 aside, correctly and
consistently). But the combination of invariant 5 ("a well-formed commit for
the matching open turn is always accepted, including zero-RNNT-token turns")
and 1C ("a matching commit … then latch `request_user_eou()`") means the
noise turn *must* end in a BOS: cough → current answer dies → empty turn →
Smart Turn fallback fires within 3 s → zero-evidence commit → model responds
to nothing. The 1D gate text ("one Smart Turn commit produces one BOS")
bakes the spurious response in as expected behavior, and the "noise false
start" live case has no defined expected outcome.

Required change: give evidence-free turns a defined non-response close. The
server has the evidence to decide: a committed turn with zero RNNT tokens
*and* zero above-gate frames can be closed with `committed` +
`speech_stopped` + an empty `…transcription.completed` and **no** EOU latch,
without violating invariant 5's acceptance guarantee (accepted ≠ must
generate BOS — but the invariant should say so explicitly). If the authors
instead choose to accept spurious responses deliberately, the plan must say
that, pin the Silero confidence/min-volume parameters it is betting on
(only `stop_secs=0.2` is pinned today), and make the noise live case's
expected outcome explicit in the gate. Either resolution also needs the
client story: if no response follows a commit, 1B currently treats a missing
response boundary as session-fatal — an evidence-free close must count as
the turn's terminal event.

### F4 (should-fix). Turn-scoped transcript state must be re-derived for start-barrier-opened turns

Today the protocol turn opens at RNNT `speech_confirmed`, which correlates
with the RNNT display reset (y_sequence and `output_asr_text_str` clear only
after 10 blank frames ≈ 800 ms — `streaming_s2s_pipeline.py:1414-1420`), and
the retention rule in `protocol.py::_update_turn_transcript` leans on that
correlation ("a changed string is the new turn's complete partial…
`decoded_token_count` disambiguates"). Revision 2 opens turns at
`turn_start`, i.e. VAD-fast — potentially inside the 800 ms window after the
previous turn, precisely the `speak → commit → immediately speak` sequence
1D tests for *event ordering*. In that window `decoded_token_count > 0`
still reflects the previous utterance, so the new turn's transcript can seed
itself with stale text and emit it as this turn's first delta.

Required change: 1C should state that `begin_input_turn` resets the
turn-transcript baseline (and ideally clears or fences the RNNT display
state for the new turn), and the 1D rapid-re-speak case must assert
*transcript* correctness, not just event ordering.

### F5 (nit). `settings_mutable_until: "first_above_gate_model_frame"` becomes stale

Start barriers force-open the gate, so the settings lock now engages at the
first replayed pre-roll frame of the first turn, which is earlier and
differently-caused than the capability string describes
(`protocol.py:29`). Update the capability value or its documentation when the
negotiation flip lands.

### F6 (nit). Client "cancelling" state needs a sentence-buffer rule

1B keeps the response ID until the server terminal event while ignoring
stale deltas. The service's new sentence aggregation (`4972c86`) flushes
pending TTS text on interruption today. Specify whether the flush happens at
local interruption (current behavior) or at the deferred terminal event, so
the cancelled response's tail text is neither dropped nor pushed twice.

## Verdict

Revision 2 is a substantially stronger plan: all five prior blockers and all
seven should-fixes are genuinely resolved in the text, the new
`request_agent_eos` and Pocket-barrier designs sit on verified seams, and
Step 2's transaction-inventory framing with objective corpus thresholds is
the right shape. The remaining problems are concentrated where revision 2's
new fail-fast turn machinery meets the two concurrency sources it doesn't
yet model: typed jobs (F1) and open function-call cycles (F2), plus the
deliberate-but-unstated cost of always-BOS commits on false VAD starts (F3).

**VERDICT: APPROVE WITH CHANGES**

Required changes:
1. F1 — replace fatal typed/microphone overlap with a specified non-fatal
   arbitration (defer, cancel-typed, or external-stop), plus a
   speak-during-typed-job test in 1D.
2. F2 — specify `turn_start` behavior during an open function-call cycle and
   add barge-in-during-tool-call tests.
3. F3 — define the evidence-free-turn outcome (non-response close or an
   explicit, parameter-pinned acceptance of spurious responses), align
   invariant 5 and the 1B missing-response rule with it, and give the noise
   live case an expected outcome.
4. F4 — reset turn-transcript state at `begin_input_turn` and assert
   transcript correctness in the rapid-re-speak test.

F5–F6 are editorial and can land in the same revision without re-review of
their substance.
