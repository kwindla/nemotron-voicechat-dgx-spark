# Smart Turn and direct-text input plan

Status: revision 7, complete; production Smart Turn and typed-input parity qualified  
Date: 2026-08-10

## Outcome

Move microphone turn ownership to Pipecat VAD + Smart Turn and carry its start
and end decisions to the model server as ordered barriers. Retain the Pocket TTS
typed-input bridge for production user turns: the direct-position experiment
proved that token injection against this frozen checkpoint/interface is not
equivalent to acoustic conditioning. Keep that experiment diagnostic-only and
use it as a qualification harness for a future trained text-to-perception
adapter or a checkpoint with a native multimodal text interface.

The target remains one stateful, full-duplex VoiceChat session. This is not a
stateless chat-completion path, and Pipecat context is not replayed into Nano.

## Recorded architecture decision

The direct-position transaction is structurally sound, but its semantic gate
failed: only 10 of 20 clean-token cases met their predicates. The decisive c014
function-call comparison was then repeated against a frozen voice carrier under
a neutral off/on/off trace ABA. Voice produced the exact `get_weather(Tokyo)`
call in all three runs. At the like-for-like settlement/EOU decision surface,
voice reached raw SOTC argmax twice (rank 1, maximum SOTC-minus-PAD logit
`+0.9375`) and staged/replayed SOTC once. Direct text never selected SOTC
(rank 2, maximum margin `-2.9375`) and emitted no tool call. The trace was neutral
across 861 stable fields with zero violations.

This does not prove that native text input is impossible. It proves that the
current inference-only source-token injection is distribution-shifted for this
checkpoint and cannot be shipped as an acoustic-user-turn replacement. Initial
system-prompt text remains a supported prefill surface and is not evidence for
mid-session user turns. Production typed input therefore remains:
`text -> Pocket TTS -> perception/RNNT -> explicit EOU -> Nano/EarTTS`.

The final typed-versus-speech qualification used canonical ordinary-English
scenario v2 (`2e95857362d4e74c3bb7cd4b2d8edcdfecfc87fdf2bf90a913b078132d700f28`).
It removed clock wording, acronyms, digits, and alphanumeric strings from the
ASR-scored surface without changing any WER, heard-input, semantic-parity, or
margin threshold. The fresh typed-then-speech run passed both the original
strict-margin verdict and `parity_margin_v2`, with no blocking reason or review
hold. Its retained top report SHA-256 is
`6226986167813ed1aa79f0e57cdf372f1e763bb6e7dc147df1db4fd36ac6f371`.

That result qualifies the production typed carrier as no worse than its exact
speech twin. It does not claim that this research model succeeds absolutely on
every requested readback: the later three turns produced the same per-turn
short answer across both arms and the same semantic miss in both arms. The
function-call structure, response audio integrity, stale-result exclusions,
speech-input recognition, and every modality-direction predicate nevertheless
passed. Absolute model capability and typed-versus-speech equivalence remain
deliberately separate claims.

## Why `tokenizer(text) -> Nano` is not already available

The released VoiceChat Nano runtime does not expose a conventional `input_ids`
request. Its published vLLM artifact accepts `combined_embeds` with width 4,480.
For every 80 ms timeline position the wrapper fuses:

- the previous agent-text token embedding;
- the current user-audio perception embedding;
- an optional user-text/ASR embedding (absent in this checkpoint because
  `predict_user_text=false`); and
- the previous function-channel token embedding.

The system prompt demonstrates that tokenizer embeddings can occupy the
user-source slot: `_prepare_system_prompt_embeddings()` puts prompt token
embeddings in the user-audio channel and fuses PAD into the other channels.
That makes native typed input technically plausible, but does not make Nano a
normal text LLM. A production path must preserve timeline state, both previous
output feedback channels, Nano and EarTTS request positions, function state,
response controls, and every live cache. It must also establish that
mid-session user text in this slot behaves acceptably with both released
artifacts; system-prompt prefill is not evidence of that behavior.

## Cross-cutting invariants

1. Pipecat VAD owns microphone turn start and Pipecat Smart Turn owns microphone
   end-of-turn. RNNT remains the transcription/speech-evidence model, but does
   not autonomously endpoint input in this mode.
2. The client sends live WebRTC frames immediately whenever the half-duplex
   input arbiter is available. During a typed/tool atomic epoch it holds the
   complete microphone turn in bounded memory, then releases it without pacing.
   Audio and turn barriers share one ordered path, so a start or commit cannot
   overtake previously observed audio. Neither client nor server adds
   wall-clock sleeps.
3. The server owns exact 80 ms PCM framing, transport-gate/pre-roll replay,
   serialized model steps, session-position admission, and the authoritative
   input-turn state machine.
4. Every source uses explicit turn control before RNNT endpointing is disabled:
   microphone uses wire start/commit barriers, Pocket uses equivalent internal
   barriers, and direct text later uses protocol start plus the common EOU
   primitive.
5. A well-formed commit that reaches the server for the matching open turn is
   always accepted, including zero-RNNT-token turns, and produces one BOS.
   Audio that passes the explicitly
   pinned client VAD is authoritative because Nano may understand perception
   features even when RNNT emits no text. Duplicate start/commit events are
   idempotent; malformed order is session-fatal rather than leaving a closed
   Pipecat aggregator waiting forever.
6. All buffered onset and tail audio is either consumed in order or explicitly
   discarded with a source transition. A partial tail consumes at most one
   padded audio position. At commit, the server advances only the missing
   bounded zero-input positions needed to reach the pinned RNNT ten-blank
   phrase/predictor settlement fence, then runs exactly one explicit EOU
   position. The commit remains the sole boundary decision; settlement is
   serialized model advancement, never wall-clock pacing or endpointing.
7. A forced boundary traverses normal model/protocol response machinery. The
   client may suppress stale deltas after local interruption, but it retains the
   response correlation until the server emits its terminal event.
8. Typed input is serialized with microphone input. Exact application context
   is committed only when typed injection actually starts, never at admission.
9. Every new capability and its behavior flip atomically. A client refuses
   client-turn mode when the advertised capability is absent.
10. Failure after the first direct state mutation is session-fatal unless an
    exact rollback transaction has been implemented and qualified.
11. Published checkpoints and converted weights are immutable. Runtime source
    changes live in the checked-in NVIDIA Speech patch/hook layer and reproduce
    from a clean checkout.

## Step 1 — Pipecat Smart Turn owns microphone turns (complete)

### 1A. Configure the pinned Pipecat stack explicitly

- Construct `VoicechatUserAggregator` with `LLMUserAggregatorParams` containing
  `SileroVADAnalyzer(VADParams(confidence=0.7, min_volume=0.6,
  start_secs=0.2, stop_secs=0.2))` and explicit
  `UserTurnStrategies(start=[VADUserTurnStartStrategy()],
  stop=[TurnAnalyzerUserTurnStopStrategy(
  turn_analyzer=LocalSmartTurnAnalyzerV3(cpu_count=1),
  wait_for_transcript=False)])`, plus `user_turn_stop_timeout=5.0`.
- Treat the explicit 5 s controller timeout as crash recovery behind Smart
  Turn's 3 s silence fallback. Add a test proving it cannot win the ordinary
  stop race.
- Use the bundled CPU Smart Turn model. Measure its host latency/CPU contention
  before deciding whether an affinity setting is needed; do not consume GPU for
  the always-on analyzer merely because GPU is available for qualification.
- Keep `LLMServiceMetadataFrame` and `is_realtime_service=True`, but set its
  `user_turn_strategies` field to `None`. Application-provided strategies are
  authoritative.
- Local VAD emits the sole Pipecat `UserStartedSpeakingFrame`/interruption;
  Smart Turn emits the sole Pipecat `UserStoppedSpeakingFrame`. Server RNNT
  speech events remain source-tagged telemetry and are not translated back into
  duplicate microphone turn frames. Pocket lifecycle remains server-owned until
  it is removed.

### 1B. Define an ordered, capability-gated client-turn protocol

- Add strict-v3 extensions `input_audio_buffer.turn_start` and
  `input_audio_buffer.commit`, correlated by a monotonically increasing
  `client_turn_id` and normal wire `event_id`. Acknowledge with
  `input_audio_buffer.turn_started` and `input_audio_buffer.committed`, echoing
  both correlations. These are additive v3 extensions because capability
  negotiation is mandatory; a client that does not see
  `input_turn_detection: "client_smart_turn_v1"` must not send them.
- Advertise the new capability only when external EOU behavior is active;
  otherwise continue advertising RNNT turn detection and rejecting the new
  events. Rewrite the existing strict-v3 negative contract tests so capability
  and behavior change atomically. Change
  `settings_mutable_until: "first_above_gate_model_frame"` to
  `"first_model_input_frame"`, because start-triggered pre-roll can now be the
  first mutation.
- In Pipecat, allocate `client_turn_id` on local VAD start. Enqueue
  `turn_start`, subsequent audio, and the eventual Smart Turn `commit` into one
  FIFO. Audio that triggered VAD can precede `turn_start`; the server retains it
  in pre-roll and the start barrier releases it. A commit is enqueued only after
  every audio frame already observed by the service.
- Replace the audio-only pacing task with an immediate sender over a bounded
  FIFO. Limit raw queued PCM to 256 KiB (about 8 s at 16 kHz mono PCM16), bound
  serialized JSON/base64 event bytes and control items separately, and never
  materialize an unbounded intermediate WebSocket queue. Wait for capacity for
  at most a short configured send deadline and fail the session on sustained
  overflow. Control barriers are never dropped and audio is never selectively
  dropped around them. Test a prerecorded burst larger than the queue and prove
  bounded process memory, backpressure, and audio-before-commit ordering.
- Client state is `idle -> start_queued -> open -> commit_queued ->
  awaiting_committed -> awaiting_response`. Duplicate acknowledgements are
  harmless. The arbiter keeps one such record per locally completed held turn,
  in FIFO order. Acknowledgement deadlines begin only when the sender actually
  transmits the held barrier, not while it is legitimately deferred. An
  out-of-order/conflict error, post-send acknowledgement timeout, or missing
  response boundary cancels Pipecat metrics and fails the pipeline/session; it
  is not recoverably re-armed after its aggregator has closed.
- On local interruption, stop playback, flush already-delivered sentence text
  into assistant context exactly once, discard its incomplete local sentence
  tail, and mark the response as cancelling. Keep its response ID until the
  matching server `response.done(status="cancelled")` is consumed; do not flush
  again at that terminal. Ignore delivery of intervening stale deltas without
  losing terminal correlation.
- The Pipecat service owns a half-duplex outbound-input arbiter. From typed-job
  request enqueue through `input_text.injection_finished`, microphone silence is
  retained only as bounded rolling pre-roll and a detected microphone turn
  (pre-roll, start, audio, commit) is held intact. Multiple locally completed
  turns may queue within a separate 1 MiB held-input budget (about 32 s of
  actual PCM, comfortably beyond the 20 s tool timeout plus Smart Turn fallback)
  and are released in order immediately after the typed terminal event. Apply
  the same rule while an external function call is awaiting/result-injecting:
  the function result bypasses the held mic FIFO,
  and the server acknowledges `conversation.item.function_call_output.applied`
  only after result injection and post-FC recovery leave function state idle.
  Release the held turn on that correlated acknowledgement. Typed requests that
  arrive during a function cycle are likewise admitted to the bounded local
  arbiter but not sent until that acknowledgement. Bound the acknowledgement to
  the configured 20 s tool timeout plus a 5 s post-FC recovery allowance; a
  missing acknowledgement is fatal.
- Held-input capacity exhaustion is a non-fatal overload outcome, not a wire
  protocol failure. Drop the newly overflowing held turn atomically (never send
  its start, partial audio, or commit), ignore its remaining audio through the
  local commit, cancel its metrics/ack timers, return its per-turn client state
  to a terminal `dropped` state, and emit a surfaced
  `voicechat.input_turn_dropped(reason="arbiter_capacity")` diagnostic. Preserve
  already complete earlier held turns and accept later turns if capacity becomes
  available. The live sender queue still fails the session on sustained
  backpressure because partial transmission cannot be rolled back.
- Record Smart Turn result/reason/probability in diagnostics when exposed by the
  pinned analyzer. Start turn-latency metrics at commit enqueue.

### 1C. Give the server an explicit input-turn state machine

- Add protocol-owned `begin_input_turn(source, correlation)` and
  `commit_input_turn(source, correlation)`. A microphone `turn_start` opens the
  protocol turn even if RNNT later yields zero tokens; RNNT speech confirmation
  enriches that turn and transcript but no longer allocates it.
- In external-turn mode, RNNT evidence generated by triggering audio that
  precedes the asynchronously emitted `turn_start` is accumulated in a bounded
  pending-turn ledger, without public speech/transcript lifecycle events. The
  matching start opens the turn and attaches that evidence; already-consumed
  model audio is never replayed. Each explicit commit snapshots the final
  transcript, resets/fences RNNT display/hypothesis state for the next turn, and
  clears the pending ledger so rapid re-speech cannot inherit stale text. While
  that client turn remains open, promote phrases across the pipeline's
  ten-blank predictor resets into a monotonic turn-local transcript; preserve
  legacy RNNT mode's original per-hypothesis replacement so text never
  accumulates across legacy turns.
- Process all audio and barriers under the existing serialized input path. On a
  new microphone start, if an ordinary agent response is open, call a common
  `cancel_response_to_terminal(reason="client_vad")` operation. Under the model
  lock it latches `request_agent_eos`, runs position-guarded zero-input model
  steps until `AcousticResponseBoundary` emits `end`, suppresses newly produced
  text/audio delivery, and emits exactly one terminal response with cancelled
  status. Bound it by the configured acoustic tail deadline; a position-limit
  or missing terminal is session-fatal. Only then open the new input turn. This
  makes local VAD authoritative even when RNNT misses barge-in.
- Snapshot/freeze pending RNNT evidence before the cancellation drain. Tag its
  zero-input positions as synthetic control so their blanks cannot erase,
  finalize, or create pending user evidence; resume the ledger only for real
  queued audio after `begin_input_turn`.
- A function-call cycle is an atomic exception: client arbitration normally
  withholds the microphone turn until function-result injection and post-FC
  recovery complete. If a buggy/mismatched client sends `turn_start` during an
  open cycle, the server buffers it and following audio/commit in a bounded
  deferred-input queue while continuing to receive the function result, then
  processes it; it never forces agent EOS through live FC state. Capacity
  exhaustion in this defensive server queue is protocol-fatal because a
  negotiated client bypassed its required arbiter and may already have sent a
  partial turn.
- A start barrier force-opens `TransportModelGate`, starts the engine if needed,
  and replays the bounded `GatePrerollBuffer` in FIFO order through normal PCM
  processing. Assert the configured pre-roll covers the VAD-start detection
  window. From start until commit, all input frames bypass the onset threshold;
  commit defensively performs the same release if a start acknowledgement was
  delayed.
- A matching commit is accepted even with no transcript/RNNT tokens. Under
  `model_input_lock`, replay remaining pre-roll and process a zero-padded
  partial PCM tail only when `pending_samples > 0`. Then advance only the
  missing zero-input positions required to reach the pinned ten-consecutive-
  blank RNNT phrase/predictor settlement fence, bounded at twice that fence.
  Enforce the position guard for every replay, tail, and settlement step. Once
  settled, latch `request_user_eou()` and run exactly one guarded zero-audio EOU
  step. Failure to settle, or any position-limit failure, is session-fatal.
  Live isolation is the reason for this fence: zero pre-commit settlement
  produced two empty responses in three turns, five blanks failed in the
  browser, and ten blanks produced three semantically correct turns. The client
  commit is still the only EOU authority; the blank fence cannot independently
  close a turn.
- While external EOU mode is active and the RNNT agent state is still idle,
  retain the function head's raw token for telemetry but force its effective
  token to PAD before the function-call state machine. Otherwise an early SOTC
  can enter asynchronous function calling and freeze the RNNT/model clocks
  before the settlement fence is reachable. If the raw token is SOTC on the
  exact position consuming the explicit EOU, stage exactly that one token;
  never stage an earlier settlement SOTC. The EOU position remains effective
  PAD and forces the one agent BOS. After Nano predicts the first post-BOS
  position, replay the staged SOTC before the ordinary function-call state
  machine and clear the latch exactly once. This preserves Nano's effective-
  token feedback chain while allowing normal post-BOS asynchronous function
  calling. Reset the latch at session boundaries, reject direct injection while
  it is pending, and expose staged/replayed telemetry. The server also fails closed with
  `pre_eou_function_activity` if any function cycle is effective during
  settlement despite this model-side invariant.
- The wrapper's external-EOU mode disables RNNT blank-count, first-turn, and
  transport-energy EOU branches while preserving RNNT transcription/evidence,
  native agent EOS, BOU telemetry, tools, and response watchdogs. The explicit
  user EOU overwrites the appropriate current agent control with BOS and clears
  the same user counters as the old EOU branch.
- Move Pocket to this policy in the same step, before enabling external EOU.
  `input_text.accepted` becomes admission-only; the existing
  `input_text.injection_started` commits exact Pipecat context/transcription and
  issues internal turn start at the first actual model mutation—not merely when
  a sub-frame PCM chunk enters `PcmFrameBuffer`. Under `model_input_lock`,
  recheck cancellation, emit started/begin, and process the first complete or
  final-padded 80 ms frame as one non-cancellable transition. Forward remaining
  PCM immediately through serialized model calls with no pacing sleeps, then
  issue internal commit after the last PCM using the same bounded settlement
  fence and explicit EOU. Remove the older duration/margin-based closure tail.
  Cancellation/replacement is allowed before that transition; afterward the
  job runs to commit and a replacement queues behind it. A
  post-mutation failure or disconnect aborts the session rather than committing
  a partial typed utterance. Pocket events remain tagged `typed` and never
  create local microphone frames.
- Make start and commit idempotent by retaining a small last-completed
  correlation record. A matching retry repeats its acknowledgement without
  model mutation. A mismatched ID, an overlap that bypassed both documented
  arbiters, or another impossible state is protocol-fatal; ordinary typed/tool
  overlap follows the bounded deferral policy. Reset latches, correlations,
  pending RNNT evidence, gate state, and partial buffers at teardown.
- Define `audio(turn N) -> commit N -> turn_start N+1 -> audio` as two FIFO
  turns: commit N creates its BOS/response, start N+1 immediately closes that
  response through agent EOS, then opens the next user turn. No event is
  relabelled based on whichever server turn RNNT happens to infer.

### 1D. Prove and qualify the seam

- Unit tests cover exact Smart Turn/VAD wiring, controller timeout ordering,
  realtime metadata, no duplicate local frames, FIFO capacity/ordering,
  `turn_start`/commit retries, fatal conflicts, acknowledgement timeouts, false
  starts, mute, disconnect during analyzer work, response-correlation cleanup,
  sentence-buffer single flush, and serialized-memory bounds.
- Runtime tests cover gate force-open/pre-roll replay, quiet onset, optional
  partial-tail drain, missing-to-fence settlement, its twice-fence fatal bound,
  exactly one EOU step, zero-RNNT-token commit, position
  limit at every synthetic step, missed-RNNT barge-in, false-positive VAD while
  the agent speaks, cancellation tail silence/deadline completion, and
  `speak -> commit -> immediately speak` event and transcript correctness.
  Cancellation-drain tests also prove synthetic blanks cannot alter the frozen
  pending transcript/evidence ledger.
- Rewrite the previous “commit unsupported” and RNNT-capability tests. Add
  microphone + retained Pocket coexistence, cancellation/replacement before
  Pocket start, non-cancellable Pocket completion after first PCM, speak during
  typed injection, response completion without closure silence, and microphone
  barge-in while a function result is pending and after post-FC recovery. Hold
  two complete microphone turns during one typed/tool epoch and prove ordered
  release and deferred acknowledgement deadlines. Overflow a third/current
  held turn and prove it is dropped atomically with one diagnostic while the
  session and earlier held turns remain usable; separately prove live-send and
  defensive-server queue overflow remain fatal.
- Run pinned Pipecat and pure runtime suites, then the existing multiturn and
  browser qualification ladder in Smart Turn mode. Add six fresh-session
  memory cases spanning `voice|typed|system-prompt` context establishment by
  `voice|typed` recall. The recall prompt must omit the secret, and only the
  recall response—not the seed acknowledgement—is required to contain it.
  This distinguishes system-prefill text support from mid-session typed-turn
  support while covering the full prompting surface. Re-baseline only
  intentional endpoint latency changes.
- Run live GPU cases for complete/incomplete phrases, fallback stop, quiet/short
  speech, ambient-noise rejection, a deliberately induced VAD false positive,
  mute, barge-in with RNNT intentionally not relied upon, rapid
  speak/commit/speak, speak during Pocket typed input, and barge-in during a
  tool call. Capture
  classifier/fallback reason, Smart Turn probability if available, Smart Turn
  p50/p95 CPU latency, commit-to-BOS, cancellation-to-agent-EOS, transcript,
  response lifecycle, queue high-water mark, duplicates, and host CPU load.

Gate: one VAD start opens one server input turn, one Smart Turn commit produces
one BOS, and one local interruption produces a matching cancelled server
terminal. Ambient/AEC noise remains below the pinned VAD policy; a deliberately
induced false positive is explicitly treated as an authoritative audio-native
turn and may yield an empty/odd response rather than silently dropping possible
quiet speech. Onset/tail audio is preserved, held turns survive typed/tool
epochs, incomplete phrases use classifier/fallback as designed, Pocket still
works, and existing multiturn/tool behavior passes.

## Step 2 — Evaluate a direct user-source position transaction (complete; gate failed)

### 2A. Inventory and expose the complete transaction seam

- Before reusing the memory gate for direct-text or changed-weight evidence,
  parameterize its secret per case and add a fresh unseeded negative-control
  session. The negative control must not return any case secret, and each seeded
  recall must return only its own secret in both model text and independent ASR.
- Before injecting text, inventory mutable ownership in
  `StreamingS2SPipeline`, its slot context, the wrapper, Nano and EarTTS vLLM
  requests, perception/RNNT caches, codec/audio buffers, subword/pad-pair state,
  function state, frame index, generated tensors, and request positions. Check
  this inventory into the implementation notes with exact field names.
- Add a diagnostic pipeline-level `inject_user_source_position` seam only if it
  can transact one position across all owners. Perception cache and RNNT
  hypotheses intentionally do not advance. Timeline/generated tensors, Nano,
  first-position feedback, function outputs/state, EarTTS/codec state, subword
  masks, request positions, and frame index do advance exactly once.
- Disable drafted pad-pair optimization for the diagnostic prototype. Capture
  before/after assertions for every inventoried field and both model request
  positions. If no safe seam exists, fail before mutation and stop the direct
  path rather than implementing a wrapper-only partial advance.

### 2B. Establish the token/fusion contract experimentally

- Tokenize bounded text and compare at least plain text and
  `[BOS] + text + [EOS]` source sequences; never mutate the system prompt.
- For the first injected position, fuse the exact previous agent and function
  tokens from live context. At later positions those inputs are PAD because the
  injection transaction deliberately forced the preceding outputs to PAD. Fuse
  the source token through the checkpoint's existing source channel and assert
  BF16 `[1, 1, 4480]` output for `combined_embeds`.
- Each transaction runs Nano, forces effective agent/function outputs to PAD,
  advances EarTTS with PAD, suppresses audio delivery, and atomically commits
  the pipeline context. During every hidden-injection epoch, after each position
  assert Nano and EarTTS request positions advanced by one and preserved their
  epoch offset. That offset is zero before the first assistant BOS. No perception
  or RNNT call occurs. At the later assistant BOS edge, the qualified
  `VOICECHAT_EARTTS_RESET_ON_BOS` runtime intentionally aborts and re-prefills
  EarTTS once: Nano remains in the conversation epoch while EarTTS starts a new
  acoustic epoch at position one. The live gate asserts exactly one reset, the
  new EarTTS baseline, and epoch-relative alignment thereafter; it must not
  mistake that required transition for clock drift.
- Use a checked-in, blinded semantic corpus spanning names/numbers, punctuation,
  reference resolution, interruptions, safety-sensitive intents, and every
  tool/argument schema. Record structured predicates rather than exact wording.
  Mechanically classify whether Pocket+RNNT preserved each case's
  predicate-relevant prompt content. Pocket is a diagnostic reference lane:
  carrier-inconclusive cases are never counted green, and carrier-preserved
  semantic failures remain explicit, but neither can veto clean-token direct
  evidence. Independently require each FP32/W8 direct lane to pass at least 95%
  of intent predicates, 100% of expected tool names and required argument
  values, and every safety-critical case. W8 may have no critical regression
  and at most one non-critical regression relative to FP32.

Gate result: the transaction and trace instrumentation passed, but the semantic
threshold did not. The direct lane reached only 10/20 semantic cases and missed
the c014 tool call. The frozen acoustic positive control passed and established
the SOTC distribution shift described above. Stop the production direct path;
Smart Turn remains independently shippable and a trained adapter or different
checkpoint/interface is required before reconsidering native user text.

### 2C. Preserve the primitive as a diagnostic only

- Do not expose `VoiceChatEngine.inject_text()` through the production server
  or advertise a direct-text capability. Keep the single-position primitive
  reachable only through explicit qualification probe modes.
- Preserve transaction assertions, semantic corpus identity, neutral tracing,
  and frozen voice-carrier comparison so a future adapter can be evaluated
  without weakening the gate or forcing model tokens.
- Diagnostic hidden positions continue advancing Nano/EarTTS recurrence while
  discarding only their intermediate waveform. This is not a serving path and
  does not suppress the eventual assistant response in the probes.

## Step 3 — Retain and harden the acoustic typed-input path (complete)

- Preserve `input_text.request` admission/job semantics and keep Pocket as the
  sole advertised production typed-input mode. Do not add a runtime switch that
  could accidentally activate the failed direct path.
- Continue to run typed jobs through the server-owned half-duplex arbiter with
  bounded IDs/text, deterministic synthesis, cancel-then-replace before model
  mutation, microphone gating, source tags, and terminal dispositions.
- Inject synthesized PCM as fast as the serialized server can consume it. The
  server—not the client—owns exact 80 ms framing, bounded buffering, model-step
  serialization, settlement, and the explicit EOU. No wall-clock pacing is
  required for synthetic typed input.
- `input_text.accepted` means admitted only. `input_text.injection_started`
  commits exact Pipecat context at the first acoustic model mutation. Pocket's
  RNNT text is telemetry and is never committed as a duplicate user message.

Gate: the Pocket path passes typed-only, microphone-only, mixed modality, tool
use, pre-start replacement, maximum input, fatal mid-injection disconnect, and
exact-context tests. Direct remains non-serving diagnostic code.

## Step 4 — End-to-end qualification and publication (complete)

- Cover complete/incomplete Smart Turn phrases, quiet/short speech, barge-in,
  typed-only multi-turn, typed then microphone and reverse, rapid replacement,
  Unicode/limits, every tool continuation, long idle, and disconnect at every
  lifecycle phase.
- Require balanced event brackets, monotonic turn/response IDs, exact-once
  context, no uncorrelated barriers, and no source leakage. Require both clocks
  to advance once and preserve their epoch offset throughout each hidden-input
  transaction; after each assistant BOS, require the qualified one-time EarTTS
  acoustic-epoch reset and epoch-relative alignment rather than absolute
  request-position equality.
- Compare p50/p95 Smart Turn inference, stop-to-BOS, cancellation-to-agent-EOS,
  typed admitted/started/BOS timings, server step time, queue high-water marks,
  underruns, CPU contention, and GPU/CPU memory against captured Pocket/RNNT
  baselines.
- Update protocol, architecture, deployment, limitations, qualification,
  provenance, release checklist, and README. Document Smart Turn as an
  audio-native classifier with silence fallback, Pocket as the production
  typed-user path, and direct injection as a failed-gate diagnostic fused-
  embedding experiment rather than a generic Nano text API.
- Run clean-checkout/bootstrap gates if the NVIDIA Speech patch or quantized
  artifact changes. Never publish changed weights without deterministic
  reproduction and manifest updates.

Gate: all static/component/browser/live suites and the release checklist pass
from a clean checkout; Fable and Codex final reviews have no blocker or
important finding.

Completion evidence: the exact-checkout bootstrap rebuilt and audited immutable
runtime image
`sha256:97f6f9d21b49fdc6c1fbe8c75aae37eca14bc71dd2a3eb52e48442b960c1c6a7`;
718 host tests, 29 subtests, Ruff, the canonical three-turn voice run, the
actual-tool continuation, and the Chromium microphone-plus-typed flow passed.
The immutable Nemotron English evaluator independently classified all retained
no-tools and tool responses as speech with zero silence and WER 0.0. The direct
token-injection lane remains a diagnostic-only failed semantic experiment; the
production text path remains Pocket-to-acoustic conditioning as specified
above.

## Review and implementation cadence

1. Fable and Codex independently review each plan revision until both return
   unconditional approval.
2. Implement one numbered step at a time. After its local gates, request a
   Fable adversarial diff review and perform a parallel Codex review. Fix and
   re-review all blocker/important findings before advancing.
3. Within Step 1, land Pocket's explicit internal commit before switching off
   autonomous RNNT EOU. Capability advertising is the final atomic enablement
   change after all Step 1 pieces are present.
4. Use GPU only for the named live/experimental gates; static and CPU tests run
   first. Preserve unrelated worktree changes.

## Primary references

- Pipecat Smart Turn overview and recommended Silero/strategy wiring:
  <https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview>
- Pipecat user-turn strategies:
  <https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies>
- Pipecat Smart Turn model and runtime:
  <https://github.com/pipecat-ai/smart-turn>
