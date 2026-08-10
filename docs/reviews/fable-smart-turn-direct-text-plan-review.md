# Fable adversarial review — Smart Turn and direct-text input plan

Reviewed document: `docs/direct-text-input-plan.md` (dated 2026-08-08)
Reviewed against: repo `codex/post-fc-pair-recovery` at `e19f906`; pinned
Pipecat in `.venv` (1.7.1.dev21, commit 5415f28); patched NVIDIA Speech
checkout at `/home/khkramer/src/Speech-nemotron-voicechat`.
Reviewer: Fable (adversarial pass; no implementation code modified).

## Method

Every API, checkpoint, and protocol claim in the plan was checked against the
actual sources. Claims verified correct are listed first so the findings below
can be read as genuine deltas, not general suspicion.

## Claims verified correct

- `LLMUserAggregatorParams` accepts both `vad_analyzer` and
  `user_turn_strategies` (`llm_response_universal.py:116-175`), so the 1A
  wiring point is real. `VADUserTurnStartStrategy`,
  `TurnAnalyzerUserTurnStopStrategy(wait_for_transcript=...)`,
  `UserTurnStrategies`, and `LocalSmartTurnAnalyzerV3` all exist with the
  claimed signatures.
- User-supplied strategies beat service metadata: `Honored only when the user
  did not pass their own user_turn_strategies, which always wins`
  (`llm_response_universal.py:948-960`). The plan's authority claim in 1A is
  accurate.
- Smart Turn v3.2 ships a bundled CPU ONNX model inside the wheel
  (`pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx`), loads it
  without network access, and resamples to 16 kHz via `soxr`
  (`local_smart_turn_v3.py`). `onnxruntime` 1.24.4 and `soxr` are already
  installed in the repo venv. The 3 s fallback is real
  (`base_smart_turn.py:27`, `STOP_SECS = 3`).
- The checkpoint story in "Why tokenizer(text) -> Nano is not already
  available" matches the Speech checkout: `combined_embeds` artifact
  (`inference/vllm/scripts/convert_nemotronllm_checkpoint.py:157`),
  `_prepare_system_prompt_embeddings()` placing prompt token embeddings in the
  user-audio channel with PAD in agent/function channels
  (`nemotron_voicechat_inference_wrapper.py:1948-2030`), fusion module factory
  (`parts/fusion.py`), ASR/user-text channel optional.
- The RNNT EOU branches named in 1C (blank-count, first-turn fallback,
  transport-energy backstop) exist as separable code paths in the checked-in
  patch (`patches/nemotron-voicechat-rnnt-turn-taking.patch`), and a
  forced-BOS latch precedent (`rnnt_state['forced_bos']`) already exists.
- The Pipecat service's sender is indeed a second wall-clock pacer today
  (`llm.py::_audio_send_loop` sleeps per chunk duration), so invariant 2 is a
  real change, correctly described.
- `input_audio_buffer.commit` is currently a documented, tested recoverable
  *unsupported* event (`docs/protocol-v3.md:25`,
  `tests/runtime/test_realtime_websocket_endpoint.py:302-308`), and
  capabilities advertise `server_turn_detection: "rnnt"`
  (`protocol.py:18-36`). The plan's 1B correctly identifies both as claims to
  reverse.

## Blockers

### B1. Step 1 breaks Pocket typed input: RNNT silence EOU is still its only endpoint

Invariant 1 and 1C disable "autonomous RNNT silence endpointing" in
external-EOU mode. But the Pocket typed-input bridge — which Step 3 only
replaces *later* — depends on exactly that mechanism today:
`inject_trailing_silence()` appends `rnnt_eou_frames × margin` of silence so
the RNNT blank counter closes the typed turn
(`server.py:2487-2496`, `VOICECHAT_RNNT_EOU_FRAMES`). With the blank-count
branch disabled globally, a typed job's turn never ends: no BOS, no response,
and the half-duplex arbiter keeps the microphone gated until the job dies.
Between the end of Step 1 and the end of Step 3, typed input is simply broken.

Required change: external-EOU mode must be **source-conditional** — disable
autonomous silence endpointing for microphone-source turns only, or have the
typed job issue its own `request_user_eou()` after trailing-silence
injection. Either way, Step 1D's gate must include a Pocket typed-input live
case (it currently tests only microphone paths, and Step 1 explicitly ships
alone).

### B2. VAD-driven interruption desynchronizes the client from a still-running server response

1A gives the local VAD start strategy "the one Pipecat interruption" while
invariant 1 keeps barge-in (BOU) ownership at RNNT. These two authorities are
not reconciled, and the current service code makes the failure concrete:
`NemotronVoicechatLLMService.process_frame` reacts to `InterruptionFrame` with
`_reset_response_after_interruption()` (`llm.py`), which clears
`_response_id`. Every subsequent `response.output_text.delta` /
`response.output_audio.delta` / `response.done` from the **still-open server
response** is then dropped by `_is_current_response()`.

Today that is safe because the only interruption source is the server itself
(RNNT speech-confirmed barge-in), so the server response is being torn down at
the same moment. Under 1A, Silero VAD fires on *unconfirmed* speech — a cough,
a breath, room noise. RNNT never confirms, the server keeps generating, and
the client silently discards the rest of the answer: the user hears the bot
stop mid-sentence, Nano's context contains an answer that was never delivered,
and the transcript pane truncates. Nothing re-synchronizes until the next
turn.

Required change: the plan must specify the interruption contract explicitly.
Options include: (a) service ignores pipeline-originated `InterruptionFrame`
for response-state purposes and continues to reset response state only on
server `speech_started` (VAD interruption then only ducks local playback —
which still leaves audio/UI divergence to be justified); (b) client forwards
an explicit cancel/barge-in event to the server on VAD start so the server
closes the response through the normal machinery; or (c) interruption remains
RNNT-gated in Step 1 and Smart Turn owns only end-of-turn. Whichever is
chosen, 1D must add a false-positive VAD (noise during agent speech) live
case. The current text ships a latent data-loss bug while claiming invariant 5
("the same transition") is preserved.

### B3. Commit races the transport energy gate and the new pre-roll; quiet turns can produce empty commits the plan cannot handle

The plan never mentions `TransportModelGate` (`server.py:939`) or the
gate pre-roll replay added in `7c0816a`. In client-Smart-Turn mode the server
still refuses to advance the model until 3 consecutive frames reach
−40 dBFS. Silero VAD and Smart Turn will happily detect and commit speech that
never crosses that gate (quiet speaker, distant mic, AGC ramping). At commit
time the model session may not even be started (`session_started=False`,
`engine.start` unrun), RNNT has decoded nothing, and the buffered pre-roll —
plus any older-than-pre-roll gated audio — has never been fed. 1C's
"drain the final partial PCM buffer... run one zero-audio model step" acts on
a model that heard no speech at all: forced BOS on zero user evidence, or a
recoverable "empty commit" rejection.

Neither outcome is specified client-side (see B4), and the interaction is
worse than a corner case: it is precisely the quiet-onset regime the pre-roll
fix was just built for. Required change: define commit semantics against the
gate explicitly. The natural rule is that an accepted commit force-opens the
gate, drains the pre-roll buffer and pending partial frame through the normal
replay path (the `7c0816a` machinery already does this ordering correctly),
starts the engine if needed, and only then applies the EOU latch. The plan
must also state what happens when, even after that, RNNT decoded zero tokens
(commit-with-no-speech): forced BOS anyway, or reject — and the client's
recovery either way.

### B4. No client-side recovery contract for rejected commits

1C requires rejecting "empty, duplicate, post-response, typed-job, or
pre-configuration commits recoverably", but the plan never says what Pipecat
does when its commit is rejected. At that point the aggregator has already
closed the user turn (Smart Turn fired, `UserStoppedSpeakingFrame` emitted,
metrics started). If the server answers with a recoverable error and no
response ever arrives, the conversation deadlocks from the user's point of
view. The Pipecat side needs an explicit state machine: on rejection, re-open
or re-arm the turn, cancel the latency metric, and surface a diagnostic; on
`committed`, keep the correlation until the BOS-bearing response arrives or a
timeout fires. 1D's gate should include at least one rejected-commit live
case. Without this, invariant 3's idempotence guarantee is untestable
end-to-end.

### B5. "Stop advertising ExternalUserTurnStrategies" is underspecified and, taken literally, disables realtime-mode aggregation behavior

`service_metadata_frame()` (`llm.py:145-151`) carries three things: the
service name, `is_realtime_service=True`, and the strategies. The pinned
aggregator uses `is_realtime_service` for realtime-mode mutations — dropping
transcription-driven turn starts and flipping `wait_for_transcript=False`
(`llm_response_universal.py:881-928`) — and for other realtime-mode handling
(deferred handoff flush sizing). If the service stops emitting the metadata
frame entirely (one plausible reading of 1A), realtime mode is silently lost.
The plan must say: keep emitting `LLMServiceMetadataFrame` with
`is_realtime_service=True` and `user_turn_strategies=None`; only the
strategies field is dropped. One sentence, but the difference between a
correct step and a subtle aggregation regression.

## Should-fix

### S1. Step 1's gate is too light for a change that replaces the qualified turn-taking behavior

`docs/known-limitations.md` states plainly that changing the RNNT 1.6 s
endpoint "changes the qualified turn-taking behavior." Step 1 replaces the
microphone endpoint wholesale, yet its gate is one live session with a handful
of cases, while Steps 2–4 carry corpus and qualification gates. The repo
already has a multiturn qualification ladder
(`tests/runtime/test_multiturn_qualification.py`, browser e2e suite). Step 1D
should either run the existing ladder in Smart Turn mode or explicitly state
that qualification re-baselining is deferred to Step 4 — silently narrowing
the gate for the riskiest behavioral change in the plan is exactly the kind of
thing this review exists to catch.

### S2. `VOICECHAT_TYPED_INPUT_MODE` will not reach the production container as an env var

Step 3 introduces "a server setting `VOICECHAT_TYPED_INPUT_MODE=direct|pocket`".
The production container receives only the frozen TOML
`runtime.environment` (equality-checked by `load_config`,
`artifacts.py:29-46`; `cli.py::model_container_command` passes nothing else).
A host-side env var never reaches the server. The plan must follow the same
pattern just established for `--speech-gate-preroll-frames`: a CLI argument
with an env-var default, with the compiled-in default being the operative
production value, and the rollout flip (`pocket` → `direct`) being a source
change, not a config change. As written, the "rollout switch" cannot actually
be switched in a supported deployment without editing frozen config or
rebuilding — say which one is intended.

### S3. Contract reversals need their negative tests flipped, not just docs

1B reverses two explicit, tested contracts: commit-as-unsupported
(`test_commit_and_session_close_are_recoverable_unsupported_events`) and
`server_turn_detection: "rnnt"` capabilities. Step 4 mentions doc updates, but
the plan should name the contract tests to be rewritten in Step 1, and state
that `input_turn_detection: "client_smart_turn"` is advertised *only* when the
feature gate is on — otherwise a gated server still advertises RNNT ownership
while accepting commits, or vice versa. Capabilities and behavior must flip
atomically with the gate.

### S4. The commit's zero-audio step and injected typed tokens must respect the session position limit

1C and 2B both add model steps that are not driven by arriving audio (the
zero-audio commit step; one step per injected token plus EOU). The plan
mentions "session-position guard per token" for `inject_text()` but not for
the commit step, and neither says what the client sees if the limit fires
inside one of these synthetic steps (the `7c0816a` mid-drain limit behavior is
a good template: stop, emit the limit error, close cleanly). Specify it.

### S5. Barge-in during pre-roll replay and commit application is unordered in the plan

Commit handling runs under `model_input_lock` (good), but the plan doesn't say
what happens if new microphone audio (VAD start; user resumed speaking) is
already queued behind the commit in the FIFO when the commit is being applied.
Concretely: audio-commit-audio arrives; the commit forces BOS; the trailing
audio now lands *during the agent response* and RNNT correctly treats it as
barge-in — but the client aggregator considers this a *new* turn while the
server treats it as BOU on the response the commit just started. That is
probably the desired behavior, but the plan should say so and 1D should test
the speak-commit-immediately-speak-again sequence explicitly, because the
turn-sequence number in the commit (1B) will not match what the server thinks
the turn is.

### S6. Direct-injection cancellation window needs a size bound

Step 3 makes any post-first-token cancellation session-fatal (honest, good).
But nothing bounds how long injection takes and therefore how large the
fatality window is. Injection is "not wall-clock paced", so N tokens = N
serialized model steps ≈ N × step-time. A 1,000-char paste can hold the
session hostage for seconds during which any cancel/disconnect kills the
session. Either bound accepted direct-mode text well below the current
`MAX_TYPED_INPUT_CHARS`, or require the fatal-vs-rollback decision (and its
test) before Step 3's gate rather than "unless the wrapper provides and tests
an exact rollback snapshot" — as written, the choice is deferred to exactly
the moment someone is tempted to skip it.

### S7. CPU budget for Silero + Smart Turn is unaccounted

The host already pins the CPU codec and Pocket TTS to disjoint core sets
(frozen affinity config). Step 1 adds Silero VAD and an ONNX transformer
inference (default `cpu_count=1`) on unpinned host cores, on the same machine
running the GPU stack's Python loops. Probably fine — but the plan's Step 4
measures GPU/CPU memory, not CPU contention, and Smart Turn inference latency
lands directly on the new commit-to-BOS metric. Add p50/p95 Smart Turn
inference time on the DGX host to the Step 1D capture list (the plan measures
it only in Step 4) and state whether affinity pinning is required.

## Nits

- N1. 1A: "bundled int8 CPU ONNX model" — the bundled file is
  `smart-turn-v3.2-cpu.onnx`; nothing in the pinned source calls it int8.
  Drop the precision claim or verify it; the review gate should not encode
  unverifiable claims.
- N2. Invariant 4 says "No speech tail is discarded" — with the energy gate
  still present (B3), leading speech *can* be discarded below the gate; after
  B3 is fixed, restate the invariant to cover onset as well as tail.
- N3. 1B "retain only capacity/backpressure limits" — state the intended
  behavior on FIFO overflow. Today's `_audio_queue` overflow is a fatal
  session error (`llm.py::_queue_audio`); presumably that stays, but the
  barrier items must never be dropped while audio is (or the ordering
  invariant silently breaks under pressure).
- N4. Step 2A's semantic-reference corpus runs "against the current Pocket
  bridge as the semantic reference" — Pocket produces *audio* that RNNT
  transcribes, so the reference includes RNNT transcription noise. Fine, but
  record the Pocket-side transcripts in the artifacts so corpus comparisons
  are reproducible.
- N5. The review cadence says "both reviewers repeat until each returns an
  unconditional approval" — this review is therefore not the last; findings
  B1–B5 are expected to produce a plan revision, not implementation.

## Verdict

The architecture is sound, the checkpoint analysis is honest (2A's
stop-without-shipping gate is exactly right), and most API claims survived
hostile checking. But the plan as written breaks typed input for the entire
Step-1-to-Step-3 window (B1), ships a client/server desync on false VAD
triggers (B2), ignores the transport gate and week-old pre-roll machinery it
must compose with (B3), and leaves rejected commits unrecoverable client-side
(B4).

**APPROVE WITH CHANGES** — revise the plan to resolve B1–B5 (S1–S3 strongly
recommended in the same revision), then re-submit for the repeat adversarial
pass the plan itself mandates.

## Addendum 122 — voice-to-voice latency measurement and decomposition (2026-08-10)

Adversarial review of the browser latency probe and the trace
`~/.local/state/nemotron-voicechat/traces/model/34c7f882-…/events.jsonl`,
including the mid-review correction on `client_input_turn_committed`
semantics. Read-only; every number below rederived from the trace.

### Measurement: valid

The probe is one-clock (`performance.now()` in one page), symmetric
(outbound via wrapped `getUserMedia`, inbound at the remote audio element),
RMS-gated at −40 dBFS in 256-sample/48 kHz callbacks (5.3 ms resolution),
and correctly measures last-outbound-speech → first-inbound-speech. Two runs
agree within 42 ms (3434.9 / 3477.3 ms) — the end-to-end number is real and
reproducible. Caveats: `ScriptProcessor` is main-thread (no evidence of
observer effect — the dominant gap is server-side, visible in the server's
own trace); the endpoint is the media element, excluding output-device
latency; inbound gating on first speech correctly skips delivered leading
silence but therefore *includes* it in the measured interval.

### The old post-commit decomposition is rejected as misleading

`client_input_turn_committed` is written only **after** tail flush +
`process_pcm(tail)` + `settle_user_eou` + `request_user_eou` + the BOS
`process_pcm` (server.py ~5138–5158, code-verified). In the trace the event
coincides with BOS frame 64 itself (−0.1 ms). Therefore "commit → first
delivered audio 88.1 ms" measured the instrumentation point, not the client
barrier: the 88.1 ms interval is frame 64→65, and "160 ms leading silence"
undercounts — frames 64/65/66 are all delivered at −120 dBFS, so
**delivered leading silence is 240 ms**, with first audible output at frame
67 (+263.9 ms, −20.1 dBFS). Anchored at the true client commit arrival
(last 320-sample packet, ~4522344.765; later "input packets" are 1280-sample
internal tail frames), the server handler path to first audible speech is
**≈1.03 s**: 8 tail frames (~610 ms) + EOU settle/BOS (~160 ms) + 240 ms
generated silence + first speech frame.

### Corrected attribution of ≈3.45 s

- **≈1.7 s: engine startup at speech onset — the dominant cost.** Speech
  reaches the server at rel 5.677; the engine's first step (frame 0) is at
  rel 7.471 — a 1.715 s dead gap — followed by exactly 12 preroll-replay
  frames (0–11) before live stepping at frame 12. Mic packets queue
  (FIFO/socket) during the gap and drain in bursts; the trace logs packets
  at consumption, not arrival, so the apparent "2.478 s mic gap mid-speech"
  is this same consumption lag, not a transport stall. Pipecat's TTFB
  (2.725 s to first delta) is ≈1.95 s backlog/startup + ≈0.77 s handler
  path — it is **not** model inference time.
- **Per-frame inference is not the bottleneck**: steps alternate ~46/~88 ms
  (~67 ms mean per 80 ms frame) — faster than real time once running.
- **+370 ms** browser speech-end → user-stopped (VAD 200 ms + Smart Turn +
  event path): near budget.
- **≈1.03 s** server handler path after commit arrival (above).
- **≈396 ms** bot-started → browser audible = 240 ms in-band silence +
  ≈156 ms downlink/jitter/playout.

### Verdict

**Measurement APPROVED; presented decomposition REJECTED.** The 390/400 ms
budget (200 VAD + 30 Smart Turn + 160 model) is not comparable to this
pipeline: it omits engine startup + preroll (~1.7 s), tail-frame stepping +
EOU/BOS (~0.77 s), 240 ms generated leading silence, and playout. The
engineering targets, in order of yield: keep the engine hot / start stepping
at speech onset (≈1.7 s), overlap tail stepping with the VAD window and
shorten the serial EOU/BOS path (up to ~0.7 s), and trim generated leading
silence (up to 240 ms). Trace instrumentation should log socket arrival and
consumption separately, and any future latency claim must anchor on client
barrier arrival, not the post-processing `client_input_turn_committed`
event. Nothing here affects the parity qualification, which measured
different properties.

## Addendum 123 — latency instrumentation review (pre-rebuild) (2026-08-10)

Adversarial review of the just-landed instrumentation across `events.py`,
`llm.py`, `server.py`, the browser e2e test, and targeted tests. Read-only.

### Verified correct

1. **True receive anchor.** `commit_received_at = time.monotonic()` is
   captured at the message-type match, before every validation and branch —
   the earliest application-level point. All three new durations
   (`receive_to_eou_settled_ms`, `eou_settled_to_bos_ms`,
   `receive_to_bos_ms`) use it on a single server clock, exactly answering
   A122's rejected post-commit decomposition.
2. **Client queue visibility.** The commit event's diagnostics are stamped
   twice: at enqueue (`client_enqueue_monotonic_s`, queue depth/bytes) and
   at dequeue/send (`client_send_monotonic_s`, after-get depth/bytes). The
   server derives `client_queue_wait_ms` (single client clock — the backlog
   A122 could only infer) and `client_send_to_server_receive_ms`
   (socket/read wait).
3. **Sanitization is fail-closed.** Seven-key allowlist, numeric-only,
   bools excluded, `math.isfinite` filters NaN/inf, non-dict tolerated;
   values are trace-only — never control flow, never echoed to clients.
4. **Startup/preroll traces sit at the measured work.**
   `model_session_start_requested/completed` (elapsed_ms) wraps
   `engine.start` at both call sites; `speech_gate_preroll_started/
   completed` (frames, elapsed_ms) brackets the drain loop — these directly
   measure the ~1.7 s dominant cost A122 identified.
5. **TTFA semantics are the honest ones.** TTFB anchors at commit
   *enqueue* (so it includes client queue wait, matching A122's finding);
   pipecat's `process_ttfa_metrics` extends TTFB by onset-scanned leading
   silence with a bounded buffer — the 240 ms delivered silence becomes a
   first-class metric. Gated on `metrics_enabled`; per-delta call matches
   the documented contract; present in the pinned pipecat commit
   (`5415f28a…`, install predates the change — not hand-patched).
6. **One-clock browser decomposition.** RTVI messages are now captured
   with `performance.now()` (`__voicechatRtviTimedMessages`) on the same
   clock as the audio probe.
7. **Compatibility and default impact.** The commit `diagnostics` field is
   additive: this server tolerates absence (qualification probes
   unchanged), and the deployed qualified server never reads the key
   (tolerant `body.get` handler) — no protocol break in either direction.
   No per-frame trace additions; stdout emission follows the pre-existing
   `SessionTrace.event` pattern; file writes remain gated by the trace
   root; tool-freshness conjuncts are unaffected.
8. **Tests.** `events.py` asserts the schema-1 diagnostics shape and
   barrier correlation; the endpoint test asserts the
   `client_input_turn_commit_received` trace, exact
   `client_queue_wait_ms` arithmetic (250.0), sanitized passthrough, and
   `receive_to_bos_ms >= 0`.

### Findings — none blocking

- **F1 (should-fix before rebuild, cheap):** no hostile-diagnostics kill
  test. Add one commit carrying NaN, Infinity, booleans, strings, nested
  dicts, unknown keys, and a non-dict `diagnostics`, asserting only
  sanitized allowlisted numerics survive and the session continues. The
  filter handles all of these by inspection; the project's own discipline
  wants the red-path proof.
- **F2 (documentation):** `client_send_to_server_receive_ms` subtracts a
  client-process monotonic from a server-container monotonic. Valid on
  this deployment (Linux containers share the kernel CLOCK_MONOTONIC);
  arbitrary-but-finite garbage for a remote client. Document the same-host
  validity assumption where the field is consumed.
- **F3 (nit):** `eou_settled_to_bos_ms` includes the committed-ack
  websocket send, and two separate `time.monotonic()` calls mean the two
  sub-durations don't exactly sum to `receive_to_bos_ms`. Capture one
  `bos_at` timestamp and derive both.

### Verdict

**APPROVE** — the instrumentation measures the right things at the right
places on valid clocks, changes no default behavior or protocol
compatibility, and directly operationalizes every attribution A122 had to
infer. F1 is recommended before the rebuild; F2/F3 are notes.

## Addendum 124 — second-turn steady-state experiment: design review (2026-08-10)

The instrumented run validates A123's placement end-to-end and the numbers
reconcile: 370 (VAD/turn) + 7555.4 (send→receive wait) + 510.9 (receive→EOU)
+ 172.5 (EOU→BOS) + 240 (delivered silence) + ~150 (playout) ≈ 8999 vs
browser 8971.2; TTFA − TTFB = 243.4 ms independently confirms the delivered
leading silence; commit queue wait of 0.114 ms proves the client is not the
bottleneck — every wait is server-side. Note 7877.4 preroll total **contains**
the 6939.9 session start (engine.start runs inside the drain loop), leaving
937.5 ms ≈ 78.1 ms/frame of true 12-frame replay.

### Design review of the same-session second spoken turn

The experiment is the right next step — it cleanly separates per-session
`engine.start` from recurring per-turn cost — with these binding
requirements:

1. **The probe cannot measure turn 2 as written.** `inboundSpeechFirstMs`
   latches once (`=== null` guard) and `latencyMs` is computed once. Either
   reset the probe state between turns from the test, or (better) record
   arrays of speech segments per direction with a close-gap hysteresis of
   ~700–1000 ms (the bot's intra-utterance pauses are shorter; a small gap
   threshold would split one reply into two segments) and pair segments to
   turns using the one-clock RTVI timestamps already captured.
2. **Overlap guard in the fixture.** The fake-capture WAV plays linearly;
   speech B must start only after reply 1 fully ends or the half-duplex gate
   invalidates turn 2. Leave ≥10–12 s between utterances and assert
   `bot-stopped-speaking` (turn 1) precedes speech-B outbound onset on the
   probe clock, plus turn-2 user-transcription content, before trusting the
   turn-2 number.
3. **No double-counting in the decomposition.** The 7555 ms send→receive
   wait and the 6940+937 ms start+preroll are the *same wall-clock window*
   viewed from two sides — the busy handler is why the commit waited. Sum
   the serial path only once; never add the wait to the work that caused it.
4. **Two cold layers, one experiment.** This isolates per-session start vs
   per-turn cost, not per-server-boot warmup vs per-session start (6.9 s
   here vs ~1.7 s in the A122 trace suggests boot-order variance). Record
   the session ordinal since server start alongside the result.
5. **Preregistered predictions**, so either outcome is interpretable:
   if warm stepping (~67 ms/frame) keeps pace with live speech, turn-2
   send→receive collapses to milliseconds and the recurring floor is
   ≈ VAD 370 + preroll replay ~940 + receive→EOU ~510 + EOU→BOS ~170 +
   silence 240 + playout ~150 ≈ **2.4 s** — still 6× the 390/400 ms budget,
   with preroll replay and the serial EOU tail as the engineering targets.
   If send→receive does **not** collapse, steady-state ingest is slower
   than real time and backlog is structural, which changes the fix entirely.
   Turn-1 `speech_gate_preroll_completed.elapsed_ms` must not be quoted as
   replay cost (it contains `engine.start`); turn 2's value is the clean
   number.
6. **Hold everything else fixed**: same session, no restart, unchanged
   inference and tracing config, and report the full in-trace decomposition
   for both turns, at least twice, rather than browser totals alone.

### Verdict

**APPROVE the design with the modifications above** — (1) and (2) are
required for the turn-2 number to exist and be valid; (3)–(5) are required
for the interpretation to be honest. Root-cause interpretation is reserved
until the results arrive.

## Addendum 125 — two-turn results: root-cause adjudication (2026-08-10)

Every reported number rederives exactly from trace `bd98038f…` (session
start 1802.656 ms; single 12-frame preroll, 2832.474 ms including
`engine.start`; commit waits 0.104/0.293 ms; send→receive 2511.209/1936.369
ms; receive→EOU 540.181/421.782 ms; EOU→BOS 211.642/222.050 ms; 108 ordered
320-sample packets = 2160 ms of audio consumed inside the turn-2
send→receive window). A124's design conditions were met: speech-B onset
5084.2 ms after bot-1 stopped, second transcription and second bot-stopped
verified. The DNS-failed replication attempts produced no model session and
are correctly excluded as non-evidence.

### The interpretation is confirmed, with three sharpenings

1. **Preregistered prediction B obtained — backlog is structural** — and
   the mechanism is now quantitative. Backlog is *created* wherever the
   serial loop's consumption pauses (engine.start; response generation) and
   *persists* because the warm catch-up margin is only ~11 ms per 80 ms
   frame (~13.75%). Cross-check that closes the loop: the ~5.08 s idle gap
   drained 2511 → 1936 ms of backlog — an 11.3% observed drain rate against
   the 13.75% theoretical margin. Likewise the turn totals reconcile:
   4021.3 − 3329.0 = 692 ms ≈ backlog Δ575 + EOU Δ118. Nothing about the
   model is slow — 69 ms active per 80 ms frame — the architecture simply
   cannot lose accumulated debt faster than ~1.1×.
2. **A124's predicted recurring ~940 ms preroll replay is retired.** Code
   and trace agree preroll is first-gate-open-only (frames buffer to
   `gate_preroll` only while the transport gate holds the model idle; after
   the first drain every frame advances the model — the same property that
   sustains the backlog). Steady state is better than predicted on preroll
   and worse on backlog.
3. **Cold has two layers, still unquantified separately**: 1.803 s here vs
   6.940 s in the prior run for the same `engine.start` — consistent with
   per-boot warmup (CUDA graph/alloc) dominating the first session. The
   engine-at-config fix relocates this wait; only a server-boot warmup
   removes the 6.9 s outlier from any user's first turn.

### The recommended architecture, adversarially

- **Start engine at session config: yes** — it removes the largest backlog
  *source* (mic audio arriving while `engine.start` blocks). Pair it with
  boot-time warmup per (3).
- **Decoupled bounded ordered ingestion queue: yes, but it is not the fix
  by itself.** Ordered semantics still require consuming the queued audio
  before the commit; decoupling yields true arrival timestamps, bounded
  memory, and visibility. Backlog *elimination* additionally requires a
  catch-up mechanism — batching multiple perception frames per iteration
  when behind, and/or keeping input consumption ≥ real time during
  response generation. Any frame-shedding shortcut (dropping silent
  backlog) alters qualified perception behavior and needs prospective
  qualification under the A90/A118 discipline before it ships.
- **Preserving the 10-blank EOU contract unless prospectively requalified:
  correct**, and the same conservatism must extend to every shortcut above.
- **Speculative blank advancement during the VAD/Smart-Turn window:
  promising and correctly framed as investigate-first.** The trailing
  silence the fence needs is already streaming during the ~370 ms endpoint
  window, so settlement could largely pre-complete — but early settlement
  commits model state before the client confirms the turn end; the
  false-trigger path (user resumes speaking) needs an explicit
  reversibility story before this leaves diagnostics.
- **BOS and leading silence: measure/optimize, with two cautions.**
  EOU→BOS (~220 ms) includes the committed-ack websocket send inside the
  measured span (A123 F3) — reorder is cheap. Trimming the ~250 ms
  delivered leading silence changes the delivered audio stream that the
  qualification gates hash and rederive — prospective requalification
  required, not a quiet optimization.

### Budget outlook

With engine-at-config plus backlog elimination, the measured warm floor is
≈ endpoint 499 + receive→EOU ~420 + BOS ~220 + silence ~250 ≈ **1.4 s**.
Speculative EOU, BOS reordering, and silence trimming — each gated on the
qualification discipline above — plausibly reach ~0.7–0.8 s. The 390/400 ms
budget remains beyond the current serial architecture even fully optimized;
say so rather than promising it.

### Verdict

**Interpretation APPROVED as sharpened; architecture APPROVED as direction**
with the binding caveats: decoupling must be paired with catch-up to
eliminate backlog; boot warmup addresses the 6.9 s outlier that
engine-at-config only relocates; and every behavior-visible optimization
(frame shedding, speculative EOU, silence trimming) passes prospective
qualification before deployment.

## Addendum 126 — latency Step 1 (eager engine start) design review (2026-08-10)

Pre-implementation review against the actual code. The design is the right
smallest change: the client barrier already exists (`_context_ready` gates
Pipecat's audio-send loop on `session.updated`), so starting the engine
before the ack removes the largest backlog source (A125) for every
conforming client with no wire-format change. `engine.start`'s prefill
explicitly consumes no transport/perception frame (`frame_index` stays 0),
so the proposed pre-frame abort criterion (`frame_index == 0` /
`last_model_result is None`) is well-defined. The settings-lock error
message ("must precede the first model frame") becomes *more* accurate
under the new predicate, and the one contract test pinning
`settings_locked` keeps passing since it drives audio first.

### R1 (binding) — the abort path has no existing mechanism; build it explicitly

`VoiceChatEngine` has **no stop/abort API**: `started` is never reset, and
`start()` raises "session already started". Meanwhile each `start()`
increments `stream_id` and `prefill_for_new_stream` primes **two vLLM
request streams** (Nano + EarTTS, including the 37-token speaker prompt)
for that stream id. A pre-frame restart therefore needs an explicit
`abort_unused_session()` that (a) releases or provably supersedes the
prefilled vLLM requests for the abandoned stream id — otherwise every
extra `session.update` leaks primed KV state on the GPU, (b) resets
`started`, and (c) is exercised by a leak canary: N abort/restart cycles
with vLLM request-position baselines and request counts asserted stable.
"Call start again" without this is not implementable against the current
engine.

### R2 (binding) — restart only on material change

Under the proposal, every pre-frame `session.update` costs a full engine
restart (1.8–6.9 s measured). Re-ack without restart when the rendered
final prompt and normalized tools are byte-identical to the running empty
session's; restart only on material change. This bounds cost, removes the
trivial restart-spam DoS, and preserves final-update-wins semantics
exactly.

### R3 — remaining requirements

- **Start-failure path**: eager `engine.start` failure must produce the
  fatal error + `session_closed` + websocket close (the lazy path's
  `model_call` discipline), with a test.
- **Phase telemetry**: `start()` returning trace-only phase durations is
  fine; server must consume it as `phases = engine.start(...) or {}` so
  `FakeEngine`s returning `None` stay valid, and the schema gets a version
  field.
- **Client-timeout audit**: `session.updated` moves from immediate to
  1.8–6.9 s after `session.update`. Pipecat's `_context_ready.wait()` is
  unbounded (fine), but any client/test asserting a fast ack must be
  audited; mocked-engine tests are instant and unaffected.
- **Test list amendments**: add (i) the R1 leak canary, (ii) the R2
  identical-update no-restart case, (iii) typed-input-before-audio under
  eager start (lock must transition on the first *consumed* frame from
  either source), and (iv) a run of the mocked qualification tier proving
  probe flows (session.update → ack → turn barriers) unchanged.

### Race/reset analysis — clean

The websocket message loop is serial, so no audio processing interleaves
with the eager start; conforming clients hold audio until the ack, so no
backlog accumulates during it (the A125 mechanism). Non-conforming clients
that stream audio pre-ack merely recreate today's backlog behavior — the
lazy-start fallback stays as the fail-closed net and a
`session_start_reason` trace field distinguishes eager/lazy/fallback.
Disabled-path neutrality holds: after the first consumed frame, behavior
and the settings lock are byte-identical to today.

### Verdict

**APPROVE the design — implementation gated on R1 and R2.** R1 is the load-
bearing piece: without a real abort-with-release mechanism the multi-update
window either leaks GPU state or must be dropped (single pre-frame update,
reject subsequent ones — an acceptable fallback scope if abort proves
costly). R3 items land with the change. The claimed latency effect
(first-turn send→receive collapses to tail-only for conforming clients) is
the A125 prediction and should be verified by one instrumented run after
implementation.

## Addendum 127 — Step 1 eager-start implementation review (2026-08-10)

Adversarial review of the landed implementation against A126 R1/R2 and the
self-review's empty-abort concern. Focused Ruff clean; contract/live-suite/
function-calling suites 82 green; endpoint+web-server suites 122 + 6
subtests green in 10 of 11 runs (see F2).

### R1 — resolved, verified at the source

The empty abort (`frame_index == 0` → `pipeline.reset_session()`) releases
the prefilled request state for real: upstream
`streaming_s2s_pipeline.reset_session()` calls `_abort_stream_request` for
**every registered stream id**, then resets the bufferer, context manager,
and state pool. The engine-level `abort()` is fail-closed at three depths
(exception → `reset_session` fallback → scalar reset in `finally`) and
idempotent (`started` guard), so the connection-teardown call after a
completed stop is a safe no-op. The synthetic-`is_last` hazard the
self-review flagged is exactly avoided — an empty session resets directly,
so no old-prompt model position can be consumed during reconfiguration —
and the post-frame path retains the ordered codec drain. `start()` is now a
true transaction: phase-tagged timings, and on failure after `open_session`
it resets pipeline and wrapper state and raises
`VoiceChatEngineStartFailure`; the server's handler traces
phase/timings/reason and the exception propagates to the outer fatal path
(`internal_error` + `session_closed(failed)` + close 1011) — test-covered
pre-ack.

### R2 — resolved

`configuration_changed` compares candidate message, normalized tools, and
rendered prompt byte-exactly; identical pre-frame updates re-ack without
restart (test), changed ones abort-then-restart with paired trace events,
and every invalid-update reject path provably precedes any mutation
(dedicated neutrality test). The settings lock now keys on
`last_model_result`, so idle/preroll frames don't lock and any consumed
model frame from either input source does.

### Findings

- **F1 (required before merge): the restart path lacks the activity guard
  the turn_start handler has.** `session.update` with a changed config
  while `active_typed is not None` (typed job admitted, Pocket synthesis in
  flight, no model frame yet) or while a silence-only client turn is open
  passes the `last_model_result` lock and aborts the engine under the
  in-flight work. Worst interleavings: the typed frame hits `process()`'s
  lazy `self.start("")` and the subsequent eager start collides
  ("session already started" → fatal close — fail-closed but wasteful), or
  typed audio synthesized under the old configuration plays into the new
  session. Unreachable by conforming clients (Pipecat locks context first;
  probes never do this), so it does not block live qualification — but the
  fix is the same three-line guard used at turn_start (reject non-fatally
  while typed input or an open client turn is active) plus one test.
- **F2 (required): deflake the model-ready test.** It asserts
  `abort_count`/trace immediately after the websocket context exits while
  the server's `finally` (executor abort + trace close) may still be
  running — observed 1 failure in 11 full-suite runs. Synchronize on
  `session_trace_stopped` (or poll with a deadline) before asserting.
- **F3 (note):** `process()`'s internal empty-prompt lazy start remains as
  the last-resort fallback; it is unreachable through `process_pcm` (which
  routes through `start_model_session`) and correct for configured clients.

### Verdict

**APPROVE for live GPU qualification** — R1 and R2 from A126 are genuinely
resolved and the transaction semantics are fail-closed at every probed
seam. F1 and F2 are required before merge/promotion. The live run should
include at least one changed pre-frame `session.update` so abort/restart is
exercised once against the real pipeline, and should confirm the A125
prediction: first-turn send→receive collapsing to tail-only for conforming
clients.

## Addendum 127b — F1/F2 closure re-review: merge/live verdict (2026-08-10)

All fixes verified in the current code, not from the description.

**F1 closed twice over.** The `session.update` handler now early-rejects
(`settings_locked`, non-fatal) when `last_model_result`, `active_typed`, or
`open_client_turn_id` is set — and, decisively, re-validates all three
**after acquiring `model_input_lock`** (the same lock `process_pcm` holds
per frame), with the entire abort → configure → eager-start → commit
transaction inside the lock and a `settings_locked_after_validation` path
that exits mutation-free. The A127 race is closed at both the fast path and
the interleaving window; lock request/acquired trace events add
observability, and holding the lock across a multi-second start is safe —
conforming clients send no audio pre-ack, nonconforming frames simply queue
on the lock, and `input_text.request` is serialized behind the same message
loop, so no deadlock path exists.

**F2 closed correctly.** The model-ready test now drives a no-audio
`session.stop` and synchronizes on `model_session_empty_stop_completed` —
the exact post-abort trace event — instead of racing the endpoint's
`finally`. Four consecutive combined runs here (125 + 6 subtests each) plus
the reported ten are green.

**Parallel self-review items verified**: `user_text`, `user_text_prefix`,
and `user_text_segment` are cleared in the start reset phase, the start
failure path, and the abort `finally` (plus `reset_user_transcript`) — no
transcript bleed across an empty restart. Pipecat now separates the 15 s
protocol handshake (`session_created`) from a 45 s model-ready wait
(`session_ready`) — ample headroom over the measured 1.8–6.9 s eager start
— with a behavioral timeout-path test; pipecat suite 60 green, protocol
contract + live suite 74 green, Ruff clean.

### Verdict

**APPROVE — merge, and proceed to live GPU qualification.** Carry-over
conditions from A127 for the live run remain: exercise at least one changed
pre-frame `session.update` against the real pipeline, and confirm the A125
prediction that first-turn send→receive collapses to tail-only for
conforming clients.

## Addendum 128 — pre-ready audio drop: design review (2026-08-10)

Live evidence rederived against the actual lifecycle: `connect()` creates
`_audio_send_task` only **after** `_session_ready` resolves, so during the
3314 ms real prefill nothing drains the queue while `_queue_audio` enqueues
WebRTC audio at 50 chunks/s — the 128-slot bounded queue fills in 2.56 s and
`_enqueue_live_input` fails closed before any turn. The diagnosis is
correct and the failure is the queue design working as intended against an
input that cannot yet be consumed.

### The design choice is right

Dropping pre-ready capture at `_queue_audio` is the only option consistent
with A125: enlarging the queue or buffering-then-replaying reconstructs the
ordered startup backlog that Step 1 exists to eliminate (the server would
repay 3.3 s of stale audio at ~1.1×, i.e. nearly never), and
`session.updated` is already the documented model-ready boundary, so audio
before it has no defined consumer. The whole-turn drop rule — a turn whose
`turn_start` predates readiness stays dropped even if readiness arrives
mid-turn — is the important subtlety done correctly: forwarding a turn's
tail without its head would hand the server a truncated utterance and
produce a wrong transcription rather than a clean absence. Non-contiguous
first client_turn_id is safe server-side (ids are tracked by value with no
contiguity requirement).

### Binding requirements on the implementation

1. **Dropped-turn commit must be metrics- and barrier-silent**: skip
   `start_ttfb_metrics`/`start_processing_metrics` (a started TTFB with no
   response pollutes the new latency metrics), register nothing in
   `_turn_barrier_event_ids`, and arm no barrier timeout — the map must not
   leak entries for turns the server never sees.
2. **One bounded summary** (dropped chunks/bytes/turn count) plus a
   trace-visible signal, so a silently-dropped first utterance is
   diagnosable from logs alone.
3. **Fixture/e2e interaction — the sharp edge**: the fake-capture WAV plays
   from browser startup, so if first speech begins inside the ready window
   the sapphire turn is now *silently dropped* and the e2e fails confusingly
   downstream. The live tests must guarantee first speech starts post-ready
   (leading silence with margin over worst-case measured start, 6.9 s, or an
   explicit ready gate before the utterance) — and one test should assert
   the drop behavior deliberately (readiness mid-turn ⇒ whole first turn
   dropped, next turn normal), which the proposed test list already covers.
4. **Byte-exactness after readiness** (proposed) and the >128-chunk
   no-failure test (proposed) — both required.
5. **Document the semantic** in the protocol/plan docs: microphone speech
   during model startup is discarded by design; `session.updated` is the
   first consumable instant. Users of the raw protocol deserve the sentence.
6. Nit: drop before the resample call, not after — no reason to spend CPU
   on audio being discarded.

### Verdict

**APPROVE the design** — implement with requirements 1–5 (6 optional), then
one live browser run re-verifying: no pre-turn overflow, first post-ready
turn byte-exact and normal, and the dropped-turn summary present when
speech deliberately starts early.

## Addendum 128b — pre-ready drop implementation review (2026-08-10)

Verified in the current tree, all A128 requirements plus the self-review
hardening:

- **R1 exact**: the dropped-turn commit returns before event construction,
  `_turn_barrier_event_ids` registration, and both metrics starts —
  metrics- and barrier-silent, no map leak. The dropped turn consumes a
  client_turn_id; the first server-visible id is simply later (safe by
  value-tracking).
- **R2 with correct deferral**: one bounded warning
  (`voicechat.pre_ready_input_dropped` chunks/bytes/turns) that refuses to
  emit while a dropped turn is open and fires on the dropped commit, first
  ready input, context-ready, or disconnect.
- **Drop point**: before downmix/resample (A128 nit adopted), counting raw
  chunks/bytes. Readiness is the strict conjunction
  `session_ready ∧ context_ready ∧ audio_send_task` — stricter than the
  ack alone, and right, since the send task is the actual drain.
- **Self-review hardening holds up**: typed input during a normal *or*
  dropped open turn parks in `_pending_typed_requests` (duplicate detection
  spans the pending set) and flushes at dropped commit, normal commit, and
  function-epoch end, each flush re-checking the guards;
  `_update_session_and_wait` serializes send+ready-wait under
  `_session_update_lock` across all three call sites; context/tool changes
  reject during normal or dropped open turns — closing the
  context-change-during-dropped-turn seam this review had not asked for.
- **R5 done**: protocol-v3.md documents `session.updated` as the
  model-ready barrier, the startup discard, and the whole-turn discard
  through its matching commit, framed as a bounded logged race fallback.
- **Tests**: 6 pre-ready/dropped behavioral tests; service suite 46,
  pipecat 70, endpoint/web/contract 138 + 6 subtests, Ruff clean — all
  rerun here.

**Carry-over for the live run (A128 R3)**: the fake-capture fixtures must
guarantee first speech begins post-ready (leading silence with margin over
the worst measured 6.9 s start, or an explicit ready gate), or the first
utterance is now dropped by design and downstream assertions fail
confusingly.

### Verdict

**APPROVE — merge; proceed to the live browser re-verification** (no
pre-turn overflow; first post-ready turn byte-exact and normal; summary
present when speech deliberately starts early), alongside the A127b live
conditions.

## Addendum 129 — next latency step: suffix replay (A) and fused blank+BOS (B) design review (2026-08-10)

Reviewed against source, the RNNT patch, and two mid-review corrections from
exact live traces. Evidence baseline accepted: warm V2V 1536 ms with
send→receive 8.3 ms (Step 1's queue-lag-zero prediction confirmed); first
turn still pays 1018 ms of synchronous 12-frame preroll replay; the
0/5/10-blank isolation (2/3 empty responses at zero, failure at five,
3/3 correct at ten) stands as the reason the ten-blank fence is untouchable
without prospective requalification.

### A) Onset-suffix preroll replay — APPROVE as finally specified

The design survived its own falsification cycle, which is to its credit:
the first correction (marker impossible — trigger frame never buffered)
would be true at `min_active_frames=1`, but the deployed gate uses
hysteresis 3, and `should_advance` returns False for the first two
above-threshold frames — so they **are** buffered (trace rows 10/11), the
third triggers outside the buffer, and the marker search is well-defined.
Code-verified: `consecutive_active_frames` in `TransportSpeechGate`.
Binding requirements:

1. **Marker predicate from stored stats, not the stored flag**: buffered
   pre-trigger active frames have `input_active=False` (the flag flips only
   at the hysteresis threshold). The marker must be
   `entry.rms_dbfs >= gate.threshold_dbfs`, with the threshold
   single-sourced from the gate — no second tuning knob.
2. **Retention rule**: first active buffered entry, minus 2 context frames,
   plus everything later, plus the live trigger frame (trace: 4 buffered +
   trigger instead of 12 — ~660 ms of the 1018 ms recovered). All buffered
   *active* frames are retained by construction, so the
   `rnnt_bou_frames=3` BOU evidence (two buffered + trigger) is complete.
   The residual acoustic risk is a soft sub-threshold onset longer than
   160 ms before the first active frame; the input-WER heard gate and live
   browser qualification are the preregistered arbiters, and 2-vs-3
   context frames should be decided by that evidence, not argument.
3. **Fail-safe full replay is load-bearing, not an edge case**: the
   client-mode `turn_start` drain can fire with an all-silent buffer
   (client VAD latency), where no marker exists — full replay preserves
   today's behavior there. FIFO order, byte-exactness of retained entries,
   source/job boundaries, and protocol events unchanged; add a trace event
   with retained/skipped counts and the marker index.
4. **This changes what perception hears** → prospective qualification
   (input ASR + live browser) before promotion, per the standing
   discipline.

### B) Fused final-blank+BOS — APPROVE only as the narrowed no-tools fast path

The correction's ordering hazard is decisive against the original form:
the raw function guard and FC state machine run **before** the RNNT
step/turn-taking in the wrapper's position loop, so arming the legacy
`_external_user_eou_requested` early could commit a boundary SOTC before
the tenth blank is proven — exactly the invariant the qualification chain
depends on. The narrowed design is sound: a **separate** latch
(`request_user_eou_at_blank_fence`), armed only for sessions with no
advertised tools, invisible to the FC guard, checked **after** the RNNT
decode updates `blank_count` within the same position — the slot provably
exists, because the post-FC client-BOS block already mutates
`gen_text[b, t]` at exactly that point in the turn-taking function.
Binding requirements:

1. Tool sessions remain **byte-identical** (existing settlement + separate
   EOU), including traces; the no-tools predicate is re-derived after every
   pre-frame empty restart, never cached across reconfiguration.
2. The fence value is single-sourced from
   `nonblank_reset_after_silence` via `user_eou_settlement_blank_frames()`
   and asserted equal in the settlement evidence — no second literal 10.
3. Latch lifecycle: cleared on engine start, abort, mode-disable, and
   session reset (the 127b `user_text` precedent); mutually exclusive with
   `_post_fc_client_bos_requested` and the legacy request flag, asserted
   fatally.
4. Controller invariants preserved: bound of 20 settlement positions with
   the existing fatal path, pre-BOS position validation, terminal
   exactly-one-BOS, pre-EOU function suppression untouched.
5. **Intra-step ordering evidence, not argument**: one instrumented run
   logging (position, blank_count at the latch check, chosen source token)
   proving the fused BOS lands on the position whose decode reached ten —
   attached to the implementation review.
6. Honest scope note: the saving (~150–220 ms, one position plus
   scheduling) applies only to no-tools sessions. If the Playground demo
   session advertises tools, the headline V2V number does not move —
   confirm the demo's tool registry before treating B as a headline win.

### Verdict

**APPROVE A as finally specified and B as the narrowed no-tools fast
path** — both gated on the enumerated requirements, the standing
prospective-qualification discipline (both change perception/turn-taking
behavior and the pinned patch), and implementation review with the exact
tests above. The ten-blank contract itself is not touched by either change.

## Addendum 129b — Step A (onset-suffix replay) implementation review (2026-08-10)

Verified in the current tree against the A129 contract; all suites rerun
here.

- **Marker predicate exact**: `drain_onset_suffix` tests
  `entry.frame_stats["rms_dbfs"] >= threshold_dbfs` — the stored stats, not
  the stored `input_active` flag — and both call sites pass
  `transport_gate.threshold_dbfs`, so the threshold is single-sourced with
  no second knob. `first_active_index is None → start = 0` is the fail-safe
  full replay, and `max(0, first_active - context)` bounds the context
  slice at the buffer head.
- **Every drain is selected**: both drain sites — `transport_gate` (gate
  release) and `client_turn_start` — route through one `select_gate_preroll`
  helper; no raw `drain()` call remains in the serving flow. The trace
  event carries reason, buffered/replayed/skipped counts, marker index,
  context frames, and the threshold — everything needed to audit a live
  selection from the trace alone.
- **Entry preservation**: the selection returns a slice of the same
  `GatePrerollEntry` objects in FIFO order; byte identity, source/job
  boundaries (pre-existing cross-source clearing retained and tested), and
  transport-frame IDs are asserted in tests.
- **Tests mirror the live trace**: the unit case reproduces the exact
  observed shape — eight −120 dBFS frames, soft onset at −62/−45 (correctly
  *below* the −40 threshold, retained only via the context window), actives
  at indices 10/11 → marker 10, retained [8..11], four frames — plus
  all-quiet fail-safe, zero-preroll, FIFO/state behavioral replay through
  the endpoint, typed-acceptance clearing, and position-limit mid-drain.
  Gate-focused: 10 + 3 subtests and 5 endpoint tests green; combined
  endpoint/web suites 128 + 6 subtests green; Ruff clean.
- **Config**: `--speech-gate-onset-context-frames` defaults to 2 with no
  production-config override — the knob the A129 qualification A/B needs;
  the effective value is recorded in every selection trace event.

### Verdict

**APPROVE.** The implementation matches the reviewed contract exactly,
with the marker-from-stored-stats subtlety handled and the fail-safe
load-bearing path preserved. Promotion remains gated on the standing
prospective qualification (input-WER heard gate + live browser) since this
changes what perception hears on first turns — that run also decides
2-vs-3 context frames on evidence.

## Addendum 129c — replay-path parity fix re-review (2026-08-10)

The self-review's finding was real and my 129b missed it: that review
verified both drain sites shared the *selection* helper but did not verify
replay-path parity — the client `turn_start` site was replaying selected
entries through a PCM16 roundtrip, losing per-entry `input_active`, source/
job identity, and the transport snapshot/frame binding that the
transport-gate site preserved. Recorded as a review miss.

The fix is verified in the current tree and is the right shape:

- **One shared `replay_gate_preroll`** used by both sites, processing the
  original `GatePrerollEntry` objects directly:
  `engine.process(entry.frame_f32, input_active=entry.input_active)` — no
  int16 roundtrip — with `set_input_source(entry.source, entry.job_id)` per
  entry, the retained transport snapshot attached to each result, and
  `send_step` bound to the entry's original `transport_frame` with
  `preroll_replay=True`.
- **Lock discipline is fail-closed**: the helper raises unless
  `model_input_lock` is held; both call sites hold it. The `turn_start`
  sequencing is coherent — selection and partial-onset capture under one
  lock acquisition, barrier acks, then `force_open` + replay under the
  lock; the sub-frame partial onset still correctly goes through
  `process_pcm` (it is pending raw PCM, not a preroll entry); mid-replay
  session-start fallback and the position-limit fatal path are preserved.
- **Bounded trace identity**: `speech_gate_preroll_started` now records
  per-entry transport_frames/sources/job_ids/input_active, bounded by the
  12-entry buffer.
- **Tests**: behavioral client-turn marker selection and all-quiet
  full-replay fallback tests present; endpoint/server suites 130 + 6
  subtests green rerun here; Ruff clean.

### Verdict

**APPROVE.** Replay parity now holds across both drain sites with the
original-object contract enforced by construction and by lock assertion.
The 129b promotion gate is unchanged: prospective qualification (input-WER
heard gate + live browser) before promotion, which also adjudicates the
2-vs-3 context-frame choice.

## Addendum 130 — Step A live qualification adjudication and remaining budget (2026-08-10)

Runtime image `sha256:523b0387…`. Every reported number rederives exactly
from trace `6067e401…`: preroll selection buffered 12 / marker 10 /
context 2 / replayed 4 / skipped 8, replay 290.485 ms; turn 1
0.545 / 510.045 / 197.544 ms, turn 2 47.942 / 430.089 / 180.533 ms.
Two details the report understates, both favorable: (a) both
`client_turn_start` drains ran with **empty** buffers — the fail-safe path
exercised, the live selection occurring on the `transport_gate` drain
exactly as the A129 analysis expected; (b) the eager start absorbed a
**4190.9 ms** prefill — the largest cold layer yet observed — entirely
before `session.updated`, exactly the Step-1 design intent; under the old
lazy start this run's first turn would have been ~5.5 s.

### Step A: PROMOTED

First V2V 1461.3 vs 2260.8 ms eager baseline (−799.5 ms; replay 1018 →
290 ms accounts for ~728 of it), warm 1407.5 vs 1536.2, input semantics
green, selection trace exactly per contract, fail-safe exercised.
Context-frames = 2 is adjudicated by this green evidence per A129b; keep 2,
revisit only on a heard-gate regression. No blockers.

### Independently derived warm budget (turn 2, browser clock)

endpoint 377.7 + send→receive 47.9 + ten-blank settle 430.1 + settle→BOS
180.5 + BOS→audible ≈250 (EarTTS leading silence 244.4 + delivery) +
downstream residual 121.3 = **1407.5 ✓ exact**. Of these: the 430–510 ms
fence is pinned by the 0/5/10 isolation evidence and untouchable without
prospective requalification; the 377 ms endpoint is client-side Smart
Turn/VAD scope; the fused-fence win is confined to no-tools sessions and
this session advertises `get_current_utc_time` — correctly excluded from
the headline path.

### Smallest safe next implementation: bounded leading-silence delivery suppression

The largest remaining server-side, fence-neutral, tool-neutral term is the
~244–248 ms of EarTTS-generated leading silence delivered in-band every
turn. Recommend a **delivery-side** suppressor: from BOS, withhold output
frames while decoded RMS stays below the silence boundary (the existing
−90 dBFS convention), up to a hard cap of ~4 frames (320 ms); on the first
audible frame — or at the cap — deliver everything from that point and
never suppress again within the response (no mid-response gaps; fail-safe
passthrough at the cap). Properties: zero model/perception state change,
ten-blank fence and tool machinery untouched, per-response bounded,
expected ~240 ms on every turn (warm ≈1.16 s). Costs to respect: it
changes the delivered audio stream, so the response-audio integrity
artifacts and the EarTTS epoch-relative alignment checks must be
prospectively requalified (A118 discipline — new runtime image, fresh
run), and the protocol doc gains one sentence (output deltas begin at
first audible frame). Pair it with the cheap A123-F3 item: overlap the
`committed_ack` send with the BOS executor step (~30–60 ms). Together the
warm floor approaches ~1.1 s; anything lower requires the fence, the
client endpoint window, or speculative settlement — each previously
adjudicated as not-smallest or not-safe.

### Verdict

**APPROVE — Step A promoted on this evidence; no blockers.** Next
implementation recommendation as above, gated on its own prospective
qualification run.

## Addendum 130a — correction: delivery suppression recovers ~0 ms; revised recommendation (2026-08-10)

**Retraction.** A130's claim that bounded leading-silence delivery
suppression would recover ~240 ms was false, and the correction's mechanism
is right: the three leading silent positions are *generation* time — serial
EarTTS steps that must exist autoregressively before the audible frame can
be computed — not delivery artifacts. The silent frames arrive
generation-paced (~85 ms apart), so no playout buffer accumulates ahead of
the audible frame; suppressing their delivery changes when
`bot-started-speaking` fires and what sits in the jitter buffer, while the
first above-threshold PCM — the browser metric's anchor — arrives at the
same wall time. Expected V2V effect ≈ 0. Delivery suppression is demoted to
a metrics/UX nicety and is not recommended as next work.

**No safe pre-generation position removal exists for tool sessions.** The
three EarTTS lead positions are autoregressively required; the fused
blank+BOS fence remains no-tools-only (A129's FC-ordering hazard); and
speculative settlement was already adjudicated unsafe. The honest statement
is that for tool-advertising sessions no position can be safely removed
from the post-commit chain.

**Revised recommendation — compress position time, not position count.**
Rederived from the same qualification trace (`6067e401…`): the 652 model
steps are strongly bimodal — 276 fast (mean **45.6 ms**) vs 376 slow (mean
**89.2 ms**) — in a near-perfect slow/fast alternation
(`SFSFSF…`), pointing at a structural every-other-position cost (plausibly
codec decode windowing or a perception-side stage). The post-commit serial
chain is ~10 positions (≈6 settle + 1 BOS + 3 EarTTS lead); closing the
slow positions toward the fast ones is worth ≈**240–430 ms per turn**,
compresses the ten-blank fence's *wall time* without touching its *count*,
and is fence-, tool-, and conditioning-neutral. Smallest next step, in
order: (1) a **measurement-only run** attributing the slow-position cost
with the existing A108/A109 heartbeat stages
(nano/eartts/codec_decode/codec_cpu_copy/rnnt) — no code changes; (2) a
targeted overlap/pipelining of the attributed stage under the existing
runtime-optimizations qualification gates; (3) alongside, the cheap
A123-F3 ack-overlap (~30–60 ms). Warm outlook if the asymmetry closes:
≈1.0–1.15 s.

### Verdict

A130's Step-A promotion and budget derivation stand; its next-step
recommendation is **withdrawn** and replaced by the
attribution-then-pipelining path above.

## Addendum 131 — Step B (no-tools fused blank-fence/BOS) implementation review (2026-08-10)

Adversarial review against the A129(B) contract; everything below verified
in the current tree, with the full CPU suite rerun here (675 passed /
15 skipped / 10 subtests excluding e2e collection; Ruff clean) and patch
parity against the live Speech checkout re-derived byte-exact (filtered
diff empty).

### Requirements, point by point

1. **Separate, guard-invisible latch**: `_external_user_eou_blank_fence_target`
   is distinct from the legacy request and post-FC flags, lives only inside
   the RNNT turn-taking function (after the FC machinery has run its
   position), and carries three fatal mutual-exclusion raises (legacy
   overlap, post-FC overlap, agent-open) plus the mode guard.
2. **Post-RNNT, exact fence**: the check reads the just-updated
   `blank_count`; `>= target` immediately followed by a `!= target` raise
   makes the fence *exact* with an overshoot detector — unit increments
   guarantee landing on it, and any logic drift dies fatally.
3. **One BOS, one EarTTS reset — proven per position**:
   `observe_fused_terminal` is a 13-conjunct terminal contract: fence
   proven exactly, frame identity, chosen token == `agent_bos` control id,
   `agent_control == "agent_bos"`, response boundary started/responding,
   deliverable-and-silent audio at the BOS position, assistant and function
   text unmutated, no effective function cycle or non-PAD function token
   (pre-EOU suppression intact), and EarTTS reset count exactly
   starting + 1. Any violation raises `fused_eou_contract_failure` with
   full evidence. This also satisfies A129 §B.5: the intra-step ordering
   evidence (`frame`, `blank_count_at_check`, `target`, `chosen_token_id`)
   is recorded in-band on every fused turn, not in a one-off run.
4. **Ack before response, by construction**: `acknowledge_fused_terminal`
   executes via the settlement's `before_send` hook, so the committed ack
   is sent before the fused BOS position's own `response.created` — while
   the non-fused path keeps the legacy ack-after-settlement ordering.
5. **Fail-closed bounded cleanup**: just-in-time arming (only at
   target − 1) bounds latch exposure to the proving position; the
   `finally` cancels an armed-but-unfused latch; the wrapper clears both
   fence fields at init and every mode transition; the existing settlement
   bound and fatal paths are unchanged.
6. **Shared controller**: typed and microphone commits both settle through
   the one `settle_user_eou(fuse_final_bos=not session_tools, …)`.
7. **Tool sessions byte-identical**: `fuse_final_bos` is re-derived per
   commit from the current `session_tools` (correct across pre-frame
   eager restarts); with any advertised tool the legacy settlement +
   separate `request_user_eou()` + dedicated BOS position + legacy ack
   ordering run unchanged. The fused trace field appears only on fused
   turns.

### Live-qualification conditions (not blockers)

The Playground session advertises a tool, so the fused path gets **zero
live coverage** from the standard demo run: the live qualification must
include one no-tools session exercising a fused turn (fence marker in the
trace, timing delta vs the ~610 ms legacy receive→BOS), alongside the
standing new-image requalification.

### Verdict

**APPROVE — no blockers.** The implementation is stricter than the
reviewed design at every point where it differs: exact-fence overshoot
detection, a 13-conjunct terminal contract instead of spot checks, in-band
ordering evidence on every fused turn, and single-position latch exposure.

## Addendum 131a — fused-EOU transcript duplication fix review (2026-08-10)

The live no-tools qualification on image `decee053…` proved the fused
contract on every turn (exact blank-10, same-frame BOS and EarTTS reset)
and surfaced one real defect outside the A131 contract's scope: the
wrapper's fence-BOS clears `y_sequence`, but the streaming display string
for that same position still carries the pre-reset phrase, and the
segment-promotion logic then re-adopted it — duplicating the full phrase
in the UI transcript.

The fix is verified in the tree and is correctly scoped:

- **Same-position binding**: suppression triggers only when the wrapper's
  fence marker frame equals the engine's current `frame_index` — later
  positions are untouched, and tool sessions never produce the marker, so
  legacy and tool behavior are byte-identical (dedicated legacy
  non-accumulation test alongside).
- **Fail-closed invariant, not a workaround**: if the marker is present
  but `y_sequence` is *not* empty, the engine raises ("fused client EOU
  marker did not clear the RNNT hypothesis") rather than guessing — the
  suppression path only ever runs on the proven post-clear state. The
  condition is unreachable in normal operation (a nonblank decode at the
  fence position would have reset the blank count and prevented the fence
  from firing), which is exactly what a guard should be.
- **One-time promotion**: at the marker position the standing segment
  promotes into the prefix once and only the marker-bound stale display is
  ignored; the next hypothesis extends normally. The unit test replicates
  the live failure shape exactly (phrase at count 8, same phrase re-offered
  at count 0 with suppression → single phrase, prefix promoted, segment
  empty).
- Focused suites 136 + 10 subtests green rerun here; Ruff clean.

### Verdict

**APPROVE.** A minimal, marker-bound, fail-closed display fix that leaves
the fused acoustic contract, tool sessions, and legacy endpointing
untouched. The live evidence stands: the fused path is proven on-image
with the transcript defect now closed.

## Addendum 131b — Step B live promotion adjudication (2026-08-10)

Every claim rederived from the artifacts, none taken on faith. The
`latency-stepb-no-tools-r4/events.jsonl` is the client wire capture; the
fence evidence lives in server trace `de68dc67…`, where I rederived:
**exactly three `client_eou_blank_fence` markers, one per response, at
frames 53/133/209, each blank 10 == target 10, chosen token 1 (BOS),
`agent_control == "agent_bos"` at the same frame**; all three settlements
`fused_terminal_bos=true, fence_reached=true` (5/5/4 steps);
`session_tools == []` with `model_ready=true`. The timing signature is the
design's own fingerprint: **`eou_settled_to_bos_ms = 0.0` on every turn** —
the separate BOS position is provably gone — with receive→BOS
505.8/515.2/372.6 ms against the ~610–707 ms legacy shape.

Semantics: input WERs 0.2/0/0.25 (codeword-tokenization noise), displayed
transcripts single-phrase — the 131a duplication is absent live; model
texts all correct; and the strongest item: **cross-turn recall survived
two fused BOS/reset positions** ("The code word I asked you to remember is
sapphire"), transcribed by the promoted immutable evaluator
(`sha256:59854214…`) at **WER 0.0 on all three retained WAVs**, all
speech-classified, semantics true, aggregate passed.

Identity chain, closed: image `sha256:b289c9bf…` is present and labeled
with payload `bc60cba3…`, recipe `aa041a84…`, and combined source
`10e1ba8d…` — all three **equal to this tree's independent recomputation**;
the quoted health value `606f53cd…` resolves to the `server.py` file hash,
also matching; the canonical patch hashes to `d5a6cd39…` as claimed.

### Verdict

**APPROVE — Step B is promoted on this evidence; no blocker found.** One
condition before `b289c9bf…` replaces the deployed production candidate:
run one standard tool-session browser E2E (or parity smoke) on this image.
The tool path's *sequence* is byte-identical by construction, but the
shared settlement controller changed shape (`fuse_final_bos=False` now
routes through the new `observe_settlement` wrapper), and the standing
discipline requires the tool path's live evidence to come from the image
that ships, not from its predecessor.

## Addendum 131c — final image promotion confirmed (2026-08-10)

The A131b condition is satisfied: the tool-advertised browser E2E
(`test_live_playground_typed_turn_and_audio`) passed on the exact
`sha256:b289c9bf…` image — verified here as the running container, health
ready — exercising two voice turns under the advertised
`get_current_utc_time` configuration plus typed math and typed memory
recall, with all voice/audio/text assertions green. The tool path's live
evidence now comes from the image that ships.

Metric coherence checks out: TTFA − TTFB is 248.4 ms cold and 244.5 ms
warm — the invariant EarTTS leading silence, matching every prior trace —
and warm V2V 1451.3 ms on a tool session sits where expected (tool
sessions retain legacy settlement by design; the fused saving applies to
no-tools sessions, measured in A131b at receive→BOS 372–515 ms with
`eou→bos = 0`).

### Verdict

**CONFIRMED — image `sha256:b289c9bf…` is promoted.** The full chain on
this image: Step A onset-suffix replay (A130), Step B no-tools fused
blank-fence/BOS with the transcript fix (A131/131a/131b), eager start and
pre-ready drop (A127b/128b), tool-session live coverage (this addendum),
and the closed identity chain (labels ≡ tree recomputation, patch
`d5a6cd39…`, server source `606f53cd…`). No open conditions remain in the
latency workstream's promotion chain.

## Addendum 132 — C1 stage-timing observability design review (2026-08-10)

This is the A130a measurement step done in the right order — attribution
before optimization — and the design's core choices are correct: reuse of
existing synchronization points (zero added CUDA syncs, hence zero
performance footprint to distort the very thing being measured), a
versioned scalar-only schema with a derived `unaccounted` residual,
frame-matched exposure (the fence-marker freshness pattern), per-session
reset, and a nested-but-separate timer for the EarTTS BOS reset so
reset-vs-infer attribution falls out arithmetically from an intact outer
total. Client compatibility is verified: the Pipecat service explicitly
ignores `voicechat.metrics` content, so the additive field is safe by
construction, and the structural qualification conjuncts count event
types, not metrics schema.

### Smallest corrections (binding)

1. **State the measurement semantics in the schema itself.** Without new
   syncs, each stage value is a *host-observed interval between
   pre-existing synchronization boundaries* — GPU work queued in one stage
   but drained at the next stage's sync point is attributed to the latter.
   That is acceptable precisely because the pipeline is serialized at
   those boundaries and the A130a bimodality was itself host-observed —
   but the schema doc must say so, and `unaccounted` must be **emitted**
   as `total − Σ(stages)` in one place, never recomputed by consumers.
   If the analysis finds the bimodality hiding in `unaccounted`, the
   answer is a follow-up design, not ad-hoc syncs added to this one.
2. **Enumerate the reset sites.** "Reset per session" must mean the same
   lifecycle sites as the fence latch and user-text fields (wrapper init,
   mode enable/disable transitions, engine start/failure/abort) — the
   131/127b precedent; a stale timing dict surviving an eager empty
   restart would attribute the old session's last frame to the new one.
3. **Byte-neutrality test, not just "no functional changes."** One mocked
   step asserting `StepResult`/`turn_state` minus the new keys is
   byte-identical to the pre-change output — the established
   disabled-path-neutrality pattern.
4. **Preregister the analysis, don't post-hoc it.** The probe analysis
   must (a) bin stage timings by the A130a fast/slow clusters
   (45.6 / 89.2 ms) and test the codec-windowing hypothesis explicitly —
   if the alternation is real, exactly one stage should carry it; (b)
   report distributions per position class (settlement zeros, fused/legacy
   BOS, post-BOS lead positions, steady speech) for both the no-tools and
   tool probes; (c) commit that the optimization choice follows the
   attribution, with the ranking stated before the run.
5. **Reset-timer scope**: measure strictly around the existing abort and
   speaker-prefill calls inside `install_eartts_reset_on_bos` — no
   reordering — and record count plus request identity so the timing
   evidence composes with the settlement contract's exactly-one-reset
   conjunct.

### Verdict

**APPROVE the C1 design with corrections 1–5** — all five are small and
none changes the architecture. The instrumented rebuild is a diagnostic
image: it needs no promotion of its own, and whatever optimization follows
it re-enters the standing qualification chain (new image, patch-parity,
probes, browser E2E) as usual.

## Addendum 132a — C1 implementation review (2026-08-10)

Verified in the actual diffs; suites rerun here (focused 125 + 15 skipped +
15 subtests, full CPU 682, Ruff clean); canonical patch `896c6b3d…` is
byte-exact against the live Speech checkout.

**Zero added synchronization, proven**: the only `+synchronize` line in the
patch is a comment; every stage clock is a `time.time()→perf_counter` swap
or a new counter at a pre-existing sync boundary, and the wrapper total
closes at the pre-existing `torch.cuda.synchronize()`.

**A132 corrections, point by point**: (1) the measurement semantics live
*in the schema* (`measurement: host_intervals_between_existing_cuda_syncs`,
`codec_pipeline: waits_for_previous_frame_and_submits_current`) and the
server validator pins both strings; the residual is emitted once, fatal
below −0.05 ms, negative-zero clamped. (2) Lifecycle: the timing dict
lives on the per-stream context (dies with the session), the producer
clears `_reset_on_bos_last_timing_ms` at the top of **every** call and
assigns it only after the wrapped call returns — a failing BOS call or
non-BOS call cannot leak the previous reset's timing, stronger than the
requirement. (5) The reset timer wraps strictly the existing abort and
speaker-prefill calls with no reordering, records count and request
identity, and a cross-request timing raises fatally.

**The server validator is the strongest link**: closed key-set, pinned
schema/semantic strings, integer and finite/nonnegative checks, and three
fatal arithmetic cross-checks — named-sum consistency, residual-total
consistency, and reset-nesting consistency (`eartts_total ==
reset_total + non_reset_remainder`) — all at 0.05 ms tolerance. An
internally inconsistent timing dict cannot reach `turn_state`. The
freshness rule (`frame_end == frame_position` else silently omit) is the
right choice for multi-position chunks, with range-consistency violations
fatal rather than silent.

**Carried condition**: A132 correction 4 — the preregistered analysis
protocol (bin by the 45.6/89.2 ms clusters, codec-windowing hypothesis
first, per-position-class distributions for both probes, optimization
ranking committed before the run) — applies to the diagnostic runs that
follow this rebuild, not to this diff.

### Verdict

**APPROVE — proceed to the diagnostic rebuild and probe runs.** No
blockers; the implementation is stricter than the reviewed design at the
validator and leak-prevention seams.

## Addendum 132b — multi-position remainder correction verdict (2026-08-10)

The concurrent correction was necessary and is verified in the current tree
(canonical patch `128e41a2…`, byte-exact against the live Speech checkout).
Recording the review miss first: A132a praised the server validator's
reset-nesting cross-check but did not trace that the producer's remainder
was derived from the single BOS *call* while `eartts_total_ms` accumulates
**all positions in the chunk** — on any multi-position BOS chunk that very
validator would have raised `"EarTTS reset timing exceeds the enclosing
call"` at serving time. Fail-closed, but a production crash; the
self-review caught it before the rebuild.

The fix is the right shape: the in-loop BOS block now captures only the
reset facts (request-bound with the fatal cross-request check retained,
count, abort/prefill/total), and `eartts_non_reset_remainder_ms` is derived
**after the position loop** from the accumulated `eartts_total_ms −
reset_total_ms`, with the same −0.05 ms fatal bound and negative-zero
clamp. The server's nesting arithmetic is now satisfied by construction for
single- and multi-position chunks alike, and the remainder is truthful:
"all EarTTS positions in this chunk minus the one reset."

Suites: full CPU 682 + 15 skipped + 15 subtests green; the focused pair
green in 9 of 10 runs — one unreproduced intermittent failure in the first
run (10 clean repeats after; the flake did not recur and no test name was
captured). Ruff clean; patch parity exact.

### Verdict

**APPROVE — the corrected tree at patch `128e41a2…` proceeds to the
diagnostic rebuild.** If the focused-pair intermittent reappears in CI or
during the rebuild, capture the test name and deflake before relying on
that suite as a gate (the A127-F2 synchronization pattern is the likely
fix); it does not block the diagnostic image.

## Addendum 132c — C1 final verdict (2026-08-10)

Final correction verified in the tree at canonical patch `232195f6…`
(byte-exact against the live Speech checkout; focused 125 + 15 subtests and
full CPU 682 green here; Ruff clean).

The entry-point clear closes the last staleness seam properly:
`inner_generate_step` nulls `context.last_step_stage_timings` immediately
after `get_context`, **before any FC/background early-return branch is
reachable** — with an in-code comment stating the invariant ("timing
evidence is current-invocation only"). This makes freshness independent of
frame-number coincidences on FC-async and background paths that return
without running `infer_one_step`; the server's `frame_end` binding becomes
a second, redundant guard rather than the only one. The context-update site
overwrites dict-or-None on every successful normal/direct update, and the
host validator's `math.isclose` checks now pin `rel_tol=0.0` — pure
absolute 0.05 ms tolerance, so large totals cannot silently loosen the
arithmetic gates. The multi-position remainder stands as corrected in 132b.

### Verdict

**APPROVE — C1 is complete at patch `232195f6…`; proceed to the diagnostic
rebuild and the preregistered probe analysis** (A132 correction 4: fast/slow
cluster binning with the codec-windowing hypothesis first, per-position-
class distributions for no-tools and tool probes, optimization ranking
committed before the run). The 132b note on the unreproduced focused-pair
intermittent stands unchanged.

## Addendum 133 — C2 design review: EarTTS idle bypass + early reset (2026-08-10)

The C1 attribution did its job and redirected the target: not codec
windowing but (a) EarTTS advancing on agent-idle positions whose output is
then discarded, and (b) the ~76–133 ms speaker prefill sitting on the BOS
critical path. Source confirms the foundation: `install_eartts_decode_pad_
silence` runs the full EarTTS step and then **replaces its output** with
`codec_silence_tokens` — idle-position EarTTS is provably dead work today,
so bypassing it changes nothing downstream of the existing substitution
point, and the codec keeps receiving the identical silence codes
(cache/pipelining untouched). Expected recovery from the C1 numbers:
~97 ms across six settlement positions plus 76–133 ms off the BOS path —
receive→audible improves ~170–230 ms; warm V2V toward ~1.25–1.30 s.

**APPROVE the architecture with the following binding requirements:**

1. **Bypass predicate = the idle arm only.** `eartts_pad_should_decode_
   silence` has two arms; the `tail_done` arm fires with the agent *not*
   idle, mid-response, where subsequent positions still consume EarTTS
   autoregressive state — bypassing there is unsafe. The bypass condition
   must be `current == PAD and agent_idle` (single-sourced with the
   existing gate's idle arm), exactly as the design states — this
   narrowing is load-bearing, not stylistic.
2. **The prev-tokens trap.** Today the at-BOS reset also injects
   `prev_audio_tokens = first_tts_code_input`. A clean-skip BOS must still
   receive that injection, and the `past_key_values` contract for bypassed
   positions must be defined (pass-through sentinel the post-reset BOS
   call provably ignores). Required test: the BOS call's effective inputs
   are identical between the early-reset and at-BOS-reset paths.
3. **Monotonic clean/dirty rule, no exceptions**: `clean` is set only by a
   completed abort+prefill (early or at-BOS) with recorded request id;
   *any* EarTTS vLLM invocation clears it — which automatically marks FC
   hidden cycles dirty (their forced tokens are not effective-PAD, so they
   run EarTTS) and routes the post-FC BOS through today's fallback reset.
   Request-id mismatch at a clean-skip BOS raises fatally; never degrade
   silently.
4. **Early-reset scheduling guard**: run the synchronous abort+prefill
   after `response_boundary event=end` only when the agent is idle *and*
   no user turn is open and no typed job is active — a barge-in that
   cancelled the response must not pay the reset inside its own turn;
   staying dirty and falling back at BOS is the correct price. First
   session is clean via `engine.start`'s own prefill.
5. **Lifecycle enumeration** (the 127b/131 precedent): clean/dirty and the
   early-reset record reset at engine start, start-failure, abort, empty
   restart, and wrapper mode transitions.
6. **Telemetry fail-closed**: clean/dirty and bypassed-position counts in
   `turn_state`; the C1 reset-timing schema gains a `site` field
   (`at_bos` / `post_response`); C1 stage timings will independently show
   `eartts_total_ms ≈ 0` on bypassed positions — the attribution
   instrument doubles as the regression detector.
7. **Qualification**: acoustic equivalence on idle positions is by
   construction (identical silence codes), but BOS-onward trajectories
   must be revalidated on the new image — the 13-conjunct fused contract,
   response-integrity and parity ASR gates, both probes per the C1
   protocol, and the tool-advertised browser E2E, per the standing
   discipline.

### Verdict

**APPROVE — proceed to implementation** under requirements 1–7. The two
sharp edges are (1) and (2); everything else is established pattern. The
design's own narrowing (idle-arm only, dirty-fallback everywhere doubtful)
is what makes it safe: at no point does correctness depend on the bypass —
it only depends on the reset, which is today's proven mechanism relocated
to idle time.

## Addendum 133a — C2 split decision: C2a first (2026-08-10)

**Prefer the split — implement C2a now, defer C2b.** The split is better
than the combined design, not merely smaller: it removes both A133 sharp
edges rather than mitigating them. With the at-BOS reset retained
byte-identically, requirement 2 (the prev-tokens trap) vanishes — the
reset still overwrites `prev_audio_tokens` unconditionally — and
requirement 3 (the clean/dirty machine) has nothing to track, because the
synchronization argument closes structurally: every *real* EarTTS call
sequence is anchored at a BOS (a response's content follows its opening
BOS; the post-FC forced BOS is a BOS), and every BOS aborts, prefills, and
overwrites prev state. A bypassed position's frozen PKV can therefore
never be consumed by a real call. FC hidden cycles need no special case:
their agent-channel positions satisfy the same idle-PAD predicate the
existing silence substitution already fires on, and the post-FC BOS
resynchronizes through the unchanged reset.

C2a carries forward from A133 only: requirement 1 (bypass predicate =
idle arm only, single-sourced), requirement 6's telemetry (bypassed-
position count; the C1 stage timings showing `eartts_total_ms ≈ 0` on
bypassed positions are the regression detector), the byte-equivalence
test that the bypass returns exactly `codec_silence_tokens` and the
incoming PKV unchanged, and requirement 7's qualification (probes + tool
E2E on the new image). Expected recovery ~97 ms across settlement
positions, live-measured before C2b is designed against that evidence —
at which point A133 requirements 2–5 apply to the reset move.

### Verdict

**APPROVE C2a as scoped; C2b deferred pending C2a live evidence.**

## Addendum 133b — C2a implementation review (2026-08-10)

Reviewed across three tree states as concurrent hardening landed; final
state: canonical patch `11f5f89c…`, focused 129 + 17 subtests and full CPU
686 green, Ruff clean.

### A real blocker was present mid-review and is now fixed — verified by execution

At intermediate patch `f23aeb44…`, the eligibility conjunct referenced
`decoder_silence_eligible` while its computation had been misplaced into
`perception_uses_captured_graph` — my direct probe raised `NameError`, and
short-circuit evaluation hid it from the CPU tier (`use_vllm_eartts=False`
in fakes never reaches the name) while the live GPU path would have crashed
on the first ordinary position. The concurrent hardening fixed both
functions; I re-probed the exact failing argument set on the final tree:
non-PAD/speaking-agent → False, idle-PAD → True. The predicate is now
computed inside the eligibility function via
`eartts_pad_should_decode_silence(..., prior_*=0, tail_ratio=0.0)` — the
tail arm disabled by construction, single-sourced idle arm, A133
requirement 1 exactly.

### The implementation meets every A133a gate

Eligibility requires, conjunctively: enabled, decode-audio, vLLM EarTTS,
**ordinary PCM only** (source-embedding positions excluded — direct-text
hidden transactions still advance EarTTS), no forced output PAD, the
single-sourced idle-PAD predicate, **full FC quiescence** (active/awaiting-
response/awaiting-EOTR/injecting/forced-tokens all clear, fail-closed on
any), and acoustic-capture disabled (captured codes stay real). Half-
installed hooks raise. The transition validates shape/quantizer match and
returns cloned `codec_silence_tokens` for the decoder while the wrapper
asserts **object identity** on the recurrent code and PKV — the pass-
through contract enforced at runtime, not by convention. The at-BOS reset
is untouched, preserving the A133a synchronization argument. Telemetry is
schema-2 with per-invocation bypassed frames plus a cumulative counter, and
the hardening moved the cumulative reset from the generic RNNT reset (which
FC cycles reuse internally and would have zeroed it mid-session) to
`reset_transport_session_state` only. The server validator pins mode ∈
{disabled, enabled_no_skip, bypassed}. The 14-mutation eligibility kill
matrix and identity-preservation tests cover the contract.

### One blocker remains: patch/live-tree parity

The canonical patch and the live Speech checkout differ by exactly one
line — the checkout is missing the
`eartts_idle_pad_bypassed_frames = []` initialization (all other deltas are
hunk-offset drift from it). The image build applies the canonical patch
and would be correct, but the sibling checkout — the environment live
probes run from — would `NameError` at the first bypassed position, and
the parity gate exists precisely so the patch matches what was live-tested.
**Exact fix: apply that one line to the sibling (or regenerate the diff)
and re-verify parity EXACT.**

### Verdict

**APPROVE the C2a implementation at patch `11f5f89c…`; BLOCK the rebuild
solely on the one-line live-tree sync above.** Once parity is EXACT,
proceed to the diagnostic rebuild and the C2a live gates (bypassed-position
stage timings ≈ 0, probes, tool E2E) per A133a.

## Addendum 133c — parity blocker withdrawn: wrong referent (2026-08-10)

The A133b rebuild blocker is **withdrawn as founded on a wrong referent**,
and the correction is recorded as a review error. The host sibling
checkout at `~/src/Speech-nemotron-voicechat` is a dev convenience that
happened to be in sync during every prior parity check; it is not the
probe environment. The authoritative chain is: bootstrap clones pinned
clean Speech and applies the retained canonical patch — and, decisively,
**the Dockerfile enforces the parity gate at build time itself**
(`git apply --check`, apply, then `git diff … | cmp - /tmp/speech.patch`):
an image cannot build unless the applied tree rederives the canonical
patch byte-exactly. Live probes run in the rebuilt container, never in the
old sibling; the currently running container is intentionally the C1
image until rebuild and must differ.

Verified here on the exact evidence offered: the verification worktree
`/tmp/voicechat-c2a-speech.Y8vPyB` (docker-cp of the exact C1 image's
`/opt/Speech` with C2a applied) regenerates a `git diff` that is
**byte-identical by `cmp`** to the retained canonical patch `11f5f89c…`,
and it contains the `eartts_idle_pad_bypassed_frames` initialization whose
absence from the stale sibling triggered the A133b concern.

The composed behavioral coverage added since 133b is present and green in
the focused pair (130 + 17 subtests rerun here): baseline installed
PAD-decoder-silence bytes equal the bypass decoder bytes; recurrent
code/PKV identity preserved; fallback audio history byte-equal; and the
unchanged next-BOS reset aborts/prefills and overrides the frozen
recurrent code with `first_tts_code_input` — the A133a synchronization
argument now proven behaviorally, not just structurally.

### Verdict

**C2a is APPROVED for the diagnostic rebuild with no remaining blockers.**
The build-time cmp gate is the parity authority going forward; future
addenda should cite it (or a fresh image-copy worktree) rather than the
host sibling. Live gates after rebuild per A133a: bypassed-position stage
timings ≈ 0, both probes, tool-advertised browser E2E.

## Addendum 133d — C2a final verdict (2026-08-10)

Final tree verified at canonical patch `3805cbac…`. The shared sibling
checkout now carries the full C2a delta — its five-file `git diff` is
byte-identical by `cmp` to the retained patch — so both parity referents
(the authoritative build-time cmp gate and the dev sibling) agree. Full
CPU 687 and the focused pair 130 + 17 subtests green here; Ruff clean.

The final telemetry hardening closes the last honesty seam: `reason`,
`recurrent_state` (`preserved_input_identity`), and `decoder_code`
(`codec_silence_tokens_clone`) are **null unless positions were actually
bypassed**, and the server validator derives the expected mode from the
frame evidence itself — monotonic unique frames, count equal to
skipped_positions, bounded by chunk positions, cumulative ≥ current — then
requires the claimed mode, reason, and both semantic strings to match
mode-dependently, with `skipped while disabled` fatal. The telemetry
cannot claim bypass semantics on a step that did not bypass, and cannot
bypass without declaring it.

### Verdict

**APPROVE — C2a is complete at patch `3805cbac…`; proceed to the
diagnostic rebuild.** Live gates unchanged from A133a/133c: bypassed
settlement positions showing `eartts_total_ms ≈ 0` in the C1 stage
timings, the no-tools and tool probes, and the tool-advertised browser
E2E on the rebuilt image. Expected recovery remains ~97 ms across
settlement positions, with C2b (the reset move, A133 requirements 2–5)
designed afterward against that measured evidence.

## Addendum 133e — C2a live promotion adjudication (2026-08-10)

All artifact hashes verified (browser log `c9de563f…`, tool-single live
`0efb159f…` / ASR `52230315…`, no-tools multiturn `b1eae528…` / ASR
`b7b5a682…`, trace `1a8a4d2d…`); image `c7308768…` present as
production-candidate-1; patch `3805cbac…` unchanged from 133d.

**The mechanism is proven live, rederived from the trace**: 642 model
steps split exactly into 439 `bypassed` + 203 `enabled_no_skip`,
cumulative counter max 439, and **`eartts_total_ms = 0.0 on every one of
the 439 bypassed positions`** — the A133a gate met with zero exceptions —
while the four legacy at-BOS resets ran intact at 72.9–78.3 ms. Browser
tool E2E green with V2V 1360.0/1322.5 ms vs 1470.7/1451.3 pre-C2a
(−110/−129 ms, consistent with the ~97 ms settlement prediction), and
TTFA − TTFB at 243.5/244.5 ms — the invariant leading silence, metrics
internally coherent. Tool-single live and ASR reports green (WER 0, exact
transcript, promoted evaluator).

**The four-turn no-tools red is single-cause, proven mechanically.** Every
substantive per-turn gate is green (input WER 0 ×4, semantics true ×4,
external ASR passed, 4 speech / 0 near-silent, response WER 0.0 ×4). The
top-level false comes from `multiturn_runtime_gate_passed`'s hardcoded
`expected_turn_count == 3`; with report count 4 and `len(responses) == 4`,
conjunct isolation shows relaxing only that one pin flips the gate to
True. Per the no-laundering discipline the artifact **stays red as
retained** and is treated as supporting evidence only; promotion rests on
the green artifacts. Follow-up (not a C2a blocker): the offline multiturn
gate needs prospective, versioned `expected_turn_count` support before it
is next used as a promotion gate for non-3-turn runs. The 70 s stress run
stays red as retained — its failure is a fixed client stop cutting the
fourth verbal response, an orchestration artifact, cited as evidence for
nothing.

### Verdict

**C2a is PROMOTED on this evidence — no blockers.** Image `c7308768…`
carries the full chain: bypass proven at zero EarTTS cost on 439 live
positions with reset synchronization intact, tool and no-tools semantics
green per-turn everywhere, browser E2E green with the predicted latency
recovery.

**C2b prepared-epoch design may begin**, under A133 requirements 2–5, with
one honest correction to its cost-benefit: the measured at-BOS reset on
this image is **~73–78 ms** (both fused and legacy), not the ~133 ms seen
in the C1 trace's first-session measurement — C2b's expected recovery is
~75 ms, and its design should be sized and judged against that number.

## Addendum 134 — C2b prepared-epoch design review (2026-08-10)

The design answers every A133 carry-over requirement explicitly: the
prev-tokens trap (clean BOS still clones `first_tts_code_input` into
`prev_audio_tokens`), the monotonic dirty rule strengthened to
**dirty-before-call** (an exception mid-call can never leave a stale
clean), lifecycle invalidation enumerated (reset/abort/reconfigure/
disconnect), first-turn cleanliness attested from `engine.start`'s own
prefill, fail-open prepare failure (legacy at-BOS reset remains the
byte-identical fallback for every doubt), C2a-bypass compatibility (bypass
never calls `infer_codes_one_step`, so clean survives idle positions), and
scope discipline (microphone turn_start only; typed/direct keep legacy).
Moving the prepare from post-response-idle (A133 req 4) to turn_start is
an improvement, not a deviation: the ~75 ms lands inside the user's own
multi-second speech window, off the settlement/BOS critical path, with no
idle-window race and the response-cancel already sequenced ahead of it.
The exact-clean snapshot (request id, RequestState id/active, iterator id,
generated-token count, structural tensor identity/version/shape/dtype/
device, exact prompt IDs) is adequate: rebinding is caught by object id,
in-place mutation by tensor version, consumption by token count and
iterator identity, reconfiguration by prompt IDs and invalidation.

### BLOCKER — the qualified reset-count conjuncts are not addressed

`observe_fused_terminal` (server.py:447) requires
`eartts_reset_count == starting + 1` **at the fused BOS position**, and
the settlement evidence records `eartts_reset_count_after` per step; a
further consumer reads `_reset_on_bos_count` at server.py:4198. A
prepared-clean BOS deliberately skips the at-BOS reset, so the counter
stays flat and the first clean no-tools fused turn dies with
`fused_eou_contract_failure`. The prepare's own increment cannot fix this:
it lands before `UserEouSettlement.begin` captures the baseline, so the
at-BOS delta is zero either way. Required design element: a
**prospective, versioned extension of the fused/settlement contracts** —
the honest evidence shape is an epoch-consumption marker (prepare epoch
id, consumed-at-BOS flag) such that the terminal contract accepts exactly
one of: `reset_count == starting + 1` (legacy reset at BOS), or
`reset_count == starting AND prepared_epoch_consumed(epoch)` with the
epoch's prefill bound to the same request id. Every consumer of
`_reset_on_bos_count` must be audited for the same assumption. Without
this the feature cannot pass its own qualification gates.

### Corrections (non-blocking)

1. **Partial-failure idempotence**: if prepare's abort succeeds and the
   prefill fails, the request is aborted-but-unprefilled; the legacy BOS
   fallback then aborts again — add a test that double-abort is safe and
   the fallback prefill recovers.
2. **Mismatch diagnosability**: on snapshot-revalidation failure, the
   telemetry should carry a reason code enumerating *which* field
   mismatched (no contents), else live fallbacks are undiagnosable.
3. **Trace the prepare cost** with the C1 reset-timing schema plus a
   `site: turn_start_prepare` tag, so the ~75 ms inside the turn_start
   handler is attributed and never rediscovered as an anomaly.
4. **State the payoff honestly in the doc**: ~73–78 ms, microphone-
   turn-start turns only; typed input and post-FC turns keep the legacy
   reset by design.

### Arming amendment (incorporated into this verdict)

The two-phase refinement — prepare publishes **clean but UNARMED** at
turn_start; after the microphone commit has flushed every pending PCM
sample and immediately before settlement, a cheap exact-validation **arm**
runs under `model_input_lock` on the model executor, binding the
`client_turn_id`; only armed+clean may reuse at BOS — is a material
strengthening, endorsed. It shrinks the trust interval from
turn_start→BOS (seconds of user speech) to commit→BOS (the locked
settlement window), makes consumption turn-scoped so typed/direct input or
an unexpected pre-commit BOS can never consume the prepared epoch, and
costs only comparisons on the critical path. Any real EarTTS call during
the utterance makes the arm refuse (legacy BOS); C2a skips preserve clean.
Two additions it implies: the arm must **disarm** on consumption, turn
end, and every invalidation event (single-consumption semantics), and the
BLOCKER's contract evidence should bind the consumed epoch to the armed
`client_turn_id` — which makes the extended fused-contract conjunct
stronger than the legacy one, not merely alternative to it.

### Verdict

**APPROVE the design — with the arming amendment — contingent on the
BLOCKER being resolved in the design before implementation** — the
epoch-consumption contract extension is qualified-surface work
(prospective, versioned, with its own kill tests), not an implementation
detail, and its evidence should carry the armed `client_turn_id` binding
per the amendment. Corrections 1–4 land with the implementation.
Promotion gates as proposed: browser E2E, canonical 3-turn no-tools +
ASR, actual tool + ASR — all on the rebuilt image.

## Addendum 134b — blocker resolution accepted: BOS-epoch transition invariant (2026-08-10)

The A134 BLOCKER is **closed by this design**. The reset-only terminal
invariant is replaced — prospectively, as schema-v2 settlement evidence —
by an exact BOS-epoch transition invariant, and the replacement is
strictly stronger than what it supersedes:

- **Legacy path loses nothing and gains constraints**: reset delta 1 is
  still required, now conjoined with transition delta exactly 1 and reuse
  delta 0.
- **Prepared path gets exact evidence instead of an exemption**: reset
  delta 0 with reuse delta exactly 1, request-id match, armed
  `client_turn_id` match, and disarm-on-consumption — the A134 amendment's
  turn binding realized in the contract itself.
- **The exclusive-mode requirement closes every corner the old conjunct
  could not see**: double reset, reset-plus-reuse, and the
  neither-transition case (a BOS call that raises increments nothing and
  fails the delta-1 requirement fail-closed).
- **Coverage extends, not just transfers**: the fused BOS keeps its
  in-contract validation inside `observe_fused_terminal`, and the separate
  legacy/tool BOS position — previously outside any per-position reset
  contract — is now validated immediately after its `process_pcm`, with
  the terminal event recorded in the evidence.
- **Scope edges are consistent**: typed carries a null expected turn id
  and can therefore only ever take the legacy fallback; the direct-probe
  audit keeps its reset assertion unchanged because direct never arms;
  the remaining generic reset-only consumers are enumerated for update.
- `last_bos_event` (mode, request, armed turn, epochs, all three counts,
  `fallback_reason`) delivers A134 correction 2's diagnosability; the
  double-abort test and `site: turn_start_prepare` tracing cover
  corrections 1 and 3. The proposed mutation matrix — both modes,
  double/neither transitions, wrong request/turn/epoch, reuse-without-arm,
  separate and fused shapes — is the right kill-test set.

One hygiene note, non-blocking: the new cumulative counters are
delta-validated so session zeroing is not load-bearing, but they should
still reset in `reset_transport_session_state` alongside the C2a bypass
counter for telemetry coherence.

### Verdict

**Blocker closed — C2b implementation is APPROVED** under the A134 design
plus arming amendment plus this transition-invariant contract, with the
A134 corrections 1–4 landing in the implementation and the promotion
gates unchanged (browser E2E, canonical 3-turn no-tools + ASR, actual
tool + ASR on the rebuilt image).

## Addendum 134c — C2b implementation review (2026-08-10)

Verified in the current tree; canonical Speech patch unchanged from
promoted C2a (`3805cbac…` — C2b is entirely host-side, as designed); full
CPU 697 + 15 skipped + 20 subtests green here; Ruff clean across
src/tools/tests. Attack-surface findings:

- **RLock/FC race — clean.** The wrapped `infer_codes_one_step` takes
  `controller.lock` around the *entire* call, so `consume_prepared_bos`
  and `mark_real_call` always run locked (the RLock's re-entrancy is why
  prepare/arm can also lock); server-side, prepare and arm run under
  `model_input_lock` via the single model executor — dual serialization.
  The prepare checks FC quiescence **before the abort and again after the
  prefill**, invalidating if a cycle started mid-prepare.
- **Identity — exact.** The snapshot binds request id, RequestState and
  iterator object ids, generated-token count, structural tensor identity
  for speaker inputs and initial code, and exact prompt token IDs; every
  mismatch enumerates its field as the invalidation reason
  (A134 correction 2 delivered).
- **Unarmed/commit binding — per the amendment.** Attestation at
  `engine.start`; turn_start prepare gated on no-FC/finished/response-
  closed/session-started, sequenced after cancel-to-terminal and before
  the barrier acks; the arm runs in the commit handler **after the tail
  PCM flush**, under the lock, bound to `client_turn_id`, evidence traced.
  Double-arm refuses; consume disarms; typed can never reuse (null
  expected turn fails the reuse-event check), and typed/direct never arm.
- **Dirty-before-call/failure — stronger than spec.** Every real non-BOS
  call marks dirty before invocation; BOS inference failure invalidates
  with a mode-specific reason; prepare failure invalidates and falls back
  legacy (abort-then-prefill-failed leaves legacy's own abort+prefill to
  recover, covered by the double-abort test).
- **Routing and arithmetic — 134b realized exactly.** Both BOS paths
  inject `prev_audio_tokens = first_tts_code_input.clone()`;
  `transition_count` increments only after a successful BOS call;
  `last_bos_event` carries mode/request/armed-turn/epochs/all counts/
  fallback-reason; the settlement validator enforces transition delta 1,
  the accounting identity (transition = reset + reuse), the exclusive
  mode set, event freshness against all three after-counts, armed-turn
  binding per mode, and post-consumption dirty/disarmed state — with the
  controller-absent path preserving the old reset-only invariant
  byte-for-byte. The separate legacy/tool BOS gains pre-publication
  validation it never had.
- **C1 timing composition** — a prepared BOS leaves
  `_reset_on_bos_last_timing_ms` null and the C1 validator accepts null;
  legacy BOS emits timing as before. **C2a composition** — bypassed
  positions never reach the wrapper, so clean survives idle, and the
  install gate refuses prepared-epoch without C2a. Production config opts
  in; health/provenance expose controller status.

Non-blocking note carried from 134b: cumulative counters survive
`reset_session` (delta validation makes this sound), but zeroing them at
the transport session boundary would keep telemetry coherent with the
C2a bypass counter.

### Verdict

**APPROVE — proceed to the rebuild and the C2b promotion gates** (browser
E2E, canonical 3-turn no-tools + ASR, actual tool + ASR, all on the
rebuilt image), expecting ~73–78 ms recovery on microphone turns and the
first live `prepared_reuse` events in the settlement evidence.

## Addendum 134d — C2b re-review after independent BLOCK (2026-08-10)

Reviewed fresh against the current tree without reliance on 134c — which
is owed an honest accounting first: **the independent BLOCK was justified,
and 134c under-inspected exactly the surfaces it hit.** 134c verified the
prepare's happy path but not that `abort_request()` could report failure,
nor that a stale RequestState could survive the abort; it took the
deferred post-FC path as "consistent" without tracing the timeout
lifecycle; and it did not ask what happens to a pending BOS expectation
when a tool cycle dies. Those were real seams. The fixes are now landed
and verified here:

1. **Prepare is transactional against the backend, not just the
   controller**: `abort_request()` must return true; the pre-abort
   RequestState must be absent-or-replaced after the abort *and* not
   reused after the prefill (three distinct raises); the post-prefill
   quiescence failure now retains the measured prefill timing instead of
   reporting None for work that ran.
2. **Request identity is authoritative**: prepare/arm/BOS validation
   derive the request id from `pipeline._request_id_for_stream`, and the
   settlement validator now binds `last_bos_event.request_id` to that
   expectation plus the event's session/prepare epochs to the controller
   status — the cross-request and cross-epoch seams are closed with
   enumerated violation codes.
3. **The deferred post-FC lifecycle is fail-closed at every exit.**
   `observe_separate_bos_or_defer`: immediate BOS validates and clears;
   FC-active retains the expectation with `overlapping_bos_expectation`
   fatal on double-set; neither is `missing_terminal_bos` fatal. During
   recovery, `validate_pending_terminal` runs as `before_send` — **an
   invalid post-FC BOS raises before its frame is published**, tolerated
   intermediates are enumerated (cycle active; forced BOS still pending),
   and recovery completing without its BOS is
   `missing_post_fc_terminal_bos` fatal. **Tool timeout with a pending
   expectation clears exactly that expectation, invalidates the EarTTS
   epoch, attaches the settlement evidence to a fatal error, and closes
   the session** — no stale expectation can ever meet a later turn's BOS.
4. **Eager-start failure aborts the partial session** (the pre-existing
   start-transaction cleanup extended to the attestation path).

Suites rerun here: focused 199 + 15 skipped + 23 subtests, full CPU 707,
Ruff clean.

### Verdict

**APPROVE — proceed to the rebuild.** The hardened tree is materially
stronger than the 134c-approved one, and the strengthening is of the
right kind: backend-verified transactions, authoritative identity,
pre-publication validation, and fatal cleanup on every abnormal exit of
the deferred path. Promotion gates unchanged (browser E2E, canonical
3-turn no-tools + ASR, actual tool + ASR on the rebuilt image).

## Addendum 134e — deferred-timeout fail-open closure review (2026-08-10)

The independent finding was real and again sat one layer below my prior
pass: 134d verified the timeout path's *ordering* but not the
exception-safety of the awaited epoch invalidation — an
`invalidate_eartts_epoch` raise between clearing the expectation and
sending the fatal error would have skipped the close and failed open.
Recorded as a second review miss on this path.

The fix is verified in the tree and is correctly shaped as a single
shared fail-closed helper (`terminate_pending_bos_expectation`), used by
both the tool timeout (5761) and the bounded post-output recovery timeout
(6142):

- **Pending cleared first** — before any await, so no failure mode can
  leave a stale expectation; the early return on already-cleared makes
  the helper idempotent against double invocation.
- **Invalidation cannot fail open**: wrapped to catch both `Exception`
  and `asyncio.CancelledError`, recording the error name into the trace
  event *and* the fatal wire error rather than propagating.
- **Send/close discipline**: the fatal send runs under `send_lock` inside
  `try/finally` — a dead client cannot prevent the close — and
  `terminal_session.set()` wakes the main receiver to run the endpoint's
  abort cleanup. Sessions without a pending expectation (feature off, or
  ordinary tool sessions) return False and keep today's behavior exactly.
- **Tests match the failure modes**: ordinary deferred timeout, the
  throwing invalidator
  (`test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails`),
  and the bounded recovery timeout
  (`test_deferred_tool_recovery_timeout_clears_epoch_and_closes`).
  Focused 201 + 25 subtests and full CPU 709 green here; Ruff clean.

One non-blocking nit: inside the helper's `finally`, `terminal_session.set()`
follows `await websocket.close()`; if the close itself raises (dead socket,
task cancellation) the wake is skipped. The receiver still wakes via the
native disconnect in those states, so this is not fail-open — but setting
the event before the close await (or wrapping the close) would make the
wake unconditional at zero cost.

### Verdict

**APPROVE — proceed to the rebuild.** The deferred-BOS contract now fails
closed under every probed fault: invalidation raise, cancellation, dead
client, double invocation, and recovery overrun — with default-path
behavior byte-preserved for sessions that never arm.

## Addendum 134f — turn-2 stale-event fix review (2026-08-10)

The first C2b live failure is, read correctly, a success of the contract
layer: turn 2's prepare advanced `prepare_epoch` 2→3 while the controller
retained turn 1's consumed BOS event, and the per-frame validator's
staleness rule (`event.prepare_epoch == status.prepare_epoch`,
server.py:2499) rejected the very first frame rather than exposing stale
terminal evidence. The validator needed no change; the producer did.

The fix is minimal and in the right place: `_publish_clean` now clears
`last_bos_event` whenever a new clean epoch is published, with the
invariant stated in-code — a BOS event is evidence for the epoch that
produced it, and the next successful BOS writes a new event before any
frame can be exposed.

**No required terminal evidence can be lost by this clear**, verified
against every consumer: the fused, separate, and deferred validators all
read the event at the BOS step itself (before publication), and a prepare
can only run at a microphone turn_start with the function cycle quiescent
and the response closed — which cannot interleave between a BOS step and
its same-step validation in the serial loop. In the deferred case the
prepare precondition (`function_cycle_call_id is None`) blocks preparation
while an expectation is pending, so the clear cannot race the post-FC
validation either. The per-frame validator accepts a null event, so
post-publish frames pass cleanly.

Suites rerun here: focused 202 + 25 subtests, full CPU 710, all green.
The regression test drives arm+BOS then a second prepare and asserts the
event nulls with the new epoch. The timing-sensitive timeout subtest that
flaked once in the broader run is the known intermittent family
(A132b/A127-F2); it remains on the deflake list and gates nothing here.

### Verdict

**APPROVE — proceed to the rebuild.** The producer lifecycle now matches
the validator's epoch semantics, the fix provably cannot orphan a pending
terminal validation, and the live failure it addresses was caught by the
system's own fail-closed layer at the first frame — which is exactly the
behavior this review series has been building toward.

## Addendum 134g — vLLM abort/re-add ABA: root cause and fix selection (2026-08-10)

**Root cause verified against the pinned voicechat vLLM source, not just
the report.** `Scheduler.add_request` installs a request with no live-
duplicate check and no tombstone hygiene; `finished_req_ids` is by design
carried into the *next* `SchedulerOutput` ("finished in between the
previous and the current steps") and replaced only inside a scheduling
step. C2b's prepare aborts request '2' and immediately re-adds the same
ID; C2a's idle bypass means **zero scheduler iterations run** until the
BOS ~4 s later, whose output then broadcasts '2' as finished — evicting
the replacement — and `set_custom_inputs` dies with "Request 2 not found",
taking the EarTTS EngineCore with it. Host-side RequestState remained
ACTIVE throughout because the death was core-side, which is why the
controller's exact-match snapshot could not see it. Turn 1 succeeded
because attestation reuses the initial prefill with no abort. Decisive
corollary: **the legacy at-BOS reset has carried the same latent race all
along**, surviving only because generation cadence happened to flush the
tombstone between abort and add.

### Fix selection: (A), and it is not close

**(A) — patch `Scheduler.add_request`** to (i) reject adding an ID that is
currently live (explicit error, fail-closed at the misuse site) and (ii)
discard the exact ID from **both** `finished_req_ids` and
`finished_req_ids_dict` before installing a replacement. This fixes the
invariant at its home — a new request must never inherit its
predecessor's death sentence — for every reuse site at once: C2b prepare,
the legacy at-BOS reset's latent race, and any future path. It is
default-neutral (per-ID tombstone discard is a no-op when no tombstone
exists; Nano's engine is a separate build artifact and untouched), and it
downgrades the failure severity class: residual misuse becomes an
immediate add-time error that the prepare's except path converts into
invalid + legacy fallback, instead of a delayed EngineCore death.

**(B) — generation-unique internal IDs** is rejected: it leaves the
scheduler landmine armed for any same-ID sequence (including the legacy
reset unless it also migrates), scatters a stream→generation mapping
across every host consumer that binds request identity (controller
snapshot, C1 timing, settlement validators, context-manager
abort-by-id), and converts an invariant into a convention.

**No barrier is needed or wanted**: an abort-completion barrier cannot
close the race — the tombstone flush requires a scheduling *iteration*,
not message delivery — and with (A) correct the sequence
abort→add→idle→step is safe at any gap length.

### Exact requirements

1. **Red-first deterministic reproduction** at scheduler level: abort an
   ID, re-add the same ID with zero intervening iterations, run one step,
   assert the replacement survives and receives custom inputs — failing
   before the patch, passing after; plus a live-duplicate rejection test.
2. **Per-ID hygiene only**: discard exactly the re-added ID from both
   tombstone structures; never clear whole sets (the scheduler's own
   comment warns the set is shared with the in-flight output).
3. **Provenance discipline**: the fix lands as a pinned voicechat-vLLM
   fork revision or a build-applied patch under the same apply+cmp gate
   as the Speech patch, with labels/provenance updated; scope explicitly
   excludes the native/Nano vLLM artifact.
4. **Requalification**: inference-engine change ⇒ full standing chain on
   the rebuilt image — C2b promotion gates (browser E2E, canonical 3-turn
   no-tools + ASR, actual tool + ASR) plus one no-tools run specifically
   proving turn-2-and-later prepared reuse across the formerly fatal
   idle-gap sequence.
5. The controller/wrapper stay unchanged — their fail-closed behavior was
   correct; the defect was below them.

### Verdict

**BLOCK the rebuild until fix (A) lands** — the current tree detonates
deterministically on turn 2 of any no-tools session. **APPROVE direction
(A)** with requirements 1–5; (B) rejected as symptom-level.

## Addendum 134h — corrected root cause on the installed source; fix selection reversed (2026-08-10)

Two review errors in 134g are corrected here, both mine. First, **I read
the wrong source tree**: my scheduler citations came from the
`/opt/src/vllm-voicechat` checkout, not the installed package. The running
engine is `vllm 0.17.1.dev0+gb31e9326a.fi065`
(`/usr/local/lib/python3.12/dist-packages/vllm`), whose `v1/engine/core.py`
contains **zero** occurrences of `aborts_queue` — ABORT is enqueued exactly
once into the FIFO input queue — and whose pinned `gpu_model_runner`
explicitly handles a `finished_req_ids` entry overlapping a same-ID
`scheduled_new_reqs` entry (old cleanup, then new). The 134g mechanism
(runner-side tombstone eviction) is therefore wrong for the pinned build.

**The corrected mechanism, rederived from the installed source**: the
first abort's `_free_request` records the ID in
`finished_req_ids_dict[client]` (scheduler.py:1529–1530), and that finish
notification travels to the host asynchronously. It arrives after the
prepare has re-registered the replacement under the same ID. The host
`AsyncLLM` output handler then sends
`abort_requests_async(processed_outputs.reqs_to_abort)`
(async_llm.py:517) — a **second, host-synthesized abort** born from the
stale finish — which the core processes normally and which kills the
replacement, while the host `RequestState` stays ACTIVE
(`generated_tokens=1` from the completed prefill). At the BOS ~4 s later,
`set_custom_inputs` finds neither a live request nor (post-flush) a
tombstone and raises "Request 2 not found", killing the EngineCore. Every
observed datum fits, including turn 1's success (attestation performs no
abort) and the silent-ignore branch not firing (an iteration flushed the
second abort's tombstone).

### Fix selection reversed: (B) unique physical IDs

134g's (A) — scheduler `add_request` tombstone hygiene plus live-duplicate
rejection — **does not close the actual chain**: the stale finish is
emitted legitimately at the first abort and reacted to host-side;
tombstone clearing at add is upstream of the kill, and no duplicate exists
at add time. (B) closes it **by construction at every layer**: a stale
finish for generation N can never name generation N+1, the host's
synthesized abort targets a dead ID (harmless no-op), and every
unaudited consumer of finish notifications is covered without being
audited. The pinned vLLM artifact stays untouched — a provenance win.
Exact requirements:

1. **Single-source physical-ID allocation**: one wrapper-owned mapping
   stream → current physical id (monotonic generation suffix); every
   reset/prepare site — C2b prepare, legacy at-BOS reset (which carries
   the same latent chain, cadence-masked), and engine-start prefill —
   allocates through it; `_request_id_for_stream` returns the current
   physical id so the C2b snapshot, settlement validators, and C1 timing
   bind the unique id with no schema change.
2. **Lifecycle**: context-manager/session teardown aborts the *current*
   physical id via the mapping; stale-generation aborts must be proven
   no-op-safe (test).
3. **Red-first reproduction against the true mechanism**: deliver a
   stale finish for generation N after generation N+1 is registered and
   assert the replacement survives to a successful custom-inputs step —
   failing under ID reuse, passing under unique IDs; plus the ~idle-gap
   integration shape (abort → re-add → bypassed idle → BOS) live.
4. **Telemetry**: generation number in the prepare/BOS events so a stale
   in-flight finish is diagnosable from traces.
5. The scheduler's add-time behavior is recorded as a documented sharp
   edge of the pinned build; unique IDs become the standing rule for all
   EarTTS request recreation, and no vLLM patch ships.

### Verdict

**BLOCK stands until (B) lands as specified; 134g's fix direction is
retracted.** The 134g requirements 3–5 (provenance, requalification, and
controller/wrapper unchanged above the ID layer) carry over intact.

## Addendum 134i — frontend async-generator ABA confirmed; (B) verdict strengthened (2026-08-10)

The additional evidence is verified in the installed sources and supplies
the most direct killing chain yet — sharper than 134h's output-processor
reaction, and with live corroboration:

1. `AsyncLLM.generate` catches `(asyncio.CancelledError, GeneratorExit)`
   and executes `await self.abort(request_id)` — the request id captured
   **by value** at generator creation (verified at both catch sites in the
   installed `async_llm.py`).
2. NeMo `abort_generation` deletes the local RequestState and
   generation-iterator bookkeeping after `AsyncLLM.abort`, but the old
   async generator object finalizes **later**, on the event loop's or the
   GC's schedule. When it does, its `GeneratorExit` handler fires a fresh,
   legitimately-formed ABORT for the captured id — which, under id reuse,
   now names the replacement.
3. The live log's **second GPU block-zeroing event on the first
   C2a-bypassed frame after the replacement prefill** is the delayed
   cleanup landing core-side — timing consistent with finalizer-driven
   abort during the idle window, followed by the BOS finding the request
   gone.

This settles the 134g question categorically: the killing message is
indistinguishable from a real abort by the time it reaches the core, so
**no scheduler- or runner-side hygiene can protect the replacement** — the
class of hazard is "asynchronous host-side cleanup paths holding captured
id strings", of which at least two now exist (generator finalizer,
output-processor reaction), and only generation-unique physical IDs
neutralize the entire class by construction.

Amendments to the 134h requirements: the red-first reproduction must
simulate **this** chain specifically — create a generation, abort and
delete its bookkeeping while the generator remains unfinalized, reuse the
id, then drive finalization and assert the replacement dies under id
reuse and survives under unique ids. Recommended hygiene (non-blocking,
harmless under unique ids): `abort_generation` should explicitly
`aclose()`/drain the old generation iterator rather than leaving
finalization to GC, so cleanup timing is owned rather than incidental.

### Verdict

**134h stands, strengthened: BLOCK until generation-unique physical
EarTTS IDs land per 134h requirements 1–5 with the reproduction amended
as above.**

## Addendum 134j — ABA fix implementation review (superseding) (2026-08-10)

Inspected both trees. The implementation is the 134h/134i design realized
correctly, with defense in depth:

- **Engine (sibling Speech tree)**: `RequestState` gains
  `backend_request_id`/`backend_generation`; allocation returns the
  logical id with generation 0 when the mode is off — Nano and non-opted
  EarTTS are byte-neutral by construction, and the opt-in shares the one
  `VOICECHAT_EARTTS_PREPARED_EPOCH` flag. All generate/append/abort call
  sites route the physical id; `abort_generation` awaits the backend
  abort and then **synchronously `aclose()`s the old async generator**,
  with the in-code comment naming the GeneratorExit second-abort — the
  134i chain is closed twice: finalization is owned (no GC-timed abort)
  and, even if a future path defers it, the stale abort names a dead
  generation.
- **Host**: prepared startup refuses without unique-mode + resolver
  (attestation, fail-closed); the controller snapshot, C1 reset-timing,
  and the reset-on-BOS wrapper all resolve the physical id; the legacy
  at-BOS reset rotates generations, closing its cadence-masked latent
  race per 134h requirement 1; provenance records
  `eartts_unique_backend_request_ids`.
- **Server**: settlement terminal validation binds
  `last_bos_event.backend_request_id` against the engine's live backend
  identity — the physical binding survives to the qualified evidence
  layer. Keeping `pipeline._request_id_for_stream` **logical** is the
  right layering: Nano shares that contract, and the logical→physical
  translation happens exactly once, at the EarTTS boundary resolver.
- **Verification**: focused 204 + 25 subtests green; full CPU 712 green in
  three consecutive runs (the one earlier failure was the standing
  timing-intermittent, which also dropped its subtest count — still on
  the deflake list); Ruff clean. The no-GPU container canary
  (gen1 abort + aclose + simulated late stale abort → gen2 remains
  active) matches the 134i reproduction shape; its script path should be
  cited in the promotion evidence.

### BLOCKER — the canonical patch is stale

The canonical Speech patch is **unchanged at C2a's `3805cbac…` and
contains none of the ABA machinery**; the sibling checkout carries the
full fix (29 physical-id/aclose references). Because the build-time cmp
gate validates patch⇔applied-tree — not patch⇔sibling — **a rebuild today
would succeed and silently ship without the Speech half of the fix**,
recreating the live failure. Required before rebuild: mechanically
regenerate the canonical patch from the sibling's five-file diff, verify
byte-exact `cmp`, and propagate the new patch SHA through the
provenance/fixture chain (the patch file is a pinned fixture source
input).

### Verdict

**Implementation content APPROVED on every inspected surface; rebuild
BLOCKED solely on the canonical-patch regeneration above.** Once parity
is EXACT, proceed to the rebuild and the amended promotion gates (134h
req 4): browser E2E, canonical 3-turn no-tools + ASR, actual tool + ASR,
plus the no-tools run proving turn-2+ prepared reuse across the formerly
fatal abort→re-add→idle→BOS sequence.

## Addendum 134k — default-neutrality refinement review (2026-08-10)

The refinement is verified in the sibling and corrects a soft spot in
134j: the unconditional `aclose()` I credited there as defense-in-depth
was itself a default-path behavior change. Now: `aclose` runs only under
`_unique_backend_request_ids`; the default/Nano abort path keeps the
byte-original logging line (restored in an explicit `else`), the original
iterator lifecycle, and the original `get_status` schema; the server's
backend-identity helper requires `backend_generation >= 1`, so default
(generation 0) yields no backend telemetry fields and the validators fall
back to their pre-ABA behavior; the audit canary asserts default identity
mapping and the unchanged iterator lifecycle. Focused suites green.

Stated plainly for the record: **the default path retains the latent
finalizer race by design** — it is today's promoted, cadence-masked
behavior, and changing it silently would violate the neutrality
discipline this series enforces. The race is fully closed (unique IDs +
owned `aclose`) exactly where the new abort→re-add→idle sequence opens
the window: prepared EarTTS mode.

### Verdict

**Refinement APPROVED.** The 134j blocker stands unchanged and now covers
the refined sibling state: regenerate the canonical patch byte-exact from
the sibling before any rebuild.

## Addendum 134l — final pre-rebuild re-review: parity restored, blockers closed (2026-08-10)

Verified on the final tree. **The 134j blocker is closed**: the canonical
five-file Speech patch (`c11d2b6f…`) is byte-exact by `cmp` against the
sibling checkout, now containing the full ABA machinery (31 physical-id
references) including the 134k default-neutrality gating; the reported
clean apply to a detached `911ec674` worktree with exact post-apply cmp
matches the build gate's own enforcement path.

**Both self-review blockers are real fixes, verified in source**:

1. **Unique-mode start retires-or-refuses**: an existing state must be
   retired via `abort_generation` returning True *and* be absent from the
   registry afterward, else start raises — a failed abort/aclose can never
   be papered over by overwriting the registration (which would have
   orphaned a live generator holding a captured id: the exact 134i hazard
   shape recreated at the start boundary).
2. **The unique resolver is fail-closed**: missing logical state raises;
   a backend id that is absent, empty, or **equal to the logical id**
   raises (with generation type checks) — the earlier silent
   `return request_id` fallback would have routed operations to a
   physical id that names nothing. Inactive iterators are closed rather
   than leaked.

The direct-position continuity check now binds backend ID + generation
alongside status + iterator, and the build audit canary
(`eartts-request-generation-canary.json`) exercises all six declared
behaviors — abort-failure no-overwrite, inactive close, invalid resolver,
late stale abort (`stale_abort_isolated`), direct ID/generation
mutations, and an explicit default-neutrality engine with unique mode
off. Suites: focused 205 + 25 subtests green; full CPU 713 green in
consecutive reruns (the single earlier failure was the standing timing
intermittent, same dropped-subtest signature — deflake item unchanged);
Ruff clean.

### Verdict

**APPROVE — rebuild unblocked.** Proceed to the rebuild and the amended
promotion gates: browser E2E, canonical 3-turn no-tools + ASR, actual
tool + ASR, and the no-tools run proving turn-2+ prepared reuse across
the formerly fatal abort→re-add→idle→BOS sequence, with the
`eartts-request-generation-canary.json` retained in the image audit
evidence.

## Addendum 135 — bootstrap `--network host` build fix review (2026-08-10)

Reviewed in the tree: both BuildKit invocations in
`container/build-public-runtime.sh` now pass `--network host`. The
diagnosis is sound (Docker's bridge/embedded DNS failing while the host
namespace resolves), the fix is the smallest correct one, and the
alternatives are worse: pinning `--add-host` IPs is brittle against CDN
rotation, and repairing the daemon's DNS is a host-configuration mutation
outside the repo's scope (worth an ops note as the eventual clean fix).

**Reproducibility — unchanged, and provenance-visible.** Network mode is
not an image input; every clone checks out `--detach` at a pinned
revision (content self-verifying by commit hash), the in-image Speech
patch apply + byte-cmp gate is untouched, and the payload/recipe/source
audits still recompute identity. Decisively:
`build-public-runtime.sh` is itself a pinned **recipe input** in
`public_runtime_identity`, so this edit changes the recipe SHA and
surfaces in the image labels — the change cannot ship silently.

**Security — a marginal, honest delta.** Build-time RUN steps already had
full outbound network (apt/pip/git); host mode additionally exposes
host-local sockets (the model server port, ngrok's local API) to build
processes. Exploiting that requires a compromised build input executing
code at build time — a risk that exists identically under bridge
networking — so the delta is reachability of localhost, on a single-user
DGX, with outputs pinned by the audit chain. Acceptable; the pip layer is
version-pinned but not hash-pinned, which is the pre-existing weakest
input and unchanged by this fix.

**Consistency note (non-blocking)**: `build-asr-evaluator.sh` invokes
`docker build` without `--network host` and will hit the same DNS failure
on its next rebuild; the evaluator image is promoted-immutable so this is
dormant, but the same one-line fix should land there before any evaluator
rebuild.

**Tests**: none unit-testable; the gates are the rebuild completing plus
the in-image apply/cmp and identity audits — which is the correct
evidence for a network-mode change.

### Verdict

**APPROVE.** The bootstrap's stale-source detection did its job, the fix
restores buildability without touching any pinned revision or audit gate,
and the recipe-identity system makes the change self-documenting.
Proceed with the rebuild.

## Addendum 136 — audit canary import-path fix review (2026-08-10)

Verified in the tree. The serving truth is server.py:3681 —
`sys.path.insert(0, speech_root)` — so `/opt/Speech` is the exact import
surface production serves from, and the canary's new
`-e PYTHONPATH=/opt/Speech` (on that one `--network none` docker run
only) makes it test the patched tree that actually runs, where before it
imported the installed, unpatched NeMo. Two properties worth recording:

1. **The failure was fail-closed, not false-green.** The wrong-tree
   canary went red (missing patched machinery) rather than silently
   passing against the wrong code — the audit harness behaved exactly as
   its design intends, and the fix corrects aim, not discipline.
2. **Rerun-without-rebuild is legitimate**, verified against the identity
   module: `audit-public-runtime.sh` is in neither
   `PUBLIC_RUNTIME_RECIPE_FILES` nor the payload inventory, so image
   `b0eca1…` and its labels are untouched by this edit. The audit is
   evidence tooling, not an image input.

The static drift-lock exists (`test_bootstrap.py:173` asserts the exact
flag in the script) and passes. One hygiene note, non-blocking: the
canary JSON could self-record the audit script's own SHA the way the
evaluator backend self-asserts its hash, so the evidence artifact carries
its generator's identity.

### Verdict

**APPROVE — rerun the audit on the existing `b0eca1…` image without
rebuild**, then proceed to the promotion gates.

## Addendum 137 — tool-path terminal contract: FC-internal BOS transitions break the exactly-one invariant (2026-08-10)

Root cause pinned from trace `9a4e5fed…`, not the report's summary. The
settlement snapshot (`transition=3, reuse=3, reset=0`) is **cumulative
carry-over from the prior green no-tools session** — the counters survive
session reset, which is the 134c hygiene note now actively confusing
evidence readers. The failing window's real structure: boundary SOTC
commits at frame 58 (FC_ACTIVE), a **legacy BOS transition fires at frame
60 while the function cycle is active** (gen 8, transitions 3→4) — an
FC-cycle-internal EarTTS BOS, not a terminal — and the true post-FC
terminal lands at frame 90 (gen 9, transitions 4→5). The deferred
expectation, snapshotted before the FC cycle, therefore sees
`transition_delta = 2` and the 134b exactly-one invariant fails fatally.

**The invariant is wrong-by-design for windows that span a function
cycle.** The 134b contract assumed the only BOS between settlement-begin
and terminal is the terminal itself; that holds for fused, immediate
separate, and typed-immediate shapes (a window containing an FC cycle is
always deferred, by `observe_separate_bos_or_defer`'s own construction),
but a deferred window legitimately contains the FC machinery's hidden
BOS activity. My 134b/134c/134d reviews validated the deferred
*lifecycle* without ever asking whether the FC cycle itself produces
wrapper-visible BOS transitions. It does.

### Smallest principled correction

1. **Re-baseline the deferred expectation at recovery entry**: when the
   FC cycle drains and `validate_pending_terminal` is installed, snapshot
   transition/reset/reuse again on the pending settlement; the terminal
   contract then requires exactly one transition **from recovery-start to
   terminal**. FC-internal transitions fall between park and re-baseline
   and are excluded; the mode-exclusive delta arithmetic, armed-turn
   binding, and same-step backend-identity binding are unchanged; the
   evidence records both baselines (pre-FC and post-FC) so the internal
   transitions remain visible, just not misattributed. No wrapper
   changes, no new counters.
2. **Consume-refusal at FC-active BOS** (evidence honesty, small): the
   controller's `consume_prepared_bos` should refuse with reason
   `function_cycle_active` when the FC state machine is live, so an armed
   epoch can never be recorded as "consumed" by a hidden internal BOS.
   It cannot save the reuse — the internal BOS is a real EarTTS call and
   dirties the epoch regardless, so post-FC terminals are always
   `legacy_fallback` — but the `last_bos_event` then tells the truth
   about who consumed what.
3. **Promote the counter-hygiene note to required**: zero the cumulative
   transition/reuse counters in `reset_transport_session_state` alongside
   the bypass counter. Delta validation never needed it; evidence
   legibility now does — this failure's first reading was wrong because
   of cross-session carry-over.

### Tests

(i) A regression reproducing this trace's exact shape: typed tool turn →
SOTC at the post-EOU position → FC cycle containing ≥1 internal BOS →
post-FC terminal validates green under the re-baseline and red without
it; (ii) two FC cycles in one turn (re-baseline per drain); (iii) an
extra BOS **after** recovery entry still fatal (the exactly-one force is
retained where it is meaningful); (iv) voice tool turn with armed epoch:
internal BOS refuses consumption, terminal is legacy, evidence shows
`function_cycle_active`; (v) fused/no-tools/typed-immediate shapes
byte-unchanged.

### Verdict

**BLOCK — tool turns are currently fatal under prepared-epoch mode, and
the candidate config ships that mode on.** The no-tools promotion
evidence (r2 green, reuse generations 4/5/6) stands and is unaffected.
Implement corrections 1–3 with tests (i)–(v), then rerun the actual-tool
gate alongside the standing promotion set.

## Addendum 137a — FC terminal contract fix implementation review (2026-08-10)

Verified in the tree. The implementation chose a **step-local snapshot**
design over my 137 recovery-entry re-baseline, and it is superior — finer
where it matters and stricter everywhere else:

- **Terminal-step exactness**: the engine captures
  reset/transition/reuse plus logical/physical/session/prepare identity
  immediately before each generate_step and after agent BOS
  (`_eartts_epoch_step_snapshot`, engine call sites 3372/3466); the
  deferred terminal validates exactly-one-transition against its own
  step's before-snapshot — no lifecycle hook, no new counters, immune to
  anything the FC cycle did earlier.
- **Nothing is ignored**: the whole-turn counters are retained with their
  accounting identity (`transition == reset + reuse`), and the new
  **preterminal FC invariant** requires the pre-terminal window to be all
  legacy (`reuse == 0` and `transition == reset`) — internal transitions
  are constrained, not merely excluded, which is stronger than my
  proposal.
- **The ABA machinery is bound into the terminal contract**: a legacy
  terminal must rotate the backend id with generation exactly +1; a reuse
  terminal must preserve id and generation; step epoch/request/backend
  identity mismatches each carry their own violation code.
- **Correction 3 fully delivered**: `reset_session` zeroes prepare,
  failure, reuse, and transition counters plus the EarTTS
  `_reset_on_bos_count` — the cross-session carry-over that misled the
  137 evidence reading is gone.

**On consume-refusal (137 correction 2): the explicit hook is waived,
with reasons.** The structural claim holds on inspection — the boundary
SOTC position carries a non-PAD effective token, so the C2a bypass is
ineligible there, the wrapper's `mark_real_call` fires, and the epoch is
dirty before any FC-internal BOS can observe `clean_armed`. Decisively,
the property is now *enforced at validation* rather than assumed: an
internal reuse would trip `preterminal_reuse_delta != 0` fatally, with
evidence — and the internal-reuse mutation test pins that red path.
Enforcement-with-evidence beats prevention-without; no hook needed.

**Tests and suites**: the settlement reproduction mirrors the 137 trace
shape (two internal legacy resets, then the final legacy BOS with a
gen 9→10 rotation), plus zero/double/internal-reuse/stale-physical
mutations; immediate/fused shapes byte-unchanged; focused 207 + 28
subtests, full CPU 717, Ruff clean — all rerun here.

### Verdict

**APPROVE — proceed to the rebuild and rerun the actual-tool gate**
alongside the standing promotion set (browser E2E, canonical 3-turn
no-tools + ASR, actual tool + ASR). The 137 BLOCK lifts contingent only
on that live tool run reproducing this contract green on-image.

## Addendum 137b — C2b final promotion and deployment adjudication (2026-08-10)

Every claim rederived from the retained artifacts. **Identity chain
closed**: image `sha256:97f6f9d2…` is present as production-candidate-1
and its source label `4f84836a…` equals this tree's independent
recomputation — valid even after the docs edits, which are outside the
payload inventory. Host full suite 718 + 15 skipped + 29 subtests green
rerun here; Ruff clean.

**The formerly fatal tool shape is green live, in exact detail.** Trace
`6ca259b4…` rederives: session counters start 0/0/0 (the 137 zeroing
working — a fresh, legible baseline), then exactly three legacy
transitions — generations 5 and 6 inside the FC cycle (preterminal deltas
`reset = transition = 2, reuse = 0`, precisely the 137a invariant) and
generation 7 as the terminal with step-local `(transition, reset, reuse)
= (1, 1, 0)` and the backend rotating 6→7 as the legacy mode demands.
Every event carries the honest `missing_prepared_snapshot` fallback
reason (typed turns never arm). One request/call/EOTR/post-FC BOS, no
errors, clean `client_stop`. The tool response is the scenario-v2 payload
spoken correctly — "The fresh verification word is River." — transcribed
by the promoted immutable evaluator at WER 0, semantic true.

**The rest of the promotion slate**: canonical no-tools 3/3 with
semantics and the input-WER gate green; all four retained WAVs (three
no-tools + one tool) speech-classified, zero near-silence, text-channel
WER 0.0, semantics true under evaluator `sha256:59854214…`; browser real-
microphone E2E green with typed math and voice-to-typed sapphire recall,
V2V 1425.6 ms, TTFB 0.6482 s, TTFA 0.8967 s — TTFA − TTFB = 248.4 ms,
the invariant leading silence, metrics coherent.

### Promotion vs the latency target — kept distinct

**C2b is correct and PROMOTED; the 400 ms target is not met and is not
claimed.** Warm V2V stands at ≈1.32–1.43 s. The remaining budget is
dominated by terms this series has already adjudicated as pinned or
out of scope: the ten-blank fence (~420–510 ms, pinned by the 0/5/10
isolation evidence), the client endpoint window (~370 ms, Smart-Turn/VAD
scope), EarTTS generated leading silence (~248 ms, generation-time — the
130a retraction stands), and downstream playout (~120 ms). The
C1→C2a→C2b sequence delivered its measured components (eager start,
zero-cost idle positions, prepared-epoch reuse where armable, ABA
elimination with unique physical IDs); further reduction requires
re-opening decisions this record has deliberately closed, and any such
re-opening must arrive as a new prospective design, not as pressure on
the promoted gates.

### Verdict

**APPROVE — C2b is promoted and image `sha256:97f6f9d2…` is the deployed
production candidate.** The latency workstream's promotion chain
(A122–A137b) is closed: every optimization shipped is live-proven under
its own preregistered contract, every red predecessor is retained and
hash-addressed, and the unmet 400 ms target is recorded as unmet rather
than redefined.
