# Fable adversarial plan review — Pocket TTS text-input bridge

Plan: `docs/pocket-tts-text-input-plan.md` · Date: 2026-08-04 · Read-only review;
no code, services, or deployment touched. Verdict at end.

Sources verified (pinned Pipecat core at `/home/khkramer/src/nvidia-voice-chat/pipecat`,
this repo, Pocket TTS source at `/home/khkramer/src/nemotron-nano-omni/pocket-tts`):
`rtvi/processor.py:468-494` (`_handle_send_text`), `rtvi/models.py:230-231`
(defaults `run_immediately=True`, `audio_response=True`), `frames.py:1017`
(`InterruptionFrame(SystemFrame)` — no source field), `llm_response_universal.py:1114/1624`
(append consumption), `pipeline/worker.py:411-479` (RTVI prepended before the
pipeline), `base_input.py:259` (background-task `push_frame` precedent),
`soxr_stream_resampler.py:84-98` (rate-switch behavior), this repo's
`llm.py:128,181-192,290-320,348-353`, `demo.py:84-115`, `aggregators.py:39-46`,
`.env` (`VOICECHAT_RNNT_EOU_FRAMES=20`), `server.py:1700-1704`, and Pocket TTS
`tts_model.py:233-560` + README (CPU-only, ~2 cores, ~6× realtime on M4).

---

## Blockers

### B1 — The FIFO queue promise contradicts the plan's own interruption rule (sharpest race)

Plan line 125: "A second typed submission waits in order; it does not overlap
generation." Plan lines 126-128: an `InterruptionFrame` "cancels generation,
clears pending jobs, stops synthetic audio."

Under the Playground default (`run_immediately=True`, models.py:230), a second
send-text calls `interrupt_bot()` **before** pushing the append
(processor.py:471-473). The resulting `InterruptionFrame` is a bare
`SystemFrame` with no source field (frames.py:1017-1027; models.py:603-612
confirms programmatic and VAD interruptions are indistinguishable), and RTVI is
prepended ahead of `transport.input()` (worker.py:479), so it reaches the
bridge **before** append #2. Per the plan's own rule it cancels active job A
mid-injection and clears the queue — the FIFO branch is unreachable in the
default configuration.

The deeper failure: cancelling mid-injection leaves the server-side RNNT with
unterminated partial speech. Job B's audio then arrives inside the 1.6 s EOU
window, so RNNT merges partial-A + B into one garbled user turn, while the
Pipecat context holds two clean typed messages — context/model divergence that
poisons subsequent turns.

**Smallest correction:** define the semantics as **cancel-then-replace** under
`run_immediately` (drop or explicitly gate the FIFO claim on
`run_immediately=false`), and require that whenever an actively-injecting job
is cancelled — by a second send **or** by a live barge-in — the bridge first
emits the turn-closing trailing silence before starting the next job or
un-ducking the microphone. Rewrite E2E case 6 to assert cancel-then-replace,
not queuing.

### B2 — Default trailing silence exactly equals the deployed EOU threshold; zero margin

Deployed `.env` sets `VOICECHAT_RNNT_EOU_FRAMES=20` → 20 × 80 ms = 1.6 s
(server.py:1700-1704 maps this to `rnnt_eou_frames`, `asr_eou_first_turn`, and
`transport_vad_eou_frames`). Plan line 159 defaults trailing silence to exactly
1.6 s. Frame-boundary alignment, 20 ms pacing jitter, and resampler settling
mean the 20th consecutive blank frame is not guaranteed to complete inside the
injected silence; turn closure then depends on post-un-duck room audio being
silent. In a noisy room the typed turn may close late or never.

**Smallest correction:** derive the default from the deployed setting with
margin — `VOICECHAT_RNNT_EOU_FRAMES × 80 ms × 1.25 + one injection frame`
(= 2.0 s at the current 20-frame profile; 1.0 s at the interactive 10-frame
default) — and plumb the env var (or an explicit bridge setting sourced from
the same deploy config) into the Pipecat unit so the two values cannot drift.

---

## Important

### I1 — `audio_response` IS observable; remove the plan's "if it cannot be observed" branch

Plan lines 142-146 hedge that the option may not be observable "without
changing read-only Pipecat core." False: an explicit `audio_response: false`
send pushes `LLMConfigureOutputFrame(skip_tts=True)` immediately **before** the
append and restores it after (processor.py:479-494), and these frames flow
through `transport.input()` to the bridge in order. The bridge can track the
latest `LLMConfigureOutputFrame` value and reject an append arriving while
`skip_tts=True` with the plan's non-fatal unsupported-operation event, without
forwarding the append (no context entry) and without synthesizing. Note the
default send (`audio_response=True`, models.py:231) pushes **no** toggle
frames, so the observation is purely for the explicit-false case. Commit to
this mechanism and delete the fallback branch.

### I2 — `run_llm=false` appends silently diverge context from the model

Plan line 90: "If `run_llm` is false … update context but do not synthesize."
The checkpoint has no user-text tensors — a context-only message is invisible
to the model forever. Later turns then run against a Pipecat context the model
never saw. **Correction:** reject `run_immediately=false` sends with the same
non-fatal unsupported-operation event used for `audio_response: false`
(the plan already establishes that precedent), or document the divergence in
capital letters as accepted. Rejection is the clean choice for release 1.

### I3 — Ducking must begin at append-forward time, not at first synthesized sample

Plan line 71 says "duck live microphone during injection." Pocket TTS
time-to-first-chunk on the Grace CPU will be tens to hundreds of ms; if
ducking begins when the first synthetic frame is pushed, live mic frames
interleave ahead of the typed turn during that gap. **Correction:** specify
that the bridge sets its duck flag synchronously inside `process_frame` when it
forwards the append/enqueues the job (dropping mic `InputAudioRawFrame`s at
the bridge — deterministic ordering), and releases it only after trailing
silence completes or cancellation-closure runs (per B1).

### I4 — Prewarm/offline reproducibility needs concrete deploy hooks

`TTSModel.load_model` (tts_model.py:233) downloads weights and voice prompts
from HF Hub. **Correction:** pin the pocket-tts version in `uv.lock`, prewarm
the HF cache in the deploy flow (`deploy/check-environment.py` can verify the
cache exists), warm the cached voice state at unit startup (the
`PocketTTSService._get_voice_state` `asyncio.to_thread` pattern, pinned
`tts.py:174-185`), and gate unit readiness on one throwaway synthesis so the
first typed message never pays model-load latency.

### I5 — CPU contention: cap threads and gate on live-session RTF

Pocket TTS is CPU-only by design (README: GPU showed no speedup; ~2 cores;
~6× realtime on an M4). The model container reserves 2 codec cores
(`EA_CPU_CODEC_THREADS=2` in `.env`) in a separate process; the bridge
competes host-wide with the container and in-process with SmallWebRTC
encode/decode. On the 20-core Grace this is feasible, but **correction:** set
`torch.set_num_threads(2)` in the TTS backend and add a phase gate measuring
synthesis RTF ≥ 2× realtime **while a live session is streaming**, plus a
check that the model service's pair-latency gate still passes during synthesis.

---

## Suggestions

- **S1 — Inject at 16 kHz.** 24 kHz injection works: the service resamples
  every input frame (llm.py:348-353), and the shared `SOXRStreamAudioResampler`
  never sees a rate flip because 16 kHz mic frames short-circuit on
  `in_rate == out_rate` before stream initialization, and `clear_after_secs=0.2`
  resets state between jobs (soxr_stream_resampler.py:99-117). But resampling
  once in the bridge worker to 16 kHz removes the dependence on that
  short-circuit subtlety and takes VHQ soxr work off the per-frame hot path.
- **S2 — Discard-stale-output guard.** `asyncio.to_thread` synthesis cannot be
  interrupted mid-call; cancellation must mark the job generation stale so a
  completing thread's chunks are dropped rather than pushed. State this in the
  concurrency section.
- **S3 — E2E matrix additions.** The real-Playground/SmallWebRTC approach is
  right and proven feasible (the repo already runs a real-Chromium browser
  gate). Add: explicit `audio_response: false` rejection; typed send while the
  user is actively speaking into the mic; EOU closure with continuous room
  noise after un-duck (exercises B2); and rewrite case 6 per B1.

---

## The ten determinations

1. **Exactly-one context message — YES, structurally sound.** RTVI is
   prepended before the pipeline (worker.py:479); the append passes
   `transport.input()` to the bridge; the bridge forwards once; the stock
   `LLMUserAggregator` (aggregators.py:41) **consumes** the frame
   (`add_messages` + one downstream `push_context_frame` when `run_llm`,
   llm_response_universal.py:1114) — it is not re-forwarded. The service's
   non-first `_handle_context` only syncs function-call results
   (llm.py:290-320), so the grown-message context frame is benign, and the
   service's append rejection (llm.py:185-192) never fires on this path — keep
   it as defense-in-depth; the plan's phase-3 "relax the rejection" note is
   unnecessary.
2. **Background `push_frame` + `InputAudioRawFrame` subclass — YES, safe.**
   Exact precedent: `BaseInputTransport` pushes `InputAudioRawFrame` from a
   managed background task (base_input.py:259→193). Requirements: create the
   worker via `self.create_task` and cancel it in the processor's own
   `cleanup` (the base class does not auto-cancel). A bridge-owned subclass
   passes every downstream `isinstance` check on the base class (llm.py:181);
   only the bridge branches on the subtype.
3. **Interruption/cancel/end races — B1 is the finding.** Two rapid sends:
   cancel-then-replace with silence-closure required. Barge-in during
   synthetic playback: VAD runs in `transport.input()` upstream of the bridge,
   so the `InterruptionFrame` arrives and cancels the job — same partial-turn
   garble, same B1 closure fix. `CancelFrame`/`EndFrame`: forward promptly,
   never block `process_frame` on job completion, and apply S2's stale-output
   guard.
4. **Ducking before first sample — NOT as written.** See I3: duck at enqueue.
5. **Trailing silence ≥ deployed EOU — NO.** Deployed profile is the manual
   20-frame/1.6 s setting; plan default equals it exactly. See B2.
6. **`audio_response` — supportable as always-voiced; observable.** See I1:
   the skip_tts bracket is visible at the bridge; reject explicit-false
   non-fatally; document that responses are always voiced.
7. **Dependency/offline/prewarm — feasible with I4's additions.**
8. **CPU contention — feasible with I5's caps and live-session gate.**
9. **E2E matrix — right approach, feasible; amend per S3 and B1.**
10. **Reuse vs custom, singleton vs per-browser:** A **custom bridge is
    correct**. `PocketTTSService` is a bot-side `TTSService`: it emits
    `TTSAudioRawFrame` with `TTSStartedFrame`/`TTSStoppedFrame` brackets
    downstream of an LLM — the wrong pipeline side, wrong frame types, wrong
    lifecycle; adapting it means suppressing most of its behavior. But do not
    reimplement its internals: compose them — `TTSModel.load_model`, the
    cached voice-state via `asyncio.to_thread` (tts.py:174-185), and the
    `generate_audio_stream(copy_state=True)` + sync-next-in-thread streaming
    pattern (tts.py:198-224). **Process-wide singleton backend
    (model + voice state), per-connection bridge processor instance** is the
    right split: load once (seconds, large RSS), keep per-session queue/duck
    state in the processor. `copy_state=True` makes generations state-safe,
    but keep the single serialized worker — the stack is single-session and
    synthesis is CPU-bound.

---

## Verdict

**APPROVE WITH CHANGES.** The architecture is sound — the injection path,
frame contract, and exactly-once context handling all check out against the
pinned core and this repo. B1 (cancel-then-replace + silence-closure on
mid-injection cancellation) and B2 (trailing-silence margin derived from the
deployed EOU setting) must be folded into the plan before implementation;
I1-I5 are required plan amendments that do not change the architecture.
