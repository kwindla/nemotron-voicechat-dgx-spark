# Step 2 live-phase preregistration

Registered 2026-08-13, before any live fixture was sent. Governing
documents: `docs/generation-frame-latency-plan.md` (Step 2 gates) and
`docs/reviews/step2-implementation-review-r4.md` (preconditions and
evidence-retention manifest). This file's git commit timestamps the
preregistration.

## Runtime identity

- Image: `pipecat-ai/nemotron-voicechat-dgx-spark:generation-step2`, built
  from commit `dd5f0f8` (digest recorded in the session manifest at run
  time).
- Environment: production-candidate-1 managed environment exactly, with
  precisely three deltas:
  - `VOICECHAT_QUALIFICATION_MODE=step2-longform-fixture`
  - `VOICECHAT_QUAL_NO_TEXT_WATCHDOG_OVERRIDE_FRAMES=160` (12.8 s no-text
    tail allowance)
  - `VOICECHAT_WEB_SYSTEM_PROMPT=Qualification latency fixture: when the
    user supplies a script, speak the supplied script aloud completely and
    exactly, then stop.`
- Host Pipecat from the same checkout with
  `NEMOTRON_VOICECHAT_PLAYOUT_TRACE=<session>/pipecat-playout.jsonl`.
- Deployed prebuffer for this run: repo default 160 ms (Step 3, not this
  run, decides changes).
- Transport: local loopback; Playwright headless Chromium with fake mic;
  typed input path for scripts. No tunnel.

## Fixtures (typed input; two independent responses per script)

- Script L1 (long bin, target 8–10 s spoken): "The quick brown fox jumps
  over the lazy dog while seventeen green dragons circle the ancient stone
  tower, counting every window, every door, and every flag that flies
  above the northern gate."
- Script L2 (long bin): "On Tuesday morning the harbor filled with small
  boats carrying baskets of silver fish, and the old lighthouse keeper
  wrote down the number of gulls, the color of the clouds, and the exact
  minute the tide turned."
- Script S1 (short control, ~3 s): "The red balloon rose over the quiet
  village square."
- Interruption fixture I1: send Script L1; at approximately 4 s after
  first response audio, send typed request "Stop. What is two plus two?"
  (exercises typed cancel-then-replace and the interruption-clear trace
  path). One run.
- Capture-off control: Script L1 once with browser capture disabled
  (paired `BROWSER_PLAYOUT_CAPTURE_OVERHEAD` comparison).

Run order: S1, S1, L1, L1, L2, L2, I1, then capture-off L1.

### RTVI transcript identity boundary

The installed Pipecat 1.7.1.dev21 protocol (`PROTOCOL_VERSION = 2.1.0`)
defines `BotTranscriptionMessage` as only `label`, `type`, and
`data: TextMessageData`; `TextMessageData` contains only `text`. The bundled
Pipecat client dispatches only that `data` object to `onBotTranscript`. There
is therefore no response ID, item ID, or message ID that the fixture driver can
bind to the traced response.

The driver instead uses the ordered RTVI data channel plus a fixed
post-terminal quiescence barrier. After every fixture's
`bot-stopped-speaking`, it admits no next typed send until a 2000 ms window
(`VOICECHAT_LIVE_RTVI_QUIET_WINDOW_MS`, default `2000`, valid range
`500..60000`) completes with no `bot-*` message. Each result retains the window
start/end, total-message counts, and bot-message counts/types. Both retained
clock durations must cover the configured interval; the wall-clock comparison
allows at most 1 ms for `Date.now()` quantization. A barrier with any bot activity fails the
fixture. Every accepted speaking/transcript bracket must also begin strictly
after its own retained typed-send edge. Pipecat emits the accumulated
`bot-transcription` before the terminal speaking transition, and RTC data
channels deliver messages in order; consequently all text sent for the prior
response must be delivered before that prior terminal. The terminal-aligned
quiet window then proves the channel stayed drained until the next send, so
prior-response text cannot enter the next fixture's bracket, including when
the two scripts have identical bytes.

## Preregistered gates (Step 2 scope)

1. **Fixture validity / audio completeness** (per long-bin response),
   adjudicated by the retained ASR evaluator image
   (`pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b`) over the
   rendered response audio: (a) tail completeness — the final 8 content
   words of the script appear, in order, in the evaluator transcript;
   (b) WER(script, transcript) ≤ 0.30 after standard normalization.
   Text-channel transcript alone is insufficient by construction.
2. **Watchdog evidence**: per-frame watchdog snapshots carry the
   qualification identity; no `no_text_since_bos_watchdog` EOS inside an
   audible long-bin rendition.
3. **Trace validity**: Pipecat artifact terminal exactly successful
   (`valid=true`, null reason/error, `dropped_records=0`);
   `observed_playout_trace.valid=true` from the checked consumer;
   provable delivered/delta reconciliation.
4. **Browser reconciliation**: per-worklet persisted last dequeue ordinal
   equals acknowledged final ordinal; browser underrun-candidate and
   onset numbers reported and reconciled qualitatively against the Step 0
   replay of the same session (quantitative underrun bounds are Step 3
   scope, after the reserve decision).
5. **Structural/queue reporting** (informational for this fixture):
   normative arrival-clock K=12 and max-gap results; full reserve sweep
   grid 0–560 ms step 80.
6. Interruption run: trace remains valid; no post-clear audio accepted;
   cancel-then-replace completes; second answer delivered.
7. Every fixture has a successful retained RTVI ordered-channel quiescence
   barrier, and every response bracket is strictly after its own send edge.

Failure of gate 1 on both long-bin attempts of a script is a finding
(fixture-design or model behavior), not silently retried; everything is
retained either way.

## Evidence retention

One immutable session directory `reports/step2-live/session-<utc>/` with
SHA-256 manifest covering: this preregistration, image digest + source
commit, container env, pre-fixture `/health` (including qualification
identity) and single-client check, Pipecat and browser JSONL artifacts,
live-test stdout/stderr with both capture-overhead reports, server
event/metrics traces and input/output audio, evaluator transcripts and
adjudications, and `arrival-analysis.json` with command parameters.
The driver retains the no-follow session-directory descriptor and its
`(st_dev, st_ino)` reservation until terminal summary publication. Browser
artifacts and atomic summary children are opened relative to that descriptor;
every necessarily pathname-based Pipecat read, publication poll,
reconstruction, and terminal-set validation first requires the current parent
pathname to match the reserved identity.
