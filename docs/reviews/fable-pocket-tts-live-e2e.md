# Fable live E2E — Pocket TTS typed-input bridge (Playground send-text)

Date: 2026-08-04, three sessions — text-only passes 16:00:39–16:02:08 UTC and
16:05:24–16:06:56 UTC, mixed voice+text 16:11:30–16:12:29 UTC · Public
origin: `https://5c70541c5b0c.ngrok.app/client/`
Session ID: `261d7c09-8b5d-425f-88f…` (RTVI client v1.13.0 / server v2.1.0) · Evidence:
`/tmp/fable-pocket-tts-live-e2e/` (driver script, full bidirectional RTVI
data-channel capture, per-turn RTP stats, 84.8 s Opus recording of assistant
audio, screenshots per turn, UI conversation text, console log, timeline).
No services restarted or reconfigured; one real-Chromium session through the
actual Pipecat Playground UI and its own send-text control over SmallWebRTC.
The live microphone stayed connected the whole session, fed by a silent fake
capture device, so microphone arbitration was genuinely exercised.

## Verdict: PASS — all required assertions, twice consecutively

Four typed turns (three-turn memory flow, then a typed tool request) drove
real `send-text` RTVI messages through the Playground input box →
SmallWebRTC data channel → PocketTTSInputBridge → synthetic 16 kHz user
speech → RNNT → native Voicechat turns, with responses returned as native
assistant text and audio.

| Assertion | Result |
|---|---|
| Public page loads; WebRTC connects; `bot-ready` | ✓ connected in ~3 s, `bot-ready` immediately |
| 10 s pre-conversation silent-mic window | ✓ zero turn/transcription events |
| Exact typed message appears once | ✓ 4/4: one `user-llm-text` context event per turn, byte-exact; UI shows each typed message exactly once |
| RNNT transcription semantically equivalent | ✓ 4/4 (differences limited to casing/terminal punctuation, e.g. "…current Utc Time.") |
| Each turn closes | ✓ 4× clean `user-started` → `user-stopped` (final transcript) → `bot-started` → `bot-stopped`; no overlap, no stuck state |
| Assistant text arrives | ✓ per-turn `bot-transcription`/`bot-llm-text` (R1 "Your code word is Indigo Horizon. I will remember that." … R4 "It is currently sixteen o one UTC.") |
| Assistant audio arrives | ✓ inbound RTP deltas 43.8/83.1/32.2/33.0 kB per turn; decoded session recording is real speech (Opus 48 kHz, peak 0.22) |
| Turn-3 memory recall | ✓ "Your code word is Indigo Horizon." |
| Typed `get_current_utc_time` request | ✓ one REAL call (`call_a5c9504d…`), result in 0.007 s, `cancelled: false` |
| Spoken tool answer correct | ✓ "sixteen o one UTC" vs. actual 16:01 UTC at call time |
| No extra microphone turn | ✓ exactly 4 user turns for 4 typed messages; silent live mic produced none before, between, or in the 8 s window after the last turn |
| Clean browser-owned teardown | ✓ Disconnect → Connect restored; zero console errors/page errors all session |

Typed-submit latencies (send-text → event): synthetic speech opened at
+4.4/+2.4/+1.6/+2.5 s; RNNT turn close (includes the documented 2.08 s
trailing-silence cost) at +8.0/+5.6/+3.8/+16.1 s; bot audio started ≤20 ms
after each turn close.

## Mixed voice+text session (16:11:30–16:12:29 UTC): PASS

An abbreviated third session (artifacts in `run3-mixed/`) closed the mixed
voice+text gate. A real fake-microphone timeline (20 s silence, then the
Pocket TTS replay fixture `turn-math.wav` "What is two plus three?" played
once, then silence) drove a genuine voice turn: RNNT transcribed exactly
"what is two plus three" and the bot answered "Two plus three is five."
A typed follow-up in the same session — "What was the answer to the math
question I just asked by voice?" — produced exactly one byte-exact context
item, a semantically equivalent RNNT transcript of the synthetic speech, and
the reply "The answer was five.", proving one continuous model session
across modalities. Exactly 2 user turns and 2 bot turns occurred (no extras
during the leading-silence or post-session windows), zero errors, and the
browser disconnected cleanly.

Pass 2 (artifacts in `run2/`) repeated the identical flow and matched pass 1
on every assertion: 4 `send-text` → 4 byte-exact once-only context items →
4 closed turns, zero errors, memory recall intact, one real tool call
(`call_fec2ab6f…`) with the spoken answer "sixteen o six UTC" matching the
actual 16:06 UTC, and a clean disconnect. An input inventory captured via
DOM inspection before/after connect confirms the send control's contract:
disconnected UI shows a disabled input with placeholder "Connect to send";
connected UI enables it as "Type message..." (rendered once per tab panel;
only one instance is visible).

Notes: each typed turn shows two `bot-interrupted` events (RTVI
`run_immediately` interruption at append acceptance, then the standard
speech-start interruption) — expected semantics, not duplicate turns. The
tool turn's close was late (+16.1 s) because the model held the terminal
transcription/response until ~8.6 s after the instant tool result; content
and ordering were still correct — model-side pacing, not a bridge defect.
Turn 2's reply declined travel planning per the system prompt's strict tool
policy; the turn itself closed normally and did not disturb memory recall.
Out of scope for this live qualification: barge-in/replacement closure and
`audio_response:false` / `run_immediately:false` rejection are covered by
the automated suite; a real mixed voice+text live session is now covered by
the session above. Synthesis-error recovery and disconnect-mid-synthesis
are covered by deterministic component gates
(`test_synthesis_failure_reports_error_and_recovers_for_next_typed_turn`,
`test_disconnect_during_synthesis_cancels_job_and_releases_microphone`;
focused bridge suite 10/10, independently re-run 2026-08-04). Those two
failure behaviors were not injected into the live public-model browser
sessions and therefore carry component-level, not live-session, evidence.
