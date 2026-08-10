# Fable live E2E — standalone stack, Pipecat debug UI (SmallWebRTC)

Date: 2026-08-04 04:19–04:21 UTC · Public origin: `https://5c70541c5b0c.ngrok.app/client/`
Model trace session: `bf048167-27c8-49ae-bbaf-ac0f67d09e7f` · Evidence:
`/tmp/fable-final-e2e-20260804/` (combined mic WAV + timeline, console log,
RTP stats, screenshot, archived trace events). No services restarted or
reconfigured; single real-Chromium session via the actual Pipecat prebuilt UI.

## Verdict: PASS — all assertions

One 83.5 s fake-microphone timeline (15 s silence → 10 s −35 dBFS noise → three
Pocket TTS turns with 12 s gaps → 20 s tail) driven through real Chromium
`getUserMedia` → SmallWebRTC → Pipecat → strict-v2 WebSocket → model and back.

| Assertion | Result |
|---|---|
| Public page loads (200) | ✓ |
| WebRTC connects (7 s) | ✓ |
| 15 s silence — no self-trigger | ✓ zero agent activity |
| 10 s −35 dBFS noise — no self-trigger | ✓ zero agent activity (first delivered audio is turn A's response) |
| Turn A "code word sapphire / two plus three" | ✓ "The code word is sapphire. Two plus three is five." |
| Turn B memory recall | ✓ "The code word is sapphire." |
| Turn C explicit tool prompt | ✓ REAL `get_current_utc_time` call emitted (`call_38f…`) |
| Tool result returned | ✓ 89-byte output, round trip **0.002 s** |
| Result spoken correctly | ✓ "The current UTC time is four hours twenty minutes and fifty-eight … seconds" — matches actual 04:20:58 UTC |
| Model audio delivered/rendered | ✓ 158,899 inbound RTP bytes, live audio track, audio element playing (t=100.3 s) |
| Clean teardown | ✓ UI disconnect → Connect restored |
| `/health` `active_client` after | ✓ `false` |

Notes: outbound RTP 213,614 bytes (mic path); agent turns opened at model
frames 63/240/557 with no activity before frame 63, proving both no-trigger
phases; the spoken time's verbosity (full fractional seconds) is model phrasing,
not an error. The public origin serves without authentication — posture is
documented in the repository's deployment guidance.
