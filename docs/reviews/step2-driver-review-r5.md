# Gate review: Step 2 live fixture driver, round 5

Review date: 2026-08-13 UTC. Scope: the Step 2 preregistration (including
the RTVI identity boundary and gate 7), the two round-4 deltas, the round-4
fix note, the live fixture driver and its browser orchestration, and the
installed Pipecat 1.7.1.dev21 source/client bundle.

## Verdict

**FAIL — the driver is not qualified and the live campaign must not execute.**

The two round-4 defects are substantively closed: the installed protocol and
ordered channel support the repeated-script drain boundary, and the retained
session descriptor protects both browser sessions and terminal summary writes
while pathname-based Pipecat operations reject a changed parent identity. One
new gate-7 fail-open remains in the barrier's configuration and retained
arithmetic.

## Remaining delta

### 1. A zero-length barrier passes, and its retained wall duration is not checked

The live barrier accepts `quiet_window_ms == 0`
(`tests/e2e/test_browser_smallwebrtc_e2e.py:935-940`) and immediately samples
the channel. With no already-queued bot message it returns `passed=true`
(`:941-969`). The completed-summary consumer independently permits the same
zero interval (`tools/qualification/step2_live_fixture_driver.py:1659-1661`).
A fresh probe supplied an empty post-terminal slice at zero milliseconds; the
runtime returned a successful barrier and `_validate_quiescence_barrier`
accepted it. Therefore
`VOICECHAT_LIVE_RTVI_QUIET_WINDOW_MS=0` removes the preregistered quiescence
interval while still satisfying gate 7.

The retained arithmetic has a second inconsistency at the same boundary. The
consumer requires
`window_end_performance_ms >= window_start_performance_ms + quiet_window_ms`,
but checks only `window_end_wall_time_s >= window_start_wall_time_s`
(`tools/qualification/step2_live_fixture_driver.py:1696-1699`). A fresh
mutation claiming a 2000 ms barrier with a correct performance-clock duration
but identical wall-clock start and end was accepted. Thus one of the two
retained duration claims can be false without invalidating the terminal
summary.

Negative, non-finite, and greater-than-60000 values were rejected by the
barrier function. Negative values are nevertheless parsed at the per-fixture
call site (`tests/e2e/test_browser_smallwebrtc_e2e.py:3395-3414`) and are not
validated until after a response has already reached its terminal. Invalid
barrier configuration should fail before path reservation, browser setup,
connection, or any fixture send.

Required delta: parse and validate the environment value once during live-test
preflight, require a finite `0 < quiet_window_ms <= 60000`, and pass only that
checked value to every fixture. Make completed-summary validation enforce the
same positive range and require both retained end clocks to cover the claimed
interval, including
`end_wall >= start_wall + quiet_window_ms / 1000`. Add direct negatives for
zero, negative, non-finite/non-numeric configuration and for a shortened wall
interval. Tests may inject a positive short interval; they must not use zero
as a successful barrier surrogate.

## Fresh protocol and adversarial record

- Installed `PROTOCOL_VERSION` is `2.1.0`. `BotTranscriptionMessage` has only
  `label`, `type`, and `data`; `TextMessageData` has only `text`. The bundled
  client dispatches `e.data` to `onBotTranscript`, so no response identity is
  available on the transcript payload.
- The installed observer's synchronous probe emitted `bot-llm-text`, then
  accumulated `bot-transcription`, then `bot-stopped-speaking`. The canonical
  one-sentence fixtures end in punctuation, so their accumulated transcript is
  emitted from `LLMTextFrame` before the output transport's terminal speaking
  transition. SmallWebRTC sends messages immediately or flushes its pending
  list in order, and the bundled client explicitly creates `chat` with
  `{ordered: true}`.
- All 14 installed `Bot*Message` wire types begin with `bot-`; the driver's
  prefix filter missed none. Fresh `bot-output`, `bot-llm-text`, `bot-tts-text`,
  `bot-audio-level`, and `bot-ready` samples were all counted. A
  `bot-transcription` timestamped exactly at the window end was also counted
  and failed the barrier.
- Repeated-S1 probes used the exact canonical bytes. A late S1 transcript in
  the preceding barrier failed; one arriving after the barrier but before the
  next baseline failed the attribution cursor; a pre-start duplicate was not
  admitted to the current transcript bracket; a late duplicate plus the owned
  current transcript failed joined-text equality. The otherwise indistinguishable
  inside-bracket replacement remains content-valid in isolation, as expected,
  but the verified Pipecat send order plus ordered data channel makes that
  insertion unreachable after a successful positive barrier.
- A directory swap between capture-on and capture-off writes left both browser
  artifacts and the summary in the descriptor-owned displaced directory; the
  attacker directory remained empty. A second swap between publication polls
  preserved the pending artifact in the retained directory and the next poll
  rejected the changed pathname parent. A later Pipecat read likewise failed
  closed. Descriptor retention is closed at the requested boundaries.

Probe outputs are retained in `/tmp/step2-driver-r5-barrier-probes.txt`,
`/tmp/step2-driver-r5-s1-insertion-probes.txt`,
`/tmp/step2-driver-r5-pipecat-protocol-probe.txt`, and
`/tmp/step2-driver-r5-descriptor-probes.txt`.

## Verification record

No `browser_e2e` entry, live URL, browser campaign, running stack, container,
service endpoint, commit, or repository file other than this review was
touched.

- Full offline suite with `-m 'not browser_e2e'`: **1022 passed, 15 skipped,
  2 deselected, 29 subtests passed** (four existing warnings).
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff over the service, driver, analyzer, browser driver/tests,
  runtime Step 2/driver/analyzer tests, and playout-trace tests: **passed**.
- Task-scoped `git diff --check`: **passed**.
- `node --check` on the generated RTC probe, voice-latency probe,
  AudioWorklet installer, and extracted embedded worklet processor: **passed**.
- Trace-off identity against `HEAD`: the base service has the same 71-method
  AST and method set; all 71 `co_code` values match, and `__new__`,
  `_handle_server_event`, `_audio_delta`, `_release_playout_buffer`, and
  `_clear_playout_buffer` retain identical constants and names.

Canonical anchors are retained at
`/tmp/step2-driver-r5-canonical-anchors.json`; the full suite and static-check
outputs are retained under `/tmp/step2-driver-r5-*`.
