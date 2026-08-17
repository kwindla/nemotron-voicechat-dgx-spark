# Session repair results — corrected acceptance adjudication — 2026-08-17

## Outcome

The first-party runtime repairs for I1 and I2 are implemented. I3 was measured
without changing a response boundary. The corrected browser replay is accepted:
completion, audible-watchdog behavior, response attribution, I3 runtime
classification, and rendered-audio ASR all pass their fail-closed gates.

Tool elicitation in the conversational replay is now reported as a model-propensity
metric, not used as an acceptance gate. The exact three-call requirement had been
conflating repair correctness with whether the model elects to call a tool for a
casual re-ask. Attempt 5 itself demonstrates that distinction.

A separate directive-phrased fixture now owns the I4 freshness gate. It elicited the
exact `get_current_time`, `set_timezone`, `get_current_time` sequence in 3/3
replicates before the grounding-instruction change and 3/3 after it. No assistant
text after the Pacific result reported the superseded UTC result in either arm:
0/6 stale before and 0/6 stale after. However, every replicate omitted the exact
fresh Pacific result on the post-tool response, so the corrected post-state gate
fails. The measured stale rate is also unchanged (0.0 to 0.0), and these results
**do not establish that I4 was fixed by the instruction change**.

## I1 — session termination versus an unclosed completed response

`response_completion_gate.py` now parses and validates the terminal
`session_trace_stopped` lifecycle record. An open response is recorded as
`session_terminated` only when a matching producer terminal claim follows its BOS
and precedes the stop, with a narrow exception for the real unclaimed transport
`disconnect` lifecycle. A clean `client_stop`, unknown or typo reason without a
claim, missing lifecycle evidence, duplicate stop, or records after the stop remain
fail-closed.

The retained trace passes the repaired gate with one recorded terminated open
response at BOS frame 975 and no schema errors. Unit mutations cover legitimate
termination, unknown/typo reasons, a duplicate stop, a completed-session open
response, malformed lifecycle, and the retained-trace case. This changes the old
false positive; it does not make absence of an EOS generic success.

Evidence: `retained-trace-completion-gate.json`.

## I2 — transcript attribution and its tool-affordance interaction

Root cause: the committed turn transcript lived only in the mutable RNNT
accumulator. On a tool-bearing turn, response BOS can be deferred until after
function-result/EOTR processing, while the accumulator is reset for the next user
turn before that BOS. Recognition succeeded, but the already recognized text had
lost its binding to the response.

The runtime now freezes the transcript and source identity at typed or microphone
commit, carries it independently through EOU/function settlement, and consumes it
exactly once at BOS. It does not restore frozen text into the live RNNT accumulator.
A BOS caused solely by a tool result is explicitly tagged
`post_tool_continuation`; a newer user commit supersedes an unopened continuation.
At the locked edge where a new microphone turn interrupts an unpublished deferred
function cycle, the runtime now clears the superseded frozen text/source before
audio can produce a pre-commit BOS. The publication-race/deferred-input branch
retains its binding. Endpoint tests cover both branches, typed and microphone
function-output/EOTR/forced-BOS recovery, and a two-tool continuation.

Fixing that attribution bug also changed observable tool-calling behavior:

- The original live buggy trace and attempt-5 browser replay each contain nine
  assistant responses. Their BOS frames track closely: live
  `[29, 185, 370, 547, 676, 788, 871, 942, 975]`; replay
  `[28, 191, 380, 502, 692, 805, 893, 963, 1005]`. This is evidence against replay
  desynchronization.
- The original live trace elicited all three calls. Its third call occurred in the
  response window at frame 547, where `user_text` was empty because of I2.
- Attempt 5, with I2 repaired, exposed the user text at the corresponding casual
  re-ask: frame 502, `okay cool what time is it now`. It elicited
  `get_current_time` at frame 28 and `set_timezone` at frame 191, but no third call
  at frame 502; the model answered from conversation memory: `It is currently
  eight fifty five PM UTC.`

Thus the missing third replay call measures the model's propensity under a
correctly attributed casual request. It is a real interaction between input
attribution and tool affordance, not evidence that the repaired replay drifted from
the live conversation. Attempt 5 has no recognized-speech attribution failures.

## I3 — unfinished sentence classification

The retained case splits as follows:

- Response 3, `Your time zone is`: `model_silence`; at least ten terminal decoded
  frames were silent before the existing silence boundary.
- Response 4, `It`: `model_silence`; the model stopped yielding audio and the
  no-audio boundary closed it.
- Response 8, `I am here if`: `model_turn_taking`; the model emitted its turn
  terminator.

Runtime-caused productive cutoffs: **0**. No response threshold or boundary
contract was loosened. Attempt 5 and the corrected replay also report zero
runtime-caused incomplete responses.

## I4 — directive-phrased freshness gate

The runtime and demo grounding instructions require the exact **MOST RECENT** tool
result for the capability and prohibit restating a superseded result. The original
retained live session remains useful descriptive evidence: after its later Pacific
result, 3 of 6 eligible assistant responses restated the superseded UTC result
(50%). That session also contains the I2 attribution defect, so it is not the
instruction-only control.

`tools/qualification/session_directive_freshness_fixture.py` is the separate
instruction-controlled I4 gate. It uses explicit directive carriers, production
tool schemas, deterministic results (`8:55 PM UTC`, then `1:55 PM PDT`), and a
final no-tool restatement turn. It fails unless each replicate:

- emits exactly one correctly shaped call on each directive turn and none on the
  final restatement turn;
- applies the Pacific-converted result and completes the response/session
  lifecycle cleanly; and
- states the fresh Pacific result in both the post-tool response and the no-tool
  restatement, with no later assistant text containing the superseded UTC result.

The matched pre arm used the repaired r3 runtime with only the new grounding clause
removed. Its retained control image is
`pipecat-ai/nemotron-voicechat-dgx-spark:session-repair-directive-pre-control`
(`sha256:262f10ed7edfb55afb31ec1d5134533b6ac173987bf23556c100ed03927635df`).
The post arm used the final r3 image.

Results:

| Arm | Exact sequence | Eligible responses | Stale responses | Stale rate | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Pre-instruction control | 3/3 | 6 | 0 | 0.0 | fail (fresh result omitted on turn 3) |
| Post-instruction | 3/3 | 6 | 0 | 0.0 | fail (fresh result omitted on turn 3) |

The retained fixture does not establish the post I4 invariant. The comparison records
`rate_changed=false`, `rate_improved=false`, and `i4_fix_supported=false` with the
finding `stale-result rate was unchanged; do not claim I4 fixed`. The honest result
is therefore a failing correctness gate with no measured improvement attributable
to the instruction.

Evidence: `directive-freshness-pre-final-v2/`,
`directive-freshness-post-final-v2/`, and
`directive-freshness-comparison.json`. Earlier fixture-development attempts are
also retained rather than overwritten.

## Browser and rendered-audio evidence

`tools/qualification/session_replay_browser_suite.py` preserves the retained PCM
byte-for-byte (apart from an explicit connection preroll in the generated fake-mic
WAV), drives real Chromium through the Playground/WebRTC path, and retains ordered
RTVI, Pipecat playout, model trace, browser remote-track audio, logs, screenshot,
reports, and hashed manifests.

Its hard, fail-closed acceptance gates are now exactly:

- response completion;
- audible-watchdog closure;
- recognized-speech attribution;
- absence of runtime-caused I3 cutoffs; and
- qualified-ASR rendered-audio/text agreement.

Tool behavior is a reported propensity metric: calls elicited, each utterance and
its call/no-call outcome, observed sequence, Pacific-conversion eligibility, and
per-response stale/fresh classification and rates when eligible. It does not alter
the replay pass/fail result.

The corrected rerun is retained at `post-fix-browser-corrected/`. It completed with
nine assistant responses and all five acceptance gates true. Rendered-audio WER is
0.075 against the 0.35 limit. This stochastic run elicited one call
(`set_timezone`), a reported per-utterance call rate of 0.125, and was not eligible
for replay freshness measurement; it still correctly passes acceptance.

The input PCM SHA-256 is
`22536283817f2e148612fb6c182281e6f8f0d29c751ec26a6a0533bc3599af04`.
Each complete browser replay directory contains a `capture-manifest.json`; runs
with ASR also contain `manifest.json`.

## Validation and handoff state

- Focused Ruff over the corrected replay fixture, directive fixture, and their
  tests: pass.
- Corrected replay/directive tests plus completion, Pipecat, and websocket suites:
  172 passed, 27 subtests passed.
- The earlier full pytest result remains 1,285 passed, 15 skipped, 29 subtests
  passed, with one unrelated sealed step-5 protocol-race fixture revision mismatch.
- The final runtime image is
  `pipecat-ai/nemotron-voicechat-dgx-spark:session-repair-20260817-r3`, image ID
  `sha256:c79934692503646531c9be56c67813e473944fc53e21728bdb72db1d5926c53d`.
- At handoff, r3 is restored on `127.0.0.1:8786`; typed input is ready; the demo
  service is restored on `127.0.0.1:7860/client/`.
- All replay and directive attempts were retained. No commit was created.
