# Plan: repair the issues found in the 2026-08-17 manual session

Reference trace: `c42c6701-f14d-4960-83fc-3c24bd269194` (retained; includes
`events.jsonl`, `input-16000-mono-s16le.pcm`, `output-22050-mono-s16le.pcm`).
Acceptance test for the whole plan is a **browser fake-mic replay of that
session's user audio** (below).

## Established evidence

- Tool plumbing is CORRECT: 3 calls, 3 correct results
  (`get_current_time`→6:40 PM UTC; `set_timezone{Pacific}`→America/Los_Angeles;
  `get_current_time`→**11:40 AM PDT**, the right conversion).
- The no-text watchdog repair HELD: zero `no_text_since_bos_watchdog`.
- Bot audio ASR matches the traced text, so rendering is faithful; the
  repetition and the unfinished sentences are real, not transcript artifacts.

## Issues, in priority order

### I1 — Completion-gate false positive on legitimate disconnect (OURS, must fix)

Running `response_completion_gate.py` on the session yields
`input_schema_valid=false`, sole error `response_open_at_eof (bos_frame=975)`.
The user disconnected mid-response; that is legitimate. The gate must
distinguish **session-terminated-with-open-response** (record, do not fail)
from **response never closed inside a completed session** (fail). Determine
termination from the trace's own session-lifecycle records, not from absence
of data. Add tests for both shapes, and re-verify the pre-fix trace still
FAILS and all repaired traces still PASS.

### I2 — Empty `user_text` at BOS on post-function-cycle turns (OURS, root-cause required)

Turns 2, 3, 4 opened with `user_text=""`. Correlation is exact: each of those
BOS frames had **39/40 preceding frames with an active/background function
cycle**, while the two turns with correct user text (BOS 29, 676) had zero.
The user's speech WAS transcribed (it appears in the stream), so this is
attribution/binding at BOS, not recognition failure. Root-cause in the
post-function-call BOS path (`post_fc_client_bos_*`, transcript binding,
`reset_user_transcript` and the RNNT turn-text lifecycle). Determine whether
the transcript is cleared, not yet bound, or bound to the wrong turn, then fix
so a response opened after a function cycle carries the user text that caused
it. If the correct disposition is that no new user turn exists (the model
self-continues after a tool result), then the response must be attributed as a
continuation rather than a new user turn, and the gate/driver must record that
distinction explicitly.

### I3 — Responses ending mid-sentence (MEASURE; fix only if runtime-caused)

Turns 3, 4, 8 ended mid-phrase ("Your time zone is", "It", "I am here if"),
closed by decoded silence / no-audio / turn-taking. The no-text watchdog is
NOT implicated. Determine per case whether EarTTS genuinely stopped producing
audible audio (the documented stochastic long-rendition truncation — model
behavior, record it) or whether a runtime boundary closed a still-productive
response (defect — fix it). Report the split with per-case evidence. Do not
"fix" model behavior by loosening a boundary contract.

### I4 — Model reports stale tool results and confabulates (MITIGATE + MEASURE)

After receiving 11:40 AM PDT, the model repeatedly reported the older 6:40 PM
UTC and invented "nine forty PM" for Pacific. This is model behavior in the
documented confabulation family; it is NOT expected to be fully fixable in the
runtime. Permitted mitigation: strengthen the tool-grounding instruction to
require reporting the MOST RECENT tool result and prohibit restating a
superseded one (`TOOL_GROUNDING_INSTRUCTIONS` in server.py and/or the demo
system message). Measure the effect with the replay; report the rate honestly
and do not claim a fix if the rate is unchanged.

## Acceptance test: browser fake-mic replay of the retained session

Build `tools/qualification/session_replay_browser_suite.py`:

- input: the retained session's `input-16000-mono-s16le.pcm` converted to WAV;
- drive real Chromium via the existing Playwright harness with
  `--use-file-for-fake-audio-capture` against the live production stack with
  the demo tools registered (this is the user's actual path, not direct WS);
- retain: model trace, RTVI messages, browser playout, and rendered audio;
- assert, fail-closed:
  1. completion gate PASSES (post-I1), zero audible watchdog closures;
  2. every response that follows recognized user speech carries non-empty
     attributed user text, or is explicitly classified as a post-tool
     continuation (post-I2);
  3. tool sequence reproduces: `get_current_time`, `set_timezone`, then a
     `get_current_time` whose result is Pacific-converted;
  4. **freshness**: after the Pacific-converted result arrives, no subsequent
     assistant text reports the superseded UTC result (this is the I4 metric —
     report it; treat as a measured rate, and gate only if I4 mitigation is
     accepted as effective);
  5. sentence-completion: report per-response whether text ended at a sentence
     boundary, with the I3 classification (model-silence vs runtime).
- Also transcribe the rendered bot audio with the qualified ASR evaluator and
  assert it matches the traced assistant text within a stated WER bound, so
  audio/text divergence cannot hide.

Run the replay before and after the fixes; retain both, and show the
difference. Pre-fix must reproduce the observed failures — a replay that
cannot reproduce them is not a valid acceptance test.

## Discipline

Adversarial review before commit. Evidence under `reports/session-repair/`
with manifests. Model-behavior findings are reported as findings, never
absorbed as passes.
