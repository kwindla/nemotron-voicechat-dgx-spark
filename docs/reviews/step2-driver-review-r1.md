# Adversarial review: Step 2 live fixture driver, round 1

Review date: 2026-08-13 UTC. Scope: `reports/step2-live/preregistration.md`,
the supplied completion note, `tools/qualification/step2_live_fixture_driver.py`,
the `test_step2_live_fixture_plan` diff in
`tests/e2e/test_browser_smallwebrtc_e2e.py`, and
`tests/runtime/test_step2_live_fixture_driver.py`.

## Verdict

**FAIL — do not run the preregistered live campaign with this driver yet.**

The ordinary timeout path does fail the pytest and retain a failed terminal
summary, capture-on and capture-off use separate browser contexts, and the
current sink tolerates the new marker records. Those good properties are not
enough for qualification evidence. The runner does not enforce the
preregistered eight-fixture plan, cannot attribute speaking boundaries to a
fixture or response, can validate a `completed` summary whose session failed,
and has unguarded teardown/error-boundary paths that can leave the only summary
nonterminal. The offline tests do not exercise these failure modes.

## Blocking findings

### 1. Speaking events and interruption timing are not response-attributed

`_run_step2_fixture` takes a message-list length before the send, then selects
the first later `bot-started-speaking` solely by type
(`tests/e2e/test_browser_smallwebrtc_e2e.py:725-749`). Completion is likewise
the first later `bot-stopped-speaking` after that start
(`tests/e2e/test_browser_smallwebrtc_e2e.py:796-817`). `_rtvi_events` has no
fixture, turn, or response predicate (`:704-712`).

This baseline excludes already-observed events but does not identify events
that arrive late from the preceding turn. Once one fixture is falsely
completed on a stale start/stop pair, the real response can be shifted into the
next fixture's window, so consecutive fixtures can be misattributed. Serial
sending prevents the next fixture's intended response from starting before the
next send in the healthy case; it does not make the evidence robust to the
failure being qualified.

The installed Pipecat RTVI contract makes the problem unavoidable on this data
source: `BotStartedSpeakingMessage` and `BotStoppedSpeakingMessage` contain
only `label` and `type`; they carry no response ID. `response_ids_in` therefore
cannot correlate those events. Its unit test invents nested response IDs that
the relevant RTVI speaking messages do not have
(`tests/runtime/test_step2_live_fixture_driver.py:215-221`). In the actual
browser stream, `response_ids_observed` can remain empty while a fixture is
accepted as completed.

For I1, the clock conversion and deadline arithmetic are reasonable *after*
the correct first-audio event has been chosen (`tests/e2e/...py:752-766`). But
the supposed replacement is merely the first global `bot-started-speaking`
after the interruption marker (`:786-794`), and success is any later stop.
The driver does not require `bot-interrupted`, identify the cancelled and
replacement response IDs, or check that the replacement transcript/audio is
the second answer. It therefore does not establish the preregistered
cancel-then-replace/second-answer gate.

Required fix:

- Add an observable, ordered turn/response identity at the browser boundary,
  or join each typed-send boundary to the Pipecat/server trace before a fixture
  can become complete. Speaking start, speaking stop, transcript, audio
  membership, and response terminal must agree on that identity.
- For I1, retain the original response ID, interruption/clear edge, cancelled
  terminal, replacement response ID, replacement start/stop, and the second
  answer's transcript/audio evidence. Reject extra, missing, or out-of-order
  transitions.
- Persist those IDs as structured per-fixture fields; do not accept an empty
  response-ID set for a completed fixture.

### 2. The parser accepts the wrong campaign and rejects literal repeated script labels

`parse_fixture_plan` accepts any nonempty ordered list and validates only local
field shape (`tools/qualification/step2_live_fixture_driver.py:148-194`). It
does not enforce the exact preregistered sequence, script bytes, counts,
capture modes, I1 replacement text, or approximately four-second offset.

The sole `name` field conflates a unique run identifier with the repeated
script identity, and duplicate names are rejected (`:158-167`). A literal
`S1, S1` plan fails parsing. Operators can invent names such as `S1-a` and
`S1-b`, but then the artifact has no checked `script_id == S1` relationship.

`FixturePlan.sessions()` also groups all capture-on fixtures before all
capture-off fixtures regardless of declaration order (`:71-79`). That happens
to reproduce the preregistration only if the supplied plan is already the
right seven-on/one-off plan; an interleaved or capture-off-first declaration is
silently executed in a different order from `plan.fixtures`. An adversarial
probe confirmed `['off', 'on']` is accepted but executed as `['on', 'off']`.

Required fix:

- Separate unique `fixture_id` from repeated `script_id`, and validate one
  canonical Step 2 plan: `S1, S1, L1, L1, L2, L2, I1, capture-off L1`, with
  the exact preregistered text bytes, seven capture-on entries, one final
  capture-off entry, and exact I1 request/offset policy.
- Reject any mode transition other than the single final on-to-off boundary;
  do not silently reorder an invalid plan.
- Check and retain hashes of the canonical serialized plan and script bytes.

### 3. Terminal-summary and error-boundary handling is not fail-closed

The normal timeout path is sound in isolation: it sets the fixture to
`timeout`, emits a timeout marker, persists, returns false, marks the session
failed, and ends with summary status `failed` (`tests/e2e/...py:820-826`,
`:1776-1787`, `:1841-1845`). A normal timeout should not look complete.

There are nevertheless three evidence-integrity holes:

1. `validate_fixture_summary` accepts `status="completed"` when every fixture
   says completed even if a session says `failed`, has an error, and has no
   `capture_shutdown` (`tools/qualification/step2_live_fixture_driver.py:372-387`,
   `:472-476`). A direct adversarial construction was accepted. It also does
   not require nonempty response IDs or a completed interruption send.
2. Error-marker emission is wrapped in `except Exception: pass`
   (`tests/e2e/...py:1797-1807`). The summary records neither the missing error
   boundary nor the marker-write failure, exactly when page/sink failure makes
   that boundary most likely to be absent.
3. `context.close()` and `browser.close()` are awaited outside the session
   exception handler (`:1810-1837`). Either exception, or failure while leaving
   `async_playwright`, bypasses the final `completed_wall_time_s`/terminal
   status write at `:1841-1843`. Also, `completion_status="running"` is not
   persisted before the first awaited fill/marker operations (`:725-733`).

Required fix:

- Make terminal validation cross-field strict. `completed` must require every
  session completed, null session/fixture errors, successful non-null capture
  shutdown, required marker counts, all fixtures completed, nonempty and
  correctly cardinalized response IDs, and a completed I1 replacement.
- Persist `running` before any awaited fixture operation. Record send attempt
  and successful send separately; the current `send_wall_time_s` is captured
  before `Enter` is pressed.
- Put setup, fixture execution, each teardown operation, and Playwright exit
  behind one outer terminalization boundary. Catch and record teardown errors
  independently, attempt all remaining cleanup, and atomically write a failed
  terminal summary in the outermost `finally`.
- Never silently discard an error-boundary failure. Retain a summary field for
  marker emission status/error so missing boundary evidence is explicit.

### 4. Session objects are separate, but artifact paths are not protected

The capture modes correctly get distinct browser/context/page/sink instances,
and the capture-off artifact name is derived separately
(`tests/e2e/...py:1713-1753`). No in-memory worklet state is shared across
those sessions.

However, there is no disjointness or existence check for the configured
microphone, capture-on artifact, derived capture-off artifact, and summary
paths (`:1657-1687`). In particular, an explicitly configured summary can be
the capture-on or capture-off JSONL path. The sink opens the artifact with
`"w"`, while atomic summary replacement can unlink/replace that pathname;
subsequent sink writes continue to the old inode. The visible path then holds
the summary rather than the browser evidence. Reusing a session directory also
silently truncates prior artifacts, contrary to the preregistered immutable
session-directory rule.

Required fix: resolve and prove all input/output paths pairwise distinct,
reject existing output files (or create the whole session directory
exclusively), and record the resolved path set before launch. Add collision,
symlink, and rerun tests.

## Marker/schema assessment

The markers do **not** currently break `_BrowserPlayoutTraceSink` ordinal
reconciliation. `emit_marker` validates the marker, writes it directly, and
does not mutate worklet registration/dequeue/ack state
(`tests/e2e/...py:527-540`). It carries the same browser trace schema and a
distinct `browser.fixture_boundary` type. Capture-off close deliberately skips
worklet-final-ordinal requirements.

This is tolerance, not checked whole-artifact validity. `emit_batch` is not a
strict browser-record schema validator: the existing test proves it accepts a
record with no `type` at all (`tests/e2e/...py:852-881`). No repository
consumer loads the mixed capture-on JSONL and validates the complete record
union including `browser.fixture_boundary`; the only new marker/sink test uses
a capture-off marker-only file (`:891-921`). Therefore the current sink will
not invalidate the artifact, but compatibility with the preregistered checked
consumer has not been established.

Required fix: define one strict browser-artifact record union that includes
fixture boundaries, validate every persisted line plus ordering and terminal
ordinals, and run a mixed capture-on artifact (capability, clock mapping,
registered/dequeue/onset/ack, and boundaries) through that checked consumer.

## Offline-test quality

The focused tests cover local JSON parsing, stable grouping, basic marker and
summary shape, pure deadline arithmetic, recursive synthetic ID discovery,
and sink ordinal failure. They do not call `_run_step2_fixture` with a fake
page, and they do not test:

- stale/cross-fixture starts or response-ID correlation;
- I1 cancel/replacement attribution and second-answer proof;
- timeout/error-marker persistence and outer teardown failures;
- strict terminal summary cross-field invariants;
- the exact eight-entry preregistered plan and repeated script identity;
- output-path collision/immutability;
- capture-on/capture-off orchestration and state isolation; or
- a mixed marker plus worklet artifact through a strict consumer.

The fake-clock test proves only `first + delay` waiting in the helper; it does
not prove that the live runner selected the right first audio event. The
synthetic response-ID test likewise does not match the actual speaking-event
schema. Add orchestration-level offline fakes for every case above, including
injected failures at every awaited boundary.

## Independent verification

No `browser_e2e`-marked entry was run, and no live URL, stack, container, or
service was touched.

- Full offline suite: `915 passed, 15 skipped, 2 deselected, 29 subtests
  passed` with `-m 'not browser_e2e'`.
- Focused driver/e2e-offline selection: `18 passed, 2 deselected`.
- Canonical retained evidence: `verify-anchors: PASS (67 anchors)`.
- Task-scoped Ruff on the three reviewed Python files: passed.
- Repository-wide Ruff: failed on seven unrelated existing findings in
  `patch_eartts_exact_constants.py`, `analyze_eartts_exact_ab.py`,
  `eartts_exact_constants_gate.py`, and `step7_post_fc_fixture.py`. None is in
  this review's driver diff.
- `git diff --check`: passed.
- `node --check`: passed for the generated RTC probe, voice-latency probe,
  AudioWorklet installer, and extracted embedded worklet processor.

The live campaign should remain blocked until findings 1-4 are fixed and the
new orchestration/consumer tests pass.
