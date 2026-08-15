# Gate review: Step 2 live fixture driver, round 3

Review date: 2026-08-13 UTC. Scope: the Step 2 preregistration, round-2
review and fix note, `tools/qualification/step2_live_fixture_driver.py`,
the current diffs to `src/nemotron_voicechat_pipecat/llm.py` and
`tests/e2e/test_browser_smallwebrtc_e2e.py`, and the Step 2/playout tests.

## Verdict

**FAIL — the driver is not qualified and the live campaign must not execute.**

The four round-2 areas are substantially improved, and the requested offline
suite is green. Fresh probes still found four fail-open boundaries.

## Remaining deltas

### 1. The exact-two-connection gate ignores a live third trace

`validate_connection_playout_trace_set` considers only published paths for
which `Path.is_file()` is true
(`tools/qualification/step2_live_fixture_driver.py:922-934`). It does not
inspect the writer staging namespace.

A fresh real-writer probe closed connections 1 and 2, constructed connection
3, and left it live. The directory then contained the successful `0001` and
`0002` artifacts plus
`.trace-0003.jsonl.<pid>.<token>.pending`; the exact-two check returned
success. It rejected the same third connection only after that writer closed
and published `trace-0003.jsonl`. A separate missing-`0002` probe correctly
failed.

Required delta: at terminal validation, reject every matching pending,
quarantined, staged, or published connection trace outside the exact closed
`0001`/`0002` set. Add a real-writer test that calls the terminal set check
while connection 3 is still live.

### 2. Response and transcript attribution still accepts non-causal evidence

The new I1 checks reject an interruption edge before an original delta, any
original delta after that edge, no original audible push, and an original
push after clear. They still reconcile delta/release/push membership as sets
without requiring each delta to precede its release and downstream push
(`tools/qualification/step2_live_fixture_driver.py:999-1037`). A fresh I1
trace with an original downstream push before its corresponding delta and
release was accepted by `correlate_response_lifecycles`. The terminal
canonical consumer would reject that artifact later, but the fixture
correlator itself records it as an exact attributed lifecycle.

Transcript filtering is inclusive and clock-only
(`tools/qualification/step2_live_fixture_driver.py:1725-1746`;
`tests/e2e/test_browser_smallwebrtc_e2e.py:1008-1019`). Fresh fake-page probes
showed all of the following ordinary fixtures complete with the supplied text
as current-response evidence:

- a transcript listed before `bot-started-speaking` with the same receipt
  clock as the start;
- a transcript listed before the start event but carrying a clock numerically
  inside the start/stop window;
- a transcript listed after `bot-stopped-speaking` with the same clock as the
  stop; and
- a late prior-response transcript arriving strictly inside the current
  speaking window, with no current-response transcript.

The original round-2 stale-before-start attack with a strictly earlier clock
is now rejected.

Required delta: enforce per-ordinal delta -> release/clear -> downstream-push
causality in the fixture correlator. For RTVI evidence, validate message clock
monotonicity, use strict event-index boundaries, retain the bracket metadata,
and identity-correlate transcripts (or add an equivalent protocol-backed
quiescence/identity guarantee) so late prior text cannot satisfy the current
fixture.

### 3. Completed summaries remain internally forgeable

`_validate_response_attribution` requires nonempty delta and release lists but
does not reconstruct their membership, require downstream-push membership, or
join the claimed response IDs back to the retained Pipecat records
(`tools/qualification/step2_live_fixture_driver.py:1260-1398`). The top-level
and nested transcript evidence are each checked for nonemptiness but never for
equality (`:1322-1324`, `:1396-1398`, `:1563-1569`). Response IDs are not
required to be unique across fixtures.

Fresh mutations of an otherwise accepted completed summary were accepted:

- make the top-level transcript disagree with the attributed response's
  transcript, or mutate only the nested transcript;
- reuse fixture 1's response ID for fixture 2 while updating only fixture 2's
  redundant summary fields; and
- replace a response's delta ordinals, release reasons, and downstream-push
  ordinals with mutually inconsistent invented values.

The exact canonical marker identity/boundary sequence is now enforced: fresh
sequence swaps and wrong fixture identities failed. Marker chronology is not
bound to the fixture lifecycle, however. Replacing every capture-on marker's
browser performance and wall clocks with the same value, earlier than the
summary start, then retaining the full newly checked browser-consumer result
still produced an accepted completed summary
(`tools/qualification/step2_live_fixture_driver.py:1652-1691`).

Required delta: reconstruct every claimed response attribution from the
retained per-session Pipecat artifact; enforce campaign-wide response-ID
uniqueness and equality of redundant transcript fields. Cross-check each
persisted marker's strict clock/order bracket against its fixture send,
interruption, and terminal lifecycle rather than comparing only the reduced
identity/boundary sequence.

### 4. Session reservation still has a symlink check/use race

`resolve_step2_paths` first calls `_reject_symlink_components` on every raw
path and only afterward resolves every path with `Path.resolve()`
(`tools/qualification/step2_live_fixture_driver.py:764-776`). The no-follow
directory reservation protects the already-resolved path, not the supplied
path identity that was checked.

A deterministic race probe replaced the checked real session directory with
a symlink immediately after the last component check and before the resolve
loop. The function accepted the configuration, returned the symlink target as
`output_dir`, and created `.step2-live-fixture-session` in that target. A
second probe that introduced the symlink after resolution, immediately before
`_reserve_session_directory`, was correctly rejected by `O_NOFOLLOW`; the
earlier check/resolve window remains open.

Required delta: do not resolve checked pathnames through a later pathname
lookup. Anchor validation and reservation to no-follow directory descriptors,
verify the opened directory identity, and derive/open output children relative
to that retained descriptor. Add deterministic races both before resolution
and immediately before reservation.

## Verification record

No `browser_e2e` entry, live URL, browser campaign, running stack, container,
or service endpoint was touched.

- Full offline suite: **1000 passed, 15 skipped, 2 deselected, 29 subtests
  passed** with `-m 'not browser_e2e'`.
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff on `llm.py`, the driver, the browser test, the runtime
  driver tests, and the playout-trace tests: **passed**.
- Task-scoped `git diff --check`: **passed**.
- `node --check` on the generated RTC probe, voice-latency probe,
  AudioWorklet installer, and extracted embedded worklet processor: **passed**.
- `{connection}` construction race: **24/24 writers published** with the exact
  unique ordinal set `0001` through `0024`. Token-free and malformed-token
  spellings were byte-for-byte inert in the service helper; the driver rejected
  every spelling lacking the exact literal token before creating a session.
- Trace-off code-object comparison against `HEAD`: all **71/71** methods on
  `NemotronVoicechatLLMService` had identical `co_code`; the constructor
  selector and four hot methods also had identical constants and names.
- Fresh self-consistent rehashing of altered canonical plan text was rejected;
  fresh marker sequence swaps and identity substitutions were rejected.

