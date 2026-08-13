# Gate review: Step 2 live fixture driver, round 2

Review date: 2026-08-13 UTC. Scope: the Step 2 preregistration, round-1
review, supplied fix note, `tools/qualification/step2_live_fixture_driver.py`,
the Step 2 diff in `tests/e2e/test_browser_smallwebrtc_e2e.py`, and
`tests/runtime/test_step2_live_fixture_driver.py`.

## Verdict

**FAIL — the driver is not qualified and the live campaign must not execute.**

The canonical-plan fix, ordinary cross-response rejection, outer teardown
recording, and mixed-record browser consumer are materially improved. Fresh
probes still found fail-open response attribution, summary, marker, and path
cases. Independently, the two browser sessions cannot safely use the one
configured Pipecat trace path, and no terminal Pipecat artifact gate is run.

## Remaining deltas

### 1. The two-session campaign has no valid Pipecat trace lifecycle or terminal gate

The driver resolves one `NEMOTRON_VOICECHAT_PLAYOUT_TRACE` path and passes that
same path to both independently connected browser sessions
(`tests/e2e/test_browser_smallwebrtc_e2e.py:2815-2818`, `:2840-2854`,
`:2890-2936`). Each SmallWebRTC worker constructs its own traced service, while
the trace writer exclusively publishes its configured artifact and refuses an
existing published path (`src/nemotron_voicechat_pipecat/llm.py:201-214`).

A fresh writer-level probe of the exact production writer showed both failure
modes:

- while the first session is still live, a second writer creates a second
  pending file; `active_playout_trace_path` rejects the two pending artifacts,
  and the second writer later terminates with `publish-error`;
- after the first writer publishes successfully, the second writer constructor
  raises `FileExistsError` for the existing artifact.

If a single accumulating trace were somehow shared instead, the driver still
resets `playout_index` to zero for the capture-off session (`:2921-2924`), so
its first fixture rejects all capture-on response records as unattributed gap
transitions (`:913-929`).

There is also no final validation of `voicechat.playout.trace_status`,
`valid=true`, null reason/error, `dropped_records=0`, or the preregistered
`observed_playout_trace.valid=true` checked-consumer result. None of those
fields or checks appears in the driver, live test, or summary schema.

Required delta: define disjoint, immutable Pipecat evidence for both browser
sessions (or a demonstrably single persistent writer with a correctly seeded
cursor), retain the session-to-trace mapping, wait for publication, and run the
canonical observed-playout consumer plus exact terminal-status checks before a
summary can complete. Add a full two-session offline orchestration test against
the real trace-writer lifecycle.

### 2. Response/transcript attribution still accepts impossible or stale evidence

The requested ordinary Pipecat attacks now fail closed: two response
lifecycles straddling one fixture window are rejected as extra, and a stale
terminal arriving inside the next window is rejected as foreign. I1 ordering
and browser transcript membership are still incomplete.

`correlate_response_lifecycles` orders only interruption edge, clear, original
terminal, and replacement creation (`tools/qualification/step2_live_fixture_driver.py:848-879`).
It does not require original audio to precede the interruption edge. A fresh
plausible I1 trace with

`original created -> typed interruption -> original audio delta -> clear -> cancelled terminal -> replacement`

was accepted, even though I1 sends the typed interruption four seconds after
browser first audio and therefore cannot have its interruption edge before the
original's first delta. The same incomplete ordering does not prove that all
original downstream pushes precede the clear.

For ordinary fixtures, transcript collection starts at the fixture start
marker rather than the attributed speaking start (`tests/e2e/test_browser_smallwebrtc_e2e.py:994-1019`).
A fake-page probe supplied a late transcript from the preceding response after
the new start marker but before the current `bot-started-speaking`, followed by
a valid current start/stop and Pipecat lifecycle with no current transcript.
The fixture completed and persisted `"stale prior response"` as the current
response's transcript evidence.

Required delta: enforce the full I1 causal order, including original audio
before the interruption and no post-clear accepted push, and bracket or
identity-correlate every persisted transcript to the same response lifecycle.
Add the accepted traces above as negative orchestration tests.

### 3. Completed-summary and browser-marker validation remain attestable rather than strict

`validate_fixture_summary` trusts the embedded canonical hash strings but does
not reconstruct and hash the embedded plan (`tools/qualification/step2_live_fixture_driver.py:1077-1084`).
It checks response clocks for numeric shape but not their lifecycle order
(`:959-1049`), checks fixture times only within each fixture (`:1165-1199`),
and accepts any mapping with `valid=true` plus the expected marker count as a
checked browser artifact (`:1242-1268`).

Fresh mutations of an otherwise accepted completed summary were all accepted:

- change a script byte inside the embedded plan while retaining the claimed
  canonical plan and script hashes;
- place a response terminal before its `response.created` clock;
- change the persisted I1 replacement request text while retaining its hash;
- overlap fixture 2's send with fixture 1 despite serial execution; and
- replace capture-on consumer evidence with only
  `{valid: true, fixture_marker_records: 15}`.

The browser consumer itself validates local marker transitions but never
requires every marker state to terminate or checks marker identity against the
canonical session (`:539-627`). A mixed capture-on artifact with all real
record variants and the correct 15 markers passed the checked consumer, as it
should. A second artifact containing capability/worklet/flush evidence plus 15
invented fixture indices and 15 `start`-only markers also returned
`valid=true`, `fixture_marker_records=15`; that result is sufficient for the
current completed-summary check.

Required delta: recompute the canonical plan and text hashes from the embedded
bytes; validate complete response and serial fixture ordering; require the
full checked-consumer result rather than a shallow attestation; and require the
exact per-session marker sequence, canonical index/fixture/script/capture
identity, and terminal boundary for every fixture. Cross-check the summary's
marker-emission claims against the persisted artifact.

### 4. A symlinked session directory is accepted and modified

`resolve_step2_paths` resolves all paths before reserving the session sentinel
(`tools/qualification/step2_live_fixture_driver.py:641-680`), erasing whether
the supplied session directory was a symlink. A fresh probe passed a symlinked
empty session directory. The function accepted it, returned the resolved
target paths, and created `.step2-live-fixture-session` inside the symlink
target.

Required delta: reject a symlinked session directory and symlinked path
components before mutation, then reserve/create the real session directory
exclusively without a resolve/check/use race. Add a session-directory symlink
test, not only a file-alias collision test.

## Verification record

No `browser_e2e` entry, live URL, browser campaign, stack, container, or
service was touched.

- Full offline suite: **988 passed, 15 skipped, 2 deselected** with
  `-m 'not browser_e2e'`.
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff on the three reviewed Python files: **passed**.
- `node --check` on the generated RTC probe, voice-latency probe, AudioWorklet
  installer, and extracted embedded worklet processor: **passed**.
- Task-scoped `git diff --check`: **passed**.
- Full valid mixed capture-on artifact through the checked consumer:
  **passed**, 23 records, 15 fixture markers, and matching persisted/final
  worklet ordinal 1.

The offline tests remain green because they do not cover the accepted cases
above or a complete two-session Pipecat writer lifecycle.
