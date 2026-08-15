# Gate review: Step 2 live fixture driver, round 6

Review date: 2026-08-13 UTC. Scope: the single round-5 quiescence-window
delta, the round-5 fix note, fresh range/tolerance/preflight probes, targeted
regression checks for previously closed areas, and the requested offline
verification.

## Verdict

**FAIL — the driver is not qualified and the live campaign must not execute.**

The inclusive 500–60000 ms range, one-time preflight ordering, and completed-
summary tolerance comparison behave as specified. One evidence-generation
hole remains within the same round-5 delta: the live barrier replaces deficient
sampled end clocks with values that necessarily satisfy the duration checks.

## Remaining delta

### 1. The runtime manufactures passing end-clock duration claims

After the wait, `_await_rtvi_quiescence_barrier` samples the browser clocks but
does not retain those measurements as the window ends. It writes
`max(sampled_performance, start_performance + quiet_window_ms)` and
`max(sampled_wall, start_wall + quiet_window_ms / 1000)` instead
(`tests/e2e/test_browser_smallwebrtc_e2e.py:952-965`). Consequently the
completed-summary consumer's dual-clock checks
(`tools/qualification/step2_live_fixture_driver.py:1722-1725`) cannot detect a
short or frozen clock in evidence emitted by the live driver: the producer has
already raised both values to the minimum passing claims.

A fresh runtime probe advanced the barrier's event-loop deadline through the
500 ms floor, returned no RTVI messages, and made the browser clock sampler
return the terminal values unchanged: performance `1000.0 -> 1000.0` and wall
`100.0 -> 100.0`. The barrier rewrote those samples to `1500.0` and `100.5`,
returned `passed=true`, and `_validate_quiescence_barrier` accepted the result.
Thus the retained fields claim two browser-clock durations that were not
observed. This defeats the newly required evidence check even though the
consumer arithmetic itself is correct.

Required delta: retain the sampled performance and wall end clocks without
clamping. Fail the live barrier if the performance sample is short or if the
wall sample is short by more than the documented 1 ms tolerance, and keep the
same checks at the completed-summary boundary. Add runtime-generation probes
for a frozen clock and a wall duration short by 1.001 ms; retain an exactly
1 ms-short case as accepted.

## Fresh probe and regression record

- The preflight parser accepted both inclusive endpoints, `500` and `60000`.
  Fast-clock runtime probes accepted both endpoints, and completed summaries
  using each endpoint passed the checked consumer.
- A completed barrier whose wall duration was exactly 1 ms short passed; one
  short by 1.001 ms failed with `RTVI quiet-window evidence is inconsistent`.
- An invalid preflight value of `499` failed before any patched microphone
  write, path reservation, Playwright/browser setup or connection, or fixture
  send boundary was reached.
- Eleven targeted checks covering repeated-S1 drain attribution, I1 causality,
  response membership/uniqueness, marker lifecycle binding, descriptor-owned
  child writes, the two-session real-writer lifecycle, live third-connection
  rejection, and trace-off isolation passed.

Fresh delta-probe output is retained at
`/tmp/step2-driver-r6-delta-probes.json` and the probe source at
`/tmp/step2_driver_r6_probe.py`.

## Verification record

No `browser_e2e` entry, live URL, browser campaign, running stack, container,
service endpoint, or commit was touched. No repository file other than this
review was written by the review.

- Full offline suite with every `browser_e2e` entry deselected: **1045 passed,
  15 skipped, 2 deselected, 29 subtests passed** (four existing warnings).
  Two preliminary collection attempts exposed the already documented
  deferred-tool-timeout background-abort race (`abort_count == 0`); its exact
  test passed immediately in isolation, and the complete file-first retry was
  green.
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff over the service, driver, analyzer, browser driver/tests,
  Step 2/runtime driver/analyzer tests, and playout-trace tests: **passed**.
- Task-scoped `git diff --check`: **passed**.
