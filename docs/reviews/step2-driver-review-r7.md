# Confirmation gate: Step 2 live fixture driver, round 7

Review date: 2026-08-13 UTC. Scope: the single round-6 raw-clock delta,
fresh producer and completed-summary barrier probes, targeted prior-round
regressions, and the requested offline verification. No live or browser-E2E
entry was run.

## Verdict

**PASS - DRIVER QUALIFIED**

The round-6 delta is closed. The live producer retains the browser's raw
sampled performance and wall end clocks, marks a barrier failed when either
duration predicate is deficient, and uses the same predicates as the checked
consumer. A failed barrier remains valid retained evidence, but cannot satisfy
a completed fixture summary.

This verdict qualifies the driver to execute the preregistered campaign. It
does not pre-adjudicate the live fixture outputs or any Step 2 gate.

## Fresh barrier evidence

All probes used a 500 ms barrier beginning at performance `1000.0` ms and wall
time `100.0` s. The event-loop deadline advanced normally while the sampled
browser clocks were independently controlled.

- Frozen clocks were retained unchanged as `1000.0` and `100.0`; the producer
  returned `passed=false`, the evidence-consistency consumer accepted that
  truthful failed record, and a completed summary rejected it.
- With a complete performance clock and wall end `100.498999` (short by
  1.001 ms), the raw wall sample was retained and the same producer/consumer/
  completed-summary path failed closed.
- With wall end `100.499` (short by exactly 1 ms), the raw sample was retained,
  the producer returned `passed=true`, and the completed summary accepted it.
- New independent evidence-path probe: performance end `1499.999` (short by
  0.001 ms) with a complete wall interval was retained unchanged, returned
  `passed=false`, and was rejected at the completed-summary boundary.

The probe source is retained at `/tmp/step2_driver_r7_probe.py` and its compact
result at `/tmp/step2-driver-r7-probes.json`.

## Regression and verification record

- Targeted quiescence tests: **24 passed, 128 deselected**.
- Prior-round spot checks covering repeated-S1 drain attribution, per-ordinal
  response causality and I1 transition faults, reconstructed response
  membership, lifecycle-bound markers, descriptor-owned child writes, the
  real two-session writer lifecycle, rejection of a live third connection,
  and trace-off isolation: **13 passed** (one dependency warning).
- Full offline suite with every `browser_e2e` entry deselected: **1048 passed,
  15 skipped, 2 deselected, 29 subtests passed** (four existing warnings).
  The documented deferred-tool-timeout collection-order flake did not occur,
  so no isolated rerun was needed.
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff over the service, driver, analyzer, browser driver/tests,
  Step 2/runtime driver/analyzer tests, and playout-trace tests: **passed**.
- Task-scoped `git diff --check`: **passed**.

## Consolidated live-execution cautions

1. Execute only the hash-pinned canonical JSON plan, supplied as the JSON value
   of `VOICECHAT_LIVE_FIXTURE_PLAN` rather than a filename: `S1, S1, L1, L1,
   L2, L2, I1, capture-off L1`, with the registered exact text, wait policies,
   capture modes, replacement request, and 4.0 s interruption policy. Do not
   reorder, add, omit, or silently retry fixtures. Failure of both registered
   long-bin attempts is a retained finding, not permission for another run.
2. Before the first connection, verify the preregistered image digest/source
   commit, production-candidate environment plus only the three registered
   overrides, 160 ms prebuffer, local-loopback/headless-Chromium fake-mic and
   typed-input transport, evaluator image, `/health` qualification identity,
   and single-client state. Do not substitute a tunnel or an unregistered
   runtime configuration.
3. Start the host Pipecat trace configuration before campaign connections with
   `NEMOTRON_VOICECHAT_PLAYOUT_TRACE` containing the exact literal
   `{connection}` token in the immutable session directory. The campaign must
   exclusively own the fresh process's first two connections (`0001` capture
   on and `0002` capture off); any extra published, pending, staged, or
   quarantined matching trace, including a live third connection, invalidates
   terminal success.
4. Use a new, unused, direct-child session directory and mutually disjoint
   microphone, capture-on, derived capture-off, summary, and Pipecat paths.
   Do not reuse an old session, precreate output files, point through symlinks,
   move or replace the reserved directory during execution, or write other
   data into the trace namespace. Preserve the descriptor reservation until
   the terminal summary is published.
5. Complete all one-time preflight configuration before launch: live URL,
   installed Chromium path if overridden, an existing regular microphone WAV
   if supplied, output/summary paths, canonical plan, and tokenized Pipecat
   trace. A preflight failure is terminal; do not work around its path, plan,
   or environment rejection.
6. Keep `VOICECHAT_LIVE_RTVI_QUIET_WINDOW_MS` at the registered 2000 ms default
   unless a deliberate preregistration-compatible value in the inclusive
   500–60000 ms range is required. Never use zero or a test-only short
   surrogate. Any `bot-*` activity, frozen/short performance duration, or wall
   shortfall beyond the sole 1 ms quantization tolerance fails the fixture;
   retain the raw clocks and do not edit or reinterpret the failed evidence.
7. Preserve the driver's two isolated browser sessions and their ordering.
   Capture-off disables only browser worklet capture; it still requires its
   markers, response attribution, Pipecat trace, teardown evidence, and checked
   terminal validation. Do not admit another browser/client during the run.
8. On any timeout, marker failure, setup/teardown error, trace-publication
   failure, reconciliation failure, or barrier failure, stop accepting further
   fixtures and preserve the failed terminal summary plus all partial/staging
   evidence. Do not overwrite, repair, truncate, merge, or resume that session.
9. Treat campaign completion as provisional until the strict summary,
   browser-artifact consumer, both terminal Pipecat consumers, exact-two trace
   set, response/text/audio attribution, marker brackets, I1 cancel/clear/
   replacement chain, worklet ordinal reconciliation, and SHA-256 manifest all
   pass. Then adjudicate the registered ASR completeness/WER gates and run the
   registered arrival/reserve analysis; driver qualification alone is not a
   Step 2 result.

No live URL, running stack, container, service endpoint, browser campaign, or
commit was touched. No repository file other than this review was written by
the round-7 review.
