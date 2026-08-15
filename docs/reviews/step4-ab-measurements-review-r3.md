# Step 4a+4b measurement-validity review — round 3

**Verdict: PASS - 4a/4b admissible as descriptive Step 4 inputs.** No
remaining closure delta was found. Their admitted scope remains descriptive:
neither report supports promoting or demoting a Step 5 candidate.

This review was entirely offline. I did not query health, connect to the model
or browser, run a live probe, or modify the running stack. Apart from this
review file, I made no file change.

## Closure verification

1. **The stale completion-note surface is closed.** The former
   `step4a-done.md` is now headed “Withdrawn,” declares the repaired reports the
   sole measurement record, and quotes and withdraws all three stale statements:
   `"bounded null"`, `"require >=30 response clusters"`, and the claimed
   thermal/DVFS/memory-pressure deprioritization, including the observational
   repeated-session follow-up (lines 1–11). Both repaired reports independently
   name `step4a-done.md` as withdrawn and identify the two reports as the sole
   measurement record (each report line 3). The reports themselves retain the
   corrected conclusions: server trace cost is `UNMEASURED`, the client A/B is
   conditional descriptive evidence, and telemetry supplies neither a causal
   finding nor a robust null.

2. **The repaired analysis paths now have genuine end-to-end coverage.** Fresh
   direct execution of the three new tests passed **3/3**. The two telemetry
   tests construct raw traces, regenerate qualified analyzer results, provide
   v2 overall and per-source acquisition boundaries, and call public
   `step4_telemetry_regression.analyze()`; the first reaches the complete
   analyzer and rejects nonmonotonic acquisition ordering, while the second
   distinguishes GPU, CPU-frequency, and pressure availability and verifies the
   emitted causally preceding PSI interval result, including r=1 and the
   10.0-ms/unit slope
   (`tests/runtime/test_step4_telemetry_regression.py:158-246`). The A/B test
   likewise regenerates both analyzer inputs and calls public
   `step4_ab_compare.analyze()` twice; unequal response lengths distinguish the
   emitted frame-weighted/equal-response/equal-run server estimands
   (7.5/5.0/7.5 ms) and residual estimands (0.75/0.5/0.75 ms), while checking
   response cardinalities and deterministic output
   (`tests/runtime/test_step4_ab_compare.py:101-129`). These paths include raw
   source identity, exact population-membership, prompt/job pairing, full joins
   or regressions, and public result emission rather than stopping at private
   helpers.

## Regression gates

- Four Step 4 modules: **25 passed**.
- Full offline suite with `-m 'not browser_e2e'`: **1124 passed, 15 skipped, 2
  deselected, 29 subtests passed**; the two deselections are the browser-E2E
  tests. Four non-failing dependency/runtime warnings were reported.
- Canonical retained verifier: **PASS (67 anchors)**.
- Task-scoped `ruff check` over the four Step 4 tools and four test modules:
  **All checks passed!**
- Both Step 4 `artifacts.sha256` manifests: every listed local and external
  retained artifact **OK**.

## Guidance for Step 4c and 4d

Step 4c should measure **EarTTS-high incidence first**, not compare pooled means
or assume a browser interaction: use counterbalanced independently assigned run
blocks across direct WebSocket, native Pipecat, and browser/WebRTC with one
deterministic long-PAD-tail fixture, matched `delivered-nonBOS` membership, and
fixed image/input/prompt/response/warm-up/duration; treat run block as the
assignment and uncertainty unit, and proceed to matched high/low PAD-position
Nsight capture only if the mode recurs before a preregistered exposure cap
(otherwise report `INSUFFICIENT` and pivot to the older reproducible state).
Step 4d should first decompose the dominant approximately 52-ms Nano interface
on captured, matched content- and PAD-position states into GPU kernel time versus
CPU/launch/gap time, paired with an exact-output component benchmark; that first
measurement should yield the recoverable ceiling and correctness constraint for
kernel tuning, fusion, or launch-gap work before any such candidate is ranked.
