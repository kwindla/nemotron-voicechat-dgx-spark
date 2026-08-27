# Session repair adversarial review R3

## Verdict: FAIL — not merge-ready

The I2 ownership repair and runtime promotion are now correct, and all retained
browser manifest records match their final artifacts. One requested blocking
condition remains: the added browser-manifest test does not genuinely kill the
old `run_asr` generation ordering. It tests the new helper in isolation and
survives an ordering-only production mutation that recreates the stale manifests.

## Blocking delta

1. **The generation-order mutation is not pinned by the added test.**
   `test_final_report_rewrite_refreshes_stale_capture_manifest_record`
   (`tests/runtime/test_session_replay_browser_suite.py:132-150`) constructs an
   in-memory manifest and calls `refresh_manifest_artifact` directly. It never
   calls `run_asr`, never observes `capture-manifest.json`, and never observes
   final `manifest.json`. Consequently it proves that the helper works when a
   caller invokes it, but not that final report generation invokes it before
   either manifest is written.

   I restored the old ordering in an in-memory production mutant by removing the
   `refresh_manifest_artifact` call and refreshed capture-manifest write at
   `tools/qualification/session_replay_browser_suite.py:910-913`, while retaining
   the helper. A mocked end-to-end `run_asr` invocation produced:

   ```text
   current_run_asr_capture_final_match (True, True)
   old_order_run_asr_capture_final_match (False, False)
   checked_in_helper_test_survives_old_order_mutation True
   ```

   The checked-in pytest also passes. A wholesale source revert would remove the
   helper and fail with a missing attribute, but that is not an ordering-sensitive
   test: the production call site can regress while the helper and test remain.

   Replace or extend the test with a mocked `run_asr` execution that starts from a
   captured-pending-ASR report and stale capture record, then asserts that both
   `capture-manifest.json` and `manifest.json` hash the finalized `report.json`.
   Removing or moving the production refresh/write must make that test fail.

## Confirmed closed deltas

### 1. I2 typed ownership and symmetric paths: PASS

I reran the R2 endpoint reproduction through the real FastAPI websocket,
protocol, locks, typed job, settlement, and `send_step` paths with the same
deterministic engine/Pocket boundary. The sequence was microphone commit ->
deferred FC before publication -> typed request -> typed BOS. Immediately before
the typed request the binding was `old deferred tool request`, source
`microphone`, source ID `1`. At the typed BOS:

```text
unpublished_interrupts 1
cleared_response_bindings 1
typed_bos_response_attribution None
old_text_absent True
```

The new typed gate takes `model_input_lock`, gives a newly published call the
race, and otherwise interrupts the unpublished owner and clears its binding
before accepting the typed job or allowing model mutation
(`server.py:8638-8699`). The microphone gate remains symmetric
(`server.py:8261-8307`). Both use a direct, fail-loud
`clear_pending_response_binding` call.

I enumerated the other ownership-adjacent paths. `session.update` is rejected
after the first model result/turn and rechecks under `model_input_lock`; function
output is continuation of an already published owner; commit and forced-BOS paths
settle the already selected microphone or typed owner; response cancellation,
session stop, disconnect, and abort are terminal/drain paths rather than new-turn
ownership; reconnect cannot overlap because the active-client lock is released
only after engine abort; ordinary audio append is transport for the currently
declared turn or the silence/preroll clock, not a distinct ownership declaration.
I found no remaining model-owner entry that bypasses the interrupt-and-clear
transition.

The published-call typed probe still rejects with
`reason=function_cycle_active` and leaves the microphone binding and clear count
unchanged. The existing publication-race/deferred-input mutation still preserves
the microphone attribution and performs zero clears. Typed/microphone EOTR,
forced BOS, two-tool continuation, immediate post-typed readiness, and
microphone pre-commit interruption tests all pass.

### 2. Promoted runtime image: PASS

The host server and all three in-image runtime copies have SHA-256
`4106fcb0a9b2e444f037f4ceef0c0c2087d16255510b73799b35c02cfbe22034`:

- `src/nemotron_voicechat_runtime/server.py`
- `/opt/project/src/nemotron_voicechat_runtime/server.py`
- `/usr/local/lib/python3.12/dist-packages/nemotron_voicechat_runtime/server.py`
- `/opt/pocket-tts/lib/python3.12/site-packages/nemotron_voicechat_runtime/server.py`

The running container and local image both resolve to
`pipecat-ai/nemotron-voicechat-dgx-spark:session-repair-20260818-r4`, immutable ID
`sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
The live health response reports ready, protocol v3, typed input ready, and the
same image/server provenance. `final-stack/status.json` and
`campaign-manifest.json` record those exact values, and the campaign digest for
the status file matches its bytes.

### 3. Retained browser manifest bytes: PASS; regression test: FAIL above

An exhaustive recomputation covered 17 manifests and 193 total artifact
records (182 browser records plus 11 campaign records): zero byte-count or SHA-256
mismatches. Every one of the 16 browser manifest records for `report.json`
matches the final report. This includes both capture/final pairs for attempt 5
and corrected replay, plus the additional repaired pair in
`pre-fix-browser-invalid-attempt2`. The six specialist digests (directive fixture,
directive script, replay script, retained PCM, demo, and server) also match.

The production `run_asr` order currently refreshes and writes
`capture-manifest.json` before deriving final `manifest.json`, so current
generation is correct. Only the required mutation guard remains open.

## Regression dispositions

I1 remains closed under its positive termination-evidence contract; I3 remains
classified as model silence/turn-taking with no runtime-caused cutoff; I4 remains
honestly unsupported with both arms at 0/3,
`post_freshness_gate_passed=false`, `rate_changed=false`, and
`i4_fix_supported=false`. No-text watchdog, model-frame-independent wall
deadline, exact twice-fence bound and fused/separate settlement, and promotion
manifest behavior remain intact.

The focused completion/directive/server/endpoint/replay/function/Pipecat set
passed 356 tests and 31 subtests. The exact nine ownership/publication/update
probes passed, the promotion manifest verified all 8 files, focused Ruff passed,
and `git diff --check` passed.
