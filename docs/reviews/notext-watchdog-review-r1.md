# Production no-text watchdog repair — adversarial review R1

**Verdict: FAIL.** The local watchdog state transition is correct and the retained traces convincingly reproduce the fixed text-quiet/audible shape, but the production repair is not yet promotable. Its new safety complement—the 120-second response cap—is not an absolute wall-time cap, is neither behaviorally tested nor exercised by the retained validation, and has no defensible 120-second service budget. The change also rewrites frozen candidate identities, leaves the promotion-script checksum seal invalid, and reports the browser plan as 8/8 despite three transcript-inequality adjudications. The narrow fix passes; the production contract and provenance do not.

## Blocking findings

### 1. Reset-on-audible is correct locally, but the replacement global bound is not proved or adequately specified

`AgentSilenceEosWatchdog.observe()` now resets `no_text_frames` on a function delta, text delta, or `output_dbfs > threshold_dbfs` (`server.py:2628-2648`). That is the right repair for decoupled Nano text and EarTTS output. The `no_audio_frames` expression is algebraically unchanged—function delta or audible audio resets it—and decoded silence is still consecutive below-threshold output after `audible_seen` (`server.py:2645-2661`). Existing and new tests continue to pin short-text/no-audio and decoded-silence behavior.

The old guard did, however, bound **audible but semantic-free** output at 30 model frames (2.4 nominal seconds). The repair deliberately removes that bound and relies on `VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC=120`. Source tracing confirms the nominal wiring:

- environment to model config: `server.py:4763-4766`;
- pinned wrapper conversion: `int(max_agent_response_sec / 0.08)`, hence 1,500 frames;
- forced EOS when continuous `agent_speaking` reaches that count: retained wrapper patch around `nemotron-voicechat-rnnt-turn-taking.patch:3155-3184`;
- response-boundary classification as `max_response_cap`: `server.py:2959-2971`.

That does bound the example of one continuously speaking response emitting audible audio forever. It does **not** justify the completion note's stronger “absolute-duration” description. Enforcement counts processed frames while `agent_speaking`; it resets when that state is false and does not use monotonic time. Slow/stalled frame processing can exceed 120 wall seconds, and response-open gaps are not part of the counter. The value also allows up to two minutes of above-−90 dBFS noise or nonsemantic speech where the old guard allowed 2.4 seconds. No SLO, product requirement, observed valid-response distribution, or tail quantile supports 120. In the retained traces, the maximum `agent_talking_frames` was only 164 (13.12 nominal seconds) in capture-on, 110 capture-off, and at most 50 in the rapid sessions; no run approached 1,500.

No changed test drives the wrapper to the cap. `test_acoustic_boundary_waits_for_decoded_silence_after_text_eos` (`test_realtime_web_server.py:2824-2840`) merely labels an already supplied `agent_eos` based on frame distance; it does not prove the wrapper emits EOS. There is also an activated dormant-path defect in the pinned wrapper: `_max_agent_response_frames` is initialized to zero, the redirect is pre-tokenized only while it is positive, and the configured value is assigned later. Forced EOS works, but `_max_response_redirect_tokens` remains empty and the logged/advertised redirect is never queued. The stale patch comment still says production disables the cap.

**Required delta:** define the actual safety contract (wall-time or model-frame time, continuous-speaking or whole-response), derive and document a production budget, fail closed on invalid/non-finite production values, and add a CPU unit test at a small configured limit that proves config propagation, exact forced-EOS boundary, counter reset semantics, response-boundary reason, and the intended redirect/no-redirect behavior. Include an audible/no-semantic response in that test. Exercise the production value in retained evidence or explicitly limit the live claim to wiring plus unit proof.

### 2. Frozen deployment identities and checksum provenance were mutated in place

Changing `PRODUCTION_ENVIRONMENT` and all three TOMLs in lockstep makes the current equality tests pass, but those tests compare mutable copies of the same newly edited value. It does not preserve historical identity. The code itself describes this mapping as the “precise qualified set” (`provenance.py:172-175`), `load_config()` rejects an environment that differs from the frozen candidate, and the promotion runbook calls candidate 1's 61-key environment frozen.

Candidate 3 is particularly unsafe to rewrite in place. `production-candidate-3.toml` names `production-candidate-3-fhw8`, pins the published fhw8 revision and release hash, and names a candidate-3 runtime image (`config/production-candidate-3.toml:1-20`). Changing its runtime environment does not alter the remote weight bytes or invalidate `release_sha256`; it does make the repository recipe and unchanged image/candidate label describe behavior that was not the published/qualified candidate-3 environment. The managed loader does not currently accept candidate 3, and the runbook says its fresh-machine managed path has not been tested. It was unnecessary to mutate this pin to validate the promotion script.

The three edited launch scripts are covered by `deploy/promotion-candidate/manifest.sha256:2-4`, but the manifest was not updated. `sha256sum -c deploy/promotion-candidate/manifest.sha256` fails for all three, as well as the independently stale qualification script. The production-script seal was already stale at `HEAD` because earlier commits changed those scripts without resealing it; this repair did not originate all of the drift, but it changed three sealed files again and still cannot make a clean provenance claim.

**Required delta:** do not retrofit candidate 1/2/3 identities. Introduce an explicitly versioned production hotfix/new candidate environment and runtime image identity, or document and implement a formal supersession mechanism that preserves the old bytes and qualification linkage. Keep the published fhw8 artifact pin separate from the new runtime contract. Recreate and verify the promotion manifest for the chosen launch surface, including all production launch scripts, and add a test/check that fails on checksum drift.

### 3. The 8/8 browser headline masks adjudicated client-visible failures, and the proposed completion gate is not executable

The retained `fixture-summary.json` has top-level `status="completed"` and eight fixtures with `completion_status="completed"`, but no top-level boolean pass. Three fixtures have `transcript_equal=false`:

- `L1-2`: browser transcript empty while traced response text is nonempty;
- `L2-1`: browser transcript includes the prior L1 prefix;
- `I1`: transcript includes interrupted-response text before the replacement answer (explicitly adjudicated).

The interruption case may be intentional policy. The two non-interruption cases mean “8/8 completed” is not a clean browser end-to-end pass and cannot establish that the repair introduced no client delivery/attribution regression. `L1-2` also ends normally through decoded silence after the traced text stops at “green dragons”; that may be the known model-silence truncation, but the present gate adjudicates it rather than distinguishing a fully rendered response.

`reports/notext-watchdog/test-matrix-amendment.md` correctly calls for a fail-closed response-completion gate and separation of model-silence from live-audio watchdog truncation. No implementation of those rules exists in `rapid_turn_settlement_suite.py`, the browser driver, or their tests. Thus future automation remains capable of repeating the original false green.

**Required delta:** implement and mutation-test the amendment's trace-derived completion gate; demonstrate it fails the retained pre-fix trace and passes the repaired traces. Make the browser result report a real aggregate pass/fail, fail or explicitly classify every non-interruption transcript mismatch, and retain a counterfactual/baseline showing these mismatches are not new.

### 4. Retained runtime provenance is incomplete

The rapid traces do bind the used `server.py` and `provenance.py` hashes to the current working-tree hashes and record the production values `no_text=30`, `no_audio=30`, decoded silence 20, threshold −90 dBFS, and max response 120. However, both retained `session.created` records have `runtime_image=null` and `runtime_image_id=null`. That fails this repository's own `valid_runtime_provenance()` requirement for an immutable `sha256:<64 hex>` image identity. The completion note's image ID exists only in prose, not in the retained validation bundle. The browser playout traces do not retain `session.created`, so their sessions are not independently bound to any image or source manifest.

**Required delta:** have the launcher pass the resolved immutable image ID and image name, retain them in each session artifact, and make the browser summary retain or hash-bind the server `session.created` provenance. Validate the bundle with `valid_runtime_provenance()` before calling it production evidence.

## What is solid

- The new regression test genuinely pins the failing shape rather than restating the implementation. With `no_text_required_frames=3`, six audible/text-quiet frames after initial text do not fire (`test_realtime_web_server.py:3062-3078`); the old implementation fires on the third such frame. The dead BOS and decoded-silence tests are also meaningful. The missing piece is the cap behavior that replaces the old audible/no-text protection.
- Independent parsing reproduced the live headline exactly. Capture-on frames 543–584 are 42 consecutive PAD frames for one response, all above −90 dBFS (−84.054 to −28.010), all with `no_text_frames=0`, and no pending request. Capture-on contributes 454 text-quiet/audible-after-text frames and capture-off 53, totaling 507 with zero violations.
- Browser response boundaries are seven decoded-silence completions plus one interruption boundary in capture-on, and one decoded-silence completion capture-off: eight normal decoded-silence ends plus the expected interruption. There are zero no-text or no-audio watchdog requests.
- Both rapid summaries genuinely show 8/8 responses completed, nonempty transcripts, and no errors. Their 16 boundaries all close through decoded silence. Tool mode made zero tool calls, so it supplies no tool-cycle evidence, as the completion note correctly discloses.
- CPU-only focused verification passed: `184 passed, 27 subtests passed`; `git diff --check` passed. No live service was started for this review.

## Acceptance checklist

1. Specify and test the global response bound, including audible/no-semantic output, exact EOS enforcement, invalid config, state resets, and redirect policy; justify the production number.
2. Preserve frozen candidates and publish a new/superseding runtime identity for the hotfix; do not rewrite the published candidate-3 recipe in place.
3. Repair and enforce the promotion checksum manifest.
4. Implement the trace-derived completion gate and make the known-bad pre-fix trace fail it.
5. Resolve or explicitly gate the two non-interruption browser transcript mismatches.
6. Retain immutable runtime image identity in rapid and browser evidence.

Until those deltas are closed, the correct production verdict remains **FAIL**, despite the narrow watchdog transition itself being correct.
