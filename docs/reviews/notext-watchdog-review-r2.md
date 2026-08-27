# No-text watchdog repair — confirmation review R2

**Verdict: FAIL — not promotable yet.** Findings 1, 2, and 4 from R1 are
closed, and the repaired traces genuinely pass while the retained pre-fix
trace genuinely fails. The remaining load-bearing response-completion gate is
still fail-open on malformed or incomplete acoustic telemetry, however. That
directly contradicts the requested non-evadable/fail-closed contract, so the
stack should not be restarted for manual promotion testing yet.

## Blocking delta

### Response completion can be made green by removing the evidence it needs

`response_completion_gate.py` silently supplies safety-favorable defaults for
missing acoustic evidence: a missing watchdog threshold becomes `-90.0` and a
missing `model_output_audio`/`output_audio.rms_dbfs` becomes `-120.0`
(`tools/qualification/response_completion_gate.py:61-67`). It also accepts
non-finite floats. Consequently, the gate interprets unavailable or invalid
audibility evidence as proven silence instead of failing closed.

I mutation-checked this against the actual retained pre-fix trace
`2f88c304-b895-4b81-aa1d-adde6043d58c`:

- unmodified: **FAIL**, 5 watchdog closures, 4 acoustically live and 1 dead
  rescue;
- after removing `model_output_audio` and `output_audio` from every model step:
  **PASS**, 0 live watchdog closures and 5 purported dead rescues;
- after changing every watchdog threshold to `NaN`: **PASS**, 0 live watchdog
  closures and 5 purported dead rescues.

There is a second fail-open path. An overlapping response start records
`incomplete_reason="overlapping_response_start"` and appends the abandoned
response (`response_completion_gate.py:71-84`), but it never adds that response
to the `incomplete` failure list (`response_completion_gate.py:138-145`). A
three-step synthetic trace `start, start, decoded-silence end` therefore returns
`passed=true` with two responses, one of which has no EOS.

The current mutation test does genuinely mutate the watchdog reason/request
edge, and its omitted-`request_pending` fallback works
(`tests/runtime/test_response_completion_gate.py:60-93`). It does not mutate or
reject missing acoustic fields, non-finite values, or invalid boundary
topology. The retained-evidence test at lines 96-109 checks a precomputed JSON
summary rather than recomputing the gate over retained trace bytes, so it also
cannot catch these parser/schema evasions.

**Required fix:** validate the gate's input schema before classification. At a
minimum, require finite numeric threshold and output dBFS values for every
model step used by an audibility window; reject missing/invalid frame and
session identity; and fail on overlapping starts, end-without-start, open-at-EOF,
or other impossible boundary topology. Add mutations of the known-bad trace
that delete acoustic telemetry and inject `NaN`, plus an overlapping-start
test, and prove all fail. Keep the existing omitted-request-edge fallback.
Also add a direct browser-aggregate mutation that changes an adjudicated
mismatch to a no-text/no-audio watchdog reason, pinning the already-correct
non-absorption behavior.

## Confirmed closed

### 1. Frozen identities, new hotfix identity, artifact pin, and launch seal

- `config/production-candidate-{1,2,3}.toml` are byte-identical to `HEAD`.
  Their SHA-256 values are respectively
  `e63d22b3ba26370a880266457e0a3a55d41d6e8c689132faadc6fbb25caebddc`,
  `00377c19b1a33ed93e3e3eec15a452541ee051ec20395760878cff78ee5eed50`,
  and `c65b110c58656d3ddf65ac3829a57f2bbb38c3c369b9d2391d8d2878af37e977`.
- The literal `PRODUCTION_ENVIRONMENT` block is byte-identical to `HEAD`
  (SHA-256 `ce2adf800c4f5369c30e7a406ae2729d1d1565d79d746e99aca5d5f7fbf9fedb`).
- `production-hotfix-notext-watchdog-v1` is a distinct config, image tag,
  launcher, and runtime contract. Its `[supersedes]` section names only
  `production-candidate-3-fhw8`'s runtime contract and declares
  `artifact_bytes_changed=false`. Its fhw8 repository revision and
  `release_sha256=426eecc0...d947` exactly match frozen candidate 3.
- The manifest contains all four on-disk `start-production-*.sh` launchers.
  The verifier checks omission, duplicate entries, repository escape, missing
  files, and digest drift (`verify_promotion_manifest.py:22-64`); the launcher
  invokes it before Docker inspection or launch. Independent verification
  reports `promotion manifest verified: 8 files`. The drift test actually
  appends bytes to the hotfix launcher and observes checksum rejection.

### 2. Whole-response wall deadline and strict configuration

`ResponseWallClockDeadline` is a real monotonic timer, not a model-frame
counter (`server.py:2772-2859`). It arms only on the observed agent BOS
(`server.py:4388-4389`), cannot be rearmed by text, audible output, or duplicate
BOS, remains active through response/tail gaps, latches one external EOS, and
uses a generation guard to ignore cancelled old timers. Expiration is carried
in turn state; `response_audio_is_deliverable()` rejects all later output while
expired (`server.py:2448-2458`). Acoustic terminal completion cancels and
resets the timer (`server.py:4403-4410`). The upstream continuous-frame cap is
disabled and no redirect text is synthesized.

The named hotfix contract accepts finite numeric 30 seconds and rejects its
missing/default zero, negative, non-finite, or wrong values before model load
(`server.py:1255-1276`). The real-clock 30 ms test proves expiration without
model frames; fake-timer tests prove one EOS, cancelled-generation isolation,
boundary reason `max_response_wall_time`, and no redirect.

Thirty seconds is conservative relative to the retained concise-response
population. One documentation number should be corrected: the advertised
13.12-second basis is `agent_talking_frames * 0.08`, not the whole-response
wall measurement. The final capture-on trace's maximum retained deadline
elapsed time is 13.423 seconds, so the observed wall multiplier is about
2.235x rather than 2.287x. This does not undermine the 30-second choice, but
the evidence file/runbook should use the metric named by the contract.

### 3. What is correct in the completion/browser gates

- Independent execution on the retained source trace reproduces the promised
  pre-fix **FAIL** exactly: 5 no-text closures, 4 live-audio defects, 1 dead
  rescue, SHA-256
  `bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`.
- Independent execution on all eight referenced repaired traces passes. The
  final browser sessions reproduce 8 responses / 451 quiet-audible frames /
  max consecutive 61 and 1 response / 49 / 34; the final rapid sessions each
  close 8/8 through decoded silence. Their trace hashes match
  `live-validation-r2.json`.
- The omitted-request-edge fallback uses the five-frame closure window and
  fails when that retained window is audible (`response_completion_gate.py:114-131`).
- Browser mismatch classification is shape- and baseline-bound for L1-2 and
  L2-1. A direct mutation of the final L1-2 result from decoded-silence to
  `no_text_since_bos_watchdog` produces aggregate `passed=false`, one
  `watchdog_truncation`, and an unclassified failing fixture
  (`test_browser_smallwebrtc_e2e.py:125-163`). Thus browser adjudication cannot
  absorb a watchdog truncation. The remaining blocker is the upstream trace
  gate's schema fail-open described above.

### 4. Runtime provenance

The launcher resolves and validates an immutable `sha256:<64 hex>` image ID,
then passes image name, image ID, and versioned contract into the container
(`start-production-notext-watchdog-hotfix-v1.sh:5-12,84-86`). Server provenance
retains those fields, the semantic environment, response deadline, Speech
commit, policy data, and hashes of the direct runtime sources
(`server.py:5320-5394`). Pipecat retains exactly one session-created provenance
record per browser session, and browser/rapid gates require valid immutable
provenance, the exact hotfix contract, and exact production watchdog/deadline
values.

I checked all four final session records (two rapid and two browser): they are
identical; `valid_runtime_provenance()` accepts them; all required source hashes
match the current working tree; their 52-key semantic environments exactly
match `NOTEXT_WATCHDOG_HOTFIX_V1_ENVIRONMENT`; and they retain image
`pipecat-ai/nemotron-voicechat-dgx-spark:production-hotfix-notext-watchdog-v1`
at `sha256:b9004a53a22c0a1b5b54295221dbb35f552f13b6366a4ecf26899339312d84cb`.
The summary and model-trace hashes in `live-validation-r2.json` also match the
available artifacts.

## Regression confirmation

The reset-on-audible repair is unchanged from R1: only the no-text progress
condition adds `audio_audible`; the no-audio expression is algebraically
unchanged, and decoded silence remains consecutive below-threshold output
after audible output (`server.py:2662-2679`). Dead BOS, short text/no audio,
audible-then-silent, long quiet-text/live-audio, and prior response-boundary
behaviors remain covered.

CPU-only verification performed for R2, with no live service run:

- focused changed-path/non-live suite: `322 passed, 3 deselected, 27 subtests passed`;
- broad non-browser suite, excluding only the documented revision-pinned
  phase-0 assertion: `1342 passed, 15 skipped, 4 deselected, 29 subtests passed`;
- promotion manifest verifier: PASS, 8 files;
- `git diff --check`: PASS.

No repository file other than this R2 review was written, and no commit was
created.
