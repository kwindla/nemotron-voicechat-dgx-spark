# No-text watchdog repair — confirmation review R3

**Verdict: FAIL — not promotable yet.** The three concrete R2 evasions are
closed, the mutation tests now recompute from retained trace bytes, and the
browser aggregate plus R1 controls have not regressed. The remaining
fail-closed blocker is not fully closed, however: the validator trusts the
very `event` discriminator that decides whether a record is validated. Omitting
that discriminator from selected model steps removes their acoustic evidence
from consideration and can still turn the pinned known-bad trace green.

## Blocking delta

### A model step can evade validation by omitting its discriminator

`_validate_model_steps()` rejects a partial record only after
`record.get("event") == "model_step"`. Every other object, including an object
with no `event`, is silently skipped (`tools/qualification/response_completion_gate.py:50-56`).
Consequently, required session/frame/turn/watchdog/acoustic/boundary fields are
not actually required until an unvalidated field has opted the record into the
schema.

I reproduced this against the retained known-bad trace
`2f88c304-b895-4b81-aa1d-adde6043d58c`, directly loaded from its 4,518,961
JSONL bytes at the pinned SHA-256
`bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`:

- unmodified: schema-valid **FAIL**, 5 responses, 4 audible watchdog defects,
  1 dead watchdog rescue;
- remove only `event` from the 208 acoustically audible interior model steps
  (leaving all response starts and ends intact): schema-valid **PASS**, 5
  responses, 0 audible watchdog defects, 5 purported dead rescues;
- insert a partial record containing only `session_id` and `frame`, with no
  `event`, into an otherwise legitimate `start / interior / end` trace:
  schema-valid **PASS** with no schema errors. The same partial record with
  `event="model_step"` correctly fails.

This is the same safety failure as R2: unavailable acoustic evidence is being
treated as if it never existed rather than failing schema-closed. It also
directly fails the requested check that omitted inspected fields and partially
valid steps cannot evade validation.

**Required delta:** validate the record envelope before dispatching on event
type. At minimum, every retained record must have a valid event discriminator,
and a model-shaped/partial record must not be able to escape the model-step
schema by omitting or changing that discriminator. Prefer validating the
supported trace record variants sufficiently to reject a relabeled partial
model step as well. Add a retained-byte mutation that removes `event` from the
audible interior steps of the pinned known-bad trace and require
`input_schema_valid=false` and `passed=false`.

## Confirmed closed in R3

### Exact R2 evasions

Independent execution against the current gate produced the required
schema-closed failures:

- deleting both `model_output_audio` and `output_audio` from every one of the
  666 model steps: `input_schema_valid=false`, `passed=false`, with 666
  `missing_or_non_finite_output_rms_dbfs` errors;
- replacing every threshold with `NaN`: `input_schema_valid=false`,
  `passed=false`, with 666 `missing_or_non_finite_threshold_dbfs` errors;
- the exact synthetic `start / start / end` topology: schema-invalid **FAIL**
  with `overlapping_response_start`.

The current validator also rejects an explicit model step missing session,
frame, audio, turn state, watchdog, threshold, or boundary; non-finite output
dBFS; non-monotonic frames; multiple sessions; end-without-start; and
open-at-EOF. The remaining defect is the pre-dispatch discriminator bypass
above.

### Retained bytes, SHA pin, and legitimate traces

The tests no longer decide the known-bad result from the precomputed evidence
summary. `retained_trace()` opens the retained `events.jsonl` and calls
`load_jsonl()` (`tests/runtime/test_response_completion_gate.py:23-26`); the
known-bad test hashes those bytes against the literal SHA pin before running
the gate (`tests/runtime/test_response_completion_gate.py:111-126`); and the
acoustic/NaN mutations operate on fresh parsed records from that same retained
path (`tests/runtime/test_response_completion_gate.py:137-165`). The evidence
JSON supplies repaired session IDs and expected response counts, but the gate
pass/fail is recomputed from each retained trace.

There is no observed false positive on the legitimate evidence population. I
reran the gate on all eight referenced repaired model traces: all eight hashes
match their evidence, all are schema-valid, all pass, and all report zero
audible watchdog defects. The synthetic 41-frame text-quiet/live-audio case
also remains schema-valid and passes. Thus the new field and topology checks
are compatible with the retained legitimate producer output; they just do not
cover a missing discriminator.

### Browser aggregate mutation

The browser mutation is load-bearing. It starts from the retained summary,
normalizes completion reasons to `decoded_silence_watchdog`, then changes the
already-adjudicated L1-2 mismatch to `no_text_since_bos_watchdog`
(`tests/e2e/test_browser_smallwebrtc_e2e.py:4770-4789`). The aggregate returns
`passed=false`, `watchdog_truncations=1`, and L1-2 in
`unclassified_transcript_mismatches`; its prior flush-gap classification does
not absorb the watchdog reason (`tests/e2e/test_browser_smallwebrtc_e2e.py:4791-4798`).

## Regression confirmation

R1 findings 1, 2, and 4 and the reset-on-audible transition have not regressed:

- The response bound remains a 30-second whole-response monotonic timer armed
  at agent BOS and reset only at the acoustic response end. It latches one EOS,
  rejects cancelled generations, records `max_response_wall_time`, emits no
  redirect, and leaves the upstream frame cap disabled. Strict production
  configuration and the real-clock no-model-frame expiry test still pass.
- Candidate 1/2/3 TOMLs remain byte-identical to `HEAD`, with SHA-256 values
  `e63d22b3...ddc`, `00377c19...50b`, and `c65b110c...977`. The distinct hotfix
  identity still reuses the frozen fhw8 revision/release pin, resolves an
  immutable image ID, and passes it into the container. The promotion manifest
  verifies all 8 files and covers all four production launchers.
- All four final retained provenance records (two rapid, two browser) still
  pass `valid_runtime_provenance()`, retain contract
  `production-hotfix-notext-watchdog-v1`, retain image ID
  `sha256:b9004a53a22c0a1b5b54295221dbb35f552f13b6366a4ecf26899339312d84cb`,
  retain the 30-second deadline, and have source hashes matching the current
  tree.
- `AgentSilenceEosWatchdog.observe()` still adds `audio_audible` only to the
  no-text progress reset. The no-audio expression remains function-or-audible,
  and decoded silence remains consecutive below-threshold output after audio
  has been heard (`server.py:2664-2690`). Dead BOS, short text/no audio,
  audible-then-silent completion, long text-quiet/live-audio, duplicate BOS,
  and reason precedence remain covered and passing.

CPU-only verification, with no live service run:

- response-completion/browser aggregate selection: **12 passed, 85
  deselected**;
- full non-browser suite, excluding only the documented unrelated
  revision-pinned phase-0 assertion: **1,348 passed, 15 skipped, 4 deselected,
  29 subtests passed**;
- promotion manifest verifier: **PASS, 8 files**;
- changed-file Ruff: **PASS**;
- `git diff --check`: **PASS**.

No repository file other than this R3 review was written, and no commit or live
run was performed.
