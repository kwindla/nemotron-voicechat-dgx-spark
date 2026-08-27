# No-text watchdog repair — confirmation review R4

**Verdict: FAIL — not promotable.** The exact R3 omitted-`event` mutation is
now schema-closed, all prior mutation and retained-trace results still hold,
and the earlier watchdog, deadline, deployment, provenance, and browser fixes
have not regressed. The discriminator fix is nevertheless not tamper-resistant:
removing one field in addition to using a recognized non-model discriminator
still hides all 208 audible interior steps and turns the pinned known-bad trace
green. Independently, the allowlist rejects a legitimate diagnostic event that
the current `SessionTrace` producer can actually serialize.

## Blocking deltas

### 1. A recognized relabel plus one missing shape key still evades validation

`_is_model_step_shaped()` requires the conjunction of `frame`, `turn_state`,
and an audio/model-exclusive key
(`tools/qualification/response_completion_gate.py:111-118`). A record with a
recognized non-model `event` is skipped unless that whole conjunction remains
true (`response_completion_gate.py:137-150`). Thus deleting any conjunct makes
the remaining model-only payload invisible to both model validation and
acoustic classification. There is no schema validation for the recognized
non-model variants.

I reproduced this from fresh retained bytes for
`2f88c304-b895-4b81-aa1d-adde6043d58c`: 4,518,961 bytes, 3,655 records, and
SHA-256
`bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`.
The mutation selected the same 208 acoustically audible interior model steps
as R3 and, on each selected record, performed only:

1. `record["event"] = "client_event"`;
2. `record.pop("frame")`.

All other model payload remains, including `turn_state`, both acoustic
objects, text/function fields, and the common trace envelope. The result is
`input_schema_valid=true`, **PASS**, 5 responses, 0 audible watchdog defects,
and 5 purported dead rescues. The unmodified bytes correctly produce
schema-valid **FAIL**, 5 responses, 4 audible defects, and 1 dead rescue.

An envelope variant also evades: replace each selected record by a recognized
`client_event` common envelope with the original complete model record under a
`payload` field. Because unexpected fields and nested payloads are not checked,
that mutation also returns the same schema-valid **PASS**. Relabeled/skipped
records disappear before frame-order and response-topology validation, so
ordering checks do not recover either defect.

This is the same safety failure as R2/R3: acoustic evidence can still be made
unavailable and then treated as nonexistent. The present model-shape heuristic
only raises the cost from one edited field to two.

**Required delta:** validate supported non-model variants rather than merely
allowlisting their discriminator; reject unexpected model-only fields and
arbitrary payload envelopes; and make model-footprint detection monotonic so
deleting one required model key cannot disable it while other model-exclusive
keys remain. Add the exact retained-byte `client_event` + removed-`frame`
mutation and the nested-envelope mutation as required FAIL tests. If arbitrary
record deletion or substitution is in the threat model, schema heuristics
alone cannot prove tamper resistance; the retained stream also needs an
integrity/sequence binding.

### 2. The allowlist is incomplete for an event the producer actually emits

A static comparison finds 60 literal `SessionTrace.event(...)` kinds in
`src/nemotron_voicechat_runtime/server.py` and 60 allowlisted kinds, with no
literal call-site difference. That comparison misses the producer's envelope
merge semantics. `SessionTrace.event()` builds the common `event=kind` envelope
and then applies `**fields`, so a payload can overwrite the discriminator
(`server.py:2481-2488`).

The `nano_pad_pair_watchdog` call expands its watchdog mapping into those top-
level fields (`server.py:6098-6104`), while the mapping itself contains
`event="idle_agent_without_pair_buffer"` (`server.py:3972-3978`). Therefore a
legitimate firing serializes as `idle_agent_without_pair_buffer`, not
`nano_pad_pair_watchdog`. The former is absent from
`RECOGNIZED_TRACE_EVENTS`; the latter is present but is not the emitted value
on this path.

I invoked `SessionTrace` in memory with that exact producer payload. It
serialized the event sequence `session_trace_started`,
`idle_agent_without_pair_buffer`, `session_trace_stopped`. In an otherwise
valid two-step response trace, the gate returned `input_schema_valid=false`,
`passed=false`, with
`record[2]: unrecognized_event ('idle_agent_without_pair_buffer')`. This is the
production-blocking false positive requested in check (b), even though none of
the retained repaired traces happened to fire this opt-in diagnostic.

**Required delta:** prevent `SessionTrace.event()` fields from overriding
reserved envelope keys (at least `trace`, `event`, `session_id`, and
`monotonic_s`), rename the nested diagnostic discriminator if needed, and
derive/test the gate vocabulary against actual serialized records rather than
only literal declared kinds. A producer-level test should exercise every
supported diagnostic variant through serialization and then through the gate.

## Confirmed closed or unchanged

### Exact R3 evasion and prior mutations

- Removing `event` from the 208 audible interior steps now gives schema-invalid
  **FAIL**. It reports 208 `missing_or_invalid_event` errors plus 208
  `model_step_shape_event_mismatch` errors.
- Relabeling those 208 intact steps as recognized `client_event` records gives
  208 shape-mismatch errors and schema-invalid **FAIL**. Relabeling them with an
  unknown event gives 208 unknown-event plus 208 shape-mismatch errors and
  schema-invalid **FAIL**.
- Removing both acoustic objects from all 666 model steps gives 666 missing/non-
  finite output errors and **FAIL**. Replacing all thresholds with `NaN` gives
  666 non-finite-threshold errors and **FAIL**.
- Missing session, frame, turn state, watchdog, threshold, boundary, or output;
  non-finite output; non-monotonic frames; multiple sessions; overlapping
  starts; end-without-start; and open-at-EOF all remain schema-invalid FAIL.
  A watchdog closure with its request edge omitted remains schema-valid but
  semantically **FAIL** through the five-frame audible closure fallback.
- The current automated response-completion/browser selection passes: **16
  passed, 85 deselected**. The suite covers every prior mutation above except
  the two new evasions.

### Retained bad/repaired evidence

- The pinned unmodified pre-fix trace remains schema-valid **FAIL** with the
  expected 5 / 4 audible / 1 dead result.
- The four completion-evidence repaired traces remain schema-valid **PASS** at
  their expected response counts: 8, 1, 8, and 8.
- The two final rapid and two final browser traces all retain matching evidence
  hashes and remain schema-valid **PASS** at expected response counts 8, 8, 8,
  and 1. No retained legitimate trace produced a false positive; the producer
  diagnostic path above is the uncovered false-positive case.

### R1 findings 1, 2, and 4; reset transition; browser aggregate

- The 30-second bound remains a whole-response monotonic timer armed at BOS,
  reset at the acoustic end, generation-guarded against stale timers, and
  configured strictly for the hotfix contract with no redirect. The focused
  deadline/watchdog selection passes **10 tests**. The real-clock no-model-frame
  expiry is included.
- The reset-on-audible transition is unchanged: audible output resets only the
  no-text progress counter in addition to its preexisting role in no-audio;
  decoded silence remains consecutive below-threshold output after audio has
  been heard (`server.py:2662-2690`). Dead BOS, short text/no audio, live audio,
  audible-then-silent completion, duplicate BOS, and reason precedence remain
  covered by the passing suite.
- Candidate 1/2/3 remain byte-identical to `HEAD`, with SHA-256 values
  `e63d22b3...ddc`, `00377c19...50b`, and `c65b110c...977`. The distinct hotfix
  identity remains in place, and the promotion manifest verifier passes all 8
  files.
- All four final rapid/browser provenance records still pass
  `valid_runtime_provenance()`, retain contract
  `production-hotfix-notext-watchdog-v1`, immutable image ID
  `sha256:b9004a53a22c0a1b5b54295221dbb35f552f13b6366a4ecf26899339312d84cb`,
  the 30-second production value, and source hashes matching the current tree.
- The browser aggregate mutation remains load-bearing: changing the retained
  adjudicated L1-2 completion reason to `no_text_since_bos_watchdog` still
  yields aggregate `passed=false`, one watchdog truncation, and L1-2 in the
  unclassified mismatch list.

CPU-only verification, with browser/live markers excluded and no service run:

- full offline suite excluding only the documented revision-pinned phase-0
  assertion: **1,352 passed, 15 skipped, 4 deselected, 29 subtests passed**;
- response-completion/browser aggregate selection: **16 passed, 85
  deselected**;
- focused watchdog/deadline selection: **10 passed, 115 deselected**;
- promotion manifest verifier: **PASS, 8 files**;
- task-file Ruff lint and format checks: **PASS**;
- `git diff --check`: **PASS**.

The discriminator repair is improved but is not genuinely tamper-resistant,
and its serialized producer vocabulary is not yet false-positive-safe. The
correct promotion verdict remains **FAIL**.
