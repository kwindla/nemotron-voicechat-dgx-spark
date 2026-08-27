# No-text watchdog repair — confirmation review R5

**Verdict: FAIL — not promotable.** The strict total parser closes the exact R4
evasion, the current diagnostic false positive is fixed, every retained unmutated
trace has zero schema errors, and the earlier watchdog/deadline/deployment/
provenance/browser results remain intact. One in-scope delta remains: the new
producer-vocabulary test is not actually bound to the producer's serialized
diagnostic discriminator. A first-party source change can therefore drift the
producer vocabulary while the asserted AST equality and producer-level test both
remain green.

## Blocking delta

### The serialized diagnostic vocabulary is still a synchronized test constant

The literal-call portion of
`test_gate_vocabulary_covers_every_source_and_serialized_producer_event()` is
source-derived, but the nonliteral serialized variant is not. The test assigns
`serialized_variants = {"idle_agent_without_pair_buffer"}` directly and unions it
with the AST result (`tests/runtime/test_response_completion_gate.py:326-343`).
The producer obtains that value from a separate diagnostic mapping
(`src/nemotron_voicechat_runtime/server.py:3972-3978`), which is expanded through
`SessionTrace.event()` and overwrites the declared `nano_pad_pair_watchdog`
discriminator (`server.py:2481-2488`, `6098-6104`). The AST test never reads or
derives the mapping's `event` value.

The producer-level test does not close that gap. It imports `SessionTrace`, but
then manually constructs the same hard-coded overriding payload, including
`"event": "idle_agent_without_pair_buffer"`
(`tests/runtime/test_response_completion_gate.py:346-387`). It proves the present
serializer merge behavior and present gate schema agree; it does not exercise the
code that creates `result.trace_diagnostics["watchdog"]`.

I tested the exact drift condition without changing repository files: in an
in-memory copy of `server.py`, I changed the producer mapping value to
`"renamed_serialized_diagnostic"` and reran the test's AST extraction logic. The
results were:

- the producer source changed;
- the extracted/test vocabulary was unchanged;
- `extracted == RECOGNIZED_TRACE_EVENTS` still evaluated true;
- the new serialized discriminator was not enumerated.

There is a second prospective blind spot in the same test: it collects only calls
whose first argument is a string literal but does not assert that all relevant
`SessionTrace.event()` calls have literal first arguments. All 63 current `.event()`
calls are literal (60 unique kinds), so there is no present discrepancy, but a new
`trace.event(kind, ...)` call would be silently omitted rather than failing the
test.

This is not an arbitrary trace-byte-writer concern. It is reachable through an
ordinary first-party producer edit and can recreate the legitimate-trace false
positive that this round is intended to prevent. It is therefore in scope for the
documented goal of detecting silent runtime/telemetry regressions.

**Required delta:** make the serialized event vocabulary mechanically derive from
the producer. The simplest robust option is to prevent `SessionTrace.event()`
payload fields from overriding reserved envelope keys, rename the diagnostic's
nested `event` field, and require every relevant producer call discriminator to be
a literal. Alternatively, exercise the actual diagnostic-producing path and derive
every overriding serialized discriminator from source/runtime producer output
rather than restating it in the test. In either design, add an assertion that fails
on any nonliteral relevant event call and a test proving that changing the
diagnostic producer value alone fails vocabulary synchronization.

## Confirmed closed

### Strict total parse and the exact R4 evasion

`TRACE_EVENT_SCHEMAS` is a discriminator-keyed map of closed top-level shapes
(`tools/qualification/response_completion_gate.py:32-48`, `84-280`). For every
record, `_parse_trace_records()` rejects a missing/unknown discriminator, checks
required and unexpected fields, validates the common envelope, and only then
excludes a validated non-model record from model-step semantic classification
(`response_completion_gate.py:302-412`). Any accumulated error returns the
schema-failure result before classification (`response_completion_gate.py:457-468`).
The former `_is_model_step_shaped` heuristic and pre-validation skip path are gone.

From freshly loaded pinned bytes, the exact R4 mutation selected 208 audible
interior model steps, changed each discriminator to recognized `client_event`, and
removed `frame`. It now returns `passed=false`, `input_schema_valid=false`, and 416
schema errors: 208 missing-required-field errors and 208 unexpected-field errors.
The nested-envelope mutation is also covered and rejected. There is no remaining
skip path through partial deletion/relabeling: a recognized record must match its
one selected closed schema or the entire gate fails.

An actor able to delete a record completely or replace it with a wholly valid,
different record can still forge the stream. That requires arbitrary write access
to retained trace bytes and is explicitly outside the supplied threat model; it is
not a blocker here.

### Current producer false positive and retained traces

The gate now recognizes the producer's current serialized
`idle_agent_without_pair_buffer` diagnostic with its actual closed field set
(`response_completion_gate.py:137-146`). The producer-level serialization test
passes it through `SessionTrace` and the gate with zero schema errors. Thus the
specific current false positive from R4 is fixed, even though the drift test above
is not future-binding.

I reran the gate directly over all nine unmodified retained model traces. Every
trace returned `input_schema_valid=true` and an empty `input_schema_errors` list:

- pre-fix `2f88c304-b895-4b81-aa1d-adde6043d58c`: expected FAIL, 5 responses,
  4 audible watchdog defects, 1 dead rescue; pinned SHA-256
  `bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`;
- completion-evidence repaired traces
  `5ca16bcb-9c4a-4074-a69e-951c173e591e`,
  `5a09f1bd-c027-453e-814a-0e8716dad110`,
  `60d0ecf8-6a5e-4ab4-ba21-6f163bcd6c16`, and
  `93d79622-9d69-4f88-8c6c-f5e5774d9245`: expected PASS at 8, 1, 8, and 8
  responses, with zero audible watchdog defects;
- final rapid/browser repaired traces
  `9549ed8a-22e6-4261-9bd0-21427e87ac3b`,
  `1d188f45-a7bd-40e0-b039-0ef4d17c1c52`,
  `793f01ab-42b3-4644-821b-0d12e882f683`, and
  `c203bca9-0f7c-4544-925a-f1d7cb9ab962`: expected PASS at 8, 8, 8, and 1
  responses, with zero audible watchdog defects; all four retained trace hashes
  match `reports/notext-watchdog/live-validation-r2.json`.

### No regression in the earlier accepted findings

- All 17 response-completion tests pass, including the exact R4 mutation, nested
  payload mutation, all prior schema/acoustic/order/topology mutations, and the
  retained pre-fix FAIL/repaired PASS assertions.
- The response-completion/browser aggregate selection passes: 21 passed, 85
  deselected. The retained browser mutation still turns the adjudicated L1-2 reason
  into `no_text_since_bos_watchdog` and forces aggregate failure.
- The focused whole-response deadline and watchdog selection passes all 10 tests,
  including real-clock expiry without model frames, strict propagation, no
  redirect, reset-on-audible, dead BOS, short text/no audio, decoded silence, live
  audible audio, and duplicate BOS behavior.
- Candidate 1/2/3 SHA-256 values remain
  `e63d22b3ba26370a880266457e0a3a55d41d6e8c689132faadc6fbb25caebddc`,
  `00377c19b1a33ed93e3e3eec15a452541ee051ec20395760878cff78ee5eed50`,
  and `c65b110c58656d3ddf65ac3829a57f2bbb38c3c369b9d2391d8d2878af37e977`.
  The promotion-manifest verifier passes all 8 files.
- The two rapid and two final browser provenance records all pass
  `valid_runtime_provenance()`, retain contract
  `production-hotfix-notext-watchdog-v1`, immutable image ID
  `sha256:b9004a53a22c0a1b5b54295221dbb35f552f13b6366a4ecf26899339312d84cb`,
  the 30-second whole-response budget, and no mismatches among source-hash entries
  present in the current tree.

## Threat-model assessment

The module documentation is honest and appropriately bounded for a first-party
qualification tool (`response_completion_gate.py:2-11`). It claims detection of
silent runtime regressions, malformed/incomplete telemetry, and adjudication
absorption in `SessionTrace` output; it explicitly says mutation coverage is
robustness evidence rather than an integrity guarantee against arbitrary trace-byte
writers. Cryptographic stream integrity, record-deletion resistance, and a fully
valid forged replacement stream are therefore out of scope and do not affect this
verdict. The sole blocker above concerns ordinary first-party producer drift and is
inside the documented boundary.

CPU-only verification was used throughout. No live service was started, and no
repository file other than this review was written.
