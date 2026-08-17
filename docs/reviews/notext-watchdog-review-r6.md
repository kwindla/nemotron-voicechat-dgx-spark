# No-text watchdog repair — confirmation review R6

**Verdict: PASS — promotable.** The exact R5 producer-vocabulary drift now
fails the synchronization test, the vocabulary comparison contains no
test-owned serialized-event constant, and the producer-level test exercises
the real diagnostic producer, `send_step()`, and `SessionTrace` JSONL
serialization. All nine retained traces parse with zero schema errors, the
pre-fix FAIL and all eight repaired PASS results reproduce, and no in-scope
regression remains in the watchdog, wall deadline, deployment/provenance,
browser aggregate, or mutation coverage.

## R5 blocker is closed

The vocabulary test now derives both kinds of serialized discriminator from
`src/nemotron_voicechat_runtime/server.py`:

- `_source_trace_event_vocabulary()` parses every `.event()` call and asserts
  that every first argument is a string literal
  (`tests/runtime/test_response_completion_gate.py:173-189`). A future
  nonliteral call therefore fails rather than disappearing from enumeration.
- `_serialized_trace_event_nodes()` discovers the mapping subscript expanded
  through `**result.trace_diagnostics["watchdog"]`, resolves the corresponding
  producer mapping assignment, and extracts its literal `"event"` value from
  that assignment (`test_response_completion_gate.py:135-170`). Missing,
  indirect, or nonliteral mapping resolution fails closed.
- The vocabulary assertion is now solely the union of those AST-derived
  values against `RECOGNIZED_TRACE_EVENTS`
  (`test_response_completion_gate.py:401-403`). The former
  `serialized_variants = {"idle_agent_without_pair_buffer"}` synchronized
  test constant is gone. The separate producer-behavior assertion for that
  string in `test_realtime_web_server.py` does not participate in vocabulary
  construction and cannot keep this equality green after producer drift.

I independently reproduced the exact R5 condition without modifying a
repository file. In a temporary copy of `server.py`, I changed only:

```text
"event": "idle_agent_without_pair_buffer"
```

to:

```text
"event": "renamed_serialized_diagnostic"
```

I then pointed the unchanged
`test_gate_vocabulary_covers_every_source_and_serialized_producer_event()` at
that copy. The baseline and mutated derived vocabularies both contained 61
elements, but the precise set delta was:

```text
removed=['idle_agent_without_pair_buffer']
added=['renamed_serialized_diagnostic']
```

The vocabulary test raised `AssertionError`: **FAIL as required**. The added
drift regression also passes in the unmodified tree. This directly reverses
the R5 result where the producer source changed while the asserted vocabulary
remained unchanged.

## Producer test uses the real diagnostic and serialization paths

`test_current_session_trace_diagnostic_serialization_passes_total_parse()` no
longer constructs the watchdog payload. It:

1. creates a minimal `VoiceChatEngine` instance and feeds 25 enabled,
   unbuffered observations through the real
   `VoiceChatEngine._pad_pair_trace_diagnostics()` threshold/state logic
   (`test_response_completion_gate.py:438-482`);
2. takes the resulting `trace_diagnostics["watchdog"]` mapping and places it
   on a real `StepResult` (`test_response_completion_gate.py:501-516`);
3. invokes the real asynchronous `send_step()` twice, which emits the
   `model_step` and expands the diagnostic into the watchdog trace call
   (`server.py:6003-6105`);
4. uses a real file-backed `SessionTrace`, whose `event()` merge and JSONL
   serialization produce the retained records (`server.py:2461-2493`); and
5. reloads that JSONL with `load_jsonl()` and passes those exact records to
   `response_completion_gate()` (`test_response_completion_gate.py:518-575`).

The resulting event sequence is
`session_trace_started`, `model_step`,
`idle_agent_without_pair_buffer`, `model_step`, `session_trace_stopped`.
The serialized diagnostic fields equal the actual producer mapping, and the
real gate reports `input_schema_valid=true`, an empty schema-error list, and
`passed=true`. The surrounding two-step response is synthetic, but the
watchdog payload, producer threshold transition, `send_step()` dispatch, and
`SessionTrace` serialization under review are not hand-built substitutes.

## Retained traces reproduce with zero schema errors

I loaded all nine `events.jsonl` files directly and reran the current gate.
Every result had `input_schema_valid=true` and
`input_schema_errors=[]`:

- Pre-fix `2f88c304-b895-4b81-aa1d-adde6043d58c`: expected **FAIL**, 5
  responses, 4 audible watchdog defects, and 1 dead rescue. Its bytes still
  match pinned SHA-256
  `bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`.
- Completion-evidence repairs
  `5ca16bcb-9c4a-4074-a69e-951c173e591e`,
  `5a09f1bd-c027-453e-814a-0e8716dad110`,
  `60d0ecf8-6a5e-4ab4-ba21-6f163bcd6c16`, and
  `93d79622-9d69-4f88-8c6c-f5e5774d9245`: expected **PASS** at 8, 1, 8,
  and 8 responses, each with zero audible watchdog defects.
- Final rapid/browser repairs
  `9549ed8a-22e6-4261-9bd0-21427e87ac3b`,
  `1d188f45-a7bd-40e0-b039-0ef4d17c1c52`,
  `793f01ab-42b3-4644-821b-0d12e882f683`, and
  `c203bca9-0f7c-4544-925a-f1d7cb9ab962`: expected **PASS** at 8, 8, 8,
  and 1 responses, each with zero audible watchdog defects. Their raw hashes
  match `reports/notext-watchdog/live-validation-r2.json`.

## No repair regression

- The complete response-gate suite passes: **18 passed**. This includes the
  exact R4 relabel-plus-deleted-frame and nested-envelope mutations, omitted,
  relabeled, and unknown discriminators, deleted acoustic evidence, non-finite
  thresholds/output, invalid frame/session data, response topology failures,
  the audible-watchdog mutation, and the newly bound producer drift.
- The response-completion/browser-aggregate selection passes: **22 passed, 85
  deselected**. Mutating the retained adjudicated browser mismatch to
  `no_text_since_bos_watchdog` still produces a failed aggregate, one watchdog
  truncation, and an unclassified failing fixture.
- The focused real-time watchdog/deadline selection passes: **19 passed, 106
  deselected**. The reset-on-audible transition, dead BOS, short text/no audio,
  live audible audio, audible-then-silent decoding, whole-response wall timer,
  strict 30-second hotfix configuration, real-clock expiry without model
  frames, one-shot EOS, stale-generation protection, and no-redirect behavior
  remain covered.
- The promotion/config/latency/analyzer selection passes: **109 passed**. The
  promotion verifier independently reports **PASS, 8 files**, including the
  fail-closed manifest mutation test.
- Frozen candidate 1/2/3 TOMLs remain byte-identical to `HEAD` at SHA-256
  `e63d22b3ba26370a880266457e0a3a55d41d6e8c689132faadc6fbb25caebddc`,
  `00377c19b1a33ed93e3e3eec15a452541ee051ec20395760878cff78ee5eed50`,
  and `c65b110c58656d3ddf65ac3829a57f2bbb38c3c369b9d2391d8d2878af37e977`.
- The two rapid and two browser provenance records all pass
  `valid_runtime_provenance()`, retain the versioned hotfix contract, immutable
  image ID
  `sha256:b9004a53a22c0a1b5b54295221dbb35f552f13b6366a4ecf26899339312d84cb`,
  exact hotfix semantic environment, and 30-second wall budget. All 11 source
  hashes present in the current tree match for every record.
- The broad CPU-only suite, with the three live `browser_e2e` entries and the
  documented unrelated revision-pinned phase-0 assertion deselected, passes:
  **1,358 passed, 15 skipped, 4 deselected, 29 subtests passed**. The four
  warnings are unchanged. Ruff lint, round-file format checks, compileall, and
  `git diff --check` pass.

The broad run caused no tracked-file changes. No live URL, browser campaign,
service, container, or model execution was used.

## Scope conclusion

The remaining theoretical cases that require arbitrary rewriting or
replacement of retained trace bytes, or arbitrary repository-source control,
are outside the supplied first-party qualification threat model and do not
affect this verdict. Within scope, ordinary producer vocabulary drift now
fails closed through source derivation and the real producer/serializer/gate
integration test. There are no remaining in-scope deltas.

**Final verdict: PASS — promotable.**
