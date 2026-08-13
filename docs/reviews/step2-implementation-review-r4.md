# Adversarial review: Step 2 implementation, round 4

Review date: 2026-08-13 UTC. Scope: the r3.3 Step 2 contract and Round-3
implementation-log entry in `docs/generation-frame-latency-plan.md`, including
its recorded threat-model boundary; deltas 1, 2, 3, and 5 from
`docs/reviews/step2-implementation-review-r3.md`; and the Round-3 fix note. All
probes were fresh and offline. I did not touch the running stack, containers,
model WebSocket, or any live endpoint.

## Verdict

**PASS - Step 2 code-complete - the separately coordinated live phase may
proceed only under the preconditions and retention requirements below.**

All four remaining Round-3 deltas are closed. Delta 4 remains closed through
the focused browser tests. No remaining implementation delta was found.

The review applied the threat model exactly as recorded in the plan:
misconfiguration and careless in-process use are in scope; arbitrary hostile
code already executing in the Python process is not. In particular, I did not
fail the gate on attacks that require mutating the private registry or its
authority with `object.__setattr__`, or patching the provenance/server modules.
That remains an informational pure-Python exposure, not a Step 2 blocker.

## 1. Registry authority, consume-once behavior, visible rewrites, and labels

**Closed.** `ValidatedQualificationMode` is now an identity-keyed, exact-type
caller token (`provenance.py:93-114`). Validator-owned values live in the
private locked registry, and the engine boundary atomically removes one entry
before constructing its own frozen identity (`provenance.py:127-158`,
`server.py:2908-2930`). Caller-visible values are not used as authority.

Fresh probes established all of the following:

- I rewrote all four visible fields on a legitimate frozen result to
  `production`, `forged-candidate`, 999999 frames, and a zero digest. Two
  threads then raced to construct engines from that same identity. Exactly one
  succeeded, it installed the original validator-owned values
  (`step2-latency`, `production-candidate-1`, 125 frames, original digest), and
  the other received `TypeError`. A later validity check was false.
- Two distinct validator invocations with identical visible fields retained
  independent authority and each constructed one engine. Reusing either
  consumed result was refused.
- `copy.copy`, `copy.deepcopy`, and `dataclasses.replace` results had no
  authority; direct and subclass identities remained rejected by the focused
  tests; pickle was explicitly refused.
- Raw labels with a leading space, trailing space, or trailing tab were all
  rejected. `production` was also rejected. The validator applies the grammar
  directly to the raw value without stripping (`provenance.py:282-290`).

This closes the requested consume-once, reachable visible-field rewrite,
cross-invocation, copy/replace/subclass, and verbatim-label cases within the
recorded boundary.

## 2. Concurrent publication, orphan quarantine, and rollback

**Closed.** The writer uses a unique sibling staging file with an owned
`flock`, serializes orphan scans per requested destination, publishes with
same-filesystem `os.link`, and cleans staging on every handled failure
(`llm.py:201-267`). Publication cannot replace an existing destination.

The independent filesystem probe used actual concurrent Python processes, not
threads, for the publication race:

- Two processes opened separate staging files for one initially absent
  destination, recorded distinct markers, and crossed a shared start gate.
  Exactly one returned `valid=true` and `artifact_published=true`; the loser
  returned `publish-error`. The requested artifact contained only the winner's
  marker, and no matching pending files remained.
- A separate live process held an exclusive lease on its pending file while a
  contender initialized for the same destination. The live pending file
  remained at its original path and was not copied or moved into quarantine.
  The contender published; when released, the live writer lost the no-replace
  race and synchronously removed its own staging file.
- A separate writer was killed while its leased pending file existed. The next
  initializer moved that unlocked orphan to
  `.voicechat-playout-trace-quarantine/<pending-name>.orphan`, never to the
  requested evidence path, and published only its fresh marker.
- An injected failure at `os.link` returned `publish-error`, left the
  destination absent, and removed the staging file. A separate injected first
  directory-`fsync` failure after link creation rolled the destination link
  back and also removed staging.

The focused suite independently covers destination creation between open and
publish plus handled write, flush, close, link, and durability failures. It
also repeats killed-writer recovery. Failed or orphaned staging bytes are never
accepted as the requested evidence artifact.

## 3. Producer-exact observed-trace consumer

**Closed.** The consumer enforces exact per-event keys and scalar types, a
single exact success terminal, positive audio sample counts, response-local
ordinal/release/push accounting, and a persistent cancelling state
(`stratified_latency_analyzer.py:1251-1356,1359-1628`).

Fresh reconstruction of the four Round-3 traces produced these failures:

| Trace | Fresh rejection |
|---|---|
| Audio/release/push after interruption clear | `audio delta contradicts lifecycle` |
| Schema-tagged microphone speech-start | `typed speech-start schema mismatch` |
| Zero-sample PCM chain | `audio delta field schema mismatch` |
| `valid=true` terminal carrying writer failure | `terminal trace status is inconsistent` |

Two new producer-impossible traces of my own design also failed closed:

1. A response ordinal jumped from 1 directly to 3, with the later
   release/push chain altered to look internally complete. It was rejected as
   `audio delta ordinal mismatch`.
2. After a valid threshold release and downstream push of ordinal 1, a typed
   interruption claimed to clear that already-pushed ordinal. The producer's
   held buffer is empty at that point. It was rejected as
   `release ordinals contradict held frames`.

These are distinct from the four Round-3 cases and exercise the producer's
monotonic ordinal generator and mutually exclusive held/pushed ownership.

## 5. Copy, deepcopy, and pickle contracts

**Closed.** The service base class explicitly rejects shallow copy, deep copy,
`__reduce__`, and `__reduce_ex__` (`llm.py:385-395`); the contract is inherited
by traced and derived variants.

A fresh matrix rejected `copy.copy`, `copy.deepcopy`, and highest-protocol
pickle for an ordinary service, traced service, ordinary subclass, and traced
subclass. Deep-copying an ordinary instance after enabling the trace
environment was also rejected, so environment-sensitive `__new__` could not
reconstruct a half-initialized traced instance. This closes the original
off-to-off, off-to-trace, traced, pickle, and subclass cases.

## Verification record

- Closure-focused suite: **106 passed, 1 skipped**; one dependency
  deprecation warning.
- Canonical anchor verifier: **PASS (67 anchors)**. Output was written only to
  `/tmp/step2-r4-canonical-anchors.json`.
- Full suite: **902 passed, 16 skipped, 29 subtests passed**; four existing
  dependency/async-resource warnings.
- Task-scoped Ruff over the established eight implementation/test files:
  **all checks passed**.

No live test was run and no commit was created.

## Preconditions for the separately coordinated live phase

The live phase must fail closed unless every item below is true before the
first fixture is sent:

1. Use a reviewed immutable runtime image containing this exact source, and
   record its image digest and source manifest. The host Pipecat/analyzer code
   must be the same reviewed checkout.
2. Check the single-client health/lock immediately before the probe. Abort if
   the stateful batch-1 service is occupied or not clean. Do not use
   `./voicechat up` as part of the probe and do not implicitly restart the
   current stack.
3. Start from fresh, absent Pipecat and browser artifact paths in a new
   session directory. A `.pending` file or quarantine entry is never evidence.
4. Supply an explicit non-production label matching
   `^[a-z0-9][a-z0-9-]{2,63}$` through
   `VOICECHAT_QUALIFICATION_MODE`, along with the qualification system-prompt
   delta and a positive
   `VOICECHAT_QUAL_NO_TEXT_WATCHDOG_OVERRIDE_FRAMES`. Every other managed
   environment value must exactly match one checked-in candidate.
5. Before sending media, capture `/health` and require its qualification
   identity to show the intended mode, candidate, frame threshold, and
   environment SHA-256. Refusal, absence, or mismatch aborts the run.
6. Preregister the known 8-10 second continuous-speech scripts, duration bins,
   transport conditions, independent-response counts, reserve sweep,
   interruption positions, structural K=12 rule, maximum-gap bound, and audio
   completeness adjudication. Transcript presence alone is not a fixture
   validity criterion.
7. Enable both Pipecat playout JSONL and browser AudioWorklet capture. Treat
   browser/Pipecat intervals as relative unless the retained clock mappings
   prove an offset. Run the paired capture-off control needed to quantify
   capture overhead.

## Exact evidence the live phase must retain

Retain one immutable session directory plus a SHA-256 manifest covering every
item below. Preserve failures and negative results as well as successful runs.

- The preregistration: fixture/script bytes and hashes, expected audio content,
  duration/population declarations, response IDs, transport setup, reserve
  grid, interruption sweep, K=12/max-gap gates, and audio-completeness rule.
- Runtime provenance: image digest, checked-out source hashes, checked-in
  candidate/config identity, complete effective managed environment or its
  canonical manifest, and the exact qualification prompt, label, override
  threshold, and environment digest.
- The pre-fixture single-client health/lock result and full `/health` JSON,
  including `mode`, `candidate`, `frames`, and `environment_sha256`; server
  per-frame watchdog snapshots must carry the matching qualification identity.
- The requested Pipecat artifact (conventionally
  `pipecat-playout.jsonl`) with its final
  `voicechat.playout.trace_status` exactly successful: `valid=true`,
  `invalid_reason=null`, `error=null`, and integer `dropped_records=0`.
  Retain all config, response, delta, metric, interruption, release, and
  downstream-push records and the artifact SHA-256. Do not retain a pending or
  quarantined file as a substitute.
- The browser artifact (conventionally `browser-playout.jsonl`) with capability
  and clock-mapping records, remote-audio enqueue records, contiguous worklet
  dequeue/onset records, and one `browser.audio_worklet.flush_ack` per
  registered worklet. For every worklet, the persisted last dequeue ordinal,
  acknowledged final ordinal, and worklet final ordinal must be identical.
- The complete live-test stdout/stderr and
  `BROWSER_PLAYOUT_CAPTURE_OVERHEAD` reports for capture-on and paired
  capture-off runs, including elapsed time, exported chunk/record counts,
  binding-handler time, and any shutdown/sink-close error.
- Raw server event/metrics traces and the exact input and rendered output audio
  files with response/ordinal/sample-count membership. Retain `response.done`
  and all interruption edges so completeness and barge-in accounting can be
  independently replayed.
- The audio-completeness adjudication for every known script, including the
  rendered WAV hash, scorer/evaluator identity and version, inputs, detailed
  result, and the evidence that the complete script was spoken. A text-channel
  transcript by itself is insufficient.
- The analyzer output (conventionally `arrival-analysis.json`) and command
  parameters. It must retain `observed_playout_trace.valid=true`, provable
  delivered/delta ordered reconciliation and membership hash, normative delta
  receipt-clock structural result, configured-prebuffer identity, the complete
  reserve replay sweep, interruption outcomes, and all declared analyzer gate
  verdicts. Retain the browser-derived enqueue/dequeue/underrun summary and its
  linkage to the raw browser artifact as a separate result on the same session.

Step 2 is code-complete; those retained live artifacts, not this offline pass,
will decide the live fixture and measurement claims.
