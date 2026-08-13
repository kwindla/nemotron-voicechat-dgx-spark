# Adversarial review: Step 2 implementation, round 3

Review date: 2026-08-13 UTC. Scope: the r3.3 Step 2 contract and Round-2
implementation-log entry in `docs/generation-frame-latency-plan.md`, the five
required deltas in `docs/reviews/step2-implementation-review-r2.md`, and the
Round-2 fix note. All probes were offline. I did not touch the running stack,
containers, or model WebSocket.

## Verdict

**FAIL. Step 2 is not code-complete and the live phase must not be scheduled.**
The browser flush-ack delta is closed, and all requested repository checks are
green. Fresh attack shapes leave deltas 1, 2, 3, and 5 open.

## 1. The retained proof seal is mutable, and mode labels are not verbatim

The exact-type check, original-result back-reference, engine-owned copy, normal
copy rejection, and cross-validator proof transplant all work. Weak references
are unavailable for both the result and seal, so a weakref/GC resurrection
attack was not plausible. Reusing one legitimate result to create two engines
also produces two independent engine-owned identities.

The seal is nevertheless reachable and mutable. `ValidatedQualificationMode`
retains `_proof` (`provenance.py:127-135`), while
`_QualificationValidationProof` exposes writable slots for every sealed value
and its result back-reference (`provenance.py:92-115`). The boundary check only
compares the result fields with those mutable slots (`provenance.py:138-153`).
A fresh probe performed this sequence:

```python
proof = object.__getattribute__(validated, "_proof")
proof.label = "production"
proof.candidate = "forged-candidate"
proof.no_text_watchdog_override_frames = 999999
proof.environment_sha256 = "0" * 64
object.__setattr__(validated, "label", proof.label)
object.__setattr__(validated, "candidate", proof.candidate)
object.__setattr__(validated, "no_text_watchdog_override_frames", 999999)
object.__setattr__(validated, "environment_sha256", proof.environment_sha256)
```

`is_validated_qualification_mode(validated)` then returned true. Threading that
same forged object into two `VoiceChatEngine` constructors installed and
retained `production`, `forged-candidate`, and 999999 frames in both engines.
The engine-owned copy at `server.py:2908-2930` prevents later aliasing, but it
copies the already-forged values.

The r3.3 label grammar also is not applied to the verbatim environment value.
The validator strips it before matching and hashing (`provenance.py:293-328`).
Fresh probes accepted both `" step2-latency "` and
`"step2-latency\t"`, storing `"step2-latency"`. The three-character minimum,
64-character maximum, 65-character rejection, and exact `production` refusal
otherwise behaved correctly. Silent normalization violates the implementation
log's requirement that the label match
`^[a-z0-9][a-z0-9-]{2,63}$` and be recorded verbatim.

Required delta: make the validator result an unforgeable capability under the
established in-process attack model, without a caller-mutable seal/value pair.
At the engine boundary, recover validator-owned values from an identity-keyed
private registry (or an equivalent design whose authority is not reachable
through the result), then consume the authority once or copy its sealed values
without consulting caller-mutable state. Reject the raw label unless the raw
value itself matches the grammar; do not strip or normalize it. Pin the
reachable-seal rewrite, identical-result cross-invocation transplant, same-proof
two-engine use, and whitespace-wrapped labels.

## 2. Artifact publication is not no-clobber and failed staging files persist

The initial pre-existing-destination check works: a sentinel destination was
refused with `FileExistsError` and remained byte-identical. A process killed
inside a deliberately stalled real staging-file write left the requested
destination absent, so an interrupted batch was not published as evidence.
Close timeout validation and the close-after-flush false-valid case remain
closed.

The initial existence test (`llm.py:97-105`) and later `Path.replace()`
(`llm.py:189-196`) do not form a no-clobber atomic publication protocol. Two
writers opened the same absent destination, each wrote a different marker, and
each returned `valid=true` and `artifact_published=true`. The second close
silently replaced the first complete artifact; the final file contained only
the second marker. A destination introduced after construction has the same
race.

Staging cleanup is also absent from all failure paths. Fresh probes observed:

- injected mid-batch write error: `writer-error`, no destination, pending file
  still present;
- publish error induced after construction: `publish-error`, pending file still
  present; and
- `SIGKILL` during a batch write: no destination, one pending file still
  present.

The last case cannot be cleaned by the killed process, but a later writer does
not quarantine or clean the orphan either. The first two are ordinary handled
failures and should clean synchronously. The safety property “failed evidence
is not published” holds, but the claimed destination exclusivity and staging
cleanup do not.

Required delta: publish with an atomic no-replace primitive so competing writers
cannot both succeed (for example, a same-filesystem link/no-replace operation
with explicit durability handling), and unlink the staging file after every
handled write/flush/close/publish failure. Define and test recovery or
quarantine of crash-orphaned `.pending` files without ever treating them as
evidence. Pin two writers racing one initially absent destination, destination
creation between open and publish, handled write/publish failures, and killed
writer recovery.

## 3. The checked consumer accepts four new producer-impossible traces

The Round-2 clock, unknown-type, release/push atomicity, done/release atomicity,
and prebuffer checks are closed. Four new traces, distinct from the pinned
Round-2 contradictions, each returned
`observed_playout_trace.valid=true`:

1. **Audio after interruption clear.** A held frame was interrupted and cleared,
   then ordinal 2 arrived and was done-released/pushed. The service sets
   `_response_cancelling` at interruption and ignores subsequent deltas, but the
   consumer only rejects a delta while `interruption_pending`; it permits one
   after `interruption_cleared` (`stratified_latency_analyzer.py:1257-1275,
   1311-1324`).
2. **Non-typed speech-start trace record.** A schema-tagged
   `input_audio_buffer.speech_started` with `source="microphone"` was accepted as
   a no-op. The traced producer records this event type only for `source="typed"`;
   the consumer admits the type globally, but its interruption branch handles
   only typed source and lets every other source fall through
   (`stratified_latency_analyzer.py:1276-1289`).
3. **Zero-sample audio chain.** An empty base64 PCM delta with `sample_count=0`,
   followed by a zero-sample done release and downstream push, was accepted. The
   producer drops empty decoded audio before recording it; the consumer validates
   equality but never requires a positive decoded sample count
   (`stratified_latency_analyzer.py:1257-1274`).
4. **Self-contradictory terminal.** A terminal with `valid=true`,
   `dropped_records=0`, `invalid_reason="writer-error"`, and a non-null error was
   accepted because only the first two fields are checked
   (`stratified_latency_analyzer.py:1166-1178`). Such a terminal cannot belong to
   a successfully published writer artifact.

Required delta: model interruption as a cancelling terminal state and reject all
later deltas/releases except the matching completion; reject producer-impossible
variants of known event types rather than treating them as no-ops; require a
positive audio payload/sample count; and validate the terminal status fields as
one consistent producer state. Add these four traces as negative tests and
prefer exact per-event field/type schemas so Python `bool`/`int` equality and
extraneous fields cannot create the next bypass.

## 5. `deepcopy` still reconstructs a half-initialized traced service

Explicit `copy.copy()` refusal works for ordinary, traced, and subclass
instances. Pickle currently fails on Pipecat's local lambda, and a derived-class
deep copy preserved the derived type. The base class's environment-sensitive
`__new__` remains in the reconstruction path, however
(`llm.py:293-305`).

A fresh probe constructed an ordinary base service with tracing off, enabled
`NEMOTRON_VOICECHAT_PLAYOUT_TRACE`, and called `copy.deepcopy(service)`. It
returned `_TracedNemotronVoicechatLLMService` with no `_playout_trace` or traced
ordinal state because deep copy does not run the traced initializer
(`llm.py:1619-1651`). Dispatching one audio delta failed with:

```text
AttributeError: '_TracedNemotronVoicechatLLMService' object has no attribute
'_playout_trace_ordinals'
```

Required delta: reject or explicitly preserve variant identity for deep copy as
well as shallow copy. If service instances are intentionally non-serializable,
make deepcopy and pickle refusal explicit service contracts rather than relying
on a dependency's incidental local lambda. Pin off-to-off deep copy,
off-to-trace-environment deep copy, traced deep copy, pickle, and shallow/deep
copy of a subclass.

## Closed delta and verification record

Round-2 delta 4 is closed. In real headless Chromium, a generated worklet
modified to acknowledge final ordinal 67 after delivering 60 failed immediately
with the ordinal mismatch. A second generated harness changed the acknowledgement
kind so the worklet never acknowledged; with the same bounded-wait path shortened
to 100 ms for the probe, it failed closed after 104 ms with
`worklet flush acknowledgements timed out` (one registered worklet, 92 delivered
dequeues, zero acknowledgements). The persisted sink also requires every
registration to have one matching final ordinal (`test_browser_smallwebrtc_e2e.py:
526-540`). No remaining browser delta was found.

- Canonical anchor verifier: **PASS (67 anchors)**.
- Full suite: **882 passed, 16 skipped, 29 subtests passed**; four existing
  dependency/async-resource warnings. The known `deferred_tool_timeout` flake did
  not occur, so no isolated rerun was needed.
- Closure-focused files: **86 passed, 1 skipped**.
- Task-scoped Ruff over all eight implementation/test files: **all checks
  passed**.
- Generated browser capture JavaScript: **`node --check` passed**.
- Task-file `git diff --check`: **clean**.

No commit was created.
