# Adversarial review: Step 2 implementation, round 2

Review date: 2026-08-13 UTC. Scope: the r3.3 Step 2 contract, all six round-1
closure conditions, the fix note, and the requested implementation/test files.
I did not touch the running stack, containers, or model WebSocket.

## Verdict

**FAIL. Step 2 is not code-complete and the live phase must not yet be
scheduled.** The ordinary hot paths, watchdog behavior tests, receipt-boundary
timestamp, bounded record queue, browser batching, 67 anchors, and full suite
are green. Five offline boundaries remain open.

## 1. The qualification proof is copy-forgeable at the engine boundary

The environment matcher itself now rejects the fresh candidate-cross-mixing,
HF/offline, codec, typed-input, and extra managed-key attacks. Both checked-in
candidate environments match only with the three declared deltas. The retained
object is not, however, a proof of those validated fields.

`ValidatedQualificationMode` is a public frozen dataclass carrying a private
sentinel (`provenance.py:90-109`), and the engine verifies only `isinstance`
plus sentinel identity (`server.py:2897-2912`). A legitimately minted object
can therefore be changed while retaining the sentinel:

```python
forged = dataclasses.replace(
    validated,
    no_text_watchdog_override_frames=999999,
)
assert is_validated_qualification_mode(forged)
engine = VoiceChatEngine(pipeline, qualification_mode=forged)
assert engine.agent_silence_watchdog.no_text_required_frames == 999999
```

The digest remains the digest for the original 120-frame environment. The
same attack changes the label to `production` while retaining a valid result.
An inherited class overriding `__getattribute__` for the candidate and frame
fields is also accepted because the check uses `isinstance`; my probe made
health/snapshot report a forged candidate and installed a 777777-frame
threshold. Shallow copy preserves validity, while deep copy and pickle happen
to lose sentinel identity and are rejected. Direct construction with an
unrelated sentinel is rejected.

Mode labels have syntax but no semantic identity: `production`,
`not-qualification`, and `x` all validate. r3.3 writes `<label>`, so this is not
a separate bypass of the environment comparison, but it does not supply a
positive named mode contract.

Required delta: bind every consumed field to the validator result at the
engine boundary and require the exact proof type, not `isinstance`. A copied or
subclassed object with any changed label, candidate, threshold, or digest must
fail. Add `dataclasses.replace`, shallow/deep copy, pickle, direct-construction,
and subclass probes. If the label is intended to identify a preregistered mode,
pin its allowed value(s).

## 2. Trace finalization can publish a false-valid artifact, and timeout input is not safe

The queue is record-bounded, overflow is nonblocking, ordinary write failure is
isolated, and a positive finite close timeout keeps cleanup off the event loop.
The terminal artifact is created before the worker finishes and before the
stream is closed (`llm.py:154-174,192-223`). A stream that successfully
writes/flushes but raises from `close()` produced:

- returned writer status: `valid=false`, `invalid_reason=close-error`;
- persisted terminal `voicechat.playout.trace_status`: `valid=true`, no error.

The observed-trace consumer trusts the persisted terminal and can therefore
qualify an artifact whose writer's authoritative result is invalid.

The traced service also parses its close timeout without requiring a finite
positive value (`llm.py:1601-1603`) and catches only timeout during cleanup
(`llm.py:1608-1621`). Fresh probes found:

- `0`: cleanup raises `ValueError`;
- `-1`: cleanup returns after marking `async-cleanup-timeout` while the writer
  remains open/alive;
- `nan`: cleanup returns with the writer alive and queued control records; and
- `inf`: cleanup raises `OverflowError`.

Required delta: validate a finite positive timeout at construction, and make
artifact validity reflect completion of write, flush, and close. The checked
consumer must not be able to accept the on-disk file when finalization failed;
use an externally checked finalization result/sidecar or an equivalent atomic
artifact protocol. Pin close-after-flush failure and invalid timeout values.

## 3. The observed-trace consumer still accepts contradictory or clockless traces

The new consumer correctly rejects the round-1 impossible-record injection,
off-by-one audio ordinal, orphan/duplicate push, count/format errors, and a
release that lists an already pushed frame. Equal monotonic timestamps are
accepted, which is appropriate.

It does not enforce the exact record schema or the service's event-handler
ordering. The clock check selects the first present field from a global list
and treats no clock as valid (`stratified_latency_analyzer.py:1164-1177`);
schema-tagged unknown event types fall through unhandled
(`stratified_latency_analyzer.py:1113-1120`). With a valid metrics row so the
whole analyzer ran, each of these fresh traces returned
`observed_playout_trace.valid=true`:

- release and downstream-push records with their clocks removed;
- a downstream push with `client_received_monotonic_s=1.22` masking its actual
  `client_downstream_push_monotonic_s=0.1` regression;
- an unknown `voicechat.playout.teleport` schema record;
- `response.done` interleaved between a threshold release and its synchronous
  downstream push; and
- interruption/clear after `response.done` but before the required done
  release, an ordering the traced handler cannot emit.

The trace also carries no configured prebuffer identity, so a `threshold`
release is accepted at any held sample count. The consumer validates linkage,
but cannot reconcile the reason against the configured theoretical release
threshold as claimed.

Required delta: define an exact event-type-to-clock schema, require exactly the
appropriate finite clock on every record, reject unknown schema records and
extraneous clock fields, and encode the atomic ordering constraints of the
traced service. Include the configured prebuffer/reserve identity and verify a
threshold release against it. Add the five accepted contradictions above as
negative tests.

## 4. Browser teardown does not guarantee export of the final worklet batch

Steady-state load is substantially improved: at 48 kHz/128 samples, the
worklet posts about 5.86 batches/s and dequeue records cross the Playwright
binding in about 2.93 128-record chunks/s, rather than 375 calls/s. Clock names,
the parallel-silent-tap caveat, startup/teardown output timestamp mapping,
capture-on/off counters, Python sink `finally` coverage, and the live test's
browser/context `finally` coverage are present. The generated JavaScript passes
`node --check`.

The final worklet batch is not synchronized. Shutdown posts `kind: flush` to
each worklet, waits only one `setTimeout(0)`, then flushes the current main-thread
buffer and closes the context (`test_browser_smallwebrtc_e2e.py:321-336`). A
zero-delay main-thread timer does not acknowledge that the AudioWorklet handled
the message or that its response reached `port.onmessage`. Late records can be
added after `flushExport()`/`Promise.all()` and never exported, or can call the
binding after the Python sink closes.

The real Chromium test passed, but its retained artifact ended at dequeue
ordinal 1344—exactly 21 full 64-quantum worklet batches—with no acknowledged
partial render batch before the teardown clock record. Repeated full-suite and
standalone runs showed the same full-batch boundary. The existing test checks
only that at least one dequeue exists and cannot detect tail loss.

Required delta: add a per-worklet flush acknowledgement carrying the final
ordinal, await every acknowledgement and delivered `render-batch`, then flush
and await a bounded set of binding writes before closing the context/sink.
Assert the persisted last ordinal equals each worklet's acknowledged final
ordinal on normal and injected-failure `finally` paths.

## 5. The construction selector can produce a malformed shallow copy

The ordinary hot methods themselves are functionally byte-identical to HEAD;
there is no remaining per-event or per-frame trace-off branch. The intended
construction selector in `NemotronVoicechatLLMService.__new__`
(`llm.py:274-277`) does change reconstruction semantics that the current test
misses.

I constructed an ordinary service with tracing off, enabled the trace
environment, and called `copy.copy(service)`. Reconstruction invoked the base
`__new__` and returned `_TracedNemotronVoicechatLLMService`, but shallow copy
does not run the traced initializer. The result dispatches to traced hot
methods while lacking `_playout_trace` and all other trace state. With tracing
still off, shallow copy correctly preserves the ordinary type. Pickling was
already unsupported by an existing local lambda.

Required delta: select the variant through an explicit construction factory
that does not participate in copy reconstruction, or explicitly reject/preserve
shallow-copy type. Add off→off, off→trace-env, and traced-copy probes so the
selector cannot silently create a half-initialized service.

## Verification record

- Canonical anchor verifier: **PASS, 67/67 anchors**.
- Full suite: **858 passed, 16 skipped, 29 subtests passed**; four existing
  dependency/async-resource warnings.
- Closure-focused files: **62 passed, 1 skipped**.
- Task-scoped Ruff: **all checks passed**.
- Generated browser capture JavaScript: `node --check` passed.
- Tracked task-file `git diff --check`: clean.
- Independent pre-Step-2 code-object comparison: the functional bytecode,
  constants, names, locals, free/cell variables, arguments, and flags match for
  the ordinary server watchdog `observe`/`snapshot` and Pipecat
  `cleanup`, receive, dispatch, audio, release, clear, interruption-reset, and
  done methods. Step 0/1 anchor semantics are unperturbed.
