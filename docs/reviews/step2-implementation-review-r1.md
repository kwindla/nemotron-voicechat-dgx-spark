# Adversarial review: Step 2 implementation, round 1

Review date: 2026-08-12. Scope: the Step 2 plan and implementation brief, the
completion note, the requested production/test diffs, both checked-in candidate
TOMLs, and the Step 0 analyzer. I did not touch the running stack, containers,
or model WebSocket.

## Verdict

**FAIL.** The watchdog threshold wiring is narrow, and the task-scoped tests,
Ruff, and full suite pass. Those results do not close several production-path
defects:

1. the production refusal is fail-open under any one-key drift and the supplied
   live recipe deliberately exploits that property;
2. tracing changes protocol behavior, timestamps receipt after parsing and
   decoding, has an unbounded writer queue, and can block the event loop during
   cleanup;
3. the trace-off path has added branches on both Pipecat and model-server hot
   paths, contrary to the explicit zero-hot-path-cost requirement;
4. the Step 0 analyzer consumes the wire-arrival subset but ignores the new
   release and downstream-push records entirely; and
5. the browser tap is qualification-only, but its per-render-quantum messaging
   and synchronous per-record flushing can perturb the latency and underruns it
   is meant to measure. Its fidelity caveats are incomplete.

These are implementation blockers, not documentation nits.

## 1. Production refusal is not fail-closed

`canonical_production_environment_active()` returns true only when every key in
its subset has the exact expected value
(`src/nemotron_voicechat_runtime/provenance.py:166-172`). Consequently, changing
or removing any one of those keys turns a nearly production deployment into
“non-production” and admits the override. I reproduced successful acceptance
with each of these single changes:

- `VOICECHAT_WEB_SYSTEM_PROMPT=Qualification latency fixture`
- `HF_HUB_OFFLINE=0`
- `EA_CPU_CODEC_THREADS=3`

In all three cases,
`validate_qualification_watchdog_production_policy()` returned `160`. The
completion note's proposed live command relies on exactly this weakness: it
changes only `VOICECHAT_WEB_SYSTEM_PROMPT` so the production signature no longer
matches. A changed prompt may be intentional in that recipe, but the mechanism
cannot distinguish it from accidental or hostile drift of an unrelated key.
This does not satisfy “only usable in explicitly non-production/qualification
serving configurations.” It treats failure to prove production as proof of
qualification.

The claimed TOML derivation is also not the actual intersection. Independent
`tomllib` parsing found:

- candidate 1 environment: 61 keys;
- candidate 2 environment: 67 keys;
- equal-valued key intersection: 57 keys; and
- `CANONICAL_PRODUCTION_ENVIRONMENT`: 48 keys.

The nine shared typed-input keys are absent from the purported intersection:
`VOICECHAT_TYPED_INPUT_CPU_CORES`, `VOICECHAT_TYPED_INPUT_EOU_MARGIN`,
`VOICECHAT_TYPED_INPUT_FRAME_MS`, `VOICECHAT_TYPED_INPUT_LANGUAGE`,
`VOICECHAT_TYPED_INPUT_MAX_TEXT_CHARS`, `VOICECHAT_TYPED_INPUT_PREWARM`,
`VOICECHAT_TYPED_INPUT_QUANTIZE`, `VOICECHAT_TYPED_INPUT_TORCH_THREADS`, and
`VOICECHAT_TYPED_INPUT_VOICE`. The four documented legacy exclusions are indeed
candidate-1-only, and all 48 implemented canonical values match both TOMLs, but
“`PRODUCTION_ENVIRONMENT` minus four keys is the environment common to both
TOMLs” is false. The new test proves only that the implemented subset matches
both configurations; it does not compute and compare the intersection
independently (`tests/runtime/test_generation_latency_step2.py:62-70`).

There is also a check/use gap. Validation occurs in `build_public_pipeline()`
before model construction (`server.py:4425-4429`), but its return value is
discarded. `main()` reparses the mutable process environment only after pipeline
construction and warmup (`server.py:7914-7921`). External processes normally
cannot mutate a running process's environment, so this is not a shell-level
bypass, but imported/in-process code can, and direct construction of
`VoiceChatEngine(...qualification_no_text_watchdog_override_frames=N)` bypasses
the policy altogether. The new health test uses precisely that unvalidated
path. After engine construction, health reports the effective captured value,
so later environment changes do not make health stale about actual behavior;
the dangerous interval is between validation and parsing.

Required fix:

- Make qualification a positively validated configuration identity, not the
  negation of an exact production-subset match. Prefer a separate validated
  qualification manifest/mode. If no new identity is permitted, accept only an
  explicit allowlist of exact qualification deviations from either complete
  candidate configuration and reject all other drift.
- Derive or independently test the complete intended TOML intersection. Reuse
  the candidate/environment definitions in `artifacts.py` rather than copying a
  second exclusion contract.
- Parse and validate the override once, before model construction, retain the
  validated immutable value, and pass that value into the engine. Enforce the
  proof at the engine boundary as defense in depth so alternate construction
  paths cannot silently enable it.
- Expose the validated qualification identity in health alongside the frame
  count.

## 2. The zero-cost-when-off claim is false

No extra clock read, JSON work, writer allocation, or `await` occurs per audio
frame when the Pipecat env is unset. That narrower claim is supported. The
stronger required claim—zero hot-path cost—is not.

With tracing off, the implementation adds:

- a trace-presence branch for every received WebSocket event
  (`llm.py:1020-1025`);
- trace locals/branches on immediate audio pushes (`llm.py:1371-1383`);
- an ordinal-map branch on every buffered audio delta (`llm.py:1386-1391`);
- a trace branch at every buffer release and another inside the per-frame push
  loop (`llm.py:1407-1422`); and
- trace branches in every buffer clear (`llm.py:1461-1470`).

The server side also adds an optional-field branch to
`AgentSilenceEosWatchdog.snapshot()` (`server.py:2519-2539`). That snapshot is
created on every model step at `server.py:4041`, so this is a production model
hot-path change even when the override is unset.

The trace-off unit test checks one happy prebuffer-zero frame sequence. It
cannot prove absence of these branches or cover threshold release, done flush,
interruption, stale events, or cleanup.

Required fix: keep the existing uninstrumented methods byte-for-byte on the
off path and select an instrumented implementation once at construction. For
the watchdog, instantiate the unchanged base watchdog when the override is
zero and a qualification-only subclass/variant with the augmented snapshot
when active. For Pipecat, use separate traced receive/audio/release methods or a
traced service variant so ordinary event/frame processing does not test a flag
repeatedly.

## 3. Trace-on is not observational and its receipt clock is misplaced

The supposed `client_received_monotonic_s` is sampled inside
`_PlayoutTraceWriter.record()` (`llm.py:94-112`). The caller invokes it only
after the async iterator yielded, JSON parsing completed, and—for audio
deltas—`events.decode_audio_delta()` completed (`llm.py:1020-1025,1058-1077`).
Accepted audio is decoded again in `_audio_delta()` (`llm.py:1358-1362`). Thus
the recorded clock is a post-parse/post-base64-decode clock, not the Pipecat
receipt boundary required by the plan. Payload-size- and CPU-dependent decode
time contaminates arrival intervals.

More seriously, decoding before the service's lifecycle filter changes
behavior. I supplied a stale malformed audio delta while another response was
current:

- tracing off: `_handle_server_event()` ignored it, as the existing lifecycle
  contract requires;
- tracing on: `_trace_server_playout_event()` raised `ProtocolError` before the
  ignore check. In the real receive loop this becomes a fatal receive failure.

This directly contradicts both observational tracing and the Step 0 analyzer's
documented rule that stale/cancelling deltas are ignored before wire decoding
(`tools/qualification/stratified_latency_analyzer.py:889-898,948-958`).

Writer failure also changes service behavior. A `/dev/full` probe caused the
writer thread to set `OSError`; the next `record()` and `close()` raised
`RuntimeError`, and the stream close emitted an uncaught thread traceback. A
qualification run should be invalidated when evidence fails, but tracing must
not turn an otherwise intentionally ignored protocol event or media operation
into a different service outcome.

Required fix:

- Capture one timestamp immediately after the WebSocket iterator yields (the
  existing liveness timestamp can share that value), before parsing, and carry
  it explicitly into records.
- Apply the normal response/cancellation lifecycle filter first. Validate and
  decode each accepted delta once, then use the resulting bytes/sample count
  for both playout and tracing.
- Treat trace failure as invalid qualification evidence exposed in
  health/artifact status, without changing media/protocol semantics. The live
  gate should fail closed on that status.
- Add stale-current, cancelling, malformed, and clock-placement tests. The fake
  writer-clock test currently only proves that `record()` calls its injected
  clock; it does not prove where receipt is measured.

## 4. Writer design is unbounded and cleanup can block the event loop

`queue.SimpleQueue` is unbounded (`llm.py:84`). Every audio record retains the
full base64 payload, and the writer flushes after every JSON line
(`llm.py:123-128`). If storage stalls or the writer merely falls behind, the
event loop continues enqueueing without backpressure or a memory ceiling. The
ordinal dictionary is also never pruned on `response.done`, so it grows for the
process lifetime.

File serialization/write is on the worker thread, but not all blocking behavior
is off the async loop: `cleanup()` calls synchronous `close()`, which performs an
unbounded `thread.join()` (`llm.py:114-121`). A stuck filesystem can therefore
freeze the Pipecat event loop indefinitely during teardown. Directory creation
and file opening are synchronous too, though those are construction-time rather
than frame-hot-path operations.

Required fix: use a bounded queue with an explicit overflow state that
invalidates the trace without blocking audio, prune per-response state, batch
writes/flushes, and perform close/join off the event loop with a finite timeout.
Capture both write and close failures without an uncaught thread exception.
Add deterministic stalled-writer, queue-overflow, disk-error, and bounded-close
tests.

## 5. Watchdog mutation itself is narrow, but tests do not prove behavior

Within the normal environment-driven path, the constructor replaces only
`no_text_required_frames`; `required_frames` (decoded silence) and
`no_audio_required_frames` are passed through unchanged
(`server.py:2882-2892`). The existing `observe()` logic is otherwise untouched.
The optional snapshot field is appended only for a positive override and is
omitted for unset/zero, preserving the old snapshot dictionary and key order.
Health similarly reports the engine's effective value only when active. I found
no strict retained-trace reader that rejects the additive active-only field.

This subpart is semantically correct apart from the production fence and the
off-path snapshot branch above. The new test, however, compares constructor
snapshots rather than exercising watchdog behavior. Add boundary sequences that
show:

- continuous audible/no-text output trips only at the overridden threshold;
- no-audio still trips at the original threshold;
- audible-then-silent decoded silence still trips at the original threshold;
- function/text resets are unchanged; and
- request-reason precedence at coincident thresholds is understood.

Also reject nonpositive or otherwise invalid direct constructor values, or make
the constructor accept only an already-validated qualification configuration.

## 6. Step 0 consumes arrivals, not the new playout records

The traced `response.created`, `voicechat.metrics`, wire audio delta, and
`response.done` rows can drive the analyzer's existing arrival-clock path when
metrics are present. The analyzer validates and decodes the original base64
wire payload, so the duplicated delta is sufficient for arrival reconciliation.
It does not validate `trace_schema`, and its wire path ignores the trace's
`ordinal` and declared `sample_count` in favor of event order and re-decoding.

Neither `voicechat.playout.release` nor
`voicechat.playout.downstream_push` is referenced anywhere in the Step 0
analyzer. I appended impossible records (release time `999`, 999 released and
cleared frames; push time `-999`, wrong response, invalid rate/channels) to an
otherwise valid synthetic trace. `queue_diagnostics` and `structural_gate` were
identical with and without them. The analyzer is replaying a theoretical
release from arrivals; it is not consuming or validating the observed release
or push chain. The completion note's “arrival/release replay” command and
direct-consumability claim therefore overstate the contract.

Required fix: extend the analyzer (or add a dedicated checked consumer) to
validate schema/version, response/ordinal/sample-count linkage, release reason
and counts, monotonic clock order, one-to-one downstream pushes, done flush, and
interruption clear. Reconcile observed releases/pushes with the theoretical
state machine and fail on orphaned, duplicate, missing, or contradictory
records. Add a negative synthetic test like the probe above, not only a happy
trace that happens to contain ignored records.

## 7. Browser capture is isolated from served code but can perturb results

The served client is untouched: all changes are in the Playwright test harness,
installed after page load and before connection. The capture is therefore
qualification-only as requested.

It is not low-impact measurement. At a 48 kHz AudioContext, a 128-sample render
quantum runs about 375 times per second. Every quantum scans samples for RMS,
posts a message to the main thread, calls a Playwright exposed binding, performs
Python JSON serialization, writes a line, and synchronously flushes it
(`test_browser_smallwebrtc_e2e.py:173-200,222-238,299-311`). The cloned-track
processor adds another consumer and binding stream. Binding calls are not
awaited, so slow transport/storage can also accumulate pending work. This load
can cause or hide scheduling delay and underruns in the same browser under
measurement.

The captured render path is a newly created parallel AudioContext graph feeding
a zero-gain destination (`lines 208-241`), not the served `<audio>` element's
audible render graph. Its resampling/scheduling may differ from the actual
element. In addition, `browser_performance_ms` for dequeue is sampled in the
main-thread `port.onmessage`, not in the AudioWorklet render quantum; the actual
quantum clock is `audio_context_time_s`/`audio_context_frame`. Main-thread delay
therefore contaminates the apparent dequeue `performance.now()` time. The
completion caveats mention a silent tap and non-DAC timing, but not these two
measurement distinctions or the tap's load.

The zero-PCM label is honestly called an underrun candidate, and the cloned
`MediaStreamTrackProcessor` limitation is documented. The generated JavaScript
also passes an independent `node --check`.

Required fix: batch compact quantum records in the page/worklet and export them
in bounded chunks or once at teardown; do not cross the Playwright binding and
flush storage for every quantum. Record a browser clock mapping (for example,
AudioContext output timestamp where supported), name main-thread receipt and
render clocks distinctly, and state explicitly that this is a parallel render
tap rather than the audible element. Quantify capture-on/off overhead before
using the trace as a normative underrun gate. Ensure capture shutdown and sink
close run in `finally` on all browser-test failure paths.

## 8. Test and lint results

Independent verification:

- full suite: **842 passed, 16 skipped, 29 subtests passed**, four warnings;
- Step 2/runtime/Pipecat/browser scope: **16 passed, 1 skipped**;
- task-scoped Ruff: **all checks passed**;
- tracked task-file `git diff --check`: clean; and
- generated browser capture JavaScript: `node --check` passed.

The tests are green but mirror implementation more than they attack the
contract. In particular, they positively encode the one-key production bypass,
do not compute the TOML intersection, do not test the real receipt boundary,
do not test stale/cancelling deltas or writer failure/stall, do not prove
release/push analyzer consumption, and only substring-check most browser schema
features. Green status is therefore not evidence that the blockers above are
closed.

## Closure gate for round 2

Do not run live qualification on this implementation. A reviewable revision
must provide all of the following offline first:

1. positive, fail-closed qualification identity and single-snapshot startup
   validation, with one-key near-production attacks rejected;
2. a genuinely unchanged off hot path in both server and Pipecat code;
3. boundary-correct receipt timestamps, lifecycle-preserving single decode,
   bounded writer memory, and bounded non-event-loop cleanup;
4. analyzer validation/reconciliation of actual release and downstream-push
   records, including adversarial negative traces;
5. dynamic watchdog interaction tests; and
6. batched browser capture with explicit clock/path semantics and measured
   capture overhead.

