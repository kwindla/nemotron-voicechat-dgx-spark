# Step 5 wedge increment 1 adversarial review — round 1

## Verdict

**FAIL.** The retained artifacts are internally hash-consistent and useful as a
deterministic bookkeeping dry run, but they do not satisfy the preregistered
Phase 0 protocol-race contract. The decisive defects are:

1. the million schedules execute synchronously without the real asyncio
   output-handler task, client/core transport, queue, or scheduler-to-worker
   wiring;
2. B0–B9 loss injection suppresses only the harness's audit record, so all ten
   injections are record-suppression tests rather than induced protocol losses;
3. the runner does not exercise the patched B1–B8 sites and only partially
   exercises the qualified scheduler/iterator paths it claims are exact; and
4. the patcher mutates the supplied root in place, can leave a partially
   patched tree after a later failure, and has no mechanism that enforces
   patch-on-copy.

Consequently the report's `PASS under the preregistered Phase 0 contract` and
`Phase 0 passed` statements are not supportable. The bounded final hypothesis
wording is correct, but it cannot rescue an invalid Phase 0 pass.

## Required deltas before a Phase 0 PASS may be restored

1. **Replace the synchronous driver with a real task-path race harness.** Run
   the exact qualified `AsyncLLM.generate`, `_run_output_handler`,
   `OutputProcessor`, `RequestOutputCollector`, core request dispatch, scheduler
   transition/descriptor construction, and B6–B8 queue/send/receive wiring.
   The executor/model output may remain fake. A CPU-local real ZMQ transport is
   preferable; at minimum, use the real task topology and queue interfaces with
   explicit, seeded yield points on both sides of every required transition.
   Do not manually `send(None)` into an already-ready generator.
2. **Make schedules describe actual interleavings.** Retain a versioned expanded
   schedule record or replayable decision vector, an equivalence-class ID, and
   per-batch histograms for append/wait order, request order, output order and
   batch partition, cancellation target/phase, error target/phase, and every
   seeded yield point. Preregister a minimum for distinct realized task
   interleavings, not merely one million seed indices.
3. **Induce real losses.** Keep record-suppression as a trace self-test, but add
   an independently observed operation-loss injection for every boundary. For
   example: drop the append message/core dispatch; withhold the scheduler
   install or runnable transition; make the fake execute/forward/sample stage
   fail or never publish; drop the real core queue item; suppress/close the real
   send/receive path; prevent `OutputProcessor` queue publication; and cancel or
   strand the actual waiting iterator. The detector must consume boundary
   records independently of the injection switch. Classify and retain both
   matrices.
4. **Correct boundary placement and truth conditions.** B2 must be after the
   complete install, waiting-to-running transition, and real finalized
   scheduler descriptor. B3 must record the graph mode and token count after
   all overrides. B6 must be recorded only after `put_nowait` succeeds, not
   before it. B8 must be recorded only when that request was actually published
   to its collector; an ignored already-aborted output is not `routed=True`.
5. **Make the patcher structurally patch-on-copy and atomic.** Accept distinct
   `--qualified-source-root` and nonexistent `--output-root` arguments; reject
   equality, aliasing, and unsafe nesting; verify all source hashes and all
   anchors before any write; copy to a staging directory; patch and compile the
   complete staging tree; then atomically publish the output. Never write the
   source root. Test source immutability on success and at every failure point.
6. **Enforce the preregistration in code.** Include and check preregistration,
   application revision, design, runner, all six frozen harness files, patch,
   configuration, and artifact hashes in the compatibility identity and pass
   predicate. Resume must validate a contiguous, nonoverlapping `0..999999`
   prefix plus all compatibility hashes. Use the preregistered domain-separated
   `fault/B0` … `fault/B9` seeds rather than schedule indices `1010000..1010009`.
7. **Rerun trace noninterference through the production call sites.** The CPU
   fixture must call the instrumented packed host function and patched
   scheduler/output path, not only `BoundaryAudit.mark`. Report trace-off hot
   path allocations/branches or measured overhead explicitly. Preserve the
   already-correct GPU/ZMQ residual wording.
8. **Amend the report before rerun.** Relabel this campaign as a failed/invalid
   Phase 0 attempt or synchronous bookkeeping dry run, retract `PASS`, disclose
   the 86-sequence effective coverage estimate and all shimmed boundaries, and
   replace the incomplete suite-baseline explanation with the exact +19 bridge
   documented below. A fresh compliant campaign is required; the current one
   must not be reinterpreted post hoc as a pass.

## 1. Schedule realism

The wall time is consistent with the code because there is no asynchronous
race. `run_schedule()` is an ordinary synchronous function. It serially marks
B0–B5 for each lane, serially marks B6/B7, directly calls
`OutputProcessor.process_outputs()`, and only then manually advances each
already-ready async generator with `operation.send(None)`
([runner lines 430–438 and 490–667](../../tools/qualification/step5_phase0_protocol_race.py)).
No `asyncio.create_task`, task scheduler, random yield, core input/output queue,
or client receive loop participates. The exact output collector's
`get_nowait()` fast path wins; the iterator does not wait and later awaken.

Enumerating all one million SplitMix64 words under the runner's actual control
flow gives only **86 distinct normalized synchronous operation sequences**
after collapsing payload-only values and bits with no executed distinction
(for example, batch versus individual processing of a single output). Those 86
are an upper bound on meaningful interleaving coverage because several differ
only in which lane undergoes the same serial operation. The largest equivalence
class contains 227,202 schedules. Including the fake B5 accepted-count values
and all categorical choices increases the apparent combinations, but not task
interleavings. The headline elapsed value is also **34.264 microseconds per
schedule**, not approximately 29 microseconds (34.264 seconds / 1,000,000).

The retained `batches.jsonl` proves ten contiguous 100,000-index ranges and
retains aggregate request/cancellation/error counts plus one routing digest per
batch. It retains no expanded schedules, realized ordering histograms,
equivalence-class counts, or representative replay indices. Thus the retained
artifacts alone cannot audit the claimed diversity; it can only be recomputed
from the current unsealed runner source. The report's statement that the run
randomized categorical choices is literally true, but calling the result a
protocol **race** or treating one million iterations as one million race
schedules is materially misleading and fails §6.3's intent
([design lines 301–307](../step5-pairing-wedge-closure-design.md)).

## 2. Exact-path fidelity

The loader does compile whole, hash-verified class definitions selected by AST,
and the retained provenance hashes match the qualified index. That establishes
source origin; it does not establish that the production path was wired or
executed. The actual boundary fidelity is:

| Boundary | What Phase 0 actually executes | Classification |
|---|---|---|
| B0 | `BoundaryAudit.mark` in the harness; it never calls production `_generate_packed_pad_pair()` | harness-only |
| B1 | constructs and Python-unpacks the exact `EngineCoreAppendRequest` class using the `_MsgspecStruct` shim | shimmed; no serialization, ZMQ, core receive, request-type decode, or `_handle_client_request` |
| B2 | exact `Request.set_custom_inputs`, `read_custom_inputs`, `Scheduler.set_custom_inputs`, and the beginning of `Scheduler.schedule`; `_TransitionSet.add()` deliberately raises at the first waiting transition | partial exact method bodies; no complete schedule, `SchedulerOutput`, or finalized dispatch descriptor |
| B3 | `BoundaryAudit.mark` with a fabricated descriptor key/epoch/graph mode | fake source; no worker execute entry |
| B4 | `BoundaryAudit.mark` | fake source; no forward call/return |
| B5 | `BoundaryAudit.mark` with a seed-derived accepted count | fake source; no sampler, rejection sampler, or Mamba commit/restore |
| B6 | `BoundaryAudit.mark` | fake source; no core output queue |
| B7 | `BoundaryAudit.mark` | fake source; no encoder or ZMQ send |
| B8 | direct calls into the exact `OutputProcessor`/collector methods | real class methods, but no `AsyncLLM._run_output_handler`, `get_output_async`, receive, output slicing task, or patched B8 site |
| B9 | exact `AsyncLLM.generate` body grafted onto `_IteratorEngine`; `add_request` returns a prefilled collector, `_run_output_handler` is a no-op, and the coroutine is manually advanced | partial generator fast path; no waiting iterator, task interleaving, or production host return |

The dependency replacement is broader than executor/model output: it replaces
`msgspec.Struct` construction/decoding, output value classes, LoRA state,
logging, the engine client, `AsyncLLM.add_request`, `AsyncLLM.abort`, and the
output-handler wiring. The runner never imports or executes the patched copied
tree at all. Its B3–B7 “delivery” is a sequence of audit marks followed by a
directly constructed `EngineCoreOutput`. Therefore report lines 49–57—especially
“Only the executor/model-output source was faked” and the claim that the
output-routing/iterator path was exact—overstate fidelity. The CPU run does
exercise useful exact leaf methods, but it bypasses precisely the B6–B9 loss
surface implicated by the retained wedge.

There are also instrumentation truth defects relevant to Phase 1:

- the patcher records B6 with `enqueued=True` **before** `put_nowait()`
  ([patcher lines 106–117](../../src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py));
- B8 marks every item after `process_outputs`, including an item that the exact
  processor can ignore because its request was already aborted
  ([patcher lines 145–161](../../src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py)); and
- B3 captures graph mode and token count before later qualified-code overrides,
  so they need not be the actual dispatch values
  ([patcher lines 173–191](../../src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py)).

## 3. Fault-injection strength

All ten injections—B0, B1, B2, B3, B4, B5, B6, B7, B8, and B9—are
**record-suppression**. `BoundaryAudit.mark()` returns immediately when the
boundary equals `injected_loss`; no request, transition, task, queue item,
socket message, collector notification, or iterator operation is altered
([runner lines 374–392](../../tools/qualification/step5_phase0_protocol_race.py)).
The detector then scans that same audit mask. The focused fault test repeats
the same tautology without running a schedule.

The report accurately discloses that it “skipped exactly one boundary record
while leaving all operations in place” (lines 59–64), but then labels the
matrix “isolated B0–B9 losses.” The 10/10 result proves only that a missing bit
is reported as a missing bit. It does not prove that the harness catches a real
protocol loss, which is the purpose of §6.3's injection requirement.

## 4. Noninterference

The trace implementation is off by default, uses a fixed preallocated mmap,
saturates rather than overwriting the causal prefix, and performs no explicit
lock/pipe/socket operation in `record_boundary`. The retained CPU fixture is
internally consistent: both arms cover 64 schedules/95 lanes, the output digest
is identical, and the enabled arm retains 950 records with zero drops. The
report also explicitly states the GPU worker, CUDA graph, Mamba, device-tensor,
and real-ZMQ residual (lines 125–128), satisfying the bounded residual wording.

The disabled production cost is **bounded but not zero-allocation or
zero-branch**. `_generate_packed_pad_pair()` unconditionally performs the local
import and makes two function calls, constructing keyword-detail arguments and
computing fields before `record_boundary` takes its `_RING is None` branch
([runtime lines 729–738 and 758–772](../../src/nemotron_voicechat_runtime/runtime_optimizations.py)).
The patched B1–B8 hot sites likewise call the function and construct arguments
when the environment gate is unset. With a successful module import the
functional return construction is preserved and record errors are swallowed,
but production timing is changed and this fixture does not execute those
production sites. It therefore does not prove noninterference for the modified
host/scheduler/output path; it proves only CPU equality for the synthetic audit
fixture.

## 5. Preregistration integrity

The current preregistration recomputes exactly to
`7508866a213915634af6f8675bc5565c1f94e6a77b5b3b7206fdeaa7aa5f4528`.
The design hash, qualified-tree hash, diagnostic-patch-manifest hash, and raw
manifest hash also recompute to their reported values, and all eight raw
manifest entries pass size/SHA-256 verification. Filesystem chronology places
the current preregistration before the batch artifacts, but the retained
evidence contains no immutable timestamp/signature that independently proves
the pre-run seal. The current completion note was necessarily rewritten after
the campaign because it contains the results; the assertion that the hash was
written there before the run is therefore not independently auditable from its
present contents.

The run does match the declared root seed, index range, aggregate schedule
count, nominal acceptance thresholds, and ten-boundary matrix. However, the
implementation violates other frozen clauses:

- the sentinel schedules are indices `1,010,000..1,010,009`, not values derived
  using domains `fault/B0` … `fault/B9` as preregistered;
- resume checks only `root_seed`, then trusts stored counts and the last end
  index; it does not reject gaps, overlaps, reordering, source drift,
  configuration drift, harness drift, or incompatible batch size;
- the `passed` predicate does not check the preregistration/application/design,
  runner/harness, diagnostic patch, or artifact hashes, despite criterion 1;
  the raw manifest does not cover the runner, runtime host change, or both test
  modules; and
- the exact-path, schedule-realism, and real-loss-injection requirements were
  weakened in execution while the report claims “no campaign deviations.”

No Phase 1 or Phase 2 result was smuggled in. The B3–B5 instrumentation is
authorized preparation for later GPU work, and both preregistration and report
correctly say those phases were not run.

## 6. Patcher safety

The four retained `before` hashes match the pinned qualified index exactly:

- scheduler `988ecb28f4383e6a9b8ab9595ab6ec5b9d08e267409c014304bb14fabe528ff3`;
- async LLM `9834b3325cee6fe263545d642d4497310fda40179c3c503bc5ec2a7600fac3ca`;
- core `590925b3823b0bcb4153329441e8e83b1a3ca06b4746082956bdb8afd4004f29`;
- GPU runner `2ce42b6fcafa1ff3320b78efe1d9c36462e646982c47ab088eed60d47605925f`.

Anchor cardinality and per-target baseline hashes fail closed for the target
currently being processed, and each patched file is compiled. But
`apply_patch(root, manifest_path)` writes each `root / relative` directly,
copies the trace directly into `root`, and the CLI accepts any resolved root
([patcher lines 255–297](../../src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py)).
It neither creates a copy nor proves that the caller supplied one. If target 2,
3, or 4 fails its hash/anchor/compile check, earlier targets have already been
mutated. It can also overwrite an existing `v1/wedge_boundary_trace.py` before
the manifest is written. Thus the “can never mutate the qualified tree in
place” claim is false, and “fail-closed patch-on-copy” describes an operator
convention, not a safety property. The only patcher-focused test checks
`replace_once()` anchor drift; it does not test `apply_patch()` safety.

## 7. Suite delta

The numerical bridge from the brief's 1,177 baseline to 1,196 is accountable,
but the report's “superseded” explanation is incomplete:

| Contribution | Passed-test delta | Evidence |
|---|---:|---|
| Step 5 graph-repro review round 1 to round 2 | +13 | focused count grew from 14 to 27 and full count from 1,177 to 1,190 in the retained round-1/round-2 reviews |
| Wedge increment 1 | +6 | three tests in each of the two new wedge test modules |
| **Total** | **+19** | **1,177 → 1,196** |

Commit `c6128e7` adds `tests/runtime/test_graph_reproducibility.py` as a new
file; it does not modify any pre-existing test file. This wedge increment adds
only `test_step5_wedge_boundary_trace.py` and
`test_step5_phase0_protocol_race.py`, with six test functions total. Therefore
the +19 consists of new Step 5 test cases, not changed pre-existing tests or
subtests. The report should state this complete bridge rather than beginning at
the later 1,190 base. In this review, the six focused wedge tests pass in 0.62
seconds; the fault test remains too weak for the reason above.

## 8. Bounded conclusion wording

The required sentence appears verbatim in substance in the preregistration
(lines 59–61) and verbatim in the report (lines 156–158): a pass only weakens
the transport/scheduler/output hypothesis and cannot clear the GPU/graph
hypothesis or reliability precondition. The report also correctly says Phase 1
and Phase 2 remain unrun. There is no Phase 1/2 or GPU-clearance overclaim.

Other report sentences do overclaim Phase 0 itself and must be corrected:

- `PASS under the preregistered Phase 0 contract` (line 4);
- “All ten isolated B0–B9 losses were detected” (lines 7–10), when only audit
  records were suppressed;
- “Only the executor/model-output source was faked” and the exact-path claims
  at lines 37–57;
- “fail-closed qualified-tree patcher” at lines 74–81; and
- `Phase 0 passed` at line 156.

The preregistration's wording is appropriately bounded; the problem is that
the executed harness did not conform to it.

## Creative, cheap additions that improve Phase 1 readiness

1. Retain one expanded schedule JSON per normalized equivalence class, plus the
   first/last index and class count. Also retain all sentinel expansions and a
   small deterministic “coverage canary” set that a focused test can replay.
2. Add a boundary schema version, campaign/run UUID, engine epoch, frontend and
   backend request IDs, logical transaction/pair ordinal, process role, thread
   or asyncio-task identity, worker/rank/virtual-engine identity, output-message
   and item ordinals, exception/cancellation kind, and actual graph construction
   key. Phase 1 needs these to join per-process records without guessing.
3. At B2 retain request status, waiting/running membership before and after,
   custom-input ready/consumed counts, proposal token IDs, and a canonical
   scheduler-descriptor digest. At B3 retain the final post-override graph mode,
   token count, batch size, and canonical descriptor digest. At B5 retain
   proposed/accepted counts and commit-versus-restore outcome per request.
4. Replace `sha256(repr(value))` with a versioned canonical serialization; a
   `repr` can change across implementations or contain unstable values. When a
   512-byte record truncates, store the actual SHA-256 of the omitted details,
   not only `details_sha256_required=True`.
5. Put a small header/status area in each sidecar with schema, process role,
   epoch, capacity, committed-record count, and drop count. Use a run/epoch
   nonce in the filename so PID reuse or re-import cannot truncate an earlier
   trace. Preserve enough metadata to interpret a sidecar after a hung process
   is killed.
6. Add cross-boundary invariants to the offline analyzer: one monotone sequence
   per request/epoch, descriptor-key continuity B2→B4, exactly one successful
   B6/B7/B8 publication for the relevant completion, and no B9 without a
   collector publication. This will make Phase 1 localization independent of a
   single in-process audit mask.

## Verification performed

- Recomputed preregistration, design, diagnostic-patch-manifest, raw-manifest,
  and every raw-artifact SHA-256/size: all matched.
- Compared the four patcher baseline hashes with the pinned qualified escrow
  index: all matched.
- Inspected the exact pinned qualified method bodies and the runner/patcher
  call sites; no live service, container, GPU, or inference probe was run.
- Recomputed the million-index schedule equivalence coverage from the frozen
  SplitMix64/control-flow definition without launching a campaign.
- Ran only the two focused CPU test modules: `6 passed in 0.62s`.
- `git diff --check` was clean before this review file was added.
