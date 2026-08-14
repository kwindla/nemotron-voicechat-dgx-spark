# Step 5 wedge increment 1 confirmation review — round 2

## Verdict

**FAIL.** Increment 1 is not concluded and Phase 1 may not yet be designed.
The v2 campaign is materially better than v1: it uses real asyncio tasks and
the exact qualified generator/output-handler/output-processor methods, its
100-batch chain and frozen-file identity recompute, and its class IDs do not
contain the schedule index or payload. The focused 23-test suite also passes.
However, six of the eight round-1 required deltas are not genuinely closed:

1. scheduler admission is still replaced by a harness shortcut, and eight of
   the 24 claimed yield points are never executed;
2. 502,930 is not the number of distinct realized event vectors and is inflated
   by inactive seeded phase bits;
3. the B9 “operation loss” closes an already-awakened iterator and then returns,
   so detection still depends on omitting the B9 audit mark rather than losing
   the required host return;
4. the patcher verifies only four target files before copying a tree whose full
   qualified-tree hash it merely asserts;
5. compatibility omits run-shaping CLI configuration and the PASS artifact gate
   omits three retained files, including the trace ring; and
6. the enabled-arm cost probe runs after trace metadata is captured, filling
   the retained ring and making the reported 952-record/zero-drop final trace
   state false.

No new million-schedule campaign was run for this review.

## Required-delta disposition

| Round-1 required delta | Round-2 status | Confirmation result |
|---|---|---|
| 1. Real task-path race harness | **OPEN** | Manual generator driving is gone, and exact `AsyncLLM.generate`, `_run_output_handler`, `OutputProcessor`, and collector code executes. But core ADD dispatch lands in the harness `_CoreDispatch.add_request`, which constructs a `Request`, forces it to `RUNNING`, installs it directly in `scheduler.running`, and uses an always-empty waiting queue. This bypasses exact `Scheduler.add_request` and its WAITING-to-RUNNING admission path. Only the `wait_then_append` half later exercises the waiting-input transition. Also only yield indices 0–5, 8–12, 16–18, and 20–21 are reachable; 6, 7, 13–15, 19, 22, and 23 are never passed to `yield_at`. Thus the claim that 24 decisions cover both sides of all required transitions is false. |
| 2. Actual realized interleavings | **OPEN** | The retained 502,930 IDs all recompute, and the digest basis excludes index and payload. But it includes cancellation/error phase for every schedule even when the corresponding target is `-1`; those inactive seed bits cannot be realized topology. Both phase values also execute at the same point after `iterator.__anext__()`. Across all representatives there are 169,206 distinct realized-event vectors, 353,984 classes after normalizing inactive phases, and 502,930 only under the runner's unnormalized categorical digest. The threshold of 64 is amply exceeded by the retained event vectors, but the reported and PASS-gated 502,930 “distinct realized interleavings” is not the claimed quantity. Histograms for the eight nonexistent yield sites are likewise seed-bit histograms, not realized-yield histograms. |
| 3. Independent real operation losses | **OPEN** | B0–B8 change real harness operations, the record-suppression matrix is separate, and `BoundaryAudit` has no operation-loss switch. B9 does not meet the preregistered loss: the consumer first awaits and receives `iterator.__anext__()`, records the awakened yield, then calls `aclose()` and immediately returns `{"kind": "stranded"}`. The host-side consumer therefore returns; only `_audit_record("B9", ...)` is skipped. The advertised operation matrix is at most 9/10 genuine operation losses. |
| 4. B2/B3/B6/B8 truth conditions | **CLOSED for the patched call sites** | The patch inserts B2 after `SchedulerOutput`, connector/event finalization, and `_update_after_schedule`; B3 follows the graph-mode and u-batch token-count overrides; B6 follows successful `put_nowait`; B8 follows successful collector `put`, after the already-aborted guard. The scheduler-admission shortcut remains a delta-1 harness defect, not a placement regression in these four patches. |
| 5. Atomic patch-on-copy and source immutability | **OPEN** | Distinct/nonexistent roots, staging, atomic `os.replace`, cleanup, compilation, and the nine injected-failure immutability tests are present and pass. But “verify every hash” loops only over the four `TARGETS`; the other 962 indexed files are copied without hash verification, while the manifest writes the full-tree digest as a constant. A valid-syntax drift in any non-target file can therefore be published under the asserted qualified-tree identity. |
| 6. Identity enforcement and resume audit | **OPEN** | Current frozen hashes and the preregistered identity recompute; all 100 batch hashes form `0..999999`; fault seeds are correctly domain-separated. However, identity configuration contains only schedule count/version, root seed, and minimum diversity. It omits the user-selectable `--batch-size` and `--concurrency`, so resume cannot reject those compatibility changes. `APPLICATION_REVISION` is copied as a constant rather than checked. The eight-entry artifact manifest excludes `phase0-summary-v2.json`, its own manifest, and the nested trace ring, yet `artifact_verified` is the PASS gate. This violates the v2 preregistration's “every retained v2 artifact” requirement. A resumed run also initializes representatives empty instead of restoring prefix representatives before rewriting `equivalence-classes.json`. |
| 7. Production-call-site noninterference | **OPEN as retained evidence** | The functional fixture does execute production `_generate_packed_pad_pair` plus the patched scheduler/output sites, and off/on functional digests match. The 78.940 ns number is genuinely taken from the trace-off subprocess and the method reports the kwargs construction and `_RING is None` branch. But fixture main captures `trace_metadata()` and then executes the same 20,000-call cost loop before closing. In the enabled subprocess those are enabled B0 records. The retained ring with the reported SHA contains all 4,096 slots, not 952: 952 functional records plus 3,144 cost records; the remaining 16,856 calls necessarily dropped after the saved zero-drop snapshot. The report's final retained-ring statement is therefore false, and the ring is not covered by the artifact PASS gate. The GPU/CUDA/Mamba/device/real-ZMQ residual wording remains correct. |
| 8. v1 retraction and +19 bridge | **CLOSED** | v1 is explicitly INVALID/FAIL, its former PASS is retracted, the 86-sequence estimate and boundary shims are disclosed, and the historical bridge is exactly 1,177 + 13 + 6 = 1,196. v2 is presented separately, with preregistration/seal files predating its retained batch output. |

## Detailed confirmation evidence

### 1. Task topology and scheduler transition

The exact async output path is real in the useful sense missing from v1:
`generate()` starts the exact qualified output handler; persistent core,
executor, and sender actors use `asyncio.Queue`; consumers are real tasks; and
there is no `send(None)` in the runner. This portion of delta 1 is closed.

The admission topology is not exact. `_CoreDispatch.add_request()` changes the
exact newly constructed request from its qualified WAITING default to RUNNING,
and `_new_scheduler()` places it directly in `running` with `_EmptyRequestQueue`
as `waiting` ([runner lines 571–639](../../tools/qualification/step5_phase0_protocol_race.py)).
The exact qualified `Scheduler.add_request()`—which puts a request into its
waiting queue—is not executed. The later preliminary `scheduler.schedule()` is
conditional on `wait_then_append` ([runner lines 925–943](../../tools/qualification/step5_phase0_protocol_race.py));
`append_then_wait` schedules a request that the harness already declared
runnable.

The 24-point statement is also mechanically false. Calls use 0, 1, 2, 3,
`4+lane`, `8+lane`, `10+lane`, 12, 16, 17, 18, and `20+lane`
([runner lines 662–678 and 733–840](../../tools/qualification/step5_phase0_protocol_race.py)).
With at most two lanes, eight declared decisions are unreachable. Nevertheless
all 24 seed bits receive batch histograms
([runner lines 1024–1033](../../tools/qualification/step5_phase0_protocol_race.py)).

### 2. What the 502,930 count measures

The positive result is that this is not a seed-index count. I streamed all
502,930 representatives, reconstructed the digest object at runner lines
964–978, and obtained zero ID mismatches. `index`, `replay_vector`, and routing
payload are absent from that object. The 100 batch records also sum to exactly
1,000,000 class memberships.

It is still not a count of distinct realized vectors. The digest adds
`(target, phase)` pairs from the seed whether or not a target exists
([runner lines 471–493 and 964–978](../../tools/qualification/step5_phase0_protocol_race.py)).
For `target == -1`, changing phase has no possible execution effect. Even when
a cancellation/error target exists, both named phases are implemented at the
same post-output point ([runner lines 834–879](../../tools/qualification/step5_phase0_protocol_race.py)).
A complete streaming audit of the retained representatives produced:

- 502,930 stored digest classes;
- 353,984 SHA-256-distinct classes after normalizing inactive target phases;
- 169,206 SHA-256-distinct `realized_events` vectors.

This does not threaten the preregistered minimum numerically. It does mean the
headline count and current PASS predicate are not the preregistered realized-
interleaving count.

### 3. Operation loss and detector independence

The split between `ScheduleContext.operation_loss` and
`BoundaryAudit.record_suppression` is real, and the detector itself only scans
its mask ([runner lines 498–532](../../tools/qualification/step5_phase0_protocol_race.py)).
B0–B8 suppress the corresponding enqueue, dispatch, install, stage, core queue,
wire queue, or collector publication.

B9 is the exception. Lines 839–843 await the actual iterator output before the
switch and then return normally after closing it. That is not a close/strand of
the waiting iterator before host return as sealed in preregistration v2 lines
72–79. The matrix says “detected” only because the B9 mark later in `_consume`
is bypassed.

### 4. Patcher and identity gates

The focused wedge suite passed **23/23 in 2.39 s**, including atomic publication
and all nine injected failure points. The safety hole is outside those tests:
`apply_patch()` obtains the qualified index but hashes only `TARGETS` before
copying the entire source tree ([patcher lines 330–385](../../src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py)).
Compilation detects syntax errors, not a hash-valid qualified tree. The
campaign separately calls `verify_qualified_tree`, but delta 5 required the
patcher itself to be structurally fail-closed.

The current compatibility identity recomputed exactly to
`010ceec82d973f6521d5d73a3955d009f26143566acdb23f26a7c270967b8fbb`,
and `_validate_resume` recovered 1,000,000 schedules, 1,500,113 lanes, and
502,930 stored classes from a valid contiguous batch chain. Those are useful
checks. They do not cover the CLI values at runner lines 1441–1444 because the
identity configuration at lines 995–1014 omits batch size and concurrency.

The retained output directory has 11 files. The manifest verifies eight
top-level inputs, but runner lines 1398–1408 deliberately exclude the summary
and manifest and never descend into the trace directory. In particular, the
PASS predicate remains true if the trace ring is absent or changed.

### 5. Noninterference and the 78.940 ns method

The functional off/on digest equality is credible. A fresh 16-schedule CPU
spot fixture also completed with zero missing boundaries and zero routing
mismatches. Repeated disabled direct-call measurements in the same environment
were approximately 63–64 ns/call, consistent in scale with the retained
78.940 ns/call; the retained value is selected from the genuinely disabled arm.

The retained ring contradicts its metadata because of call ordering in fixture
main: `trace_metadata()` is evaluated before `trace_off_cost()`, and only then
is the ring closed ([runner lines 1276–1290 and 1449–1453](../../tools/qualification/step5_phase0_protocol_race.py)).
In the enabled arm the named “trace-off” loop is enabled. Direct parsing of the
hash-recorded 2,097,152-byte ring found 4,096 nonempty records: B1–B8 each 95,
B9 96, and B0 3,240. The excess 3,144 B0 entries are the capacity-filling prefix
of the 20,000 cost calls. Thus the saved metadata describes an intermediate
state, not the retained artifact.

## Required deltas before PASS can be restored

1. Run scheduler admission through the exact qualified waiting queue and
   transition; add actual before/after yield hooks for every declared point or
   remove nonexistent points from the sealed schedule model and evidence.
2. Normalize away inactive cancellation/error choices and count the sealed
   realized-interleaving definition. The retained representative file appears
   sufficient for a corrected audit, but the report and PASS predicate must use
   that corrected quantity.
3. Make B9 actually cancel or strand the iterator while it is waiting, with the
   host call not returning normally, while keeping the detector independent.
4. Have the patcher verify the full qualified source-tree identity before its
   first write, and test non-target drift through `apply_patch()`.
5. Bind all run-shaping configuration, validate the application revision rather
   than copying a constant, make resume preserve prefix representatives, and
   seal/verify every retained artifact including the final summary and ring.
6. Measure disabled cost only in the disabled subprocess, or close/snapshot the
   enabled ring after all enabled calls and report its actual records/drops.

Until these are closed, the v2 `PASS` and “Phase 0 passed” statements must not
authorize Phase 1 design.
