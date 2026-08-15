# Step 5 wedge increment 1 confirmation review — round 3

## Verdict

**PASS. Increment 1 is concluded and Phase 1 design may begin.**

The v3 campaign genuinely closes the six deltas left OPEN in the round-2
review. I streamed the complete 2,438,244,471-byte batch ledger, independently
recomputed its batch and realized-vector class hashes, rehashed the complete
retained artifact set and all 966 qualified source files, and parsed the ring
record by record. No new campaign was run; all review work was CPU-only.

The recorded post-run test failure is the expected fail-closed application
revision guard, not a suppressed campaign or runtime defect. It rejects current
HEAD `0955bf095a607a2b7e7ffe2dfbc409e1b86cbf57` because the campaign was sealed
to `7e2e47fddb0da8288651d3158fc5a93b46ca202e`. The later commit postdates the
final artifact gate, changes no frozen v3 file, and all six frozen hashes still
match the preregistered identity. The correct disposition is to preserve and
record that failure; rerunning at the new revision would require a new identity
and seal, not weakening this one.

## Round-2 delta disposition

| Round-2 OPEN delta | Round-3 result | Confirmation |
|---|---|---|
| 1. Exact task path and scheduler admission | **CLOSED** | The harness AST-loads the qualified `Request`, `Scheduler`, and both `RequestQueue`/`FCFSRequestQueue` definitions. `_new_scheduler` starts with exact FCFS `waiting`, empty `running`, and calls exact `Scheduler.add_request`; the qualified method appends the still-WAITING request to the FCFS queue. After exact `set_custom_inputs`, exact `Scheduler.schedule` peeks and `popleft`s that queue, appends the request to `running`, and changes it from WAITING to RUNNING before the patched B2 site. B2 remains after `SchedulerOutput` construction, connector/event finalization, and `_update_after_schedule`. The retained histograms contain actual invocations of every yield ID 0–23. |
| 2. Actual realized interleavings | **CLOSED** | `equivalence_class` is computed only as `SHA-256(realized_events)[:20]`; schedule index, replay vector, routing payload, and categorical seed fields are absent. The complete retained stream produced 829,062 class IDs, 829,062 distinct retained full realized-vector hashes, 829,062 representatives, and 1,000,000 memberships, with zero representative-ID or local-count mismatches. All inactive cancellation/error choices are stored as `inactive`; there were zero inactive-phase violations. The campaign PASS predicate gates `len(class_counts)`, the same realized-vector-derived quantity reported as 829,062. |
| 3. Genuine independent B9 loss | **CLOSED** | B9 now sends a real non-final `EngineCoreOutput`, allowing B0–B8 and the first exact iterator awakening to occur, then publishes no final output. The host awaits a second `iterator.__anext__()` and cannot return before cleanup cancellation. Detection requires both `first_missing == B9` and one pending consumer per lane. Retained fault schedule 347336 has `first_missing=["B9"]` and pending task `phase0-generate-347336-0`. `BoundaryAudit` has only record suppression, not an operation-loss control, and the separately run record-suppression matrix also detects 10/10. |
| 5. Full-tree pre-write verification | **CLOSED** | `_qualified_hashes` requires exactly 966 indexed paths. `apply_patch` verifies every path is a regular nonsymlink file, verifies every file hash, recomputes tree digest `d66baad…`, and checks all patch anchors before its first write (`output.parent.mkdir`). It reverifies the staged copy before patching and atomically publishes with `os.replace`. My post-run read-only rehash found all 966 files correct and independently reproduced `d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`. The focused tests cover non-target drift through `apply_patch` and failure cleanup. |
| 6. Identity, artifacts, and representative resume | **CLOSED** | The preregistered identity includes batch size 10,000 and concurrency 128 as well as the live application revision; its canonical digest independently recomputes to `da11f267a6aad3de556d5c75899a16ae6206cb816bb3a69cc54dac3514bd3507`. All current frozen-file hashes match it. `_validate_resume` validates the compatible contiguous batch chain and reconstructs count/first/last/vector/schedule representative state. Independently restoring the first two batches yielded exactly 19,911 representatives for 20,000 memberships, equal coverage, and no count or first-representative errors. The recursive artifact manifest covers all ten non-manifest files—including summary and nested ring—and its own canonical semantic hash. |
| 7. Production-call-site noninterference | **CLOSED** | Enabled and disabled 64-schedule fixture digests are identical, with zero missing boundaries and routing mismatches, and both execute the production packed-host function. The 20,000-call cost loop is a third subprocess forced trace-off, so it cannot modify the enabled ring. Direct parsing of the retained 2,097,152-byte ring found exactly 952 JSON-valid records with contiguous sequence 0–951: B0 and B9 have 96 each, B1–B8 have 95 each, and every boundary is present. This matches final metadata with zero drops and SHA-256 `4ef032e5…`. The ring is hash-covered by the artifact PASS gate. The stated residual—GPU worker/CUDA graph/Mamba device state/device tensors/real-ZMQ timing—is truthful for this CPU/stubbed fixture. |

Round-2 deltas 4 and 8 remain closed. The patched B2/B3/B6/B8 truth
conditions have not regressed, and v1/v2 remain explicitly INVALID/FAIL with
their former PASS claims retracted.

## Independent retained-data checks

### Ledger and realized vectors

The complete 100-line ledger formed an exact contiguous `0..999999` range. I
recomputed every canonical batch hash and every local representative ID from
its `realized_events` vector. Results:

- 100 batches and 1,000,000 schedules;
- 1,500,113 request lanes;
- zero missing ordinary completions and zero routing mismatches;
- 829,062 class IDs, retained representatives, and distinct retained full
  vector hashes;
- 1,000,000 summed class memberships;
- zero bad batch hashes, compatibility identities, ranges, class coverage,
  local counts, or representative IDs;
- zero inactive cancellation/error phase violations.

Every yield ID 0–23 had both seed decisions represented and a nonzero actual
invocation count. In particular, the former phantom IDs 6, 7, 13, 14, 15, 19,
22, and 23 had 1,000,000; 500,113; 1,250,713; 1,250,713; 1,250,713;
1,250,713; 1,000,000; and 500,113 actual invocations respectively.

Inactive phases cannot inflate this count: `decision_for` normalizes the phase
to `inactive` when its target is `-1`, and only `ctx.events` supplies the class
digest. For active targets, the `before_B9` and `after_B8` branches place the
yield and cancellation/error events in different orders, so the active phase
choice is realized rather than merely labeled.

### Artifact and ring gate

The v3 data directory contains exactly the manifest plus the ten files listed
by it. Independent byte-size and SHA-256 verification found no mismatch. The
manifest file hashes to
`d9dd7ae41d8ff1265b10a1e949e2e6a5ec77f463f8a22ed104aa2ee4863fee90`,
and its canonical self-hash independently recomputes to its claimed
`3a3b77ae4a974033071e2bfdb58066c743cb7abeb61fdb96d08bfe82fd9d780a`.
The manifest includes the final summary and
`noninterference-trace/wedge-trace-2211720-phase0-v3-on.ring`; exact recursive
file-set comparison is part of `verify_artifact_manifest`.

The ring statement is therefore both semantically true (valid contiguous
records, exact reported count, all boundaries, zero recorded drops) and tied
to the PASS evidence by the recursive file hash. This corrects both v2 defects:
the enabled arm is no longer contaminated by the cost loop, and the retained
ring can no longer be changed or removed while the artifact gate stays true.

## Post-run guard failure

The sole post-run failure is
`test_compatibility_identity_enforces_revision_and_run_shape`. Its first call
to `compatibility_identity` calls `verify_application_revision`, which executes
`git rev-parse HEAD` and raises:

```text
ValueError: application revision mismatch: expected
7e2e47fddb0da8288651d3158fc5a93b46ca202e, got
0955bf095a607a2b7e7ffe2dfbc409e1b86cbf57
```

This is the sealed guard doing exactly what delta 6 required. The final v3
artifact manifest was written at 05:14:52 UTC; the later commit is timestamped
06:54:02 UTC and changes only the separate function-head W8 qualification
documents, data, and benchmark tools. It changes none of the six frozen v3
files or Step 5 retained outputs. Their hashes still exactly match the sealed
identity.

The failure is therefore not evidence that the completed campaign ran under
the wrong identity, and it does not conceal a failed runtime check. It is also
not appropriate to mark it xfail, update `APPLICATION_REVISION`, or bypass the
check merely to make a later checkout green; any of those would weaken the
sealed compatibility contract. The practical limitation is simply that this
campaign-specific identity test is intentionally not an evergreen test across
repository revisions. Validation at the sealed revision passed 28/28; a new
campaign at a later revision must be separately preregistered and sealed.

## Bounded conclusion

The six round-2 OPEN deltas are closed by v3. **Step 5 wedge increment 1 is
concluded, and Phase 1 design may begin.** This result remains bounded to the
transport/scheduler/output hypothesis and does not clear GPU/graph behavior or
the overall reliability precondition.
