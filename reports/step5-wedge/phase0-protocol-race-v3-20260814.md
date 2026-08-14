# Step 5 pairing wedge: Phase 0 v3 closure amendment

**Date:** 2026-08-14  
**Result:** **PASS — increment 1 concluded; Phase 1 may be designed**  
**Scope:** offline CPU only; no container, GPU, inference service, or live endpoint

The sealed v3 campaign completed exactly 1,000,000 schedules and 1,500,113
request lanes with zero missing ordinary B0--B9 completions and zero routing
mismatches. It produced **829,062 distinct realized ordered event vectors**, far
above the preregistered minimum of 64. Honest campaign wall time was **646.052
seconds**. Both independent fault matrices detected 10/10 boundaries,
production-call-site noninterference passed, and the complete recursive artifact
gate passed.

This is a fresh v3 result. The retained v1 and v2 attempts remain INVALID/FAIL;
their former PASS claims are retracted and their raw artifacts are not
reinterpreted.

## Pre-run seal and compatibility identity

The v3 preregistration was sealed before any v3 batch, fault, noninterference,
summary, or ring result at SHA-256
`cd8a184b6042c628d0f6ddb39d2d6007e3f94f87740a09b7f0414d2152837038`.
The pre-run seal file itself is SHA-256
`61036ace0ef8d8d71d01795d8f6715f8f5f2268af393cccbd1f37849e67fdc48`.
It records that only the two preregistered identity JSON files existed in the
v3 data directory before execution.

The enforced compatibility identity is
`da11f267a6aad3de556d5c75899a16ae6206cb816bb3a69cc54dac3514bd3507`.
It binds application revision
`7e2e47fddb0da8288651d3158fc5a93b46ca202e`, batch size 10,000,
concurrency 128, schedule range/version/root seed, all yield IDs, the
preregistration and design, the 966-file qualified tree, the deterministic
patch manifest, and all six frozen harness files. The campaign compared the
live identity with the preregistered JSON before starting and failed closed on
any difference.

The repository later moved to unrelated commit `0955bf0...` at 06:54:02 UTC,
after the v3 artifact gate finalized at 05:14:52 UTC. No v3 frozen harness or
runtime file changed in that commit. A post-run identity call now rejects the
later HEAD, which is the required revision enforcement; it does not alter the
already sealed campaign's identity or artifacts.

## Round-2 required-delta closure

| Delta | v3 closure evidence |
|---|---|
| 1. Exact task topology and reachable yields | Core admission calls the exact qualified `Scheduler.add_request` into an exact `FCFSRequestQueue`; the exact request remains WAITING until custom-input installation and exact `Scheduler.schedule` moves it to RUNNING. Exact qualified async generator, output handler, output processor, collector, core dispatch/step, request, scheduler, queue, and scheduler-output definitions execute through persistent asyncio tasks and queues. There is no manual generator driving. Every ID 0--23 is called by a real topology transition and was observed in retained realized-yield histograms. |
| 2. Realized interleavings | The class ID is derived solely from the ordered `realized_events` vector. Inactive cancellation/error phases are normalized to `inactive`; active phases have different event ordering. A complete post-run stream audit recomputed every local representative ID from its retained event vector, matched every class count, and obtained exactly 829,062 distinct vectors and 1,000,000 memberships. The PASS gate uses 829,062. |
| 3. Genuine B9 loss | B9 publishes a real non-final result so B0--B8 complete and the exact iterator awakens once, then withholds the final publication. The host awaits the second exact generator item and remains pending until cleanup. Detection requires `first_missing == B9` and one pending consumer per lane. The detector only reads its boundary mask; the separate record-suppression campaign is 10/10. |
| 5. Patcher atomicity and source immutability | Before its first write, the patcher hashes all 966 regular files against the qualified index and recomputes tree SHA-256 `d66baad...`; it verifies anchors, reverifies staging, compiles the copy, and atomically publishes with `os.replace`. Success, nine injected failure points, alias/nesting rejection, anchor drift, and non-target hash drift leave source unchanged and failed output absent. A post-run independent 966-file verification recovered the same tree digest. |
| 6. Identity, artifacts, and resume | Application revision, batch size, and concurrency are enforced. Every batch carries the sealed identity and a canonical batch hash. Each batch retains local representative first/last/count, expanded schedule, and realized vector; resume reconstructs and validates global representatives. A post-run two-batch audit restored all 19,911 representatives for the 20,000-schedule prefix, and the complete 100-batch stream matched 829,062 representative/class totals. The recursive artifact manifest covers all ten non-manifest files, including final summary and nested ring, and self-covers its semantic content. |
| 7. Production noninterference | Trace-off and trace-on ran the exact same 64-schedule functional fixture and production packed-host call site, producing identical output SHA-256 `00280c0a...`, zero missing boundaries, and zero routing mismatches. The disabled-cost loop ran in a separate forced-off subprocess. The measured cost was 66.80905 ns/disabled call using the same direct-call method as the superseded v2 78.940 ns observation. The retained ring contains exactly the reported 952 contiguous records, zero drops, all B0--B9 values, and SHA-256 `4ef032e5...`; it is included in the artifact gate. |

Round-2 deltas 4 and 8 remain closed: B2/B3/B6/B8 retain their already-confirmed
truth conditions, and the v1 retraction and historical suite bridge remain
explicit. The v2 report now also ends with an unambiguous INVALID/FAIL
conclusion, consistent with its supersession amendment.

## Campaign and artifact audit

The 100 retained batches form a contiguous, nonoverlapping `0..999999` chain.
A complete independent stream audit verified every canonical batch hash and
compatibility identity, every representative/event-vector class digest, global
class/representative counts, inactive phase normalization, all 24 actual-yield
histograms, totals, and the 2,438,244,471-byte ledger SHA-256
`d34822e2614904916d2ff68bc3a1e68da9c458705d02096b0f085be74b6ff006`.

Actual yield observation counts ranged from 500,113 to 3,000,226. Every ID
0--23 had a nonzero count; the former phantom sites 6, 7, 13--15, 19, 22, and
23 were respectively observed 1,000,000, 500,113, 1,250,713, 1,250,713,
1,250,713, 1,250,713, 1,000,000, and 500,113 times. Inactive normalization
was retained for 984,322 cancellation schedules and 992,298 error schedules.

The artifact manifest file SHA-256 is
`d9dd7ae41d8ff1265b10a1e949e2e6a5ec77f463f8a22ed104aa2ee4863fee90`;
its canonical self hash is
`3a3b77ae4a974033071e2bfdb58066c743cb7abeb61fdb96d08bfe82fd9d780a`.
Exact file-set, byte-size, SHA-256, and self-hash verification passed after the
campaign.

## Validation

- pre-seal task-scoped Ruff: PASS;
- pre-seal focused wedge suite: 28/28 passed;
- post-run task-scoped Ruff: PASS;
- canonical retained verifier: 67/67 anchors PASS;
- post-run full non-browser suite: 1,221 passed, 15 skipped, 3 deselected,
  29 subtests passed, with exactly one expected failure because the enforced
  v3 revision guard rejected the later `0955bf0...` HEAD;
- post-run artifact, full-ledger, 966-file source, resume-prefix, and ring audits:
  PASS.

At the sealed application revision, the exact suite bridge is
**1,177 + 13 graph-repro tests + 4 bandwidth-attribution tests + 28 v3 wedge
tests = 1,222**. Compared with v2's 23-test wedge suite, v3 adds five closure
tests. The later revision-guard failure is retained as enforcement evidence,
not hidden by weakening or rewriting the sealed identity.

## Bounded conclusion

Increment 1 is concluded and Phase 1 may be designed. This PASS only weakens
the transport/scheduler/output hypothesis. It does not clear the GPU/graph
hypothesis or the reliability precondition; GPU worker execution, CUDA graph
dispatch, GPU-side Mamba state, device tensors, and real-ZMQ process timing
remain untested.
