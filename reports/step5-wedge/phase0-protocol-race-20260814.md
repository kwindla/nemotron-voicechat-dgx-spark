# Step 5 pairing wedge: v1 synchronous bookkeeping dry run

**Date:** 2026-08-14  
**Amendment:** recorded before the v2 rerun  
**Result:** **INVALID / FAIL — the former PASS is retracted**

The retained v1 artifacts are hash-consistent deterministic bookkeeping, but
the run did not execute the preregistered Phase 0 task-path race contract. Its
1,000,000 seed indices collapsed to an estimated **86 normalized synchronous
operation sequences**. The reported 34.264 seconds was 34.264 microseconds per
schedule, not evidence of asynchronous task interleavings.

The synchronous runner manually called `send(None)` on already-ready async
generators. It did not run the real `_run_output_handler` task, core input and
output queues, client receive loop, sender, or scheduler-to-worker topology.
Boundary fidelity was B0 harness-only; B1 shimmed; B2 partial and stopped at
the first waiting transition; B3--B7 harness-only fabricated records; B8 a
direct exact leaf-method call without the receive task; and B9 a partial exact
generator fast path without a waiting iterator. Thus the claim that only the
executor/model-output source was faked was incorrect.

Every alleged B0--B9 loss merely suppressed the corresponding `BoundaryAudit`
record while leaving the protocol operation intact. The v1 10/10 matrix is a
record-suppression trace self-test, not an operation-loss matrix. It cannot
support a Phase 0 PASS.

The v1 noninterference fixture called `BoundaryAudit.mark` rather than the
instrumented packed host and patched scheduler/output production call sites.
It established equality only for that synthetic audit fixture. GPU worker,
CUDA graph, Mamba, device tensor, and real-ZMQ timing remained residuals.

The original patcher modified its supplied root in place and could leave a
partially patched tree after a later target failed. Operator convention did not
make it structurally patch-on-copy or atomic.

## Exact suite-count bridge

The earlier report’s baseline explanation was incomplete. The exact bridge is:

| Contribution | Passed-test delta |
|---|---:|
| Step 5 graph-repro review round 1 to round 2 | +13 |
| Wedge increment v1 (three tests in each of two new modules) | +6 |
| **Total** | **+19: 1,177 to 1,196** |

All 19 were new Step 5 tests: commit `c6128e7` added the graph-repro test file,
and wedge v1 added the two six-test modules. No pre-existing test or subtest was
changed to create that bridge.

## Disposition

The raw v1 data remain under `phase0-data/` without reinterpretation. A fresh,
separately preregistered v2 campaign is required. **A pass only weakens the
transport/scheduler/output hypothesis; it cannot clear the GPU/graph hypothesis
or the reliability precondition.** Phase 1 and Phase 2 remain unrun.
