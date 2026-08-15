# Step 5 pairing-wedge Phase 0 preregistration — v2 task-path race

**Preregistered:** 2026-08-14 UTC, after the v1 retraction and before any v2
campaign, v2 fault-matrix, or v2 noninterference result.

## Scope and immutable identity

This is a fresh amendment, not a reinterpretation of v1. It is CPU-only and
uses no container, GPU, inference service, or live endpoint. Application
revision is `a22d237cb38897c135e9d11d0eb4ab3336bfee2e`; qualified vLLM tree SHA-256
is `d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.
The governing design is `docs/step5-pairing-wedge-closure-design.md` §6.3.

The pre-run compatibility identity must contain and the pass predicate must
verify: this preregistration SHA-256; application revision; design SHA-256;
qualified-tree SHA-256; runner SHA-256; the SHA-256 of all six frozen harness
files; deterministic diagnostic-patch manifest SHA-256; and the configuration
below. The post-run artifact manifest must hash every retained v2 artifact and
must verify before PASS. Any mismatch fails closed.

Frozen harness files:

- `src/nemotron_voicechat_runtime/wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/runtime_optimizations.py`
- `tools/qualification/step5_phase0_protocol_race.py`
- `tests/runtime/test_step5_wedge_boundary_trace.py`
- `tests/runtime/test_step5_phase0_protocol_race.py`

## Real task topology

The exact qualified `AsyncLLM.generate`, `AsyncLLM.add_request`,
`AsyncLLM._add_request`, `AsyncLLM._run_output_handler`, `AsyncLLM.abort`,
`OutputProcessor`, `RequestOutputCollector`, `Request`, complete
`Scheduler.schedule`, finalized `SchedulerOutput`, `EngineCoreProc` request
dispatch, and `EngineCoreProc._process_engine_step` are AST-selected from the
hash-qualified source (or its deterministic instrumented copy) and executed.
Only the executor/model-output source is fake.

Persistent named asyncio tasks implement core dispatch, fake execution, core
output enqueue, send, receive, exact output handling, collector publication,
and per-request iterator wait/awakening. CPU-local pyzmq is unavailable in the
locked environment, so the minimum allowed topology is used: real
`asyncio.Queue` interfaces at core input, core output, and wire receive, with
24 deterministic seeded yield decisions covering both sides of all required
transitions. No generator is manually driven with `send(None)`.

## Schedules and retained diversity

- Schedule format/version: v2 expanded JSON plus four-word replay vector.
- Root seed: `0x5EED5A17B0B9C0DE`, counter-based SplitMix64.
- Exact range: contiguous, nonoverlapping `0..999999` (1,000,000 schedules).
- Randomized/retained dimensions: append-vs-wait order; request order; output
  order; actual output batch partition; cancellation target/phase; task-error
  target/phase; and every one of 24 yield points.
- Required distinct realized task-interleaving equivalence classes: **at least
  64**. A class is derived from the realized event/yield sequence and normalized
  categorical topology, never the schedule index or payload.
- Each batch retains all dimension/yield histograms and equivalence counts.
  One expanded replay schedule, realized event vector, first index, last index,
  and count is retained per equivalence class.
- Resume accepts only a hash-verified contiguous prefix beginning at zero, with
  exact compatibility identity and no gap, overlap, reordering, or batch hash
  drift.

## Fault matrices

Fault seeds are domain-separated as `fault/B0` through `fault/B9`; schedule
indices beyond the campaign range are forbidden. The operation-loss switch is
not present in the boundary detector.

The operation-loss matrix must independently induce: B0 no host append
operation; B1 dropped core dispatch; B2 withheld scheduler install/runnable
transition; B3 non-publishing execute entry; B4 forward non-return; B5 sample
non-publication; B6 dropped real core queue item; B7 suppressed real queue send;
B8 removal/prevention of actual collector publication; and B9 close/strand of
the real waiting iterator before host return. All ten must be detected at the
injected first missing boundary. A separate record-suppression trace-self-test
matrix must also detect 10/10 without being called operation loss.

## Boundary truth and noninterference

B2 is recorded only after complete install, waiting-to-running transition,
final `SchedulerOutput`, and post-schedule bookkeeping. B3 observes final
post-override graph mode/token count. B6 follows successful `put_nowait`. B8
is inside actual collector publication; ignored aborted output is not routed.

Trace-off/trace-on must be exactly equal on 64 fixed schedules through the
instrumented production `_generate_packed_pad_pair` host function and patched
scheduler/output path. The report must state disabled hot-path allocations and
branches or measured overhead. GPU worker, CUDA graph, Mamba device state,
device tensors, and real-ZMQ process timing remain explicit residuals.

## Pass predicate

PASS requires all identity and artifact hashes to verify; exactly 1,000,000
schedules; zero missing B0--B9 completions; zero routing mismatches; at least 64
distinct realized interleavings; operation-loss 10/10; record-suppression
10/10; and exact noninterference. Criteria cannot be weakened after this file
is sealed.

Interpretation is fixed: **A pass only weakens the transport/scheduler/output
hypothesis; it cannot clear the GPU/graph hypothesis or the reliability
precondition.**
