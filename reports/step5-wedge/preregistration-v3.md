# Step 5 pairing-wedge Phase 0 preregistration — v3 closure amendment

**Preregistered:** 2026-08-14 UTC, after the v2 INVALID/FAIL retraction and
before any v3 campaign, v3 fault-matrix, or v3 noninterference result.

## Scope and immutable identity

This is a fresh v3 campaign, not a reinterpretation of v1 or v2. It is CPU-only
and uses no container, GPU, inference service, or live endpoint. The enforced
application base revision is
`7e2e47fddb0da8288651d3158fc5a93b46ca202e`. The qualified 966-file vLLM tree
SHA-256 is
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.
The governing design remains `docs/step5-pairing-wedge-closure-design.md` §6.3,
with the round-2 closure requirements authoritative for this amendment.

The pre-run compatibility identity must contain and the PASS predicate must
verify this preregistration SHA-256, the live Git application revision, design
SHA-256, qualified-tree SHA-256, runner SHA-256, all six frozen harness hashes,
the deterministic patch-manifest SHA-256, and the complete run-shaping
configuration below. Any mismatch fails closed.

Frozen harness files:

- `src/nemotron_voicechat_runtime/wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/runtime_optimizations.py`
- `tools/qualification/step5_phase0_protocol_race.py`
- `tests/runtime/test_step5_wedge_boundary_trace.py`
- `tests/runtime/test_step5_phase0_protocol_race.py`

Configuration:

- schedule schema: v3;
- root seed: `0x5EED5A17B0B9C0DE`;
- exact schedule range: contiguous `0..999999`;
- batch size: exactly 10,000;
- concurrency: exactly 128;
- required realized interleavings: at least 64;
- reachable seeded yield IDs: every integer `0..23`.

## Exact task path and scheduler admission

The harness executes the exact qualified `AsyncLLM.generate`, `add_request`,
`_add_request`, `_run_output_handler`, and `abort`; exact `OutputProcessor` and
`RequestOutputCollector`; exact `Request`; complete exact `Scheduler.schedule`
and final `SchedulerOutput`; exact `EngineCoreProc` request dispatch and
`_process_engine_step`; and the exact qualified `FCFSRequestQueue`. Core ADD
must call exact `Scheduler.add_request`, leaving the exact request in WAITING
and in the real FCFS waiting queue. After the append installs custom inputs,
exact `Scheduler.schedule` must pop that queue and perform WAITING-to-RUNNING
admission before B2. Only executor/model output is fake.

Persistent asyncio tasks and real `asyncio.Queue` interfaces implement core
dispatch, fake execution, core output enqueue, sender, wire receive, exact
output handler, collector publication, and waiting iterator. All 24 seeded
yield IDs must be observed in a qualifying representative run and per-batch
yield histograms count actual invocations only. No generator is manually driven
with `send(None)`.

## Realized-interleaving definition and resume

An equivalence class is the SHA-256-derived ID of the realized ordered event
vector only. The vector includes lane-qualified task/yield events, actual
output-message partition, scheduler descriptor production, forward/sample
publication, iterator awakening, and any realized cancellation or error.
Schedule index, replay words, payload values, and inactive categorical seed
bits are excluded. Cancellation/error phase is normalized to `inactive` when
there is no target; active phases are represented only by their different
realized ordering.

Each batch retains versioned expanded schedules, four-word replay vectors,
actual realized event vectors, actual yield invocations, class counts, and one
representative with local first/last/count for every class in that batch.
Resume must validate the ordered hash-chained compatibility-identical prefix
and reconstruct global representative first/last/count state from those batch
records before appending. A representative/class coverage or count mismatch
fails closed.

## Independent fault matrices

Fault schedules use domain-separated `fault/B0` through `fault/B9` seeds and
in-range derived schedules. The boundary detector has no operation-loss switch.
B0–B8 lose the corresponding append, dispatch, install/transition, execute,
forward, sample, core queue, wire queue, or collector publication operation.

For B9, a real non-final output must complete B0–B8 and awaken the exact
generator once. The fake source then withholds the final output while the exact
generator and host consumer wait for the next collector item. Detection
requires every host consumer to remain pending before cleanup cancellation; an
already-awakened `aclose()` followed by normal host return is forbidden. A
separate record-suppression matrix must independently detect 10/10.

## Patcher and retained-artifact gate

Before its first write, the patcher must verify every one of the 966 regular
source files against the qualified index and recompute the full-tree digest,
then verify every patch anchor. The staging copy is reverified before patching,
fully compiled, manifested, and atomically published to a distinct nonexistent
output. Success, injected failures, target drift, and non-target drift must
leave the source unchanged.

The post-run artifact manifest recursively enumerates every retained v3 data
file, including the final summary and trace ring. The manifest covers its own
semantic content with `self_canonical_sha256`, defined as SHA-256 of canonical
manifest JSON with only that field omitted. PASS requires exact file-set,
size/hash, and manifest-self verification. There may be no unenumerated v3 data
artifact.

## Noninterference

Trace-off and trace-on must be exactly equal on 64 schedules through production
`_generate_packed_pad_pair` and the patched exact scheduler/output call sites.
The disabled-cost loop runs only in a separate trace-disabled subprocess and
must not touch the enabled functional ring. The retained ring's parsed record
count must equal its final metadata with zero drops, and the ring must be in the
artifact gate. Disabled allocations/branches and measured cost are reported.
GPU worker, CUDA graph, Mamba device state, device tensors, and real-ZMQ process
timing remain explicit residuals.

## Pass predicate

PASS requires exact compatibility identity; exactly 1,000,000 schedules; zero
missing ordinary B0–B9 completions; zero routing mismatches; at least 64
distinct realized event vectors; complete representative coverage; genuine
operation-loss detection 10/10 including pending-before-cleanup B9;
record-suppression detection 10/10; exact production-call-site
noninterference; a truthful zero-drop retained ring; and the complete
self-covered artifact gate. Criteria cannot be weakened after sealing.

Interpretation remains bounded: **a pass only weakens the
transport/scheduler/output hypothesis; it cannot clear the GPU/graph hypothesis
or the reliability precondition.**
