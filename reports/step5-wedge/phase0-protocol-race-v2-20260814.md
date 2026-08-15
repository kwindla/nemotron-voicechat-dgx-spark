# Step 5 pairing wedge: Phase 0 v2 task-path protocol race

> **Superseded / INVALID — 2026-08-14 round-2 amendment.** The round-2
> confirmation review found that v2 did not close six required deltas. It
> bypassed exact scheduler admission, advertised eight unreachable yield sites,
> counted inactive seeded phases as realized topology, implemented B9 as an
> already-awakened `aclose()` plus omitted record, verified only four patch
> targets, omitted run-shaping identity and retained artifacts from the PASS
> gate, and polluted the final enabled ring after taking its metadata snapshot.
> The former v2 PASS is retracted. Its raw artifacts remain retained exactly as
> evidence of that superseded attempt and are not reinterpreted post hoc. A
> separately preregistered v3 campaign is required.

**Date:** 2026-08-14  
**Result:** **INVALID / FAIL — former v2 PASS retracted**  
**Scope:** offline CPU only; no container, GPU, inference service, or live endpoint

The fresh v2 campaign completed 1,000,000 schedules and 1,500,113 request
lanes with zero missing B0--B9 completions and zero exact-routing mismatches.
It realized **502,930 distinct task-interleaving equivalence classes** against
the preregistered minimum of 64. Wall time was **269.482 seconds**.

This result does not rehabilitate v1. The v1 synchronous bookkeeping dry run
remains INVALID/FAIL and its PASS remains retracted.

## Seal and identity

The v1 retraction was recorded before v2. Then `preregistration-v2.md` was
sealed at SHA-256
`f6aeab5469f9e1f6b4590f5119d3371c6b81891e720e41624d95baf4e4256148`
before any v2 run. The pre-run compatibility identity was
`010ceec82d973f6521d5d73a3955d009f26143566acdb23f26a7c270967b8fbb`.
It binds the preregistration, qualified application base, governing design,
qualified vLLM tree, runner, all six frozen harness files, deterministic patch
manifest, and configuration. It recomputed exactly after the campaign.

The diagnostic patch manifest digest was
`840543de91f7e43e80289a8ab871d8895d6e5267106cd7b256e2101debd6cb5c`.
All 100 batch summaries formed a hash-verified contiguous, nonoverlapping
`0..999999` range. Resume validation recovered exactly 1,000,000 schedules,
1,500,113 lanes, and 502,930 distinct classes with no identity or batch drift.

## Real task path and schedule evidence

The harness executed AST-selected exact qualified definitions for
`AsyncLLM.generate`, `add_request`, `_add_request`, `_run_output_handler`, and
`abort`; `OutputProcessor` and `RequestOutputCollector`; `Request`; complete
`Scheduler.schedule` and final `SchedulerOutput`; and
`EngineCoreProc._handle_client_request` and `_process_engine_step`. Persistent
named asyncio tasks supplied core dispatch, the allowed fake executor/model
output, core output enqueue, sender, receive, exact output handler, collector,
and waiting per-request iterator. No generator used manual `send(None)`.

The locked CPU environment had no pyzmq, so the preregistered minimum topology
used real `asyncio.Queue` interfaces at core input, core output, and wire
receive. Twenty-four seeded yield decisions cover both sides of required
transitions. Output batch partition changes the actual `EngineCoreOutputs`
message topology. Each batch retains histograms for append/wait order, request
and output order, batch partition, cancellation and task-error target/phase,
and all yield points.

`equivalence-classes.json` retains a versioned replay schedule, realized event
vector, first/last index, and count for every normalized class. Its size is
1,057,810,101 bytes and SHA-256 is
`3a21dadd776a8734a4235b2eb667bb74d82208febd6b703c15fcba78fb46d626`.

## Boundary truth conditions

- B2 is emitted after custom-input install, waiting-to-running transition,
  final `SchedulerOutput` construction, connector/event finalization, and
  post-schedule bookkeeping; it includes status, membership, proposal tokens,
  input consumption, token count, and a canonical descriptor digest.
- B3 is immediately before forward after graph/token overrides and includes
  final graph mode, token count, batch size, epoch, and descriptor digest.
- B6 is emitted only after `put_nowait` succeeds.
- B8 is inside each successful collector `put`; ignored already-aborted output
  is not marked routed. Abort and propagated-error publications are explicit.

## Independent fault matrices

Fault schedules use domain-separated `fault/B0` through `fault/B9` seeds. The
operation switch is absent from the boundary detector. The operation matrix
drops or strands the real relevant operation; the separate trace self-test
suppresses only a record.

| Boundary | Operation-loss first missing | Operation detected | Record-only detected |
|---|---|---:|---:|
| B0 | B0 | yes | yes |
| B1 | B1 | yes | yes |
| B2 | B2 | yes | yes |
| B3 | B3 | yes | yes |
| B4 | B4 | yes | yes |
| B5 | B5 | yes | yes |
| B6 | B6 | yes | yes |
| B7 | B7 | yes | yes |
| B8 | B8 | yes | yes |
| B9 | B9 | yes | yes |

The retained matrix records every domain seed, derived in-range schedule, and
per-lane first missing boundary. Results were 10/10 for operation loss and
10/10 for record suppression.

## Atomic patch-on-copy

The patcher accepts distinct `--qualified-source-root` and nonexistent
`--output-root` arguments; rejects equality, existing outputs, aliasing, and
unsafe nesting; verifies all target hashes and all anchors in memory before
writing; copies into sibling staging; patches and compiles the complete staging
tree; writes the manifest inside staging; and atomically publishes with
`os.replace`. It never writes the source root. Tests hash the source before and
after success and after nine injected failures spanning verification, copy,
each target, compile, and pre-publish.

## Production-call-site noninterference

The 64-schedule fixture executes the instrumented production
`_generate_packed_pad_pair` host function plus the patched exact scheduler and
output path. Trace-off and trace-on produced identical SHA-256
`3d7577d4f38041051a31a96d7866c06c748ee895909907c9a8f047fc4260fbd2`,
zero missing completions, and zero routing mismatches. Trace-on retained 952
schema-v2 records with zero drops. The ring SHA-256 is
`15233daa4b7f63ddcd444a53384b8165dc0d61cc8bf9690dd0f8bb440a12606d`.

Trace-off has one kwargs allocation (plus detail scalar construction) at each
instrumented call and one `_RING is None` branch inside `record_boundary`.
The measured disabled call cost in this fixture was **78.940 ns/call**.

Residual: GPU worker execution, CUDA graph dispatch, GPU-side Mamba
commit/restore, device tensors, and real-ZMQ process timing were unavailable
and are not claimed checked.

## Phase 1 readiness additions

Adopted now: representative replay schedules and sentinel expansions;
schema/campaign UUID, process role, task and thread identity fields; canonical
descriptor serialization and real truncated-detail SHA-256; run nonce in ring
filenames; B2/B3/B5 Phase 1 fields; and the two independent fault matrices.

Deferred: a live-updated sidecar header (would add a write to every hot record),
worker/rank/output ordinals that are not truthful in the CPU-only topology, and
cross-process B2--B4/exactly-once analyzer invariants (they require Phase 1
multi-process records). These are deferred to Phase 1 rather than populated
with synthetic values.

## Validation and retained hashes

- task-scoped Ruff: PASS;
- focused wedge suite: 23 passed in 1.96 seconds;
- full non-browser suite: 1,217 passed, 15 skipped, 3 deselected, 4 warnings,
  and 29 subtests passed in 35.43 seconds;
- canonical retained verifier: 67/67 anchors PASS;
- eight-entry v2 raw manifest: all size/SHA-256 entries PASS, manifest SHA-256
  `fd38eab63d1caa8486fd201ef5f709fd307a56d1421fc301c175ee6c65f1341d`;
- `git diff --check`: recorded in the completion note.

The exact full-suite bridge is 1,177 + 13 graph-repro tests + 4 tests from the
later bandwidth-attribution commit at current HEAD + 23 v2 wedge tests = 1,217.
The historical v1 bridge remains exactly 1,177 + 13 + 6 = 1,196; v1’s six tests
were replaced by the 23-test v2 suite, not reinterpreted.

## Bounded conclusion

Phase 0 v2 is **INVALID / FAIL and superseded**. Its raw artifacts are retained
as historical evidence only; none of its former PASS claims authorize Phase 1
and none are reinterpreted post hoc. A separately sealed v3 campaign is the
only admissible successor.
