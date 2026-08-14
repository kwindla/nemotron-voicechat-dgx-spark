# Step 5 pairing-wedge campaign preregistration — v1 invalid dry run

> **Amendment recorded before the v2 rerun (2026-08-14 UTC).** The execution
> made under this contract was a synchronous bookkeeping dry run, not a valid
> Phase 0 protocol-race campaign. Its reported PASS is retracted. The runner
> manually advanced already-ready generators, retained only an estimated 86
> normalized synchronous sequences, shimmed B0 and B3--B7, partially executed
> B1/B2/B8/B9, and suppressed audit records instead of protocol operations for
> all ten alleged fault injections. This v1 record is retained only as history;
> it is not reinterpreted as satisfying the contract below. The fresh v2
> contract is `preregistration-v2.md` and is sealed independently before v2.

**Preregistered:** 2026-08-14 UTC, before any Phase 0 or noninterference run

**Campaign scope:** increment 1 only: B0–B9 diagnostic instrumentation,
trace-off/trace-on noninterference, and the Phase 0 CPU-only protocol race
test. Phase 1 GPU discrimination and Phase 2 post-fix stress are excluded.

## Immutable source qualification

- Application qualified tree revision: Git commit
  `a22d237cb38897c135e9d11d0eb4ab3336bfee2e` (the clean tracked tree before
  this increment; unrelated untracked retained evidence is outside scope).
- Qualified vLLM tree: 966-file reconstruction SHA-256
  `d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`,
  reconstructed from native vLLM revision
  `b31e9326a7d9394aab8c767f8ebe225c65594b60`, VoiceChat vLLM revision
  `237ba3cafc14d514551539ad980f17d40d273dfa`, and escrow index SHA-256
  `8854e6eb87c7adec5ed0cdba9a9b9d5f16cecf62182c8e928be91058aabc1117`.
- Governing design: `docs/step5-pairing-wedge-closure-design.md`, SHA-256
  `a7c8c44419d9d7eb53d559e82e2cb0402985601487e9702751a29253dff322af`.
- Static/offline execution only. No container, GPU, inference, live service,
  or running qualification environment is in scope.

The diagnostic patch and every retained artifact are hash-recorded in the
Phase 0 manifest. Any source/hash mismatch fails closed.

## Governing §6.3 clauses pinned for increment 1

Boundary localization records are required at exactly these boundaries:

| ID | Required boundary |
|---|---|
| B0 | host packed-call entry and append intent |
| B1 | `EngineCoreAppendRequest` decoded by the core |
| B2 | scheduler custom-input install, waiting/running transition, and finalized descriptor |
| B3 | worker execute entry with request/epoch, token count, graph mode, and descriptor key |
| B4 | model forward returned |
| B5 | sampling plus speculative accept/reject and Mamba commit/restore returned |
| B6 | core output enqueued |
| B7 | core output sent |
| B8 | async client received and output processor routed the request |
| B9 | per-request iterator awakened and host call returned |

Records must be bounded and nonblocking, using a preallocated ring or sidecar.
Instrumentation is environment-gated off by default. Trace off and trace on
must produce exactly equal outputs on a fixed CPU-representable fixture. A
GPU-only residual is reported explicitly and cannot be called checked. A
timeout is diagnostic only and never recovery.

Phase 0 uses a fake executor/model-output source against the exact qualified
request, scheduler, output-processor, and per-request iterator classes. It
randomizes append arrival, waiting-input transition, output batches,
cancellation, and task-error timing. The pass contract is at least 1,000,000
schedules, zero lost B0-to-B9 completions, and exact request routing. A loss is
then injected independently at each B0–B9 boundary; all 10 injections must be
detected, and the first-missing-boundary histogram is retained.

Interpretation is fixed: a failure localizes the transport/scheduler/output
hypothesis to the first missing boundary. A pass only weakens that hypothesis;
it cannot clear the GPU/graph hypothesis or the reliability precondition.

## Frozen harness and diagnostic-patch file list

- `src/nemotron_voicechat_runtime/wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/patch_step5_wedge_boundary_trace.py`
- `src/nemotron_voicechat_runtime/runtime_optimizations.py` (B0/B9 only)
- `tools/qualification/step5_phase0_protocol_race.py`
- `tests/runtime/test_step5_wedge_boundary_trace.py`
- `tests/runtime/test_step5_phase0_protocol_race.py`

The vLLM patcher may modify only a copied/reconstructed offline tree and is
never run against a live environment in this increment. Tests may use
temporary fixture trees. Any necessary change outside this list is a recorded
deviation and does not silently enter the campaign.

## Seeds, schedules, and resumability

- Seed algorithm: counter-based SplitMix64; no interpreter or platform PRNG
  state is admitted.
- Root seed: hexadecimal `0x5EED5A17B0B9C0DE`.
- Schedule `i` uses SplitMix64(root seed + `i`) and deterministically derives
  request count/order, append arrival, waiting-input transition, output batch
  boundaries/order, cancellation point, and task-error point.
- Required schedules: exactly 1,000,000 valid schedules numbered
  `0..999999`; a resumed run must retain contiguous, nonoverlapping batch
  summaries and reject incompatible seed/source/config hashes.
- Fixture request IDs and outputs are deterministic functions of the schedule
  and lane. Exact routing compares each awakened iterator's request ID and
  payload digest with that schedule's expected mapping.
- Fault-injection runs use one deterministic sentinel schedule per boundary,
  derived from the same root seed with domain `fault/B0` through `fault/B9`.

## Phase 0 pass/fail contract

Phase 0 passes only if all of the following are true:

1. Qualified source identity and all harness/artifact hashes verify.
2. At least 1,000,000 schedules complete with no missing B0-to-B9 completion.
3. Exact request routing has zero mismatches, including randomized output
   batches, cancellation, and propagated task errors.
4. Each isolated B0–B9 loss is detected (10/10), with its first missing
   boundary equal to the injected boundary.
5. Trace-off and trace-on fixture outputs are byte-for-byte/exactly equal.
6. The report uses the bounded §6.3 interpretation verbatim in substance and
   records all residuals and deviations.

No acceptance criterion may be weakened after execution begins. A deviation,
source drift, invalid schedule, missing artifact, routing ambiguity, or
undetected injected loss is reported and fails closed.
