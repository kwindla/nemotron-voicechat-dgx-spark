# Deferred-tool abort liveness and Step 4d confirmation review — round 3

Date: 2026-08-13 UTC

**A VERDICT: PASS — the transferred terminal owner now drains executor-side
abort to actual completion under direct and all-task cancellation, and the
typed/owner wait cycle is removed. All prior reproductions and a new repeated-
cancellation/raising-abort attack completed within their bounds with one abort
and no terminal effect overtaking abort completion.**

**STEP 4D VERDICT: PASS FOR DESCRIPTIVE ADMISSIBILITY, WITH `INCOMPLETE`
STANDING — three fresh non-browser suite runs passed. The corrected ten-block
splits are admissible as descriptive input, including in the Step 4e table, but
the failed exact-output gate keeps Step 4d aggregate standing `INCOMPLETE` and
the values remain inadmissible for candidate ranking.**

No live request, browser, container, profiler, health query against a running
stack, or other live probe was performed. The repository was read-only except
for this review file. Temporary trace data used by the offline probes was
created only under `/tmp` and removed by the probes.

## A. Deferred-tool abort liveness gate

### Static audit and pinned coverage

The round-3 implementation addresses both round-2 blockers.
`drain_cancellation_safe()` clears pending cancellation requests, repeatedly
shields the protected future, and returns only after it is done. The owner
submits `engine.abort()` with `run_in_executor()` and protects that returned
`asyncio.Future`; it is not an asyncio `Task` and therefore is not selected by
an `asyncio.all_tasks()` shutdown cancellation wave. An owner cancellation can
cancel a shield wrapper but cannot cancel or detach the executor work. Abort
exceptions are observed only after that work has completed and are recorded
before terminal publication proceeds.

The two-fatal-branch cycle is also removed. If a typed task encounters a fatal
condition after another task has claimed terminal ownership, it emits
`terminal_cleanup_wait_skipped` and returns into its `SessionTerminated` unwind
instead of awaiting the owner that is waiting for or gathering that typed task.
The first claimant still exclusively selects the reason, event sequence, close
code, and single abort.

The three requested pinned regressions exist in
`tests/runtime/test_realtime_websocket_endpoint.py`:

- `test_terminal_owner_drains_abort_when_directly_cancelled`;
- `test_terminal_owner_drains_abort_during_all_task_shutdown_cancellation`; and
- `test_racing_endpoint_and_typed_fatals_complete_without_wait_cycle`.

The earlier production-shaped regression
`test_deferred_tool_timeout_is_fatal_even_if_epoch_invalidation_fails` also
remains present with both invalidation subcases, registered FC-worker draining,
abort-before-send sampling, parent cancellation, bounded fatal delivery, trace
closure, and registry cleanup assertions intact.

### Fresh historical probes

I ran all four named tests together in ten consecutive fresh pytest processes
under a 45-second outer bound. Every iteration reported **4 passed, 2 subtests
passed** in 0.34–0.37 seconds. Thus each of the following passed **10/10**:

- the prior deferred FC-worker/lost-parent-abort reproduction, including both
  invalidation subcases;
- the round-2 direct cancelled-owner reproduction;
- the round-2 all-task shutdown cancellation wave; and
- the exact endpoint-wins/typed-position-limit two-fatal-branch race.

The race retained its one-second fatal boundary and required one abort, the
endpoint's `input_turn_conflict` as first winner, the typed task's
`session_position_limit` skip trace, fatal send, close, trace stop, active-lock
release, and completion of endpoint, cleanup, and typed tasks. The direct-owner
and shutdown-wave tests both held fatal send, close, trace stop, and lock release
behind a deliberately blocked abort, then required successful non-cancelled
owner completion after abort was released.

### New attack: repeated cancellation plus a completed abort exception

I added no test file for the reviewer-only attack. An inline isolated-loop probe
used an executor-side abort that blocked, marked actual completion, and then
raised `RuntimeError`. While it was blocked, the probe delivered two waves of
32 direct cancellation requests to the transferred owner.

After all 64 requests, and before releasing abort, there was no fatal send,
WebSocket close, `session_trace_stopped`, or active-client release. After
release, the endpoint and owner completed within one second; the owner was not
cancelled, had no exception, and had a zero cancellation count. The probe
observed exactly one abort, one `model_abort_cleanup_error` carrying the
synthetic `RuntimeError`, fatal send and close both sampling completed abort,
trace stop, and active-client release. This covers accumulated cancellation
requests together with the executor-failure branch, beyond the three pinned
probes.

No remaining abort-causality or cleanup-cycle blocker was reproduced or found.

## B. Three-run non-browser confirmation and Step 4d disposition

After A passed, I ran the complete suite three times from separate pytest
processes with:

```text
.venv/bin/pytest -q -m 'not browser_e2e'
```

All three runs passed:

| run | passed | skipped | deselected | subtests | pytest time |
|---:|---:|---:|---:|---:|---:|
| 1 | 1171 | 15 | 3 | 29 | 33.52 s |
| 2 | 1171 | 15 | 3 | 29 | 36.58 s |
| 3 | 1171 | 15 | 3 | 29 | 33.52 s |

Each run reported the same four warnings: three dependency/deprecation warnings
and the previously observed unawaited `FrameProcessor.__process_frame_task_handler`
runtime warning in
`test_local_interruption_discards_incomplete_sentence_tail`. They did not alter
the test results or the three-run gate.

The corrected, previously independently verified ten-block descriptive values
are therefore admitted unchanged:

| class | interface wall, ms | graph envelope, ms | wall − envelope, ms | graph / residual split |
|---|---:|---:|---:|---:|
| content | **53.560786** | **45.847206** | **7.713580** | **85.608% / 14.392%** |
| PAD | **53.595332** | **45.579065** | **8.016267** | **85.053% / 14.947%** |

These are descriptive impossible-zero-cost ceilings. The confirmation gate
does not change the retained exact-output result: all measured blocks remain
invalid on the documented call-39 token mismatch, so aggregate Step 4d remains
`INCOMPLETE`. The table may inform Step 4e's descriptive decomposition, but it
does not establish a realizable gain, identify a tunable kernel family, or
authorize candidate ranking.
