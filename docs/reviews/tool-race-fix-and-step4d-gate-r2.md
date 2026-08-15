# Deferred-tool abort liveness and Step 4d confirmation review — round 2

Date: 2026-08-13 UTC

**A VERDICT: FAIL — the corrected owner closes the original FC-worker circular
wait and lost-parent-abort reproduction, and every named branch routes through
it, but the transferred task is not itself cancellation-safe and a two-fatal-
branch race produces an unbounded cleanup/typed-task wait.**

**STEP 4D VERDICT: NOT REACHED — A failed, so the conditional three-run
non-browser confirmation gate was not run. Step 4d remains `INCOMPLETE`, with
the corrected ten-block descriptive splits retained below and no candidate-
ranking authority.**

No live request, browser, container, profiler, health, or running-stack probe
was performed. The repository was read-only except for this review file.

## What the correction does close

The revised deferred-timeout regression is production-shaped for the round-1
cycle. Its real worker is registered in `pipeline._fc_async_bg`; fake `abort()`
joins that worker with the production five-second bound; and the timeout
callback transfers ownership and returns instead of awaiting that join. In 20
fresh focused runs, both invalidation subcases passed each time. Every run
reported `1 passed, 2 subtests passed` in 0.29–0.31 seconds. The asserted fatal
boundary is under one second, the worker finishes, the join does not time out,
the registry is empty, close is 1011, and trace closure is retained.

The test also recreates the original prompt-peer cancellation window. Its
`send_json` hook records `abort_count` at the fatal
`function_call_output_timeout` publication boundary, sends the event, and then
cancels the endpoint task. `fatal_abort_counts == [1]`, cleanup completion, and
`session_trace_stopped` prove that ordinary cancellation of the parent endpoint
does not lose the already-transferred cleanup. This is a real improvement over
round 1.

Static audit also confirms that all model-owning fatal branches named in round
1 now enter the common owner:

- both session-position-limit sites (`server.py:6250`, `6391`);
- user-EOU settlement failure (`6541`) and response-cancel timeout (`6672`);
- typed-input failure after mutation (`6771`) and function-call loop limit
  (`6892`);
- deferred-input overflow (`7053`) and function-calling unavailable (`7186`);
- function-output-apply timeout, including its no-pending fallback
  (`7412`, `7419`), and function-cycle mismatch (`7431`);
- fatal turn-start validation/conflict (`7466`, `7478`, `7499`);
- fatal commit validation/conflict (`7611`, `7623`, `7644`); and
- generic internal error (`7977`).

The pre-ownership rejects, unsupported-protocol reject before model start, and
clean `session.stop` path appropriately remain outside this model-owning fatal
primitive.

## Blocking finding 1: cancelling the transferred owner breaks causality

The endpoint keeps a strong reference to the task. Synchronous fatal callers
await it through `asyncio.shield()` (`server.py:6124-6139`), while fire-and-
transfer callers are ultimately drained by the endpoint's cancellation-safe
`finally` (`6141-6156`, `7988-8003`). This protects the child from cancellation
directed only at the endpoint task. It does not protect the cleanup task from
being cancelled directly, as can happen when loop/process shutdown cancels all
remaining tasks.

`run_session_cleanup()` catches `CancelledError` from
`await model_call(engine.abort)` and treats it as a completed cleanup failure
(`server.py:6067-6073`). Cancellation of the asyncio executor future does not
stop the already-running executor thread. The task therefore proceeds to send
the fatal event, close the socket, close the trace, and release `active_client`
while `engine.abort()` can still be running.

A fresh bounded probe made `abort()` block after recording entry, captured the
named `voicechat-terminal-cleanup-*` task, and cancelled that task at its abort
await. At the fatal send boundary, `cleanup_completed` was false. The trace
recorded `model_abort_cleanup_error: CancelledError`; fatal close 1011 and
`session_trace_stopped` followed; only after the probe released the executor
thread did abort complete. Thus the advertised abort-before-fatal causality is
false under cancellation of the transferred owner, and the executor-side abort
can outlive both its asyncio owner and the active-client/trace lifetime.

The endpoint regression does not cover this window: it cancels the parent
endpoint after fatal send, never the transferred task while abort is pending.
It is the only endpoint assertion that samples abort state at a fatal send
boundary. The other common-owner endpoint tests assert error/close outcomes,
not the new primitive's abort-before-publication invariant.

## Blocking finding 2: two fatal branches can circular-wait

`start_terminal_cleanup()` is idempotent in the narrow sense: a second caller
gets the existing task, so two racing fatal branches do not create two cleanup
tasks or invoke abort twice (`server.py:6107-6122`). However, every synchronous
second caller then awaits that existing task. That creates a cycle when the
owner is already waiting for the calling typed task.

The reproduction starts a typed input with its model call blocked, then raises
an `input_turn_conflict` on the endpoint task. The common cleanup task claims
terminal ownership and waits for `typed_task` because the endpoint, not the
typed task, is `terminal_origin_task` (`server.py:6044-6053`). After the model
call is released, the typed task reaches `session_position_limit`, calls
`terminate_model_session()`, receives the existing cleanup task, and awaits it.
Cleanup is waiting for typed; typed is waiting for cleanup.

The nominal three-second typed-task timeout does not resolve this. Cleanup
cancels and gathers typed, but `run_typed_job()` enters its async `finally`,
reaches the position-limit fatal path again, and awaits the same cleanup task
while cleanup awaits the gather. A formally capped reproduction recorded the
first terminal claim and then exited with GNU `timeout` status 124 at eight
seconds; no fatal error/close boundary completed. The earlier trace showed the
single `input_turn_conflict` claim followed by two position-limit entries.

This is a new production-reachable circular wait in the common owner, and it
also shows why first-winner idempotence alone is insufficient for two fatal
branches racing.

## Suite and causality validation

- Full unfiltered suite: **1168 passed, 18 skipped, 29 subtests passed** in
  33.20 seconds.
- One full suite with every `browser_e2e` test deselected: **1168 passed, 15
  skipped, 3 deselected, 29 subtests passed** in 33.29 seconds.
- Focused corrected timeout regression: **20/20 fresh runs passed**, each with
  both subtests.
- The test-side causality assertion is meaningful for successful cleanup plus
  parent cancellation, but neither blocking probe above is represented in the
  suite. The green suite therefore does not discharge them.

## Step 4d gate and corrected ten-block splits

Because A fails, I did not convert the incidental single green non-browser run
into the requested three-run Step 4d streak. The conditional confirmation gate
remains deferred. The already-reviewed corrected ten-block descriptive values
are unchanged:

| class | interface wall, ms | graph envelope, ms | wall − envelope, ms | graph / residual split |
|---|---:|---:|---:|---:|
| content | **53.560786** | **45.847206** | **7.713580** | **85.608% / 14.392%** |
| PAD | **53.595332** | **45.579065** | **8.016267** | **85.053% / 14.947%** |

These remain descriptive impossible-zero-cost ceilings. The exact-output gate
remains failed, aggregate standing remains `INCOMPLETE`, and the values remain
inadmissible for Step 4e candidate ranking.

## Required disposition

Make the transferred task itself cancellation-safe: cancellation must not let
fatal publication, close, trace stop, or active-client release overtake the
actual completion of the executor-side abort. Shield and drain the abort work
to completion inside the owner before publishing, including during all-task
shutdown cancellation.

Also remove the cleanup/typed-task wait cycle. Once terminal ownership is
claimed, a typed task must not enter and await the same terminal owner while
that owner is waiting or gathering it. Add bounded regressions that (1) cancel
the cleanup task at the abort await and require abort completion before fatal
send and lock release, and (2) race a typed-origin fatal branch against an
endpoint-origin fatal branch and require exactly one abort plus bounded fatal
send, close, trace stop, and task completion. Only after those pass should the
three-run Step 4d confirmation gate be executed.
