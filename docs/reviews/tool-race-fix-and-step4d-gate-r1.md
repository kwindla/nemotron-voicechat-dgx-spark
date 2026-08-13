# Deferred-tool abort liveness and Step 4d confirmation review — round 1

Date: 2026-08-13 UTC

**A VERDICT: FAIL — the send-boundary ordering is proven, but the change creates a production FC-worker/abort circular wait before that boundary and does not close the broader lost-abort class.**

**STEP 4D VERDICT: NOT REACHED — A failed, so the conditional three-run non-browser confirmation gate was not run and Step 4d is not newly PASS-admissible from this round.**

## Blocking production liveness finding

The new `await model_call(engine.abort)` is executed by the timeout coroutine that
the timed-out function-call worker itself is synchronously waiting for:

1. `ExternalToolBridge._invoke()` schedules `emit_timeout(call)` and blocks the
   FC worker in `timeout_future.result(timeout=5.0)` (`server.py:2206-2216`).
2. `emit_external_tool_timeout()` enters
   `terminate_pending_bos_expectation()` (`server.py:6765-6797`).
3. The reordered code submits `engine.abort()` to the single model executor
   before publishing the fatal error (`server.py:6465-6496`).
4. Production `VoiceChatEngine.abort()` sees the same stream in
   `pipeline._fc_async_bg` and calls `pipeline.reset_session()`
   (`server.py:4108-4117`).
5. The shipped NVIDIA patch makes `reset_session()` abort and join every FC
   worker for up to five seconds
   (`nemotron-voicechat-rnnt-turn-taking.patch:4241-4260`). The worker being
   joined cannot finish while it is waiting for the timeout coroutine, and that
   coroutine cannot finish while abort is joining the worker.

This is bounded by the two five-second waits rather than necessarily permanent,
but it adds an approximately five-second circular wait on the production timeout
path and can take a second reset attempt if the first join times out. Throughout
that interval the fatal error has not been sent, the socket is still open, and
`terminal_session` is still clear. Worse, `function_cycle_call_id` and
`pending_bos_settlement` have already been cleared, so the main receive loop can
admit another client event while model teardown is underway.

The focused fake cannot reveal this. `DeferredToolTimeoutFakeEngine` starts an
unregistered daemon `Thread`, has no production-shaped `_fc_async_bg` ownership,
and overrides `abort()` with an immediate counter increment. The new assertion
therefore proves ordering only for the fake; it does not prove that the fatal
boundary remains live on the production termination path.

There is also no `try/finally` spanning the new abort await and
`terminal_session.set()`. Cancellation or executor failure during that await can
still skip the terminal signal, fatal publication, and close entirely. Moving
the signal after close means the endpoint's receive task remains live for the
whole cleanup wait.

## Other requested checks

The retained outer abort is not itself a double-abort correctness problem.
Production `VoiceChatEngine.abort()` sets `started = False` in its `finally`, and
the single-worker executor serializes an inner and outer invocation. The second
call is therefore normally a no-op. The test appropriately requires exactly one
abort at the send boundary but only at least one after teardown.

The retained instrumentation supports the central diagnosis that abort delivery
was genuinely lost: the decisive run recorded completed invalidation, fatal
receipt, client-context exit, no `abort()` entry after a two-second event wait,
and no `session_trace_stopped`. The old-runtime send-boundary reproduction below
also rules out a merely slow post-send abort. The more specific statement that
cancellation occurred *before the outer `model_call()` submitted work* remains
an inference, however: the retained record has no explicit events for outer-
`finally` entry, abort submission, `terminal_session.set()`, or the cancellation
site.

## Complete WebSocket-close audit

I inspected every `websocket.close()` call in `create_app()`.

- The pre-ownership `model_cleanup_failed` and `server_busy` rejects do not own a
  live model session. The initial unsupported-protocol reject also precedes
  model start.
- `session.stop` cleans the model before close: it either aborts an empty prefill
  or executes the final `is_last=True` model step. This path does not have the
  lost-abort window.
- The two session-position-limit branches, user-EOU settlement failure,
  response-cancel timeout, typed-input failure after mutation, function-call
  loop limit, deferred-input overflow, function-calling-unavailable after an
  existing tool-free prefill, function-output-apply timeout fallback,
  function-cycle mismatch, the turn-start and commit fatal validation/conflict
  branches, and the generic internal-error handler all publish an error and/or
  close before relying on the outer `finally` for abort. They retain the same
  structural lost-abort window.
- The typed-input fatal path and function-call-loop-limit path originate in
  background tasks and do not set `terminal_session`, so their ownership handoff
  is weaker still. `terminal_session` is used only by the changed deferred-BOS
  helper.

Thus the patch is a local ordering repair, not a safe common terminalization
primitive, and the same cancellation-sensitive cleanup dependency remains in
multiple reachable model-owning branches.

## Reproduction and validation

- Fixed working tree, focused test:
  `1 passed, 2 subtests passed` in 0.29 s.
- Detached worktree at `HEAD` (`f005150`) with only the new test-side
  send-boundary assertion applied: both invalidation subcases deterministically
  captured `fatal_abort_counts == [0]` and failed against `[1]`; pytest reported
  `2 failed, 1 passed` in 0.38 s. The temporary worktree was removed afterward.
- No browser test, live request, container query, profiler, or running-stack
  probe was performed.
- The three requested full-suite runs excluding `browser_e2e` were intentionally
  not run because their prerequisite, A passing, was not satisfied.

## Required disposition

Terminate input ownership before waiting for cleanup, and decouple timeout
notification from teardown that joins the notifying FC worker. The production-
shaped test must register the worker in `_fc_async_bg` and prove a bounded fatal
send/close plus completed cleanup without a circular wait. A common terminal
path should also make model abort and trace/lock release cancellation-safe for
the other model-owning fatal branches identified above. Only after A passes
should the Step 4d non-browser suite be run three times and its admissibility
reconsidered using the already-verified corrected ten-block splits.
