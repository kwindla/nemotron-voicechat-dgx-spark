# Final gate review: generation-frame latency plan, r3

Review date: 2026-08-12. Scope was offline/read-only inspection of r3,
`docs/reviews/generation-latency-plan-review-r2.md`, the retained sequential
preflight and pair-era traces, and the repository code needed to identify the
trace boundaries. I did not inspect or change the running stack, connect to the
model WebSocket, or run a live probe.

## Verdict

**READY-TO-IMPLEMENT.** All six r2 closure conditions are met. The replacement
structural target also has the required retained-trace discrimination: it
accepts the bursty pair-era schedule and rejects the sequential schedule. No
new implementation blocker emerged.

## Closure-condition verification

| # | Verdict | R3 evidence and verification |
|---:|---|---|
| 1 | **CLOSED** | R3 publishes one normative schema with explicit `delivered`, `delivered-nonBOS`, and comparable `idle` predicates; a separate BOS stratum; stage-based high-mode predicates; named server, arrival, and browser boundaries; nearest-rank percentiles; and response-clustered inference. The anchor table uses `server_step_ms` consistently. Its values match the r2 recomputation: sequential delivered n=34, 86.168 ms; aged delivered n=193, 85.403 ms; aged delivered-nonBOS n=188, 82.919 ms; pair delivered n=32, 75.116 ms; fresh idle n=2,692, 70.926 ms; aged idle n=2,454, 70.627 ms. I independently rechecked the currently retained sequential, fresh-idle, and pair values as 86.168441, 70.926184, and 75.116344 ms. The old unnamed 184-row rendition window is no longer used. |
| 2 | **CLOSED** | The exact release-aware state machine is normative in Problem 3, Target 1, Step 0, Step 3, and A6. The contract includes exact sample counts, threshold release, short-response `response.done` flush, interruption at every arrival, and Pipecat plus AudioWorklet timestamps. Zero-reserve debt is explicitly informational. A6 now calls its 116/203/290/376 ms quantity added release delay, not audible onset, and gives the corrected pooled cadence of 86.589 ms over 221 intervals. |
| 3 | **CLOSED** | The physical-call p95 gate is removed. Target 2 instead requires K=12 logical-frame throughput plus a default maximum arrival gap of 160 ms, while Target 1 separately requires the normative release-aware no-underrun result. The retained-trace acceptance/rejection calculation is reproduced below. The pair-era 118.27 ms physical-call p95 therefore no longer disqualifies the leading historical schedule. |
| 4 | **CLOSED** | R3 chooses the qualification-only branch: the no-text override is env-gated and refused in production images. Fixture validity is audio completeness against a known script. The production watchdog repair and its two-dimensional liveness contract, independent audio-only/max-response bound, and boundary matrix are explicitly moved to a separate correctness plan. |
| 5 | **CLOSED** | The non-recovery timeout example is removed and explicitly rejected: a bounded timeout plus session-fatal close cannot restore a wedged EngineCore. Pairing is inadmissible until one of exactly three recovery closures holds: eliminate the packed two-position request while retaining amortization; fix the worker/graph/output-wakeup defect and prove deterministic replay plus long worker stress; or supervise kill/recreate with clean reinitialization, single-session failure, and a service-recovery SLO. R3 also requires an executable tail state machine, all named control/function/abort boundaries, and a same-image counterbalanced queue-clock A/B over independent responses. |
| 6 | **CLOSED** | Step 4 now profiles the dominant Nano term for content and PAD states, requires an exact-output component benchmark, and produces a table containing recoverable ceiling, correctness scope, reliability risk, and promotion floor. Step 5 says selection is made **by** that table, declares no winner, and keeps pairing inadmissible until its reliability precondition closes. |

## Retained-trace structural-gate check

I evaluated the written K=12 rule literally for each consecutive 12-frame
window as `t[i+11] - t[i] <= 960 ms`, and evaluated every adjacent arrival gap
against 160 ms.

| Retained schedule | Delivered frames | 12-frame windows | Worst window | Windows over 960 ms | Maximum gap | Result |
|---|---:|---:|---:|---:|---:|---|
| Pair era `f110f063…` | 32 | 21 | 877.871 ms | 0 | 120.939 ms | **ACCEPT** |
| Sequential preflight | 34 | 23 | 1009.957 ms | 13 | 151.672 ms | **REJECT** |

The sequential trace passes the seam bound and fails specifically on logical
throughput, as intended. The result is not an off-by-one artifact: treating K
as 12 complete interarrival intervals instead gives pair 925.214 ms with zero
failures and sequential 1097.344 ms with all 22 windows failing. Step 0 should
encode the literal 12-frame convention above so the implementation has only
one definition.

For the historical pair artifact, the retained timestamp is the server trace
written after the output events are sent; the sequential artifact also has
direct client-receipt timestamps. This is sufficient to establish that the
gate shape admits the retained bursty production schedule, but it is not
substituted for promotion evidence. R3 correctly requires browser-clock
qualification and a same-image queue-clock A/B before any Step 5 candidate is
promoted.

The pair trace's reported approximately 4.5 ms prefix debt also recomputes:
the maximum cumulative `sum(gap_ms - 80)` is 4.530001 ms. The sequential
client-arrival prefix debt remains 293.889 ms.

## Ordered implementation sequence

1. Land Step 0's offline schema/analyzers, including membership manifests,
   source hashes, the exact K=12 convention above, release-aware replay, and
   response-clustered reporting; require reproduction of all anchors.
2. Complete Step 1's aligned retained-evidence inventories and record the
   already-retained negative controls without rerunning them.
3. Implement Step 2a's production-refused qualification override, then Step
   2b's Pipecat/AudioWorklet receipt, release, queue, and underrun capture.
4. Run Step 3's preregistered 8–10 s fixtures and interruption sweep, size the
   temporary reserve from the exact replay, and confirm it at the browser.
5. Execute Step 4's Nano content/PAD and remaining stage profiles and publish
   the candidate ranking table before selecting any structural change.
6. In Step 5, select the highest-ranked **admissible** candidate. If pairing
   ranks first, close one of its three reliability preconditions and its full
   boundary state-machine qualification before implementation and the
   counterbalanced A/B.
