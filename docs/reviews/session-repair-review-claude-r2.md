# Independent review of commit fb4654d (session repair) — Claude

Reviewer: Claude (in-window), independent of the Codex review at
`docs/reviews/session-repair-review-r2.md`. Method: source reading plus
executable adversarial probes of the retained gate over real trace bytes.
CPU-only; no live runs; no files changed except this review.

## Verdict: PASS — merge-ready, with two recorded notes (neither blocking)

### I1 — termination evidence: CLOSED

The prior fail-open formulation ("any reason != `client_stop`") is gone.
Acceptance now requires positive evidence: a `terminal_session_claimed`
record positioned after the response BOS and before the stop, whose reason
equals the stop reason — or the single explicitly allowlisted unclaimed
transport lifecycle (`disconnect`).

I probed the retained corrected trace directly:

| Mutation | Result |
|---|---|
| baseline (unmutated) | PASS, schema-valid |
| stop reason typo'd (`client_stpo`) | **FAIL** — `terminal_claim_stop_reason_mismatch` |
| stop reason unknown (`totally_made_up`) | **FAIL** — same |
| abnormal stop with claim removed (`internal_error`) | **FAIL** — `response_open_at_eof` |
| claim + stop both forged to a matching arbitrary reason | PASS |

The first three are the review's actual delta and are closed. Cross-checked
against the producer: the only `close_reason` values are `disconnect`
(allowlisted), `client_stop` (clean → open response correctly fails),
`unsupported_protocol_version` (fails closed — the safe direction), and the
dynamic `terminate_model_session` reason, which always emits a matching
`terminal_session_claimed` first. So every real producer path is classified
correctly.

**Note 1 (out of scope, do not "fix").** The forged matching claim+stop pair
passes. That requires coordinated modification of two records, which is
tampering, not the malformed-telemetry/silent-regression threat the module
documents. The Codex R1 delta explicitly permitted this design ("a closed set
of producer-owned abnormal reasons **or** a matching preceding terminal
claim"). I recommend against adding a reason allowlist to close it: the
producer's terminate reasons are numerous and dynamic, so an allowlist would
require maintenance and would reintroduce exactly the false-positive failure
mode that the event-vocabulary allowlist caused earlier in this repair (a
legitimate record rejected, blocking production). Evidence-based matching is
the more robust contract.

### I2 — transcript attribution: CLOSED

`clear_pending_response_binding()` is invoked at the locked interruption edge
immediately after `interrupt_unpublished_function_cycle`, and returns evidence
that is folded into the interrupt record. It is not invoked on the
publication-race/deferred-input branch, per the delta.

`_bind_pending_transcript_at_bos()` has correct consume-once semantics: it
acts only on `agent_bos`, and clears pending text, attribution, and post-tool
continuation before deciding, so no later BOS can re-consume a stale binding.
The freeze at commit overwrites any earlier pending value, and client input
arriving during a function cycle is deferred rather than committed
(`client_input_deferred_for_function`), which prevents a second commit from
overwriting a deferred turn's binding before its BOS.

The endpoint test
`test_new_turn_interrupt_clears_superseded_binding_before_precommit_bos`
drives the real endpoint via `make_client`/`websocket_connect` with a
publication delay, i.e. it exercises the deferred-FC → new-turn interruption →
pre-commit BOS shape rather than calling the helper directly.

**Note 2 (minor robustness).** The clear is reached through
`getattr(engine, "clear_pending_response_binding", None)` guarded by
`if callable(...)`. If that method were ever renamed, the clear would be
silently skipped and the superseded-binding bug would return without any test
necessarily failing. A direct attribute call would fail loudly instead. Not
blocking — the method exists and is covered — but worth tightening when the
file is next touched.

### I3 / I4 — dispositions honest

I3 classifies all three retained cases as model behavior (`model_silence`,
`model_turn_taking`) with zero runtime-caused, and no boundary contract was
relaxed. I4 keeps `i4_fix_supported=false` with 0/3 in both arms; the
directive gate now requires the fresh result to be stated rather than merely
checking for absence of a stale string. Neither is claimed as a fix.

### Regression surface

Full non-browser suite 1,382 passed (sole failure is the documented
non-evergreen sealed revision guard), promotion manifest verifies 8 files,
and retained-evidence anchors 67/67. No changes to the no-text watchdog
repair, wall deadline, or fence contracts.

---

## Correction (post-hoc, after the independent Codex R2 review)

**My PASS above was wrong.** The independent review found a live I2 hole I
did not check: the **typed-input** request path had no ownership transition,
so a typed request arriving while a function cycle was deferred-but-
unpublished was accepted, mutated the model without interrupting the worker,
and its BOS consumed the superseded microphone turn's binding — reproduced
through the real endpoint with `source=microphone, source_id=1,
user_text_bound="old deferred tool request"`, `unpublished_interrupts=0`.

My review error was scoping to the path the fix had touched (microphone)
instead of asking where else a new turn takes model ownership. Symmetric-path
enumeration is now a required step for any fix of this shape. The two other
findings I also missed — the promoted image not carrying the checked-in
server, and browser manifests hashing pre-final reports — were likewise
outside the surface I probed.

I stand by the I1 analysis and probes above, and by Note 1 (do not add a
reason allowlist) and Note 2 (the `getattr`/`callable` guard).
