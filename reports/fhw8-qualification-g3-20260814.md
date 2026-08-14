# function_head W8 qualification: G3 live behavioral phase

**Date:** 2026-08-14 UTC
**Preregistration:** `reports/fhw8-qualification/g3-preregistration.md`,
SHA-256 `671784e2e17318c3adbb9d577e732376d8541dddd11acb6b6207e970c1cf10b8`,
sealed before any leg ran.
**Verdict: G3 PASS — G3-1, G3-2, and G3-3 all pass.** With G0–G2
(`reports/fhw8-qualification-20260814.md`), the candidate has completed every
preregistered qualification gate. Production promotion now requires only the
plan's counterbalanced A/B latency protocol and an explicit user decision.

## Candidate service

Started from a copy of the retained v4 script with exactly the preregistered
deltas: candidates mount, `--nano-vllm-path /models/candidates/nano-fhw8`,
`--vllm-manifest /models/candidates/fhw8-candidate-manifest.json`. The
manifest (SHA-256 `fbb121328a9c…`) is kind
`exact_public_vllm_quantized_nano_eartts` with the nano component computed
from the actual candidate files and the eartts component carried from the
signed release; its `byte_identical` reproducibility claim was earned by a
second independent builder run producing byte-identical files (13/13;
`data/double-build-evidence.json`). The server's own fail-closed artifact
validation accepted the manifest and loaded the candidate. Startup attempt 1
stalled at the first EarTTS warmup generation — the same intermittent stall
retained for the unmodified release in the Step 4d operations log — and was
retained (`g3-session…/preflight/candidate-attempt1-eartts-warmup-stall.log`);
the unchanged-script retry reached `ready`.

## Legs and gates

**Leg B — baseline function-calling (release service, before the swap):**
speech-mode tool-call cells (spoken carrier → transcription → tool call →
injected result → spoken answer), 2 replicates × 10 cells. **20/20** correct:
exact transcript, correct tool (`get_benchmark_word`), correct fresh word
spoken, zero errors. The harness's `tool_response_positions_match` sub-check
is inert in this environment (it requires the vLLM tool-latency ledger,
which is not installed in the qualification container); it is inert
identically in Leg C, so it does not affect the B-vs-C comparison. A first
invocation with `--replicates 2` raced session teardown (`server_busy`) and
is retained; replicates were rerun as separate invocations with idle
barriers.

**Leg A — candidate longform campaign (G3-1, G3-2):** the canonical
hash-pinned 8-fixture plan (verified `3e056280…`) via the unchanged Step 2
driver and browser harness, with pipecat playout tracing. Result: **passed,
8/8 fixtures, zero terminalization errors, 216.6 s** (baseline session:
215.9 s). Per-fixture categorical comparison against the retained baseline
session (`reports/step2-live/session-20260813T083303Z`): every completion
status, error field, transcript gate evidence, response attribution, marker
emission, and quiescence outcome is **equal**; the only differing fields are
`rtvi_window_start_index` values (session-position offsets, non-categorical).
**No new failure class — G3-2 PASS.**

**Leg C — candidate function-calling (G3-3):** same cells as Leg B on the
candidate service: **20/20** correct with identical transcripts, tool calls,
and spoken answers, zero errors. **G3-3 PASS.**

**G3-4 (descriptive latency observation):** per-fixture time-to-first-audio,
candidate vs baseline session (single samples, stochastic rendition lengths):
S1-1 7.28 vs 5.98 s; S1-2 4.39 vs 4.79; L1-1 14.34 vs 14.40; L1-2 13.67 vs
15.23; L2-1 15.03 vs 15.87; L2-2 14.95 vs 16.17; I1 13.74 vs 15.24;
capture-off-L1 15.65 vs 15.52. Five of eight fixtures faster, long fixtures
consistently ~1–1.5 s faster — the direction expected from −4.4 ms/frame
compounding over long generations. Promotion-grade latency remains the
plan's counterbalanced A/B protocol.

## Operations

Release service identity recorded before the swap; candidate service run
only for Legs A and C; release restored from the retained v4 script with
stable-field identity verification
(`g3-session…/preflight/release-identity-verification.json`). All failed or
stalled attempts retained. Evidence under
`reports/fhw8-qualification/g3-session-20260814T0530Z/` with SHA-256
manifest.

## Standing

The fhw8 candidate is now **fully qualified under its preregistered G0–G3
contract**: construction-exact, decision-invisible on every retained and
live decision surface exercised (offline corpus, live longform, live
function-calling), and measured at −4.437 ms/frame. Remaining before
production: the counterbalanced A/B latency qualification from the plan's
promotion contract, and the user's promotion decision. Follow-ons recorded:
attention-projection W8 (~1 ms), W4 head variant, basin-bias evidence feeding
the graph exact-gate repair.
