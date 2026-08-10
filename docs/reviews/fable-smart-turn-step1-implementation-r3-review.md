# Fable final gate review — Step 1 implementation, round 3 (live-qualified)

Reviewed: the current uncommitted worktree against `codex/post-fc-pair-recovery`
`e19f906`, including plan revision 5, the settlement-fence implementation, the
monotonic transcript accumulator with legacy gating, the six-cell memory
matrix harness, and the live artifacts named below.
Prior reviews: `fable-smart-turn-step1-implementation-review.md` (r1, CHANGES
REQUIRED), `fable-smart-turn-step1-implementation-r2-review.md` (r2, APPROVE
pending live gate). This review closes the live gate.
No implementation code modified. All artifact claims below were re-derived
from the raw event streams and reports, not from summaries.

## Plan revision 5 — settlement amendment

Verified congruent with the implementation (details in the pane assessment of
2026-08-08): invariant 6 frames settlement as "serialized model advancement,
never wall-clock pacing or endpointing"; commit remains the only EOU
authority; the missing-to-ten-blank fence, twice-fence fatal bound, per-step
position guards, Pocket parity, and the preserved exactly-one post-settlement
EOU step are all in both plan text and code. The plan records the empirical
basis inline (0 blanks → 2/3 empty responses; ~5 → browser failure; 10 → 3/3
pass), which is exactly how a live-derived constant should be documented.
`settle_user_eou` reads the fence from the wrapper's own
`nonblank_reset_after_silence`, injects only missing frames, and traces
`user_eou_settled` with start/end blank counts and step count.

The transcript accumulator sits at the right seam (engine display state, not
protocol slicing, not the pipeline's functional predictor reset), promotes
segments on `y_sequence` emptying, and — after my adversarial catch in the
previous round — is gated to external-EOU mode with the legacy replacement
behavior preserved and regression-tested from both the engine and protocol
sides (`test_legacy_user_transcript_does_not_accumulate_across_phrase_clears`,
`test_second_turn_transcript_is_not_session_cumulative`).

CPU suites at review time: 278 passed, 2 skipped.

## Live artifacts verified

### Browser gate (`test_live_playground_typed_turn_and_audio`, session `6ba3d357…`)

Re-derived from `/tmp/browser-smart-turn-r6-events.jsonl` (2,998 events):

- Settlement ran live in **both** modalities with partial natural blanks —
  microphone 4→10 in 6 steps, typed 3→10 in 7 and 2→10 in 8 — so the new
  code path is exercised, not bypassed by long-tail fixtures (the
  `model_steps == 0` long-tail branch is pinned by the endpoint tests).
- Exactly one forced BOS per turn (frames 59/187/363), each response reached
  agent EOS (95/231/407), zero error events.
- All three responses non-empty and correct: "I am listening." / "Two plus
  three equals five." / "The codeword is sapphire." — the empty-response
  failure that killed the first browser gate is gone at its point of origin.
- The sapphire seed was **spoken** (mic turn transcript: "remember that the
  code word…", acknowledged without repeating the secret, per the seed rule)
  and recalled **typed** — real cross-modality session memory through the
  actual Playground/Pipecat/Smart Turn/WebRTC path, which is the half of
  Step 1 no direct harness can cover.

### Six-cell memory matrix (`/tmp/voicechat-memory-matrix-20260808-r4`)

- 6/6 cases green across `{voice,text,system} × {voice,text}`, each in a
  verifiably fresh session (distinct session IDs; `fresh_session_per_case`
  asserted and recomputed by the gate).
- Recall prompts verifiably omit the secret
  (`recall_prompt_contains_secret: false` per case); system cases carry it in
  instructions only; typed turns use the typed_pocket path with byte-exact
  transcripts (WER thresholds correctly scoped per modality — null for typed).
- Independent audio verification: the DGX Parakeet post-pass classified
  10/10 response WAVs as speech (0 near-silent), all within the 0.5 WER
  bound against the model text channel, and every recall **audio** transcript
  independently contains "sapphire". This closes the text-without-audio
  false-positive class at the evidence level.
- Word-boundary token matching (`normalize()` + membership) — no substring
  false-positive exposure.

### Wiring

`run_live_suite.py` now runs the matrix as a permanent ladder stage with
model-idle barriers, and `transcribe_sustained.py::runtime_gate_passed`
dispatches to a matrix-specific recompute so evaluator re-runs are
idempotent — with a regression test flipping a case to prove the gate
recomputes rather than trusts. The qualification surface expansion is
integrated, not bolted on.

## Findings

### Q1 (required before Step 2C weight requalification; recommended now).
The matrix uses one shared codeword and has no no-seed negative control

`memory_matrix_strict_v3.py` hardcodes `SECRET = "sapphire"` for all six
cases and runs no recall-without-seed session. For the current frozen,
published checkpoint the residual risk is low — fresh sessions are verified,
the matcher is word-boundary and independently ASR-confirmed, and the frozen
weights cannot have absorbed the fixture corpus. But both weaknesses become
material the moment the matrix qualifies *new* weights: Step 2C explicitly
contemplates adding direct-text rows to the Nano calibration corpus, and a
recalibrated model that has seen sapphire-adjacent rows could pass recall
without session memory, while a session-isolation regression would be
invisible with a shared secret.

**Accepted placement:** per-case secret parameterization plus an unseeded
negative-control session are recorded as a **Step 2A prerequisite** — they
must land before the matrix is used to qualify any direct-text or
recalibrated artifact. Step 1 remains frozen as qualified; this review's
approval is unaffected.

### Q2 — withdrawn (was: plan missing the transcript accumulator)

The current revision-5 text already covers it: 1C's transcript bullet states
that, while a client turn remains open, phrases across the pipeline's
ten-blank predictor resets are promoted into a monotonic turn-local
transcript, with legacy RNNT mode's per-hypothesis replacement preserved so
text never accumulates across legacy turns — exactly the implemented and
regression-tested semantics. Confirmed against the file at review close; no
action required.

## Verdict

Step 1 is complete against plan revision 5. The implementation survived three
adversarial rounds; every blocker and important finding raised across them
was remediated and re-verified in source; the two live failure modes the
gates surfaced (prefill mode reset, BOS settlement timing) were root-caused
against the pinned NVIDIA sources, fixed at the right seams, amended into the
plan with their evidence, and regression-tested; and the final live artifacts
— browser path and six-cell matrix with independent ASR — check out against
their raw event streams. Q1 is accepted with its remediation placed as a
Step 2A prerequisite ahead of any direct-text or recalibrated-artifact
qualification; Q2 was verified already addressed in the plan and is
withdrawn. Nothing remains open against Step 1.

VERDICT: APPROVE
