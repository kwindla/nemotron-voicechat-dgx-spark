# Codex final adversarial review — Smart Turn Step 1

Date: 2026-08-08  
Scope: plan revision 5 and the current Step 1 implementation, tests,
qualification harnesses, NVIDIA Speech patch, and retained live artifacts.

## Result

No blocker or important finding remains for Step 1.

The implementation assigns microphone start to Pipecat VAD and stop to the
pinned local Smart Turn analyzer, preserves ordered audio/start/commit delivery
through bounded queues, and gives the model server one explicit, idempotent
turn state machine. Typed-via-Pocket input uses the same settlement and EOU
contract. Cancellation, function epochs, typed/microphone conflicts, barrier
acknowledgements, and response correlation all have bounded terminal paths.

The two live failures found during review were resolved at their source:

- model prefill reset external-EOU mode, so the mode is now enabled after
  prefill and covered by a regression;
- early forced BOS could produce an empty response when the model had observed
  too little silence, so commit now advances only the missing positions to the
  wrapper's pinned ten-blank phrase fence before exactly one explicit EOU
  position. It does not add wall-clock pacing or a second endpoint detector.

The RNNT phrase-reset transcript loss is fixed with a turn-local monotonic
segment accumulator that is active only in external-EOU mode. Legacy RNNT mode
retains its original replacement semantics and has a regression against
cross-turn transcript accumulation.

## Evidence checked

- Host suite: 345 passed, 3 skipped, 4 upstream/runtime warnings, 3 subtests.
- Ruff and `git diff --check`: pass.
- The retained NVIDIA Speech patch byte-matches the reviewed dirty source
  checkout for all three patched files.
- Browser/SmallWebRTC/Pipecat gate: 1 passed in 45.00 seconds. Session
  `6ba3d357-e36c-4c7a-a3db-07a4549aaf0b` settled microphone blanks 4→10 in six
  steps, emitted one forced BOS, answered typed math with five, and recalled
  the voice-supplied codeword as sapphire through typed input. All three turns
  reached one EOS with no error event.
- Fresh-session memory matrix: 6/6 across
  `{voice, typed-via-Pocket, system prompt} × {voice, typed-via-Pocket}`. Recall
  prompts omit the secret; only assistant recall output is used for the
  semantic assertion.
- Independent DGX Parakeet pass: 10/10 response WAVs are non-silent speech,
  all meet the 0.5 text-channel WER bound, and all six recall WAVs independently
  contain sapphire.

## Residual follow-up

Before the matrix is used to qualify direct-text calibration or changed
weights, Step 2A should parameterize the secret per case and add an unseeded
negative-control session. The frozen published checkpoint and fresh-session
Step 1 evidence are not blocked by that future-hardening item.

VERDICT: APPROVE
