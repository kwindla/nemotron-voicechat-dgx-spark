# EarTTS alignment r1 evidence

This directory records the safe preflight for the alignment/state-hygiene
brief. The live portion is intentionally absent: production had to remain up,
and available GB10 unified memory was insufficient to start a second full
model without risking that workload.

## Architecture evidence

The inspected files are from production image
`sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`:

- vLLM `eartts.py` SHA-256:
  `e86ff2197e5b086ae1c6b3e91c1df778ae6ade8a8386a3e38f84495cfd3e9f3a`
- vLLM `gemma3.py` SHA-256:
  `356e772db636d0f95b6492069b6b24a82069bd4d2450931497b1a7d558e263c0`
- EarTTS artifact: `EarTTSForCausalLM`, hidden size 1152, 28 layers,
  16 attention heads and 16 KV heads; sole custom output
  `acoustic_tokens`.
- `EarTTSModel.forward` computes the fused transformer's `hidden_states`,
  samples `codes`, and returns only those two values. `Gemma3Attention.forward`
  computes Q/K/V and calls the selected vLLM attention backend, but returns
  only the projected attention output.

The existing same-position contrast is sourced from
`../eartts-text-feed-r1/analysis.json`; the pinned-ASR method and acoustic
boundary evidence are in `../eartts-diagnostic-r1/`.

## Pending maintenance-window protocol

1. Preserve the exact production container/image identity and stop only that
   container after confirming no active client.
2. Run the existing replay workload in an isolated container/port with the
   frozen prepared-reuse configuration for the baseline count.
3. Repeat with `--disable-prepared-epoch`, which sets prepared epoch to zero
   while leaving reset-on-BOS at one.
4. Retain reset attestations and per-BOS `abort_call_ms`, `prefill_call_ms`, and
   `reset_total_ms`.
5. Transcribe all response WAVs with the same pinned offline ASR evaluator.
6. Report raw mismatch counts for each campaign, not a rate inferred from one
   campaign, and state the first-response counterexample.
7. Remove the diagnostic container, restart the exact production container,
   and verify image identity, restart count, readiness, and contract.

Estimated window: 45--60 minutes. It has not been authorized or started.
