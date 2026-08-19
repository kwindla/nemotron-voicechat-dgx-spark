# EarTTS quantization A/B r1 evidence

Status: offline preflight complete; live A/B not started because production
must remain up.

`preflight.json` records the sampling contract, exhaustive retained-field
audit, quantized/FP32 artifact provenance, static sampler-exclusion cost, and
maintenance-window design.

The primary review is
`docs/reviews/eartts-quantization-ab-r1.md`.

## Pending live protocol

For each of two fresh-start campaign pairs:

1. Confirm no active production client, record the exact production
   container/image, and stop only that container.
2. Launch an isolated released-W8A32 diagnostic container and run 10 attempts
   of source turns 1, 2, and 3.
3. Remove it, launch an isolated FP32 diagnostic container with
   `--eartts-path ~/.cache/nemotron-voicechat/conversion/fp32/eartts-vllm-fp32`,
   and run the identical 10 attempts.
4. Transcribe all 60 WAVs with the same pinned offline ASR evaluator and retain
   generated/ASR text, suffix classification, startup provenance, and call
   counts.
5. Repeat from fresh model starts for replication.
6. Report counts per arm and per campaign. Do not infer a rate from either
   single campaign.
7. Remove diagnostics, restart the exact production container, and verify
   image, restart count, health, contract, and client availability.

Estimated maintenance window: 90--120 minutes.
