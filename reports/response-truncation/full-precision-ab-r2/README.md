# Full-precision A/B r2 evidence

Status: complete. FP32 Nano plus FP32 EarTTS produced 17 suffix losses in 20
story responses. Full precision did not remove truncation, so project-added
W8A32 quantization is exonerated as its cause.

- `analysis.json` retains every generated text, pinned-ASR transcript, missing
  suffix classification, audio hash, termination reason, and model-step
  distribution.
- `asr-execution.json` attests the network-disabled pinned ASR execution.
- `live/full-precision-lock-step/replay/` contains 20 complete fresh sessions,
  their WAVs, transcripts, event streams, and campaign receipt.
- `live/full-precision-lock-step/server-traces/diagnostic-vllm-manifest.json`
  attests the FP32 Nano and EarTTS artifacts.

The arm used source PCM SHA-256
`e0cfde068241c098d1bc615202c11ecf037134cd1ba7fe889fb3d5eebd57395a`,
lock-step pacing, and production frame guards 20/30/30. The primary review is
`docs/reviews/full-precision-ab-r2.md`.
