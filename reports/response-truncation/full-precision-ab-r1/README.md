# Full-precision A/B r1 evidence

Superseded by `../full-precision-ab-r2/`. The r1 equality gate was
mis-specified; r2 retains these controls and interprets them against the pooled
quantized envelope.

Status: stopped at the mandatory harness-control gate. Quantized wall pacing
had 8 suffix losses in 10 story responses; quantized lock-step had 5 in 10.
The full-precision arm was not run.

- `control-analysis.json` contains all classifications and step-time summaries.
- `control-asr-execution.json` contains pinned offline ASR provenance for all
  20 control WAVs.
- `live/quantized-wall/` contains the 10-response wall-paced campaign.
- `live/quantized-lock-step/` contains the 10-response lock-step campaign and
  3,687 processed-event acknowledgments.

Both controls used source PCM SHA-256
`e0cfde068241c098d1bc615202c11ecf037134cd1ba7fe889fb3d5eebd57395a` and
production frame guards 20/30/30. The exact production container was preserved,
restarted after the control failed, and verified healthy.

The primary review is `docs/reviews/full-precision-ab-r1.md`.
