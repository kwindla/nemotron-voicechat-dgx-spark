# EarTTS quantization A/B r1 evidence

Status: complete. The two-pair live A/B ran during the authorized maintenance
window on 2026-08-19/20. The result is inconclusive and cannot be treated as a
response-level paired comparison because generated text diverged between arms.

`preflight.json` records the sampling contract, exhaustive retained-field
audit, quantized/FP32 artifact provenance, static sampler-exclusion cost, and
maintenance-window design.

The primary review is
`docs/reviews/eartts-quantization-ab-r1.md`.

## Live result

Each fresh-start arm completed 10 attempts with three responses per attempt.
The pinned offline ASR evaluator classified generated-text/ASR suffix losses as
follows; the campaigns are intentionally not averaged:

| Campaign | W8A32 losses / 30 | FP32 losses / 30 | W8A32 long-response losses / 10 | FP32 long-response losses / 10 |
| --- | ---: | ---: | ---: | ---: |
| pair 1 | 8 | 8 | 5 | 5 |
| pair 2 | 3 | 15 | 2 | 9 |

Only 19/30 generated texts matched between arms in pair 1 and 12/30 in pair 2.
Pair 1 tied while pair 2 favored W8A32, so direction did not replicate. FP32
EarTTS also retained clear losses in both campaigns while Nano remained FHW8.
This rules out EarTTS W8A32 as a necessary cause of truncation but neither
exonerates whole-stack quantization nor identifies a precision effect on
incidence.

- `analysis.json` retains every generated/ASR comparison and classification.
- `asr-execution.json` retains the immutable evaluator image and model
  provenance for all 120 WAVs.
- `live/` retains all valid campaigns and the two discarded preflights.
- `preflight.json` retains the sampling and artifact contracts.

The exact production container was restarted after ASR and verified ready.
The user-stopped Pipecat, ngrok, and clips-server processes were left for the
user to restore as requested.
