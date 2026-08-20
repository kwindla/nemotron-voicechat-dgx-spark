# Full-precision checkpoint A/B r2

Date: 2026-08-20

Status: **complete. FP32 Nano plus FP32 EarTTS truncated in 17 of 20 story
responses. Project-added W8A32 quantization is exonerated as the cause.**

## Result

The corrected comparison used the retained quantized campaign envelope rather
than requiring two stochastic campaigns to have equal incidence:

| Configuration/campaign | Suffix losses | Rate |
| --- | ---: | ---: |
| EarTTS quantization r1 pair 1, fully quantized arm | 5/10 | 50% |
| EarTTS quantization r1 pair 2, fully quantized arm | 2/10 | 20% |
| Full-precision r1 quantized wall control | 8/10 | 80% |
| Full-precision r1 quantized lock-step control | 5/10 | 50% |
| **FP32 Nano + FP32 EarTTS, lock-step** | **17/20** | **85%** |

The quantized evidence pools to 20/40 (50%), with individual campaign rates
from 20% to 80%. The FP32 result is not a clean zero or a reduction below that
range; it is one response above its upper edge. This exonerates project-added
quantization as a necessary cause of truncation. It does not establish that
FP32 increases loss: the acoustic generator is stochastic, campaign sample
sizes differ, and pacing remains an uncertainty.

The retained normalized-word classifier requires a missing suffix of at least
three words, matched-prefix coverage of 0.70, and a two-word ASR endpoint
tolerance. `analysis.json` retains all 20 generated texts, ASR transcripts,
audio hashes, missing suffixes, and classifications. Generated text was
identical in all 20 attempts. The three non-loss classifications were attempts
5, 11, and 19; the other 17 omitted between 14 and 57 classified suffix words.

All 20 sessions completed normally. Every response terminated by
`decoded_silence_watchdog`; there were no response-wall, no-text, no-audio, or
session-failure terminations. Lock-step delivery recorded 7,554 processed-event
acknowledgments.

## FP32 timing

Across 5,394 model frames, uncaptured FP32 step time was:

| Mean | Median | p95 | Maximum | Realtime frame budget |
| ---: | ---: | ---: | ---: | ---: |
| 130.536 ms | 132.703 ms | 139.508 ms | 161.953 ms | 80 ms |

Acoustic tensor capture was deliberately disabled for this arm because the r1
capture increased timing tails. The diagnostic timing overrides were instead
gated by the doubly attested full-precision A/B identity. This preserves the
production computation path while allowing exact event acknowledgments and a
900-second whole-response deadline. The measured p95 is 1.74 times the 80 ms
realtime budget and bounds future FP32 experiments: wall-paced replay would
overrun the model, so closed-loop pacing is required.

## Experimental contract

- FP32 Nano: 37,901,759,424 bytes, SHA-256
  `1b473452103362817e9408cf864aa8ef8ca92a48ef8fd94cb3d40fa7cf14e44b`.
- FP32 EarTTS: 3,337,214,236 bytes, SHA-256
  `1cb6f131fc92df2b36fd5888ebe9df3c49e9a7e422f356dc924cce801d6e53ee`.
- Both derive from public checkpoint SHA-256
  `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`.
- Source was retained story turn 3 with PCM SHA-256
  `e0cfde068241c098d1bc615202c11ecf037134cd1ba7fe889fb3d5eebd57395a`.
- Production guards were decoded silence 20, no text 30, and no audio 30.
- Nano prompt/runtime, codec, prepared-epoch policy, and all non-artifact
  settings remained at production values.
- Pinned offline ASR ran with no network using immutable evaluator image
  `sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d`.

Because loss remains abundant with both components in FP32, neither a
Nano-only nor EarTTS-only precision swap is needed to separate a precision
effect. The prior EarTTS-only r1 result independently found that FP32 EarTTS
did not remove loss while Nano remained FHW8. Future work belongs in the
generation/conditioning mechanism, not a precision-component split.

## Audit of the former implicit 100-frame capture default

The exact affected campaigns are limited to the two live acoustic-capture
campaigns in `eartts-diagnostic-r1`:

| Earlier campaign | Effective decoded/no-text/no-audio guards | Effect |
| --- | --- | --- |
| `control-replay-live` | 20/30/30 | Production guards; unaffected. |
| `captured-replay-live` | **100**/30/30 | Non-production decoded-silence guard; all nine responses ended at a 30-frame progress guard. |
| `captured-extended-replay-live` | **100/100/100** | All three guards non-production; eight responses reached 100 silent frames and one reached the 30-second wall deadline. |

The container inspect receipts and per-frame events independently show these
values. Failed pre-window startup containers are not completed campaigns and
are excluded. Every valid arm in `eartts-quantization-ab-r1`, both quantized
pacing controls in `full-precision-ab-r1`, and this FP32 arm report 20/30/30 in
their retained model events.

The 100-frame default does **not** invalidate the mechanistic conclusions, but
it limits their scope:

- The r1 truncation review's claim that audio did not resume after production
  frame 20 is explicitly a guard counterfactual; extending the guard was what
  made frames 21--100 observable. Its production-guard control reproduced the
  substantive suffix loss first. Its response counts, timing, and termination
  mix are not production incidence evidence, as that review already states.
- The text-feed and alignment reviews reuse the captured positions to show
  that full text arrived before acoustic silence and that no explicit pointer
  exists. The extended guard exposes later PAD-state history but does not alter
  the retained full-delivery or silence-onset facts. Their mechanism verdicts
  remain valid; their captured durations cannot be treated as production
  timing or rates.
- The codec-silence offline review replays the same captured code histories.
  Its sample-exact production-codec control and decoder counterfactual remain
  valid. It is not an incidence estimate.
- The EarTTS quantization A/B and the r1/r2 full-precision comparisons did not
  use the implicit 100-frame override, so their suffix-loss counts are
  unaffected.

Thus no earlier quantization conclusion quietly rests on a non-production
guard. The reviews that do reuse the 100-frame captures are mechanism and
counterfactual reviews; they must not be reinterpreted as production-rate or
latency measurements.

## Integrity and operation

Evidence is under `reports/response-truncation/full-precision-ab-r2/`.
`analysis.json` is the machine-readable result and `asr-execution.json` is the
ASR provenance receipt. The exact production container
`042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
was never removed and was restarted from image
`sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
No frozen configuration or released artifact was modified, and no commit was
created.
