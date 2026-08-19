# EarTTS codec-silence offline diagnostic r1

## Verdict

The production CPU codec is not turning otherwise audible EarTTS codes into
silence. Across all four ASR-confirmed captured suffix losses, the same raw code
sequence remained silent under the exact production codec, pointwise-linear
disabled, persistence disabled, and the native RVQVAE convolution reference.
No configuration produced a single frame above -90 dBFS at or after the live
silence onset. Across every response position, the four required variants were
within `1.5e-5` dB of one another.

The retained non-canonical raw codes therefore encode silence for this codec;
they are not audible codes corrupted by the CPU optimizations. The next fix
belongs in EarTTS generation or its conditioning. Disabling pointwise linear,
disabling worker persistence, or switching to native convolutions is not a
production fix for these losses.

## Faithfulness gate

The offline harness loaded the exact production checkpoint
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`
from the running image's combined-checkpoint layout. The production variant
used CPU float32, all 18 sliced-depthwise replacements, all 38 pointwise-linear
replacements, and one continuous `CausalConv1dCache`.

For each case, the harness replayed the captured decoder-code history from
session frame 0 through the response end and compared every decoded sample to
the retained live `codec_pcm`. All four cases had:

- maximum absolute sample error: `0.0`;
- maximum per-position dBFS error: `0.0`;
- mean per-position mean absolute error: `0.0`.

Only after that exact reproduction passed did the raw-code experiment replace
the target response's decoder codes with its captured raw generated codes. The
preceding session history stayed intact, so the causal decoder entered each
response with the same history as production. In the fully extended case this
also prevents the later canonical PAD-tail substitutions beginning at frame 803
from obscuring what the raw model output means.

## Captures and results

The cases are exactly the four suffix mismatches accepted in the r1 diagnostic:

| Capture | Frames | Live final-silence onset | ASR-confirmed missing suffix |
|---|---:|---:|---|
| regular attempt 1, response 1 | 55-114 | 83 | 6 words: “How can I help you today?” |
| regular attempt 1, response 3 | 448-697 | 666 | 14 words beginning “From then on…” through “The end.” |
| regular attempt 3, response 3 | 484-813 | 782 | 2 words: “The End.” |
| fully extended attempt 1, response 3 | 485-823 | 722 | 2 words: “The end.” |

At each onset, all required decoders were already below -90 dBFS:

| Capture | Production | Pointwise off | Persistence off | Native reference | Audible frames after onset, each |
|---|---:|---:|---:|---:|---:|
| regular A1/R1 | -95.710803 | -95.710803 | -95.710803 | -95.710803 | 0 |
| regular A1/R3 | -125.826356 | -125.826354 | -125.826356 | -125.826356 | 0 |
| regular A3/R3 | -148.820437 | -148.820440 | -148.820437 | -148.820434 | 0 |
| extended A1/R3 | -146.742730 | -146.742728 | -146.742730 | -146.742736 | 0 |

The consolidated `matrix.jsonl` in each case contains every raw code vector,
its original shape, raw/post-policy/decoder equality flags, captured dBFS, and
the per-position dBFS from every variant. No required-variant spread exceeded
0.1 dB at any position; the actual maximum was `1.450844817441066e-05` dB.

## Variant semantics

- **Production:** exact frozen CPU process implementation and continuous causal
  cache; it is the sample-exact faithfulness control.
- **Pointwise off:** retains sliced depthwise convolutions but uses native 1x1
  `Conv1d` instead of the pointwise-linear substitution.
- **Persistence off:** uses the production codec math in a fresh worker process.
  In the implementation, `EA_CPU_CODEC_PERSISTENT=0` changes worker lifetime at
  session finalization; it does not discard causal state between decode calls
  inside a session. The full-session replay is therefore the correct toggle
  counterfactual.
- **Native reference:** uses the image's RVQVAE implementation with native
  `Conv1d` for both depthwise and pointwise layers. It ran CPU float32 without
  GPU access, which removes both CPU optimization substitutions while honoring
  the no-GPU-contention constraint.
- **Fresh-response-cache control:** for the fully extended loss only, an extra
  run discarded all prior-response causal state at frame 485. It still had zero
  audible frames after frame 722 and matched production there to the displayed
  precision. This independently rules out drift inherited from earlier turns.

No variant recovered audio, so there was no recovered suffix to submit to ASR.
The emitted WAVs retain the same acoustic loss as the live captures.

## Evidence and reproduction

The offline decoder is
`tools/qualification/eartts_codec_silence_offline.py`; the evidence consolidator
is `tools/qualification/summarize_eartts_codec_silence_offline.py`. Evidence is
under `reports/response-truncation/codec-silence-offline-r1/`:

- each variant directory has `summary.json`, `positions.jsonl`, and
  `raw-response.wav`;
- each production directory also has `actual-decoder-response.wav`;
- each case has `matrix-summary.json` and `matrix.jsonl` for direct cross-variant
  inspection.

All decoder jobs used the exact production image
`sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`,
`--network none`, no GPU device, private IPC, and only host CPUs 16-19. The live
codec remains on CPUs 5-6.

## Production safety

The production container was never stopped, restarted, reconfigured, or used
for decoding. At the final check, `nemotron-voicechat-model` was still container
`042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`,
running the same image with restart count 0 and a ready health response on
127.0.0.1:8786. `config/production-candidate-1.toml` has no diff. No production
change and no commit were made.
