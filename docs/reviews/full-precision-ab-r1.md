# Full-precision checkpoint A/B r1

> Superseded by `full-precision-ab-r2.md`. The equality stop rule below was
> mis-specified; the retained controls are valid bounded-uncertainty evidence,
> and r2 ran the full-precision arm.

Date: 2026-08-20

Status: **stopped at the mandatory harness gate. Quantized wall pacing produced
8 suffix losses in 10 story responses; quantized lock-step produced 5 in 10.
The full-precision arm was not run because it would not have a valid control.**

## Question and correction to r1

The earlier A/B replaced only EarTTS. Nano remained the released FHW8 artifact
in both arms, so that experiment supports only the narrow conclusion that
EarTTS W8A32 is not a necessary cause. It did not answer whether the complete
FP32 NVIDIA checkpoint truncates.

The complete FP32 artifacts were attested without rebuilding:

| Component | Bytes | SHA-256 |
| --- | ---: | --- |
| Nano FP32 | 37,901,759,424 | `1b473452103362817e9408cf864aa8ef8ca92a48ef8fd94cb3d40fa7cf14e44b` |
| EarTTS FP32 | 3,337,214,236 | `1cb6f131fc92df2b36fd5888ebe9df3c49e9a7e422f356dc924cce801d6e53ee` |

Both extraction records derive from public checkpoint SHA-256
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`.
The launcher can now bind and attest both components in an isolated diagnostic
manifest; this path was compiled and unit-gated but was deliberately not
launched after the control failed.

## Pacing implementation

Lock-step mode sends each retained input packet, waits for the server to finish
all work for that exact client event, then sends the next packet. Model-paced
silence uses the same processed-event barrier. Turn-start and commit barriers
are also acknowledged only after their server-side model work completes. The
acknowledgment exists only when `VOICECHAT_DIAGNOSTIC_LOCKSTEP=1` and EarTTS
acoustic capture is active; startup fails closed without capture.

The response wall deadline was extended from 30 to 900 seconds through a
separate capture-gated diagnostic override. The production frame guards were
unchanged in both controls: decoded silence 20 frames, no text 30 frames, and
no audio 30 frames. A launcher correction was required because its former
capture default implicitly changed decoded silence to 100 frames; capture now
changes no guard unless an override is explicitly supplied.

## Mandatory harness control

Both arms used 10 fresh sessions, the retained story turn 3, identical source
PCM SHA-256
`e0cfde068241c098d1bc615202c11ecf037134cd1ba7fe889fb3d5eebd57395a`,
the same quantized Nano/EarTTS artifacts, worker seed, warmup, prompt, codec,
prepared-epoch policy, and frame guards. All 20 responses and sessions
completed with `decoded_silence_watchdog`; lock-step recorded 3,687 processed
event acknowledgments.

Pinned offline ASR classified normalized generated-text/ASR suffix losses using
the retained three-word minimum, 0.70 matched-prefix coverage, and two-word ASR
endpoint tolerance:

| Quantized control | Suffix losses / 10 story responses | Model step mean | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| wall-paced | 8 | 77.305 ms | 76.129 ms | 82.171 ms |
| lock-step | 5 | 77.007 ms | 75.385 ms | 81.855 ms |

The same deterministic classifier caught every response whose generated text
exceeded the ASR result by at least three words. `control-analysis.json`
retains every generated text, ASR transcript, source-audio hash, missing
suffix, termination reason, and step-time distribution.

## Decision

The mandatory control **failed**: lock-step changed observed incidence from
8/10 to 5/10. These campaigns are not averaged and are not treated as
response-level pairs. Per the preregistered stop rule, a full-precision
lock-step observation cannot be compared with the wall-paced baseline, so the
FP32 arm was not run.

This run therefore does not answer whether the full-precision NVIDIA model
truncates. It identifies pacing as a material experimental factor that must be
characterized or controlled before whole-stack quantization can be implicated
or exonerated. No FP32 step-time measurement exists because starting that arm
after the failed gate would have produced a non-interpretable result.

## Operational integrity

- Both diagnostic containers were removed.
- The exact production container
  `042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
  was preserved and restarted on image
  `sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
- Its restart count remained zero and `/health` returned `status=ready`,
  `model=ready`, the production checkpoint/engine identity, and
  `active_client=false`.
- No released artifact or frozen configuration was modified, and no commit was
  created.

Evidence is under `reports/response-truncation/full-precision-ab-r1/`.
