# attnW8-GPTQ campaign terminal result

Recorded UTC: `2026-08-17T07:05:58Z`

## Verdict

**FAIL-CLOSED AT A0; CAMPAIGN STOPPED BEFORE SEAL-1.**

The active Seal-0 revision was r40:

- Seal-0 SHA-256: `780bdac05690dc02e8657f9770d2862a466120b1ddb15678d71b288f42dc0c37`
- Seal-0 payload SHA-256: `1a9d11505a1423e68f8de11e7312cfc3459f57ce137d32b5ebb1d49fed66e1ef`
- Seal-0 manifest SHA-256: `2610408be50ef266f058d0bf187b4067afeda5a74a494408f2a732165757104a`
- Seal-1: **not created**

The r40 pooled live fhw8 baseline completed over the entire sealed pool. All
48 candidates validated: 24/24 positive-intent and 24/24 no-call-intent. All
tool outcomes were exact. The minimum normalized transcript similarity was
`0.8888888888888888`, and every verbatim transcript is retained in its
scenario event log. The deterministic selected set was the first 12 validated
positives and first 12 validated no-calls in sealed order.

A0 reconstructed fhw8 eagerly, completed the preflight propagation and hook
path, propagated the selected/index corpora, and stopped during exact eager
tool-window classification. Sealed no-call scenario
`a2-nocall-r25-01` contained one or more eager function tokens, raising:

`RuntimeError: a2-nocall-r25-01: eager unexpected function token`

The same audio's packed-runtime pooled baseline was an exact no-call with zero
injections, no errors, no fatal event, exact transcript, and audio SHA-256
`5de8732c8d286fc907864393abee00462e3f6a051a03c1ee0f9cf3ec28238279`.
The eager categorical disagreement is therefore the first true fail-closed
boundary. No A0 PASS report is eligible for Seal-1.

## Gate disposition

- Seal-0: **PASS** (r40)
- Pooled baseline: **PASS**, 48/48 candidates validated; deterministic 12/12
  selection sufficient
- A0 eager/preflight/index stage: **FAIL** on unexpected eager function token
  in a sealed no-call scenario
- Seal-1: **NOT RUN / NOT CREATED**
- Sequential GPTQ build and double-build proof: **NOT RUN**
- C0 construction/numeric gate: **NOT RUN**
- C1 live tool gate: **NOT RUN**
- C2 dual replay: **NOT RUN**
- C3 latency: **NOT RUN**
- C4 battery: **NOT RUN**

## Protocol deviations requiring adversarial-review adjudication

The prospective criterion deviations in
`reports/attnw8-gptq/seal-criterion-deviation.md` remain explicitly subject to
results adversarial review:

1. punctuation/hyphen/explicit-compound normalization;
2. the sealed 48-member pool and first-12-per-intent selection rule; and
3. normalized word-sequence similarity threshold `>= 0.8` with exact audio
   SHA-256 and exact tool outcomes.

All prior invalidations stand. No failed or superseded evidence was
resurrected. No Hessian, quantized candidate, or qualification result was
produced.

## Final stack state

The `fence-r2` service was restored after the A0 stop and attested:

- health/model: `ready` / `ready`
- active client: `false`
- Nano fhw8 SHA-256:
  `dd4ce6a99d03847c8da10708a0214b1815df078a5c43cae9846b7fe136085ce6`
- NVIDIA driver: `580.142`
- kernel: `6.17.0-1014-nvidia`

No commit was created.
