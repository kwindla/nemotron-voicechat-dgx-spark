# EarTTS alignment and state-hygiene diagnostic r1

Date: 2026-08-19

Status: **alignment part 1 checkpointed. The prepared-reuse experiment was
superseded by the user-prioritized quantization A/B and will not be run.
Production was not touched.**

## Current verdict

EarTTS does not implement the alignment-pointer model assumed by the question.
There is no monotonic text cursor, finished flag, end-of-text probability, or
rendering-debt output in the deployed model. At each causal position it adds
the current text embedding to an embedding of the preceding acoustic codes,
runs the fused embedding through a 28-layer Gemma3 decoder, and samples the
next 31 acoustic codes from the final hidden state. Remaining text survives
only implicitly in the transformer's hidden/KV state.

Consequently, the observed silence transition cannot be classified as a
pointer reaching the end early, stalling, or jumping: no such pointer exists.
The production Triton attention backend returns only the attention result, not
attention weights, and the EarTTS vLLM artifact exposes only
`acoustic_tokens`. A final-hidden-state capture could show a latent-state
change, but it could not turn that state into a calibrated text position or an
outstanding-debt count. Calling it an alignment or debt signal would overstate
what is observable.

The existing per-position capture does establish the external distinction:

| Case | Role | Text tokens ingested | Generated / ASR words | Silence position within response |
|---|---|---:|---:|---:|
| regular-a1-r1 | truncated | 16 | 13 / 7 | 28 / 60 (46.7%) |
| regular-a1-r3 | truncated | 83 | 65 / 51 | 218 / 250 (87.2%) |
| regular-a3-r3 | truncated | 98 | 79 / 77 | 298 / 330 (90.3%) |
| extended-a1-r3 | truncated | 79 | 63 / 61 | 237 / 339 (69.9%) |
| extended-a2-r3 | ASR-complete control | 104 | 86 / 86 | 347 / 356 (97.5%) |

Every case had ingested its complete text sequence before the final silence
transition. Thus text-feed debt was zero in both losses and control, while
rendering debt measured retrospectively by generated/ASR suffix disagreement
was nonzero only in the losses. That retrospective ASR mismatch is not
available as a per-position runtime quantity. The raw acoustic codes also do
not supply it: the prior capture showed noncanonical, changing codes decoding
to silence before any PAD-tail substitution.

This is a negative but operationally important result: the deployed EarTTS
interface has no trustworthy internal debt signal on which to base a runtime
mitigation. A mitigation would need a new trained alignment/debt head, a
separate online speech-text aligner, or an explicitly validated proxy. Hidden
state norms or uncalibrated attention mass are not sufficient by themselves.

The existing capture perturbs timing tails. That is acceptable for the
mechanism evidence above because the conclusions use exact token delivery,
acoustic tensors, decoded PCM, and architecture, not latency or elapsed-time
distributions.

## State-hygiene experiment (superseded)

The hypothesis remains open but is no longer scheduled. Counter-evidence is
explicit: regular attempt 1,
response 1 truncated even though it was the first response of its session, so
prepared-state reuse cannot be the whole cause.

The diagnostic launcher now has `--disable-prepared-epoch`. It overrides only
the isolated diagnostic container to
`VOICECHAT_EARTTS_PREPARED_EPOCH=0` while retaining
`VOICECHAT_EARTTS_RESET_ON_BOS=1`. The resulting BOS path aborts any active
EarTTS request, speaker-prefills a new request, and restores the initial audio
code before generation. A dry construction check confirmed both environment
values, the non-production container name, and port 8796.

The requested same-workload campaign, pinned offline ASR confirmation,
incidence counts, and current reset-cost distribution were not run before the
quantization brief superseded this work.
Historical genuine-reset evidence from a different workload contains 20
unique resets: reset-total median 127.320 ms, p95 141.026 ms, range
79.937--188.967 ms. This is only a window-sizing prior, not the requested cost
result and not an incidence comparison.

## Why a window is required

On preflight the GB10 host had 121.7 GiB unified memory, 34.0 GiB available
(about 6.7 GiB immediately free), with the production model resident and
ready. Prior attempts already showed that a second full diagnostic model could
not coexist reliably. Starting one now would risk memory pressure and disrupt
the user's production measurement campaign, violating the brief.

The reset experiment would have required a **45--60 minute maintenance
window**. No part of that window was started. The active maintenance estimate
now belongs to the quantization A/B review.

## Integrity and operational state

- Production container:
  `042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
- Production image:
  `sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`
- At final preflight: running, restart count 0, `/health` reported ready,
  `active_client=false`.
- No diagnostic container was launched. No frozen configuration was changed.
  No commit was created.
- Launcher validation passed Python compilation, Ruff, and `git diff --check`.

Evidence and the exact pending protocol are under
`reports/response-truncation/eartts-alignment-r1/`.
