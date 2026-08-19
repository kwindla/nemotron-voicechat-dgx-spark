# EarTTS W8A32 quantization A/B r1

Date: 2026-08-19

Status: **offline preflight complete; end-to-end A/B requires a 90--120 minute
maintenance window. Production was not touched.**

## Priority and scope

Alignment part 1 was close and is now checkpointed in
`eartts-alignment-r1.md`. Its prepared-reuse/state-hygiene experiment was
superseded and will not be run. This review proceeds with the user-prioritized
W8A32 question.

## Sampling contract

EarTTS acoustic sampling is stochastic, despite the surrounding vLLM
`temperature=0` setting:

1. `MoGHead` samples a mixture component with Gumbel-Max.
2. `MaskGITSampler` adds Gaussian noise scaled by `noise_scale=0.001`.
3. The resulting latent is mapped to acoustic codes by nearest-code selection.
4. This repeats for eight MaskGIT iterations per acoustic position.

The EarTTS engine uses `skip_sampling=True`, so vLLM's ordinary token sampler
and its per-request `torch.Generator` do not select the acoustic codes. No seed
is supplied in the EarTTS `SamplingParams`. The model's `torch.randn_like` and
Gumbel draws therefore use the worker-wide Torch generator. vLLM initializes
that generator from the model seed (default 0) at worker startup and resets it
after profiling, after which calls across warmup, requests, and responses
advance one shared random stream. There is no per-response acoustic seed.

This makes sample counts and call-history matching essential. One stochastic
response cannot support a quantization claim.

## Experiment 1: retained identical-state replay

**Not possible from the retained captures, and not approximated.**

All 4,801 retained acoustic frame records were audited. They retain current
text/token state, selected raw/recurrent/decoder acoustic codes, codec PCM, and
policy counters. They do not retain:

- the final 1,152-dimensional backbone hidden state;
- MoG post-MLP state, component logits, means, or log standard deviations;
- the actual Gumbel draws and Gaussian draws, or the worker RNG state.

The hidden state is the input needed to compare the two sampler heads at an
identical causal position. The stochastic draws are required to reproduce the
captured quantized selection exactly. Re-seeding a new worker at zero cannot
recover the captured draw because the retained record does not encode the
preceding worker-wide random-call history, including warmup and other streams.

Accordingly, there is no faithful way to prove an offline quantized replay
matches the captured code before crediting an unquantized difference. Comparing
fresh hidden states or newly sampled draws would answer a different question.

## Experiment 2: end-to-end A/B preparation

A stock FP32 EarTTS vLLM artifact already exists at
`~/.cache/nemotron-voicechat/conversion/fp32/eartts-vllm-fp32`. It was extracted
from the exact 44.4 GB public checkpoint, has 453 tensors and 3,337,214,236
bytes, and records the public checkpoint SHA-256
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`.
It has no quantization configuration or W8A32 sampler flag. The retained
extraction report gives model SHA-256
`1cb6f131fc92df2b36fd5888ebe9df3c49e9a7e422f356dc924cce801d6e53ee`.

The diagnostic launcher now accepts `--eartts-path`. A dry construction check
confirmed that the FP32 directory is mounted read-only at
`/models/eartts-diagnostic`, passed to `--eartts-vllm-path`, and attested by its
config hash. The frozen production configuration remains the common runtime
contract; only the isolated container's EarTTS artifact path differs.

The planned design is two fresh-start matched campaign pairs. Each arm runs 10
replay attempts over the same three retained source turns: 30 responses per arm
per campaign pair, 60 responses per arm after replication. Within each pair:

- quantized and FP32 containers start from the same default worker seed;
- warmup, replay ordering, prompts, source PCM, guards, prepared-epoch policy,
  Nano artifact, and codec path are held fixed;
- all WAVs are transcribed with the pinned offline
  `pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b` image;
- results are reported as generated/ASR suffix-loss counts for each campaign,
  never as a rate inferred from one campaign;
- any difference is interpreted only if generated text and model-call schedule
  remain comparable. Divergent schedules are reported rather than treated as
  paired samples.

The first 30-response pair is exploratory. A quantization implication or
exoneration requires the second fresh-start pair to reproduce the direction;
otherwise r1 remains inconclusive and reports the raw counts.

## What the A/B can and cannot isolate

The released conversion quantizes 205 projections: seven projections in each
of 28 backbone layers plus nine projections in the three-layer sampler MLP.
Therefore a full FP32 pass can implicate or exonerate project-added W8A32 as a
whole, but cannot attribute an effect specifically to the sampler MLP.

If full FP32 removes the loss, a third diagnostic arm should retain the W8A32
backbone while restoring only the sampler MLP to FP32. The static cost is
small: the nine sampler matrices occupy 191,102,976 bytes in FP32 versus
47,900,160 bytes as INT8 plus scales, an increase of 143,202,816 bytes
(136.6 MiB, 0.133 GiB). Runtime latency still requires measurement: sampler
exclusion changes 72 linear calls per acoustic position (eight iterations,
three MLP layers, three projections) from the custom W8A32 path to FP32.

## Operational blocker

The host has one unified 121.7 GiB memory pool. With production resident,
preflight found only about 34 GiB available and prior co-residency attempts
already failed to start a second full model reliably. The FP32 artifact also
adds about 1.79 GiB of stored weights over the released W8A32 artifact before
runtime allocations. Launching either diagnostic arm now would risk the active
production campaign and violate the brief.

A **90--120 minute maintenance window** is required for four sequential model
starts (two quantized/FP32 campaign pairs), 120 response observations, pinned
ASR, cleanup, and exact production restart verification. No window was started.

## Integrity

- Production container remained
  `042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
  on image
  `sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
- No diagnostic container was launched and no released artifact was modified.
- No frozen configuration changed and no commit was created.
- Launcher Python compilation, Ruff, dry command construction, JSON validation,
  and `git diff --check` passed.

Evidence and the exact pending protocol are under
`reports/response-truncation/eartts-quantization-ab-r1/`.
