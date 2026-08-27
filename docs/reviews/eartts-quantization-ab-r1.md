# EarTTS W8A32 quantization A/B r1

Date: 2026-08-19/20

Status: **complete; EarTTS-only result inconclusive and not response-level
paired because generated text diverged between arms. FP32 EarTTS did not
remove suffix loss; Nano remained FHW8 in both arms.**

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

## Live A/B result

The authorized maintenance window was used for two complete fresh-start pairs.
All four valid arms used the same retained source PCM SHA-256
`e0cfde068241c098d1bc615202c11ecf037134cd1ba7fe889fb3d5eebd57395a`,
source turns 1--3, frozen runtime contract, Nano and codec artifacts, worker
seed 0, prepared-epoch policy, warmup, replay order, two-second inter-attempt
gap, and termination guards. Each arm completed 10 sessions with three
responses, yielding 30 WAVs per arm and 60 per precision.

The pinned offline evaluator ran over all 120 WAVs with network disabled. Its
immutable image identity was
`sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d`;
`asr-execution.json` retains its backend, model snapshot, decoder, device, and
determinism provenance. A suffix loss requires a normalized generated-text
suffix of at least three words, at least 0.70 matched-prefix coverage, and an
ASR endpoint within two words of the final aligned block. Every positive is
retained with its generated text, ASR text, source-audio hash, and missing
suffix in `analysis.json`.

| Fresh-start campaign | W8A32 losses / 30 | FP32 losses / 30 | W8A32 response-3 losses / 10 | FP32 response-3 losses / 10 |
| --- | ---: | ---: | ---: | ---: |
| pair 1 | 8 | 8 | 5 | 5 |
| pair 2 | 3 | 15 | 2 | 9 |

The campaigns are not averaged. Pair 1 tied, while pair 2 had more losses in
FP32, so there is no replicated direction. More importantly, only 19 of 30
generated texts matched between arms in pair 1 and 12 of 30 in pair 2. The
response-level pairing condition therefore failed before any causal
quantization comparison could be credited. The model-call schedule was not
used to rescue pairing once generated text had diverged.

This is an **inconclusive EarTTS-precision A/B**. It establishes only that FP32
EarTTS does not remove the failure while Nano remains FHW8: clear suffix losses
occurred in both FP32-EarTTS campaigns. EarTTS W8A32 is therefore not a
necessary cause of truncation. It does not exonerate quantization across the
whole stack, because Nano's FHW8 outputs and conditioning stream were common to
both arms. The stochastic arms also left different generated-text workloads,
and the two campaign directions did not replicate. The sampler-only third arm
was not run; its trigger condition (FP32 EarTTS removes the loss) was not met.

Two preflights were excluded before any valid comparison: an initial W8A32 run
without an inter-attempt gap hit WebSocket admission cleanup races, and the
first FP32 launch correctly rejected a released-manifest mismatch before model
load. Both are retained under explicitly named `*-invalid-*` directories and
do not contribute to the table.

## Integrity

- The exact production container
  `042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
  was preserved, restarted on image
  `sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
- Its restart count remained zero and `/health` returned `status=ready`,
  `model=ready`, the production checkpoint contract, and `active_client=false`.
- All diagnostic containers were removed and no released artifact was modified.
- No frozen configuration changed and no commit was created.
- Python compilation, Ruff, campaign/ASR/analysis validation, and
  `git diff --check` passed.

Evidence and the complete result are under
`reports/response-truncation/eartts-quantization-ab-r1/`.
