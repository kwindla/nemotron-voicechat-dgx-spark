# EarTTS response truncation: characterization

**Status: characterized, not fixed.** Long spoken responses lose their final
sentences. The cause is in EarTTS acoustic generation, upstream of everything
this project controls, and it reproduces on the stock NVIDIA checkpoint at full
precision. This document records what was measured, what was eliminated, and
what would be needed to go further.

## Symptom

The model generates a complete text response, delivers all of it to EarTTS, and
then stops producing audible audio partway through rendering it. The listener
hears the response end mid-thought. There is no error, no dropped connection,
and no crackle; the audio is clean right up to the point where it stops.

Two responses in a live session on 2026-08-19 lost their tails:

| response | spoken / generated | never spoken |
|---|---:|---|
| 2 | 22 / 41 words (54%) | "I can help with anything from answering questions to generating creative content. What would you like to talk about?" |
| 3 | 68 / 86 words (79%) | "From then on the forest was filled with laughter and magic, and everyone lived in peace. The end." |

Losses are clean tail truncations at a sentence boundary, never gaps in the
middle of a response.

## Incidence

Long responses are where the risk concentrates. Short responses almost always
complete. Measured on the retained story turn, 10 responses per campaign,
production frame guards (decoded silence 20, no text 30, no audio 30):

| configuration | suffix losses | rate |
|---|---:|---:|
| released stack (quantized), campaign A | 5/10 | 50% |
| released stack (quantized), campaign B | 2/10 | 20% |
| released stack (quantized), campaign C | 8/10 | 80% |
| released stack (quantized), campaign D | 5/10 | 50% |
| **pooled released stack** | **20/40** | **50%** |
| **FP32 Nano + FP32 EarTTS** | **17/20** | **85%** |

A separate browser-driven baseline over mixed conversational turns saw 1 loss
in 7 completed responses, and 1 in 3 long responses.

The 20% to 80% spread across identical configurations is not measurement noise
in the classifier; it is the model. Acoustic sampling is stochastic (below), so
no single campaign establishes a rate, and these are reported as counts.

## What was eliminated

Each layer below EarTTS generation was ruled out by direct measurement rather
than by argument.

**WebRTC transport.** The output track's chunk queue never ran dry while a
speech stream was live: zero in-speech underflow ticks across two 95-second
replays, with a clean 1/2/3/4 depth sawtooth. A negative control that withheld
audio for 200 ms every 20 writes produced 26 runs of 19-20 ticks against 26
injections, proving the counter detects the fault it reports as absent.

**Output chunking.** An apparent advantage for `audio_out_10ms_chunks=4` over 1
was withdrawn: the arms had unequal model output and the traces predicted a
larger difference than was observed, before any chunk-size effect.

**The decoded-silence watchdog.** It fires 20 frames (1.6 s) after audio has
already stopped, and it is not withholding recoverable audio. With the
threshold raised to 100 frames, zero of 17 responses that crossed frame 20 ever
produced audio above -90 dBFS again. Raising it recovers nothing.

**The PAD-tail policy.** `tail_done` substitution begins 81 frames *after* the
PCM has already gone silent. It is a consequence, not a cause.

**The CPU codec.** The offline decoder reproduces live production PCM with a
maximum absolute sample error of 0.0. On that gate, the same raw codes were
decoded with pointwise-linear disabled, persistence disabled, and the native
RVQVAE reference: all variants agree within 1.45e-5 dB and all are silent. A
fresh-cache control rules out drift inherited across turns.

**The text feed.** EarTTS received every response in full. The production
tokenizer requires 16, 83, 98 and 79 content tokens for the four analyzed
losses; exactly 16, 83, 98 and 79 were delivered, and decoding the captured
sequences reconstructs the generated text ID-for-ID. The missing suffixes were
delivered too, at identifiable frames -- for example "How can I help you
today?" arrived at frames 65-71, PAD began at 72, and silence began at 83.

**Quantization.** The project quantizes both components (Nano `fhw8`, EarTTS
`ea_w8a32`, the latter including the `mog_head` sampler MLP). Running the stock
public checkpoint at full precision for *both* components truncated 17 of 20
story responses, above the top of the released stack's envelope. Project-added
quantization is exonerated as a necessary cause.

## The mechanism, as far as it is observable

At the silence transition the raw acoustic codes sampled by EarTTS are *not*
the canonical codec-silence vector -- zero raw matches across 2,572 positions --
yet they decode to digital silence (-95 to -148 dBFS). Raw, post-policy and
decoder codes are identical at that position, `tail_done` is false, and no
policy has been applied. In other words, EarTTS is emitting varied,
non-canonical codes that the codec faithfully renders as silence, while text it
has already received remains unspoken.

Acoustic sampling is stochastic despite the surrounding vLLM `temperature=0`:
`MoGHead` selects a mixture component by Gumbel-Max, `MaskGITSampler` adds
Gaussian noise at `noise_scale=0.001`, and this repeats for eight MaskGIT
iterations per acoustic position. The engine uses `skip_sampling=True`, so
vLLM's per-request generator is not involved; draws come from the worker-wide
Torch generator with no per-response seed. This is why the same prompt
truncates only sometimes, and why the incidence spread above is expected rather
than anomalous.

Once silent, the state does not recover. Silence onset also occurs earlier in
truncated responses than in complete ones (46.7%, 69.9%, 87.2% and 90.3% of the
way through, versus 97.5% for an ASR-complete control).

## Why there is no runtime mitigation

EarTTS exposes no rendering-debt signal. There is no monotonic text cursor, no
finished flag, no end-of-text probability, and no outstanding-text output. At
each causal position it adds the current text embedding to an embedding of the
preceding acoustic codes, runs the fused embedding through a 28-layer Gemma3
decoder, and samples the next 31 acoustic codes. Remaining text survives only
implicitly in hidden and KV state, and the production Triton attention backend
returns only the attention result, not attention weights.

A detect-and-recover mitigation therefore has nothing to trigger on. Detection
would require either a model change that exposes rendering progress, or an
out-of-band check such as running ASR against the emitted audio, which is not
viable in the realtime path.

Full precision is also not an available fallback for a different reason: its
measured step time is a mean of 130.5 ms and p95 of 139.5 ms against an 80 ms
frame budget, or 1.74x realtime. The released quantized stack runs comfortably
inside budget at a mean of 72.8 ms and p95 of 75.1 ms.

## Untested lever

The one remaining parameter under our control is the sampling path itself:
`noise_scale`, the Gumbel-Max component selection, and the eight MaskGIT
iterations per position. These are generation parameters rather than weights.
Reducing stochasticity may reduce the chance of the first mis-step into the
silence regime, but noise may be load-bearing for audio quality, so any such
change needs an audio-quality check alongside an incidence measurement.

## Reproduction

    # incidence over a browser-driven session, against a running bot
    python3 reports/response-truncation/truncation_rate.py --sessions 3

    # incidence over the retained story turn, against a diagnostic container
    tools/qualification/launch_eartts_diagnostic.py    # --eartts-path/--nano-path
    tools/qualification/eartts_truncation_replay.py    # --lockstep for slow models

Loss is confirmed by transcribing emitted audio with the pinned offline
evaluator (`pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b`, network
disabled) and diffing against the generated text. A loss requires a missing
normalized suffix of at least three words, matched-prefix coverage of at least
0.70, and an ASR endpoint within two words of the final aligned block.

Note that a full-precision arm must be driven closed-loop: the model cannot
meet the 80 ms frame budget, and a wall-clock-paced replay would additionally
trip `VOICECHAT_WEB_MAX_AGENT_RESPONSE_SEC`, manufacturing terminations that
mimic or mask truncation.

## Caveats

- Counts, not rates. Every number here is an observed count from a named
  campaign; stochastic sampling means none of them extrapolates to a production
  rate.
- The FP32 result exonerates quantization as a *necessary* cause. It does not
  establish that full precision is worse, since sample sizes differ and pacing
  differs between arms.
- Pacing remains an uncontrolled variable. Wall-paced and lock-step quantized
  campaigns gave 8/10 and 5/10 (Fisher p = 0.35), which is indistinguishable at
  this sample size; both sit inside the pooled envelope, which is what allows
  the FP32 comparison to stand despite the imperfect control.
- Two early acoustic-capture campaigns ran with a non-production decoded-silence
  guard of 100 frames (`captured-replay-live`, `captured-extended-replay-live`).
  Their mechanism and counterfactual conclusions hold; their termination and
  incidence figures are not production-representative and are not used above.
- Acoustic capture measurably increases model step-time tails, so no latency
  conclusion is drawn from captured runs.

## Evidence

Reviews in `docs/reviews/`: `eartts-truncation-diagnostic-r1`,
`eartts-codec-silence-r1`, `eartts-text-feed-r1`, `eartts-alignment-r1`,
`eartts-quantization-ab-r1`, `full-precision-ab-r1`, `full-precision-ab-r2`,
`audio-crackle-root-cause-r1..r3`, `live-session-431ac6df-analysis-r1`.

Data and per-case verdicts in `reports/response-truncation/`, including the
seven clips of the original live session with their transcripts.
