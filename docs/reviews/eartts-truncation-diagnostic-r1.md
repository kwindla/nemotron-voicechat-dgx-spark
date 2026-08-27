# EarTTS truncation diagnostic r1

Date: 2026-08-19

Status: **complete. The loss is at the EarTTS acoustic-code/codec boundary,
not the 20-frame decoded-silence guard.**

## Verdict

The requested three-way split is:

1. **Raw EarTTS selected the canonical codec-silence code: no.** Across the
   fully extended run, none of 2,572 raw generated positions equaled the
   31-code codec-silence vector. The earlier 30-frame capture likewise had
   zero raw matches across 2,229 positions.
2. **Silence appeared downstream while raw codes remained non-silence: yes.**
   In the ASR-confirmed truncated fully extended story, PCM entered its final
   silence at frame 722 even though raw, post-policy, and decoder codes were
   identical, `tail_done=false`, and no policy was applied. The PAD-tail policy
   did not begin replacing codes with the canonical silence vector until frame
   803, after 81 already-silent frames. The immediate fault domain is therefore
   the acoustic-code-to-PCM boundary: varied noncanonical EarTTS codes reached
   the codec unchanged and decoded to silence.
3. **The production 20-frame guard cut off later resumed audio: no.** In the
   fully extended run, eight responses reached 100 continuous silent frames
   and none resumed after frame 20. The ninth hit the independent 30-second
   response wall deadline after only seven silent frames. In the preceding
   capture, all nine responses were observed from frame 20 through frame 30
   with no resume. Thus zero of 17 captured responses that crossed frame 20
   resumed afterward.

The policy is still visible later in many tails and forces the canonical
silence vector once `pads > 3 * content`, but it is downstream of the initial
PCM silence in the diagnostic truncation. Extending the watchdog only prolongs
already-silent PCM; it does not recover the missing generated suffix.

## Exact per-position example

The fully extended attempt 1 story generated 63 normalized words. Pinned ASR
heard 61 and omitted the final “The end.” Its response occupied server frames
485--823.

| Frame | PCM | Text/counters before position | `tail_done` / policy | Code relation | Watchdog |
|---:|---:|---|---|---|---:|
| 721 | last > -90 dBFS PCM | PAD tail | false / false | raw = post-policy = decoder | 0 |
| 722 | -146.743 dBFS | PAD, content=79, pads=157 | false / false | raw = post-policy = decoder | 0 |
| 742 | -146.370 dBFS | PAD, content=79, pads=177 | false / false | raw = post-policy = decoder | 20 |
| 802 | -146.332 dBFS | PAD, content=79, pads=237 | false / false | raw = post-policy = decoder | 80 |
| 803 | -146.747 dBFS | PAD, content=79, pads=238 | true / true | raw != canonical post-policy; post-policy = decoder | 81 |
| 822 | silent | PAD tail | true / true | policy-forced canonical silence | 100 |
| 823 | -146.581 dBFS | EOS | true / false | raw = post-policy = decoder | EOS |

The final silent run is 102 positions. For its first 81 positions, including
the production cutoff point at frame 742, the lower path received the raw code
unchanged. No PCM resumed before the extended threshold.

The canonical silence vector inferred from every policy substitution is
retained in `acoustic-analysis-extended.json`. Every policy-applied decoder
position used exactly that one vector; raw output used it zero times. All 2,572
capture files are contiguous and all contain codec PCM.

## Reproduction and ASR counts

All clips were transcribed using the pinned offline
`pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b` image, GPU access,
host IPC, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `--network none`. Counts below
are descriptive experiment counts, not estimated production rates.

| Campaign | Replay attempts | Responses | Generated/audio suffix mismatches | Termination |
|---|---:|---:|---:|---|
| Uninstrumented, production guards | 1 | 3 | 1 | 3 decoded-silence watchdog |
| Capture, decoded-silence=100, progress guards=30 | 3 | 9 | 3 | 9 no-text watchdog |
| Capture, decoded-silence/no-text/no-audio=100 | 3 | 9 | 1 | 8 no-text watchdog, 1 wall deadline |
| **Total** | **7** | **21** | **5** | grouped above |

The uninstrumented control reproduced the substantive loss first: the story
generated 71 words, while ASR heard only the first 28 and stopped after “out
of reach,” omitting 43 generated words. In the first captured campaign, the
three suffix losses were 6, 14, and 2 words. In the fully extended campaign,
the only mismatch was the two-word “The end” suffix; its exact acoustic tail is
the per-position example above. The other fully extended 86-word story was
complete in ASR even though the 30-second wall deadline closed it.

## Guard counterfactual

Raising only decoded-silence to 100 exposed a masking condition: the frozen
30-frame no-text/no-audio progress watchdogs ended every response first. A
second capture-gated diagnostic override raised both progress thresholds to
100 after first validating the frozen production values. This did not modify
the TOML contract or the production container.

With all three frame thresholds at 100, eight responses accumulated the full
100 silent frames. None showed PCM above -90 dBFS after first reaching 20.
Therefore the observed condition required to indict the production guard—audio
resuming only after frame 20—did not occur.

This does not prove that no future stochastic response can ever resume after
20 frames. It does show that the current reproduced losses are already silent
at the codec PCM boundary and remain silent through the extended observation.

## Capture perturbation

“Steady agent-speaking” uses the same trace predicate as the prior analysis:
`rnnt.agent_speaking=true`.

| Run | n | p50 ms | mean ms | p95 ms | p99 ms | max ms | >80 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior production baseline | 633 | 72.852 | 72.821 | 75.080 | 76.085 | 78.250 | 0 |
| Current uninstrumented control | 400 | 78.515 | 77.989 | 80.168 | 80.614 | 96.123 | 36 |
| Capture, progress=30 | 1,613 | 78.215 | 77.962 | 81.082 | 84.438 | 157.503 | 304 |
| Capture, fully extended | 1,943 | 76.865 | 77.221 | 82.764 | 84.339 | 159.204 | 563 |

Mean timing did not regress against the same-window control, but tail timing
did: capture p95 increased by 0.914--2.596 ms, the >80 ms share increased from
9.0% to 18.8--29.0%, and maximum steps exceeded 157 ms. This is a material
tail perturbation. Consequently, these captures are invalid for latency,
wall-clock, or production-rate conclusions.

The three-way verdict is not inferred from those durations. It uses exact
same-position raw/post-policy/decoder tensors, aligned codec PCM, control
counters, and frame-count state. Attempt counts are reported only as observed
counts and are not extrapolated.

## Implementation and integrity

- `eartts_acoustic_diagnostic.py` uses a bounded queue and CPU-pinned writer;
  device-to-host materialization and `torch.save` run off the realtime call
  site. Queue overflow is fatal rather than silently dropping evidence.
- The runtime hook clones the true raw generated code before the first PAD
  policy, records exact before/after counters and `tail_done`, and accounts for
  the process codec's one-call PCM delay plus final drain.
- The launcher enforces an isolated container and port. Diagnostic guard
  overrides require capture and must exceed validated production thresholds.
- The replay maps all 4,277 source packet events to the retained 2,904,256 PCM
  bytes and preserves original turn boundaries.
- Both captures have contiguous frame sequences and zero missing PCM records.
  Docker SIGTERM did not run the Python `atexit` summary hook, so integrity is
  established from the retained frame sequence and analyzer rather than a
  `summary.json` claim.

Validation passed: Ruff, Python compilation, `git diff --check`, 51 focused
runtime tests (15 skipped), and the capture codec-delay/final-drain unit test.
No commit was created and `config/` is unchanged.

## Operational state

The maintenance window preserved production container
`042d10e1d6d5fc264d7ee7aaf0c9dbeec9da5512e65b79d030bd7be2f0d2c444`
and image
`sha256:4607d8eb4b99a690644e35af1e75102f75fbfe148b4aee04dd9c278b4f1b0dd9`.
The exact container was restarted after the diagnostic.
