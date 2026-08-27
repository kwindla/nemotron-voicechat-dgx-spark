# Two responses lost their spoken tails (session 431ac6df)

A live manual session over the ngrok tunnel produced two responses the user
reported as "did not fully render as audio". This is the evidence that they
really were truncated, measured rather than inferred.

## Verdict

Independent transcription of the audio the model actually emitted, diffed
against the text the model actually generated:

| response | words spoken / generated | rendered | never spoken |
|---|---:|---:|---|
| 1 | 13 / 13 | 100% | — |
| 2 | 22 / 41 | **54%** | "I can help with anything from answering questions to generating creative content. What would you like to talk about?" |
| 3 | 68 / 86 | **79%** | "From then on the forest was filled with laughter and magic, and everyone lived in peace. The end." |
| 4 | 10 / 14 | 71% | cut by the user's interruption (expected) |
| 5 | 11 / 11 | 100% | — |
| 6 | 11 / 11 | 100% | — |
| 7 | 11 / 11 | 100% | — |

Both losses are clean tail truncations at a sentence boundary, not gaps in the
middle.

## What this rules in and out

The `decoded_silence_watchdog` fired three times (frames 303, 667, 891), each
one model step before that response's forced EOS. See
`watchdog-and-truncation.txt` for the retained counters.

- **Not yesterday's no-text watchdog defect.** At every firing the no-text and
  no-audio predicates stood at 20/30 and never reached threshold. What fired
  was the decoded-silence predicate at 20/20, a path that fix deliberately left
  alone.
- **Watchdog firing is not the same thing as truncation.** Response 5 fired the
  same watchdog and is 100% complete. The watchdog is downstream of the loss in
  observed causal order; what it cannot rule out is the counterfactual that
  EarTTS would have resumed after more than 20 silent frames.
- **An earlier rate-based argument (characters per second) is superseded** by
  the ASR result and should not be cited.

Full analysis, including the real-time margin reconciliation and the separate
tool-call finding, is in `docs/reviews/live-session-431ac6df-analysis-r1.md`.

## How the clips were produced

`output-22050-mono-s16le.pcm` in the model trace is exactly the frames the
model emitted, in emission order, 1764 samples each. The playout trace gives
each response's frame count, so the clips are frame-exact cuts at those
offsets: 51, 109, 260, 33, 48, 49, 46 frames respectively, in order.

Source trace (outside the repo, root-owned, rotates):
`~/.local/state/nemotron-voicechat/traces/model/431ac6df-573c-45af-8746-7cd9f66bd8c3/`
Client playout trace:
`~/.local/state/nemotron-voicechat/playout-traces/pipecat-playout-r3-0002.jsonl`

Transcription used the pinned offline evaluator, image-owned backend, no
network:

    docker run --rm --gpus all --ipc host --network none \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v <clips>:/clips \
      pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b \
      python3 -P /clips/transcribe_clips.py

`transcripts.json` holds its raw output. The evaluator is the same one the
qualification gate uses, so it is independent of the voice model under test.
