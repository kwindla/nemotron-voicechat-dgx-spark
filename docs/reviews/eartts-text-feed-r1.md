# EarTTS text-feed diagnostic r1

## Verdict

The evidence selects **B: EarTTS received the whole generated response and its
acoustic continuation stopped early**. None of the four ASR-confirmed losses is
short on delivered content tokens. The production tokenizer requires 16, 83,
98, and 79 content tokens; the captures delivered exactly 16, 83, 98, and 79,
with the counter ending at the same values.

The specific observed failure is in EarTTS's PAD-conditioned acoustic
continuation. After receiving the full text densely, EarTTS received a valid,
masked PAD run while it rendered the queued utterance from recurrent state. In
the truncated cases its raw acoustic generator entered a silence-code regime
while the ASR-confirmed suffix was still acoustically owed. There was no text
feed shortage, text EOS, ratio-cap event, or decoder substitution at the
silence transition.

This evidence names the failing boundary and behavior, but it does not expose
the deeper model-internal reason that the recurrent acoustic alignment lost or
prematurely discharged its remaining text. Attention/alignment state was not
retained. That is the next field to capture if a mechanism below the observed
PAD-continuation failure is required.

## Tokenization and full delivery

The analysis used the production Nano tokenizer (`PreTrainedTokenizerFast`,
vocabulary 131,072) whose `tokenizer.json` SHA-256 is
`3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f`.
That is identical to the tokenizer in the running `nano-fhw8` artifact.

| Case | Full generated text tokens | Delivered content tokens | Final content counter | Exact full sequence |
|---|---:|---:|---:|---|
| regular A1/R1 | 16 | 16 | 16 | yes |
| regular A1/R3 | 83 | 83 | 83 | yes |
| regular A3/R3 | 98 | 98 | 98 | yes |
| extended A1/R3 | 79 | 79 | 79 | yes |
| extended A2/R3, ASR-complete control | 104 | 104 | 104 | yes |

Four replay report strings begin with one whitespace separator. Tokenizing the
reported string with or without that separator requires the same token count;
the separator changes only the first token ID. Removing only that leading
whitespace makes the complete tokenizer sequence ID-for-ID identical to
the captured `current_subword_id` sequence. The first case has no separator and
matches directly. Decoding every captured content sequence reconstructs the
entire generated response.

All captured `current_subword_mask` values are true, including the PAD
continuation. The counters increment by exactly one on every content position.

## Missing-suffix alignment

The missing suffix tokens were not merely somewhere in the full token count;
they can be aligned directly to their captured delivery frames:

| Case | ASR-missing suffix | Suffix tokens | Delivery frames | PAD begins | Final silence begins |
|---|---|---:|---:|---:|---:|
| regular A1/R1 | “How can I help you today?” | 7 | 65-71 | 72 | 83 |
| regular A1/R3 | “From then on…” through “The end.” | 18 | 514-531 | 532 | 666 |
| regular A3/R3 | “The End.” | 3 | 580-582 | 583 | 782 |
| extended A1/R3 | “The end.” | 3 | 562-564 | 565 | 722 |

Every missing suffix was completely delivered before the first terminal PAD
and between 12 and 200 frames before final PCM silence. Therefore the delivered
content covers more than the spoken prefix: it explicitly includes every token
that ASR did not hear.

## PAD and acoustic latency

PAD begins on the frame immediately after the last content increment in every
case. PCM continues after that handoff, as expected for EarTTS rendering
latency:

| Case | Last content frame | PAD start | Final-silence onset | Onset minus last content | Audible PAD frames | Last audible frame |
|---|---:|---:|---:|---:|---:|---:|
| regular A1/R1 | 71 | 72 | 83 | 12 | 11 | 82 |
| regular A1/R3 | 531 | 532 | 666 | 135 | 121 | 665 |
| regular A3/R3 | 582 | 583 | 782 | 200 | 178 | 781 |
| extended A1/R3 | 564 | 565 | 722 | 158 | 142 | 721 |
| extended A2/R3 complete control | 594 | 595 | 837 | 243 | 223 | 836 |

Each acoustic frame is 80 ms. The audible-frame count can be lower than the
frame difference because natural pauses dip below -90 dBFS before the final
continuous silence run. In all five cases the last audible PCM is the frame
immediately before final silence.

At the truncated onsets, the preceding PAD counters are 11, 134, 199, and 157.
The input remains PAD and the mask remains true; raw, post-policy, and decoder
codes are equal there. In extended A1/R3, the diagnostic PAD-tail policy begins
only at frame 803, 81 frames after raw acoustic output became silent at frame
722. It cannot explain the transition.

## Complete long-response counterexample

Extended A2/R3 generated 86 words and ASR recovered all 86. Its full 104-token
sequence was delivered exactly, content ended at frame 594, and EarTTS remained
audible during its PAD continuation through frame 836 before entering final
silence at 837. This is the same feed shape as the truncated cases—dense full
content followed by PAD—but its recurrent acoustic generation preserved enough
rendering state to finish.

The contrast is not “truncated responses received too few tokens, complete
responses received them all”: all five received the exact full sequence. The
contrast lies in EarTTS's acoustic continuation after the feed has completed.

## Ratio-cap cross-check

`S2S_TTS_TEXT_TOKEN_MIN=5` and `S2S_TTS_TEXT_TOKEN_RATIO_CAP=16` do not pace or
truncate the text feed. The code counts non-control content tokens and can force
text EOS only when talking frames reach `16 × content_tokens`.

For the four losses, those thresholds are 256, 1,328, 1,568, and 1,264 talking
frames. The entire captured responses contain only 60, 250, 330, and 339
positions, respectively, so the guard cannot have fired. The sole EOS in each
capture is the final termination frame requested by the no-text watchdog, well
after the full content sequence and final-silence transition. The complete
control likewise has 356 positions versus a 1,664-frame ratio threshold and
ends on its wall deadline.

Thus the two suspected settings did not stop content delivery and are not the
mechanism for these losses.

## Evidence and safety

The reproducible offline analyzer is
`tools/qualification/analyze_eartts_text_feed.py`. Its output is under
`reports/response-truncation/eartts-text-feed-r1/`:

- `analysis.json` contains tokenizer identity, full/delivered ID sequences,
  suffix offsets and frames, timing, guard bounds, and the five-case verdict;
- each `*-positions.jsonl` contains frame-level token kind and ID, decoded token
  text, mask, counters before/after, PCM dBFS, and policy state.

The analyzer used only retained captures in an offline, network-disabled,
CPU-only container. No new live run was required. Production was not stopped,
restarted, entered, or reconfigured; the frozen contract is unchanged and no
commit was made.
