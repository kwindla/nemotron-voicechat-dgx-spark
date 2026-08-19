# EarTTS text-feed evidence r1

Verdict: **B**. Every truncated response delivered the exact full production
tokenizer sequence to EarTTS, including all ASR-missing suffix tokens. EarTTS's
PAD-conditioned acoustic continuation subsequently entered silence while
speech remained owed.

Files:

- `analysis.json`: consolidated five-case analysis, including the ASR-complete
  86-word control;
- `regular-a1-r1-positions.jsonl`: 6-word suffix loss;
- `regular-a1-r3-positions.jsonl`: 14-word suffix loss;
- `regular-a3-r3-positions.jsonl`: 2-word suffix loss;
- `extended-a1-r3-positions.jsonl`: fully extended 2-word suffix loss;
- `extended-a2-r3-complete-control-positions.jsonl`: ASR-complete 86-word
  counterexample.

Each position row records token identity and kind, content/PAD counters,
subword mask, retained codec PCM dBFS, and PAD-policy state. See
`docs/reviews/eartts-text-feed-r1.md` for the interpretation.
