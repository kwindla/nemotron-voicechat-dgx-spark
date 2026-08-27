# Seal-0 revision 20 reason

Seal-0 revision 19 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r19-11` made the correct single tool call and
injection and matched all transcript words, but the observed transcript added
a comma after `now` that was not present in the sealed expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r19/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by independently generated/audio-hashed
`a2-pos-r20-11`, whose prompt and expected transcript declare that comma before
capture. All punctuation-bearing shared expected transcripts are preserved
explicitly. Nothing from the invalid revision-19 baseline is eligible for
Seal-1.
