# Seal-0 revision 16 reason

Seal-0 revision 15 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r15-09` made the correct single tool call and
injection and matched all transcript words, but the observed transcript added
a terminal period not present in the sealed expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r15/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the independently generated/audio-hashed
`a2-pos-r16-09`, whose period-bearing `now` form and expected transcript are
fixed before capture. The punctuation-bearing expected transcripts from
revisions 9 and 10 are preserved explicitly. Nothing from the invalid
revision-15 baseline is eligible for Seal-1.
