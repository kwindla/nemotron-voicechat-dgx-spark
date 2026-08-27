# Seal-0 revision 17 reason

Seal-0 revision 16 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-10` made the correct single tool call and injection,
but the observed transcript rendered `tool's` as `tools`, so it did not exactly
match the sealed expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r16/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the punctuation-free, independently generated and
audio-hashed `a2-pos-r17-10`, whose wording avoids the possessive. The three
current punctuation-bearing expected transcripts are preserved explicitly.
Nothing from the invalid revision-16 baseline is eligible for Seal-1.
