# Seal-0 revision 18 reason

Seal-0 revision 17 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r17-10` matched its transcript words apart from a
terminal period but produced zero tool calls, failing the required categorical
outcome.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r17/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the explicit, independently generated/audio-hashed
`a2-pos-r18-10`, whose period-bearing prompt and expected transcript are fixed
before capture. All punctuation-bearing shared expected transcripts are
preserved explicitly. Nothing from the invalid revision-17 baseline is
eligible for Seal-1.
