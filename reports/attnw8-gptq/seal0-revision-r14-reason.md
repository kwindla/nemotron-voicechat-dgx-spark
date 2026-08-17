# Seal-0 revision 14 reason

Seal-0 revision 13 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r13-08` matched its expected transcript exactly but
produced zero tool calls, failing the required categorical outcome.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r13/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the longer, explicit, independently generated and
audio-hashed `a2-pos-r14-08`. Its punctuation-free prompt and transcript are
fixed before capture. The punctuation-bearing expected transcripts from
revisions 9 and 10 are preserved explicitly. Nothing from the invalid
revision-13 baseline is eligible for Seal-1.
