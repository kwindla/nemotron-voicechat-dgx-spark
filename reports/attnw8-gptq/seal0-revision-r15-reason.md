# Seal-0 revision 15 reason

Seal-0 revision 14 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-09` made the correct single tool call and injection,
but its observed transcript clipped the opening of `ask` and added a terminal
period, so it did not exactly match the sealed expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r14/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the longer, punctuation-free, independently generated
and audio-hashed `a2-pos-r15-09`. The punctuation-bearing expected transcripts
from revisions 9 and 10 are preserved explicitly. Nothing from the invalid
revision-14 baseline is eligible for Seal-1.
