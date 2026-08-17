# Seal-0 revision 9 reason

Seal-0 revision 8 was invalidated during its baseline-only A0 live capture.
The positive scenario `a2-pos-r6-05` was transcribed with its initial sound
truncated (`s get benchmark word now.`) and produced zero tool calls, failing
both its sealed transcript and required positive categorical outcome.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r8/`, and the
fence-r2 stack was restored and re-attested before this revision. In accordance
with the pre-Seal-1 disposition, the complete scenario is replaced by the
longer, independently recorded/audio-hashed `a2-pos-r9-05`. Its expected
transcript explicitly retains the sentence-final period. Nothing from the
invalid revision-8 baseline is eligible for Seal-1.
