# Seal-0 revision 22 reason

Seal-0 revision 21 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r21-11` made the correct single tool call and
injection and matched all transcript words, but the observed transcript added
a terminal period absent from the sealed expected transcript.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r21/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-pos-r22-11`, whose prompt and expected transcript explicitly
declare the observed terminal period. Nothing from the invalid revision-21
baseline is eligible for Seal-1.
