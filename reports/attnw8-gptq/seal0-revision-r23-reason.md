# Seal-0 revision 23 reason

Seal-0 revision 22 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r22-11` made the correct single tool call and
injection and matched all transcript words, but the observed transcript omitted
the terminal period declared by the sealed expected transcript.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r22/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-pos-r23-11`, using a longer single-clause carrier aligned with
the stable positive set. Nothing from the invalid revision-22 baseline is
eligible for Seal-1.
