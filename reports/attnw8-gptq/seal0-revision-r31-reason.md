# Seal-0 revision 31 reason

Seal-0 revision 30 was invalidated during its baseline-only A0 live capture
under the prospectively amended transcript criterion. No-call scenario
`a2-nocall-09` correctly made zero calls and zero injections, but the observed
word sequence truncated the opening `give me a synonym` to `y`. This is a word
difference and therefore remains a failure under the amended criterion.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r30/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-nocall-r31-09`, using a simpler synonym request. Nothing from
the invalid revision-30 baseline is eligible for Seal-1.
