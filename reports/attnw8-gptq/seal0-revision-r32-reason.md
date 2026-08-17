# Seal-0 revision 32 reason

Seal-0 revision 31 was invalidated during its baseline-only A0 live capture
under the prospectively amended transcript criterion. No-call scenario
`a2-nocall-12` correctly made zero calls and zero injections, but the observed
word sequence rendered sealed `harbor` as `harbour`. This is a word difference
and therefore remains a failure under the amended criterion.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r31/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-nocall-r32-12`, using the unambiguous noun `marina`. Nothing
from the invalid revision-31 baseline is eligible for Seal-1.
