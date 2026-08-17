# Seal-0 revision 28 reason

Seal-0 revision 27 was invalidated during its baseline-only A0 live capture
under the prospectively amended transcript criterion. No-call scenario
`a2-nocall-02` correctly made zero calls and zero injections, but its observed
word sequence dropped `do` and rendered `harbor` as `harbour`. These are word
differences and therefore remain failures under the amended criterion.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r27/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-nocall-r28-02`, using an unambiguous no-call sentence. Nothing
from the invalid revision-27 baseline is eligible for Seal-1.
