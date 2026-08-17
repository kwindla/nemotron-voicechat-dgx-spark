# Seal-0 revision 24 reason

Seal-0 revision 23 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-12` made the correct single tool call and injection,
but the observed transcript rendered the sealed hyphenated compound
`fresh-word` as the unpunctuated token `freshword`.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r23/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-pos-r24-12`, which avoids the ambiguous compound. Nothing
from the invalid revision-23 baseline is eligible for Seal-1.
