# Seal-0 revision 21 reason

Seal-0 revision 20 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r20-11` made the correct single tool call and
injection and matched all transcript words, but the observed transcript
rendered the declared comma after `now` as a period.

The fail-closed monitor stopped the runner at discovery, the partial evidence
and capture-service log were retained under `reports/attnw8-gptq/a0-r20/`, and
the fence-r2 stack was restored before this revision. The complete scenario is
replaced by independently generated/audio-hashed `a2-pos-r21-11`, using a
punctuation-free prompt and expected transcript. Nothing from the invalid
revision-20 baseline is eligible for Seal-1.
