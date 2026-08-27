# Seal-0 revision 25 reason

Seal-0 revision 24 was invalidated during its baseline-only A0 live capture.
No-call scenario `a2-nocall-01` correctly made zero calls and zero injections,
but the observed transcript dropped the opening syllable of `without` and did
not match the sealed punctuation.

The fail-closed monitor stopped the runner at discovery, and the partial
evidence and capture-service log were retained under
`reports/attnw8-gptq/a0-r24/`. The fence-r2 stack was restored before this
revision. The complete scenario is replaced by independently generated and
audio-hashed `a2-nocall-r25-01`, using a simpler single-clause no-call carrier.
Nothing from the invalid revision-24 baseline is eligible for Seal-1.
