# Seal-0 revision 13 reason

Seal-0 revision 12 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-r12-08` made the correct single tool call and
injection, but the observed transcript joined the intended two clauses without
the sealed sentence break.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r12/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by independently generated/audio-hashed
`a2-pos-r13-08`, whose prompt uses the continuous phrasing before capture. The
punctuation-bearing expected transcripts from revisions 9 and 10 are preserved
explicitly. Nothing from the invalid revision-12 baseline is eligible for
Seal-1.
