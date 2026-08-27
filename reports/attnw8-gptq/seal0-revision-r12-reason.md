# Seal-0 revision 12 reason

Seal-0 revision 11 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-08` made the correct single tool call and injection,
but the observed transcript used a sentence break (`word. please`) where the
sealed expected transcript used a semicolon (`word; please`).

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r11/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by independently generated/audio-hashed
`a2-pos-r12-08`, whose prompt fixes the sentence break before capture. The
punctuation-bearing expected transcripts from revisions 9 and 10 are also
preserved explicitly. Nothing from the invalid revision-11 baseline is
eligible for Seal-1.
