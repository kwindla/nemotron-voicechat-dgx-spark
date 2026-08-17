# Seal-0 revision 19 reason

Seal-0 revision 18 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-11` made the correct single tool call and injection,
but the observed transcript rendered `use` as `used`, so it did not exactly
match the sealed expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r18/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the simpler, punctuation-free, independently generated
and audio-hashed `a2-pos-r19-11`. All punctuation-bearing shared expected
transcripts are preserved explicitly. Nothing from the invalid revision-18
baseline is eligible for Seal-1.
