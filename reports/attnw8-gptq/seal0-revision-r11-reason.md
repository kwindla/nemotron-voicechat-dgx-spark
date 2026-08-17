# Seal-0 revision 11 reason

Seal-0 revision 10 was invalidated during its baseline-only A0 live capture.
The regenerated shared manifest record for `a2-pos-r9-05` silently reverted
its expected transcript from the explicitly sealed punctuation-bearing value
in revision 9 to the fixture generator's punctuation-stripped value. The
observed revision-10 baseline transcript and positive categorical outcome were
otherwise correct.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r10/`, and the
fence-r2 stack was restored and re-attested before this revision. Revision 11
changes no scenario ID, prompt, carrier, audio hash, or outcome. It restores
the terminal periods for both current scenarios whose expected transcripts
were explicitly declared punctuation-bearing (`a2-pos-r9-05` and
`a2-pos-r10-06`) and restarts the complete baseline stage. Nothing from the
invalid revision-10 baseline is eligible for Seal-1.
