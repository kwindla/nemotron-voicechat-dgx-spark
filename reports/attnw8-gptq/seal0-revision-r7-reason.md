# Seal-0 revision 7 reason

Seal-0 revision 6 was invalidated during its baseline-only A0 live capture.
The positive scenario `a2-pos-r6-02` produced the correct single
`get_benchmark_word({})` call and injection, but the observed transcript ended
with a period while its sealed expected transcript did not.

The runner was stopped at discovery, the partial evidence and capture-service
log were retained under `reports/attnw8-gptq/a0-r6/`, and the fence-r2 stack was
restored and re-attested before this revision. In accordance with the
pre-Seal-1 replacement disposition, the complete scenario is replaced by the
independently worded, newly generated/audio-hashed `a2-pos-r7-02`. Its expected
transcript explicitly retains the terminal period instead of applying the
fixture generator's terminal-punctuation stripping. Nothing from the invalid
revision-6 baseline is eligible for Seal-1.
