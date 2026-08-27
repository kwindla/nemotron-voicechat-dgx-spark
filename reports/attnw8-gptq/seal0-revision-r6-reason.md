# Seal-0 revision 6 reason

Seal-0 revision 5 was invalidated during its baseline-only A0 live capture.
The positive scenarios `a2-pos-r5-02` and `a2-pos-r5-05` produced the correct
single `get_benchmark_word({})` call and injection, but their observed baseline
transcripts did not match their sealed expected transcripts exactly:

- `a2-pos-r5-02`: expected `returned`, observed `return`.
- `a2-pos-r5-05`: expected no terminal punctuation, observed a terminal period.

The runner was stopped at discovery, the partial evidence and capture-service
log were retained under `reports/attnw8-gptq/a0-r5/`, and the fence-r2 stack was
restored before this revision. In accordance with the pre-Seal-1 replacement
disposition, both complete scenarios are replaced with independently worded,
newly generated/audio-hashed scenarios `a2-pos-r6-02` and `a2-pos-r6-05`.
Nothing from the invalid revision-5 baseline is eligible for Seal-1.
