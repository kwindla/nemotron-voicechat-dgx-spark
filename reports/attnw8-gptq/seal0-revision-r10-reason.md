# Seal-0 revision 10 reason

Seal-0 revision 9 was invalidated during its baseline-only A0 live capture.
Positive scenario `a2-pos-06` made the correct single tool call and injection,
but its observed transcript dropped the initial `could`, changed `its` to
`it's`, and added a terminal period, so it did not exactly match its sealed
expected transcript.

The monitor stopped the runner at discovery, the partial evidence and
capture-service log were retained under `reports/attnw8-gptq/a0-r9/`, and the
fence-r2 stack was restored and re-attested before this revision. The complete
scenario is replaced by the longer, independently recorded/audio-hashed
`a2-pos-r10-06`, with a punctuation-bearing expected transcript sealed before
capture. Nothing from the invalid revision-9 baseline is eligible for Seal-1.
