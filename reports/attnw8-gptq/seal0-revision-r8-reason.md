# Seal-0 revision 8 reason

Seal-0 revision 7 was invalidated during its baseline-only A0 live capture.
The replacement positive scenario `a2-pos-r7-02` had an exact matching
transcript but produced zero tool calls, failing the required positive
categorical outcome.

The runner was stopped at discovery, the partial evidence and capture-service
log were retained under `reports/attnw8-gptq/a0-r7/`, and the fence-r2 stack was
restored and re-attested before this revision. In accordance with the
pre-Seal-1 replacement disposition, the complete scenario is replaced by the
independently recorded/audio-hashed `a2-pos-r8-02`. Its longer request form is
specified before capture and uses `return`, matching the stable synthesized
speech wording. Nothing from the invalid revision-7 baseline is eligible for
Seal-1.
