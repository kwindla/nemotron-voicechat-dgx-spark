# Seal-0 revision 5 reason

Seal-0 revision 4 baseline evidence is invalidated in full. The positive
scenario `a2-pos-02` produced zero tool calls. Before the runner interruption
completed, the evidence also showed that `a2-pos-04` lost its initial phoneme
in the RNNT transcript and `a2-pos-05` had a terminal-punctuation transcript
mismatch. All three therefore fail the sealed baseline scenario-acceptance
contract.

No Hessian, quantized weight, candidate artifact/output, threshold change, or
qualification measurement was produced. Fence-r2 was restored before this
revision.

Revision 5 replaces the complete records for those three scenarios with
independently worded and recorded positive scenarios whose prompts name the
registered benchmark function directly. Their new IDs, prompts, expected
transcripts, audio bytes/hashes, tool payloads, and expected exact call are
included in the new manifest and partition audit. The other 46 records and
all partitions, rules, algorithms, and thresholds remain fixed. The complete
baseline-only A0/index stage restarts from empty revision-5 directories; no
revision-4 evidence may enter Seal-1.
