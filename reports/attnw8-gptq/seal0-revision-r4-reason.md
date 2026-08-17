# Seal-0 revision 4 reason

Seal-0 revision 3 baseline evidence is invalidated in full. At scenario 23 of
48, the hard no-call calibration scenario `a2-nocall-11` produced one
`get_benchmark_word` call. The runner was stopped immediately after the
categorical mismatch was observed.

No Hessian, quantized weight, candidate artifact/output, threshold change, or
qualification measurement was produced. Fence-r2 was restored before this
revision.

Revision 4 replaces the entire rejected scenario with
`a2-nocall-r4-11`, independently wording and recording a direct no-function
instruction. The replacement's prompt, expected transcript, audio bytes and
hashes, scenario ID, tool payloads, and expected no-call outcome are included
in the new complete fixture manifest and partition audit. All other scenarios,
partitions, rules, algorithms, and thresholds remain fixed. The complete
baseline-only A0/index stage restarts from empty revision-4 directories, and
no revision-3 evidence may enter Seal-1.
