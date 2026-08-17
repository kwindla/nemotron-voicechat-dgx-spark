# Seal-0 revision 40 reason

Seal-0 revision 39 is invalidated during the first A0 preflight propagation,
before hook proofs, indexes, Seal-1, Hessians, quantized weights, candidates,
or gates. Eager reconstruction overlaid the packed runtime's cached-streaming
`chunk_size=1` onto the uncached full-sequence Mamba torch path. That mode
combination formed a pathological broadcast and requested a 1,033 GiB
temporary allocation in layer 0. This was not ordinary memory pressure and no
model decision was produced.

Revision 40 changes only eager full-sequence reconstruction: after overlaying
the exact artifact architecture and weights, it retains the Nano skeleton's
`chunk_size=128` for uncached eager propagation. The packed public runtime
continues to use its unchanged `chunk_size=1` cached streaming contract. The
campaign retains the deterministic torch path required for transparent module
hooks; a synthetic audit at the maximum sealed effective sequence length of
1,545 completed with a 24,423,130,112-byte peak allocation. This matches the
in-tree full-sequence calibration/conversion mode and does not change weights,
effective positions, attention targets, corpus inputs, scenarios, criteria,
the numerical contract, or gates.

The complete r39 pooled baseline and capture are retained under
`reports/attnw8-gptq/a0-r39/` as invalidated evidence and are ineligible for
Seal-1. Every prior invalidation stands, and all prospective criterion
deviations remain explicitly subject to the results adversarial review.
