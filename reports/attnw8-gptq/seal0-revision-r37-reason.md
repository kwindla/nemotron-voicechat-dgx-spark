# Seal-0 revision 37 reason

Seal-0 revision 36 is invalidated before A0/Seal-1 by a post-capture audit of
the newly pooled capture-index mapping. The raw capture contains one
non-`-1` public warmup sequence followed by the 48 pool sequences. The sealed
r36 A0 slice would have included that warmup and omitted the final pool member.
No eager A0, Hessian, candidate, or gate work ran under r36.

The complete r36 pool run and capture are retained under
`reports/attnw8-gptq/a0-r36/` but are ineligible for Seal-1. Revision 37 changes
only the sealed offline mapping to take the final 48 captured sequences. The
48-member pool, manifest order, audio hashes, intent quotas, similarity
criterion, exact categorical requirements, and deterministic first-12-valid
selection rule remain unchanged. This is an implementation correction, not a
scenario replacement or a resurrection of prior evidence.

All prospective deviations remain explicitly subject to the results
adversarial review, and every prior invalidation stands.
