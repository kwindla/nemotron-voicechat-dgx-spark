# Seal-0 revision 38 reason

Seal-0 revision 37 is invalidated during A0, before eager model reconstruction
and before Seal-1. The sealed capture loader assumed that every public Nano
interface result was a token mapping. The public runtime instead returns the
boolean readiness sentinel `true` on each request's `decode_steps=0` prefill
call. The loader raised `AttributeError` when it attempted mapping access on
that sentinel. No Hessian, quantized weight, candidate, or gate work ran under
revision 37.

Revision 38 changes only the sealed offline loader: a mapping retains its
emitted text and function tokens, while a non-mapping prefill readiness
sentinel records no emitted token. The supplied prefill embeddings continue
to contribute exactly once under the existing effective-position convention.
The 48-member pool, manifest order, audio hashes, intent quotas, similarity
criterion, exact categorical requirements, and deterministic first-12-valid
selection rule remain unchanged.

The complete r37 pooled baseline and raw capture remain retained under
`reports/attnw8-gptq/a0-r37/` as invalidated evidence. They are ineligible for
Seal-1 and will not be reused. Every prior invalidation stands, and all
prospective criterion deviations remain explicitly subject to the results
adversarial review.
