# Seal-0 revision 39 reason

Seal-0 revision 38 is invalidated during A0, before eager reconstruction
completed and before any model inference, effective-position index, Seal-1,
Hessian, quantized weight, candidate, or gate work. The sealed eager loader
dequantized the qualified artifact's shape-padded MLP down projections at
their packed Marlin K dimension of 15,744, while the eager HF propagation
module correctly has the original K dimension of 15,680. The explicit shape
check stopped at layer 1 with `15744 != 15680`.

Revision 39 changes only the sealed eager reconstruction loader. For an MLP
`down_proj` carrying the qualified artifact's recorded 64-column Marlin shape
padding, it removes those padding columns before copying the BF16 propagation
weight into the eager module. This matches the in-tree converter's existing
contract: quantization pads K for packing, while eager propagation installs
only the original 15,680 columns. No attention target, corpus input, scenario,
criterion, numerical contract, or gate rule changes.

The complete r38 pooled baseline and capture are retained under
`reports/attnw8-gptq/a0-r38/` as invalidated evidence and are ineligible for
Seal-1. Every prior invalidation stands, and all prospective criterion
deviations remain explicitly subject to the results adversarial review.
