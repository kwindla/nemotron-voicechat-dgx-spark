# Step 5 wedge Phase 1 post-drain KV divergence adjudication

**Adjudication date:** 2026-08-15 UTC  
**Disposition:** **(ii) GENUINE PACKED-PATH POST-DRAIN KV DIVERGENCE**  
**Diagnostic run:** **not used**; retained evidence and sealed source decide the question offline  
**Campaign status:** unchanged — calibration remains rejected, no envelope is authorized, and the held-out Phase 1 replicates remain unrun

## Decision

The ten post-first-boundary KV failures are not an oracle position-mapping
artifact. Lane S and packed lane A execute the same corrected pending row, with
the same retained row identity, as the one accepted drain position at the same
logical KV index. The scheduler derives that position from the post-rejection
committed count in both lanes, and the receipt shows both snapshots advancing
from 377 to 378 committed positions. The committed-KV projection preserves
request block order and selects logical positions `[0, committed_tokens)`; it
does not apply a lane-dependent position transform.

The first drain therefore compares S position 377 with A position 377. Those
correctly aligned BF16 key/value vectors differ in all four attention leaves.
Later positions are then computed over different attention history and the
divergent support grows. This is a new bug-class finding about the retained
pair implementation: after a rejected packed second position is corrected and
drained, the packed path does not reproduce the sequential path's committed
attention KV content, even though observed token decisions and exact
transaction metadata remain equal.

No projection correction or waiver is justified. Skipping A position 377,
shifting it, or substituting an uncommitted cache-tail position would compare a
different token and conceal the retained failure.

## Offline adjudication of the position hypothesis

### Same drain-row identity and same committed boundary

For every construction and both lanes:

- immediately before the first drain, `text-only-rejection` has four KV leaves
  of shape `[2, 377, 8, 128]` and is exact between S and A;
- the rejected source-row identity is
  `8c814df8dd7acd5532f28c7214218729bc31275d647ec917ec268b7db7372d45`;
- the drain source, `selected_source_row_identity`,
  `pending_row_sha256_before`, and the operation-evidence source identity are
  the same hash:
  `2a45d382465463f4ea177e4f106ba2a8b7583d7641b5b874dd12ddbbe15371e2`;
- the drain accepts exactly one position, selects row 0, and advances each KV
  leaf to `[2, 378, 8, 128]` in both S and A.

These facts repeat identically across the 5 constructions (10 lane
observations). They exclude a different drained token, a different accepted
count, or a different logical committed length.

### Scheduler and projection mapping

The retained worker executes `self._original_step(pending, request_id)` for a
pending drain without a lane-specific branch. Scheduler rejection accounting
first subtracts the one rejected speculative token from
`num_computed_tokens`. On the next one-row drain, the model runner constructs:

```text
logical_position = num_computed_tokens_before_drain + row_arange
                 = 377 + 0
                 = 377
```

That position is supplied to `compute_slot_mapping`. In request-relative form,
the KV slot rule is:

```text
block_ordinal = floor(logical_position / block_size)
intra_block   = logical_position mod block_size
physical_slot = request_block_ids[group][block_ordinal] * block_size
                + intra_block
```

S and A may own different physical block IDs, but the mapping from logical
position 377 to their respective request-owned slot uses the same rule. The
capture selects request blocks in their recorded order, flattens only the
block/token axes, and projects `selected[:, :committed_tokens]`. The canonical
comparison therefore maps request-relative position `p` to projection position
`p`, for `0 <= p < committed_tokens`; at the first drain it includes exactly
positions 0 through 377.

This is already the correct mapping rule. There is no retained evidence for a
packed drain at logical position 378 or for a physical-slot permutation. A
hypothetical projection `[0..376, 378]` would reach beyond the declared
378-position committed boundary and select uncommitted cache-tail content. It
is not a correction.

## Leaf-level evidence at the first drain

Each leaf has 2,048 BF16 elements per logical position: key/value axis 2,
8 heads, and head width 128. With shape `[2, 378, 8, 128]`, flat element
386,048 is the first key element at logical position 377
(`377 * 8 * 128`). Construction 4, layer 21 first differs one element later;
all other first offsets are 386,048. Since the pre-drain 377 positions are
exact and the first-drain tensor adds only position 377, all first-drain
differences are confined to that correctly aligned new position.

| KV leaf | Constructions | Different elements at position 377 (range) | First flat element | Maximum absolute delta |
|---|---:|---:|---:|---:|
| `model.layers.14.mixer.attn[0]` | 5/5 | 1,696–1,962 / 2,048 | 386,048 | 0.21875 |
| `model.layers.21.mixer.attn[0]` | 5/5 | 1,861–1,984 / 2,048 | 386,048–386,049 | 0.328125 |
| `model.layers.30.mixer.attn[0]` | 5/5 | 1,710–1,942 / 2,048 | 386,048–386,049 | 0.21875 |
| `model.layers.39.mixer.attn[0]` | 5/5 | 1,706–1,883 / 2,048 | 386,048 | 0.1875 |

The first differing element lies in the key plane. Every per-leaf differing
count exceeds 1,024, so divergence also reaches the value plane; this is not
merely a rotary-key presentation difference. The differing content is the
committed key and value generated for the identical corrected fused row at the
same position, under S versus packed-path history/execution.

## Accumulation after the drain

This table aggregates the four leaves over all five constructions (20 leaf
comparisons per boundary). `First flat element` remains at the original drain
position; the one-element variation is construction 4, layer 21.

| Boundary | Committed positions | Different elements per leaf (range) | First flat element | Maximum absolute delta |
|---|---:|---:|---:|---:|
| `text-only-rejection-pending-drain` | 378 | 1,696–1,984 | 386,048–386,049 | 0.328125 |
| `function-only-rejection` | 379 | 3,074–3,591 | 386,048–386,049 | 0.328125 |
| `function-only-rejection-pending-drain` | 380 | 4,747–5,328 | 386,048–386,049 | 0.328125 |
| `conjunctive-rejection` | 381 | 6,286–6,962 | 386,048–386,049 | 0.328125 |
| `conjunctive-rejection-pending-drain` | 382 | 8,023–8,784 | 386,048–386,049 | 0.328125 |
| `agent-bos` | 383 | 9,594–10,483 | 386,048–386,049 | 0.328125 |
| `agent-eos` | 384 | 11,339–12,218 | 386,048–386,049 | 0.375 |
| `function-sotc` | 385 | 12,999–14,066 | 386,048–386,049 | 0.4375 |
| `function-eotr` | 386 | 14,874–16,029 | 386,048–386,049 | 0.84375 |
| `post-fc-bos` | 387 | 16,796–17,898 | 386,048–386,049 | 0.84375 |

The unchanged first offset demonstrates persistence of the original drain
failure. Growing different-element counts demonstrate propagation into later
committed content. The receipt's exact comparator does not retain a complete
per-position support bitmap, so this adjudication does not claim an exact
different-element count for each later token.

## Controls and scope

- The first `text-only-rejection` boundary is exact in all five constructions,
  including committed KV, at 377 positions.
- Mamba conv layer 0 is exact zero drift in all 60 applicable
  construction/boundary observations. This rules out a universal capture or
  comparison offset but does not make all recurrent leaves exact.
- The only exact-field failures are the four committed-KV leaves at the ten
  listed post-drain boundaries: 40 failures per construction, 200 retained
  leaf failures total.
- Emitted text/function decisions, conjunction, accepted count, selected row,
  source-row identity, pending-row join, and other exact transaction metadata
  held. Numeric logits and non-control Mamba leaves belonged to the
  calibration's directional-statistic population, not to a claim of bitwise
  identity.

This is state-content divergence, not demonstrated decision divergence in this
finite sequence. It nevertheless defeats the pairing candidate's claimed
post-rejection state exactness. Different committed attention history can
influence later hidden states and decisions outside the retained sequence, so
decision equality here cannot authorize the candidate as an exact replacement
or justify a KV tolerance.

## Disposition and required next action

1. Classify the result as a genuine retained-pair implementation bug class:
   **post-rejection corrected-drain committed-attention-KV non-equivalence**.
2. Keep the calibration verdict **REJECTED**. Emit no Phase 1 envelope and do
   not schedule the held-out campaign.
3. Do not amend the projection, shift positions, omit drain KV, or add a
   waiver. The projection compared the intended committed token.
4. Any repaired pairing implementation must be evaluated under a new,
   independently sealed contract and a calibration rerun from scratch. The
   failed calibration cannot be reused to set tolerances.

No campaign was run and the one authorized diagnostic run remains unused.

## Evidence seals

| Evidence | SHA-256 |
|---|---|
| calibration contract JSON | `c22f3cb477c898fe3f4caa655c751345b363df47c509aa5960b166fb36b4ee38` |
| calibration report | `8f8916b8a29b77cd998a58b3778477bae1d11e2d86ddcf5095f9f5f7bf5ef197` |
| five-construction calibration receipt | `a5a45dc07ad5981c77600b7bd68b9f3bb40899835937168fcf4a4602897f0ec8` |
| calibration artifact manifest | `e6451d8713e2f4f8ce6542660408dd39957b03f964f2a6998e235ab969182fe8` |
| retained Phase 1 GPU worker / projection source | `a1dd31b6a7841f798865f23cd2a44a7bd3952d5095f826fad5d6694e54e054f5` |
| retained calibration runner (constructions 1–4) | `2887e3d22157844553c727ab7471eb9789145bfd1bea2afcf6569b5283074298` |
| retained calibration runner (construction 0, receipt-recorded) | `18788387c4dae0b733771efdb5d0e92832e210233465fa34a4b1d72bb5b30c8b` |
| receipt-recorded patched vLLM tree | `7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260` |
| receipt-recorded patched model runner | `32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447` |
| receipt-recorded diagnostic patch manifest | `f1e472e597dc846a3d366fe1514fbc6bcb50599ef5cd3d56a9d88a9c514559d2` |

The adjudication report's own SHA-256 is recorded in the required completion
note, avoiding a circular self-hash field.
