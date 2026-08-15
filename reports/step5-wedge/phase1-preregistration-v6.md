# Step 5 wedge Phase 1 preregistration — v6 executor round-4 reseal

**Seal date:** 2026-08-15 UTC  
**Status:** round-4 real-executor reseal; campaign not run

This v6 seal incorporates v5, SHA-256
`c685a247623233522993cd0511193fc959e97b3debd42f65a4427cde740c370a`,
and implements the remaining campaign blockers in the authoritative round-3
executor review. It does not change lane order S/B/A, the frozen case matrix,
12-replicate shape, terminal-A rule, discordance threshold, disposition, or
resume rule.

## Cross-lane exactness boundary

Cross-lane equality is evaluated only over the preregistered committed logical
boundary: emitted text/function/conjunction, accepted count, selected packed
row and executed source-row identity, retained custom outputs, committed KV,
committed Mamba conv/SSM, transaction rollback state, request-relative shadow
index/pointer and virtual-engine identity, host pending rows, correction flags,
and correction assumptions.

Transition IDs, rejection provenance/digests, pre/post whole-state digests,
position-one and uncommitted-position digests, raw physical shadow locations,
raw inactive shadow leaves, and live shared-control-owner state are retained as
within-lane evidence. They are not cross-lane equality fields because they
contain lane-absolute allocation, frame, or history values by construction.

KV comparison is the ordered request-owned cache projected to the exact
committed-token count; unused physical cache-block tail bytes are excluded.
The pair rollback buffers are projected to the same names, shapes, dtypes, and
logical layout as committed Mamba state. For the S transaction oracle the
rollback boundary is the captured committed state after logical position one;
for B/A it is the actual pair shadow holding that same boundary. Raw shadow
buffers remain sealed evidence. Rollback is not compared on controls, drains,
or aborts because those operations do not refresh the pair shadow.

No selected output or committed logical tensor may be normalized away. Any
remaining function-logit, committed KV, committed Mamba, or canonical rollback
difference after these representation projections is a genuine semantic
divergence and must stop the campaign with leaf-level byte/value evidence.

## Historical typed-carrier replay

The fixture no longer supplies a trusted carrier string. Materialization reads
the sealed historical browser-failure witness, extracts the unique UTF-8 byte
sequence whose SHA-256 is
`c9118304a83285d240206af1ab0d651b9211ce29d59484dbf8b6d1a1bc5d5007`,
and verifies its 58-byte count and historical accepted-event job identity. Those
exact bytes are replayed through the production deterministic Pocket backend
(English 2026-04, Alba, quantized, base seed zero). The synthesized PCM, rather
than retained PCM, drives perception/fusion. Materialization fails unless its
first 61,440 bytes are byte-exact with the retained post-discard 24-frame PCM
prefix. The witness, carrier, Pocket assets/package, synthesized PCM, and exact
prefix are retained in the materialization seal.

## Gated structural smoke

The single-engine structural smoke passes the completed S result map as the
oracle argument to both B and A `_run_lane` calls. It atomically retains its
receipt even when exactness fails. The receipt classifies every remaining
cross-lane comparator difference as genuine semantic divergence and includes
leaf-level differing-byte, first-value, and maximum-absolute-delta evidence.
The campaign is viable only if at least the first B case and then every B case
is exact under this gate; otherwise work stops and reports the scientific
finding without changing the comparator.

## Frozen implementation seals

| Item | SHA-256 / value |
|---|---|
| application HEAD | `1d8163831ef0ba29a1a41e6cab9f0cb4a411bd87` |
| oracle | `be23e35df47155d4790941f89605346a60e5377b7de7954c61b66fddce2d0ebd` |
| campaign | `b99b9b55883ff98a6e3deb2953d3ca8517659eccdbfbb461bcae451c50899905` |
| real executor | `5c7e34b81f8afec52d59c819d2f9e3ada500f01afd1ec881e0a3f9e3c525ca18` |
| GPU worker | `a1dd31b6a7841f798865f23cd2a44a7bd3952d5095f826fad5d6694e54e054f5` |
| typed-carrier helper | `7515583a5a4814625e880d853c427461ecc91c130f6ff0157e52c3f4209d4b9f` |
| Pocket production worker | `8a4d316d424b6eaa2a375cbf20bbec97d4857d5a4e0f080f7156ee3e5cb926ea` |
| frozen executor config | `a07b87401d2233180df899910d6a05fd7273446190e8654c2b5a1cb3af2a8cf7` |
| frozen fixture | `ed3f8f00aa503d91275167f4a968b232b8d07b7e3a9cb1021af715fe78cf6bf0` |
| atomic patch manifest | `f1e472e597dc846a3d366fe1514fbc6bcb50599ef5cd3d56a9d88a9c514559d2` |
| patched runner | `32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447` |
| atomic patched tree | 1,439 files; `7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260` |

An independent rebuild from the 2,401-file qualified source recovered the
same 1,437-file atomic source selection and 471 explicitly inventoried binary
or data grafts. Its 1,439-file output reproduced the manifest, runner, and full
tree seals above.
