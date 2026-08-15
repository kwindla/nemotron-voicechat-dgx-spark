# Step 5 wedge Phase 1 GPU campaign — 2026-08-14

**Verdict: INCOMPLETE — zero scientific replicates; B1 is neither supported nor rejected**

The round-2 design review passed and the GPU window was available, but the
sealed preflight stopped before fixture materialization and before
`create_epoch(0)`. The repository contains no real `Phase1Executor` adapter.
Its only concrete executor is the explicitly synthetic CPU `FakeExecutor`.
Using that executor, inventing the unavailable fixture-materialization hash,
or substituting the unrelated Step-7 fused-input gate would violate the
preregistration. No S, B, or terminal-A lane was run.

This is the preregistered zero-record `INCOMPLETE` disposition. It is not a
negative B1 result, a claim that the wedge disappeared, or a
rejected-with-boundary result.

## V3 compatibility reseal

The source/protocol compatibility identity was resealed at requested HEAD
`cf0e9ca89d2048717a83c15570c2c39630b7e558`. The v3 amendment is
`reports/step5-wedge/phase1-preregistration-v3.md`, SHA-256
`8c3494e8f755f9b6f252ba2306b34af7fcd8cecd38e18285a83ee65f202c1d64`.
It preserves every scientific rule from v2 and changes only the admitted
application revision.

The retained v2 seals are:

- preregistration SHA-256:
  `f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5`;
- dry-run compatibility identity canonical SHA-256:
  `311a251849b62cc359feccfb6d89a1b7d5e71cd864af9a4111e28ba422bd73b9`;
- dry-run compatibility identity file SHA-256:
  `a4263c391b52bcef512d220684e745c83ee6529f471a321d43e66eb1fe187606`.

Before amending the revision seal, all twelve round-2 frozen Phase 1 files
were rehashed. Twelve matched and zero differed. The commit-range path diff
from `15bcb8e...` through `cf0e9ca...` had an empty intersection with those
files. The intervening commits are FHW8 qualification/promotion, the retained
tool-loop diagnosis, and the tool-loop/pre-EOU defect fixes described in the
operator's reseal rationale.

The qualified escrow was also checked independently: 966/966 indexed files
matched, index SHA-256 remained `8854e6eb...`, and the reconstructed tree was
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.
The exact Phase 1 patch manifest was retained byte-for-byte at SHA-256
`818906e4c83ecbd8a62a82193a3df86b563f8fc402d07fbf3453deec1f946786`.

The v3 harness now enforces `cf0e9ca...`, freezes both review rounds plus the
v2 and v3 preregistrations, records the prior v2 seal and reseal rationale,
and emits identity schema 3. Its focused Phase 1/boundary validation passed
62 tests. This amendment does not manufacture the missing real executor.

## Sealed preflight disposition

| Gate | Result | Retained evidence |
|---|---|---|
| GPU available and production stopped | PASS | GB10 visible, 0% utilization, no active compute process, zero running containers; prior production identity retained at `phase1-preflight/production-health-before-stop.json` (SHA-256 `da0c5e17...`) |
| Frozen sources and qualified tree | PASS | 12/12 frozen hashes and 966/966 escrow files matched |
| Pair-only patch identity | PASS | manifest SHA-256 `818906e4...`, runner `2ce42b6f...` to `1cc698d1...` |
| Real executor binding | **INCOMPLETE** | zero real `Phase1Executor` subclasses; only `FakeExecutor` exists |
| Two fused 4480-wide rows | NOT RUN | no admissible materializer |
| Complete frame-22 KV/Mamba/shadow state | NOT RUN | no admissible exporter/materializer |
| Fixture materialization seal | NOT WRITTEN | fail-closed stop |
| Replicate ledger | NOT WRITTEN | `create_epoch(0)` was not reached |

The nearby `step7_pair_bitwise_gate.py` is not an adapter substitute. It reads
a different, already-materialized silent-turn Nano corpus and compares
PIECEWISE with a dedicated FULL graph. It does not replay the retained Phase 1
PCM and typed carrier, export the required frame-22 state, exercise the Phase
1 rejection/control/abort matrix, implement S/B/A restoration, or provide the
terminal-A first-missing-B4 lifecycle.

## Campaign outcome

- planned fresh epochs: 12;
- completed epochs: 0;
- qualifying discordances: 0 (not observed; no denominator);
- B missing/inexact replicates: 0 (not observed);
- exactness failures: 0 (not observed);
- post-forward losses: 0 (not observed);
- causal verdict: `INCOMPLETE`.

No claim can be made about B1 until a reviewed real executor implements every
abstract operation, obtains a real non-null image/model/config/materialization
identity, and passes the existing sealed materialization gate. At that point
the campaign must start again at fresh replicate 0; none of this zero-record
attempt is resumable as a completed epoch.

## Retained evidence and manifest

Evidence is under `reports/step5-wedge/phase1-campaign-20260814-data/`:

- `compatibility-identity-preregistered-v3.json` — v3 source/protocol reseal,
  canonical identity SHA-256 `c9cb992c...`; it explicitly records that the
  executor identity is incomplete rather than presenting a fake seal;
- `frozen-hash-check.json` — the 12-file and 966-file checks;
- `preflight-audit.json` — GPU/stack state and the exact unrecoverable gate;
- `patch-manifest.json` — byte-identical reviewed patch manifest;
- `phase1-summary.json` — zero-record `INCOMPLETE` summary;
- `artifact-manifest.json` — recursive retained-file hashes, canonical
  self-digest `0e7a30c0...`.

The artifact manifest verifies in full. No production service was restored,
as ownership of restoration remains with the operator. No commit was made.
