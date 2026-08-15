# Step 5 wedge Phase 1 independent calibration campaign — v1

**Run date:** 2026-08-15 UTC  
**Verdict:** **REJECTED — no Phase 1 envelope is authorized**

## Pre-run seal and population

The calibration contract was sealed before any engine construction at
`reports/step5-wedge/phase1-calibration-contract-v1.json`, SHA-256
`c22f3cb477c898fe3f4caa655c751345b363df47c509aa5960b166fb36b4ee38`.
It predeclared exactly five fresh single-engine S+A constructions, all twelve
applicable non-causal case/boundary classes, delta orientation
`candidate_packed_minus_sequential_s`, the complete signed/count/norm statistic
set, exact non-waived fields including committed logical KV, and a fail-closed
acceptance rule. The twelve Phase 1 replicates were not used and remain held
out.

The first launch failed before worker construction because a host-absolute
read-only trace mount was absent. The corrected mount then selected an
incomplete 968-file temporary patch tree and failed in materialization before
S execution because `vllm._C` was absent. Neither launch produced a lane
observation or receipt. The successful launcher used the authoritative
1,439-file tree with patch manifest
`f1e472e597dc846a3d366fe1514fbc6bcb50599ef5cd3d56a9d88a9c514559d2`
and runner
`32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447`.

## Retained campaign

The campaign retained five fresh worker PIDs and five distinct live-PCM
fixture-materialization seals. Every construction completed all 13 S cases and
all 13 released-A cases; the twelve non-causal case/boundary comparisons each
retained full directional per-leaf statistics. A returned from the terminal
causal case in these calibration constructions, but that case remained outside
the predeclared numeric-envelope population and cannot supply an
artifact-matched causal oracle.

The first runner version stopped after the first rejected construction. That
observation was preserved verbatim. A mechanical resume correction then ran
the remaining four rather than discarding or replacing the first result. The
receipt records the runner SHA per construction:

- construction 0:
  `18788387c4dae0b733771efdb5d0e92832e210233465fa34a4b1d72bb5b30c8b`;
- constructions 1–4:
  `2887e3d22157844553c727ab7471eb9789145bfd1bea2afcf6569b5283074298`.

This tooling correction cannot turn the failed acceptance rule into a pass.

## Decisive rejection

All five constructions produced the same 40 exact-field failures: all four
committed logical KV leaves diverged at each of these ten boundaries:

1. `text-only-rejection-pending-drain`;
2. `function-only-rejection`;
3. `function-only-rejection-pending-drain`;
4. `conjunctive-rejection`;
5. `conjunctive-rejection-pending-drain`;
6. `agent-bos`;
7. `agent-eos`;
8. `function-sotc`;
9. `function-eotr`; and
10. `post-fc-bos`.

The first `text-only-rejection` retained exact committed KV, matching the two
pre-amendment smoke observations. The independent calibration shows that this
first-boundary result does not persist after the packed/sequential recurrent
states advance. For example, at the first pending drain the four KV leaves
already differ by roughly 1,883–1,984 BF16 elements in each construction, with
per-leaf maximum absolute deltas up to `0.328125`. Divergence then accumulates
across later cases.

The sealed acceptance rule required exact committed logical KV. It therefore
rejects every construction and the campaign as a whole. Widening the numeric
weakening to KV after seeing this result, discarding the first observation, or
changing the case sequence would be post-hoc recalibration and is not done.

## Disposition

- No `phase1-directional-envelopes.json` was emitted.
- No calibration statistic is authorized as a Phase 1 acceptance envelope.
- The twelve Phase 1 replicates remain unrun and held out.
- Phase 1 scheduling remains prohibited.
- The only safe next action is adversarial review and, if desired, a newly
  justified independently sealed oracle/contract. This failed campaign may
  not be silently reused as a pilot for widening the same acceptance rule.

## Evidence seals

| Artifact | SHA-256 |
|---|---|
| calibration contract | `c22f3cb477c898fe3f4caa655c751345b363df47c509aa5960b166fb36b4ee38` |
| five-construction receipt | `a5a45dc07ad5981c77600b7bd68b9f3bb40899835937168fcf4a4602897f0ec8` |
| artifact manifest | `e6451d8713e2f4f8ce6542660408dd39957b03f964f2a6998e235ab969182fe8` |

