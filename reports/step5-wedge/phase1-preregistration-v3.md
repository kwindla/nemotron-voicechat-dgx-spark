# Step 5 wedge Phase 1 preregistration — v3 compatibility reseal

**Seal date:** 2026-08-14 UTC  
**Status:** pre-execution compatibility reseal only

This v3 seal incorporates the complete v2 preregistration at
`reports/step5-wedge/phase1-preregistration.md`, whose SHA-256 remains
`f38b02c8c171489199685aa802d9ff176c689e762ff9851eeedce6c7556879a5`.
The round-2 review returned PASS. No lane, case, oracle, exactness,
materialization, terminal-A, two-discordance, disposition, fallback, or resume
rule is changed by this amendment.

The sole compatibility change is the application revision:

- v2 revision: `15bcb8e5fb16dfa1fa987def60e7c130c00c82bc`
- v3 revision: `cf0e9ca89d2048717a83c15570c2c39630b7e558`

The revision moved for unrelated committed work: FHW8 qualification and
promotion, the finding that the observed tool-call loop was pre-existing, and
the bounded tool-loop/pre-EOU recovery defect fixes. An independent pre-seal
audit rehashed every round-2 frozen Phase 1 input. All twelve hashes matched
the v2 completion record exactly. The commit-range path intersection with
those frozen inputs was empty. The qualified vLLM index remained
`8854e6eb87c7adec5ed0cdba9a9b9d5f16cecf62182c8e928be91058aabc1117`,
and all 966 indexed files independently recovered qualified-tree SHA-256
`d66baad763ef5fdd021047391f8b76718f7c97115993bb080a663709c34259e2`.

The campaign must still bind a non-null real executor image, model,
configuration, and harness-computed fixture-materialization SHA-256 before
replicate 0. Synthetic fixture hashes or the CPU fake executor are inadmissible.
Failure to materialize and seal exactly two real 4480-wide rows and complete
frame-22 state remains a zero-record `INCOMPLETE` disposition.
