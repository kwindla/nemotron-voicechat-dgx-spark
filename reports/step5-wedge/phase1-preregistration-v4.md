# Step 5 wedge Phase 1 preregistration — v4 executor reseal

**Seal date:** 2026-08-14 UTC  
**Status:** real-executor and trace-compatibility reseal; campaign not run

This v4 seal incorporates the complete v3 compatibility reseal at
`reports/step5-wedge/phase1-preregistration-v3.md`, SHA-256
`8c3494e8f755f9b6f252ba2306b34af7fcd8cecd38e18285a83ee65f202c1d64`.
It does not change the v2/v3 lane order, case matrix, oracle, exactness items,
replicate count, terminal-A rule, two-discordance threshold, disposition, or
resume rule.

The compatibility identity now freezes the implemented process-isolated GPU
executor and its device worker, in addition to the existing harness files:

- real executor SHA-256:
  `de5449f47f3e6f54d184ac9436c87a3f1b92b8cb07f15561853de1cb38e663c9`
- GPU worker SHA-256:
  `35a735a59beb951620c850e97fbaa339a1d5aea9b13a4e9a6cbbe21b9d276bbe`
- sealed executor configuration SHA-256:
  `3f1e179791634d14221d5d0ed8a9961072307c7100ead2c0f11d2ccdceef118e`

The executor identity returned before campaign execution additionally binds the
runtime image, model composite, canonical configuration, patched vLLM tree,
these executor/worker sources, and the harness-validated live fixture
materialization SHA-256. A missing implementation hash is now rejected by the
compatibility-identity constructor.

The authorized structural GPU smoke exposed one type-compatibility error in
the B5 trace detail: this vLLM revision represents `num_draft_tokens` as a
Python list, while the original trace-only expression invoked tensor `.sum()`.
The trace adapter now uses built-in `sum(...)`; it changes only the recorded
`proposed_count` detail and leaves scheduling, dispatch selection, model
forward, sampling, Mamba commit/rollback, and output routing unchanged. The
atomic patcher was rerun from the unchanged 966-file qualified escrow tree.

- patcher SHA-256:
  `7ca45a0b7db8070859caaf8d2236be0aa5f1efabcf174c00912db0a377b2afde`
- rebuilt patch manifest SHA-256:
  `2e5ab58a0970ea15a3fe65ea514994ab5638dbd3abdb3bcb51818dcdadf156a6`
- rebuilt patched runner SHA-256:
  `b73d2876650d38f7acd0da823f3320ad4b5ab80bf9b87555f3fecaa0f5ddb2d4`

The final one-engine S-lane structural smoke receipt is
`reports/step5-wedge/phase1-executor-smoke/smoke-result-v9.json`, SHA-256
`f51f9de93153d041a2d1ad8e30c159cba9ad4568f389cedc92820693a10e4c49`.
It records exactly one real engine construction at GPU memory utilization
0.42, all 13 S cases complete with snapshots, and canonical B0--B9 for every
case. Its synthetic real-model PAD embedding is explicitly marked
`campaign_admissible: false`; it cannot substitute for the live PCM-derived
materialization seal required before replicate 0.
