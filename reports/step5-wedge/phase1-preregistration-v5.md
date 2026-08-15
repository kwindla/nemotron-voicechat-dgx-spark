# Step 5 wedge Phase 1 preregistration — v5 executor round-3 reseal

**Seal date:** 2026-08-15 UTC  
**Status:** round-3 real-executor reseal; campaign not run

This v5 seal incorporates the complete v4 executor reseal at
`reports/step5-wedge/phase1-preregistration-v4.md`, SHA-256
`a97b197b1f80c61b438fa5f0c6c15627ca407f56664ef809a1b79a2203e6160d`.
It does not change the frozen lane order, case matrix, oracle plan, exactness
items, replicate count, terminal-A rule, two-discordance threshold,
disposition, or resume rule. It binds only the narrowed round-3 corrections
required by `docs/reviews/step5-wedge-phase1-executor-review-r2.md`.

The executor now seals `selected_source_row_identity` from the tensor actually
sent to the worker. A pending drain therefore seals its corrected pending row,
and the identity must equal `pending_row_sha256_before`. Correction assumptions
come from the recorded runtime call context captured during fixture
materialization; no harness-created PAD constants can stand in for that state.

The five named control cases now drive the production Nano agent wrapper or
function-call state machine and retain the observed wrapper, turn-context, and
streaming-state transition evidence. Harness-local `control_state` bookkeeping
is absent.

Every materializer and epoch child independently rebuilds and compares the
complete model-artifact identity immediately before constructing Nano or the
pipeline. The parent identity now includes the complete checkpoint tree
(including `config.json`, `model.safetensors`, and
`rnnt_tokenizer/tokenizer.model`), the complete Nano and EarTTS trees, the
complete Hugging Face skeleton tree, the candidate manifest, and the two Speech
runtime files read by `server.py`. The resulting composite model SHA-256 is
`bd12062f1afc92079c92b739c56b542c50778f8edab44b2a3896d5a79fbe6dc2`.
Each completed large-file hash also issues a Linux `POSIX_FADV_DONTNEED`
advisory. This does not weaken or sample the identity; it prevents the complete
model read from retaining tens of gigabytes of page cache immediately before
GPU allocation on the unified-memory host.

Frozen implementation hashes at this reseal are:

- real executor:
  `4ab46deee2857d8f9035df8899ec14ba24b92d85c850b5ebbb0937fc44363779`
- GPU worker:
  `361b275ca56752d2549401dcebf7453092fb59147b33850678a26acfedce6469`
- campaign harness:
  `c6006471147fa3d96c0f2ca3fa8cd83c7bfbf652736a92e1f4d526a6f6bf59ad`
- oracle:
  `5d85f485aaccb2691ef4e5a049d41f7bb68b31714da91b2fb35599471327446b`
- sealed executor configuration:
  `461a6ae7d98b47b845eff0f8e76832ac566c1b459f14d7bec3474e16bd82d518`

The atomic patch was independently rebuilt twice from the unchanged 966-file
qualified escrow. Both 1,439-file results had patch-manifest SHA-256
`f1e472e597dc846a3d366fe1514fbc6bcb50599ef5cd3d56a9d88a9c514559d2`,
patched-runner SHA-256
`32e0c36cd1cf257d622636e79b45e539e72ba7494188b6c69906cd8ed4a81447`,
and complete-tree SHA-256
`7d3614f9d29f002277913e19c80522893d821ae820881e7a7022959762028260`.
This preserves the round-2 closure of item (a).
