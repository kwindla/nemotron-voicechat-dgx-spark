# Attention-projection W8 qualification report

Date: 2026-08-16 UTC  
Branch: `agent/reduce-eou-settlement-latency`  
HEAD observed at closeout: `c034bb9bb667c52a6fc7c435c11b00b9096a338b`  
Verdict: **FAIL — G0, G1, and G2 passed; G3 failed on tool routing.**  
Promotion status: **not measured and not eligible for promotion.**

## Seal and construction

The preregistration SHA-256 is
`a21a3773a3096b35692a321c46e3a66a1f8c7f9848b46912a12cbba6719787ba`.
Its recorded timestamp (`2026-08-16T14:16:17.379838488Z`) precedes the
first build report (`2026-08-16T14:20:07.184409175Z`) and every gate run.

`tools/benchmark/build_attnw8_candidate.py` builds from the fhw8 artifact
with exactly 16 attention projection weights. All 16 GPTQ-W8 RTN
round-trips were exact. The worst source-to-dequantized max absolute error
was `0.00341796875`; the worst mean absolute error was
`0.00020855861657764763`.

Two independent builds produced the same SHA-256 for all 13 artifact files.
The artifact manifest SHA-256 is
`b54efd1b35df6f63ad678682c7a11948bfe89c0fc4e62d0fec4f77d0d37d9dc1`;
the live candidate manifest SHA-256 is
`0ee43bcd419c5aa01f5ebaf887114d1eb62cb7d476c4c328e124fc4598be7dbe`.

## Gates

- **G0 PASS.** A fresh graph engine loaded the candidate. Unquantized bf16
  GEMV instances fell from 8 to 0 per measured step; Marlin instances rose
  from 156 to 168. All selected replay tokens were correct.
- **G1 PASS.** The unchanged `fhw8_decision_replay.py` comparator accepted
  three exact-basin blocks per lane and checked 10,548 paired decisions with
  zero text-token, function-token, or conjunction flips. A first fhw8
  attempt with the characterized call-39-divergent signature is retained;
  the fresh exact-basin rerun is the paired baseline used by the comparator.
- **G2 PASS.** Candidate mean content-interface wall time was
  `49.0028746667 ms`, improving `1.5460725333 ms` over the retained fhw8
  baseline (`50.5489472 ms`). The largest non-target family regression was
  `0.0076288 ms`, below the `0.3 ms` limit. The target bf16-plus-Marlin net
  change was `-0.7221424 ms`.
- **G3 FAIL.** Rapid-turn settlement passed 10/10 sessions and 80/80 turns
  with zero empty responses and zero fatals. The canonical browser plan
  passed 8/8 fixtures in one live candidate-stack session. However, the
  unchanged 12-cell tool-call interrupt matrix passed 0/12 cells: every
  correctly transcribed carrier request completed as a normal text response
  without emitting `get_benchmark_word`. There were zero fatal cells, but
  the missing initiating call means the preregistered bounded-call/phase
  gate was not satisfied.

The failed matrix is deterministic across all four requested phase labels
and three replicates. Its retained precedent carrier has SHA-256
`1c7856088512ab3b865665682af1eaa43947ef3b1b403f6f8a13654be0a545ee`.
An earlier invalid attempt using a different carrier is preserved under
`data/g3/tool-interrupt-invalid-wrong-carrier/` and is excluded from the
verdict. Likewise, a browser attempt where the bridge process was reaped
before fixtures began is preserved under
`data/g3/browser-invalid-bridge-lifecycle/` and excluded; the corrected
canonical run used the same plan SHA and passed.

## Validation and operations

The full non-browser suite produced the expected sealed-campaign identity
guard failure and otherwise passed: `1 failed, 1323 passed, 15 skipped,
3 deselected, 29 subtests passed`. Deselecting exactly that non-evergreen
identity guard yielded `1323 passed, 15 skipped, 4 deselected, 29 subtests
passed`. Ruff over `src`, `tests`, and `tools`, bytecode compilation,
`git diff --check`, and the retained 67-anchor verifier all passed.

The original fhw8 service configuration was reconstructed from its captured
preflight `docker inspect` after the candidate session. No commit was
created. Because G3 failed, no counterbalanced promotion A/B was started and
no promotion decision was requested.
