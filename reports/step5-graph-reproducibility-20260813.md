# Step 5 W8 graph-path reproducibility — 2026-08-13

## Preregistration seal

This section and the machine-readable
`step5-graph-reproducibility-20260813-data/preregistration.json` were written
before any Step 5 standalone replay arm was run. Read-only inspection of the
Step 4 evidence, installed source, the retained launch scripts, the running
container identity, and `/health` occurred first so that the arms below use
controls the released stack actually exposes. Those inspections are design
inputs, not Step 5 replay observations. The sealed report and JSON hashes are
retained in `preregistration.sha256`; all later results are appended below the
explicit results boundary without changing this preregistered section.

Preregistered at `2026-08-13T23:00:30Z` (UTC). The evidence inputs fixed at
design time are:

- Step 4 ranking SHA-256
  `83e35fcbda04c2ce3e97aa08cb0cb51ce439fffc13f3f42b4b3bd8229c8624b9`;
- Step 4d report SHA-256
  `757d3d905a4551b3119f12f3351e7c7063fac48990c2b09f701f29beb0214d72`;
- authoritative failed exact replay SHA-256
  `a75b9efe41f5f83f3fb3017c1396a62154f1c32bb43643916bfdcd477198b7d7`;
- retained state selection SHA-256
  `fd3fe3338e61f89f3eab80f6a5276d07cd9a724d5c87038bf7131e9595504daa`;
- Step 4d harness SHA-256
  `ab138889cfac0249b0b5f7f06c88f180f8985e4520e14f74ab0586e82a5efd73`.

## Preregistered question and invariant

The question is why fresh graph-enabled released W8 engines sometimes replay
the retained 687-call causal prefix exactly and sometimes produce token 1044 at
call 39 instead of retained token 10592. Step 4d established that all 180/180
measured selected production-interface states matched bit for bit even in the
divergent engine, but every authoritative block in that engine had the call-39
prefix mismatch. This study must distinguish harmless numerical variability
in a near-tied greedy decision from a graph/runtime state defect. It does not
tune or rank a performance candidate.

Every arm uses the unchanged released W8 weights, BF16 compute, one client and
one active request, the retained call-0-through-call-46 input sequence for the
localization stage, and call 0 through 686 when the confirmation stage is
triggered. No image is rebuilt. The capture supplies inputs and expected tokens
only; its synchronized timings are never used. A replay block uses a new logical
request ID, aborts it at the end, and records cleanup. Any exception, missing
call, failed cleanup, identity mismatch, non-finite diagnostic, or unexpected
concurrent request invalidates the whole engine run but remains retained.

The primary observable is the complete actual token vector and first mismatch.
Secondary observables are per-call production-interface digests at the six Step
4d selected states, engine/request configuration, process and artifact identity,
and (only in the diagnostic overlay arm) top logits and the 10592-versus-1044
margin at call 39. The diagnostic overlay is weight-identical but changes the
graph workload by adding the normally omitted custom text-logit output; it may
localize a decision margin but can never establish production exactness.

## Hypotheses and discriminating experiments

### H1 — graph-capture state dependence

The captured graph may bind or replay state established during graph capture,
including hybrid Mamba/KV descriptors or buffers, so fresh graph engines can
settle into different deterministic basins.

Discriminator: compare independent fresh released engines with identical seed
and inputs under production graph mode (`enforce_eager=false`) against fresh
eager engines (`enforce_eager=true`). Within each engine, run three independent
request blocks. Run three fresh engines per mode (18 localization blocks total).
Also compare each engine's first block with later blocks. H1 is supported if
cross-engine or first-versus-later variation occurs only in graph mode, or if a
graph mode deterministically produces a different token vector while eager is
stable across all engines. It is weakened if eager and graph have the same
between-engine distribution.

### H2 — nondeterministic kernels or atomics during graph replay

A graph-contained reduction, quantized kernel, or atomic may produce small
run-to-run numerical changes. Greedy output changes are expected only where the
top decision margin is sufficiently small.

Discriminator: use the H1 replication distribution. Nondeterminism is supported
if identically configured blocks from the same fresh engine vary, or independent
fresh engines produce multiple token vectors without any seed or history change.
If graph variability is found, run a weight-identical diagnostic text-logit
overlay for one matching and one divergent graph engine when obtainable, plus
one eager engine, recording top-20 and logits for tokens 10592 and 1044 at call
39. Because the overlay changes capture shape, its result is localization only.
Evidence is strongest when the two tokens exchange rank with a small margin and
no cache/history manipulation is needed. Stable engine-specific basins point
more strongly to H1/H4 than per-replay atomics.

### H3 — sampler or seed state

The ordinary vLLM sampler may consume seed/request state even though the public
contract is greedy (`temperature=0`, `top_p=1`, `repetition_penalty=1`).

Discriminator: after the seed-0 graph baseline, run two fresh graph engines with
sampling seed 1, three blocks each. Engine seed remains the released default 0;
only `SamplingParams.seed` changes. If the call-39 or any other token distribution
changes by sampling seed, H3 is supported and the greedy sampler contract is
defective. If it does not, H3 is rejected for this divergence. Python and Torch
seeds are set and recorded as the same arm seed before construction so incidental
parent-side randomness is controlled.

### H4 — KV/SSM cache initialization or cleanup differences

Fresh-engine dummy/capture initialization, block zeroing, request allocation, or
abort cleanup may leave recurrent or KV state different from a long-lived
engine. Correct per-request cache initialization must make replay independent of
prior request history.

Discriminator: in two seed-0 graph engines, compare the first baseline block to
three blocks after a sacrificial full 47-call request and to three blocks after a
different-length 17-call request; all sacrificial requests are aborted and
cleanup is checked. The primary comparison is exact token-vector equality and
first-mismatch location by history class. H4 is supported if prior request
length/history changes the subsequent vector, if the first block differs from
later clean blocks, or if stale-state effects track block allocation. It is
rejected for the observed call-39 divergence if every history class within an
engine is identical and the between-engine state is explained by another arm.

### H5 — vLLM scheduling state

The asynchronous scheduler, request identity reuse, batching/capture-size
selection, or outstanding work may alter the graph path despite a single client.

Discriminator: every primary arm asserts one live logical request. In two graph
engines compare unique request IDs with repeated logical request ID reuse after
verified abort; add an explicit CUDA synchronization after every returned call
in one three-block run. Record backend request counts where accessible. H5 is
supported if ID reuse or synchronization changes the token vector or eliminates
within-engine variability. It is weakened if all scheduling controls are
invariant while eager/graph or cache-history controls discriminate.

## Fixed execution order and adaptive stopping

The order is fixed to avoid result-dependent arm selection:

1. retain pre-stop `/health`, identity, mounts, image ID, GPU state, source and
   artifact hashes; require `active_client:false`;
2. stop (do not remove or rebuild) `nemotron-voicechat-model` only if standalone
   GPU memory requires it;
3. run H1 graph seed-0, then H1 eager seed-0;
4. run H3 graph seed-1;
5. run H4 history arms;
6. run H5 request-ID reuse and per-call synchronization arms;
7. run H2 diagnostic overlays only if production arms exhibit more than one
   token vector or graph/eager disagreement;
8. run full 687-call confirmation on three independent fresh engines for each
   production mode needed by the classification: always graph seed-0; also the
   discriminating control/fix mode when one is identified. Each engine has one
   warm-up and two measured blocks. Require all six production-interface
   digests against the retained Step 4d canonical digests;
9. restore the model with the retained `start-step2-container-v4.sh` under Bash,
   including its read-only config mount, then verify `/health` identity and
   `active_client:false` against the pre-stop record.

An arm may stop early only for a repeated engine-fatal failure or resource limit;
the failure is retained and classification becomes `INCOMPLETE` unless the
remaining evidence already satisfies every clause of one classification below.
No failed or inconvenient engine is replaced silently. Extra localization arms
may be added only after the fixed matrix and must be labeled exploratory.

## Metrics and bounds

For each mode/history/seed cell report engines, blocks, valid blocks, distinct
token-vector digests, exact-prefix rate, first-mismatch call/token distribution,
within-engine agreement, and between-engine agreement. Confirmation reports the
same metrics over 687 calls plus selected production-output digest agreement.

If exact bitwise graph replay is shown impossible in principle, the replacement
gate is preregistered now as a bounded behavioral gate, not chosen after seeing
the observed distances:

- 100% valid blocks and successful cleanup;
- 100% exact agreement for the six selected dtype/shape/byte-sensitive
  production-interface digests;
- no mismatch for BOS, EOU, PAD, SOTC, EOTC, or EOTR control tokens;
- token mismatch rate at most `1 / 687` per block;
- first mismatch no earlier than call 39;
- all divergent tokens must be lexical (`token_id > 12`);
- all blocks must reconverge to the retained token stream within one subsequent
  call, so maximum divergence-run length is one;
- the call-39 diagnostic top-two margin, when measurable without changing the
  production result, must be at most `0.05` logit; overlay-only measurements are
  labeled supportive and cannot satisfy this clause by themselves;
- the bound must hold in 30 measured blocks spanning at least five fresh graph
  engines, with no post-selection or invalid-block deletion.

If the production sampler cannot expose the margin without altering workload,
the margin clause is unresolved and the exact gate remains blocked; it is not
silently waived.

## Preregistered classification rule

Classify **BENIGN-NONDETERMINISM** only when all of the following hold:

1. multiple outputs occur under bit-identical artifact, input, configuration,
   seed, request history, and scheduling controls, including within-engine
   replication or an equivalent kernel-level nondeterminism demonstration;
2. eager/deterministic controls do not reveal stale state, uninitialized cache,
   seed consumption, request-ID dependence, or scheduling dependence;
3. diagnostic evidence locates the change to a numerically near-tied greedy
   choice rather than a corrupted cache/control path;
4. the 30-block/five-engine bounded behavioral gate above passes in full.

Under this classification, bitwise replay is impossible in principle for this
released graph mode and the old exact-prefix gate is replaced only by the
preregistered bounded gate above. Failure of any clause is not benign.

Classify **REAL STATE DEFECT** when a graph/eager, cache/history, seed, request-ID,
synchronization, initialization, capture, or scheduling control reproducibly
selects the divergence, or when divergence violates any control-token,
reconvergence, output-digest, validity, cleanup, or margin safety bound. The
report must identify the narrowest supported faulty layer and a concrete fix
path; a workaround such as eager mode may demonstrate the cause but does not by
itself qualify the released graph baseline.

Classify **INCOMPLETE** if neither rule is satisfied. `INCOMPLETE` leaves the
Step 4 exact gate and both W8 graph-path candidate tracks blocked.

## Operational and validation contract

Everything produced by the study, including stdout/stderr, failed attempts,
pre/post health, container inspection, commands, environment, hashes, and
summaries, is retained under
`reports/step5-graph-reproducibility-20260813-data/` and inventoried by SHA-256.
No image is rebuilt. The final work must pass focused tooling tests, task-scoped
Ruff, the canonical 67-anchor verifier, and the full non-browser suite. No commit
is created.

<!-- STEP5_RESULTS_BEGIN: append observations below; do not edit preregistration above. -->

## Results

**Classification: `REAL STATE DEFECT`.** The released W8 graph-enabled path has
two fresh-engine output basins. A fresh engine is internally stable, but its
call-39 result can be either retained token 10592 or alternate token 1044. The
same released artifact in eager mode reproduced the retained prefix in every
observed engine and block. This graph/eager control selects the divergence and
therefore satisfies the preregistered real-defect rule. It fails the benign rule;
the exact gate is **not** redefined.

### Primary discrimination

The primary 47-call seed-0 matrix produced:

| mode | fresh engines | blocks | exact blocks | call-39 divergent blocks | distinct vectors | within-engine stable |
|---|---:|---:|---:|---:|---:|---|
| production graph | 3 | 9 | 3 | 6 | 2 | yes, 3/3 engines |
| production eager | 3 | 9 | 9 | 0 | 1 | yes, 3/3 engines |

Graph engine `g0-1` returned 10592 at call 39 in all three requests. Graph
engines `g0-2` and `g0-3` returned 1044 there in all three requests. All other
calls through 46 matched. All three eager engines returned the complete retained
vector in all three requests. The 18 graph and 18 eager selected-state digest
comparisons available at calls 45/46 all matched the Step 4d canonical
production-interface digests.

This is not per-request random sampling. Each fresh engine remained in one basin
across new request IDs, newly allocated cache blocks, abort/cleanup, and repeated
replay. The between-engine change occurs only with the graph-enabled/compiled
path in the primary discriminator.

### Hypothesis decisions

| hypothesis | result | discriminating evidence |
|---|---|---|
| graph-capture state dependence | **supported; narrowest demonstrated boundary** | graph seed-0 produced two fresh-engine vectors while eager produced one exact vector; the workload-changing diagnostic overlay supplies supportive greedy-boundary localization only |
| nondeterministic kernels/atomics under graph replay | **not supported as per-replay nondeterminism; construction-time numerical variation remains plausible** | every engine was bit-stable within itself; no block-to-block variation occurred, but fresh graph engines differed |
| sampler/seed state | **not supported; not isolated** | the two fresh seed-1 engines split one exact and one divergent, so there was no observed seed association; different fresh engines were used for seed 0 and seed 1, leaving construction confounded and licensing neither rejection nor a distribution claim |
| KV/SSM cache initialization differences | **history did not move the observed exact basin; cleanup evidence is reassuring, but divergent-engine history and fresh capture initialization remain unresolved** | two graph engines stayed exact across baseline, a full 47-call sacrificial request, a 17-call sacrificial request, and seven observed blocks each; ordinary cleanup left zero requests, but the history arm obtained no divergent engine |
| request-ID reuse and per-call synchronization | **not observed as selectors in the tested single-client, both-basin controls** | one divergent engine remained divergent in 9/9 blocks and one exact engine remained exact in 9/9 across unique IDs, reused IDs after abort, and explicit CUDA synchronization after every call; the sequence was fixed |

Across all authoritative production arms, graph mode had **56 valid blocks across
12 fresh engines: 38 exact and 18 divergent**. All 18 divergences were exactly
`call 39: expected 10592, actual 1044`; no engine changed basin. Eager mode had
**18/18 exact blocks across six fresh engines**. Every one of the **220 compared
production-interface output digests** matched: 148/148 graph and 72/72 eager.
Thus the selected outputs reconverge, as Step 4d observed, but that does not cure
the causal-prefix defect.

### Logit localization

The conditional diagnostic used a weight-identical overlay whose only artifact
change was adding `text_logits` to `custom_outputs`. Its config SHA-256 is
`1a673171131c362a52b7cfee8b03294f46592fd614c902957ba75d0955208dea`;
the source production config remains
`2a0576a8b71aa09a5a6683d5a0d2e3fc099839ea9fe559d69a3458e828530391`.
The overlay therefore localizes but cannot qualify production exactness.

Two of three diagnostic graph engines and the eager diagnostic engine returned
10592 with `logit(10592) - logit(1044) = +0.125` in every block. The third graph
engine returned 1044 with the two recorded BF16 logits exactly tied, margin
`0.0`, in every block. This directly locates the token flip to graph-engine
numerical/state variation at the greedy decision boundary. It does not support
`BENIGN-NONDETERMINISM`: the eager control selects the exact result and engines
are internally stable rather than intrinsically variable per replay. The
overlay's `+0.125` and `0.0` values are supportive localization only because the
overlay changes graph workload. They do not establish production margins and
cannot dispositively pass or fail the sealed `0.05` clause. The production
margin is unresolved; only an unaltered production-sampler measurement can
close it.

### Full-prefix confirmation

The confirmation phase ran three fresh production graph engines and three fresh
eager engines. Each performed three complete 687-call blocks (the first serving
as the warm-up-like observation and two as measured-like observations):

| mode | fresh engines | 687-call blocks | exact prefixes | selected digests |
|---|---:|---:|---:|---:|
| production graph | 3 | 9 | 9/9 | 54/54 match |
| production eager | 3 | 9 | 9/9 | 54/54 match |

All three confirmation graph engines happened to land in the exact basin. This
does not erase the independently replicated graph defect in the primary, seed,
scheduling, and diagnostic graph arms, nor the retained Step 4d failures. It
does show that an exact graph basin stays exact through call 686. It is also
negative evidence against a high or stationary divergence prevalence: across
all production graph arms, 4/12 engines were divergent, rather than the primary
cell's apparent 2/3. Because the arms ran in fixed order and the later
confirmation cell was all exact, a temporal or other uncontrolled construction-
epoch factor remains plausible. These correlated, unequal-block engine samples
do not estimate a random construction probability. No divergent full-prefix
engine was obtained, so post-call-46 behavior in a Step 5 divergent engine and
the preregistered 30-block/five-engine gate remain unmeasured. Eager mode is
exact both in the primary matrix and at full-prefix scope.

### Root cause and fix path

The narrowest supported root-cause boundary is the **fresh vLLM graph-enabled /
compiled decode construction for the hybrid W8 Nano model**. The evidence rules
out the public input capture and released weights. It finds no seed association,
but seed was not isolated from fresh-engine construction. History did not move
the observed exact basin, while request-ID reuse and synchronization did not
move either tested basin; divergent-engine history and capture/fresh
initialization remain unresolved. The evidence does not yet distinguish a full-
graph buffer binding error, capture-time recurrent-state contamination,
compiled-kernel/autotuner choice, or a graph-contained reduction difference.
Calling one child kernel the root cause would exceed the evidence.

The concrete fix path is:

1. use Nano `enforce_eager=true` as the correctness control and immediate safe
   workaround; it must receive its own latency/decomposition and full runtime
   qualification before deployment because it is not the released graph path;
2. bisect vLLM compilation/capture modes on the unchanged artifact: full decode
   graph versus piecewise, input copying (`cudagraph_copy_inputs=true`), capture
   warm-up count, and graph-disabled compile/eager controls;
3. instrument the actual production sampler to retain bounded top-two IDs/logits
   from its existing LM-head evaluation, avoiding the duplicate-head overlay,
   and inspect call 39 across fresh engines;
4. audit graph-capture dummy requests and hybrid Mamba/KV buffer identity and
   zeroing before the first real request, especially fixed-address input/state
   buffers; and
5. accept a repair only after the original exact-prefix gate passes across at
   least five fresh graph engines and 30 measured 687-call blocks, including all
   six production-interface digests and profiling context. This characterized
   defect requires exact-gate repair; the benign bounded gate is excluded, not a
   fallback acceptance route.

Until that repair passes, released graph-enabled W8 kernel/fusion and wrapper
graph work remain blocked. Switching to eager would create a different baseline
whose timing envelopes must be remeasured; it does not unlock ranking by itself.

### Long-lived production-instance qualification

Production uses a long-lived graph-enabled engine, so a construction-associated
basin is operationally relevant rather than averaged independently per session.
On the tested replay, exact- and divergent-basin engines remained instance-
stable across new request IDs, abort/cleanup, block allocation, reuse, and
synchronization. Restarts, worker replacement, scale-out, or new replicas can
construct a different basin.

This does not establish an 18/56 per-session defect rate, a 4/12 production
incidence, or divergence at the 39th call of arbitrary conversations. The
retained causal prefix is one controlled sequence, its observed mismatch
reconverges at the next call through the measured 47-call divergent runs, and
general user-visible semantic or audio impact is unmeasured. Step 5 did not
replay the prefix against the restored long-lived production service, so that
particular running engine's basin is unknown. Production is therefore
potentially affected at the released graph-path correctness level, not proven
universally wrong and not proven unaffected by long-lived operation.

### Tooling, retained evidence, and deviations

`tools/benchmark/graph_reproducibility.py` provides fresh-engine run capture,
token-vector and production-output hashing, divergence-run analysis, history
and scheduling sequences, diagnostic top-logit capture, and aggregate
classification. Selector conclusions are derived from their retained strata.
The aggregate recomputes sealed block/engine validity, including call and block
counts, cleanup/request emptiness, selected-digest agreement, artifact/capture/
control identity, and finite diagnostics. The revised evaluator emits N/A
bounded-rate results for localization blocks. The complete bounded-gate
evaluator enforces 30 measured 687-call blocks, five fresh engines, and a
production-workload margin; here it reports
3 engines, 6 measured blocks, and an unresolved production margin, then excludes
the gate because the classification is `REAL STATE DEFECT`. Unit coverage is in
`tests/runtime/test_graph_reproducibility.py`.

All raw JSON and stdout logs, including failed/superseded attempts, are retained
under `step5-graph-reproducibility-20260813-data/`. The first localization run
used a digest encoder missing the Step 4 tensor type marker; its token vector is
valid but its digest fields are superseded, and the arm was rerun after a
regression test proved encoder equivalence. The first history sequence completed
all intended observations but its report-level `passed` flag compared seven
blocks with the standard three-block count; it too is retained and was rerun
after correction. Neither superseded attempt enters the aggregate counts above.

The service was stopped only after `/health` proved `active_client:false`. No
image was rebuilt. Restoration used the retained v4 script under Bash with its
read-only config mount. The first restoration attempt completed model loading
but stalled at its first EarTTS pipeline-warm-up request before binding the port;
its logs and inspect are retained, it was stopped, and the same unchanged v4
path was retried. Final restoration identity is recorded under `preflight/`.
The retry reached `status:"ready"` with `active_client:false`; revision
`fb0f94eaf4d03ddc430f39565229393fa1b50c26`, model, Nano, EarTTS, and image
digests all match pre-stop, and the repository config is mounted read-only at
`/usr/local/lib/python3.12/config`.

### Validation

Task-scoped Ruff passed. The focused Step 5 plus Step 4d regression set passed
with **27 tests**. The canonical verifier passed **67/67 anchors**. The full
non-browser suite passed with **1190 passed, 15 skipped, three deselected, and 29
subtests passed**. Exact commands and results are retained in the validation
files. `artifacts.sha256` inventories the retained data directory.
