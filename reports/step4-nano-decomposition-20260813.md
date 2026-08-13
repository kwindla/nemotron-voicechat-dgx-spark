# Step 4d Nano interface decomposition — 2026-08-13

## Preregistration design and provenance

The machine-readable `preregistration.json` was written at 18:28:08 UTC, before
the 18:33:59 preflight health and container-inspect records and before any
component timing or output run. The detailed report prose was sealed later, at
18:34:53, after that health/inspect preflight and after the 18:34:33–18:34:36
container stop. The earlier claim that this prose preceded every Step 4d health
query is withdrawn.

The stored report seal is
`b90074e0e455dca2fa0fb4c7a54fcfea1a7220189e34d4035abd0e40e5b06ccf`,
but no byte-exact copy of the sealed report was retained and the current,
subsequently revised preregistration section does not reproduce that hash. An
auditable prose diff is therefore unavailable. The JSON is the authoritative
retained proof for the fixed pairs, warm-ups, 30 exact blocks, ten Nsight blocks,
block uncertainty unit, seed, and bootstrap count. It does not preregister a
graph-table-purity rule or exclusion of a range when graph and kernel activity
tables coexist. No Step 4d timing or output observation preceded the JSON.

The design of record is the Step 4d paragraph in
`docs/reviews/step4-ab-measurements-review-r3.md`: first decompose the dominant
approximately 52 ms delivered-generation Nano interface on captured, matched
content- and PAD-position states into GPU kernel time versus CPU/launch/gap time,
and pair that decomposition with an exact-output component benchmark. Candidate
ranking is out of scope until this measurement supplies both a recoverable ceiling
and a correctness constraint.

### Question, captured state, and matching rule

The primary question is how much of the released W8 Nano component's one-position
interface wall time is occupied by GPU kernels and how much is non-kernel
CPU/launch/gap time. The source state is the already retained deterministic public
Nano evaluation capture at
`~/.cache/nemotron-voicechat/conversion-corpus/evaluation/0`. Its metadata marks
capture timings invalid because tensor retention synchronized the GPU; therefore
the old capture supplies inputs and expected outputs only, never timing evidence.
The measured component is the released artifact at
`~/.cache/nemotron-voicechat/artifacts/release/nano` loaded through the same
`create_model(engine_type="vllm_llm", ...)` pattern as
`tools/conversion/nano_component_gate.py`.

A content state is a captured one-token decode whose greedy predicted text token
is lexical (`token_id > 12`). A PAD state has predicted text token 12. Controls
BOS=1 and EOU=2 are neither content nor PAD. A matched pair must:

- belong to the same captured request and sequence epoch;
- use `decode_steps=1`, no prompt-token override, identical one-token input-embed
  shape and dtype, and consecutive `current_step` values;
- be the last lexical content state in a contiguous lexical run followed
  immediately by a PAD state; and
- be replayed causally from call zero so vLLM's hybrid Mamba/KV state is the state
  induced by the exact retained prefix, rather than reconstructed from the final
  embedding alone.

Applying that rule to the manifest before timing selects exactly three pairs:

| pair | content call / step / expected token | PAD call / step / expected token |
|---:|---|---|
| 1 | 45 / 45 / 1046 | 46 / 46 / 12 |
| 2 | 326 / 326 / 1046 | 327 / 327 / 12 |
| 3 | 685 / 685 / 1046 | 686 / 686 / 12 |

The selection is fixed. No state will be replaced or dropped after observing a
timing. Every replay block gets a fresh public request ID and advances calls
0 through 686 in manifest order. A block is valid only if all 687 calls complete,
the six selected calls retain their expected token and exact output contract, and
request cleanup succeeds. Invalid blocks remain visible and are excluded as whole
blocks; no individual fast or slow call is deleted.

### Decomposition method and run counts

Two standalone component phases use the unchanged released W8 artifact. Each
phase performs **two complete causal warm-up blocks**, excluded from estimates.

1. **Exact-output phase:** 30 independent measured replay blocks. The first valid
   measured block fixes a canonical, dtype- and shape-sensitive byte digest of
   every production-observable Nano interface output at the six selected states.
   All remaining blocks must match each reference digest bit for bit. The harness
   also checks every replayed predicted token against the retained capture. Hashing
   and host copies occur after the interface timing boundary; exact-output timing
   is not used for the kernel decomposition.
2. **Nsight Systems phase:** 10 independent measured replay blocks. CUDA profiling
   is enabled only around each of the six selected interface calls, and each call
   has a unique NVTX range containing block, pair, class, call, and step. Nsight
   Systems traces CUDA, NVTX, and OS runtime activity for the process tree so the
   vLLM worker's kernels and the caller's interface wall range share one global
   timestamp domain. The raw `.nsys-rep` is exported to SQLite and retained.

For each selected NVTX wall range, the analyzer clips GPU kernel intervals to the
range and takes their interval union; overlapping kernels are never double
counted. It reports interface wall ms, summed kernel ms, kernel-union busy ms,
kernel count, launch/API count and duration, GPU memory-operation time when
present, and **non-kernel residual ms = interface wall ms - kernel-union busy ms**.
The requested CPU/launch/gap term is this non-kernel residual. Because that
residual can also contain non-kernel GPU memory activity, such activity is
reported separately and prevents an unsupported claim that every residual
microsecond is CPU work. Range/kernel correlation is by timestamp containment in
the shared Nsight clock, not by caller CUDA correlation ID across processes.

Nsight Compute is not required for the first decomposition. It will be invoked
only if the Systems result leaves a specific kernel family eligible for later
tuning; no Compute metric is allowed to substitute for the preregistered Systems
wall/kernel split.

### Estimands and uncertainty units

The independent replay **block** is the uncertainty unit in both phases. Calls,
kernels, and the three pairs inside a block are repeated measures, not independent
samples. For content and PAD separately, the primary block value is the mean of
the three matched states. The primary contrast is the within-block PAD minus
content difference. Report the 10 block values, their median and arithmetic mean,
and deterministic 10,000-resample block-bootstrap 95% percentile intervals for
the mean (seed 20260813). State/pair summaries are descriptive and use no
frame- or kernel-level confidence interval. Timing units are milliseconds;
fractions are percentages of interface wall time. Exactness units are valid
blocks/states and bitwise digest agreement counts.

The kernel-tuning recoverable ceiling is the observed kernel-union busy time and
fraction (the deliberately impossible limit where selected kernels cost zero).
The launch/gap/fusion ceiling is the non-kernel residual and fraction (the
deliberately impossible limit where uncovered interface time costs zero). These
are upper bounds for candidate classes, not predicted speedups and not additive
when a candidate affects both terms. A candidate cannot be ranked above its
applicable measured ceiling.

### Correctness constraint and decision rule

The exact-output reference includes the released production interface result:
predicted text token plus every returned function/control output, tensor dtype,
shape, and row-major bytes. The released artifact intentionally omits full text
logits under its qualified greedy-sampling contract; this benchmark does not add a
diagnostic text-logit output that would change the production kernel workload.
Future kernel tuning, fusion, or launch-gap candidates must reproduce all six
reference output digests in all 30 independent causal replay blocks, preserve all
687 captured greedy tokens per block, and introduce no invalid block. Anything
less is not exact output and is ineligible for ranking. This is a necessary
captured-state constraint, not a claim of correctness on arbitrary prompts.

The measurement is complete only if all 10 Nsight blocks and all 30 exact-output
blocks are valid, all selected states are present, each Nsight range has a positive
wall duration and at least one GPU kernel, every decomposition closes within
0.05 ms after interval clipping, and the exact-output gate passes. Otherwise the
result is `INCOMPLETE`; no candidate ranking follows from partial traces.

### Operational safeguards

Immediately before any live capture or standalone component work, `/health` must
prove `active_client:false`; that response and container inspect are retained.
Standalone profiling is expected to require the GPU memory occupied by
`nemotron-voicechat-model`. If so, the container may be stopped (never rebuilt),
and the stop time, prior identity, inspect record, and exact command are retained.
After profiling it must be recreated with the retained
`start-step2-container-v4.sh` invocation plus the read-only config mount present
in `start-step2-container-v3.sh`; inspection shows the v4 script already includes
that exact mount. The restarted `/health` must match the qualified checkpoint,
model, Nano, EarTTS, image, configuration, and optimization identity recorded
before the stop, and must again report no active client.

No image will be rebuilt. The report will explicitly record whether a stop was
needed, every deviation or failed run, raw profiler and benchmark outputs,
artifact hashes, validation results, and the final service-restoration evidence.

## Results

**Decision: `INCOMPLETE`; do not rank a Nano kernel, fusion, or launch-gap
candidate from this run.** The corrected analyzer produces a complete ten-block
descriptive timing split. The production W8 replay nevertheless diverged from
the retained causal token prefix at call 39 in every authoritative exact-output
block, so the exact-output completion gate remains failed. All observations,
including failed and superseded attempts, are retained.

### Captured states and component identity

The fixed selector reproduced the three preregistered pairs without substitution.
All six payloads are BF16 `[1, 1, 4480]`, have `decode_steps=1`, carry no prompt
override, and passed their payload hashes. The 1,172-call source manifest and the
selection record are in `state-selection.json`; the old capture timing was not
used.

The measured artifact was
`~/.cache/nemotron-voicechat/artifacts/release/nano`, whose `config.json` SHA-256
is `2a0576a8b71aa09a5a6683d5a0d2e3fc099839ea9fe559d69a3458e828530391`.
It was loaded as `vllm_llm` with CUDA graphs enabled, BF16 compute, and the
qualified greedy sampling contract. Its composite Nano identity is
`3acecd057f7367768d621bec649cbe6cbe327ca0857c17f86f7e0929f2bb2d99`.

### Descriptive interface decomposition

Nsight Systems 2025.6.3 observed all 60 expected NVTX interface ranges, and all
**60/60** contain a nonempty `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` execution
envelope. Seven ranges in blocks 2, 3, 6, and 8 also contain explicit `KERNEL`
records. The original analyzer inspected `KERNEL`, `CONCURRENT_KERNEL`, and
`GRAPH_TRACE` in that order and selected the first existing table. It therefore
discarded the graph table in those seven dual-table exports, mislabeled them
non-graph, and excluded four whole blocks when `require_cuda_graph` was enabled.
Those exclusions were an analyzer table-precedence artifact, not CUPTI
graph-envelope loss and not a preregistered rule.

The corrected analyzer inspects every nonempty GPU activity table. A graph trace
is authoritative for the graph-envelope estimand; any coexisting explicit
kernels are retained separately, and their union outside the graph plus the
combined observed-GPU union are calculated without double counting. All ten
planned timing blocks are consequently valid and enter the estimates below.

For graph-enabled vLLM calls, Nsight exposed the device-side CUDA-graph execution
envelope rather than every child kernel as separately attributable intervals.
The reported GPU term is consequently a conservative **graph-envelope upper
bound** on kernel-tunable time: it can include gaps internal to the graph. The
residual is wall minus that envelope and is a conservative upper bound for
caller/launch/outside-graph gap work. It must not be interpreted as a pure CPU
measurement. This preserves the ceiling interpretation but does not establish a
complete per-kernel decomposition.

All values are the mean of the three matched positions within each independent
causal replay block. Intervals are the preregistered 10,000-resample block
bootstrap 95% percentile intervals for the mean, seed 20260813.

| term | content mean (95% CI), ms | PAD mean (95% CI), ms | PAD − content mean (95% CI), ms |
|---|---:|---:|---:|
| interface wall | 53.561 (53.259, 53.867) | 53.595 (53.295, 53.945) | 0.035 (-0.462, 0.518) |
| GPU graph envelope | 45.847 (45.581, 46.159) | 45.579 (45.402, 45.777) | -0.268 (-0.683, 0.114) |
| wall − graph envelope | 7.714 (7.462, 7.970) | 8.016 (7.781, 8.256) | 0.303 (-0.064, 0.652) |

The corresponding mean state-level GPU-envelope fractions are 85.61% for
content (95% CI 85.16–86.03%) and 85.05% for PAD (84.68–85.43%). Wall-minus-
envelope fractions are 14.39% (13.97–14.84%) and 14.95% (14.57–15.32%),
respectively. The ten block-level values retained in
`nano-interface-decomposition.json` are:

| block | content wall | PAD wall | content GPU envelope | PAD GPU envelope | content residual | PAD residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.477 | 52.974 | 46.954 | 45.422 | 7.523 | 7.553 |
| 1 | 52.691 | 53.191 | 45.573 | 45.500 | 7.118 | 7.691 |
| 2 | 54.200 | 53.928 | 45.793 | 45.617 | 8.407 | 8.311 |
| 3 | 53.602 | 53.578 | 45.679 | 45.140 | 7.923 | 8.438 |
| 4 | 53.853 | 53.075 | 46.266 | 45.223 | 7.587 | 7.852 |
| 5 | 53.335 | 53.183 | 45.276 | 45.673 | 8.060 | 7.510 |
| 6 | 53.039 | 54.201 | 45.746 | 45.672 | 7.293 | 8.529 |
| 7 | 53.426 | 53.557 | 46.072 | 45.429 | 7.355 | 8.128 |
| 8 | 53.504 | 53.562 | 45.257 | 45.903 | 8.246 | 7.659 |
| 9 | 53.480 | 54.703 | 45.856 | 46.213 | 7.623 | 8.491 |

These observations place an impossible-zero-cost ceiling of approximately
45.85 ms (85.6%) on the content graph envelope and 7.71 ms (14.4%) on content
wall-minus-envelope work; the corresponding PAD ceilings are 45.58 ms (85.1%)
and 8.02 ms (14.9%). They are neither additive predictions nor ranking evidence.
The wall-minus-envelope means exceed the plan's 1.5 ms wrapper/launch screening
floor, so a later launch/gap investigation is not screened out, but the failed
exact baseline keeps it blocked and the residual is not a predicted gain.

Across all ten blocks, explicit kernels add an average 0.087 ms for content and
0.263 ms for PAD outside the graph envelope. Unioning those auxiliary intervals
with the graph without double counting gives all-observed-GPU means of 45.934 ms
for content and 45.842 ms for PAD, leaving 7.627 ms and 7.754 ms respectively.
This sensitivity view does not replace the authoritative graph-envelope
estimand. In particular, wall minus graph is not pure CPU time in the seven
dual-table ranges. The matched contrast does not support a material content/PAD
wall-time difference in these ten blocks.

Nsight Compute 2026.1 was available but was not run: no kernel family remained
eligible for ranking after the failed correctness gate, exactly as
preregistered.

### Exact-output component benchmark and constraint

The authoritative exact-output phase completed two warm-up and 30 measured
687-call causal replays. At the six selected states, all 180/180 measured
production interface outputs matched bit for bit, including token, function
tokens, function logits, dtype, and shape. The content digest at all three pairs
is `750c398d605368abb14d36d0fe25893ac416236edee88bb1cd2079cdd2bae8b5`;
the PAD digest is
`742dd46efe70fa686228ce62858717a2752355388b70446352998998bc7c9a1f`.

Nevertheless, every authoritative block—including both warm-ups—produced token
1044 at non-selected prefix call 39 where the retained capture expects 10592.
Thus all 30 measured blocks are invalid under the preregistered causal-state
constraint. The six later selected outputs happen to reconverge, but their state
cannot be claimed to be the matched retained state after an earlier token
divergence. `exact-output-benchmark.json` therefore correctly records
`passed:false`.

The attempt history exposes an additional reproducibility concern. The first
run, which incorrectly included private engine full-logit tensors in its digest,
matched the retained 687-token prefix; the corrected production-interface run
did not. The first profiler run matched tokens but perturbed the private full
logits, the second profiler run matched all tokens and selected outputs, and the
third profiler run again diverged at call 39. These are separate fresh-engine
runs of the same released graph-enabled artifact. The overbroad digest was fixed
because private engine logits are not part of the production interface and would
alter the requested workload; the original result remains retained rather than
discarded.

The correctness constraint for any future candidate is consequently stricter
than “the six outputs look the same”: first establish a reproducible released
baseline across fresh graph-enabled engines, then require every retained prefix
token plus the six dtype/shape/byte-sensitive production output digests to match
in every replay block and profiling context. Kernel tuning, fusion, or gap work
that does not clear that gate is ineligible for ranking.

### Operational ledger

The mandatory pre-capture `/health` check reported `status:"ready"` and
`active_client:false`. At 2026-08-13 18:34:33 UTC, the running
`nemotron-voicechat-model` container was stopped to release GPU memory for the
standalone component phases. Docker completed the stop at 18:34:36 UTC; the
container exited 143 after the requested SIGTERM. Its full pre-stop inspect and
health response and a structured stop record are retained under `preflight/`.
The stopped container ID was
`6efe0b3ed9e24df9830a3b5666652ec57d3991d01a3bbf4a876b2dfd1b7ff650` and
its image ID was
`sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`.
No image was rebuilt or removed.

The first standalone startup failed before inference because Transformers tried
to write its dynamic-module cache beneath a read-only Hugging Face mount. The
retry set `HF_MODULES_CACHE=/tmp/voicechat-hf-modules`; this changes only the
writable Python-module cache location, not model files or inference settings.
The failure log is retained.

Three profiler attempts were retained. Attempt 1 used the superseded private
engine-output digest. Attempt 2 used the correct production boundary and is the
source of the descriptive timing above. Its original analysis incorrectly called
seven dual-table ranges intermittent CUPTI CUDA-graph propagation loss; raw
inspection and the corrected analyzer show graph envelopes in all 60 ranges.
Attempt 3 added explicit profiler settling outside the timed NVTX range. Its old
analysis similarly gave the kernel table precedence, although all 60 retained
ranges contain graph and explicit-kernel tables; attempt 3 also reproduced the
call-39 prefix divergence and is not substituted for the designated attempt-2
timing population. The attempt-specific JSON files preserve those superseded
analyzer outputs as history; the canonical corrected files are
`nano-interface-decomposition.json` and `summary.json`.

After profiling, invoking the retained v4 path directly failed because its file
mode is non-executable and created no container. Running the same retained script
through Bash succeeded. The exact working invocation was:

```text
bash /tmp/claude-1000/-home-khkramer-src-nemotron-voicechat-dgx-spark/9dab1982-03af-4f77-9d25-ddf7139af66f/scratchpad/start-step2-container-v4.sh
```

Inspection of both retained scripts confirmed that v4 carries the v3 config
mount, and post-restart inspection proves
`/home/khkramer/src/nemotron-voicechat-dgx-spark/config` is mounted read-only at
`/usr/local/lib/python3.12/config`. The recreated container is
`3cdd25015787cb8b83dfadaef13a4e153b788915bcdcbf8cb287c33a7b13e95d`,
using the unchanged image ID
`sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`.
The qualified post-restart `/health` response is `ready` with
`active_client:false` and matches the pre-stop revision
`fb0f94eaf4d03ddc430f39565229393fa1b50c26`, model SHA-256
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`,
Nano SHA-256 `3acecd057f7367768d621bec649cbe6cbe327ca0857c17f86f7e0929f2bb2d99`,
and EarTTS SHA-256
`6e103b3cbcbb9594c9f01a37328ff72e9628962ccaf3fbd01e0b1937a40faf3e`.
Only process IDs and warm-up timing changed. No image was rebuilt.

## Artifacts and reproducibility

The machine-readable summary is `summary.json`; the authoritative exact gate is
`exact-output-benchmark.json`; and the authoritative Systems analysis is
`nano-interface-decomposition.json`. Raw `.nsys-rep` files, SQLite exports,
stdout logs, attempt-specific analyses, selection evidence, preregistration JSON,
pre/post service identity, container inspection, and stop evidence are retained
under `reports/step4-nano-decomposition-20260813-data/`. The pre-run report
SHA-256 in `preregistration-report.sha256` is the historical seal of an earlier
report state. Because that exact state was not retained, it does not prove that
the current revised prose preceded preflight or permit an auditable diff. The
earlier `preregistration.json` provides the narrower machine-readable provenance
described above. `artifacts.sha256` inventories the retained data files.

The reusable harness is `tools/benchmark/nano_interface_decomposition.py`; it
provides fixed state selection, causal exact/profile replay, dtype/shape/byte
digests, NVTX labeling, Nsight SQLite decomposition, whole-block validation, and
deterministic block-bootstrap summarization. Unit coverage is in
`tests/runtime/test_nano_interface_decomposition.py`.

## Validation

The original retained validation record reports task-scoped Ruff passing, six
focused unit tests passing, 67/67 qualification anchors passing, and a final full
suite excluding `browser_e2e` of 1166 passed, 15 skipped, three deselected, and
29 subtests passed. The first full run had one recorded asynchronous websocket
poll failure; its isolated rerun and the final full rerun passed.

For this closure, the updated task-scoped Ruff check, eight focused Step 4d tests
(including mixed graph-plus-kernel precedence and prefix-divergence exact-gate
regressions), the canonical 67-anchor verifier, and the full non-browser suite
were rerun. They passed with 8/8 focused tests, 67/67 anchors, and 1168 passed,
15 skipped, three deselected, and 29 subtests passed in the full suite. The exact
commands and results are recorded in `validation.txt`; the structured anchor
output is `validation-anchors.json`.
