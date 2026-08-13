# Step 4d decomposition measurement-validity review — round 1

**Verdict: FAIL for the published six-block decomposition and for use as a
Step 4e ranking input.** The `INCOMPLETE` standing is correct and must remain.
The exact-output failure is real and honestly gates ranking, the service was
restored to the qualified identity, and the quoted six-block arithmetic
reproduces. The failure is narrower but material: the retained attempt-2 SQLite
files contain a CUDA graph-trace envelope in **all 60/60 ranges**, not 53/60.
Seven ranges also contain explicit kernel records. The analyzer gives the
kernel table precedence over the graph table, labels those seven ranges
non-graph, and then excludes their four blocks. The report consequently calls a
dual-table trace an “envelope loss,” and calls the resulting exclusion
preregistered even though the retained preregistration proof does not establish
that rule.

The raw timing evidence is salvageable without touching the running stack or
necessarily rerunning Nsight. Correct the analyzer and regenerate the summaries
from the retained attempt-2 SQLite files. This does **not** clear the exact-output
gate or make a Nano candidate rankable.

## Scope and evidence integrity

I reviewed the Step 4d guidance in
`docs/reviews/step4-ab-measurements-review-r3.md`, the report, the completion
note, the harness and unit tests, all machine-readable summaries, all 120
attempt-2/attempt-3 SQLite exports, the relevant raw Nsight inventory, the
captured source manifest, and the retained pre/post service evidence. I made no
live request, Docker query, profiler run, or other probe against the restored
stack.

The data directory has 335 physical files: 334 payload files plus the
self-excluding `artifacts.sha256`. All **334/334** manifest entries verify. This
agrees with the completion note's “334 hashed files”; it is not a count defect.

## 1. Preregistration ordering and the four-block exclusion

The outcome-facing design was substantially written before component
measurement, but the current report does not prove the specific exclusion that
was applied:

- `preregistration.json` is timestamped 18:28:08 UTC, before the 18:33:59
  preflight records and the first retained component result. It fixes the three
  pairs, two warm-ups, 30 exact blocks, ten Nsight blocks, block uncertainty
  unit, seed, and bootstrap count.
- The report seal is timestamped 18:34:53, after the retained preflight health
  and inspect and after the 18:34:33–18:34:36 stop. It is still before the first
  retained output/timing run, but it contradicts the report's literal claim
  that the sealed prose preceded “any Step 4d health query.”
- More importantly, the stored seal is
  `b90074e0e455dca2fa0fb4c7a54fcfea1a7220189e34d4035abd0e40e5b06ccf`.
  The current report content through the end of the preregistration section
  hashes to
  `79d1a1bf600ba8f36794a1fd9e20a9d5e986b0d2d2769781e07f27b7935adac1`
  (and to `279d42f...` after normalizing terminal blank lines), so the current
  detailed preregistration is not the sealed text. No retained copy permits a
  diff.
- The early JSON does not specify “exclude a block if any range has a kernel
  table as well as a graph table,” or even a graph-envelope-only eligibility
  rule. The current prose instead says a range needs positive wall time and at
  least one GPU kernel. Every disputed range meets those written conditions.
- `nano-interface-decomposition-attempt2-cupti-propagation-loss.json` initially
  records `passed:true` with no `require_cuda_graph` flag. The later canonical
  analysis sets `require_cuda_graph:true`, and the later summary excludes blocks
  2, 3, 6, and 8. That artifact sequence confirms that the graph-table purity
  rule was introduced after examining attempt 2.

The applied exclusion is outcome-independent in the limited sense that it uses
trace table structure, not fast/slow wall values, content/PAD class, or output
tokens. The excluded blocks have ordinary wall means, and all four blocks are
removed whole. That prevents favorable-call cherry-picking. It does not make the
rule preregistered, and the raw data do not satisfy its stated factual premise.

## 2. Independent recomputation from raw SQLite

For each of the 60 attempt-2 SQLite files, I independently read the sole Step 4d
NVTX range, clipped device intervals to that range, unioned overlaps, and then
formed the preregistered mean of the three positions within each block and class.
The raw table inventory is:

| raw condition | ranges | affected blocks |
|---|---:|---|
| nonempty `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` | **60/60** | 0–9 |
| graph trace only | 53/60 | 0–9 |
| graph trace plus explicit `KERNEL` rows | 7/60 | 2, 3, 6, 8 |
| graph trace absent | **0/60** | none |

The seven dual-table ranges are exactly those listed by the report as losses.
Each nevertheless contains one graph record of roughly 44.8–47.3 ms. The
explicit kernel union outside or adjacent to that envelope ranges from 0.0013 to
2.65 ms. It is reasonable to report those extra kernels separately and to warn
that wall-minus-envelope is not pure CPU time; it is not reasonable to say the
graph envelope is absent.

The implementation cause is direct. `_extract_nsys_sqlite()` searches activity
tables in `KERNEL`, `CONCURRENT_KERNEL`, `GRAPH_TRACE` order and selects the
first table that merely exists. Thus a dual-table export discards the nonempty
graph table from analysis. `summarize()` then excludes the entire block because
the chosen row is labeled `cuda_kernel` rather than
`cuda_graph_execution_envelope`.

Applying that post hoc purity filter exactly does reproduce the published
six-block numbers:

| class | interface wall, ms | graph envelope, ms | wall − envelope, ms | envelope / residual |
|---|---:|---:|---:|---:|
| content | 53.543812 | 45.999404 | 7.544407 | 85.917% / 14.083% |
| PAD | 53.447378 | 45.576580 | 7.870798 | 85.283% / 14.717% |

So the requested rounded splits—**53.5/46.0/7.5 content** and
**53.4/45.6/7.9 PAD**—are numerically correct for blocks 0, 1, 4, 5, 7, and 9.
They are a transparent descriptive sensitivity subset, not the preregistered
ten-block estimand.

Using the graph envelope that is actually present in every attempt-2 range gives
the following ten-block descriptive recomputation before new bootstrap
intervals are generated:

| class | interface wall, ms | graph envelope, ms | wall − envelope, ms | mean state-level envelope / residual fraction |
|---|---:|---:|---:|---:|
| content | 53.560786 | 45.847206 | 7.713580 | 85.608% / 14.392% |
| PAD | 53.595332 | 45.579065 | 8.016267 | 85.053% / 14.947% |

If total observed GPU busy is desired instead, the analyzer must union graph
envelopes with non-overlapping explicit kernels. On all ten blocks that produces
45.933728 ms content and 45.841732 ms PAD GPU union, with residuals of 7.627058
and 7.753599 ms. The report must choose and name one estimand: graph-envelope
ceiling with separately reported outside-graph GPU work, or all-observed-GPU
union. It must not switch tables by schema presence.

## 3. Exact-output gate

The gate is real and is not laundered:

- The authoritative file contains two warm-up plus 30 measured 687-call causal
  replays. All **32/32** blocks record exactly one mismatch at call 39:
  expected token 10592, actual token 1044. The retained source manifest
  independently confirms token 10592 at call 39 and matches the selector's
  manifest hash.
- All **180/180** selected measured production-interface digests match their
  first-measured-block references. Each of the six reference keys has one digest
  across all 30 measured blocks, and the selected tokens reconverge to 1046 for
  content and 12 for PAD.
- Those 180 matches demonstrate repeatability of the six later interface
  outputs within that engine run. They do **not** demonstrate replay of the
  retained causal states after the earlier prefix diverged. The harness marks
  every measured block invalid, `exact-output-benchmark.json` says
  `passed:false`, and `summary.json` says `INCOMPLETE`.
- Across fresh `enforce_eager:false` engines of the same W8 config hash,
  attempt-1 exact and attempt-2 profile runs match the retained prefix, while
  authoritative exact attempt 2 and profile attempt 3 reproducibly emit
  1044 at call 39. This is strong evidence of a real **fresh graph-enabled
  engine reproducibility failure**. It is not enough to claim that CUDA graph
  execution, rather than initialization or another fresh-engine effect, is the
  causal mechanism; no matched eager control was retained. The report's wording
  is acceptable where it describes the graph-enabled path and requires
  reproducibility, not root-cause attribution.

The exact-output gate alone is sufficient to keep Step 4d `INCOMPLETE`, even
after the timing analyzer is repaired. No Nano kernel, fusion, or launch-gap
candidate may be ranked until a fresh-engine released baseline first clears the
full 687-token prefix plus the six dtype/shape/byte-sensitive interface digests
under the relevant profiling context.

## 4. Container stop, restoration, and health identity

The retained operational evidence passes review.

- Pre-stop health is `ready` with `active_client:false`; pre-stop inspect shows
  container `6efe0b3e...`, image ID `sha256:ea15693c...`, and the config bind
  mounted read-only at `/usr/local/lib/python3.12/config`.
- `container-stop.json` records the exact `docker stop --time 30` command,
  18:34:33 start, 18:34:36 completion, and expected SIGTERM exit 143. No retained
  evidence suggests an image rebuild or deletion.
- The v4 script is mode 0664, explaining why direct execution failed. Running it
  through Bash is consistent with the retained command. Both v3 and v4 scripts
  contain the required read-only config mount.
- Post-restart inspect shows new container `3cdd2501...` with the same image ID,
  command, normalized environment, bind set, and read-only config mount. The
  differences are lifecycle identity/order fields such as container hostname
  and mount/environment ordering.
- Post-restart health is again `ready` and `active_client:false`. The retained
  identity diff changes only process IDs and CPU-codec warm-up duration. The
  checkpoint revision, parent model hash, Nano hash, EarTTS hash, runtime
  optimizations, typed-input asset hashes, and disabled pairing policy match.

This review did not query the current container; the conclusion is about the
retained stop/restore evidence, as requested.

## 5. Tooling and tests

The recorded validation is credible: task Ruff passed, six scoped unit tests
passed, 67/67 qualification anchors passed, and the final full non-browser suite
is recorded as 1166 passed, 15 skipped, three deselected, and 29 subtests passed.
I independently reran only the read-only/offline scoped checks: Ruff passed and
the six Step 4d unit tests passed in 0.65 s.

Coverage is insufficient for the defect found here. The SQLite fixtures test a
kernel-only database and a graph-only database, but never a database containing
both nonempty tables—the exact shape of seven attempt-2 ranges and all 60
attempt-3 ranges. There is also no focused test proving that a prefix mismatch
keeps the summary `INCOMPLETE` when every selected output digest matches. The
main report's `## Validation` section is empty even though `validation.txt` and
the completion note contain the results.

## 6. What this does and does not license for Step 4e

After correction, the retained timing data can support a **descriptive ceiling
row**, not a candidate ordering:

- The graph envelope is an impossible-zero-cost ceiling of about 45.85 ms for
  content and 45.58 ms for PAD. It does not identify a tunable kernel family,
  predict a speedup, or establish that the whole envelope is removable.
- Wall-minus-envelope is about 7.71 ms content and 8.02 ms PAD on all ten blocks.
  That is above the plan's 1.5 ms wrapper-launch-path screening floor, so a
  launch/gap investigation is **not screened out**. It is still only a ceiling:
  it includes CPU/wrapper delay, graph-internal interpretation limits,
  synchronization, and in the dual-table ranges up to about 2.65 ms of explicit
  outside-graph GPU kernels. It is neither a predicted gain nor a basis to rank
  wrapper fusion over another candidate.
- Nano kernel/Marlin/fusion candidates remain **blocked**, both because Systems
  gives no per-kernel candidate measurement and because the released fresh-engine
  exact baseline is not reproducible. Nsight Compute was correctly not used to
  bypass that gate.
- **Nano W4 remains deferred.** This W8 decomposition supplies neither a W4
  quality/non-inferiority result nor an exact W4 correctness result, and the W8
  baseline itself fails fresh-engine reproducibility. Kernel-bound appearance is
  not authority to promote a precision change.
- The **tail-pairing row is unchanged**. A one-position component split does not
  measure the amortized two-position schedule, queue-clock recovery, tail-state
  machine, or packed-call wedge risk. It neither promotes nor demotes pairing and
  does not close any of pairing's three reliability preconditions. Pairing may
  retain its separately measured historical latency precedent while remaining
  inadmissible.

Step 4e may therefore list corrected descriptive Nano ceilings with status
`BLOCKED — exact baseline unreproducible`; it may not use them to order kernel,
launch-gap, W4, or tail-pairing rows.

## Required fixes

1. Change the SQLite analyzer to inspect all nonempty GPU activity tables. When
   a graph envelope and explicit kernels coexist, retain the envelope, separately
   report explicit outside-graph kernel union, and define any combined GPU-busy
   union without double counting.
2. Add mixed graph-plus-kernel SQLite coverage and a prefix-divergence/exact-gate
   regression test. The test must fail if a nonempty graph table is discarded
   merely because a kernel table also exists.
3. Regenerate the canonical decomposition, bootstrap intervals, summary, report,
   completion note, and hash manifest from the retained attempt-2 SQLite files.
   Report all ten blocks if the nonempty graph envelopes are the estimand. If a
   profiler-domain reason makes a dual-table range unusable, state that concrete
   criterion and present the six-block result only as a post hoc sensitivity
   analysis—not as envelope loss or a preregistered exclusion.
4. Correct the preregistration provenance narrative. Preserve the exact sealed
   preregistration text or an auditable diff; acknowledge that the health/stop
   preflight preceded the report seal and that the graph-table purity rule is not
   established by the retained preregistration JSON.
5. Keep `INCOMPLETE`, `passed:false`, and the no-ranking decision. Before any
   future candidate measurement, qualify reproducibility across fresh
   graph-enabled engines; use a matched eager control only to investigate cause,
   not to waive the production graph-path gate.
6. Populate the report's empty Validation section from retained evidence and
   distinguish recorded full-suite results from checks independently rerun by a
   reviewer.

No live rerun is required to correct the present decomposition arithmetic. A
future live baseline qualification is required only before Step 4e is allowed to
rank a Nano-changing candidate.
