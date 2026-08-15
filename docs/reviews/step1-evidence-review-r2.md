# Step 1 retained-evidence final gate review — r2

Date: 2026-08-12 UTC

## Verdict

**PASS. Step 1 is complete.** All four closure conditions from round 1 are
closed under fresh probes. The corrected phase split is exactly 209 true text
rows, 615 PAD rows, and seven agent-control rows; the seven rows removed from
the old text set are exactly the seven retained `agent_eos` rows identified in
round 1. The two bad neighbor labels are fixed, the added tests cover the
requested retained shapes and CLI validation, all 67 anchors and all 30
analyzer tests pass, the reports reproduce exactly, and all 14 source hashes
match. No new finding was uncovered.

## Closure status

| Round-1 condition | Result | Fresh evidence |
|---|---|---|
| 1. Separate agent controls and correct neighbor labels | **CLOSED** | Independent raw parsing gives 209 text / 615 PAD / 7 agent-control, totaling 831. Current JSON memberships match all three raw sets exactly. Both formerly incorrect neighbors are now `agent-control/t2`; no exact `text-emission/t2` label remains in JSON or Markdown. |
| 2. Add retained-shape and CLI tests | **CLOSED** | The synthetic corpus pins explicit EOS, EOS found only through `control_ids`, delivered PAD, non-delivered internal-drain PAD, BOS exclusion, ordinary text, and unavailable-token reconciliation. Separate CLI tests pin successful dual-flag output and rejection before any write when Markdown is requested without Step 1 details. |
| 3. Regenerate reports and rerun gates | **CLOSED** | Fresh analysis of all 14 retained sources is object-identical to the saved JSON. Canonical JSON serialization and analyzer-rendered Markdown are byte-identical to the saved files. `verify-anchors` passes 67/67; the analyzer suite passes 30/30. |
| 4. Correct plan A4 prose | **CLOSED** | A4 now states 12 buffered plus 12 packed-pair rows, packed p95 118.710 ms nearest-rank, and eight sequential/control rows. The stale `13 packed` / `118.27` wording is absent from the plan. |

## Independent raw-artifact phase recomputation

I parsed the 14 source paths in the retained JSON directly with a standalone
standard-library probe. The probe did not import or invoke the analyzer. It
reimplemented canonical delivery (`audio_delivered`, with the documented
legacy positive-byte fallback), BOS recognition, stage-timing eligibility,
PAD recognition, and non-PAD control recognition from `agent_control` and
`control_ids`.

The raw result is:

| phase | n | server mean | nearest-rank p95 | >80 ms |
|---|---:|---:|---:|---:|
| text-emission | 209 | 82.364440 ms | 90.569 ms | 156 |
| PAD-tail | 615 | 82.440286 ms | 91.611 ms | 492 |
| agent-control | 7 | 82.321857 ms | 84.829 ms | 7 |
| total | 831 | 82.420213 ms | 90.990 ms | 655 |

For the exact-movement check, I also applied the round-1 implementation's old
rule, “integer token other than PAD is text.” Its 216-member text set equals
the corrected 209-member text set union the seven-member control set. The set
difference is exactly the control set, the PAD set is unchanged at 615, and
the current report memberships equal the independently derived text, PAD, and
control sets member-for-member.

The seven moved rows are unchanged from round 1:

| artifact | frame | event ordinal | server ms |
|---|---:|---:|---:|
| `sustained-aged-26h-20260812` | 1792 | 2085 | 83.038 |
| `209fe177-bbb8-403d-ba50-740d3b6f3bf4` | 74 | 515 | 81.569 |
| `de64cc34-2394-490e-a03c-cd920f96af25` | 288 | 858 | 84.829 |
| `feca4ec0-2962-48df-b219-e175858659c8` | 285 | 852 | 84.398 |
| `97f9bacc-4d2f-4e55-9026-dc2590919023` | 285 | 852 | 80.197 |
| `543bfc97-d9d1-40ae-84de-1a26b78a3756` | 290 | 862 | 80.665 |
| `70d3f7b5-2d3c-4998-846d-c85c539f2c59` | 285 | 852 | 81.557 |

Every row has `agent_token_id == 2`, `agent_control == "agent_eos"`, an
`agent_eos: 2` entry in `control_ids`, `response_boundary.phase ==
"tail_draining"`, and 1,764 delivered audio samples. No other row moved.

## Label and implementation check

`_token_phase()` now checks PAD first, then nonempty `agent_control`, then a
token match against non-PAD `control_ids`, before falling through to
`text-emission` (`stratified_latency_analyzer.py:335-364`). The Step 1 builder
publishes `agent-control` as a separate stratum alongside text, PAD, and
token-unavailable (`stratified_latency_analyzer.py:624-640`).

The two round-1 neighbor defects now resolve as:

- EarTTS-high row `2a0fa2ae…` frame 152 → frame 153
  `agent-control/t2`;
- EarTTS-high row `ac88123a…` frame 150 → frame 151
  `agent-control/t2`.

A recursive JSON scan found zero neighbor objects with
`phase == "text-emission"` and `agent_token_id == 2`. An exact Markdown scan
found zero `text-emission/t2` labels; token strings such as `t2534` were not
treated as matches.

## Test coverage and fresh execution

The retained-shape test at
`tests/runtime/test_stratified_latency_analyzer.py:638-757` contains both an
explicit `agent_eos` row and a second token-2 row whose `agent_control` is null,
so its expected two-row `agent-control` stratum pins the `control_ids` route.
It also includes a non-delivered token-12 internal-drain row with its
`agent_control` cleared and asserts that row is absent from every delivered
non-BOS phase membership. Two delivered BOS rows are likewise asserted absent
from those memberships and present in the separate BOS stratum.

The positive CLI test at lines 760-792 exercises `--step1-details` together
with `--markdown-output` and checks both files. The invalid-combination test at
lines 795-819 requires exit 2, the validation message, no analyzer invocation,
and no JSON or Markdown output. The implementation performs that validation
before analysis or writes at `stratified_latency_analyzer.py:2295-2307`.

Fresh commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/qualification/stratified_latency_analyzer.py verify-anchors
verify-anchors: PASS (67 anchors)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/runtime/test_stratified_latency_analyzer.py
30 passed in 0.08s
```

## Report and source integrity

- Fresh `analyze_paths(..., step1_details=True)` over all 14 pinned paths is
  equal to the complete saved JSON object.
- `json.dumps(report, indent=2, sort_keys=True) + "\n"` is byte-for-byte equal
  to the 38,486,601-byte JSON file.
- Rendering either the saved JSON object or the fresh analysis result with
  `render_step1_markdown()` is byte-for-byte equal to the 46,120-byte Markdown
  file.
- Fresh SHA-256 computation over each of the 14 raw source files matches its
  retained JSON identity; there are zero missing or mismatched sources.

The regenerated artifacts therefore carry the corrected semantics without a
source, membership, serialization, or renderer discrepancy. Together with the
corrected A4 prose, this closes the Step 1 gate.
