# Step 4c EarTTS-high incidence — 2026-08-13

## Preregistration — written before live probes

The local filesystem chronology supports that this file existed on 2026-08-13
before any Step 4c health query, model connection, or live fixture. It does not
provide a tamper-evident pre-run copy or hash of the preregistration bytes. The
design of record is the Step 4c paragraph in
`docs/reviews/step4-ab-measurements-review-r3.md` together with “Matched client
populations and corrected Step 4c design” in
`reports/step4-telemetry-regression-20260813.md`. The independently assigned run
block—not a response or frame—is the assignment and uncertainty unit.

### Question, predicate, and fixed exposure

The primary question is whether the qualified `EarTTS-high` mode recurs on the
unchanged running image under any of three client transports: direct strict-v3
WebSocket through the sustained runner, native Pipecat without a browser, and
browser/SmallWebRTC through the existing live-E2E machinery. This is an
incidence-first campaign. It does not compare pooled latency means and does not
presume a browser interaction.

The sole event predicate is the qualified analyzer's `EarTTS-high` membership:
`eartts_ms >= 18` within exactly its `delivered-nonBOS` population. Per block,
the retained outcomes are the event count, matched population count, rate, an
any-event indicator, membership SHA-256, response grouping, PAD/text phase, and
stage rows. Transport summaries use run-block resampling/uncertainty; frames are
never treated as independent assignments. Invalid blocks remain visible and do
not contribute timing or incidence estimates.

The live exposure cap is **nine attempted model sessions/run blocks total,
three per transport**, with no extension after inspecting outcomes. A block
that reaches model session creation consumes the cap even if it later fails a
correctness gate. Pre-connection infrastructure failures consume no model
exposure and may be corrected before retrying that same scheduled block. There
is no incidence-driven early stop: all nine scheduled blocks run unless the
unchanged stack becomes unsafe or unavailable.

Every valid block fixes:

- running model container/image/configuration; it will be inspected before and
  after, never rebuilt, replaced, or restarted;
- system instruction: `Qualification latency fixture: when the user supplies a
  script, speak the supplied script aloud completely and exactly, then stop.`;
- one typed request, Script L1, exactly: `The quick brown fox jumps over the
  lazy dog while seventeen green dragons circle the ancient stone tower,
  counting every window, every door, and every flag that flies above the
  northern gate.`;
- all-zero mono PCM acoustic input throughout the client-native path;
- 10.0 s connected zero-audio warm-up before the typed request, 120.0 s total
  connected/source window measured from the start of paced input, and 15.0 s
  terminal drain/settlement;
- no tools, fixed typed-input seed/runtime prompt, fixed response target, and
  the deployed 160 ms Pipecat playout prebuffer where Pipecat is present;
- a `/health` response immediately before every connection proving
  `active_client:false`; the full response is retained in that block directory;
- server trace, client report/trace, stage rows, exact session ID, telemetry,
  and block validity evidence.

### Counterbalanced independently assigned block schedule

Transport labels are D = direct WebSocket sustained runner, P = native
Pipecat, and B = browser/SmallWebRTC. The fixed sequence is three balanced
three-block periods. Each transport occurs once in each within-period ordinal
position, three times overall, with no adjacent repeated transport.

| block | period | position | assigned transport |
|---:|---:|---:|---|
| 01 | 1 | 1 | D — direct WebSocket |
| 02 | 1 | 2 | P — native Pipecat |
| 03 | 1 | 3 | B — browser/WebRTC |
| 04 | 2 | 1 | P — native Pipecat |
| 05 | 2 | 2 | B — browser/WebRTC |
| 06 | 2 | 3 | D — direct WebSocket |
| 07 | 3 | 1 | B — browser/WebRTC |
| 08 | 3 | 2 | D — direct WebSocket |
| 09 | 3 | 3 | P — native Pipecat |

### Qualification, analysis, and decision rules

A block is valid only if its immediate single-client check passes; exactly one
new model session is attributable to it; L1 has one completed response with
nonempty delivered audio and no extra/cancelled response; client trace
validation and delivered-delta reconciliation pass where applicable; metric
transport frames are contiguous; the session tears down cleanly; and the
qualified analyzer accepts the server trace. A prompt/output mismatch, trace
drop, ambiguous session identity, active-client overlap, or model-container
identity change invalidates the entire block. Browser blocks use a separate
Step 4c qualification-plan schema containing one capture-on L1 fixture; the
canonical eight-entry Step 2 plan/schema/hash and all its validation remain
unchanged and cannot accept this variant.

Primary reporting is the nine-vector of valid-block any-event indicators and
the per-block `EarTTS-high / delivered-nonBOS` counts. For each transport and
overall, report equal-block mean incidence and a deterministic 10,000-resample
cluster bootstrap percentile interval (seed 20260813), resampling whole valid
blocks within transport. Also report total counts/rates only as descriptive
exposure summaries. With three blocks per transport, uncertainty is expected to
be coarse; no frame-level confidence interval is admissible.

If at least one qualified event recurs within the cap, select and retain a
matched high/low PAD-position pair from the same transport and preferably the
same block/response, matched first on response, PAD-tail phase, and nearest
frame position. Retain the exact raw trace rows, analyzer memberships, neighbor
window, response/session identity, and available replay/profiler inputs. Run
Nsight only if suitable tooling can capture the selected state without
rebuilding, replacing, or restarting the model container; otherwise explicitly
defer the Nsight execution to Step 4d while retaining the selected states.

If no qualified event recurs in any valid block by the nine-session cap, the
result is **INSUFFICIENT**, not a null or transport comparison. The recommended
pivot is the older reproducible state difference, beginning with
`sustained-aged-26h-20260812` and its retained PAD-tail high/low rows, rather
than increasing exposure or attributing the absence to a client path.

## Results

**Decision: INSUFFICIENT.** The campaign reached the preregistered cap of nine
attempted model sessions in the exact `D,P,B / P,B,D / B,D,P` sequence. No
retained trace contains a qualified `EarTTS-high` row. Five blocks pass every
preregistered validity gate and contain **0 / 764** matched
`delivered-nonBOS` rows at or above the 18 ms EarTTS threshold. Four blocks
consumed exposure and are excluded for evidence-backed qualification defects.
The corrected population still yields `INSUFFICIENT` under the preregistered
rule; the cap is consumed, so no replacement or additional live block is run.

The valid-block any-event vector, in scheduled order, is
`[0, X, X, 0, X, 0, X, 0, 0]`, where `X` denotes an invalid and excluded block.
For completeness, none of the four invalid traces contains an EarTTS-high
member, but those rows do not enter any estimate.

### Executed blocks and session ledger

Every block created exactly one distinct model session. The retained health
response immediately before each connection reported `active_client:false`.
All nine traces had contiguous model-step transport frames and reproduced the
qualified analyzer's `delivered-nonBOS` and `EarTTS-high` membership manifests
exactly. The validity column applies the additional client/fixture gates.

| block | transport | model session ID | valid | delivered-nonBOS | EarTTS-high | telemetry rows |
|---:|---|---|:---:|---:|---:|---:|
| 01 | direct WebSocket | `0405013f-dbd5-4344-a8ff-9a58f71a46e7` | yes | 153 | 0 | 151 |
| 02 | native Pipecat | `3db461ca-d4d3-43f4-ac71-d16f22a6ec3d` | **no** | 150 | 0 | 151 |
| 03 | browser/WebRTC | `9228250b-80bb-4a64-b5fb-eb070d52316f` | **no** | 87 | 0 | 171 |
| 04 | native Pipecat | `b624df5b-1814-4251-a734-cbc75ad57083` | yes | 152 | 0 | 151 |
| 05 | browser/WebRTC | `c354d3d0-b4b5-4d6f-a73a-d0208b6bf0ff` | **no** | 150 | 0 | 171 |
| 06 | direct WebSocket | `a0dd12aa-7765-481b-a69b-9d675fe46533` | yes | 155 | 0 | 151 |
| 07 | browser/WebRTC | `b0a6c5f8-f561-4d24-8325-a9d0bcdecfaf` | **no** | 156 | 0 | 171 |
| 08 | direct WebSocket | `61f051fd-ef42-468a-800d-3f3006efa078` | yes | 150 | 0 | 151 |
| 09 | native Pipecat | `43bc13d9-d946-40e9-b545-496a559f1960` | yes | 154 | 0 | 151 |

Block 02 is invalid because the requested native-Pipecat playout trace was not
published and its client report has no terminal playout-trace status. The
server trace and completed one-response client report are retained, but the
preregistered client-trace validation gate cannot pass. Inspection found that
the soak runner selected its base subclass instead of the traced factory when
the trace environment was enabled. The factory selection and terminal-status
backfill were corrected before later Pipecat blocks; blocks 04 and 09 retained
valid zero-drop playout traces with 1,803 and 1,799 records respectively. No
replacement block was run because block 02 had already consumed exposure.

Block 03 is invalid because the first browser-side Pipecat process used the
normal demo instruction and tools and returned a capability refusal rather
than the fixed L1 response target. Its browser capture, response attribution,
server trace, and telemetry remain retained. A qualification-only environment
gate for the exact Step 4c instruction/no-tools configuration was then added;
blocks 05 and 07 passed it. The ordinary demo remains unchanged unless that
exact qualification label is supplied. Again, no replacement exposure was
added.

Blocks 05 and 07 are invalid because all 6,088 and 6,070 retained
`source:"microphone"` server-ingress packets, respectively, have peak
`3.052e-05`: exactly one PCM16 least-significant bit rather than the registered
exact-zero acoustic input. Block 03 has the same acoustic defect in all 6,018
microphone packets in addition to its response/configuration failure. The
source WAV is exact-zero mono PCM, so this is documented as a Chrome
file-backed fake-microphone/WebRTC fixture limitation, not treated as a
retroactive tolerance. The future browser fixture contract is now separately
versioned and explicitly registers a one-LSB server-ingress bound; it does not
change this campaign's zero-tolerance gate.

The five valid direct/native blocks converged on the same spoken target, with
punctuation and one connective normalization (`they count`) relative to the typed L1. The
qualification matcher requires the ordered terminal L1 phrase and at least
0.90 normalized word-sequence agreement; it rejects block 03's refusal. This
criterion checks target identity without treating generated punctuation as a
new fixture.

### Run-block incidence and uncertainty

| scope | valid blocks | matched rows | high rows | equal-block mean incidence | block-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| direct WebSocket | 3 | 458 | 0 | 0.000 | [0.000, 0.000] |
| native Pipecat | 2 | 306 | 0 | 0.000 | [0.000, 0.000] |
| browser/WebRTC | 0 | unavailable | unavailable | unavailable | unavailable |
| all transports | 5 | 764 | 0 | unavailable | unavailable |

The direct and native intervals are the preregistered 10,000-resample
percentile results with seed 20260813, resampling whole valid run blocks. The
browser summary and stratified-overall interval are unavailable because no
browser block is valid. The empirical `[0,0]` intervals for the two available
strata describe only their observed capped all-zero blocks; they are not
population bounds. The pooled `0 / 764` is retained only as a descriptive
exposure count.

### Container, fixture, and measurement integrity

- The model container remained running and was not rebuilt, replaced, or
  restarted. Before/after inspect records have identical canonical content
  after sorting Docker's nondeterministically ordered mount list: container ID
  `6efe0b3ed9e24df9830a3b5666652ec57d3991d01a3bbf4a876b2dfd1b7ff650`,
  image ID
  `sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`,
  and image tag `pipecat-ai/nemotron-voicechat-dgx-spark:generation-step2`.
- The retained preflight health record pins checkpoint revision
  `fb0f94eaf4d03ddc430f39565229393fa1b50c26`, model SHA-256
  `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`,
  Nano component SHA-256
  `3acecd057f7367768d621bec649cbe6cbe327ca0857c17f86f7e0929f2bb2d99`,
  and EarTTS component SHA-256
  `6e103b3cbcbb9594c9f01a37328ff72e9628962ccaf3fbd01e0b1937a40faf3e`.
- All five valid direct/native clients used L1 SHA-256
  `2300ce7a5211655815abcdc1e24ab5b696000eebc553dc37071d806820c36035`,
  exact-zero PCM at server ingress, 10 s warm-up, 120 s duration, and 15 s
  drain. Block 01's original client report omitted the drain field; a derived
  timing sidecar explicitly records 15 s and is checked against its 1,500
  source frames and 135.047 s first-metric-to-clean-close interval rather than
  supplying an analyzer default. Native Pipecat used the deployed 160 ms
  prebuffer.
- The retained browser plan remains the isolated
  `nemotron_voicechat.step4c_fixture_plan.v1` document and hash. Future browser
  work uses v2, which records the fake-microphone/WebRTC one-LSB ingress
  limitation and bound. Neither parser accepts the canonical Step 2 plan.
- Server stage rows, analyzer memberships and hashes, client artifacts,
  response/session attribution, per-block health, and telemetry are retained
  below the data directory. The empty high-membership SHA-256 is
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
  in every block.

### Nsight decision and pivot

Nsight Systems 2025.6.3 and Nsight Compute 2026.1 were present, but the
preregistered trigger did not fire: there is no new EarTTS-high row to pair
with a matched low PAD-tail state. No Nsight run was performed, and
`matched-states.json` records `not-triggered`; there are no fabricated or
post-selected new states to carry into Step 4d.

The required pivot is therefore the older retained state difference, starting
with `sustained-aged-26h-20260812`, which retains 10 EarTTS-high rows among 188
delivered-nonBOS rows and PAD-tail high/low evidence for reconstruction. It is
not a same-condition comparator: it used a different image ID/tag, server
source, approximately 26-hour age, and 220-second five-response workload. Its
own sustained report also has `passed:false`. It is a profiler/reconstruction
starting point, not proof that age or image caused the difference or that the
older state remains reproducible.

The defensible conclusion is narrow: under this campaign's observed runtime
state and L1 protocol, no retained trace contains a qualified high row, so the
mode was not available for selecting a new matched high/low pair. The five valid
direct/native blocks weaken any claim that EarTTS-high is persistent or common
under those current conditions. The browser traces are sensitivity
observations, not valid evidence about preregistered browser incidence. The
campaign does not establish removal, bound future incidence, show transport
equivalence, or identify a cause.

### Retained artifacts, tooling, and validation

The data root is `reports/step4-eartts-incidence-20260813-data/`. Its primary
machine-readable records are:

- `campaign-manifest.json`: schedule, validity adjudication, paths, and every
  model session ID;
- `incidence-analysis.json`: independently reproduced qualified memberships,
  per-block counts/hashes, run-block bootstrap summaries, and decision;
- `matched-states.json`: the preregistered Nsight trigger outcome;
- `artifact-sha256.json`: SHA-256/byte inventory sealing the preregistration,
  health, client reports/traces, telemetry, server traces, analyzer results,
  campaign manifest, and before/after container records;
- `preflight/`: fixture plan/audio, source/tool versions, health, and unchanged
  before/after model-container inspection;
- `block-01-*` through `block-09-*`: health, telemetry, client evidence,
  server trace, and analyzer output for every attempted exposure.

New qualification tooling consists of the shared Step 4c fixture contract,
the campaign incidence analyzer, Step 4c modes in the sustained and native
Pipecat runners, and a clearly separate Step 4c browser fixture-plan/test path.
The analyzer regenerates qualified membership and adjudicates all registered
gates from retained evidence rather than accepting manifest validity flags. It
fails closed on schedule/session/health/container/source identity, transport
continuity, exact server-ingress acoustic input, fixture timing/identity,
response cardinality/cancellation/audio, teardown, browser artifact status,
native/browser playout publication/validation/drop/identity, and contradictory
enumerated invalid reasons. Public `analyze()` tests exercise a synthetic
nine-block campaign across all three transports, both decisions, matched
PAD-tail selection, honest exclusions, and negative mutations for every listed
gate, including the manifest-flipped missing-playout probe.

Post-campaign gates:

- full offline suite excluding live `browser_e2e`: recorded in the closure
  completion note;
- canonical retained verifier: recorded in the closure completion note;
- task-scoped Ruff and `git diff --check`: recorded in the closure completion
  note.

No commit was created.
