# Step 4c incidence measurement-validity review — round 1

**Verdict: FAIL.** The retained traces independently reproduce **0 / 1,070** for
the seven blocks *declared* valid, the two disclosed exclusions are
outcome-independent and honest, and `INSUFFICIENT` plus the older-state pivot
still follows the registered decision rule. But 0/1,070 is not the defensible
preregistered valid population: the two declared-valid browser blocks delivered
one-LSB microphone PCM rather than the registered all-zero PCM. The closure also
fails the requested tooling criterion: the Step 4c incidence analyzer does not
enforce all preregistered validity gates, accepts the known-invalid block 02 if
its manifest flag is changed to valid, and has no end-to-end test of `analyze()`.
The report's claim that seven blocks passed every gate and that the analyzer
refuses invalid fixed-fixture reports and undeclared invalid blocks is therefore
false.

This review was entirely offline. I made no health query, model connection,
browser connection, live probe, container change, or commit. Apart from this
review file, I made no intentional file change.

## Blocking findings and required fixes

### Browser acoustic input violates the exact fixed-input condition

The preregistration fixes “all-zero mono PCM acoustic input throughout the
client-native path” (`reports/step4-eartts-incidence-20260813.md:47`) and the
result claims every valid client used zero PCM (`:209-214`). The retained source
WAV is indeed mono 48 kHz and contains 7,920,000 samples, all exactly zero. The
server traces do not preserve that condition through browser/WebRTC:

- block 05 has 6,088 `source:"microphone"` input packets, all with
  `peak:3.052e-05`, exactly one PCM16 least-significant bit;
- block 07 has 6,070 such packets, again all one LSB; and
- block 03 has the same defect in all 6,018 microphone packets, although it was
  already invalid for its response/configuration failure.

By contrast, every direct/native `source:"microphone"` packet is exactly zero.
The browser value is plausibly introduced by the WebRTC/audio pipeline and is
far below the speech gate, but neither provenance nor small magnitude turns it
into the preregistered exact-zero condition. No zero-tolerance relaxation was
registered. Blocks 05 and 07 therefore fail the fixed-input gate and must be
reclassified invalid. The existing strict valid population becomes five blocks:
direct 0/458 plus native Pipecat 0/306, or **0/764**. Browser incidence is
unavailable because no browser block is valid. All nine attempts remain consumed,
so `INSUFFICIENT` and the pivot do not change.

For the retained campaign, correct the manifest/report/analysis and completion
note to 5 valid, 4 invalid, and 0/764; do not replace or add exposures. For a
future campaign, either disable the browser transformation and verify exact zero
at server ingress or preregister a physically justified near-zero tolerance and
apply it identically as far through each client-native path as the estimand
requires. The incidence analyzer must check the retained server-ingress acoustic
predicate rather than trusting a source-WAV hash.

### Incidence validity enforcement is not fail closed

`tools/qualification/step4c_incidence.py:141-162` validates direct/native client
reports without checking the native playout-trace contract. In particular, it
does not require `playout_trace_enabled:true`, a non-null terminal
`playout_trace_status`, `valid:true`, zero drops, or the retained trace itself.
The manifest's free-form `valid` and `invalid_reason` values are not read until
after that generic validation (`:326-338`). Consequently:

- retained block 02 says `playout_trace_enabled:true` but has
  `playout_trace_status:null` and no playout artifact;
- changing only block 02's in-memory manifest declaration to `valid:true` and
  `invalid_reason:null` is accepted by public `analyze()`, producing eight valid
  blocks and `0 / 1,220` rather than rejecting the undeclared trace defect;
- an empty string is accepted as an invalid reason (`:335-336`);
- block 01 has no `drain_seconds` field, but the direct-report validator silently
  substitutes 15 seconds (`:150-153`) and accepts it; and
- conversely, because generic client validation happens before validity
  adjudication, an honestly retained invalid block whose client report itself
  says `passed:false` can abort the whole analysis instead of being retained and
  excluded.

The only incidence-tool tests are five helper-level tests in
`tests/runtime/test_step4c_incidence.py:22-82`: bootstrap determinism, empty
bootstrap rejection, the schedule constant, response matching, and mount-order
canonicalization. No test constructs a manifest, invokes `analyze()`, regenerates
qualified membership from traces, exercises validity/exclusion, or tests the
matched-state trigger. The fresh broader scoped suite passing does not fill that
coverage gap.

Additional required fixes:

1. Split artifact readability from validity adjudication. Every attempted block
   must remain analyzable, while every *valid* block must pass transport-specific,
   machine-verifiable gates. For native Pipecat Step 4c, require a configured
   playout path, a published/closed terminal status with `valid:true`, zero
   dropped or queued records, matching session/response identity, and successful
   playout validation. Require explicit 10/120/15 fields rather than defaults,
   explicit clean teardown, exactly one non-cancelled response, and nonempty
   delivered audio for all applicable transports. Use enumerated invalid reason
   codes whose claimed defect is checked against the retained evidence; reject
   empty or contradictory reasons.
2. Make the native Step 4c runner itself fail closed when no playout-trace path
   is configured. Its current optional base-service path can still produce a
   passing Step 4c report without the preregistered trace evidence.
3. Add end-to-end tests over a synthetic nine-block campaign through public
   `analyze()`. They must cover exact qualified membership regeneration, all
   three transports, valid and invalid blocks, high and all-zero decisions,
   matched high/low PAD-tail selection, and negative mutations for schedule,
   session, health, container, trace hash, transport-frame continuity, fixture,
   response, teardown, playout publication/drop, and invalid-reason evidence.
4. Correct `reports/step4-eartts-incidence-20260813.md:116-123,209-214,250-255`
   and the completion note so they use the strict 0/764 population and do not
   claim gates the analyzer does not enforce. Regenerate the analysis after the
   fixes, preserving blocks 02, 03, 05, and 07 as consumed and excluded. This
   does **not** require more live exposure.
5. Add a complete artifact SHA-256 manifest, including the preregistration,
   health responses, client reports/traces, telemetry, server traces, analyzer
   results, campaign manifest, and before/after container records. The present
   result hashes server traces inside the analysis but does not seal the entire
   evidence package.

## 1. Preregistration chronology

**Supported by the local chronology, but not immutably proven.** The report inode
was created at 16:40:58 UTC. That precedes the retained preflight health file at
16:50:27 and block 01's health file at 16:50:43. The prior round-3 review was
completed at 16:36:37 and already required incidence first, counterbalanced
direct/native/browser run blocks, a preregistered cap, matched
`delivered-nonBOS`, run-block uncertainty, `INSUFFICIENT`, and the older-state
pivot (`docs/reviews/step4-ab-measurements-review-r3.md:57-66`). The final report
places the nine-block cap and exact `D,P,B / P,B,D / B,D,P` schedule before its
results, and all retained sessions follow that order.

The nine health artifacts all report `active_client:false`. Relating their file
times to each server trace's first monotonic timestamp through the retained
telemetry wall/monotonic offset places the health writes 0.002–2.951 seconds
before their respective session starts. Thus the report's per-connection health
and ordering claims agree with the evidence.

The limitation is that the report was untracked and was modified at 17:30:30,
after the campaign, to add results. There is no pre-run copy, pre-run hash,
commit, append-only log, or external timestamp that proves the *bytes containing
the numeric cap and schedule* were already present at inode creation. Filesystem
birth time proves that a file existed, not its earlier contents. I found no
contrary evidence, and the observed execution is consistent with the claimed
preregistration, but “genuinely preceded” is not independently auditable to the
same standard as a sealed preregistration. Future campaigns should retain and
hash a standalone preregistration before the first health query; this campaign
must describe its chronology as filesystem-supported rather than tamper-evident.

## 2. Independent incidence and membership recomputation

**PASS for row classification; FAIL for block-level eligibility.** I reran the
qualified analyzer's normalization, population classification, and
membership-manifest construction directly over each of the nine retained server
traces. Every fresh `delivered-nonBOS` and `EarTTS-high`
membership manifest, including every ordered member and SHA-256, exactly equals
its retained analyzer manifest. The predicate is the designed one: canonically
delivered, non-BOS rows form the denominator, and a member is high only when its
`eartts_total_ms >= 18`.

| block | valid | fresh delivered-nonBOS | fresh EarTTS-high | maximum eligible EarTTS ms |
|---:|:---:|---:|---:|---:|
| 01 | yes | 153 | 0 | 13.404 |
| 02 | no | 150 | 0 | 14.818 |
| 03 | no | 87 | 0 | 13.628 |
| 04 | yes | 152 | 0 | 14.621 |
| 05 | yes | 150 | 0 | 13.537 |
| 06 | yes | 155 | 0 | 13.474 |
| 07 | yes | 156 | 0 | 15.704 |
| 08 | yes | 150 | 0 | 16.032 |
| 09 | yes | 154 | 0 | 14.023 |

The seven *manifest-declared* valid blocks therefore recompute to direct 0/458,
native Pipecat 0/306, browser 0/306, and **0/1,070**. Even their largest eligible
row value is 1.968 ms below the threshold. Public campaign analysis reproduces
those declarations, two invalid, `INSUFFICIENT`, and
`matched-states: not-triggered`. There is no threshold-rounding or
delivered-nonBOS row-matching ambiguity. Applying all registered block gates,
however, excludes browser blocks 05 and 07 and leaves the strict valid result at
**0/764**. Thus the report's exact row arithmetic is reproducible while its
valid-population label is not.

## 3. Invalid blocks and cap accounting

**PASS for the honesty of the two disclosed exclusions; FAIL for the complete
adjudication.** Blocks 02 and 03 each created one unique model session, occupy
their scheduled positions, and remain retained. Neither was replaced, so the
campaign contains exactly nine model exposures.

- Block 02 is genuinely invalid. Its client report records one completed prompt
  and response with delivered audio, but it records tracing as enabled while its
  terminal playout status is null and no playout artifact exists. Blocks 04 and
  09, after the disclosed runner correction, retain valid closed zero-drop traces
  with 1,803 and 1,799 records.
- Block 03 is genuinely invalid. Its server trace configures
  `get_current_utc_time` rather than an empty tool set, and its retained response
  is the capability refusal quoted in the report, not L1. Later browser blocks
  show an empty tool set and the intended L1 rendition.

Both invalid traces also independently contain zero high members (0/150 and
0/87). Excluding them cannot manufacture the absence of recurrence or hide a
positive event; it only removes unqualified exposure from the denominator and
block summaries. The fixes made after each failed exposure restored the
preregistered condition for later scheduled blocks, and treating the failed
sessions as consumed rather than replacing them is exactly the registered rule.
Blocks 05 and 07 were not cherry-picked either; they must now be excluded for the
independently observed exact-zero-input violation. Their combined 0/306 also
contains no event, so correcting the oversight changes eligibility and exposure,
not the all-zero direction of the finding.

## 4. Decision rule and pivot

**PASS after correcting the valid population.** The registered rule says no
event in any valid block by nine consumed sessions yields `INSUFFICIENT`, not a
null, equivalence claim, or transport
comparison, followed by a pivot to the older retained state. Five strict valid
blocks have no event and all nine scheduled sessions were consumed, so both the
verdict and `matched-states.json`'s `not-triggered` state follow mechanically. No
Nsight capture was warranted. The currently reported empirical `[0,0]` bootstrap is
mechanically correct for the seven declared-valid all-zero rates, but the
corrected analysis must mark the browser summary and any stratified-overall
interval unavailable rather than treating browser as a valid stratum. No such
empirical interval is a population bound.

## 5. Other tooling and canonical Step 2 isolation

**PASS for contract identity and Step 2 isolation, not runtime acoustic
conformance or incidence validity enforcement.** The shared contract contains
the registered instruction, exact L1 bytes/hash, and 10/120/15 timings. The
current direct, native, and browser qualification modes select one L1 request,
zero-source-audio pacing, the qualification instruction, and no tools. Browser
transport does not retain exact zero at server ingress. The runner fixes made
after blocks 02 and 03 are visible in chronology rather than concealed.

The browser plan is genuinely isolated from canonical Step 2. Fresh checks give:

- canonical Step 2: schema `nemotron_voicechat.step2_fixture_plan.v2`, eight
  entries, hash
  `3e056280f4042aacd4209983fa5252019cbe0bc92820059f351dda70fee67e26`;
- Step 4c: schema `nemotron_voicechat.step4c_fixture_plan.v1`, one entry, hash
  `c2ff7156d9bb8892e197a44184f61619566ffa9166ad1dced8f403d3dec0afb2`;
- each parser rejects the other plan, and the retained Step 4c plan exactly
  matches the separately generated Step 4c document.

Fresh offline checks passed **154 tests** across the incidence, sustained-runner,
native-runner, browser-driver, and Pipecat bot modules. The canonical retained
verifier also passed **67/67 anchors**. Those results substantiate the runner and
schema isolation, but they do not cure the missing public incidence-analysis
coverage identified above.

## 6. What the non-recurrence means

The defensible conclusion is narrow: under this current campaign's observed
runtime state and L1 protocol, no retained trace contains a qualified high row,
so the mode was not available for selecting a new matched high/low pair. The
five valid direct/native blocks weaken any claim that EarTTS-high is persistent
or common under those current conditions. The browser traces are useful
sensitivity observations but, after the acoustic-input violation, are not valid
evidence about preregistered browser incidence. The campaign does not establish
that the mode was removed, bound its future incidence, show equivalence among
transports, or identify a cause.

In particular, the old and new observations are not same-condition replications:

- They use the same checkpoint revision, model SHA-256, Nano component SHA-256,
  and EarTTS component SHA-256.
- They do **not** use the same runtime image/code identity. The older
  `sustained-aged-26h-20260812` session records image ID
  `sha256:a6f9d56782e96038dced040f8643f7f215000893fbaf8f46c849b6fc5ad6d63c`,
  tag `production-candidate-1`, and server source SHA-256
  `eacb6ff77fbf5e37bdd4f6879d5da457d7dbab8e7ae058b05ebad645c3f4d2a6`;
  Step 4c records container image ID
  `sha256:ea15693c2c7931516a366fe886a00caabe09f441abadc2a461cdaa51d0ed232a`,
  tag `generation-step2`, and server source SHA-256
  `8f5855d5d62529367d574836caae99b36dda31c06a2aea60953bcf117a975a87`.
- The current container was about 10.5–11.0 hours old during Step 4c, while the
  old artifact is explicitly an approximately 26-hour-aged state.
- The old direct session used a 220-second, five-response mixed-prompt workload;
  Step 4c used separate 120-second source windows, one fixed long L1 response,
  10-second warm-up, and three client transports.

The larger current raw exposure (1,307 non-BOS rows across all attempts, 1,070
under the manifest declarations, and 764 under strict validity, versus 188 in
the aged artifact) makes simple exchangeability with the old 10/188 observation
implausible, but frames are not independent trials and the changed image, code,
age, workload, prompt, and period prevent causal attribution. Image difference
is a credible candidate explanation, as are aged/session state and workload;
none is identified by these data. The pivot should therefore recreate and pin
the older image/code/age/workload state and then vary one factor at a time. It
must not treat “image fixed it,” “browser has no effect,” or “more current
exposure is unnecessary” as an established conclusion.

The older artifact's EarTTS-high rows are qualified analyzer members, but its
own sustained-runner report has `passed:false` because the queue-slope gate and
expected session-limit condition did not pass. It is suitable as a retained
high-state starting point for reconstruction and profiler targeting, not as a
qualified incidence comparator or proof that the approximately 26-hour state is
currently reproducible.
