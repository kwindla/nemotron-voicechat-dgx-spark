# Preregistration: fhw8 G3 — live behavioral promotion phase

Status: sealed before any G3 run. Funded by user "Yes. Proceed." following
the committed G0–G2 report (`reports/fhw8-qualification-20260814.md`,
commit 0955bf0). G3 is the final preregistered gate before the candidate
may be considered for production promotion.

## Candidate service definition

The retained v4 qualification service configuration with exactly these
deltas:

1. `--nano-vllm-path` → the fhw8 candidate artifact (read-only mount of
   `~/.cache/nemotron-voicechat/artifacts/candidates`).
2. `--vllm-manifest` → a **candidate manifest** of accepted kind
   `exact_public_vllm_quantized_nano_eartts`: source identity copied from
   the signed release parent (repository, revision, parent model SHA-256,
   `derived_from_ea: false`); `components.nano` built from the actual
   candidate files with the validator's composite algorithm;
   `components.eartts` carried over unchanged from the signed release
   inventory; `quantization` records the fhw8 delta and provenance
   (base release, builder script SHA, G0–G2 report). The
   `reproducibility: {runs: 2, byte_identical: true}` claim must be EARNED
   before the manifest is written: a second independent builder run into a
   fresh directory must produce byte-identical files (all SHA-256 equal);
   the double-build evidence is retained. No claim is copied that is not
   locally true.
3. Everything else — image, env (including
   `VOICECHAT_QUALIFICATION_MODE=step2-longform-fixture`, which is baked
   into the canonical plan hash), EarTTS artifact, config mount — is
   byte-identical to the retained v4 script.

The released production artifact and its manifest are never modified.

## Campaign legs (order fixed)

**Leg B (baseline function-calling), on the currently running release
service:** the voice function-calling fixture
(`tools/qualification/generate_voice_function_fixture.py` output, hash
recorded) exercised through the established browser harness; retained as
the baseline comparator for Leg C's function-calling gate. Run twice
(fixture repetition) to expose obvious nondeterminism.

**Leg A (candidate longform), on the candidate service:** the canonical
hash-pinned 8-fixture plan (CANONICAL_PLAN_SHA256
`3e056280…`, S1,S1,L1,L1,L2,L2,I1,capture-off-L1) via
`tools/qualification/step2_live_fixture_driver.py` and
`tests/e2e/test_browser_smallwebrtc_e2e.py::test_step2_live_fixture_plan`,
with the driver's full strict validation, quiescence barriers, and
response-lifecycle correlation exactly as retained from the Step 2
campaign. No driver or harness code changes are in scope; any gap that
forces a change is a recorded deviation with the campaign restarted.

**Leg C (candidate function-calling), same candidate service session:**
the same function-calling fixture as Leg B, run twice.

## Gates (fail-closed)

- **G3-1 Campaign integrity:** Leg A completes 8/8 fixtures under the
  driver's strict summary validation. Engine-construction basin is
  recorded from service logs/traces where observable; a wedge, crash, or
  teardown failure is a FAIL (not retried silently; retained).
- **G3-2 Structural/audio parity:** the Step 2 adjudication gates (literal
  K=12 structural gate t[i+11]−t[i] ≤ 960 ms with max-gap ≤ 160 ms,
  transcript/audio adjudication rules) applied to Leg A must show **no new
  failure class** relative to the retained baseline Step 2 session
  (`reports/step2-live/session-20260813T083303Z`): fixture-level outcomes
  may not be worse than the retained baseline outcome for the same fixture
  slot. The known baseline findings (stochastic long-rendition truncation;
  Gate-1 FAIL as a model finding) do not count against the candidate when
  they recur at comparable incidence; a NEW class (e.g., wrong tool call,
  garbled function acknowledgment, systematic early stop not present in
  baseline) is a FAIL.
- **G3-3 Function-calling correctness:** in Leg C, every tool call that the
  fixture elicits must carry the correct function name and arguments and
  produce a coherent spoken acknowledgment, with outcomes not worse than
  Leg B on the same fixture (tool-call presence, correctness, and
  latency-to-tool-call within normal variation). Any decision-level
  divergence from Leg B beyond nondeterministic wording is adjudicated and
  a wrong/missing/extra tool call is a FAIL.
- **G3-4 Latency observation (descriptive, not a pass/fail gate here):**
  per-frame generation intervals from Leg A traces are summarized against
  the retained baseline session's distribution to confirm the direction of
  the G2 improvement end-to-end; promotion-grade A/B latency remains the
  plan's separate counterbalanced protocol.

**G3 verdict = PASS iff G3-1, G3-2, and G3-3 all pass.** On PASS the
candidate is eligible for promotion, which still requires the plan's
remaining promotion contract (counterbalanced A/B latency qualification)
and an explicit user decision. On FAIL the candidate returns to analysis
with all evidence retained.

## Discipline

Preregistration sealed by SHA-256 before any leg runs. Deviations recorded,
never silently absorbed; failed runs retained; the release service is
restored (v4 script, stable-field identity verification) at campaign end
regardless of outcome. Evidence under
`reports/fhw8-qualification/g3-session-<UTC>/` with a SHA-256 manifest.
