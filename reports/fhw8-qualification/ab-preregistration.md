# Preregistration: fhw8 promotion counterbalanced A/B latency qualification

Status: sealed before any session runs. This is the plan's promotion-contract
latency qualification for the fully G0–G3-qualified fhw8 candidate
(`reports/fhw8-qualification-20260814.md`, `-g3-20260814.md`).

## Design

Four fresh service sessions in fixed counterbalanced order **R1, C1, C2,
R2** (R = released artifact via the retained v4 script; C = candidate via
the retained G3 start script; identical image, env, EarTTS, config;
manifest/nano-path are the only deltas — both scripts already retained and
hash-recorded). Each session, after `ready`, runs the identical canonical
hash-pinned 8-fixture browser campaign (plan `3e056280…`, unchanged Step 2
driver/harness, quiescence barriers, playout tracing). Model server traces
(per-connection `events.jsonl`) are collected per session by
before/after snapshot of `~/.local/state/nemotron-voicechat/traces/model/`.
A session whose campaign fails driver validation, or whose service startup
requires more than two attempts (known intermittent EarTTS warmup stall:
one retained retry allowed per session, matching release-artifact
precedent), is retained-invalid and rerun as a fresh appended session of
the same arm; the counterbalanced order of VALID sessions must end
R,C,C,R.

## Analysis (fixed)

Each valid session's collected `events.jsonl` files are analyzed with the
unchanged qualified analyzer
(`tools/qualification/stratified_latency_analyzer.py analyze`, r3.2 schema,
default reserve sweep, max-gap 160 ms). Primary population: the normative
delivered-nonBOS generation-phase interval population as defined by the
analyzer's existing schema; primary per-session statistic: its mean
interval in ms; secondary: nearest-rank P95 and the structural-gate
outcomes.

- **Primary contrast:** mean over R sessions minus mean over C sessions of
  the per-session generation-phase mean interval (Δ, ms/frame).
- **Uncertainty:** responses are clusters; deterministic response-clustered
  bootstrap of Δ, 10,000 resamples, seed 20260814, percentile 95% CI.
- **Session count note:** two sessions per arm is the preregistered
  minimum; the session pair per arm exists to absorb order/thermal drift
  via counterbalancing, and the response-clustered CI carries the
  uncertainty.

## Gates (fail-closed)

- **AB-1 (latency):** PASS iff Δ ≥ **3.0 ms/frame** and the 95% CI
  excludes 0. (G2 traced measurement predicts ≈ 4.4; end-to-end includes
  non-Nano per-frame work, so the floor is set below the traced value but
  high enough to be promotion-meaningful.)
- **AB-2 (behavioral parity):** every valid session shows the same
  categorical fixture outcomes as the retained baseline (the G3-2 rule: no
  new failure class in any arm, either direction).
- **AB verdict = PASS iff AB-1 and AB-2 pass.** On PASS the candidate has
  completed the full promotion contract and awaits the explicit user
  promotion decision. On FAIL all evidence is retained and the candidate
  returns to analysis.

## Operations

Stop/start via retained scripts only; stable-field identity verification
for the final release restore; every stall/failure retained. Evidence under
`reports/fhw8-qualification/ab-session-20260814/` with SHA-256 manifest.
Sealed by SHA-256 recorded before R1 starts.
