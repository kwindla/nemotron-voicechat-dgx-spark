# fhw8 promotion counterbalanced A/B latency qualification

**Date:** 2026-08-14 UTC
**Preregistration:** `reports/fhw8-qualification/ab-preregistration.md`,
SHA-256 `1eb8f6dc3535ac1395607abb6923b9c8b347cea524144ad4927aa85feee89667`,
sealed before R1 started.
**Verdict: AB PASS (AB-1 PASS, AB-2 PASS with a recorded release-arm
observation).** The fhw8 candidate has now completed the entire promotion
contract: G0–G3 plus this counterbalanced A/B. The only remaining step to
production is the explicit user promotion decision.

## Campaign

Four fresh service sessions in the preregistered counterbalanced order
R1, C1, C2, R2 (R = released artifact, C = candidate; identical image/env;
retained start scripts). Every session ran the canonical hash-pinned
8-fixture browser campaign; all four passed the driver's strict validation
first-try (exit 0; no startup stalls this campaign). Per-session pipecat
playout traces and model server traces retained.

**Recorded deviation (analysis artifact):** the preregistration named the
server `events.jsonl` artifacts as the analysis input; those artifacts
cannot evaluate the structural gate (no client audio-delta receipts), and
the retained Step 2 baseline analysis had consumed the pipecat playout
traces. The normative analysis therefore uses the per-session pipecat
playout traces, exactly matching the baseline's artifact choice; the
server-clock analyses are retained alongside
(`arrival-analysis.json` vs `arrival-analysis-pipecat.json`) and give the
same contrast to 0.001 ms.

## AB-1 — latency (PASS)

Response-clustered delivered `server_step_ms`, per the analyzer's r3.2
schema; 18 response clusters per arm:

| | R (release) | C (candidate) |
|---|---:|---:|
| session lane means (ms/frame) | 85.30, 88.80, 82.75, 82.53 | 77.24, 76.89, 80.05, 76.48 |
| arm mean | **84.12** | **78.67** |

**Δ = 5.452 ms/frame**, response-clustered bootstrap 95% CI
**[3.55, 7.20]** (seed 20260814, 10,000 resamples). Gate: Δ ≥ 3.0 with CI
excluding 0 → **PASS**. The end-to-end improvement exceeds the G2 traced
prediction (4.44 ms), and the candidate arm's ~78.7 ms/frame is **under the
80 ms realtime budget**.

## AB-2 — behavioral parity (PASS, with observation)

All eight fixture campaigns (4 sessions × strict driver validation) passed
with the same categorical outcomes as the retained baseline; no new failure
class in either arm.

Structural gate (literal K=12, 960 ms window) on the evaluable capture-off
lanes: **both candidate sessions ACCEPT; both release sessions REJECT.**
The release REJECTs are the chronic ~84–86 ms generation overrun sitting
above the 80 ms/frame gate line — the very defect this plan exists to fix —
and are consistent with Step 0/1 findings (the retained baseline session's
single ACCEPT on this lane was borderline). This is recorded as a
release-arm observation, not a candidate failure class: the candidate
strictly improves the gate outcome, turning the realtime structural gate
from REJECT to ACCEPT. Capture-on lanes are UNAVAILABLE for the gate in
both arms alike (delivered/delta reconciliation, same as baseline).

## Standing

- The generation-frame realtime budget is met end-to-end by the candidate:
  ~78.7 ms/frame delivered vs the 80 ms budget, with the structural gate
  ACCEPTing.
- The candidate's full qualification chain: G0 (construction + head
  numerics), G1 (zero decision flips, 439,500 paired decisions), G2
  (−4.437 ms/frame traced), G3 (live behavioral parity + function-calling
  20/20), AB (−5.452 ms/frame end-to-end, CI-excluding-zero, structural
  gate ACCEPT).
- Pairing (~11 ms additional) remains in root-cause (Phase 1 design in
  progress) and would add margin on top.

**Promotion is now solely a user decision.** Evidence under
`reports/fhw8-qualification/ab-session-20260814/` with SHA-256 manifest.
