# Step 5 pairing wedge-closure design review — round 2

**Review date:** 2026-08-14 UTC  
**Verdict:** **PASS - track 2 concluded**

The round-1 Required deltas are closed. This PASS concludes the static pairing
wedge-closure **design study**; it does not close the reliability precondition,
qualify pairing, authorize implementation, or promote the ranking row. The
standing remains **NOT CLOSED**: repaired tail pairing is retained precedent
only and remains **INADMISSIBLE**. Option (b) remains **NEEDS-EXPERIMENT**, and
the repaired preregistered designs are the only bounded route left in track 2.
Options (a) and (c) remain NO-GO for the present Step 5 scope.

This was a document-only confirmation against the round-1 review, the revised
design, the supplied closure note, and the retained documents cited by the
design. No live service, GPU, container, model, or runtime probe was used.

## Required-delta confirmation

| Round-1 delta | Round-2 disposition | Confirmation |
|---|---|---|
| Matched construction and terminal ordering | **CLOSED** | Phase 1 now preregisters 12 fresh-engine replicates with independent S, B, optional C, and A request lanes reconstructed from the same frozen prefix inside one worker/engine epoch. The executable order is S, B, optional C, then terminal A. S and B must complete before A. An unknown A completion ends the epoch and is never resumed, cancelled-and-retried, or replayed there. A qualifying replicate requires exact S and B completion followed by A loss at the same preregistered boundary; no post-A flip-back is attempted. |
| Construction-confound fallback | **CLOSED** | If lane-local dispatch would change more than the intervention, the design requires a non-destructive construction discriminator and preregistered matched A/B epoch pairs within each observed construction stratum. Unstratified counterbalanced epochs are explicitly non-causal and cannot advance to stress. |
| Rejection-aware sequential oracle | **CLOSED** | S is defined as a logical-position transaction. PAD acceptance advances two positions and compares packed row 1 at the position-two committed boundary. Every rejection class advances only position one and compares packed row 0 at that boundary; the corrected second row remains pending until the next logical scheduler action. The equality contract covers text and function tokens and their conjunction, accepted-token count, selected-row custom outputs, committed KV and Mamba state, rollback shadows and pointer/identity, correction state, pending-row contents, and restoration/absence of uncommitted position-two state. |
| Retained-exposure census and base-rate disposition | **CLOSED** | The design de-duplicates the repeated Step-7 exposure and reports the recoverable artifact-level counts: 1,217 selected packed attempts with one exact missing-completion signature, while identifying the missing audited union of eligible calls, sessions/requests, and construction identities. It correctly refuses to pool these into `1/1217`, declares the historical base rate unidentified, and requires a manifest and identity-de-duplicated future census before historical calibration. |
| Clustered reliability estimands and power | **CLOSED** | Phase 2 separates descriptive per-call exposure from per-engine and per-session failure estimands. It declares 1% per engine/session unacceptable and derives 299 blocks for 95% detection power, rounded to 300 independent fresh engines. One frozen-selected session per engine supplies 300 powered session blocks; all 600 exposed sessions must be event-free. The design retains 24 hours and 250,000 calls only as floors and confines `3/N_calls` to descriptive accounting. |
| State-stratified stress exposure | **CLOSED** | The gate fixes minima for accepted pairs, all three rejection classes, every rejection's pending drain, abort, cancel, teardown, and repeated request/session creation, with per-engine/session distribution rules. PAD acceptance cannot replace a transition minimum, and deterministic failing-prefix replay remains mandatory before and after every engine segment. |
| Recovery evidence and 240-second ceiling | **CLOSED** | The lifecycle section reports the admitted 115.571–154.015-second Nano initialization range and uses 247.954–256.016-second full-service readiness only as context. The 240-second value is expressly a conservative preregistered experiment ceiling with named margin components, not an evidence-derived SLO or lower bound; measured recovery distribution and product acceptance remain required. |
| Standing and stop/go rules | **CLOSED** | The design retains option (a) NO-GO, whole-Nano-epoch option (c) NO-GO, the no-timeout-retry discipline, pairing inadmissibility, and historical-precedent-only ranking status. Option (b) can become GO only after causal localization, exact transactional replay, and the powered stress gate; failure to reproduce/localize, preserve exactness, or preserve amortized latency retires pairing from Step 5. |

## Final disposition

There are **no remaining round-1 deltas**. Track 2 is concluded at **NOT
CLOSED**, with option (b) **NEEDS-EXPERIMENT** under the repaired matched and
construction-stratified preregistrations as the only bounded reopening route.
