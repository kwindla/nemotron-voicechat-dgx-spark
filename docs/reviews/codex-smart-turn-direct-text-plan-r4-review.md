# Codex final plan review — revision 4

Date: 2026-08-08  
Plan: `docs/direct-text-input-plan.md`, revision 4

Revision 4 resolves the sole revision-3 important finding without weakening
wire integrity:

- The immediate send queue remains bounded at 256 KiB raw PCM plus explicit
  serialized/control budgets and is fatal only after sustained backpressure,
  where partial wire transmission cannot be rolled back.
- The separate held-input arbiter is 1 MiB (about 32 s PCM), derived above the
  20 s tool timeout plus Smart Turn fallback. It supports multiple complete
  turn records and starts acknowledgement deadlines only on actual send.
- A newly overflowing held turn is dropped whole and locally terminated with a
  surfaced diagnostic; earlier turns and the session survive. No start or
  partial audio for that turn reaches the wire.
- The defensive server queue remains fatal because its use means a negotiated
  client violated arbitration and may already have transmitted a partial turn.
- Function-result application has a correlated 25 s acknowledgement deadline.
- The Smart Turn stop strategy uses its required keyword-only analyzer
  argument. Pocket's start/context/first 80 ms model mutation is one locked,
  cancellation-rechecked transition.

**VERDICT: APPROVE**

No blocker or important/should-fix plan finding remains.
