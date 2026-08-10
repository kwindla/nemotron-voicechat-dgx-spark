# Codex adversarial review — revision 3

Date: 2026-08-08  
Plan: `docs/direct-text-input-plan.md`, revision 3

## Disposition

- The response-cancellation seam is now a bounded acoustic drain to an observed
  terminal response, not a one-step EOS assumption. It guards every synthetic
  position, suppresses cancelled output, freezes early RNNT evidence, and fails
  closed on tail/position limits.
- Audio processed before the asynchronously emitted start barrier is retained as
  bounded pending evidence and attached only after the explicit turn opens.
  Explicit commit snapshots and fences the RNNT display/hypothesis, covering the
  rapid re-speech stale-transcript case.
- Pocket admission and exact context commit are separated at the existing
  `injection_started` barrier. Pocket loses both of its redundant clocks, uses
  explicit internal start/commit, and becomes non-cancellable after mutation.
- Raw PCM, serialized JSON/base64, controls, held turns, and deferred tool/typed
  inputs all have bounded accounting and fatal overflow behavior.
- Typed and function-call epochs now use one half-duplex input arbiter. Normal
  microphone overlap is held and released in order rather than treated as a
  protocol fault; a correlated post-FC-applied acknowledgement supplies the
  otherwise-missing safe release point. Multiple held turns and post-send-only
  acknowledgement deadlines are explicitly tested.
- The VAD false-positive tradeoff is explicit rather than accidental. Exact
  Silero parameters are pinned, ambient rejection is qualified, and any signal
  that passes VAD remains authoritative so quiet audio-native evidence is not
  discarded merely because RNNT produced no text.
- Direct injection is gated on a pipeline-level state inventory and per-position
  transaction, preserves both feedback channels on its first position, has
  objective semantic/tool thresholds, and has a bounded non-cancellable window.

## Residual risks accepted by the plan

The plan intentionally accepts that a deliberately induced false-positive VAD
turn can cancel an answer and elicit an odd response. This is observable and
qualified, and is preferable at this stage to an RNNT-token filter that would
silently reject quiet speech understood through Nano's audio perception path.
Likewise, typed/tool atomic epochs can add microphone latency, but their buffers,
metrics, and overflow behavior are explicit.

## Verdict

**VERDICT: APPROVE**

No blocker or important/should-fix finding remains. Implementation should still
follow the per-numbered-step review cadence and stop the direct path at its
semantic gate if neither candidate token form qualifies.
