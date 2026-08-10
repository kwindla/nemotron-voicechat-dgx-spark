# Post-function-call PAD-pair recovery investigation

## Goal

Explain and eliminate the stochastic state transition that disabled conditional PAD-pair
drafting after a function-call cycle in one retained qualification run. The defect matters
because it changes an otherwise paired PAD stretch into sequential Nano calls, raising frame
time above the 80 ms input cadence and accumulating playback debt.

This work does not change tool selection, function schemas, prompts, timeouts, or normal
function-call behavior. A behavioral fix will be proposed only after an instrumented failing
capture identifies the first divergent state.

## Retained evidence

The original failing and passing traces were copied out of volatile `/tmp` before further
work. They are retained at:

`~/.local/state/nemotron-voicechat/retained/post-fc-pair-recovery-20260806`

The retained tree contains 1,279 files (911 MB). An `rsync -acn` source/destination
comparison was empty after the copy. `SHA256SUMS` hashes the retained contents; its SHA-256
is `79892d86c64bf39ace12fbcf510839bcb8fe9b9c6d2595f770319502bd0a7c3f`.

The failing r2 trace proves a 510-frame sequential-single stretch after the last repeated
function-call cycle. Mean Nano inference was about 82.7 ms in that stretch, and the implied
queue debt peaked at 2.98 seconds. Pairing resumed at the next typed/user turn. The passing
r4 run is a weak control because it produced only one function call.

## Corrected causal hypothesis

The original scheduler-latch hypotheses are not supported by the retained trace or current
code. `pending`, `needs_correction`, `sequential_mode`, and the per-call bypass context all
have bounded clearing paths and cannot explain 510 ordinary engine calls. The leading
hypothesis is upstream of the scheduler: the effective function-channel token did not return
to PAD after the malformed/repeated function-call sequence. That makes conditional drafting
ineligible by policy on every subsequent frame. The old trace did not record the effective
token IDs, so this remains a hypothesis until reproduced with telemetry.

## Diagnostic boundary

Commit `8e42188` adds opt-in, scalar-only diagnostics behind
`VOICECHAT_NANO_PAD_PAIR_TRACE=1` / `./voicechat up --trace-pad-pair`. Production behavior and
the default trace schema are unchanged when the flag is absent. Enabled traces record:

- every engine-call scheduling decision and its request/context identity;
- pending, correction, sequential, bypass, and eligibility state;
- actual effective text and function token IDs plus the scheduler PAD ID;
- a log-only 25-frame watchdog for an idle agent with no pair-buffer event; and
- diagnostic error/drop counters, with observation failures isolated from inference.

The boundary passed 242 host tests (7 visible dependency skips), ruff, and 86 authoritative
offline container tests with the pinned Speech source on `PYTHONPATH`. Fable's adversarial
telemetry review approved the corrected implementation for live capture.

## Live capture campaign

The trace-only image is
`pipecat-ai/nemotron-voicechat-dgx-spark:post-fc-trace`
(`sha256:6dffcafbe23b52b15009179b89855e9b89fc5e951a1334ce4dfe80a6ccf5eba7`). It is separate
from the qualified production image. Campaign evidence is written under the retained tree's
`live-campaign-1` directory.

The checkpoint has a known limited conversational training horizon. Long sessions can
therefore create model-state degradation that is a confound for this transition. The live
discriminator is fixed in advance at **10 independent sessions**, each no longer than about
two minutes. Each session drives this sequence:

1. two calls to `get_current_utc_time` within one response;
2. a fully separated, explicit non-tool math turn; and
3. a second response containing two calls to the same tool.

All sessions run to their declared boundary even if a stall is observed. Evaluation uses the
effective-token and pair-decision stream, not aggregate queue debt. A clean session requires
the function channel to return to PAD and pairing to resume at the first eligible boundary
after each function response closes. Every watchdog, trace error, request-ID mismatch, or
frame-level invariant violation is retained and investigated.

The interpretation is also fixed in advance:

- An instrumented failure localizes the first divergent effective token/state and becomes
  the source for a deterministic state-machine replay and the smallest evidence-driven fix.
- Ten clean short sessions do **not** prove the historic defect absent. They mean the trigger
  requires the long-context condition. In that case, one instrumented long session is the
  necessary follow-up diagnostic despite the quality confound; its result cannot be used as
  a short-session product-quality qualification.

### Experiments

- **Repro 1 (exploratory, stopped at about five minutes):** the identical repeated prompt
  produced two healthy double-call cycles, then the model answered later repetitions from
  context without calling the tool. No watchdog fired. This established a telemetry-complete
  healthy control but was stopped after the user clarified the two-minute session limit.
- **Repro 2 (exploratory, stopped near two minutes):** varied tool and non-tool prompts
  produced two double-call cycles before stop. No watchdog fired. It motivated the fixed
  independent-session campaign above.
- **Short-session campaign (10/10 complete):** the aggregate gate failed honestly. Only
  `bounded-session-07` produced three separately answered/closed turns with exactly two tool
  calls in each tool response. The other sessions produced only two responses for three
  completed typed injections; four also over-called (response groups ranged up to 2+6).
  These are trigger-shape/turn-separation failures, not scheduler passes. Across the 19
  response-close recovery fences that were observable, however, the effective channels were
  PAD on the first subsequent model frame and pairing buffered within 0–1 frame. There were
  zero watchdogs, effective-token mismatches, request-ID mismatches, dropped diagnostic
  events, trace errors, record errors, or scheduler invariant violations. One over-call
  response closed on the final model frame and was classified fail-closed as
  `no_model_step_after_response_close`. The machine-readable report is
  `live-campaign-1/bounded-campaign-analysis.json` in the retained evidence tree.
  Its SHA-256 is
  `47d29b3d5902cd07f253bda681df41c4c67a680fe0b8e6706d7f0379335a0d33`.

  This campaign did not reproduce the historic recovery defect within the supported
  two-minute horizon. It also did not satisfy the precommitted all-session fixture gate, so
  it cannot by itself prove the trigger requires long context. The next discriminator must
  serialize prompts on observed `response.done` boundaries rather than fixed wall-clock
  intervals, while continuing paced silence so each recovery fence remains observable.

- **Serialized short-session campaign (3/3 complete):** prompts were sent only after the
  attributed `response.done` plus two seconds, while 80 ms silence frames continued for the
  full session. Sessions 01 and 03 passed the complete fixture: three answered turns, exact
  2+2 calls, and four recovery fences with immediate PAD return and buffering within one
  frame. Session 02 failed closed. Its raw events show that the first prompt produced an
  early spoken response without a tool call; after the non-tool prompt arrived, the model
  emitted the two delayed tool calls in the original turn and did not close that response
  until the session boundary. The serialized sender therefore timed out, recorded the error,
  and did not issue prompt three. This is a model/tool sequencing failure, not a recovered
  scheduler stall or a mere attribution gap. No watchdog or trace-integrity fault occurred.
  The serialized campaign report SHA-256 is
  `128467bd41d24a6ffff481e1114a7e68a92a7bee3b676caf0eeda9c9bdca7cb0`.

## Disposition

The recovery mechanism is healthy in the supported short-context evidence, including under
repeated and over-called function cycles: 23 observable response-close fences returned to PAD
on the first subsequent model frame and buffered within 0–1 frame. The historic stall remains
a real stochastic late-session weakness inside the 12,000-frame production envelope, but its
old trace lacks the effective function-token IDs needed to localize an upstream token-stream
error versus a policy decision.

No scheduler behavior is changed on incomplete evidence. The opt-in trace remains available
for future long qualification runs; a natural recurrence will provide the deterministic token
sequence needed for replay, localization, and a narrowly gated fix. Until then, the limitation
and its observed self-recovery on the next typed/user turn remain documented.
