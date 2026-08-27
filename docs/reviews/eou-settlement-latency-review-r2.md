# User-EOU settlement latency review — round 2

Date: 2026-08-16 UTC

**Verdict: PASS — commit is authorized.** All three round-1 deltas are closed.
The arming path is now confined to complete, pre-commit model frames made from
ordinary microphone audio; the missing boundary cases are pinned; and the r2
candidate/source evidence is fresh, retained, repository-relative, and
independently reproducible from the manifest and source streams. No fence-target
or terminal contract drift was found, and fresh CPU-only regression runs are
green.

No campaign, browser run, model connection, GPU use, container mutation, or
commit was performed. The worktree was read-only except for this review file.

## Round-1 delta closure

### 1. Commit-edge and padded-frame provenance: closed

`process_pcm()` derives or accepts a model-input provenance and admits the early
arm only when all of these conditions hold (`server.py:6494-6648`): the source
is `microphone`, provenance is exactly `microphone_audio`, a client turn is
open, and the call is neither synthetic control nor fence settlement. The arm
logic is inside the `pcm_buffer.push()` frame loop, so an incomplete packet
cannot arm until it forms a complete 1,280-sample model frame. Quiet eligibility
remains strict (`rms_dbfs < speech_gate_dbfs`); a non-finite, equal-gate, or
above-gate complete frame revokes an existing arm.

Every `process_pcm()` call site was inspected. The only default
`microphone_audio` sites are ordinary microphone append and real partial-onset
replay. Ordinary typed synthesis defaults to `typed_audio`, which fails the
microphone-source condition. Every control/padded path is explicit:
`queued_microphone_audio` or `synthetic_settlement`,
`synthetic_response_cancel`, `synthetic_response_drain`,
`typed_trailing_silence`, `typed_tail_padding`,
`synthetic_function_recovery`, `commit_tail_padding`, and
`synthetic_terminal_bos`. The two settlement provenances are additionally
guarded by `fence_settlement=True`; microphone response/function/terminal
control frames also carry `synthetic_control=True` where applicable. No
synthetic or padded call site can satisfy the arming predicate.

At the commit-receipt edge, the handler copies `early_fence_arm` to
`commit_prearm` and clears the live arm before emitting
`client_input_turn_commit_received` or doing any flush, EarTTS, recovery, or
settlement work (`server.py:8112-8117`). The partial commit tail is processed
with provenance `commit_tail_padding`; within settlement evidence, its effect
appears only in `post_commit_steps`, with real/padded samples, blank counts
before/after, blank advance, and model frame (`server.py:8155-8183`). It is not
credited to `prearmed_blank_frames` or the settlement `steps`/source counters.

The committed round-1 reproduction test
`test_commit_tail_padding_cannot_arm_after_commit_receipt` asserts the exact
evidence, not just event presence:

```text
prearmed_blank_frames=0
starting_blank_frames=1
post_commit_blank_deficit=2
model_steps=2
real_audio_model_steps=0
synthetic_model_steps=2
post_commit_steps=[{
  model_input_provenance=commit_tail_padding,
  real_samples=640,
  padded_samples=640,
  blank_frames_before=0,
  blank_frames_after=1,
  blank_frames_advanced=1,
  model_frame=1
}]
```

It also proves the padded-frame event follows commit receipt and that no prearm
event occurs afterward. The separate tool-session terminal test proves the
last post-commit model input has `synthetic_terminal_bos` provenance and cannot
recreate a stale arm.

### 2. Boundary coverage: closed

The test set now pins each requested boundary:

| Boundary | Pinned evidence |
|---|---|
| Gate equality and above-gate RMS | `test_fence_audio_eligibility_is_strict_at_speech_gate` uses the quantized frame's measured RMS as the exact threshold. `test_precommit_arm_is_revoked_at_and_above_the_speech_gate` arms with quiet audio, observes two revocations (equal then greater), and requires a full three-step post-commit deficit. |
| Every partial size | `test_fence_audio_eligibility_preserves_every_partial_frame_size` iterates 1 through 1,279 whole samples, both alone and after one complete eligible frame, and requires the partial bytes to remain unconsumed. Endpoint tests cover the partial-before-commit padded flush and partial-after-commit ordinary-path flush. |
| FIFO event ordering | `test_settlement_does_not_cross_non_audio_event_in_receive_fifo` queues quiet audio, `session.update`, then later speech. It requires one real plus two synthetic settlement steps, settlement before the event, the event before later audio, and one later-audio model call. |
| Barge-in requeue once | `test_postcommit_speech_is_not_fence_eligible` requires a speech-gate block, zero real settlement steps, exactly one model call for the rejected PCM, and ordering `settled < barge audio < session.stop`. |
| Pump-bound release | `test_bounded_receive_pump_releases_and_shutdown_makes_progress` blocks the model step, queues `REALTIME_RECEIVE_QUEUE_MESSAGES + 1` events plus stop against the 256-message internal bound, proves the endpoint remains blocked while the receive pump fills, then releases the step and requires bounded endpoint completion, websocket close, abort, and observed stop. |
| Terminal-BOS provenance | `test_tool_session_keeps_separate_settlement_and_legacy_eou_position` requires no post-commit prearm and exact `synthetic_terminal_bos` provenance for the separate terminal position. |

Fresh execution of the endpoint/server pair produced **178 passed, 29 subtests
passed**. Fresh execution of the complete targeted endpoint, settlement,
function, generation-watchdog, and sustained set produced **227 passed, 29
subtests passed**.

### 3. Retained r2 evidence: closed

The manifest SHA-256 recomputes to
`4956d89c1c22819a9ce7e36bc0e886e14172c43387408518f8d384a370c4993c`.
All 53 listed paths are unique, repository-relative paths that resolve inside
this repository, and every listed byte size and SHA-256 matches:

- 20 protocol streams covering the complete baseline/candidate-r2
  no-tool/with-tool, r1-r5 matrix;
- 20 matching per-session summaries;
- 10 focused settlement-source projections;
- the two r2 aggregates; and
- the fixture.

The ten source streams named by session UUID remain available. Their bytes and
SHA-256 values match each manifest `source_stream_*` identity. Independently
filtering each source for the seven declared event names produces a byte-for-byte
exact match to its retained projection, in the same line order; this is stronger
than set membership. The projections retain 80/80 settlements, their declared
line/event counts match, and no `user_eou_fence_prearmed` occurs between a
commit-receipt event and its settlement.

Freshness is also supported independently of filenames: none of the 20 r2
candidate protocol/summary hashes equals any of the 20 round-1 candidate
hashes; the r2 session UUIDs and receipt times are new; the candidate receipts
postdate the final source edit; both r2 health receipts report ready with
`active_client=false` and server source SHA-256
`b34ef7d9bb89b78354ab463b7f0e89c15c75c49f903ef6157563a6b4815858f3`,
which matches the reviewed file; and read-only image inspection returns the
manifest image ID
`sha256:f85d441b15e7b9d11e3cc40821c819692eeef1342c1a1887585d795035ffde50`.
The changed source totals (round 1: 704 real/109 synthetic; r2: 677/135)
provide an additional non-reuse check.

Independent r2 aggregation confirms 80 settlements, target set `[10]`, all
fences reached and bounds satisfied, 78 settlements using real audio, 677 real
steps, 135 synthetic steps, two zero-deficit/zero-step settlements, zero speech
blocks, zero post-commit padded steps in the full-frame fixture, and exact
agreement among `model_steps`, source counters, and step-list lengths.

The retained V1/V2 claims also recompute: both baseline and r2 candidate have
80 `response.created`, 80 completed `response.done`, 80 nonempty
`response.output_text.done`, zero empty text, and zero fatal errors. Candidate
six-turn medians recompute exactly as
`[893.627, 859.649, 760.600, 756.522, 756.745, 753.005]` ms; every replicate is
non-monotonic and every turn 2-6 is at most its own first turn plus 200 ms.

## Regression and unchanged-contract confirmation

AST comparison against base `664526bdde08db6bf95ef5e61150f8e943cf8c6f`
is exact for `user_eou_settlement_blank_frames`, `max_model_steps`,
`_validate_terminal_bos_epoch`, `observe_separate_terminal`,
`_reject_initial_activity`, and `observe_separate_bos_or_defer`. The target
still comes directly from `engine.user_eou_settlement_blank_frames()` and the
fatal settlement bound remains exactly `target_blank_frames * 2`. The fused
observer's response/audio/text/function/terminal guards are unchanged apart
from source accounting around them.

Fresh bridge-accounted full non-browser execution, deselecting exactly the
sealed non-evergreen Step-5 revision guard, produced **1323 passed, 15 skipped,
4 deselected, 29 subtests passed**. The canonical retained verifier freshly
reports **PASS (67 anchors)**. Ruff over `src`, `tests`, and `tools` with cache
disabled and `git diff --check` both pass. These results cover the behavior
that passed round 1 while adding the requested boundary protections; no
round-1-passing behavior regressed.
