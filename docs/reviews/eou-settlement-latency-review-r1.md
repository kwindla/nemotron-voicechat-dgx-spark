# User-EOU settlement latency review — round 1

Date: 2026-08-16 UTC

**Verdict: FAIL — commit is not authorized.** The latency and empty-response
receipts support their headline claims, and no receive-pump FIFO crossing or
model-path deadlock was found in the reviewed implementation. However, the
early arm is not confined to complete real frames observed before the commit
edge: the commit handler's zero-padded partial-frame flush can create
`prearmed_blank_frames` after `client_input_turn_commit_received` and reduce
the recorded post-commit deficit. Required partial-frame and event-ordering
coverage also does not pin the claimed boundaries.

No live campaign, browser run, model connection, container action, or commit
was performed. Review probes were CPU-only and used self-cleaning temporary
directories. The worktree was read-only except for this review file.

## Blocking findings and required deltas

### 1. A zero-padded partial frame can arm after the commit edge

The commit path emits `client_input_turn_commit_received`, then, when
`pcm_buffer.pending_samples` is nonzero, calls `pcm_buffer.flush()` and feeds
the resulting zero-padded 80 ms frame through `process_pcm()` as ordinary
microphone audio (`server.py:8104-8112`). That call carries neither
`fence_settlement=True` nor `synthetic_control=True`. Consequently it enters
the early-arm branch at `server.py:6595-6633`, despite being generated after
the server observed commit and despite containing synthetic zero padding.

A reviewer-only endpoint reproduction sent one full speech frame, one 40 ms
quiet PCM append, commit, and stop with a three-frame fence. The trace order
and settlement evidence were:

```text
client_input_turn_commit_received
user_eou_fence_prearmed: prearmed_blank_frames=1
user_eou_settled:
  starting_blank_frames=1
  prearmed_blank_frames=1
  post_commit_blank_deficit=2
  model_steps=2
  real_audio_model_steps=0
  synthetic_model_steps=2
```

Thus the evidence labels an after-commit, half-real/half-zero-padded model
frame as prearmed, and settlement takes two rather than three recorded steps.
This directly contradicts the claimed precommit early-arm provenance and the
claim that synthetic frames cannot arm. It also leaves the source accounting
without a category for the padded frame that advanced the physical blank
count before `UserEouSettlement.begin()`.

There is a second, presently harmless instance of the same provenance hole:
the separate terminal-BOS zero frame at `server.py:8217-8223` is sent while
`open_client_turn_id` is still set and is not marked synthetic or settlement,
so it can create a stale early arm before the ID is cleared. The next turn
currently clears that arm, but the categorical no-synthetic-arm invariant is
not true.

Required delta:

- Freeze the eligible prearm at the actual commit-receipt edge, before any
  partial-buffer flush or recovery work that occurs after that edge.
- Give every padded/synthetic model frame explicit provenance that cannot
  create or advance an early arm, including the commit-tail flush and separate
  terminal-BOS frame.
- Account honestly for any after-commit padded frame that advances the blank
  fence. It must not appear as `prearmed_blank_frames`; its effect and source
  must be represented in the post-commit settlement evidence.

### 2. The new tests do not pin the required partial/event boundaries

The exact-gate unit test does exercise the strict `>=` rejection in
`split_fence_eligible_pcm()`, and the helper test preserves a partial suffix
that occurs *after an already ineligible speech frame*. Neither test exercises
a quiet incomplete frame at the commit boundary or in the bounded receive
queue.

The postcommit speech endpoint test proves only that settlement reports zero
real-audio steps and sets its blocked flag. It does not assert that the exact
speech PCM is subsequently processed once, nor that it remains ahead of the
following `session.stop`. There is no quiet-audio / non-audio-event / audio
interleaving test. There is also no early-arm quiet / exact-gate frame / commit
test proving that the exact-gate frame revokes the arm.

Required delta:

- Add a precommit quiet-arm then exact-gate (and above-gate) frame regression;
  commit must retain no prearm and must expose the full resulting deficit.
- Add the reproduced partial-before-commit case and a partial-after-commit
  case. A partial must not be credited as a complete real fence frame, and a
  padded flush must not be reported as prearmed real evidence.
- Add an endpoint FIFO regression with queued quiet audio, a non-audio client
  event, and more audio. Prove settlement does not cross the event and prove
  the event precedes later audio on the ordinary path.
- Strengthen the barge-in regression to prove the rejected PCM is requeued and
  processed exactly once before the next queued client event.
- Add a bounded-pump liveness test with the model step blocked and queued input
  at or near the pump bound, proving release makes forward progress and
  shutdown cannot strand the endpoint.

### 3. Settlement-source raw receipts are not retained with the report

`settlement-source-summary.json` lists ten mutable absolute paths below
`~/.local/state/nemotron-voicechat/traces/model/`. Those files currently exist,
and independent recomputation from them exactly matches the summary. They are
not retained below `reports/fence-latency/`, have no hashes in the aggregate,
and will not travel with the proposed repository commit. The brief required
all receipts to be retained under the report directory.

Required delta: copy the exact ten source event streams (or a lossless focused
extraction containing every `user_eou_settled` and block event) below
`reports/fence-latency/`, record hashes, and regenerate the aggregate from
those retained paths.

## Receive-pump and fence-consumption audit

Apart from the commit-tail issue above, the queued-frame path is fail-closed on
the reviewed boundaries:

- `split_fence_eligible_pcm()` considers only complete 2,560-byte PCM16 model
  frames, rejects non-finite RMS, and rejects RMS equal to or above the gate.
- `take_queued_settlement_frame()` stops on the first non-audio or invalid
  event and puts it back at the head. If the inspected full frame is
  speech-bearing, it puts the frame and remainder back and permanently stops
  real-audio settlement consumption for that commit.
- Requeue order is correct for the cases inspected: a remainder is placed
  before a non-audio event already put into `received_ahead`, while that event
  remains before later messages still in `inbound_messages`. No path that
  crosses a non-audio event was found.
- There is one websocket reader and FIFO `asyncio.Queue` insertion. A full
  256-message queue suspends only the receive pump; it holds neither
  `model_input_lock` nor `send_lock`. Model calls run in the executor, allowing
  the event loop and pump to run. No static lock cycle or starvation path was
  found.

An at/above-gate queued barge-in frame therefore cannot be consumed as fence
silence by this code. A sub-gate frame can be consumed, but by the specified
RMS contract it is not speech-bearing. The missing endpoint assertions in
finding 2 still need to protect this conclusion against regression.

## Unchanged-contract and typed-input audit

The fence target still comes directly from
`engine.user_eou_settlement_blank_frames()`, and `max_model_steps` remains
exactly `target_blank_frames * 2`. Direct diff and AST comparison found the
terminal epoch validator, `_reject_initial_activity`, and separate-terminal
validator unchanged. The ordinary and fused observers add only source counters
and a per-step source field around the existing response, audio, assistant
text, function text, and terminal-BOS guards; those guards retain their prior
conditions.

The typed worker path is unchanged: AST comparison found `run_typed_job()` and
`inject_trailing_silence()` identical to base `664526b`. Typed settlement
cannot take queued microphone frames because real-audio polling remains
guarded by `source == "microphone"`; it continues to use synthetic fallback.
The endpoint module's existing typed tests passed in the fresh run.

## V1/V2 receipt verification

Independent reads of all 20 retained protocol event streams produced:

| leg | sessions | response.created | response.done | nonempty assistant text | empty responses | fatal errors |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 10 | 80 | 80 | 80 | 0 | 0 |
| candidate | 10 | 80 | 80 | 80 | 0 | 0 |

The published six-turn medians recompute exactly from the per-session
summaries:

- baseline: `[861.144, 1339.872, 1750.133, 2157.566, 2525.084, 2919.301]` ms;
- candidate: `[877.188, 830.789, 770.575, 758.007, 754.541, 761.040]` ms.

The baseline turn-1-to-turn-6 increase is 2,058.157 ms using the displayed
rounded medians (the completion note reports 2,058.156 ms from unrounded
values). The candidate change is -116.148 ms. Every candidate replicate is
non-monotonic and every turn 2–6 is within its own first-turn value + 200 ms.
The claimed accumulation-slope elimination is supported.

Independent aggregation of the ten currently available model trace streams
found 80 settlements, target set `{10}`, all fence/bound checks true, 78 using
real audio, 704 real steps, 109 synthetic steps, two zero-step/zero-deficit
settlements, and zero speech-gate blocks. For every settlement,
`model_steps == real_audio_model_steps + synthetic_model_steps == len(steps)`,
and the per-step source labels reproduce the two aggregate counters. The
numbers in `settlement-source-summary.json` are internally and externally
consistent; finding 3 concerns retention, not numerical drift.

## CPU-only checks

- Targeted endpoint/server modules: **172 passed, 29 subtests passed**.
- Three named new boundary tests: **3 passed**.
- Reviewer partial-frame endpoint reproduction: **defect reproduced** with
  `prearmed_blank_frames=1` after commit.
- Ruff over the three changed tracked files plus the untracked rapid-turn tool:
  **all checks passed**.
- `git diff --check`: **passed**.

The green suite does not override the reproduced invariant violation because
the missing boundary cases are precisely outside its assertions.
