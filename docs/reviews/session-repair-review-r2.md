# Session repair adversarial review R2

## Verdict: FAIL — commit is not merge-ready

Commit reviewed: `fb4654d3ef89718e1162c5be47acde8bdbd65b01`.

I1 and the revised I4 adjudicator now satisfy the R1 source requirements. The
attempt-5 narrative inconsistency is also reconciled. I2 is not closed, however:
there is a new, independently reproduced typed-input interruption path that binds
the superseded microphone turn to a new response. In addition, the promoted image
provably omits the final I2 repair, and the browser evidence manifests do not hash
their final reports. Those are blocking deltas.

## Blocking deltas

1. **I2 remains fail-open when the interrupting turn is typed.** The microphone
   `input_audio_buffer.turn_start` path performs the new clear under
   `model_input_lock` after interrupting an unpublished function cycle
   (`src/nemotron_voicechat_runtime/server.py:8261-8289`). The typed request path
   has no corresponding ownership transition. It rejects only a published
   `function_cycle_call_id` (`server.py:8608-8622`), so a request arriving while
   `pending_bos_settlement` is set but before function publication is accepted.
   `run_typed_job` then begins model mutation and settlement without interrupting
   the unpublished worker or clearing `pending_response_user_text` /
   `pending_response_attribution` (`server.py:7393-7525`). A BOS on that new typed
   audio therefore reaches `_bind_pending_transcript_at_bos` and consumes the old
   turn's binding (`server.py:4609-4635`).

   I reproduced this through the real FastAPI websocket endpoint and its real
   locking, protocol, typed-job, settlement, and `send_step` paths, with only the
   model engine and Pocket synthesis made deterministic. The sequence was:
   microphone commit -> deferred FC before publication -> new typed request ->
   typed-audio BOS. Immediately before the typed request the pending binding was
   `old deferred tool request`, source `microphone`, source ID `1`, and the function
   was still unpublished. The endpoint accepted the typed request. Its BOS was
   emitted with:

   ```text
   response_attribution = {
     "kind": "user_turn",
     "source": "microphone",
     "source_id": 1,
     "user_text_bound": "old deferred tool request"
   }
   unpublished_interrupts = 0
   cleared_response_bindings = 0
   ```

   The current endpoint tests do drive the server rather than calling attribution
   helpers directly: they cover microphone interruption/pre-commit BOS, the
   publication-race/deferred-input branch, typed and microphone EOTR/forced-BOS
   recovery, fence settlement, and a two-tool continuation. But none drives an old
   deferred turn interrupted by a **new typed** turn. In particular,
   `test_typed_finished_is_immediate_readiness_barrier_during_deferred_tool` is old
   typed -> new microphone, so it still reaches the protected microphone edge.

   Define and enforce typed ownership at this unpublished-cycle edge. Either
   reject/defer the typed request fail-closed while `pending_bos_settlement` owns an
   unpublished function cycle, or perform the same bounded interrupt and binding
   clear before the first typed model mutation, with an explicit publication-wins
   branch. Add an endpoint mutation that forces a typed-audio pre-commit BOS and
   proves the old text/source cannot bind.

2. **The promoted runtime image does not contain the final I2 repair.** The checked-in
   server at `fb4654d` hashes to
   `a8333c5016e36775c68a6d6c3cc6d9a32e38ce4d23149cf59259bd0e62eb2877`.
   `campaign-manifest.json` and `final-stack/status.json` identify promoted image
   `sha256:c79934692503646531c9be56c67813e473944fc53e21728bdb72db1d5926c53d`
   with server hash
   `6db56d76532e77a821a73ce78d36ee3ef1b5d17fadb6beff72f27cfc9b041764`.
   This is not an unrelated build difference: removing exactly
   `clear_pending_response_binding` and its locked-edge invocation from the current
   source reproduces `6db56d...` byte-for-byte. Thus the final r3 promotion, the
   corrected browser replay, and the post directive arm do not exercise the R1 I2
   delta. Rebuild/promote from the repaired source, record the new immutable image
   and server hashes, and regenerate the affected retained evidence.

3. **The campaign hashes match only at the top level; the retained browser manifests
   are internally stale.** Every direct artifact digest in
   `reports/session-repair/campaign-manifest.json` matches its current file, as do
   the directive/replay scripts, demo source, and retained input PCM. But both
   `capture-manifest.json` and final `manifest.json` retain a pre-final-report entry:

   - attempt 5 records `report.json` as 27,394 bytes / `c18701e9...`; the retained
     report is 49,637 bytes / `84d9a8e4...`;
   - corrected replay records 30,278 bytes / `8c9d55c2...`; the retained report is
     53,258 bytes / `b2ae8de8...`.

   The source cause is visible at
   `tools/qualification/session_replay_browser_suite.py:881-897`: `run_asr` rewrites
   the final report, then copies the capture manifest without refreshing its
   `report.json` record. Regenerate the final manifest from the finalized report
   (and pin this with a mutation test) so the campaign is transitively
   self-consistent, not merely wrapped by a matching top-level digest.

## Confirmed closed items

### I1 — positive, evidence-based termination: PASS

The gate totally parses the trace and recognizes an open response as
`session_terminated` only when the stop is after BOS, is not the clean
`client_stop`, and is supported by either (a) one matching
`terminal_session_claimed` strictly after BOS and before the stop or (b) the narrow
unclaimed `disconnect` lifecycle
(`tools/qualification/response_completion_gate.py:304-480`). Schema errors are
terminal gate failures.

Independent mutations confirmed fail-closed behavior for unclaimed `client_stpo`
and `test`, clean `client_stop`, claim-before-BOS, claim/stop mismatch, duplicate
claim, duplicate stop, missing/invalid claim or stop reasons, claim after stop, and
any post-stop record. The explicit unclaimed `disconnect` passes. A novel claimed
fatal reason passes only with the required matching producer claim and ordering,
which is the intended positive-evidence contract rather than the old unknown-reason
shortcut.

The pinned known-bad watchdog trace still fails with valid schema and four audible
watchdog defects (SHA-256 `bd093498395adf3222fcd90c34fd7de3b5830015a41b414c712998eee1f0b703`).
All eight repaired retained traces pass with zero audible watchdog defects. The
session-repair trace passes with nine responses and one evidenced termination at
BOS 975; its matching `internal_error` claim is record 4728 and stop is record
4729. The pre-existing deletion, relabeling, nesting, unknown-event, acoustic,
threshold, and boundary-topology mutations also still fail. I did not find a new
I1 fail-open.

### I2 — covered branches versus remaining hole

For a new microphone turn, the superseded response binding is cleared at the
locked unpublished-cycle interruption edge and is not cleared when
`function_cycle_call_id` has already published and the input is deferred. The
endpoint tests genuinely traverse these branches. Normal typed and microphone
commit -> function output -> EOTR -> forced BOS bind the correct source/text, and
the two-tool test consumes the user binding once then labels the second BOS as an
explicit `post_tool_continuation`. New commits also supersede an unopened
post-tool-only continuation. These checks pass, but they do not close the typed
interruption blocker above.

### I4 — stated-result adjudication: PASS (the measured post arm still fails)

`evaluate_replicate` now requires both eligible ordinals 3 and 4 to contain a
normalized complete Pacific result, rejects any superseded UTC result, and requires
exact call/lifecycle/error conditions
(`tools/qualification/session_directive_freshness_fixture.py:153-223`). The unit
mutations omit the turn-3 result entirely and omit `PDT` from turn 4; both fail with
`missing_fresh_result_responses` and no reliance on a stale-string hit.

The retained v2 reports agree with the new gate. In all three pre and all three
post replicates, turn 3 is only `" It"`; turn 4 states the Pacific result. Both arms
therefore have 0/3 passing replicates and `freshness_gate_passed=false` despite
0/6 stale responses. The comparison retains `rate_changed=false`,
`rate_improved=false`, `post_freshness_gate_passed=false`, and
`i4_fix_supported=false`. No I4 fix is claimed.

### Attempt 5 narrative: reconciled

`results.md` no longer says attempt 5 passed a regenerated gate. It labels the
separately retained `post-fix-browser-corrected/` run as accepted, uses attempt 5
only as tool-propensity evidence, and the campaign manifest explicitly calls the
attempt-5 report `under_original_gate`. The retained attempt-5 report remains
honestly `passed=false` with `phase_expectation_passed=false`. That resolves the R1
semantic inconsistency, apart from the manifest-digest defect above.

### No-text watchdog, wall deadline, fence, and other regressions

The server delta does not alter the watchdog, whole-response wall-deadline, or EOU
fence algorithms. CPU-only execution passed the completion/directive/endpoint
group (90 tests, 4 subtests) and the realtime-server/Pipecat/replay group (148
tests, 27 subtests). These include the retained watchdog campaign and mutations,
wall-time-without-model-frames deadline behavior, exact twice-fence bounds,
prearm/queued-audio/fused-BOS settlement, publication and EOTR recovery, prompt
grounding, and replay adjudication. `git diff --check` also passes. No live service,
GPU, browser, model, or network run was performed.

The code-level no-text, deadline, and fence work did not regress. The promotion
manifest cannot be confirmed, because its exact server hash proves that the
promoted image predates the final locked-edge repair.
