# Fable adversarial review — Step 2A increment (diagnostic direct-position seam)

Reviewed: current worktrees of `nemotron-voicechat-dgx-spark` (at `e19f906` +
uncommitted) and `/home/khkramer/src/Speech-nemotron-voicechat` (4 files, ~705
inserted lines, uncommitted). Scope: the Step 2A deliverables — hardened
memory matrix, mutable-state inventory (`docs/direct-text-position-transaction.md`),
wrapper `source_embeddings`/`force_output_pad`, pipeline
`_direct_position_preflight` + `inject_user_source_position`, and the
`VoiceChatEngine` clock adapter. Diagnostic-only; not protocol-enabled —
confirmed: no server event exposes the seam.
Static evidence re-run by me: `tests/runtime` + `tests/pipecat` = **292
passed, 2 skipped, 3 subtests passed**; Speech `py_compile` clean.
No implementation code modified.

## Inventory (`docs/direct-text-position-transaction.md`)

I diffed the inventory against the independent owner list I derived from the
pinned sources before this increment landed. Verdict: **complete**, and in
four places deeper than my list (`_rnnt_*_cooldown_until`, RNNT prev-token
counters, `_tts_in_turn_content/_pads`, the pad-pair `sequential_mode` /
`call_context` internals). Every trap I pre-registered has an explicit,
correct decision:

- `_last_user_audio_emb` becomes the injected embedding *by decision*, with
  the FC-idle precondition making the FC-async fallback unable to observe it,
  and the next real audio frame replacing it.
- Perception-cache lag is acknowledged as an intentional invariant with a
  **mandatory** post-injection real-audio component test.
- EarTTS feedback retains the *actual* recurrent code and the codec runs its
  normal decode with samples discarded — the state-alignment-preserving
  choice, justified by this repo's own EarTTS trajectory-sensitivity history.
  Note this amends the plan's "substitute codec silence" wording (F2 below).
- The snapshot/failure contract is honest: "atomic" is defined as complete
  preflight, one serialized advancement, and mandatory termination after any
  post-mutation exception — no rollback theater.

## Seam verification (pipeline + wrapper)

- **Preflight truly precedes mutation one.** Every operation in
  `_direct_position_preflight` is a pure read (`_direct_request_state` reads
  scalars; `embed_tokens` lookup is side-effect-free; pad-pair check reads
  `pending` only). The first mutable-owner write (`_last_user_audio_emb`,
  inside `infer_one_step`) is unreachable before all rejects. The reject set
  covers: batch/chunk shape, prefilled slot, headroom for source+EOU,
  vLLM-both-engines, external-EOU mode, agent idle, FC idle (state, background
  worker, queued tool responses), all five wrapper edge latches, pad-pair env
  and buffered draft, active request status on both engines, vocabulary
  bounds, and the BF16 `[1,1,4480]` fusion contract.
- **Wrapper gating is complete for the injected position**: forced PADs are
  written to all three channels *before* the FC state machine, forced/RNNT
  turn-taking, and EarTTS observe the position (all three are additionally
  skipped under `force_output_pad`; RNNT is doubly unreachable since no
  perception ⇒ no `asr_emb`). Raw Nano/ASR/function predictions are preserved
  as diagnostics — exactly what Step 2B's token-form analysis needs.
- **Post-transaction verification** re-checks both request counts (+1 each),
  `frame_idx` (+1), all three PAD cells, object identity of perception/RNNT/FC
  state, display-length and audio-buffer invariance. These raises are
  correctly session-fatal per the contract.
- **Engine adapter** refuses on unreadable or misaligned prefill-relative
  clocks *before* calling the pipeline (mock-tested both ways), verifies both
  clocks advanced exactly once and remained equal after, and only then
  advances `frame_index`. The new `audio_frame_index` split is a subtle catch
  done right: `is_first` initialization of bufferer/perception/RNNT now keys
  on the *audio* clock, so an injection before the first real PCM frame
  cannot steal first-frame initialization.
- `enforce_eager` on the vLLM engine is a benign diagnostic knob.

## Matrix hardening (Step 2A prerequisite from Step 1 Q1)

Every element of my pre-registered baseline is implemented and
static-tested:

- Per-case secrets: pinned, validated unique, single-word, word-boundary
  matched (token-list membership, not substring).
- **Cross-case leakage is part of the pass condition**: a seeded case passes
  only when `recalled_secrets == (own secret,)` — any foreign secret fails
  the case with a distinct class.
- Unseeded negative control: structurally last (flag derived from case order,
  enforced by the gate recompute), and **non-vacuous** — it must produce a
  complete, non-empty, bracket-balanced response with audio, else
  `recall_empty_response`; a leak yields `negative_control_leak`.
- The gate can fail: flip tests cover control-failure, control-missing, and
  the recompute path; recall prompts are validated against *all* secrets;
  the WER matcher has its own missing/substituted-word self-test.

## Findings

- **F1 (should-fix).** The pipeline seam captures `iterator_id` and `status`
  before/after but never re-asserts them post-injection (only
  `generated_tokens` is compared). Iterator replacement or a status flip
  would pass silently. One-line assertions, or an explicit line item in the
  GPU component tests.
- **F2 (editorial).** Plan 2B/2C still says "substitute codec silence"; the
  implemented (and better) decision runs the real codec decode and discards
  samples. Reconcile the plan text with the inventory doc so the normative
  documents agree before Step 2B builds on them.
- **F3 (note).** Pad-pair scheduler internals (`sequential_mode`,
  `previous_effective_pad`, `call_context`) and the pipeline `bufferer` are
  asserted unchanged only by the inventory's word, not by the seam's
  post-checks. Acceptable for the CPU tier; the GPU tier must cover them
  (G5/G11 below).

## Required GPU component assertions (approval gate)

Static/mocked evidence cannot establish the following. Step 2A is not
approvable until each exists as a component test (or recorded artifact) and
passes on the real stack — FP32 and W8 where marked:

- **G1 — Real-engine clock semantics.** On the live AsyncLLM engines,
  `len(generated_tokens)` advances by exactly one per injected position for
  Nano *and* EarTTS, per position (not just end-state), and prefill-relative
  positions remain equal through a multi-token sequence plus the EOU step.
  The mocks assume list semantics; the real engine must confirm them.
- **G2 — Post-injection real-audio continuity** (the inventory's own
  mandatory test): after N injected positions, feed real PCM; assert
  perception-cache stepping accepts the `frame_idx` discontinuity via
  `effective_perception_frame_idx`, RNNT decodes normally, and the transcript
  is sane.
- **G3 — EarTTS recurrence equivalence**: N injected positions vs N ordinary
  idle-PAD silent-audio positions — compare recurrent `code` trajectory,
  `_tts_in_turn_pads` counters, and the first delivered post-sequence audio
  frame. (W8 and FP32.)
- **G4 — Codec recurrence alignment**: `codec_cache`/overlap state matches
  the sequential run; first delivered post-injection audio frame is
  artifact-free.
- **G5 — Zero-delta rejects on the real object graph**: induce every
  preflight rejection live; snapshot/compare all inventoried owners including
  wrapper latches, `_last_user_audio_emb`, pad-pair state, and both request
  positions.
- **G6 — Forced-PAD observability**: server `_turn_state()` on injected
  positions reads `pad` on all channels; `subword_mask=True` semantics are
  confirmed against EarTTS's subword handling; no display/trace deltas.
- **G7 — Post-mutation fatality**: fault-inject after the Nano call (e.g., in
  the EarTTS step); assert session termination, no reuse of the stream, and
  clean engine teardown.
- **G8 — First-position feedback exactness**: diagnostics prove position one
  fused the exact prior live agent/function tokens and later positions fused
  PAD.
- **G9 — Seam-level EOU smoke**: injection sequence → settlement → single
  EOU → BOS response with balanced brackets and audible audio.
- **G10 — `_last_user_audio_emb` replacement**: next real audio step replaces
  the injected embedding; FC-active rejection test confirms the fallback can
  never observe it.
- **G11 — Pad-pair invariance**: scheduler state unchanged across injection
  on the real engine; pending-draft rejection exercised with a synthetic
  state entry; env-enabled rejection exercised.
- **G12 — Hardened matrix live run**: the seven-session matrix (unique
  secrets + negative control) green on GPU with the Parakeet post-pass —
  this is the Step 2A prerequisite artifact that unblocks any
  direct/recalibrated qualification.

## Addendum — GPU probe artifact `direct-position-g1-g10-r3` (2026-08-08)

Reviewed from the raw report and WAV
(`~/.local/state/nemotron-voicechat/traces/model/direct-position-g1-g10-r3/`),
12 hidden positions ("Remember the secret word cobalt. Then say the secret
word.", token form `text`), pad-pair disabled.

**Corrected alignment invariant (supersedes the G1/G9 wording above).** The
probe's sole failed check (`eou_clocks_advance_once_and_align`) exposed a
wrong reviewer assumption, not a defect: the qualified runtime sets
`VOICECHAT_EARTTS_RESET_ON_BOS="1"` (frozen TOML line 63; provenance line 42),
so EarTTS intentionally aborts and re-prefills at agent BOS. Verified in the
artifact: aligned 12/12 through injection and settlement (both engines
session-position 12 before the EOU request), then after the BOS-bearing first
PCM, Nano is at session position 13 while EarTTS restarts a new acoustic
epoch (generated 2, baseline 1, session 1). The correct invariant is
therefore: **per-position Nano/EarTTS equality holds within an acoustic
epoch — through all hidden positions, settlement, and up to the BOS edge; at
BOS, EarTTS must abort and re-prefill exactly once, and thereafter
`nano_session_pos − bos_session_pos` must equal EarTTS's epoch-relative
position.** The corrected local gate must assert that full shape (exactly one
reset, correct epoch baseline, epoch-relative equality afterward), not merely
tolerate inequality.

**Satisfied by this artifact:**
- **G1** — per-position before/after clock pairs on the live AsyncLLM engines,
  12/12 advancing exactly once each and equal through the epoch; settlement
  and EOU verified under the corrected invariant.
- **G2** — first real PCM after injection initialized perception
  (`perception_frame_idx` 0→1) and RNNT (`rnnt_initialized` false→true) on
  the independent audio clock (0→1) despite `context.frame_idx`=12 — the
  discontinuity mapping works on the real stack.
- **G6 (seam half)** — all 12 positions committed effective PAD
  (`effective_token_id` 12) with `subword_mask=True`; 1,764 decoded samples
  discarded per position; zero display/delivered audio during injection.
- **G9** — settlement → single EOU → BOS → complete response with balanced
  brackets and real audio (61,740 samples, 2.8 s, −35.4 dBFS RMS / 0.111
  peak), text "The secret word is cobalt." — i.e., the model *recalled a
  secret it only ever saw as hidden direct-text positions*, which is the
  first end-to-end semantic proof of the direct path.
- **G10** — `_last_user_audio_emb` SHA-256 equals the last injected token's
  embedding post-injection, and the replaced-by-first-PCM check passed.

**Partially satisfied / corrected scope:**
- **G8** — `first_feedback_is_pad=True` is *correct here* because injection
  began at frame 0, where the position-zero path legitimately feeds PAD (BOS
  shares `pad_id` in this checkpoint). The exact-prior-live-token feedback
  case is therefore **not yet exercised**: a mid-session injection (after a
  real audio turn and response) must show non-PAD prior agent feedback fused
  at the first injected position.
- **G4** — the codec demonstrably ran per hidden position and post-injection
  delivery is clean, but the sequential idle-PAD equivalence comparison
  (G3/G4 core) is not in this artifact.

**Unverified claim:** the stated independent Parakeet transcript match for
`text.wav` is not in the report (no ASR block) and no transcript artifact was
found; provide it with the follow-up runs — the review does not count it yet.

**Still open after this artifact:** G3, G4 (equivalence half), G5, G7, G8
(mid-session variant), G11, G12, plus F1/F2 and the corrected-gate assertion
shape above.

## Addendum 2 — safety, recurrence, and token-form GPU artifacts (2026-08-08, late)

Reviewed from raw reports: `direct-position-safety-r3`,
`direct-position-recurrence-r1`, `direct-position-token-forms-r4` (+
`external-asr.json`), plus the current code deltas. CPU suites re-run: 294
passed, 2 skipped.

**G5 — satisfied.** Five live reject cases (boolean token, out-of-vocabulary,
pending user-EOU latch, open agent response, pad-pair enabled) each carry
before/after owner snapshots with `owners_unchanged: true`, including tensor
SHA-256 fingerprints. Better still, the G5 work caught a real reject-purity
hole: preflight previously read agent-idle via `_get_agent_idle()`, whose
configured getter is `get_or_create_state()` — a rejected preflight would
have *allocated* streaming state. The fix reads `_state_pool.get()`
non-creatively with an explicit comment citing the zero-delta contract. This
is exactly the class of defect the gate existed to force out.

**G7 — satisfied.** The `fault_after_nano` case injects a failure after Nano
advancement: `nano_position_delta=1, eartts_position_delta=0`,
`owners_changed: true` honestly recorded, session aborted, both vLLM requests
removed. The forward-only fatality contract is demonstrated, not asserted.

**G3/G4 — satisfied, stronger than required.** The recurrence probe runs two
*independent sessions* (verified in source: inject-arm then fresh-start
control-arm — no self-comparison): four direct hidden positions vs four
ordinary idle-PAD silent-audio positions produce **bit-identical** recurrent
`code`, codec cache state, and decoded waveforms per position (SHA-256
equality; audio hashes differ across positions while matching across arms,
confirming a non-trivial comparison). The review asked for "equivalence";
the artifact delivers determinism.

**Parakeet durability — closed.** `external-asr.json` pins the ASR model by
SHA-256, records each source WAV's SHA-256, and shows exact transcript
matches (`text_channel_wer: 0.0`, required phrase "cobalt" present, sane
dBFS) for all four token forms. The Addendum-1 unverified-claim item is
resolved with content-addressed provenance.

**Token forms (2B preview).** All four candidate forms (`text`,
`user_bos_text`, `text_user_eos`, `user_bos_text_user_eos`) produced the
identical correct recall on this probe. The 2B semantic corpus remains the
discriminator; no form can be selected from this evidence alone.

**Code deltas verified:**
- *Epoch-offset admission* (engine adapter): admission and post-checks now
  require the Nano−EarTTS **epoch offset** to be unchanged by an injection
  rather than absolute equality — the exact generalization my revised
  invariant called for under `VOICECHAT_EARTTS_RESET_ON_BOS`.
- *Stale-EOS parity*: the direct path now clears the one-frame stale agent-EOS
  notification exactly as the ordinary PCM path does.
- *Reset counter*: the probe records EarTTS reset counts before EOU / after
  first PCM / after response with an `eartts_bos_reset_exactly_once` check —
  the corrected-gate assertion shape required by Addendum 1.
- *F1 resolved*: the seam now asserts request-iterator identity (and request
  liveness) post-injection (`request iterator changed during direct
  position`).
- *Pad-draft safety case*: `pad_pair_pending_draft` (synthetic buffered-draft
  rejection) exists in the probe code but is **not present in the r3 safety
  artifact** — its evidence is pending the next safety run.

**Pending evidence (unchanged status until artifacts land):** r7 (G8
mid-session non-PAD feedback + exactly-one BOS reset under the corrected
gate), the pad-draft rejection artifact (G11's second half; the env-enabled
half is green in r3), G12 (hardened seven-session matrix rerun), and F2
(plan "codec silence" wording reconciliation).

## Addendum 3 — r7 clock artifact and safety-r4 (2026-08-08, final)

Reviewed from raw reports: `direct-position-clock-r7` and
`direct-position-safety-r4`.

**G8 — satisfied.** The mid-session injection (after a completed first
response) fused the *exact* prior live context at its first position:
`expected_previous == observed_previous` with agent token 2 — the agent EOS
of the just-completed response, i.e., genuinely non-PAD live feedback, not a
synthesized stand-in (`agent_previous_is_non_pad`,
`exact_previous_feedback` both true). The first-epoch `first_feedback_is_pad`
behavior is retained alongside, so both feedback regimes are now pinned.

**Epoch-offset preservation and exactly-one reset — satisfied under the
corrected gate.** The Nano−EarTTS epoch offset (12, from the first epoch's
hidden positions) is unchanged across the mid-session injection
(`epoch_offset_before == epoch_offset_after == 12`), and the EarTTS BOS
reset counter reads 0 before EOU, 1 after the BOS-bearing first PCM, and
still 1 after the full response (`eartts_bos_reset_exactly_once`,
`eou_clock_transition_valid`). This is the full assertion shape Addendum 1
required — one reset, correct epoch baseline, epoch-relative equality after —
now demonstrated live rather than tolerated.

**G11 — fully satisfied.** safety-r4 runs seven cases, all green with owner
snapshots: the five Addendum-2 rejects, the fault-after-Nano fatality, and
the previously missing `pad_pair_pending_draft` case — a synthetic buffered
Nano draft rejected (`owners_unchanged: true`) alongside the env-enabled
rejection. Both halves of G11 are now evidenced.

**F2 — resolved.** No "codec silence" claim remains in
`direct-text-input-plan.md`, `direct-text-position-transaction.md`, or
`protocol-v3.md`; plan 2C now states the implemented behavior exactly ("run
the ordinary idle-PAD EarTTS/codec recurrence, and suppress only the decoded
waveform samples for hidden injection positions").

**FP32 scope correction (revising this review's own G3 wording).** G3's
"(W8 and FP32)" over-reached. The recurrence property the artifacts prove —
bit-identical direct-vs-idle-PAD state advance across independent sessions —
is structural: both arms necessarily share whatever precision the engine
runs, so repeating it on FP32 would test the same identity, not a new risk.
Precision *behavioral* differences are exactly what the plan's 2B semantic
gate already covers (FP32/W8 thresholds, W8 no more than one non-critical
case worse). G3 is therefore satisfied by the qualified-stack artifact, and
FP32 involvement is correctly scoped to 2B — where it remains mandatory.

**New minor finding (N-prov).** The probe reports carry no runtime
provenance — unlike the matrix report (which pins its manifest), none of
`g1-g10-r3`, `recurrence-r1`, `token-forms-r4`, `clock-r7`, or `safety-r4`
records the checkpoint manifest, image, or precision it ran against. Add the
same manifest/provenance block the matrix harness writes before these
reports are cited by the release ladder. Not a blocker for closing the
gates they evidence (the single-stack, single-day provenance is clear from
context), but required hygiene before Step 2B citations.

**Remaining Step 2A blocker: G12 only.** The hardened seven-session memory
matrix (unique per-case secrets + unseeded negative control, Parakeet
post-pass) has not yet been re-run on GPU. It is the designated Step 2A
prerequisite artifact standing between this review and closure; everything
else in G1–G11, F1–F2, and the corrected-gate obligations is now green and
verified from raw artifacts.

## Addendum 4 — G12 attempt review: the empty-recall failures are a production blocker, not gate noise

Reviewed from raw artifacts under `memory-matrix-g12-r1/`
(`memory-matrix-r2`, `memory-matrix-r3`, `text-to-text-diagnostic-r1/r2`)
plus the harness diff. One narrative correction first: **memory-matrix-r3 —
the full run that already included the two-frame inter-turn idle — failed
BOTH `voice-to-text` AND `text-to-text`** with `recall_empty_response`, not
only voice-to-text. The idle-frame change did not hold even for the cell it
was derived from.

### Root cause, from the wire clocks

Comparing the recall-turn BOS windows of the failing r3 `text-to-text` and
the passing diagnostic-r2 traces (both with unique secret "amber"):

- In both runs the Nano clock advances in the two-frame PAD-pair pattern
  (…187→189→189→191→191…) — drafting is active throughout.
- **PASS (diag-r2)**: at the BOS edge the clock is sequential —
  nano 139→140→141(BOS)→142→143. The draft boundary happened to fall before
  the BOS frame.
- **FAIL (r3)**: nano reads 195→195→**197→197(BOS)**→198→198. The pair
  committed at f197 *pre-drafted the BOS frame (f198) as PAD*. The wrapper
  then forced BOS into `gen_text[f198]`, and the protocol/EarTTS side dutifully
  observed it (EarTTS reset 154→2) — but Nano's continuation had already
  consumed PAD at that position. **The forced BOS never entered Nano's
  actual token stream.** Nano, never having been handed the floor, emits
  PAD for 30 frames until the silence watchdog closes an empty response.

This explains every observed symptom: failures are parity-dependent (which
is why Pocket's stochastic audio lengths make the failing cell move), the
two-frame idle "fix" worked once by *shifting parity* (diagnostic-r2) and
failed at full-matrix scale (r3), and the signature is always BOS + PAD×30 +
watchdog. It is the same hazard class this branch already knows: a drafted
row crossing a control edge — the exact reason the post-FC recovery fences
exist and the exact reason the Step 2A direct seam *rejects* pending drafts.
The external-EOU forced BOS simply never received the same fence.

### Verdicts on the three questions

1. **The two-frame settlement is not correct as a remedy.** As transport
   realism (continuous WebRTC always supplies inter-turn frames; the
   discrete harness was unrealistic) it is a legitimate harness change and
   may stay. As a fix it is parity roulette — r3 proves it — and adopting it
   as the remedy would be gaming the gate with extra steps. Rejected.
2. **This is a production blocker requiring a server/model fix.** The
   explicit-EOU BOS (and by the same mechanism, plausibly
   `request_agent_eos` during cancellation drains) must be fenced against
   pad-pair drafts: before consuming the latch, drain or correct any pending
   two-frame draft (the scheduler's mispredict-correction machinery exists
   precisely for a drafted PAD that turns out wrong) and force a sequential
   window across the BOS edge, mirroring the post-FC fence contract
   ("a drafted row cannot cross into or out of" a control edge — the plan's
   own 2A language). The narrow window is real in production: it does not
   require the matrix harness to trigger, only unlucky parity between a
   commit and the draft cycle. Step 1's green browser gate passed inside
   this same roulette.
3. **G12 remains strict one-attempt, seven sessions**, re-run after the
   fence fix on an unmodified gate. Retry policy: the ladder's existing
   one-recorded-retry precedent may cover *semantic* variance (wrong words),
   but `recall_empty_response` — the BOS+PAD signature — must be a
   **non-retryable hard fail**, because it is this defect's fingerprint, not
   model stochasticity.

### Also verified

- The bounded `server_busy` acquisition retry is acceptable: it precedes
  session start, touches no model state, and addresses a real lease race.
- The hardened matrix is doing exactly what it was built for: unique
  per-case secrets (sapphire/topaz/cobalt/amber/garnet/indigo + unseeded
  control) made a moving, parity-dependent failure *visible and
  classifiable* instead of a mysterious flaky cell. This failure being
  caught here, before any policy change, is the Step 2A control investment
  paying off.

**G12 status: open, blocked on the BOS/pad-pair fence fix. Do not change
the compiled typed-input default, and do not cite any matrix run as green,
until a strict one-attempt seven-session run passes against the fixed
server with `recall_empty_response` counted as hard failure.**

## Addendum 5 — G12 fence patch review (external-EOU pad-pair barrier)

Reviewed: the working-tree diff to
`src/nemotron_voicechat_runtime/runtime_optimizations.py` and
`tests/runtime/test_public_runtime_optimizations.py`, read against the full
pad-pair engine path and the retained-patch external-EOU consumption. Suite:
18 passed, 3 skipped.

### Mechanics — sound, verified by code reading

- **Barrier placement and ordering.** `pad_pair_precontrol_reason` now checks
  `_external_user_eou_requested` first; `_prepare_pad_pair_call` runs before
  NVIDIA's frame loop, i.e., before the same call's turn-taking consumes the
  latch — so the fence observes the latch on exactly the frame that will
  force BOS. Precedence over `external_eos`/`redirect`/`rnnt` affects only
  the counter label; all paths set the same request-scoped
  `sequential_mode`.
- **The pending row is genuinely drained before BOS.** With
  `sequential_mode` set, the engine call takes the bypass path: it submits
  the *pending* row to Nano and discards the *current* row. Latch timing
  makes this safe by construction: the only frames processed with the
  user-EOU latch set are the synthetic zero-audio EOU frames (serialized
  under the model lock), so the discarded current row is always replaceable
  silence — while the pending row, which can be the **last real microphone
  audio row** when settlement is skipped at `blank_count ≥ 10`, is
  submitted, not dropped. Real speech cannot be lost at the edge; the
  one-position attribution lag that poisoned the failing runs cannot open,
  because the pending row is cleared before BOS is written and the response
  keeps drafting disabled naturally (`previous_effective_pad` goes false at
  BOS).
- **Settlement interplay.** Settlement frames run before the latch is set,
  so drafting during settlement is permitted and harmless: any pending row
  it creates is exactly what the fence drains on the EOU frame.
- **Edge sweep.** External agent EOS (cancellation drains), redirect
  injection, and RNNT-threshold proximity were already fenced; BOU/EOS edges
  inside responses cannot meet a pending row (drafting is inactive while
  effective tokens are non-PAD); legacy mode cannot latch user-EOU at all
  (`request_user_eou` raises outside external mode), so the capability
  atomicity invariant holds. I found no unfenced control edge.
- **Production reach.** `VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER="1"` is
  present in both the frozen TOML and `PRODUCTION_ENVIRONMENT`, so the fence
  is live in the qualified configuration, not just in probes. The EarTTS
  reset counters added to `install_eartts_reset_on_bos` are benign
  instrumentation supporting the exactly-once check.

### Test strength — one concrete gap

The new tests prove the two links separately: the barrier arms
`sequential_mode`/counters with the latch set and a pending row present, and
(pre-existing) the bypass path drains a pending row. **No test joins them
into the composite that actually failed live**: pending row buffered on
frame N (previous effective PAD/PAD), latch set, frame N+1 → assert the
engine received exactly the pending tensor (identity), the synthetic current
row was never submitted, state cleared, and the subsequent sequential call
carries BOS feedback. The pending-real-audio-preserved property is likewise
verified only by my code reading. Required: **one composite CPU regression
test at the wrapper/engine seam reproducing the failing scenario**, before
this fix is cited as review-complete. Cheap to write against the existing
mocks; it is the regression test for a production-blocker class and must not
be skipped on the strength of a code-reading review.

### Status

Fence mechanics: **approved for GPU validation** — no blocker found in the
implementation. Two items stand between this and G12 closure:

1. the composite regression test above (pre-GPU, required);
2. the strict G12 re-run — one attempt, seven sessions,
   `recall_empty_response` as non-retryable hard failure — against the fixed
   server.

## Addendum 6 — composite regression test review: Addendum-5 blocker resolved

Reviewed: `test_external_user_eou_precontrol_drains_pending_row_sequentially`
and the accompanying dependency-injection change to
`_install_public_pad_pair_engine`. Suites re-run: focused file 19 passed /
2 skipped; full `tests/runtime` + `tests/pipecat` 297 passed / 2 skipped.

The test closes the gap as specified, and it does so against the *real*
code under test, not mocks of it: `_install_public_pad_pair_engine` is
invoked with an injected fake streaming-engine module, so the installed
`generate_next_token` is the genuine scheduler closure wrapping a recording
`original`; `_prepare_pad_pair_call`, `pad_pair_precontrol_reason`, and
`_finalize_pad_pair_call` all execute for real. The composite sequence then
verifies every required property with value-distinct tensors (pending=1.0,
current=2.0, post-BOS=3.0):

1. pending row planted, draft-eligible state, latch set → precontrol fires;
2. the first engine call submits **exactly the pending tensor**
   (`torch.equal(calls[0], pending)`) and the synthetic current row is
   **never submitted** (`not torch.equal(calls[0], current)`; `calls`
   length 2 total);
3. latch consumption and the BOS overwrite are simulated at the correct
   point in NVIDIA's hook ordering (after Nano returns, before
   finalization), and the real finalize observes BOS →
   `previous_effective_pad is False`;
4. the second prepare resets `sequential_mode` (latch gone, no pending) and
   the second engine call submits **the distinct post-BOS fused row
   immediately** (`torch.equal(calls[1], after_bos)`) — no one-position
   lag survives the edge;
5. terminal state: `pending is None`, `sequential_pending_drains == 1`,
   `precontrol_pending_drains == 1`.

That is precisely the failing composite from Addendum 4, reproduced and
pinned at the wrapper/engine seam. The dependency injection is clean:
optional parameters defaulting to the original imports, and the sole
production call site (`runtime_optimizations.py:1112`) remains
argument-free — production behavior is unchanged by construction.

**The Addendum-5 test blocker is resolved.** Remaining before G12 closure:
the strict re-run only — one attempt, seven sessions,
`recall_empty_response` as a non-retryable hard failure, on the fixed
server, with the Parakeet post-pass.

VERDICT (fence + regression scope): APPROVE

## Addendum 7 — G12 closure review (final)

Reviewed from raw artifacts only: `memory-matrix-g12-r1/memory-matrix-r4/`
(`report.json`, `runtime-health-after.json`, all seven per-case
`events.jsonl`/reports) and `pocket-determinism-r2.json`. Run timestamps
2026-08-08 ~21:51 UTC, post-fence server.

### Verified, item by item

- **Strict one-attempt, seven sessions, correct order.** Seven distinct
  session IDs, strictly sequential per-case timestamps (~10–20 s apart, no
  repeats), the six seeded modality cells first and the unseeded control
  last (`negative_control_ran_after_seeded` recomputed true; no attempt or
  retry fields exist in the schema). r2/r3 remain retained *pre-fix*
  diagnostic evidence; r4 is the first and only attempt against the fixed
  server, which is exactly what the policy required.
- **Six unique seeded cells + non-vacuous control.** sapphire/topaz/cobalt/
  amber/garnet/indigo across {voice,text,system}×{voice,text};
  `recalled_secrets` equals exactly the own secret in all six; the unseeded
  control recalled nothing real — the model *hallucinated "cinnamon"*, a
  complete, speech-classified response leaking none of the six secrets,
  which is the control doing precisely its job.
- **Zero empty responses.** All `failure_class: null`; ASR layer: 11/11
  responses classified speech, 0 near-silent. The non-retryable
  `recall_empty_response` rule stood armed and was never triggered.
- **Fence operation at every commit edge.** The wire clocks show the
  Addendum-4 failure signature is gone in all seven recall-BOS windows: the
  two-frame pair pattern always terminates before the edge, and every BOS
  frame advances the Nano clock by exactly one (sequential submission) —
  including in `voice-to-text` and `text-to-text`, the two previously
  failing cells. Direct precontrol counters are trace-gated and not on the
  wire; the clock signature plus the Addendum-6 composite regression are
  jointly decisive evidence that the pending row is drained before BOS.
- **Deterministic Pocket stimulus.** `pocket-determinism-r2.json`: identical
  SHA-256 across repeated syntheses of the same text, different hash for
  different text; health confirms the seeded scheme
  (`sha256-text-plus-base-v1`) and content-addressed Pocket assets.
- **Independent Parakeet with expected/excluded-secret semantics.** Every
  recall entry requires its own secret (`expected_response_any`) and
  excludes the other five (`expected_response_none`) on the *independent*
  transcript, with both `semantic_match` and `negative_semantic_match`
  asserted; the control excludes all six. Model text and rendered voice are
  therefore verified separately, WER within bounds per modality.
- **Runtime identity.** `runtime-health-after.json` reports a ready,
  client-free server post-run. The stated image
  `sha256:ee4e622c…5dec2a71` exists locally tagged
  `pipecat-ai/nemotron-voicechat-dgx-spark:direct-probe` — matching the
  declared ID. The report itself still does not embed image/source
  provenance: the Addendum-3 **N-prov** obligation stands and carries
  forward — bake image ID and source fingerprint into matrix and probe
  reports before any Step 2B citation of these artifacts.

### Closure

G12 is closed on its own strict terms. With it, every Step 2A obligation in
this review is discharged: G1–G12, F1–F2, the corrected epoch-alignment
gate, the composite fence regression, and the hardened-matrix prerequisite
that Step 1's Q1 placed here. The single carried item is N-prov — report
provenance hygiene — which was scoped to Step 2B citation, not 2A closure.

The gate also earned its existence twice over during closure: it forced out
a live production blocker (the BOS/pad-pair parity race) that three earlier
green ladders had sailed past, and its unique-secret + negative-control
design converted that heisenbug into a classified, mechanically diagnosed,
regression-tested fix.

**VERDICT: APPROVE — Step 2A closed.** Step 2B may begin under its
already-defined gates (token-form corpus with FP32/W8 semantic thresholds;
stop rule intact), with N-prov landing before 2B cites these artifacts.

## Addendum 8 — N-prov implementation review (final carried item)

Reviewed: current worktree diffs and full files in
`src/nemotron_voicechat_runtime/{server.py,cli.py}`,
`tools/qualification/{memory_matrix_strict_v3.py,transcribe_sustained.py}`,
and their tests. Independently verified: focused suite **128 passed +
3 subtests**, full `tests/runtime`+`tests/pipecat` **301 passed /
2 skipped**, Ruff clean on all four files.

### Verified

1. **Fail-closed immutable image identity, end to end.** `command_up` raises
   before starting the container unless the bootstrap state carries a
   `sha256:`-prefixed image ID; `_bootstrap_ready` already equality-checks
   that ID against live `docker image inspect`, so the value passed into the
   container (`VOICECHAT_RUNTIME_IMAGE_ID`, added by
   `model_container_command` and unit-tested in the emitted command) is
   verified-current, content-addressed, and retag-immune. Manual/dev
   container runs that skip the env yield `runtime_image_id: null`, which
   the matrix validator rejects — fail-closed at the consumer too.
2. **Identical, non-recursive provenance in health and session.created.**
   Both expose the same `create_app`-scoped `checkpoint_provenance` object
   with `runtime_provenance` embedded once. The shallow-copy discipline
   (`dict(...)` before mutation) keeps the pipeline's own checkpoint dict
   clean, so probe reports (`_direct_probe_runtime_provenance`) embed
   checkpoint + reusable image/source fingerprint side by side with no
   recursive nesting — confirmed at all three probe call sites.
3. **Matrix enforcement is strict and complete.**
   `session_runtime_provenance` reads the correct
   `session.checkpoint.runtime_provenance` envelope (unit-tested against the
   wrong nesting); provenance is embedded in every case and the top report;
   the gate requires strict format (`sha256:` + 64-hex image ID, all six
   `REQUIRED_RUNTIME_SOURCES` present as 64-hex) **and** deep equality
   across all selected sessions (`runtime_provenance_consistent`), and that
   flag feeds `runtime_gate_passed`.
4. **Parakeet cannot accidentally re-green.** The transcribe-side recompute
   requires `runtime_provenance_consistent is True`, which only the matrix's
   strict validator can produce; the ASR pass cannot introduce validity.
5. **Probe reports** carry checkpoint plus the reusable fingerprint
   (Addendum-3 N-prov requirement), non-recursively.
6. **No vacuous tests**: every new test flips both directions (gate
   accepts/rejects, validator accepts/rejects mutable tag and missing hash,
   envelope reader returns None on wrong nesting, env var asserted in the
   actual command list).

### Findings (severity order)

- **Nit 1.** `transcribe_sustained.memory_matrix_runtime_gate_passed`
  checks provenance presence by truthiness rather than reusing the strict
  validator. Accidental re-greening is impossible (the consistent flag is
  strict-sourced), but the recompute's stated independence would be
  stronger — and hand-edited reports would be caught — if it re-validated
  format. Recommended, not required.
- **Nit 2.** The CLI plumbing test uses `sha256:immutable-image` (not
  64-hex); it tests transport, not format, so this is fine — a realistic ID
  would let one assertion cover both.

### Verdict

The identity chain is closed at both ends (fail-closed launch, fail-closed
gate), the schema is consistent across health, session, matrix, and probes,
and no bypass, mutable identity, hidden nondeterminism, or compatibility
regression was found.

**VERDICT: APPROVE — N-prov closed.** Step 2A has no remaining obligations;
Step 2B may cite the G12 and probe artifacts.

## Addendum 9 — N-prov tightening review (post-Addendum-8)

Reviewed: the shared-validator refactor in the worktree. Independently
verified: full `tests/runtime`+`tests/pipecat` **301 passed / 2 skipped**,
Ruff clean on all touched files (consistent with the reported 131+3 focused
result).

### Verified

- **Single-sourced strict validation.** `valid_runtime_image_id` /
  `valid_runtime_provenance` live in `provenance.py`;
  `memory_matrix_strict_v3` and `transcribe_sustained` both import the exact
  same functions — the Addendum-8 Nit 1 truthiness gap is gone.
  `command_up` now enforces the complete `sha256:<64 lowercase hex>` shape
  via the shared predicate.
- **Producer/validator agreement.** The server's
  `_runtime_source_provenance` hash map now includes `provenance.py`, so the
  seven required sources emitted match
  `RUNTIME_PROVENANCE_REQUIRED_SOURCES` exactly. Self-fingerprinting is
  sound: the file's content hash is computed at runtime and never embedded
  in the file, so no fixed-point problem exists.
- **No circular imports; import-light chain.** `provenance.py` needs only
  `re`; the package `__init__` pulls only `protocol` (stdlib); host-side
  tools and tests resolve it cleanly.
- **Tests are valid and realistic.** The validator tests flip both
  directions (mutable tag, missing hash); the CLI plumbing and state-file
  tests now use a realistic `sha256:` + 64-hex digest, so the same strings
  the gate accepts are the strings the transport tests assert.

### Findings

- **S1 (should-fix, before the next ASR pass runs).** The ASR
  postprocessing container is the *runtime image* with the repo bind-mounted
  read-only, and `transcribe_sustained.py` (loaded from the mount) imports
  `nemotron_voicechat_runtime.provenance` from the image's *baked* package.
  I demonstrated the failure against the current `direct-probe` image:
  `ImportError: cannot import name 'valid_runtime_provenance'`. Any image
  predating this change breaks the ASR pass. It fails **closed** (a crash,
  never a re-green), so gate integrity is intact — but the next Parakeet
  invocation will fault until either the image is rebuilt or, better, the
  script/launcher pins the mounted tree (`PYTHONPATH=/workspace/project/src`
  in `asr_command`, or a `sys.path` insert at the top of the tool) so the
  gate policy always travels with the reviewed worktree rather than the
  image's build vintage.
- **Compatibility note (must be stated, no action).** The retained G12
  artifact `memory-matrix-r4` carries **no** `runtime_provenance` block —
  it predates N-prov entirely. Under the tightened gate, any re-grade of
  that report now fails, by design. r4 remains valid *as reviewed* (its
  image identity was verified externally in Addendum 7); it must not be
  re-graded under the new schema nor cited as a provenance-bearing
  artifact. The first provenance-bearing matrix run will be the citation of
  record going forward.

### Verdict

The tightening is correct, single-sourced, fail-closed in every path
examined, and free of circular-import or self-fingerprint hazards. S1 is an
operational follow-up with fail-closed behavior, not a validity defect.

**VERDICT: APPROVE** — with S1 required before the next ASR-pass execution,
and the r4 grandfathering note recorded above.

## Addendum 10 — Step 2B static corpus/harness review (pre-GPU)

Reviewed: `direct_text_semantic_corpus.json`, `semantic_corpus.py`,
`direct_semantic_probe.py`, `evaluate_direct_text_semantics.py`,
`typed_semantic_baseline_v3.py`, `run_live_suite.py`, and the four test
files. Independently verified: focused 2B tests **25 passed**, full
`tests/runtime`+`tests/pipecat` **316 passed / 2 skipped**, Ruff clean.
(The harness was tightened mid-review — a `required_literal` predicate for
c003, a list-preserving "every tool exactly" check, and a `PAD_PAIR=0`
guard in the probe; all reviewed in their current form.)

### Verified

- **S1 (Addendum-9) is genuinely closed.** `asr_command` sets
  `PYTHONPATH=/workspace/project/src` *before* the read-only repo mount, so
  `transcribe_sustained` (and now the 2B tools) import the reviewed worktree
  package, not the image's baked vintage — the gate policy travels with the
  source.
- **Comparator math is correct.** Pocket gate: intent rate ≥ 0.95, all tools
  pass, all safety-critical pass, structural. Direct lanes: parity =
  `pocket.passed ∧ direct.passed` per case / 20 ≥ 0.95 (genuine parity, not
  standalone pass). W8 regression: fp32-pass-and-w8-fail classified critical
  iff `safety_critical ∨ tool`, gate = zero critical ∧ ≤1 noncritical.
  Overall `passed` = corpus identity ∧ provenance ∧ source consistency ∧
  Pocket gate ∧ all four lanes structural ∧ a qualified token form. The
  100%-tools and zero-safety-misroute requirements are exact (single call,
  dash-normalized name, argument-key-set equality, finite-float scalar
  equivalence).
- **Anti-regrade is real.** `_regrade` recomputes every predicate from the
  checked-in corpus against each report's raw `assistant_text`/`tool_calls`,
  so a hand-forged `"passed": true` with failing text is caught;
  `corpus_sha256` pinning prevents predicate tampering; `_case_map` rejects
  any id set/order/count drift. Provenance validity and cross-report source
  consistency reuse the Addendum-9 strict validator.
- **Transaction assertions cover every hidden position.**
  `_transaction_checks` asserts, for *all* positions: BF16 `[1,1,4480]`
  source, forced-PAD effective output + PAD token + subword mask, discarded
  audio, per-position clock-advance-once with epoch-offset preservation,
  perception/RNNT/function held, and exact PAD feedback chain — plus
  per-case perception-hold, audio-clock-hold, first-PCM re-init, and
  exactly-one EarTTS BOS reset. Raw per-position records are retained.
- **Production protocol is not mutated.** The direct probe is engine-level
  only (`inject_user_source_position`, `request_user_eou`, `process`,
  `configure_external_tools`) and never touches `RealtimeProtocolSession` or
  the websocket server; the Pocket baseline drives the *existing* unchanged
  `input_text.request` path with a correct tool round-trip
  (`response_id` correlation checked, `function_call_output` returned). The
  `PAD_PAIR=0` guard and lazy `.server` import keep it runnable inside the
  built image. No test is vacuous — every evaluator test flips a real gate.

### Findings (severity order)

- **B1 (blocker for the ladder, not the math).** The 2B stage is **not
  wired into `run_live_suite.py`** — no reference to the Pocket baseline,
  the FP32/W8 probes, or `evaluate_direct_text_semantics`. The pieces are
  individually correct and unit-tested but nothing orchestrates
  Pocket→FP32→W8→compare as a gated, provenance-stamped ladder stage with
  model-idle barriers. Given the S1 lesson (orchestration wiring is exactly
  where a correct component silently fails to run), GPU execution must not
  proceed on manual invocation alone: wire the stage, with the same
  provenance/corpus-identity stamping the comparator requires, and a
  live-suite test asserting its command shape — before the run.
- **S1 (should-fix). Keyword-only forbidden nets on safety cases.** The
  `forbidden_any` lists are short exact phrases (e.g. c008 forbids only
  "here is the ransomware"/"encrypt every file"/"payment wallet code"); a
  harmful completion phrased differently passes the forbidden check and
  rides on the `required_any` refusal term alone. Adequate as a diagnostic
  signal for a frozen checkpoint, but do not let this corpus's
  zero-safety-misroute result be read as a safety *audit*. Broaden the
  forbidden nets or state the limitation in the 2B report.
- **S2 (should-fix). "Blinded" corpus is checked-in plaintext.** The plan's
  2A/2B language called the corpus "blinded"; this one is visible in-repo.
  Harmless for the frozen published weights (no training exposure possible),
  but the moment 2C adds direct-text rows to calibration and requalifies,
  a visible corpus is no longer a clean semantic gate. Reconcile the wording
  or hold out a private split before any recalibrated qualification —
  mirroring the Q1 lineage.
- **N1 (nit). Structural trust boundary.** The comparator consumes each
  report's `structural_passed` boolean rather than re-deriving it (it
  cannot — the evidence is GPU-side). Fine given the probe records raw
  per-position data, but a spot re-derivation of `structural_passed` from
  the retained `direct_positions` in the comparator would harden it against
  a malformed probe report; optional.

### Verdict

The corpus, predicates, transaction assertions, comparator arithmetic,
anti-regrade, and provenance reuse are all correct and non-vacuously tested,
and the diagnostic is confirmed engine-only and image-runnable without
touching production protocol. The one thing standing between this and a
trustworthy GPU run is orchestration: the stage is not in the live ladder.

**VERDICT: REJECT for GPU execution as-wired — APPROVE the harness/corpus on
the merits.** Clear B1 (wire the gated ladder stage with provenance stamping
and a live-suite command-shape test); land S1/S2 or record them explicitly
in the 2B report. With B1 closed, GPU execution is warranted.

## Addendum 11 — Step 2B orchestrator review (B1 remediation)

Reviewed: `--direct-text-semantics-only` and its helpers in
`run_live_suite.py`, the probe/comparator changes, and the tests.
Independently verified: focused **29 passed**, full
`tests/runtime`+`tests/pipecat` **321 passed / 2 skipped**, Ruff clean.

### Addendum-10 findings — disposition

- **B1 (blocker) — resolved.** The stage is now a first-class, standalone
  mode: `run()` dispatches `run_direct_semantics_only` before touching any
  browser/conversion machinery, so it runs without the rest of the suite.
  Ordering is correct and unit-pinned
  (`["ready","idle","pocket","idle","stack-stop","fp32-w8-compare"]`): the
  fresh-session Pocket baseline runs on the live stack, a model-idle barrier
  fires, the stack is torn down in a `finally`, and only then does the
  offline ladder run. The FP32→W8 barrier is a genuine
  previous-container-exited edge — each is a foreground `docker run --rm`
  under `run_checked(check=True, timeout=…)`, so W8 cannot start until FP32's
  container has exited. `ladder.json` is written atomically at every
  transition with image ID, corpus hash, per-stage command, report
  path+SHA-256, status, and barrier label.
- **N1 (structural trust) — resolved.** The comparator now re-derives every
  hidden transaction from each case's retained `direct_positions` and the
  report's `pad_token_id` (verified emitted by the probe and consumed by the
  evaluator), and folds `hidden_transaction_rederived_passed` into each
  lane's `passed`. Report-trust on structural integrity is gone.
- **S1/S2 (safety net; public corpus) — resolved as recorded.** The final
  comparison report carries an explicit `limitations` block stating the
  safety predicates are lexical intent checks for a frozen checkpoint (not a
  safety audit) and that the corpus is public/model-blind and must not be
  reused as a holdout after training or calibration exposure — exactly the
  Q1-lineage caveat, now travelling with the artifact.

### Verified in the new orchestrator

- **Fail-closed throughout.** Missing FP32 root files raise before any
  container; `runtime_image` must equal the configured image (Pocket
  baseline integrity); `runtime_image_id` validates the full
  `sha256:`+71-char shape; `output.mkdir(exist_ok=False)` refuses to
  overwrite a prior run; the probe writes `passed:false`+traceback and the
  server re-raises on any exception, and `run_checked` propagates non-zero.
  No path re-greens on failure.
- **Artifact mounts are precision-correct.** W8 mounts the release tree
  (`/models/derived` → release manifest/nano/eartts); FP32 mounts the
  derived root (`/derived` → fp32 manifest/nano/eartts). Both force
  `VOICECHAT_NANO_PAD_PAIR=0` and stamp precision + image ID into the
  container env; the probe emits `precision` and `runtime_provenance`
  accordingly.
- **FP32 diagnostic manifest is scoped.** `accepted_vllm_manifest_kinds`
  admits the extraction manifest only when the semantic-probe dir is set
  **and** precision is fp32 — never in production serving — and this is
  unit-tested
  (`test_fp32_manifest_is_accepted_only_in_explicit_semantic_diagnostic`).
- **Provenance split is principled and consistent.** The comparator (host,
  worktree `args.python`) is the evaluator; the probe containers run the
  baked image package and *fingerprint it* — image identity is the artifact
  under test, so baked-wins is correct here, the mirror of the ASR case
  where worktree-wins was correct. All three reports run the same baked
  image, so the comparator's identical-`source_sha256` requirement holds.

### Findings (severity order)

- **B1 (blocker for execution, operational). The pinned image must be
  rebuilt to contain the 2B probe, and nothing pre-flights that.** The
  FP32/W8 containers run `python3 -m nemotron_voicechat_runtime.server` from
  the **baked** package; I confirmed the current `:direct-probe` image lacks
  `direct_semantic_probe` entirely (`ModuleNotFoundError`). The orchestrator
  resolves and pins the image ID but never verifies the image actually
  contains the probe entrypoint — so a stale image fails **closed** (crash,
  never a false green) but hard-fails the whole ladder mid-run after standing
  up the stack. Before GPU execution: rebuild the runtime image from the
  reviewed worktree, and add a cheap pre-flight (a `docker run --rm <image>
  python3 -c "import nemotron_voicechat_runtime.direct_semantic_probe"`
  before the stack starts) so a missing entrypoint fails in one second
  instead of after the Pocket baseline. This is the S1 lesson at image-content
  granularity: pinning identity is not the same as verifying capability.
- **S1 (should-fix). `ladder.json` records the *intended* command, not the
  container's own image digest.** It logs `runtime_image_id` resolved on the
  host before the run; it does not capture, per stage, the image digest the
  container actually reported (the probe's own `runtime_provenance` does, in
  each stage report). Cross-checking the ladder's `report_sha256` targets'
  embedded image ID against the top-level pin would close the last gap
  between "the ID I meant to run" and "the ID that ran." Minor; the
  comparator's provenance consistency already enforces agreement across the
  three reports.

### Verdict

B1 from Addendum 10 is genuinely closed: the stage is wired, standalone,
correctly ordered with real barriers, fail-closed, provenance-stamped, and
its structural and safety-limitation gaps are shut. The one thing between
this and a trustworthy run is operational, not architectural — the reviewed
probe code is not in the pinned image, and there is no capability pre-flight.

**VERDICT: APPROVE the orchestrator; GPU execution gated on (a) rebuilding
the runtime image from the reviewed worktree and (b) adding the one-line
probe-import pre-flight (new B1).** With those, the Step 2B run is warranted;
S1 is a recommended hardening for the same pass.

## Addendum 12 — Step 2B execution-gate closure (Addendum-11 remediation)

Reviewed: the operational-gate implementation and the claimed rebuild/pin.
Independently verified, not taken from the summary.

### Facts confirmed on this host

- **Image digest.** `docker image inspect` of
  `pipecat-ai/nemotron-voicechat-dgx-spark:production-candidate-1` returns
  `sha256:d160fd29…68dabe` — exactly the stated rebuild.
- **Bootstrap re-pin.** `bootstrap-state.json` `runtime_image_id` equals that
  same digest (atomic repin confirmed; the historical `*.pc1-backup` still
  holds the prior `sha256:62940535…` digest, which is correct — backups are
  point-in-time snapshots, not live pins).
- **Baked 2B capability under `--network none`.** `docker run --rm --network
  none` of the rebuilt image imports `direct_semantic_probe`,
  `semantic_corpus`, and `valid_runtime_provenance` successfully
  (`baked-2b-ok`) — the Addendum-11 B1 root cause (probe absent from the
  pinned image) is gone, proven offline.

### Addendum-11 gates — disposition

- **B1 (rebuild + preflight) — closed.** The reviewed 2B code is in the
  pinned image (verified above), and `run_direct_semantics_only` runs
  `direct_semantic_image_preflight_command` — an offline (`--network none`)
  `python3 -c "import …direct_semantic_probe; import …semantic_corpus"` —
  **before** `subprocess.Popen(up_command)`. A capability-missing image now
  fails in ~1 s, before the stack is stood up, instead of mid-ladder. Command
  shape is unit-tested.
- **S1 (stage image cross-check) — implemented.** The ladder reads each stage
  report's self-reported `runtime_provenance.runtime_image_id` and raises on
  any mismatch with the host pin — Pocket (before the ladder proceeds), then
  FP32 and W8 (each immediately after its container exits), recording
  `reported_runtime_image_id` per stage in `ladder.json`. "The ID I meant to
  run" and "the ID that ran" are now reconciled at every stage.

### Findings

- **S1 (should-fix, test coverage). The stage image-ID cross-check is
  untested.** The only test that reaches `run_direct_semantic_gpu_ladder`
  monkeypatches the whole function out, so the three new mismatch-raises
  (the mechanism that closes Addendum-11 S1) have no flip test. The code
  reads correctly and fails closed, but by the standard held throughout this
  review — a guard that catches a wrong-image-ran condition should have a
  test proving it fires — add one unit test feeding a stage report whose
  `runtime_image_id` differs and asserting the ladder raises. Cheap; not a
  GPU-execution blocker.
- **N1 (nit). Preflight and pin are separate `docker` queries.**
  `runtime_image_id(args.runtime_image)` and the preflight both resolve the
  image by tag; a tag re-point between the two calls is a theoretical TOCTOU.
  Irrelevant in this single-operator flow and closed in practice by the
  per-stage digest cross-check, but pinning the preflight to `image_id@digest`
  rather than the tag would make it airtight.

### Verdict

Both Addendum-11 execution gates are genuinely closed and independently
verified on this host: the pinned image contains the reviewed probe, the
offline preflight fails fast on a stale image, and every ladder stage now
reconciles its executed image digest against the host pin. No blocker
remains before the GPU Pocket→FP32→W8 ladder. The residual items are a
should-fix test for the new cross-check and a nit-level TOCTOU hardening —
both landable alongside, neither gating execution.

**VERDICT: APPROVE — GPU execution of the Step 2B ladder is warranted.**
Run it; capture the Pocket/FP32/W8 reports, `ladder.json`, and the
comparison for the closing Step 2B review, and land the cross-check test in
the same pass.

## Addendum 13 — Addendum-12 residuals closed

Independently verified (16 focused passed, Ruff clean):

- **Stage cross-check now tested, non-vacuously.**
  `test_direct_semantic_ladder_rejects_pocket_image_id_mismatch` writes a
  Pocket report with a `b…` digest and asserts the real
  `run_direct_semantic_gpu_ladder` raises `Pocket report image ID mismatch`
  against an `a…` pin; `…rejects_probe_image_id_mismatch` drives the ladder
  past Pocket (matching pin) with a stubbed `run_checked` that writes an
  FP32 report bearing a mismatched digest, and asserts `fp32 report image ID
  mismatch`. Both exercise the actual guard (not a monkeypatched-out ladder)
  and assert the raise — the Addendum-12 S1 gap is closed.
- **Preflight TOCTOU removed.** `direct_semantic_image_preflight_command`
  now takes an `image_ref`, the caller passes the already-resolved
  `image_id` (line 508), and
  `test_direct_semantic_image_preflight_is_offline_and_imports_probe`
  asserts that exact `sha256:` reference appears in the command. The
  preflight and the ladder now bind the same immutable digest, so no
  tag re-point can slip between resolve and use — Addendum-12 N1 closed.

Both Addendum-12 residuals are resolved with real, passing tests, and
nothing new was introduced. **VERDICT: APPROVE — the GPU Step 2B
Pocket→FP32→W8 ladder remains approved for execution.** Capture the three
stage reports, `ladder.json`, and `comparison.json` for the closing Step 2B
review.

## Addendum 14 — Step 2B live-run failure (r1) and the paced-recovery fix

Reviewed raw artifacts under `direct-text-semantics-r1/` and the fix diff.
Full suite **326 passed / 2 skipped**, Ruff clean.

### Failure confirmed from the artifact

The Pocket baseline died at c009: the report's terminal is a fatal
`function_output_apply_timeout` (call `call_7f12ad85…`), and c009's event
stream shows the exact `get_weather` call, `injection_started`/`finished`,
then the error — with `function_calling.background_active` true across **105**
of the case's metrics frames. The old applied-drain
(`for _ in range(64): process_pcm(...)`) is unpaced, so its 64 frames
complete in ~70 ms while the FC background thread — which had just finished a
2.964 s phase-1 and needed scheduling for phase-2 — never got wall-clock time
to clear `background_active` inside the window. `applied` stays false → fatal
close. This is a real latent defect in Step-1-qualified code: G12/browser
tool cycles were light enough to clear within an unpaced 64-frame burst;
c009's heavier `get_weather` cycle is the first to expose it. Correctly
root-caused; a genuine blocker for the ladder (Pocket never reaches c010+).

### Fix assessment

- **Mechanism is right.** `advance_function_output_recovery` paces each
  recovery step on an absolute `FRAME_SECONDS` accumulator (drift-free), so
  the 64-frame bound now spans a real ~5.12 s of wall clock during which the
  background thread is repeatedly yielded to via `await asyncio.sleep`. The
  budget goes from ~70 ms to ~5.12 s — the quantity that was actually wrong.
- **Ordering is correct.** It steps, checks `function_cycle_active`, and
  returns True *before* sleeping; the sleep is guarded by
  `frame_index + 1 < max_frames`, so a cleared cycle exits immediately with
  no trailing sleep and the fast common case pays ~1 frame, not 5 s.
- **Still fail-closed.** The 64-frame bound is retained; a genuinely stuck
  cycle returns False → the same fatal `function_output_apply_timeout`. The
  bound was never the bug, so keeping it is correct.
- **No Step-1 regression surface.** `function_cycle_active` is a single
  extracted predicate reused by both the recovery loop and the
  `input_text.request` `function_cycle_active` rejection — no divergent copy.
  The recovery still sits behind `if not client_turn_detection: continue`, so
  legacy mode is untouched (Addendum-2 I1 atomicity preserved).
- **Tests are non-vacuous.** The pacing test patches `asyncio.sleep`, feeds
  busy→busy→clear, and asserts exactly 3 steps with 2 positive delays and an
  immediate clear-return; the bound test asserts 4 steps then False on a
  permanently-open cycle. Both exercise the real helper.

### Findings

- **S1 (should-fix, must be shown by the rerun, not a code blocker).** The
  fix makes the recovery budget *real* (5.12 s) rather than *adequate by
  proof*. c009's phase-1 was 2.964 s; there is no artifact yet showing
  phase-2 completes within 5.12 s under load. The rerun must demonstrate
  c009 (and every tool case: c013–c020, plus the tool-bearing corpus cells)
  clears recovery well inside the bound, and the closing review should read
  the recovered `voicechat.metrics`/model-step timing to confirm margin. If
  any real cycle ever needs >5.12 s the fix converts a race into a
  deterministic timeout — still fail-closed, but a real ceiling to size
  against observed phase-2 wall time.
- **N1 (nit). Production latency note.** This is production `server.py`; the
  paced loop adds up to ~5.12 s to a *stuck* tool round-trip and real-time
  pacing to a slow one (it returns immediately once the cycle clears, so
  fast calls are unaffected). Acceptable and correct, but it changes
  worst-case production tool-call timing — worth one line in the protocol/
  known-limitations notes.

### Verdict

The failure is correctly diagnosed from the raw evidence, the fix corrects
the actual defect (unpaced budget), preserves the fail-closed bound and
Step-1 gating, and is covered by real tests. Its one unproven property —
that 5.12 s is a sufficient budget for a heavy phase-2 — is exactly what the
rerun exists to establish.

**VERDICT: APPROVE the fix; rebuild the runtime image from this worktree and
rerun the Pocket→FP32→W8 ladder.** Gate the closing Step 2B review on the
rerun showing every tool cycle (c009 included) clearing recovery inside the
bound with margin, alongside the semantic/parity/regression evidence.

## Addendum 15 — r2 re-failure and the FC-boundary fix

Reviewed raw `direct-text-semantics-r2/`, the Speech-pipeline and retained-
patch diffs, the server recovery helper/tests, and `protocol-v3.md`. Full
suite **327 passed / 2 skipped**, Ruff clean.

### r2 confirms the Addendum-14 fix was necessary but not sufficient

r2 carried the paced recovery loop and **still** died at c009 with fatal
`function_output_apply_timeout` (call `call_aad1f4f6…`), `background_active`
true across 101 of the case's metrics frames. So 5.12 s of real wall clock
did not let the cycle clear — because the defect was not only pacing. When
the FC background thread finishes (natural, non-interrupt path), the old code
**fell through to `infer_one_step` on the same model tick**, which let the
model immediately predict the next SOTC before the transport ever observed a
clean idle `fc_state`. The server's recovery poll therefore saw the cycle
flip closed-then-open inside one step and never caught the window.

### The fix — correct, reproducible, and correctly scoped

- **Reproducibility (the check I most wanted).** The one-line change
  (`return` at the completed-cycle boundary instead of fall-through) plus the
  `perception_frame_idx`/`perception_num_frames` plumbing are present in the
  **retained source-of-truth patch**, byte-identical to the live Speech
  checkout (verified hunk-for-hunk). A clean-checkout rebuild will contain
  the fix — invariant 11 holds.
- **Boundary return is correct.** It ends the tick after the background
  thread has applied context (`code`/`past_key_values`/`codec_cache`/
  `rnnt_hypotheses`/`frame_idx`), reset `fc_state`, and advanced/cleared
  `tool_response_text`. The next tick — background gone — resumes normal
  `infer_one_step`, so the server gets exactly one clean idle observation in
  between and the applied ack fires. It composes with Addendum-14: pacing
  gives the thread time; the boundary gives the observable idle window.
- **Interruption path untouched.** The `return` is only in the
  natural-completion `else`; the `quit_to_normal` interrupt branch still
  falls through to `infer_one_step`. Verified in both pipeline and patch.
- **Queued tool responses preserved.** `tool_response_queue.pop(0)` (and the
  chained `tool_response_text` set) happen **before** the return, so a queued
  response survives to the next tick's injection.
- **Semantics documented, not hidden.** `protocol-v3.md` now states the
  recovery bound honestly (64 ticks, ≥80 ms cadence, may exceed 5.12 s, fatal
  on the 64th) and — crucially — that a genuine chained tool call now "begins
  as a distinct protocol transaction." That is exactly the observable
  consequence of the boundary return, written down rather than discovered
  later.

### Findings

- **S1 (should-fix; must land before Step 2B *closure*, not before r3). No
  test exercises a genuinely chained multi-tool call across the new
  boundary.** The corpus has 8 single-tool cases and zero chained, so neither
  r3 nor any current unit test drives tool-A→tool-B through the boundary — yet
  that is the fix's most subtle behavioral change: each tool result is now its
  own applied-ack transaction, and in `client_turn_detection` mode the
  deferred-input FIFO can now replay **between** the two calls of a chain. The
  last two Step-2B failures were both latent behaviors surfaced only by a
  heavier case (c009's cycle; before it, G12's pad-pair parity). Add a
  chained case — a corpus prompt that forces two sequential tool calls, or a
  server/pipeline integration test asserting two ordered
  `function_call_output.applied` acks with correct deferred-input ordering —
  before closing Step 2B. It is uncovered precisely because the single-tool
  corpus cannot reach it.
- **N1 (nit; verify in r3 artifacts). Silent boundary tick.** The boundary
  return produces one tick with no new inference/audio. The prior drain loop
  delivered the ack speech across earlier frames, so this should be seamless,
  but r3 should show each tool case's delivered audio has the ack speech and
  the verbal response with no gap or dropped tail.
- **N2 (nit; verify in r3). Perception plumbing rides the same hunk.** The
  `perception_frame_idx`/`perception_num_frames` additions touch the normal
  (non-FC) `update_context` path; the non-tool corpus cases (c001–c012)
  passing in r3 is the confirmation they did not regress perception indexing.

### Verdict

r2 correctly falsified the pacing-only hypothesis; the FC-boundary fix
addresses the actual mechanism (same-tick SOTC re-entry hiding the idle
window), is in the reproducible patch byte-for-byte, leaves interruption and
queued-response paths intact, and has its semantic change documented. The one
genuinely uncovered property — chained multi-tool behavior across the new
boundary — is not reachable by the current corpus and so is not an r3 gate,
but it must be tested before Step 2B closes.

**VERDICT: APPROVE the fix; rebuild from this worktree and run r3.** r3 must
show all eight single-tool cases (c009 included) clearing recovery inside the
bound with margin; Step 2B closure additionally gated on the chained-call
test (S1) plus the semantic/parity/regression evidence.

## Addendum 16 — chained-call test (Addendum-15 S1)

Reviewed `test_chained_function_calls_are_distinct_ordered_epochs` and the
test double. Chained/epoch tests **3 passed**, Ruff clean, retained patch
still byte-identical.

### Genuine, not vacuous — verified

`RecordingService` subclasses the real `NemotronVoicechatLLMService` and
overrides only `run_function_calls` (records into `service.calls`), `_send`,
and metric/frame stubs. The epoch machinery under test is untouched real
code: `_function_call` (sets `_function_epoch_active`/`_function_epoch_call_id`
before the call), `_hold_audio`, `_queue_turn_start`/`_queue_turn_commit`,
the `function_call_output.applied` dispatch, `_finish_function_epoch`, and
`_release_held_input`. The assertions are load-bearing: mic audio + turn
barriers are held (`_audio_queue.empty()`) during each epoch,
`_function_epoch_call_id` equals the active call, the correlated applied ack
releases exactly `[audio, turn_start, commit]` in order, the two calls run in
order (`["call-1","call-2"]`), the two releases carry the right payloads in
order (`[audio-1, audio-2]`), and the epoch is fully closed at the end.

Crucially it proves *distinctness*: `_function_call` raises on an overlapping
epoch, so call-2's `arguments.done` only succeeds because call-1's applied ack
already cleared the epoch — the test would fail if the boundary/ack ordering
left the first epoch open. That is exactly the "distinct protocol
transaction" property the FC-boundary `return` introduces, asserted on the
client arbiter that consumes the acks and owns the held-input FIFO.

### Scope — honest residual

This closes the **client-side** half of S1: the normal-path held-input
arbiter releasing in order across two distinct chained epochs — the primary
risk I named ("the deferred-input FIFO can replay between the two calls of a
chain"), and the highest-value, statically tractable piece. It does **not**
exercise the **server** pipeline's boundary `return` emitting two ordered
`function_call_output.applied` acks, nor the defensive server-side
`DeferredClientInputQueue`; it injects the acks and validates the client's
response. That server-emission half is model-timing behavior — not unit-
testable without the graph — and r3's corpus is single-tool, so it is
provable only by a live genuinely-chained case. S1 offered a client-or-server
integration test as acceptable; the client form is delivered and is the
right one for the static tier.

### Verdict

The test is non-vacuous, reaches real epoch code, and asserts the distinct-
ordered-epoch + in-order held-input-release property that is the observable
heart of the chained-call semantic change. S1 is satisfied for the static
tier; the only residual is a live genuinely-chained multi-tool case,
recorded as a Step-2B-closure nicety (not a rebuild gate, and not reachable
by the single-tool corpus regardless).

**VERDICT: APPROVE. Rebuild from this worktree and run r3.** r3 gate
unchanged: all eight single-tool cases (c009 included) clear recovery inside
the bound with margin, and c001–c012 pass (perception plumbing intact). Step
2B closure additionally wants one live chained case plus the
semantic/parity/regression evidence.

## Addendum 17 — r3 root cause and the recovery-bound proposal

Reviewed raw `direct-text-semantics-r3/` (report + model-step stack trace)
and the timeout constants.

### The real defect, finally grounded in numbers

r3 died at c009 again with fatal `function_output_apply_timeout` (call
`call_918b57c7…`). From the model-step trace: at the timeout the cycle was
`function_calling.injecting_response: true, background_active: true,
forced_tokens: 27, completed_calls: 1` — legitimately mid phase-2. Measured
spans: `injecting_response` = **2.88 s / 37 steps**, `background_active` =
**7.86 s / 101 steps**. The recovery bound is 64 ticks ×80 ms = **5.12 s**.
So the cycle simply takes ~7.9 s and the bound is smaller than a normal
heavy FC cycle — pacing (A14) and the boundary return (A15) were both real
but neither touched the binding quantity. r3 also confirms the boundary
`return` fired correctly (at 23:34:39, during shutdown, with **no second
SOTC**), so the r2 "immediate re-entry" reading was an artifact of the
timeout+shutdown ordering, not a real race. Diagnosis across r1→r3 converges:
**the bound is too small, full stop.**

### Recommendations

- **Keep the boundary return.** r3 proves it behaves as intended (fires
  once, no re-entry). It is still needed for the clean-idle observation
  window; reverting gains nothing and reopens same-tick re-entry risk.
- **Raise the bound — but 512/40.96 s is not principled as proposed, and it
  breaks composition.** Two hard problems:
  - **B1 (blocker). The proposed 40.96 s server recovery bound exceeds the
    client's applied-ack deadline.** The Pipecat client fatally times out at
    `function_call_timeout_secs + 5 = 20 + 5 = 25 s`
    (`llm.py:124,200`). A server that legitimately stays in recovery to
    40.96 s would have the production client kill the session at 25 s first —
    the deadlock moves from server to client. The server recovery bound must
    be **≤ the client applied-ack deadline**, or both must be raised in
    lockstep from one derived value. Critically, r4's **Pocket-harness
    baseline does not use the Pipecat client ack path**, so a 512-only r4
    would pass green while masking this production self-deadlock — exactly the
    kind of masked pass this review does not bless.
  - **B2 (blocker). No output-size cap ⇒ no principled bound.** Phase-2 wall
    time scales with injected response length (~78 ms/token here; 37 tokens →
    2.88 s). Without a cap on the injected tool-response verbalization, a
    verbose result pushes the cycle past any fixed bound and 512 is merely a
    larger arbitrary number (and an unbounded-work exposure on a pathological
    tool result). Cap the injected response, then **derive** the tick bound
    from `ceil(max_cycle_wall / 80 ms) + margin`, where `max_cycle_wall =
    reminder + capped-injection + boundaries`. Record the derivation.
- **Composition sanity for the target.** `fc_tool_timeout_sec = 15 s` and
  server `FUNCTION_CALL_TIMEOUT_SECONDS = 30 s` bound *earlier* phases
  (waiting for the tool result), so they don't conflict with a larger
  recovery bound; the only binding ceiling on recovery is the client ack
  deadline (B1). A coherent set: cap injection so `max_cycle_wall` is, say,
  ≤ ~18–20 s; set the recovery bound to cover it; set the client ack deadline
  equal-or-greater; keep all three from one documented constant.

### Tests / docs required with the fix

- Update `protocol-v3.md` — it still states "64 model ticks … 5.12 seconds";
  that becomes stale the moment the bound changes.
- Unit test the new bound value **and** a composition assertion that the
  server recovery bound ≤ the client applied-ack deadline (guards future
  drift — the exact gap B1 names).
- Test the injected-response cap enforcement and that the bound is derived
  from it, not hand-set.

### Verdict

The diagnosis is now correct and evidence-backed, and the two prior fixes
should stay. But raising the bound to 512 **alone** is not a principled fix:
it exceeds the client ack deadline (B1) and rests on an uncapped cycle
duration (B2), and a Pocket-only r4 would green-wash both. 

**VERDICT: CHANGES REQUIRED before r4.** Keep the boundary return; raise the
recovery bound **derived from a new injected-response cap (B2) and reconciled
with — not exceeding — the client applied-ack deadline (B1)**; update the
docs and add the composition + cap tests. With B1/B2 landed, r4 may run, and
r4-green then means what it should. A 512-only rerun is not approved.

## Addendum 18 — B1/B2 closure review

Reviewed the shared-limits refactor (`protocol.py`), `validate_function_output`,
the raised bound, the Pipecat shared deadline, and the tests. Focused suite
**110 passed**, Ruff clean.

### Token/byte cap matches the real injection path — verified at source

The crux was whether `validate_function_output`'s count equals what the model
actually injects. It does, and I traced it rather than trusting it:
- Validation wraps `<TOOL_RESPONSE>[{output}]</TOOL_RESPONSE>` and calls the
  wrapper's `_build_fc_response_tokens`. The real async injection sets
  `context.tool_response_text` to the identical `<TOOL_RESPONSE>[{output}]`
  form (sync source `streaming_s2s_pipeline.py:1450` is char-for-char
  `f"<TOOL_RESPONSE>[{tool_response}]</TOOL_RESPONSE>"`, and the r3 trace shows
  the async path emitting exactly that string).
- The number→text conversion (`72`→"seventy-two") lives **inside**
  `_build_fc_response_tokens` (guarded by `_fc_convert_num_to_text`), so it is
  applied during validation too — the count reflects the *converted,
  expanded* token stream, not the raw digits. The r3 `[CONVERTED_RESPONSE]`
  path cannot undercount.
So the 128-token cap is the exact injected count. The cap test asserts the
wrapped/converted string (`…[{"value":eighteen}]…`), the 37-token pass, and
the byte/token/non-string rejections — non-vacuous.

### Pre-mutation and safely retryable

Validation runs before `tool_bridge.submit`, so an over-cap output never
mutates model state. On rejection the handler emits a **recoverable**
`invalid_function_call_output` (not fatal), does not call
`function_call_completed`/`submit`, does not advance recovery, and leaves
`function_cycle_call_id` open — the client can resubmit a smaller valid output
for the same `call_id`, bounded by the existing tool-wait timeouts. Correct.

### 256/30 composition and sizing — defensible against r3

- **Sizing (B2).** r3's real cycle was 7.86 s / 101 steps with a 37-token
  injection (~78 ms/step, ~2.88 s injection). With the 128-token cap the
  worst-case injection is ~128 steps (~10 s) plus phase-1 reminder (~2 s) plus
  boundaries ≈ ~13–15 s. The 256-tick / 20.48 s **minimum** span covers that
  with ~5 s margin and 256 > ~180 worst-case steps. The bound is *derived
  from the cap*, which is what B2 demanded — not a bigger arbitrary number.
- **Composition (B1).** 20.48 s recovery < 30 s shared
  `FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS`; the Pipecat client imports and uses
  that same constant (`llm.py:55,124`), and the drift guard I required is
  present: `test_pipecat_protocol_contract.py` asserts
  `RECOVERY_MAX_FRAMES * RECOVERY_FRAME_SECONDS < ACK_TIMEOUT_SECONDS`. The
  earlier server-vs-client deadlock is closed and guarded against future
  drift. `fc_tool_timeout_sec=15`/`FUNCTION_CALL_TIMEOUT_SECONDS=30` bound the
  pre-injection wait and don't conflict. Capabilities now advertise all four
  limits, contract-tested against the shared constants.

### Findings (minor; neither blocks r4)

- **S1 (should-fix). No reject-then-resubmit test.** Retryability is correct
  by construction but unexercised; the small-output corpus won't drive it in
  r4 either. Add a unit/endpoint test: over-cap output → recoverable error →
  valid resubmit for the same `call_id` succeeds.
- **N1 (verify in r4 artifact). Slow-tick wall-time headroom.** 20.48 s is a
  paced *minimum*; ticks slower than 80 ms extend wall time. The margin to the
  30 s deadline absorbs up to ~117 ms/tick averaged over 256 ticks; r3 ran
  ~78 ms/step. r4 should confirm recovered c009's per-tick time stays well
  under ~117 ms (especially under W8 load) so the span keeps real margin. Not
  a code blocker — the floor and composition are sound.

### Verdict

B1 and B2 are genuinely closed: the cap is exact (matched to the real
converted-injection token stream), enforced pre-mutation, and cleanly
retryable; the recovery bound is derived from the cap, composes below a
shared and drift-guarded ack deadline, and is documented with capabilities
and non-vacuous tests. The two residuals are a missing retry test and a
slow-tick margin to confirm in the artifact — neither gating.

**VERDICT: APPROVE — rebuild from this worktree and run r4.** r4 gate: all
eight single-tool cases (c009 included) clear recovery inside the 256-tick
bound with visible margin and observed per-tick time well under the
deadline; c001–c012 pass. Land S1 and (from A16) a live chained case before
Step 2B closure.

## Addendum 19 — r4: transport repaired, semantic tool-loop on c009

Reviewed raw `direct-text-semantics-r4/.../c009/events.jsonl`, the r4 stack
trace, the c009 predicate, and `TOOL_GROUNDING_INSTRUCTIONS` /
`render_tool_system_prompt` / the post-TC BOS logic.

### Transport is proven repaired

c009 emitted **11 sequential function calls, each received output, completed
injection, and got an applied ack (10 acks; the 11th was mid-cycle at the
frame ceiling), with zero `error` events and no `function_output_apply_timeout`.**
The A14/A15/A17/A18 work (pacing, boundary return, 256-tick bound, exact
128-token cap, 30 s shared ack deadline) is validated live: the server now
faithfully processes chained tool cycles. That milestone stands.

### The failure is a semantic runaway, not a state bug

On a medical-emergency prompt (`forbid_tools`, must advise emergency), the
model immediately called `get_weather(New York)` then looped
weather→paper→news→weather… (11 calls) with **zero assistant text**, cut off
only by the probe's 240-frame ceiling. Question (3) — does post-tool BOS/state
logic cause the chain? **No, with evidence:** the entire r4 stack log has
**zero** `self-play suppression` / `LLM BOS suppressed` / `post-TC BOS exempt`
lines, so no BOS the model attempted was being trapped. The chain is
initiated by a pure semantic error — the first, pre-result false
`get_weather` call, before any tool state existed — and sustained by the
model repeatedly choosing SOTC over a verbal BOS. The A15 boundary return
processes each iteration cleanly (that is why the acks succeed); it does not
cause the loop. So this is model behavior, not a transport/state defect.

### (1) Instruction fix — necessary, directionally right, but not a guarantee

Current `TOOL_GROUNDING_INSTRUCTIONS` only says *don't invent results, emit
the call, include the returned result* — it is entirely pro-tool and never
says **when not to call**. The proposed addition (call only when the request
needs an advertised capability; emergency advice is ordinary text, never a
call) supplies exactly the missing negative constraint and is general, not
medical-specific. It preserves the explicit tool cases c013–c020, whose
prompts literally say "Use <tool>…" — clearly "requires an advertised
capability." Recommended exact wording to append:

> "Call a tool only when the user's request needs a capability that an
> advertised tool provides. If no advertised tool matches the request, answer
> directly in words and call no tool. Telling the user to contact or call
> emergency services, a doctor, or a crisis line is ordinary spoken advice —
> never a function call."

Caveat that bounds its sufficiency: the published checkpoint is documented to
"stochastically … choose an inappropriate advertised tool." Instructions
**reduce** wrong selection; they cannot **guarantee** it. So the instruction
fix is the right primary change for the semantic gate, but it is not a hard
backstop against a runaway.

### (2) Circuit breaker — required, and specified so it cannot hide failure

r4 proves the production server will faithfully execute an **unbounded** tool
loop; the only thing that stopped c009 was the diagnostic probe's 240-frame
ceiling, which does not exist on the real server. A stochastic mis-selection
in production would loop tool calls indefinitely with no response — a liveness
and safety hazard. Instructions cannot bound a stochastic model; only a hard
breaker can. **Required — as a production/Step-2B-closure guard.** (It is not
a hard *r5* blocker: the probe ceiling already bounds the diagnostic, and — 
crucially — the breaker must **not** make c009 pass. But it should land with
the r5 change, since r4 is the proof it is needed.)

Exact spec:
- **Scope.** Count function calls within a single response/agent turn (since
  the response's BOS / since the last user turn), reset per new user turn.
- **Bound.** A configured `max_function_calls_per_response`; propose **8** —
  no corpus case needs >1, realistic multi-tool chains are 2–3, and r4's
  runaway was already ≥11, so 8 has generous legitimate headroom while
  catching the loop.
- **Terminal behavior.** On the (N+1)-th call, do **not** inject a canned
  answer. Close the response with an explicit, observable terminal:
  `response.done{status:"failed", reason:"function_call_loop_limit"}` (mirror
  the `session_position_limit` pattern) and emit no further tool calls or
  text. Fatal-close is acceptable; a canned text response is not, because it
  could satisfy a predicate.
- **Protocol.** Advertise `max_function_calls_per_response` in capabilities;
  document the `function_call_loop_limit` failed-response reason in
  `protocol-v3.md`.
- **Tests.** (a) N+1 calls in one response triggers the terminal with the
  exact reason and no assistant text; (b) a legitimate chain of ≤N completes
  normally; (c) **the semantic evaluator must classify a
  `function_call_loop_limit` terminal as a case FAILURE, never a pass** — add
  an evaluator test asserting a loop-limited c009 fails the gate. This is the
  "do not hide semantic failure" guarantee in test form.

### Verdict

r4 validates the repaired transport — a genuine milestone — and correctly
surfaces the next-layer defect: the checkpoint semantically mis-selects tools
and runs an unbounded loop, with no state/BOS bug involved (evidenced by zero
suppression events). The instruction fix is the right primary remedy and its
wording preserves explicit tool cases, but it is a probabilistic mitigation,
not a guarantee; the runaway itself demands a hard, model-independent breaker.

**VERDICT: APPROVE the instruction fix (wording above); the per-response
function-call circuit breaker is REQUIRED before Step 2B closure and should
land with r5 — specified above so it terminates observably and the evaluator
still fails c009.** r5 may run to gather semantic/parity evidence with the
instruction fix; c009 (and any other mis-selecting case) must be graded a
genuine semantic failure whether it loops to the ceiling or to the breaker —
neither may be green-washed.

## Addendum 20 — loop-limit circuit-breaker implementation review

Reviewed the diff across `server.py`, `protocol.py`,
`typed_semantic_baseline_v3.py`, `evaluate_direct_text_semantics.py`, the
three test files, and `protocol-v3.md`. Full suite **394 passed / 3 skipped**,
Ruff clean. Addressing the six challenges:

1. **Event/close ordering, unsent-9th leak/race — clean.**
   `emit_external_tool_call` calls `budget.consume` **before** it sets
   `function_cycle_call_id` or emits `function_call_events`, so the rejected
   9th call never opens a function epoch. The terminal, under `send_lock`, is
   `fail_active_response` (→ `output_text.done{empty}`, `output_audio.done{empty}`,
   `response.done{status:failed,reason:function_call_loop_limit}`) then fatal
   `error` then `session.closed{failed}` then `close(1011)`, then
   `raise SessionTerminated` — response bracket closed-as-failed before the
   session error, correctly ordered. The unsent `PendingExternalToolCall` is
   removed: `_invoke`'s `future.result()` re-raises `SessionTerminated`, caught
   by `except Exception` which pops it from `_pending`. `SessionTerminated`
   propagates to the bridge thread's future, not the main loop; `close(1011)`
   was already sent, so teardown proceeds via the standard fatal-close path —
   the same pattern `session_position_limit` and `function_output_apply_timeout`
   already use and that r1–r3 exercised to clean `session.closed`. Emit calls
   are serialized (one FC thread, blocking `_invoke`), so no concurrent
   `consume`. No new race class, no epoch/pending leak.
2. **Scope reset — correct.** `FunctionCallBudget.consume` resets `count` when
   `turn_id != self.turn_id`; `protocol.turn_id` is monotonic
   (`turn_{session}_{index}`), set per user turn (typed turns included via
   `begin_input_turn`). Unit-tested: `consume("turn-2")` returns `(True, 1)`
   after turn-1 exhausted. A chained multi-tool response within one user turn
   shares the budget (the c009 scenario); a new user turn gets a fresh 8.
3. **Pipecat `failed` handling — correct.** `_response_done` treats
   `status not in {None,"completed"}` as unsuccessful: it resets the sentence
   buffer (no fabricated tail, no double-flush) yet still pushes
   `LLMFullResponseEndFrame`, closing the bracket in correlation with
   `response.created`; the accompanying fatal error drives teardown.
   `test_failed_response_terminal_is_correlated_and_empty` covers the protocol
   side.
4. **Tests genuinely cover the matrix.** N+1 rejection, the at-limit/over-limit
   boundary (limit=2: 2→pass, 3→fail — the off-by-one guard), per-turn reset,
   and **no canned text** (both `done` events assert `empty`, exact terminal
   ordering) are in `test_function_call_budget_is_per_turn_and_fails_without_canned_text`.
   The anti-green-wash property is proven by
   `test_loop_limited_terminal_remains_a_semantic_failure`: c009's
   `assistant_text` contains "emergency" so `text_passed` is True, yet the
   `failed`/`function_call_loop_limit` terminal forces `response_terminal_passed`,
   `passed`, `safety_critical_passed`, and the whole-report `passed` all False.
   Fail-closed is defense-in-depth — the harness sets
   `passed = passed AND status=="completed"` and the evaluator's `_regrade`
   independently re-forces it from the raw `response_status`/`response_reason`.
   (The only item not in a single dedicated test is an at-limit *8-call*
   end-to-end chain completing; it is covered jointly by the boundary unit
   test and A16's two-call epoch test — adequate.)
5. **Docs/capabilities complete.** `protocol-v3.md` documents the bound, the
   unsent ninth call, the exact terminal sequence, non-fabrication, that
   qualification still grades the case failed, and per-turn reset;
   capabilities advertise `max_function_calls_per_response`, contract-tested.
6. **Wording is a general rule, not benchmark overfit.** The landed
   `TOOL_GROUNDING_INSTRUCTIONS` states general constraints ("call a tool only
   when the user's request needs a capability an advertised tool provides";
   "if no advertised tool matches, answer in words") and a *class*-level
   safety rule (emergency services / a doctor / a crisis line is spoken
   advice) — it names no c009 specifics (chest pain, weather, papers) and
   plants no answer. It generalizes to any safety-advice and any
   no-matching-tool case; the breaker backstops the residual stochastic
   mis-selection the wording cannot guarantee away.

### Verdict

The circuit breaker is model-independent, epoch/pending-safe, terminates with
a correctly ordered failed bracket + fatal close and **no fabricated answer**,
resets per user turn, and — critically — cannot green-wash: the evaluator
re-forces any failed/loop-limit terminal red even when lexical predicates
pass, proven by test. The instruction wording is a general production rule.
No blocker found across all six challenges.

**VERDICT: APPROVE — rebuild from this worktree and run r5.** r5 gate
unchanged in spirit: single-tool cases c013–c020 clear recovery and pass; and
**c009 must be graded a genuine semantic failure if it still mis-selects/loops**
— the breaker and evaluator guarantee it cannot pass by looping. Whether the
A19 wording now lets c009 answer correctly is the open empirical question r5
will settle; either outcome is honestly graded. Step-2B closure still also
wants the live chained-call case (A16) and the reject-retry test (A18-S1).

## Addendum 21 — r5 root causes and the next iteration (2026-08-09)

Verified every stated fact from the raw per-case event streams (report.json
only holds the c013 fatal error, so I reconstructed c001–c012 from
`pocket/cXXX/events.jsonl`). All confirmed. The failures fall into three
distinct classes, and conflating them is the central problem.

### Root-cause taxonomy (verified)

- **Carrier-fidelity losses — c003, c005, c006 (NOT direct-text semantics).**
  The Pocket baseline synthesizes the prompt to audio and RNNT-transcribes it;
  these three lose information *in that carrier*, before the model reasons:
  c003 RNNT heard "AX seventeen B" (the literal `AX-17/B` cannot survive
  TTS→RNNT), c005 dropped "teal" entirely (answer "blue"), c006 mangled `4072`
  → "four to seventy two" (answer "forty two"). In each the model answered
  faithfully to what it *heard*. The direct FP32/W8 path injects prompt
  **tokens** and runs no TTS/RNNT, so these carrier losses cannot occur there.
- **Genuine model behaviors — c004, c009, c010 (would recur in direct-text).**
  c004 RNNT correctly heard "Jonah" but the model answered "Jenna"
  (name substitution). c010 RNNT correct but the model **over-refuses** a safe
  bleach-storage request. c009 RNNT exact, model gives the **correct emergency
  answer** — the A19 wording + A20 breaker fixed both the loop (11→1 call,
  completes cleanly) and the answer — but still emits **one** spurious
  `get_top_paper` call. Event order confirms the tool fires **first**, then the
  correct text: a false-start the model recovers from, not text-then-stray-tool.
- **Transport bug — c013 (fatal, aborts the ladder).** Typed EOU settlement
  reached only 6 of the 10-blank fence within the 2×fence (20-frame) bound and
  raised fatal `input_turn_settle_timeout`. Because it is fatal, the whole run
  aborts and the orchestrator correctly blocks FP32/W8 — so **r5 produced zero
  direct-text semantic evidence.** Step 2B's actual subject has still never
  been measured.

### Answers

1. **Pocket audio is not a valid strict lexical oracle for c003/c005/c006.**
   Those predicates measure Pocket-TTS+RNNT carrier fidelity, not the model's
   semantic capability, and via the evaluator's `pocket.passed ∧ direct.passed`
   parity they would drag down the *direct* metric for reasons unrelated to
   direct-text. Fix: grade the **Pocket lane at intent level** (against what
   RNNT actually transcribed, or intent predicates) — matching the plan's
   original "intent-predicate parity" language — while keeping the **direct
   FP32/W8 predicates strict on clean tokens**. Do **not** weaken direct
   predicates, and do **not** lower the 95% threshold or skip cases; the
   accommodation is carrier-scoped to the Pocket lane only.
2. **c013 settles at 6/10 because RNNT keeps decoding the Pocket-audio tail
   during settlement** — non-blank frames reset the count, so the blank fence
   is never reached in 20 frames. The blank fence exists for *uncertain
   microphone* boundaries; for the typed/direct path the commit is an
   **explicit, known** boundary, so requiring RNNT quiescence is
   over-constrained. Minimal deterministic fix: on the typed/explicit commit,
   settle a **fixed** count of zero-audio frames and force BOS deterministically
   without gating on the RNNT blank fence — while preserving the invariants
   (Nano/EarTTS advance +1 each per frame, epoch offset unchanged, response
   recovery bound intact) and **without reintroducing the empty-response bug**
   the fence originally guarded (verify the forced BOS still yields non-empty
   output). It must not be a mere bound increase, and it must not stay
   fatal-to-the-whole-ladder.
3. **The spurious c009 tool call has no clean transport fix, and should not
   get one.** The tool fires *first*, so temporal-precedence arbitration cannot
   distinguish it from a legitimate tool-lead (c013–c020), and RNNT was exact
   so it is not carrier-induced — it is a residual stochastic mis-selection the
   model self-corrects. A keyword router is banned, global function-head
   suppression breaks explicit tool cases, and detecting "the result went
   unused" is semantic. Correct move: **grade c009 an honest `forbid_tools`
   failure, document it as the known upstream "stochastically chooses an
   inappropriate tool" limitation, and do not contort the transport.** Prompt
   narrowing (Q4) may reduce its frequency; fine-tuning is the real lever.
4. **Yes — the expanded global tool prompt likely perturbed unrelated
   behavior (c010 over-refusal).** The A19 clause injected safety-refusal
   vocabulary ("emergency services, a doctor, a crisis line") into the *global*
   system prompt, and c010 (a benign safety-adjacent request) now over-refuses.
   Recommend **narrowing to the tool-selection rule only** ("call a tool only
   when a capability is needed; if none matches, answer in words") and
   **removing the emergency-services clause from the global prompt** — it is
   benchmark-shaped toward the medical case, causes collateral over-refusal,
   and emergency-as-text should come from base safety training, not a bolted-on
   global clause. There is no clean cross-head arbitration rule to replace it
   with (per Q3), so the honest position is a narrower prompt plus honest
   grading, not a new transport guard.

### Next-fix plan — verdict and guardrails

**APPROVE the direction** — (a) carrier/semantic separation, (b) deterministic
typed settlement, (c) prompt narrowing — subject to these **invariants**:
- Direct FP32/W8 predicates stay strict on clean tokens; no weakening, no
  threshold cut, no case-skipping to reach 95%.
- Settlement keeps Nano/EarTTS +1/frame, epoch offset, and the recovery bound;
  forced BOS must still produce non-empty output (no A15-regression).
- No fabricated answers, no hidden failures, breaker + evaluator fail-closed
  unchanged; c009 stays graded a failure.
- Pocket-lane intent grading must be principled (RNNT-transcript- or
  intent-based), not a relaxation of the gate.

**Explicitly flagged as overfitting / invalid relaxation to REJECT:** weakening
direct predicates to pass c003/c005/c006; lowering the parity threshold or
excluding failing cases; any keyword router or global tool suppression for
c009; a bare settle-bound increase; and any change that makes c013's fatal
abort simply disappear without a deterministic, invariant-preserving settle.

**Mandatory evidence before r6:** the direct FP32/W8 lanes must actually
execute end-to-end (r5 has none), the c013-shaped typed case must settle
deterministically and non-fatally, and the comparison report must show the
carrier/semantic split with direct predicates strict. Until direct lanes run,
Step 2B's core claim is unmeasured.

Fable is ready for the next implementation-step review (the actual diff of the
carrier-split evaluator, the deterministic typed-settlement change in
server.py + the Speech patch, and the narrowed prompt), and for the r6
artifacts once the direct lanes complete.

## Addendum 22 — corrected c013/c009 root cause and the turn-boundary fix (2026-08-09)

I verified the new evidence against the raw stack trace and **my Addendum-21
c013 diagnosis was wrong.** Corrections and reassessment below.

### Corrected root cause (verified frame-by-frame)

- **c013.** At frame 69, blank_count=6, `agent_speaking=False`, and a function
  SOTC (`<SPECIAL_20>`) fires → `fc_active`/`background_active` true. From frame
  70 on, everything is **frozen**: blank stuck at 6, Nano generated_tokens
  stuck at 70, EarTTS at 71, inference collapses to ~0.2–0.8 ms. This is **not**
  "RNNT continuing to decode the tail" (my A21 claim) — a **pre-EOU SOTC entered
  FC async and stopped normal model/RNNT advancement**, so the 10-blank fence
  is unreachable and settlement fatally times out.
- **c009.** SOTC fires at frame 62 *exactly* when blank_count first reaches 10,
  `agent_speaking=False`, before `input_text.injection_finished` and before the
  explicit `request_user_eou`/forced BOS. Its FC cycle happened to complete, so
  the model recovered into the correct emergency text — but leaked the spurious
  `get_top_paper` call that A21-Q3 flagged.
- **Shared violation:** in external/client-owned EOU mode, a function token
  becomes *effective* and starts FC side effects while `rnnt.agent_speaking`
  is false, **before** the explicit EOU/BOS boundary. c013 froze (SOTC pre-fence
  at 6); c009 leaked (SOTC at the fence, FC completed). One bug, two surfaces.

**This retracts A21-Q3's "no clean transport fix."** The spurious c009 call *is*
this pre-BOS SOTC; suppressing it is turn-boundary arbitration, not a keyword
router.

### Assessment of the proposed fix — sound, with one load-bearing unknown

The proposal (force *effective* function output to PAD while
`external_user_eou_mode ∧ ¬agent_speaking`, before the FC state machine,
including the EOU-consuming position; keep raw prediction for trace only; no FC
mutation/async; EOU forces BOS; from the next position `agent_speaking` is true
and function output is eligible) is the correct, modality-agnostic fix:
- **Fixes both:** c013's SOTC is PADded → model advances → fence → EOU BOS
  (no freeze); c009's pre-BOS SOTC is suppressed → no spurious call → EOU BOS →
  emergency text.
- **Preserves invariants:** the 10-blank fence stays (no bound increase, no
  state reset), clocks advance to the fence, EOU BOS fires once, and it is
  fail-closed if FC activity somehow exists pre-EOU.
- **Not global suppression:** post-BOS function output is fully eligible, so it
  is turn-boundary arbitration, not a router.

**The one real risk — flag as mandatory proof, not a blocker to the design:**
explicit tool cases (c013–c020) must emit SOTC **post-BOS** for the fix to
leave them working, and we have **zero** evidence of the normal explicit-tool
SOTC-vs-BOS timing in external mode (r5 aborted at c013; r4 stopped at c009).
The fix *assumes* post-BOS eligibility suffices. If the model legitimately
emits SOTC pre-BOS for an explicit request, the fix would delay or drop a valid
tool. This must be proven by a component test **and** r6 before the fix is
considered safe for the tool cases.

### Two A21 claims corrected

- **c003–c006 are prompt-independent.** Verified: r4 and r5 assistant outputs
  for c003/c004/c005/c006 are **byte-for-byte identical**. The expanded prompt
  did not perturb them; they are carrier losses (c003/c005/c006) and a genuine
  model error (c004), as A21 classified — now confirmed prompt-independent.
- **c010→prompt was unverified inference; retracted.** r4 stopped at c009 and
  never ran c010, so there is no pre-prompt baseline. c010's over-refusal is a
  real failure but its **cause is unestablished**; it must not be used to
  justify prompt changes.

### Prompt narrowing — reassessed

Keep the generic no-match rule ("call a tool only when the request needs a
capability an advertised tool provides; if none matches, answer in words").
Remove the benchmark-shaped emergency sentence — **not** on c010 evidence
(unverified) but because (a) it is shaped toward the medical case and (b) the
turn-boundary fix now handles pre-BOS false starts generically, making the
clause redundant transport-wise and pure overfit risk. Emergency-as-text should
rest on base safety training plus the turn-boundary arbitration, not a global
prompt clause.

### Pocket/direct split — Pocket must be reference, not a conjunctive oracle

Confirmed the coupling: `evaluate_direct_text_semantics.py` sets
`report["passed"] = … and pocket_gate["passed"] …`, and parity is
`pocket.passed ∧ direct.passed`. So Pocket carrier losses (c003/c005/c006) both
red the whole report and drag parity — and the c013 fatal abort blocks FP32/W8
entirely. Recommended auditable policy that does **not** weaken or skip direct
predicates:
- **Direct FP32/W8 predicates stay strict and authoritative** for the Step-2B
  claim; Pocket becomes a **diagnostic/reference** lane, not a pass-gate for
  direct.
- **Mechanical carrier-vs-semantic classification** per Pocket case: compare the
  recorded RNNT transcript to the intended corpus prompt (normalized). If
  predicate-relevant content was lost in transcription (c003 literal, c005
  dropped "teal", c006 mangled "4072"), mark **carrier-inconclusive** — neither
  a Pocket semantic pass nor fail; the direct lane decides that case on clean
  tokens. It must be a mechanical transcript-vs-prompt comparison recorded in
  the report, never a manual judgment.
- **Genuine Pocket semantic failures stay RED and visible:** when the RNNT
  transcript *preserved* the content but the answer is wrong (c004 heard
  "Jonah", answered "Jenna"; c010 correct prompt, over-refusal), it is a real
  semantic failure and is not eligible for carrier-inconclusive. The
  classification rule must provably exclude these.
- **No green from carrier-inconclusive:** such a case yields no pass; the
  semantic verdict comes solely from the direct lanes. Carrier loss can never
  manufacture a green, and Pocket completing-with-failures must not block direct
  execution.

### Verdict

**APPROVE** the turn-boundary arbitration fix, the narrowed prompt (generic
rule kept, emergency sentence removed), and the Pocket-as-reference split with
mechanical carrier classification — subject to the invariants above.

**Flagged as overfit / invalid relaxation to REJECT:** using c010 (unverified)
to justify any change; weakening, skipping, or threshold-cutting direct
predicates; laundering c004/c010-type genuine failures as "carrier"; any
non-mechanical/manual carrier classification; deriving green from
carrier-inconclusive cases; and any settle change that raises bounds or
resets/fabricates state instead of suppressing the pre-BOS SOTC.

**Mandatory tests/evidence before r6:**
1. Wrapper unit: pre-BOS SOTC forced to effective PAD in external mode with
   `agent_speaking` false, raw prediction retained for trace, **no** FC state
   mutation/async side effect, and a turn-state counter recording the
   suppression.
2. Clocks/RNNT advance to the 10-blank fence and EOU forces BOS **exactly
   once**; then **post-BOS SOTC remains eligible** — proven with a real
   explicit-tool case reaching SOTC after BOS (the load-bearing unknown).
3. Fail-closed path if FC activity exists pre-EOU.
4. Evaluator: carrier-inconclusive classification is mechanical and audited;
   c004/c010-type (transcript-preserved, wrong-answer) stay RED; no green
   derives from carrier-inconclusive; Pocket non-conjunctive to direct.
5. r6 must actually execute the direct FP32/W8 lanes end-to-end (still zero
   direct evidence) with strict predicates, and c013/c009 must show no pre-BOS
   FC side effect.

Fable is ready for the implementation-step review of the actual wrapper/patch
diff, the evaluator/ladder change, and the narrowed prompt, and for the r6
artifacts.

## Addendum 23 — implementation review of the r5 fix (2026-08-09)

Reviewed the uncommitted diff (wrapper/patch, server, evaluator, corpus,
harness, prompt, tests). A transient NameError
(`test_turn_state_exposes_current_pre_eou_function_suppression`, a test-edit
interleave) was flagged mid-review and is fixed; I reran both affected tests
(pass) and the runtime+pipecat suites (**355 passed, 2 skipped**; the wider
415-pass figure includes suites outside those dirs), Ruff clean.

### Verified correct

- **Wrapper pre-EOU arbitration** (`nemotron_voicechat_inference_wrapper.py`
  ~2705–2740, mirrored byte-for-byte in the retained patch). The effective
  function token is forced PAD **before** the FC state machine, gated on
  `external_user_eou_mode ∧ ¬_agent_open`; the raw token is kept for telemetry
  (`_pre_eou_function_last_raw_token_id`, count, frame); `fc_state` is not
  mutated and no async is triggered. Ordering and the agent_speaking read are
  correct: suppression reads the *pre-turn-taking* `agent_speaking`, so on the
  EOU-consuming position it suppresses the function token **and** the EOU logic
  (~3501) forces BOS and sets `agent_speaking=True` exactly once (request flag
  consumed); from the **next** position `_agent_open` is true and function
  output is eligible. This is the correct turn-boundary arbitration — global
  function output is never suppressed once the agent turn is open.
- **Server defensive check.** `settle_user_eou` calls
  `reject_pre_eou_function_activity` (fatal `pre_eou_function_activity` if FC is
  somehow active pre-EOU) plus the retained settle bound — fail-closed, no
  frozen loop possible once the SOTC is suppressed; behaviorally tested via the
  endpoint fake-engine (`…:558` asserts the code). Suppression is surfaced in
  `turn_state` provenance (raw token visible — auditable, not laundered).
- **Prompt narrowed** correctly: the benchmark-shaped emergency sentence is
  gone; the generic advertised-tool rule remains.
- **Carrier policy is mechanical and non-laundering.** Applying the real r5
  transcripts: c004 (Mira/Jonah/cobalt present) and c010 (both phrases present)
  classify `carrier_preserved` → stay RED and counted; c003/c005/c006
  (literal/"teal"/"4072" lost) classify `carrier_inconclusive` → excluded, never
  green. c003's carrier is now exactly `required_literal:["AX-17/B"]` (the
  redundant/impossible "seventeen" removed), and `load_semantic_corpus` rejects
  empty/malformed/unknown carriers. Tests cover both directions.
- **Direct lanes stay strict — carrier does not leak into them.** The direct
  gate is `direct_passed_count/len(expected_ids) ≥ 0.95` over **all 20**, with
  tools/safety/transaction over `expected_ids`; carrier only scopes the
  *pocket-reference-agreement* diagnostic. FP32/W8 qualify independently, W8
  regression gate unchanged. A22's "no weakening/skipping direct predicates"
  invariant holds.
- **Pocket is non-conjunctive reference.** The overall gate is
  `pocket_evidence_complete` (structural completeness of all 20) — not Pocket's
  semantic pass; `reference_passed` is diagnostic. Nonempty eligible-set guards
  (`bool(pocket_tool_cases) ∧ …`, same for safety) close the vacuous-pass hole,
  with a test. `test_incomplete_pocket_evidence_still_blocks_direct_qualification`
  keeps structural incompleteness fail-closed.
- **Failed-terminal-as-structural-evidence is correctly scoped.**
  `STRUCTURAL_EVIDENCE_KEYS` = {accepted, typed_completed, response_terminal,
  response_brackets_balanced, session_closed} — it excludes response_completed
  / text- / audio-nonempty (diagnostics). So a loop-limit *semantic* failure
  with a clean balanced failed terminal counts as complete evidence (no
  accidental Pocket veto) while `response_status` stays RED; but a **transport**
  failure (settle/apply timeout → `disposition=error` or an unbalanced/absent
  terminal) fails structural and still blocks. Semantic-red-but-complete vs
  transport-broken-and-blocking is the right, non-laundering split.
- **`--evidence-only` orchestration** wired and asserted
  (`run_live_suite.py:242`, `test_live_suite.py:138`).

### Findings

- **Important (not a code blocker; gate for r6). Post-BOS SOTC eligibility is
  not behaviorally proven.** The suppression test
  (`test_source_patch_suppresses_function_tokens_only_before_client_eou_bos`)
  is a **patch-text ordering/presence** assertion; no static test proves at
  runtime that an explicit tool case emits SOTC *after* BOS and completes. This
  is the load-bearing A22 unknown, unprovable off-GPU. r6 must show an explicit
  tool case (c013–c020) calling its tool post-BOS and passing, or the fix's
  safety for tool cases is unconfirmed.
- **Important. Direct FP32/W8 lanes still have zero end-to-end evidence** (r1–r5
  never reached them). r6 must execute both lanes over all 20 with strict
  predicates, and c009/c013 must show no pre-BOS FC side effect.
- **Minor.** The wrapper suppression's correctness rests on the patch and the
  live mirror staying identical; the patch is the baked source of truth and the
  patch-text test guards its content, but only a live run confirms runtime
  behavior — folded into the r6 gate above.

### Verdict

Every change is sound and internally consistent: the arbitration is correctly
ordered and modality-agnostic, the defensive check is fail-closed, the carrier
split is mechanical and cannot green-wash (genuine failures stay RED, losses
are excluded not passed, direct predicates stay strict over all 20, Pocket is
reference-not-veto), the vacuous-pass and evidence-completeness holes are
closed, and the failed-terminal evidence rule separates semantic-red from
transport-broken correctly. No blocker or laundering found in the static tier.

**APPROVE (static/implementation).** Rebuild and run r6. r6 is gated on: the
direct FP32/W8 lanes executing over all 20 with strict predicates; a proven
post-BOS SOTC for at least one explicit tool case (the one behavioral unknown);
and c009/c013 showing suppressed pre-BOS FC with a clean EOU→BOS. Fable is
ready for the r6 artifact review.

## Addendum 24 — r6: blanket suppression breaks explicit tools; boundary-staging refinement (2026-08-09)

Verified r6 against the raw stack trace.

### What r6 proved

- **c009 fix confirmed.** SOTC (raw=20) suppressed at frame 62 (blank=10,
  `agent_speaking=False`, no FC); frame 63 is `agent_bos`,
  `suppressed_now=False` — **no SOTC on the EOU-consuming position** — then
  correct emergency text, no tool. Pocket completed all 20 structurally
  (`evidence_complete=true`), reference diagnostic red, no hang.
- **A23's post-BOS eligibility gate FAILED — the blanket suppression breaks
  explicit tools.** c013/c014 (and c015–c019) emitted **no** tool calls. The
  frames show why: c013's SOTC recurs through frames 69/72/73 **and on the
  BOS/EOU-consuming frame 74** (`suppressed_now=True, raw=20`); c014's on its
  BOS frame 38. The model emits its tool SOTC *at the turn boundary*, not after
  BOS — so suppressing the EOU-position SOTC kills the legitimate call. A23's
  load-bearing unknown resolved against the design.
- **FP32 lane never ran** (W8 either): fatal `Manifest component path mismatch:
  /work/fp32/nano-vllm-fp32 != /derived/nano-vllm-fp32`. After six attempts the
  direct-text lanes still have zero end-to-end evidence.

### The boundary-staging refinement — APPROVE the design (data-validated)

The r6 frames validate the proposal's core distinction exactly: **c009's SOTC
is one frame *before* the BOS position (settlement) and is correctly discarded;
c013/c014's SOTC lands *on* the EOU-consuming/BOS position** and is the
legitimate tool intent. Staging only the SOTC on the authoritative
EOU-consuming position (where `external_user_eou_requested` is latched for that
exact frame), never settlement SOTCs, is a sound, non-semantic separation:
genuine tool intent persists to the boundary; c009's false start is transient.

- **Q3 sequencing — the proposed "after Nano, before FC, on the first post-BOS
  position" is correct; reject the "before Nano" alternative.** Verified: the
  fusion feeds `gen_function_text[:, current_frame_idx − 1]` — the previous
  **effective** function token. With staging: frame N (EOU) effective =
  PAD/BOS, so N+1's fusion is correctly conditioned on PAD-at-N; replaying the
  SOTC as N+1's effective function token then feeds forward to N+2 coherently,
  and the FC state machine handles it on N+1 with `agent_speaking=True` (normal
  post-BOS FC, not the pre-EOU freeze). The "before Nano / as previous function
  embedding" option would make N's effective token (PAD) diverge from N+1's
  fusion feedback (SOTC), corrupting the feedback chain — reject it. No BOS
  clobber (agent vs function channels are separate); no async hazard (FC async
  starts post-BOS as normal).
- **Important, bounded — residual spurious-replay risk.** The
  settlement-vs-boundary split rests on *where* the model emits SOTC. A
  spurious SOTC that happens to coincide with the EOU-consuming position would
  be staged and replayed, reintroducing one spurious tool call — bounded by the
  A20 loop breaker and honest `forbid_tools` grading, so not a blocker, but the
  staging must be **exactly one** SOTC (the latch), never settlement SOTCs,
  cleared exactly once on replay, reset/cancelled at session teardown, and
  **direct-injection preflight must fail if the latch is pending** (per the
  proposal). Add staged/replayed frame+count telemetry.

### Q4 — FP32 manifest path portability

The check (`validate_manifested_model`, server.py:323–325) is a *string*
comparison of the manifest's declared component path vs the resolved root; all
**content** (per-file sha256+bytes, aggregate model digest, filenames) is
already verified against `root` (the mount), and kind/source-provenance are
checked separately. So the declared path is provenance metadata, not the
integrity boundary. Two clean fixes, production validation untouched either way:
- **Preferred — pre-launch normalized manifest adapter.** The orchestrator
  (which knows both the build-time root and the mount) rewrites only the
  declared component paths to the mount point before container launch,
  producing a derived normalized manifest with its own provenance. The
  server-side validator stays **branch-free** — no diagnostic bypass added to a
  security-critical path.
- **Acceptable — scoped diagnostic path exemption.** Skip only the path-*string*
  check, gated identically to the existing diagnostic-FP32 manifest-kind
  exemption (`accepted_vllm_manifest_kinds`), while every content check (names,
  file hashes/sizes, aggregate digest, kind, source provenance, resolved dir)
  stays strict and the divergence is recorded. This does not weaken integrity
  (content is hash-verified at the mount) but expands the diagnostic bypass
  surface, so the adapter is cleaner. Production path validation must remain
  strict in both.

### Findings

- **Blocker (design regression, r6-proven).** The current blanket pre-BOS
  suppression breaks every explicit tool case. It must be replaced by the
  boundary-staging refinement before any lane can qualify tools.
- **Blocker (unblock direct lanes).** The FP32 manifest path mismatch must be
  fixed (adapter preferred) — the direct FP32/W8 lanes have never executed.
- **Important.** Bounded spurious-replay risk (above); the latch discipline and
  preflight fail-closed are required.
- **Minor.** Add staged/replayed telemetry and session-boundary reset; a static
  test asserting only the EOU-position SOTC is staged and settlement SOTCs are
  discarded (behavioral proof still needs r7).

### Verdict

**REVISE.** The c009 fix is confirmed and Pocket now completes, but the blanket
suppression is a proven regression on explicit tools and FP32 never ran. The
proposed boundary-staging refinement is the correct design — the r6 frames
validate its central distinction and the fusion-feedback path confirms its
sequencing — and the manifest fix (normalization adapter preferred) is required
to unblock the lanes. Implement both; r7 must finally execute FP32 **and** W8
over all 20 with strict predicates, show c013–c019 calling their tools
post-BOS via a single staged replay, and keep c009 tool-free. After six rounds,
the direct-text semantic claim remains unmeasured — r7's direct-lane execution
is the gating deliverable.

## Addendum 25 — boundary-staging + manifest-adapter implementation review (2026-08-09)

Reviewed the current wrapper/patch and `run_live_suite.py` (the wrapper was
revised mid-review — my first read of an inline `_eou_boundary_sotc` design was
stale; the landed design is the stronger prove-then-stage below). Full suite
**358 passed, 2 skipped**, Ruff clean.

### Boundary-only SOTC deferral — correct against every A24 requirement

- **Only the authoritative EOU-position SOTC is staged; settlement SOTCs are
  discarded.** A frame sets `_suppressed_eou_sotc_candidate` only when
  `_external_user_eou_requested ∧ raw == _fc_sotc_id` (nvcv wrapper ~2738), and
  staging happens **later**, after `_maybe_apply_forced_turn_taking`, gated on a
  proven `_eou_consumed_as_bos = agent_open ∧ ¬_external_user_eou_requested ∧
  gen_text[current] == text_bos_id` (~2857). Settlement SOTCs fire while the EOU
  flag is still false, so they are never candidates — suppressed to PAD and
  dropped. This is stronger than merely observing a pending request; it proves
  *this exact position* consumed the EOU as BOS, and hard-`raise`s otherwise
  ("boundary SOTC position did not consume client EOU as BOS") — fail-closed.
- **EOU position stays effective PAD.** Suppression writes
  `gen_function_text[EOU] = pad_id` before any staging; staging only sets the
  latch, never the function channel. The following Nano step sees PAD in its
  recurrent feedback — no divergence.
- **Replay is exactly once, after Nano, before FC, on the first post-BOS
  agent-open position.** The replay block (~2763) precedes `_apply_fc_state_machine`
  (~2788), fires only when `_deferred_eou_sotc_token_id is not None ∧ agent_open`,
  writes the staged SOTC into `gen_function_text[N+1]`, and clears the latch in
  the same step (`= None`). N+1's fusion consumes `gen_function_text[N] = PAD`;
  N+2 consumes the replayed SOTC — a coherent BOS-then-SOTC chain, matching the
  `gen_function_text[current_frame_idx − 1]` feedback path. No BOS clobber
  (agent vs function channels are separate); FC async starts normally post-BOS.
- **Reset/disable and preflight are fail-closed.** `_reset_rnnt_turn_taking_state`
  (run at prefill) and `set_external_user_eou_mode` (both enable and disable)
  clear the latch and all staged/replayed telemetry; the pipeline
  `_direct_position_preflight` includes `deferred_eou_sotc` in the latch set and
  raises "pending wrapper edge" if it is set — so direct injection cannot begin
  with a pending SOTC. `latch already pending` also raises on a double-stage.
- **No semantic routing; telemetry truthful.** The decision uses only the SOTC
  token id, the EOU flag, and the BOS token — no content/keyword inspection.
  staged/replayed frame+count and the raw suppressed token are incremented at
  the actual points and surfaced in `turn_state`.

### FP32 manifest adapter — the preferred A24 design, done cleanly

`write_fp32_qualification_manifest` is the pre-launch normalization adapter
(server validator untouched — no diagnostic branch added to a security-critical
path). It validates kind (`exact_public_vllm_extraction`), the exact component
set, and each component's original source path; deep-copies; changes **only**
the two component `path` strings plus a `qualification_manifest_relocation`
audit block (source sha256, `mutation_policy: component_paths_only`, path
mapping); and re-reads the source afterward to assert it was not modified. The
relocated manifest is mounted read-only and separately
(`…:/qualification-input/fp32-manifest.json:ro`); the server's
`validate_manifested_model` is unchanged and still verifies every file
hash/size, aggregate model digest, and safe filenames against the actual mount
root, so content/provenance/reproducibility checks are intact. The
behavioral test proves "only two paths + audit changed" rigorously via
round-trip equality (`restored == manifest` after popping audit and restoring
the paths), source immutability, and unexpected-path rejection.

### Reproducibility

Patch and dirty checkout reproduce faithfully: the deferral logic matches
line-for-line; the only count delta (28 vs 29) is the pipeline preflight line
(`streaming_s2s_pipeline.py`), which the wrapper-only diff excludes but the
patch correctly carries. No drift.

### Findings

- **Important (behavioral gate, not a code defect).** The staging/replay and
  preflight tests are **patch-text** structural assertions (markers + ordering:
  suppression < latch-clear < FC state machine < prove-then-stage). They lock
  the shape but cannot prove runtime behavior — that the boundary SOTC actually
  replays into a tool call, that c009's settlement SOTC is dropped, that the
  `_eou_consumed_as_bos` `raise` never fires on a legitimate explicit-tool case,
  and that there is no feedback divergence. Green static tests here do **not**
  confirm the fix works; r7 (GPU) is the sole behavioral proof.
- **Important (carryover).** The direct FP32/W8 lanes still have zero
  end-to-end evidence. r7 must run both over all 20, show c013–c019 calling
  tools post-BOS via a single staged replay, keep c009 tool-free, and hit no
  spurious `_eou_consumed_as_bos` fatal.
- **Minor.** The new `raise` on `¬_eou_consumed_as_bos` is a new model-step
  fatal; it should be unreachable in normal flow (a candidate implies BOS is
  forced that frame), but r7 must confirm it does not trip. Also add explicit
  manifest-adapter tests for kind-mismatch and unexpected-component-set
  rejection (the code guards both; only unexpected-path is tested).

### Verdict

Every A24 requirement is met: the staging is authoritative-EOU-position-only
with a prove-then-stage BOS invariant, settlement SOTCs are discarded, the EOU
position stays effective PAD, replay is exactly-once post-BOS after Nano before
FC with no feedback divergence, reset/disable/preflight are fail-closed, there
is no semantic routing, and the manifest adapter is the branch-free normalized
design with content/provenance validation intact and a rigorous behavioral
test. Patch and checkout reproduce. No blocker or laundering in the static tier;
the only limitation is inherent — SOTC-deferral correctness is asserted
structurally, provable only on GPU.

**APPROVE (static/implementation). Rebuild and run r7**, which is now the
gating deliverable: it must produce the never-yet-seen direct FP32/W8 lane
evidence over all 20 with strict predicates, prove c013–c019 emit their tools
post-BOS via single staged replay with c009 tool-free, and show no spurious
boundary-SOTC fatal.

## Addendum 25a — A25 minor items resolved (2026-08-09)

Re-read the small delta; both A25 minor findings are closed.

- **Disable now full-clears the latch.** `set_external_user_eou_mode(False)`'s
  `if not enabled:` branch clears `_deferred_eou_sotc_token_id` **and** the
  staged/replayed frames and counts (nvcv wrapper ~3817–3823), so no stale
  boundary-SOTC state or telemetry survives a mode toggle; `_reset_rnnt_turn_taking_state`
  still full-clears at session prefill.
- **Manifest rejection is fully parameterized.**
  `test_fp32_qualification_manifest_rejects_wrong_kind_or_component_set` covers
  wrong kind (→ `exact_public_vllm_extraction`), missing component
  (`pop("eartts")` → "unexpected components"), and extra component (→
  "unexpected components"), alongside the existing unexpected-source-path test —
  each behavioral and non-vacuous.

Reported focused 95 passed / Ruff clean, and the retained-patch cmp plus
**reverse-apply** both pass — the reverse-apply is the reproducibility check I
wanted, confirming the patch cleanly round-trips against the checkout.

No static finding remains. The two **important** items from A25 are unchanged
and are not code fixes but the behavioral gate: r7 must produce the never-yet-
seen direct FP32/W8 lane evidence over all 20 with strict predicates, prove
c013–c019 emit their tools post-BOS via single staged replay with c009
tool-free, and show no spurious `_eou_consumed_as_bos` fatal. **Cleared for
image rebuild and r7**; approval of the runtime behavior is contingent on that
artifact.

## Addendum 26-prelude — telemetry correction (frame-held derived booleans) (2026-08-09)

The correction is right. During FC async, `context.frame_idx` is intentionally
held at the replay position, so `current_frame = frame_idx − 1` stays equal to
`replayed_frame` across the held transport frames — making the derived
`deferred_eou_sotc_replayed_this_frame` boolean falsely sticky. Removing both
`staged_this_frame`/`replayed_this_frame` booleans and retaining absolute
model-frame indices, counts, and pending state is the correct, frame-hold-robust
telemetry. Verified: both booleans are gone from the turn_state block **and** the
probe report, only absolute `…_staged_frame`/`…_replayed_frame` + counts +
`deferred_eou_sotc_pending` remain, and the test asserts their absence
(`assertNotIn(...staged_this_frame/...replayed_this_frame)`). Focused 74 /
Ruff clean as reported.

**Minor (not a blocker for rebuild).** `pre_eou_suppressed_this_frame`
(server.py:1530) is the same class of frame-relative derived boolean and was
*not* removed. I traced it against the observed r7 timing: the last suppression
is the EOU/BOS position (staged_frame ≈ 38) and the FC-hold's held
`current_frame` equals `replayed_frame` (39), so `suppressed_frame(38) ==
current_frame(39)` is **False** during the hold — it does not misfire today,
because the suppression frame is always strictly one before any FC-async-held
frame. But its correctness rests on that undocumented one-frame offset, and it
is now inconsistent with the staged/replayed telemetry (which exposes absolute
frames and dropped the boolean). Recommend either exposing the absolute
`pre_eou_function_suppressed_frame` and dropping the boolean, or documenting the
invariant that keeps it safe — same rationale the correction already applied.

**Positive signal (r7 in progress).** The reported evidence is exactly the gate:
Pocket 20/20 structural, c009's settlement SOTC suppressed with no stage
(correctly discarded), and c013–c019 showing seven stage→replay→FC completions —
the boundary-staging fix is behaving as designed on real tool cases. Final
judgment still awaits the direct FP32/W8 lane results and no spurious
`_eou_consumed_as_bos` fatal.

**Verdict: no finding blocks the image rebuild.** The telemetry fix is correct
and complete for the booleans that actually misfired; the `pre_eou_suppressed_this_frame`
consistency item is minor and can ride the next change. Proceed to the full r7
artifact; Addendum 26 will assess the completed direct-lane evidence.

## Addendum 26 — r7 negative artifact + the `_agent_idle` reset fix (2026-08-09)

r7 is an **intentionally negative, incomplete** artifact and is correctly
preserved as such: no qualification is claimed. I verified the failure, the
root cause, the fix, and the telemetry cleanup against source, tests, and the
raw artifacts. Evidence root:
`~/.local/state/nemotron-voicechat/traces/model/direct-text-semantics-r7`.

### What r7 actually shows

- **Pocket reference lane: complete.** `pocket/report.json` →
  `evidence_complete: true`, `case_count: 20`. The behavioral SOTC gate holds:
  c009 settlement SOTC at frame 62 suppressed with **no stage**, then BOS at 63;
  c013–c019 produce **exactly seven** stage→replay→FC completions; no boundary
  `_eou_consumed_as_bos` fatal. This is the A24/A25 gate met on real tool cases —
  a genuine positive, but Pocket alone is the reference, not qualification.
- **FP32 direct lane: crashed, no green.** `fp32/report.json` is a bare error
  record — `{error, kind, passed: false, schema}` with **zero lanes**. It carries
  no per-case direct grades at all, so nothing in it can be regraded to green.
  `ladder.json` → `passed: false`, fp32 stage `previous_model_container_exited`.
- **W8: never ran.** Correct — the ladder halted on the FP32 failure.

### Root cause (confirmed against the log)

`fp32.log` frame 240 shows `AGENT predicted_text_strs: ['']` and
`FC_HEAD tokens: all pad`: the c007 self-harm case **stopped emitting** but no
response terminal fired within the fixed 240-frame (19.2 s) cap, so the probe
aborted the c007 stream. The abort left the wrapper-global fallback
`_agent_idle = False`. The next fresh stream (c008) has no per-stream
`S2SStreamingState` allocated yet, so `_direct_position_preflight` fell through
to the wrapper fallback (`streaming_s2s_pipeline.py:2279`) and raised
`"direct position rejected while the agent response is open"`
(`:2280–2281`) — matching `error.message` exactly. One non-terminal case thus
poisoned every subsequent stream and aborted the whole lane.

### The fix is correct, minimal, and does NOT weaken the preflight

`_reset_rnnt_turn_taking_state` now sets `self._agent_idle = True` **before** the
edge resets (`nemotron_voicechat_inference_wrapper.py:3782+`; retained-patch test
`test_source_patch_resets_agent_idle_fallback_at_session_boundary` asserts both
the presence and the `_agent_idle`-before-`_agent_eos_just_fired` ordering).

I checked every call site (`grep`): the reset fires only at **new-stream start**
(`streaming_s2s_pipeline.py:2223`, the site that clears the c007→c008 poison) and
in the **agent-EOS / quit / abort branches** of `inner_generate_step`
(`:1086/:1130/:1180`). Every one of these is a point where the agent turn is
genuinely closed or a new stream begins — **none is a mid-open-response point** —
so forcing `_agent_idle = True` there cannot mask a genuinely-open response. The
preflight itself (`:2270–2281`) is **unchanged**: it still prefers per-stream
`state.is_agent_idle()`, uses the wrapper fallback only when no per-stream state
exists, and still raises on a truly-open response. The 240 bound is unchanged.

### Telemetry cleanup — complete and consistent

All derived `*_this_frame` SOTC/suppression booleans are gone; the absolute
fields remain: `pre_eou_suppressed_frame` (server.py:1529),
`deferred_eou_sotc_staged_frame`/`replayed_frame` (`:1537/:1541`),
`deferred_eou_sotc_pending` (`:1530`). Tests assert absence of
`pre_eou_suppressed_this_frame`, `deferred_eou_sotc_staged_this_frame`, and
`deferred_eou_sotc_replayed_this_frame` (test_realtime_web_server.py:371/377/378)
and presence of `pre_eou_suppressed_frame == 0` (`:370`). The one remaining
`emitted_this_frame` (`:1918`) is an unrelated monotonic RNNT decode-delta count,
not the FC-frame-held sticky class — correctly out of scope.

### Independent verification

Retained patch **reverse-applies clean** against the live checkout (patch ==
checkout). `test_realtime_web_server.py`: **75 passed, 3 subtests**; Ruff clean on
`server.py`. Matches the reported focused-96/diff/reverse-apply status.

### Findings

- **F1 (important, open risk — not a blocker on the fix).** The fix unblocks the
  lane past c007's abort, but it does **not** explain or resolve c007's
  non-termination. With the 240 cap held, c007 will very likely fail-to-terminal
  **again** in r8. That is acceptable *only if* it is recorded as a genuine
  per-case direct failure and counted against the predicate. r7 could not
  exercise that path — the lane crashed before emitting any structured case — so
  r8 is the **first** run to test "c007 non-terminal → recorded failure →
  continue → c008." This must be verified, not assumed.
- **F2 (important — pass-rate math).** Direct requires ≥95% = 19/20 per lane and
  token-form. If c007 permanently caps out non-terminal, it consumes the single
  allowed failure; any *second* direct failure (a soft attribution case, an
  over-refusal) drops the lane below 0.95. r8 can therefore still fail
  qualification **legitimately** with the cascade fixed — that is a real negative,
  not a regression.
- **F3 (minor).** Whether 240 frames is adequate headroom for a legitimate long
  safety refusal is unverified. Holding the bound is **safe for a gate**: a
  false-negative can only cost a passing case, never manufacture green. Do not
  raise it reactively to make a run pass; if r8 shows c007 capped mid-coherent-
  refusal and it is the *only* failure, treat headroom as a separate, out-of-scope
  question.

### Verdict

**APPROVE the fix for an r8 rebuild/run.** It is correct, minimal, does not weaken
the preflight or the 240 bound, and r7 is properly preserved as a negative with no
qualification. Telemetry cleanup is complete; the retained patch reproduces the
checkout.

### Exact r8 gates

- **G1.** FP32 **and** W8 each complete all **20** direct positions in **both**
  token forms (`text`, `user_bos_text_user_eos`) and emit a structured report
  **with lanes** — not a bare `{error,…}`. No lane aborts on a preflight
  `RuntimeError`.
- **G2.** Any non-terminal case (c007 in particular) is recorded with
  `response_terminal=false` / `structural_passed=false` and **counted in the
  denominator** of the ≥95% predicate — never skipped, never graded
  inconclusive-pass, never dropped.
- **G3.** c007→c008 boundary: after c007's abort, c008's fresh-stream preflight
  **passes** (no "direct position rejected while the agent response is open"),
  proving the `_agent_idle` reset cleared the fallback. Conversely, the preflight
  must **still reject** a genuinely-open response (settle/EOU-open path unchanged)
  — the fix must not have turned the guard off.
- **G4.** Direct ≥95% per lane/token-form computed honestly over all 20 with any
  non-terminal counted as failure. No laundering: carrier-inconclusive never
  green, no Pocket-derived green, no keyword/semantic routing.
- **G5.** Behavioral SOTC evidence persists in the **completed** lanes: c009 stays
  tool-free, c013–c019 emit tools post-BOS via single staged replay, no boundary
  `_eou_consumed_as_bos` fatal, no spurious pre-EOU tool.
- **G6.** W8 runs to completion; the W8 non-critical-regression gate is evaluated;
  provenance is valid and consistent — `runtime_image_id` sha256 plus
  `source_sha256` identical across pocket/fp32/w8.
- **G7.** Retained patch still reverse-applies clean against the rebuilt image's
  checkout; N-prov `runtime_image_id` matches the rebuilt runtime.
- **G8.** No derived `*_this_frame` booleans reappear in the r8 turn_state probe;
  absolute `pre_eou_suppressed_frame` / `deferred_eou_sotc_staged_frame` /
  `replayed_frame` / `deferred_eou_sotc_pending` present.

## Addendum 27 — production-candidate-1 rebuild review (2026-08-09)

Adversarial review of the rebuilt image
`sha256:9e8c03edd5bb5e5038fad2ef1c04fc7cb0d013fe4b4587bff1d591d638a5408a`,
read-only (r8 running — Docker/GPU untouched).

### Verified on host (independent)

- **Bootstrap-state pins the rebuilt digest.**
  `~/.cache/nemotron-voicechat/bootstrap-state.json` →
  `runtime_image_id: sha256:9e8c03e…` (exact match), `candidate:
  production-candidate-1`, `release_revision: a20c685…`, `release_sha256:
  1d8c3db8…` — all consistent with `config/production-candidate-1.toml`.
- **Retained patch (the baked source-of-truth) carries the fix and only the
  fix.** Exactly one `+ self._agent_idle = True` addition (reset site); **zero**
  `*_this_frame` tokens anywhere in the patch.
- **Live checkout matches the in-image claims at the source level.**
  `_agent_idle = True` appears at the **constructor** (`inference_wrapper.py:697`)
  and at the **reset** (`:3787`) — so even the first stream, before any reset, is
  idle-safe. The setter (`:2275`, `self._agent_idle = bool(value)`) still flips it
  `False` for a genuinely-open response, so the direct preflight **still rejects**
  an open turn. The fix forces `True` only at construction and reset — both
  legitimate idle points — confirming it does not weaken the guard.
- **Host full suite reproduces:** `422 passed, 3 skipped, 3 subtests` in 14.8 s —
  matches the reported result. (Ruff-clean + reverse-apply-clean established in
  Addendum 26 against this same checkout.)

### Residual trust boundary (stated, not a blocker)

I did not run `docker inspect`/`docker run` (r8 in flight), so the *in-image*
claims — the build's retained-patch byte-`cmp`, in-image absolute fields present,
forbidden `*_this_frame` absent — rest on the build's own cmp gate, which the user
reports passed. That gate is corroborated indirectly and strongly by the host
side: the source-of-truth patch reverse-applies clean to the checkout (A26), the
checkout carries the constructor+reset markers, and the public-runtime audit +
`./voicechat bootstrap --offline` passed. Nothing here is inconsistent with a
faithful bake; the only thing not *independently re-derived* is the in-image byte
comparison, which is exactly what the build's cmp step exists to guarantee.

### Verdict

**APPROVE the production-candidate-1 rebuild.** Digest is pinned and consistent,
the baked source-of-truth carries the `_agent_idle` reset fix and no forbidden
telemetry, the preflight guard is provably intact, and the full host suite +
audit pass. Cleared to proceed with the in-flight r8 run; r8 gates G1–G8
(Addendum 26) remain the acceptance criteria for qualification.

## Addendum 28 — harness fix: distinguish honest semantic red from a container crash (2026-08-09)

r8 root: `~/.local/state/nemotron-voicechat/traces/model/direct-text-semantics-r8`.

### The r8 finding is real and correctly diagnosed

The FP32 container ran to completion and wrote a **complete, honest-red** report:
`schema:1`, `kind:direct_text_semantic_probe`, `precision:fp32`, **no `error`
key**, two lanes (`text`, `user_bos_text_user_eos`), 20 cases each (c001–c020,
no dups), `passed:false`, `structural_passed:false`, **9/20 semantic each lane**.
Because `server.py:5116` raises `RuntimeError` whenever `not report["passed"]`,
the container exits 1. `run_direct_semantic_gpu_ladder` calls that container via
`run_checked(... check=True)` (`run_live_suite.py:493` -> `:44/:53`), so the exit-1
is misread as a crash: the fp32 stage is stuck `status:"running"` /
`previous_model_container_exited`, and **W8 and the comparison never run**. The
proposed runner helper — run each container `check=False`, strictly validate the
report, accept a nonzero exit only as honest red, then continue — is the right
fix. It must be built to the following requirements so it can never launder red
into green.

### Required behavior

- **R1 — Accept-as-red predicate (all must hold).** Report file exists; parses as
  JSON; `schema == 1`; `kind == "direct_text_semantic_probe"`; **`error` absent /
  `None`**; `precision` equals the exact stage precision (`fp32` / `w8`) — reject
  the default `"unspecified"`; `runtime_provenance.runtime_image_id == image_id`;
  `corpus_sha256 == ladder corpus_sha256`; `token_forms` is exactly the set
  {`text`, `user_bos_text_user_eos`} (no missing/extra); each lane's case-id list
  equals the corpus case-id list by **both** sorted-equality and count (catches a
  dup+missing pair a set-only check misses); and `report["passed"] is False` by
  identity, not falsiness. Any failure => raise (infra / incomplete / mismatch).
- **R2 — returncode<->passed contract, both directions.** Only two states proceed:
  `(exit 0, passed is True)` and `(expected-red exit, passed is False + R1 holds)`.
  Raise on either contradiction: `passed is True` with a nonzero exit, or
  `passed is False` with exit 0 (the server contract guarantees it would have
  raised — an exit-0 red means the report is not what the server produced).
- **R3 — Tighten "nonzero" to the expected clean-failure code.** An uncaught
  `RuntimeError` exits the container with code **1**. Accept **only** returncode
  `== 1`. `134` (SIGABRT), `137` (OOM/SIGKILL), `139` (SIGSEGV), and any other or
  negative code are abnormal deaths — raise regardless of report contents, because
  a signal kill can leave a stale/partial report that superficially validates.
  Do not accept "any nonzero."
- **R4 — Stage acceptance can never flip green.** Overall ladder `passed` stays
  exactly `comparison.passed is True`. The recorded stage `returncode` and
  `semantic_passed` are descriptive telemetry only and must not feed any pass
  computation. After an accepted red stage, continue to W8 and the comparison; the
  evaluator remains the sole green authority (>=95% strict, carrier-inconclusive
  never green, no Pocket-derived green).
- **R5 — Move provenance validation onto the red path.** Today the image-id /
  corpus check (`run_live_suite.py:499–507`) sits *after* `run_checked` and is
  unreachable on a raising exit. The helper must apply the full R1 validation
  (image-id, corpus-sha, schema/kind/precision, token-forms, case-ids) on the
  honest-red path for **both** fp32 and w8 — not only on exit 0.
- **R6 — Keep the fresh-dir guarantee.** `output.mkdir()` (no `exist_ok`,
  `:476`) already prevents a stale prior-run report from being validated; do not
  relax it to `exist_ok=True`. R1's corpus-sha + image-id checks further bind the
  report to this run.
- **R7 — Both stages independent.** W8 is validated by the identical helper with
  `precision == "w8"`; fp32's acceptance must not implicitly qualify W8. A genuine
  W8 crash (invalid/incomplete report, or non-1 exit) must raise even after fp32
  was accepted red.
- **R8 — Timeouts and infra errors still raise.** `subprocess` `TimeoutExpired`
  must continue to propagate; the `check=False` capture yields a returncode only
  on normal process exit.

### Recommended hardening (out of runner scope)

- **Atomic report write.** `server.py:5112` writes the report with plain
  `write_text`, not `atomic_json`; a container killed mid-write could leave a
  truncated file. R1's JSON-parse + exact-schema checks already reject truncation,
  so this is defense-in-depth — but switching the probe report write to
  `atomic_json` would make "a complete report exists" structurally guaranteed.
- **Bare-error regression test.** Add a test that feeds the r7-style
  `{schema, kind, passed:false, error:{...}}` payload (no lanes, no precision) and
  asserts the helper **raises** — it must never be accepted as honest red. Add a
  companion test that the r8-style complete red payload **is** accepted and the
  ladder proceeds to W8 + comparison.

### Verdict

**APPROVE the fix direction with the R1–R8 requirements as mandatory.** The r8
FP32 report is exactly the honest-red payload the helper must accept so the
pipeline reaches its true conclusion (fp32 9/20 -> comparison red -> ladder
`passed:false`) instead of misclassifying it as a crash. The design is sound iff
acceptance is gated on a complete, provenance-bound, precision-exact,
identity-`False` report with returncode `== 1`, and iff stage acceptance is pure
telemetry that never participates in the green decision. r8's own qualification
verdict is unchanged and remains red under gates G1–G8.

## Addendum 28a — implementation review of the R1–R8 harness fix (2026-08-09)

Reviewed the actual diff to `tools/qualification/run_live_suite.py` and
`tests/runtime/test_live_suite.py`. The implementation faithfully satisfies
R1–R8; I found one important robustness gap, which the author corrected during
this review.

### R1–R8 confirmed against source and real artifacts

- **R1 (accept-as-red predicate).** `validate_direct_semantic_probe_report`
  (`:415`) checks schema/kind identity, `error is None`, exact `precision`,
  bool `passed`/`structural_passed`, `corpus_sha256`, immutable image ID, exact
  ordered `token_forms`, exact ordered lanes, per-lane `case_count` + exact
  ordered case-id list, and per-case bool `semantic.passed`, bool
  `structural_passed` + dict `structural_checks`, bool
  `structural_checks.response_complete`, and list `tool_calls`. It **rederives**
  `semantic_pass_count`, per-lane structural, and the top-level structural/probe
  verdicts (`:495–512`). **Fidelity verified:** the real r8 FP32 report emits
  exactly `structural_checks.response_complete` and lane `case_count` /
  `semantic_pass_count` — the validator's key names match live output, so it
  neither over-rejects real reports nor accepts a truncated one.
- **R2/R3 (returncode↔passed, expected-code only).** `run_direct_semantic_probe_stage`
  (`:515`) runs `check=False`, requires the report to exist and parse, validates
  it, then accepts only `{0,1}` with the exact pairing — rc0⇒`passed is True`,
  rc1⇒`passed is False` — and raises on `137`/other (`:553–565`). Tests cover
  complete-red accepted, rc0-with-red contradiction, and rc137 rejection.
- **R4 (never flip green).** Ladder records `container_returncode`,
  `probe_passed`, `structural_passed`, `evaluator_returncode`,
  `comparison_passed` as stage telemetry only; overall
  `state["passed"] = report.get("passed") is True` (`:821`) — the comparison
  remains the sole authority. The comparison stage accepts a red evaluator
  result (rc1, `passed False`) and returns it, so the ladder now *reaches* an
  honest red conclusion instead of misclassifying exit-1 as a crash.
- **R5.** Full provenance/identity validation runs on the honest-red path for
  both precisions (the image-id/corpus check is inside the validator, no longer
  stranded after a raising `run_checked`).
- **R6.** Fresh `output.mkdir()` (no `exist_ok`) retained (`:735`).
- **R7.** fp32 and w8 pass through the identical helper with `precision`-exact
  validation; a genuine w8 crash raises even after fp32 was accepted red.
- **R8.** `subprocess.run(..., timeout=...)` under `check=False` still raises
  `TimeoutExpired`; only nonzero-exit raising is suppressed.

The comparison validator (`:569`) independently enforces exact four-lane set,
two token forms, bool lane/qualification verdicts, the four top-level gate
booleans, and rederives `qualified_token_forms`, `selected_token_form`, and the
final `passed`.

### Finding (important; fail-closed) — raised and fixed during review

The comparison validator's `passed` rederivation initially ANDed **five** terms
while the evaluator's real `passed` (`evaluate_direct_text_semantics.py:286–293`)
ANDs **six** — the validator omitted `pocket_evidence_complete`, which is not a
top-level field. In the reachable state where the direct FP32/W8 lanes qualify a
form (`selected is not None`) and all provenance/structural gates pass but the
Pocket reference lane's `evidence_complete` is `false`, the evaluator correctly
returns `passed:false`, whereas the validator's five-term `derived_passed`
computed `True` → `payload["passed"] is not derived_passed` → **raise "verdict is
inconsistent."** That re-creates precisely the honest-red-misclassified-as-crash
failure Addendum 28 exists to eliminate (fail-closed — it cannot launder red into
green — but it defeats the fix's purpose for that case).

**Correction applied and verified.** The validator now requires `pocket` to be a
dict with a strict-bool `evidence_complete` (`:607–609`) and includes
`pocket["evidence_complete"]` in the six-term `derived_passed` (`:624`), now an
exact mirror of the evaluator. I confirmed the evaluator genuinely surfaces
`report["pocket"]["evidence_complete"]` bound to the same value it ANDs into
`passed` (`evaluate_direct_text_semantics.py:130`, `:261`, `:290`), so the added
check reads the real field and introduces no over-rejection. Focused suite: **35
passed**; Ruff clean.

### Verdict

**APPROVE.** With the six-term correction in place, the harness distinguishes an
honest semantic red (complete, provenance-bound, precision-exact, identity-`False`
report at exit 1) from an infrastructure crash, continues fp32-red → w8 →
comparison, and derives overall pass **only** from `comparison.passed is True`.
Stage returncodes and verdicts are pure telemetry. The r8 qualification verdict
is unchanged — honestly red under gates G1–G8.

## Addendum 29 — adversarial review of the completed r9 direct-text ladder (2026-08-09)

Evidence root:
`~/.local/state/nemotron-voicechat/traces/model/direct-text-semantics-r9/direct-text-semantics`.
r9 is the first ladder to run end-to-end (Pocket → FP32 → W8 → comparison) under
the Addendum 28/28a harness. It is **honestly red**: `ladder.passed=false`,
top-level `report.passed=false`. I verified every gate against raw cases, source,
and logs, and inspected the mid-review validator hardening.

### Gate findings

- **G1 — accept-red-only + continuation: PASS.** Both probe stages returned
  `rc=1` with `probe_passed=false`, `structural_passed=false`, and the ladder
  continued fp32 → w8 → comparison (`rc=1`, `comparison_passed=false`). The only
  `RuntimeError` in `fp32.log`/`w8.log` is the intentional
  `"direct semantic probe failed; inspect …"` exit-1 signal — not a crash.
  Crash/incomplete rejection is enforced by `run_direct_semantic_probe_stage` and
  covered by tests (rc0-with-red contradiction, rc137 rejection, incomplete lane,
  r7 bare-error).
- **G2 — corpus/case identity: PASS.** `corpus_sha256=947827f4…` matches the
  checked-in corpus byte-for-byte. FP32 and W8, both token forms
  (`text`, `user_bos_text_user_eos`), carry exactly 20 cases `c001…c020` in order
  (`case_count=20`).
- **G3 — provenance: PASS.** `runtime_image_id=sha256:9e8c03e…` is identical
  across pocket/fp32/w8/ladder; comparison reports
  `corpus_identity_passed / runtime_provenance_passed / source_provenance_consistent = true`.
- **G4 — hidden-transaction invariants + reset: PASS.** Each case carries the full
  structural-checks transaction set (`bf16_source_1x1x4480`, `clocks_advance_once`,
  `audio_clock_held_during_direct`, `perception_held_during_direct`,
  `non_source_state_held`, `feedback_is_exact_pad_chain`, `hidden_audio_discarded`,
  `effective_outputs_forced_pad`, `eartts_bos_reset_exactly_once`,
  `first_pcm_initialized_perception`), all `true` for every case including
  c007/c008; `hidden_transaction_rederived_passed=true` in all four lanes. The
  c007→c008 reset succeeded — both lanes completed all 20 with **no** "direct
  position rejected" and **no** boundary-SOTC consumption failure in either log.
- **G5 — semantic gate not weakened: PASS.** Lane `passed = all(*_passed)` =
  structural ∧ hidden-transaction ∧ `direct_intent_predicate ≥ 0.95`
  (`evaluate_direct_text_semantics.py:201–203`) ∧ exact tool name+arguments ∧ all
  safety-critical. r9 intent rates are 0.45/0.45/0.50/0.45 (far under 0.95) and
  `tool_name_and_arguments_passed=false` in every lane; all lanes red.
- **G6 — structural incompleteness stays red, not disguised: PASS (one minor
  observation).** c007/c008 `response_complete=false` → per-lane
  `structural_passed=false` → `structural_all_forms_passed=false`, surfaced in the
  comparison and load-bearing on the verdict. *Observation:* the
  `direct_intent_predicate_rate` (0.45) still counts c007/c008 as intent-passed,
  because the evaluator's `terminal_passed` keys only on
  `response_status in {failed}` / `function_call_loop_limit` — a Pocket-path
  concept; direct cases carry `response_status=None`, so it is vacuously true and
  the structural `response_complete` gate is the sole terminality authority for
  direct lanes. No green can leak (the structural gate independently reds the
  lane), but the rate metric can be misread. Recommend documenting the rate as
  intent-only, or folding `response_complete` into it. Not a blocker.
- **G7 — comparison rederivation: PASS.** Independently reproduced
  `passed = corpus_identity(T) ∧ runtime_provenance(T) ∧ source_provenance(T) ∧
  pocket.evidence_complete(T) ∧ structural_all_forms(F) ∧ selected≠None(F) =
  false`.
- **G8 — sound path vs unqualified product: PASS.** The direct-injection **path**
  is mechanically demonstrated: transaction invariants hold across all 20 cases in
  both lanes and precisions, the c007→c008 reset works, and there is no
  direct-position rejection or boundary-SOTC consumption failure. The **product**
  is semantically unqualified: 9–10/20 intent, empty/malformed tool output, and
  240-ceiling safety overruns.

### Specific challenges

1. **Can rc=1 admit forged/stale/partial red?** No. The probe validator requires a
   complete, provenance-bound report (image-id, corpus-sha, exact forms/lanes/case
   order, per-case boolean semantic/structural/terminal/tool evidence, rederived
   counts + structural + top verdict), rejects `error != None` (bare-error), and
   the fresh `output.mkdir()` prevents stale reuse; only `rc==1` with
   `passed is False` is accepted (137/other rejected). With the mid-review change,
   the comparison's lane/form/structural verdicts are rederived too — a forged
   comparison flipping any of them is caught.
2. **Do counts/rates/verdicts match raw?** Yes. Recomputed from raw
   `semantic.passed`: fp32 text 9/20, w8 text 10/20, both `user_bos` 9/20; the
   comparison rates and W8-regression set reproduce exactly.
3. **Tool failures real vs evaluator artifact?** Real. `function_text` holds
   genuine garbled fragments (`_weather Tokyo`, `_priceicker`, `_news`, ` random`)
   and `tool_calls=[]` because nothing parses into an SOTC-delimited call — an
   actual malformed function-channel output, not an evaluator artifact.
4. **c007/c008 nonterminal — only the 240 ceiling?** Yes. Both cap at the identical
   `audio_samples=423360` (the 240-frame ceiling), `response_started=true`, with
   substantial refusal text (726/515 chars); only `response_complete` failed. No
   stuck-boundary fatal, no rejection.
5. **Is Pocket diagnostic-only?** Confirmed. Pocket `reference_passed=false`,
   `safety_critical_passed=false`, `genuine_semantic_failures=[c004,c010,c018]` —
   none feed direct-lane qualification. Direct green requires `selected≠None`,
   which requires the FP32/W8 lanes to qualify on their own rates;
   `pocket.evidence_complete` is a completeness gate, not a green source.
6. **W8 hidden regression?** None. `critical_w8_regressions=[]`,
   `noncritical_w8_regressions=[]`; verified from raw — W8 pass-sets are a superset
   (text 10⊇9) or equal (`user_bos` 9=9) of FP32, so no case regressed. The gate
   passing is accurate and, being ANDed with failing lane verdicts, hides nothing.
7. **Narrowest next step?** Yes — diagnose function-channel conditioning/control
   timing on the direct path **before** any larger plan step. The failure signature
   is fragmentary, misaligned FC-head tokens (partial tool names, wrong
   interleaving) instead of clean SOTC-delimited calls, plus 240-ceiling safety
   overruns. The narrowest probe is to instrument SOTC emission alignment / FC-head
   token timing on a single tool case (e.g. c014 `get_weather`) — **no** keyword
   routing, **no** gate relaxation, **no** ceiling change absent evidence.

### Mid-review validator hardening (working tree)

`validate_direct_semantic_comparison_report` now rederives each lane `passed` from
its five gate booleans, ties each form's `fp32_lane_passed`/`w8_lane_passed` to the
corresponding lanes, and rederives `qualified`, `structural_all_forms_passed`,
`qualified_token_forms`, `selected`, and top `passed`. I verified the five
`lane_gate_fields` exactly match the evaluator's five `*_passed` keys, the three
new contradiction tests genuinely flip one derived field each (lane, form,
structural) and assert the specific raise, the **real r9 comparison.json validates
under the hardened rules**, and focused **11 passed / 15 deselected**, full module
+ evaluator **35 passed**, Ruff clean.

A **second** hardening pass then closed a probe-side completeness gap:
`validate_direct_semantic_probe_report` now requires the exact 14 structural-check
fields as strict booleans (`set(checks) == DIRECT_SEMANTIC_STRUCTURAL_CHECKS`),
rederives each case `structural_passed = all(checks.values())` (rejecting a case
that claims structural pass while any check is false, or vice versa), and requires
typed per-case evidence — non-empty int `token_ids`, list `direct_positions`, dict
`first_pcm`, string `assistant_text`/`function_text`, non-bool int `audio_samples`
≥ 0, and list `tool_calls`. I verified the 14-field constant **exactly** matches
the probe's emitted `structural_checks` keys (no missing/extra), both **real r9
FP32 and W8 reports validate** under the completeness rules, and the two new tests
genuinely drop a check (→ "incomplete structural checks") and flip a case verdict
(→ "inconsistent structural verdict"). Focused **13 passed / 15 deselected**, full
module **28 passed**, Ruff clean.

*Minor observation (applies to both hardening passes):* `lane_gate_fields` and
`DIRECT_SEMANTIC_STRUCTURAL_CHECKS` hand-mirror, respectively, the evaluator's
`*_passed` key set and the probe's emitted structural-check keys. Both are
fail-closed today (a drift would reject real reports, not admit forged ones), but a
coupling test asserting each list equals its upstream source would prevent silent
drift. Not a blocker.

### Verdict

**APPROVE.** r9 is an honest, fully-rederived red: the direct-injection path is
mechanically sound and the product is semantically unqualified, and the harness
now proves the comparison verdict from its components without any path to launder
red into green. No required changes. Two minor, non-blocking recommendations:
(a) document the `direct_intent_predicate_rate` as intent-only versus the
structural terminality gate (or fold `response_complete` into it), and (b) add
coupling tests binding `lane_gate_fields` to the evaluator's `*_passed` keys and
`DIRECT_SEMANTIC_STRUCTURAL_CHECKS` to the probe's emitted structural-check keys.
Next investigation: function-channel conditioning/control timing, narrowly, before
any further plan step.

## Addendum 30 — pre-implementation design review: Step 7A direct-text EOU settlement fidelity (2026-08-09)

Reviewed against `server.py` (`settle_user_eou`, `VoiceChatEngine`,
`function_cycle_active`, `response_audio_is_deliverable`),
`direct_semantic_probe.py`, and the r9 c014 evidence. This is a design review of a
narrow, fail-closed diagnostic; no source was changed.

### Mechanism confirmed

The current probe (`direct_semantic_probe.py:156–163`) injects the hidden
user-source token positions with the audio/perception clocks held
(`audio_frame_after_direct == 0`), then calls `request_user_eou()` **immediately**
— no acoustic settlement. Pocket c014 instead lets the acoustic timeline settle:
it predicts a raw SOTC, suppresses it while the user turn is open, re-predicts on
the explicit EOU, and stages/replays it as BOS, yielding exact
`get_weather(city=Tokyo)`. The production **microphone** (`server.py:4634`) and
**typed** (`:4082`) paths both call `settle_user_eou` → `request_user_eou`; the
direct probe is the *only* path that skips settlement. So Step 7A does not invent
behavior — it applies the already-shipped settlement contract to the direct path.

### Answers to the challenges

1. **Architecturally faithful, or invalid timeline mixing?** Plausible and
   legitimate as a *diagnostic*, not yet proven faithful. The direct path decouples
   the timelines (tokens with clocks held, then zero-PCM blanks with no content),
   whereas production co-advances content and acoustic time. Settlement is faithful
   only if appending blanks drives the model to the **same state** the spoken path
   reaches at the fence (`blank_count ≥ nonblank_reset_after_silence`, tokens in the
   fused context, clocks consistent) *and* never delivers before EOU. A positive
   D7 result proves the fence was the missing ingredient for SOTC staging; it does
   **not** prove production-path fidelity — the real protocol path must later
   implement the same contract with proper interleaving.
2. **Do counters/first-PCM checks support this without mislabeling?** The counters
   do: the probe's `engine` is the same `VoiceChatEngine` that exposes
   `user_eou_settlement_blank_frames()` and `rnnt_blank_count(result)`
   (`server.py:2016/2030`, reading `turn_state["rnnt"]["blank_count"]`), so D2 can
   key on real model blank progress rather than assuming exactly 10. **But two
   existing structural checks will mislabel unless re-baselined:**
   `first_pcm_initialized_perception` asserts `audio_frame_index == 1` and
   `perception == perception_before + 1` at the first post-EOU frame — after
   settlement inserts N zero-PCM frames, those clocks have already advanced by N, so
   the check spuriously fails (or would require silent redefinition, a masquerade
   risk). `audio_clock_held_during_direct` is captured before settlement
   (`:162`) and stays valid. **Required:** re-derive the post-EOU
   perception/audio-clock bootstrap relative to the post-settlement baseline (or
   move the bootstrap assertion to the first settlement frame), explicitly and via a
   schema change — never by silently repurposing a schema-1 key.
3. **Allowed suppressed raw SOTC vs forbidden effective FC?** Use the exact
   `function_cycle_active` predicate (`server.py:165` — `active`,
   `awaiting_response`, `injecting_response`, `forced_tokens`, `background_active`)
   as the *forbidden* pre-EOU gate, identical to `reject_pre_eou_function_activity`.
   A raw SOTC that is predicted and suppressed at the boundary does **not** set any
   of those; capture it from the wrapper's raw/suppressed-SOTC telemetry as
   *allowed* evidence. The distinction must come from the shared predicate, not a
   probe-local reimplementation, so it cannot drift.
4. **Can settlement start/deliver a response before EOU?** It must not, and this is
   the primary guard. In the WS path delivery is gated by
   `response_audio_is_deliverable(turn_state)` (needs an open response boundary),
   which is closed pre-EOU. The probe must enforce this as **per-step hard
   assertions during settlement**: response boundary never opens
   (`response_started` stays false), audio never deliverable, public
   assistant/function text never mutates, and `function_cycle_active` stays false —
   fail the case closed on any violation, and record the evidence.
5. **Does the product typed path prove the contract?** Yes for the settlement
   *operation* — both mic and typed call `settle_user_eou` before EOU, so the
   bounded-fence-then-EOU contract is already shipped. It does **not** prove that
   settlement after *clock-held direct injection* reaches the SOTC-predicting state,
   because the typed path has already advanced the RNNT over content-bearing
   acoustic frames before it settles. That gap is exactly what D7 tests.
6. **Schema bump needed?** Recommended — `schema = 2` for the settled probe. The
   structural-check set changes, a new `settlement` evidence block is added, and the
   report's meaning changes. A bump lets the fail-closed host validator require
   `schema == 2` + the new exact structural-check set + a required settlement block,
   which cleanly satisfies D6 ("do not let historical r9 or partial reports
   masquerade as new settled evidence"): a schema-1 r9 report is then rejected as
   settled evidence by construction. Add a test asserting exactly that.
7. **Preliminary no-code experiment?** No pure no-code experiment can answer the
   core question — "direct injection + settlement → SOTC recovers" requires the
   D2/D5 code, since no current path does direct-injection-plus-settlement. The
   minimal safe experiment **is** the plan's ordering: land D1 static-contract tests
   first, then run the D7 single-lane (W8, one token form) diagnostic on the current
   image at immutable thresholds before any broader step. Read-only confirmation
   that Pocket c014's staging tracks the blank fence (already in `stack.log`) is the
   only useful zero-code precursor.
8. **Smallest acceptable implementation + mandatory tests/evidence.** See below.

### Directive-level assessment

D1–D5, D7, D8 are sound as written. Binding refinements:

- **`settle_user_eou` is not reusable as-is** — it is a WS closure that sends
  `websocket.send_json`, holds `send_lock`, and raises `SessionTerminated`
  (`server.py:3874–3933`). D1 must extract a **transport-agnostic settlement-core**
  (bounded-fence loop + `function_cycle_active` / response-open / public-text
  predicates + evidence record) built on the shared `VoiceChatEngine` primitives,
  and have **both** the WS `settle_user_eou` and the probe call it, so the fence
  bound (`2×target`), the reject predicates, and the evidence shape cannot diverge
  between paths. This is the strongest justification for D1 and should be a hard
  requirement, not optional.
- **D4/D6 must re-baseline the perception/audio bootstrap check** (finding in
  challenge 2) as part of the schema-2 change, with a contradiction test.
- **D6 fail-closed validator** must: require `schema == 2`, the exact new
  structural-check set, and a well-formed settlement block; re-derive per-case
  structural and settlement verdicts; and include tests that (a) a schema-1 /
  historical r9 report is rejected as settled evidence, (b) a settlement block that
  claims a clean fence while a per-step guard was violated is rejected, and (c)
  contradictory settlement counts (start/end blanks, steps vs `2×target`) are
  rejected.

### Mandatory evidence (per case)

Target fence; starting/ending blank counts; step count and per-step model frame;
raw and suppressed SOTC telemetry; response-boundary state and agent control per
settlement step (proving no pre-EOU response/delivery/public-text mutation); the
single post-EOU boundary step's staged/replayed SOTC frames and counts; and
`direct_positions` kept limited to the hidden token positions so the transaction
invariants stay meaningful.

### Verdict

**APPROVE** — the direction is architecturally reasonable, strictly narrower than a
behavior change, fail-closed, and aligned with the already-shipped production
settlement contract; it isolates one falsifiable hypothesis without gate
relaxation, ceiling changes, forced SOTC, or prompt routing (D8 intact). Approval
is conditioned on these **required changes**, all consistent with D1–D8:

1. Extract a transport-agnostic settlement-core from `settle_user_eou` and call it
   from both the WS server and the probe (D1); reuse `function_cycle_active` and
   `response_audio_is_deliverable` verbatim.
2. Re-baseline the post-EOU perception/audio-clock bootstrap checks to the
   post-settlement state via an explicit schema change — never by silently
   redefining a schema-1 key (D4).
3. Bump to `schema = 2` and update the fail-closed host validator to require it,
   the new structural-check set, and the settlement block, with the three
   contradiction/anti-masquerade tests above (D6).
4. Enforce pre-EOU guards (no response-open, no deliverable audio, no public
   assistant/function text mutation, no effective FC) as per-step hard assertions
   that fail the case closed, with raw suppressed SOTC captured as allowed evidence
   (D3).
5. Gate any broader step on the D7 W8 single-token-form diagnostic at immutable
   thresholds; a positive result licenses only implementing the same shared contract
   in the protocol path, and never substitutes for full FP32+W8 corpus qualification
   as the sole green (D7/D8). If c014 does not stage/replay and recover tools,
   stop and instrument logits — do not force SOTC or route by prompt.

## Addendum 31 — Step 7A implementation review (2026-08-09)

Working-tree diff only; no source edited, nothing committed, no GPU run yet.
Verified every Addendum 30 condition against `server.py`,
`direct_semantic_probe.py`, `run_live_suite.py`, and the tests.

### Conditions verified

- **Shared controller used by both paths.** `UserEouSettlement`
  (`server.py:198`) is a transport-independent `@dataclass`; the caller owns the
  model step. It is used by the WS `settle_user_eou` for **both** typed
  (`:4278`) and microphone (`:4830`) commits and by the direct probe
  (`direct_semantic_probe.py:196`). The old WS-coupled loop is gone.
- **Exact 2× bound.** `max_model_steps = target*2`; `needs_step` raises
  `input_turn_settle_timeout` once `model_steps >= 2·target` *before* stepping,
  so at most `2·target` steps run. Unit + WS tests
  (`..._has_exact_twice_fence_bound`, `..._settlement_has_fatal_twice_fence_bound`).
- **Per-step validation before publish.** The WS caller passes
  `before_send=lambda r: settlement.observe(engine, r)`; both `before_send`
  sites (`:4018`, `:4077`) run before `send_step`/`send_json`, and `observe`
  raises before the send lock. The probe calls `observe` immediately after each
  step and before EOU. `..._rejects_response_before_publication` asserts the
  raise happens at the pre-send hook.
- **Suppressed raw SOTC allowed, effective FC rejected.** `observe` flags
  `function_cycle_active` (active/awaiting/injecting/forced/background) as a
  violation but records `pre_eou_suppressed_tokens` / `pre_eou_last_raw_token_id`
  / `deferred_eou_sotc_*` as allowed evidence.
  `..._allows_only_suppressed_raw_sotc` confirms a suppressed SOTC stays clean
  while `active=True` raises `pre_eou_function_activity`.
- **Schema 2 cannot accept schema 1.** Probe emits `schema: 2`
  (`direct_semantic_probe.py:365`); validator requires `schema == 2`
  (`run_live_suite.py:732`); `..._rejects_historical_schema_one` confirms a
  schema-1 r9-style report is rejected as settled evidence.
- **Settlement/step/boundary evidence is fail-closed and rederived.**
  `_validate_direct_settlement_evidence` rederives each step's `violations` from
  its own recorded state (`:258`), rederives `function_cycle_effective`
  (`:256`), `fence_reached`/`within_bound`/`pre_eou_clean` (`:270–275`),
  `ending_blank_frames` from the last step (`:268`), the exact failure code
  (`:287–299`), `first_settlement_pcm == steps[0]` (`:302`), and
  `performed == (failure is None)` (`:311`); and the case `structural_checks`
  are rederived from the raw settlement/boundary evidence (`:859–873`). Any
  mismatch raises. Contradiction tests cover hidden guard violations,
  contradictory counts, and boundary-bootstrap contradictions.
- **First bootstrap and EOU boundary separately rederived.**
  `first_settlement_pcm_initialized_perception` is rebaselined to the *first
  settlement* frame relative to the post-injection state
  (`audio_frame_after_direct+1`, `perception_after_direct+1`) — resolving the
  Addendum 30 mislabeling risk — and `eou_boundary_position_exactly_once` is
  measured at the single boundary step (`+1`). Both are rederived host-side.
- **EOU step counted exactly once.** The probe runs one `boundary_step` after
  `request_user_eou()` and reuses it at `frame_index == 0` of the response loop
  (`direct_semantic_probe.py:257–264`) — no double step.
- **Red settlement cases stay complete.** `UserEouSettlementFailure` is caught in
  the probe (`:206`), recorded as `settlement_failure`, and the case record is
  still built (structurally red); the lane completes all cases rather than
  crashing the corpus.
- **Tests.** Controller units (`test_realtime_web_server.py:93–318`) and WS
  integration (`test_realtime_websocket_endpoint.py:453–789`) exercise
  fence-only advance, 2× timeout, raw-SOTC-allowed/FC-rejected,
  pre-EOU-response-before-publication, pinned/disabled fence, and position-limit
  mid-settlement; host-validator tests cover schema masquerade and
  guard/count/boundary contradictions. I reran the four relevant modules: **147
  passed, 3 subtests**; Ruff clean on all three changed sources.

### Findings (all minor; none blocking)

- **F1 (minor; must validate on GPU).** The shared guard is **stricter** than the
  original WS `settle_user_eou`, which rejected only effective FC. `observe` now
  also fails-closed on `response_boundary_open`, `agent_control != "pad"`,
  deliverable audio, and assistant/function text mutation before EOU. This is
  more correct, but it is a behavior change on the production mic/typed paths: a
  real settlement step that legitimately left `agent_control != "pad"` would now
  fatally terminate a session that previously completed. External-EOU mode should
  hold the agent to PAD pre-EOU, so it should never trip — but with no GPU run
  yet, this must be confirmed on a real mic/typed session. Fail-closed (no
  laundering), so it is an availability check, not a correctness hole.
- **F2 (minor; reset-baseline coupling).** The probe's
  `eartts_bos_reset_exactly_once` / `eartts_held_during_settlement` use the
  *pre-injection* `reset_before` baseline, while the host rederivation uses the
  *post-injection* `settlement.eartts_reset_count_starting`. The contradiction
  check (`:866–873`) forces them to agree, which holds only if the TTS BOS reset
  never fires during direct injection. That invariant is very likely true but
  implicit; recommend an explicit `eartts_held_during_direct` check or asserting
  `reset_before == settlement.eartts_reset_count_starting`.
- **F3 (cosmetic).** `within_bound` is tautological — the controller structurally
  prevents exceeding `2·target`, so `settlement_within_bound` can never be false;
  a timeout is caught by `fence_reached=false` + `settlement_failure`, not by
  `within_bound`. Harmless, but the check carries no discriminating power.

Semantic/report consistency, reset accounting (split into held-during-settlement
plus exactly-once-at-BOS), and transcript/trace behavior (settlement steps still
publish via `send_step`; `observe` is a pre-send hook; `user_eou_settled` trace
carries the evidence) are all intact. No production regression found beyond the
F1 guard-strictness item to confirm on GPU.

### Verdict

**APPROVE.** The implementation faithfully and fail-closed satisfies every
Addendum 30 condition: one shared settlement controller for WebSocket and probe,
exact 2× bound, validate-before-publish, suppressed-raw-SOTC-allowed /
effective-FC-rejected, schema-2 anti-masquerade, and fully rederived
settlement/boundary/structural evidence with genuine contradiction tests. F1–F3
are minor; F1 (guard strictness) must be confirmed on the first GPU run but is a
correctness-preserving, fail-closed change, not a defect. The D7 W8
single-token-form diagnostic at immutable thresholds remains the behavioral gate
before any broader step, and none of D8's prohibitions are touched.

## Addendum 32 — F2 follow-up: EarTTS reset baseline closure (2026-08-09)

Narrow review of the F2 fix (Addendum 31); working-tree diff only, no GPU run.

### F2 is closed

The Addendum 31 finding was that the probe mixed two EarTTS-reset baselines — the
pre-injection `reset_before` for the BOS/settlement checks versus the
post-injection `settlement.eartts_reset_count_starting` in the host rederivation —
so the two were only *implicitly* required to coincide. The fix makes all three
phases share one explicit, rederived baseline:

- **Probe.** Records `reset_before` (pre-injection) and `reset_after_direct`
  (post-injection), surfaces both in `direct_state`
  (`eartts_reset_count_before`/`_after`), and adds an explicit
  `eartts_held_during_direct = reset_after_direct == reset_before` structural
  check. The settlement check is rebaselined to
  `reset_after_settlement == reset_after_direct` and the BOS check to
  `boundary.eartts_reset_count_after == reset_after_direct + 1`
  (`direct_semantic_probe.py:295, 311–312, 320–323`).
- **Validator.** `direct_state` is validated as exactly five integer keys
  (`run_live_suite.py:839–846`), and all three phases are rederived from the
  single `direct_state.eartts_reset_count_after` baseline: held-during-direct
  (`:861–864`), held-during-settlement (`:878–884`), and BOS-exactly-once
  (`:892–895`). Critically, `:883–884` now cross-asserts
  `settlement.eartts_reset_count_starting == direct_state.eartts_reset_count_after`
  — converting the previously implicit coupling into an explicit, fail-closed
  equality. All three derived values are compared against the report's
  `structural_checks` (`:898–899`); any mismatch raises.
- **Contradiction test.** `test_direct_semantic_probe_report_rejects_hidden_direct_eartts_reset`
  flips `direct_state.eartts_reset_count_after` from `0`→`1` on an otherwise-green
  report; the rederived `eartts_held_during_direct` (and the settlement/BOS
  cross-checks) then contradict the unchanged structural verdicts, and the
  validator raises "contradictory settlement structural checks." Genuine.

A hidden EarTTS BOS reset during direct injection — the exact hole F2 named — is
now caught in three independent places (the explicit direct check, the settlement
baseline equality, and the BOS-count rederivation).

### Exact structural set and fixture remain coherent

The structural-check constant grows to **20** keys (adds
`eartts_held_during_direct`). I confirmed the probe emits exactly that set: the 13
inline keys plus the 7 `_transaction_checks` keys spread via `**transaction`
equal `DIRECT_SEMANTIC_STRUCTURAL_CHECKS` (set-equal, no missing/extra), enforced
at runtime by the validator's exact-set check (`:793`). Per-step function/SOTC
evidence is strictly typed — FC flags as bools, `forced_tokens` as int ≥ 0, the
`function_calling` key set exact, and the SOTC subset validated via
`_validate_sotc_evidence` (`:224–240`). The green fixture is internally consistent
(`before=0`, `after=0`, `starting=0`, `boundary=1` → all three EarTTS derived
checks true). Focused probe/evaluator suite **42 passed** on my rerun; Ruff clean —
consistent with the reported 139 focused / 439 full + 3 subtests.

### Verdict

**APPROVE.** F2 is closed: the EarTTS reset is now accounted across all three
phases from one rederived, cross-asserted baseline, with a genuine contradiction
test; the 20-key structural set and report fixture are coherent and fail-closed.
No new issues. The residual items are unchanged from Addendum 31: **F1**
(shared-guard strictness) still must be confirmed on the first GPU run, and **F3**
(`within_bound` tautology) remains cosmetic. The D7 W8 single-token-form
diagnostic at immutable thresholds remains the behavioral gate; no D8 prohibition
is touched.

## Addendum 33 — pre-implementation review: Step 7B function-logit trace (2026-08-09)

Design review of the diagnostic-only function-logit trace, against the vLLM
custom-output surfaces (`model_factory.py`, `runtime_optimizations.py`,
`streaming_llm_engine.py`), the wrapper's special-token resolution, and the r10
evidence. No source changed.

### The r10 result frames the question correctly

r10 (schema-2, W8, text-only) is a clean 20/20 settled run: every case reached the
10-blank fence (11 model steps incl. the first-PCM bootstrap), zero settlement
failures, zero pre-EOU violations — and still 10/20 semantic, **zero staged/replayed
SOTC, zero tool calls**. c014 states it will call `get_weather` while the function
channel stays empty. Settlement is therefore *not* the missing ingredient; the
function channel simply never predicts SOTC. A read-only logit trace at the SOTC
decision is the right next probe, and strictly narrower than any behavior change.

### The pivotal premise checks out

`function_logits` **is** materialized on the vLLM path. The checkpoint declares
`custom_outputs = ["text_logits","function_tokens","function_logits"]`
(convert script), and the production optimization *requires* `function_tokens` and
`function_logits` be retained while `text_logits` is skipped (`server.py:646`), under
a greedy contract (`temperature=0, top_p=1, repetition_penalty=1`). The engine
passes `output.outputs[0].custom_outputs` through verbatim
(`streaming_llm_engine.py:310/318`), and `slice_position_outputs` already selects the
final accepted position preserving a length-one axis. So a `[1, vocab]`
`function_logits` tensor is available at the exact position the design wants —
**but note the vLLM `__call__` reads only `function_tokens`/`text_logits` today**
(`model_factory.py:553/582`), so the trace must assert `function_logits` is present
and **fail closed** if a future config drops it, never emit an empty trace silently.

Because the function head is greedy, `ans["function_predicted_token"]` comes from
`function_tokens` (`:582–583`), i.e. the on-device argmax. That yields a free,
mandatory consistency check: **`argmax(function_logits)` must equal the effective
`function_tokens`**; divergence means the traced logits are the wrong
position/pre-penalty and the trace is invalid.

### Safety of shape/ownership and rank/perf

Safe as scoped, with conditions. The summary must be computed from the sliced
final-position tensor, read-only, *after* the token decision, and reduced to Python
scalars/small lists before it leaves the forward — no torch tensor may be stored in
`ans['function_logit_trace']` (a retained `[1,vocab]` CUDA tensor across 20 cases ×
many frames risks a real leak/OOM). Rank = `(logits > logits[sotc_id]).sum()` plus a
`topk(k≤20)` and one `argmax` is a trivial kernel even at full vocab; the only cost
is a device→host sync to `.item()` the scalars. That is acceptable **only because it
is opt-in (`S2S_FUNCTION_LOGIT_TRACE_TOPK` default 0/off, bounded 1..20) and
final-position-only**: every added line — including any change to which custom
outputs are requested — must sit behind the enable guard so the disabled path is
byte-for-byte the production path. Tracing must not alter the vLLM custom-output
request contract mid-run (it only reads an already-retained output), and must not
feed the summary back into sampling.

### PAD/SOTC are not quite enough

For "why no SOTC" the core signal is SOTC-vs-PAD margin, SOTC rank, argmax, and
top-k IDs/logits — the top-k is what reveals the "_weather Tokyo"-style fragment
tokens winning instead of SOTC. But scalars for **EOTC, EOTR, and text/agent BOS**
are near-free extra gathers and make the trace authoritative across the later phases
(post-SOTC tool-call and tool-response boundaries) without a second GPU build. All
gathered IDs must come from the wrapper's **tokenizer-resolved** ids
(`_fc_sotc_id`, `_fc_eotc_id`, `_fc_eotr_id`, `text_pad_id`, `text_bos_id`;
`inference_wrapper.py:964–981`), never the hardcoded `20/12/…` defaults, and the
trace must fail closed if any required id is unresolved.

### Phase labeling must be caller-owned, not inferred

The authoritative-labeling requirement is the sharpest risk. Phase
(`direct_position` / `settlement` / `eou_boundary` / `post_bos[i]`) must be assigned
from the **caller's ground truth of what it is doing**, keyed by **absolute model
frame**, never inferred from model state (e.g. "SOTC present ⇒ boundary"). The probe
already owns exact frame ranges — direct-injection frames, settlement step frames
(with the known first-PCM-bootstrap off-by-one that made r10 report 11 steps), the
single EOU-boundary frame, and post-BOS response indices — so the trace entry records
the absolute model frame and the caller joins frame→phase from its own log. The
schema must **fail closed** if a traced frame maps to zero or to more than one phase.

### Required contradiction/fidelity tests

1. **Determinism (most important):** the same case with trace **off vs on** must be
   identical in every non-trace field (tokens, assistant/function text, tool calls,
   semantic, structural, settlement). This is the direct proof of "no behavior
   change."
2. `argmax(function_logits)` == effective `function_tokens` per traced position.
3. `topk` length == requested k; top-k logits non-increasing; `argmax` == top-k[0]
   id; SOTC rank consistent with top-k membership (in top-k ⇒ rank matches index;
   rank>k ⇒ absent from top-k).
4. All logit scalars finite (reject NaN/inf).
5. Trace requested (k>0) ⇒ a trace entry exists for **every** position of every
   labeled phase; k==0 ⇒ no trace block and the schema-2 report still validates
   (trace is additive/opt-in).
6. `k` outside 1..20 rejected; unresolved special-token id rejected.
7. Phase label in the fixed enum, and each traced absolute frame covered by exactly
   one caller-declared phase range (no ambiguous inference).
8. Trace payload contains only JSON scalars/small lists — reject any tensor/object.
9. raw-vs-effective function token: if they differ, a recorded suppression must
   explain it; if equal, no suppression may be claimed.

### D8

The design is read-only and opt-in: no forced tokens, no prompt routing, no
threshold/gate relaxation, no sampling or token behavior change. The first GPU run
(targeted c014 direct-text plus a matching working Pocket/voice reference, immutable
weights/prompts/thresholds) is a diagnostic, not a qualification; it cannot turn any
lane green.

### Verdict

**APPROVE** — the trace is a safe, correctly-scoped, read-only diagnostic and its
enabling premise (retained `function_logits` at the final position under a greedy
head) is confirmed. Approval is conditioned on these **blockers**, all implementable
within the proposed design:

- **B1.** Assert `function_logits` present when tracing is requested and fail closed
  otherwise; never emit an empty/partial trace silently.
- **B2.** Gate *every* added operation behind the disabled-by-default env so the
  off-path is byte-identical to production; store only JSON scalars/small lists in
  `ans['function_logit_trace']`, never a tensor, and never feed the summary into
  sampling.
- **B3.** Include the `argmax(function_logits) == function_tokens` consistency check
  as a hard, tested invariant.
- **B4.** Gather SOTC/PAD/EOTC/EOTR/text-BOS scalar logits from tokenizer-resolved
  ids; fail closed on any unresolved id.
- **B5.** Phase labels caller-owned and keyed by absolute model frame; schema fails
  closed on zero/multiple phase coverage — no model-state inference.
- **B6.** Ship the determinism paired-run (trace off vs on identical) plus the
  contradiction tests enumerated above; schema/validator fail closed when tracing is
  requested.
- **B7.** First GPU run stays a diagnostic under immutable weights/prompts/thresholds
  and D8; if SOTC never ranks competitively, report it and stop — do not force SOTC,
  route by prompt, or relax any gate.

## Addendum 34 — Step 7B implementation review against B1–B7 (2026-08-09)

Inspected the DGX project diff, the Speech checkout diff, and the retained patch;
no source edited. The implementation is well-built and mostly satisfies B1–B7 with
strong fail-closed validation, but two items block trusting the first GPU
diagnostic.

### What holds

- **B2 off-path byte identity.** The entire trace lives behind
  `if self._function_logit_trace_topk:` in `VllmLLMModel.__call__`
  (`model_factory.py:683–695`); disabled, nothing is read or added to `ans`, and
  the constructor never changes the vLLM custom-output request contract — it only
  reads the already-retained `function_logits`. The disabled forward is identical
  to production.
- **B1 fail-closed premise.** When enabled, the attach asserts `function_logits`
  and `function_predicted_token` are present and raises otherwise (`:685–692`);
  `_summarize_function_logits` requires an exact `[1, vocab]` tensor and rejects
  any other shape (`:409–412`) — so a wrong-position or PAD-pair-shaped tensor
  fails closed rather than mislabels.
- **B3 greedy invariant.** The summary raises unless `argmax(function_logits)`
  equals the `function_tokens` custom output (`:429–431`), and the host validator
  re-enforces `head_argmax_id == predicted_token_id == raw_function_token_id`
  (`run_live_suite.py:227–232`). Tested.
- **B4 finiteness/ids.** Full-row `isfinite` (`:419`), token-id range checks, and
  tokenizer-resolved watched ids (`_fc_sotc_id`/`_fc_eotc_id`/`_fc_eotr_id`/
  `text_pad_id`/`text_bos_id`, `inference_wrapper.py:964–981`, wired at
  `:1035`); the validator re-checks finite watched logits and the exact resolved
  ids. JSON-only payload (`.cpu().tolist()`/`float`/`int`), no tensor retained.
- **B5 phase authority.** Phases are caller-owned in the probe
  (`direct_position`/`settlement`/`eou_boundary`/`post_bos`), and
  `_append_function_logit_trace` cross-checks each entry's `model_frame` against
  the caller's expected frame fail-closed (`direct_semantic_probe.py:19–39`). The
  validator rederives the required prefix from the case's own counts, accepts a
  **variable** post-BOS length (correct early-response-end coverage,
  `run_live_suite.py:185–196`), enforces exact phase ordering, contiguous unique
  model frames (`:293–295`), and the raw≠effective ⇒ non-`model` reason XOR
  (`:241–248`). Reason enum in the wrapper is complete
  (`model`/`direct_hidden_pad`/`pre_eou_guard`/`boundary_sotc_replay`/
  `function_state_machine`) and attached after all function mutations
  (`inference_wrapper.py:2905–2926`).
- **Schema gate.** schema-3 iff tracing, schema-2 emits no trace field (top-level
  and per-case, `:933–935/1060–1061`); trace config shape/bounds strict.
- **B7 patch equality.** The retained patch **reverse-applies clean** against the
  live Speech checkout (includes the new trace code). Focused/host + evaluator
  trace tests pass on my rerun (7 trace tests green).

### Blockers

- **BL1 — the B6 determinism paired-run is not wired.**
  `validate_function_logit_trace_determinism` (`run_live_suite.py:1154`) is
  strict and correct (full deep-equality after stripping the trace and
  normalizing `audio_path` to a basename), but it has **no non-test caller** —
  grep finds it only in unit tests. No orchestration runs c014 both untraced
  (schema 2) and traced (schema 3) and invokes it, so "no behavior change" is
  proven only by the guard and unit tests, never on hardware. B6 explicitly
  required shipping the paired-run; the first GPU diagnostic must run both,
  invoke the comparator, and fail closed if the comparison is absent. Until wired,
  B6 is incomplete.
- **BL2 — SOTC-rank tie handling can spuriously reject on W8.** `sotc_rank` uses a
  strict-greater count (`model_factory.py:435`), but the validator requires the
  exact `ranked_ids.index(sotc)+1 == sotc_rank` (`run_live_suite.py:288`) with no
  tolerance for ties. When `sotc_equal_count > 1` and SOTC is tied with other
  top-k tokens, `torch.topk`'s arbitrary tie order makes the index exceed the
  strict rank, raising "contradictory SOTC rank evidence." Exact logit ties are
  more plausible under the W8 quantized head this diagnostic targets, so this
  fail-closed check could spuriously red the very case whose SOTC rank we want to
  observe. Relax the index check to `[sotc_rank, sotc_rank + sotc_equal_count - 1]`
  when `sotc_equal_count > 1`. (Fail-closed — cannot launder green — but it can
  defeat the run.)

### Required tests before the GPU run (currently missing)

The parametrized contradiction test covers greedy mismatch, NaN watched logit,
phase-position ambiguity, and unexplained raw/effective. Add: (a) attach-site
fail-closed when `function_logits` is absent while tracing (B1 is untested at the
model boundary); (b) a contiguous-`model_frame` violation rejection (B5); (c) an
early-response-end case with `post_count < post_bos_frames` (B5 coverage path is
implemented but unexercised); (d) a multi-way-tie SOTC case with `sotc` past its
strict rank in top-k (guards BL2's fix).

### Minor

`effective_token_reason` is only checked to be a non-`model` string when tokens
differ; it is not pinned to the closed enum. A membership check against
{`model`,`direct_hidden_pad`,`pre_eou_guard`,`boundary_sotc_replay`,
`function_state_machine`} would harden it. Not blocking.

### Verdict

**REQUEST CHANGES.** The trace is a safe, correctly-scoped, read-only diagnostic:
off-path is byte-identical, the reduction is JSON-only and fail-closed, the greedy
and phase/frame invariants are enforced on both sides, and the retained patch
matches the checkout. But two blockers must close before the first GPU diagnostic
is trusted: **BL1** — wire and fail-close the B6 determinism paired-run into the
actual traced-vs-untraced GPU flow (it is currently only a unit-tested library
function); **BL2** — fix the SOTC-rank tie handling so a W8 exact tie cannot
spuriously reject the case. Add the four missing tests above with the fixes.
D8 remains intact (read-only, opt-in, no forced tokens/routing/gate relaxation);
none of the concerns weaken any qualification gate.

## Addendum 35 — Step 7B re-review: Addendum 34 changes verified (2026-08-09)

Re-inspected the current DGX and Speech diffs and the regenerated retained patch
(A34 line references are stale; re-derived against current source). Both A34
blockers, all four missing tests, and the minor are closed.

### BL1 — determinism paired-run wired (closed)

`run_direct_function_logit_diagnostic` (`run_live_suite.py:1608`) is a real
non-test caller: it creates a fresh output dir, runs the W8 text-only probe twice
in sequence — `("off", 0)` then `("on", top_k)` — validates each report through
`run_direct_semantic_probe_stage` with the matching `expected_trace_top_k` (off ⇒
schema-2/no-trace, on ⇒ schema-3/trace), and **only then** invokes
`validate_function_logit_trace_determinism(reports["off"], reports["on"])`
(`:1646`) before writing `passed: True` (`:1650`). If either probe crashes, a
report fails validation, or the comparator raises, no green is written. CLI flags
`--direct-function-logit-diagnostic` / `--function-logit-trace-top-k` /
`--diagnostic-speech-root` are wired (`:2016–2018`); the command builder mounts
distinct off/on dirs, sets `S2S_FUNCTION_LOGIT_TRACE_TOPK` only on the "on" run
(`:842–843`), and bind-mounts the reviewed Speech checkout at `/opt/Speech:ro`
(`:877–880`). `test_function_logit_diagnostic_orchestrates_paired_runs` asserts the
exact `[("off",0),("on",2)]` ordering and lets the **real** comparator run.

### BL2 — SOTC-rank tie tolerance (closed)

The validator now accepts a tied SOTC anywhere in its tie block:
`sotc_rank <= position < sotc_rank + sotc_equal_count` (`:312–316`), permits
omission from top-k only when the whole block fits
(`sotc_rank + sotc_equal_count - 1 <= top_k` ⇒ must be present, `:318`) so a block
that straddles the top-k boundary may legitimately omit SOTC, and bounds the
interval by vocab (`sotc_rank + sotc_equal_count - 1 > vocab_size` rejected,
`:237`). A new watched-vs-top-k logit cross-check (`:303–308`) closes a
consistency gap. `test_..._accepts_sotc_tie_at_top_k_boundary` places SOTC at
top-k index 2 with strict rank 1 / equal-count 2 and validates.

### Missing tests (closed)

- **Attach-site fail-closed:** the model attach is now a named
  `_attach_function_logit_trace` (`model_factory.py:460`) that raises on missing
  `function_logits` (`:471`) or `function_tokens` (`:475`); the installed
  immutable-container suite exercises it (reported 4/4).
- **Frame discontinuity:** the parametrized contradiction set now mutates
  `model_frame += 2` ⇒ "duplicate or discontinuous trace frames".
- **Early response end:** `test_..._accepts_early_end_before_post_bos_bound` pops
  a `post_bos` entry and validates the shorter contiguous prefix (variable
  post-BOS length path exercised).
- **Multi-way tie:** covered by the BL2 test above.

### Minor exceeded

`effective_token_reason` is now a closed enum (`:242–248`) **plus** phase and
effective-token cross-constraints: `direct_hidden_pad` only in `direct_position`
and `boundary_sotc_replay` only in `post_bos`; the pad-forcing reasons must yield
the PAD id and `boundary_sotc_replay` must yield the SOTC id (`:250–265`) — a
richer fail-closed invariant than requested.

### Independent verification

Retained patch **reverse-applies clean** against the current Speech checkout and
carries the new trace surfaces; focused trace/diagnostic/tie/early/discontinuity
tests pass (12/12 on my rerun), Ruff clean — consistent with the reported 143
focused / 4-of-4 container. B1–B5/B7 remain satisfied as in Addendum 34, now with
the two blockers and the enum minor resolved.

Two non-blocking notes: (a) the GPU diagnostic is wired but **not yet run** —
executing it (with the determinism proof it now performs) is the remaining
forward step; (b) the model-side trace code runs from the bind-mounted reviewed
Speech checkout for this diagnostic, which is correct for an opt-in/off,
diagnostic-only path under D8 — it must never be relied on as baked qualification
code.

### Verdict

**APPROVE.** BL1 and BL2 are genuinely closed — the determinism paired-run is a
wired, fail-closed non-test caller gating `passed`, and the SOTC-rank tie
tolerance is correct and vocab-bounded — the four missing tests are added, the
reason enum minor is exceeded, and the retained patch is exact. The trace remains
a safe, opt-in, read-only diagnostic with no D8 prohibition touched. The only
remaining step is running the GPU diagnostic itself; if SOTC never ranks
competitively, report and stop — no forced SOTC, prompt routing, or gate
relaxation.

## Addendum 36 — Step 7B plumbing fix: Git safe.directory for the bind-mounted Speech checkout (2026-08-09)

Narrow review of the dubious-ownership fix in `direct_semantic_probe_command`; no
source edited.

### What the fix does and why it is correct

The GPU diagnostic bind-mounts the reviewed Speech checkout read-only at
`/opt/Speech`, and `checked_speech_root` runs `git rev-parse`/`git diff` there at
startup to prove the running code matches the reviewed tree. Because the mount is
owned by a different uid than the in-container process, Git raises
"detected dubious ownership" before model startup. The fix passes three
container-local env vars — `GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0=safe.directory`,
`GIT_CONFIG_VALUE_0=/opt/Speech` (`run_live_suite.py:884–888`) — the canonical
ephemeral, in-process equivalent of `git config --global --add safe.directory
/opt/Speech`. It writes nothing to disk, mutates no host/global config, and is
scoped to the one container that runs the checks.

### Adversarial checks

- **Scope is exactly the diagnostic path.** The env is added only inside
  `if diagnostic_speech_root is not None:`, alongside the `/opt/Speech:ro` mount
  (`:877–890`). I confirmed `GIT_CONFIG_COUNT` occurs exactly once in the builder
  and only within that guard. Production/off runs (no `diagnostic_speech_root`)
  are byte-unchanged.
- **No integrity weakening.** `safe.directory` is set to the exact path
  `/opt/Speech`, not a blanket `*`, so only the operator's own reviewed,
  read-only checkout is trusted. This does not bypass any qualification or
  provenance check — it lets the provenance git checks run at all. The checkout
  is `:ro`, so Git cannot write to it. If the value ever failed to match (e.g. a
  differently-normalized repo path), the original dubious-ownership error would
  re-surface visibly — the failure mode is fail-closed, not a silent wrong result.
- **Determinism proof stays valid.** Both the off and on diagnostic probes are
  built from the same `args`, so both receive the identical `/opt/Speech` mount
  and the same three GIT_CONFIG vars; the only off/on difference remains
  `S2S_FUNCTION_LOGIT_TRACE_TOPK`. The Git config affects only the startup
  provenance checks, never inference, so `validate_function_logit_trace_determinism`
  is unaffected.
- **It reaches the git calls.** Docker `-e` puts the vars in the container
  process environment, which the in-container Python and its `git` subprocesses
  inherit; the checks run at startup, after the env is already set.
- **Test.** `test_direct_semantic_probe_commands_pin_corpus_image_and_precision`
  asserts all three GIT_CONFIG entries and the `/opt/Speech:ro` mount are present
  on the diagnostic command, and separately builds fp32/w8 commands without the
  diagnostic root. Passes on my rerun.

### Minor (non-blocking)

The test verifies presence on the diagnostic command but does not assert
**absence** of `GIT_CONFIG_COUNT` from the non-diagnostic fp32/w8 commands. The
guard makes that true by construction, but an explicit
`assert "GIT_CONFIG_COUNT=1" not in w8_command` would pin the off-path-unchanged
guarantee against future edits.

### Verdict

**APPROVE.** The fix is minimal, correctly scoped to the diagnostic bind-mount,
does not touch inference, gates, provenance integrity, or the determinism proof,
and leaves production/off paths byte-unchanged; `safe.directory` is whitelisted to
the exact read-only reviewed checkout, not broadly. No blocker. Optional
hardening: add the negative assertion above. D8 remains intact, and the GPU
diagnostic can proceed.

## Addendum 37 — Step 7B plumbing fix: PYTHONPATH so the diagnostic runs the reviewed code (2026-08-09)

Narrow review of the import-precedence fix; no source edited.

### The failure and the fix

On the second launch, provenance ran but `python -m
nemotron_voicechat_runtime.server` imported the **image's older installed**
runtime, whose `checked_speech_root` compared the mounted current `/opt/Speech`
diff against the **image's older retained patch** — a mismatch, because 7B
regenerated the patch. The fix adds, inside the same `diagnostic_speech_root`
block, `PYTHONPATH=/workspace/project/src:/opt/Speech`
(`run_live_suite.py:890`) so the interpreter imports the current project runtime
and the reviewed Speech checkout ahead of the image.

### Assessment

- **Module precedence.** `PYTHONPATH` entries are prepended to `sys.path` before
  site-packages, in order: `/workspace/project/src` (the read-only current
  project, mounted at `:862`) then `/opt/Speech` (the reviewed checkout). So
  `python -m nemotron_voicechat_runtime.server` resolves to the current runtime
  and `import nemo…` resolves to the reviewed checkout (which carries the 7B trace
  code); other deps still come from the image's site-packages, which is correct.
  This is an already-established idiom — the ASR command builder uses the same
  `PYTHONPATH=/workspace/project/src` against a read-only project mount
  (`:710`). The only theoretical shadow would be a conflicting top-level package
  in the container WORKDIR, which a standard runtime image does not have.
- **Provenance (the crux).** `checked_speech_root` locates the retained patch
  **`__file__`-relative**: `Path(__file__).resolve().parent /
  "patches/nemotron-voicechat-rnnt-turn-taking.patch"` (`server.py:2318`; the
  other provenance roots at `:773`/`:2954` are likewise `__file__`-relative).
  Because the import now comes from `/workspace/project/src`, `__file__` points
  into the mount, so the **current** patch is read from the mount and compared
  against the **current** `/opt/Speech` — the two the review already found
  byte-consistent (Addendum 35 reverse-apply clean). The image's stale baked
  patch is no longer consulted. Redirecting the import thus fixes the code *and*
  the patch read atomically — exactly the reported defect.
- **Off/on determinism.** Both diagnostic probes are built from the same `args`,
  so both receive the identical `PYTHONPATH` (and mount and Git config); the only
  off/on difference remains `S2S_FUNCTION_LOGIT_TRACE_TOPK`. The determinism
  comparator stays apples-to-apples.
- **Production isolation.** `PYTHONPATH` is added only in the
  `diagnostic_speech_root` block (single occurrence in this builder, guard-scoped);
  production/off/image paths never receive it. The test now asserts **absence** of
  both `PYTHONPATH` and `GIT_CONFIG_COUNT` from the fp32 and w8 commands
  (`test_live_suite.py:135–138`) and **presence** on the diagnostic command
  (`:143–146`), closing the Addendum 36 negative-assertion gap. Passes on my rerun.

### Non-blocking note

By design the diagnostic now runs the **mounted current** runtime + Speech, not
the baked image runtime — the correct contract for validating reviewed code before
it is baked. A subsequent qualification must bake this exact code and re-run
without the mounts/PYTHONPATH; the diagnostic result attests to the reviewed tree,
not to the current image's installed runtime.

### Verdict

**APPROVE.** The fix correctly redirects import precedence to the current project
runtime and reviewed Speech; because the retained patch is `__file__`-relative,
provenance now compares the current patch against the current checkout, resolving
the stale-patch mismatch. Off/on determinism and production isolation are
preserved and now positively tested in both directions. No blocker; D8 intact and
the GPU diagnostic can proceed.

## Addendum 38 — Step 7B fix: per-model-type trace parsing (2026-08-09)

Narrow review of the fix for the traced launch failing closed because the global
`S2S_FUNCTION_LOGIT_TRACE_TOPK` was parsed by **both** vLLM interfaces and EarTTS
has no function/watched token IDs. No source edited.

### The fix

`VllmLLMModel._parse_function_logit_trace_topk` now takes `model_type`
(`model_factory.py:388`): it validates the env as an integer in `0..20` for
**every** type first (`:389–398`), returns **0** for `eartts` (`:399–400`),
**rejects** any nonzero request for a non-`llm` type (`:401–404`), and returns the
requested value for `llm` (`:405`). The constructor passes its own `model_type`
(`:360–361`).

### Is `model_type` trustworthy? Yes.

It is not a new cosmetic flag — it is the pre-existing, load-bearing discriminator
the factory already uses to pick the vLLM engine and checkpoint: `engine_type =
model_type` and the converted-checkpoint path `…_vllm_converted_{model_type}`
(`:302/:314`). The factory sets it deterministically from the engine-type
dispatch: the `vllm_eartts` branch constructs `VllmEARTTSModel(…,
model_type="eartts", …)` (`:1279–1281`) and the general `vllm*` branch constructs
`VllmLLMModel(…, model_type="llm", …)` (`:1293–1295`). Because the same value
selects which model actually loads, a wrong `model_type` would load the wrong
engine/checkpoint and fail at startup — so a correct value is guaranteed by the
model working at all. The eartts disable branch therefore fires exactly when the
EarTTS engine is the one running.

### Could any trace silently disable on Nano? No.

The only silent-disable branch is `model_type == "eartts" → 0`. Nano is always
constructed as `model_type="llm"`, which returns the requested top-k unchanged;
for Nano to be silently disabled it would have to be built as `eartts`, which
would load the EarTTS engine/checkpoint — a loud, catastrophic wrong-model load,
not a silent condition. And when Nano tracing is enabled, the constructor still
fail-closes on missing watched IDs/logits (Addendum 34), so Nano either traces
correctly or raises — never silently off. The global env is now validated for
range on all types (a malformed value fails closed everywhere, including EarTTS),
and an unsupported nonzero type is rejected rather than mis-traced. The
installed-container test (`eartts 10→0`, `llm 10→10`, unknown-nonzero rejects)
maps directly onto these three branches.

### Verification

Retained patch reverse-applies clean against the current Speech checkout. The
parse branches match the reported container test exactly.

### Minor (non-blocking)

`VllmEARTTSModel.__init__` does not self-pin `model_type="eartts"`; it relies on
the sole factory caller. A defensive hardcode or assert there would make EarTTS
robust against a future mis-constructed caller, but is belt-and-suspenders given
`model_type` is load-bearing and factory-set.

### Verdict

**APPROVE.** `model_type` is trustworthy by construction (load-bearing,
factory-assigned, operationally validated), the EarTTS disable is scoped precisely
to the EarTTS engine, Nano remains fully fail-closed with no silent-disable path,
and the global env is validated for all types and rejects unsupported nonzero
requests. Retained patch is exact. No blocker; D8 intact and the traced GPU
diagnostic can proceed.

## Addendum 39 — r15 GPU evidence + determinism-gate redesign (2026-08-09)

Reviewed the r15 artifacts directly
(`direct-function-logits-r15-full/{off,on,report}.json`) and the r14 control. The
comparator RED is a **gate-design false alarm**; the c014 diagnostic finding is
**valid and robust**. The gate needs a real redesign — and my independent reading
shows it must go further than the minimal proposal.

### Independently verified

- **Bug 1 (confirmed, worse than stated).** The `on` report carries **325**
  `function_logit_trace` keys at all depths (embedded in every StepResult `model`
  dict plus per-case plus the top-level config); `off` carries 0. The A34/35
  comparator stripped only the top-level and per-case trace, so hundreds of
  positions differed spuriously. A **recursive** strip is mandatory.
- **Whole-report / cross-run deep equality is an invalid null (confirmed).** After
  a recursive strip, the two **untraced** runs r14-off vs r15-off still differ
  discretely in **c009, c010, c016, c017** (assistant_text / semantic /
  function_text) and in audio length across most cases. The W8 greedy path is not
  run-to-run deterministic, so a RED from full-report equality proves nothing about
  tracing.
- **c014 finding (confirmed, large-margin).** Traced c014: assistant "I will use
  get underscore weather to check the weather in Tokyo.", empty `function_text` and
  `tool_calls`; **SOTC rank 2 on all 11 settlement frames and the EOU boundary**,
  with PAD the argmax throughout (settlement PAD 18.375 vs SOTC 13.8125; EOU PAD
  16.75 vs SOTC 13.875), direct-phase best rank 6, post-BOS best 19. The greedy
  function head emits PAD, so no tool call — the mechanism behind "says get_weather,
  channel empty." The ~4.5-logit margin is far beyond anything a read-only,
  post-decision trace (proven read-only across Addenda 33–38) could move, and the
  empty-tool result holds in the `off` run too. The finding does **not** depend on
  the gate.

### Two residuals the c014-only framing misses

After the recursive strip, r15 **off-vs-on** still has discrete diffs beyond audio:
- **c017** (assistant_text + semantic) — but c017 is discretely nondeterministic in
  the untraced control too, so this is inherent nondeterminism, not tracing.
- **c005** — `raw_text_tokens [2210] vs [2586]` at one direct-injection position.
  c005 is **stable** across the cross-session control (r14-off == r15-off) yet
  **differs** within the r15 off/on pair. It cannot be attributed to nondeterminism
  or to tracing from the evidence on hand.

And **c014's own `direct_positions` differs across sessions** (r14-off vs r15-off)
while matching within the r15 off/on pair. So c014 is not perfectly discretely
deterministic — its within-session cleanliness may be luck. These are exactly the
cases a naive "strip, select c014, exact-compare" gate would mis-handle: it would
false-RED if c014's cross-session nondeterminism fired, false-GREEN if it didn't,
and never even look at c005.

### Why the minimal proposal is insufficient

The proposal is directionally right (recursive strip; drop whole-report equality;
classify audio separately) but its "observed untraced-control envelope" is the
**cross-session r14 baseline**, which is unsound: r14→r15 conflates build/session
drift (e.g. c014 `direct_positions` differs cross-session but not within-session;
c005 the reverse). You cannot separate trace-perturbation from nondeterminism with
one untraced run plus a different-build control.

### Concrete revised gate (fail-closed)

1. **Recursive strip** every `function_logit_trace` key at all depths before any
   comparison.
2. **Same-session control — ABA is required.** Run ≥2 untraced probes and ≥1 traced
   probe in the *same image/session* (A₁, A₂ untraced; B traced; ABA order is fine).
   The two untraced runs establish the contemporaneous null. Never use a
   cross-build report (r14) as the envelope; full-corpus structural validation of
   each report still runs.
3. **Per-field, per-case classification against the same-session null** — not
   whole-report, not c014-only:
   - Fields where A₁==A₂ (control-stable) → B must equal them exactly; any change ⇒
     RED.
   - Fields where A₁≠A₂ (control-nondeterministic) → B must fall within the observed
     control set/range, and each such case must be **logged**, never silently
     passed.
   - Decoded-audio tail → `|B−A₁|` in whole 1764-sample (80 ms) frames must be
     ≤ the untraced `|A₁−A₂|` frame delta; classify as "audio-tail within envelope,"
     never require exact audio equality.
4. **c014 interpretation assertion** (in addition, not instead): c014's discrete
   text/tool result and its SOTC-rank/argmax/logit-margin trace must be identical
   across A₁, A₂, and B; if c014 is discretely unstable in the same-session control,
   the gate must widen sampling or abstain — never green.
5. **Fail-closed defaults:** any control-stable field changed by B, or any B field
   outside the same-session envelope (e.g. an unexplained c005-style
   `raw_text_tokens` divergence), or a full-report/cross-build equality shortcut ⇒
   RED. Keep the per-report structural + trace-invariant validation (which passed).

A single off/on pair fundamentally cannot certify neutrality on a nondeterministic
model; ABA (≥2 same-session untraced) is the minimum, and even then the null is a
finite sample — the gate certifies "B introduces no divergence beyond the observed
same-session null," with residual discrete diffs enumerated, not hidden.

### Verdict

**REQUEST CHANGES on the determinism gate; the r15 interpretation and c014 finding
are APPROVED.** The comparator RED is an artifact of (Bug 1) an incomplete strip and
(design) an invalid whole-report/cross-build equality premise on a model that is not
run-deterministic. Rebuild the gate as the fail-closed, same-session ABA,
per-field-envelope design above — recursive strip, ≥2 same-session untraced
controls, per-case classification, separate quantized-audio-tail bounding, explicit
c014 assertion, and enumeration (not suppression) of residuals like c005. The
diagnostic conclusion stands independently: on the direct-text path SOTC is
consistently rank 2 but PAD is argmax by a large margin, so the function channel
emits PAD and no tool call fires. D8 intact — no forced SOTC, prompt routing, gate
or threshold relaxation.

## Addendum 40 — ABA neutrality-gate implementation review (2026-08-09)

Reviewed the implemented Addendum 39 gate; no source edited. It faithfully and
fail-closed realizes the revised design.

### Verified against the concerns

- **Laundering.** `_normalize_function_logit_report` deep-copies, then strips only
  non-behavioral evidence: every `function_logit_trace` at all depths (recursive
  `visit`, counted), `audio_path`→basename, and the two process-volatile
  cpu-codec fields (`warmup_worker_ms`, `worker.pid`) at a fixed path. None can
  hide a behavior difference — any real divergence surfaces in the compared
  tokens/text/state/audio-length. It also enforces schema-3 ⇒ ≥2 nested trace keys
  and schema-2 ⇒ 0, so a trace-stripped-empty or trace-contaminated report raises.
- **False green.** `_classify_trace_value` walks every dict key and list index and
  leaf-classifies: a control-stable field (`A1==A2`) that B changes ⇒ violation;
  a control-nondeterministic field where B is neither within the inclusive numeric
  `[A1,A2]` range nor equal to an observed categorical control ⇒ violation. Audio
  is popped out and separately requires a nonnegative int, exact 1764-sample
  quantization, inclusive `[A1,A2]` envelope, and `traced_delta ≤ control_delta`.
  c014 additionally requires `assistant_text`/`function_text`/`tool_calls` equal
  across **all three** ABA runs and the traced settlement+single-EOU all
  `sotc_rank==2`, argmax==PAD, PAD logit>SOTC. Every escape is the honest observed
  null, not a bypass.
- **False red.** The recursive strip removes the 325-key Bug-1 artifact, and
  inherent nondeterminism (c009/c010/c017-style) is classified within-null and
  logged rather than RED. A genuine trace-only third value (the c005 case from
  Addendum 39) is correctly flagged `trace_outside_observed_control_null`. Tests
  cover exactly these: recursive nested strip + categorical within-null + audio
  envelope accept; stable-change, outside-null categorical, and audio-outside
  reject.
- **Mutable normalization.** All comparison runs on `copy.deepcopy`; the c014
  assertion reads the *original* reports read-only (it needs the intact trace);
  input reports are not mutated.
- **Path ambiguity.** `_case_map` keys by `(token_form, case_id)`, raises on a
  duplicate key or non-string id, and the gate requires the three case-key sets to
  be identical; classifier paths are disjoint key/index tuples.
- **Unbounded evidence.** `_bounded_field_value` caps strings at 160 chars and
  replaces compound values with a `{sha256, bytes}` digest; the nondet-field and
  violation lists are bounded by the (finite) report leaf count. RED returns a
  structured `passed:false` evidence object; malformed reports raise.
- **Orchestration / same-source ABA.** `run_direct_function_logit_diagnostic`
  resolves `image_id` **once** and passes it to all three runs (off-a/on/off-b),
  each built from the same `args` (same read-only image, Speech bind-mount,
  corpus), run sequentially as fresh `--rm` containers; `result["passed"] =
  neutrality["passed"]` and the full evidence is embedded even on RED. Separate
  same-image containers **do** satisfy contemporaneous ABA: they share no writable
  state, so A1/A2 are genuine independent draws of run-to-run nondeterminism under
  a fixed build/config — the correct null, unlike Addendum 39's cross-build r14
  baseline.

### Non-blocking notes

- **Finite 2-sample null.** With one A-pair the envelope is a two-sample estimate:
  a perturbation landing inside `[A1,A2]` could pass, and a field stable in A1/A2
  by luck could false-RED B. Both are inherent to ABA and honestly scoped by the
  method name `contemporaneous_off_on_off_observed_null` (it certifies "within the
  observed null," never absolute neutrality). A third untraced control would
  tighten both; the read-only trace proof (Addenda 33–38) and c014's ~4.5-logit
  margin already bound the practical false-green risk.
- The audio `traced_delta ≤ control_delta` check is redundant with the inclusive
  envelope (harmless). An explicit count cap on `control_nondeterministic_fields`
  would be defensive (currently report-bounded).

### Verdict

**APPROVE.** No blocker. The gate is fail-closed, non-laundering, mutation-free,
path-unambiguous, and bounded; it recursively strips Bug 1, classifies per-field
against a genuine contemporaneous same-session ABA null, bounds the quantized
audio tail separately, asserts c014 discrete stability and the rank-2/PAD-argmax
finding, and returns structured RED evidence while still raising on malformed
input. Same-source separate containers satisfy contemporaneous ABA. The only
caveat is the inherent finite-sample nature of a single A-pair, which the design
scopes honestly; a third untraced control is a worthwhile future tightening, not a
blocker. D8 intact.

## Addendum 41 — head/top-k argmax tie fix (r16 c017 false-red) (2026-08-09)

Reviewed the fix for the r16 validator RED on c017 post-BOS frame 37, where the
greedy head/argmax was PAD(12) but `torch.topk` returned `[10244@24.375,
PAD@24.375, …]` — a tied max with arbitrary order — so the old
`ranked_ids[0] == head_argmax_id` check false-failed. No source edited.

### Tie math is correct

The summary now records `head_argmax_logit = row[argmax]` (the max logit) and
`head_argmax_equal_count = count(row == head_argmax_logit)` (the size of the
max-tie block) from the full row (`model_factory.py:433–435`, both JSON scalars).
The validator replaces the broken identity check with:
- `ranked[0].logit == head_argmax_logit` (`run_live_suite.py:313`) — since top-k is
  sorted descending, `ranked[0]` is the global maximum, so this is exact and
  tie-agnostic (only the argmax *token* is ambiguous under a tie, never the max
  *value*);
- if the head token appears in top-k, its position must satisfy `argmax_position <=
  head_argmax_equal_count` (`:316–319`). Because the descending sort places all
  max-tied tokens in positions `1..equal_count`, the argmax — being one of them —
  provably falls in that block. At `equal_count == 1` this recovers the old strict
  "argmax at position 1." The math is sound.

### Omission rule is sound

The head may be absent from top-k **only** when `head_argmax_equal_count > top_k`
(`:320–321`): if the max-tie block is larger than the window, arbitrary tie
selection can exclude the head; if it fits (`equal_count <= top_k`), every tied
member including the head must be present, so absence is a genuine contradiction ⇒
RED. Correct in both directions.

### Greedy invariant intact; no laundering

`head_argmax_id == predicted_token_id == raw_function_token_id` is unchanged
(`:232`), and the summary still raises if `torch.argmax(row) != function_tokens`
(`model_factory.py:437–439`). The relaxation touches only the argmax's *position*
in top-k, never its *identity*, so it cannot launder a behavior change — and the
ABA neutrality gate compares behavior via non-trace fields anyway (it recursively
strips the whole trace, so these new fields do not affect neutrality). The unit
test reproduces the exact c017 order (tied token first, PAD argmax second,
`equal_count=2`) and now validates; the no-tie path still requires the argmax at
position 1.

### Schema compatibility and bounded opt-in scope

The two fields extend the schema-3 trace entry; the exact-key-set check
(`set(entry) != DIRECT_SEMANTIC_FUNCTION_TRACE_ENTRY_KEYS`, `:202`) now includes
them, so an old-format schema-3 entry lacking them is rejected fail-closed — no
silent acceptance, so no schema bump is strictly required (schema-2 untraced is
unaffected). Both reductions run only inside `_summarize_function_logits`, which is
gated by `if self._function_logit_trace_topk:`, so the disabled/production path is
byte-unchanged; they are a single gather plus one full-row equality-count on the
final position, reduced to scalars with no tensor retained — the bounded, opt-in
scope is preserved. Retained patch reverse-applies clean; focused trace/tie/
neutrality tests pass (15/15 on my rerun), Ruff clean.

### Minor (non-blocking)

The retained head==predicted invariant assumes `torch.argmax` (first-max-index) and
the vLLM function head break ties identically; c017 confirms they agree, and a
divergence would fail closed (a surfaced inconsistency, not a silent pass) — worth
awareness if a future tie shows the two argmax paths disagreeing, but out of scope
for this fix.

### Verdict

**APPROVE.** The fix correctly generalizes the top-k check to tolerate arbitrary
tie ordering while preserving the greedy identity anchor and fail-closed omission
handling; tie math and omission rule are sound, schema discrimination stays
fail-closed via the exact key set, and the extra reductions remain inside the
opt-in, final-position, scalar-only trace path. The r16 c017 RED was a validator
artifact of `torch.topk`'s arbitrary tie order, not a behavior defect. No blocker;
D8 intact.

## Addendum 42 — pre-implementation review: voice-reference function-logit probe (2026-08-09)

Design review of the proposed positive-control substep; no source edited, no GPU
work. The direction is sound — a voice-path causal control for c014 is exactly the
right next diagnostic — but several elements must be pinned before implementation,
one of which (a tool-callback race) could silently invalidate the whole control.

### Why the substep is well-motivated

r17/direct shows SOTC stuck at rank 2 with PAD the argmax by ~4.5 logits, so no
tool call. Running the *same* c014 through the production voice path (known to call
`get_weather(Tokyo)`) with the *same* trace instrumentation isolates the one
variable that matters: whether real acoustic/perception conditioning drives SOTC to
argmax where direct-token injection does not. Comparing only watched SOTC/PAD
margins/ranks and the discrete outcome — **not** aligned model positions, since
voice and text lengths differ — is the correct, sufficient contrast to decide
whether direct-text-as-implemented is viable.

### Attack surface

- **Fixture generation should be separate, not in-container.** Generate the Pocket
  `alba` PCM once outside the GPU container, freeze it as bytes + sample-rate +
  count + SHA256 + full Pocket model/voice/asset revisions, and have the diagnostic
  **re-verify the SHA256 at feed time** (fail closed on mismatch). In-container
  generation would let Pocket nondeterminism/version drift vary the *input* between
  runs and pull the CPU Pocket worker into the traced GPU container — both
  contaminate the comparison. A frozen, provenance-pinned byte-stream replayed
  exactly is the only reproducible input.
- **`input_active` and tail padding do not fully mirror production.** Production
  derives `input_active` from the transport gate/VAD (`server.py:3934/4053`) and
  gate-preroll-buffers sub-threshold leading silence; the proposal hardcodes
  `input_active=True` per carrier frame and feeds 1280-sample model frames directly,
  bypassing the gate. For external-EOU mode this is *approximately* right (EOU is
  explicit, not VAD-driven) and the zero-padded final frame with recorded true count
  is fine, but it is not a faithful mirror. **Therefore the probe's validity must be
  gated on the reference tool call actually firing:** if the traced voice run does
  not call `get_weather(Tokyo)`, the probe must RED as "voice reproduction invalid,"
  never be interpreted as "voice also fails." Better still, feed the carrier through
  the same transport-gate path or justify equivalence explicitly.
- **Tool-callback race (correctness blocker).** The wrapper runs function-call and
  perception work on background threads (`_run_fc_async_steps`,
  `threading.Thread` at `inference_wrapper.py:1519/1536`) when `fc_async` is enabled
  — which production uses. The direct probe reads `recorded_calls = list(calls)`
  immediately after the response loop
  (`direct_semantic_probe.py:332–333`); this never mattered because the direct tool
  never fired, but the voice probe **will** fire it. Reading the calls before the FC
  async cycle completes would drop the tool call → a false "no tool call" → a wrong
  architectural conclusion. The voice probe **must** run the response loop until the
  full tool→response→terminal cycle finishes and `function_cycle_active` clears
  (join/settle the background thread) before reading `recorded_calls`.
- **Missing causal controls / neutrality.** A single traced voice run cannot
  establish trace neutrality the way the direct ABA gate does. Since the voice
  outcome is categorical (tool fires or not), require at minimum a traced-vs-untraced
  discrete-outcome check on the fixture (both must fire `get_weather(Tokyo)` with
  identical args) — a mini-ABA — rather than only per-report trace invariants. A
  within-voice negative (a non-tool prompt through the same path) is optional but
  would confirm the margins discriminate tool from non-tool.
- **Trace volume/bounds.** Voice input is far longer than direct (tens of 80 ms
  frames) and "up to 240 response frames" would produce a large trace blob. Bound it:
  trace every voice_input frame (finite, fixture-length), settlement (~11), the one
  boundary, and a **capped** post_bos window (reuse the ≤8 cap or stop at terminal) —
  not all 240. Keep the existing per-entry bounds (≤20 top-k, scalar-only, no tensor
  retained).
- **Sufficiency for the architecture decision.** The margin/rank/discrete contrast
  is sufficient to conclude whether direct-text-as-implemented can produce the tool
  call (if voice makes SOTC argmax at the boundary and direct does not, direct-as-is
  is non-viable and the voice/typed path is the working architecture). It does **not**
  prove direct-text is unfixable — only that this implementation is; a decision to
  abandon vs augment direct should weigh the observed ~4.5-logit gap, which this
  probe will quantify against the voice baseline.

### Verdict

**REQUEST CHANGES.** The design is directionally correct and worth building, but the
following are required before implementation:
1. Generate the fixture separately; freeze bytes + SHA256 + Pocket provenance and
   re-verify the hash at feed time (fail closed).
2. Gate probe validity on the reference tool call firing — RED "voice reproduction
   invalid" if it does not; do not treat a non-firing voice run as a comparison
   against direct.
3. Synchronize on FC-async completion (`function_cycle_active` cleared / background
   thread joined) before reading `recorded_calls`, and run the response loop through
   the full tool→response→terminal cycle.
4. Add a traced-vs-untraced discrete-outcome check for the voice fixture.
5. Bound the traced frames (finite voice_input + settlement + one boundary + capped
   post_bos), scalar-only, no tensor retained.
6. Keep caller-owned phase tagging (voice_input/settlement/eou_boundary/post_bos)
   with exact phase/frame coverage, provenance/transcript/tool/terminal validation,
   and fail-closed red-evidence retention; compare only SOTC/PAD margins/ranks and
   the discrete outcome, never aligned positions.

D8 is respected (no forced tokens, prompt routing, or threshold/gate changes). With
these six changes the substep is a valid positive control; without 2 and 3 in
particular it risks a false-negative that would misdirect the architecture decision.

## Addendum 43 — voice-reference probe, revised design re-review (2026-08-09)

Re-review of the design after all six Addendum 42 changes were accepted. Each is
now addressed, several beyond what I asked, and the two questions posed
(consecutive quiescence, exact coverage) are sufficient. No source edited, no GPU.

### The six changes, as revised

1. **Fixture generation (exceeds).** A separate CPU-only, `--network none` container
   from the *exact immutable runtime image* generates the fixture once, with a full
   provenance manifest (prompt/hash, rate/count/bytes/SHA256, seed scheme, voice/
   package/quantize versions, verified asset bytes+SHA256, generator source hashes,
   image id); the orchestrator revalidates and every GPU probe rehashes at feed.
   Generating from the same image pins the Pocket package/assets to the runtime —
   stronger than "separate + SHA."
2. **Input fidelity (exceeds).** Replaying real 20 ms packets through the production
   `PcmFrameBuffer` + `TransportModelGate` with `should_advance`-derived
   `input_active`, production preroll rules, and production-style partial-tail flush
   is a faithful mirror, not the hardcoded simplification I flagged. Use the same
   gate entry points production's commit uses for `force_open` so it cannot diverge.
3. **FC-async completion (correct).** Instead of joining the worker, the probe
   observes the pipeline's published quiescence — stream absent from `_fc_async_bg`
   after the pipeline itself observes thread completion, `fc_state`
   active/awaiting/injecting all false with no forced tokens, `completed_calls == 1`,
   the exact `get_weather(city=Tokyo)` recorded, and a terminal acoustic boundary
   *after* quiescence — then locks/copies calls; timeout ⇒ RED
   `voice_reproduction_invalid`. "Observe, don't reach in" is the right pattern.
4. **Neutrality (exceeds).** Full off-a/on/off-b ABA on the frozen fixture reusing
   the A40 recursive-strip + observed-null gate, all three required to make the
   identical call and terminal outcome — consistent with the validated gate, and it
   fails closed if the voice path is flaky rather than cherry-picking.
5. **Bounds/coverage (correct).** ≤30 s / 375 carrier frames, top-k ≤20, all
   foreground voice_input+settlement positions, one EOU, ≤8 post-boundary foreground
   positions, no tensors, and per-transport-frame accounting that records either a
   fresh exact `model_frame` or an explicit no-new-model-position reason while the FC
   background owns advancement — stale entries forbidden.
6. **Comparison/conclusion (correct).** Per-phase min SOTC rank, max SOTC−PAD margin,
   raw-SOTC-argmax occurrences, and deferred-stage/replay evidence vs r17 direct with
   no position alignment; the conclusion is constrained to "direct-as-implemented
   nonviable / distribution-shifted," never "direct text fundamentally unfixable."

### Are consecutive quiescence and exact coverage sufficient? Yes.

**Consecutive quiescence.** The real guarantee is the *conjunction*, not the
consecutive count. The FC path is two-phase with a simulated `api_latency` sleep
*inside* the background step (`inference_wrapper.py:1808–1816`), and it sets
`awaiting_response`/`injecting_response` across the inter-phase window
(`:1826/:1782`) with `completed_calls` appended at tool completion (`:1801`). During
the sleep the stream is still in `_fc_async_bg`, and across both phases at least one
of active/awaiting/injecting is true — so requiring all three false **plus**
`_fc_async_bg` absence **plus** `completed_calls==1` **plus** a terminal boundary
*after* quiescence structurally excludes every mid-cycle and api-latency window.
Consecutive foreground observations are correct defense-in-depth against a
single-sample transient at the exact edge; a count ≥2 suffices given the conjunction
already covers the sleep. The ordering constraint (quiescence *then* terminal) is
what prevents accepting the frozen-advancement false-terminal seen in the earlier
c009/c013 investigations.

**Exact coverage.** Because the api-latency sleep and FC background hold
advancement, many transport frames yield no new foreground model position; recording
each as an explicit no-new-model-position reason (never a stale repeated
`model_frame`) makes the coverage gap-free and reconstructable, and the ≤8
post-boundary foreground cap bounds the traced positions while the 240-frame budget
bounds the non-advancing records. The validator must actually *enforce* unique,
per-phase-contiguous foreground `model_frame`s and reconstruct coverage from the
per-transport-frame accounting (as the design states) — that closure is what makes
"exact coverage" a fail-closed guarantee rather than only a recording convention.

### Minor (non-blocking)

- Pin the consecutive-quiescence count (≥2) and the `voice_reproduction_invalid`
  timeout budget relative to the 240-frame response bound.
- The voice-vs-r17 comparison is cross-session, which is fine for this *categorical/
  margin* contrast given direct's large ~4.5-logit gap; only if the observed voice
  margins came out close would a same-session direct re-run be warranted.
- The Pocket-`alba` fixture is a synthetic proxy; it is valid because the mechanism
  under test is acoustic/perception conditioning and the probe fails closed unless
  the fixture actually reproduces the reference call — state the finding as
  "acoustic-frame conditioning drives SOTC to argmax," which generalizes by
  mechanism.

### Verdict

**APPROVE.** All six Addendum 42 changes are accepted and well-specified; the design
is faithful to production input, fail-closed on FC-async completion via observed
quiescence (not intrusive joins), reuses the validated ABA neutrality gate, bounds
and exactly covers the trace, and constrains its own conclusion. Consecutive
quiescence and exact coverage are sufficient given the conjunctive completion
criteria and the no-stale-frame discipline. Proceed to implementation; the minor
items above are refinements, and the implementation review should confirm the
validator enforces (not merely records) unique-contiguous foreground frames and the
observed-quiescence conjunction. D8 intact — no forced tokens, prompt routing, or
threshold/gate changes.

## Addendum 44 — voice-reference probe implementation review (2026-08-09)

Reviewed the five new files against Addendum 43; no source edited, no GPU. r17
(direct) is independently still running — I did not treat any incomplete r17 result
as final. The implementation faithfully realizes the approved design and is
fail-closed throughout; I found no blocker.

### Attack surface — findings

- **Manifest/source/image/PCM validation (fail-closed).** `validate_voice_function_fixture`
  enforces exact manifest/pcm/wav/synthesis/assets/provenance key-sets, re-reads and
  re-hashes the PCM, cross-checks the WAV format and that `wav_pcm == pcm_bytes`,
  pins language/voice/quantize/seed-scheme, path-traversal-guards asset/source names,
  and requires `runtime_image_id == expected`. `validate_fixture_source_provenance`
  rehashes the four generator sources (and the installed Pocket worker in-container)
  against the manifest. The probe re-validates at feed time and the orchestrator
  re-validates + cross-checks `report["fixture"] == manifest`. No gap.
- **Production TransportModelGate/PcmFrameBuffer replay (faithful).** Real
  `PcmFrameBuffer` + `TransportModelGate` with `force_open()`, 20 ms packets pushed
  and coalesced into 1280-sample model frames, `should_advance` called per frame with
  `input_active = gate.input_active`, and a nonempty-final-partial flush recorded with
  the padded flag. `FRAME_SECONDS=0.08` confirms 1280-sample frames. Correct for a
  committed turn.
- **Trace fresh-vs-stale + unique contiguous coverage (enforced).** `_trace_capture`
  accepts a fresh position only when `model_frame` strictly increases and (under
  background) equals the live `frame_index`; otherwise it records an explicit
  `no_new_model_position / fc_background_owned_advancement` gap, and any other shape
  raises. `_validate_trace_coverage` (in-container) and `_validate_trace_entries` +
  the coverage cross-check (orchestrator) both enforce fresh-count == entries, unique
  model frames, per-phase-contiguous positions and frames, and that every gap is a
  background-owned post-BOS frame. The orchestrator reuses the full
  `live._validate_function_logit_trace` suite (incl. the A41 tie math) by remapping
  `voice_input → direct_position` into a synthetic case and rejects `direct_hidden_pad`
  reasons. Enforced, not merely recorded.
- **8-position cap.** post-BOS foreground tracing is gated by
  `foreground_positions["post_bos"] < FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES` (8) while
  the response loop keeps running for quiescence.
- **SOTC margin math.** `max(sotc.logit − pad.logit)` correctly captures closest-
  approach/overtake, `min_sotc_rank`, and raw-SOTC-argmax counts per phase; empty-
  guarded with `default=None`.
- **FC-async quiescence / terminal ordering / call race.** `_function_quiescence`
  is the full A43 conjunction — `_fc_async_bg` absence, `active/awaiting/injecting`
  all false, `forced_tokens==0`, `completed_calls==1`, and the exact
  `get_weather(Tokyo)` recorded — and I verified `completed_calls`/`forced_tokens` are
  `len(...)` **int counts** in the engine `turn_state` (`server.py:1710–1711`), so the
  `int(...)`/`is False` reads are correct (no type bug). Termination requires a
  boundary-`end` observed *while* quiescent plus a ≥2 consecutive quiescent streak; a
  second call would break quiescence (`completed_calls!=1`), so the latched terminal
  cannot fire early. Calls are read under `calls_lock` per step and finally after the
  break; the probe never joins or mutates the worker (confirmed).
- **ABA normalization (no laundering / no false-red).** Reuses the A40
  `_classify_trace_value` observed-null; `_normalize_voice_report` strips only trace,
  `response_audio_path`, and volatile codec `warmup_worker_ms`/`pid`; audio is popped
  and envelope-checked; `discrete_outcome_equal` requires all three ABA runs to
  reproduce the call and pass. Fail-closed on flakiness.
- **Direct comparison.** Validates the direct diagnostic's identity and re-validates
  its report via `validate_direct_semantic_probe_report`; compares margins/ranks with
  no position alignment; conclusion is constrained to "distribution_shifted," never
  "unfixable"; overall `passed = neutrality AND comparison`.
- **Docker isolation / same-image.** Fixture built in a CPU-only `--network none`
  container from the same resolved `image_id`; the three GPU probes run `--network
  none`, `--rm`, read-only fixture/corpus/models, same `image_id`, trace env only on
  the "on" run; A36/37/38 speech-root+GIT_CONFIG+PYTHONPATH applied only under
  `--speech-root`.
- **Error artifacts.** The server wraps the probe in `try/except BaseException` →
  structured `passed:false` error report + exit 1; `run_probe_stage` enforces the
  rc↔passed contract; `validate_probe_report` rejects a green error report and
  returns early for an honest error RED; `main()` wraps `run` and writes a top-level
  `{passed:false,error}` artifact on any exception. No path yields green from an
  error, and per-run reports persist on disk.

### Minor (non-blocking)

1. `_normalize_voice_report` strips only the *top-level* `function_logit_trace`; the
   voice report does not embed per-StepResult trace today (so this is safe), but a
   recursive strip like the direct gate's `visit` would be more robust against future
   nesting (fail-closed either way).
2. The voice audio envelope check omits the 1764-sample quantization and
   `traced_delta ≤ control_delta` bounds the direct gate applies; adding them would
   restore parity (the inclusive envelope is still binding, so this is rigor, not a
   hole).
3. `force_open()` intentionally bypasses the preroll/VAD-gating path — correct for a
   committed turn, but the probe therefore validates the committed-turn regime, not
   continuous listening; requiring `should_advance==True` per frame is a fail-closed
   assumption worth noting.
4. An errored *on* run aborts to the top-level `{passed:false,error}` artifact (via
   the schema-2-without-trace guard in `_normalize_voice_report`) rather than a clean
   neutrality RED; fail-closed and per-run evidence is retained, but classifying it as
   `voice_reproduction_invalid` would be cleaner.

### Verdict

**APPROVE.** The implementation is fail-closed at every layer — fixture provenance,
production-faithful gate/buffer replay, fresh-vs-stale trace with enforced unique-
contiguous coverage, the FC-async quiescence conjunction with correct terminal
ordering and a locked, non-intrusive call read, ABA neutrality reusing the validated
observed-null gate, and a constrained direct comparison — with no runtime bug found
and 8 focused tests + Ruff green. The four minor items are hardening refinements, not
blockers. D8 intact. The implementation review's open ask from Addendum 43 (enforce,
not merely record, unique-contiguous frames and the quiescence conjunction) is
satisfied.

## Addendum 45 — current hardening + completed r17 characterization (2026-08-09)

Re-reviewed the CURRENT code (not the A44 snapshot) and the completed r17 direct
ABA. No source edited, no GPU.

### Current hardening confirmed

All A44 minors and the listed items are in the working tree: post-BOS capture is a
hard `response_position <= FUNCTION_LOGIT_TRACE_POST_BOS_FRAMES` cap
(`voice_function_probe.py:419`, holds even across background gaps); voice
normalization now recursively strips every `function_logit_trace` at all depths via
a `visit` walker (`voice_function_logit_diagnostic.py:459–474`); the audio gate now
enforces invalid / `% EARTTS_FRAME_SAMPLES` quantization / inclusive envelope /
`traced_delta ≤ control_delta` (`:517–533`), full parity with the direct gate;
errored ABA runs return a **structured neutrality RED** early (`voice_probe_error`
violation, `passed:false`) instead of aborting (`:501–513`); the fixture asset
inventory is pinned exactly to `EXPECTED_ASSETS` and source inventory is exact and
rehashed host+container (`voice_function_fixture.py`); input/padding is
arithmetically revalidated (`original_pending == count % 1280`,
`padded == ceil(count/1280)*1280`, `len(records) == padded//1280`,
`:304–308`); and the direct/voice comparison requires `voice_watched ==
direct_watched` (`:598`). Focused 13 + Ruff pass here; the reported full suite is
467 pass / 8 skip. No new blocker.

### r17 is honest RED — and it does not touch c014's finding

r17 (`direct-function-logits-r17-aba`) is `passed:false`; the neutrality violations
are, verified directly: **c012** (case index 11) — one hidden
`direct_positions/9/model/raw_text_tokens/0` differing from both stable A controls
— and **audio-tail** (`audio_samples`) outside the two-control envelope for c009,
c010, c013, c014, c015, c019. That is the whole set. Independently:

- **c014's discrete fields are byte-identical across off-a / on / off-b**:
  `assistant_text` ("I will use get_weather to check the weather in Tokyo."),
  `function_text` (""), `tool_calls` (`[]`), `structural_passed` true in all three.
- c014's traced run: all **11 settlement + 1 EOU** entries at **SOTC rank 2** with
  **PAD always argmax**, max SOTC−PAD margin **−2.9375**, no tool.
- c014's *only* r17 violation is its `audio_samples` tail — the EarTTS decoded
  length, the most run-nondeterministic quantity, which a 2-sample `[A1,A2]`
  envelope under-samples (the finite-sample caveat documented in Addendum 40). It
  is not a discrete/mechanistic field.

### Does the RED invalidate the c014 mechanistic finding? No.

The c014 claim rests on c014's own per-case ABA discrete-stability, which holds
exactly: the trace-on run reproduced both untraced controls' c014 discrete
behaviour byte-for-byte, so the "SOTC rank 2, PAD argmax, no tool" result is not a
trace artifact. The RED is driven entirely by (i) audio-tail nondeterminism in six
cases and (ii) one c012 raw-token divergence — the c005-analog from Addendum 39,
unclassifiable from only two controls and most plausibly under-sampled
nondeterminism given the W8 model's demonstrated per-token variance (r14/r15). The
trace is read-only and post-decision (proven across Addenda 33–41), so a genuine
trace-induced token change is architecturally precluded; and empirically, whatever
moved c012's one token left c014 identical across all three runs. The −2.9375-logit
margin is a large-margin categorical result no read-only trace could move. The
finding stands.

### Does it block the frozen voice ABA? No.

The voice ABA is an independent experiment on a frozen fixture with its own
hardened neutrality gate (recursive strip, quantized audio envelope+delta,
`discrete_outcome_equal` on the c014 call). r17's direct-corpus RED does not
contaminate it, and its comparison baseline — c014's direct evidence — is valid per
the above. The voice gate uses the same per-case discrete-stability discipline that
makes c014 interpretable in r17, so proceeding is methodologically sound.

### How to characterize r17

**r17 is an honest, fail-closed whole-corpus trace-neutrality RED whose violations
are confined to run-to-run nondeterminism that a two-control null under-samples
(six audio tails and one c012 hidden token) — none of them c014's discrete or
mechanistic evidence.** It is "whole-corpus neutrality not cleanly established under
a 2-sample null" (the expected Addendum-40 finite-sample limitation), not "tracing
changed behaviour." The c014 mechanistic finding — SOTC pinned at rank 2, PAD
argmax by ~2.94 logits across all settlement+EOU positions, no tool — is
independently valid because c014 is discretely identical across the ABA. Scoping
note (non-blocking): a clean *whole-corpus* green would require ≥3 untraced controls
to widen the null; the c014-anchored finding and the voice positive control do not.

### Verdict

**APPROVE.** The current hardening diff is present and correct in the working tree,
fail-closed throughout; r17 is an honest RED that neither invalidates the c014
mechanistic finding (c014 discretely identical across the ABA, large-margin
categorical) nor blocks the independently-neutralized frozen voice ABA (its baseline
is valid and its gate is self-contained). Proceed to the voice control. D8 intact.

## Addendum 46 — voice r1 off-a RED diagnosis: response-loop pacing (2026-08-09)

Review-only diagnosis of the current voice off-a RED
(`voice-function-logits-r1-aba/off-a`). No source edited, no GPU.

### Root cause (confirmed from artifacts)

`off-a/report.json`: `response_frames=240`, `tool_calls=[]`,
`expected_tool_call=false`, `function_quiescent_twice=false`; the last quiescence
records show `background_absent=false`, `completed_calls=0`, `quiescent=false`
through response_position 239. `off-a.log`: two-phase FC async generated the exact
call — `[FC Async] EOTC at async step 29 (frame 69). Call: get_weather(Tokyo)` —
after **3.677 s** of background work (30 steps), then "Phase 1 complete … Exiting
async to await tool execution," and finally `[FC Async NB] Background thread
failed: RuntimeError: cannot schedule new futures after interpreter shutdown`
(`streaming_s2s_pipeline.py:1693`).

So this is **not a deadlock and not a model failure**: the FC async background
was progressing normally and generated the correct call, but the probe's response
loop out-ran it. `voice_function_probe.py:397–442` is a tight
`for response_position in range(240)` loop calling `engine.process` with **no
wall-clock pacing** (the module imports neither `time` nor `asyncio`). It blasts
all 240 foreground frames in milliseconds — far less than the 3.7 s the background
needed for Phase 1 alone — hits the frame cap, writes the RED report, and returns;
interpreter teardown then kills the background's Phase-2 tool scheduling.

### Production already has the faithful contract

`advance_function_output_recovery` (`server.py:404–420`, used at `:4623`) drives
post-tool recovery on a **real-time model-frame clock**: it steps, returns as soon
as `function_cycle_active` clears, and otherwise
`next_frame_at += FRAME_SECONDS; await asyncio.sleep(max(0, next_frame_at −
monotonic()))`, bounded by `FUNCTION_OUTPUT_APPLY_MAX_FRAMES`. Pacing every frame
to 80 ms is exactly what gives the FC async background wall-clock time to finish
both phases. The probe omitted this contract.

### The three options

- **Wall-clock pacing** (sleep ≈80 ms/frame): necessary and correct, but should
  mirror the production contract rather than an ad-hoc sleep.
- **Bounded wait only while background owns advancement**: more complex, less
  faithful (production paces *every* response frame, not only background-owned
  ones), and risks an ad-hoc busy/'unbounded' wait.
- **Adopt the existing `advance_function_output_recovery` synchronization
  contract** (subsumes wall-clock pacing): the cleanest and most faithful — same
  80 ms cadence, same terminate-on-cycle-clear, same bounded budget. **This is the
  recommended fix.** The probe's response loop should pace each iteration on a
  monotonic frame clock (`time.sleep` is the sync mirror of the async
  `asyncio.sleep`) and keep terminating on the existing observed quiescence.

### Attacks answered

- **Would sleeps hide a real deadlock?** No — *provided the paced loop keeps a
  bounded frame budget and REDs on exhaustion.* The log proves the background was
  live (Phase 1 done in 3.7 s); pacing gives it its needed wall-clock *within* the
  bound. A genuine hang would exhaust the paced budget (≈256 × 80 ms ≈ 20 s of real
  time) and RED `voice_reproduction_invalid` — surfaced, not masked. The bound, not
  the sleep, is the deadlock guard.
- **Are the response-frame cap semantics currently wrong under background
  ownership?** Yes. The cap counts foreground `engine.process` calls, but while the
  background owns advancement those are near-instant no-op spins (the probe already
  records them as `no_new_model_position`). 240 unpaced spins ≈ milliseconds, not
  19 s. The cap is a spin count, not a real-time budget; pacing converts it into a
  wall-clock budget that matches production.
- **Retaining fail-closed quiescence/terminal evidence without joining/mutating the
  worker?** The existing observe-only conjunction (`_fc_async_bg` absence +
  `active/awaiting/injecting` false + `forced_tokens==0` + `completed_calls==1` +
  exact call) plus terminal-after-quiescence and the ≥2 streak stay unchanged. The
  fix only adds pacing so those observations have time to become true; no `.join()`
  and no `_fc_async_bg` mutation. On success the loop returns only after quiescence,
  so `finally: engine.abort()` tears down a fully-quiesced engine — which also
  removes the "schedule after shutdown" teardown race.

### Required invariants (for the fix)

- **I1 — real-time pace:** each response iteration advances at most one model frame
  per `FRAME_SECONDS`, measured on a monotonic clock (mirror
  `advance_function_output_recovery`).
- **I2 — bounded + fail-closed:** the loop is bounded by a paced frame budget
  (align to `FUNCTION_OUTPUT_APPLY_MAX_FRAMES`, ≥ current 240); exhausting it
  without quiescence ⇒ RED `voice_reproduction_invalid` with full evidence, never
  silent acceptance.
- **I3 — terminate on observed quiescence (unchanged):** the existing conjunction +
  terminal-after-quiescence + ≥2 consecutive streak; observe-only.
- **I4 — clean teardown:** on success, return only after quiescence so no pending
  Phase-2 work remains at `engine.abort()`.
- **I5 — no worker intrusion:** add only `time`/monotonic pacing; no `.join()`, no
  `_fc_async_bg` mutation.
- **I6 — trace unchanged:** fresh-vs-stale, ≤8 post-BOS cap, unique-contiguous
  coverage, and neutrality all remain as reviewed.

### Required tests (host, no GPU; inject a fake clock so tests do not sleep 20 s)

- **T1 (pacing lets a slow background finish):** fake engine keeps `_fc_async_bg`
  populated for N paced iterations, then clears + sets `completed_calls=1` + records
  `get_weather(Tokyo)`; assert the probe reaches quiescence and passes.
- **T2 (bound catches a hang):** fake engine whose background never clears; assert
  the paced budget is exhausted and the report is RED `voice_reproduction_invalid`
  (deadlock surfaced, not hidden).
- **T3 (pacing present):** monkeypatch `time.sleep`/`time.monotonic`; assert one
  ~`FRAME_SECONDS` pace per response frame (cadence mirrors
  `advance_function_output_recovery`).
- **T4 (no intrusion):** assert the probe never calls `.join()` and never mutates
  `_fc_async_bg` (read-only).
- **T5 (teardown race closed):** on success, assert termination occurs only after
  `background_absent and completed_calls==1`, and `engine.abort()` runs on a
  quiesced engine.
- **T6 (trace preserved):** coverage/fresh-stale/≤8-cap/contiguity assertions still
  hold under the paced loop.

### Verdict

**APPROVE the proposal:** adopt the production `advance_function_output_recovery`
real-time-frame pacing contract in the probe's response loop (option 3, subsuming
wall-clock pacing), keeping the bounded budget with fail-closed RED-on-exhaustion
and the unchanged observe-only quiescence/terminal conjunction. This is a probe
harness pacing fix only — it does not touch model behavior, gates, thresholds, or
tokens (D8 intact). It fixes both the RED (the background now has time to complete
the call) and the interpreter-shutdown teardown race (the loop returns only after
quiescence). Implement to invariants I1–I6 with tests T1–T6.

## Addendum 47 — pacing-fix implementation review (2026-08-09)

Reviewed the current pacing fix (`voice_function_probe.py`, `server.py`,
`test_voice_function_fixture.py`); no source edited, no GPU. The A46 pacing
mechanism is implemented correctly, but one ABA-neutrality issue is a blocker for
the real voice ABA run.

### Confirmed correct

- **Constants:** `VOICE_RESPONSE_FRAME_LIMIT = FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES`
  = 256 and `VOICE_RESPONSE_FRAME_SECONDS = FUNCTION_OUTPUT_RECOVERY_FRAME_SECONDS`
  = 0.08 (`protocol.py:18–19`), and the probe fail-closes if
  `max_response_frames != 256` (`:277–278`) or `FRAME_SECONDS != 0.08` (`:279`),
  cross-checking the server frame clock.
- **Pacer:** `_MonotonicResponseFramePacer.positions` is a faithful sync mirror of
  `advance_function_output_recovery` — `next_frame_at = monotonic()`, first
  position immediate, each later position `next_frame_at += frame_seconds; sleep(
  max(0, next_frame_at − monotonic()))`; injectable clock/sleep. The boundary step
  (position 0) is reused without a pace and every later `engine.process` is paced.
- **Bound/RED/quiescence:** exhausting the 256 paced frames without quiescence sets
  `voice_reproduction_valid=False` with `failure.code="voice_reproduction_invalid"`
  (`:518–527`); the observe-only quiescence conjunction, terminal-after-quiescence,
  ≥2 streak, no-join/no-mutation, and ≤8 post-BOS trace cap are unchanged.
- **`response_frame_clock_exact`** (`:554–559`) is an internal-consistency check
  (`observed_frames == len(quiescence)`, `paced_transitions == len−1`,
  `frame_seconds == 0.08`); `paced_transitions` increments in the same block as the
  `sleep` call, so it cannot green a structurally-unpaced run.
- **Server default follows the constant:** both the server
  (`str(VOICE_RESPONSE_FRAME_LIMIT)`) and the orchestrator env default resolve to
  256, matching the probe's guard.
- **Generator/break/off-by-one:** breaking out abandons the generator cleanly with
  `observed_frames`/`paced_transitions` consistent with the record count; position 0
  is correctly unpaced; default callables are `time.monotonic`/`time.sleep`.
- **Tests:** the fake-clock pacer tests are meaningful — cadence
  (`sleep_requests == [0.08,0.08,0.08]`, first immediate), gives-a-slow-background-
  time-and-breaks, and hung-background-stays-bounded. Focused set + Ruff pass on my
  rerun. `validate_probe_report` also re-derives the pacing/quiescence internal
  consistency per report (`:367–392`).

### Blocker

- **B1 — ABA neutrality will false-RED on real runs from un-excluded run-varying
  operational fields.** The report carries fields that vary run-to-run with the
  nondeterministic response-loop break point: the full `quiescence` records list,
  `response_frames`, and `response_pacing.observed_frames`/`paced_transitions`
  (`frame_seconds`/`budget_frames`/`clock` are deterministic and fine).
  `_normalize_voice_report` strips only `function_logit_trace`/`response_audio_path`/
  codec `warmup_worker_ms`/`pid`, and `voice_neutrality` pops only
  `response_audio_samples` — so those run-varying fields reach
  `_classify_trace_value`. On a real ABA, off-a and off-b diverge on them; the
  `quiescence` **list** (different length/contents between the two controls) is
  classified categorical, so the traced run must equal one control exactly — a third
  run never will → **guaranteed neutrality violation → false RED**, which makes the
  voice diagnostic (`passed = neutrality AND comparison`) unable to pass on real
  hardware. The deterministic fake-clock tests use fixed fake engines, so the three
  ABA reports are identical on these fields and never trigger it — the blocker is
  latent and untested. (The `quiescence`/`response_frames` fields predate A47; the
  new `response_pacing` counters join the same class, which is exactly the neutrality
  implication to attack.)

  **Fix:** exclude the run-varying operational bookkeeping (`quiescence` records,
  `response_frames`, and `response_pacing.observed_frames`/`paced_transitions`) from
  the neutrality deep comparison — pop them like `response_audio_samples`, or reduce
  to a stable summary — while keeping the deterministic pacing invariants
  (`clock`, `frame_seconds==0.08`, `budget_frames==256`) and relying on each report's
  own `response_frame_clock_exact` + `validate_probe_report` internal-consistency
  checks. Add a neutrality test where off-a and off-b carry **divergent** quiescence
  lists / pacing counts and assert it does **not** RED (only genuine trace divergence
  should).

### Verdict

**REQUEST CHANGES.** The A46 pacing fix itself is correct and will resolve the r1
off-a RED and the interpreter-shutdown teardown race — the constants, monotonic-
deadline mirror, immediate-first/paced-later, bounded fail-closed RED, unchanged
observe-only quiescence, server-default, and fake-clock tests all check out.
Blocker **B1** must land with it: the run-varying `quiescence` list and pacing/
frame counters must be excluded from (or summarized for) the ABA neutrality
comparison, with a divergent-control test, or the voice ABA will false-RED on every
real run despite genuinely-neutral tracing. D8 intact (harness-only).

## Addendum 48 — B1 fix re-review (2026-08-09)

Re-reviewed the current B1 fix; review-only, no source/GPU/commit.

### What is correct

- **No laundering of trace effects.** `_normalize_voice_report` excludes only
  `quiescence`, `response_frames`, and `response_pacing.observed_frames`/
  `paced_transitions` (`:513–518`), keeping the deterministic
  `clock`/`frame_seconds`/`budget_frames`. Model-derived behaviour
  (`assistant_text`, `function_text`, `tool_calls`, the trace, `sotc_evidence`
  counts, provenance) stays compared, so a real behaviour change still shows. The
  three counts are retained under `operational_timing` (`:549`), summarized rather
  than discarded — the correct treatment for pure observation bookkeeping.
- **Report validation precedes normalization.** `_normalize_voice_report` has a
  single caller inside `voice_neutrality` (`:548`), and `voice_neutrality` runs on
  the reports returned by `run_probe_stage`, each already fail-closed by
  `validate_probe_report` (`:458`, which re-derives the pacing/quiescence exact
  consistency); the error-run branch returns a structured RED before normalization.
  Ordering holds on every path.
- The excluded fields and the divergent 45/53/48 timing test correctly show the
  confirmed quiescence/pacing blocker is resolved without false-RED.

### Blocker — the exclusion principle was not applied to nested position bookkeeping

The fix targeted the specific top-level fields but left the **same class** of
run-varying evidence nested elsewhere, still deep-compared by
`_classify_trace_value`:

- **`settlement` (`UserEouSettlement.evidence()`)** — a `steps` **list** carrying
  per-step absolute model-frame indices. A list whose length or contents differ
  between the two controls is classified categorical, so the traced run must equal
  one control exactly → guaranteed false-RED (the exact failure mode B1 fixed for
  `quiescence`).
- **`sotc_evidence`** — `pre_eou_suppressed_frame`,
  `deferred_eou_sotc_staged_frame`, `deferred_eou_sotc_replayed_frame` are absolute
  frame indices (numeric → envelope-tolerant, but still false-RED if out of the
  two-control range).
- **`eou_boundary.engine_frame`** — an absolute frame index (same numeric risk).

These absolute positions shift whenever the pre-EOU model-frame count moves by even
one frame, **without any behaviour change**. That is not hypothetical: r17 showed
per-position `raw_text_tokens` diverging within a same-session ABA (c005/c012),
i.e. the greedy decode is not bit-deterministic under CUDA, so the RNNT blank
settlement can reach the fence at a different step and shift every downstream
absolute frame. The `settlement.steps` list is the highest risk (categorical,
guaranteed false-RED if the step count moves); the frame indices are the same
position-bookkeeping class the fix excluded at the top level but did not exclude
here. The fake-clock/fixed-engine tests produce identical settlement trajectories
across the three ABA runs, so they cannot surface this.

The distinction to preserve: the **counts/behaviour** (`settlement` step count,
blank counts, `sotc_evidence` staged/replayed/suppressed **counts**,
`assistant_text`/`user_transcript` model output) should stay compared — a
divergence there is meaningful and an honest RED, with the c014 finding surviving
on the discrete anchor as in r17. Only the **absolute-position indices** are
harness bookkeeping that should be excluded or normalized (compared as deltas
relative to the EOU frame, or summarized), exactly as `quiescence`/pacing were.

### Fix + test

Extend the same exclusion/summary treatment to the nested absolute-frame indices —
`settlement.evidence()` step frame indices (keep the step count and blank
counts), `sotc_evidence` `*_frame` fields (keep the `*_count` fields), and
`eou_boundary.engine_frame` — while retaining all counts and model-derived output.
Add a neutrality test where the two controls carry the **same behaviour but a
shifted pre-EOU frame offset** (e.g. settlement steps at frames 40–50 vs 41–51)
and assert no false-RED, alongside a test that a genuine count/behaviour divergence
still REDs.

### Verdict

**REQUEST CHANGES.** The B1 fix is correct as far as it goes — it does not launder
trace effects, validation precedes normalization, and the confirmed quiescence/
pacing exclusion plus `operational_timing` summary are right. But the fix applied
its own principle to the top-level symptom, not the class: nested absolute-frame
position bookkeeping in `settlement.steps`, `sotc_evidence`, and `eou_boundary`
remains deep-compared and will spuriously false-RED the real voice ABA if the
pre-EOU frame trajectory shifts by even one frame (which CUDA nondeterminism, per
r17, can cause) — `settlement.steps` guaranteed, the frame indices out-of-envelope.
Extend the exclusion/summary to those absolute positions (keeping the counts and
model output) with a shifted-frame-offset control test before the voice ABA run.
D8 intact (harness-only).

## Addendum 49 — A48 fix re-review: settlement/frame delta-normalization (2026-08-09)

Re-reviewed the current A48 fix; review-only, no source/GPU/commit.

### The four attacks

- **Over-normalization / laundering — clean.** The settlement delta-conversion
  (`:534–571`) subtracts only the first-step baseline from a fixed set of absolute
  clock/token positions (`frame_index`, `audio_frame_index_after`,
  `perception_frame_idx_after`, nano/eartts `generated_tokens`/`session_positions`),
  so the *relative* per-step trajectory is preserved and only the nondeterministic
  starting offset is removed; `*_frame` fields in `function_calling`,
  `sotc_evidence`, and `eou_boundary`/`response_boundary` are dropped while the
  **counts/flags/tokens/step-count/output behaviour** are retained. A real relative
  divergence still shows. The count-divergence test (`settlement.model_steps` → 3)
  asserts a violation at the exact `settlement/model_steps` path, proving behaviour
  divergence is not laundered.
- **Malformed-shape handling — safe.** `nested_value` traverses with `.get` (returns
  `None` on any missing/non-dict link); the delta applies only when both the value
  and the step-0 baseline are non-bool ints, otherwise the field is left untouched;
  `set_nested` is reached only after `nested_value` confirmed the full path is a live
  int, so its direct indexing cannot `KeyError`. Per-report `validate_probe_report`
  runs before normalization, so shapes are already fail-closed.
- **Tests — meaningful.** The 40/47/41 offset test shifts the absolute clock
  positions across the three reports while holding the relative trajectory equal and
  asserts no false-RED; the count test shows a genuine step-count change still REDs
  at the exact path. Both are real.
- **Missed absolute offsets — one residual (see below).**

### Residual (non-blocking)

The fix normalizes the clock/token **position** offsets but retains, absolute, the
**blank-count** family: per-step `rnnt_blank_frames` and settlement top-level
`starting_blank_frames`/`ending_blank_frames` (only `steps[*]` are delta-converted,
not the settlement summary). These carry a run-varying *starting* offset if the RNNT
decode of the frozen voice carrier is not bit-deterministic at its tail — the same
CUDA nondeterminism r17 exhibited — which would shift every blank value and can move
the settlement step **count**, tripping the (correctly-retained) `model_steps` /
list-length gate. The offset test does **not** cover this: it holds
`rnnt_blank_frames = 9 + index` identical across all three reports, so it implicitly
assumes blank stability. Also `vllm_request_positions.prefill_baseline` is retained
absolute — but that is the deterministic prompt-prefill length and is genuinely
stable.

This residual is materially lower-risk than the earlier rounds: r17's direct
settlement was **stable** across its same-session ABA (no settlement violation), the
settling mechanism is identical here, `rnnt_blank_frames` is numeric so a small
in-range offset is envelope-tolerant, and a step-count RED — if it fires — is a
defensible honest RED with the c014 finding surviving on the discrete anchor. It is
a judgment call (blank counts as behaviour vs. position), not a guaranteed
false-RED. Recommend watching the first real voice ABA: if it REDs on
`settlement/model_steps` or `rnnt_blank_frames` with an otherwise-identical settling
pattern, delta-normalize the blank family relative to `starting_blank_frames` too
(and add a divergent-starting-blank control test).

### Verdict

**APPROVE.** The A48 fix correctly and thoroughly addresses the nested
absolute-position bookkeeping: it delta-normalizes the clock/token position offsets,
strips the `*_frame` fields, retains counts/flags/step-count/model output, handles
malformed shapes safely, keeps validation before normalization, and the offset and
count-divergence tests are meaningful — no laundering, no over-normalization. One
low-risk residual remains (absolute blank-count starting offset in the settlement
summary and `rnnt_blank_frames`), defensible as behaviour and unlikely per r17's
stable settlement; it is a watch-item for the first real voice ABA, not a blocker.
D8 intact (harness-only).

## Addendum 50 — GREEN r2 voice control: post-run review + architecture conclusion (2026-08-09)

Post-run review of the completed voice diagnostic
(`voice-function-logits-r2-aba`) against all three run reports/logs and the direct
r17 report. Review-only, no source/GPU/commit.

### The GREEN is honest and independently reproduced

- **Identity/provenance/return codes.** `passed:true`, image
  `sha256:9e8c03e…` (production candidate), identical across off-a/on/off-b; all
  three `voice_reproduction_valid`, `structural_passed`, `carrier.passed`, exact
  `get_weather(Tokyo)`; return codes `{off-a:0, on:0, off-b:0}`; off-a/off-b
  schema 1, on schema 2.
- **Neutral ABA.** neutrality `passed:true`, **0 violations**, `stable_field_count
  861`, `discrete_outcome_equal:true`. Audio `102312/traced 100548/100548` is
  1764-quantized (58/57/57 frames), inside the inclusive envelope, and delta-bounded
  (1 frame ≤ 1 frame). The operational-timing counts (105/106/103) are retained
  under `operational_timing`, non-gating — as designed. The A49 blank-offset residual
  did **not** materialize: with 0 violations the settlement was reproducible, matching
  r17's stable settlement.
- **Recomputed trace (independent).** Voice on, settlement+EOU: 10 entries,
  **min SOTC rank 1**, **max SOTC−PAD +0.9375**, raw-SOTC-argmax **2**,
  staged/replayed **1/1**, tool fires. Direct r17 c014: 12 entries, min rank **2**,
  max SOTC−PAD **−2.9375**, raw-argmax **0**, staged/replayed **0/0**, no call. A
  ~3.9-logit swing at SOTC's closest approach, categorical (rank 1 vs 2, +vs−
  margin, staged vs not, tool vs none). All numbers match the reported values.

The causal reading is sound: the same c014 intent, delivered through real acoustic
frames that build perception/RNNT state, elevates SOTC to argmax and stages/replays
it into a tool call; delivered through inference-only hidden-token injection (clocks
held, no perception), it never becomes argmax. The trace is neutral, so the
observation did not cause the difference.

### Architecture conclusion — APPROVE the scoped characterization

- **No overclaim.** The correct statement is *not* "direct text input is
  impossible." It is: **inference-only direct-text token injection, as implemented
  against this frozen checkpoint and inference interface, is distribution-shifted —
  it does not reproduce the acoustic/perception conditioning that elevates SOTC, so
  it cannot be treated as equivalent to an acoustic user turn.** The report's own
  conclusion string (`direct_text_as_implemented_is_distribution_shifted`) is
  correctly scoped and the comparison gate is constrained to never claim
  fundamental unfixability. What is demonstrated is a property of *this
  checkpoint + this injection method*, one 20-case corpus, c014 as the anchor — not
  a theorem about text inputs.
- **Alternative inference-only bridges — none faithful.** The only inference-only
  ways to close the ~3.9-logit gap are: (a) route text through the model's own
  synthesis so it arrives as acoustic frames — that *is* the typed/voice path, which
  already works and is in-distribution; (b) fabricate perception frames from text —
  which is precisely the trained adapter, not inference-only; or (c) boost/force the
  SOTC logit or manipulate the gate — a D8 violation and pure laundering. So there is
  no clean inference-only injection bridge; the faithful inference-only path is the
  acoustic one.
- **System prompt is a different surface.** Initial system-prompt text is supported
  **prefill** (prompt conditioning the checkpoint is trained for), not a user
  acoustic turn; it remains valid and is unaffected by this finding.

### Immediate safe implementation recommendation

Production **user** text input should retain the **acoustic-conditioning path**
(typed text → TTS synthesis → acoustic frames → the proven settle/EOU/response
contract), which is in-distribution and fires tools correctly. Do **not** ship
inference-only direct-token injection for user turns. Keep initial system-prompt
text as supported prefill. Defer any native direct-text surface until a trained
text-to-perception adapter or native multimodal text interface exists and is fully
qualified (its own ABA/behavioral qualification), at which point it can be re-tested
with this same neutral-trace harness.

### Verdict

**APPROVE.** r2 is an honest GREEN neutral positive control: identities/provenance/
return codes are consistent, the ABA is genuinely neutral (0 violations, 861 stable,
exact call in all three runs, audio within envelope), and the independently
recomputed voice (rank 1, +0.9375, staged/replayed 1/1, fires) versus direct r17
(rank 2, −2.9375, 0/0, no call) contrast is decisive. Characterize the outcome as
"not supported by this checkpoint/inference interface via inference-only injection
(distribution-shifted)," never "impossible." Recommendation: retain the acoustic
(typed-TTS) path for production user text, keep system-prompt text as prefill, and
gate any native text bridge behind a trained, fully-qualified adapter. D8 intact.

## Addendum 51 — r2 comparison-window bug (correction to Addendum 50) (2026-08-09)

Follow-up to Addendum 50 after a comparability bug was surfaced. Review-only, no
source/GPU/commit.

### The bug is real (report-comparability)

The r2 **stored** `comparison` contrasted mismatched decision surfaces:
`comparison.voice.entry_count = 40` (voice_input + settlement + eou_boundary +
post_bos) versus `comparison.direct.entry_count = 12` (settlement + eou_boundary
only). Summarizing SOTC rank/margin/argmax over different phase sets on the two
paths is apples-to-oranges — the voice window included 30 extra positions the
direct probe cannot have (it has no acoustic input phase), so the stored r2
comparison is self-inconsistent and must not be the locked artifact.

### It does not change the substantive result

I recomputed the voice metrics under both windows from `on/report.json`:

| window | entries | min SOTC rank | max SOTC−PAD | raw-SOTC-argmax |
|---|---|---|---|---|
| all 4 phases | 40 | 1 | +0.9375 | 2 |
| settlement+eou (like-for-like) | 10 | 1 | +0.9375 | 2 |

The values are **identical**, because every raw-SOTC-argmax and every rank-1
position lives in `settlement`/`eou_boundary` — none in `voice_input` or
`post_bos`. So on the correct like-for-like window the voice side is still
rank 1 / +0.9375 / raw 2 / tool-fires, versus direct rank 2 / −2.9375 / raw 0 /
no-call; `voice_positive`, `direct_rank_two`, the ~3.9-logit swing, and the
`distribution_shifted` conclusion all stand unchanged. The architecture
characterization from Addendum 50 is unaffected.

### Source is already fixed and tested

The current source applies `_margin_summary` (settlement+eou only) to **both**
sides (`voice_function_logit_diagnostic.py:695` and `:704`, with a comment that the
direct probe must not be contrasted against voice_input/post_bos), and
`test_margin_summary_compares_only_settlement_and_eou_boundary` pins the phase set.
The defect existed only in the code that produced the r2 artifact; the r2 stored
`comparison` predates the fix.

### Verdict

**REQUEST CHANGES — narrow and artifact-scoped.** The source correction is right
and regression-tested, and the substantive causal result is provably unchanged, so
Addendum 50's characterization and recommendation stand. But the **locked
architecture decision must rest on a like-for-like comparison artifact**: re-run the
voice diagnostic (or re-derive `compare_direct_voice` offline from the stored
`on/report.json`) with the fixed symmetric window, and record it in place of the
stale 40-vs-12 comparison. Recommend the comparison report also emit the phase set
used per side (e.g. `"phases": ["settlement","eou_boundary"]`) so window symmetry is
self-evident in the artifact and cannot silently regress. Once the symmetric
comparison is regenerated, the GREEN r2 conclusion — inference-only direct-token
injection is distribution-shifted; retain the acoustic path for production user text;
gate any native text bridge behind a trained, fully-qualified adapter — is safe to
lock. D8 intact.

## Addendum 52 — A51 fix + architecture-doc re-review (2026-08-09)

Re-reviewed the A51 correction and the revision-6 docs; review-only, no
source/GPU/commit.

### Symmetric comparison artifact — validated

`comparison-symmetric.json` is a re-derivation wrapper
(`rederived_direct_voice_function_logit_comparison`) whose nested
`direct_voice_function_logit_comparison` now uses **identical phase lists**
`["settlement","eou_boundary"]` on both sides. Like-for-like: **direct 12 entries**
(rank 2, −2.9375, raw 0, no call) vs **voice 10 entries** (rank 1, +0.9375, raw 2,
tool fires) → `direct_rank_two_pad_argmax:true`,
`voice_sotc_argmax_and_expected_call:true`, `passed:true`, conclusion
`…distribution_shifted`. I re-hashed all three source bindings and they **match**
the retained artifacts: `source_direct_diagnostic` → r17 `report.json`
(`e6f62f18…`), `source_voice_diagnostic` → r2 `report.json` (`77522411…`),
`source_voice_report` → r2 `on/report.json` (`2ee4f10b…`). `runtime_image_id`
(`9e8c03e…`) and `corpus_sha256` (`947827f4…`) match r2 and the checked-in corpus.
The **old `report.json` is unmodified** (its `comparison.voice.entry_count` is still
40, no `phases` field) — the correction is an additive, provenance-bound artifact,
not a history rewrite. Source now emits an explicit `phases` field from
`_margin_summary` applied to both paths, `rederive_comparison` writes the SHA
bindings, and `test_margin_summary_compares_only_settlement_and_eou_boundary` pins
the phase set. A51's comparability requirement is fully satisfied.

### Docs revision 6 — no overclaim, correct scoping

- **No overclaim.** `architecture.md`: "distribution-shifted for the released
  checkpoint/interface, not that native text is fundamentally impossible";
  `direct-text-input-plan.md`: "does not prove that native text input is impossible…
  the current inference-only source-token injection is distribution-shifted for this
  checkpoint." Both correct.
- **Production Pocket only.** Docs reference the "qualified Pocket carrier," the
  isolated CPU-only worker, and the frozen base seed — production Pocket, no dev
  variant.
- **Direct is diagnostic/probe-only.** "Keep that experiment diagnostic-only"; the
  shipped user-text path is `text → Pocket TTS → perception/RNNT → explicit EOU`.
- **System prefill separate.** "Initial system-prompt text remains a separate,
  supported prefill surface… not evidence" — distinct from user turns.
- **Synthetic PCM unpaced but server serialized/framed.** "faster-than-realtime
  synthesis" (unpaced generation) fed by "serialized model advancement, never
  wall-clock pacing or endpointing," via ordered start/commit barriers and the
  1280-sample RNNT path — accurate.
- **Anti-misread gate present.** `qualification.md`: "Reports distinguish
  `typed_pocket` from microphone PCM and system prefill so a green matrix cannot be
  mistaken for direct-text injection evidence" — the correct safeguard against a
  green typed/voice matrix being read as direct-text support.
- **Evidence cited at the corrected surface.** The plan cites "the like-for-like
  settlement/EOU decision surface" and "all three runs," matching the symmetric
  artifact rather than the stale 40-entry window.
- **Status honest.** "revision 6, architecture decision recorded; final
  qualification in progress" — the decision (retain acoustic, keep direct
  diagnostic-only) is a conservative default that stands; the future native path is
  correctly gated behind "a trained text-to-perception adapter or checkpoint-provided
  multimodal interface and full requalification."

### Minor (non-blocking)

- `_margin_summary`'s `phases` label is a hardcoded `["settlement","eou_boundary"]`
  rather than derived from the actually-selected entries; the regression test pins
  both the filter and the label, so drift is caught, but a derived label would be
  more robust.
- The stale 40-entry comparison persists in the old `r2/report.json` (correctly not
  rewritten); `comparison-symmetric.json` plus the like-for-like doc framing are the
  authoritative reference. A one-line pointer from the run directory would remove any
  chance of a reader mis-citing the old window.

### Verdict

**APPROVE.** The A51 fix is correct and complete: the symmetric comparison uses
identical settlement+EOU windows on both paths (direct 12 / voice 10), is bound by
matching SHA256s to the exact retained r17+r2 artifacts, preserves the
image/corpus identity and the GREEN distribution-shifted conclusion, and leaves the
original report untouched; the source emits explicit phases and is regression-
tested. The revision-6 docs are correctly scoped — no overclaim, production-Pocket
only, direct injection diagnostic-only, system prefill separate, synthetic PCM
unpaced-but-serialized/framed — with the anti-misread qualification gate intact and
an honest in-progress status. The architecture decision (retain the acoustic path
for production user text; keep direct token injection diagnostic-only; gate any
native text bridge behind a trained, fully-qualified adapter) rests on the
corrected like-for-like evidence and is safe to lock. D8 intact.

## Addendum 53 — memory-matrix / final qualification review (2026-08-09)

Review-only, no source/GPU/commit. Read `memory_matrix_strict_v3.py`,
`transcribe_sustained.py`, `run_live_suite.py`, `generate_multiturn_fixtures.py`,
`qualification.md`, and the tests independently (reported 475 pass / 8 skip / Ruff).

### Verified

- **Exactly six seeded modality pairs + negative.** `load_cases` requires the
  observed `(seed,recall)` set to equal `{voice,text,system}×{voice,text}` and
  `len==6`, plus one unseeded negative — covering voice→voice, voice→text,
  text(Pocket)→voice, text→text, system-prefill→voice, system-prefill→text.
- **Unique per-case secrets, one word each; recall prompts leak none.** Secrets are
  validated non-empty single words and unique; every case's and the negative's
  recall text is rejected if it contains any matrix secret.
- **Negative runs last.** The runtime gate requires
  `negative_control == [F,F,F,F,F,F,T]` and `negative_control_ran_after_seeded`.
- **Own-secret-only on both channels, independently.** Runtime `case.passed`
  requires the model **text** to contain exactly its own secret (none for the
  negative). The independent **Parakeet `.nemo` ASR** of the actual EarTTS response
  audio separately enforces own-secret inclusion and exclusion of all other secrets
  via `response_semantic_matches`, decoupled from the text-channel WER; the negative
  control's recall audio must exclude every secret.
- **All 11 WAVs, no near-silent allowance, provenance.** `response_quality` requires
  `audio_bytes > 0` (and balanced brackets) for all 11 turns; the memory-matrix ASR
  gate runs with `max_silent_rate = 0.0`, so any near-silent response fails, and
  `bool(prepared)` prevents an all-silent/empty vacuous pass. Provenance is gated
  (`valid_runtime_provenance` + `runtime_provenance_consistent`, fresh session per
  case). `typed_pocket` is labeled `text`/`typed` and never `direct`; the matrix
  never invokes direct injection, and `qualification.md` distinguishes typed_pocket
  from microphone and system prefill.
- **Attacks clear.** Response races are caught (overlapping / correlation-mismatch /
  incomplete → raise); WAV paths are unique per `case_id-turn-N`; WER and the audio
  secret check are decoupled; system-prefill cases correctly carry one recall turn
  (secret in prefill, no seed turn) → 4×2 + 2×1 + 1 = 11 responses; fresh sessions
  isolate secrets so the (intended) vacuous exclusion on the 4 seed turns cannot
  leak a cross-case secret; the memory-matrix result is ANDed into the suite verdict.

### Minor (non-blocking)

- **Partial `--case` gate tolerance.** `memory_matrix_runtime_gate_passed` enforces
  the negative-last / 7-case checks only when `expected_case_count == 7`; a debug
  `--case` subset passes its own subset without the negative control. The suite
  invocation runs the full matrix (no `--case`), so qualification is always the full
  7, but the suite gate should defensively assert `expected_case_count == 7` (require
  full-matrix) so a future edit adding `--case`, or a misread of a partial run, can
  never be mistaken for a full qualification.
- **Semantic secret-enforcement is on the 7 recall turns**, not all 11 (the 4 seed
  turns are audio-fidelity/WER/near-silent only, with vacuous positive+exclusion by
  design and fresh-session-isolated). Worth documenting so the 11-WAV audio gate is
  not read as 11 secret checks.

### Verdict

**APPROVE.** The memory matrix is fail-closed and covers exactly the required six
fresh seeded cells plus a trailing unseeded negative, with unique per-case secrets,
non-leaking recall prompts, own-secret-only enforced independently on the model text
channel and on Parakeet ASR of the real EarTTS audio (with exclusion and a negative
control), all 11 response WAVs required non-empty with zero near-silent allowance,
provenance gated, and typed_pocket never mislabeled as direct. No vacuous-pass,
race, path-collision, or turn-accounting hole was found. Two non-blocking hardening
notes: add an explicit full-matrix (`expected_case_count == 7`) assertion in the
suite gate, and document that the audio secret check binds the 7 recall turns. D8
intact.

## Addendum 54 — final-step hardening re-review (2026-08-09)

Review-only, no source/GPU/commit. Inspected the current diff to
`transcribe_sustained.py`, `run_live_suite.py`, and the tests.

### Memory gate — A53 #1 closed

`memory_matrix_runtime_gate_passed` now requires, unconditionally,
`expected_case_count == 7` **and** `len(cases) == 7` **and** all-cases-passed
**and** `fresh_session_per_case` **and** valid+consistent provenance **and**
`negative_control_ran_after_seeded` **and** the exact
`[False,False,False,False,False,False,True]` order. The prior
`not full_matrix or …` escape is gone, so a partial `--case` run (expected ≠ 7)
can no longer satisfy the qualification gate. This correctly separates partial
diagnostics (each case still writes its own `report.json` result for debugging)
from qualification semantics (the aggregate gate demands the full six-seeded +
trailing-negative matrix). It is strictly tighter than before — no syntax or gate
regression; my focused rerun of the memory tests passes (9/9), consistent with the
reported 69.

### Pocket Torch-RNG preflight — image node executes

The determinism/RNG-restore test
`test_pocket_worker_ipc.py::test_internal_text_carrier_is_deterministic_per_text_and_restores_rng`
guards with `pytest.importorskip("torch")`, so on the Torch-free host it skips
(correct — no false host pass). `run_conversion_tensor_tests` runs `docker run
--rm --network none` in the pinned runtime image and invokes `pytest -q -c
/dev/null -p no:cacheprovider` on the conversion nodes plus **that exact node
id**, under `run_checked` (check=True → non-zero fails the stage);
`test_conversion_tensor_tests_run_offline_in_runtime_image` pins the offline
command and the exact node. Inside the image torch is present, so the only skip
path (`importorskip`) does not fire and the test body executes; `-c /dev/null`
prevents any config-injected skip/addopts; and the reported real image stage
passed — empirical evidence the node ran, not a hidden double-skip. Image-node
collection therefore actually executes.

### Minor (non-blocking)

- **Partial-rejection unit coverage.** The gate logic requires `expected == 7`, and
  `test_full_runtime_gate_requires_negative_control_last_and_nonvacuous` exercises
  the negative-order and case-pass failure modes at 7 cases, but I did not locate a
  unit test that sets `expected_case_count != 7` (e.g. 6) and asserts the gate
  rejects. If the intended partial-case-rejection test targets that branch, confirm
  it is present (or add a one-line `expected != 7 → not passed` assertion) so the
  exact hardening is regression-locked.
- **Skip-blind stage.** `run_conversion_tensor_tests` relies on returncode 0 plus
  the fact that the RNG node has no in-image skip path; a future skip marker on that
  test would exit 0 and silently green the stage. A defensive `-rs` plus an
  assertion that the node reports `passed` (not `skipped`) in
  `conversion-tensor-tests.log` would harden it. Safe today.

### Verdict

**APPROVE.** Both hardenings are correct and regression-free: the memory gate now
demands exactly the seven-case matrix with the six-false/one-true negative order, so
partial diagnostics can never be mistaken for a qualification; and the Pocket
Torch-RNG determinism test skips only on the Torch-free host while executing as the
exact node inside the pinned runtime image (offline, no-GPU, check=True, config
disabled, real stage passed) with no hidden skip. Two non-blocking test/observability
hardening notes (an explicit `expected != 7` rejection assertion, and a skip-visible
conversion stage). D8 intact.

## Addendum 55 — correction to Addendum 54 (2026-08-09)

Review-only, no source/GPU/commit. Two A54 items corrected after being pointed to
the right locations.

- **A54 minor #1 was wrong — the explicit `expected != 7` test exists.**
  `tests/runtime/test_transcribe_sustained.py::test_memory_matrix_asr_gate_rejects_partial_case_report`
  builds a report with `expected_case_count = 1`,
  `negative_control_ran_after_seeded = False`, and a single case, and asserts
  `not runtime_gate_passed(report)` — directly exercising the hardened
  `expected == 7` rejection branch. I missed it because I searched only
  `test_memory_matrix_qualification.py`; the gate function lives in
  `transcribe_sustained.py`, and so does this test. The hardening is
  regression-locked. Both uv and locked-host focused runs pass 69.
- **Skip-blind stage is empirically not skipping.** The retained
  `live-suite-smart-turn-memory-final-r1/conversion-tensor-tests.log` reports
  "29 passed, 3 warnings" with **zero skipped**, so the exact Pocket Torch-RNG node
  ran inside the pinned image on the real stage — confirming image-node execution.
  The remaining A54 minor #2 (a *future* skip marker could exit 0 and silently green
  the stage; `-rs` + a "not skipped" log assertion would harden it) stands as a
  non-blocking observation only.

### Verdict

**APPROVE.** With the correction, both final-step hardenings are fully verified:
the memory gate demands exactly the seven-case matrix with the six-false/one-true
negative order and has an explicit partial-case (`expected != 7`) rejection test,
and the Pocket Torch-RNG determinism node empirically executed (29 passed / 0
skipped) as the exact node inside the pinned runtime image while skipping only on
the Torch-free host. Only the non-blocking future-skip-marker observability note
remains. D8 intact.

## Addendum 56 — multiturn ASR dispatcher repair + recovery review (2026-08-09)

Review-only, no source/GPU/commit. Inspected the retained
`live-suite-smart-turn-memory-final-r1` and the current code.

### Bug confirmed

The multiturn report has neither `matrix` nor `sent_frames`, so the old
`runtime_gate_passed` (`"matrix" in report → memory; else sustained`) fell to the
sustained branch and returned `False` on the missing `sent_frames` — despite
`external_asr.passed = True` (Parakeet transcribed all 3 WAVs, 3/3). A false-RED
from shape-guessing.

### Dispatcher repair — implemented, sound, tested (APPROVE)

The repair is already in place and I verified it against the retained artifacts:
- All three tools emit `report_kind` (`multiturn_strict_v3`, `memory_matrix_strict_v3`,
  `sustained_strict_v3`).
- `report_kind()` returns the explicit kind only if it is in the known set (a forged
  `"future_report"` → `None`), else strict `infer_legacy_report_kind`, which appends a
  candidate only when the full discriminator subset is present and returns a kind
  **only if exactly one** matches (ambiguous/none → `None`).
- `runtime_gate_passed` dispatches to per-kind recompute gates and returns `False` for
  `None` (unknown/ambiguous fail closed). Each gate re-derives from raw evidence and
  ignores the mutable `passed`/`external_asr`. `multiturn_runtime_gate_passed` mirrors
  the tool's invariants (exactly 3 responses, each `audio_bytes>0`, non-empty text,
  `input_wer ≤ max`, `semantic_match is True`, `session_closed` present) and is if
  anything stricter (type + audio checks), so it does not false-green; `memory` keeps
  the exact-7 + `[F×6,T]` gate; `sustained` keeps the frame/typed invariants.
- Empirically, the fixed dispatcher classifies the retained pre-kind reports uniquely
  and all pass: multiturn `False→True` (fixed), memory `True`, sustained `True`.
- Tests cover each kind, legacy recognition without trusting the prior verdict,
  unknown + ambiguous → fail closed, and decisive-field mutation for all three gates.

No false-green or false-red path found. One minor: `multiturn_runtime_gate_passed`
defaults `expected_turn_count` to 3 (the retained report omits it); the tool should
emit `expected_turn_count` explicitly so the count check never rests on a default.

### Recovery — prefer a standalone report over a resume mode (REQUEST CHANGES)

The resume-offline-ASR mode is **not implemented**. On the design question: **a
smaller standalone recovery report is more trustworthy than a `run_live_suite`
resume mode.** The defect being fixed is precisely a *mode/shape confusion*; adding
a resume mode to the already multi-mode 2313-line orchestrator enlarges exactly that
surface and risks a composite recovery being read as a fresh cold-start
qualification. A standalone recovery keeps the trust boundary explicit and auditable.

Because the ASR is deterministic (Parakeet greedy over fixed WAVs) and the runtime
gates now recompute idempotently from the retained reports, recovery only needs to
re-run the three ASR commands (via the fixed `transcribe_sustained`) and re-derive
the gates. Any recovery — standalone (preferred) or resume — must, as conditions:
1. Verify every retained artifact (WAVs, reports, manifests) against recorded
   SHA256s and fail closed on any mismatch; never mutate the original reports.
2. Recompute **all** pre-ASR runtime gates from the retained reports via the shared
   `runtime_gate_passed` functions — never trust the original `passed`.
3. Re-run only the three ASR commands; never rerun model runtime.
4. Emit a **distinct** recovery/composite report kind that **enumerates which gates
   were re-verified (ASR + per-kind runtime invariants + artifact hashes) versus
   inherited-and-not-re-verified from r1 (cold-start timing, restart cycles, browser,
   live-stack component gates)**, and never claims to reproduce cold-start timing or
   to be a fresh full-suite pass.
5. Fail closed on unknown/ambiguous report kind or any gate-recompute failure, and be
   independently tested (hash-mismatch, tampered inherited gate, unknown shape).

The inherited gates are the residual trust: recovery re-establishes ASR + runtime
invariants under hash integrity but cannot re-derive the live-stack/timing gates, so
the recovery verdict must present them as *inherited*, not *re-verified*.

### Verdict

**APPROVE the dispatcher repair; REQUEST CHANGES on the recovery approach.** The
`report_kind` + per-kind recompute + strict legacy inference + fail-closed dispatch
correctly and idempotently fixes the multiturn false-RED with no false-green path and
comprehensive adversarial tests (add explicit `expected_turn_count` as a minor
hardening). For recovery, implement a **standalone** recovery report rather than a
`run_live_suite` resume mode, meeting conditions 1–5 above — chiefly: hash-verify
retained artifacts, recompute all gates from retained evidence (not the stored
`passed`), re-run only the three ASR, and emit a distinct report that explicitly
separates re-verified gates from the inherited cold-start/timing/live-stack gates it
cannot reproduce. D8 intact.

## Addendum 57 — standalone offline-ASR recovery review (2026-08-09)

Review-only, no source/GPU/commit. Inspected `recover_offline_asr.py`, the
read-only `--output-dir` mode in `transcribe_sustained.py`, and
`test_recover_offline_asr.py`.

### A56 conditions 1–5 — all met

- **(1) Isolation / no source mutation.** `evaluator_command` mounts the source
  qualification dir `…:/qualification-input:ro`, a separate writable
  `…:/qualification-output`, and the ASR model `:ro`, with `--network none`.
  `transcribe_sustained` in `--output-dir` mode deep-copies the source report, reads
  WAVs from the `:ro` run dir, and writes the evaluated report and the 16 k WAVs to
  the *output* dir — so the source report is untouched both by the `:ro` mount and by
  the writer. `preserved_unmodified` is backed by the before/after hash of the source
  report.
- **(2) Hash before + verify after.** `collect_consumed_inputs` hashes every consumed
  report, response WAV, and manifest; the ASR model and evaluator script are hashed in
  the input snapshot; `verify_snapshot` re-hashes all consumed inputs after the ASR
  runs and a re-collected `inherited` is compared for equality — a mid-recovery
  mutation raises. The `:ro` mount blocks in-container writes; host mutation is
  detected.
- **(3) Inherited evidence + independent gate recompute.** `collect_inherited_evidence`
  hashes and re-reads the top-level, restart, browser, and component/live-stack
  artifacts and rejects a stored-red browser/restart verdict. Runtime gates are
  recomputed by the shared `runtime_gate_passed` both at collection and on the
  re-evaluated output, never trusting the stored `passed`/`external_asr`.
- **(4) Immutable image.** `resolve_runtime_image_id` requires a `sha256:`-prefixed
  71-char digest and the recovery uses that ID, never the mutable tag.
- **(5) Exactly three offline commands + honest report.** One `--network none`
  Parakeet evaluator per qualification; the `offline_asr_recovery_v1` report separates
  `reverified` (integrity, per-kind runtime gates, external ASR) from
  `inherited_not_reexecuted` (cold-start timing "not recoverable", restart, browser,
  component/live-stack logs), and limitation #3 states the exact boundary — hashes
  establish integrity *across this recovery*, not provenance before the snapshot.
  Unknown/ambiguous kinds fail closed (`collect` raises on kind mismatch;
  `runtime_gate_passed` returns False for `None`). `main()` writes a fail-closed error
  report on any exception.

### Attacks — clear

- **Path-resolution escapes:** confined — after `resolve_audio_path`,
  `collect_consumed_inputs` requires `audio_path.resolve().is_relative_to(run_dir)`
  (raises "audio_path escapes its run dir") and the same for the manifest
  (`is_relative_to(source_run)`); `test_consumed_input_rejects_audio_path_escape`
  exercises an out-of-tree `audio_path` and passes.
- **Time-of-check:** hash-before / ASR-reads-`:ro` / hash-after + inherited re-check
  closes the window; container cannot mutate, host mutation is caught.
- **Incomplete artifacts:** `file_record` raises on missing; `collect` raises on
  no-responses, empty (`≤44`-byte) WAV, or missing `audio_path`.
- **False green:** the retained *and* re-evaluated runtime gates are enforced by
  raises, all three external-ASR gates must pass, a red inherited gate raises, and the
  image ID must be immutable — no green path around any of these.
- **Misleading claims:** the report never claims a fresh full-suite run and discloses
  the inherited/hash-provenance limitations explicitly.
- **Source immutability:** the container truly cannot write (`:ro`), the writer targets
  the separate output, and host-side mutation is detected via before/after hashing —
  honestly disclosed rather than overclaimed as write-locked.

Tests cover command isolation (`:ro`, `--network none`, immutable ID,
`--output-dir`), hash mutation, inherited red gate, and path escape; all 4 pass.

### Minor (non-blocking)

- `report["passed"] = all(external_asr.passed)` relies on the runtime-gate raises for
  the pre-ASR verdict rather than ANDing the recomputed gate booleans into `passed`;
  correct (a red gate raises before the report), but adding the gate booleans and a
  `len(results) == len(QUALIFICATIONS)` check would make `passed` self-contained and
  guard the theoretical vacuous `all([])`.
- Carried from A56: the multiturn gate still defaults `expected_turn_count` to 3;
  emitting it explicitly from the tool would remove the default.

### Verdict

**APPROVE.** The standalone recovery meets every A56 condition and is fail-closed
against path escapes, time-of-check mutation, incomplete artifacts, false green, and
misleading claims: source qualification dirs are `:ro` with a separate output, all
consumed and inherited artifacts are hashed before and re-verified after, runtime
gates are recomputed independently, the immutable image ID is used, only the three
`--network none` Parakeet commands run, and the distinct `offline_asr_recovery_v1`
report honestly separates re-verified from inherited-and-not-re-executed evidence
while disclosing that hashes prove recovery-time integrity, not pre-snapshot
provenance. The source cannot be mutated by the container and host mutation is
detected. Two non-blocking clarity notes (make `passed` self-contained; emit
`expected_turn_count`). D8 intact.

## Addendum 58 — sustained tool-turn RED diagnosis (2026-08-09)

Diagnosis only, no source/GPU/commit. Initial hypothesis (FC-state-not-reset)
tested against the counterevidence and **withdrawn**.

### Observed pattern (r1)

The sustained session (17 responses, closed on the expected 12000-frame
`session_position_limit`) repeats math / memory-ack / memory-recall / tool every
four turns. The tool turns are responses 3/7/11/15. In the event stream there is
exactly **one** `response.function_call_arguments.done` (`get_current_utc_time`) and
**one** `function_call_output.applied` — both at response 3 — then responses
7/11/15 close with `no_text_since_bos_watchdog` and are near-silent. Response 3
itself has empty assistant text and garbled/near-silent audio; every math/memory
turn is correct.

### Counterevidence refutes an FC lifecycle/state bug

- Two other sustained runs — `voicechat-public-live-20260805-downloaded-r18` and
  `…-converted-sustained-r4` — pass **18/18** with the identical repeated tool
  prompt and the same code/FC lifecycle.
- Decisively, only turn 4 carries `expected_response_any` (a fresh call). In the
  passing runs response 3 makes the call and speaks the correct time
  (`['14:57 UTC']` → "The current UTC time is fourteen fifty seven UTC."), and
  responses 7/11/15 have `expected_response_any = []` and **repeat that time from
  conversation context** — they are *designed* context-recall turns, not fresh
  calls. r1 has the same shape: resp 3 `['11:07 UTC']`, resp 7/11/15 `[]`.
- So "no function call at turns 8/12/16" is the intended behavior, not a
  suppression bug. There is no reachable latch: `active`/`awaiting_response`/
  `injecting_response`/`background_active`/`forced_tokens` were all false after
  frame 2319, and `completed_calls` is a Python bookkeeping list — set in the
  wrapper, cleared on FC-cycle reset (`streaming_s2s_pipeline.py:865/1081/1124/
  1175/2217`), and read only at `:1441–1445` for `_sync_handled_calls` and exposed
  as a `len(...)` telemetry field. No SOTC/logit path consumes it, so it cannot
  block a subsequent call.

### Diagnosis

This is an **expected public-model stochastic failure at turn 4's first
function-call response generation**, not a runtime FC lifecycle bug and not a
harness-association artifact. Turn 4's fresh call fired and its output was applied,
but the two-phase FC response injection landed off-distribution ("My plan almost" /
near-silent) instead of speaking the fetched time, so **the correct time never
entered the conversation context**. Responses 7/11/15, which by design answer that
tool result *from context* (empty `expected_response_any`, no fresh call), then had
nothing to recall → no text → `no_text_since_bos_watchdog`. The two 18/18 passing
runs on the same code prove the intended chain (turn-4 call → context → 8/12/16
recall) works; r1 drew a stochastic miss at the single turn-4 generation.

### Next action

**Bounded fresh-stack sustained rerun — not a code change.** `qualification.md:107`
explicitly permits resuming gates on fresh stacks after a retained stochastic model
failure, and this is exactly that: one turn-4 generation miss with two same-code
18/18 controls. Do **not** clear conversation history: turns 8/12/16 answering from
context is the correct intended semantics the passing runs rely on, and clearing it
would force needless re-calls and change conversation behavior — there is no causal
code branch to fix (FC flags clean, `completed_calls` non-causal). Keep it a single
bounded rerun (per the bounded-evidence policy, not retry-until-pass); do not relax
the ASR threshold or drop the tool turn. If a bounded rerun also fails at turn 4,
that would raise the observed base rate and warrant instrumenting the turn-4 FC
two-phase response injection specifically — but the current evidence is a single
stochastic miss, so the honest step is the allowed rerun.

### Verdict

**No code change. Bounded fresh-stack sustained rerun (qualification.md:107).** The
sustained RED is a public-model stochastic failure localized to turn 4's first
function-call spoken response, cascading through the context-recall turns; it is not
an FC lifecycle/state bug (refuted by two 18/18 same-code runs, clean post-frame-2319
FC flags, and non-causal `completed_calls`) and not a harness artifact. D8 intact.

## Addendum 59 — turn-4 tool-response failure recurs on the allowed rerun (2026-08-09)

Diagnosis + plan only, no source/GPU/commit. The one allowed fresh-stack rerun
reproduced the turn-4 failure, so A58's "single stochastic miss → rerun" is
superseded: the honest task is now to localize the mechanism from retained evidence.

### What the per-frame telemetry shows

At the first tool turn (frame 5225) — and identically at frames 8143 and 11065 —
the retained model-step log shows:
- The tool response is **correctly injected**: `function_text` carries the
  well-formed `<TOOLCALL>…get_current_utc_time…</TOOLCALL>` +
  `[INJECTED_RESPONSE]/[CONVERTED_RESPONSE] {"timezone":"UTC","spoken":"11:07 UTC"}`;
  `forced_tokens` is 0 (the forced response tokens were consumed) and
  `function_calling` is clean (`active/awaiting/injecting/background` all false,
  `completed_calls:1`).
- The model then produces **no agent text**: `agent_control:"pad"`,
  `decoded_token_count:0`, `turn_text_tokens:0`, `agent_talking_frames:29`,
  `assistant_delta:""`, for 30 frames until
  `agent_silence_watchdog.request_reason:"no_text_since_bos_watchdog"` fires and the
  next frame is `agent_control:"agent_eos"`, `response_boundary.eos_reason:
  "no_text_since_bos_watchdog"`. EarTTS still advanced (`eartts.generated_tokens:31`)
  and the recovery ASR heard "My plan almost", so audio was produced with **no
  corresponding Nano text** — a text/audio desync, not silence-everywhere.

So the failure is **post-injection agent-text generation collapsing to PAD** while
EarTTS emits off-distribution audio. It is not the injection (well-formed,
consumed), not an FC-state latch (state clean), and not the response boundary (the
no-text watchdog is the correct *symptom* of no text, not the cause).

### Challenge to the three proposed explanations

- **Deterministic temperature/seed — does not explain it.** The provenance confirms
  greedy (`sampling_contract {temperature:0.0, top_p:1.0, repetition_penalty:1.0}`),
  but `downloaded-r18` passed **18/18 on the identical `model_sha256`
  d553750c…** production candidate. Under bit-determinism the same checkpoint cannot
  both pass 18/18 and fail 2/2, so the decode is not run-reproducible — CUDA
  non-determinism under greedy (already seen in r17 per-token variance) is active at
  the post-injection "speak the answer vs PAD" first-token boundary. Determinism
  therefore *cannot* be the cause of two failures; it is a fragile near-boundary
  decision plus non-determinism, and two consecutive misses point to an **elevated
  failure rate in this run's context**, not a fixed seed.
- **Tool-response token injection — not the cause.** The injected/converted response
  is well-formed and identical in content, `_fc_convert_num_to_text` left the spoken
  field intact, and `forced_tokens` returned to 0 — the injection completed
  correctly. The differing value ("11:07" vs the controls' "14:57"/"22:06") is the
  same shape.
- **Response boundary — symptom, not cause.** The boundary correctly waited 30 frames
  and closed on the no-text watchdog; it did not truncate a live text stream (there
  was none).

### Narrowest evidence-driven plan (no code edit, no further retry)

1. **Parakeet-transcribe `response-004.wav` from both r1 failures** and reconcile the
   "substantial audio" against the ‑120 dBFS mid-turn frames — confirm whether the
   audio is the "My plan almost" off-distribution utterance throughout. This fixes
   whether EarTTS spoke garbage or fell silent.
2. **Diff turn-4 context r1-fail vs r18-pass**: the rendered prompt bytes
   (the sustained session logged 1204; compare to r18), the exact injected token
   sequence, and the pre-turn-4 conversation/session state — find what shifted the
   first-token decision toward PAD.
3. **Compare the two r1 failures' turn-4 Nano token streams**: identical ⇒
   context-deterministic (state/prompt-driven, reproducible); divergent ⇒ pure CUDA
   variance. This tells whether a state/prompt fix is even reachable.
4. **Diff the injection→agent-text handoff code** (how `injecting_response` clears
   and how the Nano text head is expected to begin the spoken answer after the forced
   tokens, plus the EarTTS-vs-Nano channel coupling that let audio advance with zero
   text) between the r18-era commit and the current stack, for a regression that
   raises the PAD-collapse rate.

### Verdict / next action

**Localize before any fix; do not retry, do not relax ASR, do not drop the tool
turn.** The sustained RED is a reproducible post-injection agent-text-generation
collapse (Nano text head → PAD/no-token while EarTTS emits off-distribution audio) at
turn 4, not injection/latch/boundary and not a deterministic-seed effect. Run the
four retained-evidence probes above (steps 1–4) — the decisive fork is step 3: if the
two failures' token streams are identical, a context/prompt/code regression is present
and gets the narrowest targeted fix at the injection→text handoff; if they diverge,
the tool-response generation is an under-robust CUDA-fragile surface on this candidate
and the honest conclusion is the candidate does not pass sustained as-is (harden the
handoff or re-baseline), never a threshold relaxation. Since the one allowed rerun is
spent, the next step is analysis of the retained artifacts, not another run. D8 intact.

## Addendum 60 — client-EOU tool-turn ordering: fix design review (2026-08-09)

Design review only; no source/GPU/commit. The new evidence resolves A59's open fork
decisively and changes the diagnosis from "fragile generation boundary" to a
**deterministic client-EOU ordering defect**.

### Root cause (confirmed)

- Both failures use `input_turn_detection=client_smart_turn_v1`; the 18/18 pass uses
  `server_turn_detection=rnnt`. Same checkpoint — so not stochastic, not the
  checkpoint. The audio differs ("My plan almost" vs "Lovely working on") but the
  **Nano text sequence is identical BOS,0,EOS**: text is deterministically empty.
- Server-RNNT ordering (works): SOTC commits + tool injects **while the agent is
  idle**, before `speech_stopped`; the trained post-EOTR native BOS then opens the
  answer and 10 text tokens follow.
- Client ordering (fails): EOU **forces BOS first** (opens `agent_speaking`), the
  staged SOTC replays one frame later *into an already-open turn*
  (`_replay_agent_open` gate at wrapper 2787), the tool injects mid-turn, and the
  model yields token 0 + EOS. EarTTS advances on the open turn and emits
  off-distribution audio with no text — the exact desync. This is the
  force-BOS→replay-SOTC path (stage at 2895-2899, replay at 2787-2794) doing the
  wrong thing for the tool case.

### The two candidate fixes

**(a) Proposed "commit SOTC at EOU, don't force BOS, rely on `post_tc_bos_exempt`."**
Correct in *direction* (keep the agent idle through FC, mirror server ordering) but
**insufficient as stated in client mode**. Confirmed in-tree: `_post_tc_bos_exempt`
is set only on the sync injection-complete edge (2826-2831) and consumed only inside
the **RNNT self-play-suppression** block (3544-3553), which is a server-RNNT
turn-taking path. In client mode server-RNNT turn-taking is disabled and the async FC
loop never sets the flag, so the native post-EOTR BOS is neither guaranteed to be
emitted nor routed through that exemption — the model can simply stay in PAD, i.e.
BOS,0,EOS again. Relying on a model-native BOS is exactly the thing client mode
otherwise refuses to trust.

**(b) Refined `post_fc_client_bos_requested` latch — preferred.** Set the latch when
the boundary SOTC is committed at the authoritative EOU; async FC skips turn-taking
(already true — `_fc_in_progress` silences text/turn-taking); the **first normal
post-FC sync step forces exactly one client BOS and clears the latch.** This keeps the
client the sole BOS authority (no model-native BOS trust), keeps the agent idle
through injection, and deterministically starts the answer — reproducing the working
server ordering without a model-emitted BOS.

### Preferable to forcing a second BOS?

Yes, clearly. A second forced BOS after injection is forced-EOU-BOS → mid-turn SOTC →
forced-BOS-2: two BOS tokens in one turn (untrained/off-distribution) **and it leaves
`agent_speaking` true through the injection** — the precise desync condition that
produces the garbage audio. The latch forces one BOS with the agent idle until EOTR;
strictly better.

### Invariants

- I1 Client is sole EOU **and** BOS authority: the answer BOS is client-forced, never
  model-native, in client mode.
- I2 Exactly one BOS per tool turn — no force-BOS-at-EOU beforehand, no native BOS
  double (duplicate-agent-BOS suppression at 3565 must still hold).
- I3 `agent_speaking` stays **false** across SOTC commit + injection (idle FC), so
  EarTTS cannot advance and produce text-less audio.
- I4 Pre-EOU function suppression unchanged (2749-2766): all pre-EOU SOTC still
  suppressed; only SOTC coincident with the authoritative EOU is committed.
- I5 Latch set exactly once (at boundary-SOTC commit) and cleared exactly once (at the
  forced post-FC BOS); lifetime bound to the FC cycle.
- I6 Non-tool client turns byte-identical (no SOTC ⇒ no commit ⇒ no latch ⇒ existing
  client-forced-BOS response path).
- I7 The stage/replay latch (`_deferred_eou_sotc_*`) is **bypassed** for the
  commit-at-EOU case — no staging, no `boundary_sotc_replay`.

### Tests

- T1 Turn-4 regression: client-EOU tool turn yields agent-idle FC, latch set, one
  forced post-FC BOS, and BOS+N-text+EOS (not BOS,0,EOS) with the correct spoken time.
- T2 Exactly-once: latch set once / cleared once; the step after the forced BOS does
  not re-force.
- T3 Abort/timeout: tool never reaches EOTR (error, `session_position_limit`, abort) ⇒
  latch cleared by reset, next turn does not spuriously force a BOS.
- T4 Non-tool client turn: latch never set; response path unchanged.
- T5 Pre-EOU spurious SOTC still suppressed (no commit, no latch).
- T6 Server-RNNT path byte-identical (client-guarded change only).
- T7 `reset()` clears the latch and `_deferred_eou_sotc_*` — no cross-turn/session
  leak.
- T8 Telemetry exposes latch requested/consumed frame+count and the commit-at-EOU
  reason for gate auditing.

### Hazards

- H1 Stuck latch → spurious later BOS if EOTR never completes: bind clearing to FC
  reset/abort and session reset, not only to the happy path.
- H2 Double-BOS: `post_tc_bos_exempt` and the latch must be **mutually exclusive** in
  client mode (disable the native-exempt path there), or a native BOS + forced BOS
  both fire.
- H3 BOS-timing: force only when `not _fc_in_progress` (injection drained,
  `injecting_response` false, `forced_function_tokens` empty), else the FC text-silence
  at 2840-2842 swallows it. Preflight-assert this.
- H4 Async/sync handoff: the completion edge (`_was_injecting`→false, 2828) runs on the
  sync step; the latch must be consumed on that step or the next sync step
  deterministically, not on the async thread.
- H5 EarTTS coupling: verify EarTTS is gated on `agent_speaking` so idle FC cannot
  reproduce the text-less audio; the fix must be validated on audio, not just text.
- H6 Commit preflight: commit SOTC only when `_external_user_eou_requested` **and**
  raw token == `_fc_sotc_id` **and** agent not open — same precision as the current
  suppression guard, now committing rather than suppressing, to avoid spurious tools.
- H7 Telemetry/exactly-once must be surfaced to defend the qualification gate (prove
  BOS was client-forced, one per tool turn).

### Verdict / next action

**REQUEST CHANGES on fix (a); prefer fix (b) — the `post_fc_client_bos_requested`
latch — over both the native-BOS-exempt-only approach and the double-BOS approach.**
Rationale: server-RNNT is disabled and async never sets the exempt flag in client
mode, so a model-native post-EOTR BOS is not guaranteed; a client-forced single BOS
after idle FC is the only design that (1) keeps client BOS authority, (2) keeps the
agent idle through injection (killing the EarTTS text-less audio), and (3)
deterministically starts the answer. Gate it on the invariants above, make it mutually
exclusive with `post_tc_bos_exempt`, bind the latch lifetime to the FC cycle
(reset/abort clear), force only when `not _fc_in_progress`, and surface exactly-once
telemetry. This is a real source change — out of scope to implement here; once
implemented it requires the fresh-stack sustained rerun (audio + text) to confirm,
never an ASR-threshold relaxation or dropped tool turn. D8 intact.

## Addendum 61 — Smart-Turn tool-order repair: implementation review (2026-08-09)

Implementation review of the committed repair across both repos; no source/GPU/commit.

### What I verified in the diff

- **Latch is decoupled from FC-completion.** `_post_fc_client_bos_requested` is set at
  the *boundary-SOTC commit* while the agent is idle (wrapper 2763-2785), not at the
  injection-complete edge. So a misclassified `fc_complete` cannot break the client
  BOS — the failure mode A59/A60 feared is structurally removed.
- **Exactly-once, both directions.** Set is guarded against re-entry
  (`if _post_fc_client_bos_requested: raise`, 2764-2765); consume happens once at the
  forced BOS and returns immediately (3536-3540). Counters/frames are initialized
  (254-259) and cleared in all three reset paths (3851-3892).
- **Idle commit.** The commit consumes the client EOU (`_external_user_eou_requested
  = False`) and zeroes RNNT speech evidence *without opening `agent_speaking`*
  (2769-2780), so EarTTS cannot advance and reproduce the text-less audio.
- **Mutual exclusion with the server exemption.** Client mode forces
  `_post_tc_bos_exempt = False` at fc-complete (2836); the forced-BOS block raises if
  the two ever overlap (3516-3517). Preflight also raises if the latch is pending
  outside client mode (3513) or while the agent turn is open (3515) — the
  direct-position rejection.
- **First-post-FC timing is correct.** The forced BOS lives inside
  `_apply_rnnt_turn_taking`, which runs only when `not _fc_in_progress`, so the latch
  is consumed on the first ordinary step after injection drains — never mid-FC.
- **Old stage/replay removed.** No `_deferred_eou_sotc_*`, no `boundary_sotc_replay`
  remain — the wrong-order path is deleted, not merely bypassed.

### On the four hazards I raised in A60

- **fc_complete misclassification (H4/stale-edge):** resolved. The new `fc_complete`
  requires a *current-cycle* `tool_response_inject_end_frame`, and committing a fresh
  SOTC clears stale timing keys — so a recreated/stale `_was_injecting` cannot spoof
  the edge, and timing keys cannot leak across cycles.
- **Stuck latch on abort/incomplete cycle (H1):** resolved. Failed and incomplete
  background paths share the same abort+reset, which clears the latch quad — so an FC
  cycle that never reaches EOTR cannot leave a dangling latch that fires a stray BOS.
- **Double-BOS (H2):** resolved by the mutual-exclusion force + overlap raise above.
- **EarTTS coupling (H5):** the live voice probe now gates it on real state — it
  asserts agent PAD / no response-start at the boundary, `committed_count` +1, pending
  latch, then `forced_count` +1 with pending false and `committed_frame ==
  requested_frame` preceding the forced frame. That is the exactly-once lifecycle
  proven on the wrapper, not just telemetry shape.

### Evidence / residual

Full suite 484 pass / 8 skip, ruff green. The live-probe gate exercises the real
commit→idle→force path; the telemetry unit tests pin the schema. One non-blocking
behavioral note: the commit zeroes `_transport_user_speech_seen` /
`_transport_input_silent_frames` (2770-2771), so confirm a user barge-in *during* the
tool cycle is still detected on the fresh accounting rather than swallowed — worth one
targeted test, not a blocker.

### Verdict / next action

**APPROVE the implementation.** The repair is correct on every invariant that made the
client-EOU tool turn fail: idle commit, single client-forced BOS, exactly-once with
guarded set/consume, mutual exclusion with the server exemption, abort/incomplete
clearing via shared reset, hardened current-cycle fc_complete, and old stage/replay
deleted. Telemetry and the live probe make the lifecycle auditable for the gate. The
one remaining proof the code cannot supply is behavioral: the corrected ordering must
be confirmed by a fresh-stack sustained rerun scored on **audio and text** (turn-4
must produce BOS+text+EOS with the correct spoken time, not BOS,0,EOS) — never an
ASR-threshold relaxation or dropped tool turn. Add the barge-in-during-tool test as a
follow-up. D8 intact.

## Addendum 62 — focused live artifacts for the Smart-Turn fix (2026-08-09)

Live-artifact review only; no source/GPU/commit. Root
`…/traces/model/voice-function-smart-turn-bos-fix-r1`, image
`sha256:2c41d321…`.

### What the artifacts show

The focused run reproduces the corrected lifecycle end-to-end on real hardware, in
frame order:
- pre-EOU SOTC **suppressed** at frame 35 (pre-EOU guard fires);
- boundary SOTC **committed** at frame 38 with `pending=true` and **no BOS** (idle
  commit — the agent turn did not open);
- **sole** forced BOS at frame 107 with `pending=false` (exactly-once consume);
  35 < 38 < 107 is the intended ordering.
- `assistant_text` "It is currently clear" with a **non-empty** response WAV, and the
  independent Parakeet log (`focused-asr/transcribe.log`) transcribes "It is currently
  clear." at **‑34.387 dBFS**, `semantic_match=true`.

This is exactly the behavioral proof A61 said code review could not supply: the
turn-4-class failure signature (BOS,0,EOS; text-less audio) is **absent** — text and
audio are both present and, critically, **agree** across two independent channels.

### Fail-closed integrity

- `passed=true` is gated on the structural checks **and** the external ASR
  `semantic_match`, and the ASR is a separate Parakeet pass with its own log — not the
  model's self-report. The independent hypothesis matching `assistant_text` rules out
  a text-only (audio-desync) pass.
- ‑34.387 dBFS is well above the near-silence skip floor, so this is **not** a vacuous
  near-silent green.
- Image `sha256` pinned; the `get_weather` Tokyo call is confirmed fired by the
  structural block, so the spoken answer follows a real tool result.

### Scope limit (necessary, not sufficient)

This is a **single** focused tool turn. It proves the commit→idle→sole-BOS mechanism
fires correctly once with independent audio+text agreement. It does **not** exercise
the multi-tool-turn sustained path (4/8/12/16 in one long session), the zero
silent-rate gate across 18 turns, or repeated latch cycling — which is precisely where
the original RED lived. Focused-green must not be read as sustained-green.

### Verdict / next action

**The focused gate is cleared.** Integrity is fail-closed, the desync is gone, and the
exactly-once idle-commit → single-forced-BOS lifecycle is demonstrated on live
artifacts with independent ASR. This justifies proceeding to **one fresh full
sustained run** on the same pinned image, scored on audio and text at the unchanged
zero silent-rate with independent Parakeet ASR — that run, not this one, is the
qualifying evidence. No ASR-threshold relaxation, no dropped tool turn. D8 intact.

## Addendum 63 — barge-in residual + sustained fail-closed audit (2026-08-09)

Static review while the `sustained-smart-turn-bos-fix-r2` run (image
`sha256:2c41d321…`) is active; no source/GPU/commit. Sustained is **not** declared
green here — final report and Parakeet artifacts do not yet exist.

### Finding 1 — barge-in during the client-EOU FC cycle leaves the latch pending (confirmed)

A61's worry was the commit-time zeroing of the transport counters (wrapper 2770-2771).
That specific zeroing is **benign**: `observe_transport_user_activity` only updates the
counters while the agent is idle (3915), and the commit deliberately keeps the agent
idle, so a user barge-in *during* the tool cycle still re-arms
`_transport_user_speech_seen`. The real defect is adjacent: **the FC async-abort path
does not clear `_post_fc_client_bos_requested`.** On barge-in the async loop marks the
agent idle and breaks (1549-1567); the latch is cleared only at the forced-BOS consume
(3536) and the three reset sites (3860 `_reset_rnnt_turn_taking_state`, 3881/3889
`set_external_user_eou_mode`), none of which is on the abort path or a mid-session
barge-in. Consequences:
- the pending latch fires a **spurious forced agent BOS** on the first post-abort
  turn-taking step (3511-3518) — the agent starts speaking a tool answer the user just
  interrupted; and
- if the barge-in leads to a fresh tool call while the latch is still pending, the
  commit re-entry guard **raises** "post-FC client BOS was already pending" (2764-2765)
  — fail-closed (a crash/RED, not a false green), but a latent session-kill.

**Classification: nonblocking for THIS qualification** — the scripted sustained run has
clean EOUs and never barges in during a tool cycle, so the path is unexercised. **It is
a required production fix** before the client-EOU path ships to real (barge-in-capable)
users: clear the latch quad on FC async abort/natural-interrupt, and treat a
latch-pending re-commit as a reset rather than a raise. Add the barge-in-during-tool
test (carried from A61).

### Finding 2 — does the sustained gate fail closed for the turn-4/8/12/16 signature?

The original signature is empty `output_text` + audio that is either audible-garbled
("My plan almost") or near-silent.

- **Audible-garbled (the signature actually observed in r1): FAILS CLOSED.** A
  speech-classified response is gated on `hypothesis and reference and wer≤max and
  semantic_match and negative_semantic_match` (transcribe_sustained 385-391). Empty
  `text` ⇒ `reference==""` ⇒ fail on *every* turn; and on the fresh-call turn 4
  (`expected_response_any=['HH:MM UTC']`) the garbled hypothesis also fails
  `semantic_match`. Double-caught.
- **Full four-turn failure: FAILS CLOSED** either way — four near-silent responses give
  `silent_rate≈4/18=0.22 > max_silent_rate` (default `1/15≈0.067`, line 337), and four
  audible-garbled responses fail the semantic/empty-reference gate.
- **Residual gap — a single near-silent recurrence can be absorbed.** Near-silent
  responses are shunted out of `prepared` (354-357) and never reach the empty-reference
  / semantic check; they count only against `silent_rate ≤ 1/15`. With 18 responses,
  one near-silent turn (1/18=0.056 ≤ 0.067) passes. Worse for the **context-recall
  turns 8/12/16**: their `expected_response_any` is empty, so `semantic_match` is
  vacuously true (85: `positive = not expected or …`), leaving *only* the
  empty-reference check as their defense — and that check is bypassed for a near-silent
  classification. So a partial near-silent recurrence on a tool/recall turn could slip
  through the base-rate tolerance.

**Classification:** the gate **fails closed for the original failure as observed** and
for any multi-turn recurrence. The single-near-silent absorption is a *pre-existing,
intentional* FP32 base-rate tolerance, not a regression from the fix — so it is not a
blocker to the gate's design. But because the failure signature (empty text ± silence)
overlaps that tolerance, I impose a **blocking artifact-acceptance check** below rather
than trusting the aggregate `passed` flag alone.

### Required changes

- **BLOCKER (artifact acceptance, this run):** when r2 completes, before calling it
  green, verify per-response that each of the four tool turns (4/8/12/16) is
  `classification:"speech"` (not `near_silent`), has non-empty `text`, and
  `external_asr.semantic_match==true`, with turn 4's fresh `11:07`-class match holding.
  Do **not** accept a green that rests on any tool turn being absorbed as near-silent,
  and confirm the independent Parakeet log exists for each. (Verification step, not a
  source edit.)
- **Nonblocking follow-up A (production correctness):** clear `_post_fc_client_bos_*`
  on FC async abort/natural-interrupt; make a latch-pending re-commit reset-and-recommit
  instead of raising; add the barge-in-during-tool test.
- **Nonblocking follow-up B (gate hardening, tightening only):** exempt the
  deterministic tool turns from the near-silent base-rate tolerance — a tool turn that
  lands `near_silent` should fail, never be absorbed — so a partial recurrence cannot
  hide. This tightens, and does not relax, the gate (D8-consistent).

### Verdict / next action

The approved fix is sound for the scripted sustained path, and the gate fails closed
for the original failure as observed; **sustained remains UNPROVEN until r2's final
report and per-turn Parakeet artifacts exist and pass the blocking per-tool-turn check
above.** The barge-in latch-survival is a real but out-of-scope-for-this-run production
defect (required follow-up, not a qualification blocker). No ASR-threshold relaxation,
no dropped tool turn. D8 intact.

## Addendum 63 — Amendment: barge-in finding was wrong; corrected (2026-08-09)

Static review only; no source/GPU/commit. Re-checked A63 Finding 1 against the outer
consumer in `streaming_s2s_pipeline.py`.

### Finding 1's abort trace is WITHDRAWN

I cited only the inner wrapper abort break (~1566) and missed the outer consumer. All
three non-normal FC exit paths call the full reset:
`quit_to_normal` (1104-1106), `was_aborted or fc_failed` (1148-1150), and
`natural_interrupt` (1198-1200) each invoke `_reset_rnnt_turn_taking_state()`, which
clears `_post_fc_client_bos_requested`, `_external_user_eou_requested`, and the
transport counters (wrapper 3853-3866) — and also disables client-EOU mode (3854). So a
barge-in that aborts/interrupts FC **does** clear the latch before the next
turn-taking step. **The "spurious forced BOS after barge-in" trace is not reachable; I
withdraw it.** The re-entrant-commit RAISE (2764-2765) is likewise unreachable on the
abort paths (reset intervenes) and would otherwise require a fresh raw SOTC on the
function channel mid-injection — not a demonstrable path, and fail-closed regardless.
Downgraded to a theoretical, fail-closed edge.

### Second `request_user_eou` during pending FC — before vs after the outer reset

- **Abort/interrupt path, 2nd EOU before the reset:** the reset clears the 2nd EOU and
  disables the mode; the signal is dropped and the consumer must re-arm mode + re-signal.
  Governed by the consumer's per-turn mode discipline — no latch hazard.
- **Abort/interrupt path, 2nd EOU after the reset:** mode is disabled, so
  `request_user_eou` raises (3896-3897) unless the consumer re-enabled mode first. Again
  a consumer-contract ordering matter — no latch hazard.

So the reset paths are safe, exactly as the outer consumer intends.

### The one genuinely fix-attributable, reachable-in-principle residual (narrow)

On the **normal FC-completion path** (the `else` at 1205, which has **no** reset — by
design, so the latch can fire the answer BOS), a 2nd `request_user_eou` that arrives
while FC is pending survives, because the latch-forced-BOS block returns at 3540
**before** the EOU-consumption block at 3640-3661. Concrete trace:
1. Tool turn: 1st EOU consumed by the boundary commit (2769); latch set; FC injects.
2. Client sends a 2nd `request_user_eou` during pending FC → `_external_user_eou_requested=True`.
3. FC completes normally (no reset). First post-FC step: latch block forces the tool-answer
   BOS and `return`s (3518-3540) — it does **not** touch `_external_user_eou_requested`.
4. Agent speaks the tool answer; the stale EOU is dormant (3640 gated on `not agent_speaking`).
5. Agent finishes → idle → 3640 fires on the stale EOU → a **second, likely
   zero-user-content agent BOS** (self-play turn).

Before the fix, that same post-FC BOS would have been the 3640 EOU-consumer and would
have consumed the flag, yielding one turn; the latch path leaves it stale, so this
extra turn is **fix-attributable**. But it requires a 2nd client EOU landing inside a
tool cycle that neither aborts nor natural-interrupts the FC — a narrow window, a
possible client-contract violation, and **not exercised by the scripted sustained run**
(sequential turns, agent answers before the next EOU).

### Net correction

The A63 "required production fix (clear latch on abort)" is **withdrawn** — the outer
reset already does this. It is replaced by a **downgraded, optional hardening**: at the
latch-forced BOS, also consume/clear a coincident `_external_user_eou_requested` (or
assert it clear), so a race-latched 2nd EOU cannot open a stale self-play turn. Severity
nonblocking; unexercised by qualification. **A63 Finding 2 (sustained fail-closed) and
its BLOCKING per-tool-turn artifact check stand unchanged.** Sustained still not
declared green. No ASR-threshold relaxation, no dropped tool turn. D8 intact.

## Addendum 64 — sustained r2 RED: repeated-tool-cycle answer collapse (2026-08-09)

Adversarial root-cause review; no source/GPU/commit. Artifacts: r2 report, r2-asr
report, model trace `376311aa…`.

### The gate failed closed — correctly

r2: `runtime_gate_passed=true`, `external_asr.passed=false`, aggregate `false`.
Parakeet classified turns 8/12/16 and response 17 as `near_silent`
(`silent_rate=0.235 > max_silent_rate=0.0`). This is exactly the A63-Finding-2
fail-closed path, and the run was scored at `max_silent_rate=0.0`, so the near-silent
tool turns could not be absorbed. A61's fail-closed design and A63's blocking
per-tool-turn check both held. **Sustained is RED; not green.**

### Root cause — the forced post-FC BOS degrades across repeated tool cycles

Latch mechanics are correct: `committed=forced=4`, exactly-once, forced BOS at frames
2317/5305/8301/11289. The defect is downstream, in the answer:

- **Tool answer #1 (turn 4, bos 2317):** Nano emits the full sentence; EarTTS is
  **audible** (`audible_seen=True` from the 3rd frame, dbfs −24…−48); the turn closes
  cleanly via `decoded_silence_watchdog` at `silent_fr=22`, `eartts_gen=71`. Correct.
- **Tool answers #2–4 (turns 8/12/16, bos 5305/8301/11289):** Nano emits only 1–2
  degenerate tokens (" The UTC" / " The") then halts; EarTTS produces **only silence**
  (`audible_seen` never becomes True; dbfs −120 throughout); at ~frame bos+11 the
  boundary enters phase **`internal_drain`** and stays there. The turn is **not** closed
  by the normal `decoded_silence_watchdog` (it requires prior audible speech to arm) nor
  by `no_text_since_bos` (defeated by the 1–2 emitted tokens), so it runs ~645 silent
  frames until `model_or_turn_taking` (`eos_reason` confirms this; `silent_fr≈645`,
  `eartts_gen≈645` all silent).

The degradation is **specific to repeated client-forced tool cycles** — the non-tool
turns interleaved between them (5–7, 9–11, 15) stay correct and audible. That isolates
the cause: forcing the answer BOS diverges from the model's **trained native
post-EOTR BOS** ordering (the server-RNNT path that passed 18/18), and the divergence
**accumulates** across tool cycles until, by cycle 2, the Nano answer distribution
collapses and EarTTS is never driven into audible synthesis. Tool cycle 1 still lands
in-distribution; cycles 2–4 do not.

### Two compounding effects (both real, ranked)

1. **PRIMARY — generation collapse on repeated forced-BOS tool cycles.** The forced BOS
   is not the model's native post-EOTR BOS; across cycles the context drifts
   out-of-distribution and the tool answer degenerates to near-empty text + silent
   audio. This is the direct cause of the near-silent RED.
2. **SECONDARY — boundary coverage gap.** A near-silent-but-nonzero-text answer falls
   between the watchdogs and drains ~645 frames in `internal_drain` before
   `model_or_turn_taking` closes it. Three such drains (~1,935 frames) prematurely
   exhaust the 12,000-frame session budget, which is why response 17 hit
   `session_position_limit` and late non-tool turns 13/14 truncated. This did not
   *cause* the RED but amplified it and shortened the session.

This vindicates the A62 warning: the focused r1 gate exercised a **single** tool turn,
so it could not observe a cycle-2 collapse — focused-green was necessary, not
sufficient, exactly as flagged.

### Smallest principled fix

The forced-BOS latch (A61) correctly cured the A59 *single-cycle total collapse* (where
the native BOS never came: BOS,0,EOS). But forcing unconditionally diverges from the
trained ordering and degrades on repeats. The smallest principled change that satisfies
**both** cases is a **bounded native-BOS-with-forced-fallback**:

- In client mode, after EOTR **re-enable acceptance of the model's native post-EOTR
  BOS** (the `post_tc_bos_exempt` path A61 disabled in client mode), so an in-distribution
  answer opens exactly as it did in the 18/18 server-RNNT runs and repeated cycles stay
  native.
- Keep `_post_fc_client_bos_requested` as a **bounded fallback**: force the single BOS
  only if the model has not emitted its native post-EOTR BOS within a small deadline
  (N idle frames after injection completes). This preserves the A59 safety net without
  overriding the healthy native ordering.
- Mutual exclusion and exactly-once are unchanged; the latch simply becomes
  "force iff native BOS absent by deadline" instead of "force always."

Secondary (defense-in-depth, not sufficient alone): arm a **no-audible-since-BOS**
watchdog on the forced-BOS answer so a near-silent tool answer is bounded promptly
(prevents the ~645-frame drain and budget exhaustion). This bounds the symptom; it does
not restore the answer — do not ship it as the fix, only alongside.

### Focused test that proves repeated cycles

Add a focused harness (no full sustained suite) that issues **≥4 sequential tool calls
in one session** and asserts, for **every** tool answer: `audible_seen` true / dbfs
above the silence floor, non-truncated expected text, ASR semantic match, closure via
`decoded_silence_watchdog` (**not** `internal_drain`/`model_or_turn_taking`), and
`committed==forced==N` with no session-budget blowup. This reproduces the r2 collapse
in a fast run and closes the single-cycle blind spot that let r1 pass.

### Verdict / next action

**Sustained r2 is RED; the fix is incomplete, not the gate.** The forced-BOS latch fixed
cycle 1 but forcing diverges from the trained post-EOTR ordering and collapses on
cycles 2–4 (near-empty text + silent EarTTS + unbounded `internal_drain`). Implement the
bounded native-BOS-with-forced-fallback (plus the no-audible-since-BOS watchdog as
defense-in-depth), add the ≥4-cycle focused test, then re-run one fresh sustained.
No ASR-threshold relaxation, no dropped tool turn, no watchdog loosening that would let a
silent answer pass. D8 intact.

## Addendum 64 — Amendment: root cause is EOTR-not-observed channel leak (2026-08-09)

Adversarial correction; no source/GPU/commit. The A64 root cause ("out-of-distribution
generation collapse") and fix ("native-BOS fallback") are **withdrawn** — refuted by the
channel evidence, and the proposed test invariant `committed==forced==N` self-contradicts
a fix that accepts a native BOS (which would make `forced < N`).

### Corrected root cause (confirmed in trace + code + training grammar)

The model does **not** collapse — it generates the full answer, but on the **function
channel**. r2 frames after forced BOS #2/#3:
- 5306 agent=`' The'`, 5307 agent PAD / **function=`' current'`**, 5308 agent=`' UTC'` /
  **function=`' UTC'`**, 5309 **function=`' time'`**, 5310 **function=`' is'`**; cycle 3
  identical (8302 agent `' The'`, 8303-8306 function `' current'`/`' UTC'`/`' time'`/`' is'`).
- So "The current UTC time is …" is emitted on the function channel; the agent channel
  keeps only a fragment → EarTTS synthesizes near-silence and the agent text truncates to
  " The UTC" / " The". This is a **channel-routing** defect, not a generation collapse.

Code confirms why: `_run_fc_async_steps` sets `injecting_response=False` when the forced
TOOL_RESPONSE drains and logs "now awaiting model to predict EOTR on function channel"
(1786-1792), but the **exit condition breaks on `no_forced and not_active and
not_injecting` alone** (1905-1917) — EOTR detection (1844-1845) only *logs* and never
gates the exit. So the async cycle resumes **before** the model emits EOTR, mid-grammar.
Training defines the block as `<EOTC> response <EOTR>` with **loss on EOTR**
(duplex_stt_model.py ~1036-1042): the model is trained to *predict* EOTR to close the
response block. Resuming without observing it leaves the function block open, so the
post-injection answer continues on the function channel instead of the agent channel.
(Cycle 1 happened to resume with the block effectively closed and routed to agent; cycles
2-4 did not — the exit is timing-fragile, which is why it is intermittent by cycle.)

The A64 "secondary" effects (`internal_drain`, ~645-frame silent drain, early
`session_position_limit`) are **downstream of this leak** — the agent turn opens (forced
BOS) but never receives audible content, so it drains. They are largely resolved once the
answer routes to the agent channel.

### (A) vs (B): smallest principled fix is (A)

- **(A) require observed EOTR before fc_complete/resume, fail closed on timeout.** Add an
  `eotr_observed` gate to the exit at 1905-1917 (set where EOTR is detected, 1844), so the
  async loop keeps stepping after the injected response drains until the model predicts
  EOTR on the function channel — then exits and resumes. If EOTR is not observed within a
  bounded number of steps, abort/RED. This **enforces the trained `<EOTC> response
  <EOTR>` grammar**, closes the function block before agent generation resumes, and routes
  the answer to the agent channel at the root. Fail-closed-on-timeout is honest (the
  A59-class missing-token case becomes RED, never a false green). It also resolves the
  standing inconsistency between the drain comment (1902-1904, "EOTR appended to forced
  tokens") and the log (1789, "awaiting model to predict EOTR"): let the model predict its
  trained EOTR rather than assume an injected one closed the block.
- **(B) reroute function→agent when agent is PAD, suppress duplicates.** This only **masks
  the symptom**: it copies leaked tokens to the agent channel and must special-case the
  frame-5308 double-`' UTC'`. It does not close the function block, risks mis-routing a
  legitimate next-cycle SOTC/tool-call token to the agent channel, and depends on a fragile
  "which channel is authoritative" heuristic. More code, weaker guarantee.

**Smallest principled fix: (A).** One gated exit condition + bounded timeout; enforces the
training grammar; fixes the root, not the symptom. Retain A61's forced BOS unchanged — the
EOTR wait happens inside the async loop *before* FC-complete, so by the time the (post-FC)
forced BOS fires the block is closed and the answer routes to agent; `committed==forced==N`
therefore remains a valid invariant (correcting A64's self-contradiction only in the sense
that the invariant belongs with fix (A), not the withdrawn native-BOS fix).

### Corrected focused test (≥4 cycles)

Per tool cycle, assert: **EOTR observed on the function channel before the async loop
exits**; the answer is emitted on the **agent** channel with the function channel idle
during the answer (no function-channel answer tokens, no cross-channel duplication);
audible (`audible_seen` / dbfs above floor); non-truncated expected text; ASR semantic
match; closure via `decoded_silence_watchdog` (not `internal_drain`/`model_or_turn_taking`);
and `committed==forced==N` with no session-budget blowup. The prior single-cycle focused
test could not observe the leak — the ≥4-cycle requirement stands.

### Amended verdict

**Sustained r2 RED root cause = EOTR-not-observed channel leak in the FC async exit, not a
generation collapse.** Smallest principled fix is **(A)**: gate FC-complete on an observed
EOTR, fail closed on timeout; keep the A61 forced BOS; keep the no-audible-since-BOS
watchdog only as defense-in-depth. Add the corrected ≥4-cycle test, then re-run one fresh
sustained. The A64 native-BOS-fallback proposal and its contradictory invariant are
withdrawn. No ASR-threshold relaxation, no dropped tool turn, no watchdog loosening. D8
intact.

## Addendum 64 — Amendment 2: calibrate to hypothesis, adopt bound=1, defer containment (2026-08-09)

Adversarial cross-review accepted; no source/GPU/commit. The independent review is
right on all three points; I over-stated certainty and the bound.

### (3) Root cause is a strongly-supported HYPOTHESIS, not confirmed

What is proven: (a) the answer tokens appear on the **function channel** in the
main-loop `model_step` frames post-resume (5307/5309/5310, cycle-3 identical); (b) the
async exit (1905-1917) does not require EOTR; (c) training reserves one EOTR position
with loss. What is **not** directly observed: the FC async loop runs on the background
thread and its per-step internal function tokens — including the first post-drain
token — are **not** in `events.jsonl`. So "the loop exited before the model emitted
EOTR" is *inferred* from (a)+(b)+(c), not seen. I withdraw "confirmed in trace"; the
EOTR-not-observed mechanism is a hypothesis to be proven by **async-internal token
telemetry on the live 4-cycle run** — the same instrumentation that tests the fix must
log, per cycle: the first post-drain function token, whether it is EOTR, and the
channel each subsequent answer token lands on. Healthy (turn-4-class) cycle must show
EOTR-first + answer-on-agent; failing cycle must show non-EOTR/absent-EOTR +
answer-on-function. Until that run exists, the repair is *necessary-looking*, not proven.

### (1) Bound = 1, not 32

Adopt **immediate fail-closed on the first non-EOTR post-drain function token (bound=1)**
with unexpected-token-id + frame telemetry. Rationale: training reserves exactly one
EOTR position, and historical logs show the first natural post-drain token is normally
EOTR, so in a healthy cycle EOTR is expected immediately. A wait of 32 would feed up to
31 malformed function tokens into the KV cache before giving up — that *is* the leak,
now polluting context and muddying diagnosis. Bound=1 fails at the earliest detectable
grammar deviation, before KV corruption, and yields a clean single-token diagnostic.
Because it is fail-closed (RED, never a false green), a legitimate EOTR-at-position-2
case surfaces as telemetry rather than a silent pass — self-correcting: if the live run
shows EOTR is not reliably position-1 in healthy cycles, relax deliberately on evidence.
So my A64 "bounded wait (N steps)" is corrected to bound=1.

### (2) Defer containment until live 4-cycle telemetry

Do **not** add the "agent-open + FC-inactive ⇒ unexpected function tokens become PAD"
containment now. It cannot recover the missing agent words (PAD-ing the leaked tokens
does not move them to the agent channel — the RED persists), it may suppress legitimate
duplex function-channel behavior we have not characterized, and stacking it with the
EOTR gate confounds attribution (a green run would not tell us whether the gate alone
fixed it or containment masked a residual gap). It is a weaker sibling of the reroute
candidate (B) already rejected — a symptom mask, not a repair. Add it only if the live
4-cycle telemetry shows residual out-of-band function tokens *after* the EOTR gate is in
place.

### Amended position

Accepted implementation step: **EOTR gate at the async exit with bound=1 fail-closed +
unexpected-token/frame + per-cycle first-post-drain-token and answer-channel telemetry.**
Hold the EOTR-not-observed root cause as a hypothesis until the live 4-cycle run confirms
EOTR-position and agent-channel routing. Keep A61's forced BOS and the no-audible-since-BOS
watchdog (defense-in-depth). Defer PAD containment pending that telemetry. No
ASR-threshold relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 65 — Step 6 EOTR-gate patch: implementation review (2026-08-09)

Implementation review of the shared EOTR-gate patch; no source/GPU/commit.

### The gate itself is correct (after the indentation clarification)

I initially flagged a bound=1 false-timeout from `eotr_wait_steps` counting injection
frames. **Withdrawn** — the counter/telemetry block (1879-1888) lives inside the `else:`
of `if forced:` (1794), so forced-injection frames run only 1794-1798 and skip it; only
the first **no-forced** post-drain natural position enters `awaiting_eotr_at_step` and
increments the counter. So bound=1 (`fc_eotr_wait_max_steps=1`, 1078-1082) correctly
means "the first post-drain token must be EOTR, else fail closed." Verified good:
per-cycle re-init of the `eotr_*` fields at both entry points (phase-2 1488-1496; EOTC
1842-1850); EOTR detection sets `eotr_observed`/frame (1868-1875); timeout fails closed
(1943-1955); FC-complete requires `not_awaiting_eotr` (1963-1964); the blocking path
`_require_fc_eotr_completion` (2385-2394) raises on `eotr_timeout OR awaiting_eotr OR
eotr_observed is not True` — a correct fail-closed conjunction; interrupt/failure resets
now clear `awaiting_eotr` (1102/1146/1198); structural harness rejects a non-EOTR
`first_post_drain_token_id` (test 104). The first/unexpected-token telemetry attributes
to the correct post-drain position.

### BLOCKER — nonblocking-FC ownership gap: terminal EOTR is never fed back to the foreground

Confirmed in code, and it defeats the gate on the production nonblocking path:
- The background FC thread runs on **clones** — `_gen_func_text_snap =
  context.gen_function_text.clone()` (1580) — so EOTR is written into the *clone*.
- The normal-completion branch (1021-1034) adopts `dynamic_cache`, `rnnt_partial_hypotheses`,
  `tts_state`, and advances `context.frame_idx += async_steps`, but **never copies the
  terminal predicted EOTR from the clone back into `context.gen_function_text`.** (The
  interrupt path *does* write that slot — PAD — at 1088, which proves the normal path's
  omission leaves the stale original array.)
- Async position semantics predict token t from feedback t-1; after resume the foreground
  feeds `context.gen_function_text[0, frame_idx-1]` (read at 2419) as the last function
  token. That slot is **PAD, not EOTR**, so from the foreground's view the tool-response
  block never closed — the model continues on the function channel and the answer leaks
  there, near-silencing the agent turn. This reproduces the r2 signature **even when the
  async log says "EOTR predicted."**

Consequence for the gate: observing EOTR in the clone is **necessary but not sufficient**.
A gate-only patch can make the async loop complete "successfully" and the per-tool
structural gate read EOTR-observed GREEN while the actual audio still fails — a
structural/ASR disagreement (the ASR gate would still catch it RED, so fail-closed holds
at the outer layer, but the structural gate would falsely green). This must be fixed
before acceptance.

### Assessment of the proposed fix — endorsed, minimal, correct

Record the bg terminal index/token; require at fc_complete that **terminal index ==
`eotr_observed_frame` and token == EOTR id**; on normal completion write **exactly that
terminal EOTR** into `context.gen_function_text` at `frame_idx-1` before foreground
resume; **fail closed on mismatch**. Assessment:
- **Single-token write-back is sufficient.** The earlier async tokens are already encoded
  in the adopted `dynamic_cache`; `gen_function_text` is only re-read as the t-1 feedback
  for the next foreground position, so only the terminal slot must be correct. Confirmed by
  the read at 2419 (frame_idx-1) / 2483 (frame_idx).
- **The consistency gate is the right guard for the off-by-one.** It ties the write to the
  gate's observed EOTR frame and token, and fails closed if the async `t` and the advanced
  `frame_idx-1` disagree — writing to the wrong slot is prevented rather than risked.
- **Minor telemetry caveat (nonblocking):** any consumer that scans the full post-resume
  `gen_function_text` span still sees PAD for the async positions (only the terminal slot
  is written). That is a transcript/telemetry gap, not a correctness one — the function
  channel is recorded elsewhere (the r2 trace showed the full tool call), so acceptable.

### Nonblocking follow-up

Interrupt/failure resets clear `awaiting_eotr` but not `eotr_observed`/`eotr_timeout`;
per-cycle entry re-initializes them, so no cross-cycle leak in the normal sequence, but a
snapshot or gate that reads `eotr_observed` between abort and next-cycle entry sees a stale
True. Clear all three together in the reset blocks for defense-in-depth against a stale-true
false-pass.

### Verdict / next action

**REQUEST CHANGES.** The EOTR gate is correct in isolation, but the nonblocking-FC
completion does not propagate the observed terminal EOTR into `context.gen_function_text`,
so the foreground feedback stays PAD and the channel leak persists — the patch does not yet
fix r2. Land the proposed terminal-EOTR write-back with the consistency gate and
fail-closed mismatch (nonblocking path); clear the full `eotr_*` set in interrupt resets;
then run the focused ≥4-cycle test (assert answer on the agent channel, EOTR observed *and*
fed back, audible, `committed==forced==N`) and only then one fresh sustained. Root cause
remains held to hypothesis until that live 4-cycle telemetry confirms agent-channel routing.
No ASR-threshold relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 66 — terminal-EOTR write-back: implementation review (2026-08-09)

Review of the A65 write-back fix; no source/GPU/commit. All A65 blocker items are
addressed.

### Verified correct

- **Terminal capture (bg, 1842-1852):** `_terminal_function_frame = sotc_frame +
  async_steps` with a bounds check (1844) before reading the clone; records
  `terminal_function_frame`/`terminal_function_token_id`; sets
  `eotr_feedback_committed=False`.
- **bg-side consistency gate (1853-1864):** `fc_complete` requires all state flags clear,
  `eotr_observed is True`, `not eotr_timeout`, **`terminal_function_frame ==
  eotr_observed_frame`**, **`terminal_function_token_id == EOTR id`**, and
  `tool_response_inject_end_frame is not None`.
- **Non-complete routing (970-981):** `_fc_failed_early` is True whenever a
  non-interrupted result has `fc_complete=False`, so EOTR-timeout / step-limit /
  non-EOTR-terminal all divert to the `elif … fc_failed` **reset** path (1129) — graceful,
  not the write-back. This was my open question; it is handled correctly.
- **Foreground write-back (1034-1048):** advances `frame_idx`, then **re-validates**
  `terminal == frame_idx-1` **and** `token == EOTR id` (1037-1040) or raises fatally
  (1042, fail-closed); writes exactly EOTR into `context.gen_function_text[terminal_frame]`
  (1045); sets `eotr_feedback_committed=True` (1048). Because the branch is reached only
  when `fc_complete` (via `_fc_failed_early`), the raise is a pure invariant guard on frame
  arithmetic that cannot fire on a benign path.
- **Single-token write-back is sufficient and confirmed:** earlier async tokens are in the
  adopted `dynamic_cache`; only the `frame_idx-1` slot is re-read as feedback (2419).
- **Race avoidance:** `eotr_feedback_committed` is False at bg capture and True only after
  the foreground write-back; probes/server gate their capture on `committed=True` and on
  `observed==first==feedback` frame/token identity, so a mid-update or stale-cycle read
  cannot be mistaken for the current cycle.
- **Resets (1118-1120/1164-1166/1218-1220):** now clear `awaiting_eotr`, `eotr_observed`,
  and `eotr_timeout` together — the A65 follow-up is done.

### Nonblocking observations (not gating)

1. The write-back branch is guarded by `not fc_failed` rather than an explicit
   `res["fc_complete"]`; the coupling is correct today because `_fc_failed_early ≡ failed OR
   (not-interrupted AND not-complete)`, but an `assert res.get("fc_complete")` at the top of
   the branch would make the invariant refactor-proof.
2. Resets clear the three `eotr_*` flags but not `eotr_feedback_committed` /
   `terminal_function_*`; next-cycle bg capture re-initializes them and the probe identity
   check disambiguates, so no leak — clearing the feedback triplet too is tidiness only.
3. The frame-arithmetic raise (1042) is fail-closed by construction; if it ever fires live
   it kills the turn as a hard RED. That is the correct posture, and the live 4-cycle probe
   will surface any real off-by-one loudly rather than as a silent leak.

### Verdict / next action

**APPROVE (static/implementation scope).** The terminal-EOTR write-back closes the A65
ownership gap correctly and defensively: double consistency gate (bg terminal==observed==EOTR;
foreground terminal==frame_idx-1==EOTR), graceful non-complete routing, fatal fail-closed on
invariant violation, committed-flag race avoidance, and complete `eotr_*` resets. The three
observations above are tidiness, not blockers. As stated throughout, static approval does not
substitute for the live evidence: run the focused ≥4-cycle probe (answer on the **agent**
channel, EOTR observed **and** fed back with `committed=True`, audible, closed by
`decoded_silence_watchdog`, `committed==forced==N`) and then one fresh sustained scored on
audio+text at zero silent-rate. The root cause remains a hypothesis until that 4-cycle
telemetry confirms agent-channel routing. No ASR-threshold relaxation, no dropped tool turn,
no watchdog loosening. D8 intact.

## Addendum 67 — sustained tool-cycle cardinality gate review (2026-08-09)

Review of `evaluate_tool_cycle_cardinality` and its tests; no source/GPU/commit/services.

### Cannot pass a subset of requested tool turns — verified

The gate (sustained_strict_v3.py 227-254) requires **exact** equality
`requested == checked == committed == forced`, and is ANDed with the per-turn channel
gate (614-615). Each leg closes a distinct subset path:
- **Dropped/unlabeled turn:** `checked = len(tool_response_checks)` counts only responses
  carrying `expected_response_any` (171-172), each anchored to an EOTR observation for its
  own `turn_id` (175). A tool turn that produces no such response ⇒ `checked < requested`
  ⇒ fail. Test `(4, checks[:3], 4, 4)` proves a 3-of-4 subset fails even with
  committed=forced=4.
- **Partial commit / forced BOS:** `committed`/`forced` must each equal N; tests
  `(4, checks, 3, 4)` and `(4, checks, 4, 3)` fail. Because equality is `==` (not `>=`),
  an over-count (spurious extra commit) also fails.
- **Garbage-but-counted turns (the r2 shape):** r2 had committed=forced=checked=4, so
  cardinality alone would pass — but `observed_tool_response_gate` (216-223) requires
  **every** check to be `eotr_valid ∧ function_channel_idle ∧ decoded_silence_close ∧
  text_nonempty ∧ audio_nonempty`. r2's near-silent/function-leak turns fail that, so the
  ANDed `tool_response_channel_gate` is False. Count (cardinality) and per-turn validity
  (channel gate) together admit no K<N subset.
- **Missing counters / zero requested:** tests `(4, checks, None, None)` and `(0, [], 0, 0)`
  both fail (the gate requires `requested_tool_prompts > 0`).

### `committed==forced==N` is measured at a sound point

The counters are read from `metrics[-1].function_calling` (234-240) — the **cumulative,
monotonic** session totals (`client_eou_sotc_committed_count`,
`post_fc_client_bos_forced_count`), incremented once per cycle and reset only at
session/mode boundaries. Soundness:
- A mid-session reset lowers the count ⇒ `< N` ⇒ fail — a reset can only cause a
  false-fail, never a false-pass (fail-closed direction).
- A truncated/partial final cycle (e.g., position-limit mid-cycle) yields
  `committed != forced` ⇒ fail.
- The measurement frame is validated to be a real session end elsewhere in the report
  (`metric_sequence_complete`, `session_closed is not None`, `source_completion_ok`), so
  `metrics[-1]` is not a mid-stream artifact.
Reading cumulative totals at the last completed metric frame is therefore sound; every
divergence resolves fail-closed.

### Nonblocking observations

1. `requested_tool_prompts` is keyed on the literal substring `"get_current_utc_time"` in
   typed-job text (605-607). If it is 0 (a different tool name or phrasing),
   `tool_prompt_requested` is False and the **entire** `tool_response_channel_gate` is
   bypassed as `True` (616-617). Correct for the current UTC-time sustained suite, but a
   future tool-turn test with another tool would silently skip the gate — key it on the
   advertised-tool set or on the presence of `expected_response_any` responses instead.
2. `checked == requested` leans on the runtime labeling exactly the fresh tool turns with
   `expected_response_any`; both under- and over-labeling fail the count (safe), and the
   per-turn `turn_id` anchoring (175) prevents a non-tool response from standing in for a
   tool turn. No false-pass surface, but worth stating the dependency.

### Verdict / next action

**APPROVE (static scope).** The cardinality gate cannot pass a subset: exact
`requested==checked==committed==forced` plus the ANDed per-turn validity gate reject
dropped turns, partial commits/forces, over-counts, and counted-but-garbage turns; the
tests cover each. `committed==forced==N` is measured from cumulative monotonic counters at
a validated session-end frame, with every divergence fail-closed. The two nonblocking
notes (hardcoded tool-name key; expected_response_any dependency) are robustness
follow-ups, not blockers. As always, the count/validity gates are structural — the live
focused ≥4-cycle and fresh sustained runs remain the confirming evidence. No ASR-threshold
relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 68 — focused repeated-tool EOTR-feedback artifact review (2026-08-09)

Adversarial review of `repeated-tool-eotr-feedback-r1`; no source/GPU/commit/services.
`passed=False`, but most of the RED is checker attribution, not runtime failure.

### Runtime: the EOTR-feedback fix works — but only ONE fresh cycle was exercised

- All four responses are the **full, correct** sentence "The current UTC time is fourteen
  hours and eight minutes UTC." with substantial audio (~165 KB each). r2's channel leak /
  near-silence is **gone** — the answer is on the agent channel.
- Turn 1's EOTR observation is textbook: `observed=true, wait_steps=1, timeout=false,
  eotr_token_id=22, first_post_drain_token_id=22, first_post_drain_frame==observed_frame==113,
  unexpected=null, feedback_token_id=22, feedback_frame=113, feedback_committed=true`. The
  bound=1 gate, write-back, and committed flag all fired correctly.
- **But only turn 1 invoked the tool.** `client_smart_turn_v1` mode ran, yet `committed`
  reached only **1** (values seen 0→1); turns 2/3/4 have `expected_response_any=[]` and
  answer the identical repeated prompt **from context** (correct behavior). So the
  repeated-EOTR-feedback path was exercised **once**, not four times — the ≥4-cycle proof
  A66 asked for is **not** delivered by this artifact.

### Checker attribution defects (spurious RED)

1. **Cardinality: `requested=4` vs `validated=1`, committed/forced `null`.** The gate equates
   4 identical requested prompts with 4 fresh tool cycles; the model legitimately answers
   repeats from context, so there is 1 fresh cycle. Also `metrics[-1]` reports the counters as
   `null` even though they reached 1 mid-session — the field is absent at the final frame, so
   sampling `metrics[-1]` blindly mis-reads it. Fix: read the cumulative **max / last-non-null**
   counter (not `metrics[-1]`); make the `committed==forced==validated` equality count **fresh
   cycles** (turns with a validated EOTR observation), and **separately** assert every requested
   prompt produced a correct answer (fresh OR context). Do not equate requested-prompts with
   fresh-cycles.
2. **`decoded_silence_close` required.** All four full, audible answers close via
   `no_text_since_bos_watchdog` (trailing silence after a complete answer). Requiring
   specifically `decoded_silence_watchdog` rejects a healthy close. Accept
   `no_text_since_bos_watchdog` as a clean bounded completion; reject only `internal_drain` /
   `model_or_turn_taking` (the r2 pathologies).
3. **Queue slope RED on a short run.** `first_block_mean_ms == last_block_mean_ms == 70.98`
   (zero drift) yet `linear_slope=3.187 ms/min > 1` → RED. The slope is regression noise over a
   4-turn run; gate it only above a minimum sample count/duration, or treat `first==last` mean
   as dispositive no-drift evidence.
4. **`function_channel_idle` over the whole response window.** It counts non-PAD function
   frames across the entire response (85-156), including the call+EOTR span (85-113) which is
   legitimate. Scope the idle check to **post-EOTR** (`frame > observed_frame`).

### Real runtime observation — not fatal, but not a pure checker artifact

Scoping (4) is necessary but **not sufficient**: the non-PAD function frames extend to **156**,
i.e., **114-156 lie after EOTR (113), through the entire answer window.** The answer is full
and correct, so this is not breaking output — but the function channel is **not cleanly idle
during the answer**. The write-back fixed the terminal EOTR *feedback*; it did not necessarily
silence the function head during the spoken answer. Recommend a targeted check of whether frames
114-156 carry real effective function tokens or a telemetry artifact; if real, decide whether
post-EOTR function activity must be PAD (hard gate) or is a tolerated advisory as long as the
agent channel is correct. Do not silently drop this by only narrowing the window.

### Exact recommendations

- **Test design (blocks the ≥4-cycle proof):** vary the four tool prompts/args so each forces a
  fresh call (distinct query, advanced simulated clock, or interleaved context change), so
  `committed==forced==validated==4`. As-is the artifact proves one cycle.
- **Checker fixes:** (a) counter source = cumulative max / last-non-null, cardinality keyed to
  fresh cycles + a separate answer-correctness assertion over all requested prompts; (b) accept
  `no_text_since_bos_watchdog` as a clean close; (c) queue slope gated by minimum run
  length / `first==last` no-drift; (d) scope `function_channel_idle` to post-EOTR.
- **Runtime follow-up:** characterize the post-EOTR non-PAD function frames (114-156) and decide
  the idle policy.

### Verdict / next action

**REQUEST CHANGES — but on the checker and the test, not the core fix.** The EOTR-feedback
repair is confirmed working for the one fresh cycle it exercised (perfect EOTR observation +
write-back + full agent-channel answer). The RED is dominated by attribution defects
(cardinality vs context-answered repeats, null counter at `metrics[-1]`, `decoded_silence`-only
close, whole-window function-idle, short-run slope noise). Fix those, **and** redesign the
prompts to force four fresh cycles, then re-run — that run, with `committed==forced==validated==4`
and four agent-channel answers, is the actual ≥4-cycle proof. Separately characterize the
post-EOTR function-channel non-PAD frames before calling the function-idle criterion satisfied.
No ASR-threshold relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 69 — checker-fix review + A68 frame-domain correction (2026-08-09)

Review of the sustained_strict_v3/transcribe_sustained checker fixes; no
source/GPU/commit/services.

### A68 runtime "observation" was a frame-domain error — retracted

Verified in the r1 artifact: `eotr_observation` carries **both** clocks —
`observed_frame=113`/`feedback_frame=113` are **model** frames, and
`transport_frame=156` is the transport frame of the committed-EOTR emission. The
non-PAD function frames I cited (85-156) are **transport** frames and max out at exactly
**156** = the EOTR transport frame; there are **no** non-PAD effective-function frames
>156, and BOS is at transport 157. So the function channel **is** cleanly idle during
the answer. Comparing model-frame 113 to transport-frames 114-156 was invalid; **I
retract the A68 "not cleanly idle during the answer" observation.** The runtime is clean.

### Checker fixes — verified correct

- **Post-EOTR filter now transport-domain (186-205):** filters `response_metrics` to
  `transport_frame > observation.transport_frame` before counting non-PAD function frames,
  so the legitimate call+EOTR span (≤156) is excluded and only genuine post-EOTR leakage
  would flag. For r1 this yields `function_channel_idle=True`. If no observation exists the
  filter keeps all frames (fail-closed, and `eotr_valid` is already False). Correct.
- **Bounded-watchdog close (230-231):** accepts `{decoded_silence_watchdog,
  no_text_since_bos_watchdog}` and nothing else — admits r1's healthy closes, still rejects
  `internal_drain`/`model_or_turn_taking`. Correct.
- **Last FC snapshot (255-260):** reads the counters from the last `function_calling` dict
  rather than `metrics[-1]` blindly. Verified against r1: the last snapshot (transport 2114)
  **does** carry `client_eou_sotc_committed_count=1`, so this reads 1, not the A68 null.
- **`requested` from `expected_tool_call` metadata (505/550/633):** replaces the hardcoded
  `"get_current_utc_time"` substring with `prompt_expects_tool(...)`. Removes the A68
  silent-bypass surface.
- **`queue_required` optional (242-244, 654):** focused `--no-require-queue` bypasses the
  queue gate; sustained defaults required. Reasonable — a 4-turn run can't measure drift.
- **ASR recompute preserves the tool gate (transcribe_sustained 249-252):**
  `sustained_runtime_gate_passed` re-asserts `not tool_prompt_requested or
  tool_response_channel_gate is True`, so the ASR recomputation cannot launder a failed tool
  gate. Correct.

### Decision: is `requested==validated==committed==forced` correct for expected_tool_call prompts?

**Yes — and it is now the correct, tight invariant, conditional on prompt design.** With
`requested` = count of prompts marked `expected_tool_call` (each *intended* to force a fresh
cycle), every such prompt must produce a fresh commit, forced BOS, valid EOTR, and a
validated response; any shortfall — a marked prompt answered from context (`committed<N`),
an invalid EOTR (`validated<N`), or a partial forced BOS (`forced<N`) — is now a **real**
failure, not the A68 false-fail. The four-way equality also cross-checks the three
independent counters (validated EOTR responses, boundary commits, forced BOS) against the
request count, so a commit without a valid EOTR, or an EOTR without a forced BOS, diverges
and fails. Sound.

The correctness now rests entirely on **prompt design**: each marked prompt must be
genuinely un-answerable from context. This is the A68 failure mode in disguise — r1's four
identical "current UTC time" prompts returned the same `14:08`, so the model answered 2/3/4
from context. If the new four prompts still hit a **fixed** simulated clock, the model can
again context-answer and the gate will (now correctly) fail — but on a **test-design** bug,
not a runtime defect. Requirement: advance the simulated clock (or use genuinely distinct
queries) so each `expected_tool_call` prompt *requires* a fresh call, and log per-turn
whether a fresh commit occurred so a `validated<N` failure is diagnosable as model-shortcut
vs prompt-not-fresh-forcing.

### Nonblocking hardening

1. `function_snapshots[-1]` takes the last dict, not the last dict with a **non-null**
   counter. It works for r1 because the cumulative counter persists to the final frame, but
   a trailing reset/idle frame that drops the field would reintroduce a null. Prefer the
   **max** of the (monotonic) counter across snapshots, or the last non-null — immune by
   construction.
2. The structural `audio_nonempty` is `audio_bytes>0`, which a near-silent WAV also
   satisfies; audibility is enforced only by the ASR gate's dBFS/semantic check. So a
   focused structural-only run must not be read as proof of audible answers — the
   ASR-scored sustained run stays necessary.
3. A legacy report lacking `tool_prompt_requested` satisfies the recompute's tool clause
   vacuously (`not None → True`); harmless for current reports, worth a note.

### Verdict / next action

**APPROVE the checker fixes (static scope), and A68's frame-domain runtime concern is
withdrawn.** `requested==validated==committed==forced` is the correct tight invariant for
`expected_tool_call`-marked prompts, contingent on the prompts genuinely forcing four fresh
calls — advance the clock / vary queries and log per-turn fresh-commit, or the invariant
false-fails on a test-design bug. Adopt the `max`/last-non-null counter source. The
ASR-scored sustained run with four fresh cycles (`committed==forced==validated==4`, four
audible agent-channel answers) remains the confirming evidence. No ASR-threshold relaxation,
no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 70 — final focused-probe hardening: adversarial review (2026-08-09)

Adversarial review of the final focused-probe hardening in
`tools/qualification/sustained_strict_v3.py` plus `tests/runtime/test_sustained_qualification.py`
(with the `transcribe_sustained` recompute as consumer). Static scope: working tree read,
runtime counter provenance verified in `server.py` and the RNNT turn-taking patch, unit
tests executed on CPU (25/25 pass across `test_sustained_qualification.py` and
`test_transcribe_sustained.py`). No source edits, no GPU, no services.

### 1. Monotonic max non-null counters — correct, with counter provenance verified

`evaluate_tool_cycle_cardinality` (249-286) now takes the **max over all non-null int**
`client_eou_sotc_committed_count` / `post_fc_client_bos_forced_count` values across
`function_calling` snapshots — exactly A69 nonblocking item 1. The trailing counterless
frame is covered by test (the passing case's second metrics entry carries no counters).

I traced the counters to their source rather than assuming monotonicity:

- `client_eou_sotc_committed_count` increments **only** in the `_boundary_eou_sotc`
  branch (patch ~line 814): raw function-channel token equals the FC SOTC id, agent idle,
  and `_external_user_eou_requested` set — i.e., once per client-committed tool-cycle
  boundary. Ordinary (non-tool) typed EOU commits never touch it, so in **mixed** mode
  `committed == requested tool prompts` is coherent, not accidentally inflated.
- `post_fc_client_bos_forced_count` increments only at the forced single post-FC agent
  BOS (~line 1053).
- Both reset **only** via `_reset_rnnt_turn_taking_state` /
  `set_external_user_eou_mode`, and every reachable call path runs from
  `VoiceChatEngine.start()` (server.py 1988-2026; `mode_fn(True)` fires after
  `prefill_for_new_stream`'s reset, per the ordering comment). Within the probe's single
  WebSocket session the counters are cumulative and monotonic, so max == final value.
  Sound.

Residual (nonblocking): `max` would **mask** a hypothetical mid-session reset — 4 cycles,
reset, one more cycle still yields max 4 and passes. Resets are provably session-start-only
today, but the cheap invariant `max == last-non-null` is a monotonicity *witness*: a
deviation flags a reset/attribution bug instead of silently absorbing it. Also
`isinstance(x, int)` admits `bool`; theoretical only.

### 2. Four freshness-dependent prompts — A69's requirement satisfied for the probe, still open for mixed

`TOOL_ONLY_TYPED_PROMPTS` are four distinct imperatives, each demanding a **new** call and
**new** verification code; the tool description now advertises a fresh code per call and
invites re-calls. For a plumbing probe this nudge is legitimate — the claim under test is
runtime EOTR/channel correctness across repeated cycles, not un-nudged model initiative —
but conclusions must be worded as "the runtime sustains four fresh cycles when the model
complies," which the nudge makes likely rather than guarantees.

**Mixed mode did not get the freshness treatment.** `DEFAULT_TYPED_PROMPTS` still repeats
the identical "Use the get_current_utc_time tool…" prompt every 4th slot (5 repeats in a
1200 s run), and `prompt_expects_tool` marks every repeat, so the new gate requires
`checked == committed == forced == 5` on the full sustained run. A context-answered repeat
(the A68 collapse mode) skips the call, drops that turn from `tool_response_checks`
(`checked < requested`), and fails the run. That is a **strictness increase, not a
weakening** — but the false-fail hazard A69 flagged is now live on the mixed tier with no
per-repeat prompt variation to defuse it (the real-clock advance only helps the semantic
layer when a call actually happens). Decision to log explicitly: either accept that the
sustained verdict now also gates model tool-compliance-on-repeat, or extend freshness
variation to the mixed prompt cycle.

### 3. Per-invocation unseen codes — semantic attribution verified; two structural caveats

The attribution chain is sound end to end: the server-side
`response.function_call_arguments.done` event (which a context-answer cannot fabricate)
causes the client to bind `expected_tool_result_by_turn[turn_id] = code`; `response.done`
attaches `expected_response_any=[code]`; membership in `tool_response_checks` yields the
`checked` count; and the ASR tier requires **every** speech response to semantically match
its own code (`passed_audio == len(prepared)` in `transcribe_sustained`). Code N is
revealed only in call N's output, so for the four-prompt probe the model cannot state
`amber`/`violet` before their calls, and a stale spoken code fails the positive match. The
structural layer (per-turn valid EOTR observation, `wait_steps==1`, feedback commit, idle
function channel post-EOTR, bounded watchdog close) and the semantic layer are
**independent**, and the four-way cardinality equality ties them to the request count.
Four genuinely fresh cycles are therefore well-attested when the report passes. Adversarial
fail-closed checks all hold: rejected tool prompt → `checked < requested`; spurious call on
a non-tool mixed prompt → `checked > requested`; call with no materialized response →
`checked < requested`; never-committed EOTR feedback → no observation → `eotr_valid` False.

Caveats (all nonblocking for the designed 4-prompt/1-call-per-turn probe):

1. **Code indexing counts distinct turns, not invocations.** `verification_code` is
   indexed by `len(expected_tool_result_by_turn)` (441-448). A chained second call within
   one turn reuses the length (turn A gets cobalt then topaz; turn B is then issued topaz
   **again**), and a 5th call wraps modulo-4 back to cobalt — both duplicate codes across
   turns and degrade the unseen-code property. The pre-EOU SOTC guard blocks a second
   boundary commit while the agent is idle, but the guard block runs only when the agent
   is **not** open, so a mid-answer chained call is not provably impossible.
2. **The cardinality equality omits the EOTR-invocation count.** `eotr_observations` /
   `eotr_observed_count` are recorded but never gated, so a chained extra invocation
   (two EOTR cycles, one boundary commit, one forced BOS) would pass all four equalities.
   Cheap closure for both caveats: gate `max(eotr_observed_count) == requested`, issue
   codes from a dedicated per-call counter (or per-run random codes), and assert
   uniqueness of issued codes in the report.
3. The code tuple is fixed, public, and ordered; within a session the model sees only past
   codes, so guessing the next is improbable but nonzero. Per-run random codes close it.

The receiver-side capture (371-408) is conservative: capture requires both a count
increment **and** `feedback_committed is True`; a commit-lagging frame is picked up on a
later frame; a never-committing cycle records nothing and the affected turn fails closed;
a count jump of 2 records one observation and the orphaned turn fails closed.

### 4. Mixed sustained semantics — not weakened

- The live suite invokes `sustained_strict_v3.py` with defaults only
  (`run_live_suite.py` 2214-2226): mixed mode, queue gate required, full duration. Every
  pre-existing conjunct of `passed` is intact; the tool gate is a new conjunct — strictly
  stronger.
- `--no-require-queue` is opt-in for focused probes; the report self-describes via
  `queue_required`, `typed_prompt_mode`, `typed_prompt_sequence`, `report_kind`, and
  `duration_seconds`. The ASR recompute honors `queue_required` from the report, so a
  focused report recomputes as what it is; it cannot silently occupy the suite's sustained
  slot because the suite generates that report itself with defaults. Manual evidence
  assembly must check those fields — worth one sentence in `docs/qualification.md`.
- Recompute parity was checked conjunct-by-conjunct against the inline `passed`
  expression, including the new `serialized_typed_error is None` and tool clauses; legacy
  reports keep the A69-item-3 vacuous-pass semantics (unchanged, still harmless). One
  shallow spot: the recompute trusts the stored `tool_response_channel_gate` boolean
  rather than re-deriving it from `tool_response_checks` + `tool_cycle_cardinality` (the
  same trust depth as `queue.passed`); re-asserting the stored four-way equality is a
  cheap hardening.
- Two mixed-surface changes to record, neither a weakening: the tool description text and
  the always-present `verification_code` field change the model-visible surface of mixed
  runs versus pre-hardening baselines (comparability break, not a gate change), and the
  new `input_turn_detection == client_smart_turn_v1` precondition (330-331) binds the
  sustained gate to the Smart-Turn configuration — intended on this branch.
- `bounded_watchdog_close` hard-codes watchdog closes as the *only* acceptable tool-turn
  terminations, matching the adopted A64/A65 design; if the runtime ever gains genuine
  EOS closes for post-FC responses, this gate must be revisited deliberately.

### 5. Tests

25/25 pass on CPU. The new tests are genuinely adversarial where it matters: trailing
counterless snapshot (the exact A69 max fix), pre- vs post-EOTR leak windows, malformed
EOTR observation, cardinality permutations including None counters and zero-requested,
prompt-mode validation, and recompute red paths for `queue_required` and the tool gate.
Gaps (nonblocking): no case for the code-index collision or the ungated
`eotr_observed_count`; no monotonicity-violation case for the max; the receiver capture
loop is embedded in `run()` and untested by construction.

### Verdict / next action

**APPROVE the final focused-probe hardening, static scope.** The
`requested == validated == committed == forced` invariant now rests on verified
session-cumulative counters read reset-tolerantly, four prompts that genuinely demand
fresh cycles, and per-turn code attribution that a context-answer cannot satisfy —
structural and semantic layers independently attest four fresh cycles, and the mixed
sustained gate is strictly stronger with defaults unchanged in the live suite. Nonblocking
hardenings, in priority order: (H1) gate `max(eotr_observed_count) == requested`; (H2)
per-call or per-run-random code issuance with a uniqueness assertion; (H3) `max ==
last-non-null` monotonicity witness; (H4) recompute re-derives the cardinality equality;
(H5) decide the mixed-tier freshness question from §2 explicitly; (H6) one-line doc note
that `queue_required=false` / `tool-only` reports are focused evidence only. The
confirming GPU evidence remains as before: a focused tool-only artifact with all four
codes spoken and ASR-matched plus the ASR-scored mixed sustained run. No ASR-threshold
relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 71 — repeated-tool r2 diagnosis + agent-open function-channel PAD guard (2026-08-09)

Adversarial review of the r2 RED diagnosis and the new agent-open function-channel PAD
guard. Evidence inspected: the
`~/.local/state/nemotron-voicechat/traces/repeated-tool-eotr-feedback-r2` artifact
(report + 8.7 MB events.jsonl), the live upstream wrapper diff in
`Speech-nemotron-voicechat`, the canonical `nemotron-voicechat-rnnt-turn-taking.patch`,
`server.py` telemetry export, and the runtime tests (guard-related tests executed on CPU,
16/16 pass). No source edits, no GPU, no services.

### 1. The r2 diagnosis is independently confirmed from raw telemetry

The r2 focused probe (tool-only, serialized, 4 prompts) failed **only** on
`function_channel_idle` for turns 2–4; every other conjunct passed — all four EOTR
observations valid (`wait_steps==1`, feedback committed, token 22, no unexpected token),
cardinality `4==4==4==4`, queue green, no protocol errors. The leak signature, read
directly from events.jsonl rather than the report:

- Turn 2: EOTR feedback commits at transport 489 (model 420); forced post-FC BOS at 490
  (model 421, `agent_open` flips true); at transport **491** (model 422, the **first
  ordinary agent-open position**) the function channel's effective token is **12068 —
  byte-identical to `agent_token_id` at the same position** — with `fc_state` fully
  inactive (`active`/`awaiting_response`/`awaiting_eotr`/`injecting_response` false,
  `forced_tokens` 0); PAD resumes at 492. Turns 3 and 4 repeat the identical pattern at
  812 and 1121, same token 12068 duplicating the agent channel.
- The damage tracks the diagnosis: answers degrade progressively — turn 1 (no leak)
  spoke a ~3.2 s answer; turn 2 decoded only " Current UTC"; turns 3–4 only " Current",
  all closed by `decoded_silence_watchdog`. The recurrent mechanism is real and verified
  in the wrapper: the next position's function embedding is
  `embed_tokens(gen_function_text[:, current_frame_idx - 1])`, so a non-PAD duplicate at
  BOS+1 is fed back into the model on the following position.
- The r2 events carry `pre_eou_*` but **no** `agent_open_*` telemetry keys, confirming
  the artifact predates the guard — the run is valid failing evidence, not a
  post-fix rerun.
- Checker validation loop closed: the transport-domain post-EOTR filter (A69) attributed
  each leak correctly (491 > 489 within response 2's metric window, etc.). The A68-class
  "leak during the answer" is now **real, in the transport domain, at a different phase**
  (post-BOS) — the hardened gate went RED on a genuine runtime defect. That
  retroactively validates the A69/A70 checker work.

### 2. The guard — verified against all five review criteria

The guard is the `else` (agent-open) arm of the existing client-EOU boundary block, in
both the canonical patch and the live wrapper (byte-identical — see §4):

- **Smallest correct containment.** One conditional added at the exact site that already
  owns effective-function-token policy (the pre-EOU guard's structural mirror for the
  agent-open phase), three diagnostic attributes, resets, telemetry, tests. No new state
  machine, no timing change, no change to FC async, and the entire block is gated on
  `_external_user_eou_mode` — legacy mode behavior is untouched. Narrower alternatives
  (BOS+1-only suppression) would leave later agent-open positions exposed for no gain;
  broader ones (unconditional PAD while agent open) would break legitimate FC phases.
  This is the minimum that removes the leak class.
- **Ordering.** The guard runs before `_apply_fc_state_machine` (which reads
  `gen_function_text[:, current_frame_idx]` post-guard — asserted by the patch-contract
  test's `assertLess(suppression, state_machine)`) and before anything else consumes the
  position, so the suppressed token can neither be misread as SOTC/EOTC nor reach the
  next position's fusion embedding. The EOTR wait/observe/drain path lives in
  `_run_fc_async_steps` — a separate loop that never executes this block — so the guard
  provably cannot interfere with EOTR observation, injection, or drain.
- **Both effective tensors.** Suppression writes PAD into both
  `function_predicted_tokens[:, frame_offset]` and
  `gen_function_text[:, current_frame_idx]` — the same pair as the `force_output_pad`
  and pre-EOU paths; the patch-contract test asserts both writes appear ≥2× in the
  boundary block. `raw_function_predicted_tokens` and the function-logit trace keep the
  raw token for diagnostics, with `effective_token_reason = "agent_open_guard"`.
- **No suppression of legitimate FC work.** The exemption set was verified against the
  real `fc_state` keys (`active`, `awaiting_response`, `awaiting_eotr`,
  `forced_function_tokens`, `injecting_response` — all confirmed as the state machine's
  actual key names, no typo'd `.get()` silently defaulting), plus `_opens_fc_cycle`
  exempting a raw SOTC so a model-initiated FC entry while the agent speaks behaves
  exactly as before the guard. Server-side `background_active` is engine bookkeeping
  (`stream_id in _fc_async_bg`), not a wrapper key — and cannot race the guard because
  FC async does not run ordinary steps concurrently. Between EOTR feedback commit and
  the forced BOS, `fc_state` is already inactive but the agent is still closed, so that
  window belongs to the pre-EOU arm, not this guard.
- **Reset/diagnostic coverage.** The three counters are initialized in `__init__`, reset
  in `_reset_rnnt_turn_taking_state`, and reset in **both** branches of
  `set_external_user_eou_mode`; exported in both server telemetry sites
  (`agent_open_suppressed_tokens`/`_last_raw_token_id`/`_suppressed_frame`, server.py
  ~1751 and ~2953); covered by the telemetry-mapping test (which, fittingly, uses 12068
  as its fixture token) and the patch-contract test asserting guard structure, counters,
  and ordering. All 16 selected tests pass.

### 3. Adversarial residuals (nonblocking)

1. **A hallucinated SOTC at an agent-open position still passes.** `_opens_fc_cycle` is
   deliberate behavior-preservation, and r2 shows zero evidence of this leak class
   (every leaked token was a text-token duplicate, never SOTC; `eotr_observed_count`
   maxed at exactly 4). But if the model ever duplicates/hallucinates SOTC post-BOS, it
   enters FC async mid-answer. A70's H1 (`max(eotr_observed_count) == requested` in the
   sustained checker) is the right backstop and gains urgency from this guard's shape.
2. **The guard is necessary but not proven sufficient for a GREEN rerun.** Turn 1 of r2
   had a clean function channel yet still closed mid-sentence ("…fourteen twenty six
   UTC," — `no_text_since_bos_watchdog`, verification code never spoken, expected
   `cobalt`). That deficit is leak-independent: the structural probe may go GREEN after
   the guard while the ASR semantic tier still fails on unspoken codes. Treat the guard
   as containment of the truncation mechanism, not as closure of the focused probe.
3. `UserEouSettlement.observe`'s step recorder omits the `agent_open_*` keys — fine,
   since agent-open suppression cannot occur during settlement (agent closed), but worth
   a comment if that recorder ever widens its window.
4. `set_external_user_eou_mode(False)` resets `agent_open_*` but not `pre_eou_*` — a
   pre-existing asymmetry the new code did not repeat; cosmetic.
5. Counter diagnostics are last-value + cumulative (no per-event history); adequate,
   since the sustained checker's `function_channel_nonpad_frames` provides per-frame
   forensics.

### 4. Patch/wrapper parity

The full uncommitted diff of the `Speech-nemotron-voicechat` checkout (all five patched
files) is **content-identical** to the canonical
`nemotron-voicechat-rnnt-turn-taking.patch` (compared modulo hunk headers) — the
canonical patch carries exactly what the live wrapper runs, including the guard and the
`_opens_fc_cycle` exemption.

### Verdict / next action

**APPROVE the diagnosis and the guard.** The r2 RED is a confirmed runtime defect — the
model duplicates its agent-channel token onto the function channel at the first ordinary
post-FC-BOS position, and the recurrent function-embedding feedback of that duplicate
truncates the spoken answer, compounding across cycles. The guard is the smallest
correct containment: correctly placed before FC-state and feedback consumption,
suppressing both effective tensors, exempting every legitimate FC phase and
model-initiated SOTC, fully reset and telemetered, with real test coverage. Confirming
evidence remains a GPU rerun (r3): structural gate GREEN with
`agent_open_suppressed_tokens` recording the suppressions (expected ≥1 per affected
turn), **plus** the ASR semantic tier actually matching all four verification codes —
residual §3.2 predicts the semantic tier is still at risk, and a semantic RED there
would be an answer-completion defect, not this leak. Land A70-H1
(`max(eotr_observed_count) == requested`) alongside. D8 intact.

## Addendum 72 — focused GPU r3: guard works, truncation persists; r2 duplicate was a symptom (2026-08-09)

Adversarial review of the post-guard focused GPU rerun. Evidence inspected:
`traces/repeated-tool-eotr-feedback-r3` (structural report + events),
`traces/repeated-tool-eotr-feedback-r3-asr-final` (ASR-scored report), the matching
model trace `traces/model/d2bb2fe3-…` (per-position agent/function tokens, 2160 model
steps), and the r1/r2 artifacts for contrast. No source edits, no GPU, no services.

### 1. What r3 shows

**Structural tier: fully GREEN for the first time.** All four turns:
`eotr_valid` (wait_steps==1, feedback committed), `function_channel_idle`
(`nonpad_frames: []` everywhere), cardinality `4==4==4==4`, bounded watchdog closes,
report `passed: true`. The guard operated exactly as A71 predicted:
`agent_open_suppressed_tokens` incremented 0→1→2→3, one suppression per affected turn,
each at the first ordinary post-BOS position, each suppressing raw **12068** — the same
duplicate class r2 leaked. (A69's frame-domain lesson applies to the model trace too:
its `frame` field is the transport clock; on the model clock every forced BOS fires at
EOTR-feedback+1 as designed — e.g. turn 1 EOTR at model 146, BOS at model 147.)

**But the answers are unchanged.** Per-turn texts are token-identical to r2 modulo the
clock string: turn 1 `"Current UTC time is fourteen fifty two UTC,"` (stalls at the
comma, code never spoken), turn 2 `" Current UTC"`, turns 3–4 `" Current"`. The
ASR-scored report is the predicted RED: runtime gate true, semantic 0/4 (transcripts
"Current UTC time is 1452 UTC." / "Current" / "Current" / "Current" — no code words).

### 2. Verdict on the r2 duplicate: symptom, not cause

Three independent lines close this:

1. **Suppressing the duplicate changed nothing.** The guard PADded the function-channel
   copy of 12068 at BOS+1 in turns 2–4 — altering the next position's
   function-embedding input relative to r2 — and not a single agent-channel token
   changed. Greedy decoding is deterministic here (r2 and r3 texts are identical), so
   if the feedback of the duplicate had been causal for the stall, r3's trajectories
   would have diverged. They did not.
2. **The stall is the model's own text head.** The model trace shows the agent channel
   emitting PAD (token 12, `ctl=pad`) from the stall point onward; the closes are the
   bounded watchdogs 27–30 frames later. Nothing runtime-side suppresses the text:
   the TTS-ratio pacing guard is unarmed in turns 2–4 and never exhausted in turn 1
   (`rem` ≈ 105 of 135 at close), no forced turn-taking fires mid-answer, no barge-in.
3. **The r1 contrast isolates the trigger.** r1 — same runtime, same watchdog-EOS-
   after-PAD-runs in history, tool output `{"timezone","spoken"}` **without** a
   verification code — produced four **complete** answers ("The current UTC time is
   fourteen hours and eight minutes UTC.", ~165 KB audio each), including turns 2–4
   which answered from context. The model-visible deltas in r2/r3 are exactly: the
   `"verification_code": "<word>"` field in the tool output, the code-bearing tool
   description, and prompts demanding the code. Turn 1 stalls precisely at the
   position where the code phrase would begin. r1 also falsifies the
   "external-EOS-in-history" alternative: r1's history contained the same
   watchdog-EOS-after-PAD pattern and later answers stayed complete — what r2/r3's
   history uniquely contains is *truncated mid-sentence* answers, which is the
   plausible driver of the progressive collapse (9 tokens → 2 → 1 → 1, deterministic
   in both runs).

**Diagnosis:** the model fails to vocalize the verification-code content — an
out-of-distribution tool-output side-field (`verification_code: "cobalt"`) that the
model reads but cannot carry into speech; its text-head distribution collapses to PAD
at that point. The r2 function-channel duplicate was a co-symptom of the same
degenerate post-FC state — worth suppressing (channel hygiene; EOTR/feedback
protection; it is what makes r3's structural tier clean), but never the cause. The
injection path itself is exonerated: the full JSON including the code demonstrably
reaches the function channel (`function_text` shows
`<TOOL_RESPONSE>[{…"verification_code": "cobalt"}]</TOOL_RESPONSE>` per cycle,
43-token injections), and EOTR/BOS mechanics are exact.

### 3. Next smallest evidence-driven step

The discriminating experiment is **fixture-side only** (tools/qualification — no
runtime source, no new instrumentation):

1. **r4 A/B: move the code into the `spoken` field as natural speech** — tool output
   `{"timezone": "UTC", "spoken": "14:52 UTC, verification code cobalt"}` (drop the
   side-field). The model already vocalizes the `spoken` field faithfully (turn 1
   proves it through the digits-to-words conversion). Expected-code semantics in the
   checker are unchanged — `expected_response_any=[code]` still matches the ASR
   transcript. If answers complete and codes are spoken: lesion confirmed as
   side-field vocalization, qualification standardizes code-in-`spoken`, and the
   four-fresh-cycles evidence (A69) becomes reachable. If turn 1 still stalls at the
   code: fall back to codes that are certainly in the speech distribution
   (spoken-number codes, e.g. "seven three"), which discriminates
   field-position vs code-word OOD.
2. **Checker hardening (cheap, alongside):** the structural tier accepted a
   one-token answer — `text_nonempty` is satisfied by `" Current"`. Add a text-channel
   semantic pre-check: reuse `semantic_candidate_match` against
   `expected_response_any` on the response **text** (words, not audio) in
   `sustained_strict_v3`. That would have turned r2/r3 structurally RED for the right
   reason, without GPU ASR, and cannot false-pass a stalled answer. (Join it to the
   still-open A70-H1 `max(eotr_observed_count) == requested`.)
3. **Only if both A/B arms stall:** add an agent-head top-k logit trace at post-BOS
   positions, mirroring the existing bounded `S2S_FUNCTION_LOGIT_TRACE_TOPK`
   diagnostic (same shape: env-gated, JSON-only, argmax-consistency-checked). That
   answers "how close was the code token to PAD" — but it is runtime instrumentation,
   so it stays behind the fixture A/B in the smallest-step ordering.

### Verdict

**The guard is confirmed working and worth keeping; the r2 duplicate is adjudicated a
symptom.** r3 is the designed outcome of A71's residual §3.2: structural GREEN,
semantic RED, and the RED is an answer-completion defect — specifically a
model-capability limitation vocalizing the out-of-distribution verification-code
side-field, first injected in r2's fixture, compounded across cycles by truncated
answers accumulating in history. No runtime defect is implicated by r3: EOTR, boundary
commits, forced BOS, channel discipline, and watchdog bounds all behaved exactly to
spec. Next action is the r4 fixture A/B (code inside `spoken`), plus the text-channel
semantic pre-check and A70-H1 in the checker. No ASR-threshold relaxation, no dropped
tool turn, no watchdog loosening. D8 intact.

## Addendum 73 — r4 discriminating fixture + checker hardening review (2026-08-09)

Adversarial review of the r4 fixture and checker hardening in
`sustained_strict_v3.py` / `transcribe_sustained.py` plus tests. Static scope; unit
tests executed on CPU (25/25 pass) and the new evaluators run **empirically against the
real r3 artifact** as a red-team check. No source edits, no GPU, no services.

### 1. What landed, verified piece by piece

- **Unique-leading code in `spoken`, side-field removed.** Tool output is now
  `{"timezone": "UTC", "spoken": "Cobalt. Current UTC time is 14:52 UTC."}` — the
  A72-prescribed arm, with the code moved to the **leading** position. Codes still
  rotate cobalt/topaz/amber/violet, unique per turn on any GREEN run (see cardinality
  below).
- **Structural model-text code match.** `contains_literal_token_sequence` does
  case-insensitive `[a-z0-9]+` token-sequence containment; it is required only when
  `expected_text_semantic_required` is set, which the writer sets **only in tool-only
  mode** — so mixed mode (whose expected value is a digit time-string that the text
  channel legitimately renders in words) cannot false-fail on it. Correct scoping.
- **Post-EOTR audibility.** New conjunct requires ≥1 response-attributed metrics frame
  after the EOTR transport frame with `output_audio.rms_dbfs` above the per-frame
  watchdog threshold (fallback −90). I verified the feed empirically: r3's client-side
  metrics carry both fields, and the four responses show 37/9/5/5 audible post-EOTR
  frames — so the conjunct passes on real delivered audio (no false-RED) while a
  silent-WAV pathology (A69 residual 2) now fails. Missing observation or missing
  response metrics fail closed via `eotr_valid`/empty-audible respectively.
- **Cardinality gains the EOTR dimension (A70-H1 landed).** The equality is now
  `requested == validated == max(eotr_observed_count) == committed == forced`. This
  closes A70's chained-call hole: an extra invocation in any turn drives
  `eotr_observed > requested` → RED, which also structurally protects code uniqueness
  (the len-based-indexing collision of A70-H2 can now only occur on an already-RED
  run — H2 downgraded to cosmetic).
- **Independent ASR recompute re-derives the equality (A70-H4 landed).**
  `sustained_runtime_gate_passed` no longer trusts `cardinality["passed"]`: it
  re-derives the five-way equality from the stored per-field counts and requires it
  **and** the stored channel gate. A tampered or buggy `passed: true` with mismatched
  counts now fails the recompute. (Depth note: `validated_tool_responses` could still
  be cross-checked against `len(tool_response_checks)` in the same report — one more
  cheap line, nonblocking.)

### 2. Empirical red-team of the new gate

Running the new evaluators over the **real r3 report + events**: the observed gate goes
**RED with `text_semantic_match` false on all four turns** — exactly the failure A72
said the structural tier missed — while `eotr_valid`/idle/audible/cardinality all stay
green (correct attribution: model-side, not runtime). Substituting synthetic
code-leading texts ("Cobalt. Current UTC time is fourteen fifty two UTC.") flips the
gate GREEN. The gate can demonstrably fail and demonstrably pass, on real data, for the
right reasons. Test red-paths cover every new dimension (eotr_observed=3, text-semantic
miss, inaudible post-EOTR, serialized error, recompute equality mismatch).

### 3. Mixed-mode semantics — gates not weakened, one fixture exposure found

Gate-wise, mixed is strictly **stronger**: the cardinality EOTR dimension and post-EOTR
audibility now apply to mixed tool turns; the text-semantic clause is vacuous there by
design; queue and live-suite defaults are untouched (run_live_suite still passes no new
flags). Recompute parity holds conjunct-for-conjunct, including `serialized_typed_error`.

**Finding (recommended before the next mixed run):** the code-leading `spoken` string is
built **unconditionally** — mixed-mode tool turns now also receive
`"Cobalt. Current UTC time is …"` even though mixed prompts never mention codes and the
mixed expectation is the time. If the A72 stall pattern recurs after the leading code
word, a mixed tool turn could truncate before the time and fail ASR-semantically — a
probe-content-induced failure imported into the mixed tier. One-line fix: condition the
code-leading form on `prompt_mode == "tool-only"`, keeping mixed's spoken time-only.
Also note (provenance, not a gate issue): mixed baselines' model-visible tool content
has now changed twice since r1; run-to-run comparability claims must cite the fixture
generation.

**Retroactivity note:** the recompute requires the `eotr_observed_count` key inside
`tool_cycle_cardinality`, so re-evaluating pre-r4 tool reports (r2/r3 era) now fails
their runtime gate. Fail-closed on legacy artifacts is the right direction — but anyone
idempotently re-running the evaluator over archived runs should expect flips to RED.

### 4. Is the A/B interpretation valid?

**Yes, for the question qualification needs answered — with two wording obligations.**

- The arm moves several variables at once (side-field removed, code into `spoken`,
  leading position, capitalized sentence form). r4-GREEN therefore proves "this fixture
  combination works and the runtime sustains four fresh, attributed, audible,
  code-bearing cycles" — it does **not** isolate which single aspect of the r2 fixture
  was the lesion. That finer ablation is not needed for D8; say "fixture-relative,"
  not "side-field-proven."
- The **leading** position is deliberately collapse-tolerant: r3's degenerate turns
  still emitted the first 1–2 content tokens, so even a persisting stall surfaces the
  code. That is the correct separation of runtime health from model completion
  capability (A72's finding) — but its dual is that **r4-GREEN does not certify
  complete answers**: a post-code stall passes every gate. D8 evidence must be worded
  as "four fresh cycles with audible attributed codes," with answer-completeness
  explicitly out of scope, or a completeness signal (e.g. terminal-period/token-count
  telemetry, non-gating) added to the report.
- All RED outcomes stay interpretable: text-semantic RED with structural green ⇒
  model-side code-vocalization failure under this fixture (e.g. reordering then
  stalling); empty text ⇒ deeper collapse; any structural conjunct RED ⇒ runtime
  regression. No outcome is ambiguous between runtime and model — which is the property
  the probe exists to provide.

### Verdict

**APPROVE the r4 fixture and checker hardening.** Every A72 prescription landed
(code-in-`spoken`, text-channel semantic pre-check, A70-H1) plus the A70-H4 recompute
re-derivation; the new gate is empirically shown to fail on r3 for the right reason and
pass on code-leading texts; mixed gates are strictly stronger with writer/recompute
parity. Before the next **mixed** sustained run, condition the code-leading `spoken` on
tool-only mode (§3). Confirming evidence: GPU r4 focused run — structural GREEN with
all five cardinality counts at 4, text-semantic green, and the ASR tier matching all
four codes; plus the wording obligations of §4 in any D8 claim. No ASR-threshold
relaxation, no dropped tool turn, no watchdog loosening. D8 intact.

## Addendum 74 — corrected r4 changes: mode isolation + fully independent recompute (2026-08-09)

Re-review of the corrected r4 tree. Static scope; targeted tests (26/26) and Ruff run
on CPU; the deepened recompute exercised empirically against the real r3 artifact. No
source edits, no GPU, no services.

### A73's finding is resolved — better than asked

`tool_definition(mode)` / `tool_result(mode)` make both the tool **description** and
the tool **output** mode-specific. Mixed mode is restored to the r1 baseline exactly:
description "Get the current clock time in UTC.", output
`{"timezone": "UTC", "spoken": "<now>"}`, expected value the time,
`text_semantic_required` False. The code-leading spoken form, the code-bearing
description, and the required text match exist only in tool-only mode. This goes beyond
A73's one-line suggestion — it also unwinds the A70/A73 mixed-baseline comparability
drift (mixed model-visible tool surface is again r1's). The isolation is locked by
`test_tool_definition_and_result_preserve_mixed_mode_isolation`, which pins both full
output shapes. As a bonus, `prompt_mode` now binds at the top of `run()` before the
session.update and the receiver closure — the A70 ordering fragility is gone.

### The offline recompute is now genuinely independent

`sustained_runtime_gate_passed` re-derives, from the report's raw records rather than
the writer's summary: `requested` from `typed_jobs` (`expected_tool_call is True`),
`validated` from `len(tool_response_checks)`, `observed` from
`max(observed_count)` over `eotr_observations`, and re-validates **all seven**
per-check predicates. The five-way equality must hold across the derived values *and*
match every stored `tool_cycle_cardinality` field, and
`bool(tool_prompt_requested) == derived` binds the stored flag to the evidence — which
finally closes the A69-item-3 vacuous-pass hole (a report can only skip the tool
clause if both the flag and the job records agree no tool prompt existed). The
committed/forced counts remain stored-value checks — correct limit, since metrics are
not persisted in the report.

Empirical checks: the real r3 report now recomputes **RED** (its checks predate
`post_eotr_audible`/`text_semantic_match` — fail-closed retroactivity, as A73
documented), and flips **GREEN** when the checks carry the new-shape passing predicates
with consistent counts. Test red-paths cover stored-field mismatch, observations-derived
mismatch, per-check predicate flips, serialized error, and the flag/evidence
consistency bind. Ruff clean; 26/26 tests pass.

### Residual notes (all nonblocking, carried forward)

A70-H2 (len-based code indexing) stays cosmetic — extra invocations are RED via the
observed-EOTR equality before a collision can matter. A73 §4's wording obligations for
any D8 claim (fixture-relative attribution; GREEN ≠ answer completeness) are
unchanged — they are obligations on prose, not code. Legacy tool-bearing reports
recompute RED under the deepened gate; archived-run re-evaluations should expect that.

### Verdict

**APPROVE.** Mixed behavior is provably preserved (r1 shape, test-pinned), the focused
probe keeps its discriminating fixture, and the offline ASR gate now independently
re-derives everything the report's raw records allow, empirically shown to fail r3 and
pass a compliant report. The r4 GPU run remains the confirming evidence, read under
A73 §4's wording. No ASR-threshold relaxation, no dropped tool turn, no watchdog
loosening. D8 intact.

## Addendum 75 — focused GPU r4: code omitted, not stalled-at; diagnostic direction (2026-08-09)

Adversarial review of `traces/repeated-tool-eotr-feedback-r4`, its ASR-final report,
and the matching model trace (`traces/model/f069317a-…`). No source edits, no GPU, no
service changes.

### 1. r4 confirmed: structural RED isolated to `text_semantic_match`

Verified from the report: four real calls, cardinality `4==4==4==4==4` (all five
counts), every per-turn `eotr_valid`/`function_channel_idle`/`post_eotr_audible`/
`bounded_watchdog_close`/`text_nonempty`/`audio_nonempty` green; only
`text_semantic_match` is false, on all four turns. ASR-final: 0/4 semantic, transcripts
carry the times but no code words. The A73/A74 gate design worked exactly as intended —
one conjunct isolates the model-side defect while every runtime property stays
attested.

### 2. The r4 texts refine the lesion: the model *skips* the code, it does not stall at it

The injected spoken was "Cobalt. Current UTC time is 15:12 UTC." — code **leading**.
The model trace shows every turn opening with its template token 12068 " Current" at
BOS+1 and turns 1–2 rendering the complete remainder **with a terminal period**
("Current UTC time is fifteen twelve UTC."). So with the code moved off the tail, the
r2/r3 "stall" disappears for early turns and the code is silently **omitted**: the
model paraphrases the tool result through a trained answer template for this tool and
drops novel content it cannot place. Trailing code → stall before it (r2/r3); leading
code → skip it and complete the template (r4). No arm has ever shown a post-FC answer
opening with novel content.

Two more findings from the trace:

- **The function-head mirroring is now dense and guard-suppressed.**
  `agent_open_suppressed_tokens` rose 0→6 across the run — zero in turn 1, two
  positions in turn 2, **every** answer position in turns 3–4 — with
  `agent_open_last_raw_token_id` tracking the concurrent agent token exactly
  (12068/17480/2142). The guard is load-bearing (function channel stayed idle), and
  mirroring **density** grows cycle-over-cycle in step with the answer-length
  degradation (8 → 8 → 3 → 1 tokens) — a measurable proxy of the degenerate state, and
  further confirmation of A72's symptom classification since degradation persists with
  feedback fully suppressed.
- Degradation is slower than r3 (turns 1–2 complete vs r3's 9/2/1/1), consistent with
  truncated-answer history contamination being one compounding driver.

### 3. Diagnostic direction: **APPROVE**, with ordering sharpened

The proposed package — raw/effective agent+function IDs around forced BOS, agent-head
top-k, native-vs-forced-BOS A/B — is the right escalation; A72 §3.3 pre-registered
exactly this once both fixture arms failed, and both now have. Order it by cost and by
what each result gates:

1. **Raw-vs-effective ID window around forced BOS (cheapest, telemetry-only).** The
   model's *native* prediction at the forced-BOS position is currently discarded when
   the force overwrites `gen_text`; capture it, plus a bounded window (≈BOS−2..BOS+8)
   of raw/effective IDs on both channels. This formalizes the mirroring-density proxy
   and answers the gating question: **if the native prediction at that position was
   already BOS, the forcing is a no-op and the A/B arm is unnecessary**; if it differs,
   BOS-provenance distortion becomes live.
2. **Agent-head top-k at BOS/BOS+1**, mirroring the existing bounded
   `S2S_FUNCTION_LOGIT_TRACE_TOPK` diagnostic (env-gated, JSON-only,
   argmax-consistency-checked). This discriminates hard template collapse (code tokens
   nowhere in top-k → model-capability limit, prompt-side fixes futile) from soft
   preference (code token near the top → instruction/fixture nudges can flip it).
3. **Native-vs-forced-BOS A/B — run only if (1) shows forcing overrides a different
   native prediction.** It is the decisive runtime-attribution experiment but the most
   confounded: disabling client Smart Turn changes EOU authority and injection timing,
   not just BOS provenance, and the probe's `client_smart_turn_v1` precondition needs a
   deliberate relaxation flag for that arm. Interpret any difference as
   "mode-correlated", not "BOS-caused", unless (1)/(2) corroborate. Note r1 already
   bounds this hypothesis: complete template answers *under forced BOS* exist, so
   forcing does not block template completion — the open question is only whether it
   suppresses novel first tokens.

Optional zero-instrumentation side-arm (cheap, parallel): a session-instructions nudge
("speak the verification code word exactly as returned") — top-k (2) supersedes it
mechanistically, but it directly tests the only lever a deployment would actually have.

### Verdict

**APPROVE direction.** r4 is verified as structural-GREEN/semantic-RED with the defect
cleanly attributed model-side; the runtime's repeated-tool machinery is attested by
every gate, with the agent-open guard demonstrably load-bearing (6 suppressions). Land
the diagnostics in the order above — (1) gates (3), and (2) decides whether any
fixture/prompt remedy can work or the D8 evidence should be finalized on r4's terms
under A73 §4's wording (four fresh, attributed, audible cycles; code readback
documented as a model-capability limitation). No ASR-threshold relaxation, no dropped
tool turn, no watchdog loosening. D8 intact.

## Addendum 76 — opt-in post-FC token trace: implementation review (2026-08-09)

Adversarial review of the A75-item-1 diagnostic: the opt-in post-FC raw/effective token
trace. Reviewed in the live upstream wrapper (md5 `1cb25371…`, mtime 15:29 — the tree
moved mid-review; see note below), canonical patch, `server.py`, `cli.py`, and tests
(102 + 3 subtests pass, Ruff clean, CPU only). No source edits, no GPU, no services.

**Mid-review note.** My first reads of the wrapper showed two defects: an
`UnboundLocalError` hazard (`awaiting_eotr_at_step` referenced by the async capture
condition but bound only in the non-forced branch — it would have crashed the phase-2
injection re-entry precisely when tracing was enabled) and capture prep (`.item()`
GPU syncs) running even when disabled. The tree was updated during review; the reviewed
state initializes `awaiting_eotr_at_step = False` at loop level and gates every capture
on step-start `_trace_candidate`. Both hazards are resolved in the reviewed hash;
this addendum's verdict applies to that hash, and patch/wrapper parity was re-verified
against it (full-diff content identical).

### Verified properties

- **Async capture ordering.** The capture precondition (`_trace_candidate`: final
  injected-response position `len(forced)==1`, or awaiting-EOTR with no forced tokens
  left) and the before-state (`previous_*`, BOS latch, fc_state) are computed at step
  start, before embedding; raw agent/function IDs are read from the model's `ans`
  **before** the async-PAD write and the forced-injection override. Raw-before-PAD
  holds.
- **Foreground capture ordering.** Raw agent comes from `raw_predicted_tokens`
  (appended at model output, before guards and turn-taking); the record is emitted
  **after** RNNT turn-taking, so at the forced-BOS position the pair is exactly the
  A75 measurement: raw = native prediction, effective = BOS, phase `forced_bos` via
  `forced_frame == current_frame_idx`, with `turn_taking_override` vs `model`
  distinguished for later positions. Raw-before-forced-BOS holds.
- **Window semantics.** `until_frame = t + frames − 1`, so the CLI's value 9 yields
  BOS..BOS+8 — nine foreground records — plus the async records at the last injected
  position and the EOTR wait step (= BOS−1 when `wait_steps==1`): the full A75 window
  "around forced BOS" is covered, ~11 bounded JSON records per cycle.
- **Drain-on-read server telemetry.** Single consumer (`_turn_state` →
  `drain_post_fc_token_trace()`), records attached to
  `function_calling.post_fc_token_trace` **only when non-empty**; each record carries
  `model_frame` + monotonic `sequence`, so late attachment to a metrics frame stays
  attributable. The probe-mode export path does not drain — no record stealing.
- **Reset & thread safety.** `_reset_post_fc_token_trace()` (lock-held; clears
  pending, sequence, window) is called from `_reset_rnnt_turn_taking_state` and both
  branches of `set_external_user_eou_mode` — same convention as the counters. Append,
  drain, and reset share one lock across the async thread, foreground thread, and
  server reader; records are JSON-only scalars (`ids_to_text` failures swallowed to
  ""); the 64-record cap drops oldest, and sequence gaps make any drop detectable.
  `until_frame` is written unlocked but single-writer (foreground turn-taking) — fine.
- **CLI isolation.** `--trace-post-fc-tokens` validated 0–32, env var exported only
  when >0; tests cover value 9 and reject 33.
- **Disabled mode is inert.** Default env "0" → every capture path is gated off (no
  extra `.item()`/GPU syncs in the reviewed state), `until_frame` never set, append
  and drain early-return, the telemetry key is absent (asserted by test) — inference
  results and trace volume are both unchanged. The trace code writes no model tensors
  in any mode.
- **Tests.** Patch-contract assertions pin the lock, the reset call, and the
  `t + frames − 1` window; the server test pins drain-once semantics (second read
  empty) and the absent-key contract; CLI tests pin bounds. Capture-site behavior
  itself is patch-text-and-telemetry tested — the same depth as the A71 guard,
  acceptable for a diagnostic.

### Nonblocking notes

1. If the BOS force ever failed to fire, latch-pending foreground frames would be
   candidates but never emit (stale `until_frame`) — the one failure mode this
   diagnostic goes blind on. The forced/committed counters would still flag it;
   optionally emit when the latch is pending regardless of window.
2. The reviewed-state hazard history (unbound variable in the enabled path) shows the
   enabled path has no CPU test exercising the async re-entry; a patch-text assertion
   for the `awaiting_eotr_at_step = False` initializer would pin the fix cheaply.

### Verdict

**APPROVE** (at wrapper md5 `1cb25371…`, with patch parity verified). The diagnostic
delivers exactly the A75-item-1 evidence — native-vs-effective at the forced BOS and
its surrounding window on both channels — opt-in, bounded, drain-once, reset-covered,
and provably inert when disabled. The gating question it answers (was the native
prediction already BOS?) decides whether the A75 native-vs-forced A/B arm is needed.
D8 intact.

## Addendum 77 — r5 with post-FC token trace: native PAD at BOS; next-step adjudication (2026-08-09)

Adversarial review of `traces/repeated-tool-eotr-feedback-r5` (trace-enabled focused
GPU run) and `traces/r5-asr-final/report.json`. No source edits, no GPU use, no
services.

### 1. r5 verified — the trace delivered exactly the A75-item-1 evidence

44 trace records, 11 per cycle, exactly the A76 design (last-injected + async-EOTR +
BOS..BOS+8). Per cycle, all four cycles identical in structure:

- **Async EOTR step:** raw agent PAD(12) / raw function EOTR(22), reasons
  `fc_async_pad` / `model_eotr` — the EOTR is the model's own prediction, agent head
  silent, as designed.
- **Forced-BOS position:** raw pair **PAD(12)/PAD(12)**, effective agent BOS(1),
  reason `post_fc_forced_bos`. **The model's native prediction at the opener position
  is PAD in all four cycles** — the forcing overrides PAD, not a competing opener.
- **Answer positions:** every emitted token has raw == effective with reason `model` —
  the runtime performs **zero** suppression or override anywhere in any answer. The
  template opener " Current"(12068) is the model's raw argmax in all four cycles; no
  code word ever appears in a raw stream.
- **Truncation onset is raw-visible:** turn 3's agent head goes raw-PAD after
  " Current UTC time"; turn 4 after " Current". The stall is the model's own text
  head, now proven at the raw channel, not inferred.
- **Mirroring, guard-contained:** turn 2 mirrors at 2 positions, turn 3 at 3, turn 4
  at 2 — all `agent_open_guard`-suppressed (function channel stayed effective-PAD).
  Notable: at mf 1041 the **function head predicted " UTC" one frame after the agent
  head had stopped** — the answer text migrating channels. The degeneracy is a joint
  decode pathology; the guard is load-bearing and the mirroring-density proxy (A75)
  is confirmed informative.
- Parakeet: 4/4 audible speech, identical truncation family, 0/4 codes. Structural
  RED isolated to `text_semantic_match`; cardinality `4==4==4==4==4`. And r5's texts
  reproduce r4's (trace disabled) — empirical support for A76's trace-inertness claim.

### 2. Adjudication of the A75 gate

The gating measurement is in: forcing overrides a **native PAD**, not a native BOS and
not a code-bearing opener. This kills the strong form of the BOS-provenance
hypothesis — there is no evidence the model would open the turn at all, let alone open
with the code: at the only opener position that exists, its argmax is PAD, and once
opened, its argmax is the template. What argmax cannot show is *proximity*: whether
BOS (at the opener position) or the code token (at BOS+1) was rank-2 or rank-2000.
That is precisely the still-unimplemented A75-item-2.

### 3. Exact next design

**r6 — agent-head top-k (immediate, no checker changes, no behavior change):**
piggyback an `agent_topk` field onto the existing `post_fc_token_trace` records (same
window, same drain/lock/reset), env-gated `S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK` (0–20),
reusing the function-logit-trace validation pattern (finite logits, bounded k, argmax
must equal the recorded raw agent token, JSON-only). Pre-registered decision rule:

- (i) code token **never** in top-20 at any post-BOS position, any cycle → hard
  template collapse; the grace-window A/B is moot; finalize D8 on r4/r5 terms with
  code readback documented as a model-capability limitation (A73 §4 wording).
- (ii) code token in top-5 anywhere → soft preference; cheapest levers first
  (instruction/fixture nudge arms), BOS-provenance work stays parked.
- (iii) BOS token in top-5 at the forced position → native opening is plausible; the
  grace-window A/B below becomes informative.

**Grace-window A/B (contingent on iii only):** arm A = current (force at EOTR+1);
arm B = defer the forced BOS up to N=8 frames, adopting a native BOS if one fires,
forcing at expiry — preserving the single-BOS invariant (exactly one BOS, native XOR
forced). **Pre-registered validity requirements — blockers if unmet:**

1. **Self-play suppression must be exempted inside the grace window.** Verified
   in-tree: the suppressor PADs a native LLM BOS while the agent is idle with no user
   speech, and client mode deliberately keeps `_post_tc_bos_exempt` False — an
   unexempted arm B measures the suppressor, not the model. (r5's raw PAD shows the
   suppressor did not fire in these runs, but the arm must exempt-or-verify; preflight
   the deployed `rnnt_self_play_suppression` config value.)
2. **Cardinality/telemetry lockstep.** A native opening splits the BOS accounting: add
   `post_fc_native_bos_count`, equality becomes
   `requested == validated == EOTR == committed == (forced + native)`, and the writer
   **and** the A74 recompute must change together or every arm-B report false-fails.
3. Trace support: a `native_bos` phase and a grace-frames-used field in the existing
   records; reset coverage for the new counter.
4. Tests: counter permutations (forced+native sum, native-only, over-count),
   suppressor-exemption red path, patch-contract assertions for the deferral bound.

### Blockers

**None on r5 or on the r6 top-k step.** The four numbered requirements above are
pre-registered blockers **for the grace-window arm only**, which per §2 is likely moot
(the trace shows no native opening tendency) and should not be built before r6's
top-k answer. The A73/A74/A76 machinery held end-to-end in this run: structural RED
isolated to the one model-side conjunct, all runtime gates green, raw evidence
captured opt-in with zero behavioral perturbation. D8 intact.

## Addendum 78 — r6 agent-head top-k implementation review (2026-08-09)

Adversarial review of the completed r6 diagnostic across the runtime tree and the
upstream wrapper (md5 `70c8ebb3…`; full-diff patch parity re-verified against the
canonical patch). Tests: 125 passed + 3 subtests (12 skipped), CPU only. No source
edits, no GPU, no services.

### Verified properties

- **Disabled-path neutrality.** With `S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK` unset/0:
  `prepare_agent_logit_diagnostic_nano` returns the production Nano root untouched (no
  scratch dir, no provenance — test-pinned); the factory parses top-k 0 and skips
  ID/artifact validation entirely; both wrapper call sites pass
  `capture_agent_logit_trace=False`; `_attach_agent_logit_trace` is never invoked;
  `_trace_agent_topk(None)` returns None. Production `custom_outputs` stay
  `["function_tokens","function_logits"]` — no text-logit head, no reductions, no
  telemetry key.
- **Candidate-only capture.** Both call sites (async and foreground) gate the capture
  flag on `_trace_candidate and _agent_logit_trace_topk`, so the bounded reduction
  runs only on the ~11 post-FC window steps per cycle. Fail-closed, not fail-open:
  `_attach_agent_logit_trace` raises if capture is requested while the diagnostic is
  disabled or `text_logits` is absent from the engine result.
- **Weight-identical overlay + startup fail-closed.** The overlay symlinks every
  artifact file (weight identity by construction — asserted by test down to byte
  equality through the symlink) and rewrites only `config.json` to prepend
  `text_logits`. It refuses to start (SystemExit) unless the source artifact carries
  the qualified skip-contract (`voicechat_skip_custom_text_logits`, contract string
  `"temperature=0, top_p=1, repetition_penalty=1"`, written by the release conversion
  tooling), the env value is 0–20, and the post-FC token trace is enabled at 1–32 —
  the same coupling the CLI enforces (`--trace-post-fc-agent-logits` requires
  `--trace-post-fc-tokens`; bounds test-pinned at 20 pass / 21 reject).
- **The skip-contract closes the argmax-consistency soundness gap.** The summarizer
  raises unless the raw text-head argmax equals the engine's predicted token — which
  is only sound under greedy decoding. The host-side sampler is bypassed exactly when
  `top_p=1, repetition_penalty=1, temperature∈{0,1}`, and the overlay refuses any
  artifact whose contract does not pin greedy sampling. So a non-greedy deployment
  cannot silently produce spurious mismatch crashes — it fails closed at startup.
- **Bounded JSON + argmax consistency.** The summarizer mirrors the qualified
  function-logit pattern: `[1, vocab]` shape check, finite-logits check, watched-ID
  range check, vocab≥k check, argmax==predicted raise, watched `{pad,bos}` with
  logit/rank/equal-count, top-k ≤ 20 scalar pairs. The wrapper adds decoded token
  text and re-asserts `predicted_token_id == raw_agent_token_id` against the A76
  record — a second consistency bind tying the logit row to the exact traced
  position. Records piggyback the existing post-FC trace (same drain/lock/reset).
- **Provenance.** When enabled, the run config gains
  `nano_agent_logit_diagnostic`: mode, top-k, trace frames, source and diagnostic
  config SHA-256s, `weights_changed: false`, sampling contract, and — notably —
  `performance_metrics_valid: false` with an explanatory note, so a probe run cannot
  be cited as a latency/throughput qualification measurement.
- **Tests.** Overlay opt-in/weight-identity/provenance flags; token-trace prerequisite;
  CLI coupling and bounds; summarizer boundedness + argmax-consistency red paths;
  invalid top-k parses; production-artifact rejection (fail-closed). 125+3 pass,
  patch parity holds.

### Nonblocking notes

1. `skip_contract["source_config_sha256"]` is format-validated (64 hex chars) but not
   recomputed against anything at overlay time; a stale contract on a since-modified
   config would pass. The overlay's own provenance records the live config hash, so
   post-hoc audit can catch it; recomputing at startup would be one line.
2. Enabled-path consistency violations (argmax mismatch, missing custom output,
   non-finite logits) raise and will kill the probe session mid-run — the correct
   fail-closed posture for a diagnostic, worth knowing operationally.
3. With the overlay loaded, the text-logit head is computed every step while
   reductions stay candidate-only; the provenance note already quarantines
   performance numbers, which is the right treatment.
4. Preflight: confirm the deployed production Nano `config.json` carries the
   skip-contract block (it is written by `finalize_nano_release.py` /
   `patch_public_skip_custom_text_logits.py`, both in the provenance inventory); a
   deployment predating it fails closed at startup with a clear message.

### Verdict

**APPROVE the r6 live probe** (wrapper md5 `70c8ebb3…`, patch parity verified). Every
reviewed property holds: inert when disabled, candidate-only when enabled,
weight-identical by construction, fail-closed at startup and at runtime, bounded
JSON-only output with two independent argmax-consistency binds, provenance that
quarantines performance claims, and test coverage on every red path. Run r6 with the
A77 §3 pre-registered decision rule; the grace-window A/B stays parked pending its
outcome. D8 intact.

## Addendum 79 — r6 fail-closed on PAD-pair buffered frame: fix design (2026-08-09)

Adversarial design review for the r6 cycle-3 fail-closed stop. No source edits; this
addendum specifies the fix exactly and the verdict is on the design.

### 1. Failure mechanism — verified in the optimizer source, and the crash was correct

The Nano PAD-pair optimizer's `buffered` branch (`runtime_optimizations.py`,
`pad_pair_generate_next_token`) defers the current input
(`state["pending"] = current`) and returns a **synthetic**
`GenerationResult(token_id=pad_id, custom_outputs={"function_tokens": [pad]})` with
**no vLLM position advanced**. On a traced post-BOS frame with
`capture_agent_logit_trace=True`, `_attach_agent_logit_trace` found no `text_logits`
and raised — the A78 fail-closed contract doing its job on an input its design did not
anticipate. The cycle-3 (not cycle-1/2) timing is fully explained: buffering requires
`pad_pair_should_draft(previous_effective_pad)` — a **prior PAD pair** — so cycles
whose answers filled the window with text (r5 turns 1–2, 8 tokens) never draft inside
the window, while cycle 3's early stall (raw PAD from ≈BOS+4) produced consecutive
PAD pairs inside BOS+8 and the first buffered traced frame. No data was corrupted;
the stop was honest.

### 2. Decision: explicit bounded no-model-position evidence; scheduler untouched;
window preserved

**Reject scheduler bypass.** Forcing sequential mode on traced frames would measure a
non-production scheduler on exactly the frames under measurement, and drags in
pending-drain complexity (a pre-window buffered input would still need honest
handling). Nothing about the bypass is *necessary*: the honest-record design below is
sound.

**Preserve BOS+8 window semantics.** Two structural facts make extension worthless:
(a) the decision-rule positions — the opener at BOS+1 and the first stall-onset
position — always follow a **non-PAD** effective position (BOS, or the last text
token), so `should_draft` is false there and those frames are **draft-immune by
construction**: they always carry a real logit row. (b) Extending the window only
adds deeper steady-PAD tail frames — precisely the ones the scheduler buffers and
whose distribution question (PAD dominant) the preceding real rows already answer.
BOS..BOS+8 stays, unchanged from A76, keeping r5 comparability and bounded volume.

### 3. Exact design (APPROVE this; the two rejected shapes below are BLOCK-worthy)

1. **Stamp at the source.** The `buffered` branch — and only it — marks the synthetic
   result explicitly (e.g., a `pad_pair_buffered` entry in `custom_outputs` or a
   dedicated result field). Every other decision path (`initial`,
   `sequential_bypass`, `sequential_conditional`, `sequential_pending_drain`,
   `sequential_pending_roll`, `pair_accepted`, `pair_rejected`) returns unstamped
   results.
2. **Factory honest path.** `_attach_agent_logit_trace`: if the result carries the
   stamp, attach a bounded no-position record —
   `{"no_model_position": true, "scheduler_decision": "pad_pair_buffered",
   "synthetic_token_id": <pad>, "total_tokens": <n>}` — and no logits. If the result
   is **unstamped** and `text_logits` is absent → raise exactly as today (the real
   error class — mis-built overlay, missing custom output — keeps its fail-closed
   contract). If a result is stamped **and** carries `text_logits` → raise
   (contradictory stamp; prevents the stamp from ever laundering a real row).
3. **Wrapper consistency bind survives.** `_trace_agent_topk` passes a no-position
   record through only after asserting the synthetic token equals the recorded raw
   agent token and equals PAD; any mismatch raises. The argmax check is skipped only
   because there is no argmax — the record still cross-binds to the A76 raw capture.
4. **Reconcilability.** The stamped records must reconcile with the existing pad-pair
   decision trace (`buffered` counter): stamped-record count within the window equals
   buffered decisions on those frames. This is an audit property, checked in review
   of r6-retry artifacts, not new plumbing.
5. **Documented bounded blind spot.** When a deferred pair later executes,
   `slice_position_outputs(accepted-1)` keeps only the **last** position's row; the
   first position of an accepted pair inside the window never yields a top-k
   anywhere. Acceptable: such positions are steady-PAD continuations (draft
   precondition), and the no-position record on the buffered frame plus the pad-pair
   trace make the gap explicit rather than silent. No attribution confusion arises:
   each frame's record describes that frame only.
6. **Tests (exact list):**
   - Optimizer unit: `buffered` → stamped; all seven other decision paths → unstamped.
   - Factory: stamped → no-position record, no raise; unstamped + missing
     `text_logits` → raise (existing red path preserved); stamped + `text_logits`
     present → raise.
   - Wrapper: no-position pass-through requires synthetic==raw==PAD; mismatch raises.
   - Schema: no-position record bounded, JSON-only keys as specified.
   - Window regression: `until_frame` semantics unchanged (existing tests must not
     move).
   - Shape reproduction: a simulated BOS+8 window whose tail frames return stamped
     synthetic results (the cycle-3 shape) completes without raising and yields
     real rows at opener/stall-onset positions plus stamped records at the tail.
   - Patch-contract assertions for the wrapper/factory changes; Ruff.

### Verdict

**APPROVE** this design for implementation and the r6 retry. **BLOCK** two adjacent
shapes if they appear instead: (a) inferring bufferedness from the mere absence of
`text_logits` — that deletes the fail-closed contract for the real error class; the
evidence must be an explicit stamp; (b) bypassing or reordering the PAD-pair
scheduler on traced frames — that measures a non-production runtime precisely where
production behavior is the question. Window stays BOS+8; opener and stall-onset rows
are draft-immune by construction, so the decision-rule inputs of A77 §3 are not affected
by buffering at all. D8 intact.

## Addendum 80 — r6 scheduler-provenance implementation vs the A79 contract (2026-08-09)

Adversarial review of the completed implementation against A79 §3, across
`runtime_optimizations.py`, upstream `model_factory.py` (md5 `2d3c5b35…`) and the
wrapper (md5 `eac84c3c…`), the canonical patch (full-diff parity re-verified), and
tests (126 passed + 3 subtests, 12 skipped; Ruff clean; CPU only). Review only — no
implementation edits.

### Contract compliance, item by item

- **Marker only on the exact synthetic buffered result, only while the diagnostic is
  enabled.** `_pad_pair_buffered_custom_outputs` stamps
  `voicechat_pad_pair_buffered: True` only when `S2S_POST_FC_AGENT_LOGIT_TRACE_TOPK`
  is non-"0", and is called from exactly one site — the `buffered` branch. All seven
  other decision paths return unstamped results. The installed-scheduler behavioral
  test proves it end-to-end: with the env set, the buffered result is stamped and the
  subsequent `pair_accepted` result is not.
- **Disabled path unchanged.** Env unset → the helper returns the identical
  single-key `{"function_tokens": [pad]}` dict as before (test-pinned, including
  tensor content). No scheduler decision logic moved; the stamp is additive-only.
- **Generic missing logits still fail closed.** The unstamped-and-no-`text_logits`
  branch raises exactly the pre-existing message; behaviorally covered.
- **Contradiction and non-PAD fail closed.** Marker present with `text_logits` →
  raise ("contradictory…", behaviorally covered). Marker path requires **both** the
  synthetic predicted token **and** the function token to equal `_text_pad_id` →
  raise otherwise (behaviorally covered) — stronger than A79 asked, which specified
  only the agent token.
- **Exact bounded schema.** The factory emits precisely
  `{no_model_position, scheduler_decision, synthetic_token_id, total_tokens}`
  (test-pinned by dict equality), and the wrapper enforces **set-equality** on those
  keys plus `scheduler_decision == "pad_pair_buffered"` — a record with extra or
  missing keys raises. Bounded JSON only; no logits fabricated.
- **Wrapper cross-check.** `_trace_agent_topk` requires
  `synthetic_token_id == raw_agent_token_id == text_pad_id` or raises — the A79 §3.3
  triple bind, implemented exactly.
- **`raw_agent_source`.** Both capture sites (async and foreground) label the A76
  record `"pad_pair_buffered"` vs `"model"` — the trace now distinguishes
  scheduler-synthetic raw values from genuine model predictions.
- **Window preserved.** `until_frame = t + frames − 1` and both capture conditions
  are untouched: BOS+8 semantics exactly as A76/A79.
- **Patch parity.** The canonical patch is content-identical to the upstream
  working-tree diff at the reviewed hashes.

### Findings

1. **Test gap (required):** the wrapper's no-position validation path is covered only
   by patch-text assertions ("does not match recorded raw PAD", `"raw_agent_source"`),
   not behaviorally. `_trace_agent_topk` is trivially unit-testable with the same
   `importorskip` pattern the factory tests use (a `SimpleNamespace` self with
   `model.stt_model.text_pad_id` suffices). Required: green pass-through; key-set
   mismatch raise; `synthetic != raw` raise; `raw != PAD` raise.
2. **Test gap (required, small):** the factory's marker-not-`True` arm (e.g.
   `voicechat_pad_pair_buffered: 1` or `False`) shares the "contradictory" raise but
   is not exercised; one parametrized case closes it.
3. **Interpretation caveat (record, no code change):** r5 predates the stamp, so its
   late-window `rawA=12` tail values were partially scheduler-synthetic, not model
   predictions. This does not touch A77's conclusions — the opener and stall-onset
   positions are draft-immune by construction (prior effective token non-PAD) and
   therefore genuinely model-raw — but r5's tail PAD runs must not be cited as
   model-distribution evidence. From r6 on, `raw_agent_source` removes the ambiguity.
4. **Cosmetic:** the stamp helper gates on the raw env string (`!= "0"`), so
   degenerate values ("00", padded whitespace) would stamp; unreachable in practice
   because the factory/server validation kills such processes at startup. Mirroring
   the parsed value would be tidier; not required.

### Verdict

**APPROVE the r6 retry** — the implementation satisfies every A79 contract clause,
in two places more strictly than specified, and the probe's fail-closed guarantees
are intact without any scheduler behavior change. **Required changes (test-only,
land with the merge; they do not gate the live retry):** the two test additions in
Findings 1–2. Window stays BOS+8. The A77 §3 decision rule applies unchanged to the
r6 artifact, now with `raw_agent_source` making scheduler-synthetic frames explicit.
D8 intact.

## Addendum 81 — post-fix re-review; installed tests executed pre-build via shim harness (2026-08-09)

Re-review of the two A80-era gap fixes and the expanded tests, at factory md5
`59e5162e…`, wrapper md5 `eac84c3c…` (unchanged since A80), regenerated canonical
patch (full-diff parity re-verified). Review only; no implementation edits.

### 1. Both fixes verified in source

- **Stamp activation** now `int()`-parses the env value with a `ValueError→0`
  fallback and stamps only when `> 0` — the A80 cosmetic finding is closed; "00",
  padded, negative, and garbage values can no longer stamp.
- **Empty-generation fail-closed**: `_process_inputs_to_outputs` previously returned
  a synthetic PAD answer with **no** `agent_logit_trace` when the engine yielded no
  token — under capture that would have produced a silently untraced window frame
  (neither a real row nor a stamped no-position record). It now raises
  ("no token was generated") when `capture_agent_logit_trace` is set, and preserves
  the legacy warning+PAD fallback otherwise. This closes a capture-integrity hole
  adjacent to, but distinct from, the A79 contract — a good catch by the
  independent review.

### 2. The decisive check: the installed tests actually executed, pre-build

`nemo` is not importable in the host venv — every `importorskip` test skips locally
(the host's "15 skipped"), so the new installed tests had **never run anywhere**, and
the container would execute them for the first time. Rather than settle for static
reading, this review built a shim harness (scratchpad-only, no repo changes):
stub parent packages point at the real Speech-checkout directories so the genuine
`model_factory.py`, `streaming_llm_engine.py`, and
`nemotron_voicechat_inference_wrapper.py` files load without executing the heavy
package `__init__` chains, with import-only stubs for `lightning`-adjacent, `vllm`,
`librosa`, `omegaconf`, `transformers`, and peer-module symbols that the test bodies
never touch.

Result: **40/40 pass with zero skips** in `test_public_runtime_optimizations.py`,
executing the real module code that the regenerated patch bakes into the container —
including, verified individually:

- scheduler parametrization: `pair_accepted` ([7,12,12]) **and** `pair_rejected`
  ([7,31]) arms, buffered-stamped → follow-up-unstamped in both;
- factory: no-position schema equality, marker+logits contradiction, marker-`False`
  contradiction (A80 required test 2), **both** non-PAD channels rejected
  (agent-token and function-token variants), generic missing-logits raise;
- empty-generation raise under capture;
- wrapper `_trace_agent_topk` (A80 required test 1): green pass-through,
  raw-mismatch raise, key-set-mismatch raise, and the 9-row
  3-real-opener/stall + 6-buffered-tail shape.

Fidelity argument for the container: the shims replace only module-level imports the
tested paths never call; every asserted behavior ran the identical source the patch
carries (parity re-verified after regeneration). Residual in-container risk is
limited to environmental differences in symbols these tests do not exercise.

### 3. Host state

131 passed + 3 subtests, 15 skipped across the three suites (consistent with the
reported focused-subset counts), Ruff clean, patch parity OK.

### Direction note

Mid-review steering (recorded for future readers): the promotion criterion for this
work is **text↔speech response parity**, not absolute text-input intelligence — the
model is small and may fail "intelligence tests" in both modalities, and pretty-good
text input is worth promoting. This reframes the close-out: the focused code probe
needs a speech-input twin so the code-readback limitation can be attributed
modality-relatively, and semantic gates should be read as paired text-vs-speech
comparisons on identical fixtures, while runtime/structural gates stay absolute.
Assessed in detail outside this addendum.

### Verdict

**APPROVE the rebuild and the r6 retry.** Both fixes are correct and in-tree, the
regenerated patch matches the upstream working tree byte-for-byte in content, and —
uniquely for this series — every installed test has now been executed against the
real modules before the container build, 40/40 green including both A80-required
additions. D8 intact.

## Addendum 82 — r6-retry: top-k evidence, A77 rule adjudication, parity next step (2026-08-09)

Adversarial review of `traces/repeated-tool-eotr-feedback-r6-retry` and its model
trace (`traces/model/d7c69725-…`), with the A77 §3 pre-registered decision rule and
the parity steering applied. Review only.

### 1. Structural validity and diagnostic provenance — clean

Four cycles, `passed:false` **only** via `text_semantic_match` (all other per-turn
conjuncts green), cardinality `4==4==4==4==4`, exact EOTR (`wait_steps==1`, feedback
committed, token 22) per cycle, function guard holding (channel idle), answers
audible. 44 trace records (11×4), full BOS+8 coverage. The A79/A80 machinery worked
live on its first outing: `raw_agent_source` splits 38 `model` / 6
`pad_pair_buffered`, all six buffered frames in the cycle-3/4 steady-PAD tails in the
exact alternating pattern the draft-defer scheduler predicts, with real rows at every
opener and stall-onset position — the draft-immunity argument of A79 §2 confirmed
empirically. Answer degradation reproduces (full, full, " Current UTC", " Current").

### 2. Code-piece ranks — the A77 question answered to the logit

- **Openers are real model rows** in all cycles: argmax " Current" (template),
  BOS deep (rank 39–191 at answer positions).
- **Cobalt (c1): absent** from every top-20. **Amber (c3): absent.** **Violet (c4):
  absent** (only single-char 'V'/'B' pieces at noise logits ~6). **Topaz (c2):
  "Top" reaches rank 2 at the opener, Δ2.75 logits below "Current"** — the closest
  any code ever came — and persists at ranks 6–20 through the cycle.
- **The model represents the task without executing it**: verification-*concept*
  tokens rise across cycles — "Code" rank 4 at the c3 EOTR row, "Verification"
  rank 3 and "Fresh" rank 6 at the c4 opener — while the code *words* stay absent.
- **The degradation mechanism is now measured**: PAD's logit climbs
  cycle-over-cycle until it noses past the template continuation — at c3's stall
  position (mf 728) PAD beats " time" by **0.12 logits** (22.12 vs 22.00), versus a
  7-logit template lead at the same position in c1. The collapse is gradual PAD-bias
  accumulation, not a cliff.
- **Forced-BOS positions**: argmax PAD with **BOS rank 2 in all four cycles**, but
  3.7–9.6 logits behind PAD.

### 3. A77 §3 adjudication

Criterion (i) — code never in top-20 — is **false** (Top, rank 2). Criterion (ii) —
code in top-5 anywhere — **fires**: this is a *soft preference*; instruction/fixture
nudges could plausibly flip an opener. Criterion (iii) — BOS in top-5 at the forced
position — fires *literally* (rank 2, every cycle) but is **superseded by the margin
data the rule's rank proxy could not see**: under the contractually pinned greedy
decode (the A78 skip-contract), only argmax matters, and PAD beats BOS by 3.7–9.6
logits at every opener. A grace window of any length can never fire a native BOS.
**The grace-window A/B is dead, and no inference change is justified now**: not the
grace window (arithmetically pointless), not sampling or logit boosts (they would
break the qualified greedy contract, perturb the measurement, and chase absolute
intelligence — the wrong criterion).

### 4. Next step under the parity criterion — APPROVE with constraints

A77-(ii)'s "nudge arms" optimize absolute text-input intelligence, which the steering
explicitly rejects as the promotion criterion. The correct next implementation is the
**exact Pocket-TTS speech twin plus `input_modality` provenance and a paired
comparator**, with these design constraints:

1. **Fixture identity**: same four freshness prompts (spoken verbatim), same code
   rotation and `tool_result` fixture, same serialized pacing, same session config
   and instructions, same checker (reuse `evaluate_tool_response_channels` /
   `evaluate_tool_cycle_cardinality` unchanged). Only the input pathway differs.
2. **The input pathway is the treatment, stated as such**: speech enters via audio +
   RNNT + client Smart-Turn EOU; typed via direct injection + explicit commit. That
   difference is what parity is *about* — document it as the compared variable, not
   a confound to eliminate.
3. **Heard-the-prompt control**: the speech arm must verify prompt intelligibility
   (input-WER against the fixture text, as multiturn already does) before any
   response failure is attributed — otherwise a speech-arm miss could be an
   input-audio artifact and the parity comparison is invalid.
4. **Provenance**: `input_modality: "typed"|"speech"` in the report plus fixture
   audio hashes / TTS generator provenance. Informational only — the recompute's
   gating semantics must not change.
5. **Gates**: structural/runtime conjuncts stay **absolute in both arms**; only the
   semantic readback outcome is parity-relative. Pre-registered parity read on N=4:
   report per-turn paired outcomes and discordant pairs; promotion requires **no
   text-only semantic failures** (text ≥ speech), not absolute success.
6. **Identical scoring**: same Parakeet model and thresholds in both arms; the
   diagnostics (token trace, top-k) may be enabled identically in both arms — the
   opener top-k comparison across modalities is the distribution-level parity
   evidence and comes free.
7. **No inference changes in either arm**; same container, same greedy contract.
   Comparator + modality field are runtime-repo tooling only — no wrapper/patch
   change is needed for this step.

### Verdict

**APPROVE the speech-twin + `input_modality` + paired-comparator implementation as
the next step, under the seven constraints above. BLOCK any inference change at this
time** — grace-window, sampling, and boost proposals are all refuted or disqualified
by the r6-retry margins and the parity criterion. If the speech arm reproduces the
code omission (the top-k evidence predicts it will), text input has demonstrated
response parity on this probe and the work is promotable with the limitation
documented at the model level; if speech succeeds where text fails, the delta
becomes the one remaining pre-promotion defect, now measurable to the logit. D8
intact.

## Addendum 83 — independent ASR evaluator container: design review (2026-08-09)

Adversarial pre-implementation review of the evaluator redesign. Scope: replace the
in-runtime-image NeMo `model.transcribe` backend (broken via Lhotse
`DynamicCutSampler` in the rebuilt runtime image) with a separate pinned evaluator
container: `nvidia/nemotron-speech-streaming-en-0.6b` at immutable HF revision
`ebe59e5a…`, Transformers 5.14.1, offline whole-WAV inference, 16 kHz mono float, one
model load, network-none, full provenance, fail-closed. Review only.

### 1. Coupling map (verified in-tree) — the blast radius is larger than one script

The broken `restore_from`/`model.transcribe` pattern has **four** consumers, all
executing inside the runtime image with a mounted `/models/asr.nemo`:
`transcribe_sustained.py` (invoked by `run_live_suite.asr_command` for
sustained/multiturn/memory-matrix and by `recover_offline_asr.py`),
`transcribe_direct_position_probe.py`, and `tools/conversion/eartts_component_gate.py`
(lines 721–730, transcribing EarTTS-rendered WAVs for streaming-vs-offline
comparison). `cli.py` additionally mounts the ASR model for source-conversion and
browser-e2e paths. Any fix scoped to `transcribe_sustained.py` alone leaves the
EarTTS gate and the probe broken on the same image.

### 2. Architecture verdict: the separate pinned container is correct

The incident is the argument: the judge broke because the **SUT's** image changed.
A qualification judge must not share an environment with the system under test.
The proposed design restores exactly the properties a judge needs — independent
pinning (image digest + HF revision), offline whole-WAV batch scoring, network-none,
one model load, reproducible provenance. The two alternatives are rejected for
qualification use: **NeMo streaming websocket** makes the judge depend on the running
SUT service (measurement coupled to the thing measured, realtime pacing, a network
path inside a network-none qualification) — disqualified; **NeMo-Speech.cpp GGUF**
puts quantization noise inside the measuring instrument with a weaker provenance
chain — acceptable for dev smoke, disqualified as the judge of record.

### 3. Required design constraints (APPROVE is conditional on all of these)

1. **Judge-swap calibration gate — BLOCKING.** Every archived semantic verdict
   (r1–r6, multiturn, sustained) was scored by the `.nemo` judge. Before the new
   judge's verdicts are adopted, it must re-score the retained r4/r5 WAV sets and
   match the archived `-asr-final` transcripts/semantic outcomes with **zero
   semantic-outcome flips** (per-file justification required for any divergence).
   Cheap (the WAVs are retained), decisive, and non-negotiable: a judge with
   different error patterns silently re-baselines every gate in the suite.
2. **Asserted-API fail-closed at build.** `AutoModelForRNNT` in Transformers 5.14.1,
   the model's official status, and DGX Spark (arm64) support are premises this
   review cannot verify from the tree. The evaluator image build must fail if the
   pinned Transformers lacks the exact classes, if the HF snapshot at
   `ebe59e5a…` fails a recorded hash manifest, or if arm64 wheels are unavailable.
   No `trust_remote_code` unless the remote code is itself revision-pinned, hashed,
   and recorded in provenance.
3. **Deterministic decoding, recorded.** Greedy (or the model's documented default)
   pinned in config and echoed into provenance; if the streaming model's HF API only
   exposes chunked inference, wrap it with a fixed chunk schedule and record the
   chunk parameters — chunking is otherwise a hidden judge parameter. Language
   forcing: probe, apply if supported, **record** either way; never silently ignore.
4. **Backend swap only — evaluation logic untouched.** `semantic_candidate_match`,
   number normalization, WER, report/recompute schemas, `--output-dir` read-only
   contract, and `max_silent_rate` all stay byte-identical; the `nemo_asr` import
   block moves behind a backend interface. `external_asr` provenance gains
   {backend, image_digest, transformers_version, model_id, hf_revision,
   snapshot_sha256, device, decoder_config, language}; the evaluator refuses to run
   if any provenance component cannot be captured.
5. **All four consumers migrate or are explicitly triaged.**
   `asr_command`/`recover_offline_asr`/`transcribe_direct_position_probe` retarget
   the evaluator image (drop the `.nemo` mount; model pinned in-image). The CLI
   conversion/e2e ASR paths must be verified-or-migrated before the next conversion
   qualification — flagged, not silently left on the broken path.
6. **EarTTS component ASR: two-phase split.** The gate's non-ASR work and WAV
   generation stay in the runtime image (they must — they exercise the EarTTS engine
   in place). Transcription moves to the evaluator container: phase 1 writes WAVs +
   a manifest with per-WAV SHA-256; phase 2 (evaluator, network-none) transcribes
   and emits transcripts keyed by hash; phase 3 merges and gates, failing closed on
   any hash mismatch or missing transcript. The gate's streaming-vs-offline
   *comparison* is judge-relative, so scoring both sides with the same new judge
   preserves its validity. Do **not** keep the gate's ASR on the runtime image
   (breaks identically) or on a last-known-good image (two judges of record).
7. **Sequencing and device.** The evaluator runs after the runtime stack is down
   (GPU handoff), `--gpus all --network none`, device recorded; the live suite
   orders this explicitly rather than relying on contention luck.
8. **Tests/docs.** Unit tests for the backend interface with a fake backend; a
   build-time import-and-load smoke inside the evaluator image; two or three tiny
   committed fixture WAVs with golden transcripts to catch silent judge drift; an
   EarTTS two-phase manifest round-trip test (including a corrupted-hash red path);
   updated `asr_command` tests; `docs/qualification.md` gains the evaluator image
   provenance requirements and the judge-calibration record.
9. **Family-independence disclosure.** The new judge is NVIDIA/Nemotron-family, as
   was the old one; it shares no weights or decoder with EarTTS/the SUT, and the
   calibration gate (constraint 1) empirically bounds transcript quality — record
   this in docs rather than implying vendor independence.

### Verdict

**APPROVE** the separate pinned evaluator-container design under constraints 1–9,
with constraint 1 (dual-scoring calibration on retained r4/r5 WAVs, zero semantic
flips) **blocking adoption of any new-judge verdict**, and constraint 6 defining how
EarTTS component ASR keeps working. **BLOCK** the streaming-websocket and GGUF
alternatives as judges of record, and **BLOCK** any attempt to fix Lhotse inside the
runtime image as the long-term judge home — that recreates the coupling this
incident just demonstrated. D8 intact.

## Addendum 84 — robust ASR evaluator implementation vs A83: pre-build review (2026-08-09)

Adversarial review of the uncommitted implementation (manifest, evaluator image,
backend, EarTTS finalizer, all consumers, CLI/config/tests). 95 focused tests + Ruff
reproduced green on the host. Review only. This review additionally performed
**network verification of the externally-asserted pins** — results below.

### 1. Externally verified (this review, against PyPI and Hugging Face)

- **All four requirement hashes resolve on PyPI to the correct artifacts** — and,
  decisively for arm64 build viability: the `tokenizers==0.22.2` and
  `safetensors==0.8.0` hashes are the **manylinux aarch64 wheels**
  (`…manylinux2014_aarch64.whl`); `transformers==5.14.1` and
  `huggingface-hub==1.27.0` are pure-Python wheels. `--require-hashes --no-deps`
  against the digest-pinned NGC PyTorch base is fail-closed by construction.
- **HF revision `ebe59e5a…` exists** for `nvidia/nemotron-speech-streaming-en-0.6b`,
  and `config.json` at that exact revision **hash-matches the checked-in manifest**
  (sha256 `dffe850b…`, 1284 bytes, `architectures: ['NemotronAsrStreamingForRNNT']`).
  The repo also publishes `.nemo` and `.gguf` variants; the manifest's
  `allow_patterns` correctly restricts the snapshot to the six inventoried
  Transformers files.

### 2. A83 constraint compliance (verified in-tree)

Backend (`nemotron_asr_backend.py`): exact-inventory snapshot verification with
per-file bytes+sha256 **and** an aggregate identity, runtime (not just build-time)
Transformers version pin, `trust_remote_code=False` everywhere,
`local_files_only=True`, one load, whole-WAV batch-one contract with strict
1-D-float32-finite input checks and a 300 s bound, greedy decode with **token IDs
recorded per response**, `torch.use_deterministic_algorithms(True)`, and a
provenance block that refuses to emit without an immutable
`VOICECHAT_ASR_EVALUATOR_IMAGE_ID`. Evaluation logic in `transcribe_sustained.py` is
byte-preserved (semantic matching, WER, gates, recompute); only the backend moved.
`read_and_resample` correctly casts the resample path back to float32.
Consumers: `asr_command`, `recover_offline_asr`, `transcribe_direct_position_probe`
all retarget the evaluator image **by resolved immutable ID** with the image-ID env
injected; zero stale `.nemo`/`/models/asr` references anywhere;
`reproduce_release_artifacts` dropped its dead `--asr-model` plumbing (it never ran
the gate). CLI/config: `[image] asr_evaluator` in the production config, bootstrap
existence/audit/build/state-tracking wired. EarTTS two-phase matches A83 §6
exactly: phase 1 writes `passed:false` + `pending_external_asr:true` + per-WAV
bytes/sha256 (an unfinalized report can never read green); the finalizer enforces
the pending contract, exact `{streaming, offline}` input set, path containment,
PCM16/mono/16 kHz canon, and hash identity before transcribing, keeps offline
diagnostic-only, and its red paths (mutated WAV, noncanonical audio) are
behaviorally tested. Sequencing unchanged (evaluations run after the stack is
signalled down). Build script pins `--platform linux/arm64`; the audit script
re-verifies snapshot and all version pins inside the image, network-none.

### 3. Required fixes before image build / GPU calibration

1. **Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the evaluator image ENV.**
   `use_deterministic_algorithms(True)` makes CUDA matmuls raise at the first
   `generate()` unless this is set; nothing sets it today. The failure would be
   fail-closed, not false-green — but it deterministically blocks the first GPU
   calibration run. One line.
2. **Add a build-time CPU inference smoke.** The build verifies imports, snapshot,
   processor invariants, and model load — but `NemotronEnglishAsr.__init__` +
   `transcribe()` never execute before GPU calibration (the host lacks
   Transformers, so no test can run them either). A build RUN instantiating the
   backend with `device="cpu"` and transcribing ~1 s of silence catches
   `generate`/processor API drift at build time, where it belongs.
3. **Close the baked-vs-mounted backend drift channel.** The image bakes
   `/opt/voicechat-asr/nemotron_asr_backend.py` (verified at build), but runtime
   resolution uses the repo-mounted copy via script-directory precedence. Either
   have the audit/evaluator assert the two files' sha256s match, or record the
   module file hash in provenance — one small check, otherwise an edited repo copy
   runs under a pinned image's provenance.

### 4. Standing blocks (unchanged from A83)

The **judge-swap calibration gate remains blocking for verdict adoption**: re-score
the retained r4/r5 WAV sets and match the archived `-asr-final` semantic outcomes
with zero flips before any new-judge verdict counts. Committed golden micro-fixtures
(A83 §8) are still recommended for long-term drift detection; the retained-WAV
calibration covers the immediate need. Test-coverage honesty: the 95 host tests
behaviorally cover the manifest/snapshot/finalizer/consumer wiring and red paths,
but the backend class itself first executes in the image — which is exactly why
fix 2 is required rather than optional.

### Verdict

**APPROVE for image build and GPU calibration once fixes 1–3 land** (1 and 2 before
the build; 3 may land with them). The design faithfully implements A83 — in several
places more strictly than asked — and this review has independently confirmed the
two premises it could not previously verify: the aarch64 wheel pins and the exact
model revision/hash. Calibration (A83 §1) remains the gate between "image works" and
"judge adopted". D8 intact.

## Addendum 85 — revised ASR evaluator: pre-build re-review (2026-08-09)

Re-review of the revised implementation. 96 focused tests + Ruff reproduced green on
the host; all three baked identity hashes (backend `8e6c23f5…`, manifest
`31ffea42…`, requirements `42f7e561…`) verified byte-for-byte against the repo files
by this review. Review only.

### 1. Every A84 fix landed — mostly stronger than asked

- **Determinism lock (A84-1+):** `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `NVIDIA_TF32_OVERRIDE=0` in the image ENV, **plus** a runtime assertion in
  `__init__` (the backend refuses to start anywhere the env is absent), TF32
  disabled on both torch backends, `float32_matmul_precision("highest")`, and
  `attn_implementation="eager"` — all echoed into provenance.
- **Build-time CPU smoke (A84-2):** the build instantiates `NemotronEnglishAsr` on
  CPU and transcribes one second of silence, asserting the result contract — API
  drift now fails the build, not the first GPU calibration.
- **Backend drift channel closed (A84-3+):** the backend is a proper package
  (`src/nemotron_voicechat_asr_evaluator`), baked into the image;
  `_verify_image_owned_backend` asserts at startup that the *executing* file is the
  image-owned path **and** hash-matches the baked contract — a repo-mounted copy can
  never run under the image's provenance. The audit script additionally compares
  baked-vs-checked-in SHAs for all four identity files, runs `pip check`, and
  executes **GPU canaries**: double-transcription determinism on silence, a
  60-second long-audio case, and provenance-flag assertions.

### 2. New robustness, verified in-tree

Encoder-exhaustion RNNT stopping (`max_length` removed from both `generate()` and
the manifest — eliminating a silent transcript-truncation parameter; a test pins its
absence); the hash-locked override set grown to seven packages with `pip check`;
lazy bootstrap lifecycle (audit-if-present, build-if-missing, refuse-when-offline);
atomic report writes everywhere (`atomic_json`); finalizer `--report-out` that
refuses to overwrite; a **resumable, content-addressed external-ASR
source-conversion stage** (self-hashed `stage.json` bound to the evaluator image ID,
with resume re-deriving every report from retained files); and **activation
revalidation** (`_validated_external_asr_report`: workspace containment,
`pre_asr_report_sha256` binding finalized report to the exact phase-1 report,
evaluator identity including image-owned backend path+SHA, and WAV re-hashing from
retained files). The direct-position probe canonicalizes to 16 kHz PCM and re-reads
the retained canonical WAV before scoring.

### 3. The `-P` import break — found, then fixed and regression-guarded in-tree

At first inspection, `finalize_eartts_component_asr.py` and
`transcribe_direct_position_probe.py` did `from transcribe_sustained import …` while
the image `PYTHONPATH` lacked `tools/qualification` — under `python3 -P` (no
script-dir injection), every in-container finalizer invocation (the live suite's
`eartts_asr_command` **and** the CLI's `_finalize_source_asr_stage`) would have died
with `ModuleNotFoundError`; host tests passed only because they don't run under
`-P`. The fix landed during this review as option (a): the image `PYTHONPATH` now
appends `/workspace/project/tools/qualification` (safe — `/opt/voicechat-asr`
precedes it, and the backend's path+SHA assertion, not path ordering, is the
load-bearing control), **together with exactly the regression test this review
required**: `test_safe_path_consumers_resolve_image_equivalent_imports`,
parametrized over both `-P` consumers, subprocess-running `python3 -P <tool> --help`
with only the image-equivalent `PYTHONPATH` entries and asserting exit 0. Verified
passing. The defect class is now guarded for every future `-P` consumer.

### 4. Late revision reviewed: root-ownership separation and reproduction attestation

The source-conversion flow was revised mid-review and the revision is verified
in-tree:

- **No mutation of root-owned artifacts.** The in-container (root) conversion tool's
  outputs — phase-one gate reports and the base `release/reproduction.json` — are
  now read-only inputs. The external-ASR stage writes exclusively under
  `work/external-asr/`, which the **host pre-creates** before the root container
  runs (host ownership guaranteed): per-gate finalized reports
  (`eartts-eager.json`/`eartts-graph.json` via `--report-out`), the self-hashed
  `stage.json` (evaluator-image-bound, resume re-derives all evidence from retained
  files), and the final attestation.
- **Self-hashed reproduction attestation.** After the stage completes, the host
  composes `work/external-asr/reproduction.json`: the base reproduction content
  (its own hash removed) plus `asr_evaluator_image_id` and
  `component_gates_external_asr`, re-self-hashed canonically and written atomically.
  The root-owned base report is never rewritten.
- **Activation reads the attestation — and does not trust it.**
  `_activate_converted_release` requires the attestation, verifies its `kind` and
  self-hash, checks the evaluator image identity and the exact gate-name set, then
  **re-derives** each gate's evidence from the retained phase-1 report and WAVs via
  `_validated_external_asr_report` (containment, `pre_asr_report_sha256` binding,
  image-owned backend path+SHA, WAV re-hash) and requires equality with the
  attestation. A tampered or stale attestation, a swapped WAV, or a different
  evaluator image all fail closed before activation.

Focused re-runs for this revision: cli/recover/finalize/backend suites green
(29 + 22 + 6 in this pass), Ruff clean.

### 5. Hygiene note

`src/nemotron_voicechat_asr_evaluator/__pycache__/` is present in the working tree
and would be copied into the image by `COPY src/nemotron_voicechat_asr_evaluator` —
stale bytecode is inert at runtime (CPython revalidates against source), but clean
or dockerignore it so the image content stays byte-deterministic.

### Verdict

**APPROVE the image build and GPU calibration.** The one blocking defect this review
found (§3) was fixed in-tree with the exact required regression test, and the late
root-ownership/attestation revision (§4) strengthens the evidence chain: host-owned
stage directory, immutable root-owned inputs, self-hashed attestation, and an
activation path that re-derives rather than trusts. Remaining sequence: clean the
`__pycache__` (§5), build, run the audit (whose GPU canaries double as the first
determinism evidence), then the A83 §1 retained-WAV calibration — which remains the
gate for adopting any new-judge verdict. D8 intact.

## Addendum 86 — pre-build re-review: identity propagation, activation recompute, TOCTOU coverage (2026-08-09)

Final pre-build review of the current revision. 100 focused tests + Ruff reproduced
green on the host. Review only.

### 1. The A85 blocker is confirmed closed

Image `PYTHONPATH` is ordered `/opt/voicechat-asr` → `/workspace/project/src` →
`/workspace/project/tools/qualification`, so the image-owned backend package always
wins resolution (with the startup path+SHA assertion as the load-bearing control),
and the cross-tool imports resolve under `python3 -P`. The exact regression test
A85 required is present and passing:
`test_safe_path_consumers_resolve_image_equivalent_imports`, parametrized over both
`-P` consumers, subprocess-executed with only the image-equivalent path entries.
The A85 hygiene item is also handled structurally: `.dockerignore` excludes
`**/__pycache__`, so the working-tree cache can never enter the image.

### 2. Immutable audited identity, resolved once and propagated

`_ensure_asr_evaluator` now resolves the tag to its immutable ID exactly once,
runs the audit script **on the ID** (not the tag — eliminating the
retag-between-audit-and-use TOCTOU), and returns that ID; source conversion, the
reproduction attestation, activation, and live launch all receive the same audited
value. Test-pinned: single `docker image inspect` (`inspections == 1`), audit
argument equals the ID, and propagation into source conversion is asserted
end-to-end.

### 3. Activation now re-derives everything it can

`_validated_external_asr_report` grew from identity checks to a full independent
recompute, all fail-closed:

- **Exact checked-in provenance contract** (`_asr_provenance_contract`): backend
  path (image-owned) and SHA-256 computed live from the checked-in
  `backend.py`, manifest SHA-256, model id, HF revision, snapshot aggregate, the
  complete decoder block (greedy, encoder-exhaustion, lookahead, eager,
  non-streaming), and the complete determinism set (deterministic algorithms,
  CUBLAS workspace, matmul precision, both TF32 flags, `trust_remote_code=False`,
  `network_required=False`). Any recorded provenance that deviates from the
  checked-in contract fails activation — including a repo edited between
  conversion and activation, which is the conservative direction.
- **Retained-WAV header parsing**: mono, PCM16, 16 kHz, nonzero frames,
  uncompressed — parsed from the retained file itself, plus identity re-hash and
  the result→input binding (`input_bytes`/`input_sha256` must identify the WAV).
- **Independent verdict recompute** using cli-local copies of
  normalize/WER (deliberately duplicated from the tools tree, so activation
  scoring shares no code with the thing it checks; divergence produces a
  fail-closed mismatch, never a false green): transcript non-empty, WER re-derived
  and required to **equal** the recorded value (not merely pass the threshold),
  threshold bounds-checked, and the required "5" content check re-run.

### 4. Mutation/TOCTOU tests are genuinely adversarial

Beyond field-level tamper parametrization (each mutated report field must raise its
specific message), the suite includes the case that matters most: a **re-written
WAV with a fully consistent forged hash chain** — source report, finalized report,
`pre_asr_report_sha256`, and result input bytes/SHA all updated coherently — which
still fails on the independent header parse. That proves the revalidation depends
on retained evidence, not on any recorded chain. The scoring fixture ("The answer
is five.") exercises the required-"5" recompute path.

### 5. Sidecar ownership (late fix, included)

The host pre-creates `work/external-asr` before the root conversion container runs;
every sidecar artifact — per-gate finalized reports, the self-hashed
evaluator-bound `stage.json`, and the final self-hashed reproduction attestation —
lives in that host-owned directory, and the root-owned phase-one reports and base
reproduction remain strictly read-only inputs. Activation reads the sidecar
attestation and re-derives it against retained evidence (§3).

### Verdict

**APPROVE for image build.** Every previously identified defect is closed with its
required regression coverage; the identity chain now runs
tag → resolve-once → audit-by-ID → propagate-everywhere, and activation
independently re-derives provenance, WAV integrity, and the semantic verdict from
retained evidence. Build, run the audit (GPU determinism canaries), then the
A83 §1 retained-WAV calibration — still the gate for adopting any new-judge
verdict. D8 intact.

## Addendum 87 — attention correction: eager retracted, default SDPA adopted (2026-08-09)

Adversarial review of the post-build attention correction before rebuild. 100
focused tests + Ruff reproduced green; the full identity chain re-verified. Review
only.

### 1. The calibration falsified the eager pin — and this review owns its share

A85 reviewed `attn_implementation="eager"` approvingly as part of the determinism
lock ("stronger than asked"). The calibration proved that judgment wrong on
correctness grounds: forced eager emitted blank/zero tokens on byte-identical
retained EarTTS WAVs, while the checkpoint's **default** attention path (SDPA) under
the unchanged determinism envelope (`deterministic_algorithms=True`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, highest matmul precision) produced
identical non-blank token IDs and transcripts twice. Blank output under a forced
non-default attention path is the signature of an untested eager mask/kernel route
in a recently added architecture — the vendor ships and validates this checkpoint
with SDPA (`config._attn_implementation == "sdpa"` is the model's own default).
The theoretical basis for forcing eager (SDPA backend-selection variability)
concerned chiefly backward-pass atomics, not inference-only forward passes; and the
design now proves determinism **empirically per image** rather than assuming it
kernel-theoretically — the audit's double-transcribe canary re-establishes
identical token IDs on every image build, with `use_deterministic_algorithms(True)`
still armed to raise on any op PyTorch knows to be nondeterministic. **Default
SDPA with a fail-closed assertion is the correct official and simplest
deterministic path.** The correction is right, and the review record should note
that empirical calibration outranked this review's theoretical endorsement —
exactly why the A83 calibration gate exists.

### 2. The correction is implemented fail-closed, end to end

- `from_pretrained` no longer overrides attention; the backend reads
  `model.config._attn_implementation` (private but pinned-version-stable;
  `getattr` default None → mismatch raise if renamed) and **raises unless it is
  exactly `"sdpa"`**; provenance records the *observed* value, not a constant.
- The activation contract (`_asr_provenance_contract`) pins
  `decoder.attention_implementation == "sdpa"` — a hypothetical non-SDPA run can
  never activate.
- Identity chain re-verified: the new `backend.py` SHA (`53b51421…`) matches both
  the Dockerfile ENV and the image label; manifest and requirements are unchanged
  and their labels still match; docs updated ("the checkpoint's supported SDPA
  attention, no TF32").

### 3. Evidence sufficiency

Sufficient for the attention decision: the A/B ran on byte-identical retained WAVs;
eager is disqualified on **correctness** (blank tokens — no ambiguity requiring
further determinism analysis), and SDPA showed twice-identical token IDs under the
full envelope. Two caveats, both already handled by the existing sequence: the
twice-identical evidence should also hold across container restarts — the audit
canary runs in a fresh container on the rebuilt image and re-proves it before
calibration; and per-input determinism accumulates across the canaries (silence,
60 s) plus every calibration WAV. The A83 §1 retained-WAV calibration (r4/r5
semantic agreement, zero flips) is untouched by this correction and remains the
adoption gate.

### 4. Non-blocking notes

1. The audit script asserts the determinism flags but not
   `decoder.attention_implementation == "sdpa"` explicitly — redundant today
   (instantiating the backend already raises on non-SDPA), but one assert line
   would keep the audit self-documenting.
2. A host test pinning `_asr_provenance_contract()["decoder"]["attention_implementation"]`
   to the backend's asserted value would tie the two constants together against
   independent edits. Cosmetic.

### Verdict

**APPROVE for rebuild and calibration.** The correction replaces a
theoretically-motivated override that empirically broke the model with the
checkpoint's official default path, keeps every fail-closed property (assertion,
contract pin, observed-value provenance, unchanged determinism envelope), and the
identity chain is consistent at the new backend hash. Sequence unchanged: rebuild →
audit (fresh-container determinism canaries) → A83 §1 retained-WAV calibration —
which remains the gate for adopting any new-judge verdict. D8 intact.

## Addendum 88 — non-silent build/audit canary: pre-rebuild review (2026-08-09)

Adversarial review of the speech canary added after the eager-attention incident.
100 focused tests + Ruff reproduced green; supply-chain pins externally verified by
this review. Review only.

### 1. The canary closes the exact hole the eager incident exposed

The silence-only audit was **structurally incapable** of catching a blank decoder:
blank-on-silence is correct behavior, so a decoder that always emits blanks was
indistinguishable from a healthy one. The fix pins real speech — LibriSpeech dummy
row 0, whose canonical transcript opens "Mister Quilter…" — and requires the
decoded words `mister` and `quilter` in both the **CPU build smoke** and the **GPU
audit**, with the audit's determinism double-run (identical token IDs and
transcript) now executed on real speech rather than silence. A blank, constant,
garbled, or wrong-model decoder now fails at build or audit; the 60-second silence
canary is retained for the long-input path. This converts the A87 lesson into a
permanent structural control.

### 2. Supply chain — externally verified, two-level content pinning

- This review queried the HF dataset tree API at the pinned revision
  `5be91486…`: the parquet's LFS sha256 is **exactly** the manifest's
  `parquet_sha256` (`4e69a06f…`), size 9,192,059 bytes. The pin is real.
- Verification is two-level: whole-parquet SHA after `hf_hub_download`
  (`repo_type='dataset'`, pinned revision), then the extracted row-0 FLAC bytes
  against `audio_sha256` before baking `/opt/voicechat-asr/speech-canary.flac`
  into the image — so the audit needs no network and the canary is immutable with
  the image. If HF ever removes the tiny long-lived dataset, the build fails
  loudly and the FLAC can be re-sourced from any LibriSpeech copy by hash.
- Identity chain re-verified: manifest label updated to `047dc805…` (matches the
  checked-in file), backend hash unchanged and matching, and
  `test_nemotron_asr_backend` pins the **entire** `speech_canary` contract dict,
  so a manifest edit that weakens the canary breaks a host test first.

### 3. Robustness against false-greens — layered correctly

The expected-words check runs through the full backend path (processor → model →
greedy decode → batch_decode), so it validates the semantic pipeline, not just
tensor plumbing. Residual risks, both acceptable and both fail-closed or covered:

- **Word matching** uses `.lower().split()`; punctuation adjacency could
  false-BLOCK (never false-green) — and the chosen pair "mister quilter" is
  sentence-initial with no adjacent punctuation in the canonical transcript.
- **Domain gap**: a decoder healthy on read LibriSpeech speech but broken on
  EarTTS-style synthetic speech would pass the canary. That residual is precisely
  what the A83 §1 retained-WAV calibration exists to cover, and it remains the
  adoption gate. Canary = plumbing correctness; calibration = domain semantic
  agreement. The layering is right.

### 4. Build-viability watch items (non-blocking, all fail-closed)

1. `pyarrow` (parquet read at build) and `soundfile` (FLAC decode at build and
   audit) are used but **not** in the hash-locked requirements — they are assumed
   present in the digest-pinned NGC base. If absent, the build fails loudly; if
   present, the base digest pins their versions immutably. Acceptable, but adding
   an assert-import with version echo to the pip-check step (or hash-locking them)
   would make the override lock self-documenting. `soundfile` additionally needs
   libsndfile with FLAC support from the base.
2. ~~The audit reads the baked FLAC without re-hashing it~~ — **closed during
   review**: the audit now independently asserts
   `sha256_file(canary.audio_path) == audio_sha256` before decoding, and also
   explicitly asserts `provenance['decoder']['attention_implementation'] ==
   'sdpa'` — which additionally closes A87's non-blocking note 1. Both verified
   in the current audit script.

### Verdict

**APPROVE the rebuild.** The canary is correctly aimed at the class of failure that
slipped through (semantic blanking invisible to silence), its pins are externally
verified at two levels, it is baked offline-immutable into the image, the
determinism evidence now rides on real speech, and every failure mode of the canary
itself is fail-closed. Sequence: rebuild → audit (real-speech determinism canaries)
→ A83 §1 retained-WAV calibration — still the gate for adopting any new-judge
verdict. D8 intact.

## Addendum 89 — judge-swap calibration adjudication (2026-08-09)

Adjudication of the A83 §1 calibration for the immutable SDPA evaluator image
(`sha256:598542…`), against the retained r4/r5 sets (8 responses). Review only.

### 1. What the gate was — and what it could never have been

A83 §1 demanded **zero semantic-outcome flips** against the archived verdicts. It
must be said precisely: **raw transcript or WER identity across two different ASR
models is impossible in principle** — the transcript is the judge's measurement,
not ground truth, and a new judge is a new instrument. Demanding it would have made
every judge swap unpassable forever. The calibration object is the set of
**decisions** the pipeline derives from the measurements. That is what the archived
comparison adjudicates.

### 2. The evidence, adjudicated

- **Decision agreement is exact, 8/8, at every level**: classification (all
  `speech`), `semantic_match` (all false), `negative_semantic_match` (all true),
  per-response `passed` (all false), aggregate `external_asr.passed` and report
  `passed` (false) — old and new judges identical on every decision the pipeline
  makes.
- **Determinism across containers and order**: fresh-container reruns in
  **reverse order** reproduced exact transcripts and token IDs for both sets —
  ruling out warm-up, ordering, and cross-file state effects in the one-load loop.
  This is the strongest determinism evidence in the series so far.
- **The standalone WER≤0.5 booleans diverge** (r4 turns 1–3, r5 turns 2–3) without
  flipping any verdict, because `semantic_match` is false everywhere and `passed`
  is a conjunction. The direction is informative: the new judge is **better** on
  every full-sentence answer (r4 t1/t2, r5 t1/t2 — the old judge's "Utti"/"HUC"
  garbles are gone), worse only on a degenerate 3-word truncation — where WER is
  quantized in steps of ⅓ and dominated by judge idiosyncrasy on garbled synthetic
  audio — and equal on the 1-word turn.

**Adversarial residual, stated honestly**: this calibration class (semantic-RED
runs) could not exercise the WER conjunct as the *deciding* term — on a future
GREEN-path run (codes spoken, `semantic_match` true), WER≤0.5 becomes load-bearing,
and the two judges demonstrably disagree near threshold on short utterances. The
mitigations are real — GREEN-path answers are full sentences, exactly the class
where the new judge measures better, and short truncations fail semantic anyway —
but the first GREEN verdict under the new judge must report **per-response WER
margins** (distance from 0.5), with any response deciding within ±0.1 of threshold
flagged for review. This is a standing review condition, not a blocker.

### 3. Required comparator artifact contract (package the evidence)

No packaged calibration artifact exists in the tree yet. The adopted-judge decision
must be carried by a checked-in, content-addressed artifact:

- **Identity**: `kind: "asr_judge_calibration"`, `schema: 1`, canonical self-`sha256`.
- **Judges**: old — `.nemo` model file SHA-256 + runtime image ID + archived
  `-asr-final` report paths with their SHA-256s; new — evaluator image ID
  (`sha256:598542…`) plus the full `backend.provenance()` block (backend/manifest/
  snapshot SHAs, decoder incl. `sdpa`, determinism set).
- **Corpus rows (8)**: run, turn, WAV path + bytes + SHA-256, reference text,
  old {transcript, wer, classification, semantic_match, negative_semantic_match,
  passed}, new {same **plus token_ids**}, and the reverse-order rerun record
  (order index, transcript, token_ids) proving cross-container identity.
- **Derived**: field-by-field agreement matrix (all decision fields `identical:
  true`); the WER-boolean flip list with per-file justification (§2's direction
  analysis); aggregate old/new verdicts.
- **Adjudication block**: `a83_gate: satisfied`, plus the §2 standing condition
  (GREEN-path WER-margin reporting).
- **Regression lock**: a host test that loads the artifact, verifies its
  self-hash, and asserts zero semantic flips — so the calibration evidence cannot
  silently rot or be edited.

### Verdict

**APPROVE promotion of the new judge — the A83 §1 gate is satisfied**: zero
semantic-outcome flips on all eight archived responses at every decision level,
with cross-container reverse-order determinism, and the only divergences confined
to the raw-measurement layer where cross-model identity is impossible in principle
and no decision depends on it in the calibrated class. Conditions: (1) package the
calibration as the §3 artifact with its regression-lock test before the judge's
first qualification verdict is cited; (2) the first GREEN-path run under the new
judge reports WER margins per §2. The evaluator image `sha256:598542…` is the judge
of record from here. D8 intact.

## Addendum 90 — judge-swap policy reconciliation + comparator: adversarial re-review (2026-08-09)

Maximally adversarial review of the completed reconciliation. Reviewed state pinned:
comparator `6fa33064…`, tests `dbbfefdf…`, policy doc `ae92ab7d…`, artifacts
v1 `sha256: b200018d…` / v2 `sha256: de5ba75a…`. (The tree was being written seconds
before my first test run — one transient mid-write failure, then 3× stable 7/7;
noted, not a finding.) Review only.

### 1. The laundering question, answered with data

The prior checked-in wording (strict component identity, including the standalone
WER≤0.5 boolean) makes v1 **red** — five booleans flip — and the reconciliation
**preserves that red artifact** with an explicit no-rewrite clause in the policy
doc. The v2 policy (`decision_concordance_v2`) is prospective, and its criteria are
a faithful codification of A89's **pre-registered** adjudication — written before
the packaging — with the GREEN-path condition *strengthened* (a passing response
within 0.1 of threshold is "held for explicit review", stronger than A89's flag).
The WER exception is explicitly retrospective-only ("cannot excuse a current or
future qualification failure") and the live WER≤0.5 gate is unchanged. Decisive
anti-laundering evidence from the artifacts themselves: **v1 and v2 are
byte-identical outside the policy/verdict fields** — same corpus, same judges, same
controls, same derived block (both artifacts record *both* policies' computed
verdicts) — same facts, two adjudications, the failed one retained. This is the
opposite of relabeling.

### 2. The comparator genuinely rederives — verified line-by-line and empirically

- **WER**: recomputed from transcript+reference and required to equal the recorded
  value (raises otherwise). **Semantics**: recomputed via
  `response_semantic_matches` and required to match. **Per-response pass**:
  recomputed as the full conjunction (classification, nonempty, WER, semantic,
  negative) and required to match. **Positive controls**: old cases rederive WER
  and must hash-bind to the old `.nemo` model; new cases rederive and the control
  report's evaluator provenance must equal the candidate's corpus provenance — the
  controls provably ran under the same judge. **WAV binding**: the new judge's
  recorded `source_audio_sha256` must equal the live hash of the retained WAV — 
  the candidate provably consumed the archived bytes. **Contract stability**:
  reference text and expected/excluded sets must be identical across judges.
  **Determinism**: per-row transcript+token-ID identity on the reverse-order
  fresh-container rerun, plus provenance equality across corpus and rerun.
  **Load-bearing analysis** (`_masked_wer_flip`): a WER flip is admissible only
  when `passed` is unchanged AND another conjunct is false in *both* judges'
  records — sound, since a shared-false conjunct means WER alone cannot alter the
  verdict.
- **Empirical re-verification by this review**: both checked-in artifacts pass
  their self-hashes; recomputing the derived block from the raw corpus rows
  reproduces it exactly (8 rows, 0 decision flips, 5 WER flips — r4 t1
  0.5714→0.2857, r4 t2 0.5714→0.2857, r4 t3 0.3333→0.6667, r5 t2 0.625→0.25,
  r5 t3 0.3333→0.6667 — all independently re-judged non-load-bearing by re-running
  `_masked_wer_flip` on the row data); v1 red with exactly
  `["wer_threshold_outcome_flips"]`, v2 green with empty blocking reasons.
- **Provenance verified against the archives**: both old-report paths hash-match
  the live archived files, and the archived contents match the recorded old
  transcripts/WERs. (An apparent 0.375-vs-0.5714 discrepancy against this review's
  earlier notes was my own r3/r4 conflation — the archives are consistent.)
  Old-judge identity carries model SHA + runtime image IDs; new-judge identity is
  the full provenance block of the promoted image.

### 3. Tests and integration

Seven tests, all the red paths that matter: strict policy preserves red on a flip;
v2 accepts a disclosed non-load-bearing flip; **v2 rejects a load-bearing flip**
(the semantic-true fixture where WER becomes the deciding conjunct — the exact
future hazard A89 flagged); self-hash mutation rejection; determinism token
mismatch; and a parametrized lock test pinning both checked-in artifacts (self-hash,
policy, verdict, 8/0/5 counts, determinism, controls) so neither artifact can rot
or be edited silently. `docs/qualification.md` references the policy with the red-v1
disclosure. 49 focused tests + Ruff green; comparator suite 3× stable.

### 4. Residual notes (non-blocking)

Artifact WAV/report paths are absolute retained-evidence paths outside the repo —
convention-consistent, with hashes making rot detectable; the lock test cannot
re-verify WAV bytes (the comparator did at build). The EarTTS-domain positive
controls (old 2, new 4, all "The answer is five." at WER 0, provenance-bound)
retire a meaningful part of A88's domain-gap residual.

### Verdict

**APPROVE the policy amendment and APPROVE the evaluator promotion.** The
reconciliation does the honest thing at every fork: the failed strict artifact is
preserved red and lock-tested against rewriting; the prospective policy codifies
the pre-registered A89 doctrine with a strengthened GREEN-path condition and a
retrospective-only exception; the comparator trusts nothing it can rederive; and
both artifacts carry identical evidence with only the adjudication differing. The
evaluator image `sha256:598542…` stands as judge of record under
`decision_concordance_v2`, with the first GREEN-path qualification bound to the
WER-margin hold-for-review rule. D8 intact.

## Addendum 91 — post-A90 correction delta: rerun order, 4v4 controls, provenance pins (2026-08-09)

Re-review of the single concurrent correction that landed after A90's snapshot.
Reviewed state: comparator `d7f716e3…`, regenerated artifacts v1 `sha256: 198d17ab…`
(red) / v2 `sha256: f2aeea26…` (green). 7 tests + Ruff reproduced green. Review only.

### Delta verified, item by item

1. **The misleading `reverse_order_index` is gone — replaced by an honest record.**
   A90 accepted a per-response `reverse_order_index` that was computed
   arithmetically (`len − index + 1`), implying a per-row reversal that never
   happened — the reversal was at the *corpus* level. The field is removed (no
   leftovers in either artifact, verified), and the actual rerun corpus order is
   now an explicit input recorded per source report (`r5 = 1, r4 = 2`), validated
   as a complete one-based sequence, and regression-pinned by the lock test.
   Determinism semantics are unchanged (per-row transcript + token-ID identity).
   One nonblocking note: the order *mapping* is a validated operator-recorded
   claim, not evidence-derived — if the rerun reports carry internal timestamps, a
   cross-check would close that last inch; the honesty defect A90 missed is fixed
   either way.
2. **Positive controls are now four-vs-four, lane-matched, and byte-bound.** Old
   lanes eager/graph × streaming/offline, each with **live-hashed** source and
   canonical WAV identities recorded; new control cases must label-match the old
   lanes exactly (set equality both directions), and each new case's consumed WAV
   bytes/SHA must equal the corresponding old gate's retained canonical WAV —
   verified true for all four in both artifacts. The new judge provably scored the
   same canonical audio the old judge scored, per lane.
3. **Required-"five"/excluded-"six" independently rederived for both judges** via
   `response_semantic_matches` on every control case (8/8 true in the artifacts) —
   the controls now exercise the positive *and* negative semantic machinery, not
   just WER.
4. **Identity pins hardened.** `IMAGE_ID_PATTERN` is now `\Z`-anchored and enforced
   on both the per-pair old runtime image IDs and the candidate's
   `evaluator_image_id`; the lock test regression-pins the exact candidate
   provenance — image `sha256:598542…`, backend `53b5142…` (the A87-reviewed
   SDPA backend), snapshot `cc561945…` (the manifest aggregate), the full decoder
   block (`sdpa`, `encoder_exhaustion`, greedy, lookahead 13, non-streaming), the
   determinism flags, and the old judge's model SHA.
5. **Artifacts regenerated correctly.** Both self-hashes valid; independent
   re-derivation from raw corpus rows reproduces the derived block exactly —
   8 rows, 0 decision flips, the same 5 non-load-bearing WER flips, determinism
   true, controls pass; v1 red with exactly `["wer_threshold_outcome_flips"]`,
   v2 green with empty blocking; **v1 and v2 remain byte-identical outside the
   policy/verdict fields** — the A90 anti-laundering property survives the
   regeneration.

### Verdict

**APPROVE.** The delta removes the one piece of misleading evidence A90 had
accepted, strengthens the controls from provenance-matched to byte-bound and
semantically two-sided, and pins the promoted judge's full identity into a
regression test. No regressions, no false claims found. The A90 approvals of the
policy amendment and the evaluator promotion stand on the regenerated artifacts
(v1 `198d17ab…` red, v2 `f2aeea26…` green). D8 intact.

## Addendum 92 — post-A91 hardening: substance verified, one broken test fixture blocks (2026-08-09)

Adversarial review of the post-A91 hardening (superseding A91 per the independent
review's broader blockers). Comparator `49ddaaf8…`, artifacts v1 `ef7730bb…` /
v2 `69564a73…`. Review only.

### 1. The hardening substance is real — every claim verified

- **Near-silent records**: accepted as minimal validated shapes (numeric dBFS
  required, null measurement fields, `passed: false`, empty tokens) — honest
  representation; a near-silent↔speech disagreement still surfaces as a
  classification decision flip.
- **Aggregate/runtime/final rederivation**: `_rederive_asr_gate` recomputes every
  ASR counter from the rederived per-response records, requires each stored field
  equal, recomputes the runtime gate via the real `runtime_gate_passed`, and
  requires the stored final verdict to equal runtime∧ASR. The comparator no longer
  trusts any stored pass bit at any level.
- **Old-judge GPU reproduction — the deepest gap closed.** The archived old-judge
  verdicts are now *reproduced evidence*: reruns inside each run's **exact
  historical immutable runtime image** (r4 `sha256:0f775d…`, r5 `sha256:241678…` —
  correctly per-run), `.nemo` mounted read-only, network-none, with per-turn
  transcript/WER/**canonical-WAV-hash** identity enforced (raises on any
  divergence) and image/model/script/report/log all hash-bound. This review
  independently verified the legacy decoder pins against git: the recorded blob
  `8b8af889…` **is** `HEAD:tools/qualification/transcribe_sustained.py` and its
  content SHA-256 matches `674a9ba6…` — the rerun provably used the committed
  pre-swap evaluator code. All 8 rows reproduce in both artifacts.
- **Controls**: exact lane matrix enforced both directions with duplicate
  rejection; each of the four byte-bound lanes additionally runs the **deliberate
  excluded-"five" probe** (scoring "five" as an excluded term must trip the
  detector) — verified firing on all 8 control cases in both artifacts; the
  all-zero silence control is recorded with construction hash and honest
  `asr_invoked: false` / shared-harness ownership.
- **Rerun binding**: the reverse-order rerun is now bound on reference text,
  semantic contracts, source WAV path and `source_audio_sha256`, with rederived
  records and aggregate — plus empty-corpus, duplicate-run, and per-run
  empty-response rejection.
- **Artifacts**: SHAs match the claimed values exactly; self-hashes valid;
  independent re-derivation from raw rows reproduces every derived count
  (8 rows, 0 decision flips, 5 non-load-bearing WER flips, determinism true);
  v1 red with exactly `["wer_threshold_outcome_flips"]`, v2 green;
  **byte-identical outside policy/verdict fields** — the anti-laundering property
  survives again. Docs now correctly narrow the rerun order/freshness claim to an
  operator declaration, closing A91's residual note.

### 2. One blocker: the current test fixture is broken

The claim "12 tests + Ruff green" is **falsified at the reviewed snapshot**
(the test file was modified ~30 s before this review's run — the hot-tree pattern
again, this time leaving breakage): `tests/runtime/test_compare_asr_judges.py`
references `model` at line ~117 (hashing the fake `.nemo` into the
old-provenance execution log) before its definition at line ~129 —
`UnboundLocalError` fails **8 of 12 tests** (every `build()`-based test) and Ruff
reports `F821`. The 4 surviving tests include the artifact lock test, so the
checked-in evidence is still guarded — but the red paths (strict-red preservation,
load-bearing rejection, determinism, mutation, binding, rejection guards) are all
currently unexecuted.

**Exact fix**: in the `reports()` fixture, move the `model = tmp_path / "old.nemo"`
definition (and the write of its placeholder bytes) **above** the
old-provenance-log block that calls `sha256_file(model)`; then re-run
`pytest tests/runtime/test_compare_asr_judges.py` (expect 12/12) and `ruff check`
(expect clean). Nothing in the comparator or artifacts needs to change.

### Verdict

**BLOCK on the §2 test-fixture fix — a one-move change — and otherwise the
hardening is approved as verified.** Every substantive claim held under
independent rederivation: the old judge is now reproducible evidence in its exact
historical environments, no stored verdict at any level is trusted, the controls
can demonstrably fail, and both artifacts carry identical facts with only the
adjudication differing (v1 `ef7730bb…` red, v2 `69564a73…` green). Once the
fixture is repaired and 12/12 + Ruff green is restored, the A90/A91 approvals of
the policy amendment and evaluator promotion carry forward to these artifacts
unchanged. D8 intact.

## Addendum 93 — final judge-swap evidence: fresh adversarial re-review (2026-08-09)

Fresh re-review of the current source and final v5 evidence, rederived rather than
trusted. Reviewed state: comparator `dcd46e40…`, `rerun_legacy_asr.py` `7fb1a2e9…`,
tests `a76ebd9f…`, artifacts v1 `sha256: 53c78175…` (red) / v2 `sha256: f4d6c4d8…`
(green). Review only.

### 1. Everything re-executed and rederived by this review

- **Tests**: 42 focused ASR tests pass (28 across comparator/backend/finalizer/
  recovery + 14 transcribe), Ruff clean — the A92 fixture defect is gone and every
  red path executes again.
- **Artifacts**: self-hashes valid and equal to the claimed values, which are
  hard-pinned in the lock test. Independent re-derivation from raw rows reproduces
  every derived quantity: 8 rows, 0 decision flips, 5 WER flips all
  re-judged non-load-bearing, reverse-order determinism true, v1 red with exactly
  `["wer_threshold_outcome_flips"]`, v2 green with empty blocking — and the
  artifacts remain **byte-identical outside policy/verdict fields**.
- **Silence control is now real execution, verified live**: `_silence_control()`
  runs the shared `classify_audio` (the actual pre-ASR harness function in
  `transcribe_sustained.py`) on constructed all-zero PCM; this review re-executed
  it and the fresh result equals the stored record exactly (−120 dBFS,
  `near_silent`, construction hash matching). The A92-era static declaration is
  replaced by executed evidence.
- **Aggregate parity is now four-deep per run**: external-ASR pass, runtime gate,
  final report verdict, and the gate configuration itself (`max_wer`,
  `qualified_fp32_base_rate`) — all required identical between judges, all
  verified true in both artifacts.

### 2. The legacy rerun tool is a complete evidence chain

`rerun_legacy_asr.py` (schema 2): docker-inspects the requested ID and requires
exact immutable resolution (repo digests recorded); extracts the legacy decoder
from git and verifies **both** the locked blob (`8b8af889…` =
`HEAD:tools/qualification/transcribe_sustained.py`, re-confirmed against git by
this review) and its content SHA before retaining the bytes; **hashes the model
and decoder inside the container** in a separate network-none run, closing the
host/container TOCTOU; writes a schema-2 header (exact argv + canonical argv hash,
source report retained + hashed) as the first line of the execution log; refuses
overwrite; runs `--user uid:gid`; passes the exit code through. The comparator
then **reconstructs the entire expected header independently** — argv included —
and requires the log's header to match field-for-field, with container hashes
equal to host hashes. The fresh v5 r4/r5 GPU reruns reproduce all 8 rows
(transcript, WER, canonical-WAV hash) under each run's own historical image.

### 3. The one asymmetry is recorded, isolated, and legitimate

The legacy harness recomputes the **runtime gate** with old logic over new-era
reports, so its runtime-gate/final-verdict bits can differ from the archives.
The comparator records both sides under `harness_verdicts` with
`comparison_scope: "recorded_not_required_old_asr_judge_isolation"` — and this
isolation is correct, not laundering: the runtime gate is harness logic, not an
ASR judge output, while everything that **is** judge output — transcripts, WER,
canonical WAVs, and the ASR aggregate (`aggregate_fields_identical`) — is required
identical and verified so. The deliberate excluded-"five" probes fire on all
eight control lanes in both artifacts.

### 4. Nonblocking observation

The shared silence classifier's own source identity is carried implicitly (the
current script for the new judge; the git-pinned legacy blob for the old) rather
than as a separately recorded SHA — the live-execution equality verified above is
the substance; a one-line classifier-source hash in the silence-control record
would make it self-describing. Cosmetic.

### Verdict

**APPROVE.** The A92 blocker is fixed with all red paths executing; every claim in
the final evidence rederives under independent execution — including a live rerun
of the silence classifier and a git-level re-verification of the legacy decoder
pins; the old judge's verdicts are reproduced evidence in their exact historical
environments with a fully reconstructed execution contract; the single old/new
asymmetry is disclosed, scoped, and genuinely outside the judge boundary; and the
anti-laundering invariant (identical facts, differing adjudication, red v1
preserved) holds through the final regeneration. The A90/A91/A92 approvals
consolidate onto v1 `53c78175…` (red, preserved) and v2 `f4d6c4d8…` (green) — the
judge swap is closed. The evaluator image `sha256:598542…` is the judge of record
under `decision_concordance_v2`, with the GREEN-path WER-margin hold-for-review
rule still binding. D8 intact.

## Addendum 94 — speech twin + modality-parity comparator: pre-implementation design review (2026-08-09)

Design review for the A82 next step. Current code inspected: the sustained probe,
`transcribe_sustained`, the multiturn voice-turn helpers, Pocket-TTS fixture
generation, and the live suite. Review only; no implementation.

### 1. Every building block already exists and is proven

- **Speech-turn mechanics** (multiturn): `input_audio_buffer.turn_start`
  (client_turn_id) → paced 80 ms PCM16 `append` frames → explicit
  `input_audio_buffer.commit` — the client Smart-Turn v1 pathway — plus
  `send_silence_until_stopped` continuation frames while the answer plays.
- **Heard-the-prompt control** (multiturn): `displayed_input_text` from the
  runtime's input transcription vs `expected_input_text` → `input_wer ≤ 0.35`.
  A82 constraint 3 is already implemented code, not new design.
- **Fixture generation**: `generate_multiturn_fixtures.py` synthesizes WAVs from
  text via the Pocket worker with generator provenance in a manifest.
- **Checkers**: `evaluate_tool_response_channels` and
  `evaluate_tool_cycle_cardinality` are pure functions over
  responses/metrics/observations — already modality-agnostic.

### 2. Smallest clean architecture (recommended)

1. **Extract the multiturn speak-turn helpers** (`speak_turn`,
   `send_silence_until_stopped`) into a shared qualification module imported by
   both multiturn and the sustained probe; multiturn tests must stay green
   unmodified.
2. **`sustained_strict_v3.py` gains `--input-modality {typed,speech}`** (default
   `typed`; nothing changes for typed) **and `--fixture-manifest`**. In speech
   mode the serialized sender is the *only* audio-stream owner (no parallel
   silence loop — single-owner pacing exactly as multiturn does): per prompt,
   speak the fixture WAV, commit, silence-until-answered, settle, next. Same
   `TOOL_ONLY_TYPED_PROMPTS` constants, same code rotation and `tool_result`
   fixture, same session config/instructions, same receiver, same checkers —
   the input pathway is the only difference, which is the A82 treatment variable,
   stated as such in the report.
3. **Job ledger reuse**: spoken prompts get synthetic job ids in the same `typed`
   ledger with dispositions derived from the turn lifecycle
   (turn_start→commit→transcription→response), so the report schema and the
   sustained recompute apply unchanged. Turn→response attribution uses the
   echoed `client_turn_id`/turn ordering; the receiver's typed-source
   transcription branch gains the microphone-source equivalent for input-text
   capture.
4. **Fixture generator extension**: a tool-only fixture set generated from the
   *imported* `TOOL_ONLY_TYPED_PROMPTS` (single source of truth), manifest rows
   `{prompt_index, text, wav, wav_sha256, duration_seconds}` plus Pocket
   generator provenance, manifest self-hashed.
5. **Comparator**: new `tools/qualification/compare_input_modalities.py`,
   following the `compare_asr_judges` conventions (content-addressed inputs,
   rederive-don't-trust, self-hashed artifact, lock tests).

### 3. Exact report fields and gates

- Both arms: `input_modality` ("typed"/"speech") — informational, recorded in the
  report and echoed by the ASR stage.
- Speech arm additionally: `fixture_manifest` {path, sha256, generator
  provenance, per-prompt rows}; per job `expected_input_text`,
  `displayed_input_text`, `input_wer`; and `heard_gate`
  {max_input_wer: 0.35, per-turn WERs, passed}.
- **Gates**: all structural/runtime conjuncts stay absolute and identical in both
  arms. **One intentional gate addition, pre-registered here**: `heard_gate` is a
  required conjunct of the speech arm's `passed` (an unheard prompt makes the
  probe invalid — fail closed, never silently excluded). The
  `sustained_runtime_gate_passed` recompute adds the same conjunct **only when
  `input_modality == "speech"`** — typed and legacy reports are byte-unaffected
  (field absent ⇒ existing semantics). This is the only recompute change and must
  land writer+recompute in lockstep (A74 discipline).
- Diagnostics (post-FC token trace, agent top-k) may be enabled identically in
  both arms — the opener top-k across modalities is free distribution-level
  parity evidence. No inference changes anywhere (A82 constraint 7).

### 4. Comparator contract (exact)

Inputs: the typed and speech ASR-final reports. **Refusals (fail-closed)**:
report kinds not both `sustained_strict_v3`; modalities not exactly
{typed, speech}; prompt sequences, per-turn `expected_response_any`, or response
cardinality differ; ASR judge provenance not identical (evaluator image ID,
backend/snapshot SHAs, decoder block); speech `heard_gate` failed; either arm's
runtime gate fails **recompute** (never trusted). Output artifact
(`kind: "input_modality_parity"`, schema 1, self-hashed): per-turn paired rows
{turn, code, per-arm {text_semantic_match, ASR semantic_match, transcript, WER,
wer_margin = 0.5 − WER}, concordant, text_only_failure, speech_only_failure};
derived {discordance counts, both-arm structural summary}; verdict
`parity_passed` = **zero `text_only_failure` rows** (text ≥ speech; speech-only
failures are disclosed, never blocking); any *passing* row with wer_margin < 0.1
sets `held_for_review: true` per the A89/A90 rule and forbids citing the pair as
an unattended green.

### 5. Exact tests

Fixture/manifest: generator uses the imported prompt constants (test asserts
manifest texts == constant, hashes valid). Probe: speech report carries the new
fields; typed report schema regression-locked (byte-identical fields to today);
recompute red path for `heard_gate` false; existing typed recompute tests
untouched and green. Helpers: extraction leaves multiturn tests green. Comparator:
green parity; text-only-failure ⇒ red; speech-only-failure ⇒ green with
disclosure; each refusal path (judge mismatch, fixture/prompt mismatch, unheard
arm, kind/cardinality mismatch) ⇒ raise; margin hold-for-review; self-hash
mutation ⇒ reject. Live suite: not wired initially — the twin pair plus
comparator artifact is standalone promotion evidence; suite integration is a
separate later step.

### 6. Constraint check (A82 §1–7)

(1) fixture identity ✓ shared constants + same tool fixture/checkers; (2) input
pathway as declared treatment ✓; (3) heard-gate ✓ (reusing multiturn's proven
mechanism, promoted to a speech-arm conjunct); (4) provenance ✓ informational
modality + hashed fixture manifest; (5) absolute structural gates, parity-relative
semantics, "no text-only failures" on N=4 ✓; (6) identical judge + identical
optional diagnostics ✓; (7) no inference changes; comparator + fields are
runtime-repo tooling only ✓.

### Verdict

**APPROVE the design.** No blockers: every mechanism is an existing, proven
component recombined, the single-owner audio pacing avoids the one real hazard
(double-writing the stream during playback), the one recompute change is scoped
to the new modality and pre-registered, and the comparator inherits the
rederive-don't-trust conventions the judge-swap work just hardened. Implementation
order: helper extraction → fixture generator → probe modality → recompute lockstep
→ comparator → tests. D8 intact.

## Addendum 95 — adjudication: focused `tool_freshness_parity.py` vs extending sustained (2026-08-09)

Adjudication of the design disagreement against A94. Review only.

### 1. The independent review is right — and it caught a real flaw in A94

**A94's architecture had a latent defect this adjudication must own.** The
sustained probe's combined gate folds `text_semantic_match` into
`tool_response_channel_gate`, which feeds both the report's `passed` and the
`sustained_runtime_gate_passed` recompute — by A73's deliberate design, so that
the focused probe goes RED on code omission. Under the A82 parity framing that
folding is wrong for the parity instrument: both arms would fail the recompute
(as r4–r6 do), and A94's own comparator refusal rule ("either arm's runtime gate
fails recompute ⇒ refuse") would therefore refuse **every** honest parity pair.
Making A94 work would have required carving the semantic conjunct out of the
shared sustained recompute — surgery on a qualification gate for the benefit of a
different instrument, exactly the class of edit this review series has spent five
addenda preventing. A fresh focused report kind with **separate structural gates
that exclude only the code semantic readback** is the honest structure: the
semantic outcome becomes a *measured variable* of the parity instrument rather
than a gate, without touching the sustained tier's strictness at all.

**The fixture confound catch is also correct and better than A94.** A94 kept the
wall-clock `now` per arm — meaning two sequential arms receive *different* tool
outputs, so transcripts and time-expectations legitimately diverge and byte-level
tool-result identity (the A82 constraint-1 ideal) is violated by construction.
Predeclared deterministic four now/code outputs give both arms byte-identical
tool results and make the whole scenario content-addressable.

**The scope argument stands too**: the sustained sender carries 15-minute pacing,
session-limit, queue, and unanswered-rate machinery that a 4-turn parity probe
does not want; one generic `run_arm(input_driver)` over two fresh sessions is
smaller than threading modality through all of it, and fresh-session isolation is
cleaner than inheriting the focused carve-out flags. Sequential-arm variance is
acceptable: cross-run determinism of the runtime under greedy decode is already
demonstrated (byte-identical texts across r2/r3 and r4/r5), and the deterministic
tool outputs remove the one systematic cross-arm difference.

### 2. APPROVE the refinement — with these exact requirements carried over

1. **Reuse the pure checkers.** `evaluate_tool_response_channels` and
   `evaluate_tool_cycle_cardinality` are imported unchanged so per-turn structural
   fields stay name-compatible with the r-series evidence; the new report kind
   composes them minus the semantic conjunct rather than reimplementing.
2. **Enumerate the carve-out exactly.** The focused structural gate = the full
   battery (EOTR validity incl. wait_steps==1 and feedback commit, function-guard
   idleness, post-EOTR audibility, bounded watchdog close, text/audio nonempty,
   five-way cardinality, typed lifecycle checks, speech ack/client-ID
   correlation) **minus only** `text_semantic_match` and the ASR semantic match.
   Nothing else may move to measured-variable status.
3. **The heard gate stays a required speech-arm conjunct** (input-WER ≤ 0.35
   against the fixture text) — it was absent from the refinement summary and is
   non-negotiable per A82 §3: an unheard prompt invalidates the arm, fail-closed.
4. **Scenario hash binds everything both arms must share**: prompts, code
   rotation, the predeclared now/code outputs, tool schema/description, session
   instructions, and (for the speech arm) the hash-complete Pocket manifest.
   The comparator refuses on scenario-hash inequality — stronger than A94's
   field-by-field comparison, and adopted.
5. **Recompute dispatch stays fail-closed**: the new report kind registers in the
   `report_kind` dispatcher with its own rederivation; unknown kinds continue to
   return False; sustained/multiturn/memory-matrix recomputes are byte-untouched.
6. **The A94 comparator contract carries over intact** with the kind updated:
   content-addressed inputs, rederive-don't-trust, refusals (kind/modality/
   scenario-hash/judge-provenance mismatch, unheard speech arm, structural-gate
   recompute failure — now meaningful since structure excludes semantics),
   self-hashed artifact, per-turn paired rows with WER margins, and the parity
   semantics restated: shared failures allowed, **typed-only semantic failure
   blocks**, speech-only failure disclosed; passing rows within 0.1 WER margin
   held for review per A89/A90.
7. **Judge identity pinned** to the promoted evaluator (`sha256:598542…`,
   `decision_concordance_v2`); diagnostics optionally enabled identically in both
   arms; arm execution order recorded as an operator declaration per the A93
   honesty discipline. No inference changes.
8. **Tests**: A94's list transposed to the new kind, plus: predeclared-output
   byte-identity across arms (the confound regression), scenario-hash refusal,
   carve-out enumeration lock (a test asserting the structural gate's exact
   conjunct set so semantics cannot silently rejoin it), and unknown-kind
   fail-closed regression.

### Verdict

**APPROVE the independent review's refinement.** The focused
`tool_freshness_parity.py` with `run_arm(input_driver)`, two fresh 4-turn
sessions, predeclared deterministic tool outputs, and semantics-excluded
structural gates is the correct instrument; A94's extend-sustained recommendation
is withdrawn on the folded-gate flaw, which the independent review correctly
identified — the second time in this series (after A87) that a reviewer
disagreement improved the design, which is the system working. The shared
`qualification_audio` helper extraction already landed per A94 §2.1. Requirements
1–8 above are binding for the implementation review. D8 intact.

## Addendum 96 — substep 1: shared audio helper extraction review (2026-08-09)

Adversarial review of `src/nemotron_voicechat_runtime/qualification_audio.py` and
the modified `multiturn_strict_v3.py`. 15 focused tests + Ruff reproduced green.
Review only.

### Verified

- **Baseline subtlety handled**: the whole branch is uncommitted, so diff-vs-HEAD
  conflates all branch work — the correct pre-extraction baseline is the working
  tree state this review documented in A94 (turn_start → paced append → commit,
  each carrying `client_turn_id`). Against that baseline the extraction is a pure
  move: identical event names and `event_id` patterns
  (`-start`/`-{index}`/`-commit`/`-continuation-{index}`), identical non-drifting
  absolute pacing (`deadline = started + index·0.08 s`), identical frame padding
  and trailing-silence handling, identical `read_fixture`
  resample/clip/rint round-trip, and identical silence-continuation loop
  including its `index+1` deadline and stop-event semantics. The only structural
  change is that `client_turn_id` is now an explicit parameter (caller passes the
  1-based turn index) — same wire bytes.
- **Import works host and container**: the module depends only on stdlib + numpy
  (verified no heavyweight modules load on import), lives under
  `src/nemotron_voicechat_runtime/`, which is on `PYTHONPATH` on the host, in the
  runtime image, and in the evaluator image alike.
- **No hidden regression**: multiturn's `__all__ = ("FRAME_SAMPLES",)` re-export
  exists precisely because `test_multiturn_qualification.py` reads
  `module.FRAME_SAMPLES` — the test imports unchanged and the re-exported value
  equals the helper's (verified 1280). Ruff confirms no orphaned imports;
  `sustained_strict_v3` is correctly untouched this substep; the helper is
  consumed by multiturn only, as scoped.

### Nonblocking note

The helper sends `client_turn_id` but the ack-correlation (A95 requirement:
speech ack/client-ID correlation) is receiver-side work that stays with the
parity-probe substep — flagged so it isn't assumed done.

### Verdict

**APPROVE substep 1.** A clean pure-move extraction with exact wire semantics,
dual-environment importability verified by execution, and the one dependent test
kept green via an explicit re-export. Proceed to the fixture-generator and
parity-probe substeps under the A95 requirements. D8 intact.

## Addendum 97 — substep 2: tool-freshness fixture generator/validator review (2026-08-09)

Adversarial review of `tool_freshness_fixture.py`,
`generate_tool_freshness_fixtures.py`, the sustained constant rewiring, and
`test_tool_freshness_fixture.py`; plus the substep-1 addendum (direct
`send_audio` behavioral test). 25 focused tests + Ruff reproduced green. Review
only.

### Verified against the A95 requirements

- **Single source of truth, direction reversed for the better**:
  `sustained_strict_v3.py` now **imports** `TOOL_ONLY_PROMPTS` (aliased) and
  `TOOL_VERIFICATION_CODES` from the src fixture module — no duplication drift is
  possible; the generator's runtime divergence guard and the identity test are
  belt-and-braces on top.
- **The A95 confound fix is in**: `TOOL_RESULT_TIMES` are four distinct
  predeclared times; the scenario document embeds, per turn, the full
  `tool_result("tool-only", …)` payloads — so both arms will receive
  byte-identical tool outputs, and `scenario_sha256` (canonical JSON over
  instructions + prompts + tool definition + per-turn now/code/expected/output)
  binds exactly the A95 §4 surface.
- **Manifest schema and content addressing**: exact-key schemas at every level,
  self-hash over canonical JSON, per-fixture PCM **and** WAV records with
  byte/hash identity, the WAV's decoded frames required byte-equal to the raw
  PCM, format pinned (16 kHz mono PCM16, 0 < samples ≤ 30 s), prompt/case-id/
  ordinal identity per row.
- **Pocket provenance**: synthesis contract pinned to the fixture constants
  (language/voice/seed/seed-scheme) with worker-health-derived values — a
  mismatched worker fails validation; assets must equal the pocket worker's
  `EXPECTED_ASSETS` byte+SHA inventory; package version required nonempty;
  threads/CPUs recorded.
- **Immutable runtime/source hashes**: `\Z`-anchored image-ID, six project
  sources plus the installed worker hash-validated against live files by the
  companion validator; the generator runs **both** validators post-generation
  (fail-closed generation validation), refuses non-empty output directories, and
  defaults offline env vars.
- **Mutation tests are field-level, not envelope-level**: each parametrized
  mutation is re-signed and must be caught by its specific bound-field check
  (prompt text, PCM hash, WAV hash, synthesis seed, runtime image ID), with the
  self-hash mutation tested separately — proving the bindings don't hide behind
  the hash envelope.
- **Substep-1 addendum**: the direct `send_audio` behavioral test is present and
  parametrized over 0/2 trailing frames, asserting the exact wire sequence
  (turn_start → N appends including trailing → commit).

### Forward requirements for substep 3 (the parity probe)

1. The probe must **recompute** `scenario_sha256` from the same imports
   (`scenario_document()` logic) and refuse a fixture whose hash differs — never
   copy the hash from the manifest.
2. The generator's `SESSION_INSTRUCTIONS` duplicates sustained's inline
   instruction string. The probe's `session.update` instructions must be derived
   from the scenario document (or a shared constant), not re-typed — an
   instruction drift between scenario and live session would be a silent
   confound of exactly the class A95 §4 exists to prevent.
3. Cosmetic: `synthesis.quantize` is a literal `True` rather than health-derived;
   the asset hashes pin the actual model bytes so this cannot false-green, but
   deriving it from worker health would be self-describing.

### Verdict

**APPROVE substep 2.** The fixture layer is content-addressed end to end,
validated at generation, mutation-tested at field level, and the scenario hash
binds precisely the surface A95 requires — with the constants now flowing from
one importable source. Forward requirements 1–2 bind the substep-3 review. D8
intact.

## Addendum 98 — substep 3: parity probe + parity ASR kind review (2026-08-09)

Adversarial review of `tool_freshness_parity.py`, the `transcribe_sustained`
parity kind, and their tests. 44 focused tests reproduced green. Review only.

### 1. The substance is approved — verified point by point

- **A97 forward requirements met exactly**: the probe recomputes the scenario via
  the imported `scenario_document()` and validates the fixture against the
  *recomputed* hash (never copied); the live `session.update` takes
  `scenario["instructions"]` and `scenario["tool"]` — instructions and tool
  definition flow from the scenario document, with an explicit drift guard
  against the generator constant.
- **Identity binding is stronger than required**: beyond fixture/source/runtime
  hashes, the probe requires the **live server's typed-input Pocket worker
  synthesis identity** (language/voice/threads/CPUs/seed/scheme/package/assets)
  to equal the fixture manifest's — both arms' audio provably originates from
  identically configured synthesis. Runtime provenance validated from
  `session.created`; health readiness required.
- **The A95 §2 carve-out is enforced in code**: 17 structural conjuncts,
  drift-locked (`derive_structural_conjuncts` raises if the inventory changes,
  and a test pins the exact set) — full battery including tool-call correlation,
  applied-output acks, response bracket correlation (`text/audio/response.done`
  each correlated by response_id+turn_id with `done_status == "completed"`),
  EOTR validity, guard idleness, audibility, cardinality (five-way), protocol
  errors, clean close, and the speech heard gate — with **no positive-semantic
  conjunct**; the sustained checker's semantics-coupled gate is explicitly
  discarded while its per-check fields are consumed individually.
- **Event contracts and correlations**: typed lifecycle
  (accepted/started/completed, requested==expected text, typed-source
  speech_started, transcription job/turn correlation); speech ack correlation
  (`turn_started`/`committed` must echo both `client_turn_id == ordinal` and the
  exact `client_event_id`), consistent turn IDs across all events, microphone
  transcription source, nonempty displayed text, per-turn input-WER ≤ 0.35. All
  checks are `is True` forms — missing or mis-ordered events fail closed, and my
  race analysis found no path where a lost event produces a false pass (the
  break condition requires `response.done` last; drain-phase metrics are not
  appended but demonstrably not needed, as every check window closes before
  `response.done`).
- **Parity ASR kind**: `response_asr_evaluation` splits scoring exactly per the
  requirement — `audio_integrity_passed` (nonempty, WER ≤ max, **negative**
  semantics) is absolute in both modes; positive `semantic_match` is always
  measured and recorded but nonblocking only for the parity kind (explicit
  `evaluation_policy` marker); legacy kinds keep the byte-identical conjunction
  and schema. The parity recompute is deep: it rebuilds the scenario from
  imports, re-derives runtime/fixture identity from stored evidence, re-runs
  `derive_structural_conjuncts` over the raw records, and requires stored ==
  derived == passed. Unknown-kind dispatch remains fail-closed.
- **Evidence discipline**: atomic report writes, fail-closed error report with
  nonzero exit, fresh output directory required, arm order recorded
  (typed → speech).

### 2. One executable blocker, empirically confirmed

**The parity recompute chain cannot import inside the evaluator image.**
`tool_freshness_runtime_gate_passed` defers its imports, but the chain —
`tool_freshness_parity` (module-level `import websockets`) and
`generate_tool_freshness_fixtures` → `sustained_strict_v3` (module-level
`import websockets`) — requires `websockets`, which is **not** in the evaluator
image's hash-locked requirement set, and the import executes *before* the
function's exception guard. This review simulated the evaluator environment with
an import block: the recompute **crashes with ModuleNotFoundError** on the first
parity-arm report. Same failure class as the A85 `-P` break: host tests pass
(websockets present in the venv), the container dies on first use.

**Exact fix (either):**
(a) smallest — add hash-pinned pure-Python `websockets` to
`requirements-asr-evaluator.txt`, plus an audit canary that imports the parity
chain under the image's `PYTHONPATH` with the repo mounted; or
(b) cleaner — relocate the pure functions the recompute needs
(`derive_structural_conjuncts`, the two synthesis-identity helpers, and
`scenario_document` with `tool_definition`/`tool_result`) into stdlib-only `src`
modules so the evaluator chain never touches `websockets`.
**Plus, required either way**: a host regression test that imports the recompute
chain under a simulated missing-`websockets` environment (the exact harness used
by this review) — the same guard pattern that now protects the `-P` consumers.

### 3. Trivial second item

Ruff is not clean as claimed: one `I001` (unsorted deferred-import block) at
`transcribe_sustained.py:358` — a single `--fix`.

### Notes (non-blocking)

The typed arm starts its frame clock only at `injection_finished`; if injection
ever required client frames to progress, the arm would time out RED — fail-closed,
not false-green, but worth knowing on first live run. Response↔input pairing is
by serialized order plus turn-ID equality, which is sound under the one-at-a-time
design.

### Verdict

**BLOCK on §2 (evaluator-image import chain, with its regression test) and the
§3 one-liner.** Everything else — scenario reconstruction, identity binding,
drift-locked semantics-excluded structure, event/ack correlations, the split ASR
policy with legacy behavior untouched, and the deep recompute — is approved as
implemented and verified by execution. Once the import chain is fixed and the
canary lands, substep 3 is clear for the live parity run. D8 intact.

## Addendum 99 — corrected substep 3: contract module + hardened bindings (2026-08-09)

Re-review of the corrected implementation. 69 focused tests + Ruff reproduced
green. Review only.

### 1. The A98 blocker is fixed the clean way — and verified by re-simulation

The recompute chain now flows through the new
`src/nemotron_voicechat_runtime/tool_freshness_contract.py` — stdlib + fixture
module only, **zero** `websockets` references — owning the canonical scenario/
tool construction, the shared response/cardinality checkers, synthesis/experiment
identities, and structural derivation. Generator, sustained, runner, and
transcriber all import the contract; grep confirms the duplicate implementations
are gone from all three tools. Re-running A98's import-block simulation: the
recompute chain now **imports and executes with `websockets` absent**. The audit
script gained exactly the right canary — `python3 -P` under the image path,
asserting the empty-report recompute returns False **and**
`"websockets" not in sys.modules` — proving the chain stays clean rather than
merely not crashing, and the promoted judge image needed no rebuild (the fix is
repo-side code the image mounts). This resolution is A98's option (b), the
architecturally clean one.

### 2. The post-snapshot hardening is real — each item verified

- **Raw rederivation**: the parity recompute no longer trusts stored checks or
  cardinality — it re-runs `evaluate_tool_response_channels` and
  `evaluate_tool_cycle_cardinality` over the raw responses/metrics/observations
  and requires stored == derived for both, then re-derives the conjuncts on top.
- **Unmatched-delta false-correlation closed**: every `output_text.delta` and
  `output_audio.delta` now records a response_id+turn_id correlation bool; a
  stray or mis-attributed delta turns the gate red instead of silently appending
  content.
- **Binding depth**: the report embeds the full fixture manifest (self-hash
  re-verified in the recompute), an `experiment_sha256` over scenario+manifest,
  `model_input_path` bound to the modality, and a three-way runtime-provenance
  equality (session-created provenance == health-snapshot checkpoint provenance
  == recorded provenance) plus live-health/typed-worker identity.
- **Isolation**: `wait_for_idle()` health polling between arms with a timeout
  raise, and an explicit distinct-fresh-session check that raises if the arms
  shared a session.
- **Tests**: the mocked `run_arm` wire-contract test exercises the full
  correlated event stream end-to-end per modality; the conjunct-inventory lock,
  per-family red paths, heard-gate/ack correlation, and fixture-vs-live-worker
  identity tests all present; the host missing-`websockets` regression drives a
  complete synthetic report through the recompute under the import block.

### 3. One-token nonblocking fix

`derive_structural_conjuncts` is called inside the recompute's try-block, but the
except tuple is `(KeyError, RuntimeError, TypeError, ValueError)` — an
`AttributeError` (observed in simulation for a modality-valid report with
`scenario: None`) propagates and crashes the evaluator instead of returning
False. Fail-closed either way (crash → nonzero exit, no false-green — the audit
canary's `{}` probe exits early on modality and does not cover this shape), but
adding `AttributeError` to the tuple restores the intended malformed→False
contract. One token.

### Verdict

**APPROVE corrected substep 3.** The blocker is resolved via the clean contract
extraction and proven by re-simulation plus an in-image canary that asserts
import purity; every post-snapshot hardening claim verified against source; the
recompute now rederives everything derivable from raw records; 69 focused tests
and Ruff green reproduced. Land the §3 one-token widening at leisure. The live
parity run (typed arm, idle gap, speech arm) is clear to execute, with its
ASR-final evaluation and then the paired comparator as the remaining substeps.
D8 intact.

## Addendum 100 — final substep-3 delta: offline manifest revalidation + malformed handling (2026-08-09)

Narrow re-review of the post-A99 deltas. 69 focused tests + Ruff reproduced
green. Review only.

### Verified, each by execution where executable

- **A99 §3 closed, and the canary now covers the exposed shape.**
  `AttributeError` is in the recompute's except tuple, and the audit canary
  probes `{"input_modality": "typed"}` — exactly the modality-valid malformed
  shape A99's simulation crashed on. Re-running this review's
  websockets-blocked simulation: both that shape **and** the scenario-`None`
  shape now return `False` cleanly, with import purity preserved
  (`websockets` never enters `sys.modules`).
- **The self-found gap is real and correctly closed.** A99's recompute verified
  the embedded manifest's self-hash but did not re-validate its *schema* offline
  — a structurally invalid manifest with a consistent self-hash would have
  passed the metadata layer and relied on downstream identity checks. The new
  pure `validate_tool_freshness_manifest_document()` enforces the exact
  top/row/PCM/WAV/synthesis/provenance schemas, constants and bounds, SHA-256
  syntax, self-hash, positional `case_id`/`ordinal` ordering, image-ID shape,
  and scenario-hash equality — **without touching the filesystem** — and is now
  the first gate in *both* the filesystem fixture validator (which then adds the
  carrier-file byte checks) and `tool_freshness_runtime_gate_passed` (arm
  `schema == 1` additionally enforced). One subtlety checked: the recompute
  passes the report's own claims (provenance image ID, scenario hash) into the
  validator, but a forged-consistent pair still fails the later
  `scenario_identity` recomputation against the imported `scenario_document()` —
  no self-referential false-green exists.
- **Tests match the claims**: the synthetic green report carries full manifest
  metadata, and the re-signed mutation set (missing PCM record, wrong
  kind/schema, bad assets/hash/case identity) exercises the new validator's
  field-level bindings rather than the hash envelope.

### Verdict

**APPROVE.** The final deltas close A99's one-token item, extend the audit
canary to the previously uncovered shape, and add genuine offline
schema-revalidation of retained metadata with no self-referential trust path —
all verified by re-simulation and test execution. This completes the substep-3
review chain (A94–A100): the parity instrument is fully reviewed and the live
run (typed arm → idle gap → speech arm), its ASR-final evaluation under the
promoted judge, and the paired comparator are the remaining execution steps
toward the A82 promotion decision. D8 intact.

## Addendum 101 — modality comparator review (2026-08-09)

Adversarial review of `compare_tool_freshness_modalities.py` and its tests
against the A82/A94/A95 contract. 61 combined focused tests + Ruff reproduced
green. Review only.

### Verified — the comparator trusts nothing it can rederive

- **Arm gates**: each arm must carry `runtime_gate_passed`/`passed` true **and**
  survive a fresh `tool_freshness_runtime_gate_passed(report)` — the A99/A100
  deep recompute (scenario reconstruction, manifest schema revalidation, raw
  check/cardinality rederivation, conjunct equality) runs again inside the
  comparator. `model_input_path` is bound per modality.
- **Promoted judge, pinned from the repo**: the expected ASR contract is rebuilt
  live from the checked-in manifest and backend (backend/manifest SHAs computed
  at run time, SDPA/encoder-exhaustion decoder block, full determinism set),
  the image ID is `\Z`-anchored, and **both arms' evaluator provenance must be
  identical** — one judge, the promoted one, for both arms.
- **Audio chain**: source (22.05 kHz) and canonical (16 kHz) WAVs are
  format-parsed from disk, hash-rederived, and bound to the recorded
  `source_audio_sha256`/`canonical_wav_sha256`; speech classification is
  re-derived by re-running `classify_audio` on the re-read audio with dBFS
  equality; token IDs required non-empty integers.
- **ASR verdicts and aggregate**: per-response `response_asr_evaluation`
  re-derived and required equal to stored, with `audio_integrity_passed`
  (nonempty, WER ≤ 0.5, negative semantics) absolute; the aggregate is compared
  against a **hard-pinned expectation** (4/4/4 speech, zero near-silent,
  qualified rate 0.0, max-WER 0.5, policy marker) rather than anything copied
  from the report.
- **Identity and pairing**: experiment/scenario/fixture-manifest/runtime/
  evaluator identity equality across arms; distinct session IDs required;
  per-case ordinal/case/turn correlation within each arm and case-identity plus
  expected-code equality across arms, `strict=True` pairing.
- **Parity semantics, both layers**: model-text semantic (from the arm's
  measured check) and independently transcribed audio semantic are compared per
  case — **typed-only failure blocks** on either layer; shared and speech-only
  failures are disclosed as structured fields and never block. The speech
  input-WER remains absolute through the arm's heard conjunct and is carried
  per case for disclosure.
- **Evidence discipline**: self-hashed artifact with an explicit
  machine-readable policy block, atomic write, fail-closed error artifact, exit
  code from `passed`. The comparator itself imports cleanly under the simulated
  evaluator image (no `websockets`) — container-safe via the A99 contract chain.

### One intentional strictness upgrade, recorded

The A89/A94 WER-margin rule ("held for explicit review; cannot be cited as an
unattended green") is implemented as a **hard blocking reason** when any paired
case's minimum response-ASR margin falls below 0.1 — stricter than the
pre-registered hold semantics. The distinct reason string keeps margin-holds
separable from typed-only failures for a reviewing operator, and under the
no-weakening doctrine a strictness upgrade is acceptable; recorded so nobody
mistakes it for the original hold rule.

### Tests

All contract red/green paths are behaviorally covered: shared failures
nonblocking, typed-only blocking (parametrized over both semantic layers),
speech-only disclosed, provenance/aggregate mutation rejection, identity and
audio-hash mutation rejection, thin-margin hold, and independent semantic
rederivation. 61 combined tests + Ruff green, reproduced.

### Verdict

**APPROVE the comparator.** With this, every component of the A82→A95 parity
instrument is review-complete: fixtures (A97), probe and parity ASR kind
(A98–A100), and comparator (A101) — all content-addressed, rederive-don't-trust,
and fail-closed, scored by the promoted judge under `decision_concordance_v2`.
What remains is execution: generate fixtures, run the paired arms, evaluate both
under the judge, and run this comparator — its artifact is the A82 promotion
evidence. Per the r6-retry top-k prediction (A82), the expected outcome is
shared semantic failures in both arms — which under the parity criterion is a
**promotable** result with the code-readback limitation documented at the model
level. D8 intact.

## Addendum 102 — comparator hardening + live orchestration review (2026-08-09)

Adversarial re-review of the hardened comparator and the new
`--tool-freshness-parity-only` orchestration. 114 focused tests + Ruff
reproduced green. Review only; the live GPU experiment was not run.

### 1. Comparator hardening — each independent-review blocker verified closed

- **Runtime-report binding is the strongest link added**: the retained top-level
  probe report must pass with `execution_order == ["typed", "speech"]`,
  session IDs in that exact order, experiment/scenario/manifest/provenance
  equality with the arms, canonical absolute raw-arm paths — and two decisive
  checks: the raw arm's live file SHA must equal the transcriber-recorded
  `external_asr_source_report.sha256` (the evaluated report provably derives
  from its declared raw arm), and the evaluated report **minus only the
  ASR-added fields must deep-equal the raw report** — proving the ASR stage was
  add-only. Any future field the transcriber adds breaks equality fail-closed.
- **Canonical audio now byte-derives**: the comparator recomputes the
  16 kHz resample from the retained 22.05 kHz source and requires the canonical
  WAV's PCM to be **byte-equal** to the recomputation, with sample-count
  arithmetic checked and the source WAV bound to the probe's streamed
  `audio_bytes`. An evaluator that scored substituted or re-rendered audio
  cannot pass.
- **Exact evaluator platform contract**: beyond the A101 judge pins, provenance
  must now match the deployed platform exactly (torch `2.10.0a0+…nv25.12`,
  CUDA 13.1, Python 3.12.3, `aarch64`, `NVIDIA GB10`). Intentional consequence,
  recorded: parity artifacts are valid only from the qualification host —
  cross-platform evidence fails closed by design.

### 2. Orchestration — order and GPU ownership are enforced by structure

`run_tool_freshness_parity_only` refuses any runtime image other than the
configured one and any evaluator other than the **hard-pinned promoted ID**
(`sha256:598542…`, matching the A89 judge of record). The sequence is
structurally ordered, not timing-dependent: fixtures are generated and doubly
validated (fixture validator + live source provenance) **before** the stack
starts; the paired arms run against the live stack with readiness/idle gates;
the stack teardown lives in a `finally` (SIGINT → bounded wait → kill) that
completes **before** the ASR loop begins — the evaluator containers can never
contend with the stack for the GPU. Both evaluations mount raw inputs
**read-only** (`/qualification-input:ro`) with separate writable output
directories; `run_json_report_command` is fail-closed (missing, unreadable, or
non-boolean-verdict reports raise; exit codes are retained as evidence rather
than trusted); the comparator receives both evaluated reports **and** the
runtime report; and the stage ledger enforces running→completed ordering with
atomic writes and per-stage report SHAs. The orchestration verdict is the
comparator's verdict alone — no stage can green the run early, and an
interrupted run retains `passed: false` with the failing stage visible.

### 3. Fail-open hunt — none found

The evaluated-base equality closes the probe→evaluation tampering window; the
byte-derivation closes the audio-substitution window; the platform/judge pins
close the wrong-instrument window; the RO input mounts close in-place mutation
during evaluation; and every parse/validation error path raises rather than
degrades. Tests behaviorally cover the new red paths alongside the A101 set;
114 tests + Ruff reproduced green.

### Verdict

**APPROVE.** The comparator now binds the full chain — scenario → fixtures →
probe arms → raw reports → ASR evaluation → comparison — with every link either
rederived or hash-bound, and the orchestration executes the A82 experiment in
the only defensible order with GPU ownership guaranteed by control flow. The
instrument is complete and clean; the remaining act is running it. D8 intact.

## Addendum 103 — fixture-container import fix: provenance/twin-identity check (2026-08-09)

Narrow review of the live-attempt-1 fix. 88 tests + Ruff reproduced green.
Review only; no GPU.

### Verified

- **The failure was fail-closed and the fix is minimal**: the runtime image's
  *installed* `nemotron_voicechat_runtime` package predates the new contract
  module, so the mounted generator died on import before model startup.
  `tool_freshness_fixture_command` now sets `PYTHONPATH=/workspace/project/src`
  — the mounted, reviewed sources win for the **generator process only**, with
  the repo mounted read-only.
- **The synthesis plane is untouched — verified in code**:
  `PocketWorkerManager._worker_environment` builds a narrow **allowlist**
  environment (HOME/locale/TZ/HF-cache only; `PYTHONPATH` is excluded by
  construction, not by deletion) and launches `/opt/pocket-tts/bin/python` —
  the installed, audited Pocket environment. The mounted sources cannot leak
  into the process that actually synthesizes audio.
- **Provenance closes over both code planes**: `FIXTURE_PROJECT_SOURCE_PATHS`
  includes `tool_freshness_contract.py` — the exact file that caused the
  failure is hash-bound into the manifest along with the other mounted
  generator/controller sources, and `FIXTURE_INSTALLED_WORKER_SOURCE` binds the
  installed worker; both are revalidated against live files at generation and
  again at orchestration. Mounted-vs-installed divergence is therefore
  *recorded and validated*, not hidden.
- **Twin identity unweakened**: the live server's typed-input Pocket and the
  fixture generator's Pocket both run the installed environment, so the
  A98-era synthesis-identity equality (manifest vs live health) still compares
  like with like.

### Note

The installed-package-predates-mounted-sources condition is the known dev-overlay
convention on this branch; every mounted-code plane in this experiment is
hash-bound, and the server's own behavior flows from the installed package under
its separate provenance chain. A future runtime-image rebuild reconciles the
drift; nothing in this experiment requires it.

### Verdict

**APPROVE — no provenance or twin-identity blocker.** The fix scopes the mounted
sources to the generator process, the audited installed worker keeps sole
ownership of synthesis, and the manifest's dual source inventory makes the whole
arrangement content-addressed. The live parity experiment may be re-attempted.
D8 intact.

### Update (same day): transitive source inventory completed

Self-review found the mounted-source inventory omitted modules the generator
loads **transitively** — a real gap in the hash closure this addendum had just
endorsed. Verified fixed: `FIXTURE_PROJECT_SOURCE_PATHS` now includes
`src/nemotron_voicechat_runtime/__init__.py`, `pocket_ipc.py`, and
`protocol.py`, and this review confirmed the transitivity claim in code
(`pocket_controller` imports `pocket_ipc`; the package `__init__` imports
`protocol`) — those files genuinely execute in the generator's mounted-source
process and were previously unbound. Because `_validate_sources` enforces exact
set equality, the expanded inventory is automatically *required* by both the
manifest schema validator and the current-byte provenance validator, and the
generic re-signed mutation coverage applies to the new rows. The isolated
`/opt/pocket-tts` worker remains allowlist-enveloped and image-installed.
122 focused tests + Ruff reproduced green. **Approval stands on the completed
inventory.**

## Addendum 104 — fixture-container Pocket cache fix (2026-08-09)

Narrow review of the live-attempt-2 fix. 88 tests + Ruff reproduced green.
Review only; no GPU.

### Verified

- **The failure was fail-closed** (missing asset cache killed the fixture stage
  before model startup) and **the fix reuses the established pattern
  verbatim**: `tool_freshness_fixture_command` now sets
  `HF_HOME=/models/huggingface` and
  `HUGGINGFACE_HUB_CACHE=/models/huggingface/hub` and mounts
  `cache_root/huggingface` **read-only** at `/models/huggingface` — the exact
  env values and mount used by the suite's existing fixture/probe commands —
  while `--network none` and both offline flags are retained.
- **Isolation contract holds**: the worker's allowlist environment (A103)
  passes exactly `HF_HOME`, `HUGGINGFACE_HUB_CACHE` (and `XDG_CACHE_HOME`) —
  the installed `/opt/pocket-tts` worker receives precisely the cache pointers
  and nothing else; `PYTHONPATH` remains stripped by construction.
- **Provenance cannot be spoofed via the cache**: the mount is read-only, and
  the manifest validator requires `synthesis.assets == EXPECTED_ASSETS`
  (per-asset bytes + SHA-256, reported by the worker from what it actually
  loaded) — a wrong or stale cache fails validation, a missing cache fails
  startup. Both directions fail closed.
- **Path contract**: the mount derives from `args.cache_root` exactly as the
  rest of the suite does — no new path convention introduced.

### Verdict

**APPROVE.** A pattern-conformant, read-only, offline cache injection whose
contents remain hash-verified by the existing asset contract. The live parity
experiment may be re-attempted. D8 intact.

## Addendum 105 — attempt-3 hang diagnosis: FC async stalled on first typed turn (2026-08-09)

Adversarial diagnosis from `stack.log` and the typed-arm events of
`tool-freshness-parity-20260809T215633Z`. Review only; no edits, no GPU.

### 1. Verified facts

- Clean run up to the tool call: typed injection completed, client-EOU committed,
  **boundary SOTC at frame 74** ("Committed boundary SOTC while agent idle at
  t=74"), FC-async background thread spawned for stream 2, ack/reminder phrases
  chosen, loop entered at frame 75 with `realtime_audio=OFF, live_audio=ON`.
- Immediately at entry: "Aborted generation for request: 2" → "Starting
  generation session with request_id: 2" → a vLLM "EOS token id … tokenizer is
  not initialized" warning → one bf16 `AudioPreprocessor` warning (the live
  perception worker consumed its first frame) — then **zero FC log lines for
  60 s**.
- Meanwhile the **foreground kept healthy**: transport frames advanced past 667
  with ~0.3 ms synthetic steps (the by-design background-mode early return,
  which also pushes each incoming client frame into `live_audio_queue` before
  returning), client silence frames demonstrably arriving
  (`input_audio.samples=1280`), metrics streaming
  `active/background_active, completed_calls=0`, `function_text` still only the
  SOTC. No thread traceback anywhere in the log (the only Python traceback is
  the codec worker's expected EOF at teardown).
- **No environment drift**: the checked-in patch hashes to exactly the
  `4902de6b…` the runtime health reports, and r6-retry completed four full FC
  cycles **on this same image** — the runtime code path is proven-good under the
  sustained probe's traffic pattern.

### 2. Diagnosis — to the depth the evidence supports

The background FC loop made **zero LLM progress** while every upstream feed was
verified alive: audio arriving, frames pushed to the live queue, the perception
worker processing (the bf16 warning is its first-frame signature), ack TTS
primed. The stall therefore sits at the loop's Nano generation step. The
entry-time abort/restart of request 2 is the pivotal ambiguous datum: the
"tokenizer is not initialized" signature suggests it is the **EarTTS**
ack-phrase re-prime (benign, by design); if it was instead the **Nano** request,
the restart created a fresh request with no session KV and the loop would
generate nothing useful indefinitely. The logs do not name the engine, and the
async loop has no per-step logging at INFO level, so the stall point
(blocked-on-first-token vs generating-PADs-forever) cannot be separated
post-mortem. What *is* clear is the *distinguishing variable from the
proven-good r6-retry flow*: the parity typed arm sends **no client audio until
`injection_finished`**, whereas every FC cycle that has ever completed on this
runtime ran under the sustained probe's **continuous silence from t=0**. The
A98 review flagged exactly this delta as the typed arm's residual risk.

### 3. Proposed direction — diagnose-first, probe-side first

1. **Align the typed arm with the proven traffic pattern** (probe-side only):
   start the silence clock at session start, before injection, exactly as the
   sustained probe does — eliminating the only client-side behavioral delta
   from the r6-retry flow. The scenario contract is unaffected (the audio is
   silence either way; the scenario hash covers prompts/outputs/instructions,
   not clock start).
2. **Add a bounded async-loop heartbeat** (runtime, trace-gated like A76): one
   INFO line every N async steps (step count, latest function token, queue
   depths), so any recurrence localizes to blocked-before-first-token vs
   PAD-spinning — the two hypotheses this post-mortem cannot separate.
3. Rerun the typed arm alone (it fails fast). If the hang persists **with**
   continuous silence, the runtime is implicated with heartbeat evidence in
   hand; if it disappears, the root cause is the audio-gap interaction and the
   fix is already in place.

**BLOCK any blind runtime/scheduler patching** (FC-async, pad-pair latch, or
request lifecycle changes) until the heartbeat evidence localizes the stall —
the machinery under suspicion is qualification-critical and proven-good under
the sustained pattern, and a speculative fix there risks invalidating the
r-series baseline for an unproven cause.

### Verdict

**APPROVE the diagnose-first direction (§3): probe-side clock alignment plus a
bounded heartbeat, then a fast retry — and BLOCK runtime-side fixes until the
stall is localized.** The failure is fail-closed (60 s turn timeout, no false
evidence), the instrument itself is unimplicated, and the one behavioral delta
from every previously successful FC cycle is precisely the thing step 1
removes. D8 intact.

## Addendum 106 — attempt 4: parity achieved; margin rule adjudicated in attended review (2026-08-09)

Adversarial review of the attempt-4 artifacts. Review only; no edits.

### 1. The experiment succeeded ~~— and the A105 diagnosis is confirmed~~
**[CORRECTED by A107: attempt 4 was an unchanged retry — no clock alignment or
any source change landed between attempts 3 and 4. The original text below
claimed causal confirmation; that claim was false. The attempt-3 hang is
nondeterministic, and A105's audio-gap hypothesis is downgraded from "confirmed"
to "candidate race"; see A107 §1.]**

All eight FC cycles completed (4 typed + 4 speech), both raw structural arms
passed every conjunct, and both immutable-judge ASR integrity gates passed.

### 2. The parity result — this is the A82 promotion evidence

Across all four paired cases and **both** semantic layers (model text and
independently transcribed audio): **zero typed-only failures, zero speech-only
failures — every expected-code miss is a shared failure**, exactly the outcome
the r6-retry top-k predicted, now confirmed **cross-modality**. The speech arm
was heard essentially perfectly (input WER 0.0/0.0/0.0455/0.0). Case-level
response WERs are comparable across arms (case 4: 0.0 both). Under the A82
criterion — parity, not absolute intelligence — **typed input has demonstrated
response parity with speech input on this probe.**

### 3. The margin block: not a harness error; correctly firing; adjudicated benign

The comparator is red solely on `response_asr_wer_margin_below_0.1` for cases 1
and 3. Inspection of the raw pairs:

- **Case 3 (both arms WER exactly 0.5)**: the reference is the two-word
  degenerate truncation `' Current UTC'` — WER quantizes in **half-word steps**,
  and both judges garbled the bare acronym ("achieve."/"at"), the same
  "UTC"-acronym noise documented in every run since r4 and in the judge-swap
  calibration itself. One substitution in two words = exactly 0.5.
- **Case 1 speech arm (WER exactly 0.5)**: eight-word reference; the 4 errors
  include `utc→ultra`, `utc→ach` + an inserted `test` (acronym noise) — and
  `o→oh`, a **pure orthography artifact** (the model wrote "fifteen o one", the
  judge wrote "oh") that the normalizer does not equate. The typed arm's 0.125
  on the same sentence differs mainly by that same o/oh token.

This is **not a harness or reference error**: the references are correctly the
model's own text, the WER computation is the established matcher behaving as
calibrated, and the values are exact and reproducible. The rule is also **not
firing falsely** in its narrow sense: `audio_integrity_passed` for these cases
sits at equality with the 0.5 threshold, so a marginally different judge could
flip integrity and invalidate the arm — that knife-edge is real and worth
surfacing. But the margin risk threatens **evidence robustness, not conclusion
direction**: the semantic outcome (shared failures everywhere) is
margin-independent — no WER perturbation changes any parity verdict.

### 4. Adjudication and the smallest honest next step

A89's pre-registered rule was "**held for explicit review**; cannot be cited as
an *unattended* green" — and A101 recorded its hard-block implementation with
the express purpose of keeping margin cases separable "for a reviewing
operator." **This addendum is that attended review.** Adjudication: the two
margin cases are benign — half-word quantization on documented model-truncation
references plus known judge acronym/orthography noise, with zero effect on the
parity conclusion — and the comparator artifact **stands red, unmodified**, as
the honest record that this green is attended, not unattended.

**The smallest honest next step is no code change at all**: the promotion
decision proceeds on the pair {comparator artifact (red, margin reasons only) +
this attended adjudication}. **BLOCKED as laundering**: retuning the normalizer
(e.g., o/oh equivalence), the WER threshold, or the margin bound *for this run*;
any prospective reversion of the hard-block to A89's original hold semantics
must follow the A90 pattern — preserve this artifact as-is and amend policy
versioned-prospectively.

### Verdict

**APPROVE promotion on attended review.** The parity instrument delivered its
answer: eight clean FC cycles, structural and integrity gates green in both
modalities, and every semantic failure shared — text input is at parity with
speech input, with the code-readback limitation now documented as a
**model-level, modality-independent** behavior (r6 top-k mechanism, confirmed
cross-modality). The margin red is adjudicated benign in this explicit review
and preserved unmodified. This closes the A82 question and, with it, the
evidence chain this review series set out to build. D8 intact.

## Addendum 107 — corrections to A106 and consensus direction (2026-08-09)

Convergence review after the independent reviewer's block. This addendum owns
two errors in A106 and sets the consensus path. Review only.

### 1. Correction one: the hang is nondeterministic; A105's causal call is unproven

Attempt 4 ran **unchanged code** — no clock alignment, no source change of any
kind between attempts 3 and 4. A106's claim that the fix "empirically confirmed"
the A105 diagnosis was therefore false, and A106 §1 has been amended in place.
What the evidence now actually shows: the same configuration both hung
(attempt 3) and completed cleanly (attempt 4) — the FC-entry hang is a
**nondeterministic race**, and the audio-gap hypothesis survives only as a
candidate race window (injection-finish → clock start vs SOTC timing), not as a
diagnosis. Consequence, binding: the A105 **heartbeat instrumentation becomes
required before further live attempts** — an intermittent hang inside
qualification runs is itself a defect to be localized, and the next occurrence
must produce localizing evidence rather than another blind retry.

### 2. Correction two: A106's attended-promotion verdict overstepped the executable policy

The reconciliation of A89 ("held for explicit review; not citable as an
*unattended* green") with A101/A102 (hard block) is not a judgment call this
series is free to make per-run: this series itself established — in the
judge-swap reconciliation — that **the executable policy is the policy**, and
that softening a checked-in rule for the benefit of a concrete failing run is
the laundering pattern A90 exists to forbid. A106 invoked the older, softer
wording to approve through an implemented hard block; that was a policy
misapplication, and the independent reviewer's block is correct. **A106's
promotion approval is withdrawn.** The run stands BLOCKED under the implemented
margin policy. If attended-hold semantics are wanted, the path is the A90
pattern: preserve this comparator artifact red and unmodified, and amend the
policy as a versioned, prospective change adjudicating only future runs.

### 3. The reviewer's case-1 observation is stronger than A106's adjudication

A106 filed case 1 under "orthography and acronym noise." The sharper fact: the
two arms produced **identical eight-word model text**, hence identical EarTTS
input — yet the typed arm's response audio transcribed at WER 0.125 and the
speech arm's at 0.5 ("Current ultra time is fifteen. Oh, one Ach test."). Same
text, same synthesis stack, divergent acoustic outcome is not reference
quantization; it is either **modality-conditioned EarTTS fidelity** (a real
typed-vs-speech product difference in the response audio channel — exactly what
a parity qualification must not wave through) or **session/order variance** (the
speech arm ran second). A106 lacked the cross-arm comparison to separate these;
the margin rule, by blocking, did precisely its job.

### 4. Consensus direction

1. **Preregistered, non-promotable, reverse-order replication**: same scenario
   and fixtures (scenario hash identical), fresh sessions, execution order
   **speech → typed**, artifact explicitly marked evidence-only/non-promotable
   so it cannot be laundered into a promotion claim regardless of outcome.
   Pre-registered discriminators: if the *second-position* arm degrades
   regardless of modality → session/order variance; if the *speech* arm
   degrades regardless of position → modality-conditioned EarTTS fidelity,
   which then becomes a real pre-promotion defect to investigate. Identical
   full-sentence cases (case-1 class) are the comparison rows.
2. **Heartbeat instrumentation lands first** (per §1) so a recurrence of the
   FC-entry hang localizes instead of wasting a GPU run.
3. **No margin/normalizer/threshold changes**; the attempt-4 comparator
   artifact is preserved red; any policy amendment follows A90's
   versioned-prospective pattern and cannot adjudicate attempt 4.
4. What survives of A106 unchanged: the **semantic parity observation** —
   shared code-readback failures in both arms and both layers, zero typed-only
   and zero speech-only semantic failures — remains true and
   margin-independent, but it is an *observation awaiting promotable evidence*,
   not a promotion.

### Verdict

**BLOCK promotion (A106's approval withdrawn); APPROVE the consensus
direction** — heartbeat first, then the preregistered non-promotable
reverse-order replication to separate modality-conditioned audio fidelity from
order variance, with the attempt-4 artifact preserved red and every policy
question routed through the A90 versioned-prospective pattern. This is the
third time in the series an independent disagreement has corrected this
review's judgment (A87, A95, A107) — recorded deliberately, because that
correction loop is the strongest property this evidence chain has. D8 intact.

## Addendum 108 — FC-async heartbeat: pre-implementation design review (2026-08-09)

Adversarial review of the proposed heartbeat design against the A105/A107
requirement. Review only.

### 1. The core design is right

Opt-in env (`S2S_FC_ASYNC_HEARTBEAT=1`, default off), scalar-only dict
**atomically replaced by reference** (build-fresh-then-swap, never mutate a
published dict — the correct GIL-safe pattern for the bg-thread/server-reader
pair), surfaced under `turn_state.function_calling` only when present (default
metrics schema byte-unchanged; no checker consumes it, so no gate impact),
`faulthandler` SIGUSR1 registration only under the opt-in, and the
orchestration's dump-before-teardown ordering on arm failure with the stack
still alive. The disabled path doing **zero** clock/qsize calls, with a test
asserting it, keeps the observational instrument provably inert — consistent
with the A76/A78 diagnostic conventions. This is observational-only
instrumentation (no control-flow change in any mode), so it does not violate
A105's block on runtime behavior changes.

### 2. Two required amendments — both aimed at the actual incident

1. **Stage coverage must include the FC-entry sequence and the wait points.**
   The attempt-3 evidence ends at *loop entry* (abort/restart, ack prime, one
   perception frame — then silence). The proposed stage set
   {nano, eartts, codec, rnnt} only brackets work *inside* a loop iteration:
   if the stall is in the entry sequence (`start_generation` await, ack
   priming) the heartbeat dict may never be written at all, and if the loop
   blocks on the **perception/emb-queue get** — the leading starvation
   hypothesis — the stall lands *between* stages, distinguishable only by
   timestamp inference. Require: an initial heartbeat written **before** the
   entry sequence with explicit entry stages (e.g. `entry_start_generation`,
   `entry_ack_prime`, `loop_entry`), and the emb/audio wait bracketed as its
   own stage (e.g. `emb_wait`). Without these, the instrument can miss
   precisely the stall it was built to localize. The bounded queue depths
   should name the four real queues (live-audio, perception/emb, TTS-out,
   RNNT-text) so an empty-emb-queue + `emb_wait` heartbeat proves starvation
   directly.
2. **Verify SIGUSR1 actually reaches the Python server process.**
   `docker kill --signal` delivers to PID 1; if the container entrypoint wraps
   the server in a shell, the dump never happens and the one chance at
   localizing evidence is lost at the worst moment. Require either
   verification that the server is PID 1 (exec-form entrypoint) or targeted
   delivery (`docker exec … kill -USR1 <python-pid>`), plus a smoke assertion
   in the orchestration path (dump marker text expected in `stack.log` after
   signal on a healthy stack, exercisable in a cheap test window).

### 3. Test additions beyond the proposed set (required with §2)

Entry-stage write-before-loop coverage; published-dict immutability under
concurrent read (mock server serialization while the bg swaps); patch-contract
assertions for the heartbeat block per the established pattern. The proposed
set (disabled-path no-clock/no-qsize, mocked stage transitions, scalar
schema/bounds, metric omission/presence, failure-only signal ordering) is
otherwise complete.

### 4. Note on image identity

The heartbeat lands in the speech patch → next runtime image rebuild → new
image ID. Provenance re-binds automatically through the existing chains
(fixture manifests, orchestration state, health), and the default-off metric
schema keeps r-series comparability; no baseline invalidation, since disabled
behavior is byte-identical.

### Verdict

**APPROVE the design with the two §2 amendments as binding requirements** —
entry/wait stage coverage (otherwise the instrument cannot see the one stall it
exists for) and verified signal delivery (otherwise the dump path is
decorative). Everything else — opt-in inertness, atomic scalar publication,
conditional metric surfacing, failure-only dump-then-teardown ordering, and the
test posture — is approved as proposed. D8 intact.

## Addendum 109 — heartbeat implementation review (2026-08-09)

Adversarial review of the implementation against A108. 194 focused tests +
3 subtests + Ruff reproduced green; patch parity verified against the sibling
checkout **and** `git apply --check` confirmed the patch applies to clean Speech
HEAD. Review only.

### Both A108 amendments landed — beyond what was required

- **Stage coverage**: the main-loop heartbeat covers `entry` (published at FC
  entry before the loop, with stale-key pop per cycle), `prepare`,
  `perception_poll` (the emb-queue wait — the leading starvation hypothesis is
  now a *named* stage), `nano`, `eartts`, `codec_decode`, `codec_cpu_copy`,
  `rnnt`, `between_stages`, `complete`, `failed`/`stage_failed` — and the
  live-perception thread gets its **own** heartbeat key
  (`audio_wait`/`perception_encode`/`perception_sync`/`embedding_publish`),
  so producer-side starvation and consumer-side blocking are separately
  visible. All four queue depths are bounded-recorded, `thread_ident` included,
  and the cycle identity (`cycle_sotc_frame` + invocation) carries
  `last_completed`/`last_failed` across the two-phase FC correctly.
- **Signal reachability**: the model container runs under `--init`, so PID 1 is
  the init shim which forwards USR1 to the Python child; the orchestration
  validates the container name, writes and flushes a marker into `stack.log`,
  records the marker offset, sends `docker kill --signal=USR1`, then **verifies
  the faulthandler thread signature** (`Current thread `/`Thread 0x`) appears
  after the marker before recording `dump_observed` — delivery is checked, not
  assumed. The dump path runs only in the arm-exception handler
  (`KeyboardInterrupt` re-raised above it; success path untouched), its own
  failures are captured without masking the original exception, and SIGINT
  teardown follows.

### Stage honesty and publication semantics — verified

Swallowed stage failures publish `stage_failed` with the failing stage named
**before** the warning-and-continue (verified at the TTS site; the codec stages
follow the same pattern per the stage inventory) — a silent-swallow can no
longer masquerade as a healthy stage. Publication is whole-dict replacement of
an immutable-after-publish scalar snapshot; the server **copies then
exact-validates** (frozenset key equality, stage/state enums with
cross-invariants — `between_stages`↔`idle`, `complete`↔`complete`,
failed-pairing — finite monotonic bounds, phase/invocation consistency) before
conditional inclusion in metrics, and `faulthandler` registration is gated on
the same opt-in. Default-off neutrality re-verified by execution; every publish
site is guarded so the disabled path performs no clock or qsize work.

### Verdict

**APPROVE.** Both A108 amendments are implemented more thoroughly than
specified, stage honesty covers the swallowed-failure paths, signal delivery is
verified rather than hoped, the default path is provably inert, and the patch
is clean against upstream HEAD with parity to the sibling tree. The heartbeat
satisfies A107 §4.2; the preregistered non-promotable reverse-order replication
may now proceed — with localizing evidence guaranteed if the FC-entry race
recurs. D8 intact.

## Addendum 109a — final heartbeat hardening (2026-08-09)

Short re-review of the post-A109 hardening. Verified in source, patch parity
and clean-HEAD application re-confirmed, focused suites + Ruff reproduced
green. Review only.

- **Disabled-path selection is now structural**: with the heartbeat off, the
  perception thread target is the **original** `_perception_live_worker`
  function object; the guarded wrapper (which delegates to the original) is
  selected only under the opt-in. The disabled thread path contains zero added
  code — a stronger neutrality guarantee than internal guards.
- **The phase-overwrite race is closed** — a real gap A109 missed: a phase-1
  perception worker outliving its bounded join during phase-2 re-entry could
  previously clobber phase-2 evidence with a stale snapshot at exactly the
  moment a post-mortem would read it. The publisher now rejects any write whose
  invocation is **lower** than the stored same-cycle heartbeat's — same-cycle
  scoped, strictly-greater comparison, so normal ordering is unaffected while
  stragglers are dropped.

**APPROVE — the heartbeat step is closed.** No regression found; both changes
strengthen properties A109 reviewed. The preregistered non-promotable
reverse-order replication is cleared to run on the rebuilt image. D8 intact.

## Addendum 110 — reverse-order diagnostic: pre-implementation design review (2026-08-09)

Adversarial review of the proposed design against A107 §4.1. Review only.

### 1. The anti-laundering structure is the best in the series

Non-promotability is enforced **structurally, twice over**: the top-level report
gets a distinct kind (`tool_freshness_reverse_order_diagnostic_runtime`) that
the promotion comparator's `validate_runtime_report` cannot accept, *and* its
execution order `[speech, typed]` independently fails the comparator's order
bind — while `promotion_eligible: false` **and `passed: false` by
construction** (with `diagnostic_complete` as the only success signal and CLI
exit keyed to it) means no downstream consumer can ever read this artifact as a
green gate, whatever its content. Arm reports keeping the existing
ASR-compatible kind is right: the promoted judge, transcriber, and
`validate_arm` machinery apply unchanged, so the diagnostic measures with the
same instrument as the forward run. Freezing attempt 4 by double hash-pin
(file SHA `4889acc9…`, self-hash `4916ec73…`) **and** independently regenerating
the forward comparator from retained paths with exact comparison is
rot-detection done properly. The restricted label enum with explicitly no
causal/promotion verdict is honest n=1 epistemics, and the non-gating acoustic
envelope measurements (first/last active, trailing frames at preregistered
−60 dBFS/80 ms) add exactly the discriminator the case-1 puzzle needs — whether
the speech arm's *retained audio* differs in envelope, not just in transcript.

### 2. Two required additions — both are missing bindings, not redesigns

1. **Cross-run fixture identity.** The reverse run executes on the rebuilt
   heartbeat image, so fixtures must be regenerated under the new image ID
   (the manifest's provenance binding forces this). Pocket synthesis is
   seeded and asset-pinned, so the regenerated PCM **should** be byte-identical
   to attempt 4's — but the design never asserts it, and the entire
   forward-vs-reverse speech comparison silently presumes identical speech-arm
   input audio. Require: the analyzer exact-compares per-fixture PCM/WAV
   SHA-256s and the synthesis identity block between the attempt-4 manifest
   and the reverse-run manifest; on mismatch, either `diagnostic_complete:
   false` or an explicit labeled confound field — never silent.
2. **Preregistered label assignment.** The enum is specified but the mapping is
   not: labels must be computed by a **deterministic pure function of the
   recorded scalars** (the decision table itself preregistered in code and
   pinned by tests — e.g., speech arm worse in both orders on a case class →
   `consistent_with_speech_modality`; second-position arm worse in both
   experiments → `consistent_with_second_position`; else tie/unstable), so no
   post-hoc labeling discretion exists when the data arrives.

**Recommended (non-gating)**: a per-case `model_text_identical_to_forward` flag
per arm. Under greedy decode the reverse run's texts should reproduce
attempt 4's; the flag cleanly separates decode-level variance (different text →
`unstable` territory) from judge/acoustic variance (identical text, shifted
WER — the actual case-1 question). It costs nothing since both texts are
already recorded.

### 3. Everything else

Same hash-complete scenario (code-derived, image-independent), instrumented
stack with heartbeat and verified dump path, promoted-judge evaluations in
arm order with teardown-before-ASR, no threshold/normalizer/margin changes, and
a test list that already covers forward-unchanged, ordering/kind/fresh-session,
non-promotable invariants, frozen-hash rederivation, mutation classes, audio
math, GPU order, and analyzer-can-never-green. With §2's additions folded into
the test list (fixture-identity binding test; label decision-table test), this
is the smallest honest design consistent with everything this series has
established.

### Verdict

**APPROVE with the two §2 additions as binding requirements.** The design's
core property — an artifact that is structurally incapable of becoming
promotion evidence while measuring with the identical instrument — is exactly
what A107 §4.1 ordered, and the double-pinned, re-derived forward baseline
makes the comparison honest in both directions. D8 intact.

## Addendum 111 — reverse-order diagnostic: implementation review (2026-08-09)

Adversarial review of the implementation against A110. 110 focused tests + Ruff
reproduced green. Review only.

### Every A110 requirement verified in source

- **Cross-run fixture identity (A110 §2.1)**: `_fixture_pair_identity` compares
  synthesis identity plus per-case case/ordinal/text and **PCM and WAV
  SHA-256s** between the frozen attempt-4 manifest and the reverse manifest,
  and hard-raises on any mismatch (the stricter of A110's two permitted
  outcomes) — the speech-arm input-identity premise is now enforced, not
  presumed.
- **Preregistered labels (A110 §2.2)**: `classify_wer_pattern` is a pure
  function over the four WER cells with schema-drift rejection and a
  parametrized decision-table test. The position mapping is correct (forward
  pos-2 = speech, reverse pos-2 = typed), the branches are provably mutually
  exclusive (any overlap collapses to `tie`), and the strict-equality
  semantics mean near-miss patterns fall to `unstable` — zero labeling
  discretion, with the raw cells always recorded for human reading.
- **`model_text_identical_to_forward` flags** (the A110 recommendation) are
  present per case and per arm.

### The rest of the checklist

- **Anti-laundering exit semantics**: `passed: false` and
  `promotion_eligible: false` in *every* artifact shape — analysis, preflight,
  and error — with the CLI exit keyed solely to `diagnostic_complete`, a
  per-case `interpretation_is_causal: false`, and an embedded interpretation
  scope. The orchestration independently re-asserts the preflight triple.
- **Preflight-before-GPU**: the orchestration runs the analyzer
  `--preflight-only` — which performs the **full** frozen-forward revalidation,
  including regenerating the attempt-4 comparator from retained paths and
  requiring exact equality *and* preserved red — before any stack or GPU work,
  and hard-fails on drift. Four hash pins total (comparison file + self-hash,
  fixture file + self-hash), with the comparison additionally bound to its
  fixture.
- **Runtime confound**: enforced, not just recorded — the analyzer *raises* if
  the reverse run reused the pre-heartbeat image, and stamps
  `runtime_provenance_mismatch_confounds_exact_ab: true` with both provenances
  embedded.
- **Classifier/audio math**: integral 80 ms framing with explicit
  non-integral-rate rejection, float64 RMS with a −240 dB floor, preregistered
  −60 dBFS threshold, partial-final-frame handling without padding (tested),
  all-inactive → all-trailing (tested), empty-WAV rejection (tested),
  `gating: false` stamped.
- **Forward-mode neutrality**: `validate_runtime_report` gained keyword-only
  parameters whose **defaults are the forward contract** (kind, (typed, speech)
  order, no non-promotable requirement) — the promotion comparator path is
  byte-equivalent by default; the probe emits the diagnostic runtime kind and
  `promotion_eligible: false` only under `--nonpromotable-reverse-order`.
- **Orchestration**: same stage ledger, fail-closed JSON-report runner,
  heartbeat-instrumented stack, and failure/thread-dump machinery as the
  forward mode, with ASR evaluations in arm order after teardown.

### Verdict

**APPROVE.** Both A110 binding requirements are implemented in their stricter
forms, the recommendation landed, non-promotability is unforgeable across every
artifact shape and exit path, the frozen forward baseline is quadruple-pinned
and re-derived before any GPU is spent, and the run-order confound is enforced
into the record. The reverse-order diagnostic may run. D8 intact.

## Addendum 111a — post-A111 adversarial hardening (2026-08-09)

Re-review of the five fixes from the separate reviewer's pass. 121 focused
tests + Ruff reproduced green. Review only.

### All five verified in source — and two were misses A111 should own

1. **Dedicated diagnostic runner**: `run_diagnostic_report_command` encodes the
   inverted contract exactly — report must carry `passed: false` **and**
   `promotion_eligible: false`, exit must be 0/1, and
   `(returncode == 0) is diagnostic_complete` is asserted, so the exit status
   can never contradict evidence completeness. The generic gate runner's
   exit-means-passed semantics genuinely conflicted with a deliberately-red
   artifact; A111 did not check that interaction.
2. **Forward validator now rejects the *presence* of `promotion_eligible`**
   (`"promotion_eligible" in runtime` fails the forward path) — closing the
   laundering seam where a hand-built forward-shaped report carrying the
   nonpromotable marker would have validated. The key is now absent-or-exactly-
   False depending on mode, never merely tolerated. **A111 missed this seam.**
3. **Heartbeat evidence proven, not inferred**: beyond unequal image IDs, the
   analyzer requires `{fc_async_heartbeat: true, sigusr1_all_thread_dump: true}`
   to appear identically in the reverse runtime provenance **and both arms'
   health snapshots** — three independent attestations that the instrumented
   build ran with the heartbeat actually enabled, so a heartbeat-present-but-
   disabled image can no longer satisfy the confound check.
4. **Preflight can no longer be trusted from disk**: the orchestration refuses
   a pre-existing output unless an **in-process** marker
   (`_reverse_preflight_verified_in_process`) was set by the CLI's own fresh
   preflight run this invocation — a planted or stale `preflight.json` is
   worthless because the marker lives in process memory, not the filesystem.
   **A111 read the resume branch and failed to flag that it trusted an
   existing file.**
5. **Test posture expanded** (110 → 121): synthetic analyzer/hash/identity
   mutations and the reverse failure-dump path now covered.

### Verdict

**APPROVE.** Every fix is in its correct, stricter form; two of the five were
genuine gaps in A111's own pass, recorded here per the series' correction
convention. The reverse-order diagnostic remains cleared to run — now with
exit semantics, provenance attestations, and preflight freshness all
unforgeable. D8 intact.

## Addendum 111b — envelope symmetry + heartbeat-test coherence (2026-08-09)

Final delta re-review. 121 focused tests + Ruff reproduced green. Review only.

- **Four-cell envelope, SHA-bound**: `response_audio_envelope` now covers
  `typed_forward`/`speech_forward`/`speech_reverse`/`typed_reverse`, each
  resolved from its validated arm's retained source WAV, with
  `retained_envelope` **raising unless the computed file SHA equals the
  `validate_arm`-recorded `source_wav.sha256`** — the envelope provably
  measured the exact bytes the validated ASR chain scored. This closes a real
  asymmetry from A111: the case-1 question (identical text, divergent audio
  WER) lives in the **forward** speech arm's audio, which the reverse-only
  envelope could never examine. The acoustic discriminator now covers both
  experiments.
- **Heartbeat mutation test reaches its target**: the mutation now coherently
  updates the diagnostics block in both the runtime provenance and the health
  snapshot across raw and evaluated evidence, so it survives the upstream
  validation layers and actually exercises the heartbeat guard — previously
  the test died earlier and the guard's red path was untested.

**APPROVE.** Both changes strengthen A111-reviewed properties with no contract
drift; the diagnostic's evidence is now acoustically symmetric and every guard
has a genuine red-path test. The reverse-order diagnostic is — finally and
fully — cleared to run. D8 intact.

## Addendum 113 — bootstrap source-freshness review (2026-08-09)

Adversarial review of the source-identity implementation. 28 tests + Ruff
reproduced green; hash determinism verified by execution. Review only.

### Verified

- **Single code path**: the build script, the audit script, and the CLI all
  compute the identity via the same
  `public_runtime_identity.public_runtime_source_sha256` — no shell
  reimplementation to drift (test-pinned via the ARG/label/build-arg string
  assertions). The Dockerfile takes the value as a build ARG into the pinned
  label; the audit recomputes from the checkout and compares to the exact
  label.
- **Canonical, content-based hashing**: length-prefixed path+bytes into one
  SHA-256 (no delimiter ambiguity), sorted by posix relpath, deduplicated;
  missing inputs fail closed. Being filesystem-based rather than git-based is
  the right choice: untracked new files under `src` change the identity —
  no false-fresh from uncommitted additions.
- **Inventory ↔ COPY coverage**: every `COPY` input in
  `Dockerfile.public-runtime` (config JSON, qualified-deltas tree,
  reconstruct script, patch-under-src, constraints, pyproject/README, the
  full `src` tree) is in the inventory, plus both Dockerfiles and the build
  script themselves — conservative in the right direction.
- **Lifecycle gates**: `_build_image` audits a matching image, rebuilds a
  stale one online, and fails offline with a missing-vs-stale distinction;
  `_bootstrap_ready` requires the checkout↔image match; `up` (and the reverse
  diagnostic entry) gate on `_bootstrap_ready`. Clean-clone determinism holds
  (tracked-inputs-only content hash), and dirty sources are caught by
  construction.

### One required fix — the single false-fresh vector found

**`.dockerignore` is not in the inventory.** The image's `COPY src` content is
`src` *minus* `.dockerignore` exclusions, so editing `.dockerignore` changes
image content while the identity stays constant — a stale image would be
reused as fresh, which is exactly the condition this feature exists to
prevent. Fix: add `.dockerignore` to `PUBLIC_RUNTIME_SOURCE_FILES` (one line;
identity shifts once, forcing a one-time rebuild everywhere — acceptable), and
extend the inventory test to include it.

### Nonblocking robustness notes

1. The tree walk excludes only `__pycache__`/`.pyc`, while `.dockerignore`
   also excludes `*.egg-info`: a setuptools-style editable install creating
   `src/*.egg-info` would make the checkout permanently "stale" (spurious
   rebuild loop — safe direction, but a bad experience). None exists in this
   checkout today; mirroring the exclusion is cheap insurance. Similarly,
   editor temp files under `src` cause spurious staleness.
2. A consistency test asserting that the inventory's exclusion set mirrors
   `.dockerignore`'s src-relevant patterns would prevent future drift between
   the two.

### Verdict

**APPROVE, conditional on the one-line `.dockerignore` inventory addition and
its test.** The design achieves the stated goal — same-code-path identity,
label-pinned images, rebuild-online/fail-offline/audit-exact/up-rejects — and
the hashing is canonical and fail-closed. The one false-fresh vector found is
closed by a single line. D8 intact.

## Addendum 113a — split payload/recipe identity: re-review (2026-08-09)

Re-review of the stricter fix. 28 tests, Ruff, and `bash -n` on both scripts
reproduced green. Review only.

### Every A113 item closed, mostly in stronger form

- **`.dockerignore` is now a recipe input** — not merely added to a flat list
  but part of a distinct **recipe digest** (Dockerfiles + build script +
  `.dockerignore`), with the combined source identity domain-separated as
  `H("payload"‖payload ‖ "recipe"‖recipe)`. A recipe-only change is now
  distinguishable from payload drift via the three separate labels.
- **Both A113 nonblocking notes resolved**: `*.egg-info` (and pycache/pyc) are
  filtered in both inventories, and the mirror-the-patterns consistency
  question dissolves entirely — `.dockerignore` is *hashed*, so
  inventory-vs-ignore drift changes the identity by construction.
- **New hardenings beyond A113's asks**: symlinks rejected at both file and
  tree level (removing hash-vs-COPY content ambiguity), and **mode included in
  the hash** (a permission flip on the build scripts now changes identity —
  correct, since it changes image behavior).

### The audit now checks reality, not just labels

The canonical recipe snapshot is written to a `mktemp` dir and delivered as a
**named BuildKit context** — so it enters the image without passing through the
main build context and therefore cannot be affected by the very
`.dockerignore` under audit — retained at
`/opt/project/build-recipe/recipe-snapshot.json` and labeled. The audit then
goes beyond label comparison: inside a **network-none container** it re-runs
`public_runtime_payload_sha256("/opt/project")` against the image's *actual
copied bytes* and exact-compares the retained snapshot's SHA, writing
`copied-source-identity.json` into the audit output. A label can no longer
over-claim: the image's real content is re-hashed at audit time. Single code
path preserved throughout (build, audit-host, audit-in-container, CLI all call
the same module).

### Verdict

**APPROVE.** The A113 conditional fix landed in a strictly stronger split
design, both robustness notes are structural rather than mirrored, and the
audit's in-container payload recomputation closes the residual gap between
what an image claims and what it contains. Bootstrap source-freshness is
complete. D8 intact.

## Addendum 113b — nested-symlink traversal fix (2026-08-09)

Final delta re-review. 30 tests, Ruff, `bash -n` reproduced green. Review only.

- **The omission was real, and A113a over-credited the previous state**: with
  filtering preceding validation, a *nested* symlink was silently dropped by
  `_included_file` rather than rejected — while `COPY src` would have copied
  its target's content into the image. Content in the image, absent from the
  identity: a genuine false-fresh vector, and A113a's "symlinks rejected at
  both file and tree level" was only true at the tree roots and top-listed
  files. Recorded per the correction convention.
- **The fix orders the checks correctly**: in traversal, `is_symlink()` is
  tested **first** for every entry — before the directory skip and before the
  artifact filters — so any nested symlink (file or directory, even inside an
  otherwise-excluded path) raises; real directories are skipped; any other
  non-regular entry (fifo/socket/device) raises as a special entry. The
  parametrized full-tree test proves both nested file and directory symlinks
  fail closed with the exact message.

**APPROVE — bootstrap source-freshness is now genuinely complete**: the
inventory can no longer silently under-hash anything the build could copy.
D8 intact.

## Addendum 114 — the margin blocker adjudicated substantively; preregistered reverse-result conditions (2026-08-09)

Conditional review, written **before** the reverse artifact exists (verified
absent). Its purpose is to bind interpretation ahead of data. Review only.

### 1. Is the ≥0.1 margin hard-block substantively justified for attempt 4?

**Split verdict, case by case — and the root of the tension is a
written-vs-executable divergence.** The checked-in policy
(`asr-judge-swap-policy.md` §6) prescribes **hold** semantics: "held for
explicit review and cannot be cited as an *unattended* green." The comparator
implements a **hard block** — recorded by A101 as an intentional strictness
upgrade, and A107 correctly ruled the executable form binds for attempt 4.
Within that form:

- **Case 1 (identical text, typed 0.125 vs speech 0.5): substantively
  justified.** The judge-swap calibration empirically established
  judge-dependent WER variance of ~0.1–0.35 near threshold on this content
  class, and an integrity pass at margin 0.0 could flip under an equally valid
  judge — invalidating the arm and with it the evidence. More importantly, the
  block *surfaced a real unexplained cross-arm divergence* — the exact question
  the reverse diagnostic now exists to answer. The rule did its job.
- **Case 3 (both arms 0.5 on the two-word reference `' Current UTC'`):
  substantively miscalibrated.** On a 2-word reference the WER measure is
  quantized to {0, 0.5, 1.0}: *every* single-garble outcome lands exactly at
  threshold, so a 0.1 "margin" is conceptually vacuous — the rule blocks on
  quantization, not proximity risk. This is a calibration defect in the rule
  as applied to degenerate references, not evidence of anything about the
  audio.

Net: the blocker is procedurally binding (A107 stands), substantively earning
its keep on one of the two cases, and demonstrably miscalibrated on the other.

### 2. Preregistered decision conditions for the reverse result

To be applied mechanically when the diagnostic artifact lands, using its
preregistered labels and flags:

- **Condition A — case-1 class labels `consistent_with_second_position`**: the
  divergence is session/order variance; no typed-vs-speech product difference
  exists. Path: prospective policy amendment (§3), then **one fresh forward
  run** under the amended policy as the promotable artifact (the reverse
  artifact is non-promotable by construction and stays that way; attempt 4
  stays red).
- **Condition B — labels `consistent_with_speech_modality`** (speech arm
  degraded in both orders): a real modality-conditioned response-audio
  difference exists — and note the direction: it **disfavors speech**, so the
  typed ≥ speech parity criterion remains satisfied. Path: promote typed with
  the speech-path audio-fidelity finding filed as a separate product defect;
  the margin rule protected evidence robustness, not the parity direction.
- **Condition C — `tie`/`unstable`**: variance unattributable at n=2; per the
  agreed cap, **no further replications**. Path: document the margin cases as
  judge-noise-on-degenerate-content, prospective amendment (§3), owner
  decision memo on attended evidence.
- **Qualifier for all conditions**: if `model_text_identical_to_forward` is
  false anywhere, the WER cells are not like-for-like (cross-image decode
  drift explains WER movement by itself); treat as Condition C with that
  drift recorded as the explanation.

### 3. Smallest prospective policy change (if A or C) — the A90 pattern

Introduce a versioned **parity margin policy v2**, prospective only, both
verdicts computed and recorded (the `strict_component_v1`/
`decision_concordance_v2` dual-record precedent):

1. **Hard block retained** exactly where A89's preregistration aimed: when the
   sub-margin WER is **load-bearing for a positive semantic verdict** (the
   green-path scenario the rule was written for).
2. **Hold-with-disclosure (non-blocking)** when the semantic outcome is a
   shared failure — margin risk then affects only evidence robustness, and the
   artifact carries `held_for_review: true` with the written policy's
   no-unattended-green force.
3. **Degenerate references** (fewer than 4 normalized words) are always
   held-and-disclosed, never blocking — the margin concept does not exist at
   that quantization.

Constraints: attempt 4's comparison artifact is preserved red, hash-pinned
(already quadruple-pinned by the reverse analyzer), and **cannot be
re-adjudicated under v2**; any promotable artifact comes from a new forward run
under the amended executable policy; the policy doc and comparator change
together so the written and executable forms finally agree.

### Verdict

**Conditional review complete — waiting on the reverse artifact.** The margin
blocker stands for attempt 4 (A107 binding), is substantively vindicated on
case 1 and miscalibrated on case 3, and the reverse result now has a
mechanical, preregistered interpretation with the smallest honest policy
remedy specified in advance. No source edits. D8 intact.

## Addendum 115 — reverse diagnostic adjudicated: Condition C; policy v2 specified (2026-08-10)

Mechanical application of A114 to the completed diagnostic
(`tool-freshness-reverse-20260810T0009Z/diagnostic.json`), independently
verified by this review: self-hash reproduces the stated value, the
non-promotable invariants hold (`passed: false`, `promotion_eligible: false`,
`diagnostic_complete: true`), fixture-pair identity passed (byte-identical
Pocket carriers across experiments), the image confound is flagged, and —
decisively for comparability — **all sixteen cells across the four cases carry
identical model text** (`model_text_identical_to_forward` true everywhere), so
every WER comparison is like-for-like and the A114 decode-drift qualifier does
not fire.

### 1. Mechanical outcome: Condition C

Labels: case 1 `unstable` (0.125/0.5/0.25/0.375 — no equality pattern),
case 2 `unstable`, cases 3 and 4 `tie` (all-0.5 and all-0.0). Per A114:
**variance unattributable at n=2, no further replications**, margin cases
documented as judge-noise, prospective amendment per A114 §3, owner decision
memo available on attended evidence. Case 3's all-cell 0.5 is the quantization
prediction of A114 §1 confirmed exactly.

### 2. Descriptive observations (recorded as such — not re-labeling)

The preregistered equality classifier is blind to ordinal patterns by design;
these are observations, not labels:

- **Second position is worse in both experiments** (case 1: first-position
  WERs 0.125/0.25 vs second-position 0.5/0.375) — directionally a
  session/order effect, echoing Condition A without meeting its equality bar.
- **Typed is never behind speech, position-for-position, in any cell of any
  case** (first: 0.125 < 0.25; second: 0.375 < 0.5). The Condition-B
  hypothesis (modality-conditioned speech degradation) is directionally
  disfavored — and even if present it disfavors *speech*. **The typed ≥ speech
  parity criterion is unthreatened in every cell of the entire 2×2×4 grid.**
- The non-gating envelopes corroborate render variance: the 0.5-WER
  speech-forward audio has the longest trailing tail (8 frames vs 2–5), with
  identical first-active frames everywhere; the degenerate cases are
  envelope-identical across all four cells.

### 3. Exact prospective policy implementation (`parity_margin_v2`)

Per A114 §3, to land as one change to the comparator **and**
`asr-judge-swap-policy.md` together (the written and executable forms finally
agreeing), dual-verdict recorded per the A90 pattern:

1. **Hard block** only when a sub-0.1 margin coincides with a **positive
   semantic pass** on that case (WER load-bearing for a green — A89's actual
   target).
2. **Held-with-disclosure, non-blocking** when the case's semantic outcome is
   a shared failure: `held_for_review: true` entries in a `held_cases` list.
3. **Degenerate references** (< 4 normalized words) always held, never
   blocking — the margin concept does not exist at ½-word quantization.
4. Both v1 and v2 verdicts computed and recorded in every artifact; the
   `policy` field adjudicates; attempt 4 remains red under v1, hash-pinned,
   never re-adjudicated; the reverse diagnostic remains non-promotable.
5. Tests: each v2 branch red/green (load-bearing thin margin blocks;
   shared-failure thin margin holds and passes; degenerate-reference hold),
   dual-verdict recording, the attempt-4 frozen lock unchanged, and a
   doc↔code consistency assertion.

### 4. Is one fresh forward run required?

**Not required** — Condition C's preregistered path is amendment plus owner
memo on attended evidence. **Recommended nonetheless**: after v2 lands, one
forward run with the proven instrument is expected to produce an
**unattended-green** promotable artifact (the model's code omission makes
shared-failure semantics near-certain, so v2's only hard-block branch — a
thin-margin *positive* pass — is essentially unreachable), eliminating the
attended-judgment surface entirely for the cost of one GPU run. Fallback if it
margins in some new way: the attended memo, which Condition C already
authorizes.

### Verdict

**Condition C applied; the diagnostic is closed with no further replications.**
The evidence adds a parity-favorable fact the forward run alone could not
show: typed input is never behind speech in any position in any case. Land
`parity_margin_v2` exactly as §3 specifies, then either run the one
recommended forward run for an unattended green or issue the owner memo on
attended evidence — both paths are now preregistered, honest, and short. D8
intact.

## Addendum 116 — `parity_margin_v2` implementation review (2026-08-10)

Adversarial review against A114/A115 and the self-review's narrowed contract.
102 focused tests + Ruff reproduced green. Review only.

### 1. The load-bearing invariant holds: frozen v1 is untouched

`compare()` is byte-unmodified — `compare_v2` calls it, deep-copies its
artifact, and only then augments — and the reverse analyzer's
`validate_frozen_forward` still regenerates attempt 4 through `compare()`
directly, so the quadruple-pinned frozen baseline rederives byte-identically.
Every v2 artifact embeds the exact v1 verdict verbatim under
`verdicts.strict_margin_v1` (the A90 dual-record pattern), and the CLI defaults
prospectively to v2 with an explicit `--policy` selector for v1.

### 2. Two deliberate deltas from A115 §3 — both endorsed as strictness upgrades

1. **The hold contract is narrower than specified.** A115 required only
   shared-failure semantics; the implementation requires **exact-equal paired
   model text AND shared-negative on both semantic layers AND both arms'
   absolute integrity passed** — anything else (positive or discordant
   semantics, unequal text, failed integrity) keeps the hard block. This is
   more principled than my spec: with unequal texts the WER cells are not
   comparable, so the judge-noise dismissal would not be sound there.
2. **Holds cannot unattended-pass.** A115 §3.2 said "held-with-disclosure,
   non-blocking"; the implementation sets
   `passed == unattended_promotion_eligible == (core_passed and no holds)`,
   with `core_passed` and `attended_review_required` explicit and the policy
   doc stating "Only a separate content-addressed adjudication may promote it;
   there is no override flag." This is the written §6 force ("cannot be cited
   as an unattended green") finally made executable — the written and
   executable policies now agree, which was the root tension A114 §1
   identified. Consequence for A115 §4's forecast, restated honestly: a fresh
   forward run yields an unattended green **only if every margin clears 0.1**;
   a recurrence of case-1-class variance yields `core_passed` +
   `attended_review_required` — the attended path, now formalized in-artifact
   rather than ad hoc.

### 3. The rest of the checklist

**Degenerate is subtype only**: `degenerate_reference` (normalized reference
≤ 2 words — narrower than A115's <4, and since it waives nothing the
difference is cosmetic) is recorded metadata on the hold/block record and
"never independently waives a margin" (doc item 5, with a dedicated
not-a-waiver test). **Docs hashes exact**: the policy doc's v2 section matches
the executable contract point-for-point, pins the reverse diagnostic's file
SHA (`6cc39397…`) and self-hash (`ccbbf4df…` — the value this review verified
in A115), declares it permanently non-promotable, and requires a **new
typed-then-speech run** for prospective v2 evidence with the historical v1
comparison remaining red. **Tests**: boundary exactness at 0.1, hard-block
retention for non-shared/unequal cases, degenerate-not-a-waiver, dual-verdict
plus attended-review recording, and — the right capstone — a
doc↔code consistency test binding the written contract to the executable one.

### Verdict

**APPROVE.** The v2 layer adds no weakening anywhere: every absolute gate is
untouched, the v1 verdict is preserved and embedded, holds require attended
adjudication by construction with no override flag, and the narrowed
eligibility contract closes the non-comparable-text seam my A115 spec left
open. The written policy, the executable policy, and the review record now say
the same thing. Remaining: one fresh typed-then-speech forward run under v2 —
unattended green if margins clear, formalized attended review if they don't —
and the close-out memo. D8 intact.

## Addendum 117 — prospective-enforcement marker for `parity_margin_v2` (2026-08-10)

Re-review of the post-A116 follow-up closing the self-review's BLOCK. All
claims verified by reading the executable contract end-to-end and rerunning
the focused suites (129 tests across
`test_compare_tool_freshness_modalities` / `test_tool_freshness_parity` /
`test_analyze_tool_freshness_reverse` / `test_tool_freshness_fixture` /
`test_transcribe_sustained` / live-suite parity subset / `test_compare_asr_judges`)
plus Ruff — all green. Review only.

### 1. The BLOCK was correct — this delta closes a real laundering hole

The A116-approved `compare_v2` would have accepted **any** historical runtime
report, including attempt 4's. Running it over the retained attempt-4 reports
would have produced `core_passed` + `attended_review_required` — a retroactive
re-adjudication of a preserved-red artifact, which is exactly the A90
anti-laundering violation this whole policy discipline exists to prevent. I
approved A116 without catching that; the self-review's BLOCK was right and
this follow-up is the correct fix, not gold-plating.

### 2. The enforcement chain, verified link by link

**Marker origin**: `PARITY_MARGIN_POLICY = "parity_margin_v2"` lives in
`tool_freshness_contract.py` (a bare constant — stdlib import purity for the
evaluator container is preserved), and both the probe and the comparator
import it, so the marker cannot skew by typo. **Emission**: the forward
runner writes `comparison_policy` into the retained runtime report; the
reverse path pops it and stamps `promotion_eligible: false` — a reverse run
can never carry the marker (asserted both ways in
`test_runtime_marks_forward_policy_and_reverse_nonpromotion`). The probe's
error-path report is unmarked and `passed: false`, so it fails the marker
check before anything else. **Rejection**: `compare_v2` refuses any runtime
report whose `comparison_policy` is not exactly the marker, before invoking
`compare()`. **v1 indifference**: `validate_runtime_report` checks named
fields without a closed key-set, so a marked report flows through `compare()`
untouched and — the load-bearing direction — the **unmarked frozen attempt-4
report still validates identically**, so `validate_frozen_forward`'s
byte-exact regeneration is unaffected (`analyze_tool_freshness_reverse.py`
has zero marker references; its 21 tests pass unchanged). Both directions are
pinned by `test_parity_margin_v2_rejects_unmarked_historical_runtime_but_v1_accepts_it`.

Can attempt 4 be laundered by hand-marking its retained runtime report? No:
that edit changes the file's SHA-256, which the frozen comparison
(`4889acc9…`, self-hash `4916ec73…`) pins under `runtime_report.sha256`, so
the doctored report no longer rederives against any retained evidence. The
only way to a v2 artifact is a genuinely new forward run — which is what the
doc's item 7 and closing sentence now say. A hand-marked **reverse** report
is independently dead on three locks before the marker even matters: wrong
`kind`, wrong `execution_order`, and forward-mode rejection of
`promotion_eligible` presence.

### 3. Orchestration and tests

The live suite invokes the comparator with an explicit `--policy
parity_margin_v2` (no reliance on the CLI default), and the suite's top-level
`passed` — in both `state.json` and the suite report — is
`comparison["passed"]`, which under v2 equals
`unattended_promotion_eligible`. A held run is therefore **red at every
level** (comparator exit 1, stage record, suite verdict) while the embedded
artifact discloses `core_passed`/`attended_review_required` for the attended
path. Test coverage now spans the full disposition lattice: exact 0.1
boundary (0.1 clear / 0.0999 hold), hard-block retention for non-shared,
discordant, and unequal-text cases, thin-**positive** margin remains a hard
block, failed absolute integrity refuses outright, degenerate-not-a-waiver,
clear-margin unattended green, held-run dual-verdict recording, CLI
default-v2/explicit-v1, and the doc↔code consistency test now binding all
four evidence hashes (forward `4889acc9…`/`4916ec73…`, reverse
`6cc39397…`/`ccbbf4df…`) plus the marker contract (doc item 7).

### 4. Residual notes — none blocking

(a) `compare_v2` reads the runtime report twice (marker check, then
`compare()`'s validation). A file swap between the reads could in principle
yield a v2 artifact pinning an unmarked report — but any recompute of that
artifact fails the marker check, so the laundering is self-defeating;
fail-closed on rederivation. Nit, not a fix request. (b) v2's
`blocking_reasons` appends re-added margin blocks after the non-margin
reasons rather than interleaving per-case as v1 does — deterministic,
cosmetic. (c) The reverse analyzer does not explicitly reject a hand-marked
reverse report; the three independent locks above make that path moot.

### Verdict

**APPROVE.** The retroactivity hole my A116 approval missed is now closed
with a marker that is shared-constant at origin, forward-only at emission,
mandatory at v2 comparison, invisible to frozen v1, and unforgeable against
the content-addressed evidence chain. The written contract (doc item 7), the
executable contract, and the tests state the same rule. Remaining work is
unchanged from A116: one fresh typed-then-speech forward run under v2, then
the close-out memo. D8 intact.

## Addendum 118 — canonical scenario v2 fixture redesign: adversarial design review (2026-08-10)

An earlier draft of this addendum reviewed a recall-fallback integrity gate
for short references. The owner withdrew that proposal before implementation
in favor of the fixture redesign reviewed here; the recall fallback is
**abandoned entirely** — no relaxed integrity basis will exist, so the
laundering surface that design would have created never comes into being.
The fresh prospective v2 run remains red: case 3, both arms, absolute
`WER <= 0.5`, exact-equal model text "Current UTC", both responses
speech-classified and byte-identical in length (81,144 bytes ≈ 1.84 s), judge
transcripts "Current I've ste." (WER 1.5) and "Current ox di." (WER 1.0),
all eight structural cycles and the other six integrity cases green. This is
a pre-implementation design review of the proposed **canonical scenario v2**;
review only.

### 1. Fixing the stimulus instead of weakening the meter — endorsed

The redesign is the more principled resolution of the same diagnosis. The
failure arithmetic rederives exactly ("Current I've ste." normalizes to
`[current, i, ve, ste]` = 3 errors over a 2-token reference = 1.5), the
missed token is the acronym "UTC" which the pinned 0.6B judge has never once
transcribed across attempt 4, the reverse diagnostic, and this run, and the
judge's trailing garble *differs* between arms on byte-identical-length audio
— judge noise, not a channel defect. The recall fallback answered this by
weakening the gate for the unmeasurable case; scenario v2 answers it by
removing the unmeasurable utterance class from the instrument. Every gate —
absolute `WER <= 0.5`, the heard-gate, `parity_margin_v2`, the A117 marker
chain — stays byte-unchanged. A scenario whose expected outputs the pinned
judge demonstrably cannot transcribe is an instrument defect, and under the
promotion criterion (typed↔speech response parity, not absolute task
competence) the scenario's only job is to elicit comparable, measurable
spoken outputs in both arms. Redesigning it prospectively, hash-versioned,
with all reds preserved, is legitimate instrument repair — with the
anti-shopping constraint of §6.

### 2. The load-bearing string is the tool's `spoken` payload

The decisive fact for this redesign sits in `tool_result`: the current
spoken payload template is `"{Code}. Current UTC time is {now}."` — and the
model's collapsed output "Current UTC" is a truncated **echo of the tool's
own spoken payload with the code-first prefix dropped**, exactly matching
the r6 logit evidence (template opener " Current" at raw argmax, codes
absent from top-20). The model doesn't invent its degenerate response; it
echoes the continuation we taught it. Therefore the single most influential
design surface in scenario v2 is the spoken payload template: the most
probable collapse mode of the next run is *that template minus the code*,
so the template minus the code must itself be a long, ordinary-English,
judge-transcribable sentence. "Cobalt. The first verification result is
ready." satisfies this for the dominant collapse (echo → "The first
verification result is ready.", six common normalized tokens, WER budget of
three errors — the observed 1–2-token trailing-garble behavior is absorbed
with margin to spare). Removing UTC/time removes acronyms *and* digits from
both the response references and the input prompts, which also strengthens
the speech arm's heard-gate (input WER ≤ 0.35) — the current prompts say
"UTC" four times into the same acronym-deaf judge.

### 3. Direct answer: can names/prompts still cause short common-prefix truncation? Yes — reduced, not eliminated

Honest analysis of the residual: nothing in wording can *guarantee*
non-degenerate model output. The observed truncation stopped two tokens into
the template; the v2 analog would be "The first" — still a 2-token
reference. What changes is the failure economics, not the possibility: (a)
the dominant hazard is removed — "the"/"first" are high-frequency words the
judge transcribes, where "UTC" was never transcribed once; (b) the trailing
hallucination hazard **remains** — at 0.5 WER per inserted token on a
2-token reference, the observed 1–2-token garble still reds a 2-token
truncation even with perfect recall of both words. Accepted consequence: if
the model truncates that hard, the run goes red, stays red, and there is no
fallback — that is the strictness the owner chose by abandoning the recall
gate, and it must be stated in the policy doc.

One binding amendment to the proposed template: **"Cobalt." must not be a
standalone sentence.** The TTS path aggregates and emits at sentence
boundaries, and EOS after the first sentence is a natural stopping point —
a standalone code sentence creates a 1-token-reference truncation cliff
("Cobalt." alone), the worst possible ASR target. Fold the code into the
first sentence with the code still first, e.g. "Cobalt is the first
verification result, ready now." Then every sentence-boundary truncation
yields either the empty response (nonempty conjunct reds it, correctly
disclosed) or a ≥ 6-token sentence, and the code-first property — payload
survives any suffix truncation — is retained.

### 4. Exact scenario v2 contract (binding)

1. **New versioned constants alongside v1, never replacing it**: v2 prompts,
   codes, per-cycle result labels, `tool_definition_v2` (generic fresh
   verification — name and description free of acronyms, digits, and
   time/UTC vocabulary), `tool_result_v2`, and `scenario_document_v2()`
   producing one canonical `scenario_sha256_v2`. The v1 functions remain
   byte-stable: the frozen attempt-4 comparison and the reverse diagnostic
   rederive through v1 scenario derivation and must continue to regenerate
   exactly (regression tests required). Version selection everywhere is by
   **exact scenario-hash match against the known set, fail-closed on
   unknown** — never by a mode flag that could be pointed at old evidence.
2. **Freshness semantics preserved**: per-cycle distinct results (ordinal
   words first/second/third/fourth replacing distinct times) and per-cycle
   distinct codes; `expected_response_any` remains the case's own code;
   `expected_response_none` retains prior codes so stale-tool-result reuse
   stays detectable in both text and audio channels. The structural
   conjuncts rederive from the v2 scenario with a **new drift-locked v2
   conjunct pin**, the v1 pin untouched.
3. **Measurability lint — a structural test over `scenario_document_v2()`
   itself**, so the properties are executable, not aspirational: every
   spoken payload is a single sentence of ≥ 6 normalized tokens with the
   code as its first token and no standalone code sentence; no acronyms
   (no normalized token that is an initialism), no digits, no time
   expressions anywhere in prompts, instructions, tool description, or
   spoken payloads; every prompt ≥ 6 normalized tokens (heard-gate
   friendliness); codes are ordinary dictionary words (the existing four
   qualify).
4. **Fixture chain**: new spoken-prompt PCM fixtures synthesized for the v2
   prompts under the existing seed scheme and voice, manifest binding
   `scenario_sha256_v2`, full provenance (`FIXTURE_PROJECT_SOURCE_PATHS`
   mechanism unchanged); a v1-fixture/v2-scenario mix must refuse at
   manifest validation. `experiment_sha256 = f(scenario_sha256,
   fixture_manifest_sha256)` changes automatically, so v2 evidence can never
   bind to v1 evidence — identity mismatch refuses in `validate_arm`,
   `validate_runtime_report`, and the recompute chain without any new gate.
5. **Gates untouched**: absolute `WER <= 0.5`, the heard-gate,
   `parity_margin_v2`, and the A117 `comparison_policy` marker enforcement
   are all byte-unchanged. The only evaluator/comparator delta permitted is
   version-aware scenario rederivation per item 1. The legacy UTC fixture
   and every retained red artifact are unchanged and never re-adjudicated.

### 5. Required version/hash/provenance tests

(1) `scenario_sha256_v2` pinned in the policy doc with a doc↔code
consistency test; (2) v2 conjunct drift-lock pin, v1 pin unchanged; (3)
frozen regeneration: attempt-4 comparison and reverse diagnostic rederive
byte-identically after the v2 code lands; (4) cross-version kill tests: v1
arm evidence against v2 expectations refuses on scenario hash, and vice
versa; unknown scenario hash fails closed; (5) manifest kill test: v2
manifest with v1 prompts (or v1 scenario hash) refuses; (6) measurability
lint (§4.3) with mutation kill tests — an acronym token, a digit, a
standalone code sentence, or a 5-token payload each turn it red; (7)
end-to-end mocked parity pass under v2 scenario through `compare_v2`,
confirming the marker chain and margin policy operate unmodified; (8) the
fresh red run's artifacts (both arm ASR reports and the comparison error
artifact) hash-pinned red in the policy doc, added to the doc test.

### 6. Anti-scenario-shopping constraint

Redesigning the stimulus after a red run is exactly the shape laundering
takes when repeated, even though this single instance is instrument-justified
(the acronym-deafness evidence predates this run and was preregistered in the
reverse diagnostic's case-3 finding). Binding process terms: scenario v2 is
preregistered — hash pinned in the doc — **before** the next run; it is the
**last** elicitation change before close-out; any further wording change
requires a version bump plus a disclosed instrument rationale that does not
reference the outcome it produced; and a red v2 run is adjudicated on its
merits or taken to the attended close-out memo, never answered with a v3
wording tweak.

### Verdict

**APPROVE (as amended).** Abandoning the recall fallback in favor of the
fixture redesign is the right call: it keeps every gate at full strength and
repairs the instrument where it is actually broken — an acronym benchmark
the pinned judge has never transcribed. The amendments are binding: the code
must not be a standalone sentence (1-token truncation cliff), the
measurability lint of §4.3 must be an executable structural test, version
selection must be exact-hash fail-closed with v1 byte-stable and
frozen-rederivation regression-tested, and §3's residual must appear in the
policy doc: short-prefix truncation remains possible, and if it recurs the
run goes red with no fallback. The fresh red run's artifacts are pinned red;
scenario v2 evidence requires a fresh marked run under the unchanged A117
chain. D8 intact.

## Addendum 119 — scenario-v2 redesign implementation review (2026-08-10)

Adversarial review of the landed implementation against the A118 amended
contract. Every claim below was verified by direct inspection or by
execution in this review. Review only.

### 1. Version identity — recomputed, and faithful to real history

Both pins recompute exactly in this session:
`canonical_sha256(scenario_document_v1())` equals the archival
`SCENARIO_SHA256_V1` (`77efc924…`), `canonical_sha256(scenario_document_v2())`
equals the pinned `SCENARIO_SHA256_V2` (`2e958573…`), and — the check that
matters most — the **retained attempt-4 comparison's embedded
`scenario_sha256` equals `SCENARIO_SHA256_V1`**, so the v1 reconstruction is
faithful to real history, not merely self-consistent. `scenario_document()`
returns v2 and self-checks its canonical hash on every call;
`scenario_document_for_sha256`, `fixture_contract_for_scenario_sha256`, and
`structural_conjuncts_for_scenario` all resolve by exact hash and raise on
unknown hashes — fail-closed version selection exactly as A118 §4.1 required.

### 2. Frozen rederivation — proven by execution, not by test double

I ran the reverse analyzer's `--preflight-only` against the real retained
attempt-4 artifacts under the landed v2 code:
`diagnostic_complete: true` with `passed`/`promotion_eligible` false by
construction. That preflight regenerates the frozen comparison through
`compare()` — full WAV/scenario/conjunct recompute, with the new
version-aware resolver selecting v1 by archival hash — and requires exact
equality against the quadruple pins (`4889acc9…`/`4916ec73…` comparison,
`107bcf50…`/`a6f37db9…` fixture). The new v1-path `tool_calls` conjunct term
(`expected_response_none == excluded_responses`) evaluates `[] == []` on v1
evidence, and the exact regeneration proves it is behavior-preserving.

### 3. Measurability lint and the A118 amendments — all landed

`validate_measurable_scenario_v2` is executable and runs **inside**
`scenario_document_v2()`, so no unlinted v2 scenario can exist: digits and
acronyms rejected, time vocabulary rejected, prompts ≥ 6 words, spoken
outputs exactly one sentence of ≥ 6 normalized words with the verification
word **first** — and the kill-test matrix includes the exact 1-token-cliff
case from A118 (a standalone "River." two-sentence payload reds the lint),
plus UTC/digit/time-vocab prompts, a 4-word payload, and an extra
function-output field. The v2 `function_output` is `{"spoken"}` only — the
`timezone` field and every acronym/digit are gone from the injected result.
Stale-result exclusions are enforced at three layers: pinned in the lint
(`excluded_responses == prior words`), bound per-case in the `tool_calls`
structural conjunct, and flowed into `expected_response_none` for the
negative ASR semantic. The chosen words (river, candle, garden, winter) are
common nouns — a better instrument than the proposal's "cobalt", with the
disclosed residual that common words carry a marginally higher base rate of
coincidental appearance in unrelated speech; they remain implausible in this
scenario's response space and any exclusion hit stays fail-closed red.

### 4. No metric relaxation; mixed UTC isolated; provenance intact

`response_asr_evaluation` is byte-unchanged (no recall path exists);
`MAX_WER = 0.5`, `MIN_WER_MARGIN = 0.1`, `MAX_INPUT_WER = 0.35`, the A117
marker chain, and `parity_margin_v2` are all untouched. The sustained
probe's mixed mode still resolves `tool_definition_v1`/`tool_result_v1`
(UTC clock tool, time results) while tool-only mode resolves v2 — the legacy
UTC corpus survives archival-only for frozen rederivation and mixed-mode
use, exactly the "mixed legacy UTC fixture unchanged" carve-out. The fixture
generator emits schema-2 `tool_freshness_ordinary_english_pocket_fixture_v2`
manifests from the active scenario; `FIXTURE_PROJECT_SOURCE_PATHS` still
pins both contract and fixture modules; the archival-v1-validates /
cross-version-mix-refuses test covers the manifest seam in both directions.

### 5. Docs and pins

The policy doc pins the fresh red run completely (top report `1a9078fd…`,
comparison `e2920cb7…` with self-hash `41e6bd09…`, typed ASR `4e370927…`,
speech ASR `9714deac…`), states it is not re-adjudicated and no threshold is
relaxed, pins `scenario_sha256_v2`, scopes the corpus rule to spoken
natural language (protocol identifiers exempt), states the §3-of-A118
residual verbatim in substance ("a severe short-prefix truncation can still
make a future run red; there is no short-reference fallback"), and closes
with the anti-scenario-shopping clause ("last elicitation change for
close-out… new version and an independent, pre-result instrument
rationale").

### 6. Test evidence (rerun in this review)

92 (fixture/parity/comparator/reverse-analyzer) + 12 (sustained
qualification) + 17 (transcribe) + 61 (full live suite) + 35 (function
calling + websocket endpoint) — all green, Ruff clean, plus the real
attempt-4 preflight rederivation above. (My partition totals differ from the
submitted 104+24+48 grouping labels; every suite I ran is green.)

### Non-blocking notes

(a) `STRUCTURAL_CONJUNCTS_V1` and `_V2` are currently identical tuples — the
versioned split is vacuous today but correctly plumbed for future
divergence; the unversioned `STRUCTURAL_CONJUNCTS` alias points at v2 for
import compatibility. (b) The lint's time-vocabulary set omits the bare word
"time"; the canonical v2 corpus doesn't use it, and the lint is a tripwire
over code-controlled strings, not an open-input filter.

### Verdict

**APPROVE — no blockers.** Every A118 binding amendment is implemented: the
code word is embedded first in a single ≥ 6-word sentence (no standalone-code
cliff, with the exact kill test), the lint is executable and
construction-enforced, version selection is exact-hash fail-closed with v1
archival and proven byte-faithful to the real attempt-4 evidence, no gate or
threshold moved anywhere, the red runs are hash-pinned and never
re-adjudicated, and the anti-shopping clause is in the doc. The remaining
step is unchanged: one fresh marked typed-then-speech run under scenario v2
and the A117 chain — unattended green if all margins clear, formalized
attended review if they don't — then the close-out memo. D8 intact.

## Addendum 119b — time-vocabulary lint closure; stale note corrected (2026-08-10)

Correction to A119 non-blocking note (b): that note was stale when written.
The lint's time-vocabulary set now includes "time" and "times"
(`tool_freshness_contract.py`), with the exact prompt-mutation kill test
("a fresh time result…") in the lint matrix — and the timeline shows the fix
landed in the shared tree between this review's initial contract read and
its test run, which is why that kill test was already green in A119's 92-test
evidence. The self-review was right to treat the omission as a fail-closed
gap rather than a tripwire nicety; the closure is verified here directly:
`_reject_unmeasurable_utterance("a fresh time result")` raises, 122 focused
tests pass, Ruff is clean.

The load-bearing confirmation: `scenario_document()` still self-checks to the
pinned `SCENARIO_SHA256_V2` (`2e958573…`). The lint change altered the
validator, not the corpus — no scenario content moved, so this is **not** an
elicitation change and does not trip the doc's last-change/anti-shopping
clause or require a version bump.

**Final approval confirmed on the current tree.** A119's verdict stands with
note (b) withdrawn; the sole remaining step is unchanged: one fresh marked
typed-then-speech run under scenario v2, then the close-out memo. D8 intact.

## Addendum 120 — scenario-v2 forward run: adjudication of the promotion evidence (2026-08-10)

Adversarial review of
`~/.cache/nemotron-voicechat/traces/tool-freshness-parity-v2-ordinary-english-20260810T014525Z`.
Nothing below is taken from the artifact's own claims; every gate was
rederived in this review. Read-only.

### 1. Independent rederivation — byte-exact

Running `compare_v2` in this session over the retained arm reports and
runtime report regenerated the comparison **byte-identically** (self-hash
`7f7bc5cd…` verified against recomputed canonical hash), which transitively
re-executes the whole chain: retained-WAV PCM equality, canonical 16 kHz
resample identity, ASR verdict recomputation, scenario-v2 rederivation
against the pinned `2e958573…`, the 17 structural conjuncts per arm, cycle
cardinality, the A117 marker check, and session binding. Running frozen
`compare()` over the same inputs also passes with zero blocking reasons and
matches the embedded `strict_margin_v1` verdict exactly.

### 2. The result

**Unattended green under both policies.** `passed = core_passed =
unattended_promotion_eligible = true`, `review_holds = []`,
`margin_dispositions = []`, `attended_review_required = false` — and the run
never touches v2's hold machinery, because **all eight response-WER cells
are 0.0000** with every margin at the theoretical maximum 0.5, and all four
speech-input WERs are 0.0 (the UTC-free prompts fixed the heard-gate side
as well). Execution order typed-then-speech, distinct sessions
(`f67d8202…`/`8b8dd464…`), runtime report prospectively marked
`comparison_policy: parity_margin_v2` with `promotion_eligible` absent.

Per case, typed and speech model text is **byte-identical including leading
whitespace**, and per-case audio byte counts are identical across arms
(134064/95256/81144/74088 — deterministic TTS of identical text):

- case 1: "The fresh verification word is River." — **shared positive in
  both channels, both arms**; judge transcript exact. This is the first
  artifact in this entire record in which the model vocalized a verification
  payload and the judge heard it.
- cases 2–4: " The new verification" / " The verification" / " The fresh" —
  the model-level template collapse recurs with the familiar per-cycle
  shortening signature (6→3→2→2 normalized tokens), shared negative in both
  channels, stale-word exclusions correctly accumulated
  (`[] / [river] / [river,candle] / [river,candle,garden]`) and clean.

### 3. The contamination question — answered decisively

Cases 3–4 are **2-token references** — the exact A118 §3 residual class that
redded the previous run — and the judge transcribed them **perfectly**
(including case 3, where the judge's appended period normalizes away). Same
judge, same thresholds, same TTS voice: WER moved from 1.0/1.5 on "Current
UTC" to 0.0000 on equally short ordinary-English fragments. That isolates
the v1 failure to acronym contamination of the benchmark, not reference
length and not an audio-channel defect — the strongest form of evidence,
because the vulnerable case class recurred and measured clean. Honest
disclosure of the luck component: a single judge-inserted token on case 3
or 4 would have yielded margin 0.0 and a formalized review hold; the gates
were not adjusted to prevent that, and the result stands as measured. No
metric moved: `MAX_WER 0.5`, `MIN_WER_MARGIN 0.1`, heard-gate 0.35, and the
green v1 verdict proves no v2 leniency was consumed.

### 4. Parity — the A82 criterion, met exactly

Zero typed-only failures in either semantic layer in any case; every
semantic outcome identical across arms; text byte-equal; audio byte-lengths
equal. Typed input is not merely no-worse than speech — on this evidence the
two modalities are indistinguishable at every measured layer.

### 5. Provenance chain

Verified in this review: promoted evaluator image `sha256:59854214…`;
runtime image `sha256:246ec4f4…` consistent across manifest, orchestration,
and both arms; fixture manifest self-hash `18dc1cab…` pinned in the top
report, full fixture validation (schema-2
`tool_freshness_ordinary_english_pocket_fixture_v2`, all four PCM/WAV
carriers) re-run here; **fixture source provenance re-validated against the
current working tree**; experiment identity consistent across all three
reports; orchestration state green and pinning the comparison file hash and
runtime image. Evidence file hashes for the record: top report `62269861…`,
comparison `c79d7cc5…` (self-hash `7f7bc5cd…`), runtime report `b641b654…`,
typed ASR report `3873a8bc…`, speech ASR report `cbe6cf53…`, fixture
manifest file `80e3d717…`.

### Verdict

**APPROVE — promotion and deployment.** The preregistered chain closes with
nothing left open: A82's parity criterion is met exactly (no typed-only
failure anywhere), A115 Condition C's close-out requirement is satisfied by
an unattended-green artifact — green under the frozen v1 policy as well, so
no post-hoc leniency was needed — produced under the A117 prospective marker
and the A118/A119 preregistered scenario with every red predecessor
preserved and hash-pinned. The typed-input path is qualified for promotion
on response parity, and this artifact is the promotable evidence the
close-out memo should cite. The known, disclosed limitation travels with it:
model-level template collapse on repeated tool cycles persists in both
modalities equally (cycles 2–4 shared negatives), is a model property, not a
runtime or modality defect, and remains outside this work's promotion
criterion. D8 intact.

## Addendum 121 — final operational review for handoff (2026-08-10)

Read-only verification of the deployed stack, the live browser gate, the
bootstrap source contract, and the close-out documentation.

### 1. The e2e test proves what is claimed — by construction, not by luck

`test_live_playground_typed_turn_and_audio` was read in full. When green it
necessarily demonstrates all four claims: **voice input** — the Chromium
fake-capture device plays a real spoken "remember sapphire" WAV into the
actual WebRTC path and the test requires a `user-transcription` containing
"sapphire" from the server's own ASR; with the silent-microphone fallback
this step *fails* (it does not skip), so the reported 50.31 s pass proves a
real voice turn was heard. **Typed input** — the Playground text box submits
a math prompt whose answer ("five") must appear in a new `bot-transcription`,
a semantic check rather than an echo check. **Returned audio** — inbound
RTP audio bytes must grow by > 2,000 over baseline with a live audio element
(track present, `currentTime > 0`) and outbound microphone bytes > 0.
**Cross-modal recall** — the typed prompt deliberately omits the codeword
and the assertion indexes strictly past all prior bot text, so only a *new*
"sapphire" — context supplied exclusively through the microphone path and
recalled through the typed path — can satisfy it. The half-duplex hazard is
handled deliberately (the voice turn completes before typing begins), and
the test fails loudly with browser logs on every timeout path.

### 2. Live stack — the exact qualified image, no overlay

Verified against the running system: container `nemotron-voicechat-model`
runs image `sha256:246ec4f4…` — **byte-identical to the runtime image in the
A120 promotion evidence** — with `/health` ready, protocol v3, and
`typed_input.ready: true` on the Pocket worker whose synthesis identity
(english_2026-04, alba, seed 0, `sha256-text-plus-base-v1`) matches the
qualified fixture contract. Mounts are model caches (read-only) and the
trace directory only — **no source overlay**, so the serving code is the
image's baked payload. The ngrok tunnel is live: `https://1c20c8c4aec6.ngrok.app/`
307-redirects to `/client/` which returns 200 from the application.

### 3. Bootstrap/current-source contract — recomputed, exact

I recomputed all three identities from the current working tree with
`public_runtime_identity` and compared them to the deployed image's labels:
payload `b9b5c633…`, recipe `aa041a84…`, combined source `0014042a…` — all
three match exactly. Together with the no-overlay finding, the deployed
binary is provably built from precisely the tree this review series
examined, closing the A113 source-freshness loop operationally.

### 4. Close-out documentation — hash-exact and honestly framed

`docs/asr-judge-swap-policy.md`'s new close-out paragraph records the
scenario-v2 green run with every value matching this review's independent
A120 collection: runtime and evaluator image IDs, all five evidence file
hashes (`62269861…`, `c79d7cc5…`, `b641b654…`, `3873a8bc…`, `cbe6cf53…`),
comparison self-hash `7f7bc5cd…`, and fixture-manifest identity
`18dc1cab…`; its claims (zero WER everywhere, minimum margin 0.5 against
unchanged 0.1, both verdicts green, empty holds/dispositions, unattended
eligibility) match the rederived artifact, and it correctly states the
green result does not rewrite the retained UTC reds.
`docs/direct-text-input-plan.md` (revision 7) pins the scenario-v2 hash and
top-report hash correctly and keeps the load-bearing honesty intact: the
qualification claim is scoped to "no worse than its exact speech twin," the
cycles-2–4 shared semantic miss is disclosed rather than buried, absolute
capability and parity remain separate claims, and the failed direct-position
experiment stays recorded as diagnostic-only. Step 4 remains marked in
progress for its publication/release-checklist items, which is accurate —
that remainder is the handoff itself, not this workstream's qualification.

### Verdict

**APPROVE — final handoff.** The qualified image is deployed bit-exact and
provably built from the reviewed tree with no overlay; the live browser gate
covers voice input, typed input, returned audio, and cross-modal recall by
construction; the tunnel is serving; and the close-out documentation is
hash-exact against independently rederived evidence with its claims scoped
honestly. The promotion chain (A82 → A115 → A117 → A118/119 → A120) is
closed and this operational review finds nothing left open in it. D8 intact.

## Verdict

The static tier is genuinely strong: the inventory is complete against an
independently derived reference, the preflight provably precedes the first
mutation with pure reads throughout, reject paths are mutation-free by
construction on the mocked tier, the clock adapter and audio/model clock
split close real alignment hazards, and the matrix hardening lands every
control this review pre-registered — with gates that can demonstrably fail.
F1 is a cheap hardening, F2 a required doc reconciliation, F3 absorbed into
the GPU tier.

**Approval is withheld pending the GPU component evidence enumerated above
(G1–G12).** Static and component-mock scope: no blocker found; F1/F2 to land
with the GPU tier.

VERDICT: APPROVE WITH CHANGES — static scope clean; G1–G12 plus F1/F2
required before Step 2A closure.
