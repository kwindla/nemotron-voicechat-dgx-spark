# Fable review — Phase 9 downloaded-artifact live qualification: Nano gate failure

Date: 2026-08-04 · Evidence:
`/tmp/voicechat-public-live-20260804-downloaded-r2/components/nano/report.json`
(no separate log file exists; the run dir holds `candidate-capture/`).
Read-only analysis; no code edited. (Noted: Phase 1 review is fully closed.)

## Diagnosis: runtime-configuration mismatch in the capture lane — NOT an
artifact deficiency, and NOT a bug in the gate's extraction or thresholds.

Evidence chain:

- `token_agreement = 1.0` over all 1,172 calls — the downloaded weights
  compute the same argmax path as the teacher. The artifact is healthy;
  an artifact deficiency cannot produce perfect token agreement with
  all-null logits.
- Every logit-derived field is `None` and `control_tokens.comparisons = 0`
  with `missing_allowed = False` → the gate FAILED CLOSED, exactly as
  designed. This failure is the gate working.
- The gate extracts `custom_outputs.text_logits`
  (`nano_component_gate.py:66-70`). The qualified public runtime ships the
  `skip_custom_text_logits` optimization (a qualified perf patch that
  suppresses text-logit materialization in normal serving). The gate's own
  `--allow-missing-control-margins` help text ("…omits text_logits. The
  default qualification gate remains strict.") shows this interaction was
  anticipated. The Phase 2 in-conversion gate captured through the
  converter stack, which emits logits; the Phase 9 lane captured through
  the public runtime with the optimization ACTIVE → nulls.

## Smallest correct fix

Configure the candidate CAPTURE (not the gate) to emit text logits for
qualification runs only: the capture invocation in the Phase 9 lane must
disable the skip-custom-text-logits optimization (dedicated env/flag on
the capture engine config), leaving production serving behavior untouched.
Do NOT pass `--allow-missing-control-margins` — that weakens the
control-margin gate, which this report explicitly declines to do; the
strict default stays.

## Required regression coverage

1. A capture-lane unit test asserting the qualification capture config
   sets the emit-text-logits flag, and that a capture payload lacking
   `text_logits` still fails the strict gate (preserving today's correct
   fail-closed behavior as a pinned test).
2. A fail-fast precondition: the capture engine echoes
   text-logits-enabled BEFORE replaying 1,172 calls, so a future
   misconfiguration fails in seconds, not after a full replay.
3. Keep `comparisons > 0 when missing_allowed=False` as an explicit
   asserted invariant.

## Verdict on this boundary: the gate and artifact are sound; fix the
capture configuration and re-run the Nano component lane. No gate
threshold may change.

---

## Delta — 2026-08-04, Nano capture fix (diagnostic overlay)

### Verdict: APPROVE — cleared for the full re-run.

- **Fail-closed skip contract:** a candidate omitting `text_logits` is a
  hard `ValueError` unless its config carries the qualified
  `voicechat_skip_custom_text_logits` contract with a non-empty
  `source_config_sha256` — arbitrary logit-less artifacts still fail. The
  strict gate thresholds are untouched, exactly as required.
- **Overlay is weight-identical by construction:** every non-config entry
  is symlinked (non-file entries rejected), and ONLY `config.json`
  changes — `custom_outputs` gains `text_logits` ahead of the production
  list and the skip contract is removed. Production serving configuration
  is never modified; the overlay is a per-run diagnostic view.
- **Provenance recorded:** the report carries source and diagnostic
  config SHAs plus both custom-output lists, so the evidence shows exactly
  what was captured through.
- **Tests:** two new tests; focused suite passes (7/7 host with the torch
  skip visible). The live 2-call GPU smoke (2/2 logits, cosine ≥ 0.9987,
  2 control comparisons, exact margins) proves the overlay emits logits
  end-to-end.
- Note: the 2-call smoke served as the fail-fast precheck this time;
  wiring a brief logit-presence precheck into the lane before the full
  1,172-call replay remains recommended (from the original findings) but
  does not gate the re-run.

---

## Live-evidence delta — 2026-08-04, corrected Nano gate (r3)

### Verdict: PASS CONFIRMED — the r2 observability failure is retired
without any weakening of the gate.

Independently read from
`/tmp/voicechat-public-live-20260804-downloaded-r3/components/nano/report.json`:

- `passed = true` over all 1,172 calls; token agreement 1.0.
- `minimum_logit_cosine = 0.998677` against the unchanged strict threshold
  0.995 — previously null, now measured and passing.
- Control-token evidence went from 0 comparisons (r2) to **1,172
  comparisons**: order agreement 1.0, max absolute BOS/PAD margin delta
  0.3125 against the unchanged 1.0 allowance, and the minimum teacher and
  candidate margins are IDENTICAL (−21.625 / −21.625).
- `missing_allowed = false` — the strict contract was never relaxed.
- Diagnostic-overlay provenance is recorded in-report
  (`diagnostic_config` with `source_config_sha256 = 2a0576a8…` and the
  diagnostic custom-output list), so the evidence states exactly what was
  captured through.

The downloaded Nano artifact is now fully qualified at the component
boundary on the public runtime. EarTTS gates are the next lane.

---

## Live-evidence delta 2 — 2026-08-04, EarTTS eager engine-init failure

### Diagnosis INDEPENDENTLY CONFIRMED: harness-only mount mismatch. The
proposed minimal fix is correct.

Verified from primary evidence:

- The published EarTTS release config pins
  `pretrained_tokenizer_name = /models/NVIDIA-Nemotron-Nano-9B-v2`.
- `run_live_suite.py:152` mounts the skeleton at `/models/nano-skeleton`
  and passes that path only as a CLI arg (:206-207) — the vLLM EarTTS
  engine reads its OWN release config, looks for the production path,
  finds nothing, and huggingface_hub then rejects the absolute path as a
  repo id (`HFValidationError`, confirmed in the log) under
  `--network none`. Fail-closed offline behavior working as designed.
- Production `cli.py` mounts the identical artifact at the production
  path (:332, :423) — the live demo and `voicechat up` are unaffected.

One reinforcing fact the diagnosis did not cite: the qualified Speech
stack dispatches the Nemotron backbone by the substring `Nemotron` in the
model path (the documented standalone-cutover blocker). `/models/
nano-skeleton` violates that naming contract independently of the
tokenizer lookup, so the harness mount was doubly wrong and the
production-path fix retires both hazards at once.

### Fix ruling

Mount and pass `/models/NVIDIA-Nemotron-Nano-9B-v2` in the live suite,
matching production exactly, with a regression test pinning the precise
mount string and argument. No artifact, gate, or production change is
needed or permitted. Rerun EarTTS and continue the ladder. The r3 Nano
PASS is unaffected (its lane consumed the skeleton via the CLI arg it was
given).

### Implementation verification — 2026-08-04: APPROVED, launch r4.

The uncommitted diff is exactly the Delta 2 ruling and nothing more: the
skeleton mount changes from `/models/nano-skeleton` to
`/models/NVIDIA-Nemotron-Nano-9B-v2:ro` and the CLI argument follows
(2 files, +46/−5, harness-only). The two new regression tests pin
precisely what the ruling required —
`test_gpu_component_prefix_uses_production_tokenizer_mount` asserts the
exact production mount string AND that no `/models/nano-skeleton`
reference survives anywhere in the command, and
`test_eartts_gate_receives_checkpoint_pinned_tokenizer_path` pins the
gate-side path. Focused suite 6/6, lint clean. Committing and launching
the r4 full ladder (fail-closed Nano rerun included) is approved.

---

## Delta 3 — 2026-08-04, r4 EarTTS eager: evaluator-only ASR failure

### Rulings (all three questions answered affirmatively):

1. **Evaluator compatibility, NOT EarTTS — CONFIRMED.** The lane completed
   engine init, inference, and decode: all four WAVs plus
   `acoustic-codes.pt` exist in `components/eartts-eager/`. The failure is
   exclusively in loading the supplied newer ASR checkpoint, which targets
   `nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt`
   — a module that does not exist at the pinned Speech commit `911ec674…`;
   the abstract-`ASRModel` fallback then fails correctly. EarTTS evidence
   is unimplicated.
2. **Proven-Parakeet fallback is LEGITIMATE and preferable.** The
   checkpoint at `…/oci-N-8_G-8_cacheaware-600M_granary-gsc_PC_
   multilookahead.nemo` (2,473,031,680 bytes, verified present) is the
   same independent evaluator that produced the prior EarTTS WER gates —
   using it preserves comparability with existing evidence and is more
   conservative than introducing a never-qualified newer evaluator.
   Independence holds: Parakeet is not part of the stack under test.
   Record it in the r5 report as a deliberate, documented fallback with
   the checkpoint path and size/hash.
3. **No harness change is required for r5** — `--asr-model` already
   parameterizes the evaluator. Two non-gating recommendations: (a) load
   the ASR evaluator BEFORE running EarTTS inference so an incompatible
   checkpoint fails in seconds instead of after a full lane (the standing
   fail-fast principle); (b) note in qualification docs that ASR
   checkpoints requiring post-`911ec674` NeMo classes are incompatible
   with the pinned Speech runtime.

Caution line for the record: the log also shows a non-fatal
`vllm.plugins.lora_resolvers.hf_hub_resolver` ModuleNotFoundError during
init probing; inference completed after it, so it is plugin-probe noise,
but it should be confirmed absent from failure-path significance if it
ever co-occurs with a real engine failure.

**Proceed: invoke r5 with the proven Parakeet `--asr-model`.**

### Delta 3 implementation verdict — 2026-08-04: APPROVED.

The uncommitted change is evidence-only, exactly as described:
`model_file_provenance` resolves the evaluator path, requires it to be a
regular file (a fail-fast that runs BEFORE inference — partially
satisfying recommendation (a)), and embeds resolved path + bytes + sha256
into `external_asr`. No decoding logic or pass criteria are touched.
Tests 2/2, lint clean, verified by running them myself. Committing before
the r5 EarTTS container starts is correct sequencing.

---

## Delta 4 — 2026-08-04, r5 coherent-run Nano gate

### Verdict: STRICT PASS CONFIRMED, and bit-reproducible across runs.

Independently read from the r5 report: `passed=true`, all 1,172 calls,
token agreement 1.0, minimum logit cosine 0.998677, 1,172 control
comparisons with order agreement 1.0 and max margin delta 0.3125,
`missing_allowed=false`, diagnostic-overlay provenance present
(`source_config_sha256 = 2a0576a8…`, `text_logits` leading the diagnostic
output list). Every measured value is IDENTICAL to r3 — the corrected
harness reproduces the gate bit-for-bit across independent runs, which is
itself determinism evidence. Nano stands qualified within the coherent r5
ladder; EarTTS eager with the proven Parakeet evaluator is next.

---

## Delta 5 — 2026-08-04, r5 EarTTS eager gate

### Verdict: STRICT PASS CONFIRMED, with cryptographically verified
evaluator provenance.

- `passed=true`; the checkpoint hash `d553750c…` was re-verified within
  the run itself.
- Decode evidence complete: acoustic codes `[1, 135, 31]` int64 in range
  1–1023, four WAVs produced via the production streaming codec in cache
  mode (streaming and offline variants at both rates).
- External ASR: transcript is EXACTLY the reference text
  ("The answer is five."), WER 0.0 against the 0.5 allowance.
- Parakeet provenance: the report records
  bytes 2,473,031,680 / sha256 `248e295b9f639079…` for the container's
  `/models/asr.nemo`; I independently hashed the host checkpoint and it
  matches exactly — the deliberate-fallback evaluator is cryptographically
  the proven one. Delta 3's requirements are fully satisfied.

Eager lane qualified. Graph mode in flight.

---

## Delta 6 — 2026-08-04, r5 EarTTS graph + Pocket gates

### Verdict: BOTH STRICT PASSES CONFIRMED.

- **EarTTS graph:** `passed=true` with `enforce_eager=False` (genuinely
  graph mode), transcript exactly the reference text, WER 0.0, same
  cryptographically verified Parakeet evaluator (`248e295b…`) — eager and
  graph modes agree on identical evidence.
- **Pocket (in-container RTF):** `passed=true`, minimum realtime speed
  1.456× against the 1.0× floor, measured inside the isolated container
  worker with health and per-sample evidence — discharging the
  in-container re-measurement requirement from the Phase 4/7 reviews.

## Delta 7 — 2026-08-04, restart cycle-1 fail-closed stop + HF fix

### Verdict: root cause CONFIRMED; the 2-file fix is correct. APPROVED.

- **Root cause verified verbatim in the log:** Transformers runs
  `os.makedirs(HF_MODULES_CACHE)` unconditionally (even offline) and the
  default modules path lives under `HF_HOME`, which is the read-only
  asset mount → `OSError: [Errno 30] Read-only file system:
  /models/huggingface/modules`. The dynamic-modules dir is needed because
  the Nano skeleton ships custom modeling code.
- **The restart gate earned its keep:** the component lanes run containers
  directly, so cycle 1 was the first live execution of the production
  `voicechat up` path under the read-only mount — the gate caught a REAL
  production-path defect, fail-closed, on its first outing.
- **Fix is minimal and sound:** the model launcher sets
  `HF_MODULES_CACHE=/tmp/voicechat-hf-modules` — container-ephemeral and
  writable — while `HF_HOME=/models/huggingface` stays read-only/offline;
  derived dynamic-module copies go to scratch while source-of-truth assets
  remain RO and hash-verified. The test pins all three (HF_HOME,
  HF_MODULES_CACHE, and the `:ro` mount). Focused tests pass (verified
  locally). Resume the restart cycles.

---

## Delta 8 — 2026-08-05, restart cycle-2 port-rebind race

### Verdict: diagnosis CONFIRMED, fix APPROVED.

- **Diagnosis verified in cycle-2.log:** `error: model port 127.0.0.1:8786
  is already in use` — the classic brief post-teardown window where the
  published port is released slightly after container exit. Cycle 1's
  clean SIGINT (health + READY + exit 0) proves teardown itself is sound;
  only immediate rebinding raced.
- **Fix is correct product semantics:** `_wait_ports_available` polls BOTH
  the model and Pipecat endpoints at 100 ms intervals against a 30 s
  monotonic deadline after `_stop_stack`, using the same
  `_port_available` predicate that gates startup; on timeout it raises
  with the exact unreleased endpoints rendered — fail-closed is
  preserved, and the pre-existing collision check remains the backstop.
  Immediate `Ctrl-C`-then-`up` (exactly what the restart gate and real
  operators do) is now deterministic instead of racy.
- Retry unit test present; focused tests verified passing locally.

Resume the cycles; this was the last known race in the teardown path.

---

## Delta 9 — 2026-08-05, port-probe TIME_WAIT false-busy

### Verdict: root cause CONFIRMED; fix APPROVED. Linux semantics are
exactly right.

- **Root cause:** the probe bound WITHOUT `SO_REUSEADDR`, so recently
  closed readiness-check connections in TIME_WAIT made the port look
  busy even though every real consumer (uvicorn, docker-proxy) sets
  `SO_REUSEADDR` and can rebind immediately. The probe was stricter than
  the servers it guards — a false-busy, not a real conflict (confirmed by
  the evidence: no listener and no stale container remained).
- **No false-negative exposure:** on Linux, `SO_REUSEADDR` permits
  binding over TIME_WAIT remnants but NOT over an active LISTEN socket —
  an actual collision still returns EADDRINUSE, including a wildcard
  (0.0.0.0) listener colliding with the specific 127.0.0.1 probe. The
  collision check retains full strength; the tests pin exactly this
  (option set, active-listener rejection on a real bound socket, retry
  behavior). Verified passing locally.
- The probe now matches real-server bind semantics, which is the correct
  definition of "available" for this gate.

Three restart-gate defects found and fixed in sequence (HF modules cache,
port-release race, TIME_WAIT probe) — each one a genuine production-path
issue an operator would have hit. Resume cycles.

---

## Delta 10 — 2026-08-05, r8 restart gate PASS + browser selector fix

### Part 1 — restart/recovery gate: LIVE PASS CONFIRMED (and it carries
the loopback negative).

Independently read from the r8 report: `passed=true`; three SIGINT cycles
ready in 390.5 / 387.5 s (consistent cold starts); SIGKILL cycle initial
ready 379.5 s; post-`down` recovery ready in 385.5 s; and
`raw_model_loopback_only=true` — the live loopback negative assertion
(model port unreachable from every non-loopback address) is discharged
within this same report. The residual-GPU-state failure mode that
motivated this gate did not reproduce across four consecutive cold
starts, including the SIGKILL path. B3 is retired with live evidence.

### Part 2 — browser selector fix: APPROVED, no false-green vector, no
additional regression test required before rerun.

The diff changes only the locator: `[placeholder="Type message..."]
:visible`, targeting the active Radix panel. False-green analysis:
Playwright strict mode still throws if MORE than one visible input
matches, and any action on a zero-match locator times out and fails — so
the fix cannot mask either duplicate-visible or missing-input
regressions. Every downstream assertion (READY, real UI send, assistant
text, cross-modality memory, RTP audio) is unchanged. The e2e gate itself
is the regression test for future Playground DOM changes; the in-code
comment documents the rationale. Rerun the browser gate.

---

## Delta 11 — 2026-08-05, r9 browser assertion semantics fix

### Verdict: APPROVED — no false-green vector; no additional tests
required. Two pieces of bonus live evidence recorded.

- **r9 functionality was already correct:** the model produced the exact
  text "Two plus three equals five."; only the assertion failed — the
  `\bfive\b|\b5\b` regex with `.last` matched hidden Playground clock
  timestamps ("5:09:02 AM"), a locator defect, not a product defect.
- **The fix closes every identified trap:** (1) the real microphone
  sapphire turn completes FIRST (in-code comment documents the Chrome
  fake-capture-clock rationale); (2) both locators scope
  `visible=True`, excluding hidden panels and clocks; (3) the numeric
  `\b5\b` alternative is removed so timestamps can never match the
  word-only assertion; (4) decisively, the recall assertion is a COUNT
  DELTA — `sapphire_before_recall` is captured after the mic turn and the
  test requires `count() > sapphire_before_recall`
  (test:451,469-472) — so the mic turn's own transcript cannot
  false-green the memory probe, and since the recall prompt omits the
  codeword, only the assistant's memory-based answer can add a visible
  rendering. All RTP/audio assertions retained; waits are bounded.
- **Bonus evidence 1:** the r9 trace proves the server-owned half-duplex
  gate worked LIVE — the fake mic's entire speech window fell inside
  typed_input_injection_started→finished and was correctly gated
  (mic-partial discarded), the exact D1 semantics from the plan review,
  now observed in production.
- **Bonus evidence 2:** exact deterministic math text from the public
  stack over real SmallWebRTC.

Rerun the browser gate; sustained remains the final lane of this ladder.

---

## Delta 12 — 2026-08-05, RTVI data-channel observer replaces DOM reads

### Verdict: APPROVED. The observer is strictly passive, attribution is
correct, and the real-UI typed-path requirement remains satisfied.

- **Passivity verified in the diff:** the probe patches
  `RTCPeerConnection` to hook BOTH `createDataChannel` (local) and the
  `datachannel` event (remote), covering every channel regardless of
  direction; recording is `addEventListener("message", …)` with
  `JSON.parse` into a window array inside try/catch — purely additive, no
  preventDefault/stopPropagation, and it cannot displace the app's own
  handlers. Installed via `add_init_script` before any page JS runs.
- **Assertion semantics are sound and stronger than DOM scraping:** the
  RTVI event stream is append-only, eliminating the transient-DOM
  finalization race outright; the gate waits for the attributed final
  user-transcription sapphire (mic), then bot-transcription sapphire,
  then applies COUNT-DELTA guards before typed math and before recall
  (`bot_before_math`, `bot_before_recall`, `after=` filtering) — user
  echo cannot increment bot counts, and stale events cannot false-green
  the recall. The event stream IS the RTVI product contract.
- **Real-UI requirement intact:** typed text still enters through the
  real visible Playground input; RTC audio assertions are unchanged.
- **Observer wiring is itself regression-tested** by the local
  real-Chromium mock asserting the observer sees bot-transcription.
- r10's server trace (mic sapphire → assistant sapphire; typed math →
  exact five; recall-omitting-codeword → assistant sapphire) already
  proves the product path; this delta makes the gate measure that truth
  reliably.

Rerun the browser gate. Sustained remains the final lane.

---

## Delta 13 — 2026-08-05, r11 browser PASS + multiturn silence continuation

### Verdict: browser PASS independently confirmed; the multiturn fix is
APPROVED and is the architecturally correct behavior, not a workaround.

- **r11 browser gate:** `1 passed in 43.85s` confirmed in browser.log —
  all three semantic turns over real SmallWebRTC with the passive RTVI
  observer. The browser lane is qualified.
- **Multiturn root cause is real:** the model is frame-driven and cannot
  advance a response boundary without input frames; a harness that stops
  sending after a fixed tail deadlocks by design. Production clients
  stream continuous mic silence — the harness now mirrors the product.
- **Implementation verified sound:** `send_silence_until_stopped` paces
  zero-PCM frames on an absolute monotonic schedule
  (`started + (i+1)×0.08` — drift-free), interleaves stop-responsiveness
  via `wait_for(stop.wait(), remaining)`, and halts within one frame of
  `response.done`. Cleanup is exception-safe: `finally:
  continuation_stop.set(); await continuation` on every exit path,
  awaited BEFORE `session.stop` is sent — no leaked task, no
  send-after-close race. The sender starts only after the fixture phase,
  so no concurrent-send interleaving exists.
- **No false-user-turn risk:** zero-PCM frames sit far below the
  −40 dBFS speech gate and cannot trigger BOU; silence advances the clock
  only, exactly as production mic silence does.
- Focused tests verified passing locally, including the
  stops-after-signal pacing proof.

Multiturn reruns next; sustained remains the final lane.

---

## Delta 14 — 2026-08-05, r12 multiturn: over-constrained turn-1 oracle

### Ruling: PRODUCT BEHAVED CORRECTLY; the fixture oracle is
over-constrained. Approve the minimal fixture-only change below.

Independently read from the r12 report: turn 1 "I have noted the code
word." (109,368 audio bytes) — a correct, arguably ideal acknowledgement
that declines to parrot the secret; turn 2 "The sum of two and three is
five." (match); turn 3 "The code word is sapphire." (match) — the actual
memory proof, produced although the turn-3 input omits the codeword.
Session closed cleanly; `passed=false` stems SOLELY from turn 1's
`expected_response_any=["sapphire"]` demanding an echo.

**Smallest principled change (fixture data only, no gate code, no
threshold change):** expand turn 1's `expected_response_any` to
acknowledgement markers — e.g. `["sapphire", "noted", "code word",
"remember", "acknowledged", "got it"]` — while turn 3 keeps
`["sapphire"]` alone as the memory proof. Safety analysis:

- **Silent/empty responses still fail:** the per-response requirements
  (`audio_bytes > 0`, non-empty text, semantic match) are untouched.
- **Cross-turn memory cannot false-green:** turn 3's input omits the
  codeword, so its sapphire can only come from model memory; turn 1's
  input-side sapphire is independently verified by the RNNT transcript,
  so turn 1's role is acknowledgement + audio, not memory.
- **Refusals do not pass:** the marker list is acknowledgement-specific;
  "I cannot…" matches none of them.

Rerun multiturn with the amended fixture; sustained remains the final
lane.

### Delta 14 implementation verdict — 2026-08-05: APPROVED.

The diff is one fixture line, exactly as ruled: turn 1
`expected_response_any` becomes `["sapphire", "noted", "acknowledged",
"got it"]` — notably TIGHTER than my suggested list ("code word" and
"remember" were dropped, further shrinking any refusal-adjacent surface;
none of the four markers can match an "I cannot…" refusal). Turn 3
remains exactly `["sapphire"]`, and the new pinning test asserts all
three invariants: the acknowledgement marker present, recall requiring
the secret alone, AND the recall prompt not containing "sapphire"
(`test:31-33`) — so a future fixture edit cannot silently reintroduce a
prompt-echo false green. Tests 3/3, verified locally. Rerun multiturn.

---

## Delta 15 — 2026-08-05, r13: turn-2 input-WER + Pocket determinism label

### Ruling 1: APPROVE the fixture-only simplification of turn 2 to
"What is two plus three?" — WER threshold untouched.

Independently confirmed from the r13 report: all three semantic matches
now pass (acknowledgement / five / sapphire), the model answered the math
exactly, and the SOLE failure is turn 2 `input_wer = 0.6` vs 0.5 — turns
1 and 3 sit at 0.2/0.25. The failing words are the instruction suffix
("Answer in one short sentence."), which RNNT garbled into
"threentence"; the suffix duplicates the server system prompt's standing
one-sentence requirement and carries no mathematical semantic payload.
Removing it is the smallest principled fix: the input-WER gate keeps its
threshold and keeps verifying intelligibility of the semantically
load-bearing prompt; no alternative (raising the threshold, per-turn
overrides) avoids weakening or complicating the gate. Turn 2's expected
responses (["five","5"]) do not depend on the suffix.

### Ruling 2: the Pocket byte-length variance is NOT material to gate
validity, but the "deterministic" label is now falsified and MUST change.

Fixtures are generated fresh per run, hashed into the per-run manifest,
and verified in-run — the evidence chain is self-consistent regardless of
cross-run bit-identity, so no gate conclusion is affected. However the
generator's label must say "reproducible/qualified" (per-run
self-recording) rather than "deterministic", and any manifest/doc claim
of fixture bit-determinism must be corrected to match reality. CPU
thread scheduling and/or sampling inside Pocket generation are the
expected sources; no further action beyond the honest relabel.

Rerun multiturn with the simplified turn 2; sustained remains the final
lane.

### Delta 15 implementation verdict — 2026-08-05: APPROVED.

The diff is exactly both rulings and nothing more: the docstring drops
"deterministic" for "independent … qualified Pocket TTS"; the simple-math
fixture text becomes "What is two plus three?"; the WER threshold and all
gate code are untouched; and the invariant test now additionally pins the
exact simple-math text so the suffix cannot silently return. 3/3 verified
locally. Rerun multiturn, then sustained.

---

## Delta 16 — 2026-08-05, r14 network-boundary verdict

### Verdict: BROWSER AND DIRECT STRICT-V3 LANES PASS — independently
confirmed. Deltas 14/15 closed their exact issues with zero gate
weakening.

- **Browser:** `1 passed in 40.99s` on the public image (verified in
  browser.log) — three semantic turns over real SmallWebRTC with the
  passive RTVI observer.
- **Direct multiturn:** `passed=true`, clean `session.closed`, all three
  turns semantic-matched with substantial audio (112,896 / 116,424 /
  127,008 bytes). Decisive detail: turn 2's input WER dropped from 0.6
  to **0.0** after the suffix removal — the Delta 15 fix eliminated its
  exact failure rather than sliding under the unchanged 0.5 threshold.
  Turn 1 passes via acknowledgement ("I have noted the code word.") while
  turn 3 independently proves memory ("The code word is sapphire.")
  against a codeword-free prompt — the Delta 14 split working as
  designed. Memory, audio-bytes, text, and closure requirements all
  remained enforced and were satisfied with real values.

The 1,020 s sustained run — the final lane and the project's
highest-stakes gate given the documented EarTTS drift history — is now
active; evidence monitoring only from here.

---

## Delta 17 — 2026-08-05, r14 sustained boundary

### Headline ruling: the RUNTIME sustained evidence PASSED on its merits.
The sole failure is a two-part harness expectation mismatch. And the
project's highest-stakes risk did not materialize: delivery slope
0.317 ms/min over 17 minutes (≈5.4 ms total) against the 1.0 ms/min
threshold — the historic linear EarTTS drift is ABSENT on the public
stack. Queue debt peaked at 316 ms and returned; last lag 70.76 ms;
11,837 metric frames complete; zero errors; 17/17 typed jobs completed,
16 answered vs allowed_unanswered=1; all response audio nonempty.

### The failure is harness-side, verified in code and server:

1. **Sizing:** `--expect-session-limit` defaults true, but typed-input
   server gating suppressed ~915 model advances, so model frames reached
   ~11,835 < 12,000 and the limit CORRECTLY never fired. The harness
   budget ignored gating suppression.
2. **Close-reason vocabulary:** the pass formula
   (`sustained_strict_v3.py:296-299`) expects `client_requested` on the
   explicit-stop branch, but the server's actual strict-v3 reason is
   `client_stop` — a harness contract error on both branches.

### Approved smallest fix (as proposed, with one addition):

Keep the unified ≥15-minute gate. In limit-expecting mode, allocate
source duration with explicit margin for gating suppression (e.g.
1,200 s / 15,000 source frames, guaranteeing >12,000 model advances even
with ~900+ gated), and require `session_position_limit` to actually
fire — full source completion is NOT acceptable as a limit proof.
Correct the explicit-stop branch's expected reason to the actual
contract (`client_stop`). Addition: pin the server's close-reason
vocabulary in a unit test so harness and server cannot drift again.

### Frame-2250 unanswered job: attribution gap, not silence.

The job record shows `accepted/started/completed` with a real `turn_id`,
and response-004 contains the exact tool result — the turn produced
correct content but the response events carried no `job_id` linkage. It
falls within the allowed base rate either way (16/17 vs allowed 1), so it
is not gating; recommend the harness additionally correlate responses by
`turn_id` (present in both records) before counting a typed job
unanswered — evidence-side only, no server change — and track the
missing `job_id` propagation as a strict-v3 attribution follow-up.

Resize, fix the reason vocabulary, rerun sustained.

### Delta 17 implementation verdict — 2026-08-05: APPROVED.

- **Limit branch is strictly honest:** `source_completion_passes` requires
  `sent_frames < total_frames AND close_reason ==
  "session_position_limit"` in limit-expecting mode — the limit must
  genuinely fire; full completion cannot masquerade. The explicit-stop
  branch requires full completion AND `client_stop`; zero
  `client_requested` references remain (stale vocabulary eliminated).
- **Duration sufficiency:** 1,200 s / 15,000 source frames consistent
  across the harness, run_live_suite, and CLI defaults gives ~2,085
  frames of margin over the 12,000 guard even after ~915
  gating-suppressed advances — suppression would need to more than
  triple to starve the limit. Adequate.
- **Attribution fix is evidence-side and race-correct:**
  `resolve_response_job_id` prefers the explicit `job_id` and falls back
  to the `typed_job_by_turn` mapping at `response.done` — resolving the
  FC ordering case (response.created before transcription.completed)
  without any server change; precedence and late-fallback are
  test-pinned.
- Focused tests verified locally. Rerun sustained (final lane).

---

## Delta 18 — 2026-08-05, r15 network-boundary confirmation

### Verdict: CONFIRMED. Browser 1/1 in 47.88 s (verified in log); direct
strict-v3 `passed=true` with clean `client_stop` closure, input WER
0.3/0.0/0.25, all semantic matches
(acknowledgement / five / sapphire), audio 127,008/119,952/116,424 bytes.
The corrected close-reason vocabulary is exercised and passing on its
first live outing. The 1,200 s sustained run is active; final
verification will specifically check (a) function-turn attribution via
the late turn-id mapping and (b) closure by `session_position_limit`
BEFORE all 15,000 source frames are sent.

---

## Delta 19 — 2026-08-05, r15 sustained: limit FIRED, sender race clobbered the report

### Headline: the resized gate WORKED. Events prove the structured fatal
`session_position_limit` error at exactly `model_frames=12000` followed by
`session.closed reason=session_position_limit`, well before source
exhaustion — the limit-expecting branch's precondition
(`sent_frames < total`) is genuinely satisfiable. The failure is purely a
harness shutdown race: the paced sender passed its top-of-loop
`receiver_done` check (:257), slept, and sent one more frame into the
cleanly closed (code 1000) socket; `ConnectionClosedOK` then reached the
top-level handler (:374), which OVERWROTE report.json down to
`['errors','passed']`. events.jsonl survived intact (append-mode), so the
run's truth is recoverable.

### Approved smallest fix, with one mandatory addition:

1. Catch `ConnectionClosedOK` around the sender's sends; break ONLY when
   `receiver_done` is set AND the recorded `session_closed.reason` is
   `session_position_limit` (or a declared close) — otherwise re-raise.
   Unexpected disconnects stay fail-closed.
2. **Mandatory addition:** the exception path must NEVER overwrite an
   assembled report. Write exception evidence to a sibling `error.json`
   (or merge into the partial report) so a late crash cannot destroy
   gate evidence again.

### Status=failed ruling: keep the server unchanged; key the pass
contract on REASON.

The server closes the WS cleanly (1000) while marking the session
`status=failed` with a fatal structured error — a protective by-design
stop labeled as failure. Changing server semantics is a runtime change
requiring parity gates and is not justified for a label; the harness pass
contract must key on `reason == session_position_limit` and treat status
as informational. Record `status="failed"`-for-a-by-design-stop as a
cosmetic strict-v-next contract improvement.

Fix the sender race + report preservation; rerun sustained.

### Delta 19 follow-up ruling — error classification: APPROVED as specified.

The proposed classification is exactly right and is the only shape that
preserves both evidence completeness and strictness:

- ALL errors stay recorded (evidence is never filtered at capture).
- `unexpected_errors` excludes ONLY an error that simultaneously
  satisfies all three keys: fatal `session_position_limit` code, exact
  `model_frames == max_model_frames`, AND `--expect-session-limit` with
  `session_closed.reason == session_position_limit`. The triple keying
  means an early-firing limit (wrong frame count), a limit error without
  matching closure, a malformed/missing-detail event, any other code, or
  a limit event in no-expect mode ALL remain failing — no generic
  exclusion can creep in.
- `passed` requires `not unexpected_errors`. Tests must pin each failing
  variant plus the single passing shape.

Report-preservation semantics APPROVED as stated: the exception path
writes a sibling `error.json` and creates a minimal `report.json` ONLY
when none exists; an existing assembled report is never overwritten.

This completes the Delta 19 fix specification. Implement, test, rerun
sustained.

### Delta 19 implementation verdict — 2026-08-05: APPROVED.

Verified in the diff: the sender's `ConnectionClosedOK` handler first
waits up to 1 s for `receiver_done` (closing the race where the receiver
has not yet recorded the closure), then breaks only via
`expected_sender_close(receiver_done, session_closed, expect_limit)` —
anything else re-raises, so unexpected disconnects stay fail-closed.
`classify_protocol_errors` keeps every raw error, exempts only the FIRST
triple-keyed limit event with `isinstance(model_frames, int)` exactness,
and the pass formula requires no unexpected errors AND
`expected_limit_error_count == 1` in limit mode — exactly-one semantics,
stronger than my ruling required (zero limit errors in limit mode also
fails). `write_exception_evidence` always writes the sibling
`error.json` and creates `report.json` only when absent, with the
byte-preservation of an assembled report test-pinned. Focused tests
verified locally. Launch the final sustained rerun.

---

## Delta 20 — 2026-08-05, r16 turn-1 suffix stochasticity

### Ruling: APPROVE the fixture-only simplification to "Remember that the
codeword is sapphire." — retain the acknowledgement markers; reject the
empty-oracle alternative.

Verified from the r16 report: all three responses semantically correct
("I have noted the code word." / five / sapphire), clean close, sole
failure turn-1 input WER 0.6 from RNNT dropping the "and acknowledge it
briefly" suffix — the same suffix-stochasticity class as Delta 15's math
turn, observed across r12–r16.

Marker-appropriateness analysis: removing the explicit "acknowledge"
request does NOT make the markers inappropriate. A natural assistant
response to the imperative "Remember that X" IS an acknowledgement (the
observed behavior in every run), so the oracle moves the requirement
from the input text to the response side — we stop ASKING for the
behavior while still REQUIRING that we observe it, which is strictly
cleaner. The `expected_response_any=[]` alternative weakens the semantic
gate to any-nonempty-text (refusals and non-sequiturs would pass) and is
rejected. WER margin improves decisively: the simplified 6-word expected
text against the observed RNNT rendering ("code word" split) yields
≈0.33, comfortably under the untouched 0.5 threshold. Residual risk is
fail-closed, not false-green: if the model ever declines to acknowledge
unprompted, turn 1 fails visibly and we revisit. Pin the new text in the
invariant test as before. Rerun.

### Delta 20 implementation verdict — 2026-08-05: APPROVED. One fixture
line exactly as ruled; the invariant test pins the exact new turn-1 text
alongside the retained marker, strict turn-3 oracle, and codeword-free
recall prompt. 3/3 verified locally. Rerun the ladder.

---

## Delta 21 — 2026-08-05, r17 browser wrong-tool event: release-policy ruling

Evidence verified: user-transcription contained sapphire; the bot answered
with a UTC time (tool-call activity confirmed in the model trace); the
gate correctly failed waiting for a sapphire response. Components in r17
were all strict passes with values bit-identical to r3/r5/r8.

### Ruling 1 — documented model base-rate, NOT a deployment release
blocker. Wrong-tool selection with `get_current_utc_time` registered is
the known upstream-checkpoint limitation recorded since the demo era;
this release is a community quantization and Phase 8 explicitly
distinguishes upstream model behavior from deployment defects. Two
conditions attach: the base rate must be measured and documented (current
browser-gate evidence: r11 pass, r15 pass, r17 fail ≈ 1-in-3 sessions
with at least one wrong-tool event), and the gate must handle it honestly
rather than flake.

### Ruling 2 — one recorded retry; REJECT the tool-router guard.
A Pipecat tool-router guard is rejected outright: it re-introduces the
semantic guarding the project owner explicitly removed ("model owns tool
selection"), changes qualified behavior, and would misrepresent the model
in a reproducibility release. Unbounded retry-until-pass is also
rejected. Approved: the browser gate gains max-2 attempts with EVERY
attempt's outcome recorded in the report (attempts list), two consecutive
failures still fail the ladder, and `docs/known-limitations.md` gains the
measured wrong-tool base rate with the browser-gate retry rationale.
This mirrors the multiturn `allowed_unanswered=1` base-rate philosophy
already approved.

### Ruling 3 — resuming from network gates on one fresh stack is
LEGITIMATE. Component gates are deterministic (bit-identical across four
independent runs) and their sealed r17 evidence stands; the network gates
test the stack, not the artifacts. Conditions: identical image ID and
hash-verified artifacts at startup (already enforced), and the resumed
report must reference the r17 component evidence explicitly for
coherent-ladder provenance.

### ETA implications: browser retry + direct + sustained ≈ 45–60 min if
green — the downloaded-artifact ladder can close today. Dual-source
(multi-hour conversion), clean-machine run, and CI remain; full release
checklist realistically completes tomorrow.

### Delta 21 implementation verdict — 2026-08-05: APPROVED FOR LAUNCH.
Non-atomic attempts-file write is ruled a NIT, not a blocker.

- **Atomicity ruling (correcting the request's wording):**
  `browser-attempts.json` uses plain `write_text`, not temp+rename. This
  is NOT launch-blocking: pass/fail flows from in-memory state and child
  exit codes — never from re-reading this file — so a torn write cannot
  false-green anything; the primary evidence is the per-attempt logs with
  recorded SHA256s. Recommended (NIT): switch to the existing
  `artifacts.atomic_json` helper at the next code touch.
- **Fail-closed verified:** pass requires child returncode 0; the
  `TimeoutExpired` path (600 s) records the outcome with `passed` False;
  after two failures the gate raises — retry-until-pass is structurally
  impossible at max_attempts=2 with every attempt's
  duration/returncode-or-timeout/log-SHA256 recorded.
- **Single-client protection:** `wait_model_idle` runs before the retry.
- **Base-rate wording is honest:** known-limitations reports the observed
  2-pass/1-wrong-tool three-session sample explicitly "not an estimate of
  the model's general" rate — no generalization.
- **Comparator semantics correct:** browser verdict normalized so
  stochastic attempt count is not an artifact delta; legacy form kept.
- Focused tests verified locally (8/8 in the suite file). Launch the
  resumed network gates.

---

## Delta 22 — 2026-08-05, r18 FINAL boundary: the downloaded-artifact
ladder is CLOSED. ALL GATES PASS.

Independently verified from primary reports (no client run against the
stack; manual-test window respected):

- **Top report `passed=true`.** Browser: attempt 1 failed (known
  wrong-tool), attempt 2 passed, both logs hashed — the bounded-retry
  contract exercised exactly as designed. Multiturn `passed=true`.
- **Sustained `passed=true` with every Delta 17/19 mechanism working:**
  `sent_frames 13,039 < source_frames 15,000`; exactly one structured
  limit error (`expected_session_limit_error_count = 1`) at
  `model_frames = max = 12,000` with matching
  `session.closed reason=session_position_limit`;
  `unexpected_errors = []`; queue criterion met — last-block mean
  70.087 ms (≤80), slope 0.289 ms/min (≤1), peak debt bounded (381.7 ms);
  metric stream complete at 12,002; typed jobs 18 completed / 18
  answered / 0 unanswered — the attribution fix leaves zero unanswered.
  Embedded external ASR `passed=true` (18/18 per report, near-silent 0);
  multiturn ASR log `passed=true` (3/3).
- **Drift verdict stands:** two independent 1,200 s runs (r14: slope
  0.317; r18: 0.289 ms/min) — the historic linear EarTTS drift is absent
  on the public stack, reproducibly.

**Updated wrong-tool base-rate record (full observed browser sample):**
r11 pass, r15 pass, r17 wrong-tool fail, r18 attempt-1 wrong-tool fail,
r18 attempt-2 pass — 2 wrong-tool events across 5 sessions. The
known-limitations sample must reflect this full record at the commit
boundary.

**Phase 9 downloaded-artifact qualification: COMPLETE.** Remaining before
public push: user manual ngrok test, conversion carve-out review, claims
trim, curated-main pre-push review. Dual-source and clean-machine follow
as stage 2.
