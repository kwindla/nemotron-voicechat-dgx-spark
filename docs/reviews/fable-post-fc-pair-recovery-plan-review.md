# Adversarial Plan Review: Post-Function-Call PAD-Pair Recovery Defect

- **Date:** 2026-08-06
- **Scope:** Review of the proposed 7-step plan for the intermittent post-FC pair-drafting
  recovery defect in the DGX Spark runtime. Read-only inspection of the failed sustained run
  (`/tmp/voicechat-public-live-20260805-converted-r2`, session `aaee56ee-…`), the passing repeat
  (`…-sustained-r4`, session `5945a8cf-…`), their model traces, and
  `src/nemotron_voicechat_runtime/runtime_optimizations.py`. No production code was modified.

## Verdict: APPROVE WITH CHANGES

The plan's instincts (preserve evidence, telemetry, replay, localize before fixing) are right,
but the trace evidence I extracted refutes four of the five candidate hypotheses at the code
level and strongly indicates the "poisoned state" is **not scheduler-local** — it is the
conditional-drafting *input* (per-frame effective function-channel tokens) that never returns
to PAD after the final degenerate FC cycle. Step 5 as written (harden scheduler recovery only)
therefore risks shipping a no-op that passes its own gates. Two blockers, four important
findings, three nits.

---

## Evidence section: what I verified

### FAILED run (r2, session aaee56ee, report `/tmp/voicechat-public-live-20260805-converted-r2/sustained/report.json`)

- `passed=false` solely on the queue criterion: `peak_implied_queue_debt_ms = 2979.732`
  (criterion ≤1000). `first_block_mean_ms 59.96`, `last_block_mean_ms 58.84`, slope −0.51 —
  the failure is a *transient* debt spike, not drift. The `session_position_limit` at 12000
  frames is expected/allowed (`expected_session_position_limit: true`).
- The model trace (`…-r2-traces/model/aaee56ee-…/events.jsonl`, 12000 `model_step` records)
  contains per-frame `inference_ms`, `server_step_ms`, `turn_state.function_calling` flags, and
  `vllm_request_positions.nano.generated_tokens`. **Nano token deltas per frame directly encode
  the pair scheduler's decision**: delta 0 = draft buffered, 2 = pair executed, 1 = sequential
  single. This is direct evidence, not inference:
  - **Stall window confirmed.** Last FC cycle (8th call, `call_922763…`, second
    `get_current_utc_time` of turn_16) runs rel t ≈ 961.7–977.5 s. From frame 11501
    (rel 977.62) through frame 11994 (rel 1022.62): **510 consecutive frames, delta=1 on 509
    of them** (one stray 2 at 976.02 inside injection), inference mean **82.7 ms**
    (range 82–88), `server_step_ms` mean 82.9. First buffered frame after the FC: **frame
    11995 at rel 1022.74** — 61.0 s after the FC's SOTC at 961.7, matching the "~62 s" claim.
    (The claimed 85.6 ms mean is slightly high vs my 82.7–84 ms depending on window; same
    conclusion either way.)
  - **Healthy in-session control exists.** After the *earlier* repeated double-FC of turn_14
    (calls at rel 786.5 and 795.5 — same repeated-call shape), pairing resumed within **3
    ordinary frames**: frames 9366–9368 sequential (first at 154.5 ms — KV catch-up), frame
    9369 buffered at rel 800.38. Windows 800–860 and 860–900 show full pairing
    (~50/50 delta 0/2, mean 70.8/71.4 ms). So repeated same-name calls per se recover; only
    the *last* cycle (second FC within turn_16, after the rel-901.7 malformed function-channel
    garbage: deltas `"_"`, `"utc"`, `"_time"` with `response_id: null`) failed to recover.
  - **Scheduler-visible FC flags were clean during the entire stall**: for 491/510 stall
    frames `function_calling` = {active:0, awaiting_response:0, injecting_response:0,
    background_active:0, forced_tokens:0}, `agent_control: "pad"` on all 510,
    `completed_calls` constant at 8, `vllm_request_positions.request_id` constant `"5"` for
    the whole session.
  - **Debt accounting closes.** ~510 frames × ~3–5 ms overage ≈ 2.2–2.5 s, plus a 1.55 s
    wall-clock stall between frames 11991→11992 (rel 1020.74→1022.29) ≈ the reported 2.98 s
    peak. `inference_ms ≈ server_step_ms` (82.7 vs 82.9) — the overage is inside model
    inference, not codec/transport overhead.
  - **Resume coincides with user speech**: buffering resumes at frame 11995 immediately after
    the final typed-audio injection (qual-12000, injected ≈ rel 1021–1022), then the session
    hits the 12000-frame limit at rel 1026.9.
- Quality context (out of scope but relevant to "malformed"): responses 14 and 15 truncated to
  `" The"`; response 17 is 0 audio bytes / empty text; 8 FC invocations against ~4 tool
  prompts (repeated calls in turns 14 and 16).
- The failed session's trace exists **only** under `/tmp/voicechat-public-live-20260805-converted-r2-traces/`
  — it is *not* present in `~/.local/state/nemotron-voicechat/traces/model/` (session dirs
  there do not include `aaee56ee`). `/tmp` is volatile across reboots.

### PASSING run (r4, session 5945a8cf)

- `passed=true`, peak debt 399.69 ms, 18/18 typed answered.
- **Weak control:** the r4 trace contains exactly **one** `function_call_requested` for the
  whole 1200 s (vs 8 in r2), and zero buffered-frame gaps >8 s. The fresh-stack repeat passed
  largely because the model never entered the repeated/degenerate FC regime — it does not
  exonerate the recovery path.

### Scheduler code (`/home/khkramer/src/nemotron-voicechat-dgx-spark/src/nemotron_voicechat_runtime/runtime_optimizations.py`)

Read in full: `pad_pair_generate_next_token` (369–487), `_prepare_pad_pair_call` (493–544),
`_finalize_pad_pair_call` (547–595), `pad_pair_fc_in_progress` (95–105),
`pad_pair_should_draft` (108–110), `pad_pair_precontrol_reason` (125–180),
`_pad_pair_state` (329–354), `public_pad_pair_status` (598–638). Cross-checked against the
NVIDIA wrapper's async FC loop at
`/home/khkramer/src/Speech-nemotron-voicechat/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
(async loop ~1440–1900; sync FC state machine `_apply_fc_state_machine` ~2311–2372).

---

## Challenge 1 — Causal interpretation: VERIFIED, with one correction

The evidence **does** show pair drafting stopped: 510 consecutive delta-1 frames is the
scheduler's sequential-single signature, directly observable in nano `generated_tokens`.
Alternative causes are excluded:

- Codec/CPU contention: `server_step_ms − inference_ms ≈ 0.2 ms` throughout the stall; paired
  windows immediately before (860–900) ran at 71.4 ms with identical codec/GC/transport config.
- EarTTS behavior: EarTTS request positions advance normally; agent channel is PAD (silence)
  for the whole window, and the per-frame cost matches the known unpaired step cost (the
  paired cadence amortizes a ~31–47 ms buffered frame against a ~89–95 ms pair frame; losing
  it raises every frame to ~83–87 ms — see resumed pairing at frames 11995–11999: 31.8/95.2
  alternation).

**Correction to the framing:** "resumed with user speech" discriminates *against*
scheduler-latch hypotheses and *for* the effective-token family. No code path in
`runtime_optimizations.py` clears any pad-pair state on user speech/BOS — user BOS would, via
the control barrier (561–572), set `sequential_mode = True` (more sequential, not less). The
only thing user speech plausibly changes is the *model's* text/function token streams (new
turn → channels return to PAD), which is exactly the input `previous_effective_pad` is
recomputed from.

## Challenge 2 — Hypothesis ranking with code cites

1. **`previous_effective_pad` never returns True after the final FC cycle — MOST LIKELY, but
   the stuck value lives upstream of the scheduler.**
   `_finalize_pad_pair_call` recomputes it *unconditionally every frame* from
   `gen_text[:, frame_idx]` / `gen_function[:, frame_idx]`
   (`runtime_optimizations.py:557–560`). For it to be non-True for 510 frames, the effective
   *function-channel* token at `frame_idx` must be non-PAD every frame (agent text was PAD —
   `agent_control: "pad"` in every stall record). With `pending is None`, `bypass` False and
   `should_draft` False, every frame takes the `conditional_singles` branch
   (`runtime_optimizations.py:414–425` via `pad_pair_should_draft`, 108–110) — precisely the
   observed delta-1 pattern with clean FC flags. The rel-901.7 malformed emission proves the
   function channel *does* emit garbage outside FC cycles in this session; the wrapper's async
   loop also leaves the model's natural function-channel prediction in
   `gen_function_text[:, t]` on ordinary frames (`nemotron_voicechat_inference_wrapper.py:1756`),
   and a stream of special tokens (e.g. EOTR-adjacent) would decode to the observed empty
   `function_delta`. A new user turn resets the channel to PAD → drafting resumes → matches
   resume-on-speech. **Caveat:** the trace does not record function-channel token ids per
   frame, so this is the leading hypothesis, not proof — which is why telemetry must precede
   any fix.
2. **`sequential_mode` latched — REFUTED for a 510-frame stall.** It is cleared on the first
   ordinary frame once `pending is None` (`runtime_optimizations.py:524–528`); if it were
   latched with a pending row, the very next engine call takes the drain branch
   (`:443–460`) and sets `pending = None`, so the latch cannot survive more than ~2 frames
   while engine calls are demonstrably occurring every frame (delta=1).
3. **Delayed `pending` row stuck — REFUTED.** Cleared at `:455` (bypass drain) and `:478`
   (pair accepted); a stuck pending with `sequential_mode` False would produce delta-2/0
   patterns, not uniform delta-1.
4. **`needs_correction` surviving FC exit — REFUTED.** Cleared at `:456`, `:479`, `:594`; an
   orphaned `needs_correction` (True with `pending`/`assumed` None) *raises* in finalize
   (`:580–582`) — no such error occurred (session ran to its frame limit).
5. **Stale per-call `bypass` flag — REFUTED.** `_voicechat_pad_pair_call_context` is
   overwritten on every prepared frame (`:538–543`); stall frames show `fc_state` fully idle,
   so prepare set `bypass: False` each frame.

**No state in this file is set during FC handling and reset only by user speech** — the
resume-on-speech behavior localizes the defect to the wrapper/model token stream feeding
`_finalize_pad_pair_call`, or (rank 2, cheap to check) a `request_id` keying mismatch between
the engine-side state (`_pad_pair_state(engine, request_id)` at `:387` using the call's
request id) and the wrapper-side state (`:513`, `arguments.get("request_id") or
wrapper.request_id`) — disfavored because `request_id` was `"5"` all session and pairing
worked for 960 s.

## Challenge 3 — Replay false-green risks (plan step 3)

A component replay will pass while the live defect persists if any of these hold:

- **Missing input data (fatal as planned):** the retained failing trace does **not** contain
  per-frame effective function/text token ids or pad-pair state; `function_delta` is lossy
  (specials decode to `""`). A replay that assumes the function channel returns to PAD after
  FC closure will reproduce the *healthy* recovery (as at frame 9369) and green-light a fix
  that never touches the real failure mode. The replay can only be faithful if driven by
  *captured* `effective_text`/`effective_function` per frame — which requires the telemetry
  step to run **before** a failing capture exists to replay.
- **Out-of-band engine calls:** the async FC loop calls `model_llm_interface` directly,
  bypassing `infer_one_step` and hence prepare/finalize (`runtime_optimizations.py:405–408`;
  wrapper `~1620`). The replay must issue those engine-level `generate_next_token` calls too
  (they advance `request_state.generated_tokens` and hit the engine-side latch), or scheduler
  state diverges from live.
- **Accept/reject nondeterminism:** pair acceptance (1 vs 2 tokens, `:307–311`) depends on
  live model logits; the replay must inject recorded accept counts, not recompute them.
- **Timing invisibility:** the harm (85 ms > 80 ms budget, queue debt) is emergent from
  step-time coupling; a state-machine replay must assert on the *decision sequence*
  (buffered/pair/single per frame), never on latency, and the sustained gate remains the only
  timing authority.
- **Granularity:** intra-frame ordering (prepare → 0..n engine calls → finalize; finalize's
  control-barrier check happens *after* the engine call) must be preserved; per-frame replay
  that collapses this ordering will mask the finalize-side latch arming (`:561–577`).

## Challenge 4 — Invariant safety analysis

Wording under review: *"pairing resumes within 1–2 ordinary PAD frames after FC/response
closure or remains disabled for an explicit logged reason."*

Legitimate longer-than-2-frame disable paths that would false-alarm as worded:

1. **Precontrol barriers (`:125–180`)**: near an EOU threshold, `blank_count >= eou_frames−1`
   can hold across many consecutive frames (`rnnt_eou`), and `rnnt_bou` similarly during
   confirmed user onset; each frame re-sets `sequential_mode` (`:530–535`). Post-FC closure
   frequently abuts an EOU window → guaranteed false alarms.
2. **Redirect / external EOS (barge-in family)** (`:133–136`): `_redirect_tokens_queue` and
   `_external_agent_eos_requested` legitimately force sequential operation for the duration
   of the redirect.
3. **Post-FC verbal response**: after response injection, the agent normally speaks; effective
   text is non-PAD for seconds. ("Ordinary PAD frames" partially covers this, but only if PAD
   is defined on **both** channels.)
4. **Control-barrier drain after BOS/EOS** (`:561–572`): a one-to-two-frame legitimate delay
   — inside the allowance, but only if counting starts *after* the drain frame.
5. **Correction frames** (`:462–473`): a `conditional_singles` frame that executes the
   delayed row is not a draft but is healthy.
6. **Session frame limit**: near 12000 frames, resumption may never occur.
7. **This very defect, legitimately:** under conditional policy, a function channel that
   genuinely keeps emitting non-PAD tokens means there *are no* "ordinary PAD frames" — the
   invariant as worded would never trip, silently missing the exact failure it was written
   for.

**Proposed rewording (two-part):**

- *Frame-level invariant:* "On any frame where `fc_state` is fully idle, no
  precontrol/control reason is active, `pending is None`, `sequential_mode` is False, and the
  **previous** frame's effective text **and** function tokens were both PAD, the scheduler
  MUST buffer a draft; otherwise it MUST log `pair_disabled_reason` with the failing
  condition." (This is checkable purely from scheduler inputs and would have fired on frame
  ~11503 in the failed run *only if* the function channel was actually PAD — i.e., it
  doubles as the hypothesis discriminator.)
- *Watchdog (log-only, not a gate):* "If ≥25 consecutive frames (2 s) have fc-idle state and
  agent_control=pad but zero pair-buffer events, emit one state snapshot (full
  `public_pad_pair_status` plus effective token ids)." Log-only, because a hard invariant on
  "resumes within 1–2 frames" cannot distinguish this defect from legitimate non-PAD
  function-channel activity until the telemetry exists.

## Challenge 5 — Recommended implementation sequence

Reordered and trimmed; smallest evidence-driven path:

1. **Preserve evidence NOW (plan step 1, escalated to blocking).** Copy
   `…-converted-r2/` and `…-converted-r2-traces/` (and r4 equivalents) out of `/tmp` to a
   durable location (e.g. `~/.local/state/nemotron-voicechat/retained/` or repo artifacts).
   The failed trace has no copy outside `/tmp`. Extract the compact window
   (frames 11380–12000 + the healthy 9300–9420 control) as planned.
2. **Telemetry (plan step 2), extended:** per-frame `effective_text`/`effective_function`
   token ids, pair decision (buffered/pair/single-bypass/single-conditional/drain),
   `previous_effective_pad`, `sequential_mode`, `pending` presence, `needs_correction`,
   transition reason — added to the `model_step` trace record (the counters already exist in
   `public_pad_pair_status`, `:598–638`; this is sampling, not behavior change). This is the
   root-cause discriminator between "scheduler ignores PAD frames" and "frames are not PAD."
3. **Targeted live repro *before* building the replay harness** (new step; cheapest
   discriminator): drive the observed trigger signature — a second FC within the same turn
   following a malformed function-channel emission — via the repeated-tool stress script.
   The failed session produced this shape twice in 1200 s (turn_14 recovered, turn_16 did
   not); with telemetry from step 2, one reproduction answers the hypothesis question
   outright. If it reproduces, the 20-minute-iteration problem the replay was meant to solve
   mostly disappears.
4. **Replay harness (plan step 3), demoted to fallback** if live repro is elusive — built
   from a telemetry-complete failing capture, subject to the fidelity requirements in
   Challenge 3, asserting on decision sequences only.
5. **Localize, then fix (plan steps 4–5), re-scoped:** if telemetry shows the function channel
   is non-PAD post-FC (expected), the fix is **not** scheduler recovery hardening — options
   (post-FC function-channel token policy, or counting fc-idle non-PAD function specials as
   draft-eligible) change qualified pairing semantics (`function_channel_rejection` is a
   frozen contract item in the session capability report) and need their own review. Of the
   proposed step-5 items, only the invariant/logging (as reworded above) and the
   "drain pending once" (already implemented at `:443–460`) are safely scheduler-only;
   "clear stale scheduler-only state" and "rebase eligibility from actual effective tokens"
   describe what the code already does (`:522–528`, `:557–560`) and would be no-ops.
6. **Plan step 6 (no changes to tools/schemas/prompts/timeouts) — keep**, with one caution:
   if root cause lands in the wrapper's post-FC function-channel handling, an exception to
   "scheduler-only" scope will be required and must go through re-qualification, since it
   touches `infer_one_step` semantics used by the 1172-call pairing gate.
7. **Gate ladder (plan step 7) — keep, add two controls:** (a) the r4 "passing" baseline is
   near-worthless as an FC control (1 FC in 1200 s) — the sustained gate must require a
   minimum executed-FC count and at least one repeated-same-turn FC to count as covering this
   defect; (b) add the healthy-recovery assertion from the same failed trace (pairing resumed
   ≤3 ordinary frames after turn_14's double-FC) so the fix cannot regress the working path.

---

## Findings

### BLOCKER

- **B1 — Step 5 targets state the evidence says is not poisoned.** All four scheduler-local
  hypotheses are refuted with line cites (`runtime_optimizations.py:443–460, 455–456,
  478–479, 522–528, 538–543, 580–582, 594`); the stall pattern (510 uniform delta-1 frames,
  fc flags idle, `previous_effective_pad` recomputed every frame at `:557–560`) is only
  consistent with the conditional policy correctly refusing to draft on non-PAD effective
  inputs. **Fix:** make steps 2→3(repro) produce telemetry-confirmed root cause before
  authorizing any step-5 behavior change; re-scope step 5 per Challenge 5 item 5.
- **B2 — The replay (step 3) cannot be faithful from existing data and its failing-trace
  gate can false-green.** The retained trace lacks effective token ids and pair decisions;
  a replay seeded with assumed-PAD post-FC frames reproduces the healthy path (proven to
  exist in the same trace at frames 9366–9369). **Fix:** reorder — telemetry first, targeted
  live repro second, replay only from a telemetry-complete capture, decision-sequence
  assertions only (Challenge 3 requirements).

### IMPORTANT

- **I1 — Evidence at risk:** the only copy of the failing trace is under volatile `/tmp`
  (`…-converted-r2-traces/model/aaee56ee-…`; absent from
  `~/.local/state/nemotron-voicechat/traces/model/`). Step 1 must name a durable destination
  and run first.
- **I2 — Invariant as worded false-alarms and under-detects** (Challenge 4): precontrol EOU/BOU
  windows (`:125–180`) legitimately hold sequential mode for many frames; and under this
  defect no "ordinary PAD frame" ever occurs, so the invariant never trips. Use the two-part
  rewording (frame-level conditional invariant + 2 s log-only watchdog).
- **I3 — "Resumed with user speech" was being read as supporting latched-flag hypotheses; it
  does the opposite.** No user-speech path clears any pad-pair state in
  `runtime_optimizations.py`; user BOS *arms* barriers (`:561–572`, `:530–535`). Resume is
  explained by the model's channels resetting on a new turn — record this discriminator in the
  investigation notes so the fix isn't aimed at flag-clearing.
- **I4 — Passing r4 run is a weak control** (1 executed FC vs 8 in r2; zero repeated calls).
  The gate ladder's sustained step must require minimum FC coverage including a
  repeated-same-turn call, else a recurrence stays invisible until production.

### NIT

- **N1 —** Cheap discriminator available today: sample `public_pad_pair_status()`
  (`:598–638`) once per second into the event stream; growth of `conditional_singles` vs
  `sequential_bypass` during a stall distinguishes hypothesis 1 vs 2 without new state.
- **N2 —** Verify the dual `request_id` keying (`:387` call-side vs `:513` wrapper-side) with
  a one-line assert in telemetry; disfavored (constant `"5"` all session) but two lines of
  logging retire it permanently.
- **N3 —** The report's "85.6 ms" and "62 s" should be restated precisely in the extracted
  window artifact: measured 82.7–84 ms mean over frames 11501–11994 (with excursions to 88),
  stall 961.7→1022.74 s (61.0 s SOTC-to-first-buffer), plus a separate 1.55 s wall stall at
  rel 1020.7 contributing to the 2979.7 ms peak debt.

## Recommended first implementation step

Execute step 1 immediately (durable copy of r2 + r2-traces out of `/tmp`), then land the
step-2 telemetry (per-frame effective token ids + pair decision + pad-pair state in
`model_step`) as a trace-only change, and attempt the targeted repeated-FC-same-turn live
repro before investing in the replay harness.

## Telemetry-boundary review — 2026-08-06

**Verdict: APPROVE WITH CHANGES** (0 BLOCKER, 3 IMPORTANT, 4 NIT). The uncommitted diff
(runtime_optimizations.py, server.py, cli.py + tests) delivers the mandated per-frame
effective-token capture, prepared-vs-out-of-band discrimination, and request-id check as
opt-in, log-only telemetry. No finding requires withholding the instrumentation from a
diagnostic run; the IMPORTANT items should land before results from it are treated as
authoritative evidence.

### IMPORTANT

- **I-T1 — Telemetry is not exception-hardened.** `_record_pad_pair_decision`
  (`runtime_optimizations.py:364–429`) executes *after* the engine call has produced its
  result but *before* the return on every branch (`:523`, `:560`, `:588`, `:618`, `:642`,
  `:667`); an exception there discards a computed generation result and kills the frame.
  Same for `_pad_pair_trace_diagnostics` inside `step()` (`server.py:1548`). Today both are
  de-facto non-throwing (all keys are created by `_pad_pair_state` in the same process), but
  the prior review's mandate was a guarantee, not an accident. Smallest fix: wrap the body of
  `_record_pad_pair_decision` after the `enabled()` check, and the body of
  `_pad_pair_trace_diagnostics`, in `try/except Exception` that increments a
  `trace_record_errors` counter and returns.
- **I-T2 — Two PAD-id sources feed the finalize-consistency check.** Server-side effective
  tokens use `stt_model.text_pad_id` (`server.py:1354`); `last_effective_tokens` from
  `_finalize_pad_pair_call` uses the config-gated `_pad_pair_token_id(engine)`
  (`runtime_optimizations.py:755–769`). `effective_tokens_match_finalize` compares the
  `"pad"` key across both (`server.py:1397–1400`), so any divergence between
  `voicechat_pad_pair_token_id` and `text_pad_id` reports a spurious mismatch and makes
  server-side `both_pad` disagree with the scheduler's `previous_effective_pad` — exactly
  the signal this investigation reads. Smallest fix: drop `"pad"` from the compared-keys
  tuple at `server.py:1399`, or read the pad id from `snapshot["last_effective_tokens"]`.
- **I-T3 — No test drives the wrapped scheduler with tracing on.** The new unit tests call
  `_record_pad_pair_decision` directly (`test_public_runtime_optimizations.py:168–251`) and
  the watchdog test mocks `consume_pad_pair_trace`
  (`test_realtime_web_server.py:49–108`), so a regression that deletes one of the six
  in-branch record calls would still pass the full suite — the capture path from live branch
  to drained event is never asserted end-to-end. The in-container rerun (see test line below)
  proves the real NVIDIA scheduler module loads and the unit pieces behave under real Torch;
  it does not close this wiring gap, because the same test list runs there. Smallest fix: one
  async test that installs `_install_public_pad_pair_engine` over a fake backend (the
  `Backend`/`RequestState` pattern at `test_public_runtime_optimizations.py:45–77` already
  exists), runs buffered→pair frames, and asserts the drained decision sequence.

### NIT

- **N-T1 — `call_origin` is positional, not identity-based**
  (`runtime_optimizations.py:410`): an async-FC-loop call that interleaves between
  `_prepare_pad_pair_call` and the prepared frame's own engine call (both have await points)
  claims `engine_calls == 0` and is labeled `prepared_frame`, and when the context attribute
  is absent the increment at `:382` mutates a throwaway `{}` so every call reads 0. Sequence
  numbers and `context_frame_idx` keep the stream reconstructible; document the heuristic or
  add a `context_generation` counter bumped in `_prepare_pad_pair_call`.
- **N-T2 — `request_id_match` defaults to `True` when no context exists**
  (`runtime_optimizations.py:405–407`), which hides the discriminator for pre-prepare calls;
  emit `None` (unknown) instead.
- **N-T3 — Default-path trace schema change:** `model_step` events now always carry
  `runtime_diagnostics: null` when tracing is off (`server.py:2173`). Log-only and
  parser-benign, but qualified-run trace records are no longer byte-identical; omit the key
  when `None`.
- **N-T4 — Watchdog false-positive audit (challenge 4): clean.** `watching` requires
  `agent_control == "pad"` (`server.py:1376–1378`), i.e. the effective agent *text* token is
  PAD, so agent-talking turns (text non-PAD → label `None`), BOS/EOS frames
  (`agent_bos`/`agent_eos`), and FC windows (all five `function_calling` keys at
  `server.py:1271–1281` match the idle test at `:1367–1373`, including `background_active`)
  never count. User-silence/idle-agent steady state alternates buffered/pair frames, so the
  counter oscillates 0↔1; session start reaches its first buffer well inside 25 frames.
  Reset on buffer resume and single-shot emission per episode are correct
  (`server.py:1381–1393`); output goes only to `trace.event` (`server.py:2175–2181`) — truly
  log-only, nothing reaches the websocket or control flow. The one deliberate residual:
  a request-id mismatch between `_request_id_for_stream` and the engine's request key yields
  `state_found: False` with empty events and a firing watchdog — acceptable, since that
  mismatch is itself hypothesis-relevant (prior N2), but read `state_found` before trusting
  an empty event stream.

### Perturbation analysis (challenge 1)

- **Default path (flag off): no new device sync, no new lock, no material allocation.**
  Additions are one `os.environ` lookup per engine call (`runtime_optimizations.py:378`),
  one `len()` of a Python list (`:516`, `:546`), four extra keys plus one empty `deque` per
  request state (one-time, `:356–359`), and one function call + env lookup per server frame
  (`server.py:1341`). `_prepare`/`_finalize` behavior is unchanged (the hoisted
  `precontrol_reason = None` at `:721` only feeds the context dict). No telemetry write
  touches a scheduler-read key; `consume_pad_pair_trace` drains only `trace_events`.
- **Trace path (flag on): two new `.item()` D2H syncs per frame** at `server.py:1352–1353`,
  on the same `gen_text`/`gen_function` elements `_finalize_pad_pair_call` already `.item()`s
  each frame (`runtime_optimizations.py:753–754`), so the stream is already drained and each
  costs microseconds post-step — immaterial against the ~85 ms frame but it must be stated:
  tracing is not sync-free, only sync-cheap. Per-event dict allocations are bounded by the
  128-deque (`:426–428`) with an explicit drop counter. No locks added anywhere; deque
  append/popleft are GIL-atomic, so the async-FC-loop writer vs. server-drain reader worst
  case is an event surfacing one frame late, never corruption (challenge 2's ordering is
  preserved by the per-request monotonic `trace_sequence`).

### Opt-in plumbing (challenge 5)

`--trace-pad-pair` defaults off (`cli.py:1142–1146`), threads through `command_up` →
`model_container_command` (`cli.py:812–819`, `:531–532`) as `-e
VOICECHAT_NANO_PAD_PAIR_TRACE=1` on the docker invocation, and no other command path sets the
variable — `test_model_command_enables_pad_pair_trace_only_when_requested`
(`test_cli.py:59–67`) pins both directions. Qualified default behavior is unchanged apart
from N-T3's null field.

### Test results

Host focused trio: 82 passed, 1 skipped in 0.74 s (skip = NVIDIA-Speech importorskip at
`test_public_runtime_optimizations.py:34`). Authoritative evidence per the coordinator: the
same trio rerun **inside the qualified runtime image** (repo mounted read-only,
`--network none`, real Torch/NVIDIA scheduler modules) — **82 passed, 1 unrelated skip in
33.76 s**. Ruff clean on all three touched source files (verified). The in-container run
retires any environment-stub concern for the unit assertions; I-T3 stands only as the
branch-wiring end-to-end gap, which no environment can fix without the missing test.

---

## Telemetry delta verdict — 2026-08-06: APPROVED. CLEARED for the live
failure-capture campaign.

Verified on the actual diff: **I-T1** closed with dual-layer exception
isolation — the engine recorder is wrapped with a logged-drop path whose
logging call is itself guarded, and the server collector is separately
isolated with drop counters ("caller owns exception isolation"
documented); diagnostics can no longer discard an inference result under
any exception. **I-T2** closed — the server's effective-token comparison
now consumes the scheduler finalize snapshot (`last_effective_tokens`)
as its source, eliminating the dual-source pad-id mismatch risk.
**I-T3** closed via the real installed-wrapper test driving buffered →
pair_accepted and draining the recorded sequence. NITs landed
(positional `call_origin` labels, `request_id_match: null` on missing
context, `runtime_diagnostics` omitted when tracing is off). Focused
pad-pair/trace tests verified locally (15 passed / 1 skipped);
authoritative in-container run 86/86 under `--network none` with the
real scheduler. The telemetry is cleared: proceed to the live
failure-capture campaign (repeated-FC-same-turn repro with
`--trace-pad-pair`).
