# Phase 7 Static Adversarial Review — Qualification Harness & Live-Suite Wiring

- **Date:** 2026-08-04
- **Branch:** `codex/pocket-tts-typed-input` (uncommitted Phase 7 changes)
- **Scope:** `tools/qualification/{restart_cycle_gate,multiturn_strict_v3,sustained_strict_v3,transcribe_sustained,run_live_suite}.py`; new `tests/runtime/test_{restart_cycle_gate,multiturn_qualification,sustained_qualification,transcribe_sustained,live_suite,cli}.py`; live browser gate in `tests/e2e/test_browser_smallwebrtc_e2e.py`; runtime-state/orphan recovery and `./voicechat test --live` in `src/nemotron_voicechat_runtime/{artifacts,cli}.py`. Static review only — the public image build is still in flight and no live evidence was produced.

**Verdict: REJECT** (static boundary). Three blockers: the containerized ASR step cannot resolve its input paths and will fail on every run; the sustained gate is structurally blind to silent/missing responses to typed turns (the project's documented worst failure mode); and the live browser gate can print PASS with zero assistant output and a non-functioning typed-input path. All fixes are small and local. Everything else in the harness held up well under challenge (see "Verified sound").

**Live-evidence gates still pending** (out of scope for this static verdict, required before Phase 7 closure):
1. Restart cycles (3×SIGINT + SIGKILL + `./voicechat down` recovery) on the real public image.
2. Loopback negative test executed live on a multi-interface DGX host.
3. 15-minute sustained qualification with typed interleaving on the shipped container.
4. Dual-source weight agreement (downloaded vs converted) — no harness exists yet (finding I-8).
5. Clean-machine run with no undeclared files (plan §7.2) — procedural, not automatable here.
6. Browser gate against the public stack, after B-3 is fixed.

---

## BLOCKER

### B-1. Containerized ASR gate reads host-absolute audio paths that do not exist in the container
`tools/qualification/transcribe_sustained.py:143` does `source = Path(response["audio_path"])`. Those paths are written by `multiturn_strict_v3.py:142` and `sustained_strict_v3.py:185` as **host** absolute paths (`args.output / "response-001.wav"`). `run_live_suite.py:84-98` runs the transcriber inside the runtime image with the run dir mounted at `/qualification` and `--network none`; the host path is not visible there, so `read_and_resample` raises `FileNotFoundError` on the first response and the suite always fails. (Fails loud, not silent — but the shipped `./voicechat test --live` can never pass.)

**Fix** (transcribe_sustained.py, in the response loop):
```python
source = Path(response["audio_path"])
if not source.is_file():
    source = args.run_dir / source.name
```

### B-2. Sustained gate cannot fail on silent/unanswered typed turns
`sustained_strict_v3.py:249-255` — pass criteria are `not errors`, all frames sent, `len(typed_completed) >= max(1, len(typed) - 1)`, **`bool(responses)`**, queue passed, session closed. `injection_finished disposition="completed"` (server.py:2458) means the synthetic audio finished entering RNNT — not that the model answered. With a silent mic, ~15 typed prompts should yield ~15 responses; the gate passes with **one**. The prior sustained failure mode was exactly silent turns (conditional pairing), and `transcribe_sustained.py` only rates the responses that *were* received (`transcribe_sustained.py:184-198`), so 13 missing responses out of 15 still pass end to end. The plan's "expected silent-turn base rate" (plan §7.5) is therefore unenforced against the dominant failure shape.

**Fix** (sustained_strict_v3.py report): count responses and gate them against completed typed jobs with the same 1/15 base rate, e.g.:
```python
unanswered = len(typed_completed) - len(responses)
"passed": ( ... and unanswered <= math.ceil(len(typed_completed) / 15) ... )
```
Better: attribute responses to typed jobs via `turn_id`/`job_id` on `conversation.item.input_audio_transcription.completed` (protocol.py:212-219 carries `source`/`job_id`) and report per-job answered/unanswered.

### B-3. Live browser gate can PASS with no assistant output and a broken typed path
`tests/e2e/test_browser_smallwebrtc_e2e.py` (new `test_live_playground_typed_turn_and_audio`):
- It types the prompt immediately after the Disconnect button appears — no bot-READY wait (prior lesson: readiness/transcript-completeness precondition). A dropped send only fails via the 20 s text wait, and worse:
- `page.get_by_text(prompt)` matches the *locally echoed sent message* in the Playground UI — it proves the textbox worked, not that the server accepted/injected anything.
- The audio assertion is `stats["inbound"] > 4_000` bytes accumulated over up to 90 s of an always-on WebRTC audio track. Keepalive/comfort-noise/silence payloads plausibly exceed ~44 bytes/s; this does not prove assistant speech. `tracks >= 1` and `outbound > 0` are true from connection time.
- Net: bot never responds, typed injection rejected server-side → test still passes.

**Fix:** (a) before sending, wait for a readiness indicator (RTVI ready event surfaced in the UI, or poll the model `/health` `active_client: true` via the suite); (b) assert the **assistant** reply, not the echo — e.g. `page.get_by_text(re.compile("five", re.I))` for the "two plus three" prompt, scoped to the bot transcript element; (c) make the audio check differential: snapshot `inbound` after the reply text appears and require a delta consistent with ≥1 s of speech, or decode track RMS via a WebAudio `AnalyserNode`.

## IMPORTANT

### I-1. Restart gate has no cold-start-budget assertion
`restart_cycle_gate.py:190` — the only time bound is `--startup-timeout` (default 900 s). The plan (§7.7) requires "health within the cold-start budget for three consecutive cycles." `ready_seconds` is recorded (lines 143, 147) but never compared to anything, so a restart that degrades from 90 s to 850 s passes. The known residual-GPU-state hang is caught only if it exceeds 15 minutes.
**Fix:** add `--ready-budget-seconds` (required, from the documented cold-start budget) and fail any cycle with `ready_seconds > budget`; keep `--startup-timeout` as the hard abort.

### I-2. Restart gate leaks a running stack when SIGINT teardown times out
`restart_cycle_gate.py:130-136` — `finish()` sends SIGINT and `wait(timeout=60)`; on `TimeoutExpired` the exception propagates (log closed via `finally`) with the launcher **still running**. `main()` then writes a failed report and exits — leaving the model container and Pipecat up, which poisons any subsequent gate run.
**Fix:** in `finish`, catch `TimeoutExpired`, `process.kill(); process.wait(timeout=10)`, then raise. Note also that a launcher SIGINT teardown legitimately runs `docker stop --time 20` plus Pipecat shutdown (cli.py:485-499), so 60 s is tight; 90 s is safer.

### I-3. Loopback negative probe treats HTTP errors as "unreachable" and skips silently with no interfaces
`restart_cycle_gate.py:64-71` — `urllib` raises `HTTPError` (a `URLError`/`OSError` subclass) for any non-2xx response, which hits `continue`: a model port published on `0.0.0.0` that answers 404/500 is reported as loopback-only. And if `host_non_loopback_addresses()` (lines 55-61, `gethostname()`-based, may miss secondary interfaces) returns `[]`, the check passes vacuously. The unit test `tests/runtime/test_restart_cycle_gate.py:36-39` — named `test_non_loopback_probe_rejects_reachable_model` — monkeypatches the address list to `[]` and asserts nothing about rejection; it certifies the silent-skip path.
**Fix:** catch `urllib.error.HTTPError` first and treat it as *reachable*; record probed addresses in the report and require at least one (or fall back to enumerating interface addresses); add a real rejection test with a stub opener returning a response.

### I-4. `./voicechat down` can abort before removing the stale container
`cli.py:611-621` — inside the runtime-state block, `json.loads` on a corrupt file raises after the `finally` unlink, and `_stop_orphan_pipecat` (cli.py:510-521) has a TOCTOU: the PID can exit between `_owned_pipecat_process` and `os.kill`, raising `ProcessLookupError`. Either way `down` crashes before `docker rm -f` (line 621) — the exact recovery the SIGKILL gate (`restart_cycle_gate.py:153-164`, `check=True`) depends on.
**Fix:**
```python
try:
    state = json.loads(...)
    if ...: _stop_orphan_pipecat(state["pipecat_pid"])
except (ValueError, OSError) as exc:
    print(f"runtime-state cleanup skipped: {exc}")
finally:
    layout.runtime_state_file.unlink(missing_ok=True)
```
and swallow `ProcessLookupError` around both `os.kill` calls in `_stop_orphan_pipecat`.

### I-5. No active_client pre-check before WS gates (single-client contract)
The server hard-rejects a second client with fatal `server_busy` (server.py:2222-2240) and exposes `active_client` in `/health` (server.py:2215), but `multiturn_strict_v3.py:87` and `sustained_strict_v3.py:100` connect blindly and `run_live_suite.py:149-174` runs multiturn immediately after the browser pytest exits — while the Pipecat bot's model WS teardown may still be in flight, and with no protection against an operator's live demo session. Failure is loud (protocol check at multiturn_strict_v3.py:95) but misattributed and flaky.
**Fix:** in `run_live_suite` (and/or each client), poll `http://127.0.0.1:{model_port}/health` until `active_client` is `false` (bounded, ~30 s) before each WS gate.

### I-6. Expected-content fields are recorded but never asserted
`multiturn_strict_v3.py:146-155` stores `expected_input_text`, `displayed_input_text`, `expected_response_any`; the pass criterion (lines 172-176) is only non-empty text + non-empty audio + session closed, and nothing else in the tree consumes those fields (verified by grep). `transcribe_sustained.py:171` computes WER against the model's **own** text channel only. So input-transcript correctness, conversational memory (codeword), and exact spoken tool results are unfalsifiable by these gates.
**Fix:** in multiturn's report: fail a turn when `word_error_rate(expected_input_text, displayed_input_text)` exceeds a threshold, and when `expected_response_any` is non-empty require one normalized candidate substring in `text` (and, post-ASR, in the transcript).

### I-7. `run_checked` has no timeouts — a hung sub-gate hangs `./voicechat test --live` forever
`run_live_suite.py:18-32` — no `timeout=` on any `subprocess.run` (browser pytest, multiturn, sustained, docker ASR). A wedged docker pull/GPU hang blocks indefinitely with no report written.
**Fix:** add per-step `timeout=` (e.g. browser 600 s, multiturn 900 s, sustained `duration+drain+600`, ASR 1800 s); `TimeoutExpired` then flows into the existing failure-report path.

### I-8. Missing Phase 7 requirements: dual-source agreement and graceful session limit
- Plan §7.6 (downloaded vs converted weights must agree on component outputs and verdicts): no harness anywhere in the diff; `run_live_suite.py` runs once against one cache root with no comparator.
- Graceful session limit: `max_session_model_frames=12_000` (server.py:2155) ≈ 960 s of model frames; the sustained default (900 s, `sustained_strict_v3.py:283`) plus drain never crosses it, and no gate asserts the `session_position_limit` close path (server.py:2330-2337). The limit shutdown is exercised by zero live gates.
**Fix:** (a) add a `--compare` mode or a small comparator script taking two suite output dirs and diffing verdicts + component metrics; (b) add a short session-limit gate (silence + one typed turn, duration > limit) asserting `session.closed` with `reason="session_position_limit"` and clean socket close.

## SUGGESTION

- **S-1.** `sustained_strict_v3.py:150-157`: handle `input_text.rejected` in the receiver (record `{"rejected": reason}`); today a rejected job is indistinguishable from a lost one and only soaks up the single allowed non-completed slot.
- **S-2.** `sustained_strict_v3.py:241-245`: the `> 0` filter drops idle/skip steps, so the debt integral gets no sub-cadence credit and there is no completeness check; also assert `len(metrics)` ≈ frames observed and cross-check wall-clock arrival using the recorded `client_received_monotonic_s` (line 145) so a server that stops emitting metrics can't look healthy.
- **S-3.** `multiturn_strict_v3.py:193`: default `--trailing-silence-frames 25` = 2.00 s vs the EOU-derived ≈2.08 s; a borderline EOU makes turns time out spuriously. Default to 30.
- **S-4.** `cli.py:510-521`: PID-reuse window between the cmdline check and `os.kill` remains (mitigated by the cmdline match, cli.py:502-507); comparing `/proc/<pid>/stat` start time recorded in runtime-state would close it.
- **S-5.** `cli.py:652-657`: `test --live` runs the *full* pytest tree (including `tests/e2e`) before the stack exists; the live test skips only because the env var is absent. Pass `--ignore tests/e2e` in the pre-pass and let the suite own e2e explicitly.
- **S-6.** `transcribe_sustained.py:128` `--max-wer 0.50` is generous even for TTS-vs-text-channel comparison; document the calibration or tighten once live distributions exist.

## Verified sound (held under challenge)

- **Protocol conformance (checklist 2):** client event names/fields exactly match the server: `input_text.request` handling (server.py:2701-2761), `input_text.accepted/rejected/injection_started/injection_finished` + disposition set (protocol.py:138-155), `session.created.protocol` (protocol.py:73-79), response lifecycle incl. `response.output_audio.delta` encoding fields (protocol.py:237-353), `response.function_call_arguments.done.call_id` (protocol.py:364-371), `voicechat.metrics.server_step_ms` (server.py:2053-2070), `session.closed` (protocol.py:107-108).
- **Pacing (checklist 3):** both clients pace by absolute monotonic schedule, immune to per-frame sleep drift (`sustained_strict_v3.py:200-207`, `multiturn_strict_v3.py:62-66`).
- **Drift detection:** explicit thresholds (late-mean ≤80 ms, slope ≤1 ms/min, implied serial debt ≤1 s; `sustained_strict_v3.py:69-74`), a fail-closed "insufficient settled frames" branch (line 57), and a unit test proving positive drift fails (`tests/runtime/test_sustained_qualification.py:17-23`).
- **Silent-response classification:** −55 dBFS near-silent detection with an explicit 1/15 base-rate gate and fail-closed behavior when all/zero responses exist (`transcribe_sustained.py:145-149, 184-198`) — for the responses that arrive (see B-2).
- **Audio attribution (checklist 4):** response WAVs are assembled only from `response.output_audio.delta` between `created`/`done`; typed synthesis enters the *model input* path (`process_pcm(..., source="typed")`, server.py:2399-2415) and is never echoed to output, so user/typed audio cannot leak into assistant ASR.
- **Exit-code integrity (checklist 1/9):** every sub-step in `run_live_suite` is `check=True`; the transcriber both mutates `report.json` and exits nonzero on failure (`transcribe_sustained.py:201-204`); all three tools write `passed:false` reports and `SystemExit(1)` on exception (e.g. `restart_cycle_gate.py:192-201`); the suite's `"passed": True` is reachable only after every checked step.
- **Restart/teardown semantics (checklist 5):** the wrapper `exec`s Python (`voicechat:15`) so signals hit the real CLI; `up` refuses stale containers and busy ports (cli.py:532-540), which is what makes "restart immediately" a real teardown test; SIGKILL leaves an orphan Pipecat holding port 7860 plus the container, so recovery genuinely requires `./voicechat down` (runtime-state written at cli.py:575-586, consumed at 611-617, removed in the `finally` at 604); container runs `--rm --init` (cli.py:408-414); Popen handles (not raw PIDs) are signaled in the gate.
- **Health contract:** restart gate asserts `ready` + protocol v3 + `auth_required is False` + `typed_input.ready` each cycle (`restart_cycle_gate.py:105-111`), matching server.py:2202-2217.
- **Typed interleaving (checklist 8):** sustained sends a typed prompt at frame 0 and every 60 s, cycling memory + tool prompts (`sustained_strict_v3.py:27-32, 208-216`).
- **Suite serialization (checklist 6/9):** restart gate → stack up → browser → multiturn → sustained run strictly sequentially; the stack is torn down before the GPU ASR container starts (`run_live_suite.py:175-190`).
- **ASR isolation:** `--network none`, repo and model mounted read-only (`run_live_suite.py:80-98`), with a unit test pinning both properties (`tests/runtime/test_live_suite.py:36-46`).
- **Orphan PID identity:** cmdline-gated signaling with a unit test covering accept and refuse paths (`cli.py:502-507`, `tests/runtime/test_cli.py:59-65`).

## Pytest (read-only)

`.venv/bin/python -m pytest tests/runtime/test_restart_cycle_gate.py tests/runtime/test_multiturn_qualification.py tests/runtime/test_sustained_qualification.py tests/runtime/test_transcribe_sustained.py tests/runtime/test_live_suite.py tests/runtime/test_cli.py -q` → **14 passed in 0.08s**.

---

## Delta re-review — 2026-08-04, blocker fixes + static amendment

### Verdict: APPROVE (static boundary). All blockers and IMPORTANT items
verified closed. Live-evidence gates remain pending as listed below.

- **B-1 CLOSED:** `resolve_audio_path(run_dir, recorded)`
  (transcribe_sustained.py:115, used :159) — audio paths resolve relative
  to the run directory, correct under the `/qualification` container mount.
- **B-2 CLOSED:** the sustained gate now requires, for every result,
  `audio_bytes > 0`, non-empty `text`, and `semantic_match`, with explicit
  numeric gates: delivery drift (`last ≤ 80 ms`, `slope ≤ 1.0`,
  `peak_debt ≤ 1000`) and queue-lag growth (`last ≤ 1000`,
  `last − first ≤ 500`, `slope ≤ 100`) — silent typed turns and linear
  EarTTS drift now fail loudly. Response attribution keys off
  `response.created` events.
- **B-3 CLOSED:** the browser gate now enforces runner readiness (explicit
  `TimeoutError`), fails if the Playground text input never becomes ready,
  and extracts/asserts assistant PCM plus instruction content — it can no
  longer pass on comfort noise.
- **I-3 CLOSED:** the loopback probe's logic is now correct — an
  `HTTPError` (a real HTTP response) raises "reachable through
  non-loopback"; only `OSError`/`URLError` count as unreachable, with
  interface enumeration.
- **I-4 CLOSED:** `cli.py` handles `ProcessLookupError` on both kill paths
  (:524, :534) so `down` reaches container removal after SIGKILL.
- New surfaces spot-checked: `compare_candidate_reports.py` aggregates
  per-gate `passed` fields for the dual-source agreement verdict; runtime +
  release suites pass locally (156/1 in the subset run; reported full suite
  245 passed / 2 skipped, lint clean).

**Pending live evidence (unchanged, authoritative):** restart cycles on the
real public image; live loopback negative test; 15-minute sustained gate
with typed interleaving on the public stack; dual-source
downloaded-vs-converted agreement; clean-machine run; browser gate against
the public deployment; final `--no-cache` rebuild.

---

## Delta 2 — 2026-08-04, self-synthesized fixture usability change

### Verdict: APPROVE (static). No regression of prior Phase 7 gates.

- **WAV construction correct:** 16 kHz (`INPUT_RATE = 16_000`), 16-bit mono
  via the `wave` module; the browser microphone fixture prepends
  explicit leading silence built as zero PCM (the documented
  READY-race lesson, with an in-code comment explaining why).
- **Independence preserved:** fixtures are synthesized by Pocket TTS, not
  by the Voicechat model — no coupling of gate inputs to model output.
- **Exit semantics fail-closed:** the exception path builds a report
  without `"turns"`, so `SystemExit(0 if "turns" in report else 1)` cannot
  exit 0 on failure (my earlier concern resolves cleanly).
- **Container paths consistent with the B-1 fix:** run dirs resolved then
  mounted at `/qualification`; in-container invocations use container
  paths throughout (`run_live_suite.py:106-127, 172-198`).
- **Cross-modality assertion is real:** the browser test's memory probe
  requires the typed turn to render content spoken via the generated
  microphone fixture ("sapphire" cross-modality proof), not an echo.
- Focused suites pass (4/4 here; reported full 245 passed / 2 skipped,
  lint clean).

Static boundary remains fully approved; the live ladder on the public
image is the sole remaining evidence.

---

## Delta 3 — 2026-08-04, hosted CI workflow

### Verdict: APPROVE. Security and reproducibility posture is correct.

- **Least privilege:** `permissions: contents: read` only; no secrets
  consumed; triggers limited to PRs and pushes to main.
- **Supply chain:** both actions pinned by full commit SHA (reported
  verified against official remotes); uv pinned to `0.11.21`; environment
  from `uv sync --frozen` against the committed lock; Python taken from
  the repo's `.python-version` (the Phase 5 `uv python pin`).
- **Command parity:** the gate steps are exactly the documented commands
  (`./voicechat test`, `uv run --frozen ruff check .`) — no CI-only
  variant that could drift from the release checklist.
- **Bounded:** `timeout-minutes: 20` on a 245-test suite that runs in
  seconds locally.

Notes (non-gating): (1) x86_64 GPU/Docker/model independence of
`./voicechat test` is asserted by design (the 2 local skips are
container-only) but is PROVEN only by the first green CI run on
ubuntu-24.04 — treat that first run as the closing evidence, and if any
test needs Docker it must skip by marker, visibly. (2) Consider a
`concurrency` group to cancel superseded runs; cost-only.
