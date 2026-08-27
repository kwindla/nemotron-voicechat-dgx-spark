# Test-matrix amendment: response-completion coverage

Status: plan amendment, authored after the production no-text watchdog
repair. Motivating failure: a pre-existing one-dimensional liveness bug
truncated live responses mid-audio for weeks while **every automated gate
passed**. This amendment closes the blindness, not just the bug.

## Why the matrix missed it

1. **Configuration blindness.** Every campaign ran
   `VOICECHAT_QUAL_NO_TEXT_WATCHDOG_OVERRIDE_FRAMES=160`; production runs
   30. The gates never exercised the production failure window.
2. **No completion gate.** Suites assert emptiness, fatals, latency,
   decisions — never *whether a response finished or was cut off*. A
   truncated response has text and audio, so it passes every existing
   check.
3. **An adjudication rule absorbed it.** The browser plan's long-rendition
   truncation rule treats truncation as the known stochastic model
   limitation. Written before we knew a runtime watchdog could produce the
   same symptom, it has been explaining away a defect.

## Required additions

### A. Live response-completion gate (new, load-bearing)

Add to `tools/qualification/rapid_turn_settlement_suite.py` and the Step 2
browser adjudication a gate computed from retained model traces:

- **FAIL** on any response whose closing `eos_reason` is a watchdog reason
  (`no_text_since_bos_watchdog`, `no_audio_since_bos_watchdog`) **while the
  response was acoustically live** — defined as: audible output
  (`output_audio.rms_dbfs > threshold_dbfs`) within the 5 frames preceding
  the watchdog request.
- **RECORD, do not fail**, watchdog closures with no audible audio in that
  window (the legitimate dead-response rescue) and `decoded_silence_watchdog`
  closures (the normal completion path), with counts per session.
- Report per session: total responses, completion reasons histogram, count
  of text-quiet-but-audible frames, and max consecutive such frames (the
  direct regression signal — pre-fix this could not exceed the no-text
  threshold; post-fix it must be allowed to).

### B. Production-value coverage (configuration honesty)

The behavioral suites must run at least once per campaign with the
**production** watchdog values (`VOICECHAT_WEB_AGENT_NO_TEXT_FRAMES=30`, no
qualification override). Where the override is needed for a specific
measurement, that campaign must state which behaviors it therefore cannot
gate. No gate may claim production behavioral coverage from an
override-configured run.

### C. Split the browser truncation adjudication

The long-rendition truncation rule must distinguish:

- **model-silence truncation** — EarTTS stops producing audible audio and
  `decoded_silence_watchdog` closes: the documented known limitation,
  adjudicated as today; and
- **watchdog truncation** — a watchdog closes while audio is live: a
  DEFECT, never adjudicable as expected model behavior.

Both classes are reported separately in the fixture summary so no future
run can absorb the second as the first.

### D. Unit coverage (already delivered by the repair, pinned here)

Text-quiet + audible audio must not fire; BOS with neither semantic nor
acoustic output must fire; audible-then-silent must close via decoded
silence. Retain the existing watchdog cases unchanged.

## Validation for this amendment

The new gate must be demonstrated to have **caught the original defect**:
run it against the retained pre-fix trace
`2f88c304-b895-4b81-aa1d-adde6043d58c` and show it FAILS (4 of 5 firings
audible), and against the post-fix validation traces and show it PASSES.
A gate that cannot fail on the known-bad evidence is not a gate.
