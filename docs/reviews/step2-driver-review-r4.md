# Gate review: Step 2 live fixture driver, round 4

Review date: 2026-08-13 UTC. Scope: the Step 2 preregistration, round-3
review and fix note, `tools/qualification/step2_live_fixture_driver.py`,
the current diffs to `src/nemotron_voicechat_pipecat/llm.py` and
`tests/e2e/test_browser_smallwebrtc_e2e.py`, and the Step 2/playout tests.

## Verdict

**FAIL — the driver is not qualified and the live campaign must not execute.**

The real live-third-writer terminal check, audio delta/release/push causality,
artifact-reconstructed summary membership (including response text-delta
records), marker lifecycle brackets, checked-consumer text-delta schema, and
trace-off isolation all passed fresh probes. Two fail-open boundaries remain.

## Remaining deltas

### 1. Repeated-script transcripts are content-bound but not response-bound

The strict event-index and clock brackets close the four round-3 clock/order
attacks, but transcript identity is still inferred only from equal text.
`transcription_texts` accepts every `bot-transcription` inside the bracket and
tests only whether their joined bytes equal `expected_output_text`
(`tools/qualification/step2_live_fixture_driver.py:2269-2299`).
`rtvi_transcript_evidence` then writes the caller-supplied `response_id` into
the retained bracket even though no accepted transcript record is joined to
that ID (`:2302-2359`).

This is fail-open for the canonical campaign because S1 and L1 are each run
more than once. A fresh probe placed a late S1-1 transcript strictly between
the S1-2 speaking start and stop, gave it the exact S1 bytes, and supplied the
response-owned S1-2 text as the expected output. The consumer returned the
late transcript and retained it under `response-S1-2`. Thus the late-prior
attack remains accepted whenever the prior and current response text is
identical; the current negative test uses different prior/current text and is
closed only by the content comparison.

Required delta: bind each accepted transcription event to the traced response
through a protocol response/item identity, or enforce and retain an equivalent
protocol-backed drain/ack barrier that makes prior-response delivery impossible
before the next fixture starts. Add an exact repeated-S1 late-transcript test;
different transcript bytes are not a sufficient identity attack.

### 2. The retained output descriptor does not protect later output opens

`resolve_step2_paths` correctly reserves and checks the session through an
`O_NOFOLLOW` descriptor, but closes that descriptor and returns pathname
strings (`tools/qualification/step2_live_fixture_driver.py:876-914`). The live
driver later reopens those strings by pathname: the browser sink calls
`path.parent.mkdir(...)` and `path.open("x")`
(`tests/e2e/test_browser_smallwebrtc_e2e.py:484-487`), while summary retention
uses pathname `write_text` and `replace` (`:798-805`).

A fresh deterministic race wrapped `_reserve_session_directory`, let it open
the real session descriptor, then renamed that directory and installed a
symlink at the original session pathname before `resolve_step2_paths` performed
its descriptor-relative child checks and sentinel creation. Resolution
succeeded; the sentinel was safely created in the renamed original directory,
but a subsequent write through the returned summary path landed in the symlink
target. This is the next check/use boundary after the two newly pinned races.

Required delta: retain the reserved directory identity through every browser
artifact and summary open, and open children relative to that descriptor with
no-follow semantics. Any Pipecat artifact operation that must remain
pathname-based must revalidate the current parent identity against the reserved
`(st_dev, st_ino)` before reads, publication waits, reconstruction, and terminal
set validation. Add a deterministic directory swap after the output descriptor
is opened and before the first child writer opens.

## Verification record

No `browser_e2e` entry, live URL, browser campaign, running stack, container,
or service endpoint was touched.

- Full offline suite: **1018 passed, 15 skipped, 2 deselected, 29 subtests
  passed** with `-m 'not browser_e2e'` (four warnings).
- Canonical retained evidence: **`verify-anchors: PASS (67 anchors)`**.
- Task-scoped Ruff over `llm.py`, the driver, analyzer, browser driver/tests,
  runtime driver tests, and playout-trace tests: **passed**.
- Task-scoped `git diff --check`: **passed**.
- `node --check` on the generated RTC probe, voice-latency probe, AudioWorklet
  installer, and extracted embedded worklet processor: **passed**.
- A real third `_PlayoutTraceWriter` left live while writers 1 and 2 were
  published produced a matching `.trace-0003...pending` artifact and the
  terminal exact-two check rejected it. Fresh release-before-delta and
  push-before-delta/release mutations were also rejected.
- Fresh completed-summary mutations of response-owned text plus redundant
  transcript/hash fields, release/delta/push membership, and persisted marker
  clocks were rejected against the retained artifacts and lifecycle brackets.
- New text-delta records rejected an extra field, ordinal zero, empty text, and
  foreign response ownership. The canonical consumer result was identical with
  and without a valid text-delta record, and the 67 retained anchors remained
  unchanged.
- The trace-off base service has the same 71-method AST and method set as
  `HEAD`; all 71 method `co_code` values matched, and `__new__`,
  `_handle_server_event`, `_audio_delta`, `_release_playout_buffer`, and
  `_clear_playout_buffer` also retained identical constants and names.

Fresh probe output is retained at `/tmp/step2-driver-r4-probes.json`; canonical
anchor output is retained at `/tmp/step2-driver-r4-canonical-anchors.json`.
