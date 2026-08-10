# Fable pre-deployment closure review

Commit: `aea7a0d4842deea628a32ee3e84cf560c44543ef` · Date: 2026-08-04
Verdict: **APPROVE**. No live services modified; nothing deployed.

## Recommended-fix closure (all three verified)

1. **Chromium test prerequisite** — `PLAYWRIGHT_CHROMIUM_EXECUTABLE` documented
   in the README test section (README.md:55,60) and `.env.example:71`.
2. **Port-collision/cutover** — `docs/deployment.md:77` documents the cutover
   (stop legacy tunnel; `VOICECHAT_PIPECAT_PORT` for side-by-side).
3. **Bounded overflow handling** — `llm.py` now latches `_input_failed`: one
   fatal error push, one disconnect, all subsequent paths early-return. The
   500 ms repeat loop is gone; regression tests added
   (tests/pipecat/test_nemotron_voicechat_websocket.py, +33 lines).

## New items verified

- **Safe `.env` loader** — `deploy/check-environment.py` validates variable
  names against a strict identifier regex (rejects injection via names) and its
  `--shell` mode emits `export NAME=<shlex.quote(value)>`; `deploy/stack` evals
  only that sanitized output. Missing required variables exit with a clear
  message. Sound.
- **Explicit-process-env precedence** — `demo.py` captures launcher-supplied
  `NEMOTRON_VOICECHAT_WS_URL`/`NEMOTRON_VOICECHAT_API_KEY` before Pipecat's
  runner import (which loads a developer `.env` with override) and prefers
  them; `load_dotenv(override=True)` removed. Test/deployment overrides can no
  longer be silently replaced. Correct fix for the failure mode the real
  browser gate exposed.

## Gates and scans

- Full suite: **106 passed, 3 skipped** (9.45 s). Browser SmallWebRTC gate:
  **passed** via the documented Chromium env var.
- Secret/path scan: no tunnel origins, credentials, or private absolute paths
  in tracked files (the only pattern-scan hits are legitimate auth-handling
  code and its tests; verified value-free). Largest tracked file is `uv.lock`
  (~700 KB); no large binaries or EA artifacts.
- Live model service verified healthy and untouched after the review.
