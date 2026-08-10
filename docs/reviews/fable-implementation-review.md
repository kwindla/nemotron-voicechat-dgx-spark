# Fable implementation review — standalone repository extraction

Date: 2026-08-04 · Commit reviewed: `b069af037e6b5edbe829d89bb9b6d4c75cc0a067`
Cross-checked against `docs/extraction-plan.md`, the prior extraction-plan review
(`docs/reviews/fable-extraction-plan-review.md`), the tracked source worktree, and
the live legacy-stack processes (read-only). No live service was modified or
restarted; no deployment was performed.

## Verdict

**APPROVE.** All four prior plan-review blockers are resolved in code, the
semantic diff against the tracked source is behavior-preserving, minimality and
share-safety scans are clean, and both test tiers pass (browser gate via the
documented Chromium env var — see fix 1). Three documentation/robustness fixes
are recommended below; none blocks implementation sign-off.

## Prior blocker disposition (B1–B4)

- **B1 resolved** — `patches/nemotron-voicechat-rnnt-turn-taking.patch` ships in
  `src/nemotron_voicechat_runtime/patches/`; the server resolves it relative to
  its own file (`server.py:1480`) and byte-validates the Speech working tree at
  startup, exactly as before.
- **B2 resolved** — `deploy/prepare-speech.sh` clones NVIDIA Speech at the pinned
  commit, applies the patch with `git apply --check`, and `cmp`-verifies the
  resulting diff byte-for-byte. The runtime-loaded function-calling template is
  resolved from that prepared tree (`server.py:2629-2631`) and the prerequisite
  is documented in `docs/deployment.md`.
- **B3 resolved** — no EA-extracted audio ships; `git ls-files` contains no
  `.wav`/`.pcm`/`.nemo`/checkpoint artifacts. Warm-up is the synthetic
  engine-path warm-up (`VOICECHAT_WEB_REALTIME_WARMUP`), not a sample file.
- **B4 resolved** — `patch_public_pad_pair_scheduler.py` is packaged and applied
  in `container/Dockerfile.runtime-overlay`.

## Semantic diff vs tracked source

- `runtime_optimizations.py` (988 lines vs 1181): removed functions are
  profiling-only (`install_vllm_step_profile`, `profile_public_frame`, worker
  CUDA profiler start/stop/control). No dangling references to removed
  nano-replay/profiling/native modes anywhere in `src/`, `deploy/`,
  `container/`, or docs.
- The **inlined GC/BF16 perception primitives** (`RealtimeCyclicGCGuard`,
  `install_perception_precision_override`, new helper
  `perception_uses_captured_graph`) are behavior-identical to the originals in
  the legacy adapter module: differences are docstrings, formatting, a type
  hint, and the sentinel attribute rename to `_voicechat_bf16_installed`. The
  load-bearing captured-graph autocast-skip logic is preserved
  (`runtime_optimizations.py:50,79-85`).
- `cpu_codec_offload.py` spawns the packaged worker via
  `Path(__file__).with_name("cpu_codec_worker.py")` with `PYTHONPATH` propagated
  — correct under the new package layout. The NLTK/`regex` preload is retained
  with rationale and `regex>=2024.11.6` pinned in `pyproject.toml`.
- Package renames are consistent; zero references to the old module names.

## Minimality / share-safety scans

- No EA-derived files, experiment/benchmark scripts, or legacy protocol
  branches. 57 tracked files total; largest is `uv.lock` (~700 KB); no binaries.
- No tunnel origins, credentials, API keys, or private absolute paths anywhere
  in tracked content (the only bearer string is a test literal). The committed
  prior review is scrubbed.
- `THIRD_PARTY_NOTICES.md` covers NVIDIA NeMo Speech, the vLLM/Voicechat fork,
  Pipecat, NVIDIA model materials, and the qualified container image.
- `container/Dockerfile.runtime-overlay` explicitly scopes the base image as a
  local, non-redistributable prerequisite with a pointer to
  `docs/provenance.md` — reproducibility claims are correctly repo-scoped.

## Environment parity (vs live stack, values redacted)

43 contract variables in `provenance.py` vs 47 in the live container. The four
live-only extras are removed diagnostics (step-profile, EA profile frames,
RNNT-step disable, warm-up skip) consistent with minimality; the repo adds two
explicit watchdog-threshold variables. `VOICECHAT_REALTIME_API_KEY` (server) and
`NEMOTRON_VOICECHAT_API_KEY` (client) are both documented in `.env.example`, and
the server refuses to start without auth off-loopback (`server.py:2628`).

## Deployment dry review

`deploy/stack` parameterizes ports and uses distinct unit names from the live
stack. Note: the defaults in `.env.example` (model 8786, Pipecat 7860) equal the
ports the legacy stack currently occupies — a default-value bring-up alongside
the live stack fails loudly on port bind. See fix 2.

## Tests

- `uv run --frozen pytest -q tests/runtime tests/pipecat` → **104 passed,
  3 skipped** (1.2 s).
- Browser SmallWebRTC E2E → **passed** with
  `PLAYWRIGHT_CHROMIUM_EXECUTABLE=<playwright chromium path>` (the test reads
  this variable at `tests/e2e/test_browser_smallwebrtc_e2e.py:23`). Without it,
  browser resolution is environment-dependent; on this host an alternate
  browser produced a non-connecting media path and the gate fails. This is a
  documentation gap, not a code defect.

## Recommended fixes (non-blocking)

1. **Document the browser-gate prerequisite** — README test section (add under
   the test-running instructions): `PLAYWRIGHT_CHROMIUM_EXECUTABLE` must point
   at a Playwright-provisioned Chromium; also worth a commented line in
   `.env.example`.
2. **Deployment collision note** — add an explicit paragraph to
   `docs/deployment.md` (near the port discussion): the default ports match the
   legacy stack; side-by-side bring-up requires overriding
   `VOICECHAT_MODEL_PORT`/`VOICECHAT_PIPECAT_PORT`, and cutover requires
   stopping the legacy services first.
3. **Bound the overflow error loop** — in the failed-media diagnostic run, the
   service emitted `input audio queue overflow` errors every ~500 ms and the
   loop continued after pipeline-worker cancellation. The first overflow is
   already fatal by design; ensure it terminates the input path once and that
   the error task is cancelled with the worker (`nemotron_voicechat_pipecat`
   input/send path).
