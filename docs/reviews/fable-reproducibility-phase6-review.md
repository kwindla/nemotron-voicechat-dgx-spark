# Fable adversarial review — Reproducibility Phase 6 (bootstrap) + P5-I1 walks

Date: 2026-08-04 · Scope: `voicechat`, `cli.py`, `bootstrap_download.py`,
`artifacts.py`, `pocket_controller/worker`, TOML, container scripts, four
boundary test files, docs. Boundary tests: **28 passed (0.5 s)**; full suite
previously 219/1. No implementation files modified.

## Verdict: APPROVE — no blockers; one IMPORTANT residual.

## P5-I1 line-walks (all four discharged)

1. **Server mic gate:** `server.py` gates microphone packets during typed
   injection (`microphone_packet_gated`, :2822) and discards the partial mic
   utterance on typed acceptance
   (`microphone_partial_discarded_for_typed_input`, :2741) — the gate spans
   acceptance→closure with the EOU-derived silence rule verified at Phase 5.
2. **Pocket affinity:** worker is pinned via `--cpus 7,8,9,15`
   (`pocket_controller.py:45,119-120`) and echoes
   `os.sched_getaffinity` for verification — see IMPORTANT below re codec
   disjointness.
3. **Loopback-only model port:** the docker publish is
   `127.0.0.1:{model_port}:8786` (`cli.py:419`), with pre-flight port
   checks on 127.0.0.1 (:215, :509-510) and health polled on loopback.
4. **Teardown:** SIGINT forwarded to Pipecat (:485), SIGTERM handler
   installed with prior handler preserved (:518), stale-container detection
   refuses `up` with an explicit "run ./voicechat down first" (:507).

## Bootstrap properties verified

- **Atomic/idempotent downloads:** snapshot downloads land in
  `mkdtemp(.tmp-…)` siblings then rename; state via `atomic_json`
  (`artifacts.py:102-104`); revisions pinned per call; `local_files_only`
  drives the offline path; `uv sync --frozen --offline` covered by
  `test_cli` (observed-command assertion).
- **`filter="data"` extraction** at `bootstrap_download.py:229`, into a
  temporary directory — the Phase 3 carried requirement is CLOSED.
- **Disk floors:** `require_free_space` gates download (with a printed
  ~65 GiB estimate) and the conversion path retains the 100 GiB floor
  (Phase 2 tooling).
- **TOML/env parity fail-closed:** `artifacts.py:46` raises if the TOML
  runtime environment differs from the frozen candidate.
- **Mutable state outside checkout:** asserted by tests
  (`layout.cache/traces not relative to REPO_ROOT`).
- **No-cache gate:** build script asserts `--no-cache` flags (tested).

## IMPORTANT

**P6-I1 — Pocket worker cores (7,8,9,15) vs codec cores: disjointness is
asserted nowhere.** The controller pins the worker, but I found no
machine-readable declaration of the codec worker's reserved cores to check
against, and no startup assertion that the two sets are disjoint. Add the
codec core set to the TOML/BOM and a startup check (or a comment citing the
measured core map) so a future core-map change cannot silently create
contention on the latency-critical cores.

## NITs

- Gated-HF 401/403 actionable-retry and LFS/unexpected-file rejection were
  verified via the bootstrap tests rather than live network probes —
  appropriate for a read-only pass; Phase 7's clean-cache run is the live
  proof.
- Document the worker affinity echo (`sched_getaffinity` report) in
  `docs/deployment.md` as the operator's verification hook.

Phase 6 is approved; the remaining authoritative evidence is Phase 7's
clean-machine run (restart cycles, loopback negative test, sustained gate
with typed interleaving, dual-source weight agreement).

---

## Delta — 2026-08-04, P6-I1 closure (CPU affinity disjointness)

### Verdict: APPROVE — P6-I1 CLOSED. Phase 6 has no open findings.

Verified by inspection: `config/production-candidate-1.toml:104` declares
`codec = [5, 6]` alongside Pocket `[7, 8, 9, 15]`; `artifacts.load_config`
rejects empty or overlapping affinity sets (`artifacts.py:51`);
`PocketWorkerManager` independently rejects overlap at startup
(`pocket_controller.py:59`) — defense in depth at both the config and
process layers; overlap/disjoint behavior is covered by tests (3 references
across the boundary suites). Reported full suite: 227 passed, 1 skipped.
The core map is now machine-readable and fail-closed against future drift.
