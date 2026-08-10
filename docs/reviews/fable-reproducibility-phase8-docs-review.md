# Fable adversarial review — Reproducibility Phase 8 (documentation)

Date: 2026-08-04 · Scope: README.md, docs/deployment.md,
docs/qualification.md, docs/provenance.md, docs/known-limitations.md
cross-checked against the actual `voicechat` CLI, bootstrap/build/audit
scripts, config, and the Phase 7 harness. No files edited.

## Verdict: APPROVE — no blockers, no IMPORTANT findings, one MINOR.

## Verified

- **Two-command first screen:** README opens with clone →
  `./voicechat bootstrap` → `./voicechat up` → open
  `http://127.0.0.1:7860/client/`, Ctrl-C to stop, with the ~65 GiB
  download/build estimate and gated-HF guidance (accept model terms +
  `hf auth login`, or `HF_TOKEN` supplied only to the bootstrap
  invocation). Matches the plan's Phase 8.1 exactly.
- **No stale prerequisites:** every `.env`/systemd/`deploy/stack`/
  `prepare-speech` mention across the reviewed docs is a negation
  ("there is no `.env` …", "no system service, Compose file …"). The
  retired host-Pocket architecture language is gone —
  `known-limitations.md:21` correctly states typed input is half-duplex
  and "the model server owns the gate."
- **Command/flag parity:** all six documented commands (`bootstrap`,
  `doctor`, `up`, `down`, `test`, `release`) exist in `cli.py`; documented
  invocations (`--offline`, `--no-cache`, `test --live`,
  `up --host --port`) all correspond to real flags.
- **Teardown/restart and exposure guidance present:** stale-container
  recovery via `down`, restart behavior, and loopback/LAN discussion in
  `deployment.md`.
- **Offline story:** `bootstrap --offline` documented in both deployment
  and qualification docs, matching the Phase 6 no-network proof path.

## MINOR

- The `./voicechat up --host 0.0.0.0` example in `deployment.md` should
  state in the same passage that only the Pipecat port binds to the given
  host and the model port remains `127.0.0.1`-only regardless (true in
  `cli.py:419`; nearby loopback mentions exist but the guarantee deserves
  one explicit sentence at the point of the example).

An uninvolved DGX Spark developer following these docs reaches the
Playground with no private prerequisites; qualification and provenance
stories match the implemented harness and evidence chain.

---

## Final disposition — 2026-08-04

The sole MINOR is verified fixed: `deployment.md` now states, immediately
after the `--host 0.0.0.0` example, that only the Pipecat listener changes
and the raw model port remains bound to `127.0.0.1` regardless.

**Phase 8: APPROVE, no open findings.** The full static ladder (Phases
1–8) is closed under adversarial review; live qualification on the public
image is the sole remaining evidence.

---

## Delta 2 — 2026-08-04, release checklist artifact

### Verdict: APPROVE. The checklist is complete, exact, and fail-closed.

Verified against the plan and harness: every public release obligation is
present as an unchecked item — clean-clone tests+lint, HF dry-run/upload/
immutable-revision read-back, `bootstrap --no-cache` from public inputs,
`bootstrap --offline` proof, `doctor`, `test --live` with the ASR model
path, live loopback negative ("unreachable from every non-loopback host
address"), the real Chromium gate covering microphone AND typed paths, the
>15-minute paced run with typed interleaving / no unbounded queue growth /
qualified silent-turn base rate, dual-source agreement via
`compare_candidate_reports.py`, the clean-machine two-command developer run
(voice + typed + Ctrl-C), and three restart cycles plus orphan cleanup.
Command names and flags match the implemented CLI exactly. Every box is
`[ ]` — nothing claims an unrun gate passed. README links the checklist at
the publication decision point. No findings.
