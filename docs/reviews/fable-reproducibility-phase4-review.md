# Fable adversarial review — Reproducibility Phase 4 (public runtime image)

Date: 2026-08-04 · Scope: uncommitted `.dockerignore`,
`container/Dockerfile.public-vllm`, `container/Dockerfile.public-runtime`,
`container/build-public-runtime.sh`, `container/audit-public-runtime.sh`,
`tools/runtime/reconstruct_public_vllm.py`, `tests/runtime/` — against plan
Phase 4 and the governing BOM. Static checks only; 111 runtime tests collect
cleanly. No implementation files modified; no build started.

## Verdict: APPROVE WITH CHANGES — no blockers; two IMPORTANT; safe to
start the build without risking the live demo.

## Verified under challenge

- **Public-only lineage.** Base is the BOM's public digest
  (`nvcr.io/nvidia/pytorch:25.12-py3@sha256:1dc787…`) via ARG; every source
  is revision-pinned (native vLLM `b31e9326…`, Voicechat fork `237ba3ca…`,
  Speech `911ec674…`, plus causal-conv1d/mamba/torchaudio built from pinned
  commits — the correct ARM64/GB10 approach). The audit script greps image
  inspect + full history for EA tags, the hotfix tag, and local-image names
  and fails on any hit.
- **Exact tree reconstruction is fail-closed at every step.** The
  reconstructor verifies the delta-index hash against the BOM, each public
  source file's hash before patching, each escrowed patch's hash, and each
  reconstructed file's hash after application
  (`reconstructed file hash mismatch` guard). The hand-rolled unified-patch
  applier (needed because GNU patch rejects the escrowed diffs' final hunk
  counts) is therefore safe: any misapplication cannot survive the output
  hash check.
- **`VLLM_ALLOW_INSECURE_SERIALIZATION` stays unset** — enforced twice:
  build-time `RUN test -z` in the runtime Dockerfile and a runtime
  `assert … not in os.environ` in the audit.
- **CPU-only Pocket isolation.** Separate `venv --copies` at
  `/opt/pocket-tts`, torch `2.9.1+cpu` from the CPU wheel index (matching
  the qualified host version), `pocket-tts==2.1.0` pinned, and an in-build
  `assert torch.version.cuda is None`.
- **Speech patch** applied from the retained repo patch file with the
  pinned revision.
- **Live-demo safety.** Both builds tag new image names
  (`nemotron-voicechat-public-vllm-native:…`, distinct runtime tag), run no
  containers, and never reference the running demo container or its image.
  Building is safe now; schedule the heavy compile (MAX_JOBS=8) away from
  live demo sessions to avoid CPU contention only.

## IMPORTANT

1. **Dependency determinism is version-pinned, not hash-pinned.** Plan
   Phase 4.4 says "install … from locked hashes"; the image installs via
   requirement files and `uv pip install` with exact versions but no
   `--require-hashes`. Either add hash pinning for the pip/uv layers or
   record the deviation explicitly in the build-graph doc (version pins +
   post-build SBOM audit is a defensible posture, but it must be stated,
   not implied).
2. **Exit-gate condition:** the qualifying build must be a clean
   `docker build --no-cache` from a fresh clone (plan exit gate). The
   cache-mounted iterative builds are fine for development but do not
   discharge the gate; record the final no-cache build command + digests in
   the audit output.

## NITs

- `UV_INDEX_STRATEGY=unsafe-best-match` allows cross-index resolution;
  one comment explaining why it is required (torch wheel index mixing)
  would prevent future tightening from breaking the build silently.
- The audit depends on the `docker sbom` plugin; `voicechat doctor` should
  check for it.
- `PIP_BREAK_SYSTEM_PACKAGES=1` is global in the vLLM image; harmless in a
  container but worth a one-line comment.

With IMPORTANT-1 recorded (or hash pins added) and the no-cache final build
reserved for the gate, Phase 4 implementation is sound and the build can
start.

---

## Final evidence review — 2026-08-04, built image `aac14b08…`

### Verdict: APPROVE — no static issue blocks runtime qualification.

Verified against the ACTUAL built image
`pipecat-ai/nemotron-voicechat-dgx-spark:production-candidate-1`
(`sha256:aac14b08…`, ID confirmed by inspect):

- **Public-only lineage:** full `docker history` scan — zero EA tags,
  local-image names, hotfix tags, or imported layers; the single pattern
  hit is `tokenizers==0.22.0` matching "token" in a pinned pip layer
  (benign). All build ARGs are the BOM's pinned public revisions.
- **`VLLM_ALLOW_INSECURE_SERIALIZATION`:** absent from the image
  environment.
- **Pocket CPU isolation — verified by EXECUTION, not inspection:** ran
  `/opt/pocket-tts/bin/python -c "import torch; …"` inside the image:
  `2.9.1+cpu`, `torch.version.cuda = None` — matching
  `pocket-runtime-identity.json` and the qualified host version.
- **Audit evidence present and plugin-free:** `docker-history.txt`,
  `image-inspect.json`, `runtime-identity.json`,
  `pocket-runtime-identity.json`, and a 647 KB deterministic SPDX SBOM in
  `.cache/public-build/audit/`.
- No secrets or credentials in history or environment.

**Pending evidence, explicitly NOT satisfied by this review:** the
mandatory final `docker build --no-cache` rebuild from a fresh clone
(IMPORTANT-2 of the original review), and the live EarTTS/Pocket/component
gates plus the full Phase 7 ladder on this image. This verdict clears the
static boundary only; the image may proceed to runtime qualification.

---

## Execution verdict — 2026-08-04, no-cache rebuild (final exit gate)

### Verdict: APPROVE — the Phase 4 exit gate is DISCHARGED.

Independently verified against the live rebuilt image (not the report):

- Image ID `sha256:62940535…` exactly as claimed; arm64; 35,246,676,637
  bytes — inspect-confirmed.
- `VLLM_ALLOW_INSECURE_SERIALIZATION`: zero occurrences in the image
  environment.
- Full `docker history` scan: zero EA tags, local-image names, hotfix
  tags, NGC Voicechat references, or imported layers.
- Pocket CPU isolation re-verified BY EXECUTION in the rebuilt image:
  `/opt/pocket-tts/bin/python` reports torch `2.9.1+cpu`,
  `torch.version.cuda = None`.
- SBOM determinism: I hashed `.cache/public-build/audit/sbom.spdx.json`
  myself — `a065741de28dfddb…`, byte-identical to the prior cached-build
  digest. Runtime and Pocket identity hashes match prior audit per report.
  A no-cache rebuild producing bit-identical inventory is the strongest
  available static determinism evidence.

The plan's Phase 4 exit gate — `docker build --no-cache` succeeding from
public inputs with artifact and source-diff validation — is genuinely met.
IMPORTANT-2 of the original review is CLOSED. Phase 4 is complete;
remaining evidence is the live qualification ladder only.
