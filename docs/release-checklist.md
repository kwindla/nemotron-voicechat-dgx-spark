# Release checklist

Use this checklist for a public runtime or weight release. A checked box must
name the retained report or immutable identifier that proves it; command exit
status alone is not release evidence.

## Source and publication

- [x] The public NVIDIA parent, Nano skeleton, Speech fork, Voicechat vLLM
  fork, base image, Python locks, and Pocket assets still match the immutable
  revisions in `config/production-candidate-1.toml` and the signed BOM. Proven
  by the 2026-08-05 conversion preflight and reproduction report.
- [x] `./voicechat test` and `uv run --frozen ruff check .` pass from a clean
  detached checkout: 241 passed, 7 explicit dependency skips; Ruff clean.
- [x] Two clean source conversions are byte-identical and match the candidate
  manifest, or the differing output has been assigned a new candidate identity
  and requalified from the beginning. Reproduction self-hash:
  `ab80bbcb19c5e7ba34a1936b923480701e8b594c6fc158010d801308c9dc1e83`.
- [x] The Hugging Face dry-run inventory, upload, immutable-revision read-back,
  and release signature checks pass. Record the repository revision and signed
  release hash. Revision `a20c68547d27421c1e66bd501bf295569a0cc309`;
  signed release `1d8c3db8efbb87e0da3be86f1045957018082e1e8a339edaaf7e9bf4fb70671b`.
- [x] OpenMDW-1.1, origin notices, calibration provenance, model card, and known
  limitations are present in both the repository and published weight release.
- [ ] After the source commit is public, publish the superseding model card as
  a new immutable Hugging Face metadata revision. Do not regenerate or alter
  the already-qualified weight payloads.

## Runtime build

- [x] `./voicechat bootstrap --no-cache` completes from public inputs and its
  audit proves there is no private/EA layer, credential, NGC dependency, or
  `VLLM_ALLOW_INSECURE_SERIALIZATION` setting.
- [x] Record the final image ID, architecture, history digest, SPDX 2.3 SBOM
  digest, main/Pocket runtime identities, and provenance attestation digest.
  Image `sha256:6294053537329500a17a1684421cd3909224babd7423d677c6c5d6bceead0482`
  (`arm64`); SBOM SHA-256
  `a065741de28dfddb10c57c9fc30fd206c4edb2931c8fc7e2f4691c93cd6197b2`;
  attestation SHA-256
  `a74947438476dae6476b62d5bba2b5197d5aa7a2e98c182c28b3f925732ecd53`.
- [x] `./voicechat bootstrap --offline` succeeds afterward with networking
  unavailable and the bootstrap state points at that final image ID.
- [x] `./voicechat doctor` passes on the qualified driver, kernel, Python,
  Docker/NVIDIA runtime, ports, artifacts, disk, and bootstrap state on
  2026-08-05.

## Production qualification

- [x] Run `./voicechat test --live --asr-model /absolute/path/to/asr.nemo`
  against the downloaded published artifacts and retain its entire output
  directory.
- [x] Nano (all 1,172 calls), EarTTS eager/graph, and isolated Pocket worker
  component gates pass before live execution.
- [x] Three SIGINT cycles and SIGKILL/down recovery meet the cold-start budget;
  the raw model is unreachable from every non-loopback host address.
- [x] The real Chromium Playground/SmallWebRTC gate passes microphone, typed,
  mixed-modality memory, inbound assistant PCM, and playback assertions.
- [x] The direct multi-turn voice gate passes independent input/output ASR.
- [x] The paced run exceeds 15 minutes, interleaves typed turns, has no
  unbounded queue growth, stays within the qualified silent-turn base rate,
  closes the 12,000-frame limit gracefully, and passes late-response ASR.
- [x] Repeat the live gates with locally converted artifacts and run
  `compare_candidate_reports.py`; component identities and gate verdicts agree.
  Staged source-agreement report SHA-256:
  `1bf4ea21391865f293b5dca755088e4f9fe904ba1811744ade6aa9517df578b8`.

## Developer journey and handoff

- [ ] In a separate clean clone and cache, with no old Voicechat image or source
  worktree available, an uninvolved developer runs the README commands exactly:
  `./voicechat bootstrap`, `./voicechat up`, opens the printed Playground URL,
  completes voice and typed turns, and stops everything with Ctrl-C.
- [x] Three immediate foreground restart cycles plus orphan cleanup have left
  no model container, Pipecat process, occupied port, or hidden system service.
  The retained downloaded-artifact lifecycle gate passed three SIGINT cycles,
  SIGKILL/down recovery, and a subsequent clean start.
- [ ] Documentation reflects measured download size, startup time, memory
  headroom, qualification verdict, and any newly observed upstream limitation.
- [x] Final Fable and Codex adversarial reviews have no open release blocker for
  the Stage-2 public candidate; public-history, secret, claim, conversion,
  activation, and release-tool boundaries were checked independently.
