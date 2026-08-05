# Release checklist

Use this checklist for a public runtime or weight release. A checked box must
name the retained report or immutable identifier that proves it; command exit
status alone is not release evidence.

## Source and publication

- [ ] The public NVIDIA parent, Nano skeleton, Speech fork, Voicechat vLLM
  fork, base image, Python locks, and Pocket assets still match the immutable
  revisions in `config/production-candidate-1.toml` and the signed BOM.
- [ ] `./voicechat test` and `uv run --frozen ruff check .` pass from a clean
  checkout.
- [ ] Two clean source conversions are byte-identical and match the candidate
  manifest, or the differing output has been assigned a new candidate identity
  and requalified from the beginning. **Pending stage 2:** the production
  conversion pipeline and dual-source evidence intentionally follow the
  initial inference-runtime release.
- [ ] The Hugging Face dry-run inventory, upload, immutable-revision read-back,
  and release signature checks pass. Record the repository revision and signed
  release hash.
- [ ] OpenMDW-1.1, origin notices, calibration provenance, model card, and known
  limitations are present in both the repository and published weight release.

## Runtime build

- [ ] `./voicechat bootstrap --no-cache` completes from public inputs and its
  audit proves there is no private/EA layer, credential, NGC dependency, or
  `VLLM_ALLOW_INSECURE_SERIALIZATION` setting.
- [ ] Record the final image ID, architecture, history digest, SPDX 2.3 SBOM
  digest, main/Pocket runtime identities, and provenance attestation digest.
- [ ] `./voicechat bootstrap --offline` succeeds afterward with networking
  unavailable and the bootstrap state points at that final image ID.
- [ ] `./voicechat doctor` passes on the qualified driver, kernel, Python,
  Docker/NVIDIA runtime, ports, artifacts, disk, and bootstrap state.

## Production qualification

- [ ] Run `./voicechat test --live --asr-model /absolute/path/to/asr.nemo`
  against the downloaded published artifacts and retain its entire output
  directory.
- [ ] Nano (all 1,172 calls), EarTTS eager/graph, and isolated Pocket worker
  component gates pass before live execution.
- [ ] Three SIGINT cycles and SIGKILL/down recovery meet the cold-start budget;
  the raw model is unreachable from every non-loopback host address.
- [ ] The real Chromium Playground/SmallWebRTC gate passes microphone, typed,
  mixed-modality memory, inbound assistant PCM, and playback assertions.
- [ ] The direct multi-turn voice gate passes independent input/output ASR.
- [ ] The paced run exceeds 15 minutes, interleaves typed turns, has no
  unbounded queue growth, stays within the qualified silent-turn base rate,
  closes the 12,000-frame limit gracefully, and passes late-response ASR.
- [ ] Repeat the full live suite with locally converted artifacts and run
  `compare_candidate_reports.py`; component identities and gate verdicts agree.

## Developer journey and handoff

- [ ] In a separate clean clone and cache, with no old Voicechat image or source
  worktree available, an uninvolved developer runs the README commands exactly:
  `./voicechat bootstrap`, `./voicechat up`, opens the printed Playground URL,
  completes voice and typed turns, and stops everything with Ctrl-C.
- [ ] Three immediate foreground restart cycles plus orphan cleanup have left
  no model container, Pipecat process, occupied port, or hidden system service.
- [ ] Documentation reflects measured download size, startup time, memory
  headroom, qualification verdict, and any newly observed upstream limitation.
- [ ] Final Fable and Codex adversarial reviews have no open release blocker.
