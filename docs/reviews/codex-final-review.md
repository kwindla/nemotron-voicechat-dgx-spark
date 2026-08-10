# Codex final adversarial review

Date: 2026-08-04

## Verdict

**APPROVE.** The external runtime manifest was regenerated from clean runtime
commit `8e0255135424fd007b6c69778ca1f3c9d90e12c6`, the model was restarted with
that manifest, and the post-restart browser smoke passed. No open code,
protocol, security, or repository-scope blocker remains.

## Findings and disposition

- **Backward-compatible codec loading:** both independent reviewers found that
  `cpu_codec_worker.py` still accepted the retired standalone
  `codec.safetensors` layout. The fallback and its test were removed. The
  worker now accepts only the combined public checkpoint's
  `model.safetensors` and extracts only `tts_model.audio_codec.*` tensors.
- **Browser authentication:** the public surface is Pipecat's stock debug UI
  and SmallWebRTC endpoints, with no username, password, or model credential in
  the page. The authenticated strict-v2 model socket remains loopback-only and
  is called only by Pipecat.
- **Startup prefill stall:** one launch stalled during a 37-element EarTTS mask
  device-to-host copy after an earlier failed boot. A full stack-down released
  roughly 100 GiB of unified memory; an identical clean-memory launch passed
  engine initialization, realtime warmup, and inference. A proposed
  `VLLM_ALLOW_INSECURE_SERIALIZATION` change was rejected because the passing
  discriminator did not contain it. No speculative runtime change was made.

## Evidence reviewed

- Repository contains 59 tracked files before the final review reports, with
  no model weights, captured media, traces, vendored Pipecat, custom browser
  client, training pipeline, or quantization campaign.
- Secret-pattern scan of tracked files found no credential. `.env` is ignored
  and the deployed copy is mode `0600`.
- Model container mounts the repository, NVIDIA Speech checkout, checkpoint,
  Nano skeleton, and derived artifacts read-only. Only its dedicated trace
  mount is writable. Model port 8786 is loopback-only.
- Pipecat is pinned to public commit
  `5415f28af9407937610d68abb2c195365650a862`; NVIDIA Speech and model/runtime
  provenance are pinned in `docs/provenance.md` and the external manifest.
- The model rejects missing or non-v2 negotiation and unsupported legacy
  client events. Pipecat's adapter independently checks the advertised v2
  contract.
- `uv run --frozen pytest -q`: **106 passed, 3 skipped** after removal of the
  obsolete compatibility test.
- `uv run --frozen ruff check .`: **PASS**.
- Real Chromium through the stock Pipecat debug UI and SmallWebRTC: **PASS**;
  see `fable-live-e2e.md`. This covered silence, noise, three voice turns,
  cross-turn memory, a real UTC tool call with matching spoken output, inbound
  RTP audio playback, and clean teardown.
- Post-cleanup live smoke on the regenerated manifest: **PASS**; trace
  `25e55794-5430-4a99-becc-586bd774c17d`. It reconfirmed a normal voice turn,
  the UTC tool and matching spoken result, rendered RTP audio, and clean slot
  release.

## Accepted limitations

The project remains a single-client, single-GPU production candidate. The
known stochastic model/tool behaviors, 12,000-frame session guard, late-session
quality qualification, and two-frame p95 near-miss remain explicitly recorded
in `docs/known-limitations.md`; none is hidden by adapter compatibility code.
