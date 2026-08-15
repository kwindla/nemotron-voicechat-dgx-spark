# fhw8 promotion candidate — clone + bootstrap runbook

The fhw8 candidate quantizes the release Nano's F32 `function_head` onto the
shipped lm_head GPTQ-W8/Marlin path. Qualified end to end (G0–G3 +
counterbalanced A/B: −5.45 ms/frame, 78.7 ms/frame delivered, realtime
structural gate ACCEPT); full evidence trail in
`docs/generation-frame-latency-plan.md` and `reports/fhw8-qualification*`.

## Getting the weights (either path yields byte-identical artifacts)

**A. Download (canonical):** the signed release repo
`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` carries the fhw8 nano at
the revision pinned in `config/production-candidate-3.toml`
(`release_sha256` and the nano composite
`dd4ce6a99d03847c8da10708a0214b1815df078a5c43cae9846b7fe136085ce6` are
verified by the normal bootstrap flow).

**B. Derive locally (provenance proof):** from any bootstrapped release of the
prior candidate, `./derive-fhw8-candidate.sh` rebuilds the candidate
deterministically (double-build reproducibility check included) and fails
closed unless the result matches the committed pins
(`fhw8-artifact-expected.sha256.json`, `fhw8-candidate-manifest.json`).
~10 minutes CPU. This proves the uploaded weights are exactly the committed
transform of the release.

## Runtime image

The runtime image must be built from this branch (the tool-call watchdog and
pre-EOU settlement recovery live in `src/nemotron_voicechat_runtime/`):
`container/build-public-runtime.sh` per `docs/provenance.md`, then either use
the tag directly in `start-production-fhw8-fixed.sh` (edit the image name) or
overlay an existing runtime image with `Dockerfile.defect-fixes`.

## Launch

`start-production-fhw8-fixed.sh` — the frozen production-candidate-1
environment (61 keys + launcher HF-cache keys) with the fhw8 manifest/nano
paths. The server re-verifies every artifact hash at startup. The pipecat bot
runs from the host repo: `PYTHONPATH=src python -m nemotron_voicechat_pipecat.demo
-t webrtc --host 0.0.0.0 --port 7860` with
`NEMOTRON_VOICECHAT_WS_URL=ws://127.0.0.1:8786/v1/realtime`.

Note: `production-candidate-3.toml` pins the release; switching the managed
`./voicechat up` flow to it requires the documented bootstrap-state swap and
has not yet been end-to-end tested on a fresh machine — the scripts above are
the qualified path used for promotion.
