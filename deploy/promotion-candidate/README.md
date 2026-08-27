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

## Runtime image (REQUIRED — the qualified behavior is in the source)

The qualified runtime code lives in `src/nemotron_voicechat_runtime/`: the
tool-call repetition watchdog, the pre-EOU settlement recovery, and the
fence-on-real-audio + early-arming settlement changes that eliminate
conversational latency accumulation. Weights alone do NOT provide them.

Build the image from the current checkout:

```
deploy/promotion-candidate/build-promotion-image.sh
```

It builds via `container/build-public-runtime.sh` (native vLLM base per
`docs/provenance.md`), tags `pipecat-ai/nemotron-voicechat-dgx-spark:promotion-candidate`,
and fails closed unless the built image's `server.py` hash equals the working
tree's. Override the tag with `VOICECHAT_PROMOTION_IMAGE` (the start script
honors the same variable).

`Dockerfile.defect-fixes` remains as a fast overlay for iterating on an
already-built runtime image; it is not the reproducible path.

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

## No-text watchdog runtime hotfix v1

Candidate 1, candidate 2, and the published candidate-3 fhw8 recipe remain
frozen. `config/production-hotfix-notext-watchdog-v1.toml` reuses candidate
3's exact published artifact revision and release hash but assigns the changed
runtime its own image and contract identity.

`start-production-notext-watchdog-hotfix-v1.sh` is the only launcher for this
contract. Before Docker starts it verifies `manifest.sha256` fail-closed,
resolves the image to an immutable `sha256:<64 hex>` ID, and passes both image
name and ID into retained session provenance.

The response bound is 30 seconds of monotonic wall time from agent BOS through
the whole response, including gaps where the upstream `agent_speaking` flag is
false. The budget is 2.29 times the retained maximum valid response of 13.12
seconds (16.88 seconds of headroom). At expiry the runtime latches agent EOS
and stops client audio delivery; it never synthesizes redirect text. The
upstream frame-count cap is disabled because it is neither wall time nor a
whole-response measurement.
