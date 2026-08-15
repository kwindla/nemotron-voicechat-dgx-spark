# fhw8 production promotion record

**Date:** 2026-08-14 UTC
**Authorized by:** user instruction "Promote." following the completed
qualification contract: G0–G2 (`reports/fhw8-qualification-20260814.md`),
G3 (`…-g3-20260814.md`), counterbalanced A/B (`…-ab-20260814.md`, Δ 5.452
ms/frame, candidate 78.67 ms/frame, structural gate ACCEPT).

## What is promoted

The fhw8 candidate Nano artifact
(`~/.cache/nemotron-voicechat/artifacts/candidates/nano-fhw8`, artifact
manifest `fhw8-candidate-manifest.json`, SHA-256
`fbb121328a9c7a4d5e8b374704484095364a232b79ed0001cba142fb0b86f472`,
reproducibility earned by byte-identical double build) serving as the
production Nano, with the unchanged release EarTTS and parent checkpoint.

## Promotion mode: local production

The running production service is launched by
`start-production-fhw8.sh` (SHA-256
`060cf095a36e1f718007d5678e0b1fadbac0f9f566c2e476c890b8bf8cff8832` (v2;
the v1 script `f05e8f04…` omitted the launcher-added HF cache env keys and
its pocket-worker startup failure is retained as
`startup-attempt2-pocket-env.log`),
retained in the session scratchpad and reproduced below in spirit):
the exact frozen 61-key `[runtime.environment]` from
`config/production-candidate-1.toml` (no qualification overrides: no
qualification mode, no watchdog override, no fixture system prompt), the
current branch runtime image
`pipecat-ai/nemotron-voicechat-dgx-spark:generation-step2`
(`sha256:ea15693c2c79…`, carries the tool-call and abort-liveness fixes),
and the qualified fhw8 manifest/nano paths. The released artifact and the
signed release chain are untouched.

## Recorded follow-up (release-chain assembly)

`config/production-candidate-1.toml` pins a signed Hugging Face release
(`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`, release_sha256
`1d8c3db8…`). Folding fhw8 into that chain requires assembling and
publishing a new signed release and a new candidate TOML — an external
publication step that needs the user's account/authorization and is NOT
performed here. Until then, production runs in the locally-promoted mode
above; the promotion is fully reversible by relaunching the retained v4 or
candidate-1 scripts.

## Known-limitations carried forward (pre-existing, not fhw8-related)

Stochastic long-rendition truncation (~50% at ~9 s scripts); the
production no-text watchdog repair and transport-gate quiet-speech defect
remain open follow-ups; the Step 5 graph-construction basin defect affects
fresh engine constructions of all Nano artifacts (candidate constructions
bias divergent; one-token call-39 text flip; characterized, production
impact unproven).

## Stand-up for manual testing (2026-08-14)

Promoted service ready (`post-promotion-health.json`, production
candidate-1 identity, fhw8 nano manifest in provenance). Pipecat bot
running on 0.0.0.0:7860 from the host repo (production mode, no fixture
env); ngrok tunnel active. LAN: http://192.168.7.228:7860 — tunnel:
https://3fdf30831d45.ngrok.app
