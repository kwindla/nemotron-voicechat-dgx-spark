# Architecture

```text
Browser (Pipecat /client/)
        │ SmallWebRTC media + RTVI voice/text events
        ▼
Host Pipecat bot :7860
        ├─ shared application context
        ├─ application tool handlers
        ├─ Silero VAD + local Pipecat Smart Turn v3
        └─ strict-v3 Voicechat adapter
                │ loopback WebSocket; PCM16 16 kHz in / 22.05 kHz out
                ▼
Docker model server :8786 (127.0.0.1 only)
        ├─ perception + embedded RNNT transcription/evidence
        ├─ W8 Nano vLLM + full-precision function head
        ├─ W8A32 EarTTS vLLM
        ├─ persistent two-core codec worker
        └─ isolated CPU-only Pocket TTS worker
                │ bounded length-prefixed JSON/raw-PCM Unix socket
                └─ typed text → synthetic user PCM → RNNT path
```

Voicechat owns continuous acoustic/model state. Pipecat owns microphone turn
start/end decisions and sends explicit correlated start/commit barriers. Voicechat
mirrors context for applications but never replays it as a stateless chat
completion. The qualified model is single-session and batch size one.

Typed input crosses the same model WebSocket as microphone audio. The model
server accepts one bounded job, asks the prewarmed Pocket worker for audio,
injects it through RNNT without wall-clock pacing, and issues the same explicit
model EOU used by Smart Turn commits after the server advances the missing
bounded RNNT phrase-settlement positions. Pipecat holds microphone input in a
bounded FIFO during typed and function-result mutation. Replacement is
serialized cancel-then-replace. Transcript source/job tags let
Pipecat commit the byte-exact typed text once without also committing its RNNT
transcription.

Direct mid-session source-token injection is intentionally not a production
capability. A structurally correct diagnostic reached only 10/20 semantic cases;
in the frozen c014 comparison, acoustic input selected the start-of-tool-call
token and produced the expected call while direct injection left PAD as argmax.
This is evidence that the injection method is distribution-shifted for the
released checkpoint/interface, not that native text is fundamentally
impossible. Initial system-prompt text remains a separate, supported prefill
surface. A future native user-text path requires a trained text-to-perception
adapter or checkpoint-provided multimodal interface and full requalification.

Pocket uses a separate Python executable with CPU-only PyTorch. Its environment
scrubs CUDA and Python library paths, its assets are immutable read-only HF
snapshots, and its CPU affinity excludes codec cores. Worker death degrades the
typed-input capability without corrupting the microphone path; startup is not
ready until prewarm succeeds.
Pocket's sampled latent noise is seeded from the frozen base seed plus a
SHA-256 digest of the exact text. The waveform is an internal model stimulus,
not delivered speech, so deterministic per-text synthesis removes avoidable
semantic variance without reducing any user-facing voice diversity.

Function results flow upstream immediately because Voicechat needs the result
before it can complete its response. The browser bundle is Pipecat's installed
debug client; this repository maintains no custom web application.
