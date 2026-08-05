# Architecture

```text
Browser (Pipecat /client/)
        │ SmallWebRTC media + RTVI voice/text events
        ▼
Host Pipecat bot :7860
        ├─ shared application context
        ├─ application tool handlers
        └─ strict-v3 Voicechat adapter
                │ loopback WebSocket; PCM16 16 kHz in / 22.05 kHz out
                ▼
Docker model server :8786 (127.0.0.1 only)
        ├─ perception + embedded RNNT turn detection
        ├─ W8 Nano vLLM + full-precision function head
        ├─ W8A32 EarTTS vLLM
        ├─ persistent two-core codec worker
        └─ isolated CPU-only Pocket TTS worker
                │ bounded length-prefixed JSON/raw-PCM Unix socket
                └─ typed text → synthetic user PCM → RNNT path
```

Voicechat owns continuous acoustic/model state and authoritative turns. Pipecat
mirrors context for applications but never replays it as a stateless chat
completion. The qualified model is single-session and batch size one.

Typed input crosses the same model WebSocket as microphone audio. The model
server accepts one bounded job, gates microphone PCM, asks the prewarmed Pocket
worker for audio, injects it through RNNT, and sends EOU-derived closure silence.
Replacement is serialized cancel-then-replace. Transcript source/job tags let
Pipecat commit the byte-exact typed text once without also committing its RNNT
transcription.

Pocket uses a separate Python executable with CPU-only PyTorch. Its environment
scrubs CUDA and Python library paths, its assets are immutable read-only HF
snapshots, and its CPU affinity excludes codec cores. Worker death degrades the
typed-input capability without corrupting the microphone path; startup is not
ready until prewarm succeeds.

Function results flow upstream immediately because Voicechat needs the result
before it can complete its response. The browser bundle is Pipecat's installed
debug client; this repository maintains no custom web application.
