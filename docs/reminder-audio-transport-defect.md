# Function-call reminder audio is rendered but not transported

Status: open, independently reproduced twice on 2026-08-11. This defect is
separate from the fast-tool latency change and is not a compatibility
requirement for that change.

## Observed behavior

The NVIDIA-derived function-call path renders a layer-one acknowledgement while
an external tool is running. In two flag-off runs, the worker produced about two
seconds of reminder PCM, but the WebSocket client received no audio chunks in the
correlated reminder window:

- Deliberately slow tool (3,000 ms): 44,100 samples rendered over 2,003.9 ms;
  client witness 0 chunks, 0 bytes, peak -120 dBFS. The tool result reached phase
  two 0.292 ms after completion.
- Zero-delay tool: 45,864 samples rendered over 2,082.7 ms; client witness 0
  chunks, 0 bytes, peak -120 dBFS. The tool finished 14.1 ms after it started,
  but phase two waited another 2,080.2 ms for the inaudible reminder path.

Both turns completed with the correct model text and audible final response.
The result is therefore not a general output-audio failure.

Retained evidence:

- `step1-baseline-slow/report.json`: SHA-256 `518ca4a4769efa6f1c908ac01f50e05d672751293baf450a5e3dd4bb373f24d3`
- `step1-baseline-slow/cell-001/report.json`: SHA-256 `5672e746ac87a33e2b5d1e32afccc1ffc18516b63675211c6cf084ac66e98c7c`
- `step1-baseline-zero-r2/report.json`: SHA-256 `c9f74da700521af693d26dcfa1f935dea1073739cddbc88c96af5a97534bd41f`
- `step1-baseline-zero-r2/cell-001/report.json`: SHA-256 `cd36be8b582c70e244e9bc3284bd12144ea18cdb2ac75a853c5f7170245549ec`

All paths are beneath
`~/.local/state/nemotron-voicechat/traces/model/tool-call-latency-benchmark-v2/`.

## Current mechanism evidence

`_run_tts_reminder()` decodes each reminder token and enqueues each decoded chunk
on `tts_audio_output_queue`. While the function-call worker is alive,
`StreamingS2SPipeline.inner_generate_step()` drains that queue into the stream
state. The server-visible response does not publish those chunks during the
background-function early-return interval. The exact loss boundary still needs
an output-state/return-value trace before changing inference behavior.

## Follow-up gate

Instrument one reminder cycle at these adjacent boundaries:

1. reminder queue put (samples and cumulative queue samples);
2. pipeline queue drain and `state.update_state` (samples);
3. pipeline return value for that transport tick;
4. `VoiceChatEngine.process()` output samples;
5. WebSocket `response.output_audio.delta` send samples.

The eventual fix must deliver paced reminder audio without duplicating it in the
final assistant response, preserve interruption discard behavior, and retain the
normal final response audio. Until then, fast-tool qualification records the
transport witness but does not require this defect to remain present.
