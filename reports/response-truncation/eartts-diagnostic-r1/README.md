# EarTTS truncation diagnostic r1 evidence

Verdict: raw generated codes remained noncanonical while unchanged decoder
codes produced silent codec PCM; later PAD-tail policy substitutions were
downstream. No audio resumed after the production guard's frame 20 in 17
captured responses observed beyond that point.

The complete analysis is in
`docs/reviews/eartts-truncation-diagnostic-r1.md`.

## Live experiment

- `control-replay-live/`: one uninstrumented attempt and pinned-ASR output.
- `control-server-traces-live/`: corresponding model trace.
- `captured-replay-live/`: three attempts with decoded-silence=100 and frozen
  30-frame progress guards, including pinned-ASR output.
- `acoustic-capture-live/` and `capture-server-traces-live/`: 2,229 aligned
  acoustic positions and server traces for that pass.
- `captured-extended-replay-live/`: three attempts with decoded-silence,
  no-text, and no-audio thresholds all at 100, including pinned-ASR output.
- `acoustic-capture-extended-live/` and
  `capture-extended-server-traces-live/`: 2,572 aligned acoustic positions and
  server traces for the fully extended pass.
- `acoustic-analysis.json` and `acoustic-analysis-extended.json`: generated
  per-response raw/policy/decoder/PCM and timing summaries.
- `*-live-container.log` and `*-live-container.inspect*.json`: isolated
  container provenance and guard-override evidence.

## Earlier blocked-startup evidence

The `nemotron-voicechat-eartts-control-*.{log,inspect.json}` files retain the
pre-maintenance coexistence attempts. They are not part of the completed live
verdict but explain why the maintenance window was necessary.

`diagnostic-src/nemotron_voicechat_inference_wrapper.py` is the retained
startup-only wrapper used in those earlier attempts. The completed maintenance
window runs used the unmodified baked Speech wrapper.
