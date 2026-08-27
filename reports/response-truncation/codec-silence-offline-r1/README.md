# EarTTS codec-silence offline evidence r1

This directory contains offline decodes of the four ASR-confirmed captured
suffix losses. The result is uniform: production, pointwise-off,
persistence-off, and native-reference decoding all produce zero audible frames
after the live silence onset. The production replay is sample-exact against the
retained live PCM in every case.

Case directories:

- `regular-a1-r1`: 6-word suffix loss, silence onset frame 83;
- `regular-a1-r3`: 14-word suffix loss, silence onset frame 666;
- `regular-a3-r3`: 2-word suffix loss, silence onset frame 782;
- `extended-a1-r3`: fully extended 2-word suffix loss, silence onset frame 722,
  plus the extra fresh-response-cache control.

Within each case:

- `matrix-summary.json` is the compact cross-variant result;
- `matrix.jsonl` contains every raw code vector and per-position dBFS for direct
  comparison;
- `<variant>/summary.json` records codec identity, semantics, and aggregate
  results;
- `<variant>/positions.jsonl` records the full position-level decode;
- `<variant>/raw-response.wav` is the decoded raw-code response;
- `production/actual-decoder-response.wav` is the faithfulness replay.

See `docs/reviews/eartts-codec-silence-r1.md` for method, interpretation, and
production-safety evidence.
