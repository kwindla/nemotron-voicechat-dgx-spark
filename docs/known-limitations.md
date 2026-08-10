# Known limitations

- Production Candidate 1 is single-client and single-GPU.
- Mean cadence passed, but the two-frame p95 target remains a documented near
  miss at roughly 82 ms versus 80 ms. The paced 12,000-frame qualification
  nevertheless passed the governing no-unbounded-queue-growth criterion on
  both downloaded and locally reproduced weights.
- Sessions are guarded at 12,000 model frames. The client receives a graceful
  close reason rather than an allocation overrun.
- EarTTS uses the evaluated 1,500-position sliding-window artifact. Its late-turn
  quality passed independent ASR throughout the retained long-run campaigns.
- The public model can stochastically skip tools, choose an inappropriate
  advertised tool, mis-speak a returned value, enter canned-reply/self-talk
  loops, or drop transcript words. These behaviors are also described in the
  upstream NVIDIA Voicechat known limitations. Tool-router policy belongs in an
  application layer; it is not disguised as a wire compatibility feature here.
- The UTC-time tool is a useful full-path test but is not selected reliably on
  every prompt. When selected, its result path is required to complete without
  the previous circular EOU timeout.
- Across all 15 semantically usable retained browser sessions from the
  2026-08-05 harness-development and qualification campaigns, eight selected
  the intended response path, five selected the unrelated UTC-time tool, and
  two ended in non-tool browser assertion/timeout failures. Runs where missing
  Chromium or a locator defect prevented a usable model verdict are excluded.
  This is a complete count of the retained usable sessions, not an estimate of
  the model's general error probability. The release gate permits one recorded
  retry, retains both attempt logs, and still fails after two unsuccessful
  attempts; it does not add an application tool router or retry until success.
- One retained source-reproduced run entered a malformed post-function-call
  path late in the session, stopped accepting two-frame drafts, and accumulated
  2.98 seconds of implied playback debt before the session limit. A fresh-stack
  repeat answered all 18 typed turns and passed with 400 ms peak debt. The
  failed run remains evidence of a stochastic recovery weakness rather than
  being averaged into the successful qualification. This occurred in one of
  the two retained full-length source-runtime runs inspected for this specific
  mechanism; the tiny, trigger-unequal sample is evidence, not an estimate of
  its general probability. In the failed trace, sequential execution persisted
  for 510 frames and pairing resumed only when the next typed/user turn changed
  the model token stream. Follow-up short-context stress runs observed 23
  post-function-call recovery fences, including repeated and over-called tool
  cycles: every observable fence returned to PAD on its first post-close model
  frame and buffered a pair within 0–1 frame, with no trace-integrity fault.
  The old failing trace lacks effective function-token IDs, so scheduler
  behavior is not changed speculatively; future long qualifications should use
  the opt-in PAD-pair trace to capture a natural recurrence.
- RNNT endpointing is configured for 1.6 seconds (`20 × 80 ms`) for natural
  human pauses. Changing it changes the qualified turn-taking behavior.
- Typed input is deliberately half-duplex and the model server owns the gate.
  Microphone PCM received from typed-job acceptance through its EOU-derived
  closure is discarded. This prevents RNNT from merging modalities but can drop
  a live barge-in; wait for the typed turn to finish when the utterance matters.
- The derived weights are published under OpenMDW-1.1 with origin notices. The
  runtime image is built locally from pinned public inputs rather than
  distributed as a prebuilt binary image.
