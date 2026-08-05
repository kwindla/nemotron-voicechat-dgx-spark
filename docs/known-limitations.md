# Known limitations

- Production Candidate 1 is single-client and single-GPU.
- Mean cadence passed, but the two-frame p95 target remains a documented near
  miss at roughly 82 ms versus 80 ms. Long-run queue behavior must be evaluated
  through Pipecat before declaring sustained realtime qualification complete.
- Sessions are guarded at 12,000 model frames. The client receives a graceful
  close reason rather than an allocation overrun.
- EarTTS uses the evaluated 1,500-position sliding-window artifact. Its late-turn
  quality and the wider sustained Pipecat campaign remain explicit gates.
- The public model can stochastically skip tools, choose an inappropriate
  advertised tool, mis-speak a returned value, enter canned-reply/self-talk
  loops, or drop transcript words. These behaviors are also described in the
  upstream NVIDIA Voicechat known limitations. Tool-router policy belongs in an
  application layer; it is not disguised as a wire compatibility feature here.
- The UTC-time tool is a useful full-path test but is not selected reliably on
  every prompt. When selected, its result path is required to complete without
  the previous circular EOU timeout.
- In the retained browser qualification sample, three sessions selected the
  intended response path and two selected the unrelated UTC-time tool. This is
  a measured five-session sample, not an estimate of the model's general error
  probability. The release gate permits one recorded retry, retains both
  attempt logs, and still fails after two unsuccessful attempts; it does not
  add an application tool router or retry until success.
- RNNT endpointing is configured for 1.6 seconds (`20 × 80 ms`) for natural
  human pauses. Changing it changes the qualified turn-taking behavior.
- Typed input is deliberately half-duplex and the model server owns the gate.
  Microphone PCM received from typed-job acceptance through its EOU-derived
  closure is discarded. This prevents RNNT from merging modalities but can drop
  a live barge-in; wait for the typed turn to finish when the utterance matters.
- The derived weights are published under OpenMDW-1.1 with origin notices. The
  runtime image is built locally from pinned public inputs rather than
  distributed as a prebuilt binary image.
