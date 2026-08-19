# Transport underflow: corrected measurement

Every number this project has reported for "silence spliced into speech by the
WebRTC output track" was a measurement artifact. This records the corrected
result, the instrument that produced it, and the control that shows the
instrument can still see the fault it reports as absent.

## Result

On a 95 s replay of a real captured session (5 responses, ~27-42 s of generated
audio depending on run), the output track underran **zero times while a speech
stream was live**, across two runs:

| run | in-speech ticks | runs | epoch advances | audio observations |
|-----|-----------------|------|----------------|--------------------|
| `underflow-bot2` | 0 | 0 | 15 | 4180 |
| `underflow-bot3` | 0 | 0 | 15 | 2743 |

The queue depth histogram is a clean 1/2/3/4 sawtooth with equal counts in each
bucket: `BaseOutputTransport` hands the track four 10 ms chunks at a time and
the track drains them on its own clock, never arriving late.

**Transport-side underflow is therefore not the cause of the reported
crackling**, at least on this path. The replay runs headless against
`127.0.0.1`, so it exercises the server-side playout path only -- no ngrok, no
Wi-Fi, no browser jitter buffer. Those remain untested and are now the leading
candidates.

## Why the earlier numbers were wrong

Three separate false-positive classes, each found only after the previous fix:

1. **Terminal drain.** A fixed grace period called any zero within 0.3 s of the
   last chunk "in speech", so the drain after *every* response was counted:
   ~30 ticks and one run per response, roughly half the reported total. (Found
   by the independent R2 assessment.)
2. **Short inter-response gap.** Requiring audio to resume removed the drains
   but still let the *next* response's first chunk promote a gap shorter than
   the time window.
3. **Stop-frame flush.** `BaseOutputTransport.handle_tts_stopped` flushes the
   trailing partial chunk *padded with silence* and queues it ahead of the stop
   frame. It arrives 550-670 ms after the response's real audio has drained --
   measured landing within 9-10 ms of `response.done` in all four instances --
   and looks exactly like audio resuming, from within the same response.

The current gate uses no timing at all: a run of zeros counts only when audio
resumes under the same playout epoch, and the epoch is fenced at every boundary
of a contiguous TTS stream (`TTSStartedFrame`, `TTSStoppedFrame`,
`InterruptionFrame`). A time window survives only as a backstop against a
missing epoch signal, and did not fire in either run.

## Negative control

A zero from an instrument that has never been shown to read non-zero is not
evidence. `stall_injector.py` withholds audio from the track for 200 ms every
20 writes -- a genuine mid-stream underflow, in order, with no reordering.

    in_speech_ticks 508 | runs 26 | epoch advances 11
    run tick counts: [20, 19, 20, 20, 20, 19, 20, 20, ...]

26 injections, 26 runs, 19-20 ticks each against 200 ms injected. The counter
resolves a mid-speech stall to roughly one tick and attributes exactly one run
per fault.

## Reproducing

    python replay_probe.py <48kHz-mono-wav> 95        # against a running bot
    NEMOTRON_VOICECHAT_TRACK_UNDERFLOW_TELEMETRY=1    # required; off by default

The replay WAV is built from a retained model trace
(`input-16000-mono-s16le.pcm`, resampled to 48 kHz for Chrome's fake capture
device). The summary is logged by the bot on client disconnect.
