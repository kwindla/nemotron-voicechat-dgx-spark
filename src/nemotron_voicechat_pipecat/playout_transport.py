"""Transport-side playout instrumentation and margin recovery.

Two independent concerns, both about the audio path between the Voicechat
service and the WebRTC track.

**Underflow telemetry (opt-in).**  ``RawAudioTrack`` keeps no counter for the
case that actually degrades audio: an empty chunk queue with ``auto_silence``
enabled, where it emits a 10 ms zero chunk, advances its timestamp, and leaves
the splice permanently in the content timeline.  Both the model-trace analysis
and the independent Codex assessment inferred these ticks from timestamps and
disagreed by nearly two orders of magnitude (1 large gap versus ~62 small
ticks).  This module counts them directly so the question is settled by
observation.

**Margin recovery.**  One mechanism silently consumes playout lead before audio
is playable, and reclaiming it costs no latency:

1. ``SOXRStreamAudioResampler`` clears its filter history after
   ``clear_after_secs`` of inactivity (default 0.2 s).  Gaps between responses
   are far longer than that, so every response resamples on a *fresh* stream
   and pays the initial filter latency again: the first two 80 ms deltas expose
   135.5 ms rather than 160 ms.  Keeping the stream warm reclaims that.

This does not change when audio starts playing, so it adds no voice-to-voice
latency.

``audio_out_10ms_chunks`` is exposed here as well, but it is a genuine tradeoff
rather than free margin: a smaller value strands less generated audio in the
output byte buffer, yet ``BaseOutputTransport`` feeds the track from its own
realtime clock, so that same value is the only slack absorbing jitter between
that clock and the track's.  Measure before changing it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

UNDERFLOW_TELEMETRY_ENV = "NEMOTRON_VOICECHAT_TRACK_UNDERFLOW_TELEMETRY"
OUTPUT_CHUNKS_ENV = "NEMOTRON_VOICECHAT_OUTPUT_10MS_CHUNKS"

#: Pipecat's default. ``BaseOutputTransport`` forwards only complete chunks of
#: this size, so it sets both how much generated audio waits in the output byte
#: buffer and how much slack the track queue has against clock jitter.
DEFAULT_OUTPUT_CHUNKS_10MS = 4


def output_chunks_10ms() -> int:
    """Return the configured ``audio_out_10ms_chunks`` value."""
    raw = os.environ.get(OUTPUT_CHUNKS_ENV, "").strip()
    if not raw:
        return DEFAULT_OUTPUT_CHUNKS_10MS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{OUTPUT_CHUNKS_ENV} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{OUTPUT_CHUNKS_ENV} must be a positive integer")
    return value


_underflow_installed = False
_warm_resampler_installed = False


#: A queue that empties within this long of the last real chunk drained mid
#: playout. Chunks arrive roughly every 80 ms while the bot speaks, so a longer
#: quiet stretch means the bot simply is not talking and the silence is
#: expected rather than a defect.
IN_SPEECH_GAP_SECONDS = 0.3


class PlayoutUnderflowCounter:
    """Per-track tally of auto-silence emissions and observed queue depth.

    Distinguishes the two reasons the queue is empty.  Between responses it is
    empty almost always and the emitted silence is exactly right -- counting
    that would report ~100% underflow on any session where the bot is quiet.
    Only a queue that empties while audio is still streaming splices silence
    into speech, so that case is tallied separately as ``in_speech_ticks``.
    """

    def __init__(self) -> None:
        self.recv_calls = 0
        self.silence_ticks = 0
        self.in_speech_ticks = 0
        self.in_speech_runs = 0
        self.first_in_speech_monotonic_s: float | None = None
        self.last_in_speech_monotonic_s: float | None = None
        self.min_depth_chunks: int | None = None
        self.depth_histogram: dict[int, int] = {}
        self._last_nonempty_monotonic_s: float | None = None
        self._previous_was_in_speech = False

    def observe(self, depth_chunks: int) -> None:
        self.recv_calls += 1
        now = time.monotonic()
        if depth_chunks > 0:
            self._last_nonempty_monotonic_s = now
            self._previous_was_in_speech = False
            if self.min_depth_chunks is None or depth_chunks < self.min_depth_chunks:
                self.min_depth_chunks = depth_chunks
            bucket = depth_chunks if depth_chunks <= 8 else 9
            self.depth_histogram[bucket] = self.depth_histogram.get(bucket, 0) + 1
            return

        self.silence_ticks += 1
        last = self._last_nonempty_monotonic_s
        if last is None or (now - last) > IN_SPEECH_GAP_SECONDS:
            self._previous_was_in_speech = False
            return
        self.in_speech_ticks += 1
        if not self._previous_was_in_speech:
            self.in_speech_runs += 1
        self._previous_was_in_speech = True
        if self.first_in_speech_monotonic_s is None:
            self.first_in_speech_monotonic_s = now
        self.last_in_speech_monotonic_s = now

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot; ticks are 10 ms each.

        ``in_speech_ticks`` is the number that degrades audio; ``silence_ticks``
        includes every expected between-response tick and is reported only for
        context.
        """
        return {
            "recv_calls": self.recv_calls,
            "in_speech_ticks": self.in_speech_ticks,
            "in_speech_ms": self.in_speech_ticks * 10,
            "in_speech_runs": self.in_speech_runs,
            "silence_ticks_total": self.silence_ticks,
            "min_depth_chunks_while_playing": self.min_depth_chunks,
            "min_depth_ms_while_playing": (
                None if self.min_depth_chunks is None else self.min_depth_chunks * 10
            ),
            "depth_histogram_chunks": dict(sorted(self.depth_histogram.items())),
            "first_in_speech_monotonic_s": self.first_in_speech_monotonic_s,
            "last_in_speech_monotonic_s": self.last_in_speech_monotonic_s,
        }


def install_underflow_telemetry() -> bool:
    """Count auto-silence emissions on every ``RawAudioTrack``.

    Returns True when instrumentation is active.  Enabled only when
    ``NEMOTRON_VOICECHAT_TRACK_UNDERFLOW_TELEMETRY`` is set, so production
    behaviour is unchanged by default.

    The wrapper reproduces ``recv``'s own pacing wait before sampling the
    queue.  That matters: ``recv`` sleeps until the frame is due and *then*
    tests the queue, so sampling before the sleep would count chunks that
    arrive during it.  After the replicated wait the original's own wait
    computes to <= 0 and it does not suspend again, so it observes exactly the
    queue this wrapper measured.
    """
    global _underflow_installed
    if _underflow_installed:
        return True
    if not os.environ.get(UNDERFLOW_TELEMETRY_ENV, "").strip():
        return False

    import asyncio

    from pipecat.transports.smallwebrtc.transport import RawAudioTrack

    original_recv = RawAudioTrack.recv

    async def recv(self: Any):  # type: ignore[no-untyped-def]
        counter = getattr(self, "voicechat_underflow_counter", None)
        if counter is None:
            counter = PlayoutUnderflowCounter()
            self.voicechat_underflow_counter = counter
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        counter.observe(len(self._chunk_queue))
        return await original_recv(self)

    RawAudioTrack.recv = recv  # type: ignore[method-assign]
    _underflow_installed = True
    logger.info("voicechat.playout.underflow_telemetry installed")
    return True


def attach_underflow_counter(transport: Any) -> PlayoutUnderflowCounter | None:
    """Attach (or return) the counter on ``transport``'s output track.

    Call this while the client is connected and hold the returned reference.
    The transport clears ``_audio_output_track`` in ``_handle_client_closed``
    *before* it fires ``on_client_disconnected``, so a lookup performed from
    that handler always finds nothing.
    """
    if not _underflow_installed:
        return None
    client = getattr(transport, "_client", None)
    track = getattr(client, "_audio_output_track", None) if client is not None else None
    if track is None:
        return None
    counter = getattr(track, "voicechat_underflow_counter", None)
    if counter is None:
        counter = PlayoutUnderflowCounter()
        track.voicechat_underflow_counter = counter
    return counter


def install_warm_resampler() -> bool:
    """Stop discarding SOXR filter history between responses.

    ``BaseOutputTransport`` builds its resampler with
    ``create_stream_resampler()`` and no arguments, taking the 0.2 s
    clear-after-inactivity default.  Conversational gaps always exceed that, so
    the resampler is cold at the start of every response and re-pays its
    initial filter latency exactly when playout lead is scarcest.
    """
    global _warm_resampler_installed
    if _warm_resampler_installed:
        return True

    from pipecat.transports import base_output

    original_factory = base_output.create_stream_resampler

    def create_stream_resampler(**kwargs: Any):  # type: ignore[no-untyped-def]
        kwargs.setdefault("clear_after_secs", None)
        return original_factory(**kwargs)

    base_output.create_stream_resampler = create_stream_resampler  # type: ignore[assignment]
    _warm_resampler_installed = True
    logger.info("voicechat.playout.warm_resampler installed (clear_after_secs=None)")
    return True
