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

The counter is gated on **explicit stream state**, not on timing.  Every
revision that inferred speech from timing inflated the total, in three distinct
ways: a fixed grace period counted each response's terminal drain; requiring
audio to resume still let a short inter-response gap be promoted by the next
response's first chunk; and both were still fooled by the silence-padded
trailing chunk ``BaseOutputTransport`` flushes on ``TTSStoppedFrame``, which
arrives half a second after a response's real audio has drained and looks
exactly like audio resuming.  A playout epoch, advanced at every boundary of a
contiguous TTS stream, removes all three by construction: a gap counts only when
the audio that closes it belongs to the same speech stream as the audio that
preceded it.

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
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

UNDERFLOW_TELEMETRY_ENV = "NEMOTRON_VOICECHAT_TRACK_UNDERFLOW_TELEMETRY"
OUTPUT_CHUNKS_ENV = "NEMOTRON_VOICECHAT_OUTPUT_10MS_CHUNKS"

#: Pipecat's default. ``BaseOutputTransport`` forwards only complete chunks of
#: this size, so it sets both how much generated audio waits in the output byte
#: buffer and how much slack the track queue has against clock jitter.
DEFAULT_OUTPUT_CHUNKS_10MS = 4

#: Backstop only.  The playout epoch decides whether a gap is inside a response;
#: this bounds the damage if the epoch signal is ever absent (tagger not in the
#: pipeline) or stale, in which case the counter must not silently attribute a
#: multi-second inter-response gap to speech.  Far above any stall that could
#: occur inside one response, so with the epoch wired it never fires.
MAX_MID_STREAM_GAP_SECONDS = 5.0

#: Per-gap records kept for post-hoc analysis. Bounded so a long session cannot
#: grow the counter without limit; the aggregate totals stay exact regardless.
MAX_RECORDED_RUNS = 500


class PlayoutEpoch:
    """Monotone id for the response currently feeding the output track.

    Advanced when a response starts emitting audio, so the track can tell the
    difference between "the queue emptied while this response was still
    speaking" and "this response ended and the next one has not begun".
    """

    __slots__ = ("value", "advances")

    def __init__(self) -> None:
        self.value = 0
        self.advances = 0

    def advance(self) -> int:
        self.value += 1
        self.advances += 1
        return self.value


#: Every boundary of a contiguous TTS stream.  ``TTSStoppedFrame`` is as
#: load-bearing as the start: ``BaseOutputTransport.handle_tts_stopped`` flushes
#: the trailing partial chunk *padded with silence* and queues it ahead of the
#: stop frame, so a chunk arrives at the track after the response's real audio
#: has already drained -- observed at 550-670 ms after the queue emptied, each
#: one landing within 10 ms of ``response.done``.  Without this boundary that
#: padded flush looks exactly like audio resuming, and the silence before it is
#: counted as a mid-speech gap.
_STREAM_BOUNDARY_FRAMES = (TTSStartedFrame, TTSStoppedFrame, InterruptionFrame)


class PlayoutEpochTagger(FrameProcessor):
    """Advance ``epoch`` at each boundary of a contiguous TTS stream.

    Placed between the Voicechat service and ``transport.output()``, so it sees
    every boundary before the output transport acts on it -- crucially before
    the stop-frame flush is enqueued, and before any of a new response's audio
    can reach the track.  The epoch therefore always changes *before* the audio
    it labels.  An interruption also advances it: the output transport discards
    its queued audio, so audio arriving afterwards is a different stream.
    """

    def __init__(self, epoch: PlayoutEpoch) -> None:
        super().__init__()
        self._epoch = epoch

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, _STREAM_BOUNDARY_FRAMES):
            self._epoch.advance()
        await self.push_frame(frame, direction)


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


class PlayoutUnderflowCounter:
    """Per-track tally of auto-silence emissions and observed queue depth.

    Distinguishes the two reasons the queue is empty.  Between responses it is
    empty almost always and the emitted silence is exactly right -- counting
    that would report ~100% underflow on any session where the bot is quiet.
    Only a queue that empties while audio is still streaming splices silence
    into speech, so that case is tallied separately as ``in_speech_ticks``.

    A run of zeros is attributed to a gap only when audio resumes *under the
    same playout epoch*.  Both conditions are load-bearing:

    * requiring audio to resume excludes each response's terminal drain, which
      an earlier fixed grace period counted as ~30 ticks and one run per
      response -- roughly half the reported total;
    * requiring the same epoch excludes both the inter-response gap, which the
      resume rule alone still let the *next* response's first chunk promote,
      and the stop-frame flush, whose silence-padded chunk closed a 550-670 ms
      tail from *within* the same response.

    Zeros keep the epoch in force when the run began, so the silence between a
    response's start and its first chunk (prebuffer) is discarded with the
    preceding tail rather than counted against the new response.

    The tradeoff is deliberate: a gap that ends a response, or that spans a
    response boundary, is not counted.  This undercounts rather than
    overcounts, which is the safe direction for a number used to decide whether
    to change production.
    """

    def __init__(self) -> None:
        self.recv_calls = 0
        self.silence_ticks = 0
        self.in_speech_ticks = 0
        self.in_speech_runs = 0
        self.tail_ticks_discarded = 0
        self.stale_epoch_runs_discarded = 0
        self.backstop_runs_discarded = 0
        self.first_in_speech_monotonic_s: float | None = None
        self.last_in_speech_monotonic_s: float | None = None
        self.min_depth_chunks: int | None = None
        self.depth_histogram: dict[int, int] = {}
        self.runs: list[dict[str, Any]] = []
        self.runs_truncated = 0
        self.max_epoch_seen = 0
        self._last_nonempty_monotonic_s: float | None = None
        self._pending_zero_ticks = 0
        self._pending_first_monotonic_s: float | None = None
        self._pending_epoch = 0

    def observe(self, depth_chunks: int, epoch: int = 0) -> None:
        self.recv_calls += 1
        now = time.monotonic()
        if epoch > self.max_epoch_seen:
            self.max_epoch_seen = epoch
        if depth_chunks > 0:
            self._close_pending_gap(now, epoch)
            self._last_nonempty_monotonic_s = now
            if self.min_depth_chunks is None or depth_chunks < self.min_depth_chunks:
                self.min_depth_chunks = depth_chunks
            bucket = depth_chunks if depth_chunks <= 8 else 9
            self.depth_histogram[bucket] = self.depth_histogram.get(bucket, 0) + 1
            return

        self.silence_ticks += 1
        if self._last_nonempty_monotonic_s is None:
            return
        if self._pending_zero_ticks == 0:
            self._pending_first_monotonic_s = now
            # The epoch that was speaking when the queue ran dry. Held for the
            # whole run so a later response cannot adopt these zeros.
            self._pending_epoch = epoch
        self._pending_zero_ticks += 1

    def _close_pending_gap(self, now: float, epoch: int) -> None:
        """Resolve zeros that preceded this chunk into a gap, or discard them.

        Audio resuming is what proves the preceding zeros interrupted a stream
        rather than trailed its end, and a matching epoch is what proves the
        resumed audio is the *same* stream.  Zeros never followed by more audio
        are the normal terminal drain and are discarded when the counter is
        summarised.
        """
        pending = self._pending_zero_ticks
        self._pending_zero_ticks = 0
        started = self._pending_first_monotonic_s
        self._pending_first_monotonic_s = None
        if pending == 0:
            return
        if epoch != self._pending_epoch:
            # A different response resumed: this was the previous one's tail
            # plus the new one's pre-roll, not a gap inside either.
            self.stale_epoch_runs_discarded += 1
            self.tail_ticks_discarded += pending
            return
        if started is None or (now - started) > MAX_MID_STREAM_GAP_SECONDS:
            self.backstop_runs_discarded += 1
            self.tail_ticks_discarded += pending
            return
        self.in_speech_ticks += pending
        self.in_speech_runs += 1
        if self.first_in_speech_monotonic_s is None:
            self.first_in_speech_monotonic_s = started
        self.last_in_speech_monotonic_s = now
        if len(self.runs) < MAX_RECORDED_RUNS:
            self.runs.append(
                {
                    "start_monotonic_s": round(started, 6),
                    "end_monotonic_s": round(now, 6),
                    "ticks": pending,
                    "ms": pending * 10,
                    "epoch": epoch,
                }
            )
        else:
            self.runs_truncated += 1

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot; ticks are 10 ms each.

        ``in_speech_ticks`` is the number that degrades audio; ``silence_ticks``
        includes every expected between-response tick and is reported only for
        context.  ``epoch_advances`` is a wiring check: a session with audio and
        zero advances means the tagger never ran, so the gate silently degraded
        to the backstop window and the in-speech total is not trustworthy.
        """
        return {
            "recv_calls": self.recv_calls,
            "in_speech_ticks": self.in_speech_ticks,
            "in_speech_ms": self.in_speech_ticks * 10,
            "in_speech_runs": self.in_speech_runs,
            # Zeros still awaiting resolution belong to a drain that never
            # resumed, i.e. a response tail; report them as discarded.
            "tail_ticks_discarded": self.tail_ticks_discarded + self._pending_zero_ticks,
            "stale_epoch_runs_discarded": self.stale_epoch_runs_discarded,
            "backstop_runs_discarded": self.backstop_runs_discarded,
            "silence_ticks_total": self.silence_ticks,
            "epoch_advances": self.max_epoch_seen,
            "min_depth_chunks_while_playing": self.min_depth_chunks,
            "min_depth_ms_while_playing": (
                None if self.min_depth_chunks is None else self.min_depth_chunks * 10
            ),
            "depth_histogram_chunks": dict(sorted(self.depth_histogram.items())),
            "first_in_speech_monotonic_s": self.first_in_speech_monotonic_s,
            "last_in_speech_monotonic_s": self.last_in_speech_monotonic_s,
            "runs": list(self.runs),
            "runs_truncated": self.runs_truncated,
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
        epoch = getattr(self, "voicechat_playout_epoch", None)
        counter.observe(len(self._chunk_queue), epoch.value if epoch is not None else 0)
        return await original_recv(self)

    RawAudioTrack.recv = recv  # type: ignore[method-assign]
    _underflow_installed = True
    logger.info("voicechat.playout.underflow_telemetry installed")
    return True


def attach_underflow_counter(
    transport: Any, epoch: PlayoutEpoch | None = None
) -> PlayoutUnderflowCounter | None:
    """Attach (or return) the counter on ``transport``'s output track.

    Call this while the client is connected and hold the returned reference.
    The transport clears ``_audio_output_track`` in ``_handle_client_closed``
    *before* it fires ``on_client_disconnected``, so a lookup performed from
    that handler always finds nothing.

    ``epoch`` is the same object the pipeline's :class:`PlayoutEpochTagger`
    advances; it is bound to the track here because the track is created by the
    transport, not by us.  It is attached before the first response can start,
    since the caller runs on client connect.
    """
    if not _underflow_installed:
        return None
    client = getattr(transport, "_client", None)
    track = getattr(client, "_audio_output_track", None) if client is not None else None
    if track is None:
        return None
    if epoch is None:
        epoch = getattr(transport, "voicechat_playout_epoch", None)
    if epoch is not None:
        track.voicechat_playout_epoch = epoch
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
