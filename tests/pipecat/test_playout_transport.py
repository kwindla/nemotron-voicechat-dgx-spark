"""Tests for transport playout instrumentation and margin recovery.

The underflow counter exists to adjudicate a disagreement between models of why
audio crackles, so a counter that silently miscounts is worse than none at all.
Two revisions proved that, and both failures were the same mistake -- deciding
from timing whether a zero interrupted speech:

* the first called any zero within a fixed grace period of the last chunk "in
  speech", so every response's terminal drain was counted (~30 ticks and one
  run per response, roughly half the reported total);
* its replacement required audio to resume, which removed the drains but still
  let a *short inter-response gap* be promoted by the next response's first
  chunk;
* gating on the response alone still counted the silence-padded chunk that
  ``BaseOutputTransport`` flushes on ``TTSStoppedFrame``, which arrives long
  after the response's real audio drained but before the next response starts.

These pin the contract that replaced all three: a run of zeros counts only when
audio resumes under the same playout epoch, and the epoch is fenced at every
boundary of a contiguous TTS stream.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import (
    AggregationType,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)

from nemotron_voicechat_pipecat import playout_transport
from nemotron_voicechat_pipecat.playout_transport import (
    MAX_MID_STREAM_GAP_SECONDS,
    PlayoutEpoch,
    PlayoutEpochTagger,
    PlayoutUnderflowCounter,
    install_warm_resampler,
    output_chunks_10ms,
)


@pytest.fixture
def clock(monkeypatch):
    """Deterministic monotonic clock, in seconds."""
    state = {"now": 1000.0}
    monkeypatch.setattr(playout_transport.time, "monotonic", lambda: state["now"])
    return state


def _stream(counter, clock, ticks, depth, epoch):
    for _ in range(ticks):
        counter.observe(depth, epoch)
        clock["now"] += 0.01


def test_idle_session_reports_no_in_speech_ticks():
    """A quiet bot empties the queue constantly; none of it degrades audio."""
    counter = PlayoutUnderflowCounter()
    for _ in range(50):
        counter.observe(0, 1)
    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["silence_ticks_total"] == 50


def test_response_tail_drain_is_not_counted(clock):
    """A response's terminal drain never interrupted speech."""
    counter = PlayoutUnderflowCounter()
    _stream(counter, clock, 20, 2, 1)
    _stream(counter, clock, 30, 0, 1)  # 300 ms terminal drain
    clock["now"] += 4.0
    _stream(counter, clock, 20, 2, 2)  # next response, new epoch

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["tail_ticks_discarded"] == 30
    assert summary["stale_epoch_runs_discarded"] == 1


def test_short_inter_response_gap_is_not_promoted(clock):
    """The failure that survived the previous fix.

    The bot stops, the queue drains, and the next response begins well inside
    any plausible time window. Requiring audio to resume does not exclude this;
    only the epoch does.
    """
    counter = PlayoutUnderflowCounter()
    _stream(counter, clock, 20, 2, 1)
    _stream(counter, clock, 40, 0, 1)  # 400 ms, far under the backstop
    _stream(counter, clock, 20, 3, 2)  # next response starts speaking

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["tail_ticks_discarded"] == 40
    assert summary["stale_epoch_runs_discarded"] == 1


def test_stop_frame_flush_does_not_close_a_tail(clock):
    """The third false-positive class, measured at 550-670 ms per response.

    A response's real audio drains, the queue sits empty while the model
    finalises, and then ``BaseOutputTransport`` flushes its trailing partial
    chunk padded with silence. That chunk is not speech resuming, and the
    ``TTSStoppedFrame`` that triggers it has already advanced the epoch.
    """
    counter = PlayoutUnderflowCounter()
    _stream(counter, clock, 30, 4, 5)  # response speaking
    _stream(counter, clock, 58, 0, 5)  # 580 ms drain while the response finalises
    _stream(counter, clock, 4, 1, 6)  # padded flush, fenced onto the next epoch

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["tail_ticks_discarded"] == 58
    assert summary["stale_epoch_runs_discarded"] == 1


def test_prebuffer_silence_before_a_new_response_is_not_counted(clock):
    """Zeros spanning a response boundary belong to neither response.

    The epoch advances when a response starts, but its audio only arrives after
    the prebuffer fills. Those zeros carry the *previous* epoch, so the new
    response's first chunk cannot adopt them.
    """
    counter = PlayoutUnderflowCounter()
    _stream(counter, clock, 10, 2, 1)
    _stream(counter, clock, 12, 0, 1)  # tail of response 1
    _stream(counter, clock, 16, 0, 2)  # response 2 started; prebuffer filling
    _stream(counter, clock, 10, 4, 2)  # response 2 audio arrives

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 28


def test_gap_inside_one_response_is_counted(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(2, 1)
    for _ in range(3):
        clock["now"] += 0.01
        counter.observe(0, 1)
    clock["now"] += 0.01
    counter.observe(2, 1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 3
    assert summary["in_speech_runs"] == 1
    assert summary["tail_ticks_discarded"] == 0


def test_two_gaps_in_one_response_are_two_runs(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(2, 1)
    for _ in range(4):
        clock["now"] += 0.01
        counter.observe(0, 1)
    clock["now"] += 0.01
    counter.observe(2, 1)
    for _ in range(2):
        clock["now"] += 0.01
        counter.observe(0, 1)
    clock["now"] += 0.01
    counter.observe(3, 1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 6
    assert summary["in_speech_runs"] == 2


def test_counted_runs_record_their_extent(clock):
    """Per-gap records, so a total can be re-derived rather than trusted."""
    counter = PlayoutUnderflowCounter()
    counter.observe(2, 7)
    start = clock["now"]
    for _ in range(5):
        clock["now"] += 0.01
        counter.observe(0, 7)
    clock["now"] += 0.01
    counter.observe(2, 7)

    (run,) = counter.summary()["runs"]
    assert run["ticks"] == 5
    assert run["ms"] == 50
    assert run["epoch"] == 7
    assert run["start_monotonic_s"] == pytest.approx(start + 0.01)
    assert run["end_monotonic_s"] > run["start_monotonic_s"]


def test_recorded_runs_are_bounded_but_totals_stay_exact(clock, monkeypatch):
    monkeypatch.setattr(playout_transport, "MAX_RECORDED_RUNS", 3)
    counter = PlayoutUnderflowCounter()
    for _ in range(10):
        counter.observe(2, 1)
        clock["now"] += 0.01
        counter.observe(0, 1)
        clock["now"] += 0.01

    counter.observe(2, 1)
    summary = counter.summary()
    assert len(summary["runs"]) == 3
    assert summary["runs_truncated"] == 7
    assert summary["in_speech_runs"] == 10
    assert summary["in_speech_ticks"] == 10


def test_unresolved_trailing_zeros_are_reported_as_tail(clock):
    """Zeros at session end never resume, so they are a tail, not a gap."""
    counter = PlayoutUnderflowCounter()
    counter.observe(3, 1)
    for _ in range(12):
        clock["now"] += 0.01
        counter.observe(0, 1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 12


def test_backstop_drops_an_implausibly_long_same_epoch_gap(clock):
    """Defence in depth for a missing or stale epoch signal."""
    counter = PlayoutUnderflowCounter()
    counter.observe(1, 1)
    clock["now"] += 0.01
    counter.observe(0, 1)
    clock["now"] += MAX_MID_STREAM_GAP_SECONDS + 0.05
    counter.observe(1, 1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 1
    assert summary["backstop_runs_discarded"] == 1


def test_zeros_before_any_audio_are_ignored(clock):
    """Silence before the first chunk cannot have interrupted anything."""
    counter = PlayoutUnderflowCounter()
    for _ in range(10):
        counter.observe(0, 1)
        clock["now"] += 0.01
    counter.observe(2, 1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 0
    assert summary["silence_ticks_total"] == 10


def test_summary_reports_epoch_advances_as_a_wiring_check():
    """Zero advances during a session with audio means the gate never ran."""
    counter = PlayoutUnderflowCounter()
    counter.observe(2, 0)
    assert counter.summary()["epoch_advances"] == 0
    counter.observe(2, 4)
    assert counter.summary()["epoch_advances"] == 4


def test_depth_histogram_only_records_playing_observations(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(0, 1)
    counter.observe(2, 1)
    counter.observe(5, 1)
    counter.observe(0, 1)
    hist = counter.summary()["depth_histogram_chunks"]
    assert 0 not in hist
    assert hist == {2: 1, 5: 1}


@pytest.mark.parametrize("depth", [1, 4, 9, 20])
def test_positive_depth_never_counts_as_underflow(depth):
    counter = PlayoutUnderflowCounter()
    for _ in range(5):
        counter.observe(depth, 1)
    assert counter.summary()["in_speech_ticks"] == 0
    assert counter.summary()["silence_ticks_total"] == 0


def test_real_monotonic_clock_path_counts_a_closed_gap():
    """Guard the unpatched clock path so a fixture cannot mask a regression."""
    counter = PlayoutUnderflowCounter()
    counter.observe(2, 1)
    counter.observe(0, 1)
    counter.observe(0, 1)
    counter.observe(2, 1)
    assert counter.summary()["in_speech_ticks"] == 2
    assert counter.summary()["in_speech_runs"] == 1


def test_playout_epoch_advances_monotonically():
    epoch = PlayoutEpoch()
    assert epoch.value == 0
    assert epoch.advance() == 1
    assert epoch.advance() == 2
    assert epoch.advances == 2


def _run_frames(processor, frames):
    async def drive():
        for frame in frames:
            await processor.process_frame(frame, playout_transport.FrameDirection.DOWNSTREAM)

    asyncio.run(drive())


def test_tagger_advances_on_every_tts_stream_boundary():
    """Stop is a boundary too: the stop-frame flush must land on a new epoch."""
    epoch = PlayoutEpoch()
    tagger = PlayoutEpochTagger(epoch)
    _run_frames(tagger, [TTSStartedFrame(), TTSStoppedFrame(), TTSStartedFrame()])
    assert epoch.value == 3


def test_tagger_ignores_frames_that_are_not_stream_boundaries():
    epoch = PlayoutEpoch()
    tagger = PlayoutEpochTagger(epoch)
    _run_frames(tagger, [TTSTextFrame("hello", aggregated_by=AggregationType.SENTENCE)])
    assert epoch.value == 0


def test_tagger_passes_frames_through_unchanged():
    """It observes the stream; it must not alter it."""
    epoch = PlayoutEpoch()
    tagger = PlayoutEpochTagger(epoch)
    seen: list[Frame] = []

    async def capture(frame, direction):
        seen.append(frame)

    tagger.push_frame = capture  # type: ignore[method-assign]
    sent = [TTSStartedFrame(), TTSStoppedFrame()]
    _run_frames(tagger, sent)
    assert seen == sent


def test_warm_resampler_keeps_filter_history_across_responses():
    """The default clears SOXR state after 0.2 s, which every gap exceeds."""
    install_warm_resampler()
    from pipecat.transports import base_output

    resampler = base_output.create_stream_resampler()
    assert resampler._clear_after_secs is None


def test_warm_resampler_install_is_idempotent():
    assert install_warm_resampler() is True
    assert install_warm_resampler() is True


def test_output_chunks_defaults_to_pipecat_default(monkeypatch):
    monkeypatch.delenv(playout_transport.OUTPUT_CHUNKS_ENV, raising=False)
    assert output_chunks_10ms() == playout_transport.DEFAULT_OUTPUT_CHUNKS_10MS == 4


@pytest.mark.parametrize("bad", ["0", "-1", "nonsense"])
def test_output_chunks_rejects_invalid_values(monkeypatch, bad):
    monkeypatch.setenv(playout_transport.OUTPUT_CHUNKS_ENV, bad)
    with pytest.raises(ValueError):
        output_chunks_10ms()
