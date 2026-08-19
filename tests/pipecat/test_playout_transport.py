"""Tests for transport playout instrumentation and margin recovery.

The underflow counter exists to adjudicate a disagreement between models of why
audio crackles, so a counter that silently miscounts is worse than none at all.
An earlier version proved that: it called any zero within a fixed grace period
of the last chunk "in speech", which meant the terminal drain after every
response was counted, manufacturing roughly 30 ticks and one run per response.

These pin the corrected contract: a zero counts only once more real audio
arrives to close the gap.
"""

from __future__ import annotations

import pytest

from nemotron_voicechat_pipecat import playout_transport
from nemotron_voicechat_pipecat.playout_transport import (
    MAX_MID_STREAM_GAP_SECONDS,
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


def test_idle_session_reports_no_in_speech_ticks():
    """A quiet bot empties the queue constantly; none of it degrades audio."""
    counter = PlayoutUnderflowCounter()
    for _ in range(50):
        counter.observe(0)
    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["silence_ticks_total"] == 50


def test_response_tail_drain_is_not_counted(clock):
    """The exact false positive R2 identified: ~30 ticks after the last chunk.

    A response ends, the queue stays empty through the terminal drain, and
    seconds later the next response begins. None of those zeros interrupted
    speech, so none may be reported as in-speech.
    """
    counter = PlayoutUnderflowCounter()
    for _ in range(20):
        counter.observe(2)
        clock["now"] += 0.01
    for _ in range(30):  # 300 ms terminal drain
        counter.observe(0)
        clock["now"] += 0.01
    clock["now"] += 4.0  # gap between responses
    for _ in range(20):  # next response
        counter.observe(2)
        clock["now"] += 0.01

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["tail_ticks_discarded"] == 30


def test_gap_is_counted_only_when_audio_resumes(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(2)
    for _ in range(3):
        clock["now"] += 0.01
        counter.observe(0)
    clock["now"] += 0.01
    counter.observe(2)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 3
    assert summary["in_speech_runs"] == 1
    assert summary["tail_ticks_discarded"] == 0


def test_two_gaps_are_two_runs(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(2)
    for _ in range(4):
        clock["now"] += 0.01
        counter.observe(0)
    clock["now"] += 0.01
    counter.observe(2)
    for _ in range(2):
        clock["now"] += 0.01
        counter.observe(0)
    clock["now"] += 0.01
    counter.observe(3)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 6
    assert summary["in_speech_runs"] == 2


def test_unresolved_trailing_zeros_are_reported_as_tail(clock):
    """Zeros at session end never resume, so they are a tail, not a gap."""
    counter = PlayoutUnderflowCounter()
    counter.observe(3)
    for _ in range(12):
        clock["now"] += 0.01
        counter.observe(0)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 12


def test_stall_longer_than_the_window_is_dropped_not_promoted(clock):
    """Undercount rather than let a new response absorb an older tail."""
    counter = PlayoutUnderflowCounter()
    counter.observe(1)
    clock["now"] += 0.01
    counter.observe(0)
    clock["now"] += MAX_MID_STREAM_GAP_SECONDS + 0.05
    counter.observe(1)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 1


def test_zeros_before_any_audio_are_ignored(clock):
    """Silence before the first chunk cannot have interrupted anything."""
    counter = PlayoutUnderflowCounter()
    for _ in range(10):
        counter.observe(0)
        clock["now"] += 0.01
    counter.observe(2)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["tail_ticks_discarded"] == 0
    assert summary["silence_ticks_total"] == 10


def test_depth_histogram_only_records_playing_observations(clock):
    counter = PlayoutUnderflowCounter()
    counter.observe(0)
    counter.observe(2)
    counter.observe(5)
    counter.observe(0)
    hist = counter.summary()["depth_histogram_chunks"]
    assert 0 not in hist
    assert hist == {2: 1, 5: 1}


@pytest.mark.parametrize("depth", [1, 4, 9, 20])
def test_positive_depth_never_counts_as_underflow(depth):
    counter = PlayoutUnderflowCounter()
    for _ in range(5):
        counter.observe(depth)
    assert counter.summary()["in_speech_ticks"] == 0
    assert counter.summary()["silence_ticks_total"] == 0


def test_real_monotonic_clock_path_counts_a_closed_gap():
    """Guard the unpatched clock path so a fixture cannot mask a regression."""
    counter = PlayoutUnderflowCounter()
    counter.observe(2)
    counter.observe(0)
    counter.observe(0)
    counter.observe(2)
    assert counter.summary()["in_speech_ticks"] == 2
    assert counter.summary()["in_speech_runs"] == 1


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
