"""Tests for transport playout instrumentation and margin recovery.

The underflow counter exists to adjudicate a disagreement between two
timestamp-based models, so a counter that silently miscounts would be worse
than none at all. These pin the discrimination that makes its number mean
something: silence emitted between responses is expected, silence emitted
while audio is still streaming is the defect.
"""

from __future__ import annotations

import time

import pytest

from nemotron_voicechat_pipecat import playout_transport
from nemotron_voicechat_pipecat.playout_transport import (
    IN_SPEECH_GAP_SECONDS,
    PlayoutUnderflowCounter,
    install_warm_resampler,
)


def test_idle_session_reports_no_in_speech_ticks():
    """A quiet bot empties the queue constantly; none of it degrades audio."""
    counter = PlayoutUnderflowCounter()
    for _ in range(50):
        counter.observe(0)
    summary = counter.summary()
    assert summary["in_speech_ticks"] == 0
    assert summary["in_speech_runs"] == 0
    assert summary["silence_ticks_total"] == 50


def test_drain_while_streaming_is_counted_and_grouped_into_runs():
    counter = PlayoutUnderflowCounter()
    for _ in range(10):
        counter.observe(3)
    for _ in range(4):
        counter.observe(0)
    for _ in range(10):
        counter.observe(2)
    for _ in range(2):
        counter.observe(0)

    summary = counter.summary()
    assert summary["in_speech_ticks"] == 6
    assert summary["in_speech_ms"] == 60
    # Two separate audible splices, not one six-tick event.
    assert summary["in_speech_runs"] == 2
    assert summary["min_depth_chunks_while_playing"] == 2


def test_silence_after_speech_stops_is_not_counted(monkeypatch):
    """Once the bot stops talking, an empty queue is correct behaviour."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(playout_transport.time, "monotonic", lambda: clock["now"])

    counter = PlayoutUnderflowCounter()
    counter.observe(3)
    clock["now"] += IN_SPEECH_GAP_SECONDS + 0.05
    for _ in range(5):
        counter.observe(0)

    assert counter.summary()["in_speech_ticks"] == 0
    assert counter.summary()["silence_ticks_total"] == 5


def test_gap_boundary_is_inclusive_of_recent_audio(monkeypatch):
    """A drain just inside the window still counts; just outside does not."""
    clock = {"now": 500.0}
    monkeypatch.setattr(playout_transport.time, "monotonic", lambda: clock["now"])

    inside = PlayoutUnderflowCounter()
    inside.observe(1)
    clock["now"] += IN_SPEECH_GAP_SECONDS - 0.01
    inside.observe(0)
    assert inside.summary()["in_speech_ticks"] == 1

    clock["now"] = 500.0
    outside = PlayoutUnderflowCounter()
    outside.observe(1)
    clock["now"] += IN_SPEECH_GAP_SECONDS + 0.01
    outside.observe(0)
    assert outside.summary()["in_speech_ticks"] == 0


def test_depth_histogram_only_records_playing_observations():
    counter = PlayoutUnderflowCounter()
    counter.observe(0)
    counter.observe(2)
    counter.observe(5)
    counter.observe(0)
    hist = counter.summary()["depth_histogram_chunks"]
    assert 0 not in hist
    assert hist == {2: 1, 5: 1}


def test_warm_resampler_keeps_filter_history_across_responses():
    """The default clears SOXR state after 0.2 s, which every gap exceeds."""
    install_warm_resampler()
    from pipecat.transports import base_output

    resampler = base_output.create_stream_resampler()
    assert resampler._clear_after_secs is None


def test_warm_resampler_install_is_idempotent():
    assert install_warm_resampler() is True
    assert install_warm_resampler() is True


@pytest.mark.parametrize("depth", [1, 4, 9, 20])
def test_positive_depth_never_counts_as_underflow(depth):
    counter = PlayoutUnderflowCounter()
    for _ in range(5):
        counter.observe(depth)
    assert counter.summary()["in_speech_ticks"] == 0
    assert counter.summary()["silence_ticks_total"] == 0


def test_real_monotonic_clock_path_discriminates():
    """Guard the unpatched path too, so the fixture cannot mask a regression."""
    counter = PlayoutUnderflowCounter()
    counter.observe(2)
    counter.observe(0)
    assert counter.summary()["in_speech_ticks"] == 1
    time.sleep(IN_SPEECH_GAP_SECONDS + 0.02)
    counter.observe(0)
    assert counter.summary()["in_speech_ticks"] == 1
