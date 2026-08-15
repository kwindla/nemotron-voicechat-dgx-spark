from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools" / "qualification" / "step4_telemetry_poll.py"
SPEC = importlib.util.spec_from_file_location("step4_telemetry_poll", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


def test_parse_number_preserves_unavailable_as_null() -> None:
    assert telemetry.parse_number(" [N/A] ") is None
    assert telemetry.parse_number("2398") == 2398.0


def test_parse_number_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        telemetry.parse_number("nan")


def test_parse_pressure_requires_complete_some_record() -> None:
    assert telemetry.parse_pressure("some avg10=0.01 avg60=0.02 avg300=0.03 total=123") == {
        "avg10": 0.01,
        "avg60": 0.02,
        "avg300": 0.03,
        "total": 123.0,
    }
    with pytest.raises(ValueError, match="unexpected pressure fields"):
        telemetry.parse_pressure("some avg10=0.01 total=123")


def test_sample_records_overall_and_per_source_acquisition_boundaries(monkeypatch) -> None:
    monotonic = iter(float(value) for value in range(10, 18))
    wall = iter(float(value) for value in range(110, 118))
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(telemetry.time, "time", lambda: next(wall))
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=", ".join("1" for _ in range(9))),
    )

    def fake_read(path: Path, **_kwargs) -> str:
        if "/proc/pressure/" in str(path):
            return "some avg10=0 avg60=0 avg300=0 total=0\n"
        return "3900000\n"

    monkeypatch.setattr(telemetry.Path, "read_text", fake_read)
    record = telemetry.sample()
    assert record["schema"] == "nemotron_voicechat.step4_telemetry.v2"
    assert record["acquisition"]["monotonic_start_s"] == 10.0
    assert record["acquisition"]["monotonic_end_s"] == 17.0
    assert record["source_acquisition"]["gpu"]["monotonic_start_s"] == 11.0
    assert record["source_acquisition"]["gpu"]["monotonic_end_s"] == 12.0
    assert record["source_acquisition"]["pressure"]["monotonic_end_s"] == 16.0
