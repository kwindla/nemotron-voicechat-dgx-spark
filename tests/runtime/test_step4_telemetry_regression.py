from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

QUALIFICATION = Path(__file__).parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION))
MODULE_PATH = QUALIFICATION / "step4_telemetry_regression.py"
SPEC = importlib.util.spec_from_file_location("step4_telemetry_regression", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
regression = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regression)

import stratified_latency_analyzer as analyzer  # noqa: E402


def _telemetry_sample(
    timestamp: float,
    *,
    total: float,
    varying: float | None,
    source_end_offsets: dict[str, float] | None = None,
) -> dict:
    sample = {
        "schema": "nemotron_voicechat.step4_telemetry.v1",
        "monotonic_s": timestamp,
        "wall_time_s": 1_700_000_000 + timestamp,
        "gpu": {"clock": varying, "constant": 3.0},
        "cpu_scaling_cur_freq_khz": {"5": 3_900_000},
        "pressure": {
            kind: {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": total}
            for kind in ("cpu", "memory", "io")
        },
    }
    if source_end_offsets is None:
        return sample
    sample["schema"] = "nemotron_voicechat.step4_telemetry.v2"
    source_acquisition = {}
    for source, offset in source_end_offsets.items():
        source_start = timestamp + offset - 0.05
        source_end = timestamp + offset
        source_acquisition[source] = {
            "monotonic_start_s": source_start,
            "monotonic_end_s": source_end,
            "wall_start_s": 1_700_000_000 + source_start,
            "wall_end_s": 1_700_000_000 + source_end,
            "duration_ms": 50.0,
        }
    acquisition_end = timestamp + max(source_end_offsets.values())
    sample["acquisition"] = {
        "monotonic_start_s": timestamp,
        "monotonic_end_s": acquisition_end,
        "wall_start_s": 1_700_000_000 + timestamp,
        "wall_end_s": 1_700_000_000 + acquisition_end,
        "duration_ms": (acquisition_end - timestamp) * 1000.0,
    }
    sample["source_acquisition"] = source_acquisition
    return sample


def _model_step(timestamp: float, frame: int, value: float, *, bos: bool = False) -> dict:
    timings = {"wrapper_residual_ms": 2.0, "eartts_total_ms": 10.0}
    turn_state = {"model_stage_timings_ms": timings}
    if bos:
        turn_state["eartts_bos_epoch_step"] = 0
    return {
        "event": "model_step",
        "frame": frame,
        "monotonic_s": timestamp,
        "server_step_ms": value,
        "audio_delivered": True,
        "output_audio": {"bytes": 3528, "samples": 1764},
        "turn_state": turn_state,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pearson_and_slope_handle_signal_and_constant_input() -> None:
    assert regression.pearson([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == pytest.approx(1.0)
    assert regression.slope([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == pytest.approx(2.0)
    assert regression.pearson([1.0, 1.0], [2.0, 3.0]) is None
    assert regression.slope([1.0, 1.0], [2.0, 3.0]) is None
    assert regression.linear_residuals([0.0, 1.0, 2.0], [3.0, 5.0, 7.0]) == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_join_population_selects_causally_prior_sample() -> None:
    members = [
        {
            "event_ordinal": 1,
            "frame": 7,
            "monotonic_timestamp_s": 10.49,
            "timestamp_field": "monotonic_s",
            "response_id": "r1",
        }
    ]
    events = [{"event": "model_step", "frame": 7, "monotonic_s": 10.49, "server_step_ms": 82.0}]
    telemetry, _ = regression.flatten_telemetry(
        [
            _telemetry_sample(10.0, total=0, varying=1.0),
            _telemetry_sample(11.0, total=1, varying=2.0),
        ]
    )
    joined = regression.join_population(members, events, telemetry)
    assert joined[0]["telemetry_sample_indices"]["gpu.clock"] == 0
    assert joined[0]["telemetry"]["gpu.clock"] == 1.0
    assert joined[0]["join_age_ms"] == pytest.approx(490.0)


def test_join_population_fails_closed_on_old_causal_sample() -> None:
    members = [
        {
            "event_ordinal": 1,
            "frame": 7,
            "timestamp_field": "monotonic_s",
            "monotonic_timestamp_s": 11.2,
        }
    ]
    events = [{"event": "model_step", "frame": 7, "monotonic_s": 11.2, "server_step_ms": 82.0}]
    telemetry, _ = regression.flatten_telemetry(
        [
            _telemetry_sample(10.0, total=0, varying=1.0),
            _telemetry_sample(12.0, total=1, varying=2.0),
        ]
    )
    with pytest.raises(ValueError, match="causally prior telemetry"):
        regression.join_population(members, events, telemetry)


def test_member_timestamp_must_match_raw_event() -> None:
    members = [
        {
            "event_ordinal": 1,
            "frame": 7,
            "timestamp_field": "monotonic_s",
            "monotonic_timestamp_s": 10.5,
        }
    ]
    events = [{"event": "model_step", "frame": 7, "monotonic_s": 10.4, "server_step_ms": 82.0}]
    telemetry, _ = regression.flatten_telemetry(
        [
            _telemetry_sample(9.0, total=0, varying=1.0),
            _telemetry_sample(10.0, total=1, varying=2.0),
        ]
    )
    with pytest.raises(ValueError, match="member timestamp"):
        regression.join_population(members, events, telemetry)


def test_complete_analysis_rejects_nonmonotonic_v2_acquisitions(tmp_path: Path) -> None:
    trace = tmp_path / "events.jsonl"
    events = [_model_step(10.1, 1, 150.0, bos=True), _model_step(10.4, 2, 80.0)]
    _write_jsonl(trace, events)
    analyzer_path = tmp_path / "analyzer.json"
    analyzer_path.write_text(json.dumps(analyzer.analyze_paths([trace])), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.jsonl"
    source_offsets = {"gpu": 0.1, "cpu_scaling_cur_freq": 0.2, "pressure": 0.3}
    _write_jsonl(
        telemetry_path,
        [
            _telemetry_sample(
                9.0, total=0, varying=1.0, source_end_offsets=source_offsets
            ),
            _telemetry_sample(
                8.9, total=1, varying=2.0, source_end_offsets=source_offsets
            ),
        ],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        regression.analyze(analyzer_path, trace, telemetry_path)


def test_complete_analysis_uses_v2_source_boundaries_and_causal_psi_intervals(
    tmp_path: Path, monkeypatch
) -> None:
    trace = tmp_path / "events.jsonl"
    events = [
        _model_step(10.15, 1, 150.0, bos=True),
        _model_step(10.25, 2, 80.0),
        _model_step(10.45, 3, 81.0),
        _model_step(11.25, 4, 84.0),
    ]
    _write_jsonl(trace, events)
    analyzer_path = tmp_path / "analyzer.json"
    analyzer_path.write_text(json.dumps(analyzer.analyze_paths([trace])), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            _telemetry_sample(
                9.0,
                total=0,
                varying=1.0,
                source_end_offsets={
                    "gpu": 0.1,
                    "cpu_scaling_cur_freq": 0.2,
                    "pressure": 0.3,
                },
            ),
            _telemetry_sample(
                10.0,
                total=110_000,
                varying=2.0,
                source_end_offsets={
                    "gpu": 0.1,
                    "cpu_scaling_cur_freq": 0.2,
                    "pressure": 0.4,
                },
            ),
            _telemetry_sample(
                11.0,
                total=430_000,
                varying=3.0,
                source_end_offsets={
                    "gpu": 0.1,
                    "cpu_scaling_cur_freq": 0.2,
                    "pressure": 0.2,
                },
            ),
        ],
    )
    monkeypatch.setattr(regression, "BOOTSTRAP_ITERATIONS", 50)

    result = regression.analyze(analyzer_path, trace, telemetry_path)

    delivered = result["populations"]["delivered"]["regressions"]
    assert delivered["gpu.clock"]["n_telemetry_samples"] == 2
    assert delivered["cpu5.scaling_cur_freq_khz"]["n_telemetry_samples"] == 3
    psi = result["populations"]["delivered-nonBOS"]["regressions"][
        "pressure.cpu.interval_stall_fraction"
    ]
    assert psi["status"] == "estimated"
    assert psi["n_rows"] == 2
    assert psi["n_telemetry_samples"] == 2
    assert psi["n_unique_values"] == 2
    assert psi["pearson_r_frame"] == pytest.approx(1.0)
    assert psi["ols_slope_ms_per_unit_frame"] == pytest.approx(10.0)
    assert result["clock_mapping"]["method"].startswith("direct shared-host monotonic clock")


@pytest.mark.parametrize("spike_value,expected_spikes", [(87.0, 0), (89.0, 2)])
def test_complete_analysis_handles_null_constant_and_spike_strata(
    tmp_path: Path, monkeypatch, spike_value: float, expected_spikes: int
) -> None:
    trace = tmp_path / "events.jsonl"
    events = [
        _model_step(10.1, 1, 150.0, bos=True),
        _model_step(10.4, 2, 80.0),
        _model_step(11.4, 3, spike_value),
        _model_step(12.1, 4, 150.0, bos=True),
        _model_step(12.4, 5, 81.0),
        _model_step(13.4, 6, spike_value),
    ]
    _write_jsonl(trace, events)
    analyzer_path = tmp_path / "analyzer.json"
    analyzer_path.write_text(json.dumps(analyzer.analyze_paths([trace])), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            _telemetry_sample(9.5 + index, total=index * 1000, varying=value)
            for index, value in enumerate((None, 1.0, 2.0, 3.0, 4.0))
        ],
    )
    monkeypatch.setattr(regression, "BOOTSTRAP_ITERATIONS", 50)
    result = regression.analyze(analyzer_path, trace, telemetry_path)
    population = result["populations"]["delivered-nonBOS"]
    assert population["effective_support"] == {
        "n_rows": 4,
        "n_responses": 2,
        "n_telemetry_samples": 4,
    }
    assert population["run_level_sensitivity"]["status"] == "unavailable-one-run"
    assert population["regressions"]["gpu.constant"]["status"] == "undefined-no-variation"
    assert population["regressions"]["gpu.clock"]["n_unique_values"] == 3
    assert len(population["response_level_stage_means"]) == 2
    assert (
        result["populations"]["delivered-generation->=88ms"]["membership"]["n"] == expected_spikes
    )
    first = population["regressions"]["gpu.clock"]["response_clustered_bootstrap"]
    second = regression.analyze(analyzer_path, trace, telemetry_path)["populations"][
        "delivered-nonBOS"
    ]["regressions"]["gpu.clock"]["response_clustered_bootstrap"]
    assert first == second


def test_complete_analysis_rejects_same_cardinality_wrong_membership(tmp_path: Path) -> None:
    trace = tmp_path / "events.jsonl"
    events = [_model_step(10.1, 1, 150.0, bos=True), _model_step(10.4, 2, 80.0)]
    _write_jsonl(trace, events)
    report = analyzer.analyze_paths([trace])
    membership = report["sources"][0]["populations"]["delivered-nonBOS"]["membership"]
    membership["members"][0]["frame"] = 999
    analyzer_path = tmp_path / "analyzer.json"
    analyzer_path.write_text(json.dumps(report), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            _telemetry_sample(9.0, total=0, varying=1.0),
            _telemetry_sample(10.0, total=1, varying=2.0),
        ],
    )
    with pytest.raises(ValueError, match="exact membership mismatch"):
        regression.analyze(analyzer_path, trace, telemetry_path)


def test_complete_analysis_rejects_corrupted_member_timestamp(tmp_path: Path) -> None:
    trace = tmp_path / "events.jsonl"
    events = [_model_step(10.1, 1, 150.0, bos=True), _model_step(10.4, 2, 80.0)]
    _write_jsonl(trace, events)
    report = analyzer.analyze_paths([trace])
    membership = report["sources"][0]["populations"]["delivered-nonBOS"]["membership"]
    membership["members"][0]["monotonic_timestamp_s"] = 10.5
    analyzer_path = tmp_path / "analyzer.json"
    analyzer_path.write_text(json.dumps(report), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        telemetry_path,
        [
            _telemetry_sample(9.0, total=0, varying=1.0),
            _telemetry_sample(10.0, total=1, varying=2.0),
        ],
    )
    with pytest.raises(ValueError, match="exact membership mismatch"):
        regression.analyze(analyzer_path, trace, telemetry_path)
