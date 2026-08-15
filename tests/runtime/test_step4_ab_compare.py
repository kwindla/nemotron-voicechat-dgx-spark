from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

QUALIFICATION = Path(__file__).parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION))
MODULE_PATH = QUALIFICATION / "step4_ab_compare.py"
SPEC = importlib.util.spec_from_file_location("step4_ab_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)

import stratified_latency_analyzer as analyzer  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _step(timestamp: float, frame: int, value: float, *, bos: bool = False) -> dict:
    turn_state = {
        "model_stage_timings_ms": {
            "wrapper_residual_ms": value / 10,
            "eartts_total_ms": 10.0,
        }
    }
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


def _arm(
    tmp_path: Path, name: str, values: tuple[float | list[float], float | list[float]]
) -> tuple[Path, Path, Path]:
    trace = tmp_path / f"{name}.jsonl"
    events = []
    frame = 0
    for response_index, response_values in enumerate(values, 1):
        frame += 1
        response_start = response_index * 10.0
        events.append(_step(response_start, frame, 150, bos=True))
        emitted_values = (
            response_values if isinstance(response_values, list) else [response_values]
        )
        for value_index, value in enumerate(emitted_values, 1):
            frame += 1
            events.append(_step(response_start + value_index / 10, frame, value))
    _write_jsonl(trace, events)
    analyzer_path = tmp_path / f"{name}-analyzer.json"
    analyzer_path.write_text(json.dumps(analyzer.analyze_paths([trace])), encoding="utf-8")
    report_path = tmp_path / f"{name}-report.json"
    report_path.write_text(
        json.dumps(
            {
                "job_records": [
                    {
                        "prompt_ordinal": ordinal,
                        "prompt_id": f"prompt-{ordinal:02d}",
                        "prompt_text": f"prompt {ordinal}",
                        "job_id": f"{name}-job-{ordinal}",
                        "response_id": f"{name}-response-{ordinal}",
                        "terminal_status": "completed",
                    }
                    for ordinal in (1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    return analyzer_path, trace, report_path


def test_paired_cluster_difference_preserves_arm_sign() -> None:
    report = comparison._paired_cluster_difference(
        {"a1": [5.0, 7.0], "a2": [9.0]},
        {"b1": [3.0, 5.0], "b2": [7.0]},
    )
    assert report["effect_ms"] == pytest.approx(2.0)
    assert report["n_paired_response_clusters"] == 2
    assert report["bootstrap_95_ci_ms"] == pytest.approx([2.0, 2.0])


def test_paired_cluster_difference_rejects_cardinality_mismatch() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        comparison._paired_cluster_difference({"a1": [1.0]}, {"b1": [1.0], "b2": [2.0]})


def test_complete_analysis_distinguishes_unequal_weight_estimands(
    tmp_path: Path, monkeypatch
) -> None:
    analyzer_a, trace_a, report_a = _arm(tmp_path, "a", ([10.0, 10.0, 10.0], [0.0]))
    analyzer_b, trace_b, report_b = _arm(tmp_path, "b", ([0.0], [0.0]))
    monkeypatch.setattr(comparison, "BOOTSTRAP_ITERATIONS", 50)

    first = comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)
    second = comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)

    assert first == second
    effect = first["delivered_generation_server_step_ms"][
        "paired_response_clustered_difference"
    ]
    estimands = effect["declared_estimands"]
    assert effect["effect_ms"] == pytest.approx(7.5)
    assert estimands["frame_weighted"]["effect_ms"] == pytest.approx(7.5)
    assert estimands["equal_response"]["effect_ms"] == pytest.approx(5.0)
    assert estimands["equal_run"]["effect_ms"] == pytest.approx(7.5)
    assert [(row["arm_a_n"], row["arm_b_n"]) for row in effect["per_response"]] == [
        (3, 1),
        (1, 1),
    ]
    residual = first["delivered_generation_wrapper_residual_ms"][
        "paired_response_clustered_difference"
    ]["declared_estimands"]
    assert residual["frame_weighted"]["effect_ms"] == pytest.approx(0.75)
    assert residual["equal_response"]["effect_ms"] == pytest.approx(0.5)
    assert residual["equal_run"]["effect_ms"] == pytest.approx(0.75)


def test_complete_analysis_rejects_same_cardinality_wrong_membership(tmp_path: Path) -> None:
    analyzer_a, trace_a, report_a = _arm(tmp_path, "a", (80, 82))
    analyzer_b, trace_b, report_b = _arm(tmp_path, "b", (79, 81))
    tampered = json.loads(analyzer_a.read_text(encoding="utf-8"))
    membership = tampered["sources"][0]["populations"]["delivered-nonBOS"]["membership"]
    membership["members"][0]["frame"] = 999
    analyzer_a.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="exact membership mismatch"):
        comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)


def test_complete_analysis_rejects_missing_middle_prompt_ordinal(tmp_path: Path) -> None:
    analyzer_a, trace_a, report_a = _arm(tmp_path, "a", (80, 82))
    analyzer_b, trace_b, report_b = _arm(tmp_path, "b", (79, 81))
    report = json.loads(report_a.read_text(encoding="utf-8"))
    report["job_records"][1]["prompt_ordinal"] = 3
    report_a.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or duplicated prompt ordinal"):
        comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)


def test_complete_analysis_rejects_unequal_prompt_identity(tmp_path: Path) -> None:
    analyzer_a, trace_a, report_a = _arm(tmp_path, "a", (80, 82))
    analyzer_b, trace_b, report_b = _arm(tmp_path, "b", (79, 81))
    report = json.loads(report_b.read_text(encoding="utf-8"))
    report["job_records"][1]["prompt_text"] = "different prompt"
    report_b.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt identity differs"):
        comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)


def test_complete_analysis_pairs_by_prompt_records(tmp_path: Path) -> None:
    analyzer_a, trace_a, report_a = _arm(tmp_path, "a", (80, 84))
    analyzer_b, trace_b, report_b = _arm(tmp_path, "b", (78, 82))
    result = comparison.analyze(analyzer_a, trace_a, report_a, analyzer_b, trace_b, report_b)
    assert [item["prompt_ordinal"] for item in result["pairing"]] == [1, 2]
    effect = result["delivered_generation_server_step_ms"]["paired_response_clustered_difference"]
    assert effect["effect_ms"] == pytest.approx(2.0)
    assert effect["declared_estimands"]["equal_response"]["effect_ms"] == pytest.approx(2.0)
