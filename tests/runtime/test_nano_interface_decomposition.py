from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "tools" / "benchmark" / "nano_interface_decomposition.py"
SPEC = importlib.util.spec_from_file_location("nano_interface_decomposition", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
decomposition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decomposition)


def _records() -> list[dict]:
    tokens = [12] * 687
    for content, pad in decomposition.EXPECTED_PAIRS:
        tokens[content] = 1046
        tokens[pad] = 12
    return [
        {
            "call_index": index,
            "current_step": index,
            "decode_steps": 1,
            "predicted_token": token,
            "request_id": "fixture",
            "sequence_epoch": 0,
            "file": f"nano-call-{index:06d}.pt",
            "sha256": f"{index:064x}",
        }
        for index, token in enumerate(tokens)
    ]


def test_selection_fixes_three_consecutive_lexical_to_pad_boundaries() -> None:
    pairs = decomposition.select_matched_states(_records())
    assert [(pair["content"]["call_index"], pair["pad"]["call_index"]) for pair in pairs] == list(
        decomposition.EXPECTED_PAIRS
    )
    mutated = _records()
    mutated[326]["predicted_token"] = 12
    with pytest.raises(ValueError, match="selection changed"):
        decomposition.select_matched_states(mutated)


def test_exact_digest_is_dtype_shape_and_value_sensitive_and_ignores_request_identity() -> None:
    torch = pytest.importorskip("torch")
    first = decomposition.exact_output_digest(
        {
            "predicted_token": 1046,
            "function_predicted_token": torch.tensor([12]),
            "request_id": "a",
            "nano_engine_benchmark": {"ms": 1},
        },
    )
    second = decomposition.exact_output_digest(
        {
            "predicted_token": 1046,
            "function_predicted_token": torch.tensor([12]),
            "request_id": "b",
            "nano_engine_benchmark": {"ms": 9},
        },
    )
    assert first == second
    assert (
        decomposition.exact_output_digest(
            {
                "predicted_token": 1046,
                "function_predicted_token": torch.tensor([1]),
            }
        )
        != first
    )


def _nsys_fixture(path: Path, blocks: int) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel');
        """
    )
    clock = 0
    for block in range(blocks):
        for pair, (content_call, pad_call) in enumerate(decomposition.EXPECTED_PAIRS, 1):
            for state_class, call in (("content", content_call), ("pad", pad_call)):
                start = clock
                end = start + 10_000_000
                message = (
                    f"step4d:block={block:03d};pair={pair};class={state_class};"
                    f"call={call};step={call}"
                )
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (start, end, message)
                )
                # Two overlapping kernels: summed 7 ms, union 6 ms, residual 4 ms.
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)",
                    (start + 1_000_000, start + 5_000_000),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)",
                    (start + 4_000_000, start + 7_000_000),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 1)",
                    (start + 500_000, start + 600_000),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?)",
                    (start + 8_000_000, start + 9_000_000),
                )
                clock = end + 1_000_000
    connection.commit()
    connection.close()


def test_nsys_sqlite_analysis_uses_kernel_union_and_block_range_identity(tmp_path: Path) -> None:
    source = tmp_path / "fixture.sqlite"
    _nsys_fixture(source, blocks=2)
    report = decomposition.analyze_nsys_sqlite(source, expected_blocks=2)
    assert report["passed"]
    assert report["observed_ranges"] == 12
    row = report["rows"][0]
    assert row["interface_wall_ms"] == pytest.approx(10.0)
    assert row["kernel_sum_ms"] == pytest.approx(7.0)
    assert row["kernel_union_ms"] == pytest.approx(6.0)
    assert row["non_kernel_residual_ms"] == pytest.approx(4.0)
    assert row["gpu_memory_union_ms"] == pytest.approx(1.0)
    assert row["launch_api_count"] == 1


def test_nsys_sqlite_accepts_cuda_graph_execution_as_gpu_busy_envelope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "graph.sqlite"
    _nsys_fixture(source, blocks=1)
    connection = sqlite3.connect(source)
    connection.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    )
    connection.commit()
    connection.close()
    report = decomposition.analyze_nsys_sqlite(source, expected_blocks=1)
    assert report["passed"]
    assert report["gpu_activity_tables"] == ["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"]
    assert {row["gpu_activity_kind"] for row in report["rows"]} == {"cuda_graph_execution_envelope"}


def test_nsys_sqlite_graph_envelope_is_authoritative_in_dual_table_range(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dual-table.sqlite"
    _nsys_fixture(source, blocks=1)
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)")
    ranges = connection.execute("SELECT start, end FROM NVTX_EVENTS").fetchall()
    for start, _ in ranges:
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?)",
            (start + 2_000_000, start + 8_000_000),
        )
    connection.commit()
    connection.close()

    report = decomposition.analyze_nsys_sqlites(
        [source], expected_blocks=1, require_cuda_graph=True
    )
    assert report["passed"]
    assert report["gpu_activity_tables"] == [
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        "CUPTI_ACTIVITY_KIND_KERNEL",
    ]
    row = report["rows"][0]
    assert row["gpu_activity_kind"] == "cuda_graph_execution_envelope"
    assert row["graph_envelope_count"] == 1
    assert row["graph_envelope_union_ms"] == pytest.approx(6.0)
    assert row["kernel_union_ms"] == pytest.approx(6.0)
    assert row["explicit_kernel_count"] == 2
    assert row["explicit_kernel_union_ms"] == pytest.approx(6.0)
    assert row["explicit_kernel_outside_graph_union_ms"] == pytest.approx(1.0)
    assert row["gpu_busy_union_ms"] == pytest.approx(7.0)
    assert row["gpu_busy_residual_ms"] == pytest.approx(3.0)


def test_summary_treats_blocks_as_units_and_requires_preregistered_counts(monkeypatch) -> None:
    monkeypatch.setattr(decomposition, "BOOTSTRAP_RESAMPLES", 20)
    exact = {
        "passed": True,
        "reference_output_digests": {"pair-1-content": "a"},
        "blocks": [
            {
                "phase": "measured",
                "valid": True,
                "selected": [{"exact_output_match": True}] * 6,
            }
            for _ in range(30)
        ],
    }
    rows = []
    for block in range(10):
        for pair in range(1, 4):
            for state_class, kernel in (("content", 6.0), ("pad", 7.0)):
                rows.append(
                    {
                        "block": block,
                        "pair": pair,
                        "class": state_class,
                        "interface_wall_ms": 10.0,
                        "kernel_union_ms": kernel,
                        "non_kernel_residual_ms": 10.0 - kernel,
                        "kernel_fraction": kernel / 10.0,
                        "non_kernel_fraction": 1.0 - kernel / 10.0,
                        "graph_envelope_count": 1,
                        "graph_envelope_union_ms": kernel,
                        "graph_envelope_residual_ms": 10.0 - kernel,
                        "graph_envelope_fraction": kernel / 10.0,
                        "graph_envelope_residual_fraction": 1.0 - kernel / 10.0,
                        "explicit_kernel_outside_graph_union_ms": 0.0,
                        "gpu_busy_union_ms": kernel,
                        "gpu_busy_residual_ms": 10.0 - kernel,
                        "gpu_busy_fraction": kernel / 10.0,
                        "gpu_busy_residual_fraction": 1.0 - kernel / 10.0,
                    }
                )
    result = decomposition.summarize(exact, {"passed": True, "rows": rows})
    assert result["decision"] == "COMPLETE"
    assert result["decomposition_blocks"]["valid"] == 10
    assert result["decomposition"]["kernel_union_ms"]["pad_minus_content"]["mean"] == pytest.approx(
        1.0
    )
    assert result["decomposition"]["kernel_union_ms"]["content"]["blocks"] == 10


def test_summary_stays_incomplete_when_prefix_diverges_but_selected_digests_match(
    monkeypatch,
) -> None:
    monkeypatch.setattr(decomposition, "BOOTSTRAP_RESAMPLES", 20)
    exact = {
        # Exercise the summary's own block-validity check even if an upstream
        # producer incorrectly claims the aggregate exact gate passed.
        "passed": True,
        "reference_output_digests": {"pair-1-content": "a"},
        "blocks": [
            {
                "phase": "measured",
                "valid": False,
                "token_mismatches": [
                    {"call_index": 39, "expected_token": 10592, "actual_token": 1044}
                ],
                "selected": [{"exact_output_match": True}] * 6,
            }
            for _ in range(30)
        ],
    }
    rows = []
    for block in range(10):
        for pair in range(1, 4):
            for state_class in ("content", "pad"):
                rows.append(
                    {
                        "block": block,
                        "pair": pair,
                        "class": state_class,
                        "interface_wall_ms": 10.0,
                        "kernel_union_ms": 6.0,
                        "non_kernel_residual_ms": 4.0,
                        "kernel_fraction": 0.6,
                        "non_kernel_fraction": 0.4,
                        "graph_envelope_count": 1,
                        "graph_envelope_union_ms": 6.0,
                        "graph_envelope_residual_ms": 4.0,
                        "graph_envelope_fraction": 0.6,
                        "graph_envelope_residual_fraction": 0.4,
                        "explicit_kernel_outside_graph_union_ms": 0.0,
                        "gpu_busy_union_ms": 6.0,
                        "gpu_busy_residual_ms": 4.0,
                        "gpu_busy_fraction": 0.6,
                        "gpu_busy_residual_fraction": 0.4,
                    }
                )

    result = decomposition.summarize(
        exact, {"passed": True, "require_cuda_graph": True, "rows": rows}
    )
    assert result["exact_output"]["matching_states"] == 180
    assert result["exact_output"]["passed"] is False
    assert result["passed"] is False
    assert result["decision"] == "INCOMPLETE"


def test_selection_command_writes_provenance(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "capture"
    root.mkdir()
    (root / "metadata.json").write_text('{"schema": 1}\n', encoding="utf-8")
    records = _records()
    selected_calls = {call for pair in decomposition.EXPECTED_PAIRS for call in pair}
    for call in selected_calls:
        path = root / records[call]["file"]
        torch.save(
            {
                "inputs": {
                    "input_embeds": torch.ones(1, 1, 4, dtype=torch.bfloat16),
                    "current_step": call,
                    "decode_steps": 1,
                    "prompt_token_ids": None,
                }
            },
            path,
        )
        records[call]["sha256"] = decomposition.sha256_file(path)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    payload = decomposition.selection_payload(root)
    assert payload["capture_calls"] == 687
    assert payload["max_replayed_call"] == 686
    assert len(payload["capture_manifest_sha256"]) == 64
    assert payload["pairs"][0]["input_contract"] == {
        "input_embed_shape": [1, 1, 4],
        "input_embed_dtype": "torch.bfloat16",
        "decode_steps": 1,
        "prompt_token_ids": None,
    }
