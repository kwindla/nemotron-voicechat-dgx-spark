from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip(
    "torch",
    reason="conversion tensor tests run in the public runtime container",
)
from nano_attribution import (
    _WorkerSession,
    compare_snapshot_directories,
    parse_int_set,
    target_window_summary,
    tensor_comparison,
)


def test_integer_selection_parses_ranges_and_rejects_descending():
    assert parse_int_set("20-22,30, 22") == {20, 21, 22, 30}
    with pytest.raises(ValueError, match="descending"):
        parse_int_set("4-2")


def test_target_gate_requires_every_call_order_and_margin():
    comparisons = [
        {
            "call_index": index,
            "token_agrees": True,
            "bos_pad_order_agrees": True,
            "bos_pad_margin_delta": delta,
        }
        for index, delta in ((20, 0.5), (21, -1.0))
    ]
    assert target_window_summary(comparisons, {20, 21})["passed"]
    comparisons[1]["bos_pad_order_agrees"] = False
    assert not target_window_summary(comparisons, {20, 21})["passed"]
    assert target_window_summary(comparisons[:1], {20, 21})["missing_call_indices"] == [21]


def test_tensor_comparison_reports_identity_and_relative_error():
    left = torch.tensor([1.0, 2.0])
    identity = tensor_comparison(left, left.clone())
    assert identity["cosine"] == pytest.approx(1.0)
    assert identity["relative_rmse"] == 0.0
    shifted = tensor_comparison(left, left + 1.0)
    assert shifted["relative_rmse"] > 0


class _Down(torch.nn.Module):
    def forward(self, value):
        return value + 1, None


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = SimpleNamespace(down_proj=_Down())

    def forward(self, value, residual=None):
        hidden, _ = self.mixer.down_proj(value)
        return hidden, value if residual is None else residual


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def forward(self, *, combined_embeds=None):
        hidden = combined_embeds
        residual = None
        for layer in self.model.layers:
            hidden, residual = layer(hidden, residual)
        return hidden


def test_capture_session_writes_no_model_artifact_and_is_comparable(tmp_path: Path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for output in (reference, candidate):
        model = _Model()
        session = _WorkerSession(
            model,
            {
                "mode": "capture",
                "layers": [0],
                "windows": [0],
                "output_dir": str(output),
            },
        )
        model(combined_embeds=torch.ones(1, 4))
        summary = session.finalize()
        assert not summary["missing"]
        assert not summary["writes_model_artifact"]
        assert (output / "call-000000-layer-00.pt").is_file()

    comparison = compare_snapshot_directories(reference, candidate)
    assert comparison["complete"]
    assert comparison["comparisons"][0]["layer_hidden"]["relative_rmse"] == 0.0
    manifest = json.loads((candidate / "manifest.json").read_text())
    assert manifest["mode"] == "capture"
