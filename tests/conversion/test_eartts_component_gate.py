from __future__ import annotations

from pathlib import Path


def test_component_gate_defers_asr_to_independent_evaluator() -> None:
    source = Path("tools/conversion/eartts_component_gate.py").read_text(encoding="utf-8")

    assert "nemo.collections.asr" not in source
    assert '"pending_external_asr": True' in source
    assert '"asr_inputs"' in source
    assert 'parser.add_argument("--asr-model"' not in source
