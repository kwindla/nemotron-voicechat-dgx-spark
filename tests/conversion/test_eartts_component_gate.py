from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("tools/conversion/eartts_component_gate.py")
    spec = importlib.util.spec_from_file_location("eartts_component_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asr_evaluator_provenance_records_content_identity(tmp_path: Path) -> None:
    module = load_module()
    model = tmp_path / "evaluator.nemo"
    model.write_bytes(b"independent evaluator")

    assert module.model_file_provenance(model) == {
        "model": str(model.resolve()),
        "bytes": 21,
        "sha256": "5b62c2ce9b516f3fbd116eede69c8f6bdcdb404d85346e38fecb5b1e7213456f",
    }


def test_asr_evaluator_provenance_rejects_missing_file(tmp_path: Path) -> None:
    module = load_module()

    try:
        module.model_file_provenance(tmp_path / "missing.nemo")
    except FileNotFoundError as exc:
        assert "not a regular file" in str(exc)
    else:
        raise AssertionError("missing evaluator was accepted")
