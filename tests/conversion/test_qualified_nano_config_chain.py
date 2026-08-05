from __future__ import annotations

from pathlib import Path

import finalize_nano_release as finalizer
from nano_gptq_calibration import sha256_file

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_SHA256 = "9179d7190c3708000cb9fdfa9495b7fded63259384b35185547d1fa42e6100bd"
QUALIFIED_CONFIG_SHA256 = "2a0576a8b71aa09a5a6683d5a0d2e3fc099839ea9fe559d69a3458e828530391"


def test_release_config_matches_qualified_candidate(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "nano-base"
    source.mkdir()
    source_config = ROOT / "tests/conversion/fixtures/nano-converter-base-config.json"
    (source / "config.json").write_bytes(source_config.read_bytes())
    (source / "model.safetensors").write_bytes(b"fixture")
    assert sha256_file(source / "config.json") == BASE_CONFIG_SHA256
    monkeypatch.setattr(finalizer, "MINIMUM_FREE_BYTES", 0)

    final = tmp_path / "nano-final"
    report = finalizer.finalize(source, final)

    assert sha256_file(final / "config.json") == QUALIFIED_CONFIG_SHA256
    assert (source / "model.safetensors").read_bytes() == (final / "model.safetensors").read_bytes()
    assert report["weights_changed"] is False


def test_release_config_rejects_nonqualified_converter_input(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "nano-base"
    source.mkdir()
    (source / "config.json").write_text('{"custom_outputs": []}\n')
    monkeypatch.setattr(finalizer, "MINIMUM_FREE_BYTES", 0)

    import pytest

    with pytest.raises(ValueError, match="qualified recipe input"):
        finalizer.finalize(source, tmp_path / "output")
