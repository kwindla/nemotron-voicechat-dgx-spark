import json
from pathlib import Path

import create_eartts_window_variant as module

ROOT = Path(__file__).resolve().parents[2]


def test_window_variant_preserves_weights_and_removes_production_claim(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "MINIMUM_FREE_BYTES", 0)
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "sliding_window": 7500,
        "layer_types": ["sliding_attention"] * 24 + ["full_attention"] * 4,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"exact-weight-bytes")
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "derived_from_ea": False,
                "components": {"eartts": {"path": str(source)}},
                "runtime_optimization": {"production_candidate": 1},
                "production_candidate": {"number": 1, "status": "frozen"},
            }
        )
    )
    output = tmp_path / "window1500"
    output_manifest = tmp_path / "window1500-manifest.json"

    report = module.create_variant(
        source, manifest_path, output, output_manifest, sliding_window=1500
    )

    result_config = json.loads((output / "config.json").read_text())
    result_manifest = json.loads(output_manifest.read_text())
    assert result_config["sliding_window"] == 1500
    assert result_config["voicechat_eartts_window_experiment"]["weights_changed"] is False
    assert (source / "model.safetensors").stat().st_ino == (
        output / "model.safetensors"
    ).stat().st_ino
    assert "production_candidate" not in result_manifest
    assert result_manifest["components"]["eartts"]["files"]["config.json"]["sha256"]
    assert report["all_non_config_files_hardlinked"]
    assert report["all_non_config_files_byte_identical"]
    assert report["copy_modes"] == {"model.safetensors": "hardlink"}
    assert report["production_candidate_removed"]


def test_window_recipe_matches_qualified_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "MINIMUM_FREE_BYTES", 0)
    source = tmp_path / "source"
    source.mkdir()
    fixture = ROOT / "tests/conversion/fixtures/eartts-converter-base-config.json"
    (source / "config.json").write_bytes(fixture.read_bytes())
    (source / "model.safetensors").write_bytes(b"fixture")
    assert module.sha256_file(source / "config.json") == (
        "ca185841e958b92e2f25492a0722265a0acf247ef22690e9e3fff4dffb35ab17"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "derived_from_ea": False,
                "components": {"eartts": {"path": "/derived/eartts"}},
            }
        )
    )

    output = tmp_path / "window1500"
    module.create_variant(
        source,
        manifest,
        output,
        tmp_path / "output-manifest.json",
        sliding_window=1500,
    )

    assert module.sha256_file(output / "config.json") == (
        "f1b445f141fb7db1acdf2b76d7e9da0939831bac5c2363f40af99968cce385e8"
    )
