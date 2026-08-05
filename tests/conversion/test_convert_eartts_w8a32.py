from pathlib import Path

import pytest
from convert_eartts_w8a32 import (
    GIB,
    OUTPUT_OVERHEAD_BYTES,
    conversion_disk_preflight,
    projected_output_upper_bound,
    selected,
)


def test_selects_backbone_and_sampler_mlp_projections_only():
    assert selected("model.backbone.layers.0.self_attn.q_proj.weight")
    assert selected("model.backbone.layers.27.mlp.down_proj.weight")
    assert selected("model.sampler.mog_head.mlp_stack.0.mlp.gate_proj.weight")
    assert selected("model.sampler.mog_head.mlp_stack.2.mlp.down_proj.weight")
    assert not selected("model.sampler.mog_head.proj_mus.weight")
    assert not selected("model.total_emb.embed_subword.proj_embedding.weight")


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "config.json").write_text("{}")
    return source


def test_projected_output_is_conservative_source_copy_plus_overhead(tmp_path):
    source = _source(tmp_path)
    source_bytes = sum(path.stat().st_size for path in source.iterdir())
    assert projected_output_upper_bound(source) == source_bytes + OUTPUT_OVERHEAD_BYTES


def test_disk_preflight_accepts_exact_floor_after_projected_peak(tmp_path):
    source = _source(tmp_path)
    projected = projected_output_upper_bound(source)
    result = conversion_disk_preflight(
        source,
        tmp_path / "output",
        min_remaining_gib=100,
        free_bytes=projected + 100 * GIB,
    )
    assert result["projected_remaining_bytes"] == 100 * GIB


def test_disk_preflight_refuses_below_floor_before_output_creation(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "output"
    projected = projected_output_upper_bound(source)
    with pytest.raises(RuntimeError, match="below the 100.00 GiB floor"):
        conversion_disk_preflight(
            source,
            output,
            min_remaining_gib=100,
            free_bytes=projected + 100 * GIB - 1,
        )
    assert not output.exists()


def test_disk_preflight_refuses_existing_output(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Output already exists"):
        conversion_disk_preflight(source, output, free_bytes=1000 * GIB)


def test_disk_preflight_floor_cannot_be_overridden_below_100_gib(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="hard 100 GiB conversion floor"):
        conversion_disk_preflight(
            source,
            tmp_path / "output",
            min_remaining_gib=99.999,
            free_bytes=1000 * GIB,
        )
