import json

import pytest
from create_quantized_eartts_manifest import build_manifest

IDENTITY = {
    "repository": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
    "revision": "public-revision",
    "sha256": "public-sha",
}


def artifact(tmp_path, *, derived_from_ea=False):
    root = tmp_path / "eartts-w8a32"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "voicechat_checkpoint_repository": IDENTITY["repository"],
                "voicechat_checkpoint_revision": IDENTITY["revision"],
                "voicechat_checkpoint_sha256": IDENTITY["sha256"],
                "voicechat_derived_from_ea": derived_from_ea,
                "quantization_config": {"quant_method": "ea_w8a32"},
            }
        )
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"model.layer.weight": "model-00001-of-00001.safetensors"},
            }
        )
    )
    return root


def base_manifest():
    return {
        "kind": "exact_public_vllm_extraction",
        "derived_from_ea": False,
        "source": {
            "repository": IDENTITY["repository"],
            "revision": IDENTITY["revision"],
            "model": {"sha256": IDENTITY["sha256"]},
        },
        "components": {"nano": {"unchanged": True}, "eartts": {}},
    }


def test_build_manifest_preserves_nano_and_records_reproducible_eartts(tmp_path):
    root = artifact(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")
    result = build_manifest(
        base_manifest(),
        {
            "status": "complete",
            "method": "W8A32",
            "selected_tensor_count": 205,
            "selected_source_bytes": 10,
        },
        root,
        report_path=report_path,
        reproducibility_runs=2,
        byte_identical=True,
    )
    assert result["kind"] == "exact_public_vllm_quantized_eartts"
    assert result["components"]["nano"] == {"unchanged": True}
    component = result["components"]["eartts"]
    assert component["model"]["bytes"] == 7
    assert component["model"]["inventory"]["tensor_count"] == 1
    assert component["quantization"]["selected_tensor_count"] == 205
    assert component["reproducibility"] == {"runs": 2, "byte_identical": True}


def test_build_manifest_refuses_ea_provenance(tmp_path):
    root = artifact(tmp_path, derived_from_ea=True)
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")
    with pytest.raises(ValueError, match="provenance does not match"):
        build_manifest(
            base_manifest(),
            {"status": "complete"},
            root,
            report_path=report_path,
            reproducibility_runs=2,
            byte_identical=True,
        )
