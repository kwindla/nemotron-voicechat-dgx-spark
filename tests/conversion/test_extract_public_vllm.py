from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "conversion"))

from extract_public_vllm import (  # noqa: E402
    MINIMUM_POST_CONVERSION_FREE_GIB,
    MODEL_SHA256,
    NANO_PREFIXES,
    NANO_SKELETON_REVISION,
    RUNTIME_NANO_SKELETON_PATH,
    annotate_eartts_config,
    annotate_nano_config,
    convert_reproducibly,
    tensor_inventory,
    verify_component,
    verify_hashed_files,
)


def test_conversion_floor_is_at_least_one_hundred_gib() -> None:
    assert MINIMUM_POST_CONVERSION_FREE_GIB >= 100


def test_public_source_verification_uses_payload_hashes_without_hf_cache_sidecars(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "config.json"
    payload.write_text('{"model": "public"}\n')
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    result = verify_hashed_files(
        tmp_path,
        {"config.json": digest},
        expected_sizes={"config.json": payload.stat().st_size},
    )
    assert result["config.json"] == {"bytes": payload.stat().st_size, "sha256": digest}
    assert not (tmp_path / ".cache").exists()

    payload.write_text('{"model": "tampered"}\n')
    with pytest.raises(ValueError, match="SHA256 differs"):
        verify_hashed_files(tmp_path, {"config.json": digest})


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload.extend(b"\0" * size)
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_tensor_inventory_filters_exact_public_nano_families(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    write_safetensors(
        model,
        {
            "stt_model.llm.layers.0.weight": ("F32", [2, 2], 16),
            "stt_model.function_head.weight": ("F32", [2, 2], 16),
            "tts_model.tts_model.backbone.weight": ("F32", [2, 2], 16),
        },
    )
    inventory = tensor_inventory(model, lambda name: name.startswith(NANO_PREFIXES))
    assert inventory["tensor_count"] == 2
    assert inventory["payload_bytes"] == 32
    assert inventory["payload_bytes_by_dtype"] == {"F32": 32}


def test_nano_config_declares_bf16_inputs_without_claiming_bf16_weights(
    tmp_path: Path,
) -> None:
    config = {
        "custom_input_specs": [{"name": "combined_embeds", "dtype": "bfloat16", "dim": 4480}],
        "custom_outputs": [
            "text_logits",
            "function_tokens",
            "function_logits",
        ],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    annotate_nano_config(tmp_path)
    result = json.loads((tmp_path / "config.json").read_text())
    assert result["custom_input_specs"][0]["dtype"] == "bfloat16"
    assert result["voicechat_checkpoint_sha256"] == MODEL_SHA256
    assert result["voicechat_has_asr_head"] is False
    assert result["voicechat_has_function_head"] is True


def test_nano_verifier_requires_fp32_public_function_head(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output"
    output.mkdir()
    tensors = {
        "stt_model.llm.layer.weight": ("F32", [2, 2], 16),
        "stt_model.function_head.weight": ("F32", [2, 2], 16),
    }
    write_safetensors(source, tensors)
    write_safetensors(output / "model.safetensors", tensors)
    (output / "config.json").write_text(
        json.dumps(
            {
                "voicechat_checkpoint_sha256": MODEL_SHA256,
                "voicechat_derived_from_ea": False,
            }
        )
    )
    report = verify_component("nano", source, output)
    assert report["model"]["inventory"]["payload_bytes_by_dtype"] == {"F32": 32}

    bf16 = {
        "stt_model.llm.layer.weight": ("BF16", [2, 2], 8),
        "stt_model.function_head.weight": ("BF16", [2, 2], 8),
    }
    write_safetensors(output / "model.safetensors", bf16)
    with pytest.raises(ValueError, match="shape/dtype inventory"):
        verify_component("nano", source, output)


def test_public_vllm_patch_covers_optional_asr_and_function_outputs() -> None:
    source = (
        ROOT
        / "tools"
        / "provenance"
        / "qualified-deltas"
        / "patches"
        / "model_executor"
        / "models"
        / "nemotron_h.py.patch"
    ).read_text()
    assert "stt_model.function_head" in source
    assert "if self.has_asr_head" in source
    assert '"function_tokens": function_tokens' in source
    assert "for name in self.custom_outputs" in source
    assert 'if "text_logits" in self.custom_outputs' in source


def test_eartts_config_uses_identity_locked_local_tokenizer(tmp_path: Path) -> None:
    artifact = tmp_path / "eartts"
    skeleton = tmp_path / "nano-skeleton"
    artifact.mkdir()
    skeleton.mkdir()
    (artifact / "config.json").write_text(
        json.dumps({"pretrained_tokenizer_name": "nvidia/remote"})
    )

    annotate_eartts_config(artifact, skeleton)

    config = json.loads((artifact / "config.json").read_text())
    assert config["pretrained_tokenizer_name"] == RUNTIME_NANO_SKELETON_PATH
    assert config["voicechat_tokenizer_source"] == {
        "repository": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "revision": NANO_SKELETON_REVISION,
        "path": RUNTIME_NANO_SKELETON_PATH,
    }


def test_reproducibility_resume_runs_only_one_verified_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.safetensors"
    output_root = tmp_path / "derived"
    final = output_root / "nano-vllm-fp32"
    final.mkdir(parents=True)
    tensors = {
        "stt_model.llm.layer.weight": ("F32", [2, 2], 16),
        "stt_model.function_head.weight": ("F32", [2, 2], 16),
    }
    write_safetensors(source, tensors)

    def write_output(target: Path) -> None:
        target.mkdir(parents=True)
        write_safetensors(target / "model.safetensors", tensors)
        (target / "config.json").write_text(
            json.dumps(
                {
                    "voicechat_checkpoint_sha256": MODEL_SHA256,
                    "voicechat_derived_from_ea": False,
                }
            )
        )

    write_safetensors(final / "model.safetensors", tensors)
    (final / "config.json").write_text(
        json.dumps(
            {
                "voicechat_checkpoint_sha256": MODEL_SHA256,
                "voicechat_derived_from_ea": False,
            }
        )
    )
    calls = []

    def converter(target: Path) -> None:
        calls.append(target)
        write_output(target)

    report = convert_reproducibly("nano", output_root, converter, source, runs=2)
    assert len(calls) == 1
    assert report["reproducibility"]["runs"] == 2
    assert report["reproducibility"]["byte_identical"] is True
    assert not calls[0].exists()
