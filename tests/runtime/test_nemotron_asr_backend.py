from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nemotron_voicechat_asr_evaluator.backend as asr_backend


def load_module():
    return asr_backend


def write_snapshot(root: Path, module) -> tuple[Path, dict]:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    files = {
        "config.json": {
            "bytes": (root / "config.json").stat().st_size,
            "sha256": module.sha256_file(root / "config.json"),
        }
    }
    manifest = {
        "schema": 1,
        "model_id": "fixture/asr",
        "revision": "a" * 40,
        "snapshot_sha256": module.snapshot_sha256(files),
        "files": files,
        "inference": {},
    }
    path = root.parent / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_snapshot_verification_is_exact_and_detects_mutation(tmp_path: Path) -> None:
    module = load_module()
    root = tmp_path / "model"
    _path, manifest = write_snapshot(root, module)

    assert module.verify_snapshot(root, manifest) == manifest["files"]
    (root / "config.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file mismatch"):
        module.verify_snapshot(root, manifest)


def test_snapshot_verification_rejects_unexpected_files(tmp_path: Path) -> None:
    module = load_module()
    root = tmp_path / "model"
    _path, manifest = write_snapshot(root, module)
    (root / "unexpected.bin").write_bytes(b"x")

    with pytest.raises(RuntimeError, match="inventory mismatch"):
        module.verify_snapshot(root, manifest)


def test_checked_in_manifest_pins_latest_english_weights() -> None:
    manifest = json.loads(
        Path("config/nemotron-asr-en-0.6b.json").read_text(encoding="utf-8")
    )

    assert manifest["model_id"] == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert manifest["revision"] == "ebe59e5a817142986528bbbee5dba8db7b38ed50"
    assert manifest["files"]["model.safetensors"]["sha256"] == (
        "bddd8a7300826efd19cf7e01f1c7db8402bed6786fc4c7739632894f69c71473"
    )
    assert manifest["inference"]["language_conditioning"] == "checkpoint_fixed_english"
    assert manifest["inference"]["num_lookahead_tokens"] == 13
    assert "max_length" not in manifest["inference"]
    assert manifest["speech_canary"] == {
        "dataset_id": "hf-internal-testing/librispeech_asr_dummy",
        "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a",
        "parquet_path": "clean/validation-00000-of-00001.parquet",
        "parquet_sha256": "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603",
        "row": 0,
        "audio_path": "/opt/voicechat-asr/speech-canary.flac",
        "audio_sha256": "4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41",
        "expected_words": ["mister", "quilter"],
    }


def test_generate_uses_encoder_derived_stop_without_fixed_max_length() -> None:
    observed: dict = {}

    class Inputs(dict):
        def to(self, *_args, **_kwargs):
            return self

    class Sequence:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[1, 2]]

    class Processor:
        def __call__(self, *_args, **_kwargs):
            return Inputs(input_features="features")

        def batch_decode(self, *_args, **_kwargs):
            return ["test"]

    class Model:
        dtype = "float32"

        def generate(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(sequences=Sequence())

    backend = asr_backend.NemotronEnglishAsr.__new__(asr_backend.NemotronEnglishAsr)
    backend.manifest = {
        "inference": {"sampling_rate": 16_000, "max_audio_seconds": 300}
    }
    backend.processor = Processor()
    backend.model = Model()
    backend.device = "cuda:0"
    backend._torch = SimpleNamespace(inference_mode=nullcontext)

    assert backend.transcribe(np.zeros(16_000 * 60, dtype=np.float32))["transcript"] == "test"
    assert observed == {"input_features": "features", "return_dict_in_generate": True}


@pytest.mark.parametrize(
    "tool",
    [
        "tools/qualification/finalize_eartts_component_asr.py",
        "tools/qualification/transcribe_direct_position_probe.py",
    ],
)
def test_safe_path_consumers_resolve_image_equivalent_imports(tool: str) -> None:
    repo = Path.cwd().resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo / "tools/qualification")]
    )
    result = subprocess.run(
        [sys.executable, "-P", str(repo / tool), "--help"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
