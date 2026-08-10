"""Pinned, offline Nemotron English ASR backend for qualification audio."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MANIFEST = Path(
    os.environ.get("VOICECHAT_ASR_MANIFEST", "/opt/voicechat-asr/model-manifest.json")
)
DEFAULT_MODEL = Path(os.environ.get("VOICECHAT_ASR_MODEL", "/models/nemotron-asr"))
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sha256(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_snapshot(model_root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("ASR model manifest has no file inventory")
    actual_names = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and not path.relative_to(model_root).as_posix().startswith(".cache/")
    }
    if actual_names != set(expected):
        raise RuntimeError(
            "ASR snapshot inventory mismatch: "
            f"expected {sorted(expected)}, got {sorted(actual_names)}"
        )
    observed: dict[str, dict[str, Any]] = {}
    for name, record in expected.items():
        path = model_root / name
        observed[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed[name] != record:
            raise RuntimeError(f"ASR snapshot file mismatch: {name}")
    aggregate = snapshot_sha256(observed)
    if aggregate != manifest.get("snapshot_sha256"):
        raise RuntimeError("ASR snapshot aggregate identity mismatch")
    return observed


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "model_id", "revision", "snapshot_sha256", "files", "inference"}
    if manifest.get("schema") != 1 or not required.issubset(manifest):
        raise RuntimeError(f"invalid ASR model manifest: {path}")
    return manifest


def _verify_image_owned_backend() -> tuple[Path, str]:
    active = Path(__file__).resolve()
    expected_path = os.environ.get("VOICECHAT_ASR_EXPECTED_BACKEND")
    expected_sha = os.environ.get("VOICECHAT_ASR_BACKEND_SHA256")
    if not expected_path or active != Path(expected_path).resolve():
        raise RuntimeError(f"ASR backend is not the image-owned implementation: {active}")
    observed_sha = sha256_file(active)
    if not expected_sha or observed_sha != expected_sha:
        raise RuntimeError("ASR backend identity differs from the evaluator image contract")
    return active, observed_sha


class NemotronEnglishAsr:
    """One-load, greedy full-WAV RNNT evaluator with fail-closed provenance."""

    def __init__(
        self,
        *,
        model_root: Path = DEFAULT_MODEL,
        manifest_path: Path = DEFAULT_MANIFEST,
        device: str = "cuda:0",
    ) -> None:
        self.backend_path, self.backend_sha256 = _verify_image_owned_backend()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before evaluator startup")

        import torch
        import transformers
        from transformers import AutoModelForRNNT, AutoProcessor

        self._torch = torch
        self._transformers = transformers
        self.model_root = model_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.snapshot_files = verify_snapshot(self.model_root, self.manifest)
        inference = self.manifest["inference"]
        if transformers.__version__ != "5.14.1":
            raise RuntimeError(f"unexpected Transformers version: {transformers.__version__}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested ASR device is unavailable: {device}")
        self.device = torch.device(device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.processor = AutoProcessor.from_pretrained(
            self.model_root, local_files_only=True, trust_remote_code=False
        )
        if self.processor.feature_extractor.sampling_rate != inference["sampling_rate"]:
            raise RuntimeError("ASR processor sampling rate differs from the manifest")
        if self.processor.default_num_lookahead_tokens != inference["num_lookahead_tokens"]:
            raise RuntimeError("ASR processor lookahead differs from the manifest")
        self.processor.set_num_lookahead_tokens(inference["num_lookahead_tokens"])
        self.model = AutoModelForRNNT.from_pretrained(
            self.model_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        if type(self.model).__name__ != inference["architecture"]:
            raise RuntimeError(f"unexpected ASR architecture: {type(self.model).__name__}")
        if str(self.model.dtype).removeprefix("torch.") != inference["dtype"]:
            raise RuntimeError(f"unexpected ASR dtype: {self.model.dtype}")
        self.attention_implementation = getattr(
            self.model.config, "_attn_implementation", None
        )
        if self.attention_implementation != "sdpa":
            raise RuntimeError(
                f"unexpected ASR attention implementation: {self.attention_implementation}"
            )
        self.model = self.model.to(self.device).eval()

    def transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        inference = self.manifest["inference"]
        values = np.asarray(audio)
        if values.dtype != np.float32 or values.ndim != 1:
            raise ValueError("ASR input must be a one-dimensional float32 array")
        if not values.size or not np.isfinite(values).all():
            raise ValueError("ASR input must contain finite audio samples")
        duration = values.size / inference["sampling_rate"]
        if duration > inference["max_audio_seconds"]:
            raise ValueError(f"ASR input exceeds {inference['max_audio_seconds']} seconds")
        inputs = self.processor(
            values,
            sampling_rate=inference["sampling_rate"],
            is_streaming=False,
            return_tensors="pt",
        ).to(self.device, dtype=self.model.dtype)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, return_dict_in_generate=True)
        token_ids = output.sequences.detach().cpu().tolist()
        transcripts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        if len(transcripts) != 1 or len(token_ids) != 1:
            raise RuntimeError("ASR batch-one contract was violated")
        return {
            "transcript": transcripts[0],
            "token_ids": token_ids[0],
            "audio_samples": int(values.size),
            "audio_seconds": round(duration, 6),
        }

    def provenance(self) -> dict[str, Any]:
        image_id = os.environ.get("VOICECHAT_ASR_EVALUATOR_IMAGE_ID", "")
        if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
            raise RuntimeError("VOICECHAT_ASR_EVALUATOR_IMAGE_ID is not an immutable image ID")
        inference = self.manifest["inference"]
        return {
            "backend": inference["backend"],
            "backend_path": str(self.backend_path),
            "backend_sha256": self.backend_sha256,
            "evaluator_image_id": image_id,
            "model_id": self.manifest["model_id"],
            "hf_revision": self.manifest["revision"],
            "manifest_sha256": sha256_file(self.manifest_path),
            "snapshot_sha256": self.manifest["snapshot_sha256"],
            "snapshot_files": self.snapshot_files,
            "transformers_version": self._transformers.__version__,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "torch_version": self._torch.__version__,
            "cuda_version": self._torch.version.cuda,
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "device": str(self.device),
            "device_name": (
                self._torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else platform.processor()
            ),
            "dtype": inference["dtype"],
            "sampling_rate": inference["sampling_rate"],
            "language": inference["language"],
            "language_conditioning": inference["language_conditioning"],
            "decoder": {
                "strategy": inference["strategy"],
                "stopping": "encoder_exhaustion",
                "num_lookahead_tokens": inference["num_lookahead_tokens"],
                "attention_implementation": self.attention_implementation,
                "streaming": False,
            },
            "deterministic_algorithms": self._torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "float32_matmul_precision": self._torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": self._torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": self._torch.backends.cudnn.allow_tf32,
            "trust_remote_code": False,
            "network_required": False,
        }
