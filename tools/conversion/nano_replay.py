#!/usr/bin/env python3
"""Deterministic capture primitives for the public VoiceChat Nano component."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu(value: Any) -> Any:
    """Detach tensors while retaining only replay-safe container values."""
    try:
        import torch
    except ImportError:  # pragma: no cover - capture runs in the CUDA image
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}


def _scalar(value: Any) -> int | float | bool | str | None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return None


def generation_result_payload(result: Any) -> dict[str, Any] | None:
    """Convert the vLLM GenerationResult into an inspectable tensor payload."""
    if result is None:
        return None
    custom = getattr(result, "custom_outputs", None) or {}
    return {
        "token_id": _scalar(getattr(result, "token_id", None)),
        "is_finished": bool(getattr(result, "is_finished", False)),
        "finish_reason": getattr(result, "finish_reason", None),
        "total_tokens": _scalar(getattr(result, "total_tokens", None)),
        "custom_outputs": _cpu(custom),
    }


def outputs_contain_full_text_logits(outputs: dict[str, Any]) -> bool:
    """Report whether every recorded engine result retains text logits."""
    results = outputs.get("engine_results") or []
    return bool(results) and all(
        result is not None and "text_logits" in (result.get("custom_outputs") or {})
        for result in results
    )


class NanoReplayRecorder:
    """Write every bounded Nano call in causal order, including available logits."""

    def __init__(self, root: str | os.PathLike[str], max_calls: int = 512):
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Nano replay directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_calls = int(max_calls)
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.manifest = self.root / "manifest.jsonl"
        self.call_index = 0
        self.sequence_epoch = -1
        self.request_epochs: dict[str, int] = {}
        self._contains_full_text_logits: bool | None = None
        self._lock = threading.Lock()
        (self.root / "metadata.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "public_nano_deterministic_replay",
                    "max_calls": self.max_calls,
                    "contains_full_teacher_logits": None,
                    "warning": "Capture synchronizes GPU tensors; timings are invalid.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def write(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        import torch

        with self._lock:
            if self.call_index >= self.max_calls:
                return
            if self._contains_full_text_logits is None:
                self._contains_full_text_logits = outputs_contain_full_text_logits(outputs)
                metadata_path = self.root / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["contains_full_teacher_logits"] = self._contains_full_text_logits
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            index = self.call_index
            self.call_index += 1
            request_id = str(inputs.get("request_id", "default"))
            if request_id not in self.request_epochs:
                self.sequence_epoch += 1
                self.request_epochs[request_id] = self.sequence_epoch
            epoch = self.request_epochs[request_id]
            filename = f"nano-call-{index:06d}.pt"
            path = self.root / filename
            payload = {
                "schema": 1,
                "call_index": index,
                "inputs": _cpu(inputs),
                "outputs": _cpu(outputs),
            }
            torch.save(payload, path)
            record = {
                "call_index": index,
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "current_step": inputs.get("current_step"),
                "decode_steps": inputs.get("decode_steps", 1),
                "request_id": inputs.get("request_id"),
                "sequence_epoch": epoch,
                "predicted_token": _scalar(outputs.get("predicted_token")),
            }
            with self.manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def set_provenance(self, provenance: dict[str, Any]) -> None:
        """Bind a new capture to the already validated checkpoint artifacts."""
        path = self.root / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["checkpoint_provenance"] = provenance
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class CapturingNanoInterface:
    """Transparent VllmLLMModel proxy that retains raw engine logits."""

    def __init__(self, target: Any, recorder: NanoReplayRecorder):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_engine_results", [])
        engine = target.engine
        original_generate = engine.generate_next_token

        async def traced_generate(*args, **kwargs):
            result = await original_generate(*args, **kwargs)
            self._engine_results.append(generation_result_payload(result))
            return result

        object.__setattr__(self, "_original_generate", original_generate)
        engine.generate_next_token = traced_generate

    def __call__(self, input_embeds: Any, request_id: str = "request_id_1", **kwargs):
        self._engine_results.clear()
        result = self._target(input_embeds, request_id=request_id, **kwargs)
        current_step = int(kwargs.get("current_step", 0))
        generated = kwargs.get("generated_tokens")
        if generated is not None:
            # Only the causal prefix participates in sampling. Avoid copying the
            # wrapper's unused 5,000-token allocation on every replay frame.
            generated = generated[..., :current_step]
        inputs = {
            "input_embeds": input_embeds,
            "request_id": request_id,
            "current_step": current_step,
            "decode_steps": int(kwargs.get("decode_steps", 1)),
            "prompt_token_ids": kwargs.get("prompt_token_ids"),
            "generated_tokens": generated,
        }
        outputs = {
            "interface_result": result,
            "predicted_token": result.get("predicted_token") if isinstance(result, dict) else None,
            "engine_results": list(self._engine_results),
        }
        self._recorder.write(inputs, outputs)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)


def install_public_nano_capture(s2s_model: Any) -> NanoReplayRecorder | None:
    """Install the opt-in capture after the exact-public pipeline is built."""
    root = os.environ.get("VOICECHAT_NANO_REPLAY_CAPTURE_DIR", "").strip()
    if not root:
        return None
    if not getattr(s2s_model, "use_vllm_llm", False):
        raise RuntimeError("public Nano replay capture requires a vLLM Nano engine")
    recorder = NanoReplayRecorder(
        root,
        max_calls=int(os.environ.get("VOICECHAT_NANO_REPLAY_MAX_CALLS", "512")),
    )
    s2s_model.model_llm_interface = CapturingNanoInterface(s2s_model.model_llm_interface, recorder)
    return recorder


def read_manifest(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = Path(root).expanduser().resolve() / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = list(range(len(records)))
    actual = [int(record["call_index"]) for record in records]
    if actual != expected:
        raise ValueError(f"Nano replay calls are not contiguous: {actual[:8]}")
    return records
