#!/usr/bin/env python3
"""Profiling-only activation and weight attribution for public Nano.

The component gate runs vLLM's V1 engine in another process.  These callables
are sent to that worker with ``collective_rpc`` so the probe can observe or
temporarily replace decoder values without producing another checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODES = {"capture", "oracle-layer", "teacher-down-weight"}


def parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if start > stop:
                raise ValueError(f"invalid descending range: {part}")
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    if not result:
        raise ValueError("integer selection is empty")
    return result


def target_window_summary(
    comparisons: list[dict[str, Any]],
    windows: set[int],
    *,
    max_margin_delta: float = 1.0,
) -> dict[str, Any]:
    selected = [item for item in comparisons if int(item["call_index"]) in windows]
    expected = sorted(windows)
    actual = [int(item["call_index"]) for item in selected]
    missing = sorted(set(expected) - set(actual))
    margin_deltas = [
        abs(float(item["bos_pad_margin_delta"]))
        for item in selected
        if item.get("bos_pad_margin_delta") is not None
    ]
    order = [
        bool(item["bos_pad_order_agrees"])
        for item in selected
        if item.get("bos_pad_order_agrees") is not None
    ]
    token_agreement = (
        sum(bool(item.get("token_agrees")) for item in selected) / len(selected)
        if selected
        else 0.0
    )
    maximum = max(margin_deltas, default=None)
    order_agreement = sum(order) / len(order) if order else 0.0
    passed = (
        not missing
        and maximum is not None
        and maximum <= max_margin_delta
        and order_agreement == 1.0
    )
    return {
        "passed": passed,
        "call_indices": expected,
        "missing_call_indices": missing,
        "comparisons": len(selected),
        "token_agreement": token_agreement,
        "bos_pad_order_agreement": order_agreement,
        "max_absolute_bos_pad_margin_delta": maximum,
        "allowed_max_absolute_bos_pad_margin_delta": max_margin_delta,
    }


def tensor_comparison(left: Any, right: Any) -> dict[str, Any]:
    """Return stable FP32 metrics while accepting arbitrary tensor shapes."""
    import torch

    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"tensor shapes differ: {tuple(left.shape)} != {tuple(right.shape)}")
    delta = right - left
    denominator = torch.linalg.vector_norm(left).clamp_min(torch.finfo(torch.float32).eps)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
    return {
        "elements": left.numel(),
        "cosine": float(cosine),
        "relative_rmse": float(torch.linalg.vector_norm(delta) / denominator),
        "max_absolute_error": float(delta.abs().max()) if delta.numel() else 0.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_from_worker(worker: Any) -> Any:
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("vLLM worker has no loaded model_runner.model")
    # Some vLLM loaders wrap the registered architecture once.
    if not hasattr(model, "model") and hasattr(model, "module"):
        model = model.module
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(f"unexpected Nano worker model: {type(model)!r}")
    return model


class _WorkerSession:
    def __init__(self, model: Any, config: dict[str, Any]):
        import torch

        self.torch = torch
        self.model = model
        self.mode = str(config["mode"])
        if self.mode not in MODES:
            raise ValueError(f"unsupported attribution mode: {self.mode}")
        self.layers = sorted({int(value) for value in config["layers"]})
        self.windows = {int(value) for value in config["windows"]}
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir = (
            Path(config["reference_dir"]).resolve() if config.get("reference_dir") else None
        )
        self.teacher_weight_file = (
            Path(config["teacher_weight_file"]).resolve()
            if config.get("teacher_weight_file")
            else None
        )
        self.call_index = -1
        self.current_call: int | None = None
        self.pending: dict[int, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.teacher_weights: dict[int, Any] = {}
        self.oracle: dict[tuple[int, int], tuple[Any, Any]] = {}

        if len(model.model.layers) <= max(self.layers):
            raise ValueError(
                f"layer selection exceeds model depth {len(model.model.layers)}: {self.layers}"
            )
        for layer_index in self.layers:
            layer = model.model.layers[layer_index]
            down = getattr(getattr(layer, "mixer", None), "down_proj", None)
            if down is None:
                raise ValueError(f"selected layer {layer_index} is not an MLP down_proj layer")

        if self.mode == "oracle-layer":
            self._load_oracle()
        elif self.mode == "teacher-down-weight":
            self._load_teacher_weights()

        self.handles.append(model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        for layer_index in self.layers:
            layer = model.model.layers[layer_index]
            self.handles.append(
                layer.mixer.down_proj.register_forward_hook(
                    self._down_hook(layer_index), with_kwargs=False
                )
            )
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(layer_index), with_kwargs=False)
            )

    def _load_oracle(self) -> None:
        if self.reference_dir is None:
            raise ValueError("oracle-layer mode requires reference_dir")
        for call_index in sorted(self.windows):
            for layer_index in self.layers:
                path = self.reference_dir / f"call-{call_index:06d}-layer-{layer_index:02d}.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"missing oracle activation: {path}")
                payload = self.torch.load(path, map_location="cpu", weights_only=False)
                self.oracle[(call_index, layer_index)] = (
                    payload["layer_hidden"].to("cuda"),
                    payload["layer_residual"].to("cuda"),
                )

    def _load_teacher_weights(self) -> None:
        if self.teacher_weight_file is None:
            raise ValueError("teacher-down-weight mode requires teacher_weight_file")
        if not self.teacher_weight_file.is_file():
            raise FileNotFoundError(self.teacher_weight_file)
        from safetensors import safe_open

        with safe_open(self.teacher_weight_file, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            for layer_index in self.layers:
                candidates = (
                    f"stt_model.llm.layers.{layer_index}.mixer.down_proj.weight",
                    f"model.layers.{layer_index}.mixer.down_proj.weight",
                )
                key = next((name for name in candidates if name in available), None)
                if key is None:
                    raise KeyError(f"teacher weight missing for layer {layer_index}: {candidates}")
                weight = tensors.get_tensor(key)
                if tuple(weight.shape) != (4480, 15680):
                    raise ValueError(
                        f"unexpected teacher weight shape {key}: {tuple(weight.shape)}"
                    )
                self.teacher_weights[layer_index] = weight.to(
                    device="cuda", dtype=self.torch.bfloat16
                )
                del weight

    def _model_pre_hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]):
        # Hooks are installed after vLLM initialization/warmup.  Still require
        # the released custom-input contract so an unrelated dummy forward
        # cannot shift the deterministic replay index.
        if kwargs.get("combined_embeds") is None:
            self.current_call = None
            return
        self.call_index += 1
        self.current_call = self.call_index

    @staticmethod
    def _tensor_output(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    def _down_hook(self, layer_index: int):
        def hook(_module: Any, args: tuple[Any, ...], output: Any):
            call_index = self.current_call
            if call_index not in self.windows:
                return None
            original = self._tensor_output(output)
            entry = self.pending.setdefault(layer_index, {})
            entry["down_proj_input"] = args[0].detach().to("cpu")
            entry["down_proj_candidate"] = original.detach().to("cpu")
            if self.mode != "teacher-down-weight":
                entry["down_proj_effective"] = original.detach().to("cpu")
                return None
            if (
                self.torch.distributed.is_initialized()
                and self.torch.distributed.get_world_size() != 1
            ):
                raise RuntimeError("teacher down_proj substitution is qualified only for TP=1")
            replacement = self.torch.nn.functional.linear(
                args[0].to(dtype=self.torch.bfloat16),
                self.teacher_weights[layer_index],
            ).to(dtype=original.dtype)
            entry["down_proj_effective"] = replacement.detach().to("cpu")
            if isinstance(output, tuple):
                return (replacement, *output[1:])
            return replacement

        return hook

    def _layer_hook(self, layer_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any):
            call_index = self.current_call
            if call_index not in self.windows:
                return None
            hidden, residual = output
            entry = self.pending.setdefault(layer_index, {})
            entry["layer_candidate_hidden"] = hidden.detach().to("cpu")
            entry["layer_candidate_residual"] = residual.detach().to("cpu")
            effective = output
            if self.mode == "oracle-layer":
                effective = self.oracle[(call_index, layer_index)]
            entry["layer_hidden"] = effective[0].detach().to("cpu")
            entry["layer_residual"] = effective[1].detach().to("cpu")
            path = self.output_dir / f"call-{call_index:06d}-layer-{layer_index:02d}.pt"
            self.torch.save(
                {
                    "schema": 1,
                    "mode": self.mode,
                    "call_index": call_index,
                    "layer_index": layer_index,
                    **entry,
                },
                path,
            )
            self.records.append(
                {
                    "call_index": call_index,
                    "layer_index": layer_index,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            self.pending.pop(layer_index, None)
            return effective if self.mode == "oracle-layer" else None

        return hook

    def finalize(self) -> dict[str, Any]:
        for handle in self.handles:
            handle.remove()
        expected = {(call, layer) for call in self.windows for layer in self.layers}
        actual = {(item["call_index"], item["layer_index"]) for item in self.records}
        summary = {
            "schema": 1,
            "kind": "public_nano_in_memory_attribution",
            "mode": self.mode,
            "layers": self.layers,
            "windows": sorted(self.windows),
            "records": sorted(
                self.records, key=lambda item: (item["call_index"], item["layer_index"])
            ),
            "missing": [
                {"call_index": call, "layer_index": layer}
                for call, layer in sorted(expected - actual)
            ],
            "writes_model_artifact": False,
            "substitution_scope": "selected replay calls only",
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary


def install_nano_attribution(worker: Any, config: dict[str, Any]) -> dict[str, Any]:
    """collective_rpc entry point: attach eager-only profiling hooks."""
    if hasattr(worker, "_voicechat_nano_attribution"):
        raise RuntimeError("Nano attribution is already installed")
    model = _model_from_worker(worker)
    session = _WorkerSession(model, config)
    worker._voicechat_nano_attribution = session
    return {
        "installed": True,
        "mode": session.mode,
        "layers": session.layers,
        "windows": sorted(session.windows),
    }


def finalize_nano_attribution(worker: Any) -> dict[str, Any]:
    """collective_rpc entry point: remove hooks and write the capture manifest."""
    session = getattr(worker, "_voicechat_nano_attribution", None)
    if session is None:
        raise RuntimeError("Nano attribution was not installed")
    summary = session.finalize()
    del worker._voicechat_nano_attribution
    return summary


def compare_snapshot_directories(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Compare captured effective down/layer outputs after both engines exit."""
    import torch

    reference_manifest = json.loads((reference_dir / "manifest.json").read_text())
    candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text())
    reference_records = {
        (int(item["call_index"]), int(item["layer_index"])): item
        for item in reference_manifest["records"]
    }
    comparisons = []
    for item in candidate_manifest["records"]:
        key = (int(item["call_index"]), int(item["layer_index"]))
        reference_item = reference_records.get(key)
        if reference_item is None:
            continue
        left = torch.load(
            reference_dir / reference_item["file"], map_location="cpu", weights_only=False
        )
        right = torch.load(candidate_dir / item["file"], map_location="cpu", weights_only=False)
        comparisons.append(
            {
                "call_index": key[0],
                "layer_index": key[1],
                "down_proj_effective": tensor_comparison(
                    left["down_proj_effective"], right["down_proj_effective"]
                ),
                "layer_hidden": tensor_comparison(left["layer_hidden"], right["layer_hidden"]),
                "layer_residual": tensor_comparison(
                    left["layer_residual"], right["layer_residual"]
                ),
            }
        )
    return {
        "reference": str(reference_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "comparisons": comparisons,
        "expected_comparisons": len(reference_records),
        "complete": len(comparisons) == len(reference_records),
    }
