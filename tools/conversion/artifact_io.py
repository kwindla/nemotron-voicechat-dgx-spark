"""Shared deterministic tensor-sharding primitives for qualified converters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def quantized_names(weight_name: str) -> dict[str, str]:
    prefix = weight_name.removesuffix(".weight")
    return {
        "qweight": f"{prefix}.qweight",
        "scales": f"{prefix}.scales",
        "qzeros": f"{prefix}.qzeros",
        "g_idx": f"{prefix}.g_idx",
    }


class ShardWriter:
    """Write bundles without splitting one quantized tensor across shards."""

    def __init__(self, root: Path, max_bytes: int, save_file: Callable[..., None]):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = root
        self.max_bytes = max_bytes
        self.save_file = save_file
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.shards: list[tuple[Path, list[str], int]] = []

    def add(self, name: str, tensor: Any) -> None:
        size = tensor_bytes(tensor)
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        self.pending[name] = tensor.contiguous()
        self.pending_bytes += size

    def add_bundle(self, tensors: dict[str, Any]) -> None:
        size = sum(tensor_bytes(tensor) for tensor in tensors.values())
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        for name, tensor in tensors.items():
            self.pending[name] = tensor.contiguous()
        self.pending_bytes += size

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.root / f"model-part-{len(self.shards) + 1:05d}.safetensors"
        names = list(self.pending)
        self.save_file(self.pending, path, metadata={"format": "pt"})
        self.shards.append((path, names, self.pending_bytes))
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> dict[str, Any]:
        self.flush()
        total = len(self.shards)
        weight_map: dict[str, str] = {}
        records = []
        for index, (old_path, names, size) in enumerate(self.shards, 1):
            final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
            final_path = self.root / final_name
            old_path.rename(final_path)
            weight_map.update({name: final_name for name in names})
            records.append({"file": final_name, "bytes": size, "tensors": len(names)})
        return {"weight_map": weight_map, "shards": records}
