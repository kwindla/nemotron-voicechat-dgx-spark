"""vLLM FP32-activation/INT8-weight linear method for EA EarTTS."""

from __future__ import annotations

from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)


@triton.jit
def _w8a32_linear_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row = tl.program_id(1)
    valid_n = offs_n < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        valid_k = offs_k < K
        x = tl.load(x_ptr + row * K + offs_k, mask=valid_k, other=0.0)
        weights = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=valid_n[:, None] & valid_k[None, :],
            other=0,
        ).to(tl.float32)
        accumulator += tl.sum(weights * x[None, :], axis=1)
    scales = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
    tl.store(output_ptr + row * N + offs_n, accumulator * scales, mask=valid_n)


def _launch_w8a32_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    m, k = x_2d.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if n > k:
        block_n, num_warps = 64, 4
    elif k > n:
        block_n, num_warps = 32, 8
    else:
        block_n, num_warps = 16, 8
    _w8a32_linear_kernel[(triton.cdiv(n, block_n), m)](
        x_2d,
        weight,
        weight_scale,
        output,
        M=m,
        K=k,
        N=n,
        BLOCK_K=256,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return output.reshape(*original_shape[:-1], n)


@torch.library.custom_op("ea::w8a32_linear", mutates_args=())
def w8a32_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return _launch_w8a32_linear(x, weight, weight_scale)


@w8a32_linear.register_fake
def _w8a32_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    del weight_scale
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32, device=x.device)


class EAW8A32Config(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "ea_w8a32"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EAW8A32Config":
        del config
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        del prefix
        if isinstance(layer, LinearBase):
            return EAW8A32LinearMethod()
        return None


class EAW8A32LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.pop("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(layer.weight.data.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data.reshape(-1).contiguous(), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.ops.ea.w8a32_linear.default(x, layer.weight, layer.weight_scale)
        if bias is not None:
            output = output + bias
        return output
