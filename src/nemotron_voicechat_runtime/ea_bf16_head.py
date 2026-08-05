"""Exact-weight BF16 GEMV method for the EA Nemotron ASR output head."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@triton.jit
def _bf16_head_kernel(
    x_ptr,
    weight_ptr,
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
        weight = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=valid_n[:, None] & valid_k[None, :],
            other=0.0,
        )
        accumulator += tl.sum(weight * x[None, :], axis=1)
    tl.store(output_ptr + row * N + offs_n, accumulator, mask=valid_n)


def _launch_bf16_head(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    m, k = x_2d.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _bf16_head_kernel[(triton.cdiv(n, 32), m)](
        x_2d,
        weight,
        output,
        M=m,
        K=k,
        N=n,
        BLOCK_K=256,
        BLOCK_N=32,
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], n)


@torch.library.custom_op("ea::bf16_head_linear", mutates_args=())
def bf16_head_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _launch_bf16_head(x, weight)


@bf16_head_linear.register_fake
def _bf16_head_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32, device=x.device)


class EABF16HeadLinearMethod(UnquantizedLinearMethod):
    """Keep the original BF16 parameter layout and replace only dispatch."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.ops.ea.bf16_head_linear.default(x, layer.weight)
        if bias is not None:
            output = output + bias
        return output
