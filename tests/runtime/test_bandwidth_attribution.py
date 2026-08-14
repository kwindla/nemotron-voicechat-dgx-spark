"""Focused tests for the bandwidth-attribution analyzer (preregistered side-study)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "benchmark" / "bandwidth_attribution.py"
)
_spec = importlib.util.spec_from_file_location("bandwidth_attribution", MODULE_PATH)
bandwidth_attribution = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bandwidth_attribution", bandwidth_attribution)
_spec.loader.exec_module(bandwidth_attribution)


def test_nvtx_range_regex_matches_step4d_label() -> None:
    match = bandwidth_attribution.NVTX_RANGE.match(
        "step4d:block=3;pair=2;class=content;call=326;step=326"
    )
    assert match is not None
    assert match.group("class") == "content"
    assert match.group("call") == "326"
    assert bandwidth_attribution.NVTX_RANGE.match("step4d:block=3;class=content") is None


def test_tensor_classification_covers_gptq_payloads_and_fails_closed() -> None:
    classify = bandwidth_attribution.classify_tensor
    assert classify("stt_model.llm.layers.0.mixer.in_proj.qweight") == ("mamba_proj_gptq", True)
    assert classify("stt_model.llm.layers.1.mixer.down_proj.g_idx") == ("mamba_proj_gptq", True)
    assert classify("stt_model.lm_head.qweight") == ("lm_head_gptq", True)
    assert classify("stt_model.function_head.weight") == ("function_head_f32", True)
    assert classify("stt_model.embed_tokens.weight") == ("embed_gather", False)
    assert classify("stt_model.llm.layers.9.mixer.q_proj.weight") == ("attention_f32", True)
    assert classify("totally.unknown.tensor") is None


def test_kernel_classification_separates_cublas_gemv_from_cub_sampling() -> None:
    classify = bandwidth_attribution.classify_kernel
    cublas_gemv = (
        "std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, "
        "cublasGemvParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>, float>>"
    )
    assert classify(cublas_gemv) == "unquantized_gemv_bf16"
    assert classify("void marlin::Marlin<(long)1125899906909960>") == "gptq_dequant_gemv"
    assert classify("void at::native::reduce_kernel<(int)512>") == "sampling_argmax"
    assert classify("cub::DeviceRadixSortKernel") == "sampling_argmax"
    assert classify("_selective_scan_update_kernel") == "mamba_scan"
    assert classify("totally_unknown_kernel") is None


def test_interval_union_merges_overlaps_without_double_count() -> None:
    union = bandwidth_attribution.interval_union_ns
    assert union([]) == 0
    assert union([(0, 10), (5, 15)]) == 15
    assert union([(0, 10), (20, 30)]) == 20
    assert union([(20, 30), (0, 25)]) == 30
