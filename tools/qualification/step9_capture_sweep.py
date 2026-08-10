#!/usr/bin/env python3
"""Prove that every prefill size through the Step-9 ceiling dispatches a graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-sizes", required=True)
    parser.add_argument("--ceiling", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(value) for value in args.capture_sizes.split(",")]
    if sizes != sorted(set(sizes), reverse=True):
        raise SystemExit("capture sizes must be unique and descending")
    if args.ceiling != sizes[0]:
        raise SystemExit("ceiling must equal the largest capture size")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if os.environ.get("VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE") != "1":
        raise SystemExit("sweep requires VOICECHAT_STEP9_ASSERT_CAPTURE_COVERAGE=1")
    if os.environ.get("VOICECHAT_NANO_PAIR_FULL_GRAPH") != "1":
        raise SystemExit("sweep requires VOICECHAT_NANO_PAIR_FULL_GRAPH=1")

    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationConfig, CompilationLevel
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    compilation = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=sizes,
    )
    compilation.splitting_ops = list(compilation._attention_ops)
    compilation.init_with_cudagraph_sizes(sizes)
    config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(voicechat_pad_pair_token_id=12)
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=1
    )

    histogram: dict[int, int] = {}
    failures = []
    for num_tokens in range(1, args.ceiling + 1):
        padded = compilation.bs_to_padded_graph_size[num_tokens]
        histogram[padded] = histogram.get(padded, 0) + 1
        mode, key = dispatcher.dispatch(
            BatchDescriptor(num_tokens=padded, uniform_decode=False)
        )
        if mode == CUDAGraphMode.NONE or key is None:
            failures.append(
                {"raw_num_tokens": num_tokens, "padded_num_tokens": padded}
            )

    decode_results = []
    for descriptor in (
        BatchDescriptor(num_tokens=1, uniform_decode=True),
        BatchDescriptor(num_tokens=2, uniform_decode=True, query_len=2),
    ):
        mode, key = dispatcher.dispatch(descriptor)
        decode_results.append(
            {
                "descriptor": repr(descriptor),
                "mode": mode.name,
                "key": repr(key),
            }
        )
        if mode == CUDAGraphMode.NONE or key is None:
            failures.append({"decode_descriptor": repr(descriptor)})

    report = {
        "schema": 1,
        "capture_sizes": sizes,
        "ceiling": args.ceiling,
        "prefill_shapes_tested": args.ceiling,
        "raw_to_capture_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "decode_results": decode_results,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
